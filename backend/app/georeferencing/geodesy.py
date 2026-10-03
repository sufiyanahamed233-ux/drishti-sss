"""
geodesy.py
----------
Low-level geodetic helpers for the Drishti sonar georeferencing engine.

Uses the *spherical-Earth* approximation with the WGS-84 mean radius
(6 371 008.8 m).  For typical sonar operating ranges (< 1 km) this
introduces sub-centimetre errors compared with full ellipsoidal maths,
which is well within the accuracy budget of a sonar-altimeter system.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: WGS-84 mean Earth radius in metres.
EARTH_RADIUS_M: float = 6_371_008.8

#: Degrees-to-radians conversion factor.
DEG2RAD: float = math.pi / 180.0

#: Radians-to-degrees conversion factor.
RAD2DEG: float = 180.0 / math.pi


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def bearing_to_radians(bearing_deg: float) -> float:
    """Convert a compass bearing (0–360 °, clockwise from North) to radians."""
    return bearing_deg * DEG2RAD


def normalise_bearing(bearing_deg: float) -> float:
    """Return *bearing_deg* normalised to the range [0, 360)."""
    return bearing_deg % 360.0


def destination_point(
    lat_deg: float,
    lon_deg: float,
    bearing_deg: float,
    distance_m: float,
) -> tuple[float, float]:
    """
    Compute the destination point given a start point, bearing, and distance.

    Uses the spherical-Earth forward (direct) formula.

    Parameters
    ----------
    lat_deg:
        Geodetic latitude of the start point in decimal degrees (−90 … +90).
    lon_deg:
        Geodetic longitude of the start point in decimal degrees (−180 … +180).
    bearing_deg:
        Compass bearing from start to destination, in decimal degrees,
        measured clockwise from True North (0 … 360).
    distance_m:
        Great-circle distance from start to destination in metres.

    Returns
    -------
    (lat_deg, lon_deg) of the destination point in decimal degrees.

    Raises
    ------
    ValueError
        If *lat_deg* is outside [−90, 90] or *lon_deg* is outside [−180, 180].
    """
    if not -90.0 <= lat_deg <= 90.0:
        raise ValueError(
            f"Latitude must be in [-90, 90], got {lat_deg}"
        )
    if not -180.0 <= lon_deg <= 180.0:
        raise ValueError(
            f"Longitude must be in [-180, 180], got {lon_deg}"
        )

    # Angular distance on the sphere
    delta = distance_m / EARTH_RADIUS_M  # radians

    lat1 = lat_deg * DEG2RAD
    lon1 = lon_deg * DEG2RAD
    theta = bearing_deg * DEG2RAD

    sin_lat1 = math.sin(lat1)
    cos_lat1 = math.cos(lat1)
    sin_delta = math.sin(delta)
    cos_delta = math.cos(delta)

    sin_lat2 = sin_lat1 * cos_delta + cos_lat1 * sin_delta * math.cos(theta)
    # Clamp to [-1, 1] to guard against floating-point drift at poles
    sin_lat2 = max(-1.0, min(1.0, sin_lat2))
    lat2 = math.asin(sin_lat2)

    lon2 = lon1 + math.atan2(
        math.sin(theta) * sin_delta * cos_lat1,
        cos_delta - sin_lat1 * sin_lat2,
    )

    # Normalise longitude to (−π, π]
    lon2 = (lon2 + math.pi) % (2 * math.pi) - math.pi

    return lat2 * RAD2DEG, lon2 * RAD2DEG
