"""
engine.py
---------
Sonar georeferencing engine for the Drishti project.

Public API
~~~~~~~~~~
``georeference(sonar_lat, sonar_lon, sonar_heading, sonar_altitude,
               target_slant_range, target_relative_bearing)``

    Given the six sonar observables, return the estimated geographic position
    of the detected target as ``(latitude_deg, longitude_deg)``.

Algorithm
~~~~~~~~~
1. Validate all inputs.
2. Convert slant range → horizontal ground range using the Pythagorean
   flat-bottom model.
3. Compute the absolute (True-North) bearing to the target by adding the
   sonar platform heading to the target's relative bearing.
4. Project the sonar position forward by *ground_range* along
   *absolute_bearing* using the spherical-Earth forward formula (see
   ``geodesy.destination_point``).

The result is deterministic: identical inputs always produce identical outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import absolute_bearing, slant_to_ground_range
from .geodesy import destination_point, normalise_bearing


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GeoreferencedTarget:
    """Immutable result produced by :func:`georeference`."""

    #: Estimated target latitude in decimal degrees (−90 … +90).
    latitude: float
    #: Estimated target longitude in decimal degrees (−180 … +180).
    longitude: float
    #: Computed horizontal ground range in metres.
    ground_range_m: float
    #: Computed absolute (True-North) bearing to the target in degrees [0, 360).
    absolute_bearing_deg: float


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def georeference(
    sonar_lat: float,
    sonar_lon: float,
    sonar_heading: float,
    sonar_altitude: float,
    target_slant_range: float,
    target_relative_bearing: float,
) -> GeoreferencedTarget:
    """
    Georeference a sonar target detection.

    Parameters
    ----------
    sonar_lat:
        WGS-84 latitude of the sonar in decimal degrees (−90 … +90).
    sonar_lon:
        WGS-84 longitude of the sonar in decimal degrees (−180 … +180).
    sonar_heading:
        True (magnetic-declination-corrected) heading of the sonar platform
        in decimal degrees.  Values outside [0, 360) are normalised.
    sonar_altitude:
        Height of the sonar above the seabed / target plane in metres (> 0).
    target_slant_range:
        Measured slant range from the sonar to the target in metres (≥ altitude).
    target_relative_bearing:
        Bearing to the target measured relative to the sonar bow, in decimal
        degrees (clockwise positive).  −180 … +180 or 0 … 360 are both
        accepted.

    Returns
    -------
    :class:`GeoreferencedTarget`
        The estimated geographic position of the target together with the
        intermediate computed values.

    Raises
    ------
    TypeError
        If any argument is not a real number.
    ValueError
        If *sonar_lat* is outside [−90, 90], *sonar_lon* outside [−180, 180],
        *sonar_altitude* ≤ 0, or *target_slant_range* < *sonar_altitude*.
    """
    # ------------------------------------------------------------------
    # 1. Type validation
    # ------------------------------------------------------------------
    _args = {
        "sonar_lat": sonar_lat,
        "sonar_lon": sonar_lon,
        "sonar_heading": sonar_heading,
        "sonar_altitude": sonar_altitude,
        "target_slant_range": target_slant_range,
        "target_relative_bearing": target_relative_bearing,
    }
    for name, value in _args.items():
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"'{name}' must be a real number, got {type(value).__name__!r}"
            )
        if value != value:  # NaN check (math.isnan would also work)
            raise ValueError(f"'{name}' must not be NaN")

    # ------------------------------------------------------------------
    # 2. Slant → ground range  (validates altitude & range internally)
    # ------------------------------------------------------------------
    ground_range = slant_to_ground_range(target_slant_range, sonar_altitude)

    # ------------------------------------------------------------------
    # 3. Absolute bearing
    # ------------------------------------------------------------------
    abs_bearing = normalise_bearing(
        absolute_bearing(sonar_heading, target_relative_bearing)
    )

    # ------------------------------------------------------------------
    # 4. Forward geodetic projection  (validates lat/lon internally)
    # ------------------------------------------------------------------
    target_lat, target_lon = destination_point(
        sonar_lat, sonar_lon, abs_bearing, ground_range
    )

    return GeoreferencedTarget(
        latitude=target_lat,
        longitude=target_lon,
        ground_range_m=ground_range,
        absolute_bearing_deg=abs_bearing,
    )
