"""
geometry.py
-----------
Geometric helper functions for the Drishti sonar georeferencing engine.

All functions deal with *sensor-level* geometry (altitudes, ranges, bearings)
before the results are handed to the geodetic layer.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Slant-range → ground-range conversion
# ---------------------------------------------------------------------------


def slant_to_ground_range(slant_range_m: float, altitude_m: float) -> float:
    """
    Convert sonar slant range to horizontal ground range.

    The geometry assumes a flat-bottom model:

        ground_range = sqrt(slant_range² − altitude²)

    Parameters
    ----------
    slant_range_m:
        Measured slant range from sonar to target, in metres (> 0).
    altitude_m:
        Sonar altitude above the seabed / target plane, in metres (> 0).

    Returns
    -------
    Horizontal ground range in metres.

    Raises
    ------
    ValueError
        * If *slant_range_m* ≤ 0 or *altitude_m* ≤ 0.
        * If *slant_range_m* < *altitude_m* (geometrically impossible —
          the slant is shorter than the vertical leg).
        * If *slant_range_m* equals *altitude_m* (target directly beneath
          sonar; ground range is zero — this is allowed and returns 0.0).
    """
    if slant_range_m <= 0.0:
        raise ValueError(
            f"slant_range_m must be > 0, got {slant_range_m}"
        )
    if altitude_m <= 0.0:
        raise ValueError(
            f"altitude_m must be > 0, got {altitude_m}"
        )
    if slant_range_m < altitude_m:
        raise ValueError(
            f"slant_range_m ({slant_range_m} m) must be >= altitude_m "
            f"({altitude_m} m) — slant range cannot be shorter than altitude."
        )

    radicand = slant_range_m ** 2 - altitude_m ** 2
    # Guard against tiny negative values due to floating-point equality
    return math.sqrt(max(0.0, radicand))


# ---------------------------------------------------------------------------
# Bearing arithmetic
# ---------------------------------------------------------------------------


def absolute_bearing(
    sonar_heading_deg: float,
    relative_bearing_deg: float,
) -> float:
    """
    Compute the absolute (true-North) bearing to a target.

    The sonar measures the target at *relative_bearing_deg* degrees from its
    own bow (clockwise positive).  Adding the platform heading converts this
    to a True-North bearing.

    Parameters
    ----------
    sonar_heading_deg:
        True heading of the sonar/platform in decimal degrees (0 … 360).
        Values outside this range are automatically normalised.
    relative_bearing_deg:
        Bearing to the target relative to the sonar bow, in decimal degrees.
        Typically in [−180, 180] or [0, 360]; values are normalised.

    Returns
    -------
    Absolute (True-North) bearing in decimal degrees, normalised to [0, 360).
    """
    return (sonar_heading_deg + relative_bearing_deg) % 360.0
