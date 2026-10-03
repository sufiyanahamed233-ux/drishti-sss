"""
test_georeferencing.py
----------------------
Tests for the Drishti sonar georeferencing engine (Phase 1).

Coverage
--------
geometry.slant_to_ground_range
    - Normal conversion
    - Zero ground range (slant == altitude → target directly below sonar)
    - Invalid inputs: non-positive slant / altitude, slant < altitude

geometry.absolute_bearing
    - Additive composition, normalisation to [0, 360)

geodesy.destination_point
    - Cardinal direction displacements from a known origin
    - Round-trip consistency
    - Invalid lat/lon inputs

engine.georeference  (integration tests)
    - Zero slant range equal to altitude → target at sonar position
    - North / East / South / West cardinal targets
    - Non-zero relative bearing
    - Type errors and value errors for bad inputs
"""

from __future__ import annotations

import math
import pytest

from backend.app.georeferencing.geometry import (
    absolute_bearing,
    slant_to_ground_range,
)
from backend.app.georeferencing.geodesy import (
    destination_point,
    normalise_bearing,
    EARTH_RADIUS_M,
)
from backend.app.georeferencing.engine import georeference, GeoreferencedTarget


# ===========================================================================
# Helpers
# ===========================================================================

def approx_deg(value: float, abs_tol: float = 1e-6) -> float:
    """Return a pytest.approx wrapper with a tight absolute tolerance."""
    return pytest.approx(value, abs=abs_tol)


# ===========================================================================
# geometry.slant_to_ground_range
# ===========================================================================


class TestSlantToGroundRange:
    """Unit tests for the slant→ground range conversion."""

    def test_typical_values(self) -> None:
        """3-4-5 triangle: altitude=3, slant=5 → ground=4."""
        assert slant_to_ground_range(5.0, 3.0) == approx_deg(4.0)

    def test_zero_ground_range(self) -> None:
        """Slant equals altitude means the target is directly beneath the sonar."""
        result = slant_to_ground_range(10.0, 10.0)
        assert result == approx_deg(0.0)

    def test_large_range(self) -> None:
        """1000 m slant, 5 m altitude → nearly 1000 m ground range."""
        result = slant_to_ground_range(1000.0, 5.0)
        expected = math.sqrt(1000.0 ** 2 - 5.0 ** 2)
        assert result == approx_deg(expected)

    def test_slant_less_than_altitude_raises(self) -> None:
        """Slant range shorter than altitude is geometrically impossible."""
        with pytest.raises(ValueError, match="slant_range_m"):
            slant_to_ground_range(3.0, 5.0)

    def test_zero_slant_raises(self) -> None:
        """Zero slant range is invalid."""
        with pytest.raises(ValueError, match="slant_range_m"):
            slant_to_ground_range(0.0, 5.0)

    def test_negative_slant_raises(self) -> None:
        """Negative slant range is invalid."""
        with pytest.raises(ValueError, match="slant_range_m"):
            slant_to_ground_range(-10.0, 5.0)

    def test_zero_altitude_raises(self) -> None:
        """Zero altitude is invalid (would imply sonar on the seabed)."""
        with pytest.raises(ValueError, match="altitude_m"):
            slant_to_ground_range(10.0, 0.0)

    def test_negative_altitude_raises(self) -> None:
        """Negative altitude is invalid."""
        with pytest.raises(ValueError, match="altitude_m"):
            slant_to_ground_range(10.0, -1.0)


# ===========================================================================
# geometry.absolute_bearing
# ===========================================================================


class TestAbsoluteBearing:
    """Unit tests for heading + relative bearing → absolute bearing."""

    def test_north_heading_zero_relative(self) -> None:
        """Heading 0°, target straight ahead → absolute bearing 0°."""
        assert absolute_bearing(0.0, 0.0) == approx_deg(0.0)

    def test_east_heading_zero_relative(self) -> None:
        """Heading 90° (East), target straight ahead → absolute bearing 90°."""
        assert absolute_bearing(90.0, 0.0) == approx_deg(90.0)

    def test_simple_addition(self) -> None:
        """Heading 30°, relative +45° → absolute 75°."""
        assert absolute_bearing(30.0, 45.0) == approx_deg(75.0)

    def test_wrap_around_360(self) -> None:
        """Result is normalised: 350° + 20° = 10° (not 370°)."""
        assert absolute_bearing(350.0, 20.0) == approx_deg(10.0)

    def test_negative_relative_bearing(self) -> None:
        """Heading 10°, relative −30° → absolute 340°."""
        assert absolute_bearing(10.0, -30.0) == approx_deg(340.0)

    def test_full_circle_normalisation(self) -> None:
        """720° input should normalise to 0°."""
        assert absolute_bearing(360.0, 360.0) == approx_deg(0.0)


# ===========================================================================
# geodesy.destination_point
# ===========================================================================


class TestDestinationPoint:
    """Unit tests for the spherical-Earth forward projection."""

    # One degree of latitude ≈ π * R / 180 metres
    _DEG_TO_M = math.pi * EARTH_RADIUS_M / 180.0

    def test_zero_distance_returns_origin(self) -> None:
        """Zero ground range → destination equals origin."""
        lat, lon = destination_point(10.0, 20.0, 45.0, 0.0)
        assert lat == approx_deg(10.0)
        assert lon == approx_deg(20.0)

    def test_due_north(self) -> None:
        """
        Moving due North from (0°, 0°) by ~111 km (≈ 1°) should increase
        latitude by ≈ 1° and leave longitude unchanged.
        """
        distance = self._DEG_TO_M  # metres per degree at equator
        lat, lon = destination_point(0.0, 0.0, 0.0, distance)
        assert lat == pytest.approx(1.0, abs=1e-4)
        assert lon == approx_deg(0.0, abs_tol=1e-4)

    def test_due_south(self) -> None:
        """Moving due South should decrease latitude."""
        distance = self._DEG_TO_M
        lat, lon = destination_point(0.0, 0.0, 180.0, distance)
        assert lat == pytest.approx(-1.0, abs=1e-4)
        assert lon == approx_deg(0.0, abs_tol=1e-4)

    def test_due_east_at_equator(self) -> None:
        """
        Moving due East from (0°, 0°) along the equator by one degree of
        longitude distance should increase longitude by ≈ 1°.
        """
        distance = self._DEG_TO_M  # same at equator
        lat, lon = destination_point(0.0, 0.0, 90.0, distance)
        assert lat == approx_deg(0.0, abs_tol=1e-4)
        assert lon == pytest.approx(1.0, abs=1e-4)

    def test_due_west_at_equator(self) -> None:
        """Moving due West should decrease longitude."""
        distance = self._DEG_TO_M
        lat, lon = destination_point(0.0, 0.0, 270.0, distance)
        assert lat == approx_deg(0.0, abs_tol=1e-4)
        assert lon == pytest.approx(-1.0, abs=1e-4)

    def test_invalid_latitude_raises(self) -> None:
        """Latitude outside [−90, 90] must raise ValueError."""
        with pytest.raises(ValueError, match="Latitude"):
            destination_point(91.0, 0.0, 0.0, 100.0)

    def test_invalid_longitude_raises(self) -> None:
        """Longitude outside [−180, 180] must raise ValueError."""
        with pytest.raises(ValueError, match="Longitude"):
            destination_point(0.0, 181.0, 0.0, 100.0)

    def test_southern_hemisphere(self) -> None:
        """Projection works correctly for negative (southern) latitudes."""
        distance = self._DEG_TO_M
        lat, lon = destination_point(-10.0, 30.0, 0.0, distance)
        assert lat == pytest.approx(-9.0, abs=1e-4)
        assert lon == approx_deg(30.0, abs_tol=1e-4)


# ===========================================================================
# engine.georeference  (integration)
# ===========================================================================


class TestGeoreference:
    """Integration tests for the top-level georeferencing engine."""

    # ------------------------------------------------------------------
    # Basic geometry — target directly below sonar
    # ------------------------------------------------------------------

    def test_target_directly_below_sonar(self) -> None:
        """
        When slant_range == altitude the target is directly beneath the sonar.
        The reported position must equal the sonar position.
        """
        result = georeference(
            sonar_lat=51.5,
            sonar_lon=-0.1,
            sonar_heading=0.0,
            sonar_altitude=10.0,
            target_slant_range=10.0,       # equals altitude → zero ground range
            target_relative_bearing=0.0,
        )
        assert isinstance(result, GeoreferencedTarget)
        assert result.ground_range_m == approx_deg(0.0)
        assert result.latitude == approx_deg(51.5)
        assert result.longitude == approx_deg(-0.1)

    # ------------------------------------------------------------------
    # Cardinal direction tests at the equator / prime meridian
    # ------------------------------------------------------------------

    def _cardinal_setup(
        self,
        sonar_heading: float,
        relative_bearing: float,
        ground_range_m: float = 100.0,
    ) -> GeoreferencedTarget:
        """Helper: sonar at (0°, 0°), altitude 10 m, slant to give *ground_range_m*."""
        slant = math.sqrt(ground_range_m ** 2 + 10.0 ** 2)
        return georeference(
            sonar_lat=0.0,
            sonar_lon=0.0,
            sonar_heading=sonar_heading,
            sonar_altitude=10.0,
            target_slant_range=slant,
            target_relative_bearing=relative_bearing,
        )

    def test_due_north(self) -> None:
        """Heading 0°, relative bearing 0° → target due North (lat increases)."""
        r = self._cardinal_setup(sonar_heading=0.0, relative_bearing=0.0)
        assert r.absolute_bearing_deg == approx_deg(0.0)
        assert r.latitude > 0.0
        assert r.longitude == approx_deg(0.0, abs_tol=1e-5)

    def test_due_east(self) -> None:
        """Heading 90°, relative bearing 0° → target due East (lon increases)."""
        r = self._cardinal_setup(sonar_heading=90.0, relative_bearing=0.0)
        assert r.absolute_bearing_deg == approx_deg(90.0)
        assert r.longitude > 0.0
        assert r.latitude == approx_deg(0.0, abs_tol=1e-5)

    def test_due_south(self) -> None:
        """Heading 180°, relative bearing 0° → target due South (lat decreases)."""
        r = self._cardinal_setup(sonar_heading=180.0, relative_bearing=0.0)
        assert r.absolute_bearing_deg == approx_deg(180.0)
        assert r.latitude < 0.0
        assert r.longitude == approx_deg(0.0, abs_tol=1e-5)

    def test_due_west(self) -> None:
        """Heading 270°, relative bearing 0° → target due West (lon decreases)."""
        r = self._cardinal_setup(sonar_heading=270.0, relative_bearing=0.0)
        assert r.absolute_bearing_deg == approx_deg(270.0)
        assert r.longitude < 0.0
        assert r.latitude == approx_deg(0.0, abs_tol=1e-5)

    def test_heading_north_relative_east(self) -> None:
        """Heading North (0°), target 90° to the right → target due East."""
        r = self._cardinal_setup(sonar_heading=0.0, relative_bearing=90.0)
        assert r.absolute_bearing_deg == approx_deg(90.0)
        assert r.longitude > 0.0

    def test_bearing_wraps_correctly(self) -> None:
        """Heading 350° + relative 20° = absolute 10°, not 370°."""
        r = self._cardinal_setup(sonar_heading=350.0, relative_bearing=20.0)
        assert r.absolute_bearing_deg == approx_deg(10.0)

    # ------------------------------------------------------------------
    # Ground-range round-trip
    # ------------------------------------------------------------------

    def test_ground_range_matches_pythagoras(self) -> None:
        """The reported ground_range_m must equal sqrt(slant²−alt²)."""
        slant, alt = 50.0, 30.0
        expected_gr = math.sqrt(slant ** 2 - alt ** 2)
        r = georeference(
            sonar_lat=10.0,
            sonar_lon=10.0,
            sonar_heading=0.0,
            sonar_altitude=alt,
            target_slant_range=slant,
            target_relative_bearing=0.0,
        )
        assert r.ground_range_m == pytest.approx(expected_gr, rel=1e-9)

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_slant_less_than_altitude_raises(self) -> None:
        with pytest.raises(ValueError, match="slant_range_m"):
            georeference(
                sonar_lat=0.0, sonar_lon=0.0, sonar_heading=0.0,
                sonar_altitude=20.0, target_slant_range=10.0,
                target_relative_bearing=0.0,
            )

    def test_invalid_latitude_raises(self) -> None:
        with pytest.raises(ValueError, match="Latitude"):
            georeference(
                sonar_lat=95.0, sonar_lon=0.0, sonar_heading=0.0,
                sonar_altitude=10.0, target_slant_range=15.0,
                target_relative_bearing=0.0,
            )

    def test_invalid_longitude_raises(self) -> None:
        with pytest.raises(ValueError, match="Longitude"):
            georeference(
                sonar_lat=0.0, sonar_lon=200.0, sonar_heading=0.0,
                sonar_altitude=10.0, target_slant_range=15.0,
                target_relative_bearing=0.0,
            )

    def test_zero_altitude_raises(self) -> None:
        with pytest.raises(ValueError, match="altitude_m"):
            georeference(
                sonar_lat=0.0, sonar_lon=0.0, sonar_heading=0.0,
                sonar_altitude=0.0, target_slant_range=10.0,
                target_relative_bearing=0.0,
            )

    def test_non_numeric_input_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            georeference(
                sonar_lat="north",  # type: ignore[arg-type]
                sonar_lon=0.0, sonar_heading=0.0,
                sonar_altitude=10.0, target_slant_range=15.0,
                target_relative_bearing=0.0,
            )

    def test_nan_input_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            georeference(
                sonar_lat=float("nan"),
                sonar_lon=0.0, sonar_heading=0.0,
                sonar_altitude=10.0, target_slant_range=15.0,
                target_relative_bearing=0.0,
            )

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------

    def test_deterministic(self) -> None:
        """Same inputs always produce exactly the same outputs."""
        kwargs = dict(
            sonar_lat=48.858,
            sonar_lon=2.294,
            sonar_heading=123.4,
            sonar_altitude=5.0,
            target_slant_range=15.0,
            target_relative_bearing=-30.0,
        )
        r1 = georeference(**kwargs)
        r2 = georeference(**kwargs)
        assert r1 == r2
