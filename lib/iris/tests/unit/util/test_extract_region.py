# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.util.extract_region`."""

import cartopy.crs as ccrs
import numpy as np
import pytest

from iris.coord_systems import GeogCS
from iris.coords import DimCoord
from iris.cube import Cube
import iris
import iris.util


GEOGCS = GeogCS(6371229.0)


def _make_global_latlon_cube():
    """Make a simple global lat/lon cube for testing."""
    lon_points = np.arange(-180, 181, 10.0)
    lat_points = np.arange(-90, 91, 10.0)
    lon = DimCoord(
        lon_points,
        standard_name="longitude",
        units="degrees",
        coord_system=GEOGCS,
        circular=True,
    )
    lat = DimCoord(
        lat_points,
        standard_name="latitude",
        units="degrees",
        coord_system=GEOGCS,
    )
    data = np.zeros((len(lat_points), len(lon_points)))
    cube = Cube(data, dim_coords_and_dims=[(lat, 0), (lon, 1)])
    return cube


class TestExtractRegionBasic:
    """Basic extraction tests (existing behaviour, not new parameters)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = _make_global_latlon_cube()

    def test_returns_cube(self):
        result = iris.util.extract_region(self.cube, [-10, -10, 10, 10])
        assert isinstance(result, Cube)

    def test_correct_lon_range(self):
        result = iris.util.extract_region(self.cube, [-10, -90, 10, 90])
        lon = result.coord("longitude").points
        assert lon.min() >= -10
        assert lon.max() <= 10

    def test_correct_lat_range(self):
        result = iris.util.extract_region(self.cube, [-180, -10, 180, 10])
        lat = result.coord("latitude").points
        assert lat.min() >= -10
        assert lat.max() <= 10

    def test_invalid_area_length(self):
        with pytest.raises(ValueError, match="length 2 or 4"):
            iris.util.extract_region(self.cube, [1, 2, 3])

    def test_invalid_crs(self):
        with pytest.raises(TypeError, match="not a valid coordinate reference system"):
            iris.util.extract_region(self.cube, [-10, -10, 10, 10], crs="bad")


class TestExtractRegionInclusivity:
    """Tests for min_inclusive and max_inclusive parameters.

    Inclusivity is most cleanly tested with ``ignore_bounds=True`` (points
    only) and ``crs=GEOGCS.as_cartopy_crs()`` (cube's own CRS) so that the
    coordinate transform is identity and boundary comparisons are exact.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = _make_global_latlon_cube()
        self.native_crs = GEOGCS.as_cartopy_crs()

    def test_min_inclusive_true(self):
        # lon=-10 is a grid point; it should be included when min_inclusive=True
        result = iris.util.extract_region(
            self.cube,
            [-10, -90, 10, 90],
            crs=self.native_crs,
            ignore_bounds=True,
            min_inclusive=True,
        )
        lon = result.coord("longitude").points
        assert -10.0 in lon

    def test_min_inclusive_false(self):
        # lon=-10 is a grid point; it should be excluded when min_inclusive=False
        result = iris.util.extract_region(
            self.cube,
            [-10, -90, 10, 90],
            crs=self.native_crs,
            ignore_bounds=True,
            min_inclusive=False,
        )
        lon = result.coord("longitude").points
        assert -10.0 not in lon

    def test_max_inclusive_true(self):
        # lon=10 is a grid point; it should be included when max_inclusive=True
        result = iris.util.extract_region(
            self.cube,
            [-10, -90, 10, 90],
            crs=self.native_crs,
            ignore_bounds=True,
            max_inclusive=True,
        )
        lon = result.coord("longitude").points
        assert 10.0 in lon

    def test_max_inclusive_false(self):
        # lon=10 is a grid point; it should be excluded when max_inclusive=False
        result = iris.util.extract_region(
            self.cube,
            [-10, -90, 10, 90],
            crs=self.native_crs,
            ignore_bounds=True,
            max_inclusive=False,
        )
        lon = result.coord("longitude").points
        assert 10.0 not in lon

    def test_both_exclusive(self):
        result = iris.util.extract_region(
            self.cube,
            [-10, -90, 10, 90],
            crs=self.native_crs,
            ignore_bounds=True,
            min_inclusive=False,
            max_inclusive=False,
        )
        lon = result.coord("longitude").points
        assert -10.0 not in lon
        assert 10.0 not in lon
        assert lon.min() > -10.0
        assert lon.max() < 10.0


class TestExtractRegionIgnoreBounds:
    """Tests for the ignore_bounds parameter."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = _make_global_latlon_cube()
        # Ensure bounds exist
        self.cube.coord("longitude").guess_bounds()
        self.cube.coord("latitude").guess_bounds()

    def test_ignore_bounds_false(self):
        # Default: bounds-aware extraction should include cells whose bounds
        # touch the region boundary.
        result_bounds = iris.util.extract_region(
            self.cube, [-15, -90, 15, 90], ignore_bounds=False
        )
        result_points = iris.util.extract_region(
            self.cube, [-15, -90, 15, 90], ignore_bounds=True
        )
        # With bounds, we may get the same or more cells than with points only
        assert result_bounds.shape[1] >= result_points.shape[1]

    def test_ignore_bounds_true_returns_cube(self):
        result = iris.util.extract_region(
            self.cube, [-10, -10, 10, 10], ignore_bounds=True
        )
        assert isinstance(result, Cube)


class TestExtractRegionThreshold:
    """Tests for the threshold parameter."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = _make_global_latlon_cube()
        self.cube.coord("longitude").guess_bounds()
        self.cube.coord("latitude").guess_bounds()

    def test_threshold_zero_default(self):
        result = iris.util.extract_region(
            self.cube, [-15, -90, 15, 90], threshold=0
        )
        assert isinstance(result, Cube)

    def test_high_threshold_excludes_partial_cells(self):
        # A threshold of 1.0 requires full overlap; partial boundary cells
        # should be excluded.
        result_low = iris.util.extract_region(
            self.cube, [-15, -90, 15, 90], threshold=0
        )
        result_high = iris.util.extract_region(
            self.cube, [-15, -90, 15, 90], threshold=1.0
        )
        assert result_high.shape[1] <= result_low.shape[1]
