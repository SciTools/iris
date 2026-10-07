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


class TestMakeCellCheck:
    """Unit tests for the :func:`iris.util._make_cell_check` helper.

    Tests the constraint-path logic independently of CRS transforms.
    """

    from iris.coords import Cell

    def _cell(self, point, bound=None):
        from iris.coords import Cell

        return Cell(point=point, bound=bound)

    def test_point_only_cell_inclusive(self):
        check = iris.util._make_cell_check(0, 10, True, True, False, 0)
        assert check(self._cell(0))
        assert check(self._cell(5))
        assert check(self._cell(10))
        assert not check(self._cell(-1))
        assert not check(self._cell(11))

    def test_point_only_cell_exclusive_min(self):
        check = iris.util._make_cell_check(0, 10, False, True, False, 0)
        assert not check(self._cell(0))
        assert check(self._cell(5))
        assert check(self._cell(10))

    def test_point_only_cell_exclusive_max(self):
        check = iris.util._make_cell_check(0, 10, True, False, False, 0)
        assert check(self._cell(0))
        assert check(self._cell(5))
        assert not check(self._cell(10))

    def test_bounded_cell_overlaps(self):
        # Cells whose bounds overlap the region should be included
        check = iris.util._make_cell_check(0, 10, True, True, False, 0)
        assert check(self._cell(5, bound=(-2, 12)))  # spans region
        assert check(self._cell(2, bound=(0, 5)))    # lower boundary touch
        assert check(self._cell(8, bound=(5, 10)))   # upper boundary touch
        assert not check(self._cell(-5, bound=(-10, -1)))  # entirely below

    def test_bounded_ignore_bounds_uses_point(self):
        # With ignore_bounds=True, only the point is checked, not the bounds
        check = iris.util._make_cell_check(0, 10, True, True, True, 0)
        # Cell point=5, bound spanning outside region -> included (point in range)
        assert check(self._cell(5, bound=(-20, 20)))
        # Cell point=-5, bound spanning into region -> excluded (point out of range)
        assert not check(self._cell(-5, bound=(-10, 5)))

    def test_threshold_full_overlap_required(self):
        # threshold=1.0 requires the cell to be fully within [0, 10]
        check = iris.util._make_cell_check(0, 10, True, True, False, 1.0)
        assert check(self._cell(5, bound=(2, 8)))      # fully inside
        assert not check(self._cell(5, bound=(-2, 8))) # partially outside at min
        assert not check(self._cell(5, bound=(2, 12))) # partially outside at max

    def test_threshold_half_overlap(self):
        # threshold=0.5: cell [−5, 5] has 5 units overlap with [0, 10],
        # cell size is 10, fraction = 0.5 → included at threshold=0.5
        check = iris.util._make_cell_check(0, 10, True, True, False, 0.5)
        assert check(self._cell(0, bound=(-5, 5)))
        # cell [−6, 4] has 4/10 = 0.4 overlap -> excluded
        assert not check(self._cell(-1, bound=(-6, 4)))
