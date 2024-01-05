# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for :func:`iris.experimental.regrid.regrid_conservative_via_esmpy`."""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests  # isort:skip

import contextlib
import unittest

import cf_units
import numpy as np

# Import ESMF if installed, else fail quietly + disable all the tests.
try:
    import ESMF
except ImportError:
    ESMF = None
skip_esmf = unittest.skipIf(
    condition=ESMF is None, reason="Requires ESMF, which is not available."
)

import iris
import iris.analysis
import iris.analysis.cartography as i_cartog
from iris.experimental.regrid_conservative import regrid_conservative_via_esmpy
import iris.tests.stock as istk

_PLAIN_GEODETIC_CS = iris.coord_systems.GeogCS(i_cartog.DEFAULT_SPHERICAL_EARTH_RADIUS)


def _make_test_cube(shape, xlims, ylims, pole_latlon=None):
    """Create latlon cube (optionally rotated) with given xy dimensions and bounds
    limit values.

    Produces a regular grid in source coordinates.
    Does not work for 1xN or Nx1 grids, because guess_bounds fails.

    """
    nx, ny = shape
    cube = iris.cube.Cube(np.zeros((ny, nx)))
    xvals = np.linspace(xlims[0], xlims[1], nx)
    yvals = np.linspace(ylims[0], ylims[1], ny)
    coordname_prefix = ""
    cs = _PLAIN_GEODETIC_CS
    if pole_latlon is not None:
        coordname_prefix = "grid_"
        pole_lat, pole_lon = pole_latlon
        cs = iris.coord_systems.RotatedGeogCS(
            grid_north_pole_latitude=pole_lat,
            grid_north_pole_longitude=pole_lon,
            ellipsoid=cs,
        )

    co_x = iris.coords.DimCoord(
        xvals,
        standard_name=coordname_prefix + "longitude",
        units=cf_units.Unit("degrees"),
        coord_system=cs,
    )
    co_x.guess_bounds()
    cube.add_dim_coord(co_x, 1)
    co_y = iris.coords.DimCoord(
        yvals,
        standard_name=coordname_prefix + "latitude",
        units=cf_units.Unit("degrees"),
        coord_system=cs,
    )
    co_y.guess_bounds()
    cube.add_dim_coord(co_y, 0)
    return cube


def _cube_area_sum(cube):
    """Calculate total area-sum - Iris can't do this in one operation."""
    area_sums = cube * i_cartog.area_weights(cube, normalize=False)
    area_sum = area_sums.collapsed(area_sums.coords(dim_coords=True), iris.analysis.SUM)
    return area_sum.data.flatten()[0]


def _reldiff(a, b):
    """Compute a relative-difference measure between real numbers.

    Result is:
        if a == b == 0:
            0.0
        otherwise:
            |a - b| / mean(|a|, |b|)

    """
    if a == 0.0 and b == 0.0:
        return 0.0
    return abs(a - b) * 2.0 / (abs(a) + abs(b))


def _minmax(v):
    """Calculate [min, max] of input."""
    return [f(v) for f in (np.min, np.max)]


@contextlib.contextmanager
def _donothing_context_manager():
    yield


@skip_esmf
class TestConservativeRegrid(tests.IrisTest):
    def setUp(self):
        # Compute basic test data cubes.
        shape1 = (5, 5)
        xlims1, ylims1 = ((-2, 2), (-2, 2))
        c1 = _make_test_cube(shape1, xlims1, ylims1)
        c1.data[:] = 0.0
        c1.data[2, 2] = 1.0

        shape2 = (4, 4)
        xlims2, ylims2 = ((-1.5, 1.5), (-1.5, 1.5))
        c2 = _make_test_cube(shape2, xlims2, ylims2)
        c2.data[:] = 0.0

        # Save timesaving pre-computed bits
        self.stock_c1_c2 = (c1, c2)
        self.stock_regrid_c1toc2 = regrid_conservative_via_esmpy(c1, c2)
        self.stock_c1_areasum = _cube_area_sum(c1)

    def test_simple_areas(self):
        """Test area-conserving regrid between simple "near-square" grids.

        Grids have overlapping areas in the same (lat-lon) coordinate system.
        Grids are "nearly flat" lat-lon spaces (small ranges near the equator).

        """
        c1, c2 = self.stock_c1_c2
        c1_areasum = self.stock_c1_areasum

        # main regrid
        c1to2 = regrid_conservative_via_esmpy(c1, c2)

        c1to2_areasum = _cube_area_sum(c1to2)

        # Check expected result (Cartesian equivalent, so not exact).
        d_expect = np.array(
            [
                [0.00, 0.00, 0.00, 0.00],
                [0.00, 0.25, 0.25, 0.00],
                [0.00, 0.25, 0.25, 0.00],
                [0.00, 0.00, 0.00, 0.00],
            ]
        )
        # Numbers are slightly off (~0.25000952).  This is expected.
        self.assertArrayAllClose(c1to2.data, d_expect, rtol=5.0e-5)

        # check that the area sums are equivalent, simple total is a bit off
        self.assertArrayAllClose(c1to2_areasum, c1_areasum)

        #
        # regrid back onto original grid again ...
        #
        c1to2to1 = regrid_conservative_via_esmpy(c1to2, c1)

        c1to2to1_areasum = _cube_area_sum(c1to2to1)

        # Check expected result (Cartesian/exact difference now greater)
        d_expect = np.array(
            [
                [0.0, 0.0000, 0.0000, 0.0000, 0.0],
                [0.0, 0.0625, 0.1250, 0.0625, 0.0],
                [0.0, 0.1250, 0.2500, 0.1250, 0.0],
                [0.0, 0.0625, 0.1250, 0.0625, 0.0],
                [0.0, 0.0000, 0.0000, 0.0000, 0.0],
            ]
        )
        self.assertArrayAllClose(c1to2to1.data, d_expect, atol=0.00002)

        # check area sums again
        self.assertArrayAllClose(c1to2to1_areasum, c1_areasum)

    def test_simple_missing_data(self):
        """Check for missing data handling.

        Should mask cells that either ..
          (a) go partly outside the source grid
          (b) partially overlap masked source data

        """
        c1, c2 = self.stock_c1_c2

        # regrid from c2 to c1 -- should mask all the edges...
        c2_to_c1 = regrid_conservative_via_esmpy(c2, c1)
        self.assertArrayEqual(
            c2_to_c1.data.mask,
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
        )

        # do same with a particular point masked
        c2m = c2.copy()
        c2m.data = np.ma.array(c2m.data)
        c2m.data[1, 1] = np.ma.masked
        c2m_to_c1 = regrid_conservative_via_esmpy(c2m, c1)
        self.assertArrayEqual(
            c2m_to_c1.data.mask,
            [
                [True, True, True, True, True],
                [True, True, True, False, True],
                [True, True, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
        )

    @tests.skip_data
    def test_multidimensional(self):
        """Check valid operation on a multidimensional cube.

        Calculation should repeat across multiple dimensions.
        Any attached orography is interpolated.

        NOTE: in future, extra dimensions may be passed through to ESMF:  At
        present, it repeats the calculation on 2d slices.  So we check that
        at least the results are equivalent (as it's quite easy to do).

        """
        # Get some higher-dimensional test data
        c1 = istk.realistic_4d()
        # Chop down to small size, and mask some data
        c1 = c1[:3, :4, :16, :12]
        c1.data[:, 2, :, :] = np.ma.masked
        c1.data[1, 1, 3:9, 4:7] = np.ma.masked
        # Give it a slightly more challenging indexing order: tzyx --> xzty
        c1.transpose((3, 1, 0, 2))

        # Construct a (coarser) target grid of about the same extent
        c1_cs = c1.coord(axis="x").coord_system
        xlims = _minmax(c1.coord(axis="x").contiguous_bounds())
        ylims = _minmax(c1.coord(axis="y").contiguous_bounds())
        # Reduce the dimensions slightly to avoid NaNs in regridded orography
        delta = 0.05
        # || NOTE: this is *not* a small amount.  Think there is a bug.
        # || NOTE: See https://github.com/SciTools/iris/issues/458
        xlims = np.interp([delta, 1.0 - delta], [0, 1], xlims)
        ylims = np.interp([delta, 1.0 - delta], [0, 1], ylims)
        pole_latlon = (
            c1_cs.grid_north_pole_latitude,
            c1_cs.grid_north_pole_longitude,
        )
        c2 = _make_test_cube((7, 8), xlims, ylims, pole_latlon=pole_latlon)

        # regrid onto new grid
        c1_to_c2 = regrid_conservative_via_esmpy(c1, c2)

        # check that all the original coords exist in the new cube
        # NOTE: this also effectively confirms we haven't lost the orography
        def list_coord_names(cube):
            return sorted([coord.name() for coord in cube.coords()])

        self.assertEqual(list_coord_names(c1_to_c2), list_coord_names(c1))

        # check that each xy 'slice' has same values as if done on its own.
        for i_p, i_t in np.ndindex(c1.shape[1:3]):
            c1_slice = c1[:, i_p, i_t]
            c2_slice = regrid_conservative_via_esmpy(c1_slice, c2)
            subcube = c1_to_c2[:, i_p, i_t]
            self.assertEqual(subcube, c2_slice)

        # check all other metadata
        self.assertEqual(c1_to_c2.metadata, c1.metadata)

    def test_xy_transposed(self):
        # Test effects of transposing X and Y in src/dst data.
        c1, c2 = self.stock_c1_c2
        testcube_xy = self.stock_regrid_c1toc2

        # Check that transposed data produces transposed results
        # - i.e.  regrid(data^T)^T == regrid(data)
        c1_yx = c1.copy()
        c1_yx.transpose()
        testcube_yx = regrid_conservative_via_esmpy(c1_yx, c2)
        testcube_yx.transpose()
        self.assertEqual(testcube_yx, testcube_xy)

        # Check that transposing destination does nothing
        c2_yx = c2.copy()
        c2_yx.transpose()
        testcube_dst_transpose = regrid_conservative_via_esmpy(c1, c2_yx)
        self.assertEqual(testcube_dst_transpose, testcube_xy)

    def test_same_grid(self):
        # Test regridding onto the identical grid.
        # Use regrid with self as target.
        c1, _ = self.stock_c1_c2
        testcube = regrid_conservative_via_esmpy(c1, c1)
        self.assertEqual(testcube, c1)

    def test_global(self):
        # Test global regridding.
        # Compute basic test data cubes.
        shape1 = (8, 6)
        xlim1 = 180.0 * (shape1[0] - 1) / shape1[0]
        ylim1 = 90.0 * (shape1[1] - 1) / shape1[1]
        c1 = _make_test_cube(shape1, (-xlim1, xlim1), (-ylim1, ylim1))
        # Create a small, plausible global array:
        # - top + bottom rows all the same
        # - left + right columns "mostly close" for checking across the seam
        basedata = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 4, 4, 4, 2, 2, 1],
                [2, 1, 4, 4, 4, 2, 2, 2],
                [2, 5, 5, 1, 1, 1, 5, 5],
                [5, 5, 5, 1, 1, 1, 5, 5],
                [5, 5, 5, 5, 5, 5, 5, 5],
            ]
        )
        c1.data[:] = basedata

        # Create a rotated grid to regrid this onto.
        shape2 = (14, 11)
        xlim2 = 180.0 * (shape2[0] - 1) / shape2[0]
        ylim2 = 90.0 * (shape2[1] - 1) / shape2[1]
        c2 = _make_test_cube(
            shape2, (-xlim2, xlim2), (-ylim2, ylim2), pole_latlon=(47.4, 25.7)
        )

        # Perform regridding
        c1toc2 = regrid_conservative_via_esmpy(c1, c2)

        # Check that before+after area-sums match fairly well
        c1_areasum = _cube_area_sum(c1)
        c1toc2_areasum = _cube_area_sum(c1toc2)
        self.assertArrayAllClose(c1toc2_areasum, c1_areasum, rtol=0.006)

    def test_global_collapse(self):
        # Test regridding global data to a single cell.
        # Fetch 'standard' testcube data
        c1, _ = self.stock_c1_c2
        c1_areasum = self.stock_c1_areasum

        # Condense entire globe onto a single cell
        x_coord_2 = iris.coords.DimCoord(
            [0.0],
            bounds=[-180.0, 180.0],
            standard_name="longitude",
            units="degrees",
            coord_system=_PLAIN_GEODETIC_CS,
        )
        y_coord_2 = iris.coords.DimCoord(
            [0.0],
            bounds=[-90.0, 90.0],
            standard_name="latitude",
            units="degrees",
            coord_system=_PLAIN_GEODETIC_CS,
        )
        c2 = iris.cube.Cube([[0.0]])
        c2.add_dim_coord(y_coord_2, 0)
        c2.add_dim_coord(x_coord_2, 1)

        # NOTE: at present, this causes an error inside ESMF ...
        context = self.assertRaises(ValueError)
        global_cell_supported = False
        if global_cell_supported:
            context = _donothing_context_manager()
        with context:
            c1_to_global = regrid_conservative_via_esmpy(c1, c2)
            # Check the total area sum is still the same
            self.assertArrayAllClose(c1_to_global.data[0, 0], c1_areasum)

    def test_single_cells(self):
        # Test handling of single-cell grids.
        # Fetch 'standard' testcube data
        c1, c2 = self.stock_c1_c2
        c1_areasum = self.stock_c1_areasum

        #
        # At present NxN -> 1x1 "in-place" doesn't seem to work properly
        # - result cell has missing-data ?
        #
        # Condense entire region into a single cell in the c1 grid
        xlims1 = _minmax(c1.coord(axis="x").bounds)
        ylims1 = _minmax(c1.coord(axis="y").bounds)
        x_c1x1 = iris.coords.DimCoord(
            xlims1[0],
            bounds=xlims1,
            standard_name="longitude",
            units="degrees",
            coord_system=_PLAIN_GEODETIC_CS,
        )
        y_c1x1 = iris.coords.DimCoord(
            ylims1[0],
            bounds=ylims1,
            standard_name="latitude",
            units="degrees",
            coord_system=_PLAIN_GEODETIC_CS,
        )
        c1x1_gridcube = iris.cube.Cube([[0.0]])
        c1x1_gridcube.add_dim_coord(y_c1x1, 0)
        c1x1_gridcube.add_dim_coord(x_c1x1, 1)
        c1x1 = regrid_conservative_via_esmpy(c1, c1x1_gridcube)
        c1x1_areasum = _cube_area_sum(c1x1)
        # Check the total area sum is still the same
        condense_to_1x1_supported = False
        # NOTE: currently disabled (ESMF gets this wrong)
        # NOTE ALSO: call hits numpy 1.7 bug in testing.assert_array_compare.
        if condense_to_1x1_supported:
            self.assertArrayAllClose(c1x1_areasum, c1_areasum)

        # Condense entire region onto a single cell covering the area of 'c2'
        xlims2 = _minmax(c2.coord(axis="x").bounds)
        ylims2 = _minmax(c2.coord(axis="y").bounds)
        x_c2x1 = iris.coords.DimCoord(
            xlims2[0],
            bounds=xlims2,
            standard_name="longitude",
            units=cf_units.Unit("degrees"),
            coord_system=_PLAIN_GEODETIC_CS,
        )
        y_c2x1 = iris.coords.DimCoord(
            ylims2[0],
            bounds=ylims2,
            standard_name="latitude",
            units=cf_units.Unit("degrees"),
            coord_system=_PLAIN_GEODETIC_CS,
        )
        c2x1_gridcube = iris.cube.Cube([[0.0]])
        c2x1_gridcube.add_dim_coord(y_c2x1, 0)
        c2x1_gridcube.add_dim_coord(x_c2x1, 1)
        c1_to_c2x1 = regrid_conservative_via_esmpy(c1, c2x1_gridcube)

        # Check the total area sum is still the same
        c1_to_c2x1_areasum = _cube_area_sum(c1_to_c2x1)
        self.assertArrayAllClose(c1_to_c2x1_areasum, c1_areasum, 0.0004)

        # 1x1 -> NxN : regrid single cell to NxN grid
        # construct a single-cell approximation to 'c1' with the same area sum.
        # NOTE: can't use _make_cube (see docstring)
        c1x1 = c1.copy()[0:1, 0:1]
        xlims1 = _minmax(c1.coord(axis="x").bounds)
        ylims1 = _minmax(c1.coord(axis="y").bounds)
        c1x1.coord(axis="x").bounds = xlims1
        c1x1.coord(axis="y").bounds = ylims1
        # Assign data mean as single cell value : Maybe not exact, but "close"
        c1x1.data[0, 0] = np.mean(c1.data)

        # Regrid this back onto the original NxN grid
        c1x1_to_c1 = regrid_conservative_via_esmpy(c1x1, c1)
        c1x1_to_c1_areasum = _cube_area_sum(c1x1_to_c1)

        # Check that area sum is ~unchanged, as expected
        self.assertArrayAllClose(c1x1_to_c1_areasum, c1_areasum, 0.0004)

        # Check 1x1 -> 1x1
        # NOTE: can *only* get any result with a fully overlapping cell, so
        # just regrid onto self
        c1x1toself = regrid_conservative_via_esmpy(c1x1, c1x1)
        c1x1toself_areasum = _cube_area_sum(c1x1toself)
        self.assertArrayAllClose(c1x1toself_areasum, c1_areasum, 0.0004)
        # NOTE: perhaps surprisingly, this has a similar level of error.

    def test_longitude_wraps(self):
        """Check results are independent of where the grid 'seams' are."""
        # First repeat global regrid calculation from 'test_global'.
        shape1 = (8, 6)
        xlim1 = 180.0 * (shape1[0] - 1) / shape1[0]
        ylim1 = 90.0 * (shape1[1] - 1) / shape1[1]
        xlims1 = (-xlim1, xlim1)
        ylims1 = (-ylim1, ylim1)
        c1 = _make_test_cube(shape1, xlims1, ylims1)

        # Create a small, plausible global array (see test_global).
        basedata = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 4, 4, 4, 2, 2, 1],
                [2, 1, 4, 4, 4, 2, 2, 2],
                [2, 5, 5, 1, 1, 1, 5, 5],
                [5, 5, 5, 1, 1, 1, 5, 5],
                [5, 5, 5, 5, 5, 5, 5, 5],
            ]
        )
        c1.data[:] = basedata

        shape2 = (14, 11)
        xlim2 = 180.0 * (shape2[0] - 1) / shape2[0]
        ylim2 = 90.0 * (shape2[1] - 1) / shape2[1]
        xlims_2 = (-xlim2, xlim2)
        ylims_2 = (-ylim2, ylim2)
        c2 = _make_test_cube(shape2, xlims_2, ylims_2, pole_latlon=(47.4, 25.7))

        # Perform regridding
        c1toc2 = regrid_conservative_via_esmpy(c1, c2)

        # Now redo with dst longitudes rotated, so 'seam' is somewhere else.
        x2_shift_steps = shape2[0] // 3
        xlims2_shifted = np.array(xlims_2) + 360.0 * x2_shift_steps / shape2[0]
        c2_shifted = _make_test_cube(
            shape2, xlims2_shifted, ylims_2, pole_latlon=(47.4, 25.7)
        )
        c1toc2_shifted = regrid_conservative_via_esmpy(c1, c2_shifted)

        # Show that results are the same, when output rolled by same amount
        rolled_data = np.roll(c1toc2_shifted.data, x2_shift_steps, axis=1)
        self.assertArrayAllClose(rolled_data, c1toc2.data)

        # Repeat with rolled *source* data : result should be identical
        x1_shift_steps = shape1[0] // 3
        x_shift_degrees = 360.0 * x1_shift_steps / shape1[0]
        xlims1_shifted = [x - x_shift_degrees for x in xlims1]
        c1_shifted = _make_test_cube(shape1, xlims1_shifted, ylims1)
        c1_shifted.data[:] = np.roll(basedata, x1_shift_steps, axis=1)
        c1shifted_toc2 = regrid_conservative_via_esmpy(c1_shifted, c2)
        self.assertEqual(c1shifted_toc2, c1toc2)

    def test_polar_areas(self):
        """Test area-conserving regrid between different grids.

        Grids have overlapping areas in the same (lat-lon) coordinate system.
        Cells are highly non-square (near the pole).

        """
        # Like test_basic_area, but not symmetrical + bigger overall errors.
        shape1 = (5, 5)
        xlims1, ylims1 = ((-2, 2), (84, 88))
        c1 = _make_test_cube(shape1, xlims1, ylims1)
        c1.data[:] = 0.0
        c1.data[2, 2] = 1.0
        c1_areasum = _cube_area_sum(c1)

        shape2 = (4, 4)
        xlims2, ylims2 = ((-1.5, 1.5), (84.5, 87.5))
        c2 = _make_test_cube(shape2, xlims2, ylims2)
        c2.data[:] = 0.0

        c1to2 = regrid_conservative_via_esmpy(c1, c2)

        # check for expected pattern
        d_expect = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.23614, 0.23614, 0.0],
                [0.0, 0.26784, 0.26784, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        self.assertArrayAllClose(c1to2.data, d_expect, rtol=5.0e-5)

        # check sums
        c1to2_areasum = _cube_area_sum(c1to2)
        self.assertArrayAllClose(c1to2_areasum, c1_areasum)

        #
        # transform back again ...
        #
        c1to2to1 = regrid_conservative_via_esmpy(c1to2, c1)

        # check values
        d_expect = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.056091, 0.112181, 0.056091, 0.0],
                [0.0, 0.125499, 0.250998, 0.125499, 0.0],
                [0.0, 0.072534, 0.145067, 0.072534, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        self.assertArrayAllClose(c1to2to1.data, d_expect, atol=0.0005)

        # check sums
        c1to2to1_areasum = _cube_area_sum(c1to2to1)
        self.assertArrayAllClose(c1to2to1_areasum, c1_areasum)

    def test_fail_no_cs(self):
        # Test error when one coordinate has no coord_system.
        shape1 = (5, 5)
        xlims1, ylims1 = ((-2, 2), (-2, 2))
        c1 = _make_test_cube(shape1, xlims1, ylims1)
        c1.data[:] = 0.0
        c1.data[2, 2] = 1.0

        shape2 = (4, 4)
        xlims2, ylims2 = ((-1.5, 1.5), (-1.5, 1.5))
        c2 = _make_test_cube(shape2, xlims2, ylims2)
        c2.data[:] = 0.0
        c2.coord("latitude").coord_system = None

        with self.assertRaises(ValueError):
            regrid_conservative_via_esmpy(c1, c2)

    def test_fail_different_cs(self):
        # Test error when either src or dst coords have different
        # coord_systems.
        shape1 = (5, 5)
        xlims1, ylims1 = ((-2, 2), (-2, 2))
        shape2 = (4, 4)
        xlims2, ylims2 = ((-1.5, 1.5), (-1.5, 1.5))

        # Check basic regrid between these is ok.
        c1 = _make_test_cube(shape1, xlims1, ylims1, pole_latlon=(45.0, 35.0))
        c2 = _make_test_cube(shape2, xlims2, ylims2)
        regrid_conservative_via_esmpy(c1, c2)

        # Replace the coord_system one of the source coords + check this fails.
        c1.coord("grid_longitude").coord_system = c2.coord("longitude").coord_system
        with self.assertRaises(ValueError):
            regrid_conservative_via_esmpy(c1, c2)

        # Repeat with target coordinate fiddled.
        c1 = _make_test_cube(shape1, xlims1, ylims1, pole_latlon=(45.0, 35.0))
        c2 = _make_test_cube(shape2, xlims2, ylims2)
        c2.coord("latitude").coord_system = c1.coord("grid_latitude").coord_system
        with self.assertRaises(ValueError):
            regrid_conservative_via_esmpy(c1, c2)

    def test_rotated(self):
        """Test area-weighted regrid on more complex area.

        Use two mutually rotated grids, of similar area + same dims.
        Only a small central region in each is non-zero, which maps entirely
        inside the other region.
        So the area-sum totals should match exactly.

        """
        # create source test cube on rotated form
        pole_lat = 53.4
        pole_lon = -173.2
        deg_swing = 35.3
        pole_lon += deg_swing
        c1_nx = 9 + 6
        c1_ny = 7 + 6
        c1_xlims = -60.0, 60.0
        c1_ylims = -45.0, 20.0
        c1_xlims = [x - deg_swing for x in c1_xlims]
        c1 = _make_test_cube(
            (c1_nx, c1_ny),
            c1_xlims,
            c1_ylims,
            pole_latlon=(pole_lat, pole_lon),
        )
        c1.data[3:-3, 3:-3] = np.array(
            [
                [100, 100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 199, 199, 199, 199, 100, 100, 100],
                [100, 100, 100, 100, 199, 199, 100, 100, 100],
                [100, 100, 100, 100, 199, 199, 199, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100, 100],
            ],
            dtype=np.float64,
        )

        c1_areasum = _cube_area_sum(c1)

        # construct target cube to receive
        nx2 = 9 + 6
        ny2 = 7 + 6
        c2_xlims = -100.0, 120.0
        c2_ylims = -20.0, 50.0
        c2 = _make_test_cube((nx2, ny2), c2_xlims, c2_ylims)
        c2.data = np.ma.array(c2.data, mask=True)

        # perform regrid
        c1to2 = regrid_conservative_via_esmpy(c1, c2)

        # check we have zeros (or nearly) all around the edge..
        c1toc2_zeros = np.ma.array(c1to2.data)
        c1toc2_zeros[c1toc2_zeros.mask] = 0.0
        c1toc2_zeros = np.abs(c1toc2_zeros.mask) < 1.0e-6
        self.assertArrayEqual(c1toc2_zeros[0, :], True)
        self.assertArrayEqual(c1toc2_zeros[-1, :], True)
        self.assertArrayEqual(c1toc2_zeros[:, 0], True)
        self.assertArrayEqual(c1toc2_zeros[:, -1], True)

        # check the area-sum operation
        c1to2_areasum = _cube_area_sum(c1to2)
        self.assertArrayAllClose(c1to2_areasum, c1_areasum, rtol=0.004)

        #
        # Now repeat, transforming backwards ...
        #
        c1.data = np.ma.array(c1.data, mask=True)
        c2.data[:] = 0.0
        c2.data[5:-5, 5:-5] = np.array(
            [
                [199, 199, 199, 199, 100],
                [100, 100, 199, 199, 100],
                [100, 100, 199, 199, 199],
            ],
            dtype=np.float64,
        )
        c2_areasum = _cube_area_sum(c2)

        c2toc1 = regrid_conservative_via_esmpy(c2, c1)

        # check we have zeros (or nearly) all around the edge..
        c2toc1_zeros = np.ma.array(c2toc1.data)
        c2toc1_zeros[c2toc1_zeros.mask] = 0.0
        c2toc1_zeros = np.abs(c2toc1_zeros.mask) < 1.0e-6
        self.assertArrayEqual(c2toc1_zeros[0, :], True)
        self.assertArrayEqual(c2toc1_zeros[-1, :], True)
        self.assertArrayEqual(c2toc1_zeros[:, 0], True)
        self.assertArrayEqual(c2toc1_zeros[:, -1], True)

        # check the area-sum operation
        c2toc1_areasum = _cube_area_sum(c2toc1)
        self.assertArrayAllClose(c2toc1_areasum, c2_areasum, rtol=0.004)

    def test_missing_data_rotated(self):
        """Check missing-data handling between different coordinate systems.

        Regrid between mutually rotated lat/lon systems, and check results for
        missing data due to grid edge overlap, and source-data masking.

        """
        for do_add_missing in (False, True):
            # create source test cube on rotated form
            pole_lat = 53.4
            pole_lon = -173.2
            deg_swing = 35.3
            pole_lon += deg_swing
            c1_nx = 9 + 6
            c1_ny = 7 + 6
            c1_xlims = -60.0, 60.0
            c1_ylims = -45.0, 20.0
            c1_xlims = [x - deg_swing for x in c1_xlims]
            c1 = _make_test_cube(
                (c1_nx, c1_ny),
                c1_xlims,
                c1_ylims,
                pole_latlon=(pole_lat, pole_lon),
            )
            c1.data = np.ma.array(c1.data, mask=False)
            c1.data[3:-3, 3:-3] = np.ma.array(
                [
                    [100, 100, 100, 100, 100, 100, 100, 100, 100],
                    [100, 100, 100, 100, 100, 100, 100, 100, 100],
                    [100, 100, 199, 199, 199, 199, 100, 100, 100],
                    [100, 100, 100, 100, 199, 199, 100, 100, 100],
                    [100, 100, 100, 100, 199, 199, 199, 100, 100],
                    [100, 100, 100, 100, 100, 100, 100, 100, 100],
                    [100, 100, 100, 100, 100, 100, 100, 100, 100],
                ],
                dtype=np.float64,
            )

            if do_add_missing:
                c1.data = np.ma.array(c1.data)
                c1.data[7, 7] = np.ma.masked
                c1.data[3:5, 10:12] = np.ma.masked

            # construct target cube to receive
            nx2 = 9 + 6
            ny2 = 7 + 6
            c2_xlims = -80.0, 80.0
            c2_ylims = -20.0, 50.0
            c2 = _make_test_cube((nx2, ny2), c2_xlims, c2_ylims)
            c2.data = np.ma.array(c2.data, mask=True)

            # perform regrid + snapshot test results
            c1toc2 = regrid_conservative_via_esmpy(c1, c2)

            # check masking of result is as expected
            # (generated by inspecting plot of how src+dst grids overlap)
            expected_mask_valuemap = np.array(
                # KEY: 0=masked, 7=present, 5=masked with masked datapoints
                [
                    [0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 7, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0],
                    [0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0],
                    [0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 5, 5, 7, 0, 0],
                    [0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 7, 0, 0],
                    [0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 5, 5, 7, 0, 0],
                    [0, 0, 0, 7, 7, 7, 7, 5, 5, 7, 7, 7, 7, 0, 0],
                    [0, 0, 0, 0, 7, 7, 7, 5, 5, 7, 7, 7, 7, 0, 0],
                    [0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0],
                    [0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0],
                    [0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 0],
                ]
            )

            if do_add_missing:
                expected_mask = expected_mask_valuemap < 7
            else:
                expected_mask = expected_mask_valuemap == 0

            actual_mask = c1toc2.data.mask
            self.assertArrayEqual(actual_mask, expected_mask)

            if not do_add_missing:
                # check preservation of area-sums
                # NOTE: does *not* work with missing data, even theoretically,
                # as the 'missing areas' are not the same.
                c1_areasum = _cube_area_sum(c1)
                c1to2_areasum = _cube_area_sum(c1toc2)
                self.assertArrayAllClose(c1_areasum, c1to2_areasum, rtol=0.003)


if __name__ == "__main__":
    tests.main()
