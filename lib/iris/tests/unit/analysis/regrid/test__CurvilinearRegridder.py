# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for :class:`iris.analysis._regrid.CurvilinearRegridder`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

import numpy as np

from iris.analysis._regrid import CurvilinearRegridder as Regridder
from iris.analysis.cartography import rotate_pole
from iris.aux_factory import HybridHeightFactory
from iris.coord_systems import GeogCS, RotatedGeogCS
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.fileformats.pp import EARTH_RADIUS
from iris.tests.stock import global_pp, lat_lon_cube, realistic_4d

RESULT_DIR = ("analysis", "regrid")


class Test___init__(tests.IrisTest):
    def setUp(self):
        self.src_grid = lat_lon_cube()
        self.bad = np.ones((3, 4))
        self.weights = np.ones(self.src_grid.shape, self.src_grid.dtype)

    def test_bad_src_type(self):
        with self.assertRaisesRegex(TypeError, "'src_grid_cube'"):
            Regridder(self.bad, self.src_grid, self.weights)

    def test_bad_grid_type(self):
        with self.assertRaisesRegex(TypeError, "'target_grid_cube'"):
            Regridder(self.src_grid, self.bad, self.weights)


@tests.skip_data
class Test___call__(tests.IrisTest):
    def setUp(self):
        self.func_setup = (
            "iris.analysis._regrid."
            "_regrid_weighted_curvilinear_to_rectilinear__prepare"
        )
        self.func_operate = (
            "iris.analysis._regrid."
            "_regrid_weighted_curvilinear_to_rectilinear__perform"
        )
        # Define a test source grid and target grid, basically the same.
        self.src_grid = global_pp()
        self.tgt_grid = global_pp()
        # Modify the names so we can tell them apart.
        self.src_grid.rename("src_grid")
        self.tgt_grid.rename("TARGET_GRID")
        # Replace the source-grid x and y coords with equivalent 2d versions.
        x_coord = self.src_grid.coord("longitude")
        y_coord = self.src_grid.coord("latitude")
        (nx,) = x_coord.shape
        (ny,) = y_coord.shape
        xx, yy = np.meshgrid(x_coord.points, y_coord.points)
        self.src_grid.remove_coord(x_coord)
        self.src_grid.remove_coord(y_coord)
        x_coord_2d = AuxCoord(
            xx,
            standard_name=x_coord.standard_name,
            units=x_coord.units,
            coord_system=x_coord.coord_system,
        )
        y_coord_2d = AuxCoord(
            yy,
            standard_name=y_coord.standard_name,
            units=y_coord.units,
            coord_system=y_coord.coord_system,
        )
        self.src_grid.add_aux_coord(x_coord_2d, (0, 1))
        self.src_grid.add_aux_coord(y_coord_2d, (0, 1))
        self.weights = np.ones(self.src_grid.shape, self.src_grid.dtype)
        # Define an actual, dummy cube for the internal partial result, so we
        # can do a cubelist merge on it, which is too complicated to mock out.
        self.dummy_slice_result = Cube([1])

    def test_same_src_as_init(self):
        # Check the regridder call calls the underlying routines as expected.
        src_grid = self.src_grid
        target_grid = self.tgt_grid
        regridder = Regridder(src_grid, target_grid, self.weights)
        with mock.patch(
            self.func_setup, return_value=mock.sentinel.regrid_info
        ) as patch_setup:
            with mock.patch(
                self.func_operate, return_value=self.dummy_slice_result
            ) as patch_operate:
                result = regridder(src_grid)
        patch_setup.assert_called_once_with(
            src_grid, self.weights, target_grid
        )
        patch_operate.assert_called_once_with(
            src_grid, mock.sentinel.regrid_info
        )
        # The result is a re-merged version of the internal result, so it is
        # therefore '==' but not the same object.
        self.assertEqual(result, self.dummy_slice_result)

    def test_no_weights(self):
        # Check we can use the regridder without weights.
        src_grid = self.src_grid
        target_grid = self.tgt_grid
        regridder = Regridder(src_grid, target_grid)
        with mock.patch(
            self.func_setup, return_value=mock.sentinel.regrid_info
        ) as patch_setup:
            with mock.patch(
                self.func_operate, return_value=self.dummy_slice_result
            ):
                _ = regridder(src_grid)
        patch_setup.assert_called_once_with(src_grid, None, target_grid)

    def test_diff_src_from_init(self):
        # Check we can call the regridder with a different cube from the one we
        # built it with.
        src_grid = self.src_grid
        target_grid = self.tgt_grid
        regridder = Regridder(src_grid, target_grid, self.weights)
        # Provide a "different" cube for the actual regrid.
        different_src_cube = self.src_grid.copy()
        # Rename so we can distinguish them.
        different_src_cube.rename("Different_source")
        with mock.patch(
            self.func_setup, return_value=mock.sentinel.regrid_info
        ):
            with mock.patch(
                self.func_operate, return_value=self.dummy_slice_result
            ) as patch_operate:
                _ = regridder(different_src_cube)
        patch_operate.assert_called_once_with(
            different_src_cube, mock.sentinel.regrid_info
        )

    def test_caching(self):
        # Check that it calculates regrid info just once, and re-uses it in
        # subsequent calls.
        src_grid = self.src_grid
        target_grid = self.tgt_grid
        regridder = Regridder(src_grid, target_grid, self.weights)
        different_src_cube = self.src_grid.copy()
        different_src_cube.rename("Different_source")
        with mock.patch(
            self.func_setup, return_value=mock.sentinel.regrid_info
        ) as patch_setup:
            with mock.patch(
                self.func_operate, return_value=self.dummy_slice_result
            ) as patch_operate:
                _ = regridder(src_grid)
                _ = regridder(different_src_cube)
        patch_setup.assert_called_once_with(
            src_grid, self.weights, target_grid
        )
        self.assertEqual(len(patch_operate.call_args_list), 2)
        self.assertEqual(
            patch_operate.call_args_list,
            [
                mock.call(src_grid, mock.sentinel.regrid_info),
                mock.call(different_src_cube, mock.sentinel.regrid_info),
            ],
        )


class Test__derived_coord(tests.IrisTest):
    def setUp(self):
        src = realistic_4d()[0]
        tgt = realistic_4d()
        new_lon, new_lat = np.meshgrid(
            src.coord("grid_longitude").points,
            src.coord("grid_latitude").points,
        )
        coord_system = src.coord("grid_latitude").coord_system
        lat = AuxCoord(
            new_lat, standard_name="latitude", coord_system=coord_system
        )
        lon = AuxCoord(
            new_lon, standard_name="longitude", coord_system=coord_system
        )
        lat_t = AuxCoord(
            new_lat.T, standard_name="latitude", coord_system=coord_system
        )
        lon_t = AuxCoord(
            new_lon.T, standard_name="longitude", coord_system=coord_system
        )

        src.remove_coord("grid_latitude")
        src.remove_coord("grid_longitude")
        src_t = src.copy()
        src.add_aux_coord(lat, [1, 2])
        src.add_aux_coord(lon, [1, 2])
        src_t.add_aux_coord(lat_t, [2, 1])
        src_t.add_aux_coord(lon_t, [2, 1])
        self.src = src.copy()
        self.src_t = src_t
        self.tgt = tgt
        self.altitude = src.coord("altitude")
        transposed_src = src.copy()
        transposed_src.transpose([0, 2, 1])
        self.altitude_transposed = transposed_src.coord("altitude")

    def test_no_transpose(self):
        rg = Regridder(self.src, self.tgt)
        res = rg(self.src)

        assert len(res.aux_factories) == 1 and isinstance(
            res.aux_factories[0], HybridHeightFactory
        )
        assert np.allclose(res.coord("altitude").points, self.altitude.points)

    def test_cube_transposed(self):
        rg = Regridder(self.src, self.tgt)
        transposed_cube = self.src.copy()
        transposed_cube.transpose([0, 2, 1])
        res = rg(transposed_cube)

        assert len(res.aux_factories) == 1 and isinstance(
            res.aux_factories[0], HybridHeightFactory
        )
        assert np.allclose(
            res.coord("altitude").points, self.altitude_transposed.points
        )

    def test_coord_transposed(self):
        rg = Regridder(self.src_t, self.tgt)
        res = rg(self.src_t)

        assert len(res.aux_factories) == 1 and isinstance(
            res.aux_factories[0], HybridHeightFactory
        )
        assert np.allclose(
            res.coord("altitude").points, self.altitude_transposed.points
        )

    def test_both_transposed(self):
        rg = Regridder(self.src_t, self.tgt)
        transposed_cube = self.src_t.copy()
        transposed_cube.transpose([0, 2, 1])
        res = rg(transposed_cube)

        assert len(res.aux_factories) == 1 and isinstance(
            res.aux_factories[0], HybridHeightFactory
        )
        assert np.allclose(res.coord("altitude").points, self.altitude.points)


@tests.skip_data
class Test___call____bad_src(tests.IrisTest):
    def setUp(self):
        self.src_grid = global_pp()
        y = self.src_grid.coord("latitude")
        x = self.src_grid.coord("longitude")
        self.src_grid.remove_coord("latitude")
        self.src_grid.remove_coord("longitude")
        self.src_grid.add_aux_coord(y, 0)
        self.src_grid.add_aux_coord(x, 1)
        weights = np.ones(self.src_grid.shape, self.src_grid.dtype)
        self.regridder = Regridder(self.src_grid, self.src_grid, weights)

    def test_bad_src_type(self):
        with self.assertRaisesRegex(TypeError, "must be a Cube"):
            self.regridder(np.ones((3, 4)))

    def test_bad_src_shape(self):
        with self.assertRaisesRegex(
            ValueError, "not defined on the same source grid"
        ):
            self.regridder(self.src_grid[::2, ::2])


class Test__call__multidimensional(tests.IrisTest):
    def test_multidim(self):
        # Testing with >2D data to demonstrate correct operation over
        # additional non-XY dimensions (including data masking), which is
        # handled by the PointInCell wrapper class.

        # Define a simple target grid first, in plain latlon coordinates.
        plain_latlon_cs = GeogCS(EARTH_RADIUS)
        grid_x_coord = DimCoord(
            points=[15.0, 25.0, 35.0],
            bounds=[[10.0, 20.0], [20.0, 30.0], [30.0, 40.0]],
            standard_name="longitude",
            units="degrees",
            coord_system=plain_latlon_cs,
        )
        grid_y_coord = DimCoord(
            points=[-30.0, -50.0],
            bounds=[[-20.0, -40.0], [-40.0, -60.0]],
            standard_name="latitude",
            units="degrees",
            coord_system=plain_latlon_cs,
        )
        grid_cube = Cube(np.zeros((2, 3)))
        grid_cube.add_dim_coord(grid_y_coord, 0)
        grid_cube.add_dim_coord(grid_x_coord, 1)

        # Define some key points in true-lat/lon that have known positions
        # First 3x2 points in the centre of each output cell.
        x_centres, y_centres = np.meshgrid(
            grid_x_coord.points, grid_y_coord.points
        )
        # An extra point also falling in cell 1, 1
        x_in11, y_in11 = 26.3, -48.2
        # An extra point completely outside the target grid
        x_out, y_out = 70.0, -40.0

        # Define a rotated coord system for the source data
        pole_lon, pole_lat = -125.3, 53.4
        src_cs = RotatedGeogCS(
            grid_north_pole_latitude=pole_lat,
            grid_north_pole_longitude=pole_lon,
            ellipsoid=plain_latlon_cs,
        )

        # Concatenate all the testpoints in a flat array, and find the rotated
        # equivalents.
        xx = list(x_centres.flat[:]) + [x_in11, x_out]
        yy = list(y_centres.flat[:]) + [y_in11, y_out]
        xx, yy = rotate_pole(
            lons=np.array(xx),
            lats=np.array(yy),
            pole_lon=pole_lon,
            pole_lat=pole_lat,
        )
        # Define handy index numbers for all these.
        i00, i01, i02, i10, i11, i12, i_in, i_out = range(8)

        # Build test data in the shape Z,YX = (3, 8)
        data = [
            [1, 2, 3, 11, 12, 13, 7, 99],
            [1, 2, 3, 11, 12, 13, 7, 99],
            [7, 6, 5, 51, 52, 53, 12, 1],
        ]
        mask = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
        src_data = np.ma.array(data, mask=mask, dtype=float)

        # Make the source cube.
        src_cube = Cube(src_data)
        src_x = AuxCoord(
            xx,
            standard_name="grid_longitude",
            units="degrees",
            coord_system=src_cs,
        )
        src_y = AuxCoord(
            yy,
            standard_name="grid_latitude",
            units="degrees",
            coord_system=src_cs,
        )
        src_z = DimCoord(np.arange(3), long_name="z")
        src_cube.add_dim_coord(src_z, 0)
        src_cube.add_aux_coord(src_x, 1)
        src_cube.add_aux_coord(src_y, 1)
        # Add in some extra metadata, to ensure it gets copied over.
        src_cube.add_aux_coord(DimCoord([0], long_name="extra_scalar_coord"))
        src_cube.attributes["extra_attr"] = 12.3

        # Define what the expected answers should be, shaped (3, 2, 3).
        expected_result = [
            [[1.0, 2.0, 3.0], [11.0, 0.5 * (12 + 7), 13.0]],
            [[1.0, -999, 3.0], [11.0, 12.0, 13.0]],
            [[7.0, 6.0, 5.0], [51.0, 0.5 * (52 + 12), 53.0]],
        ]
        expected_result = np.ma.masked_less(expected_result, 0)

        # Perform the calculation with the regridder.
        regridder = Regridder(src_cube, grid_cube)

        # Check all is as expected.
        result = regridder(src_cube)
        self.assertEqual(result.coord("z"), src_cube.coord("z"))
        self.assertEqual(
            result.coord("extra_scalar_coord"),
            src_cube.coord("extra_scalar_coord"),
        )
        self.assertEqual(
            result.coord("longitude"), grid_cube.coord("longitude")
        )
        self.assertEqual(result.coord("latitude"), grid_cube.coord("latitude"))
        self.assertMaskedArrayAlmostEqual(result.data, expected_result)


if __name__ == "__main__":
    tests.main()
