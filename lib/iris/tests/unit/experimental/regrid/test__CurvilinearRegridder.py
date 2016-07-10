# (C) British Crown Copyright 2015 - 2016, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""Unit tests for :class:`iris.experimental.regrid._CurvilinearRegridder`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.cube import Cube
from iris.coords import AuxCoord
from iris.experimental.regrid import _CurvilinearRegridder as Regridder
from iris.tests import mock
from iris.tests.stock import global_pp, lat_lon_cube


RESULT_DIR = ('analysis', 'regrid')


class Test___init__(tests.IrisTest):
    def setUp(self):
        self.src_grid = lat_lon_cube()
        self.bad = np.ones((3, 4))
        self.weights = np.ones(self.src_grid.shape, self.src_grid.dtype)

    def test_bad_src_type(self):
        with self.assertRaisesRegexp(TypeError, "'src_grid_cube'"):
            Regridder(self.bad, self.src_grid, self.weights)

    def test_bad_grid_type(self):
        with self.assertRaisesRegexp(TypeError, "'target_grid_cube'"):
            Regridder(self.src_grid, self.bad, self.weights)


@tests.skip_data
class Test___call__(tests.IrisTest):
    def setUp(self):
        self.func_setup = (
            'iris.experimental.regrid.'
            '_regrid_weighted_curvilinear_to_rectilinear__prepare')
        self.func_operate = (
            'iris.experimental.regrid.'
            '_regrid_weighted_curvilinear_to_rectilinear__perform')
        # Define a test source grid and target grid, basically the same.
        self.src_grid = global_pp()
        self.tgt_grid = global_pp()
        # Modify the names so we can tell them apart.
        self.src_grid.rename('src_grid')
        self.tgt_grid.rename('TARGET_GRID')
        # Replace the source-grid x and y coords with equivalent 2d versions.
        x_coord = self.src_grid.coord('longitude')
        y_coord = self.src_grid.coord('latitude')
        nx, = x_coord.shape
        ny, = y_coord.shape
        xx, yy = np.meshgrid(x_coord.points, y_coord.points)
        self.src_grid.remove_coord(x_coord)
        self.src_grid.remove_coord(y_coord)
        x_coord_2d = AuxCoord(xx,
                              standard_name=x_coord.standard_name,
                              units=x_coord.units,
                              coord_system=x_coord.coord_system)
        y_coord_2d = AuxCoord(yy,
                              standard_name=y_coord.standard_name,
                              units=y_coord.units,
                              coord_system=y_coord.coord_system)
        self.src_grid.add_aux_coord(x_coord_2d, (0, 1))
        self.src_grid.add_aux_coord(y_coord_2d, (0, 1))
        self.weights = np.ones(self.src_grid.shape, self.src_grid.dtype)
        # Define an actual, dummy cube for the internal partial result, so we
        # can do a cubelist merge on it, which is too complicated to mock out.
        self.mock_slice_result = Cube([1])

    def test_same_src_as_init(self):
        # Check the regridder call calls the underlying routines as expected.
        src_grid = self.src_grid
        target_grid = self.tgt_grid
        regridder = Regridder(src_grid, target_grid, self.weights)
        with mock.patch(self.func_setup,
                        return_value=mock.sentinel.regrid_info) as patch_setup:
            with mock.patch(
                    self.func_operate,
                    return_value=self.mock_slice_result) as patch_operate:
                result = regridder(src_grid)
        patch_setup.assert_called_once_with(
            src_grid, self.weights, target_grid)
        patch_operate.assert_called_once_with(
            src_grid, mock.sentinel.regrid_info)
        # The result is a re-merged version of the internal result, so it is
        # therefore '==' but not the same object.
        self.assertEqual(result, self.mock_slice_result)

    def test_no_weights(self):
        # Check we can use the regridder without weights.
        src_grid = self.src_grid
        target_grid = self.tgt_grid
        regridder = Regridder(src_grid, target_grid)
        # Note: for now, patch the internal partial result to a "real" cube,
        # so we can do a cubelist merge, which is too complicated to mock out.
        mock_slice_result = Cube([1])
        with mock.patch(self.func_setup,
                        return_value=mock.sentinel.regrid_info) as patch_setup:
            with mock.patch(
                    self.func_operate,
                    return_value=self.mock_slice_result) as patch_operate:
                result = regridder(src_grid)
        patch_setup.assert_called_once_with(
            src_grid, None, target_grid)

    def test_diff_src_from_init(self):
        # Check we can call the regridder with a different cube from the one we
        # built it with.
        src_grid = self.src_grid
        target_grid = self.tgt_grid
        regridder = Regridder(src_grid, target_grid, self.weights)
        # Note: for now, patch the internal partial result to a "real" cube,
        # so we can do a cubelist merge, which is too complicated to mock out.
        mock_slice_result = Cube([1])
        # Provide a "different" cube for the actual regrid.
        different_src_cube = self.src_grid.copy()
        # Rename so we can distinguish them.
        different_src_cube.rename('Different_source')
        with mock.patch(self.func_setup,
                        return_value=mock.sentinel.regrid_info) as patch_setup:
            with mock.patch(
                    self.func_operate,
                    return_value=self.mock_slice_result) as patch_operate:
                result = regridder(different_src_cube)
        patch_operate.assert_called_once_with(
            different_src_cube, mock.sentinel.regrid_info)

    def test_caching(self):
        # Check that it calculates regrid info just once, and re-uses it in
        # subsequent calls.
        src_grid = self.src_grid
        target_grid = self.tgt_grid
        regridder = Regridder(src_grid, target_grid, self.weights)
        mock_slice_result = Cube([1])
        different_src_cube = self.src_grid.copy()
        different_src_cube.rename('Different_source')
        with mock.patch(self.func_setup,
                        return_value=mock.sentinel.regrid_info) as patch_setup:
            with mock.patch(
                    self.func_operate,
                    return_value=self.mock_slice_result) as patch_operate:
                result1 = regridder(src_grid)
                result2 = regridder(different_src_cube)
        patch_setup.assert_called_once_with(
            src_grid, self.weights, target_grid)
        self.assertEqual(len(patch_operate.call_args_list), 2)
        self.assertEqual(
            patch_operate.call_args_list,
            [mock.call(src_grid, mock.sentinel.regrid_info),
             mock.call(different_src_cube, mock.sentinel.regrid_info)])


@tests.skip_data
class Test___call____bad_src(tests.IrisTest):
    def setUp(self):
        self.src_grid = global_pp()
        y = self.src_grid.coord('latitude')
        x = self.src_grid.coord('longitude')
        self.src_grid.remove_coord('latitude')
        self.src_grid.remove_coord('longitude')
        self.src_grid.add_aux_coord(y, 0)
        self.src_grid.add_aux_coord(x, 1)
        weights = np.ones(self.src_grid.shape, self.src_grid.dtype)
        self.regridder = Regridder(self.src_grid, self.src_grid, weights)

    def test_bad_src_type(self):
        with self.assertRaisesRegexp(TypeError, 'must be a Cube'):
            self.regridder(np.ones((3, 4)))

    def test_bad_src_shape(self):
        with self.assertRaisesRegexp(ValueError,
                                     'not defined on the same source grid'):
            self.regridder(self.src_grid[::2, ::2])


if __name__ == '__main__':
    tests.main()
