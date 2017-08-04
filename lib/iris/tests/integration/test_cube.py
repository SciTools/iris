# (C) British Crown Copyright 2014 - 2017, Met Office
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
"""Integration tests for :class:`iris.cube.Cube`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np
import numpy.ma as ma

import iris
from iris.analysis import MEAN
from iris.cube import Cube

from iris._lazy_data import as_lazy_data, is_lazy_data


class Test_aggregated_by(tests.IrisTest):
    @tests.skip_data
    def test_agg_by_aux_coord(self):
        problem_test_file = tests.get_data_path(('NetCDF', 'testing',
                                                'small_theta_colpex.nc'))
        cube = iris.load_cube(problem_test_file)

        # Test aggregating by aux coord, notably the `forecast_period` aux
        # coord on `cube`, whose `_points` attribute is a lazy array.
        # This test then ensures that aggregating using `points` instead is
        # successful.

        # First confirm we've got a lazy array.
        # NB. This checks the merge process in `load_cube()` hasn't
        # triggered the load of the coordinate's data.
        forecast_period_coord = cube.coord('forecast_period')

        self.assertTrue(is_lazy_data(forecast_period_coord.core_points()))

        # Now confirm we can aggregate along this coord.
        res_cube = cube.aggregated_by('forecast_period', MEAN)
        res_cell_methods = res_cube.cell_methods[0]
        self.assertEqual(res_cell_methods.coord_names, ('forecast_period',))
        self.assertEqual(res_cell_methods.method, 'mean')


class Test_data_dtype_fillvalue(tests.IrisTest):
    def _sample_data(self, dtype=('f4'), masked=False, fill_value=None,
                     lazy=False):
        data = np.arange(6).reshape((2, 3))
        dtype = np.dtype(dtype)
        data = data.astype(dtype)
        if masked:
            data = ma.masked_array(data, mask=[[0, 1, 0], [0, 0, 0]],
                                   fill_value=fill_value)
        if lazy:
            data = as_lazy_data(data)
        return data

    def _sample_cube(self, dtype=('f4'), masked=False, fill_value=None,
                     lazy=False):
        data = self._sample_data(dtype=dtype, masked=masked,
                                 fill_value=fill_value, lazy=lazy)
        cube = Cube(data)
        return cube

    def test_realdata_change(self):
        # Check re-assigning real data.
        cube = self._sample_cube()
        self.assertEqual(cube.dtype, np.float32)
        new_dtype = np.dtype('i4')
        new_data = self._sample_data(dtype=new_dtype)
        cube.data = new_data
        self.assertIs(cube.core_data(), new_data)
        self.assertEqual(cube.dtype, new_dtype)

    def test_realmaskdata_change(self):
        # Check re-assigning real masked data.
        cube = self._sample_cube(masked=True, fill_value=1234)
        self.assertEqual(cube.dtype, np.float32)
        new_dtype = np.dtype('i4')
        new_fill_value = 4321
        new_data = self._sample_data(masked=True,
                                     fill_value=new_fill_value,
                                     dtype=new_dtype)
        cube.data = new_data
        self.assertIs(cube.core_data(), new_data)
        self.assertEqual(cube.dtype, new_dtype)
        self.assertEqual(cube.data.fill_value, new_fill_value)

    def test_lazydata_change(self):
        # Check re-assigning lazy data.
        cube = self._sample_cube(lazy=True)
        self.assertEqual(cube.core_data().dtype, np.float32)
        new_dtype = np.dtype('f8')
        new_data = self._sample_data(new_dtype, lazy=True)
        cube.data = new_data
        self.assertIs(cube.core_data(), new_data)
        self.assertEqual(cube.dtype, new_dtype)

    def test_lazymaskdata_change(self):
        # Check re-assigning lazy masked data.
        cube = self._sample_cube(masked=True, fill_value=1234,
                                 lazy=True)
        self.assertEqual(cube.core_data().dtype, np.float32)
        new_dtype = np.dtype('f8')
        new_fill_value = 4321
        new_data = self._sample_data(dtype=new_dtype, masked=True,
                                     fill_value=new_fill_value, lazy=True)
        cube.data = new_data
        self.assertIs(cube.core_data(), new_data)
        self.assertEqual(cube.dtype, new_dtype)
        self.assertEqual(cube.data.fill_value, new_fill_value)

    def test_lazydata_realise(self):
        # Check touching lazy data.
        fill_value = 27.3
        cube = self._sample_cube(lazy=True)
        data = cube.data
        self.assertIs(cube.core_data(), data)
        self.assertEqual(cube.dtype, np.float32)

    def test_lazymaskdata_realise(self):
        # Check touching masked lazy data.
        fill_value = 27.3
        cube = self._sample_cube(masked=True, fill_value=fill_value, lazy=True)
        data = cube.data
        self.assertIs(cube.core_data(), data)
        self.assertEqual(cube.dtype, np.float32)
        self.assertEqual(data.fill_value, np.float32(fill_value))

    def test_realmaskedconstantint_realise(self):
        masked_data = ma.masked_array([666], mask=True)
        masked_constant = masked_data[0]
        cube = Cube(masked_constant)
        data = cube.data
        self.assertTrue(ma.isMaskedArray(data))
        self.assertNotIsInstance(data, ma.core.MaskedConstant)

    def test_lazymaskedconstantint_realise(self):
        dtype = np.dtype('i2')
        masked_data = ma.masked_array([666], mask=True, dtype=dtype)
        masked_constant = masked_data[0]
        masked_constant_lazy = as_lazy_data(masked_constant)
        cube = Cube(masked_constant_lazy)
        data = cube.data
        self.assertTrue(ma.isMaskedArray(data))
        self.assertNotIsInstance(data, ma.core.MaskedConstant)

    def test_lazydata___getitem__dtype(self):
        fill_value = 1234
        dtype = np.dtype('int16')
        masked_array = ma.masked_array(np.arange(5),
                                       mask=[0, 0, 1, 0, 0],
                                       fill_value=fill_value,
                                       dtype=dtype)
        lazy_masked_array = as_lazy_data(masked_array)
        cube = Cube(lazy_masked_array)
        subcube = cube[3:]
        self.assertEqual(subcube.dtype, dtype)
        self.assertEqual(subcube.data.fill_value, fill_value)


if __name__ == '__main__':
    tests.main()
