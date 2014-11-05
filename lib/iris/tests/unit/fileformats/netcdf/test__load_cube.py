# (C) British Crown Copyright 2014, Met Office
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
"""Unit tests for the `iris.fileformats.netcdf._load_cube` function."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import iris.fileformats.cf
import mock
import netCDF4
import numpy as np

from iris.fileformats.netcdf import _load_cube


class TestFillValue(tests.IrisTest):
    def setUp(self):
        name = 'iris.fileformats.netcdf._assert_case_specific_facts'
        patch = mock.patch(name)
        patch.start()
        self.addCleanup(patch.stop)

        self.engine = mock.Mock()
        self.cf = None
        self.filename = 'DUMMY'

    def _make_cf_var(self, dtype):
        variable = mock.Mock(spec=netCDF4.Variable, dtype=dtype)
        cf_var = mock.MagicMock(spec=iris.fileformats.cf.CFVariable,
                                cf_data=variable, cf_name='DUMMY_VAR',
                                cf_group=mock.Mock(), dtype=dtype,
                                shape=mock.MagicMock())
        return cf_var

    def _test(self, cf_var, expected_fill_value):
        cube = _load_cube(self.engine, self.cf, cf_var, self.filename)
        self.assertEqual(cube._my_data.fill_value, expected_fill_value)

    def test_from_attribute_dtype_f4(self):
        # A _FillValue attribute on the netCDF variable should end up as
        # the fill_value for the cube.
        dtype = np.dtype('f4')
        cf_var = self._make_cf_var(dtype)
        cf_var.cf_data._FillValue = mock.sentinel.FILL_VALUE
        self._test(cf_var, mock.sentinel.FILL_VALUE)

    def test_from_default_dtype_f4(self):
        # Without an explicit _FillValue attribute on the netCDF
        # variable, the fill value should be selected from the default
        # netCDF fill values.
        dtype = np.dtype('f4')
        cf_var = self._make_cf_var(dtype)
        self._test(cf_var, netCDF4.default_fillvals['f4'])

    def test_from_attribute_dtype_i4(self):
        # A _FillValue attribute on the netCDF variable should end up as
        # the fill_value for the cube.
        dtype = np.dtype('i4')
        cf_var = self._make_cf_var(dtype)
        cf_var.cf_data._FillValue = mock.sentinel.FILL_VALUE
        self._test(cf_var, mock.sentinel.FILL_VALUE)

    def test_from_default_dtype_i4(self):
        # Without an explicit _FillValue attribute on the netCDF
        # variable, the fill value should be selected from the default
        # netCDF fill values.
        dtype = np.dtype('i4')
        cf_var = self._make_cf_var(dtype)
        self._test(cf_var, netCDF4.default_fillvals['i4'])

    def test_from_attribute_with_scale_offset(self):
        # The _FillValue attribute still takes priority even when an
        # offset/scale transformation takes place on the data.
        dtype = np.dtype('i2')
        cf_var = self._make_cf_var(dtype)
        cf_var.scale_factor = np.float64(1.5)
        cf_var.cf_data._FillValue = mock.sentinel.FILL_VALUE
        self._test(cf_var, mock.sentinel.FILL_VALUE)

    def test_from_default_with_scale_offset(self):
        # The fill value should be related to the *non-scaled* dtype.
        dtype = np.dtype('i2')
        cf_var = self._make_cf_var(dtype)
        cf_var.scale_factor = np.float64(1.5)
        self._test(cf_var, netCDF4.default_fillvals['i2'])


if __name__ == "__main__":
    tests.main()
