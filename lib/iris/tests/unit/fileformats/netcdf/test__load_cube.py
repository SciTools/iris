# (C) British Crown Copyright 2014 - 2015, Met Office
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

import mock
import netCDF4
import numpy as np

from iris.coords import DimCoord
import iris.fileformats.cf
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


class TestCoordAttributes(tests.IrisTest):
    @staticmethod
    def _patcher(engine, cf, cf_group):
        coordinates = []
        for coord in cf_group:
            engine.cube.add_aux_coord(coord)
            coordinates.append((coord, coord.name()))
        engine.provides['coordinates'] = coordinates

    def setUp(self):
        this = 'iris.fileformats.netcdf._assert_case_specific_facts'
        patch = mock.patch(this, side_effect=self._patcher)
        patch.start()
        self.addCleanup(patch.stop)
        self.engine = mock.Mock()
        self.filename = 'DUMMY'
        self.flag_masks = mock.sentinel.flag_masks
        self.flag_meanings = mock.sentinel.flag_meanings
        self.flag_values = mock.sentinel.flag_values

    def _make(self, names, attrs):
        coords = [DimCoord(i, long_name=name) for i, name in enumerate(names)]

        cf_group = {}
        for name, cf_attrs in zip(names, attrs):
            cf_attrs_unused = mock.Mock(return_value=cf_attrs)
            cf_group[name] = mock.Mock(cf_attrs_unused=cf_attrs_unused)
        cf = mock.Mock(cf_group=cf_group)

        cf_var = mock.MagicMock(spec=iris.fileformats.cf.CFVariable,
                                dtype=np.dtype('i4'),
                                cf_data=mock.Mock(),
                                cf_name='DUMMY_VAR',
                                cf_group=coords,
                                shape=(1,))
        return cf, cf_var

    def test_flag_pass_thru(self):
        items = [('masks', 'flag_masks', self.flag_masks),
                 ('meanings', 'flag_meanings', self.flag_meanings),
                 ('values', 'flag_values', self.flag_values)]
        for name, attr, value in items:
            names = [name]
            attrs = [[(attr, value)]]
            cf, cf_var = self._make(names, attrs)
            cube = _load_cube(self.engine, cf, cf_var, self.filename)
            self.assertEqual(len(cube.coords(name)), 1)
            coord = cube.coord(name)
            self.assertEqual(len(coord.attributes), 1)
            self.assertEqual(list(coord.attributes.keys()), [attr])
            self.assertEqual(list(coord.attributes.values()), [value])

    def test_flag_pass_thru_multi(self):
        names = ['masks', 'meanings', 'values']
        attrs = [[('flag_masks', self.flag_masks),
                  ('wibble', 'wibble')],
                 [('flag_meanings', self.flag_meanings),
                  ('add_offset', 'add_offset')],
                 [('flag_values', self.flag_values)]]
        cf, cf_var = self._make(names, attrs)
        cube = _load_cube(self.engine, cf, cf_var, self.filename)
        self.assertEqual(len(cube.coords()), 3)
        self.assertEqual(set([c.name() for c in cube.coords()]), set(names))
        expected = [attrs[0],
                    [attrs[1][0]],
                    attrs[2]]
        for name, expect in zip(names, expected):
            attributes = cube.coord(name).attributes
            self.assertEqual(set(attributes.items()), set(expect))


class TestCubeAttributes(tests.IrisTest):
    def setUp(self):
        this = 'iris.fileformats.netcdf._assert_case_specific_facts'
        patch = mock.patch(this)
        patch.start()
        self.addCleanup(patch.stop)
        self.engine = mock.Mock()
        self.cf = None
        self.filename = 'DUMMY'
        self.flag_masks = mock.sentinel.flag_masks
        self.flag_meanings = mock.sentinel.flag_meanings
        self.flag_values = mock.sentinel.flag_values

    def _make(self, attrs):
        cf_attrs_unused = mock.Mock(return_value=attrs)
        cf_var = mock.MagicMock(spec=iris.fileformats.cf.CFVariable,
                                dtype=np.dtype('i4'),
                                cf_data=mock.Mock(),
                                cf_name='DUMMY_VAR',
                                cf_group=mock.Mock(),
                                cf_attrs_unused=cf_attrs_unused,
                                shape=mock.MagicMock())
        return cf_var

    def test_flag_pass_thru(self):
        attrs = [('flag_masks', self.flag_masks),
                 ('flag_meanings', self.flag_meanings),
                 ('flag_values', self.flag_values)]
        for key, value in attrs:
            cf_var = self._make([(key, value)])
            cube = _load_cube(self.engine, self.cf, cf_var, self.filename)
            self.assertEqual(len(cube.attributes), 1)
            self.assertEqual(list(cube.attributes.keys()), [key])
            self.assertEqual(list(cube.attributes.values()), [value])

    def test_flag_pass_thru_multi(self):
        attrs = [('flag_masks', self.flag_masks),
                 ('wibble', 'wobble'),
                 ('flag_meanings', self.flag_meanings),
                 ('add_offset', 'add_offset'),
                 ('flag_values', self.flag_values),
                 ('standard_name', 'air_temperature')]
        expected = set([attrs[0], attrs[1], attrs[2], attrs[4]])
        cf_var = self._make(attrs)
        cube = _load_cube(self.engine, self.cf, cf_var, self.filename)
        self.assertEqual(len(cube.attributes), len(expected))
        self.assertEqual(set(cube.attributes.items()), expected)


if __name__ == "__main__":
    tests.main()
