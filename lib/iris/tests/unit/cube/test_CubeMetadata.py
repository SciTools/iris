# (C) British Crown Copyright 2017, Met Office
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
"""Unit tests for the `iris.cube.CubeMetadata` class."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import biggus

from iris.cube import Cube, CubeMetadata
from iris.tests import mock


class Test(tests.IrisTest):
    def create_metadata(self,
                        standard_name=mock.sentinel.std_name,
                        long_name=mock.sentinel.long_name,
                        var_name=mock.sentinel.var_name,
                        units=mock.sentinel.units,
                        attributes=mock.sentinel.attributes,
                        cell_methods=mock.sentinel.cell_methods):
        md = CubeMetadata(standard_name, long_name, var_name, units,
                          attributes, cell_methods)
        return md

    def test_constructor(self):
        md = self.create_metadata()
        self.assertEqual(md.standard_name, mock.sentinel.std_name)
        self.assertEqual(md.long_name, mock.sentinel.long_name)
        self.assertEqual(md.var_name, mock.sentinel.var_name)
        self.assertEqual(md.attributes, mock.sentinel.attributes)
        self.assertEqual(md.cell_methods, mock.sentinel.cell_methods)

    def test_name_method(self):
        md = self.create_metadata()
        self.assertEqual(md.name(), mock.sentinel.std_name)
        self.assertNotEqual(md.name(), mock.sentinel.long_name)

        md = self.create_metadata(standard_name=None)
        self.assertEqual(md.name(), mock.sentinel.long_name)

        md = self.create_metadata(standard_name=None, long_name=None)
        self.assertEqual(md.name(), mock.sentinel.var_name)

        md = self.create_metadata(standard_name=None, long_name=None,
                                  var_name=None)
        self.assertEqual(md.name(), 'unknown')
        self.assertEqual(md.name('foobar'), 'foobar')

    def test_from_cube(self):
        cube = mock.Mock()
        md = CubeMetadata.from_cube(cube)
        self.assertEqual(md.standard_name, cube.standard_name)
        self.assertEqual(md.long_name, cube.long_name)
        self.assertEqual(md.var_name, cube.var_name)
        self.assertEqual(md.attributes, cube.attributes)
        self.assertEqual(md.cell_methods, cube.cell_methods)

    def test_apply_to_cube(self):
        md = self.create_metadata()
        cube = mock.Mock()
        md.apply_to_cube(cube)
        self.assertEqual(cube.standard_name, mock.sentinel.std_name)
        self.assertEqual(cube.long_name, mock.sentinel.long_name)
        self.assertEqual(cube.var_name, mock.sentinel.var_name)
        self.assertEqual(cube.attributes, mock.sentinel.attributes)
        self.assertEqual(cube.cell_methods, mock.sentinel.cell_methods)

    def test_to_cube(self):
        md = self.create_metadata(standard_name='air_temperature',
                                  long_name='long_name_value',
                                  units='kelvin', var_name='foobar',
                                  attributes={},
                                  cell_methods=None)
        cube = md.to_cube(mock.Mock())
        self.assertEqual(cube.standard_name, 'air_temperature')
        self.assertEqual(cube.long_name, 'long_name_value')
        self.assertEqual(cube.var_name, 'foobar')
        self.assertEqual(cube.attributes, {})
        self.assertEqual(cube.cell_methods, ())

if __name__ == '__main__':
    tests.main()
