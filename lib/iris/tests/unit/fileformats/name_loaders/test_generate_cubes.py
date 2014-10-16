# (C) British Crown Copyright 2013 - 2014, Met Office
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
"""
Unit tests for :func:`iris.analysis.name_loaders._generate_cubes`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

import iris.cube
from iris.fileformats.name_loaders import _generate_cubes


class TestCellMethods(tests.IrisTest):
    def test_cell_methods(self):
        header = mock.MagicMock()
        column_headings = {'Species': [1, 2, 3], 'Quantity': [4, 5, 6],
                           "Unit": ['m', 'm', 'm'], 'Z': [1, 2, 3]}
        coords = mock.MagicMock()
        data_arrays = [mock.Mock(), mock.Mock()]
        cell_methods = ["cell_method_1", "cell_method_2"]

        with mock.patch('iris.fileformats.name_loaders._cf_height_from_name'):
            with mock.patch('iris.cube.Cube'):
                cubes = list(_generate_cubes(header, column_headings, coords,
                                             data_arrays, cell_methods))

        cubes[0].assert_has_call(mock.call.add_cell_method('cell_method_1'))
        cubes[1].assert_has_call(mock.call.add_cell_method('cell_method_2'))


if __name__ == "__main__":
    tests.main()
