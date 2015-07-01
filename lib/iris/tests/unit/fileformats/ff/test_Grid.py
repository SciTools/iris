# (C) British Crown Copyright 2013 - 2015, Met Office
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
"""Unit tests for :class:`iris.fileformat.ff.Grid`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

from iris.fileformats.ff import Grid


class Test___init__(tests.IrisTest):
    def test_attributes(self):
        # Ensure the constructor initialises all the grid's attributes
        # correctly, including unpacking values from the REAL constants.
        reals = (mock.sentinel.ew, mock.sentinel.ns,
                 mock.sentinel.first_lat, mock.sentinel.first_lon,
                 mock.sentinel.pole_lat, mock.sentinel.pole_lon)
        grid = Grid(mock.sentinel.column, mock.sentinel.row, reals,
                    mock.sentinel.horiz_grid_type)
        self.assertIs(grid.column_dependent_constants, mock.sentinel.column)
        self.assertIs(grid.row_dependent_constants, mock.sentinel.row)
        self.assertIs(grid.ew_spacing, mock.sentinel.ew)
        self.assertIs(grid.ns_spacing, mock.sentinel.ns)
        self.assertIs(grid.first_lat, mock.sentinel.first_lat)
        self.assertIs(grid.first_lon, mock.sentinel.first_lon)
        self.assertIs(grid.pole_lat, mock.sentinel.pole_lat)
        self.assertIs(grid.pole_lon, mock.sentinel.pole_lon)
        self.assertIs(grid.horiz_grid_type, mock.sentinel.horiz_grid_type)


class Test_vectors(tests.IrisTest):
    def setUp(self):
        self.xp = mock.sentinel.xp
        self.xu = mock.sentinel.xu
        self.yp = mock.sentinel.yp
        self.yv = mock.sentinel.yv

    def _test_subgrid_vectors(self, subgrid, expected):
        grid = Grid(None, None, (None,) * 6, None)
        dummy_vectors = (self.xp, self.yp, self.xu, self.yv)
        grid._x_vectors = mock.Mock(return_value=(self.xp, self.xu))
        grid._y_vectors = mock.Mock(return_value=(self.yp, self.yv))
        result = grid.vectors(subgrid)
        self.assertEqual(result, expected)

    def test_1(self):
        # Data on atmospheric theta points.
        self._test_subgrid_vectors(1, (self.xp, self.yp))

    def test_2(self):
        # Data on atmospheric theta points, values over land only.
        self._test_subgrid_vectors(2, (self.xp, self.yp))

    def test_3(self):
        # Data on atmospheric theta points, values over sea only.
        self._test_subgrid_vectors(3, (self.xp, self.yp))

    def test_4(self):
        # Data on atmospheric zonal theta points.
        self._test_subgrid_vectors(4, (self.xp, self.yp))

    def test_5(self):
        # Data on atmospheric meridional theta points.
        self._test_subgrid_vectors(5, (self.xp, self.yp))

    def test_11(self):
        # Data on atmospheric uv points.
        self._test_subgrid_vectors(11, (self.xu, self.yv))

    def test_18(self):
        # Data on atmospheric u points on the 'c' grid.
        self._test_subgrid_vectors(18, (self.xu, self.yp))

    def test_19(self):
        # Data on atmospheric v points on the 'c' grid.
        self._test_subgrid_vectors(19, (self.xp, self.yv))

    def test_26(self):
        # Lateral boundary data at atmospheric theta points.
        self._test_subgrid_vectors(26, (self.xp, self.yp))

    def test_27(self):
        # Lateral boundary data at atmospheric u points.
        self._test_subgrid_vectors(27, (self.xu, self.yp))

    def test_28(self):
        # Lateral boundary data at atmospheric v points.
        self._test_subgrid_vectors(28, (self.xp, self.yv))

    def test_29(self):
        # Orography field for atmospheric LBCs.
        self._test_subgrid_vectors(29, (self.xp, self.yp))


if __name__ == "__main__":
    tests.main()
