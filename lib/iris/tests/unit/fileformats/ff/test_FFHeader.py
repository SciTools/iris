# (C) British Crown Copyright 2013, Met Office
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
"""Unit tests for :class:`iris.fileformat.ff.FFHeader`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import collections

import mock

from iris.fileformats.ff import FFHeader


MyGrid = collections.namedtuple('MyGrid', 'column row real horiz_grid_type')


class Test_grid(tests.IrisTest):
    def _header(self, grid_staggering):
        with mock.patch.object(FFHeader, '__init__',
                               mock.Mock(return_value=None)):
            header = FFHeader()
        header.grid_staggering = grid_staggering
        header.column_dependent_constants = mock.sentinel.column
        header.row_dependent_constants = mock.sentinel.row
        header.real_constants = mock.sentinel.real
        header.horiz_grid_type = mock.sentinel.horiz_grid_type
        return header

    def _test(self, grid_staggering):
        header = self._header(grid_staggering)
        with mock.patch.dict(FFHeader.GRID_STAGGERING_CLASS,
                             {grid_staggering: MyGrid}):
            grid = header.grid()
        self.assertIsInstance(grid, MyGrid)
        self.assertIs(grid.column, mock.sentinel.column)
        self.assertIs(grid.row, mock.sentinel.row)
        self.assertIs(grid.real, mock.sentinel.real)
        self.assertIs(grid.horiz_grid_type, mock.sentinel.horiz_grid_type)

    def test_new_dynamics(self):
        self._test(3)

    def test_end_game(self):
        self._test(6)

    def test_unknown(self):
        header = self._header(0)
        with mock.patch('iris.fileformats.ff.NewDynamics',
                        mock.Mock(return_value=mock.sentinel.grid)):
            with mock.patch('warnings.warn') as warn:
                grid = header.grid()
        warn.assert_called_with('Staggered grid type: 0 not currently'
                                ' interpreted, assuming standard C-grid')
        self.assertIs(grid, mock.sentinel.grid)


if __name__ == "__main__":
    tests.main()
