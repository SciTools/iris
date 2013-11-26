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
"""Unit tests for the :class:`iris.coords.Coord`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from collections import namedtuple
import mock

import numpy as np

from iris.coords import AuxCoord
import iris.exceptions


Pair = namedtuple('Pair', 'points bounds')


class Test_collapsed(tests.IrisTest):
    def _serialize(self, data):
        return '|'.join(str(item) for item in data.flatten())

    def setUp(self):
        # Build numeric N points and (N, 2) bounds.
        points = np.array([2, 4, 6, 8])
        bounds = np.array([[1, 3], [3, 5], [5, 7], [7, 9]])
        self.numeric = Pair(points, bounds)
        # Build string N points and (N, 2) bounds.
        points = np.array(['two', 'four', 'six', 'eight'])
        bounds = np.array([['one', 'three'],
                           ['three', 'five'],
                           ['five', 'seven'],
                           ['seven', 'nine']])
        self.string = Pair(points, bounds)
        # Build numeric N points and (N, 4) bounds.
        points = np.array([3, 6, 9])
        bounds = np.array([[1, 2, 4, 5], [4, 5, 7, 8], [7, 8, 10, 11]])
        self.numeric_multi = Pair(points, bounds)
        # Build string N points and (N, 4) bounds.
        points = np.array(['three', 'six', 'nine'])
        bounds = np.array([['one', 'two', 'four', 'five'],
                           ['four', 'five', 'seven', 'eight'],
                           ['seven', 'eight', 'ten', 'eleven']])
        self.string_multi = Pair(points, bounds)
        self.pairs = [self.numeric, self.numeric_multi,
                      self.string, self.string_multi]

    def test_serialize(self):
        for units in ['unknown', 'no_unit']:
            for points, bounds in self.pairs:
                coord = mock.MagicMock(spec_set=AuxCoord, name='AuxCoord',
                                       points=points, bounds=bounds,
                                       units=units)
                # Now perform the collase operation with the mock coordinate.
                AuxCoord.collapsed(coord)
                # Examine the operational footprint in the mock.
                self.assertEqual(coord.copy.call_count, 1)
                args, kwargs = coord.copy.call_args
                self.assertEqual(args, ())
                self.assertEqual(set(kwargs), set(['points', 'bounds']))
                self.assertArrayEqual(kwargs['points'],
                                      self._serialize(points))
                for index in np.ndindex(coord.bounds.shape[1:]):
                    index_slice = (slice(None),) + tuple(index)
                    self.assertArrayEqual(kwargs['bounds'][index_slice],
                                          self._serialize(bounds[index_slice]))


class Test_cells(tests.IrisTest):
    def test_exception_multidim(self):
        # Ensure that an exception is raised with multidimensional coordinates.
        coord = AuxCoord(np.zeros(4).reshape(2, 2))
        with self.assertRaises(iris.exceptions.CoordinateMultiDimError):
            coord.cells().next()

    def test_cell_method_call_default(self):
        # Ensure cell method is called and with the correct parameters.
        coord = AuxCoord(np.zeros(4))
        with mock.patch('iris.coords.Coord.cell') as cell_patch:
            coord.cells().next()
        cell_patch.assert_called_with(0, extended=True)

    def test_cell_method_call_extended(self):
        # Ensure cell method is called and with the correct parameters.
        coord = AuxCoord(np.zeros(4))
        with mock.patch('iris.coords.Coord.cell') as cell_patch:
            coord.cells(extended=False).next()
        cell_patch.assert_called_with(0, extended=False)


class Test_index(tests.IrisTest):
    def test_raise_exception(self):
        coord = AuxCoord(np.zeros(4))
        cell = mock.Mock(name='cell')
        with self.assertRaises(iris.exceptions.IrisError) as err:
            coord.index(cell)
        msg = ('Coord.index() is no longer available.  Use '
               'Coord.nearest_neighbour_index() instead.')
        self.assertEqual(err.exception.message, msg)


class Test_cell_extended(tests.IrisTest):
    # Use of the extended keyword.
    def setUp(self):
        slice_patch = mock.patch(
            'iris.util._build_full_slice_given_keys', return_value=0)
        slice_patch.start()
        self.addCleanup(slice_patch.stop)

        cell_patch = mock.patch('iris.coords.Cell')
        cell_patch.start()
        self.addCleanup(cell_patch.stop)

        self.numdate = mock.Mock(name='date')
        numdate_patch = mock.patch(
            'iris.unit.Unit.num2date', return_value=self.numdate)
        numdate_patch.start()
        self.addCleanup(numdate_patch.stop)

        self.coord = AuxCoord(np.zeros(4), units='unknown')

    def test_extended_false(self):
        # Supplying extended false keyword.
        self.coord.cell(0, extended=False)
        iris.coords.Cell.assert_called_with((0.0,), None)

    def test_extended_no_bound_time_ref(self):
        # Extended keyword and is time reference unit.
        with mock.patch(
                'iris.unit.Unit.is_time_reference', return_value=True):
            self.coord.cell(0, extended=True)
            iris.coords.Cell.assert_called_with(self.numdate, None)
            self.assertEqual(iris.unit.Unit.is_time_reference.call_count, 1)
            self.assertEqual(iris.unit.Unit.num2date.call_count, 1)

    def test_extended_no_bound_no_time_ref(self):
        # Extended keyword and is time reference unit.
        with mock.patch(
                'iris.unit.Unit.is_time_reference', return_value=False):
            self.coord.cell(0, extended=True)
            iris.coords.Cell.assert_called_with((0.0,), None)
            self.assertEqual(iris.unit.Unit.is_time_reference.call_count, 1)
            self.assertEqual(iris.unit.Unit.num2date.call_count, 0)

    def test_extended_bound_time_ref(self):
        # Extended keyword with bounds and has time reference unit.
        self.coord.bounds = np.zeros(8).reshape(4, 2)
        with mock.patch(
                'iris.unit.Unit.is_time_reference', return_value=True):
            self.coord.cell(0, extended=True)
            iris.coords.Cell.assert_called_with(self.numdate, self.numdate)
            self.assertEqual(iris.unit.Unit.is_time_reference.call_count, 1)
            self.assertEqual(iris.unit.Unit.num2date.call_count, 2)


if __name__ == '__main__':
    tests.main()
