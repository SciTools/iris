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
"""Unit tests for the :class:`iris.coords.Coord` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import datetime
import collections

import mock
import numpy as np

from iris.coords import AuxCoord, Coord


Pair = collections.namedtuple('Pair', 'points bounds')


class Test_cell(tests.IrisTest):
    def _mock_coord(self):
        coord = mock.Mock(spec=Coord, ndim=1,
                          points=np.array([mock.sentinel.time]),
                          bounds=np.array([[mock.sentinel.lower,
                                            mock.sentinel.upper]]))
        return coord

    def test_time_as_number(self):
        # Make sure Coord.cell() normally returns the values straight
        # out of the Coord's points/bounds arrays.
        coord = self._mock_coord()
        cell = Coord.cell(coord, 0)
        self.assertIs(cell.point, mock.sentinel.time)
        self.assertEquals(cell.bound,
                          (mock.sentinel.lower, mock.sentinel.upper))

    def test_time_as_object(self):
        # When iris.FUTURE.cell_datetime_objects is True, ensure
        # Coord.cell() converts the point/bound values to "datetime"
        # objects.
        coord = self._mock_coord()
        coord.units.num2date = mock.Mock(
            side_effect=[mock.sentinel.datetime,
                         (mock.sentinel.datetime_lower,
                          mock.sentinel.datetime_upper)])
        with mock.patch('iris.FUTURE', cell_datetime_objects=True):
            cell = Coord.cell(coord, 0)
        self.assertIs(cell.point, mock.sentinel.datetime)
        self.assertEquals(cell.bound,
                          (mock.sentinel.datetime_lower,
                           mock.sentinel.datetime_upper))
        self.assertEqual(coord.units.num2date.call_args_list,
                         [mock.call((mock.sentinel.time,)),
                          mock.call((mock.sentinel.lower,
                                     mock.sentinel.upper))])


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


class Test_is_compatible(tests.IrisTest):
    def setUp(self):
        self.test_coord = AuxCoord([1.])
        self.other_coord = self.test_coord.copy()

    def test_noncommon_array_attrs_compatible(self):
        # Non-common array attributes should be ok.
        self.test_coord.attributes['array_test'] = np.array([1.0, 2, 3])
        self.assertTrue(self.test_coord.is_compatible(self.other_coord))

    def test_matching_array_attrs_compatible(self):
        # Matching array attributes should be ok.
        self.test_coord.attributes['array_test'] = np.array([1.0, 2, 3])
        self.other_coord.attributes['array_test'] = np.array([1.0, 2, 3])
        self.assertTrue(self.test_coord.is_compatible(self.other_coord))

    def test_different_array_attrs_incompatible(self):
        # Differing array attributes should make coords incompatible.
        self.test_coord.attributes['array_test'] = np.array([1.0, 2, 3])
        self.other_coord.attributes['array_test'] = np.array([1.0, 2, 777.7])
        self.assertFalse(self.test_coord.is_compatible(self.other_coord))


if __name__ == '__main__':
    tests.main()
