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


if __name__ == '__main__':
    tests.main()
