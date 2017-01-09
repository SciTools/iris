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
"""Unit tests for :class:`iris.coords.AuxCoord`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

import biggus
from iris.coords import AuxCoord


class Test___init__(tests.IrisTest):
    def test_writeable(self):
        coord = AuxCoord([1, 2], bounds=[[1, 2], [2, 3]])
        self.assertTrue(coord.points.flags.writeable)
        self.assertTrue(coord.bounds.flags.writeable)


def fetch_base(ndarray):
    if ndarray.base is not None:
        return fetch_base(ndarray.base)
    return ndarray


class Test___getitem__(tests.IrisTest):
    def test_share_data(self):
        # Ensure that slicing a coordinate behaves like slicing a numpy array
        # i.e. that the points and bounds are views of the original.
        original = AuxCoord([1, 2], bounds=[[1, 2], [2, 3]],
                            attributes={'dummy1': None},
                            coord_system=tests.mock.sentinel.coord_system)
        sliced_coord = original[:]
        self.assertIs(fetch_base(sliced_coord._points),
                      fetch_base(original._points))
        self.assertIs(fetch_base(sliced_coord._bounds),
                      fetch_base(original._bounds))
        self.assertIsNot(sliced_coord.coord_system, original.coord_system)
        self.assertIsNot(sliced_coord.attributes, original.attributes)

    def test_lazy_data_realisation(self):
        # Capture the fact that we realise the data when slicing.
        points = np.array([1, 2])
        points = biggus.NumpyArrayAdapter(points)

        bounds = np.array([[1, 2], [2, 3]])
        bounds = biggus.NumpyArrayAdapter(bounds)

        original = AuxCoord(points, bounds=bounds,
                            attributes={'dummy1': None},
                            coord_system=tests.mock.sentinel.coord_system)
        sliced_coord = original[:]
        # Returned coord is realised.
        self.assertIsInstance(sliced_coord._points, biggus.NumpyArrayAdapter)
        self.assertIsInstance(sliced_coord._bounds, biggus.NumpyArrayAdapter)

        # Original coord remains unrealised.
        self.assertIsInstance(points, biggus.NumpyArrayAdapter)
        self.assertIsInstance(bounds, biggus.NumpyArrayAdapter)


class Test_copy(tests.IrisTest):
    def setUp(self):
        self.original = AuxCoord([1, 2], bounds=[[1, 2], [2, 3]],
                                 attributes={'dummy1': None},
                                 coord_system=tests.mock.sentinel.coord_system)

    def assert_data_no_share(self, coord_copy):
        self.assertIsNot(fetch_base(coord_copy._points),
                         fetch_base(self.original._points))
        self.assertIsNot(fetch_base(coord_copy._bounds),
                         fetch_base(self.original._bounds))
        self.assertIsNot(coord_copy.coord_system, self.original.coord_system)
        self.assertIsNot(coord_copy.attributes, self.original.attributes)

    def test_existing_points(self):
        # Ensure that copying a coordinate does not return a view of its
        # points or bounds.
        coord_copy = self.original.copy()
        self.assert_data_no_share(coord_copy)

    def test_existing_points_deepcopy_call(self):
        # Ensure that the coordinate object itself is deepcopied called.
        with tests.mock.patch('copy.deepcopy') as mock_copy:
            self.original.copy()
        mock_copy.assert_called_once_with(self.original)

    def test_new_points(self):
        coord_copy = self.original.copy([1, 2], bounds=[[1, 2], [2, 3]])
        self.assert_data_no_share(coord_copy)

    def test_new_points_shallowcopy_call(self):
        # Ensure that the coordinate object itself is shallow copied so that
        # the points and bounds are not unnecessarily copied.
        with tests.mock.patch('copy.copy') as mock_copy:
            self.original.copy([1, 2], bounds=[[1, 2], [2, 3]])
        mock_copy.assert_called_once_with(self.original)


if __name__ == '__main__':
    tests.main()
