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
"""Test function :func:`iris.coord_categorisation.add_categorised_coord`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import mock
import numpy as np

from iris.coord_categorisation import add_categorised_coord
from iris.coord_categorisation import add_day_of_year
from iris.cube import Cube
from iris.coords import DimCoord
from iris.unit import CALENDARS as calendars
from iris.unit import Unit


class Test_add_categorised_coord(tests.IrisTest):

    def setUp(self):
        # Factor out common variables and objects.
        self.cube = mock.Mock(name='cube', coords=mock.Mock(return_value=[]))
        self.coord = mock.Mock(name='coord',
                               points=np.arange(12).reshape(3, 4))
        self.units = 'units'
        self.vectorised = mock.Mock(name='vectorized_result')

    def test_vectorise_call(self):
        # Check that the function being passed through gets called with
        # numpy.vectorize, before being applied to the points array.
        # The reason we use numpy.vectorize is to support multi-dimensional
        # coordinate points.
        fn = lambda coord, v: v**2

        with mock.patch('numpy.vectorize',
                        return_value=self.vectorised) as vectorise_patch:
            with mock.patch('iris.coords.AuxCoord') as aux_coord_constructor:
                add_categorised_coord(self.cube, 'foobar', self.coord, fn,
                                      units=self.units)

        # Check the constructor of AuxCoord gets called with the
        # appropriate arguments.
        # Start with the vectorised function.
        vectorise_patch.assert_called_once_with(fn)
        # Check the vectorize wrapper gets called with the appropriate args.
        self.vectorised.assert_called_once_with(self.coord, self.coord.points)
        # Check the AuxCoord constructor itself.
        aux_coord_constructor.assert_called_once_with(
            self.vectorised(self.coord, self.coord.points),
            units=self.units,
            attributes=self.coord.attributes.copy())
        # And check adding the aux coord to the cube mock.
        self.cube.add_aux_coord.assert_called_once_with(
            aux_coord_constructor(), self.cube.coord_dims(self.coord))

    def test_string_vectorised(self):
        # Check that special case handling of a vectorized string returning
        # function is taking place.
        fn = lambda coord, v: '0123456789'[:v]

        with mock.patch('numpy.vectorize',
                        return_value=self.vectorised) as vectorise_patch:
            with mock.patch('iris.coords.AuxCoord') as aux_coord_constructor:
                add_categorised_coord(self.cube, 'foobar', self.coord, fn,
                                      units=self.units)

        self.assertEqual(
            aux_coord_constructor.call_args[0][0],
            vectorise_patch(fn, otypes=[object])(self.coord, self.coord.points)
            .astype('|S64'))


class Test_add_day_of_year(tests.IrisTest):
    def setUp(self):
        self.expected = {
            'standard': np.array(list(range(360, 367)) + list(range(1, 4))),
            'gregorian': np.array(list(range(360, 367)) + list(range(1, 4))),
            'proleptic_gregorian': np.array(list(range(360, 367)) +
                                            list(range(1, 4))),
            'noleap': np.array(list(range(359, 366)) + list(range(1, 4))),
            'julian': np.array(list(range(360, 367)) + list(range(1, 4))),
            'all_leap': np.array(list(range(360, 367)) + list(range(1, 4))),
            '365_day': np.array(list(range(359, 366)) + list(range(1, 4))),
            '366_day': np.array(list(range(360, 367)) + list(range(1, 4))),
            '360_day': np.array(list(range(355, 361)) + list(range(1, 5)))}

    def make_cube(self, calendar):
        n_times = 10
        cube = Cube(np.arange(n_times))
        time_coord = DimCoord(np.arange(n_times), standard_name='time',
                              units=Unit('days since 1980-12-25',
                                         calendar=calendar))
        cube.add_dim_coord(time_coord, 0)
        return cube

    def test_calendars(self):
        for calendar in calendars:
            cube = self.make_cube(calendar)
            add_day_of_year(cube, 'time')
            points = cube.coord('day_of_year').points
            expected_points = self.expected[calendar]
            msg = 'Test failed for the following calendar: {}.'
            self.assertArrayEqual(points, expected_points,
                                  err_msg=msg.format(calendar))


if __name__ == '__main__':
    tests.main()
