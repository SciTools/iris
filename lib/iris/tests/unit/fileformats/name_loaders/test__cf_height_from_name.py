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
Unit tests for the :class:`iris.analysis.name_loaders._cf_height_from_name`
function.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.coords import AuxCoord
from iris.fileformats.name_loaders import _cf_height_from_name


class TestAll(tests.IrisTest):
    def _default_coord(self, data):
        # This private method returns a coordinate with values expected
        # when no interpretation is made of the field header string.
        return AuxCoord(
            units='no-unit', points=data, bounds=None, standard_name=None,
            long_name='z')


class TestAll_NAMEII(TestAll):
    # NAMEII formats are defined by bounds, not points
    def test_bounded_height_above_ground(self):
        data = 'From     0 -   100m agl'
        res = _cf_height_from_name(data)
        com = AuxCoord(
            units='m', points=50.0, bounds=np.array([0., 100.]),
            standard_name='height', long_name='height above ground level')
        self.assertEqual(com, res)

    def test_bounded_flight_level(self):
        data = 'From FL0 - FL100'
        res = _cf_height_from_name(data)
        com = AuxCoord(
            units='unknown', points=50.0, bounds=np.array([0., 100.]),
            standard_name=None, long_name='flight_level')
        self.assertEqual(com, res)

    def test_bounded_height_above_sea_level(self):
        data = 'From     0 -   100m asl'
        res = _cf_height_from_name(data)
        com = AuxCoord(
            units='m', points=50.0, bounds=np.array([0., 100.]),
            standard_name='altitude', long_name='altitude above sea level')
        self.assertEqual(com, res)

    def test_malformed_height_above_ground(self):
        # Parse height above ground level with additional stuff on the end of
        # the string (agl).
        data = 'From     0 -   100m agl and stuff'
        res = _cf_height_from_name(data)
        com = self._default_coord(data)
        self.assertEqual(com, res)

    def test_malformed_height_above_sea_level(self):
        # Parse height above ground level with additional stuff on the end of
        # the string (agl).
        data = 'From     0 -   100m asl and stuff'
        res = _cf_height_from_name(data)
        com = self._default_coord(data)
        self.assertEqual(com, res)

    def test_malformed_flight_level(self):
        # Parse height above ground level with additional stuff on the end of
        # the string (agl).
        data = 'From FL0 - FL100 and stuff'
        res = _cf_height_from_name(data)
        com = self._default_coord(data)
        self.assertEqual(com, res)

    def test_float_bounded_height_above_ground(self):
        # Parse height above ground level when its a float.
        data = 'From     0.0 -   100.0m agl'
        res = _cf_height_from_name(data)
        com = AuxCoord(
            units='m', points=50.0, bounds=np.array([0., 100.]),
            standard_name='height', long_name='height above ground level')
        self.assertEqual(com, res)

    def test_float_bounded_height_flight_level(self):
        # Parse height above ground level, as a float (agl).
        data = 'From FL0.0 - FL100.0'
        res = _cf_height_from_name(data)
        com = AuxCoord(
            units='unknown', points=50.0, bounds=np.array([0., 100.]),
            standard_name=None, long_name='flight_level')
        self.assertEqual(com, res)

    def test_float_bounded_height_above_sea_level(self):
        # Parse height above ground level as a float (agl).
        data = 'From     0.0 -   100.0m asl'
        res = _cf_height_from_name(data)
        com = AuxCoord(
            units='m', points=50.0, bounds=np.array([0., 100.]),
            standard_name='altitude', long_name='altitude above sea level')
        self.assertEqual(com, res)

    def test_no_match(self):
        # Parse height information when there is no match.
        # No interpretation, just returns default values.
        data = 'Vertical integral'
        res = _cf_height_from_name(data)
        com = self._default_coord(data)
        self.assertEqual(com, res)

    def test_pressure(self):
        # Parse air_pressure string.
        data = 'From     0 -   100 Pa'
        res = _cf_height_from_name(data)
        com = AuxCoord(
            units='Pa', points=50.0, bounds=np.array([0., 100.]),
            standard_name='air_pressure', long_name=None)
        self.assertEqual(com, res)


class TestAll_NAMEIII(TestAll):
    # NAMEIII formats are defined by points, not bounds.
    def test_height_above_ground(self):
        data = 'Z = 50.00000 m agl'
        res = _cf_height_from_name(data)
        com = AuxCoord(
            units='m', points=50.0, bounds=None,
            standard_name='height', long_name='height above ground level')
        self.assertEqual(com, res)

    def test_height_flight_level(self):
        data = 'Z = 50.00000 FL'
        res = _cf_height_from_name(data)
        com = AuxCoord(
            units='unknown', points=50.0, bounds=None,
            standard_name=None, long_name='flight_level')
        self.assertEqual(com, res)

    def test_height_above_sea_level(self):
        data = 'Z = 50.00000 m asl'
        res = _cf_height_from_name(data)
        com = AuxCoord(
            units='m', points=50.0, bounds=None,
            standard_name='altitude', long_name='altitude above sea level')
        self.assertEqual(com, res)

    def test_malformed_height_above_ground(self):
        # Parse height above ground level, with additonal stuff at the string
        # end (agl).
        data = 'Z = 50.00000 m agl and stuff'
        res = _cf_height_from_name(data)
        com = self._default_coord(data)
        self.assertEqual(com, res)

    def test_malformed_height_above_sea_level(self):
        # Parse height above ground level, with additional stuff at string
        # end (agl).
        data = 'Z = 50.00000 m asl and stuff'
        res = _cf_height_from_name(data)
        com = self._default_coord(data)
        self.assertEqual(com, res)

    def test_malformed_flight_level(self):
        # Parse height above ground level (agl), with additional stuff at
        # string end.
        data = 'Z = 50.00000 FL and stuff'
        res = _cf_height_from_name(data)
        com = self._default_coord(data)
        self.assertEqual(com, res)

    def test_integer_height_above_ground(self):
        # Parse height above ground level when its an integer.
        data = 'Z = 50 m agl'
        res = _cf_height_from_name(data)
        com = AuxCoord(
            units='m', points=50.0, bounds=None,
            standard_name='height', long_name='height above ground level')
        self.assertEqual(com, res)

    def test_integer_height_flight_level(self):
        # Parse flight level when its an integer.
        data = 'Z = 50 FL'
        res = _cf_height_from_name(data)
        com = AuxCoord(
            units='unknown', points=50.0, bounds=None,
            standard_name=None, long_name='flight_level')
        self.assertEqual(com, res)

    def test_integer_height_above_sea_level(self):
        # Parse height above sea level (agl) when its an integer.
        data = 'Z = 50 m asl'
        res = _cf_height_from_name(data)
        com = AuxCoord(
            units='m', points=50.0, bounds=None,
            standard_name='altitude', long_name='altitude above sea level')
        self.assertEqual(com, res)

    def test_pressure(self):
        # Parse pressure.
        data = 'Z = 50.00000 Pa'
        res = _cf_height_from_name(data)
        com = AuxCoord(
            units='Pa', points=50.0, bounds=None,
            standard_name='air_pressure', long_name=None)
        self.assertEqual(com, res)


if __name__ == "__main__":
    tests.main()
