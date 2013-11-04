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
"""
Unit tests for the :class:`iris.analysis.name_loaders.cf_height_from_name`
function.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.fileformats.name_loaders import cf_height_from_name, NAMEIrisBridge


class Test_cf_height_from_name_NAMEII(tests.IrisTest):
    # NAMEII formats are defined by bounds, not points
    def test_bounded_height_above_ground(self):
        # Parse height above ground level (agl).
        data = 'From     0 -   100m agl'
        res = cf_height_from_name(data)
        com = NAMEIrisBridge(
            units='m', points=50.0, bounds=np.array([   0.,  100.]),
            standard_name='height', long_name='z')
        self.assertEqual(com, res)

    def test_bounded_height_flight_level(self):
        # Parse height above ground level (agl).
        data = 'From FL0 - FL100'
        res = cf_height_from_name(data)
        com = NAMEIrisBridge(
            units='unknown', points=50.0, bounds=np.array([   0.,  100.]),
            standard_name=None, long_name='flight_level')
        self.assertEqual(com, res)

    def test_bounded_height_above_sea_level(self):
        # Parse height above ground level (agl).
        data = 'From     0 -   100m asl'
        res = cf_height_from_name(data)
        com = NAMEIrisBridge(
            units='m', points=50.0, bounds=np.array([   0.,  100.]),
            standard_name='altitude', long_name='z')
        self.assertEqual(com, res)

    def test_malformed_height_above_ground(self):
        # Parse height above ground level (agl).
        data = 'From     0 -   100m agl and stuff'
        with self.assertRaises(RuntimeError):
            res = cf_height_from_name(data)

    def test_malformed_height_above_sea_level(self):
        # Parse height above ground level (agl).
        data = 'From     0 -   100m asl and stuff'
        with self.assertRaises(RuntimeError):
            res = cf_height_from_name(data)

    def test_malformed_flight_level(self):
        # Parse height above ground level (agl).
        data = 'From FL0 - FL100 and stuff'
        with self.assertRaises(RuntimeError):
            res = cf_height_from_name(data)

    def test_float_bounded_height_above_ground(self):
        # Parse height above ground level when its a float.
        data = 'From     0.0 -   100.0m agl'
        res = cf_height_from_name(data)
        com = NAMEIrisBridge(
            units='m', points=50.0, bounds=np.array([   0.,  100.]),
            standard_name='height', long_name='z')
        self.assertEqual(com, res)

    def test_float_bounded_height_flight_level(self):
        # Parse height above ground level (agl).
        data = 'From FL0.0 - FL100.0'
        res = cf_height_from_name(data)
        com = NAMEIrisBridge(
            units='unknown', points=50.0, bounds=np.array([   0.,  100.]),
            standard_name=None, long_name='flight_level')
        self.assertEqual(com, res)

    def test_float_bounded_height_above_sea_level(self):
        # Parse height above ground level (agl).
        data = 'From     0.0 -   100.0m asl'
        res = cf_height_from_name(data)
        com = NAMEIrisBridge(
            units='m', points=50.0, bounds=np.array([   0.,  100.]),
            standard_name='altitude', long_name='z')
        self.assertEqual(com, res)

    def test_no_match(self):
        # Parse height information when there is no match.
        data = 'Vertical integral'
        res = cf_height_from_name(data)
        com = NAMEIrisBridge(
            units='no-unit', points='Vertical integral', bounds=None,
            standard_name=None, long_name='z')

    def test_pressure(self):
        data = 'From     0 -   100 Pa'
        res = cf_height_from_name(data)
        com = NAMEIrisBridge(
            units='Pa', points=50.0, bounds=np.array([   0.,  100.]),
            standard_name='air_pressure', long_name='z')
        self.assertEqual(com, res)

 
class Test_cf_height_from_name_NAMEIII(tests.IrisTest):
    # NAMEII formats are defined by points, not boudns.
    def test_height_above_ground(self):
        # Parse height above ground level (agl).
        data = 'Z = 50.00000 m agl'
        res = cf_height_from_name(data)
        com = NAMEIrisBridge(
            units='m', points=50.0, bounds=None,
            standard_name='height', long_name='z')
        self.assertEqual(com, res)

    def test_height_flight_level(self):
        # Parse height above ground level (agl).
        data = 'Z = 50.00000 FL'
        res = cf_height_from_name(data)
        com = NAMEIrisBridge(
            units='unknown', points=50.0, bounds=None,
            standard_name=None, long_name='flight_level')
        self.assertEqual(com, res)

    def test_height_above_sea_level(self):
        # Parse height above ground level (agl).
        data = 'Z = 50.00000 m asl'
        res = cf_height_from_name(data)
        com = NAMEIrisBridge(
            units='m', points=50.0, bounds=None,
            standard_name='altitude', long_name='z')
        self.assertEqual(com, res)

    def test_malformed_height_above_ground(self):
        # Parse height above ground level (agl).
        data = 'Z = 50.00000 m agl and stuff'
        with self.assertRaises(RuntimeError):
            res = cf_height_from_name(data)

    def test_malformed_height_above_sea_level(self):
        # Parse height above ground level (agl).
        data = 'Z = 50.00000 m asl and stuff'
        with self.assertRaises(RuntimeError):
            res = cf_height_from_name(data)

    def test_malformed_flight_level(self):
        # Parse height above ground level (agl).
        data = 'Z = 50.00000 FL and stuff'
        with self.assertRaises(RuntimeError):
            res = cf_height_from_name(data)

    def test_integer_height_above_ground(self):
        # Parse height above ground level when its a float.
        data = 'Z = 50 m agl'
        res = cf_height_from_name(data)
        com = NAMEIrisBridge(
            units='m', points=50.0, bounds=None,
            standard_name='height', long_name='z')
        self.assertEqual(com, res)

    def test_integer_height_flight_level(self):
        # Parse height above ground level (agl).
        data = 'Z = 50 FL'
        res = cf_height_from_name(data)
        com = NAMEIrisBridge(
            units='unknown', points=50.0, bounds=None,
            standard_name=None, long_name='flight_level')
        self.assertEqual(com, res)

    def test_integer_height_above_sea_level(self):
        # Parse height above ground level (agl).
        data = 'Z = 50 m asl'
        res = cf_height_from_name(data)
        com = NAMEIrisBridge(
            units='m', points=50.0, bounds=None,
            standard_name='altitude', long_name='z')
        self.assertEqual(com, res)

    def test_pressure(self):
        data = 'Z = 50.00000 Pa'
        res = cf_height_from_name(data)
        com = NAMEIrisBridge(
            units='Pa', points=50.0, bounds=None,
            standard_name='air_pressure', long_name='z')
        self.assertEqual(com, res)


if __name__ == "__main__":
    tests.main()
