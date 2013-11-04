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

from iris.fileformats.name_loaders import cf_height_from_name, NAMECoord


class Test_cf_height_from_name_NAMEII(tests.IrisTest):
    def test_bounded_height_above_ground(self):
        # Parse height above ground level (agl).
        data = ['From     0 -   100m agl', 'From   100 -   200m agl',
                'From   200 -   300m agl', 'From   300 -   400m agl']
        res = cf_height_from_name(data)
        com = NAMECoord(name='height', dimension=None,
                        points=np.array([50., 150., 250., 350.]),
                        bounds=np.array([[0., 100.],
                                         [100., 200.],
                                         [200., 300.],
                                         [300., 400.]]))
        self.assertEqual(res, com)

    def test_bounded_height_above_sea_level(self):
        # Parse height above sea level name coordinate (asl).
        data = ['From     0 -   100m asl', 'From   100 -   200m asl',
                'From   200 -   300m asl', 'From   300 -   400m asl']
        res = cf_height_from_name(data)
        com = NAMECoord(name='altitude', dimension=None,
                        points=np.array([50., 150., 250., 350.]),
                        bounds=np.array([[0., 100.],
                                         [100., 200.],
                                         [200., 300.],
                                         [300., 400.]]))
        self.assertEqual(res, com)

    def test_flight_level(self):
        # Parse flight level name coordinate.
        data = ['From FL0 - FL10', 'From FL10 - FL20',
                'From FL20 - FL30', 'From FL30 - FL40']
        res = cf_height_from_name(data)
        com = NAMECoord(name='flight_level', dimension=None,
                        points=np.array([5., 15., 25., 35.]),
                        bounds=np.array([[0., 10.],
                                         [10., 20.],
                                         [20., 30.],
                                         [30., 40.]]))
        self.assertEqual(res, com)

    def test_malformed_agl(self):
        # Ensure that error raised on attempting to parse agl coordinate which
        # has additional information on it.
        data = ['From     0 -   100m agl should break',
                'From   100 -   200m agl cant interpret']
        with self.assertRaises(RuntimeError):
            cf_height_from_name(data)

    def test_malformed_flight_level(self):
        # Ensure that error raised on attempting to parse flight level
        # coordinate which has additional information on it.
        data = ['From     0 -   100m asl should break',
                'From   100 -   200m asl cant interpret']
        with self.assertRaises(RuntimeError):
            cf_height_from_name(data)

    def test_type_mix(self):
        # Check that error is raised when only partial parsing was possible.
        data = ['From     0 -   100m asl',
                'From FL0 - FL10']
        with self.assertRaises(RuntimeError):
            cf_height_from_name(data)

    def test_inconsistent_units(self):
        # Check that an error is raised when mixed unit interpretation results.
        data = ['From     0 -   100m asl',
                'From     0 -   100m agl']
        with self.assertRaises(TypeError):
            cf_height_from_name(data)

    def test_not_parsed(self):
        # Ensure that function returns None when no parsing is made at all.
        data = ['600m, 700m']
        res = cf_height_from_name(data)
        self.assertIs(res, None)


class Test_cf_height_from_name_NAMEIII(tests.IrisTest):
    def test_bounded_height_above_ground(self):
        # Test scalar value parse for agl.
        data = ['Z = 50.00000 m agl', 'Z = 100.00000 m agl']
        res = cf_height_from_name(data)
        com = NAMECoord(name='height', dimension=None,
                        points=np.array([50., 100.]),
                        bounds=None)
        self.assertEqual(res, com)

    def test_bounded_height_above_sea_level(self):
        # Test scalar value parse for asl.
        data = ['Z = 50.00000 m asl', 'Z = 100.00000 m asl']
        res = cf_height_from_name(data)
        com = NAMECoord(name='altitude', dimension=None,
                        points=np.array([50., 100.]),
                        bounds=None)
        self.assertEqual(res, com)

    def test_flight_level(self):
        # Test scalar value FL.
        data = ['Z = 50.0FL', 'Z = 100.0FL']
        res = cf_height_from_name(data)
        com = NAMECoord(name='flight_level', dimension=None,
                        points=np.array([50., 100.]),
                        bounds=None)
        self.assertEqual(res, com)

    def test_bounded_height_above_ground_integer(self):
        # Test scalar integer value parse for agl.
        data = ['Z = 50 m agl', 'Z = 100 m agl']
        res = cf_height_from_name(data)
        com = NAMECoord(name='height', dimension=None,
                        points=np.array([50., 100.]),
                        bounds=None)
        self.assertEqual(res, com)

    def test_flight_level_integer(self):
        # Test scalar integer value FL.
        data = ['Z = 50FL', 'Z = 100FL']
        res = cf_height_from_name(data)
        com = NAMECoord(name='flight_level', dimension=None,
                        points=np.array([50., 100.]),
                        bounds=None)
        self.assertEqual(res, com)


if __name__ == "__main__":
    tests.main()
