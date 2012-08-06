# (C) British Crown Copyright 2010 - 2012, Met Office
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


from __future__ import division

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import logging

import iris.coords
import iris.unit


logger = logging.getLogger('tests')

class TestCoordSystemSame(tests.IrisTest):

    def setUp(self):
        self.cs1 = iris.coord_systems.HorizontalCS(None)
        self.cs2 = iris.coord_systems.HorizontalCS(None)
        self.cs3 = iris.coord_systems.LatLonCS(None, None, None, None)

    def test_simple(self):
        a = self.cs1
        b = self.cs2
        self.assertEquals(a, b)

    def test_different_class(self):
        a = self.cs1
        b = self.cs3
        self.assertNotEquals(a, b)

    def test_different_public_attributes(self):
        a = self.cs1
        b = self.cs2
        a.foo = 'a'

        # check that that attribute was added (just in case)
        self.assertEqual(a.foo, 'a')

        # a and b should not be the same
        self.assertNotEquals(a, b)

        # a and b should be the same
        b.foo = 'a'
        self.assertEquals(a, b)

        b.foo = 'b'
        # a and b should not be the same
        self.assertNotEquals(a, b)


if __name__ == "__main__":
    tests.main()
