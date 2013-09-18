# (C) British Crown Copyright 2015, Met Office
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
"""Unit tests for the :class:`iris.coord_systems.LambertConformal` class."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris.coord_systems import LambertConformal


class Test__init__(tests.IrisTest):

    def test_north_cutoff(self):
        lcc = LambertConformal(0, 0, standard_parallels=(30, 60))
        ccrs = lcc.as_cartopy_crs()
        self.assertEqual(ccrs.cutoff, -30)

    def test_south_cutoff(self):
        lcc = LambertConformal(0, 0, standard_parallels=(-30, -60))
        ccrs = lcc.as_cartopy_crs()
        self.assertEqual(ccrs.cutoff, 30)

    def test_default(self):
        lcc = LambertConformal()
        self.assertNotEqual(lcc.standard_parallels[0], 0.0)
        self.assertNotEqual(lcc.standard_parallels[1], 0.0)
        self.assertNotEqual(lcc.central_lat, 0.0)
        self.assertNotEqual(lcc.central_lon, 0.0)

    def test_modified_centre(self):
        lcc = LambertConformal(75.2, -34.5)
        self.assertEqual(lcc.central_lat, 75.2)
        self.assertEqual(lcc.central_lon, -34.5)

    def test_modified_parallels(self):
        lcc = LambertConformal(standard_parallels=(25.4, 57.3))
        self.assertEqual(lcc.standard_parallels[0], 25.4)
        self.assertEqual(lcc.standard_parallels[1], 57.3)

    def test_1sp(self):
        lcc = LambertConformal(standard_parallels=43.2)
        self.assertEqual(lcc.standard_parallels, [43.2])


if __name__ == "__main__":
    tests.main()
