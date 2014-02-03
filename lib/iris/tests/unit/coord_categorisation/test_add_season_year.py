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
"""Test function :func:`iris.coord_categorisation.add_season_year`."""


# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

from datetime import datetime

from iris.coord_categorisation import add_season_year
from iris.coords import AuxCoord
import iris.cube
import iris.unit


class Test_add_season_year(tests.IrisTest):

    def setUp(self):
        # Create a cube with an unbounded time coord.
        # The second point is at the start of "next year's" djf season.
        self.units = iris.unit.Unit("days since 2000-01-01 00:00:00")
        points = [self.units.date2num(datetime(2012, 9, 1)),
                  self.units.date2num(datetime(2012, 12, 1))]
        coord = AuxCoord(points, "time", units=self.units)
        self.cube = iris.cube.Cube([1, 2])
        self.cube.add_aux_coord(coord, 0)

    def test_point(self):
        add_season_year(self.cube, 'time')
        self.assertArrayEqual(self.cube.coord('season_year').points,
                              [2012, 2013])

    def test_bounds(self):
        bounds = [[self.units.date2num(datetime(2012, 6, 1)),
                   self.units.date2num(datetime(2012, 9, 1))],
                  [self.units.date2num(datetime(2012, 9, 1)),
                   self.units.date2num(datetime(2012, 12, 1))]]
        self.cube.coord('time').bounds = bounds
        add_season_year(self.cube, 'time')
        self.assertArrayEqual(self.cube.coord('season_year').points,
                              [2012, 2012])


if __name__ == '__main__':
    tests.main()
