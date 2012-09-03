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
"""
Test the coordinate categorisation functions.
"""

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import numpy as np

import iris
import iris.coord_categorisation as ccat


class TestCategorisations(tests.IrisTest):
  
    def test_basic(self):
        #make a series of 'day numbers' for the time, that slide across month boundaries
        day_numbers =  np.arange(0, 600, 27, dtype=np.int32)
        
        cube = iris.cube.Cube(day_numbers, long_name='test cube', units='metres')

        #use day numbers as data values also (don't actually use this for anything)
        cube.data = day_numbers 
        
        time_coord = iris.coords.DimCoord(
            day_numbers, standard_name='time', units=iris.unit.Unit('days since epoch', 'gregorian'))
        cube.add_dim_coord(time_coord, 0)

        #add test coordinates for examples wanted    
        ccat.add_year(cube, time_coord)
        ccat.add_day_of_month(cube, 'time')    #NB test passing coord-name instead of coord itself

        ccat.add_month(cube, time_coord)
        ccat.add_month_shortname(cube, time_coord, name='month_short')
        ccat.add_month_fullname(cube, time_coord, name='month_full')
        ccat.add_month_number(cube, time_coord, name='month_number')
        
        ccat.add_weekday(cube, time_coord)
        ccat.add_weekday_number(cube, time_coord, name='weekday_number')
        ccat.add_weekday_shortname(cube, time_coord, name='weekday_short')
        ccat.add_weekday_fullname(cube, time_coord, name='weekday_full')

        ccat.add_season(cube, time_coord)
        ccat.add_season_number(cube, time_coord, name='season_number')
        ccat.add_season_month_initials(cube, time_coord, name='season_months')
        ccat.add_season_year(cube, time_coord, name='year_ofseason')
        
        #also test 'generic' categorisation interface
        def _month_in_quarter(coord, pt_value):
            date = coord.units.num2date(pt_value)
            return (date.month - 1) % 3

        ccat.add_categorised_coord(cube, 'month_in_quarter', time_coord, _month_in_quarter)

        for coord_name in ['month_number', 'month_in_quarter', 'weekday_number', 'season_number', 'year_ofseason', 'year', 'day']:
            cube.coord(coord_name).points = cube.coord(coord_name).points.astype(np.int64)

        #check values
        self.assertCML(cube, ('categorisation', 'quickcheck.cml'))


if __name__ == '__main__':
    tests.main()
