# (C) British Crown Copyright 2010 - 2013, Met Office
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

import warnings

import numpy as np

import iris
import iris.coord_categorisation as ccat


OK_DEFAULTS = (
    ccat.add_year,
    ccat.add_month,
    ccat.add_weekday,
    ccat.add_season,
)


DEPRECATED_DEFAULTS = (
    ccat.add_month_number,
    ccat.add_month_fullname,
    ccat.add_day_of_month,
    ccat.add_day_of_year,
    ccat.add_weekday_number,
    ccat.add_weekday_fullname,
    ccat.add_season_number,
    ccat.add_season_year,
)


DEPRECATED = tuple()


class TestCategorisations(tests.IrisTest):
    def setUp(self):
        # make a series of 'day numbers' for the time, that slide across month
        # boundaries
        day_numbers = np.arange(0, 600, 27, dtype=np.int32)

        cube = iris.cube.Cube(
            day_numbers, long_name='test cube', units='metres')

        # use day numbers as data values also (don't actually use this for
        # anything)
        cube.data = day_numbers

        time_coord = iris.coords.DimCoord(
            day_numbers, standard_name='time',
            units=iris.unit.Unit('days since epoch', 'gregorian'))
        cube.add_dim_coord(time_coord, 0)

        self.cube = cube
        self.time_coord = time_coord

    def test_bad_coord(self):
        for func in OK_DEFAULTS + DEPRECATED_DEFAULTS + DEPRECATED:
            with self.assertRaises(iris.exceptions.CoordinateNotFoundError):
                func(self.cube, 'DOES NOT EXIST', 'my_category')

    def test_deprecateds(self):
        no_warning = 'Missing deprecation warning for {0!r}'
        no_result = 'Missing/incorrectly named result for {0!r}'

        def check_deprecated(result_name, func, args=()):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                cube = self.cube.copy()
                func(cube, 'time', *args)
                self.assertEqual(len(w), 1, no_warning.format(func.func_name))
                result_coords = cube.coords(result_name)
                self.assertEqual(len(result_coords), 1,
                                 no_result.format(func.func_name))

        for func in DEPRECATED_DEFAULTS + DEPRECATED:
            if func.func_name == 'add_season_year':
                result_name = 'year'
            else:
                result_name = func.func_name.split('_')[1]
            check_deprecated(result_name, func)

        unexpected = 'Unexpected deprecation warning for {0!r}'
        for func in OK_DEFAULTS:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                cube = self.cube.copy()
                func(cube, 'time')
                self.assertEqual(len(w), 0, unexpected.format(func.func_name))
                result_name = func.func_name.split('_')[1]
                result_coords = cube.coords(result_name)
                self.assertEqual(len(result_coords), 1,
                                 no_result.format(func.func_name))

    def test_explicit_result_names(self):
        result_name = 'my_category'
        fmt = 'Missing/incorrectly named result for {0!r}'
        for func in OK_DEFAULTS + DEPRECATED_DEFAULTS + DEPRECATED:
            # Specify source coordinate by name
            cube = self.cube.copy()
            with warnings.catch_warnings(record=True):
                func(cube, 'time', result_name)
            result_coords = cube.coords(result_name)
            self.assertEqual(len(result_coords), 1, fmt.format(func.func_name))
            # Specify source coordinate by coordinate reference
            cube = self.cube.copy()
            time = cube.coord('time')
            with warnings.catch_warnings(record=True):
                func(cube, time, result_name)
            result_coords = cube.coords(result_name)
            self.assertEqual(len(result_coords), 1, fmt.format(func.func_name))

    def test_basic(self):
        cube = self.cube
        time_coord = self.time_coord

        ccat.add_year(cube, time_coord, 'my_year')
        ccat.add_day_of_month(cube, time_coord, 'my_day_of_month')
        ccat.add_day_of_year(cube, time_coord, 'my_day_of_year')

        ccat.add_month(cube, time_coord, 'my_month')
        ccat.add_month_fullname(cube, time_coord, 'my_month_fullname')
        ccat.add_month_number(cube, time_coord, 'my_month_number')

        ccat.add_weekday(cube, time_coord, 'my_weekday')
        ccat.add_weekday_number(cube, time_coord, 'my_weekday_number')
        ccat.add_weekday_fullname(cube, time_coord, 'my_weekday_fullname')

        ccat.add_season(cube, time_coord, 'my_season')
        ccat.add_season_number(cube, time_coord, 'my_season_number')
        ccat.add_season_year(cube, time_coord, 'my_season_year')

        # also test 'generic' categorisation interface
        def _month_in_quarter(coord, pt_value):
            date = coord.units.num2date(pt_value)
            return (date.month - 1) % 3

        ccat.add_categorised_coord(cube,
                                   'my_month_in_quarter',
                                   time_coord,
                                   _month_in_quarter)

        # To ensure consistent results between 32-bit and 64-bit
        # platforms, ensure all the numeric categorisation coordinates
        # are always stored as int64.
        for coord in cube.coords():
            if coord.long_name is not None and coord.points.dtype.kind == 'i':
                coord.points = coord.points.astype(np.int64)

        # check values
        self.assertCML(cube, ('categorisation', 'quickcheck.cml'))

    def test_add_season_nonstandard(self):
        # season categorisations work for non-standard seasons?
        cube = self.cube
        time_coord = self.time_coord
        seasons = ['djfm', 'amjj', 'ason']
        ccat.add_season(cube, time_coord, name='seasons', seasons=seasons)
        ccat.add_season_number(cube, time_coord, name='season_numbers',
                               seasons=seasons)
        ccat.add_season_year(cube, time_coord, name='season_years',
                             seasons=seasons)
        self.assertCML(cube, ('categorisation', 'customcheck.cml'))

    def test_add_season_membership(self):
        # season membership identifies correct seasons?
        season = 'djf'
        ccat.add_season_membership(self.cube, 'time', season,
                                   name='in_season')
        ccat.add_season(self.cube, 'time')
        coord_season = self.cube.coord('season')
        coord_membership = self.cube.coord('in_season')
        season_locations = np.where(coord_season.points == season)[0]
        membership_locations = np.where(coord_membership.points)[0]
        self.assertArrayEqual(membership_locations, season_locations)

    def test_add_season_invalid_spec(self):
        # custom seasons with an invalid season raises an error?
        seasons = ('djf', 'maj', 'jja', 'son')   # MAJ not a season!
        for func in (ccat.add_season, ccat.add_season_year,
                     ccat.add_season_number):
            with self.assertRaises(ValueError):
                func(self.cube, 'time', name='my_category', seasons=seasons)

    def test_add_season_repeated_months(self):
        # custom seasons with repeated months raises an error?
        seasons = ('djfm', 'mam', 'jja', 'son')
        for func in (ccat.add_season, ccat.add_season_year,
                     ccat.add_season_number):
            with self.assertRaises(ValueError):
                func(self.cube, 'time', name='my_category', seasons=seasons)

    def test_add_season_missing_months(self):
        # custom seasons with missing months raises an error?
        seasons = ('djfm', 'amjj')
        for func in (ccat.add_season, ccat.add_season_year,
                     ccat.add_season_number):
            with self.assertRaises(ValueError):
                func(self.cube, 'time', name='my_category', seasons=seasons)

    def test_add_season_membership_invalid_spec(self):
        season = 'maj'   # not a season!
        with self.assertRaises(ValueError):
            ccat.add_season_membership(self.cube, 'time', season,
                                       name='maj_season')


if __name__ == '__main__':
    tests.main()
