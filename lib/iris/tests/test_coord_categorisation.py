# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test the coordinate categorisation functions.
"""

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests  # isort:skip

import warnings

import cf_units
import numpy as np

import iris
import iris.coord_categorisation as ccat

CATEGORISATION_FUNCS = (
    ccat.add_day_of_month,
    ccat.add_day_of_year,
    ccat.add_weekday,
    ccat.add_weekday_fullname,
    ccat.add_weekday_number,
    ccat.add_month,
    ccat.add_month_fullname,
    ccat.add_month_number,
    ccat.add_year,
    ccat.add_season,
    ccat.add_season_number,
    ccat.add_season_year,
    ccat.add_season_membership,
)


class TestCategorisations(tests.IrisTest):
    def setUp(self):
        # make a series of 'day numbers' for the time, that slide across month
        # boundaries
        day_numbers = np.arange(0, 600, 27, dtype=np.int32)

        cube = iris.cube.Cube(
            day_numbers, long_name="test cube", units="metres"
        )

        # use day numbers as data values also (don't actually use this for
        # anything)
        cube.data = day_numbers

        time_coord = iris.coords.DimCoord(
            day_numbers,
            standard_name="time",
            units=cf_units.Unit("days since epoch", "standard"),
        )
        cube.add_dim_coord(time_coord, 0)

        self.cube = cube
        self.time_coord = time_coord

    def test_bad_coord(self):
        for func in CATEGORISATION_FUNCS:
            kwargs = {"name": "my_category"}
            if func is ccat.add_season_membership:
                kwargs["season"] = "djf"
            with self.assertRaises(iris.exceptions.CoordinateNotFoundError):
                func(self.cube, "DOES NOT EXIST", **kwargs)

    def test_explicit_result_names(self):
        result_name = "my_category"
        fmt = "Missing/incorrectly named result for {0!r}"
        for func in CATEGORISATION_FUNCS:
            # Specify source coordinate by name
            cube = self.cube.copy()
            kwargs = {"name": result_name}
            if func is ccat.add_season_membership:
                kwargs["season"] = "djf"
            with warnings.catch_warnings(record=True):
                func(cube, "time", **kwargs)
            result_coords = cube.coords(result_name)
            self.assertEqual(len(result_coords), 1, fmt.format(func.__name__))
            # Specify source coordinate by coordinate reference
            cube = self.cube.copy()
            time = cube.coord("time")
            with warnings.catch_warnings(record=True):
                func(cube, time, **kwargs)
            result_coords = cube.coords(result_name)
            self.assertEqual(len(result_coords), 1, fmt.format(func.__name__))

    def test_basic(self):
        cube = self.cube
        time_coord = self.time_coord

        ccat.add_year(cube, time_coord, "my_year")
        ccat.add_day_of_month(cube, time_coord, "my_day_of_month")
        ccat.add_day_of_year(cube, time_coord, "my_day_of_year")

        ccat.add_month(cube, time_coord, "my_month")
        ccat.add_month_fullname(cube, time_coord, "my_month_fullname")
        ccat.add_month_number(cube, time_coord, "my_month_number")

        ccat.add_weekday(cube, time_coord, "my_weekday")
        ccat.add_weekday_number(cube, time_coord, "my_weekday_number")
        ccat.add_weekday_fullname(cube, time_coord, "my_weekday_fullname")

        ccat.add_season(cube, time_coord, "my_season")
        ccat.add_season_number(cube, time_coord, "my_season_number")
        ccat.add_season_year(cube, time_coord, "my_season_year")

        # also test 'generic' categorisation interface
        def _month_in_quarter(coord, pt_value):
            date = coord.units.num2date(pt_value)
            return (date.month - 1) % 3

        ccat.add_categorised_coord(
            cube, "my_month_in_quarter", time_coord, _month_in_quarter
        )

        # To ensure consistent results between 32-bit and 64-bit
        # platforms, ensure all the numeric categorisation coordinates
        # are always stored as int64.
        for coord in cube.coords():
            if coord.long_name is not None and coord.points.dtype.kind == "i":
                coord.points = coord.points.astype(np.int64)

        # check values
        self.assertCML(cube, ("categorisation", "quickcheck.cml"))

    def test_add_season_nonstandard(self):
        # season categorisations work for non-standard seasons?
        cube = self.cube
        time_coord = self.time_coord
        seasons = ["djfm", "amjj", "ason"]
        ccat.add_season(cube, time_coord, name="seasons", seasons=seasons)
        ccat.add_season_number(
            cube, time_coord, name="season_numbers", seasons=seasons
        )
        ccat.add_season_year(
            cube, time_coord, name="season_years", seasons=seasons
        )
        self.assertCML(cube, ("categorisation", "customcheck.cml"))

    def test_add_season_membership(self):
        # season membership identifies correct seasons?
        season = "djf"
        ccat.add_season_membership(self.cube, "time", season, name="in_season")
        ccat.add_season(self.cube, "time")
        coord_season = self.cube.coord("season")
        coord_membership = self.cube.coord("in_season")
        season_locations = np.where(coord_season.points == season)[0]
        membership_locations = np.where(coord_membership.points)[0]
        self.assertArrayEqual(membership_locations, season_locations)

    def test_add_season_invalid_spec(self):
        # custom seasons with an invalid season raises an error?
        seasons = ("djf", "maj", "jja", "son")  # MAJ not a season!
        for func in (
            ccat.add_season,
            ccat.add_season_year,
            ccat.add_season_number,
        ):
            with self.assertRaises(ValueError):
                func(self.cube, "time", name="my_category", seasons=seasons)

    def test_add_season_repeated_months(self):
        # custom seasons with repeated months raises an error?
        seasons = ("djfm", "mam", "jja", "son")
        for func in (
            ccat.add_season,
            ccat.add_season_year,
            ccat.add_season_number,
        ):
            with self.assertRaises(ValueError):
                func(self.cube, "time", name="my_category", seasons=seasons)

    def test_add_season_missing_months(self):
        # custom seasons with missing months raises an error?
        seasons = ("djfm", "amjj")
        for func in (
            ccat.add_season,
            ccat.add_season_year,
            ccat.add_season_number,
        ):
            with self.assertRaises(ValueError):
                func(self.cube, "time", name="my_category", seasons=seasons)

    def test_add_season_membership_invalid_spec(self):
        season = "maj"  # not a season!
        with self.assertRaises(ValueError):
            ccat.add_season_membership(
                self.cube, "time", season, name="maj_season"
            )


if __name__ == "__main__":
    tests.main()
