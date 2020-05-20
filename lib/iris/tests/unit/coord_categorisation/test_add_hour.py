# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test coordinate categorisation function add_hour.
"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import cf_units
import numpy as np

import iris
import iris.coord_categorisation as ccat


class Test_add_hour(tests.IrisTest):
    def setUp(self):
        # make a series of 'hour numbers' for the time
        hour_numbers = np.arange(0, 200, 5, dtype=np.int32)

        # use hour numbers as data values also (don't actually use this for
        # anything)
        cube = iris.cube.Cube(
            hour_numbers, long_name="test cube", units="metres"
        )

        time_coord = iris.coords.DimCoord(
            hour_numbers,
            standard_name="time",
            units=cf_units.Unit("hours since epoch", "gregorian"),
        )
        cube.add_dim_coord(time_coord, 0)

        self.hour_numbers = hour_numbers
        self.cube = cube
        self.time_coord = time_coord

    def test_bad_coord(self):
        with self.assertRaises(iris.exceptions.CoordinateNotFoundError):
            ccat.add_hour(self.cube, "DOES NOT EXIST", name="my_hour")

    def test_explicit_result_name_specify_coord_by_name(self):
        coord_name = "my_hour"
        msg = "Missing/incorrectly named result for add_hour"

        # Specify source coordinate by name
        cube = self.cube
        ccat.add_hour(cube, "time", name=coord_name)
        result_coords = cube.coords(coord_name)
        self.assertEqual(len(result_coords), 1, msg)

    def test_explicit_result_name_specify_coord_by_reference(self):
        coord_name = "my_hour"
        msg = "Missing/incorrectly named result for add_hour"

        # Specify source coordinate by coordinate reference
        cube = self.cube
        time = cube.coord("time")
        ccat.add_hour(cube, time, name=coord_name)
        result_coords = cube.coords(coord_name)
        self.assertEqual(len(result_coords), 1, msg)

    def test_basic(self):
        coord_name = "my_hour"
        cube = self.cube
        time_coord = self.time_coord
        expected_coord = iris.coords.AuxCoord(
            self.hour_numbers % 24, long_name=coord_name, units="1"
        )

        ccat.add_hour(cube, time_coord, coord_name)

        self.assertEqual(cube.coord(coord_name), expected_coord)


if __name__ == "__main__":
    tests.main()
