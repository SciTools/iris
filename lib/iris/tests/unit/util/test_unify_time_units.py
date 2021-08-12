# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Test function :func:`iris.util.array_equal`."""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import copy

import cf_units
import numpy as np

import iris
import iris.tests.stock as stock
from iris.util import unify_time_units


class Test(tests.IrisTest):
    def simple_1d_time_cubes(self, calendar="gregorian"):
        coord_points = [1, 2, 3, 4, 5]
        data_points = [273, 275, 278, 277, 274]
        reftimes = [
            "hours since 1970-01-01 00:00:00",
            "hours since 1970-01-02 00:00:00",
        ]
        list_of_cubes = []
        for reftime in reftimes:
            cube = iris.cube.Cube(
                np.array(data_points, dtype=np.float32),
                standard_name="air_temperature",
                units="K",
            )
            unit = cf_units.Unit(reftime, calendar=calendar)
            coord = iris.coords.DimCoord(
                points=np.array(coord_points, dtype=np.float32),
                standard_name="time",
                units=unit,
            )
            cube.add_dim_coord(coord, 0)
            list_of_cubes.append(cube)
        return list_of_cubes

    def _common(self, expected, result, coord_name="time"):
        # This tests time-like coords only.
        for cube in result:
            try:
                epoch = cube.coord(coord_name).units.origin
            except iris.exceptions.CoordinateNotFoundError:
                pass
            else:
                self.assertEqual(expected, epoch)

    def test_cubelist_with_time_coords(self):
        # Tests an :class:`iris.cube.CubeList` containing cubes with time
        # coords against a time string and a time coord.
        cubelist = iris.cube.CubeList(self.simple_1d_time_cubes())
        expected = "hours since 1970-01-01 00:00:00"
        unify_time_units(cubelist)
        self._common(expected, cubelist)

    def test_list_of_cubes_with_time_coords(self):
        # Tests an iterable containing cubes with time coords against a time
        # string and a time coord.
        list_of_cubes = self.simple_1d_time_cubes()
        expected = "hours since 1970-01-01 00:00:00"
        unify_time_units(list_of_cubes)
        self._common(expected, list_of_cubes)

    @tests.skip_data
    def test_no_time_coord_in_cubes(self):
        path0 = tests.get_data_path(("PP", "aPPglob1", "global.pp"))
        path1 = tests.get_data_path(("PP", "aPPglob1", "global_t_forecast.pp"))
        cube0 = iris.load_cube(path0)
        cube1 = iris.load_cube(path1)
        cubes = iris.cube.CubeList([cube0, cube1])
        result = copy.copy(cubes)
        unify_time_units(result)
        self.assertEqual(cubes, result)

    def test_time_coord_only_in_some_cubes(self):
        list_of_cubes = self.simple_1d_time_cubes()
        cube = stock.simple_2d()
        list_of_cubes.append(cube)
        expected = "hours since 1970-01-01 00:00:00"
        unify_time_units(list_of_cubes)
        self._common(expected, list_of_cubes)

    def test_multiple_time_coords_in_cube(self):
        cube0, cube1 = self.simple_1d_time_cubes()
        units = cf_units.Unit(
            "days since 1980-05-02 00:00:00", calendar="gregorian"
        )
        aux_coord = iris.coords.AuxCoord(
            72, standard_name="forecast_reference_time", units=units
        )
        cube1.add_aux_coord(aux_coord)
        cubelist = iris.cube.CubeList([cube0, cube1])
        expected = "hours since 1970-01-01 00:00:00"
        unify_time_units(cubelist)
        self._common(expected, cubelist)
        self._common(expected, cubelist, coord_name="forecast_reference_time")

    def test_multiple_calendars(self):
        cube0, cube1 = self.simple_1d_time_cubes()
        cube2, cube3 = self.simple_1d_time_cubes(calendar="360_day")
        cubelist = iris.cube.CubeList([cube0, cube1, cube2, cube3])
        expected = "hours since 1970-01-01 00:00:00"
        unify_time_units(cubelist)
        self._common(expected, cubelist)


if __name__ == "__main__":
    tests.main()
