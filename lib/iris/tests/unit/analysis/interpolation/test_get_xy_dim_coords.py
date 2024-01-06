# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris.analysis._interpolation.get_xy_dim_coords`."""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests  # isort:skip

import copy

import numpy as np

from iris.analysis._interpolation import get_xy_dim_coords
import iris.coord_systems
import iris.coords
import iris.experimental.regrid
import iris.tests.stock


class TestGetXYCoords(tests.IrisTest):
    @tests.skip_data
    def test_grid_lat_lon(self):
        cube = iris.tests.stock.realistic_4d()
        x, y = get_xy_dim_coords(cube)
        self.assertIs(x, cube.coord("grid_longitude"))
        self.assertIs(y, cube.coord("grid_latitude"))

    def test_lat_lon(self):
        cube = iris.tests.stock.lat_lon_cube()
        x, y = get_xy_dim_coords(cube)
        self.assertIs(x, cube.coord("longitude"))
        self.assertIs(y, cube.coord("latitude"))

    def test_projection_coords(self):
        cube = iris.tests.stock.lat_lon_cube()
        cube.coord("longitude").rename("projection_x_coordinate")
        cube.coord("latitude").rename("projection_y_coordinate")
        x, y = get_xy_dim_coords(cube)
        self.assertIs(x, cube.coord("projection_x_coordinate"))
        self.assertIs(y, cube.coord("projection_y_coordinate"))

    @tests.skip_data
    def test_missing_x_coord(self):
        cube = iris.tests.stock.realistic_4d()
        cube.remove_coord("grid_longitude")
        with self.assertRaises(ValueError):
            get_xy_dim_coords(cube)

    @tests.skip_data
    def test_missing_y_coord(self):
        cube = iris.tests.stock.realistic_4d()
        cube.remove_coord("grid_latitude")
        with self.assertRaises(ValueError):
            get_xy_dim_coords(cube)

    @tests.skip_data
    def test_multiple_coords(self):
        cube = iris.tests.stock.realistic_4d()
        cs = iris.coord_systems.GeogCS(6371229)
        time_coord = cube.coord("time")
        time_dims = cube.coord_dims(time_coord)
        lat_coord = iris.coords.DimCoord(
            np.arange(time_coord.shape[0]),
            standard_name="latitude",
            units="degrees",
            coord_system=cs,
        )
        cube.remove_coord(time_coord)
        cube.add_dim_coord(lat_coord, time_dims)
        model_level_coord = cube.coord("model_level_number")
        model_level_dims = cube.coord_dims(model_level_coord)
        lon_coord = iris.coords.DimCoord(
            np.arange(model_level_coord.shape[0]),
            standard_name="longitude",
            units="degrees",
            coord_system=cs,
        )
        cube.remove_coord(model_level_coord)
        cube.add_dim_coord(lon_coord, model_level_dims)

        with self.assertRaises(ValueError):
            get_xy_dim_coords(cube)

        cube.remove_coord("grid_latitude")
        cube.remove_coord("grid_longitude")

        x, y = get_xy_dim_coords(cube)
        self.assertIs(x, lon_coord)
        self.assertIs(y, lat_coord)

    def test_no_coordsystem(self):
        cube = iris.tests.stock.lat_lon_cube()
        for coord in cube.coords():
            coord.coord_system = None
        x, y = get_xy_dim_coords(cube)
        self.assertIs(x, cube.coord("longitude"))
        self.assertIs(y, cube.coord("latitude"))

    def test_one_coordsystem(self):
        cube = iris.tests.stock.lat_lon_cube()
        cube.coord("longitude").coord_system = None
        with self.assertRaises(ValueError):
            get_xy_dim_coords(cube)

    def test_different_coordsystem(self):
        cube = iris.tests.stock.lat_lon_cube()

        lat_cs = copy.copy(cube.coord("latitude").coord_system)
        lat_cs.semi_major_axis = 7000000
        cube.coord("latitude").coord_system = lat_cs

        lon_cs = copy.copy(cube.coord("longitude").coord_system)
        lon_cs.semi_major_axis = 7000001
        cube.coord("longitude").coord_system = lon_cs

        with self.assertRaises(ValueError):
            get_xy_dim_coords(cube)


if __name__ == "__main__":
    tests.main()
