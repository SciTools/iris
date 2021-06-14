# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Test function :func:`iris.util.new_axis`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip
import copy
import unittest

import numpy as np

import iris
from iris._lazy_data import as_lazy_data
import iris.tests.stock as stock
from iris.util import new_axis


class Test(tests.IrisTest):
    def setUp(self):
        self.data = np.array([[1, 2], [1, 2]])
        self.cube = iris.cube.Cube(self.data)
        lat = iris.coords.DimCoord([1, 2], standard_name="latitude")
        lon = iris.coords.DimCoord([1, 2], standard_name="longitude")

        time = iris.coords.DimCoord([1], standard_name="time")
        wibble = iris.coords.AuxCoord([1], long_name="wibble")

        self.cube.add_dim_coord(lat, 0)
        self.cube.add_dim_coord(lon, 1)
        self.cube.add_aux_coord(time, None)
        self.cube.add_aux_coord(wibble, None)

        self.coords = {"lat": lat, "lon": lon, "time": time, "wibble": wibble}

    def _assert_cube_notis(self, cube_a, cube_b):
        for coord_a, coord_b in zip(cube_a.coords(), cube_b.coords()):
            self.assertIsNot(coord_a, coord_b)

        self.assertIsNot(cube_a.metadata, cube_b.metadata)

        for factory_a, factory_b in zip(
            cube_a.aux_factories, cube_b.aux_factories
        ):
            self.assertIsNot(factory_a, factory_b)

    def test_no_coord(self):
        # Providing no coordinate to promote.
        res = new_axis(self.cube)
        com = iris.cube.Cube(self.data[None])
        com.add_dim_coord(self.coords["lat"].copy(), 1)
        com.add_dim_coord(self.coords["lon"].copy(), 2)
        com.add_aux_coord(self.coords["time"].copy(), None)
        com.add_aux_coord(self.coords["wibble"].copy(), None)

        self.assertEqual(res, com)
        self._assert_cube_notis(res, self.cube)

    def test_scalar_dimcoord(self):
        # Providing a scalar coordinate to promote.
        res = new_axis(self.cube, "time")
        com = iris.cube.Cube(self.data[None])
        com.add_dim_coord(self.coords["lat"].copy(), 1)
        com.add_dim_coord(self.coords["lon"].copy(), 2)
        com.add_aux_coord(self.coords["time"].copy(), 0)
        com.add_aux_coord(self.coords["wibble"].copy(), None)

        self.assertEqual(res, com)
        self._assert_cube_notis(res, self.cube)

    def test_scalar_auxcoord(self):
        # Providing a scalar coordinate to promote.
        res = new_axis(self.cube, "wibble")
        com = iris.cube.Cube(self.data[None])
        com.add_dim_coord(self.coords["lat"].copy(), 1)
        com.add_dim_coord(self.coords["lon"].copy(), 2)
        com.add_aux_coord(self.coords["time"].copy(), None)
        com.add_aux_coord(self.coords["wibble"].copy(), 0)

        self.assertEqual(res, com)
        self._assert_cube_notis(res, self.cube)

    def test_maint_factory(self):
        # Ensure that aux factory persists.
        data = np.arange(12, dtype="i8").reshape((3, 4))

        orography = iris.coords.AuxCoord(
            [10, 25, 50, 5], standard_name="surface_altitude", units="m"
        )

        model_level = iris.coords.AuxCoord(
            [2, 1, 0], standard_name="model_level_number"
        )

        level_height = iris.coords.DimCoord(
            [100, 50, 10],
            long_name="level_height",
            units="m",
            attributes={"positive": "up"},
            bounds=[[150, 75], [75, 20], [20, 0]],
        )

        sigma = iris.coords.AuxCoord(
            [0.8, 0.9, 0.95],
            long_name="sigma",
            bounds=[[0.7, 0.85], [0.85, 0.97], [0.97, 1.0]],
        )

        hybrid_height = iris.aux_factory.HybridHeightFactory(
            level_height, sigma, orography
        )

        cube = iris.cube.Cube(
            data,
            standard_name="air_temperature",
            units="K",
            dim_coords_and_dims=[(level_height, 0)],
            aux_coords_and_dims=[(orography, 1), (model_level, 0), (sigma, 0)],
            aux_factories=[hybrid_height],
        )

        com = iris.cube.Cube(
            data[None],
            standard_name="air_temperature",
            units="K",
            dim_coords_and_dims=[(copy.copy(level_height), 1)],
            aux_coords_and_dims=[
                (copy.copy(orography), 2),
                (copy.copy(model_level), 1),
                (copy.copy(sigma), 1),
            ],
            aux_factories=[copy.copy(hybrid_height)],
        )
        res = new_axis(cube)

        self.assertEqual(res, com)
        self._assert_cube_notis(res, cube)

        # Check that factory dependencies are actual coords within the cube.
        # Addresses a former bug : https://github.com/SciTools/iris/pull/3263
        (factory,) = list(res.aux_factories)
        deps = factory.dependencies
        for dep_name, dep_coord in deps.items():
            coord_name = dep_coord.name()
            msg = (
                "Factory dependency {!r} is a coord named {!r}, "
                "but it is *not* the coord of that name in the new cube."
            )
            self.assertIs(
                dep_coord,
                res.coord(coord_name),
                msg.format(dep_name, coord_name),
            )

    def test_lazy_data(self):
        cube = iris.cube.Cube(as_lazy_data(self.data))
        cube.add_aux_coord(iris.coords.DimCoord([1], standard_name="time"))
        res = new_axis(cube, "time")
        self.assertTrue(cube.has_lazy_data())
        self.assertTrue(res.has_lazy_data())
        self.assertEqual(res.shape, (1,) + cube.shape)

    def test_masked_unit_array(self):
        cube = stock.simple_3d_mask()
        test_cube = cube[0, 0, 0]
        test_cube = new_axis(test_cube, "longitude")
        test_cube = new_axis(test_cube, "latitude")
        data_shape = test_cube.data.shape
        mask_shape = test_cube.data.mask.shape
        self.assertEqual(data_shape, mask_shape)


if __name__ == "__main__":
    unittest.main()
