# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for concatenating cubes with differing time coord epochs
using :func:`iris.util.unify_time_units`.

"""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests  # isort:skip

import cf_units
import dask.array as da
import numpy as np

from iris._concatenate import _DerivedCoordAndDims, concatenate
import iris.aux_factory
import iris.coords
import iris.cube
import iris.tests.stock as stock
from iris.util import unify_time_units


class Test_DerivedCoordAndDims:
    def test_equal(self):
        assert _DerivedCoordAndDims(
            "coord", "dims", "aux_factory"
        ) == _DerivedCoordAndDims("coord", "dims", "aux_factory")

    def test_non_equal_coord(self):
        assert _DerivedCoordAndDims(
            "coord_0", "dims", "aux_factory"
        ) != _DerivedCoordAndDims("coord_1", "dims", "aux_factory")

    def test_non_equal_dims(self):
        assert _DerivedCoordAndDims(
            "coord", "dims_0", "aux_factory"
        ) != _DerivedCoordAndDims("coord", "dims_1", "aux_factory")

    def test_non_equal_aux_factory(self):
        # Note: aux factories are not taken into account for equality!
        assert _DerivedCoordAndDims(
            "coord", "dims", "aux_factory_0"
        ) == _DerivedCoordAndDims("coord", "dims", "aux_factory_1")

    def test_non_equal_types(self):
        assert (
            _DerivedCoordAndDims("coord", "dims", "aux_factory")
            != "I am not a _DerivedCoordAndDims"
        )


class Test_concatenate__epoch(tests.IrisTest):
    def simple_1d_time_cubes(self, reftimes, coords_points):
        cubes = []
        data_points = [273, 275, 278, 277, 274]
        for reftime, coord_points in zip(reftimes, coords_points):
            cube = iris.cube.Cube(
                np.array(data_points, dtype=np.float32),
                standard_name="air_temperature",
                units="K",
            )
            unit = cf_units.Unit(reftime, calendar="standard")
            coord = iris.coords.DimCoord(
                points=np.array(coord_points, dtype=np.float32),
                standard_name="time",
                units=unit,
            )
            cube.add_dim_coord(coord, 0)
            cubes.append(cube)
        return cubes

    def test_concat_1d_with_differing_time_units(self):
        reftimes = [
            "hours since 1970-01-01 00:00:00",
            "hours since 1970-01-02 00:00:00",
        ]
        coords_points = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        cubes = self.simple_1d_time_cubes(reftimes, coords_points)
        unify_time_units(cubes)
        result = concatenate(cubes)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (10,))


class Test_cubes_with_aux_coord(tests.IrisTest):
    def create_cube(self):
        data = np.arange(4).reshape(2, 2)

        lat = iris.coords.DimCoord([0, 30], standard_name="latitude", units="degrees")
        lon = iris.coords.DimCoord([0, 15], standard_name="longitude", units="degrees")
        height = iris.coords.AuxCoord([1.5], standard_name="height", units="m")
        t_unit = cf_units.Unit("hours since 1970-01-01 00:00:00", calendar="standard")
        time = iris.coords.DimCoord([0, 6], standard_name="time", units=t_unit)

        cube = iris.cube.Cube(data, standard_name="air_temperature", units="K")
        cube.add_dim_coord(time, 0)
        cube.add_dim_coord(lat, 1)
        cube.add_aux_coord(lon, 1)
        cube.add_aux_coord(height)
        return cube

    def test_diff_aux_coord(self):
        cube_a = self.create_cube()
        cube_b = cube_a.copy()
        cube_b.coord("time").points = [12, 18]
        cube_b.coord("longitude").points = [120, 150]

        result = concatenate([cube_a, cube_b])
        self.assertEqual(len(result), 2)

    def test_ignore_diff_aux_coord(self):
        cube_a = self.create_cube()
        cube_b = cube_a.copy()
        cube_b.coord("time").points = [12, 18]
        cube_b.coord("longitude").points = [120, 150]

        result = concatenate([cube_a, cube_b], check_aux_coords=False)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (4, 2))


class Test_cubes_with_cell_measure(tests.IrisTest):
    def create_cube(self):
        data = np.arange(4).reshape(2, 2)

        lat = iris.coords.DimCoord([0, 30], standard_name="latitude", units="degrees")
        volume = iris.coords.CellMeasure([0, 15], measure="volume", long_name="volume")
        area = iris.coords.CellMeasure([1.5], standard_name="height", units="m")
        t_unit = cf_units.Unit("hours since 1970-01-01 00:00:00", calendar="standard")
        time = iris.coords.DimCoord([0, 6], standard_name="time", units=t_unit)

        cube = iris.cube.Cube(data, standard_name="air_temperature", units="K")
        cube.add_dim_coord(time, 0)
        cube.add_dim_coord(lat, 1)
        cube.add_cell_measure(volume, 1)
        cube.add_cell_measure(area)
        return cube

    def test_diff_cell_measure(self):
        cube_a = self.create_cube()
        cube_b = cube_a.copy()
        cube_b.coord("time").points = [12, 18]
        cube_b.cell_measure("volume").data = [120, 150]

        result = concatenate([cube_a, cube_b])
        self.assertEqual(len(result), 2)

    def test_ignore_diff_cell_measure(self):
        cube_a = self.create_cube()
        cube_b = cube_a.copy()
        cube_b.coord("time").points = [12, 18]
        cube_b.cell_measure("volume").data = [120, 150]

        result = concatenate([cube_a, cube_b], check_cell_measures=False)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (4, 2))

    def test_lazy_cell_measure(self):
        cube_a = self.create_cube()
        cube_a.cell_measure("volume").data = da.array([0, 15])
        cube_b = cube_a.copy()
        cube_b.coord("time").points = [12, 18]

        result = concatenate([cube_a, cube_b])
        assert len(result) == 1
        assert result[0].cell_measure("volume").has_lazy_data()


class Test_cubes_with_ancillary_variables(tests.IrisTest):
    def create_cube(self):
        data = np.arange(4).reshape(2, 2)

        lat = iris.coords.DimCoord([0, 30], standard_name="latitude", units="degrees")
        quality = iris.coords.AncillaryVariable([0, 15], long_name="quality")
        height = iris.coords.AncillaryVariable([1.5], standard_name="height", units="m")
        t_unit = cf_units.Unit("hours since 1970-01-01 00:00:00", calendar="standard")
        time = iris.coords.DimCoord([0, 6], standard_name="time", units=t_unit)

        cube = iris.cube.Cube(data, standard_name="air_temperature", units="K")
        cube.add_dim_coord(time, 0)
        cube.add_dim_coord(lat, 1)
        cube.add_ancillary_variable(quality, 1)
        cube.add_ancillary_variable(height)
        return cube

    def test_diff_ancillary_variables(self):
        cube_a = self.create_cube()
        cube_b = cube_a.copy()
        cube_b.coord("time").points = [12, 18]
        cube_b.ancillary_variable("quality").data = [120, 150]

        result = concatenate([cube_a, cube_b])
        self.assertEqual(len(result), 2)

    def test_ignore_diff_ancillary_variables(self):
        cube_a = self.create_cube()
        cube_b = cube_a.copy()
        cube_b.coord("time").points = [12, 18]
        cube_b.ancillary_variable("quality").data = [120, 150]

        result = concatenate([cube_a, cube_b], check_ancils=False)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (4, 2))

    def test_lazy_ancillary_variables(self):
        cube_a = self.create_cube()
        cube_a.ancillary_variable("quality").data = da.array([0, 15])
        cube_b = cube_a.copy()
        cube_b.coord("time").points = [12, 18]

        result = concatenate([cube_a, cube_b])
        assert len(result) == 1
        assert result[0].ancillary_variable("quality").has_lazy_data()


class Test_cubes_with_derived_coord(tests.IrisTest):
    def create_cube(self):
        data = np.arange(4).reshape(2, 2)
        aux_factories = []

        # DimCoords
        sigma = iris.coords.DimCoord([0.0, 10.0], var_name="sigma", units="1")
        t_unit = cf_units.Unit("hours since 1970-01-01 00:00:00", calendar="standard")
        time = iris.coords.DimCoord([0, 6], standard_name="time", units=t_unit)

        # AtmosphereSigmaFactory (does not span concatenated dim)
        ptop = iris.coords.AuxCoord(100.0, var_name="ptop", units="Pa")
        surface_p = iris.coords.AuxCoord([1.0, 2.0], var_name="ps", units="Pa")
        aux_factories.append(
            iris.aux_factory.AtmosphereSigmaFactory(ptop, sigma, surface_p)
        )

        # HybridHeightFactory (span concatenated dim)
        delta = iris.coords.AuxCoord(10.0, var_name="delta", units="m")
        orog = iris.coords.AuxCoord(data, var_name="orog", units="m")
        aux_factories.append(iris.aux_factory.HybridHeightFactory(delta, sigma, orog))

        dim_coords_and_dims = [(time, 0), (sigma, 1)]
        aux_coords_and_dims = [
            (ptop, ()),
            (delta, ()),
            (surface_p, 1),
            (orog, (0, 1)),
        ]

        cube = iris.cube.Cube(
            data,
            standard_name="air_temperature",
            units="K",
            dim_coords_and_dims=dim_coords_and_dims,
            aux_coords_and_dims=aux_coords_and_dims,
            aux_factories=aux_factories,
        )
        return cube

    def test_equal_derived_coords(self):
        cube_a = self.create_cube()
        cube_b = cube_a.copy()
        cube_b.coord("time").points = [12, 18]

        result = concatenate([cube_a, cube_b])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (4, 2))

        np.testing.assert_allclose(
            result[0].coord("air_pressure").points, [100.0, -880.0]
        )
        np.testing.assert_allclose(
            result[0].coord("altitude").points,
            [[10.0, 20.0], [10.0, 40.0], [10.0, 20.0], [10.0, 40.0]],
        )

        # Make sure indexing the resulting cube works correctly
        # (see https://github.com/SciTools/iris/issues/5339)
        self.assertEqual(result[0][0].shape, (2,))

        # Make sure ALL aux factory dependencies of the resulting cube were
        # properly updated (i.e., they are different from the original cubes).
        for aux_factory in result[0].aux_factories:
            for coord in aux_factory.dependencies.values():
                self.assertNotEqual(id(coord), id(cube_a.coord(coord.name())))
                self.assertNotEqual(id(coord), id(cube_b.coord(coord.name())))

    def test_equal_derived_coords_with_bounds(self):
        cube_a = self.create_cube()
        cube_a.coord("sigma").bounds = [[0.0, 5.0], [5.0, 20.0]]
        cube_b = cube_a.copy()
        cube_b.coord("time").points = [12, 18]

        result = concatenate([cube_a, cube_b])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (4, 2))

        np.testing.assert_allclose(
            result[0].coord("air_pressure").bounds,
            [[100.0, -395.0], [-390.0, -1860.0]],
        )

    def test_diff_altitude(self):
        """Gives one cube since altitude spans concatenation dim."""
        cube_a = self.create_cube()
        cube_b = cube_a.copy()
        cube_b.coord("time").points = [12, 18]
        cube_b.coord("orog").points = [[0, 0], [0, 0]]

        result = concatenate([cube_a, cube_b])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (4, 2))

        np.testing.assert_allclose(
            result[0].coord("altitude").points,
            [[10.0, 20.0], [10.0, 40.0], [10.0, 10.0], [10.0, 10.0]],
        )

    def test_diff_air_pressure(self):
        """Gives two cubes since altitude does not span concatenation dim."""
        cube_a = self.create_cube()
        cube_b = cube_a.copy()
        cube_b.coord("time").points = [12, 18]
        cube_b.coord("ps").points = [10.0, 20.0]

        result = concatenate([cube_a, cube_b], check_aux_coords=False)
        self.assertEqual(len(result), 2)

    def test_ignore_diff_air_pressure(self):
        cube_a = self.create_cube()
        cube_b = cube_a.copy()
        cube_b.coord("time").points = [12, 18]
        cube_b.coord("ps").points = [10.0, 20.0]

        result = concatenate(
            [cube_a, cube_b],
            check_aux_coords=False,
            check_derived_coords=False,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (4, 2))

        np.testing.assert_allclose(
            result[0].coord("air_pressure").points, [100.0, -880.0]
        )


class Test_anonymous_dims(tests.IrisTest):
    def setUp(self):
        data = np.arange(12).reshape(2, 3, 2)
        self.cube = iris.cube.Cube(data, standard_name="air_temperature", units="K")

        # Time coord
        t_unit = cf_units.Unit("hours since 1970-01-01 00:00:00", calendar="standard")
        t_coord = iris.coords.DimCoord([0, 6], standard_name="time", units=t_unit)
        self.cube.add_dim_coord(t_coord, 0)

        # Lats and lons
        self.x_coord = iris.coords.DimCoord(
            [15, 30], standard_name="longitude", units="degrees"
        )
        self.y_coord = iris.coords.DimCoord(
            [0, 30, 60], standard_name="latitude", units="degrees"
        )
        self.x_coord_2D = iris.coords.AuxCoord(
            [[0, 15], [30, 45], [60, 75]],
            standard_name="longitude",
            units="degrees",
        )
        self.y_coord_non_monotonic = iris.coords.AuxCoord(
            [0, 30, 15], standard_name="latitude", units="degrees"
        )

    def test_matching_2d_longitudes(self):
        cube1 = self.cube
        cube1.add_dim_coord(self.y_coord, 1)
        cube1.add_aux_coord(self.x_coord_2D, (1, 2))

        cube2 = cube1.copy()
        cube2.coord("time").points = [12, 18]
        result = concatenate([cube1, cube2])
        self.assertEqual(len(result), 1)

    def test_differing_2d_longitudes(self):
        cube1 = self.cube
        cube1.add_aux_coord(self.y_coord, 1)
        cube1.add_aux_coord(self.x_coord_2D, (1, 2))

        cube2 = cube1.copy()
        cube2.coord("time").points = [12, 18]
        cube2.coord("longitude").points = [[-30, -15], [0, 15], [30, 45]]

        result = concatenate([cube1, cube2])
        self.assertEqual(len(result), 2)

    def test_matching_non_monotonic_latitudes(self):
        cube1 = self.cube
        cube1.add_aux_coord(self.y_coord_non_monotonic, 1)
        cube1.add_aux_coord(self.x_coord, 2)

        cube2 = cube1.copy()
        cube2.coord("time").points = [12, 18]

        result = concatenate([cube1, cube2])
        self.assertEqual(len(result), 1)

    def test_differing_non_monotonic_latitudes(self):
        cube1 = self.cube
        cube1.add_aux_coord(self.y_coord_non_monotonic, 1)
        cube1.add_aux_coord(self.x_coord, 2)

        cube2 = cube1.copy()
        cube2.coord("time").points = [12, 18]
        cube2.coord("latitude").points = [30, 0, 15]

        result = concatenate([cube1, cube2])
        self.assertEqual(len(result), 2)

    def test_concatenate_along_anon_dim(self):
        cube1 = self.cube
        cube1.add_aux_coord(self.y_coord_non_monotonic, 1)
        cube1.add_aux_coord(self.x_coord, 2)

        cube2 = cube1.copy()
        cube2.coord("latitude").points = [30, 0, 15]

        result = concatenate([cube1, cube2])
        self.assertEqual(len(result), 2)


class Test_anonymous_dims_alternate_mapping(tests.IrisTest):
    # Ensure that anonymous concatenation is not sensitive to dimension mapping
    # of the anonymous dimension.
    def setUp(self):
        self.cube = stock.simple_3d()
        coord = self.cube.coord("wibble")
        self.cube.remove_coord(coord)
        self.cube.add_aux_coord(coord, 0)

    def test_concatenate_anom_1st_dim(self):
        # Check that concatenation along a non anonymous dimension is
        # insensitive to the dimension which is anonymous.
        # Concatenate along longitude.
        # DIM: cube(--, lat, lon)   & cube(--, lat, lon')
        # AUX: cube(wibble, --, --) & cube(wibble, --, --)
        cube1 = self.cube[..., :2]
        cube2 = self.cube[..., 2:]
        result = concatenate([cube1, cube2])
        self.assertEqual(len(result), 1)

    def test_concatenate_anom_2nd_dim(self):
        # Check that concatenation along a non anonymous dimension is
        # insensitive to the dimension which is anonymous.
        # Concatenate along longitude.
        # DIM: cube(lon, --, lat)   & cube(lon', ---, lat)
        # AUX: cube(--, wibble, --) & cube(--, wibble, --)
        cube1 = self.cube[..., :2]
        cube2 = self.cube[..., 2:]
        cube1.transpose((2, 0, 1))
        cube2.transpose((2, 0, 1))
        result = concatenate([cube1, cube2])
        self.assertEqual(len(result), 1)

    def test_concatenate_anom_3rd_dim(self):
        # Check that concatenation along a non anonymous dimension is
        # insensitive to the dimension which is anonymous.
        # Concatenate along longitude.
        # DIM: cube(lat, lon, --)   & cube(lat, lon', --)
        # AUX: cube(--, --, wibble) & cube(--, --, wibble)
        cube1 = self.cube[..., :2]
        cube2 = self.cube[..., 2:]
        cube1.transpose((1, 2, 0))
        cube2.transpose((1, 2, 0))
        result = concatenate([cube1, cube2])
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    tests.main()
