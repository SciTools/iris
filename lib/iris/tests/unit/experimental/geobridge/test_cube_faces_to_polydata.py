# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.experimental.geobridge.cube_faces_to_polydata` function."""


from unittest.mock import Mock

from geovista import Transform
import numpy as np
import pytest

import iris.analysis.cartography
import iris.coord_systems
from iris.experimental.geobridge import cube_faces_to_polydata
from iris.tests.stock import lat_lon_cube, sample_2d_latlons
from iris.tests.stock.mesh import sample_mesh_cube


@pytest.fixture()
def cube_mesh():
    return sample_mesh_cube()


@pytest.fixture()
def cube_1d():
    sample_1d_cube = lat_lon_cube()
    for coord in sample_1d_cube.dim_coords:
        coord.coord_system = None
    return sample_1d_cube


@pytest.fixture()
def cube_2d():
    return sample_2d_latlons()


class ParentClass:
    @pytest.fixture()
    def expected(self):
        pass

    @pytest.fixture()
    def operation(self):
        pass

    @pytest.fixture()
    def cube(self):
        pass

    @staticmethod
    def test_to_poly(expected, operation, cube):
        mocking = Mock()
        setattr(Transform, operation, mocking)
        cube_faces_to_polydata(cube)
        getattr(Transform, operation)
        actual = mocking.call_args.kwargs
        for key, actual_value in actual.items():
            if isinstance(actual_value, np.ndarray):
                np.testing.assert_array_equal(expected[key], actual_value)
            else:
                assert expected[key] == actual_value

    @staticmethod
    def test_to_poly_crs(expected, operation, cube):
        default_cs = iris.coord_systems.GeogCS(
            iris.analysis.cartography.DEFAULT_SPHERICAL_EARTH_RADIUS
        )
        expected["crs"] = default_cs.as_cartopy_crs().proj4_init
        for coord in cube.coords():
            coord.coord_system = default_cs
        mocking = Mock()
        setattr(Transform, operation, mocking)
        cube_faces_to_polydata(cube)
        getattr(Transform, operation)
        actual = mocking.call_args.kwargs
        print(actual)
        for key, actual_value in actual.items():
            if isinstance(actual_value, np.ndarray):
                np.testing.assert_array_equal(expected[key], actual_value)
            else:
                assert expected[key] == actual_value


class Test2dToPoly(ParentClass):
    @pytest.fixture()
    def expected(self, cube_2d):
        return {
            "xs": cube_2d.coord(axis="X").contiguous_bounds(),
            "ys": cube_2d.coord(axis="Y").contiguous_bounds(),
            "data": cube_2d.data,
            "name": cube_2d.name() + " / " + str(cube_2d.units),
        }

    @pytest.fixture()
    def operation(self):
        return "from_2d"

    @pytest.fixture()
    def cube(self, cube_2d):
        return cube_2d


class Test1dToPoly(ParentClass):
    @pytest.fixture()
    def expected(self, cube_1d):
        return {
            "xs": cube_1d.coord(axis="X").contiguous_bounds(),
            "ys": cube_1d.coord(axis="Y").contiguous_bounds(),
            "data": cube_1d.data,
            "name": cube_1d.name() + " / " + str(cube_1d.units),
        }

    @pytest.fixture()
    def operation(self):
        return "from_1d"

    @pytest.fixture()
    def cube(self, cube_1d):
        return cube_1d


class TestMeshToPoly(ParentClass):
    @pytest.fixture()
    def expected(self, cube_mesh):
        return {
            "xs": cube_mesh.coord(axis="X"),
            "ys": cube_mesh.coord(axis="Y"),
            "data": cube_mesh.data,
            "name": cube_mesh.name() + " / " + str(cube_mesh.units),
            "start_index": 0,
        }

    @pytest.fixture()
    def operation(self):
        return "from_unstructured"

    @pytest.fixture()
    def cube(self, cube_mesh):
        return cube_mesh

    @pytest.mark.skip(reason="Meshes do not support crs currently")
    def test_to_poly_crs(self, expected, actual):
        return NotImplemented


class TestExtras:
    def test_too_many_dims(self):
        pytest.raises(
            ValueError,
            match=r"There are too many dimensions on this coordinate",
        )

    def test_not_1d_or_2d(self):
        pytest.raises(
            NotImplementedError,
            match=r"Only 1D and 2D coordinates are supported",
        )

    def test_no_mesh_or_2d(self):
        pytest.raises(
            NotImplementedError,
            match=r"Cube must have a mesh or have 2 dimensions",
        )
