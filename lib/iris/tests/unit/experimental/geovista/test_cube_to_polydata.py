# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.experimental.geovista.cube_to_polydata` function."""

from typing import ClassVar
from unittest.mock import Mock

from geovista import Transform
import numpy as np
import pytest

import iris.analysis.cartography
import iris.coord_systems
from iris.experimental.geovista import cube_to_polydata
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


@pytest.fixture()
def default_cs():
    return iris.coord_systems.GeogCS(
        iris.analysis.cartography.DEFAULT_SPHERICAL_EARTH_RADIUS
    )


class ParentClass:
    MOCKED_OPERATION: ClassVar[str]

    @pytest.fixture()
    def expected(self):
        pass

    @pytest.fixture()
    def operation(self):
        pass

    @pytest.fixture()
    def cube(self):
        pass

    @pytest.fixture()
    def cube_with_crs(self, default_cs, cube):
        cube_crs = cube.copy()
        for coord in cube_crs.coords():
            coord.coord_system = default_cs
        return cube_crs

    @pytest.fixture()
    def mocked_operation(self):
        mocking = Mock()
        setattr(Transform, self.MOCKED_OPERATION, mocking)
        return mocking

    @staticmethod
    def test_to_poly(expected, mocked_operation, cube):
        cube_to_polydata(cube)
        actual = mocked_operation.call_args.kwargs
        for key, expected_value in expected.items():
            if hasattr(expected_value, "shape"):
                np.testing.assert_array_equal(actual[key], expected_value)
            else:
                assert actual[key] == expected_value

    @staticmethod
    def test_to_poly_crs(mocked_operation, default_cs, cube_with_crs):
        cube_to_polydata(cube_with_crs)
        actual = mocked_operation.call_args.kwargs
        assert actual["crs"] == default_cs.as_cartopy_crs().proj4_init

    @staticmethod
    def test_to_poly_kwargs(mocked_operation, cube):
        kwargs = {"test": "test"}
        cube_to_polydata(cube, **kwargs)
        actual = mocked_operation.call_args.kwargs
        assert actual["test"] == "test"


class Test2dToPoly(ParentClass):
    MOCKED_OPERATION = "from_2d"

    @pytest.fixture()
    def expected(self, cube_2d):
        return {
            "xs": cube_2d.coord(axis="X").contiguous_bounds(),
            "ys": cube_2d.coord(axis="Y").contiguous_bounds(),
            "data": cube_2d.data,
            "name": f"{cube_2d.name()} / ({cube_2d.units})",
        }

    @pytest.fixture()
    def cube(self, cube_2d):
        return cube_2d


class Test1dToPoly(ParentClass):
    MOCKED_OPERATION = "from_1d"

    @pytest.fixture()
    def expected(self, cube_1d):
        return {
            "xs": cube_1d.coord(axis="X").contiguous_bounds(),
            "ys": cube_1d.coord(axis="Y").contiguous_bounds(),
            "data": cube_1d.data,
            "name": f"{cube_1d.name()} / ({cube_1d.units})",
        }

    @pytest.fixture()
    def cube(self, cube_1d):
        return cube_1d


class TestMeshToPoly(ParentClass):
    MOCKED_OPERATION = "from_unstructured"

    @pytest.fixture()
    def expected(self, cube_mesh):
        return {
            "xs": cube_mesh.mesh.node_coords[0].points,
            "ys": cube_mesh.mesh.node_coords[1].points,
            "connectivity": cube_mesh.mesh.face_node_connectivity.indices_by_location(),
            "data": cube_mesh.data[0],
            "name": cube_mesh.name() + " / " + "(" + str(cube_mesh.units) + ")",
            "start_index": 0,
        }

    def test_if_1d(self, cube_mesh):
        with pytest.raises(
            NotImplementedError,
            match=r"Cubes with a mesh must be one dimensional",
        ):
            cube_to_polydata(cube_mesh)

    @pytest.fixture()
    def operation(self):
        return "from_unstructured"

    @pytest.fixture()
    def cube(self, cube_mesh):
        return cube_mesh[0]

    @pytest.mark.skip(reason="Meshes do not support crs currently")
    def test_to_poly_crs(self, expected, actual):
        return NotImplemented


class TestExtras:
    @pytest.fixture()
    def cube_1d_2d(self, cube_2d):
        my_cube = cube_2d.copy()
        lat_coord = my_cube.aux_coords[0]
        lat_coord_small = lat_coord[0]
        lat_coord_small.bounds = None
        lat_coord_small.points = np.arange(len(lat_coord_small.points))
        my_cube.remove_coord(lat_coord)
        my_cube.add_aux_coord(lat_coord_small, 1)
        return my_cube

    def test_not_1d_or_2d(self, cube_1d_2d):
        with pytest.raises(
            NotImplementedError,
            match=r"Only 1D and 2D coordinates are supported",
        ):
            cube_to_polydata(cube_1d_2d)

    def test_no_mesh_or_2d(self, cube_1d):
        cube = cube_1d[0]
        with pytest.raises(
            NotImplementedError,
            match=r"Cube must have a mesh or have 2 dimensions",
        ):
            cube_to_polydata(cube)
