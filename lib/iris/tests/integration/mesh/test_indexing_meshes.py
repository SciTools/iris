# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

import numpy as np
import pytest

from iris.coords import AuxCoord, Coord
from iris.cube import Cube
from iris.experimental import mesh_coord_indexing
from iris.loading import load_cube
from iris.mesh.components import MeshCoord, MeshXY, _MeshIndexSet
from iris.tests import _shared_utils
from iris.tests.stock.mesh import sample_mesh, sample_mesh_cube


@pytest.fixture
def cube_mesh_from_file():
    file_path = _shared_utils.get_data_path(
        [
            "NetCDF",
            "unstructured_grid",
            "lfric_ngvat_2D_72t_face_half_levels_main_conv_rain.nc",
        ]
    )
    return (load_cube(file_path, "conv_rain"), "face")


@pytest.fixture
def cube_mesh_node():
    location = "node"
    mesh = sample_mesh(n_nodes=15, n_edges=3, n_faces=3)  # Cannot have a node only mesh
    return (sample_mesh_cube(location=location, mesh=mesh), location)


@pytest.fixture
def cube_mesh_edge():
    location = "edge"
    mesh = sample_mesh(n_nodes=15, n_edges=3, n_faces=0)
    return (sample_mesh_cube(location=location, mesh=mesh), location)


@pytest.fixture
def cube_mesh_face():
    location = "face"
    mesh = sample_mesh(n_nodes=15, n_edges=0, n_faces=3)
    return (sample_mesh_cube(location=location, mesh=mesh), location)


@pytest.fixture
def cube_mesh_1_indexed(cube_mesh_from_file):
    cube = cube_mesh_from_file[0]
    mesh = cube.mesh
    for con in mesh.all_connectivities:
        if con is not None:
            con.indices[:] += 1
            con._metadata_manager.start_index = 1
    return (cube, "face")


def change_and_assert_no_change_in_right(left: Coord, right: Coord, value: int):
    left.points[0] = value
    assert right.points[0] != value


@pytest.mark.parametrize(
    "fixture",
    ["cube_mesh_from_file", "cube_mesh_node", "cube_mesh_edge", "cube_mesh_face"],
)
def test_subset_indexing_auxcoord(fixture, request):
    cube: Cube
    location: str
    (cube, location) = request.getfixturevalue(fixture)
    with mesh_coord_indexing.SETTING.context(mesh_coord_indexing.Options.AUX_COORD):
        indexed_cube = cube[0, 0:1]

    indexed_lat = indexed_cube.coord(standard_name="latitude")
    indexed_lon = indexed_cube.coord(standard_name="longitude")
    # The mesh's lat/lon should be represented as AuxCoord
    assert isinstance(indexed_lat, AuxCoord)
    assert isinstance(indexed_lon, AuxCoord)
    # And no mesh in cube
    assert indexed_cube.mesh is None

    assert cube.mesh is not None
    original_lat = cube.mesh.coord(standard_name="latitude", location=location)
    original_lon = cube.mesh.coord(standard_name="longitude", location=location)
    assert original_lat is not None
    assert original_lon is not None
    # Any changes to the indexed cube's mesh should not be reflected in the original
    value = 9999
    change_and_assert_no_change_in_right(indexed_lat, original_lat, value)
    change_and_assert_no_change_in_right(indexed_lon, original_lon, value)
    # and vice versa
    value = -9999
    change_and_assert_no_change_in_right(original_lat, indexed_lat, value)
    change_and_assert_no_change_in_right(original_lon, indexed_lon, value)


# Excluding "cube_mesh_node" from this test because it is an invalid thing to do
# Test for the raised exception exists in unit tests
@pytest.mark.parametrize(
    "fixture",
    ["cube_mesh_from_file", "cube_mesh_edge", "cube_mesh_face"],
)
def test_subset_indexing_new_mesh(fixture, request):
    cube: Cube
    location: str
    (cube, location) = request.getfixturevalue(fixture)

    with mesh_coord_indexing.SETTING.context(mesh_coord_indexing.Options.NEW_MESH):
        indexed_cube = cube[0, 0:1]
    # The mesh's lat/lon should be represented as MeshCoord with a Mesh as the mesh
    indexed_lat = indexed_cube.coord(standard_name="latitude")
    indexed_lon = indexed_cube.coord(standard_name="longitude")
    assert isinstance(indexed_lat, MeshCoord)
    assert isinstance(indexed_lon, MeshCoord)
    assert isinstance(indexed_lat.mesh, MeshXY)
    assert isinstance(indexed_lon.mesh, MeshXY)

    assert cube.mesh is not None
    original_lat = cube.mesh.coord(standard_name="latitude", location=location)
    original_lon = cube.mesh.coord(standard_name="longitude", location=location)
    assert original_lat is not None
    assert original_lon is not None
    # Any changes to the indexed cube's mesh should not be reflected in the original
    value = 9999
    change_and_assert_no_change_in_right(indexed_lat, original_lat, value)
    change_and_assert_no_change_in_right(indexed_lon, original_lon, value)
    # and vice versa
    value = -9999
    change_and_assert_no_change_in_right(original_lat, indexed_lat, value)
    change_and_assert_no_change_in_right(original_lon, indexed_lon, value)


@pytest.mark.parametrize(
    "fixture",
    ["cube_mesh_from_file", "cube_mesh_node", "cube_mesh_edge", "cube_mesh_face"],
)
def test_subset_indexing_mesh_index_set(fixture, request):
    cube: Cube
    location: str
    (cube, location) = request.getfixturevalue(fixture)
    with mesh_coord_indexing.SETTING.context(
        mesh_coord_indexing.Options.MESH_INDEX_SET
    ):
        indexed_cube = cube[0, 0:1]
    # The mesh's lat/lon should be represented as MeshCoord,
    # with a _MeshIndexSet as the mesh
    indexed_cube_lat = indexed_cube.coord(standard_name="latitude")
    indexed_cube_lon = indexed_cube.coord(standard_name="longitude")
    assert isinstance(indexed_cube_lat, MeshCoord)
    assert isinstance(indexed_cube_lon, MeshCoord)
    assert isinstance(indexed_cube_lat.mesh, _MeshIndexSet)
    assert isinstance(indexed_cube_lon.mesh, _MeshIndexSet)
    # The indexed_cube's mesh is also a _MeshIndexSet
    assert isinstance(indexed_cube.mesh, _MeshIndexSet)

    assert isinstance(cube.mesh, MeshXY)
    original_lat = cube.mesh.coord(standard_name="latitude", location=location)
    original_lon = cube.mesh.coord(standard_name="longitude", location=location)
    assert original_lat is not None
    assert original_lon is not None

    indexed_mesh_lat = indexed_cube.mesh.coord(
        standard_name="latitude", location=location
    )
    indexed_mesh_lon = indexed_cube.mesh.coord(
        standard_name="longitude", location=location
    )
    assert indexed_mesh_lat is not None
    assert indexed_mesh_lon is not None
    # Changing the values of a _MeshIndexSet's coords does nothing to the original
    value = 9999
    change_and_assert_no_change_in_right(indexed_mesh_lat, original_lat, value)
    change_and_assert_no_change_in_right(indexed_mesh_lon, original_lon, value)

    # Changing the original mesh is reflected in the _MeshIndexSet
    # Importantly, the _MeshIndexSet must be declared after the changes to the original
    value = -9999

    def change_original_mesh_reflected_in_index_set(
        original: Cube, indexed: Cube, name: str, loc: str, value: int
    ):
        assert original.mesh is not None
        original_coord = original.mesh.coord(standard_name=name, location=loc)
        assert original_coord is not None
        original_coord.points[0] = value

        assert indexed.mesh is not None
        indexed_coord = indexed.mesh.coord(standard_name=name, location=loc)
        assert indexed_coord is not None
        assert indexed_coord.points[0] == value

    change_original_mesh_reflected_in_index_set(
        cube, indexed_cube, "latitude", location, value
    )
    change_original_mesh_reflected_in_index_set(
        cube, indexed_cube, "longitude", location, value
    )


@pytest.mark.parametrize(
    "fixture",
    ["cube_mesh_from_file", "cube_mesh_edge", "cube_mesh_face", "cube_mesh_1_indexed"],
)
def test_indexing_mode_equivalency(fixture, request):
    cube: Cube
    location: str
    (cube, location) = request.getfixturevalue(fixture)

    with mesh_coord_indexing.SETTING.context(mesh_coord_indexing.Options.AUX_COORD):
        indexed_cube_aux = cube[0, 1:-1]
    with mesh_coord_indexing.SETTING.context(mesh_coord_indexing.Options.NEW_MESH):
        indexed_cube_mesh = cube[0, 1:-1]
    with mesh_coord_indexing.SETTING.context(
        mesh_coord_indexing.Options.MESH_INDEX_SET
    ):
        indexed_cube_mis = cube[0, 1:-1]

    aux_lon = indexed_cube_aux.coord("longitude")
    aux_lat = indexed_cube_aux.coord("latitude")
    mesh_lon = indexed_cube_mesh.coord("longitude")
    mesh_lat = indexed_cube_mesh.coord("latitude")
    mis_lon = indexed_cube_mis.coord("longitude")
    mis_lat = indexed_cube_mis.coord("latitude")

    assert np.array_equal(aux_lon.bounds, mesh_lon.bounds)
    assert np.array_equal(aux_lat.bounds, mesh_lat.bounds)
    assert np.array_equal(aux_lon.bounds, mis_lon.bounds)
    assert np.array_equal(aux_lat.bounds, mis_lat.bounds)
