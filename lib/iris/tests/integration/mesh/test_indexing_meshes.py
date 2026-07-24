# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

import pytest

from iris.coords import AuxCoord, Coord
from iris.cube import Cube
from iris.experimental import mesh_coord_indexing
from iris.loading import load_cube
from iris.mesh.components import Mesh, MeshCoord, _MeshIndexSet
from iris.tests import _shared_utils
from iris.tests.stock.mesh import sample_mesh, sample_mesh_cube

# using a cube with a mesh from file and building one from scratch (an example for each location):
# Index the cube to get back the auxcoord, meshxy, meshindexset (by using the setting)
# Modifications to the original mesh and check and if they are reflected
# test by looking at meshcoord and comparing
# something with as_mesh
# check that a change is not reflected


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


def change_and_assert_no_change_in_right(left: Coord, right: Coord, value: int):
    left.points[0] = value
    assert right.points[0] != value


def change_and_assert_change_in_right(left: Coord, right: Coord, value: int):
    left.points[0] = value
    assert right.points[0] == value


def assert_change_raises_exception(mesh_coord: MeshCoord):
    assert mesh_coord.points
    with pytest.raises(Exception, match="test"):
        mesh_coord.points[0] = 0


@pytest.mark.parametrize(
    "fixture",
    ["cube_mesh_from_file", "cube_mesh_node", "cube_mesh_edge", "cube_mesh_face"],
)
def test_subset_indexing_auxcoord(fixture, request):
    (cube, _) = request.getfixturevalue(fixture)
    with mesh_coord_indexing.SETTING.context(mesh_coord_indexing.Options.AUX_COORD):
        indexed_cube = cube[0, 0:1]

    indexed_lat = indexed_cube.coord(standard_name="latitude")
    indexed_lon = indexed_cube.coord(standard_name="longitude")
    # The mesh's lat/lon should be represented as AuxCoord
    assert isinstance(indexed_lat, AuxCoord)
    assert isinstance(indexed_lon, AuxCoord)
    # And no mesh in cube
    assert indexed_cube.mesh is None

    original_lat = cube.coord(standard_name="latitude")
    original_lon = cube.coord(standard_name="longitude")

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
    (cube, _) = request.getfixturevalue(fixture)

    with mesh_coord_indexing.SETTING.context(mesh_coord_indexing.Options.NEW_MESH):
        indexed_cube = cube[0, 0:1]
    # The mesh's lat/lon should be represented as MeshCoord with a Mesh as the mesh
    indexed_lat = indexed_cube.coord(standard_name="latitude")
    indexed_lon = indexed_cube.coord(standard_name="longitude")
    assert isinstance(indexed_lat, MeshCoord)
    assert isinstance(indexed_lon, MeshCoord)
    assert isinstance(indexed_lat.mesh, Mesh)
    assert isinstance(indexed_lon.mesh, Mesh)

    original_lat = cube.coord(standard_name="latitude")
    original_lon = cube.coord(standard_name="longitude")
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
    (cube, _) = request.getfixturevalue(fixture)
    with mesh_coord_indexing.SETTING.context(
        mesh_coord_indexing.Options.MESH_INDEX_SET
    ):
        indexed_cube = cube[0, 0:1]
    # The mesh's lat/lon should be represented as MeshCoord,
    # with a _MeshIndexSet as the mesh
    indexed_lat = indexed_cube.coord(standard_name="latitude")
    indexed_lon = indexed_cube.coord(standard_name="longitude")
    assert isinstance(indexed_lat, MeshCoord)
    assert isinstance(indexed_lon, MeshCoord)
    assert isinstance(indexed_lat.mesh, _MeshIndexSet)
    assert isinstance(indexed_lon.mesh, _MeshIndexSet)

    # You cannot change the values of a _MeshIndexSet
    # Not raising exception for some reason
    assert_change_raises_exception(indexed_lat)
    assert_change_raises_exception(indexed_lon)

    original_lat = cube.coord(standard_name="latitude")
    original_lon = cube.coord(standard_name="longitude")
    # Changing the original mesh is reflected in the _MeshIndexSet
    # Change not reflected for some reason
    value = 9999
    change_and_assert_change_in_right(original_lat, indexed_lat, value)
    change_and_assert_change_in_right(original_lon, indexed_lon, value)
