import pytest

from iris.coords import AuxCoord
from iris.experimental import mesh_coord_indexing
from iris.loading import load_cube
from iris.mesh.components import MeshCoord, _MeshIndexSet
from iris.tests import _shared_utils
from iris.tests.stock.mesh import sample_mesh_cube

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
    return (sample_mesh_cube(location=location), location)


@pytest.fixture
def cube_mesh_edge():
    location = "edge"
    return (sample_mesh_cube(location=location), location)


@pytest.fixture
def cube_mesh_face():
    location = "face"
    return (sample_mesh_cube(location=location), location)


@pytest.mark.parametrize(
    "fixture",
    ["cube_mesh_from_file", "cube_mesh_node", "cube_mesh_edge", "cube_mesh_face"],
)
def test_subset_indexing_auxcoord(fixture, request):
    (cube, _) = request.getfixturevalue(fixture)
    with mesh_coord_indexing.SETTING.context(mesh_coord_indexing.Options.AUX_COORD):
        indexed_cube = cube[0, 0:1]
    # The mesh's lat/lon should be represented as AuxCoord
    assert isinstance(indexed_cube.coord(standard_name="latitude"), AuxCoord)
    assert isinstance(indexed_cube.coord(standard_name="longitude"), AuxCoord)


@pytest.mark.parametrize(
    "fixture",
    ["cube_mesh_from_file", "cube_mesh_node", "cube_mesh_edge", "cube_mesh_face"],
)
def test_subset_indexing_new_mesh(fixture, request):
    (cube, _) = request.getfixturevalue(fixture)
    with mesh_coord_indexing.SETTING.context(mesh_coord_indexing.Options.NEW_MESH):
        indexed_cube = cube[0, 0:1]
    # The mesh's lat/lon should be represented as MeshCoord
    assert isinstance(indexed_cube.coord(standard_name="latitude"), MeshCoord)
    assert isinstance(indexed_cube.coord(standard_name="longitude"), MeshCoord)


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
    # The mesh's lat/lon should be represented as MeshCoord, but they are based on
    # _MeshIndexSet
    lat = indexed_cube.coord(standard_name="latitude")
    lon = indexed_cube.coord(standard_name="longitude")
    assert isinstance(lat, MeshCoord)
    assert isinstance(lon, MeshCoord)

    assert isinstance(lat.mesh, _MeshIndexSet)
    assert isinstance(lon.mesh, _MeshIndexSet)
