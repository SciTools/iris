# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.mesh.components._MeshIndexSet` class."""

import itertools
import re

from dask import array as da
import numpy as np
import pytest

from iris._lazy_data import as_concrete_data, is_lazy_data
from iris.coords import AuxCoord
from iris.mesh import MeshCoord, MeshXY
from iris.mesh.components import Connectivity, _MeshIndexSet
from iris.tests import _shared_utils
from iris.tests.stock.mesh import sample_mesh, sample_strip_mesh


@pytest.fixture(params=[False, True], ids=["real", "lazy"], autouse=True)
def lazy_values(request):
    return request.param


@pytest.fixture
def mesh_2d(lazy_values):
    return sample_mesh(lazy_values=lazy_values)


@pytest.fixture
def mesh_1d(lazy_values):
    return sample_mesh(n_faces=0, lazy_values=lazy_values)


@pytest.fixture(params=["mesh_1d", "mesh_2d"])
def meshes_all(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=["node", "edge", "face"])
def locations_all(request):
    return request.param


_MESH_LOCATION_COMBINED = [
    (mesh, loc)
    for mesh, loc in itertools.product(["mesh_1d", "mesh_2d"], ["node", "edge", "face"])
    if not (mesh == "mesh_1d" and loc == "face")
]


@pytest.fixture(params=_MESH_LOCATION_COMBINED, ids=lambda x: f"{x[0]}-{x[1]}")
def meshes_locs_all(request):
    mesh_name, loc = request.param
    mesh = request.getfixturevalue(mesh_name)
    return mesh, loc


def test_dummy(meshes_all):
    assert isinstance(meshes_all, MeshXY)


class Test___init__:
    def test_basic(self, meshes_locs_all):
        mesh, location = meshes_locs_all
        indices = [0, 2]
        index_set = _MeshIndexSet(indices=indices, mesh=mesh, location=location)

        assert index_set.cf_role == "location_index_set"
        assert index_set.mesh is mesh
        assert index_set.location == location
        assert index_set.start_index == 0
        assert index_set.topology_dimension == mesh.topology_dimension
        _shared_utils.assert_array_equal(index_set.indices, indices)

    def test_numpy_array_indices(self, meshes_locs_all):
        mesh, location = meshes_locs_all
        indices = np.array([0, 2])
        index_set = _MeshIndexSet(indices=indices, mesh=mesh, location=location)

        _shared_utils.assert_array_equal(index_set.indices, indices)

    def test_fail_multidim_indices(self, meshes_locs_all):
        mesh, location = meshes_locs_all
        with pytest.raises(ValueError, match="`indices` must be 1D"):
            _MeshIndexSet(indices=[[0, 1], [2, 3]], mesh=mesh, location=location)

    def test_fail_invalid_mesh(self):
        with pytest.raises(TypeError, match="`mesh` must be `MeshXY`"):
            _MeshIndexSet(indices=[0], mesh="not-a-mesh", location="face")

    def test_fail_invalid_location(self, mesh_2d):
        with pytest.raises(ValueError, match="`location` must be in"):
            _MeshIndexSet(indices=[0], mesh=mesh_2d, location="bad")

    def test_fail_location_mismatch(self, mesh_1d):
        with pytest.raises(ValueError, match="`location` cannot be 'face'"):
            _MeshIndexSet(indices=[0], mesh=mesh_1d, location="face")

    def test_fail_invalid_start_index(self, mesh_2d):
        with pytest.raises(ValueError, match="`start_index must be 0 or 1"):
            _MeshIndexSet(indices=[0], mesh=mesh_2d, location="face", start_index=3)

    def test___getstate____setstate__(self, meshes_locs_all):
        mesh, location = meshes_locs_all
        original = _MeshIndexSet(indices=[0, 1], mesh=mesh, location=location)
        state = original.__getstate__()

        recreated = _MeshIndexSet.__new__(_MeshIndexSet)
        # __setstate__ expects an initial values manager slot.
        object.__setattr__(recreated, "_values_dm", None)
        recreated.__setstate__(state)

        _shared_utils.assert_array_equal(state[0], original.indices)
        assert state[1].mesh is original.mesh
        assert state[1].location == original.location
        assert state[1].start_index == original.start_index
        _shared_utils.assert_array_equal(recreated.indices, original.indices)
        assert recreated.mesh is original.mesh
        assert recreated.location == original.location
        assert recreated.start_index == original.start_index

    def test_scalar_index(self, meshes_locs_all):
        mesh, location = meshes_locs_all
        index_set = _MeshIndexSet(indices=2, mesh=mesh, location=location)

        _shared_utils.assert_array_equal(index_set.indices, np.array([2]))
        coord = index_set.coord(location=location, axis="x")
        assert coord.shape == (1,)


class Test___eq__:
    def test_equal_same_object(self, meshes_locs_all):
        """An instance is equal to itself."""
        mesh, location = meshes_locs_all
        index_set = _MeshIndexSet(indices=[0, 2], mesh=mesh, location=location)
        assert index_set == index_set

    def test_equal_identical(self, meshes_locs_all):
        """Two independently constructed instances with identical args are equal."""
        mesh, location = meshes_locs_all
        a = _MeshIndexSet(indices=[0, 2], mesh=mesh, location=location)
        b = _MeshIndexSet(indices=[0, 2], mesh=mesh, location=location)
        assert a == b

    def test_equal_with_metadata(self, meshes_locs_all):
        """Equality holds when optional metadata fields also match."""
        mesh, location = meshes_locs_all
        kwargs = dict(
            indices=[1, 3],
            mesh=mesh,
            location=location,
            long_name="test",
            var_name="v",
            attributes={"source": "test"},
        )
        assert _MeshIndexSet(**kwargs) == _MeshIndexSet(**kwargs)

    def test_not_equal_different_indices(self, meshes_locs_all):
        """Instances with different indices are not equal."""
        mesh, location = meshes_locs_all
        a = _MeshIndexSet(indices=[0, 2], mesh=mesh, location=location)
        b = _MeshIndexSet(indices=[0, 3], mesh=mesh, location=location)
        assert a != b

    def test_not_equal_different_metadata(self, meshes_locs_all):
        """Instances with differing metadata are not equal even if indices match."""
        mesh, location = meshes_locs_all
        a = _MeshIndexSet(indices=[0, 2], mesh=mesh, location=location, long_name="foo")
        b = _MeshIndexSet(indices=[0, 2], mesh=mesh, location=location, long_name="bar")
        assert a != b

    def test_not_equal_different_start_index(self, meshes_locs_all):
        """Instances with different start_index values are not equal."""
        mesh, location = meshes_locs_all
        a = _MeshIndexSet(indices=[0, 2], mesh=mesh, location=location, start_index=0)
        b = _MeshIndexSet(indices=[0, 2], mesh=mesh, location=location, start_index=1)
        assert a != b

    def test_not_equal_different_mesh(self, lazy_values):
        """Instances referencing different meshes are not equal."""
        mesh_a = sample_mesh(lazy_values=lazy_values)
        mesh_b = sample_mesh(lazy_values=lazy_values)
        # Give the meshes distinct var_names so their metadata differs.
        mesh_b.var_name = "different_mesh"
        a = _MeshIndexSet(indices=[0, 2], mesh=mesh_a, location="node")
        b = _MeshIndexSet(indices=[0, 2], mesh=mesh_b, location="node")
        assert a != b

    def test_not_equal_different_location(self, mesh_2d):
        """Instances with different locations are not equal."""
        a = _MeshIndexSet(indices=[0, 2], mesh=mesh_2d, location="node")
        b = _MeshIndexSet(indices=[0, 2], mesh=mesh_2d, location="face")
        assert a != b

    def test_not_equal_non_mesh_index_set(self, meshes_locs_all):
        """Comparing with a non-_MeshIndexSet object returns NotImplemented."""
        mesh, location = meshes_locs_all
        index_set = _MeshIndexSet(indices=[0, 2], mesh=mesh, location=location)
        result = index_set.__eq__("not-a-mesh-index-set")
        assert result is NotImplemented


class Test_properties:
    def test_cf_role(self, meshes_locs_all):
        mesh, location = meshes_locs_all
        index_set = _MeshIndexSet([0], mesh=mesh, location=location)
        assert index_set.cf_role == "location_index_set"

    def test_dimension_properties(self, meshes_locs_all):
        mesh, location = meshes_locs_all
        index_set = _MeshIndexSet([0], mesh=mesh, location=location)
        assert index_set.node_dimension == "_MeshIndexSet_NotImplemented"
        assert index_set.edge_dimension == "_MeshIndexSet_NotImplemented"
        assert index_set.face_dimension == "_MeshIndexSet_NotImplemented"

    def test_metadata_properties(self, meshes_locs_all):
        mesh, location = meshes_locs_all
        indices = [0, 2]
        index_set = _MeshIndexSet(
            indices=indices,
            mesh=mesh,
            location=location,
            long_name="my-index-set",
            start_index=1,
        )
        _shared_utils.assert_array_equal(index_set.indices, indices)
        assert index_set.mesh is mesh
        assert index_set.location == location
        assert index_set.start_index == 1
        assert index_set.topology_dimension == mesh.topology_dimension
        assert index_set.long_name == "my-index-set"


class Test_index_calculations:
    def test_node_location_calculate_node_bool_index(self, meshes_all):
        index_set = _MeshIndexSet(indices=[0, 2, 4], mesh=meshes_all, location="node")
        result = index_set._calculate_node_bool_index()

        expected = np.zeros(meshes_all.node_coords.node_x.shape[0], dtype=bool)
        expected[[0, 2, 4]] = True
        _shared_utils.assert_array_equal(result, expected)

    def test_node_location_requires_monotonic_indices(self, meshes_locs_all):
        mesh, location = meshes_locs_all
        index_set = _MeshIndexSet(indices=[2, 1], mesh=mesh, location=location)

        if location == "node":
            with pytest.raises(
                ValueError, match="requires strictly increasing indices"
            ):
                index_set._calculate_node_bool_index()
        else:
            result = index_set._calculate_node_bool_index()
            assert isinstance(result, (np.ndarray, da.Array))
            assert result.dtype == bool

    def test_edge_location_calculate_node_bool_index(self, meshes_all):
        index_set = _MeshIndexSet(indices=[0, 2], mesh=meshes_all, location="edge")
        result = index_set._calculate_node_bool_index()

        expected = np.zeros(meshes_all.node_coords.node_x.shape[0], dtype=bool)
        expected[[5, 6, 9, 10]] = True
        _shared_utils.assert_array_equal(result, expected)

    def test_face_location_calculate_node_bool_index(self, mesh_2d):
        index_set = _MeshIndexSet(indices=[0, 2], mesh=mesh_2d, location="face")
        result = index_set._calculate_node_bool_index()

        expected = np.zeros(mesh_2d.node_coords.node_x.shape[0], dtype=bool)
        expected[[0, 1, 2, 3, 8, 9, 10, 11]] = True
        _shared_utils.assert_array_equal(result, expected)

    def test_calculate_edge_indices(self, meshes_locs_all):
        mesh, location = meshes_locs_all
        index_set = _MeshIndexSet(indices=[1], mesh=mesh, location=location)
        result = index_set._calculate_edge_indices()

        # Face 1 has side (5, 6), which is edge 0.
        expected = {"edge": [1], "face": [0], "node": []}[location]
        assert isinstance(result, np.ndarray)
        _shared_utils.assert_array_equal(result, np.array(expected, dtype=int))

    def test_calculate_face_indices(self, mesh_2d, locations_all):
        index_set = _MeshIndexSet(indices=[1], mesh=mesh_2d, location=locations_all)
        result = index_set._calculate_face_indices()

        expected = {"face": [1], "edge": [], "node": []}[locations_all]
        assert isinstance(result, np.ndarray)
        _shared_utils.assert_array_equal(result, np.array(expected, dtype=int))

    @pytest.mark.parametrize("start_index", [0, 1])
    @pytest.mark.parametrize("location_axis", [0, 1])
    @pytest.mark.parametrize(
        ("location", "indices", "edges", "faces"),
        [
            ("face", [0, 2], [0, 2, 3, 5, 6, 7, 8, 9], [0, 2]),
            ("face", [1, 0], [0, 1, 3, 4, 6, 7, 8], [1, 0]),
            ("edge", [0, 3, 6, 7, 9], [0, 3, 6, 7, 9], [0]),
            ("edge", [1, 4], [1, 4], []),
            ("node", [0, 1, 4, 5], [0, 3, 6, 7], [0]),
        ],
    )
    def test_strip_derived_selections(
        self, lazy_values, start_index, location_axis, location, indices, edges, faces
    ):
        mesh = sample_strip_mesh(
            lazy_values=lazy_values,
            start_index=start_index,
            location_axis=location_axis,
        )
        index_set = _MeshIndexSet(indices=indices, mesh=mesh, location=location)

        edge_result = index_set._calculate_edge_indices()
        face_result = index_set._calculate_face_indices()

        for result, expected in [(edge_result, edges), (face_result, faces)]:
            assert isinstance(result, np.ndarray)
            assert result.ndim == 1
            assert np.issubdtype(result.dtype, np.integer)
            _shared_utils.assert_array_equal(result, np.array(expected, dtype=int))

    def test_no_edges_gives_none(self, lazy_values):
        mesh = sample_mesh(n_edges=0, lazy_values=lazy_values)
        index_set = _MeshIndexSet(indices=[0, 2], mesh=mesh, location="face")
        assert index_set._calculate_edge_indices() is None

    def test_1d_mesh_no_faces(self, mesh_1d):
        index_set = _MeshIndexSet(indices=[0, 2], mesh=mesh_1d, location="node")
        assert index_set._calculate_face_indices() is None


class Test_managers_and_views:
    def test_coord_manager_subset(self, meshes_locs_all):
        mesh, location = meshes_locs_all
        index_set = _MeshIndexSet(indices=[0], mesh=mesh, location=location)

        coord_manager = index_set._coord_manager
        match location:
            case "node":
                assert coord_manager.node_x.shape < mesh.node_coords.node_x.shape
                assert coord_manager.node_y.shape < mesh.node_coords.node_y.shape
                assert (
                    coord_manager.node_x.core_points()[0]
                    in mesh.node_coords.node_x.points
                )
            case "edge":
                assert coord_manager.edge_x.shape < mesh.edge_coords.edge_x.shape
                assert coord_manager.edge_y.shape < mesh.edge_coords.edge_y.shape
                assert (
                    coord_manager.edge_x.core_points()[0]
                    in mesh.edge_coords.edge_x.points
                )
            case "face":
                assert coord_manager.face_x.shape < mesh.face_coords.face_x.shape
                assert coord_manager.face_y.shape < mesh.face_coords.face_y.shape
                assert (
                    coord_manager.face_x.core_points()[0]
                    in mesh.face_coords.face_x.points
                )

    def test_connectivity_manager_subset(self, meshes_locs_all):
        mesh, location = meshes_locs_all
        index_set = _MeshIndexSet(indices=[0, 2], mesh=mesh, location=location)

        connectivity_manager = index_set._connectivity_manager
        assert connectivity_manager.is_view
        match location:
            case "node":
                assert connectivity_manager.edge_node.shape[0] == 0
                if mesh.topology_dimension == 2:
                    assert connectivity_manager.face_node.shape[0] == 0
            case "edge":
                assert connectivity_manager.edge_node is not None
                if mesh.topology_dimension == 2:
                    assert connectivity_manager.face_node.shape[0] == 0
                else:
                    assert not hasattr(connectivity_manager, "face_node")
                assert (
                    connectivity_manager.edge_node.shape[0]
                    < mesh.edge_node_connectivity.shape[0]
                )
            case "face":
                assert connectivity_manager.face_node is not None
                # Edge 2 (nodes 9, 10) is a side of face 2.
                assert connectivity_manager.edge_node.shape[0] == 1
                assert (
                    connectivity_manager.face_node.shape[0]
                    < mesh.face_node_connectivity.shape[0]
                )

    def test_coord_manager_setter_forbidden(self, mesh_2d):
        index_set = _MeshIndexSet(indices=[0], mesh=mesh_2d, location="face")

        with pytest.raises(NotImplementedError, match="Modification of _MeshIndexSet"):
            index_set._coord_manager = mesh_2d._coord_manager

    def test_connectivity_manager_setter_forbidden(self, mesh_2d):
        index_set = _MeshIndexSet(indices=[0], mesh=mesh_2d, location="face")

        with pytest.raises(NotImplementedError, match="Modification of _MeshIndexSet"):
            index_set._connectivity_manager = mesh_2d._connectivity_manager

    def test_coord_manager_fail_non_lazy(self, mesh_2d):
        index_set = _MeshIndexSet(indices=[0], mesh=mesh_2d, location="face")

        coord_manager = index_set._coord_manager
        assert is_lazy_data(coord_manager.node_x.core_points())
        # Force realisation.
        _ = coord_manager.node_x.points
        # Attempting to access a second time triggers a laziness check,
        #  which fails.
        with pytest.raises(ValueError, match="Non-lazy coordinate detected"):
            _ = coord_manager.node_x.points

    def test_connectivity_manager_fail_non_lazy(self, mesh_2d):
        index_set = _MeshIndexSet(indices=[0], mesh=mesh_2d, location="face")

        connectivity_manager = index_set._connectivity_manager
        assert is_lazy_data(connectivity_manager.face_node.core_indices())
        # Force realisation.
        _ = connectivity_manager.face_node.indices
        # Attempting to access a second time triggers a laziness check,
        #  which fails.
        with pytest.raises(ValueError, match="Non-lazy connectivity detected"):
            _ = connectivity_manager.face_node.indices


class Test_unusual_connectivities:
    @pytest.fixture
    def mis_unusual_start_index(self, lazy_values):
        mesh = sample_mesh(lazy_values=lazy_values)
        face_node = mesh.face_node_connectivity
        assert face_node.start_index == 0
        mesh.add_connectivities(
            Connectivity(
                indices=face_node.indices + 1,
                cf_role=face_node.cf_role,
                start_index=1,
            )
        )
        index_set = _MeshIndexSet(
            indices=[0, 2], mesh=mesh, location="face", start_index=1
        )
        return index_set

    def test_unusual_start_index(self, mis_unusual_start_index):
        index_set = mis_unusual_start_index
        _shared_utils.assert_array_equal(index_set.indices, np.array([0, 2]))
        _shared_utils.assert_array_equal(
            index_set.connectivity(cf_role="face_node_connectivity").indices,
            np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
        )

    @pytest.fixture
    def mis_unusual_transposition(self, lazy_values):
        mesh = sample_mesh(lazy_values=lazy_values)
        face_node = mesh.face_node_connectivity
        mesh.add_connectivities(face_node.transpose())
        index_set = _MeshIndexSet(indices=[0, 2], mesh=mesh, location="face")
        return index_set

    def test_unusual_transposition(self, mis_unusual_transposition):
        index_set = mis_unusual_transposition
        _shared_utils.assert_array_equal(index_set.indices, np.array([0, 2]))
        _shared_utils.assert_array_equal(
            index_set.connectivity(cf_role="face_node_connectivity").indices,
            np.array([[0, 1, 2, 3], [4, 5, 6, 7]]).T,
        )

    @pytest.fixture
    def varied_mesh(self, lazy_values):
        r"""A mesh with edges and variable-sided faces, some sharing nodes.

        Two squares joined by a triangle, with a separate irregular pentagon.
        Edges to mirror the pentagon sides, plus an extra isolated edge.
         0 1 2 3 4
        0* o-o * O
           | |   |
        1* o-o * O
            \|
        2O * o-o *
         |\  | |
        3O O o-o *
         | |
        4O-O * * *
        """
        nodes_xy = {
            0: (0, -2),
            1: (0, -3),
            2: (0, -4),
            3: (1, 0),
            4: (1, -1),
            5: (1, -3),
            6: (1, -4),
            7: (2, 0),
            8: (2, -1),
            9: (2, -2),
            10: (2, -3),
            11: (3, -2),
            12: (3, -3),
            13: (4, 0),
            14: (4, -1),
        }
        face_nodes = [
            [9, 10, 12, 11, -1],
            [4, 9, 8, -1, -1],
            [3, 4, 8, 7, -1],
            [0, 1, 2, 6, 5],
        ]
        edge_nodes = [
            [0, 1],
            [1, 2],
            [2, 6],
            [6, 5],
            [13, 14],
        ]
        node_x = AuxCoord([x for x, y in nodes_xy.values()], standard_name="longitude")
        node_y = AuxCoord([y for x, y in nodes_xy.values()], standard_name="latitude")

        if lazy_values:
            node_x.points = node_x.lazy_points()
            node_y.points = node_y.lazy_points()

        # Face sizes vary (4/3/4/5), so unused slots are masked.
        face_node_indices = np.ma.masked_array(
            data=face_nodes,
            mask=np.array(
                [
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                ],
                dtype=bool,
            ),
        )
        edge_node_indices = np.array(edge_nodes)
        if lazy_values:
            face_node_indices = da.from_array(face_node_indices)
            edge_node_indices = da.from_array(edge_node_indices)

        mesh = MeshXY(
            topology_dimension=2,
            node_coords_and_axes=[(node_x, "x"), (node_y, "y")],
            connectivities=[
                Connectivity(
                    indices=face_node_indices,
                    cf_role="face_node_connectivity",
                ),
                Connectivity(
                    indices=edge_node_indices,
                    cf_role="edge_node_connectivity",
                ),
            ],
        )
        return mesh

    @pytest.fixture
    def varied_mesh_1_indexed(self, varied_mesh):
        mesh = varied_mesh
        for con in mesh.all_connectivities:
            if con is not None:
                con.indices[:] += 1
                con._metadata_manager.start_index = 1
        return mesh

    @pytest.mark.parametrize(
        "fixture",
        ["varied_mesh", "varied_mesh_1_indexed"],
    )
    def test_varied_faces(self, fixture, request):
        mesh: MeshXY = request.getfixturevalue(fixture)
        index_set = _MeshIndexSet(indices=[0, 1, 3], mesh=mesh, location="face")
        _shared_utils.assert_array_equal(index_set.indices, np.array([0, 1, 3]))
        connectivity = index_set.connectivity(cf_role="face_node_connectivity")
        normalised_indices = connectivity.indices - connectivity.start_index
        _shared_utils.assert_array_equal(
            normalised_indices,
            np.ma.masked_array(
                data=[[7, 8, 10, 9, -1], [3, 7, 6, -1, -1], [0, 1, 2, 5, 4]],
                mask=[[0, 0, 0, 0, 1], [0, 0, 0, 1, 1], [0, 0, 0, 0, 0]],
            ),
        )
        _shared_utils.assert_array_equal(
            index_set.node_coords[0].points, np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3])
        )
        _shared_utils.assert_array_equal(
            index_set.node_coords[1].points,
            np.array([-2, -3, -4, -1, -3, -4, -1, -2, -3, -2, -3]),
        )

    @pytest.mark.parametrize(
        "fixture",
        ["varied_mesh", "varied_mesh_1_indexed"],
    )
    def test_varied_edges(self, fixture, request):
        mesh: MeshXY = request.getfixturevalue(fixture)
        index_set = _MeshIndexSet(indices=[0, 1, 4], mesh=mesh, location="edge")
        _shared_utils.assert_array_equal(index_set.indices, np.array([0, 1, 4]))
        connectivity = index_set.connectivity(cf_role="edge_node_connectivity")
        normalised_indices = connectivity.indices - connectivity.start_index
        _shared_utils.assert_array_equal(
            normalised_indices,
            # Note the shared node.
            np.array([[0, 1], [1, 2], [3, 4]]),
        )
        _shared_utils.assert_array_equal(
            index_set.node_coords[0].points,
            np.array([0, 0, 0, 4, 4]),
        )
        _shared_utils.assert_array_equal(
            index_set.node_coords[1].points,
            np.array([-2, -3, -4, 0, -1]),
        )

    @pytest.mark.parametrize(
        "fixture",
        ["varied_mesh", "varied_mesh_1_indexed"],
    )
    def test_varied_derived_selections(self, fixture, request):
        mesh: MeshXY = request.getfixturevalue(fixture)

        face_set = _MeshIndexSet(indices=[3], mesh=mesh, location="face")
        _shared_utils.assert_array_equal(
            face_set._calculate_edge_indices(), np.array([0, 1, 2, 3])
        )

        # The pentagon's closing side (5, 0) has no edge, so no face is kept.
        edge_set = _MeshIndexSet(indices=[0, 1, 2, 3], mesh=mesh, location="edge")
        _shared_utils.assert_array_equal(
            edge_set._calculate_face_indices(), np.array([], dtype=int)
        )

    def test_node_selection_ignores_masked_slot_values(self, varied_mesh):
        """Masked face slots can hold out-of-range data, as when loaded from netCDF.

        Face [9, 10, 12, 11, <999999>] indexed the node mask with 999999: IndexError.
        """
        mesh: MeshXY = varied_mesh
        face_nodes = mesh.face_node_connectivity
        indices = face_nodes.indices
        indices = np.ma.masked_array(
            np.where(np.ma.getmaskarray(indices), 999999, np.ma.getdata(indices)),
            mask=np.ma.getmaskarray(indices),
        )
        mesh.add_connectivities(
            Connectivity(indices=indices, cf_role="face_node_connectivity")
        )

        node_set = _MeshIndexSet(indices=[0, 1, 2, 5, 6], mesh=mesh, location="node")
        _shared_utils.assert_array_equal(
            node_set._calculate_face_indices(), np.array([3])
        )


def _by_location_zero_based(connectivity):
    indices = connectivity.indices_by_location(connectivity.core_indices())
    return np.ma.asanyarray(as_concrete_data(indices)) - connectivity.start_index


def _masked_rows(rows, width):
    return np.ma.masked_less(np.array(rows, dtype=int).reshape(-1, width), 0)


class Test_strip_connectivities:
    @pytest.mark.parametrize("start_index", [0, 1])
    @pytest.mark.parametrize("location_axis", [0, 1])
    @pytest.mark.parametrize(
        ("location", "indices", "expected"),
        [
            pytest.param(
                "face",
                [0, 2],
                dict(
                    face_node=_masked_rows([[0, 1, 5, 4], [2, 3, 7, 6]], 4),
                    edge_node=_masked_rows(
                        [
                            [0, 1],
                            [2, 3],
                            [4, 5],
                            [6, 7],
                            [0, 4],
                            [1, 5],
                            [2, 6],
                            [3, 7],
                        ],
                        2,
                    ),
                    face_edge=_masked_rows([[0, 5, 2, 4], [1, 7, 3, 6]], 4),
                    face_face=_masked_rows([[-1] * 4] * 2, 4),
                    edge_face=_masked_rows(
                        [
                            [0, -1],
                            [1, -1],
                            [0, -1],
                            [1, -1],
                            [0, -1],
                            [0, -1],
                            [-1, 1],
                            [1, -1],
                        ],
                        2,
                    ),
                ),
                id="face-0-2",
            ),
            pytest.param(
                "face",
                [1, 0],
                dict(
                    face_node=_masked_rows([[1, 2, 5, 4], [0, 1, 4, 3]], 4),
                    edge_node=_masked_rows(
                        [[0, 1], [1, 2], [3, 4], [4, 5], [0, 3], [1, 4], [2, 5]], 2
                    ),
                    face_edge=_masked_rows([[1, 6, 3, 5], [0, 5, 2, 4]], 4),
                    face_face=_masked_rows([[-1, -1, -1, 1], [-1, 0, -1, -1]], 4),
                    edge_face=_masked_rows(
                        [[1, -1], [0, -1], [1, -1], [0, -1], [1, -1], [1, 0], [0, -1]],
                        2,
                    ),
                ),
                id="face-1-0",
            ),
            pytest.param(
                "edge",
                [0, 3, 6, 7, 9],
                dict(
                    face_node=_masked_rows([[0, 1, 4, 3]], 4),
                    edge_node=_masked_rows([[0, 1], [3, 4], [0, 3], [1, 4], [2, 5]], 2),
                    face_edge=_masked_rows([[0, 3, 1, 2]], 4),
                    face_face=_masked_rows([[-1] * 4], 4),
                    edge_face=_masked_rows(
                        [[0, -1], [0, -1], [0, -1], [0, -1], [-1, -1]], 2
                    ),
                ),
                id="edge-5",
            ),
            pytest.param(
                "edge",
                [1, 4],
                dict(
                    face_node=_masked_rows([], 4),
                    edge_node=_masked_rows([[0, 1], [2, 3]], 2),
                    face_edge=_masked_rows([], 4),
                    face_face=_masked_rows([], 4),
                    edge_face=_masked_rows([[-1, -1]] * 2, 2),
                ),
                id="edge-no-faces",
            ),
            pytest.param(
                "node",
                [0, 1, 4, 5],
                dict(
                    face_node=_masked_rows([[0, 1, 3, 2]], 4),
                    edge_node=_masked_rows([[0, 1], [2, 3], [0, 2], [1, 3]], 2),
                    face_edge=_masked_rows([[0, 3, 1, 2]], 4),
                    face_face=_masked_rows([[-1] * 4], 4),
                    edge_face=_masked_rows([[0, -1]] * 4, 2),
                ),
                id="node-4",
            ),
        ],
    )
    def test_strip_cases(
        self, lazy_values, start_index, location_axis, location, indices, expected
    ):
        mesh = sample_strip_mesh(
            lazy_values=lazy_values,
            start_index=start_index,
            location_axis=location_axis,
        )
        index_set = _MeshIndexSet(indices=indices, mesh=mesh, location=location)

        # One manager access: reading `.indices` would realise a view.
        results = index_set._connectivity_manager.all_members._asdict()
        assert set(results) - {"boundary_node"} == set(expected)
        assert results["boundary_node"] is None
        for name, expected_indices in expected.items():
            connectivity = results[name]
            original = getattr(mesh, f"{name}_connectivity")
            assert connectivity.start_index == original.start_index
            assert connectivity.location_axis == original.location_axis
            actual = _by_location_zero_based(connectivity)
            assert actual.shape == expected_indices.shape
            _shared_utils.assert_masked_array_equal(actual, expected_indices)

    @pytest.mark.parametrize("location_axis", [0, 1])
    def test_strip_repeated_faces(self, lazy_values, location_axis):
        mesh = sample_strip_mesh(lazy_values=lazy_values, location_axis=location_axis)
        index_set = _MeshIndexSet(indices=[0, 0], mesh=mesh, location="face")

        results = index_set._connectivity_manager.all_members
        _shared_utils.assert_masked_array_equal(
            _by_location_zero_based(results.face_node),
            _masked_rows([[0, 1, 3, 2]] * 2, 4),
        )
        face_face = _by_location_zero_based(results.face_face)
        assert face_face.shape == (2, 4)
        assert np.ma.getmaskarray(face_face).all()

    def test_strip_stays_lazy(self):
        mesh = sample_strip_mesh(lazy_values=True)
        index_set = _MeshIndexSet(indices=[0, 2], mesh=mesh, location="face")

        connectivities = [
            c for c in index_set._connectivity_manager.all_members if c is not None
        ]
        assert len(connectivities) == 5
        for connectivity in connectivities:
            assert is_lazy_data(connectivity.core_indices()), connectivity.cf_role

    def test_boundary_node_dropped(self, lazy_values):
        mesh = sample_strip_mesh(lazy_values=lazy_values)
        mesh.add_connectivities(
            Connectivity(
                indices=[[0, 1], [1, 2]],
                cf_role="boundary_node_connectivity",
            )
        )
        index_set = _MeshIndexSet(indices=[0, 2], mesh=mesh, location="face")

        assert index_set._connectivity_manager.boundary_node is None

    def test_no_edge_node_drops_edge_connectivities(self, lazy_values):
        mesh = sample_strip_mesh(lazy_values=lazy_values)
        mesh.remove_connectivities(cf_role="edge_node_connectivity")
        index_set = _MeshIndexSet(indices=[0, 2], mesh=mesh, location="face")

        results = index_set._connectivity_manager.all_members
        assert results.edge_node is None
        assert results.face_edge is None
        assert results.edge_face is None
        assert results.face_node is not None
        assert results.face_face is not None


class Test_as_mesh:
    def test_creation(self, meshes_locs_all):
        mesh, location = meshes_locs_all
        index_set = _MeshIndexSet(indices=[0, 2], mesh=mesh, location=location)
        new_mesh = index_set.as_mesh()

        assert isinstance(new_mesh, MeshXY)
        assert new_mesh is not mesh
        match location:
            case "node":
                assert (
                    new_mesh.node_coords.node_x.shape
                    == index_set.node_coords.node_x.shape
                )
                assert (
                    new_mesh.node_coords.node_y.shape
                    == index_set.node_coords.node_y.shape
                )
            case "edge":
                assert (
                    new_mesh.edge_coords.edge_x.shape
                    == index_set.edge_coords.edge_x.shape
                )
                assert (
                    new_mesh.edge_coords.edge_y.shape
                    == index_set.edge_coords.edge_y.shape
                )
            case "face":
                assert (
                    new_mesh.face_coords.face_x.shape
                    == index_set.face_coords.face_x.shape
                )
                assert (
                    new_mesh.face_coords.face_y.shape
                    == index_set.face_coords.face_y.shape
                )

    def test_zero_length_face_coords(self, mesh_2d):
        # Edges 0 and 2 enclose no face.
        index_set = _MeshIndexSet(indices=[0, 2], mesh=mesh_2d, location="edge")
        new_mesh = index_set.as_mesh()

        assert new_mesh.face_coords.face_x.shape == (0,)
        assert new_mesh.face_coords.face_y.shape == (0,)
        assert new_mesh.edge_coords.edge_x.shape == (2,)

    def test_deep_copy_not_view(self, mesh_2d):
        index_set = _MeshIndexSet(indices=[0, 2], mesh=mesh_2d, location="face")
        new_mesh = index_set.as_mesh()

        _shared_utils.assert_array_equal(
            new_mesh.face_coords.face_x.points, [3100, 3102]
        )
        # Changing the original mesh should not affect the new mesh.
        mesh_2d.face_coords.face_x.points = np.array([999, 998, 997])
        _shared_utils.assert_array_equal(
            new_mesh.face_coords.face_x.points, [3100, 3102]
        )


class Test_meshcoord_interop:
    @pytest.mark.parametrize("creator", [MeshCoord, _MeshIndexSet.to_MeshCoord])
    def test_meshcoord_from_index_set_location_must_match(
        self, meshes_locs_all, mesh_2d, creator
    ):
        mesh, location = meshes_locs_all
        index_set = _MeshIndexSet(indices=[0, 1], mesh=mesh, location=location)

        wrong_location = "face" if location != "face" else "edge"
        with pytest.raises(ValueError, match="does not match the location"):
            creator(index_set, location=wrong_location, axis="x")

    @pytest.mark.parametrize("creator", [MeshCoord, _MeshIndexSet.to_MeshCoord])
    def test_meshcoord_from_index_set(self, meshes_locs_all, mesh_2d, creator):
        mesh, location = meshes_locs_all
        index_set = _MeshIndexSet(indices=[0, 1], mesh=mesh, location=location)
        meshcoord = creator(index_set, location=location, axis="x")

        assert meshcoord.mesh is index_set
        assert meshcoord.shape == (2,)


class Test__str_repr:
    @pytest.fixture(autouse=True)
    def _setup(self, meshes_locs_all):
        mesh, location = meshes_locs_all
        mesh.rename("test_mesh")
        self.mesh = mesh
        self.location = location
        self.index_set = _MeshIndexSet(indices=[0, 1], mesh=mesh, location=location)

    def test_repr_unnamed(self):
        # When the index set has no name, repr mimics object.__str__ style.
        result = repr(self.index_set)
        assert re.match(r"<_MeshIndexSet object at 0x[0-9a-f]+>", result)

    def test_repr_named(self):
        # When the index set has a name, repr uses the human-readable form.
        self.index_set.long_name = "my_index_set"
        result = repr(self.index_set)
        assert result == "<_MeshIndexSet: 'my_index_set'>"

    def test_repr_var_name(self):
        # var_name is used as the name when long_name is absent.
        self.index_set.var_name = "idx"
        result = repr(self.index_set)
        assert result == "<_MeshIndexSet: 'idx'>"

    def test_str_contains_class_name(self):
        result = str(self.index_set)
        assert result.startswith("_MeshIndexSet : ")

    def test_str_contains_mesh_repr(self):
        result = str(self.index_set)
        assert "mesh: <MeshXY: 'test_mesh'>" in result

    def test_str_contains_location(self):
        result = str(self.index_set)
        assert f"location: {self.location}" in result

    def test_str_contains_start_index_default(self):
        result = str(self.index_set)
        assert "start_index: 0" in result

    def test_str_contains_start_index_nonzero(self):
        index_set = _MeshIndexSet(
            indices=[1, 2],
            mesh=self.mesh,
            location=self.location,
            start_index=1,
        )
        result = str(index_set)
        assert "start_index: 1" in result

    def test_str_contains_mesh_info_summary(self):
        result = str(self.index_set)
        assert "mesh info summary:" in result

    def test_str_mesh_info_includes_topology_dimension(self):
        result = str(self.index_set)
        assert f"topology_dimension: {self.mesh.topology_dimension}" in result

    def test_str_nameless_mesh_uses_object_repr(self):
        mesh = sample_mesh()  # no name
        index_set = _MeshIndexSet(indices=[0], mesh=mesh, location="node")
        result = str(index_set)
        assert re.search(r"mesh: <MeshXY object at 0x[0-9a-f]+>", result)

    def test_str_structure(self):
        # Coarse structure check: the key header lines appear together at the top.
        result = str(self.index_set)
        expected_header = "\n".join(
            [
                "_MeshIndexSet : 'unknown'",
                "    mesh: <MeshXY: 'test_mesh'>",
                f"    location: {self.location}",
                "    start_index: 0",
                "    mesh info summary:",
            ]
        )
        assert result.startswith(expected_header)


class Test_deferred_views:
    @pytest.mark.parametrize("update_mode", ["edit", "replace"])
    def test_coord_view_is_lazy_and_updates(self, mesh_2d, update_mode):
        index_set = _MeshIndexSet(indices=[0, 2], mesh=mesh_2d, location="face")

        coord_before = index_set.coord(location="face", axis="x")
        assert coord_before.has_lazy_points()
        _shared_utils.assert_array_equal(coord_before.points, [3100, 3102])
        assert coord_before.bounds is None

        # Attempting edits on the index set has no effect.
        ignored_points = [2000, 2002]
        ignored_bounds = [[2000, 2001], [2002, 2003]]
        coord_before.points = ignored_points
        coord_before.bounds = ignored_bounds
        coord_unchanged = index_set.coord(location="face", axis="x")
        assert coord_unchanged.has_lazy_points()
        _shared_utils.assert_array_equal(coord_unchanged.points, [3100, 3102])
        assert coord_unchanged.bounds is None

        new_points = [4400, 4401, 4402]
        new_bounds = [[4400, 4401], [4401, 4402], [4402, 4403]]
        if update_mode == "edit":
            mesh_2d.face_coords.face_x.points = new_points
            mesh_2d.face_coords.face_x.bounds = new_bounds
        else:
            replacement = mesh_2d.face_coords.face_x.copy(
                points=new_points, bounds=new_bounds
            )
            mesh_2d.add_coords(face_x=replacement)

        coord_after = index_set.coord(location="face", axis="x")
        assert coord_after.has_lazy_points()
        _shared_utils.assert_array_equal(coord_after.points, [4400, 4402])
        _shared_utils.assert_array_equal(
            coord_after.bounds, [[4400, 4401], [4402, 4403]]
        )

    @pytest.mark.parametrize("update_mode", ["edit", "replace"])
    def test_connectivity_view_is_lazy_and_updates(self, mesh_2d, update_mode):
        index_set = _MeshIndexSet(indices=[0, 2], mesh=mesh_2d, location="face")

        conn_before = index_set.connectivity(cf_role="face_node_connectivity")
        assert conn_before.has_lazy_indices()
        _shared_utils.assert_array_equal(
            conn_before.indices, [[0, 1, 2, 3], [4, 5, 6, 7]]
        )

        # Attempting edits on the index set has no effect.
        ignored_indices = [[100, 101, 102, 103], [104, 105, 106, 107]]
        conn_before._values = ignored_indices
        conn_unchanged = index_set.connectivity(cf_role="face_node_connectivity")
        assert conn_unchanged.has_lazy_indices()
        _shared_utils.assert_array_equal(
            conn_unchanged.indices, [[0, 1, 2, 3], [4, 5, 6, 7]]
        )

        new_indices = np.array(
            [
                [3, 2, 1, 0],
                [4, 5, 6, 7],
                [11, 10, 9, 8],
            ]
        )
        if update_mode == "edit":
            mesh_2d.face_node_connectivity._values = new_indices
        else:
            replacement = mesh_2d.face_node_connectivity.copy(new_indices)
            mesh_2d.add_connectivities(replacement)

        conn_after = index_set.connectivity(cf_role="face_node_connectivity")
        assert conn_after.has_lazy_indices()
        _shared_utils.assert_array_equal(
            conn_after.indices, [[3, 2, 1, 0], [7, 6, 5, 4]]
        )
