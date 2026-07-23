# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.mesh.components._MeshIndexSet` class."""

import numpy as np
import pytest

from iris._lazy_data import as_lazy_data, is_lazy_data
from iris.mesh import MeshCoord, MeshXY
from iris.mesh.components import _MeshIndexSet
from iris.tests import _shared_utils
import iris.tests.stock.mesh


@pytest.fixture
def mesh():
    return iris.tests.stock.mesh.sample_mesh()


class Test___init__:
    def test_basic(self, mesh):
        indices = np.array([0, 2])
        index_set = _MeshIndexSet(indices=indices, mesh=mesh, location="face")

        assert index_set.cf_role == "location_index_set"
        assert index_set.mesh is mesh
        assert index_set.location == "face"
        assert index_set.start_index == 0
        assert index_set.topology_dimension == mesh.topology_dimension
        _shared_utils.assert_array_equal(index_set.indices, indices)

    def test_fail_invalid_mesh(self):
        with pytest.raises(TypeError, match="`mesh` must be `MeshXY`"):
            _MeshIndexSet(indices=[0], mesh="not-a-mesh", location="face")

    def test_fail_invalid_location(self, mesh):
        with pytest.raises(ValueError, match="`location` must be in"):
            _MeshIndexSet(indices=[0], mesh=mesh, location="bad")

    def test_fail_invalid_start_index(self, mesh):
        with pytest.raises(ValueError, match="`start_index must be 0 or 1"):
            _MeshIndexSet(indices=[0], mesh=mesh, location="face", start_index=3)

    def test___getstate____setstate__(self, mesh):
        original = _MeshIndexSet(indices=np.array([0, 1]), mesh=mesh, location="edge")
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

    def test_scalar_index(self, mesh):
        index_set = _MeshIndexSet(indices=2, mesh=mesh, location="face")

        _shared_utils.assert_array_equal(index_set.indices, np.array([2]))
        coord = index_set.coord(location="face", axis="x")
        _shared_utils.assert_array_equal(coord.points, [3102])


class Test_properties:
    def test_cf_role(self, mesh):
        index_set = _MeshIndexSet(indices=np.array([0]), mesh=mesh, location="node")
        assert index_set.cf_role == "location_index_set"

    def test_dimension_properties(self, mesh):
        index_set = _MeshIndexSet(indices=np.array([0]), mesh=mesh, location="node")
        assert index_set.node_dimension == "_MeshIndexSet_NotImplemented"
        assert index_set.edge_dimension == "_MeshIndexSet_NotImplemented"
        assert index_set.face_dimension == "_MeshIndexSet_NotImplemented"

    def test_metadata_properties(self, mesh):
        indices = np.array([0, 2])
        index_set = _MeshIndexSet(
            indices=indices,
            mesh=mesh,
            location="face",
            long_name="my-index-set",
            start_index=1,
        )
        _shared_utils.assert_array_equal(index_set.indices, indices)
        assert index_set.mesh is mesh
        assert index_set.location == "face"
        assert index_set.start_index == 1
        assert index_set.topology_dimension == mesh.topology_dimension
        assert index_set.long_name == "my-index-set"


class Test_index_calculations:
    def test_node_location_calculate_node_bool_index(self, mesh):
        index_set = _MeshIndexSet(
            indices=np.array([0, 2, 4]), mesh=mesh, location="node"
        )
        result = index_set._calculate_node_bool_index()

        expected = np.zeros(mesh.node_coords.node_x.shape[0], dtype=bool)
        expected[[0, 2, 4]] = True
        _shared_utils.assert_array_equal(result, expected)

    def test_node_location_requires_monotonic_indices(self, mesh):
        index_set = _MeshIndexSet(indices=np.array([2, 1]), mesh=mesh, location="node")

        with pytest.raises(ValueError, match="requires strictly increasing indices"):
            index_set._calculate_node_bool_index()

    def test_edge_location_calculate_node_bool_index(self, mesh):
        index_set = _MeshIndexSet(indices=np.array([0, 1]), mesh=mesh, location="edge")
        result = index_set._calculate_node_bool_index()

        expected = np.zeros(mesh.node_coords.node_x.shape[0], dtype=bool)
        expected[[5, 6, 7, 8]] = True
        _shared_utils.assert_array_equal(result, expected)

    def test_face_location_calculate_node_bool_index(self, mesh):
        index_set = _MeshIndexSet(indices=np.array([1]), mesh=mesh, location="face")
        result = index_set._calculate_node_bool_index()

        expected = np.zeros(mesh.node_coords.node_x.shape[0], dtype=bool)
        expected[[4, 5, 6, 7]] = True
        _shared_utils.assert_array_equal(result, expected)

    def test_calculate_edge_indices(self, mesh):
        index_set = _MeshIndexSet(indices=np.array([1]), mesh=mesh, location="edge")
        _shared_utils.assert_array_equal(
            index_set._calculate_edge_indices(), np.array([1])
        )

        node_index_set = _MeshIndexSet(
            indices=np.array([1]), mesh=mesh, location="node"
        )
        assert node_index_set._calculate_edge_indices() is None

    def test_calculate_face_indices(self, mesh):
        index_set = _MeshIndexSet(indices=np.array([1]), mesh=mesh, location="face")
        _shared_utils.assert_array_equal(
            index_set._calculate_face_indices(), np.array([1])
        )

        node_index_set = _MeshIndexSet(
            indices=np.array([1]), mesh=mesh, location="node"
        )
        assert node_index_set._calculate_face_indices() is None


class Test_managers_and_views:
    def test_coord_manager_subset_for_face_index(self, mesh):
        index_set = _MeshIndexSet(indices=np.array([0, 2]), mesh=mesh, location="face")

        assert mesh.node_coords.node_x.shape == (15,)
        assert mesh.face_coords.face_x.shape == (3,)

        coord_manager = index_set._coord_manager
        assert coord_manager.node_x.shape == (8,)
        assert coord_manager.node_y.shape == (8,)
        assert coord_manager.face_x.shape == (2,)
        assert coord_manager.face_y.shape == (2,)

        _shared_utils.assert_array_equal(
            coord_manager.face_x.points,
            mesh.face_coords.face_x.points[[0, 2]],
        )

    def test_connectivity_manager_subset_for_face_index(self, mesh):
        index_set = _MeshIndexSet(indices=np.array([0, 2]), mesh=mesh, location="face")

        assert mesh.face_node_connectivity.shape == (3, 4)

        connectivity_manager = index_set._connectivity_manager
        assert connectivity_manager.face_node is not None
        assert connectivity_manager.edge_node is None
        assert connectivity_manager.is_view
        assert connectivity_manager.face_node.shape == (2, 4)

    def test_coord_manager_setter_forbidden(self, mesh):
        index_set = _MeshIndexSet(indices=np.array([0]), mesh=mesh, location="face")

        with pytest.raises(NotImplementedError, match="Modification of _MeshIndexSet"):
            index_set._coord_manager = mesh._coord_manager

    def test_connectivity_manager_setter_forbidden(self, mesh):
        index_set = _MeshIndexSet(indices=np.array([0]), mesh=mesh, location="face")

        with pytest.raises(NotImplementedError, match="Modification of _MeshIndexSet"):
            index_set._connectivity_manager = mesh._connectivity_manager


class Test_as_mesh:
    def test_basic(self, mesh):
        index_set = _MeshIndexSet(indices=np.array([0, 2]), mesh=mesh, location="face")
        new_mesh = index_set.as_mesh()

        assert isinstance(new_mesh, MeshXY)
        assert new_mesh is not mesh
        assert new_mesh.face_coords.face_x.shape == index_set.face_coords.face_x.shape
        assert new_mesh.face_coords.face_y.shape == index_set.face_coords.face_y.shape
        assert new_mesh.node_coords.node_x.shape == index_set.node_coords.node_x.shape
        assert new_mesh.node_coords.node_y.shape == index_set.node_coords.node_x.shape

    def test_deep_copy_not_view(self, mesh):
        index_set = _MeshIndexSet(indices=np.array([0, 2]), mesh=mesh, location="face")
        new_mesh = index_set.as_mesh()

        _shared_utils.assert_array_equal(
            new_mesh.face_coords.face_x.points, [3100, 3102]
        )
        # Changing the original mesh should not affect the new mesh.
        mesh.face_coords.face_x.points = np.array([999, 998, 997])
        _shared_utils.assert_array_equal(
            new_mesh.face_coords.face_x.points, [3100, 3102]
        )


class Test_meshcoord_interop:
    @pytest.mark.parametrize("creator", [MeshCoord, _MeshIndexSet.to_MeshCoord])
    def test_meshcoord_from_index_set_location_must_match(self, mesh, creator):
        index_set = _MeshIndexSet(indices=np.array([0, 1]), mesh=mesh, location="face")

        with pytest.raises(ValueError, match="does not match the location"):
            creator(index_set, location="edge", axis="x")

    @pytest.mark.parametrize("creator", [MeshCoord, _MeshIndexSet.to_MeshCoord])
    def test_meshcoord_from_index_set(self, mesh, creator):
        index_set = _MeshIndexSet(indices=np.array([0, 1]), mesh=mesh, location="face")
        meshcoord = creator(index_set, location="face", axis="x")

        assert meshcoord.mesh is index_set
        assert meshcoord.shape == (2,)


class Test_deferred_views:
    @pytest.mark.parametrize("update_mode", ["edit", "replace"])
    def test_coord_view_is_lazy_and_updates(self, mesh, update_mode):
        index_set = _MeshIndexSet(indices=np.array([0, 2]), mesh=mesh, location="face")

        coord_before = index_set.coord(location="face", axis="x")
        assert coord_before.has_lazy_points()
        _shared_utils.assert_array_equal(coord_before.points, [3100, 3102])
        assert coord_before.bounds is None

        new_points = [4400, 4401, 4402]
        new_bounds = [[4400, 4401], [4401, 4402], [4402, 4403]]
        if update_mode == "edit":
            mesh.face_coords.face_x.points = new_points
            mesh.face_coords.face_x.bounds = new_bounds
        else:
            replacement = mesh.face_coords.face_x.copy(
                points=new_points, bounds=new_bounds
            )
            mesh.add_coords(face_x=replacement)

        coord_after = index_set.coord(location="face", axis="x")
        assert coord_after.has_lazy_points()
        _shared_utils.assert_array_equal(coord_after.points, [4400, 4402])
        _shared_utils.assert_array_equal(
            coord_after.bounds, [[4400, 4401], [4402, 4403]]
        )

    @pytest.mark.parametrize("update_mode", ["edit", "replace"])
    def test_connectivity_view_is_lazy_and_updates(self, mesh, update_mode):
        index_set = _MeshIndexSet(indices=np.array([0, 2]), mesh=mesh, location="face")

        conn_before = index_set.connectivity(cf_role="face_node_connectivity")
        assert conn_before.has_lazy_indices()
        _shared_utils.assert_array_equal(
            conn_before.indices, [[0, 1, 2, 3], [4, 5, 6, 7]]
        )

        new_indices = np.array(
            [
                [3, 2, 1, 0],
                [4, 5, 6, 7],
                [11, 10, 9, 8],
            ]
        )
        if update_mode == "edit":
            mesh.face_node_connectivity._values = new_indices
        else:
            replacement = mesh.face_node_connectivity.copy(new_indices)
            mesh.add_connectivities(replacement)

        conn_after = index_set.connectivity(cf_role="face_node_connectivity")
        assert conn_after.has_lazy_indices()
        _shared_utils.assert_array_equal(
            conn_after.indices, [[3, 2, 1, 0], [7, 6, 5, 4]]
        )
