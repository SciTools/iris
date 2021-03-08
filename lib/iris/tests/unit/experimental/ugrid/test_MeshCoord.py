# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :class:`iris.experimental.ugrid.MeshCoord`.

"""
# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import dask.array as da
import numpy as np
import unittest.mock as mock

from iris.coords import AuxCoord, Coord
from iris.common.metadata import BaseMetadata
from iris.cube import Cube
from iris.experimental.ugrid import Connectivity, Mesh

from iris.experimental.ugrid import MeshCoord

# Default creation controls for creating a test Mesh.
# Note: we're not creating any kind of sensible 'normal' mesh here, the numbers
# of nodes/faces/edges are quite arbitrary and the connectivities we generate
# are pretty random too.
_TEST_N_NODES = 15
_TEST_N_FACES = 3
_TEST_N_EDGES = 5
_TEST_N_BOUNDS = 4

# Default actual points + bounds.
_TEST_POINTS = np.arange(_TEST_N_FACES)
_TEST_BOUNDS = np.arange(_TEST_N_FACES * _TEST_N_BOUNDS)
_TEST_BOUNDS = _TEST_BOUNDS.reshape((_TEST_N_FACES, _TEST_N_BOUNDS))


def _create_test_mesh():
    node_x = AuxCoord(
        1100 + np.arange(_TEST_N_NODES),
        standard_name="longitude",
        long_name="long-name",
        var_name="var",
    )
    node_y = AuxCoord(
        1200 + np.arange(_TEST_N_NODES), standard_name="latitude"
    )

    # Define a rather arbitrary edge-nodes connectivity.
    # Some nodes are left out, because n_edges*2 < n_nodes.
    conns = np.arange(_TEST_N_EDGES * 2, dtype=int)
    # Missing nodes include #0-5, because we add 5.
    conns = ((conns + 5) % _TEST_N_NODES).reshape((_TEST_N_EDGES, 2))
    edge_nodes = Connectivity(conns, cf_role="edge_node_connectivity")
    conns = np.arange(_TEST_N_EDGES * 2, dtype=int)

    # Some numbers for the edge coordinates.
    edge_x = AuxCoord(
        2100 + np.arange(_TEST_N_EDGES), standard_name="longitude"
    )
    edge_y = AuxCoord(
        2200 + np.arange(_TEST_N_EDGES), standard_name="latitude"
    )

    # Define a rather arbitrary face-nodes connectivity.
    # Some nodes are left out, because n_faces*n_bounds < n_nodes.
    conns = np.arange(_TEST_N_FACES * _TEST_N_BOUNDS, dtype=int)
    conns = (conns % _TEST_N_NODES).reshape((_TEST_N_FACES, _TEST_N_BOUNDS))
    face_nodes = Connectivity(conns, cf_role="face_node_connectivity")

    # Some numbers for the edge coordinates.
    face_x = AuxCoord(
        3100 + np.arange(_TEST_N_FACES), standard_name="longitude"
    )
    face_y = AuxCoord(
        3200 + np.arange(_TEST_N_FACES), standard_name="latitude"
    )

    mesh = Mesh(
        topology_dimension=2,
        node_coords_and_axes=[(node_x, "x"), (node_y, "y")],
        connectivities=[face_nodes, edge_nodes],
        edge_coords_and_axes=[(edge_x, "x"), (edge_y, "y")],
        face_coords_and_axes=[(face_x, "x"), (face_y, "y")],
    )
    return mesh


def _default_create_args():
    # Produce a minimal set of default constructor args
    kwargs = {"location": "face", "axis": "x", "mesh": _create_test_mesh()}
    # NOTE: *don't* include coord_system or climatology.
    # We expect to only set those (non-default) explicitly.
    return kwargs


def _create_test_meshcoord(**override_kwargs):
    kwargs = _default_create_args()
    # Apply requested overrides and additions.
    kwargs.update(override_kwargs)
    # Create and return the test coord.
    result = MeshCoord(**kwargs)
    return result


class Test___init__(tests.IrisTest):
    def setUp(self):
        self.meshcoord = _create_test_meshcoord()

    def test_basic(self):
        kwargs = _default_create_args()
        meshcoord = _create_test_meshcoord(**kwargs)
        for key, val in kwargs.items():
            self.assertEqual(getattr(meshcoord, key), val)
        self.assertIsInstance(meshcoord, MeshCoord)
        self.assertIsInstance(meshcoord, Coord)

    def test_derived_properties(self):
        # Check the derived properties of the meshcoord against the correct
        # underlying mesh coordinate.
        for axis in Mesh.AXES:
            meshcoord = _create_test_meshcoord(axis=axis)
            # N.B.
            node_x_coord = meshcoord.mesh.coord(include_nodes=True, axis=axis)
            for key in node_x_coord.metadata._fields:
                meshval = getattr(meshcoord, key)
                if key == "var_name":
                    # var_name is unused.
                    self.assertIsNone(meshval)
                else:
                    # names, units and attributes are derived from the node coord.
                    self.assertEqual(meshval, getattr(node_x_coord, key))

    def test_fail_bad_mesh(self):
        with self.assertRaisesRegex(TypeError, "must be a.*Mesh"):
            _create_test_meshcoord(mesh=mock.sentinel.odd)

    def test_valid_locations(self):
        for loc in Mesh.LOCATIONS:
            meshcoord = _create_test_meshcoord(location=loc)
            self.assertEqual(meshcoord.location, loc)

    def test_fail_bad_location(self):
        with self.assertRaisesRegex(ValueError, "not a valid Mesh location"):
            _create_test_meshcoord(location="bad")

    def test_fail_bad_axis(self):
        with self.assertRaisesRegex(ValueError, "not a valid Mesh axis"):
            _create_test_meshcoord(axis="q")


class Test__readonly_properties(tests.IrisTest):
    def setUp(self):
        self.meshcoord = _create_test_meshcoord()

    def test_fixed_metadata(self):
        # Check that you cannot set any of these on an existing MeshCoord.
        meshcoord = self.meshcoord
        for prop in ("mesh", "location", "axis"):
            with self.assertRaisesRegex(AttributeError, "can't set"):
                setattr(meshcoord, prop, mock.sentinel.odd)

    def test_coord_system(self):
        # The property exists, =None, can set to None, can not set otherwise.
        self.assertTrue(hasattr(self.meshcoord, "coord_system"))
        self.assertIsNone(self.meshcoord.coord_system)
        self.meshcoord.coord_system = None
        with self.assertRaisesRegex(ValueError, "Cannot set.* MeshCoord"):
            self.meshcoord.coord_system = 1

    def test_set_climatological(self):
        # The property exists, =False, can set to False, can not set otherwise.
        self.assertTrue(hasattr(self.meshcoord, "climatological"))
        self.assertFalse(self.meshcoord.climatological)
        self.meshcoord.climatological = False
        with self.assertRaisesRegex(ValueError, "Cannot set.* MeshCoord"):
            self.meshcoord.climatological = True


class Test__inherited_properties(tests.IrisTest):
    """
    Check the settability and effect on equality of the common BaseMetadata
    properties inherited from Coord : i.e. names/units/attributes.

    Though copied from the mesh at creation, they are also changeable.

    """

    def setUp(self):
        self.meshcoord = _create_test_meshcoord()

    def test_inherited_properties(self):
        # Check that these are settable, and affect equality.
        meshcoord = self.meshcoord
        # Add an existing attribute, so we can change it.
        meshcoord.attributes["thing"] = 7
        for prop in BaseMetadata._fields:
            meshcoord2 = meshcoord.copy()
            if "name" in prop:
                # Use a standard-name, can do for any of them.
                setattr(meshcoord2, prop, "height")
            elif prop == "units":
                meshcoord2.units = "Pa"
            elif prop == "attributes":
                meshcoord2.attributes["thing"] = 77
        self.assertNotEqual(meshcoord2, meshcoord)


class Test__points_and_bounds(tests.IrisTest):
    # Basic method testing only, for 3 locations with simple array values.
    # See Test_MeshCoord__dataviews for more detailed checks.
    def test_node(self):
        meshcoord = _create_test_meshcoord(location="node")
        self.assertIsNone(meshcoord.core_bounds())
        self.assertArrayAllClose(
            meshcoord.points, 1100 + np.arange(_TEST_N_NODES)
        )

    def test_edge(self):
        meshcoord = _create_test_meshcoord(location="edge")
        points, bounds = meshcoord.core_points(), meshcoord.core_bounds()
        self.assertEqual(points.shape, meshcoord.shape)
        self.assertEqual(bounds.shape, meshcoord.shape + (2,))
        self.assertArrayAllClose(
            meshcoord.points, [2100, 2101, 2102, 2103, 2104]
        )
        self.assertArrayAllClose(
            meshcoord.bounds,
            [
                (1105, 1106),
                (1107, 1108),
                (1109, 1110),
                (1111, 1112),
                (1113, 1114),
            ],
        )

    def test_face(self):
        meshcoord = _create_test_meshcoord(location="face")
        points, bounds = meshcoord.core_points(), meshcoord.core_bounds()
        self.assertEqual(points.shape, meshcoord.shape)
        self.assertEqual(bounds.shape, meshcoord.shape + (4,))
        self.assertArrayAllClose(meshcoord.points, [3100, 3101, 3102])
        self.assertArrayAllClose(
            meshcoord.bounds,
            [
                (1100, 1101, 1102, 1103),
                (1104, 1105, 1106, 1107),
                (1108, 1109, 1110, 1111),
            ],
        )


class Test___eq__(tests.IrisTest):
    def setUp(self):
        self.mesh = _create_test_mesh()

    def _create_common_mesh(self, **kwargs):
        return _create_test_meshcoord(mesh=self.mesh, **kwargs)

    def test_same_mesh(self):
        meshcoord1 = self._create_common_mesh()
        meshcoord2 = self._create_common_mesh()
        self.assertEqual(meshcoord2, meshcoord1)

    def test_different_identical_mesh(self):
        # For equality, must have the SAME mesh (at present).
        mesh1 = _create_test_mesh()
        mesh2 = _create_test_mesh()  # Presumably identical, but not the same
        meshcoord1 = _create_test_meshcoord(mesh=mesh1)
        meshcoord2 = _create_test_meshcoord(mesh=mesh2)
        # These should NOT compare, because the Meshes are not identical : at
        # present, Mesh equality is not implemented (i.e. limited to identity)
        self.assertNotEqual(meshcoord2, meshcoord1)

    def test_different_location(self):
        meshcoord = self._create_common_mesh()
        meshcoord2 = self._create_common_mesh(location="node")
        self.assertNotEqual(meshcoord2, meshcoord)

    def test_different_axis(self):
        meshcoord = self._create_common_mesh()
        meshcoord2 = self._create_common_mesh(axis="y")
        self.assertNotEqual(meshcoord2, meshcoord)


class Test__copy(tests.IrisTest):
    def test_basic(self):
        meshcoord = _create_test_meshcoord()
        meshcoord2 = meshcoord.copy()
        self.assertIsNot(meshcoord2, meshcoord)
        self.assertEqual(meshcoord2, meshcoord)
        # In this case, they should share *NOT* copy the Mesh object.
        self.assertIs(meshcoord2.mesh, meshcoord.mesh)

    def test_fail_copy_newpoints(self):
        meshcoord = _create_test_meshcoord()
        with self.assertRaisesRegex(ValueError, "Cannot change the content"):
            meshcoord.copy(points=meshcoord.points)

    def test_fail_copy_newbounds(self):
        meshcoord = _create_test_meshcoord()
        with self.assertRaisesRegex(ValueError, "Cannot change the content"):
            meshcoord.copy(bounds=meshcoord.bounds)


class Test__getitem__(tests.IrisTest):
    def test_slice_wholeslice_1tuple(self):
        # The only slicing case that we support, to enable cube slicing.
        meshcoord = _create_test_meshcoord()
        meshcoord2 = meshcoord[
            :,
        ]
        self.assertIsNot(meshcoord2, meshcoord)
        self.assertEqual(meshcoord2, meshcoord)
        # In this case, we should *NOT* copy the linked Mesh object.
        self.assertIs(meshcoord2.mesh, meshcoord.mesh)

    def test_slice_whole_slice_singlekey(self):
        # A slice(None) also fails, if not presented in a 1-tuple.
        meshcoord = _create_test_meshcoord()
        with self.assertRaisesRegex(ValueError, "Cannot index"):
            meshcoord[:]

    def test_fail_slice_part(self):
        meshcoord = _create_test_meshcoord()
        with self.assertRaisesRegex(ValueError, "Cannot index"):
            meshcoord[:1]


class Test_cube_containment(tests.IrisTest):
    # Check that we can put a MeshCoord into a cube, and have it behave just
    # like a regular AuxCoord.
    def setUp(self):
        meshcoord = _create_test_meshcoord()
        data_shape = (2,) + _TEST_POINTS.shape
        cube = Cube(np.zeros(data_shape))
        cube.add_aux_coord(meshcoord, 1)
        self.meshcoord = meshcoord
        self.cube = cube

    def test_added_to_cube(self):
        meshcoord = self.meshcoord
        cube = self.cube
        self.assertIn(meshcoord, cube.coords())

    def test_cube_dims(self):
        meshcoord = self.meshcoord
        cube = self.cube
        self.assertEqual(meshcoord.cube_dims(cube), (1,))
        self.assertEqual(cube.coord_dims(meshcoord), (1,))

    def test_find_by_name(self):
        meshcoord = self.meshcoord
        cube = self.cube
        self.assertIs(cube.coord(standard_name="longitude"), meshcoord)
        self.assertIs(cube.coord(long_name="long-name"), meshcoord)

    def test_find_by_axis(self):
        meshcoord = self.meshcoord
        cube = self.cube
        self.assertIs(cube.coord(axis="x"), meshcoord)
        self.assertEqual(cube.coords(axis="y"), [])

        # NOTE: the meshcoord.axis takes precedence over the older
        # "guessed axis" approach.  So the standard_name does not control it.
        meshcoord.rename("latitude")
        self.assertIs(cube.coord(axis="x"), meshcoord)
        self.assertEqual(cube.coords(axis="y"), [])

    def test_cube_copy(self):
        # Check that we can copy a cube, and get a MeshCoord == the original.
        # Note: currently must have the *same* mesh, as for MeshCoord.copy().
        meshcoord = self.meshcoord
        cube = self.cube
        cube2 = cube.copy()
        meshco2 = cube2.coord(meshcoord)
        self.assertIsNot(meshco2, meshcoord)
        self.assertEqual(meshco2, meshcoord)

    def test_cube_nonmesh_slice(self):
        # Check that we can slice a cube on a non-mesh dimension, and get a
        # meshcoord == original.
        # Note: currently this must have the *same* mesh, as for .copy().
        meshcoord = self.meshcoord
        cube = self.cube
        cube2 = cube[:1]  # Make a reduced copy, slicing the non-mesh dim
        meshco2 = cube2.coord(meshcoord)
        self.assertIsNot(meshco2, meshcoord)
        self.assertEqual(meshco2, meshcoord)

    def test_cube_mesh_partslice(self):
        # Check that we can *not* get a partial MeshCoord slice, as the
        # MeshCoord refuses to be sliced.
        # Instead, you get an AuxCoord created from the MeshCoord.
        meshcoord = self.meshcoord
        cube = self.cube
        cube2 = cube[:, :1]  # Make a reduced copy, slicing the mesh dim

        # The resulting coord can not be identified with the original.
        # (i.e. metadata does not match)
        co_matches = cube2.coords(meshcoord)
        self.assertEqual(co_matches, [])

        # The resulting coord is an AuxCoord instead of a MeshCoord, but the
        # values match.
        co2 = cube2.coord(meshcoord.name())
        self.assertFalse(isinstance(co2, MeshCoord))
        self.assertIsInstance(co2, AuxCoord)
        self.assertArrayAllClose(co2.points, meshcoord.points[:1])
        self.assertArrayAllClose(co2.bounds, meshcoord.bounds[:1])


class Test_auxcoord_conversion(tests.IrisTest):
    def test_basic(self):
        meshcoord = _create_test_meshcoord()
        auxcoord = AuxCoord.from_coord(meshcoord)
        for propname, auxval in auxcoord.metadata._asdict().items():
            meshval = getattr(meshcoord, propname)
            self.assertEqual(auxval, meshval)
        # Also check array content.
        self.assertArrayAllClose(auxcoord.points, meshcoord.points)
        self.assertArrayAllClose(auxcoord.bounds, meshcoord.bounds)


class Test_MeshCoord__dataviews(tests.IrisTest):
    """
    Fuller testing of points and bounds calculations and behaviour.
    Including connectivity missing-points (non-square faces).

    """

    def setUp(self):
        self._make_test_meshcoord()

    def _make_test_meshcoord(
        self,
        lazy_sources=False,
        location="face",
        inds_start_index=0,
        inds_src_dim=0,
    ):
        # Construct a miniature face-nodes mesh for testing.
        # NOTE: we will make our connectivity arrays with standard
        # start_index=0 and src_dim=0 :  We only adjust that (if required) when
        # creating the actual connectivities.
        face_nodes_array = np.array(
            [
                [0, 2, 1, 3],
                [1, 3, 10, 13],
                [2, 7, 9, 19],
                [
                    3,
                    4,
                    7,
                    -1,
                ],  # This one has a "missing" point (it's a triangle)
                [8, 1, 7, 2],
            ]
        )
        # Connectivity uses *masked* for missing points.
        face_nodes_array = np.ma.masked_less(face_nodes_array, 0)

        # Construct a miniature edge-nodes mesh for testing.
        edge_nodes_array = np.array([[0, 2], [1, 3], [1, 4], [3, 7]])
        # Connectivity uses *masked* for missing points.
        edge_nodes_array = np.ma.masked_less(edge_nodes_array, 0)

        n_faces = face_nodes_array.shape[0]
        n_edges = edge_nodes_array.shape[0]
        n_nodes = int(face_nodes_array.max() + 1)
        face_xs = 500.0 + np.arange(n_faces)
        edge_xs = 600.0 + np.arange(n_edges)
        node_xs = 100.0 + np.arange(n_nodes)

        # Record all these for re-use in tests
        self.n_faces = n_faces
        self.n_nodes = n_nodes
        self.face_xs = face_xs
        self.node_xs = node_xs
        self.edge_xs = edge_xs
        self.face_nodes_array = face_nodes_array
        self.edge_nodes_array = edge_nodes_array

        # convert source data to Dask arrays if asked.
        if lazy_sources:

            def lazify(arr):
                return da.from_array(arr, chunks=-1, meta=np.ndarray)

            node_xs = lazify(node_xs)
            face_xs = lazify(face_xs)
            edge_xs = lazify(edge_xs)
            face_nodes_array = lazify(face_nodes_array)
            edge_nodes_array = lazify(edge_nodes_array)

        # Build a mesh with this info stored in it.
        co_nodex = AuxCoord(
            node_xs, standard_name="longitude", long_name="node_x", units=1
        )
        co_facex = AuxCoord(
            face_xs, standard_name="longitude", long_name="face_x", units=1
        )
        co_edgex = AuxCoord(
            edge_xs, standard_name="longitude", long_name="edge_x", units=1
        )
        # N.B. the Mesh requires 'Y's as well.
        co_nodey = co_nodex.copy()
        co_nodey.rename("latitude")
        co_nodey.long_name = "node_y"
        co_facey = co_facex.copy()
        co_facey.rename("latitude")
        co_facey.long_name = "face_y"
        co_edgey = co_edgex.copy()
        co_edgey.rename("edge_y")
        co_edgey.long_name = "edge_y"

        face_node_conn = Connectivity(
            inds_start_index
            + (
                face_nodes_array.transpose()
                if inds_src_dim == 1
                else face_nodes_array
            ),
            cf_role="face_node_connectivity",
            long_name="face_nodes",
            start_index=inds_start_index,
            src_dim=inds_src_dim,
        )

        edge_node_conn = Connectivity(
            inds_start_index
            + (
                edge_nodes_array.transpose()
                if inds_src_dim == 1
                else edge_nodes_array
            ),
            cf_role="edge_node_connectivity",
            long_name="edge_nodes",
            start_index=inds_start_index,
            src_dim=inds_src_dim,
        )

        self.mesh = Mesh(
            topology_dimension=2,
            node_coords_and_axes=[(co_nodex, "x"), (co_nodey, "y")],
            connectivities=[face_node_conn, edge_node_conn],
            face_coords_and_axes=[(co_facex, "x"), (co_facey, "y")],
            edge_coords_and_axes=[(co_edgex, "x"), (co_edgey, "y")],
        )

        # Construct a test meshcoord.
        meshcoord = MeshCoord(mesh=self.mesh, location=location, axis="x")
        self.meshcoord = meshcoord
        return meshcoord

    def _check_expected_points_values(self):
        # The points are just the face_x-s
        meshcoord = self.meshcoord
        self.assertArrayAllClose(meshcoord.points, self.face_xs)

    def _check_expected_bounds_values(self):
        mesh_coord = self.meshcoord
        # The bounds are selected node_x-s :  all == node_number + 100.0
        result = mesh_coord.bounds
        # N.B. result should be masked where the masked indices are.
        expected = 100.0 + self.face_nodes_array
        # Check there are *some* masked points.
        self.assertTrue(np.count_nonzero(expected.mask) > 0)
        # Check results match, *including* location of masked points.
        self.assertMaskedArrayAlmostEqual(result, expected)

    def test_points_values(self):
        """Basic points content check, on real data."""
        meshcoord = self.meshcoord
        self.assertFalse(meshcoord.has_lazy_points())
        self.assertFalse(meshcoord.has_lazy_points())
        self._check_expected_points_values()

    def test_bounds_values(self):
        """Basic bounds contents check."""
        meshcoord = self.meshcoord
        self.assertFalse(meshcoord.has_lazy_points())
        self.assertFalse(meshcoord.has_lazy_bounds())
        self._check_expected_bounds_values()

    def test_lazy_points_values(self):
        """Check lazy points calculation on lazy inputs."""
        # Remake the test data with lazy source coords.
        meshcoord = self._make_test_meshcoord(lazy_sources=True)
        self.assertTrue(meshcoord.has_lazy_points())
        self.assertTrue(meshcoord.has_lazy_bounds())
        # Check values, as previous.
        self._check_expected_points_values()

    def test_lazy_bounds_values(self):
        meshcoord = self._make_test_meshcoord(lazy_sources=True)
        self.assertTrue(meshcoord.has_lazy_points())
        self.assertTrue(meshcoord.has_lazy_bounds())
        # Check values, as previous.
        self._check_expected_bounds_values()

    def test_edge_points(self):
        meshcoord = self._make_test_meshcoord(location="edge")
        result = meshcoord.points
        self.assertArrayAllClose(result, self.edge_xs)

    def test_edge_bounds(self):
        meshcoord = self._make_test_meshcoord(location="edge")
        result = meshcoord.bounds
        # The bounds are selected node_x-s :  all == node_number + 100.0
        expected = 100.0 + self.edge_nodes_array
        # NB simpler than faces : no possibility of missing points
        self.assertArrayAlmostEqual(result, expected)

    def test_bounds_connectivity__src_dim_1(self):
        # Test with a transposed indices array.
        self._make_test_meshcoord(inds_src_dim=1)
        self._check_expected_bounds_values()

    def test_bounds_connectivity__start_index_1(self):
        # Test 1-based indices.
        self._make_test_meshcoord(inds_start_index=1)
        self._check_expected_bounds_values()

    def test_meshcoord_leaves_originals_lazy(self):
        self._make_test_meshcoord(lazy_sources=True)
        mesh = self.mesh
        meshcoord = self.meshcoord

        # Fetch the relevant source objects from the mesh.
        def fetch_sources_from_mesh():
            return (
                mesh.coord(include_nodes=True, axis="x"),
                mesh.coord(include_faces=True, axis="x"),
                mesh.face_node_connectivity,
            )

        # Check all the source coords are lazy.
        for coord in fetch_sources_from_mesh():
            self.assertTrue(hasattr(coord._core_values(), "compute"))

        # Calculate both points + bounds of the meshcoord
        self.assertTrue(meshcoord.has_lazy_points())
        self.assertTrue(meshcoord.has_lazy_bounds())
        meshcoord.points
        meshcoord.bounds
        self.assertFalse(meshcoord.has_lazy_points())
        self.assertFalse(meshcoord.has_lazy_bounds())

        # Check all the source coords are still lazy.
        for coord in fetch_sources_from_mesh():
            self.assertTrue(hasattr(coord._core_values(), "compute"))


if __name__ == "__main__":
    tests.main()
