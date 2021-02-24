# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :class:`iris.experimental.ugrid.Mesh` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.coords import AuxCoord
from iris.experimental import ugrid

# A collection of minimal coords and connectivities describing an equilateral triangle.
NODE_LON = AuxCoord([0, 2, 1], standard_name="longitude", var_name="node_lon")
NODE_LAT = AuxCoord([0, 0, 1], standard_name="latitude", var_name="node_lat")
EDGE_LON = AuxCoord(
    [1, 1.5, 0.5], standard_name="longitude", var_name="edge_lon"
)
EDGE_LAT = AuxCoord(
    [0, 0.5, 0.5], standard_name="latitude", var_name="edge_lat"
)
FACE_LON = AuxCoord([0.5], standard_name="longitude", var_name="face_lon")
FACE_LAT = AuxCoord([0.5], standard_name="latitude", var_name="face_lat")

EDGE_NODE = ugrid.Connectivity(
    [[0, 1], [1, 2], [2, 0]], cf_role="edge_node_connectivity"
)
FACE_NODE = ugrid.Connectivity([[0, 1, 2]], cf_role="face_node_connectivity")
FACE_EDGE = ugrid.Connectivity([[0, 1, 2]], cf_role="face_edge_connectivity")
# Actually meaningless:
FACE_FACE = ugrid.Connectivity([[0, 0, 0]], cf_role="face_face_connectivity")
# Actually meaningless:
EDGE_FACE = ugrid.Connectivity(
    [[0, 0], [0, 0], [0, 0]], cf_role="edge_face_connectivity"
)
BOUNDARY_NODE = ugrid.Connectivity(
    [[0, 1], [1, 2], [2, 0]], cf_role="boundary_node_connectivity"
)


class TestProperties1D(tests.IrisTest):
    # Tests that can re-use a single instance for greater efficiency.

    # Mesh kwargs with topology_dimension=1 and all applicable arguments
    # populated - this tests correct property setting.
    KWARGS = {
        "topology_dimension": 1,
        "node_coords_and_axes": ((NODE_LON, "x"), (NODE_LAT, "y")),
        "connectivities": EDGE_NODE,
        "long_name": "my_topology_mesh",
        "var_name": "mesh",
        "attributes": {"notes": "this is a test"},
        "node_dimension": "NodeDim",
        "edge_dimension": "EdgeDim",
        "edge_coords_and_axes": ((EDGE_LON, "x"), (EDGE_LAT, "y")),
    }

    @classmethod
    def setUpClass(cls):
        cls.mesh = ugrid.Mesh(**cls.KWARGS)

    def test___getstate__(self):
        expected = (
            self.mesh._metadata_manager,
            self.mesh._coord_manager,
            self.mesh._connectivity_manager,
        )
        self.assertEqual(expected, self.mesh.__getstate__())

    def test_all_connectivities(self):
        expected = ugrid.Mesh1DConnectivities(EDGE_NODE)
        self.assertEqual(expected, self.mesh.all_connectivities)

    def test_all_coords(self):
        expected = ugrid.Mesh1DCoords(NODE_LON, NODE_LAT, EDGE_LON, EDGE_LAT)
        self.assertEqual(expected, self.mesh.all_coords)

    def test_boundary_node(self):
        with self.assertRaises(AttributeError):
            _ = self.mesh.boundary_node_connectivity

    def test_edge_dimension(self):
        self.assertEqual(
            self.KWARGS["edge_dimension"], self.mesh.edge_dimension
        )

    def test_edge_coords(self):
        expected = ugrid.MeshEdgeCoords(EDGE_LON, EDGE_LAT)
        self.assertEqual(expected, self.mesh.edge_coords)

    def test_edge_face(self):
        with self.assertRaises(AttributeError):
            _ = self.mesh.edge_face_connectivity

    def test_edge_node(self):
        self.assertEqual(EDGE_NODE, self.mesh.edge_node_connectivity)

    def test_face_coords(self):
        with self.assertRaises(AttributeError):
            _ = self.mesh.face_coords

    def test_face_dimension(self):
        self.assertIsNone(self.mesh.face_dimension)

    def test_face_edge(self):
        with self.assertRaises(AttributeError):
            _ = self.mesh.face_edge_connectivity

    def test_face_face(self):
        with self.assertRaises(AttributeError):
            _ = self.mesh.face_face_connectivity

    def test_face_node(self):
        with self.assertRaises(AttributeError):
            _ = self.mesh.face_node_connectivity

    def test_node_coords(self):
        expected = ugrid.MeshNodeCoords(NODE_LON, NODE_LAT)
        self.assertEqual(expected, self.mesh.node_coords)

    def test_node_dimension(self):
        self.assertEqual(
            self.KWARGS["node_dimension"], self.mesh.node_dimension
        )

    def test_topology_dimension(self):
        self.assertEqual(
            self.KWARGS["topology_dimension"], self.mesh.topology_dimension
        )


class TestProperties2D(TestProperties1D):
    # Additional/specialised tests for topology_dimension=2.
    @classmethod
    def setUpClass(cls):
        cls.KWARGS["topology_dimension"] = 2
        cls.KWARGS["connectivities"] = (
            FACE_NODE,
            EDGE_NODE,
            FACE_EDGE,
            FACE_FACE,
            EDGE_FACE,
            BOUNDARY_NODE,
        )
        cls.KWARGS["face_dimension"] = "FaceDim"
        cls.KWARGS["face_coords_and_axes"] = ((FACE_LON, "x"), (FACE_LAT, "y"))
        super().setUpClass()

    def test_all_connectivities(self):
        expected = ugrid.Mesh2DConnectivities(
            FACE_NODE,
            EDGE_NODE,
            FACE_EDGE,
            FACE_FACE,
            EDGE_FACE,
            BOUNDARY_NODE,
        )
        self.assertEqual(expected, self.mesh.all_connectivities)

    def test_all_coords(self):
        expected = ugrid.Mesh2DCoords(
            NODE_LON, NODE_LAT, EDGE_LON, EDGE_LAT, FACE_LON, FACE_LAT
        )
        self.assertEqual(expected, self.mesh.all_coords)

    def test_boundary_node(self):
        self.assertEqual(BOUNDARY_NODE, self.mesh.boundary_node_connectivity)

    def test_edge_face(self):
        self.assertEqual(EDGE_FACE, self.mesh.edge_face_connectivity)

    def test_face_coords(self):
        expected = ugrid.MeshFaceCoords(FACE_LON, FACE_LAT)
        self.assertEqual(expected, self.mesh.face_coords)

    def test_face_dimension(self):
        self.assertEqual(
            self.KWARGS["face_dimension"], self.mesh.face_dimension
        )

    def test_face_edge(self):
        self.assertEqual(FACE_EDGE, self.mesh.face_edge_connectivity)

    def test_face_face(self):
        self.assertEqual(FACE_FACE, self.mesh.face_face_connectivity)

    def test_face_node(self):
        self.assertEqual(FACE_NODE, self.mesh.face_node_connectivity)


class TestOperations1D(tests.IrisTest):
    # Tests that cannot re-use an existing Mesh instance, instead need a new
    # one each time.
    def setUp(self):
        self.mesh = ugrid.Mesh(
            topology_dimension=1,
            node_coords_and_axes=((NODE_LON, "x"), (NODE_LAT, "y")),
            connectivities=EDGE_NODE,
        )

    def test___setstate__(self):
        false_metadata_manager = "foo"
        false_coord_manager = "bar"
        false_connectivity_manager = "baz"
        self.mesh.__setstate__(
            (
                false_metadata_manager,
                false_coord_manager,
                false_connectivity_manager,
            )
        )

        self.assertEqual(false_metadata_manager, self.mesh._metadata_manager)
        self.assertEqual(false_coord_manager, self.mesh._coord_manager)
        self.assertEqual(
            false_connectivity_manager, self.mesh._connectivity_manager
        )

    def test_add_coords(self):
        # Test coord addition AND replacement.
        node_kwargs = {
            "node_x": NODE_LON.copy(np.zeros(NODE_LON.shape)),
            "node_y": NODE_LAT.copy(np.zeros(NODE_LAT.shape)),
        }
        edge_kwargs = {"edge_x": EDGE_LON, "edge_y": EDGE_LAT}
        self.mesh.add_coords(**node_kwargs, **edge_kwargs)

        self.assertEqual(
            ugrid.MeshNodeCoords(**node_kwargs), self.mesh.node_coords
        )
        self.assertEqual(
            ugrid.MeshEdgeCoords(**edge_kwargs), self.mesh.edge_coords
        )

    def test_add_coords_face(self):
        self.assertRaises(
            TypeError, self.mesh.add_coords, face_x=FACE_LON, face_y=FACE_LAT
        )

    def test_edge_dimension_set(self):
        self.mesh.edge_dimension = "foo"
        self.assertEqual("foo", self.mesh.edge_dimension)

    def test_face_dimension_set(self):
        with self.assertLogs(ugrid.logger, level="DEBUG") as log:
            self.mesh.face_dimension = "foo"
            self.assertIn("Not setting face_dimension", log.output[0])
        self.assertIsNone(self.mesh.face_dimension)

    def test_node_dimension_set(self):
        self.mesh.node_dimension = "foo"
        self.assertEqual("foo", self.mesh.node_dimension)


class TestOperations2D(TestOperations1D):
    # Additional/specialised tests for topology_dimension=2.
    def setUp(self):
        self.mesh = ugrid.Mesh(
            topology_dimension=2,
            node_coords_and_axes=((NODE_LON, "x"), (NODE_LAT, "y")),
            connectivities=(FACE_NODE, EDGE_NODE),
        )

    def test_add_coords_face(self):
        kwargs = {"face_x": FACE_LON, "face_y": FACE_LAT}
        self.mesh.add_coords(**kwargs)
        expected = ugrid.MeshFaceCoords(**kwargs)
        self.assertEqual(expected, self.mesh.face_coords)

    def test_face_dimension_set(self):
        self.mesh.face_dimension = "foo"
        self.assertEqual("foo", self.mesh.face_dimension)
