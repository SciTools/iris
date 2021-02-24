# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :class:`iris.experimental.ugrid.Mesh` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

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


class Test1DTopology(tests.IrisTest):
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

    def test_edge_dimension_set(self):
        # Don't modify self.mesh, which would prevent re-use.
        new_mesh = ugrid.Mesh(**self.KWARGS)
        new_mesh.edge_dimension = "foo"
        self.assertEqual("foo", new_mesh.edge_dimension)

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

    def test_face_dimension_set(self):
        # Don't modify self.mesh, which would prevent re-use.
        new_mesh = ugrid.Mesh(**self.KWARGS)
        with self.assertLogs(ugrid.logger, level="DEBUG") as log:
            new_mesh.face_dimension = "foo"
            self.assertIn("Not setting face_dimension", log.output[0])
        self.assertIsNone(new_mesh.face_dimension)

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

    def test_node_dimension_set(self):
        # Don't modify self.mesh, which would prevent re-use.
        new_mesh = ugrid.Mesh(**self.KWARGS)
        new_mesh.node_dimension = "foo"
        self.assertEqual("foo", new_mesh.node_dimension)

    def test_topology_dimension(self):
        self.assertEqual(
            self.KWARGS["topology_dimension"], self.mesh.topology_dimension
        )


class Test2DTopology(Test1DTopology):
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

    def test_face_dimension_set(self):
        # Don't modify self.mesh, which would prevent re-use.
        new_mesh = ugrid.Mesh(**self.KWARGS)
        new_mesh.face_dimension = "foo"
        self.assertEqual("foo", new_mesh.face_dimension)

    def test_face_edge(self):
        self.assertEqual(FACE_EDGE, self.mesh.face_edge_connectivity)

    def test_face_face(self):
        self.assertEqual(FACE_FACE, self.mesh.face_face_connectivity)

    def test_face_node(self):
        self.assertEqual(FACE_NODE, self.mesh.face_node_connectivity)
