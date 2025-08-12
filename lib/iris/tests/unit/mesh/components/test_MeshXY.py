# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.mesh.MeshXY` class."""

import re

import numpy as np
import pytest

from iris.common.metadata import MeshMetadata
from iris.coords import AuxCoord
from iris.exceptions import ConnectivityNotFoundError, CoordinateNotFoundError
from iris.mesh import components
from iris.mesh.components import logger
from iris.tests import _shared_utils


class TestMeshCommon:
    @staticmethod
    @pytest.fixture(scope="class", autouse=True)
    def make_common_inputs():
        cls = TestMeshCommon
        cls.NODE_LON = AuxCoord(
            [0, 2, 1],
            standard_name="longitude",
            long_name="long_name",
            var_name="node_lon",
            attributes={"test": 1},
        )
        cls.NODE_LAT = AuxCoord(
            [0, 0, 1], standard_name="latitude", var_name="node_lat"
        )
        cls.EDGE_LON = AuxCoord(
            [1, 1.5, 0.5], standard_name="longitude", var_name="edge_lon"
        )
        cls.EDGE_LAT = AuxCoord(
            [0, 0.5, 0.5], standard_name="latitude", var_name="edge_lat"
        )
        cls.FACE_LON = AuxCoord([0.5], standard_name="longitude", var_name="face_lon")
        cls.FACE_LAT = AuxCoord([0.5], standard_name="latitude", var_name="face_lat")

        cls.EDGE_NODE = components.Connectivity(
            [[0, 1], [1, 2], [2, 0]],
            cf_role="edge_node_connectivity",
            long_name="long_name",
            var_name="var_name",
            attributes={"test": 1},
        )
        cls.FACE_NODE = components.Connectivity(
            [[0, 1, 2]], cf_role="face_node_connectivity"
        )
        cls.FACE_EDGE = components.Connectivity(
            [[0, 1, 2]], cf_role="face_edge_connectivity"
        )
        # (Actually meaningless:)
        cls.FACE_FACE = components.Connectivity(
            [[0, 0, 0]], cf_role="face_face_connectivity"
        )
        # (Actually meaningless:)
        cls.EDGE_FACE = components.Connectivity(
            [[0, 0], [0, 0], [0, 0]], cf_role="edge_face_connectivity"
        )
        cls.BOUNDARY_NODE = components.Connectivity(
            [[0, 1], [1, 2], [2, 0]], cf_role="boundary_node_connectivity"
        )


class TestProperties1D(TestMeshCommon):
    # Tests that can reuse a single instance for greater efficiency.
    @staticmethod
    @pytest.fixture(scope="class", autouse=True)
    def _setup_class_1d(make_common_inputs):
        # MeshXY kwargs with topology_dimension=1 and all applicable
        # arguments populated - this tests correct property setting.
        cls = TestProperties1D
        cls.kwargs = {
            "topology_dimension": 1,
            "node_coords_and_axes": ((cls.NODE_LON, "x"), (cls.NODE_LAT, "y")),
            "connectivities": [cls.EDGE_NODE],
            "long_name": "my_topology_mesh",
            "var_name": "mesh",
            "attributes": {"notes": "this is a test"},
            "node_dimension": "NodeDim",
            "edge_dimension": "EdgeDim",
            "edge_coords_and_axes": ((cls.EDGE_LON, "x"), (cls.EDGE_LAT, "y")),
        }
        cls.mesh = components.MeshXY(**cls.kwargs)

    def test__metadata_manager(self):
        assert self.mesh._metadata_manager.cls.__name__ == MeshMetadata.__name__

    def test___getstate__(self):
        expected = (
            self.mesh._metadata_manager,
            self.mesh._coord_manager,
            self.mesh._connectivity_manager,
        )
        assert expected == self.mesh.__getstate__()

    def test___repr__(self):
        expected = "<MeshXY: 'my_topology_mesh'>"
        assert expected == repr(self.mesh)

    def test___str__(self):
        expected = [
            "MeshXY : 'my_topology_mesh'",
            "    topology_dimension: 1",
            "    node",
            "        node_dimension: 'NodeDim'",
            "        node coordinates",
            "            <AuxCoord: longitude / (unknown)  [...]  shape(3,)>",
            "            <AuxCoord: latitude / (unknown)  [...]  shape(3,)>",
            "    edge",
            "        edge_dimension: 'EdgeDim'",
            (
                "        edge_node_connectivity: "
                "<Connectivity: long_name / (unknown)  [...]  shape(3, 2)>"
            ),
            "        edge coordinates",
            "            <AuxCoord: longitude / (unknown)  [...]  shape(3,)>",
            "            <AuxCoord: latitude / (unknown)  [...]  shape(3,)>",
            "    long_name: 'my_topology_mesh'",
            "    var_name: 'mesh'",
            "    attributes:",
            "        notes  'this is a test'",
        ]
        assert expected == str(self.mesh).split("\n")

    def test___eq__(self):
        # The dimension names do not participate in equality.
        equivalent_kwargs = self.kwargs.copy()
        equivalent_kwargs["node_dimension"] = "something_else"
        equivalent = components.MeshXY(**equivalent_kwargs)
        assert equivalent == self.mesh

    def test_different(self):
        different_kwargs = self.kwargs.copy()
        different_kwargs["long_name"] = "new_name"
        different = components.MeshXY(**different_kwargs)
        assert different != self.mesh

        different_kwargs = self.kwargs.copy()
        ncaa = self.kwargs["node_coords_and_axes"]
        new_lat = ncaa[1][0].copy(points=ncaa[1][0].points + 1)
        new_ncaa = (ncaa[0], (new_lat, "y"))
        different_kwargs["node_coords_and_axes"] = new_ncaa
        different = components.MeshXY(**different_kwargs)
        assert different != self.mesh

        different_kwargs = self.kwargs.copy()
        conns = self.kwargs["connectivities"]
        new_conn = conns[0].copy(conns[0].indices + 1)
        different_kwargs["connectivities"] = new_conn
        different = components.MeshXY(**different_kwargs)
        assert different != self.mesh

    def test_all_connectivities(self):
        expected = components.Mesh1DConnectivities(self.EDGE_NODE)
        assert expected == self.mesh.all_connectivities

    def test_all_coords(self):
        expected = components.Mesh1DCoords(
            self.NODE_LON, self.NODE_LAT, self.EDGE_LON, self.EDGE_LAT
        )
        assert expected == self.mesh.all_coords

    def test_boundary_node(self):
        with pytest.raises(AttributeError):
            _ = self.mesh.boundary_node_connectivity

    def test_cf_role(self):
        assert "mesh_topology" == self.mesh.cf_role
        # Read only.
        with pytest.raises(AttributeError):
            self.mesh.cf_role = "foo"

    def test_connectivities(self, mocker):
        # General results. Method intended for inheritance.
        positive_kwargs = (
            {"item": self.EDGE_NODE},
            {"item": "long_name"},
            {"long_name": "long_name"},
            {"var_name": "var_name"},
            {"attributes": {"test": 1}},
            {"cf_role": "edge_node_connectivity"},
        )

        fake_connectivity = mocker.Mock(
            __class__=components.Connectivity, cf_role="fake"
        )
        negative_kwargs = (
            {"item": fake_connectivity},
            {"item": "foo"},
            {"standard_name": "air_temperature"},
            {"long_name": "foo"},
            {"var_name": "foo"},
            {"attributes": {"test": 2}},
            {"cf_role": "foo"},
        )

        func = self.mesh.connectivities
        for kwargs in positive_kwargs:
            assert [self.EDGE_NODE] == func(**kwargs)
        for kwargs in negative_kwargs:
            assert [] == func(**kwargs)

    def test_connectivities_elements(self, caplog):
        # topology_dimension-specific results. Method intended to be overridden.
        positive_kwargs = (
            {"contains_node": True},
            {"contains_edge": True},
            {"contains_node": True, "contains_edge": True},
        )
        negative_kwargs = (
            {"contains_node": False},
            {"contains_edge": False},
            {"contains_edge": True, "contains_node": False},
            {"contains_edge": False, "contains_node": False},
        )

        func = self.mesh.connectivities
        for kwargs in positive_kwargs:
            assert [self.EDGE_NODE] == func(**kwargs)
        for kwargs in negative_kwargs:
            assert [] == func(**kwargs)

        log_regex = r".*filter for non-existent.*"
        with _shared_utils.assert_logs(
            caplog, logger, level="DEBUG", msg_regex=log_regex
        ):
            assert [] == func(contains_face=True)

    def test_coord(self):
        # See MeshXY.coords tests for thorough coverage of cases.
        func = self.mesh.coord
        exception = CoordinateNotFoundError
        with pytest.raises(exception, match=".*but found 2"):
            func(location="node")
        with pytest.raises(exception, match=".*but found none"):
            func(axis="t")
        with pytest.raises(ValueError, match="Expected location.*got `foo`"):
            func(location="foo")

    def test_coords(self):
        # General results. Method intended for inheritance.
        positive_kwargs = (
            {"item": self.NODE_LON},
            {"item": "longitude"},
            {"standard_name": "longitude"},
            {"long_name": "long_name"},
            {"var_name": "node_lon"},
            {"attributes": {"test": 1}},
            {"location": "node"},
        )

        fake_coord = AuxCoord([0])
        negative_kwargs = (
            {"item": fake_coord},
            {"item": "foo"},
            {"standard_name": "air_temperature"},
            {"long_name": "foo"},
            {"var_name": "foo"},
            {"attributes": {"test": 2}},
            {"location": "edge"},
        )

        func = self.mesh.coords
        for kwargs in positive_kwargs:
            assert self.NODE_LON in func(**kwargs)
        for kwargs in negative_kwargs:
            assert self.NODE_LON not in func(**kwargs)

        with pytest.raises(ValueError, match="Expected location.*got.*foo"):
            self.mesh.coords(location="foo")

    def test_coords_elements(self, caplog):
        # topology_dimension-specific results. Method intended to be overridden.
        all_expected = {
            "node_x": self.NODE_LON,
            "node_y": self.NODE_LAT,
            "edge_x": self.EDGE_LON,
            "edge_y": self.EDGE_LAT,
        }

        kwargs_expected = (
            ({"axis": "x"}, ["node_x", "edge_x"]),
            ({"axis": "y"}, ["node_y", "edge_y"]),
            ({"location": "node"}, ["node_x", "node_y"]),
            ({"location": "edge"}, ["edge_x", "edge_y"]),
            ({"location": "face"}, ["face_x", "face_y"]),
        )

        func = self.mesh.coords
        for kwargs, expected in kwargs_expected:
            expected = [all_expected[k] for k in expected if k in all_expected]
            assert expected == func(**kwargs)

        log_regex = r".*filter non-existent.*"
        with _shared_utils.assert_logs(
            caplog, logger, level="DEBUG", msg_regex=log_regex
        ):
            assert [] == func(location="face")

    def test_edge_dimension(self):
        assert self.kwargs["edge_dimension"] == self.mesh.edge_dimension

    def test_edge_coords(self):
        expected = components.MeshEdgeCoords(self.EDGE_LON, self.EDGE_LAT)
        assert expected == self.mesh.edge_coords

    def test_edge_face(self):
        with pytest.raises(AttributeError):
            _ = self.mesh.edge_face_connectivity

    def test_edge_node(self):
        assert self.EDGE_NODE == self.mesh.edge_node_connectivity

    def test_face_coords(self):
        with pytest.raises(AttributeError):
            _ = self.mesh.face_coords

    def test_face_dimension(self):
        assert self.mesh.face_dimension is None

    def test_face_edge(self):
        with pytest.raises(AttributeError):
            _ = self.mesh.face_edge_connectivity

    def test_face_face(self):
        with pytest.raises(AttributeError):
            _ = self.mesh.face_face_connectivity

    def test_face_node(self):
        with pytest.raises(AttributeError):
            _ = self.mesh.face_node_connectivity

    def test_node_coords(self):
        expected = components.MeshNodeCoords(self.NODE_LON, self.NODE_LAT)
        assert expected == self.mesh.node_coords

    def test_node_dimension(self):
        assert self.kwargs["node_dimension"] == self.mesh.node_dimension

    def test_topology_dimension(self):
        assert self.kwargs["topology_dimension"] == self.mesh.topology_dimension
        # Read only.
        with pytest.raises(AttributeError):
            self.mesh.topology_dimension = "foo"


class TestProperties2D(TestProperties1D):
    # Additional/specialised tests for topology_dimension=2.
    @staticmethod
    @pytest.fixture(scope="class", autouse=True)
    def _setup_class_2d(_setup_class_1d):
        cls = TestProperties2D
        cls.kwargs["topology_dimension"] = 2
        cls.kwargs["connectivities"] = (
            cls.FACE_NODE,
            cls.EDGE_NODE,
            cls.FACE_EDGE,
            cls.FACE_FACE,
            cls.EDGE_FACE,
            cls.BOUNDARY_NODE,
        )
        cls.kwargs["face_dimension"] = "FaceDim"
        cls.kwargs["face_coords_and_axes"] = (
            (cls.FACE_LON, "x"),
            (cls.FACE_LAT, "y"),
        )
        cls.mesh = components.MeshXY(**cls.kwargs)

    def test___repr__(self):
        expected = "<MeshXY: 'my_topology_mesh'>"
        assert expected == repr(self.mesh)

    def test___str__(self):
        expected = [
            "MeshXY : 'my_topology_mesh'",
            "    topology_dimension: 2",
            "    node",
            "        node_dimension: 'NodeDim'",
            "        node coordinates",
            "            <AuxCoord: longitude / (unknown)  [...]  shape(3,)>",
            "            <AuxCoord: latitude / (unknown)  [...]  shape(3,)>",
            "    edge",
            "        edge_dimension: 'EdgeDim'",
            (
                "        edge_node_connectivity: "
                "<Connectivity: long_name / (unknown)  [...]  shape(3, 2)>"
            ),
            "        edge coordinates",
            "            <AuxCoord: longitude / (unknown)  [...]  shape(3,)>",
            "            <AuxCoord: latitude / (unknown)  [...]  shape(3,)>",
            "    face",
            "        face_dimension: 'FaceDim'",
            (
                "        face_node_connectivity: "
                "<Connectivity: unknown / (unknown)  [...]  shape(1, 3)>"
            ),
            "        face coordinates",
            "            <AuxCoord: longitude / (unknown)  [...]>",
            "            <AuxCoord: latitude / (unknown)  [...]>",
            "    optional connectivities",
            (
                "        face_face_connectivity: "
                "<Connectivity: unknown / (unknown)  [...]  shape(1, 3)>"
            ),
            (
                "        face_edge_connectivity: "
                "<Connectivity: unknown / (unknown)  [...]  shape(1, 3)>"
            ),
            (
                "        edge_face_connectivity: "
                "<Connectivity: unknown / (unknown)  [...]  shape(3, 2)>"
            ),
            "    long_name: 'my_topology_mesh'",
            "    var_name: 'mesh'",
            "    attributes:",
            "        notes  'this is a test'",
        ]
        assert expected == str(self.mesh).split("\n")

    # Test some different options of the str() operation here.
    def test___str__noedgecoords(self):
        mesh_kwargs = self.kwargs.copy()
        del mesh_kwargs["edge_coords_and_axes"]
        alt_mesh = components.MeshXY(**mesh_kwargs)
        expected = [
            "MeshXY : 'my_topology_mesh'",
            "    topology_dimension: 2",
            "    node",
            "        node_dimension: 'NodeDim'",
            "        node coordinates",
            "            <AuxCoord: longitude / (unknown)  [...]  shape(3,)>",
            "            <AuxCoord: latitude / (unknown)  [...]  shape(3,)>",
            "    edge",
            "        edge_dimension: 'EdgeDim'",
            (
                "        edge_node_connectivity: "
                "<Connectivity: long_name / (unknown)  [...]  shape(3, 2)>"
            ),
            "    face",
            "        face_dimension: 'FaceDim'",
            (
                "        face_node_connectivity: "
                "<Connectivity: unknown / (unknown)  [...]  shape(1, 3)>"
            ),
            "        face coordinates",
            "            <AuxCoord: longitude / (unknown)  [...]>",
            "            <AuxCoord: latitude / (unknown)  [...]>",
            "    optional connectivities",
            (
                "        face_face_connectivity: "
                "<Connectivity: unknown / (unknown)  [...]  shape(1, 3)>"
            ),
            (
                "        face_edge_connectivity: "
                "<Connectivity: unknown / (unknown)  [...]  shape(1, 3)>"
            ),
            (
                "        edge_face_connectivity: "
                "<Connectivity: unknown / (unknown)  [...]  shape(3, 2)>"
            ),
            "    long_name: 'my_topology_mesh'",
            "    var_name: 'mesh'",
            "    attributes:",
            "        notes  'this is a test'",
        ]
        assert expected == str(alt_mesh).split("\n")

    def test_all_connectivities(self):
        expected = components.Mesh2DConnectivities(
            self.FACE_NODE,
            self.EDGE_NODE,
            self.FACE_EDGE,
            self.FACE_FACE,
            self.EDGE_FACE,
            self.BOUNDARY_NODE,
        )
        assert expected == self.mesh.all_connectivities

    def test_all_coords(self):
        expected = components.Mesh2DCoords(
            self.NODE_LON,
            self.NODE_LAT,
            self.EDGE_LON,
            self.EDGE_LAT,
            self.FACE_LON,
            self.FACE_LAT,
        )
        assert expected == self.mesh.all_coords

    def test_boundary_node(self):
        assert self.BOUNDARY_NODE == self.mesh.boundary_node_connectivity

    def test_connectivity(self):
        # See MeshXY.connectivities tests for thorough coverage of cases.
        # Can only test MeshXY.connectivity for 2D since we need >1 connectivity.
        func = self.mesh.connectivity
        exception = ConnectivityNotFoundError
        with pytest.raises(exception, match=".*but found 3"):
            func(contains_node=True)
        with pytest.raises(exception, match=".*but found none"):
            func(contains_node=False, contains_edge=False, contains_face=False)

    def test_connectivities_elements(self):
        kwargs_expected = (
            (
                {"contains_node": True},
                [self.EDGE_NODE, self.FACE_NODE, self.BOUNDARY_NODE],
            ),
            (
                {"contains_edge": True},
                [self.EDGE_NODE, self.FACE_EDGE, self.EDGE_FACE],
            ),
            (
                {"contains_face": True},
                [
                    self.FACE_NODE,
                    self.FACE_EDGE,
                    self.FACE_FACE,
                    self.EDGE_FACE,
                ],
            ),
            (
                {"contains_node": False},
                [self.FACE_EDGE, self.EDGE_FACE, self.FACE_FACE],
            ),
            (
                {"contains_edge": False},
                [self.FACE_NODE, self.BOUNDARY_NODE, self.FACE_FACE],
            ),
            ({"contains_face": False}, [self.EDGE_NODE, self.BOUNDARY_NODE]),
            (
                {"contains_edge": True, "contains_face": True},
                [self.FACE_EDGE, self.EDGE_FACE],
            ),
            (
                {"contains_node": False, "contains_edge": False},
                [self.FACE_FACE],
            ),
            (
                {"contains_node": True, "contains_edge": False},
                [self.FACE_NODE, self.BOUNDARY_NODE],
            ),
            (
                {
                    "contains_node": False,
                    "contains_edge": False,
                    "contains_face": False,
                },
                [],
            ),
        )
        func = self.mesh.connectivities
        for kwargs, expected in kwargs_expected:
            result = func(**kwargs)
            assert len(expected) == len(result)
            for item in expected:
                assert item in result

    def test_coords_elements(self):
        all_expected = {
            "node_x": self.NODE_LON,
            "node_y": self.NODE_LAT,
            "edge_x": self.EDGE_LON,
            "edge_y": self.EDGE_LAT,
            "face_x": self.FACE_LON,
            "face_y": self.FACE_LAT,
        }

        kwargs_expected = (
            ({"axis": "x"}, ["node_x", "edge_x", "face_x"]),
            ({"axis": "y"}, ["node_y", "edge_y", "face_y"]),
            ({"location": "node"}, ["node_x", "node_y"]),
            ({"location": "edge"}, ["edge_x", "edge_y"]),
        )

        func = self.mesh.coords
        for kwargs, expected in kwargs_expected:
            expected = [all_expected[k] for k in expected if k in all_expected]
            assert expected == func(**kwargs)

    def test_edge_face(self):
        assert self.EDGE_FACE == self.mesh.edge_face_connectivity

    def test_face_coords(self):
        expected = components.MeshFaceCoords(self.FACE_LON, self.FACE_LAT)
        assert expected == self.mesh.face_coords

    def test_face_dimension(self):
        assert self.kwargs["face_dimension"] == self.mesh.face_dimension

    def test_face_edge(self):
        assert self.FACE_EDGE == self.mesh.face_edge_connectivity

    def test_face_face(self):
        assert self.FACE_FACE == self.mesh.face_face_connectivity

    def test_face_node(self):
        assert self.FACE_NODE == self.mesh.face_node_connectivity


class Test__str__various(TestMeshCommon):
    # Some extra testing for the str() operation : based on 1D meshes as simpler
    @pytest.fixture(autouse=True)
    def _setup(self, make_common_inputs):
        # All the tests here want modified meshes, so use standard set up to
        # create afresh for each test, allowing them to modify it.

        # MeshXY kwargs with topology_dimension=1 and all applicable
        # arguments populated - this tests correct property setting.
        self.kwargs = {
            "topology_dimension": 1,
            "node_coords_and_axes": (
                (self.NODE_LON, "x"),
                (self.NODE_LAT, "y"),
            ),
            "connectivities": [self.EDGE_NODE],
            "long_name": "my_topology_mesh",
            "var_name": "mesh",
            "attributes": {"notes": "this is a test"},
            "node_dimension": "NodeDim",
            "edge_dimension": "EdgeDim",
            "edge_coords_and_axes": (
                (self.EDGE_LON, "x"),
                (self.EDGE_LAT, "y"),
            ),
        }
        self.mesh = components.MeshXY(**self.kwargs)

    def test___repr__basic(self):
        expected = "<MeshXY: 'my_topology_mesh'>"
        assert expected == repr(self.mesh)

    def test___repr__varname(self):
        self.mesh.long_name = None
        expected = "<MeshXY: 'mesh'>"
        assert expected == repr(self.mesh)

    def test___repr__noname(self):
        self.mesh.long_name = None
        self.mesh.var_name = None
        expected = "<MeshXY object at 0x[0-9a-f]+>"
        assert re.search(expected, repr(self.mesh))

    def test___str__noattributes(self):
        self.mesh.attributes = None
        assert "attributes" not in str(self.mesh)

    def test___str__emptyattributes(self):
        self.mesh.attributes.clear()
        assert "attributes" not in str(self.mesh)

    def test__str__longstringattribute(self):
        self.mesh.attributes["long_string"] = (
            "long_x_10_long_x_20_long_x_30_long_x_40_"
            "long_x_50_long_x_60_long_x_70_long_x_80_"
        )
        result = str(self.mesh)
        # Note: initial single-quote, but no final one : this is correct !
        expected = (
            "'long_x_10_long_x_20_long_x_30_long_x_40_long_x_50_long_x_60_long_x_70..."
        )
        assert expected + ":END" in result + ":END"

    def test___str__units_stdname(self):
        # These are usually missing, but they *can* be present.
        mesh_kwargs = self.kwargs.copy()
        mesh_kwargs["standard_name"] = "height"  # Odd choice !
        mesh_kwargs["units"] = "m"
        alt_mesh = components.MeshXY(**mesh_kwargs)
        result = str(alt_mesh)
        # We expect these to appear at the end.
        expected = "\n".join(
            [
                "        edge coordinates",
                "            <AuxCoord: longitude / (unknown)  [...]  shape(3,)>",
                "            <AuxCoord: latitude / (unknown)  [...]  shape(3,)>",
                "    standard_name: 'height'",
                "    long_name: 'my_topology_mesh'",
                "    var_name: 'mesh'",
                "    units: Unit('m')",
                "    attributes:",
                "        notes  'this is a test'",
            ]
        )
        assert result.endswith(expected)


class TestOperations1D(TestMeshCommon):
    # Tests that cannot reuse an existing MeshXY instance, instead need a new
    # one each time.
    @pytest.fixture(autouse=True)
    def _setup_1d(self, make_common_inputs):
        self.mesh = components.MeshXY(
            topology_dimension=1,
            node_coords_and_axes=((self.NODE_LON, "x"), (self.NODE_LAT, "y")),
            connectivities=self.EDGE_NODE,
        )

    @staticmethod
    def new_connectivity(connectivity, new_len=False):
        """Provide a new connectivity recognisably different from the original."""
        # NOTE: assumes non-transposed connectivity (location_axis=0).
        if new_len:
            shape = (connectivity.shape[0] + 1, connectivity.shape[1])
        else:
            shape = connectivity.shape
        return connectivity.copy(np.zeros(shape, dtype=int))

    @staticmethod
    def new_coord(coord, new_shape=False):
        """Provide a new coordinate recognisably different from the original."""
        if new_shape:
            shape = tuple([i + 1 for i in coord.shape])
        else:
            shape = coord.shape
        return coord.copy(np.zeros(shape))

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

        assert false_metadata_manager == self.mesh._metadata_manager
        assert false_coord_manager == self.mesh._coord_manager
        assert false_connectivity_manager == self.mesh._connectivity_manager

    def test_add_connectivities(self):
        # Cannot test ADD - 1D - nothing extra to add beyond minimum.

        for new_len in (False, True):
            # REPLACE connectivities, first with one of the same length, then
            # with one of different length.
            edge_node = self.new_connectivity(self.EDGE_NODE, new_len)
            self.mesh.add_connectivities(edge_node)
            assert (
                components.Mesh1DConnectivities(edge_node)
                == self.mesh.all_connectivities
            )

    def test_add_connectivities_duplicates(self):
        edge_node_one = self.EDGE_NODE
        edge_node_two = self.new_connectivity(self.EDGE_NODE)
        self.mesh.add_connectivities(edge_node_one, edge_node_two)
        assert edge_node_two == self.mesh.edge_node_connectivity

    def test_add_connectivities_invalid(self, caplog):
        with pytest.raises(TypeError, match="Expected Connectivity.*"):
            self.mesh.add_connectivities("foo")

        face_node = self.FACE_NODE
        log_regex = r"Not adding connectivity.*"
        with _shared_utils.assert_logs(
            caplog, logger, level="DEBUG", msg_regex=log_regex
        ):
            self.mesh.add_connectivities(face_node)

    def test_add_coords(self):
        # ADD coords.
        edge_kwargs = {"edge_x": self.EDGE_LON, "edge_y": self.EDGE_LAT}
        self.mesh.add_coords(**edge_kwargs)
        assert components.MeshEdgeCoords(**edge_kwargs) == self.mesh.edge_coords

        for new_shape in (False, True):
            # REPLACE coords, first with ones of the same shape, then with ones
            # of different shape.
            node_kwargs = {
                "node_x": self.new_coord(self.NODE_LON, new_shape),
                "node_y": self.new_coord(self.NODE_LAT, new_shape),
            }
            edge_kwargs = {
                "edge_x": self.new_coord(self.EDGE_LON, new_shape),
                "edge_y": self.new_coord(self.EDGE_LAT, new_shape),
            }
            self.mesh.add_coords(**node_kwargs, **edge_kwargs)
            assert components.MeshNodeCoords(**node_kwargs) == self.mesh.node_coords
            assert components.MeshEdgeCoords(**edge_kwargs) == self.mesh.edge_coords

    def test_add_coords_face(self):
        with pytest.raises(TypeError):
            self.mesh.add_coords(face_x=self.FACE_LON, face_y=self.FACE_LAT)

    def test_add_coords_invalid(self):
        func = self.mesh.add_coords
        with pytest.raises(TypeError, match=".*requires to be an 'AuxCoord'.*"):
            func(node_x="foo")
        with pytest.raises(TypeError, match=".*requires a x-axis like.*"):
            func(node_x=self.NODE_LAT)
        climatological = AuxCoord(
            [0],
            bounds=[-1, 1],
            standard_name="longitude",
            climatological=True,
            units="Days since 1970",
        )
        with pytest.raises(TypeError, match=".*cannot be a climatological.*"):
            func(node_x=climatological)
        wrong_shape = self.NODE_LON.copy([0])
        with pytest.raises(ValueError, match=".*requires to have shape.*"):
            func(node_x=wrong_shape)

    def test_add_coords_single(self):
        # ADD coord.
        edge_x = self.EDGE_LON
        expected = components.MeshEdgeCoords(edge_x=edge_x, edge_y=None)
        self.mesh.add_coords(edge_x=edge_x)
        assert expected == self.mesh.edge_coords

        # REPLACE coords.
        node_x = self.new_coord(self.NODE_LON)
        edge_x = self.new_coord(self.EDGE_LON)
        expected_nodes = components.MeshNodeCoords(
            node_x=node_x, node_y=self.mesh.node_coords.node_y
        )
        expected_edges = components.MeshEdgeCoords(edge_x=edge_x, edge_y=None)
        self.mesh.add_coords(node_x=node_x, edge_x=edge_x)
        assert expected_nodes == self.mesh.node_coords
        assert expected_edges == self.mesh.edge_coords

        # Attempt to REPLACE coords with those of DIFFERENT SHAPE.
        node_x = self.new_coord(self.NODE_LON, new_shape=True)
        edge_x = self.new_coord(self.EDGE_LON, new_shape=True)
        node_kwarg = {"node_x": node_x}
        edge_kwarg = {"edge_x": edge_x}
        both_kwargs = dict(**node_kwarg, **edge_kwarg)
        for kwargs in (node_kwarg, edge_kwarg, both_kwargs):
            with pytest.raises(ValueError, match=".*requires to have shape.*"):
                self.mesh.add_coords(**kwargs)

    def test_add_coords_single_face(self):
        with pytest.raises(TypeError):
            self.mesh.add_coords(face_x=self.FACE_LON)

    def test_dimension_names(self, caplog):
        # Test defaults.
        default = components.Mesh1DNames("Mesh1d_node", "Mesh1d_edge")
        assert default == self.mesh.dimension_names()

        log_regex = r"Not setting face_dimension.*"
        with _shared_utils.assert_logs(
            caplog, logger, level="DEBUG", msg_regex=log_regex
        ):
            self.mesh.dimension_names("foo", "bar", "baz")
        assert components.Mesh1DNames("foo", "bar") == self.mesh.dimension_names()

        self.mesh.dimension_names_reset(True, True, True)
        assert default == self.mesh.dimension_names()

        # Single.
        self.mesh.dimension_names(edge="foo")
        assert "foo" == self.mesh.edge_dimension
        self.mesh.dimension_names_reset(edge=True)
        assert default == self.mesh.dimension_names()

    def test_edge_dimension_set(self):
        self.mesh.edge_dimension = "foo"
        assert "foo" == self.mesh.edge_dimension

    def test_face_dimension_set(self, caplog):
        log_regex = r"Not setting face_dimension.*"
        with _shared_utils.assert_logs(
            caplog, logger, level="DEBUG", msg_regex=log_regex
        ):
            self.mesh.face_dimension = "foo"
        assert self.mesh.face_dimension is None

    def test_node_dimension_set(self):
        self.mesh.node_dimension = "foo"
        assert "foo" == self.mesh.node_dimension

    def test_remove_connectivities(self, caplog, mocker):
        """Test that remove() mimics the connectivities() method correctly,
        and prevents removal of mandatory connectivities.

        """
        positive_kwargs = (
            {"item": self.EDGE_NODE},
            {"item": "long_name"},
            {"long_name": "long_name"},
            {"var_name": "var_name"},
            {"attributes": {"test": 1}},
            {"cf_role": "edge_node_connectivity"},
            {"contains_node": True},
            {"contains_edge": True},
            {"contains_edge": True, "contains_node": True},
        )

        fake_connectivity = mocker.Mock(
            __class__=components.Connectivity, cf_role="fake"
        )
        negative_kwargs = (
            {"item": fake_connectivity},
            {"item": "foo"},
            {"standard_name": "air_temperature"},
            {"long_name": "foo"},
            {"var_name": "foo"},
            {"attributes": {"test": 2}},
            {"cf_role": "foo"},
            {"contains_node": False},
            {"contains_edge": False},
            {"contains_edge": True, "contains_node": False},
            {"contains_edge": False, "contains_node": False},
        )

        log_regex = r"Ignoring request to remove.*"
        for kwargs in positive_kwargs:
            with _shared_utils.assert_logs(
                caplog, logger, level="DEBUG", msg_regex=log_regex
            ):
                self.mesh.remove_connectivities(**kwargs)
            assert self.EDGE_NODE == self.mesh.edge_node_connectivity
        for kwargs in negative_kwargs:
            caplog.clear()
            with caplog.at_level("DEBUG", logger=logger.name):
                logger.debug("foo", extra=dict(cls=None))
                self.mesh.remove_connectivities(**kwargs)
                assert len(caplog.records) == 1
            assert self.EDGE_NODE == self.mesh.edge_node_connectivity

    def test_remove_coords(self, caplog):
        # Test that remove() mimics the coords() method correctly,
        # and prevents removal of mandatory coords.
        positive_kwargs = (
            {"item": self.NODE_LON},
            {"item": "longitude"},
            {"standard_name": "longitude"},
            {"long_name": "long_name"},
            {"var_name": "node_lon"},
            {"attributes": {"test": 1}},
        )

        fake_coord = AuxCoord([0])
        negative_kwargs = (
            {"item": fake_coord},
            {"item": "foo"},
            {"standard_name": "air_temperature"},
            {"long_name": "foo"},
            {"var_name": "foo"},
            {"attributes": {"test": 2}},
        )

        log_regex = r"Ignoring request to remove.*"
        for kwargs in positive_kwargs:
            with _shared_utils.assert_logs(
                caplog, logger, level="DEBUG", msg_regex=log_regex
            ):
                self.mesh.remove_coords(**kwargs)
            assert self.NODE_LON == self.mesh.node_coords.node_x
        for kwargs in negative_kwargs:
            caplog.clear()
            with caplog.at_level("DEBUG", logger=logger.name):
                # Check that the only debug log is the one we inserted.
                logger.debug("foo", extra=dict(cls=None))
                self.mesh.remove_coords(**kwargs)
                assert len(caplog.records) == 1
            assert self.NODE_LON == self.mesh.node_coords.node_x

        # Test removal of optional connectivity.
        self.mesh.add_coords(edge_x=self.EDGE_LON)
        # Attempt to remove a non-existent coord.
        self.mesh.remove_coords(self.EDGE_LAT)
        # Confirm that EDGE_LON is still there.
        assert self.EDGE_LON == self.mesh.edge_coords.edge_x
        # Remove EDGE_LON and confirm success.
        self.mesh.remove_coords(self.EDGE_LON)
        assert None is self.mesh.edge_coords.edge_x

    def test_to_MeshCoord(self):
        location = "node"
        axis = "x"
        result = self.mesh.to_MeshCoord(location, axis)
        assert isinstance(result, components.MeshCoord)
        assert location == result.location
        assert axis == result.axis

    def test_to_MeshCoord_face(self):
        location = "face"
        axis = "x"
        with pytest.raises(CoordinateNotFoundError):
            self.mesh.to_MeshCoord(location, axis)

    def test_to_MeshCoords(self):
        location = "node"
        result = self.mesh.to_MeshCoords(location)
        assert len(self.mesh.AXES) == len(result)
        for ix, axis in enumerate(self.mesh.AXES):
            coord = result[ix]
            assert isinstance(coord, components.MeshCoord)
            assert location == coord.location
            assert axis == coord.axis

    def test_to_MeshCoords_face(self):
        location = "face"
        with pytest.raises(CoordinateNotFoundError):
            self.mesh.to_MeshCoords(location)


class TestOperations2D(TestOperations1D):
    # Additional/specialised tests for topology_dimension=2.
    @pytest.fixture(autouse=True)
    def _setup_2d(self, _setup_1d):
        self.mesh = components.MeshXY(
            topology_dimension=2,
            node_coords_and_axes=((self.NODE_LON, "x"), (self.NODE_LAT, "y")),
            connectivities=(self.FACE_NODE),
        )

    def test_add_connectivities(self):
        # ADD connectivities.
        kwargs = {
            "edge_node": self.EDGE_NODE,
            "face_edge": self.FACE_EDGE,
            "face_face": self.FACE_FACE,
            "edge_face": self.EDGE_FACE,
            "boundary_node": self.BOUNDARY_NODE,
        }
        expected = components.Mesh2DConnectivities(
            face_node=self.mesh.face_node_connectivity, **kwargs
        )
        self.mesh.add_connectivities(*kwargs.values())
        assert expected == self.mesh.all_connectivities

        # REPLACE connectivities.
        kwargs["face_node"] = self.FACE_NODE
        for new_len in (False, True):
            # First replace with ones of same length, then with ones of
            # different length.
            kwargs = {k: self.new_connectivity(v, new_len) for k, v in kwargs.items()}
            self.mesh.add_connectivities(*kwargs.values())
            assert (
                components.Mesh2DConnectivities(**kwargs)
                == self.mesh.all_connectivities
            )

    def test_add_connectivities_inconsistent(self):
        # ADD Connectivities.
        self.mesh.add_connectivities(self.EDGE_NODE)
        face_edge = self.new_connectivity(self.FACE_EDGE, new_len=True)
        edge_face = self.new_connectivity(self.EDGE_FACE, new_len=True)
        for args in ([face_edge], [edge_face], [face_edge, edge_face]):
            with pytest.raises(ValueError, match="inconsistent .* counts."):
                self.mesh.add_connectivities(*args)

        # REPLACE Connectivities
        self.mesh.add_connectivities(self.FACE_EDGE, self.EDGE_FACE)
        for args in ([face_edge], [edge_face], [face_edge, edge_face]):
            with pytest.raises(ValueError, match="inconsistent .* counts."):
                self.mesh.add_connectivities(*args)

    def test_add_connectivities_invalid(self, caplog, mocker):
        fake_cf_role = mocker.Mock(__class__=components.Connectivity, cf_role="foo")
        log_regex = r"Not adding connectivity.*"
        with _shared_utils.assert_logs(
            caplog, logger, level="DEBUG", msg_regex=log_regex
        ):
            self.mesh.add_connectivities(fake_cf_role)

    def test_add_coords_face(self):
        # ADD coords.
        kwargs = {"face_x": self.FACE_LON, "face_y": self.FACE_LAT}
        self.mesh.add_coords(**kwargs)
        assert components.MeshFaceCoords(**kwargs) == self.mesh.face_coords

        for new_shape in (False, True):
            # REPLACE coords, first with ones of the same shape, then with ones
            # of different shape.
            kwargs = {
                "face_x": self.new_coord(self.FACE_LON, new_shape),
                "face_y": self.new_coord(self.FACE_LAT, new_shape),
            }
            self.mesh.add_coords(**kwargs)
            assert components.MeshFaceCoords(**kwargs) == self.mesh.face_coords

    def test_add_coords_single_face(self):
        # ADD coord.
        face_x = self.FACE_LON
        expected = components.MeshFaceCoords(face_x=face_x, face_y=None)
        self.mesh.add_coords(face_x=face_x)
        assert expected == self.mesh.face_coords

        # REPLACE coord.
        face_x = self.new_coord(self.FACE_LON)
        expected = components.MeshFaceCoords(face_x=face_x, face_y=None)
        self.mesh.add_coords(face_x=face_x)
        assert expected == self.mesh.face_coords

        # Attempt to REPLACE coord with that of DIFFERENT SHAPE.
        face_x = self.new_coord(self.FACE_LON, new_shape=True)
        with pytest.raises(ValueError, match=".*requires to have shape.*"):
            self.mesh.add_coords(face_x=face_x)

    def test_dimension_names(self):
        # Test defaults.
        default = components.Mesh2DNames("Mesh2d_node", "Mesh2d_edge", "Mesh2d_face")
        assert default == self.mesh.dimension_names()

        self.mesh.dimension_names("foo", "bar", "baz")
        assert (
            components.Mesh2DNames("foo", "bar", "baz") == self.mesh.dimension_names()
        )

        self.mesh.dimension_names_reset(True, True, True)
        assert default == self.mesh.dimension_names()

        # Single.
        self.mesh.dimension_names(face="foo")
        assert "foo" == self.mesh.face_dimension
        self.mesh.dimension_names_reset(face=True)
        assert default == self.mesh.dimension_names()

    def test_face_dimension_set(self):
        self.mesh.face_dimension = "foo"
        assert "foo" == self.mesh.face_dimension

    def test_remove_connectivities(self):
        """Do what 1D test could not - test removal of optional connectivity."""
        # Add an optional connectivity.
        self.mesh.add_connectivities(self.FACE_FACE)
        # Attempt to remove a non-existent connectivity.
        self.mesh.remove_connectivities(self.EDGE_NODE)
        # Confirm that FACE_FACE is still there.
        assert self.FACE_FACE == self.mesh.face_face_connectivity
        # Remove FACE_FACE and confirm success.
        self.mesh.remove_connectivities(contains_face=True)
        assert None is self.mesh.face_face_connectivity

    def test_remove_coords(self, caplog):
        """Test the face argument."""
        super().test_remove_coords(caplog)
        self.mesh.add_coords(face_x=self.FACE_LON)
        assert self.FACE_LON == self.mesh.face_coords.face_x
        self.mesh.remove_coords(location="face")
        assert None is self.mesh.face_coords.face_x

    def test_to_MeshCoord_face(self):
        self.mesh.add_coords(face_x=self.FACE_LON)
        location = "face"
        axis = "x"
        result = self.mesh.to_MeshCoord(location, axis)
        assert isinstance(result, components.MeshCoord)
        assert location == result.location
        assert axis == result.axis

    def test_to_MeshCoords_face(self):
        self.mesh.add_coords(face_x=self.FACE_LON, face_y=self.FACE_LAT)
        location = "face"
        result = self.mesh.to_MeshCoords(location)
        assert len(self.mesh.AXES) == len(result)
        for ix, axis in enumerate(self.mesh.AXES):
            coord = result[ix]
            assert isinstance(coord, components.MeshCoord)
            assert location == coord.location
            assert axis == coord.axis


class TestInitValidation(TestMeshCommon):
    def test_invalid_topology(self):
        kwargs = {
            "topology_dimension": 0,
            "node_coords_and_axes": (
                (self.NODE_LON, "x"),
                (self.NODE_LAT, "y"),
            ),
            "connectivities": self.EDGE_NODE,
        }
        with pytest.raises(ValueError, match="Expected 'topology_dimension'.*"):
            components.MeshXY(**kwargs)

    def test_invalid_axes(self):
        kwargs = {
            "topology_dimension": 2,
            "connectivities": self.FACE_NODE,
        }
        with pytest.raises(ValueError, match="Invalid axis specified for node.*"):
            components.MeshXY(
                node_coords_and_axes=((self.NODE_LON, "foo"), (self.NODE_LAT, "y")),
                **kwargs,
            )
        kwargs["node_coords_and_axes"] = (
            (self.NODE_LON, "x"),
            (self.NODE_LAT, "y"),
        )
        with pytest.raises(ValueError, match="Invalid axis specified for edge.*"):
            components.MeshXY(
                edge_coords_and_axes=((self.EDGE_LON, "foo"),),
                **kwargs,
            )
        with pytest.raises(ValueError, match="Invalid axis specified for face.*"):
            components.MeshXY(
                face_coords_and_axes=((self.FACE_LON, "foo"),),
                **kwargs,
            )

    # Several arg safety checks in __init__ currently unreachable given earlier checks.

    def test_minimum_connectivities(self):
        # Further validations are tested in add_connectivity tests.
        kwargs = {
            "topology_dimension": 1,
            "node_coords_and_axes": (
                (self.NODE_LON, "x"),
                (self.NODE_LAT, "y"),
            ),
            "connectivities": (self.FACE_NODE,),
        }
        with pytest.raises(ValueError, match=".*requires a edge_node_connectivity.*"):
            components.MeshXY(**kwargs)

    def test_minimum_coords(self):
        # Further validations are tested in add_coord tests.
        kwargs = {
            "topology_dimension": 1,
            "node_coords_and_axes": ((self.NODE_LON, "x"), (None, "y")),
            "connectivities": (self.FACE_NODE,),
        }
        with pytest.raises(ValueError, match=".*is a required coordinate.*"):
            components.MeshXY(**kwargs)
