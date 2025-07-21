# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.mesh.MeshCoord`."""

from platform import python_version
import re
import unittest.mock as mock

import dask.array as da
import numpy as np
from packaging import version
import pytest

from iris._lazy_data import as_lazy_data, is_lazy_data
from iris.common.metadata import CoordMetadata
from iris.coords import AuxCoord, Coord
from iris.cube import Cube
from iris.mesh import Connectivity, MeshCoord, MeshXY
from iris.tests._shared_utils import (
    assert_array_all_close,
    assert_array_almost_equal,
    assert_masked_array_almost_equal,
)
import iris.tests.stock.mesh
from iris.tests.stock.mesh import sample_mesh, sample_meshcoord
from iris.warnings import IrisVagueMetadataWarning


@pytest.fixture(params=["face", "edge"])
def location_face_or_edge(request):
    # Fixture to parametrise over location = face/edge
    return request.param


class Test___init__:
    @pytest.fixture(autouse=True)
    def _setup(self):
        mesh = sample_mesh()
        self.mesh = mesh
        self.meshcoord = sample_meshcoord(mesh=mesh)

    def test_basic(self):
        meshcoord = self.meshcoord
        assert meshcoord.mesh == self.mesh
        assert meshcoord.location == "face"
        assert meshcoord.axis == "x"
        assert isinstance(meshcoord, MeshCoord)
        assert isinstance(meshcoord, Coord)

    def test_derived_properties(self):
        # Check the derived properties of the meshcoord against the correct
        # underlying mesh coordinate.
        for axis in MeshXY.AXES:
            meshcoord = sample_meshcoord(axis=axis)
            face_x_coord = meshcoord.mesh.coord(location="face", axis=axis)
            for key in face_x_coord.metadata._fields:
                meshval = getattr(meshcoord, key)
                # All relevant attributes are derived from the face coord.
                assert meshval == getattr(face_x_coord, key)

    def test_fail_bad_mesh(self):
        with pytest.raises(TypeError, match="must be a.*Mesh"):
            sample_meshcoord(mesh=mock.sentinel.odd)

    def test_valid_locations(self):
        for loc in MeshXY.ELEMENTS:
            meshcoord = sample_meshcoord(location=loc)
            assert meshcoord.location == loc

    def test_fail_bad_location(self):
        with pytest.raises(ValueError, match="not a valid MeshXY location"):
            sample_meshcoord(location="bad")

    def test_fail_bad_axis(self):
        with pytest.raises(ValueError, match="not a valid MeshXY axis"):
            sample_meshcoord(axis="q")


class Test__readonly_properties:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.meshcoord = sample_meshcoord()

    def test_fixed_metadata(self):
        # Check that you cannot set any of these on an existing MeshCoord.
        meshcoord = self.meshcoord
        if version.parse(python_version()) >= version.parse("3.11"):
            msg = "object has no setter"
        else:
            msg = "can't set attribute"
        for prop in ("mesh", "location", "axis"):
            with pytest.raises(AttributeError, match=msg):
                setattr(meshcoord, prop, mock.sentinel.odd)

    def test_set_coord_system(self):
        # The property exists, =None, can set to None, can not set otherwise.
        assert hasattr(self.meshcoord, "coord_system")
        assert self.meshcoord.coord_system is None
        self.meshcoord.coord_system = None
        with pytest.raises(ValueError, match=r"Cannot set.* MeshCoord\.$"):
            self.meshcoord.coord_system = 1

    def test_set_climatological(self):
        # The property exists, =False, can set to False, can not set otherwise.
        assert hasattr(self.meshcoord, "climatological")
        assert self.meshcoord.climatological is False
        self.meshcoord.climatological = False
        with pytest.raises(ValueError, match=r"Cannot set.* MeshCoord\.$"):
            self.meshcoord.climatological = True

    @pytest.mark.parametrize(
        "metadata_name",
        [
            "points",
            "bounds",
            "standard_name",
            "long_name",
            "var_name",
            "units",
            "attributes",
        ],
    )
    def test_set_other(self, metadata_name):
        with pytest.raises(
            ValueError, match=rf"Cannot set '{metadata_name}' on a MeshCoord\.$"
        ):
            self.meshcoord.__setattr__(metadata_name, "value")


class Test__points_and_bounds:
    # Basic method testing only, for 3 locations with simple array values.
    # See Test_MeshCoord__dataviews for more detailed checks.
    def test_node(self):
        meshcoord = sample_meshcoord(location="node")
        n_nodes = iris.tests.stock.mesh._TEST_N_NODES  # n-nodes default for sample mesh
        assert meshcoord.core_bounds() is None
        assert_array_all_close(meshcoord.points, 1100 + np.arange(n_nodes))

    def test_edge(self):
        meshcoord = sample_meshcoord(location="edge")
        points, bounds = meshcoord.core_points(), meshcoord.core_bounds()
        assert points.shape == meshcoord.shape
        assert bounds.shape == meshcoord.shape + (2,)
        assert_array_all_close(meshcoord.points, [2100, 2101, 2102, 2103, 2104])
        assert_array_all_close(
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
        meshcoord = sample_meshcoord(location="face")
        points, bounds = meshcoord.core_points(), meshcoord.core_bounds()
        assert points.shape == meshcoord.shape
        assert bounds.shape == meshcoord.shape + (4,)
        assert_array_all_close(meshcoord.points, [3100, 3101, 3102])
        assert_array_all_close(
            meshcoord.bounds,
            [
                (1100, 1101, 1102, 1103),
                (1104, 1105, 1106, 1107),
                (1108, 1109, 1110, 1111),
            ],
        )


class Test___eq__:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mesh = sample_mesh()

    def _create_common_mesh(self, **kwargs):
        return sample_meshcoord(mesh=self.mesh, **kwargs)

    def test_identical_mesh(self):
        meshcoord1 = self._create_common_mesh()
        meshcoord2 = self._create_common_mesh()
        assert meshcoord2 == meshcoord1

    def test_equal_mesh(self):
        mesh1 = sample_mesh()
        mesh2 = sample_mesh()
        meshcoord1 = sample_meshcoord(mesh=mesh1)
        meshcoord2 = sample_meshcoord(mesh=mesh2)
        assert meshcoord2 == meshcoord1

    def test_different_mesh(self):
        mesh1 = sample_mesh()
        mesh2 = sample_mesh()
        mesh2.long_name = "new_name"
        meshcoord1 = sample_meshcoord(mesh=mesh1)
        meshcoord2 = sample_meshcoord(mesh=mesh2)
        assert meshcoord2 != meshcoord1

    def test_different_location(self):
        meshcoord = self._create_common_mesh()
        meshcoord2 = self._create_common_mesh(location="node")
        assert meshcoord2 != meshcoord

    def test_different_axis(self):
        meshcoord = self._create_common_mesh()
        meshcoord2 = self._create_common_mesh(axis="y")
        assert meshcoord2 != meshcoord


class Test__copy:
    def test_basic(self):
        meshcoord = sample_meshcoord()
        meshcoord2 = meshcoord.copy()
        assert meshcoord2 is not meshcoord
        assert meshcoord2 == meshcoord
        # In this case, they should share *NOT* copy the Mesh object.
        assert meshcoord2.mesh is meshcoord.mesh

    def test_fail_copy_newpoints(self):
        meshcoord = sample_meshcoord()
        with pytest.raises(ValueError, match="Cannot change the content"):
            meshcoord.copy(points=meshcoord.points)

    def test_fail_copy_newbounds(self):
        meshcoord = sample_meshcoord()
        with pytest.raises(ValueError, match="Cannot change the content"):
            meshcoord.copy(bounds=meshcoord.bounds)


class Test__getitem__:
    def test_slice_wholeslice_1tuple(self):
        # The only slicing case that we support, to enable cube slicing.
        meshcoord = sample_meshcoord()
        meshcoord2 = meshcoord[:,]
        assert meshcoord2 is not meshcoord
        assert meshcoord2 == meshcoord
        # In this case, we should *NOT* copy the linked Mesh object.
        assert meshcoord2.mesh is meshcoord.mesh

    def test_slice_whole_slice_singlekey(self):
        # A slice(None) also fails, if not presented in a 1-tuple.
        meshcoord = sample_meshcoord()
        with pytest.raises(ValueError, match="Cannot index"):
            meshcoord[:]

    def test_fail_slice_part(self):
        meshcoord = sample_meshcoord()
        with pytest.raises(ValueError, match="Cannot index"):
            meshcoord[:1]


class Test__str_repr:
    @pytest.fixture(autouse=True)
    def _setup(self):
        mesh = sample_mesh()
        self.mesh = mesh
        # Give mesh itself a name: makes a difference between str and repr.
        self.mesh.rename("test_mesh")
        self.meshcoord = sample_meshcoord(mesh=mesh)

    def _expected_elements_regexp(
        self,
        standard_name="longitude",
        long_name=None,
        attributes=False,
        location="face",
        axis="x",
        var_name=None,
    ):
        # Printed name is standard or long -- we don't have a case with neither
        coord_name = standard_name or long_name
        # Construct regexp in 'sections'
        # NB each consumes up to first non-space in the next line
        regexp = f"MeshCoord :  {coord_name} / [^\n]+\n *"
        regexp += r"mesh: \<MeshXY: 'test_mesh'>\n *"
        regexp += f"location: '{location}'\n *"

        # Now some optional sections : whichever comes first will match
        # arbitrary content leading up to it.
        matched_upto = False

        def upto_first_expected(regexp, matched_any_upto):
            if not matched_any_upto:
                regexp += ".*"
                matched_any_upto = True
            return regexp, matched_any_upto

        if standard_name:
            regexp, matched_upto = upto_first_expected(regexp, matched_upto)
            regexp += f"standard_name: '{standard_name}'\n *"
        if long_name:
            regexp, matched_upto = upto_first_expected(regexp, matched_upto)
            regexp += f"long_name: '{long_name}'\n *"
        if var_name:
            regexp, matched_upto = upto_first_expected(regexp, matched_upto)
            regexp += f"var_name: '{var_name}'\n *"
        if attributes:
            # if we expected attributes, they should come next
            # TODO: change this when each attribute goes on a new line
            regexp, matched_upto = upto_first_expected(regexp, matched_upto)
            # match 'attributes:' followed by N*lines with larger indent
            regexp += "attributes:(\n        [^ \n]+ +[^ \n]+)+\n    "
        # After those items, expect 'axis' next
        # N.B. this FAILS if we had attributes when we didn't expect them
        regexp += f"axis: '{axis}'$"  # N.B. this is always the end

        # Compile regexp, also allowing matches across newlines
        regexp = re.compile(regexp, flags=re.DOTALL)
        return regexp

    def test_repr(self):
        # A simple check for the condensed form.
        result = repr(self.meshcoord)
        expected = (
            "<MeshCoord: longitude / (unknown)  "
            "mesh(test_mesh) location(face)  [...]+bounds  shape(3,)>"
        )
        assert expected == result

    def test_repr_lazy(self):
        # Displays lazy content (and does not realise!).
        coord_on_mesh = self.meshcoord.mesh.coord(
            standard_name=self.meshcoord.standard_name,
            axis=self.meshcoord.axis,
            location=self.meshcoord.location,
        )
        points = self.meshcoord.points
        coord_on_mesh.points = as_lazy_data(points)
        # Node coords are used to calculate the meshcoord bounds
        for nc in self.meshcoord.mesh.node_coords:
            nc.points = as_lazy_data(nc.points)

        assert self.meshcoord.has_lazy_points() is True
        assert self.meshcoord.has_lazy_bounds() is True

        result = repr(self.meshcoord)
        assert self.meshcoord.has_lazy_points() is True
        assert self.meshcoord.has_lazy_bounds() is True

        expected = (
            "<MeshCoord: longitude / (unknown)  "
            "mesh(test_mesh) location(face)  <lazy>+bounds  shape(3,)>"
        )
        assert expected == result

    def test_repr__nameless_mesh(self):
        # Check what it does when the Mesh doesn't have a name.
        self.mesh.long_name = None
        assert self.mesh.name() == "unknown"
        result = repr(self.meshcoord)
        re_expected = (
            r".MeshCoord: longitude / \(unknown\)  "
            r"mesh\(.MeshXY object at 0x[^>]+.\) location\(face\) "
        )
        assert re.match(re_expected, result)

    def test__str__(self):
        # Basic output contains mesh, location, standard_name, long_name,
        # attributes, mesh, location and axis
        result = str(self.meshcoord)
        re_expected = self._expected_elements_regexp()
        re.match(re_expected, result)

    def test__str__lazy(self):
        # Displays lazy content (and does not realise!).
        coord_on_mesh = self.meshcoord.mesh.coord(
            standard_name=self.meshcoord.standard_name,
            axis=self.meshcoord.axis,
            location=self.meshcoord.location,
        )

        # Node coords are used to calculate the meshcoord bounds
        for nc in self.meshcoord.mesh.node_coords:
            nc.points = as_lazy_data(nc.points)

        coord_on_mesh.points = as_lazy_data(self.meshcoord.points)
        result = str(self.meshcoord)
        assert self.meshcoord.has_lazy_points() is True
        assert self.meshcoord.has_lazy_bounds() is True

        assert "points: <lazy>" in result
        assert "bounds: <lazy>" in result
        re_expected = self._expected_elements_regexp()
        re.match(re_expected, result)

    def test_alternative_location_and_axis(self):
        meshcoord = sample_meshcoord(mesh=self.mesh, location="edge", axis="y")
        result = str(meshcoord)
        re_expected = self._expected_elements_regexp(
            standard_name="latitude",
            long_name=None,
            location="edge",
            axis="y",
            attributes=None,
        )
        re.match(re_expected, result)
        # Basic output contains standard_name, long_name, attributes

    def test_str_no_long_name(self):
        mesh = self.mesh
        # Remove the long_name of the node coord in the mesh.
        node_coord = mesh.coord(location="node", axis="x")
        node_coord.long_name = None
        # Make a new meshcoord, based on the modified mesh.
        meshcoord = sample_meshcoord(mesh=self.mesh)
        result = str(meshcoord)
        re_expected = self._expected_elements_regexp(long_name=False)
        re.match(re_expected, result)

    def test_str_no_attributes(self):
        mesh = self.mesh
        # No attributes on the node coord in the mesh.
        node_coord = mesh.coord(location="node", axis="x")
        node_coord.attributes = None
        # Make a new meshcoord, based on the modified mesh.
        meshcoord = sample_meshcoord(mesh=self.mesh)
        result = str(meshcoord)
        re_expected = self._expected_elements_regexp(attributes=False)
        re.match(re_expected, result)

    def test_str_empty_attributes(self):
        mesh = self.mesh
        # Empty attributes dict on the node coord in the mesh.
        node_coord = mesh.coord(location="node", axis="x")
        node_coord.attributes.clear()
        # Make a new meshcoord, based on the modified mesh.
        meshcoord = sample_meshcoord(mesh=self.mesh)
        result = str(meshcoord)
        re_expected = self._expected_elements_regexp(attributes=False)
        re.match(re_expected, result)


class Test_cube_containment:
    # Check that we can put a MeshCoord into a cube, and have it behave just
    # like a regular AuxCoord.
    @pytest.fixture(autouse=True)
    def _setup(self):
        meshcoord = sample_meshcoord()
        data_shape = (2,) + meshcoord.shape
        cube = Cube(np.zeros(data_shape))
        cube.add_aux_coord(meshcoord, 1)
        self.meshcoord = meshcoord
        self.cube = cube

    def test_added_to_cube(self):
        meshcoord = self.meshcoord
        cube = self.cube
        assert meshcoord in cube.coords()

    def test_cube_dims(self):
        meshcoord = self.meshcoord
        cube = self.cube
        assert meshcoord.cube_dims(cube) == (1,)
        assert cube.coord_dims(meshcoord) == (1,)

    def test_find_by_name(self):
        meshcoord = self.meshcoord

        # changes to meshcoords have to be done via the attached mesh
        coord_on_mesh = meshcoord.mesh.coord(
            standard_name=meshcoord.standard_name, location=meshcoord.location
        )
        coord_on_mesh.long_name = "odd_case"

        cube = self.cube
        assert cube.coord(standard_name="longitude") is meshcoord
        assert cube.coord(long_name="odd_case") is meshcoord

    def test_find_by_axis(self):
        meshcoord = self.meshcoord
        mesh = self.meshcoord.mesh
        cube = self.cube
        assert cube.coord(axis="x") is meshcoord
        assert cube.coords(axis="y") == []

        # NOTE: the meshcoord.axis takes precedence over the older
        # "guessed axis" approach.  So the standard_name does not control it.
        for c in mesh.coords(axis="x"):
            c.standard_name = "grid_longitude"
        for c in mesh.coords(axis="y"):
            c.standard_name = "grid_latitude"

        assert cube.coord(axis="x") is meshcoord
        assert cube.coords(axis="y") == []

    def test_cube_copy(self):
        # Check that we can copy a cube, and get a MeshCoord == the original.
        # Note: currently must have the *same* mesh, as for MeshCoord.copy().
        meshcoord = self.meshcoord
        cube = self.cube
        cube2 = cube.copy()
        meshco2 = cube2.coord(meshcoord)
        assert meshco2 is not meshcoord
        assert meshco2 == meshcoord

    def test_cube_nonmesh_slice(self):
        # Check that we can slice a cube on a non-mesh dimension, and get a
        # meshcoord == original.
        # Note: currently this must have the *same* mesh, as for .copy().
        meshcoord = self.meshcoord
        cube = self.cube
        cube2 = cube[:1]  # Make a reduced copy, slicing the non-mesh dim
        meshco2 = cube2.coord(meshcoord)
        assert meshco2 is not meshcoord
        assert meshco2 == meshcoord

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
        assert co_matches == []

        # The resulting coord is an AuxCoord instead of a MeshCoord, but the
        # values match.
        co2 = cube2.coord(meshcoord.name())
        assert isinstance(co2, MeshCoord) is False
        assert isinstance(co2, AuxCoord)
        assert_array_all_close(co2.points, meshcoord.points[:1])
        assert_array_all_close(co2.bounds, meshcoord.bounds[:1])


class Test_auxcoord_conversion:
    def test_basic(self):
        meshcoord = sample_meshcoord()
        auxcoord = AuxCoord.from_coord(meshcoord)
        for propname, auxval in auxcoord.metadata._asdict().items():
            meshval = getattr(meshcoord, propname)
            assert auxval == meshval
        # Also check array content.
        assert_array_all_close(auxcoord.points, meshcoord.points)
        assert_array_all_close(auxcoord.bounds, meshcoord.bounds)


class Test_MeshCoord__dataviews:
    """Fuller testing of points and bounds calculations and behaviour.
    Including connectivity missing-points (non-square faces).

    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        self._make_test_meshcoord()

    def _make_test_meshcoord(
        self,
        lazy_sources=False,
        location="face",
        inds_start_index=0,
        inds_location_axis=0,
        facenodes_changes=None,
    ):
        # Construct a miniature face-nodes mesh for testing.
        # NOTE: we will make our connectivity arrays with standard
        # start_index=0 and location_axis=0 :  We only adjust that (if required) when
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
        if facenodes_changes:
            facenodes_changes = facenodes_changes.copy()
            facenodes_changes.pop("n_extra_bad_points")
            for indices, value in facenodes_changes.items():
                face_nodes_array[indices] = value

        # Construct a miniature edge-nodes mesh for testing.
        edge_nodes_array = np.array([[0, 2], [1, 3], [1, 4], [3, 7]])
        # Connectivity uses *masked* for missing points.
        edge_nodes_array = np.ma.masked_less(edge_nodes_array, 0)

        n_faces = face_nodes_array.shape[0]
        n_edges = edge_nodes_array.shape[0]
        n_nodes = int(face_nodes_array.max() + 1)
        self.NODECOORDS_BASENUM = 1100.0
        self.EDGECOORDS_BASENUM = 1200.0
        self.FACECOORDS_BASENUM = 1300.0
        node_xs = self.NODECOORDS_BASENUM + np.arange(n_nodes)
        edge_xs = self.EDGECOORDS_BASENUM + np.arange(n_edges)
        face_xs = self.FACECOORDS_BASENUM + np.arange(n_faces)

        # Record all these for reuse in tests
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
        # N.B. the MeshXY requires 'Y's as well.
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
                if inds_location_axis == 1
                else face_nodes_array
            ),
            cf_role="face_node_connectivity",
            long_name="face_nodes",
            start_index=inds_start_index,
            location_axis=inds_location_axis,
        )

        edge_node_conn = Connectivity(
            inds_start_index
            + (
                edge_nodes_array.transpose()
                if inds_location_axis == 1
                else edge_nodes_array
            ),
            cf_role="edge_node_connectivity",
            long_name="edge_nodes",
            start_index=inds_start_index,
            location_axis=inds_location_axis,
        )

        self.mesh = MeshXY(
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
        assert_array_all_close(meshcoord.points, self.face_xs)

    def _check_expected_bounds_values(self, facenodes_changes=None):
        mesh_coord = self.meshcoord
        # The bounds are selected node_x-s, ==> node_number + coords-offset
        result = mesh_coord.bounds
        # N.B. result should be masked where the masked indices are.
        expected = self.NODECOORDS_BASENUM + self.face_nodes_array
        if facenodes_changes:
            # ALSO include any "bad" values in that calculation.
            bad_values = (self.face_nodes_array < 0) | (
                self.face_nodes_array >= self.n_nodes
            )
            expected[bad_values] = np.ma.masked
        # Check there are *some* masked points.
        n_missing_expected = 1
        if facenodes_changes:
            n_missing_expected += facenodes_changes["n_extra_bad_points"]
        assert np.count_nonzero(expected.mask) == n_missing_expected
        # Check results match, *including* location of masked points.
        assert_masked_array_almost_equal(result, expected)

    def test_points_values(self):
        """Basic points content check, on real data."""
        meshcoord = self.meshcoord
        assert meshcoord.has_lazy_points() is False
        assert meshcoord.has_lazy_bounds() is False
        self._check_expected_points_values()

    def test_bounds_values(self):
        """Basic bounds contents check."""
        meshcoord = self.meshcoord
        assert meshcoord.has_lazy_points() is False
        assert meshcoord.has_lazy_bounds() is False
        self._check_expected_bounds_values()

    def test_lazy_points_values(self):
        """Check lazy points calculation on lazy inputs."""
        # Remake the test data with lazy source coords.
        meshcoord = self._make_test_meshcoord(lazy_sources=True)
        assert meshcoord.has_lazy_points() is True
        assert meshcoord.has_lazy_bounds() is True
        # Check values, as previous.
        self._check_expected_points_values()

    def test_lazy_bounds_values(self):
        meshcoord = self._make_test_meshcoord(lazy_sources=True)
        assert meshcoord.has_lazy_points() is True
        assert meshcoord.has_lazy_bounds() is True
        # Check values, as previous.
        self._check_expected_bounds_values()

    def test_edge_points(self):
        meshcoord = self._make_test_meshcoord(location="edge")
        result = meshcoord.points
        assert_array_all_close(result, self.edge_xs)

    def test_edge_bounds(self):
        meshcoord = self._make_test_meshcoord(location="edge")
        result = meshcoord.bounds
        # The bounds are selected node_x-s :  all == node_number + 100.0
        expected = self.NODECOORDS_BASENUM + self.edge_nodes_array
        # NB simpler than faces : no possibility of missing points
        assert_array_almost_equal(result, expected)

    def test_bounds_connectivity__location_axis_1(self):
        # Test with a transposed indices array.
        self._make_test_meshcoord(inds_location_axis=1)
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
                mesh.coord(location="node", axis="x"),
                mesh.coord(location="face", axis="x"),
                mesh.face_node_connectivity,
            )

        # Check all the source coords are lazy.
        for coord in fetch_sources_from_mesh():
            # Note: not all are actual Coords, so can't use 'has_lazy_points'.
            assert is_lazy_data(coord._core_values()) is True

        # Calculate both points + bounds of the meshcoord
        assert meshcoord.has_lazy_points() is True
        assert meshcoord.has_lazy_bounds() is True
        mc_points = meshcoord.points
        mc_bounds = meshcoord.bounds
        assert meshcoord.has_lazy_points() is True
        assert meshcoord.has_lazy_bounds() is True
        assert is_lazy_data(mc_points) is False
        assert is_lazy_data(mc_bounds) is False

        # Check all the source coords are still lazy.
        for coord in fetch_sources_from_mesh():
            # Note: not all are actual Coords, so can't use 'has_lazy_points'.
            assert is_lazy_data(coord._core_values()) is True

    def _check_bounds_bad_index_values(self, lazy):
        facenodes_modify = {
            # nothing wrong with this one
            (2, 1): 1,
            # extra missing point, normal "missing" indicator
            (3, 3): np.ma.masked,
            # bad index > n_nodes
            (4, 2): 100,
            # NOTE: **can't** set an index < 0, as it is rejected by the
            # Connectivity validity check.
            # Indicate how many "extra" missing results this should cause.
            "n_extra_bad_points": 2,
        }
        self._make_test_meshcoord(facenodes_changes=facenodes_modify, lazy_sources=lazy)
        self._check_expected_bounds_values()

    def test_bounds_badvalues__real(self):
        self._check_bounds_bad_index_values(lazy=False)

    def test_bounds_badvalues__lazy(self):
        self._check_bounds_bad_index_values(lazy=True)


class Test__metadata:
    def setup_mesh(self, location, axis):
        # Create a standard test mesh + attach it to the test instance.
        mesh = sample_mesh()

        # Modify the metadata of specific coordinates used in this test.
        node_coord = mesh.coord(axis=axis, location="node")
        location_coord = mesh.coord(axis=axis, location=location)
        for i_place, coord in enumerate((node_coord, location_coord)):
            coord.standard_name = "longitude" if axis == "x" else "latitude"
            coord.units = "degrees"
            coord.long_name = f"long_name_{i_place}"
            coord.var_name = f"var_name_{i_place}"
            coord.attributes = {"att": i_place}

        # attach all the relevant testcase context to the test instance.
        self.mesh = mesh
        self.location = location
        self.axis = axis
        self.location_coord = location_coord
        self.node_coord = node_coord

    def coord_metadata_matches(self, test_coord, ref_coord):
        # Check that two coords match, in all the basic Coord identity/phenomenon
        # metadata fields -- so it works even between coords of different subclasses.
        for key in CoordMetadata._fields:
            assert getattr(test_coord, key) == getattr(ref_coord, key)

    @pytest.fixture(params=["x", "y"])
    def axis_x_or_y(self, request):
        # Fixture to parametrise over axis = X/Y
        return request.param

    def test_node_meshcoord(self, axis_x_or_y):
        # MeshCoord metadata matches that of the relevant node coord.
        self.setup_mesh(location="node", axis=axis_x_or_y)
        meshcoord = self.mesh.to_MeshCoord(location=self.location, axis=self.axis)
        self.coord_metadata_matches(meshcoord, self.node_coord)

    def test_faceedge_basic(self, location_face_or_edge, axis_x_or_y):
        # MeshCoord metadata matches that of the face/edge ("points") coord.
        self.setup_mesh(location_face_or_edge, axis_x_or_y)
        meshcoord = self.mesh.to_MeshCoord(location=self.location, axis=self.axis)
        self.coord_metadata_matches(meshcoord, self.location_coord)

    @pytest.mark.parametrize("fieldname", ["long_name", "var_name", "attributes"])
    def test_faceedge_dontcare_fields(
        self, location_face_or_edge, axis_x_or_y, fieldname
    ):
        # Check that it's ok for the face/edge and node coords to have different
        # long-name, var-name or attributes.
        self.setup_mesh(location_face_or_edge, axis_x_or_y)
        if fieldname == "attributes":
            different_value = {"myattrib": "different attributes"}
        else:
            # others are just arbitrary strings.
            different_value = "different"
        setattr(self.location_coord, fieldname, different_value)
        # Mostly.. just check this does not cause an error, as it would do if we
        # modified "standard_name" or "units" (see other tests) ...
        meshcoord = self.mesh.to_MeshCoord(location=self.location, axis=self.axis)
        # ... but also, check that the result matches the expected face/edge coord.
        self.coord_metadata_matches(meshcoord, self.location_coord)

    def test_faceedge_fail_mismatched_stdnames(
        self, location_face_or_edge, axis_x_or_y
    ):
        # Different "standard_name" for node and face/edge causes an error.
        self.setup_mesh(location_face_or_edge, axis_x_or_y)
        node_name = f"projection_{axis_x_or_y}_coordinate"
        self.node_coord.standard_name = node_name
        location_name = "longitude" if axis_x_or_y == "x" else "latitude"
        msg = (
            "Node coordinate .*"
            f"disagrees with the {location_face_or_edge} coordinate .*, "
            'in having a "standard_name" value of '
            f"'{node_name}' instead of '{location_name}'"
        )
        with pytest.raises(ValueError, match=msg):
            self.mesh.to_MeshCoord(location=location_face_or_edge, axis=axis_x_or_y)

    def test_faceedge_fail_missing_stdnames(self, location_face_or_edge, axis_x_or_y):
        # "standard_name" compared with None also causes an error.
        self.setup_mesh(location_face_or_edge, axis_x_or_y)
        self.node_coord.standard_name = None
        # N.B. in the absence of a standard-name, we **must** provide an extra ".axis"
        # property, or the coordinate cannot be correctly identified in the MeshXY.
        # This is a bit of a kludge, but works with current code.
        self.node_coord.axis = axis_x_or_y

        location_name = "longitude" if axis_x_or_y == "x" else "latitude"
        msg = (
            "Node coordinate .*"
            f"disagrees with the {location_face_or_edge} coordinate .*, "
            'in having a "standard_name" value of '
            f"None instead of '{location_name}'"
        )
        with pytest.raises(ValueError, match=msg):
            self.mesh.to_MeshCoord(location=location_face_or_edge, axis=axis_x_or_y)

    def test_faceedge_fail_mismatched_units(self, location_face_or_edge, axis_x_or_y):
        # Different "units" for node and face/edge causes an error.
        self.setup_mesh(location_face_or_edge, axis_x_or_y)
        self.node_coord.units = "hPa"
        msg = (
            "Node coordinate .*"
            f"disagrees with the {location_face_or_edge} coordinate .*, "
            'in having a "units" value of '
            "'hPa' instead of 'degrees'"
        )
        with pytest.raises(ValueError, match=msg):
            self.mesh.to_MeshCoord(location=location_face_or_edge, axis=axis_x_or_y)

    def test_faceedge_missing_units(self, location_face_or_edge, axis_x_or_y):
        # Units compared with a None ("unknown") is not an error.
        self.setup_mesh(location_face_or_edge, axis_x_or_y)
        self.node_coord.units = None
        # This is OK
        meshcoord = self.mesh.to_MeshCoord(location=self.location, axis=self.axis)
        # ... but also, check that the result matches the expected face/edge coord.
        self.coord_metadata_matches(meshcoord, self.location_coord)

    def test_updates_from_mesh(self):
        self.setup_mesh(location="node", axis="x")
        coord_on_mesh = self.mesh.coord(location=self.location, axis=self.axis)
        meshcoord = self.mesh.to_MeshCoord(location=self.location, axis=self.axis)

        coord_on_mesh.units = "radians"
        meshcoord.standard_name


class Test_collapsed:
    """Very simple operation that in theory is fully tested elsewhere
    (Test_auxcoord_conversion, and existing tests of AuxCoord.collapsed()),
    but there is still need to check that the operation is valid for any
    expected MeshCoord variety.
    """

    @pytest.fixture(params=[False, True], ids=["real", "lazy"])
    def lazy(self, request):
        return request.param

    @pytest.fixture(params=[4, 5], ids=["quads", "pentagons"])
    def nodes_per_face(self, request):
        return request.param

    @pytest.fixture(params=[False, True], ids=["conn_no_masks", "conn_has_masks"])
    def masked_connecteds(self, request):
        return request.param

    @staticmethod
    @pytest.fixture
    def mesh_coord(location_face_or_edge, lazy, nodes_per_face, masked_connecteds):
        mesh = sample_mesh(
            lazy_values=lazy,
            nodes_per_face=nodes_per_face,
            masked_connecteds=masked_connecteds,
        )
        coord = sample_meshcoord(
            mesh=mesh,
            location=location_face_or_edge,
        )
        return coord

    @staticmethod
    @pytest.fixture
    def mesh_coord_basic():
        return sample_meshcoord()

    def test_works(self, mesh_coord):
        """Just check that the operation succeeds.

        The points/bounds produced by collapsing a MeshCoord are suspect
        (hence a warning is raised), so we will not assert for 'correct'
        values.
        """
        collapsed = mesh_coord.collapsed()
        assert collapsed.points.shape == (1,)
        assert collapsed.bounds.shape == (1, 2)

    def test_warns(self, mesh_coord_basic):
        """Confirm that the correct warning has been raised.

        Also confirm that the original AuxCoord warning has NOT been raised -
        successfully suppressed.
        """
        with pytest.warns(IrisVagueMetadataWarning) as record:
            _ = mesh_coord_basic.collapsed()

        # Len 1 means that no other warnings were raised.
        assert len(record) == 1
        message = record[0].message.args[0]
        assert message.startswith("Collapsing a mesh coordinate")

    def test_aux_collapsed_called(self, mesh_coord_basic, mocker):
        mocked = mocker.patch.object(AuxCoord, "collapsed")
        _ = mesh_coord_basic.collapsed()
        mocked.assert_called_once()


class Test__updates_from_mesh:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.meshcoord = sample_meshcoord()
        self.mesh = self.meshcoord.mesh
        self.timestamp_at_creation = self.meshcoord.mesh._last_modified
        self.coord_on_mesh = self.mesh.coord(
            location=self.meshcoord.location, axis=self.meshcoord.axis
        )

    def _set_all_coords(self, name, value):
        for c in self.mesh.coords():
            c.__setattr__(name, value)

    def test__last_modified(self):
        # Ensure that mesh._last_modified is updated when you update the mesh, and
        # and meshcoord._last_modified is updated to match that.

        assert self.meshcoord.mesh._last_modified == self.timestamp_at_creation

        self.coord_on_mesh.points = np.zeros(3)

        assert self.meshcoord.mesh._last_modified == self.meshcoord._last_modified
        assert self.meshcoord.mesh._last_modified != self.timestamp_at_creation

    def test_points(self, mocker):
        zeroes = np.zeros(3)
        mocked = mocker.patch.object(MeshCoord, "_load_points_and_bounds")
        mocked.return_value = (
            zeroes,
            np.zeros((3, 3)),
        )
        assert self.meshcoord.points.all() != zeroes.all()
        self.coord_on_mesh.points = zeroes
        assert self.meshcoord.points.all() == zeroes.all()
        mocked.assert_called_once()

    def test_bounds(self, mocker):
        zero_bounds = np.zeros(3)
        zero_points = np.zeros(15)
        mocked = mocker.patch.object(MeshCoord, "_load_points_and_bounds")
        mocked.return_value = (
            zero_bounds,
            np.zeros((3, 3)),
        )
        assert self.meshcoord.bounds.all() != zero_bounds.all()
        # Node coords are used to calculate the meshcoord bounds
        for nc in self.meshcoord.mesh.node_coords:
            nc.points = zero_points
        assert self.meshcoord.bounds.all() == zero_bounds.all()
        mocked.assert_called_once()

    @pytest.mark.parametrize(
        "metadata_name, value",
        [
            ("long_name", "foo"),
            ("var_name", "foo"),
            ("attributes", {"foo": 1}),
        ],
    )
    def test_basic_metadata(self, metadata_name, value, mocker):
        mocked = mocker.patch.object(MeshCoord, "_load_points_and_bounds")

        self.coord_on_mesh.__setattr__(metadata_name, value)
        assert self.meshcoord.__getattribute__(
            metadata_name
        ) == self.coord_on_mesh.__getattribute__(metadata_name)

        # Ensure updating metadata doesn't prompt the MeshCoord
        # to update points and bounds.
        mocked.assert_not_called()

    def test_units(self, mocker):
        mocked = mocker.patch.object(MeshCoord, "_load_points_and_bounds")
        for c in self.mesh.coords():
            c.units = "radians"
        assert self.meshcoord.standard_name == self.coord_on_mesh.standard_name
        # Ensure updating metadata doesn't prompt the MeshCoord
        # to update points and bounds.
        mocked.assert_not_called()

    def test_standard_name(self, mocker):
        mocked = mocker.patch.object(MeshCoord, "_load_points_and_bounds")
        for c in self.mesh.coords(axis="x"):
            c.standard_name = "grid_longitude"
        for c in self.mesh.coords(axis="y"):
            c.standard_name = "grid_latitude"
        assert self.meshcoord.standard_name == self.coord_on_mesh.standard_name
        # Ensure updating metadata doesn't prompt the MeshCoord
        # to update points and bounds.
        mocked.assert_not_called()

    def test_updates(self, mocker):
        mocked = mocker.patch.object(MeshCoord, "_load_points_and_bounds")
        mocked.return_value = (
            np.zeros(
                3,
            ),
            np.zeros((3, 3)),
        )
        self.coord_on_mesh.points = np.zeros(3)
        # Only the first time you access an attribute should fetch
        # update the points and bounds.

        self.meshcoord.standard_name
        self.meshcoord.standard_name
        mocked.assert_called_once()

        # Ensure it updates more than once if the mesh has been updated
        # a second time.
        self.coord_on_mesh.points = np.ones(3)
        self.meshcoord.standard_name
        assert mocked.call_count == 2

    def test_update_from_mesh(self, mocker):
        mocked = mocker.patch.object(MeshCoord, "_load_points_and_bounds")
        mocked.return_value = (
            np.zeros(
                3,
            ),
            np.zeros((3, 3)),
        )
        self.coord_on_mesh.points = np.zeros(3)
        self.meshcoord.update_from_mesh()
        mocked.assert_called_once()

        # Ensure it forces an update, even if the mesh hasn't been updated.
        self.meshcoord.update_from_mesh()
        assert mocked.call_count == 2
