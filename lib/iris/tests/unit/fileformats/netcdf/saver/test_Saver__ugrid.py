# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.fileformats.netcdf.Saver` class.

WHEN MODIFYING THIS MODULE, CHECK IF ANY CORRESPONDING CHANGES ARE NEEDED IN
:mod:`iris.tests.unit.fileformats.netcdf.test_Saver__lazy.`

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from pathlib import Path
import shutil
import tempfile

import numpy as np

from iris import save
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList
from iris.experimental.ugrid.mesh import Connectivity, Mesh
from iris.experimental.ugrid.save import save_mesh
from iris.fileformats.netcdf import _thread_safe_nc
from iris.tests.stock import realistic_4d

XY_LOCS = ("x", "y")
XY_NAMES = ("longitude", "latitude")


def build_mesh(
    n_nodes=2,
    n_faces=0,
    n_edges=0,
    nodecoord_xyargs=None,
    edgecoord_xyargs=None,
    facecoord_xyargs=None,
    conn_role_kwargs=None,  # mapping {connectivity-role: connectivity-kwargs}
    mesh_kwargs=None,
):
    """Make a test mesh.

    Mesh has faces edges, face-coords and edge-coords, numbers of which can be
    controlled.

    Parameters
    ----------
    n_nodes, n_faces, n_edges : int
        Basic dimensions of mesh components.  Zero means no such location.
    nodecoord_xyargs, edgecoord_xyargs, facecoord_xyargs : pair of dict
        Pairs (x,y) of settings kwargs, applied after initial creation the
        relevant location coordinates.
    conn_role_kwargs : dict of str
        Mapping from cf_role name to settings kwargs for connectivities,
        applied after initially creating them.
    mesh_kwargs : dict
        Dictionary of key settings to apply to the Mesh, after creating it.

    """

    def applyargs(coord, kwargs):
        if kwargs:
            for key, val in kwargs.items():
                # kwargs is a dict
                setattr(coord, key, val)

    def apply_xyargs(coords, xyargs):
        if xyargs:
            for coord, kwargs in zip(coords, xyargs):
                # coords and xyargs both iterables : implicitly=(x,y)
                applyargs(coord, kwargs)

    node_coords = [
        AuxCoord(np.arange(n_nodes), standard_name=name) for name in XY_NAMES
    ]
    apply_xyargs(node_coords, nodecoord_xyargs)

    connectivities = {}
    edge_coords = []
    face_coords = []
    topology_dimension = 0
    if n_edges:
        topology_dimension = 1
        connectivities["edge_node_connectivity"] = Connectivity(
            np.zeros((n_edges, 2), np.int32), cf_role="edge_node_connectivity"
        )
        edge_coords = [
            AuxCoord(np.arange(n_edges), standard_name=name) for name in XY_NAMES
        ]
        apply_xyargs(edge_coords, edgecoord_xyargs)

    if n_faces:
        topology_dimension = 2
        connectivities["face_node_connectivity"] = Connectivity(
            np.zeros((n_faces, 4), np.int32), cf_role="face_node_connectivity"
        )
        face_coords = [
            AuxCoord(np.arange(n_faces), standard_name=name) for name in XY_NAMES
        ]
        apply_xyargs(face_coords, facecoord_xyargs)

    mesh_dims = {"node": n_nodes, "edge": n_edges, "face": n_faces}

    if conn_role_kwargs:
        for role, kwargs in conn_role_kwargs.items():
            if role in connectivities:
                conn = connectivities[role]
            else:
                loc_from, loc_to, _ = role.split("_")
                dims = [mesh_dims[loc] for loc in (loc_from, loc_to)]
                conn = Connectivity(np.zeros(dims, dtype=np.int32), cf_role=role)
                connectivities[role] = conn
            applyargs(conn, kwargs)

    mesh = Mesh(
        topology_dimension=topology_dimension,
        node_coords_and_axes=zip(node_coords, XY_LOCS),
        edge_coords_and_axes=zip(edge_coords, XY_LOCS),
        face_coords_and_axes=zip(face_coords, XY_LOCS),
        connectivities=connectivities.values(),
    )
    applyargs(mesh, mesh_kwargs)

    return mesh


def make_mesh(basic=True, **kwargs):
    """Create a test mesh, with some built-in 'standard' settings.

    Parameters
    ----------
    basic : bool
        If true (default), create with 'standard' set of test properties.
    **kwargs : dict
        Additional kwargs, passed through to 'build_mesh'.
        Items here override the 'standard' settings.

    """
    if basic:
        # Use some helpful non-minimal settings as our 'basic' mesh.
        use_kwargs = dict(
            n_nodes=5,
            n_faces=2,
            nodecoord_xyargs=tuple(dict(var_name=f"node_{loc}") for loc in XY_LOCS),
            facecoord_xyargs=tuple(dict(var_name=f"face_{loc}") for loc in XY_LOCS),
            mesh_kwargs=dict(
                var_name="Mesh2d",
                node_dimension="Mesh2d_nodes",
                face_dimension="Mesh2d_faces",
            ),
        )
        use_kwargs.update(kwargs)
    else:
        use_kwargs = kwargs

    mesh = build_mesh(**use_kwargs)
    return mesh


def mesh_location_size(mesh, location):
    """Get the length of a location-dimension from a mesh."""
    if location == "node":
        # Use a node coordinate (which always exists).
        node_coord = mesh.node_coords[0]
        result = node_coord.shape[0]
    else:
        # Use a <loc>_node_connectivity, if any.
        conn_name = f"{location}_node_connectivity"
        conn = getattr(mesh, conn_name, None)
        if conn is None:
            result = 0
        else:
            result = conn.shape[conn.location_axis]
    return result


# A simple "standard" test mesh for multiple uses, which we can use for cubes
# that *share* a mesh (since we don't support mesh equality).
# However, we defer creating this until needed, as it can cause an import loop.
_DEFAULT_MESH = None


def default_mesh():
    """Return the unique default mesh, creating it if needed."""
    global _DEFAULT_MESH
    if _DEFAULT_MESH is None:
        _DEFAULT_MESH = make_mesh()
    return _DEFAULT_MESH


def make_cube(mesh=None, location="face", **kwargs):
    """Create a test cube, based on a given mesh + location.

    Parameters
    ----------
    mesh : :class:`iris.experimental.ugrid.mesh.Mesh` or None, optional
        If None, use 'default_mesh()'
    location : str, optional, default="face"
        Which mesh element to map the cube to.
    **kwargs : dict, optional
        Additional property settings to apply to the cube (after creation).

    """
    if mesh is None:
        mesh = default_mesh()
    dim_length = mesh_location_size(mesh, location)
    cube = Cube(np.zeros(dim_length, np.float32))
    for meshco in mesh.to_MeshCoords(location):
        cube.add_aux_coord(meshco, (0,))
    for key, val in kwargs.items():
        setattr(cube, key, val)
    return cube


def add_height_dim(cube):
    """Add an extra initial 'height' dimension onto a cube."""
    cube = cube.copy()  # Avoid trashing the input cube.
    cube.add_aux_coord(AuxCoord([0.0], standard_name="height", units="m"))
    # Make three copies with different heights
    cubes = [cube.copy() for _ in range(3)]
    for i_cube, cube in enumerate(cubes):
        cube.coord("height").points = [i_cube]
    # Merge to create an additional 'height' dimension.
    cube = CubeList(cubes).merge_cube()
    return cube


# Special key-string for storing the dimensions of a variable
_VAR_DIMS = "<variable dimensions>"


def scan_dataset(filepath):
    """Snapshot a netcdf dataset (the key metadata).

    Returns
    -------
    dimsdict : dict
        A map of dimension-name: length.
    varsdict : dict
        A map of each variable's properties, {var_name: propsdict}
        Each propsdict is {attribute-name: value} over the var's ncattrs().
        Each propsdict ALSO contains a [_VAR_DIMS] entry listing the
        variable's dims.

    """
    ds = _thread_safe_nc.DatasetWrapper(filepath)
    # dims dict is {name: len}
    dimsdict = {name: dim.size for name, dim in ds.dimensions.items()}
    # vars dict is {name: {attr:val}}
    varsdict = {}
    for name, var in ds.variables.items():
        varsdict[name] = {prop: getattr(var, prop) for prop in var.ncattrs()}
        varsdict[name][_VAR_DIMS] = list(var.dimensions)
    ds.close()
    return dimsdict, varsdict


def vars_w_props(varsdict, **kwargs):
    """Subset a vars dict, {name:props}, returning only those where each
    <attribute>=<value>, defined by the given keywords.
    Except that '<key>="*"' means that '<key>' merely _exists_, with any value.

    """

    def check_attrs_match(attrs):
        result = True
        for key, val in kwargs.items():
            result = key in attrs
            if result:
                # val='*'' for a simple existence check
                result = (val == "*") or attrs[key] == val
            if not result:
                break
        return result

    varsdict = {
        name: attrs for name, attrs in varsdict.items() if check_attrs_match(attrs)
    }
    return varsdict


def vars_w_dims(varsdict, dim_names):
    """Subset a vars dict, returning those which map all the specified dims."""
    varsdict = {
        name: propsdict
        for name, propsdict in varsdict.items()
        if all(dim in propsdict[_VAR_DIMS] for dim in dim_names)
    }
    return varsdict


def vars_meshnames(vars):
    """Return the names of all the mesh variables (found by cf_role)."""
    return list(vars_w_props(vars, cf_role="mesh_topology").keys())


def vars_meshdim(vars, location, mesh_name=None):
    """Extract a dim-name for a given element location.

    Parameters
    ----------
    vars : varsdict
        file varsdict, as returned from 'snapshot_dataset'.
    location : string
        a mesh location : 'node' / 'edge' / 'face'
    mesh_name : str or None, optional, default=None
        If given, identifies the mesh var.
        Otherwise, find a unique mesh var (i.e. there must be exactly 1).

    Returns
    -------
    dim_name : str
        The dim-name of the mesh dim for the given location.

    Notes
    -----
    TODO: relies on the element having coordinates, which in future will not
        always be the case. This can be fixed

    """
    if mesh_name is None:
        # Find "the" meshvar -- assuming there is just one.
        (mesh_name,) = vars_meshnames(vars)
    mesh_props = vars[mesh_name]
    loc_coords = mesh_props[f"{location}_coordinates"].split(" ")
    (single_location_dim,) = vars[loc_coords[0]][_VAR_DIMS]
    return single_location_dim


class TestSaveUgrid__cube(tests.IrisTest):
    """Test for saving cubes which have meshes."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def check_save_cubes(self, cube_or_cubes):
        """Write cubes to a new file in the common temporary directory.

        Use a name unique to this testcase, to avoid any clashes.

        """
        # use 'result_path' to name the file after the test function
        tempfile_path = self.result_path(ext=".nc")
        # Create a file of that name, but discard the result path and put it
        # in the common temporary directory.
        tempfile_path = self.temp_dir / Path(tempfile_path).name

        # Save data to the file.
        save(cube_or_cubes, tempfile_path)

        return tempfile_path

    def test_basic_mesh(self):
        # Save a small mesh example and check aspects of the resulting file.
        cube = make_cube()  # A simple face-mapped data example.

        # Save and snapshot the result
        tempfile_path = self.check_save_cubes(cube)
        dims, vars = scan_dataset(tempfile_path)

        # There is exactly 1 mesh var.
        (mesh_name,) = vars_meshnames(vars)

        # There is exactly 1 mesh-linked (data)var
        data_vars = vars_w_props(vars, mesh="*")
        ((a_name, a_props),) = data_vars.items()
        mesh_props = vars[mesh_name]

        # The mesh var links to the mesh, with location 'faces'
        self.assertEqual(a_name, "unknown")
        self.assertEqual(a_props["mesh"], mesh_name)
        self.assertEqual(a_props["location"], "face")

        # There are 2 face coords == those listed in the mesh
        face_coords = mesh_props["face_coordinates"].split(" ")
        self.assertEqual(len(face_coords), 2)

        # The face coords should both map that single dim.
        face_dim = vars_meshdim(vars, "face")
        self.assertTrue(all(vars[co][_VAR_DIMS] == [face_dim] for co in face_coords))

        # The face coordinates should be referenced by the data variable.
        for coord in face_coords:
            self.assertIn(coord, a_props["coordinates"])

        # The dims of the datavar also == [<faces-dim>]
        self.assertEqual(a_props[_VAR_DIMS], [face_dim])

        # There are 2 node coordinates == those listed in the mesh.
        node_coords = mesh_props["node_coordinates"].split(" ")
        self.assertEqual(len(node_coords), 2)
        # These are the *only* ones using the 'nodes' dimension.
        node_dim = vars_meshdim(vars, "node")
        self.assertEqual(
            sorted(node_coords), sorted(vars_w_dims(vars, [node_dim]).keys())
        )

        # There are no edges.
        self.assertNotIn("edge_node_connectivity", mesh_props)
        self.assertEqual(len(vars_w_props(vars, cf_role="edge_node_connectivity")), 0)

        # The dims are precisely (nodes, faces, nodes-per-face), in that order.
        self.assertEqual(
            list(dims.keys()),
            ["Mesh2d_nodes", "Mesh2d_faces", "Mesh2d_face_N_nodes"],
        )

        # The variables are exactly (mesh, 2*node-coords, 2*face-coords,
        # face-nodes, data) -- in that order
        self.assertEqual(
            list(vars.keys()),
            [
                "Mesh2d",
                "node_x",
                "node_y",
                "face_x",
                "face_y",
                "mesh2d_faces",
                "unknown",
            ],
        )

        # For completeness, also check against a full CDL snapshot
        self.assertCDL(tempfile_path)

    def test_multi_cubes_common_mesh(self):
        cube1 = make_cube(var_name="a")
        cube2 = make_cube(var_name="b")

        # Save and snapshot the result
        tempfile_path = self.check_save_cubes([cube1, cube2])
        dims, vars = scan_dataset(tempfile_path)

        # there is exactly 1 mesh in the file
        (mesh_name,) = vars_meshnames(vars)

        # both the main variables reference the same mesh, and 'face' location
        v_a, v_b = vars["a"], vars["b"]
        self.assertEqual(v_a["mesh"], mesh_name)
        self.assertEqual(v_a["location"], "face")
        self.assertEqual(v_a["coordinates"], "face_x face_y")
        self.assertEqual(v_b["mesh"], mesh_name)
        self.assertEqual(v_b["location"], "face")
        self.assertEqual(v_b["coordinates"], "face_x face_y")

    def test_multi_cubes_different_locations(self):
        cube1 = make_cube(var_name="a", location="face")
        cube2 = make_cube(var_name="b", location="node")

        # Save and snapshot the result
        tempfile_path = self.check_save_cubes([cube1, cube2])
        dims, vars = scan_dataset(tempfile_path)

        # there is exactly 1 mesh in the file
        (mesh_name,) = vars_meshnames(vars)

        # the main variables reference the same mesh at different locations
        v_a, v_b = vars["a"], vars["b"]
        self.assertEqual(v_a["mesh"], mesh_name)
        self.assertEqual(v_a["location"], "face")
        self.assertEqual(v_a["coordinates"], "face_x face_y")
        self.assertEqual(v_b["mesh"], mesh_name)
        self.assertEqual(v_b["location"], "node")
        self.assertEqual(v_b["coordinates"], "node_x node_y")

        # the main variables map the face and node dimensions
        face_dim = vars_meshdim(vars, "face")
        node_dim = vars_meshdim(vars, "node")
        self.assertEqual(v_a[_VAR_DIMS], [face_dim])
        self.assertEqual(v_b[_VAR_DIMS], [node_dim])

    def test_multi_cubes_equal_meshes(self):
        # Make 2 identical meshes
        # NOTE: *can't* name these explicitly, as it stops them being identical.
        mesh1 = make_mesh()
        mesh2 = make_mesh()
        cube1 = make_cube(var_name="a", mesh=mesh1)
        cube2 = make_cube(var_name="b", mesh=mesh2)

        # Save and snapshot the result
        tempfile_path = self.check_save_cubes([cube1, cube2])
        dims, vars = scan_dataset(tempfile_path)

        # there is exactly 1 mesh in the file
        mesh_names = vars_meshnames(vars)
        self.assertEqual(sorted(mesh_names), ["Mesh2d"])

        # same dimensions
        self.assertEqual(vars_meshdim(vars, "node", mesh_name="Mesh2d"), "Mesh2d_nodes")
        self.assertEqual(vars_meshdim(vars, "face", mesh_name="Mesh2d"), "Mesh2d_faces")

        # there are exactly two data-variables with a 'mesh' property
        mesh_datavars = vars_w_props(vars, mesh="*")
        self.assertEqual(["a", "b"], list(mesh_datavars))

        # the data variables reference the same mesh
        a_props, b_props = vars["a"], vars["b"]
        for props in a_props, b_props:
            self.assertEqual(props["mesh"], "Mesh2d")
            self.assertEqual(props["location"], "face")
            self.assertEqual(props["coordinates"], "face_x face_y")

        # the data variables map the appropriate node dimension
        self.assertEqual(a_props[_VAR_DIMS], ["Mesh2d_faces"])
        self.assertEqual(b_props[_VAR_DIMS], ["Mesh2d_faces"])

    def test_multi_cubes_different_mesh(self):
        # Check that we can correctly distinguish 2 different meshes.
        cube1 = make_cube(var_name="a")
        cube2 = make_cube(var_name="b", mesh=make_mesh(n_faces=4))

        # Save and snapshot the result
        tempfile_path = self.check_save_cubes([cube1, cube2])
        dims, vars = scan_dataset(tempfile_path)

        # there are 2 meshes in the file
        mesh_names = vars_meshnames(vars)
        self.assertEqual(len(mesh_names), 2)

        # there are two (data)variables with a 'mesh' property
        mesh_datavars = vars_w_props(vars, mesh="*")
        self.assertEqual(2, len(mesh_datavars))
        self.assertEqual(["a", "b"], sorted(mesh_datavars.keys()))

        def get_props_attrs(props: dict):
            return props["mesh"], props["location"], props["coordinates"]

        # the main variables reference the correct meshes, and 'face' location
        a_props, b_props = vars["a"], vars["b"]
        mesh_a, loc_a, coords_a = get_props_attrs(a_props)
        mesh_b, loc_b, coords_b = get_props_attrs(b_props)
        self.assertNotEqual(mesh_a, mesh_b)
        self.assertNotEqual(coords_a, coords_b)
        self.assertEqual(loc_a, "face")
        self.assertEqual(loc_b, "face")

    def test_nonmesh_dim(self):
        # Check where the data variable has a 'normal' dim and a mesh dim.
        cube = make_cube()
        cube = add_height_dim(cube)

        # Save and snapshot the result
        tempfile_path = self.check_save_cubes(cube)
        dims, vars = scan_dataset(tempfile_path)

        # have just 1 mesh, including a face and node coordinates.
        (mesh_name,) = vars_meshnames(vars)
        # Check we have faces, and identify the faces dim
        face_dim = vars_meshdim(vars, "face", mesh_name)
        # Also just check we *have* a recognisable node-coordinate
        vars_meshdim(vars, "node", mesh_name)

        # have just 1 data-variable
        ((data_name, data_props),) = vars_w_props(vars, mesh="*").items()

        # data maps to the height + mesh dims
        self.assertEqual(data_props[_VAR_DIMS], ["height", face_dim])
        self.assertEqual(data_props["mesh"], mesh_name)
        self.assertEqual(data_props["location"], "face")

    @tests.skip_data
    def test_nonmesh_hybrid_dim(self):
        # Check a case with a hybrid non-mesh dimension
        cube = realistic_4d()
        # Strip off the time and longitude dims, to make it simpler.
        cube = cube[0, ..., 0]
        # Remove all the unwanted coords (also loses the coord-system)
        lose_coords = (
            "time",
            "forecast_period",
            "grid_longitude",
            "grid_latitude",
        )
        for coord in lose_coords:
            cube.remove_coord(coord)

        # Add a mesh on the remaining (now anonymous) horizontal dimension.
        i_horizontal_dim = len(cube.shape) - 1
        n_places = cube.shape[i_horizontal_dim]
        mesh = make_mesh(
            n_faces=n_places,
            n_nodes=30,  # arbitrary + unrealistic, but doesn't actually matter
        )
        # Attach the mesh by adding MeshCoords
        for coord in mesh.to_MeshCoords("face"):
            cube.add_aux_coord(coord, (i_horizontal_dim,))

        # Save and snapshot the result
        tempfile_path = self.check_save_cubes(cube)
        dims, vars = scan_dataset(tempfile_path)

        # have just 1 mesh, including face and node coordinates.
        (mesh_name,) = vars_meshnames(vars)
        face_dim = vars_meshdim(vars, "face", mesh_name)
        _ = vars_meshdim(vars, "node", mesh_name)

        # have hybrid vertical dimension, with all the usual term variables.
        self.assertIn("model_level_number", dims)
        vert_vars = list(vars_w_dims(vars, ["model_level_number"]).keys())
        # The list of file variables mapping the vertical dimension:
        # = the data-var, plus all the height terms
        self.assertEqual(
            vert_vars,
            [
                "air_potential_temperature",
                "model_level_number",
                "level_height",
                "level_height_bnds",
                "sigma",
                "sigma_bnds",
            ],
        )

        # have just 1 data-variable, which maps to hybrid-height and mesh dims
        ((data_name, data_props),) = vars_w_props(vars, mesh="*").items()
        self.assertEqual(data_props[_VAR_DIMS], ["model_level_number", face_dim])
        self.assertEqual(data_props["mesh"], mesh_name)
        self.assertEqual(data_props["location"], "face")

    def test_alternate_cube_dim_order(self):
        # A cube transposed from the 'usual' order
        # Should work much the same as the "basic" case.
        cube_1 = make_cube(var_name="a")
        cube_1 = add_height_dim(cube_1)

        cube_2 = cube_1.copy()
        cube_2.var_name = "b"
        cube_2.transpose()

        # Save and snapshot the result
        tempfile_path = self.check_save_cubes([cube_1, cube_2])
        dims, vars = scan_dataset(tempfile_path)

        # There is only 1 mesh
        (mesh_name,) = vars_meshnames(vars)

        # both variables reference the same mesh
        v_a, v_b = vars["a"], vars["b"]
        self.assertEqual(v_a["mesh"], mesh_name)
        self.assertEqual(v_a["location"], "face")
        self.assertEqual(v_b["mesh"], mesh_name)
        self.assertEqual(v_b["location"], "face")

        # Check the var dimensions
        self.assertEqual(v_a[_VAR_DIMS], ["height", "Mesh2d_faces"])
        self.assertEqual(v_b[_VAR_DIMS], ["Mesh2d_faces", "height"])

    def test_mixed_aux_coords(self):
        """``coordinates`` attribute should include mesh location coords and 'normal' coords."""
        cube = make_cube()
        mesh_dim = cube.mesh_dim()
        mesh_len = cube.shape[mesh_dim]
        coord = AuxCoord(np.arange(mesh_len), var_name="face_index")
        cube.add_aux_coord(coord, mesh_dim)

        # Save and snapshot the result
        tempfile_path = self.check_save_cubes(cube)
        dims, vars = scan_dataset(tempfile_path)

        # There is exactly 1 mesh-linked (data)var
        data_vars = vars_w_props(vars, mesh="*")
        ((_, a_props),) = data_vars.items()

        expected_coords = [c for c in cube.mesh.face_coords]
        expected_coords.append(coord)
        expected_coord_names = [c.var_name for c in expected_coords]
        expected_coord_attr = " ".join(sorted(expected_coord_names))
        self.assertEqual(a_props["coordinates"], expected_coord_attr)


class TestSaveUgrid__mesh(tests.IrisTest):
    """Tests for saving meshes to a file."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def check_save_mesh(self, mesh):
        """Write a mesh to a new file in the common temporary directory.

        Use a name unique to this testcase, to avoid any clashes.

        """
        # use 'result_path' to name the file after the test function
        tempfile_path = self.result_path(ext=".nc")
        # Create a file of that name, but discard the result path and put it
        # in the common temporary directory.
        tempfile_path = self.temp_dir / Path(tempfile_path).name

        # Save data to the file.
        save_mesh(mesh, tempfile_path)

        return tempfile_path

    def test_connectivity_dim_order(self):
        """Test a mesh with some connectivities in the 'other' order.

        This should also create a property with the dimension name.

        """
        # Make a mesh with both faces *and* some edges
        mesh = make_mesh(n_edges=7)
        # Get the face-node and edge-node connectivities
        face_nodes_conn = mesh.face_node_connectivity
        edge_nodes_conn = mesh.edge_node_connectivity
        # Transpose them : N.B. this sets location_axis=1, as it should be.
        nodesfirst_faces_conn = face_nodes_conn.transpose()
        nodesfirst_edges_conn = edge_nodes_conn.transpose()
        # Make a new mesh with both face and edge connectivities 'transposed'.
        mesh2 = Mesh(
            topology_dimension=mesh.topology_dimension,
            node_coords_and_axes=zip(mesh.node_coords, XY_LOCS),
            face_coords_and_axes=zip(mesh.face_coords, XY_LOCS),
            connectivities=[nodesfirst_faces_conn, nodesfirst_edges_conn],
        )

        # Save and snapshot the result
        tempfile_path = self.check_save_mesh(mesh2)
        dims, vars = scan_dataset(tempfile_path)

        # Check shape and dimensions of the associated connectivity variables.
        (mesh_name,) = vars_meshnames(vars)
        mesh_props = vars[mesh_name]
        faceconn_name = mesh_props["face_node_connectivity"]
        edgeconn_name = mesh_props["edge_node_connectivity"]
        faceconn_props = vars[faceconn_name]
        edgeconn_props = vars[edgeconn_name]
        self.assertEqual(
            faceconn_props[_VAR_DIMS], ["Mesh_2d_face_N_nodes", "Mesh2d_face"]
        )
        self.assertEqual(
            edgeconn_props[_VAR_DIMS], ["Mesh_2d_edge_N_nodes", "Mesh2d_edge"]
        )

        # Check the dimension lengths are also as expected
        self.assertEqual(dims["Mesh2d_face"], 2)
        self.assertEqual(dims["Mesh_2d_face_N_nodes"], 4)
        self.assertEqual(dims["Mesh2d_edge"], 7)
        self.assertEqual(dims["Mesh_2d_edge_N_nodes"], 2)

        # the mesh has extra location-dimension properties
        self.assertEqual(mesh_props["face_dimension"], "Mesh2d_face")
        self.assertEqual(mesh_props["edge_dimension"], "Mesh2d_edge")

    def test_connectivity_start_index(self):
        """Test a mesh where some connectivities have start_index = 1."""
        # Make a mesh with both faces *and* some edges
        mesh = make_mesh(n_edges=7)
        # Get the face-node and edge-node connectivities
        face_nodes_conn = mesh.face_node_connectivity
        edge_nodes_conn = mesh.edge_node_connectivity
        edge_nodes_conn2 = Connectivity(
            indices=edge_nodes_conn.indices + 1,
            cf_role=edge_nodes_conn.cf_role,
            var_name="edges_x_2",
            start_index=1,
        )
        # Make a new mesh with altered connectivities.
        mesh2 = Mesh(
            topology_dimension=mesh.topology_dimension,
            node_coords_and_axes=zip(mesh.node_coords, XY_LOCS),
            face_coords_and_axes=zip(mesh.face_coords, XY_LOCS),
            connectivities=[face_nodes_conn, edge_nodes_conn2],
        )

        # Save and snapshot the result
        tempfile_path = self.check_save_mesh(mesh2)
        dims, vars = scan_dataset(tempfile_path)

        # Check shape and dimensions of the associated connectivity variables.
        (mesh_name,) = vars_meshnames(vars)
        mesh_props = vars[mesh_name]
        faceconn_name = mesh_props["face_node_connectivity"]
        edgeconn_name = mesh_props["edge_node_connectivity"]
        faceconn_props = vars[faceconn_name]
        edgeconn_props = vars[edgeconn_name]
        self.assertEqual(faceconn_props["start_index"], 0)
        self.assertEqual(edgeconn_props["start_index"], 1)

    def test_nonuniform_connectivity(self):
        # Check handling of connectivities with missing points.
        n_faces = 7
        mesh = make_mesh(n_faces=n_faces)

        # In this case, add on a partial face-face connectivity.
        # construct a vaguely plausible face-face index array
        indices = np.ma.arange(n_faces * 4).reshape((7, 4))
        indices = indices % 7
        # make some missing points -- i.e. not all faces have 4 neighbours
        indices[(2, (2, 3))] = np.ma.masked
        indices[(3, (0, 2))] = np.ma.masked
        indices[6, :] = np.ma.masked

        conn = Connectivity(
            indices,
            cf_role="face_face_connectivity",
        )
        mesh.add_connectivities(conn)

        # Save and snapshot the result
        tempfile_path = self.check_save_mesh(mesh)
        dims, vars = scan_dataset(tempfile_path)

        # Check that the mesh saved with the additional connectivity
        (mesh_name,) = vars_meshnames(vars)
        mesh_props = vars[mesh_name]
        self.assertIn("face_face_connectivity", mesh_props)
        ff_conn_name = mesh_props["face_face_connectivity"]

        # check that the connectivity has the corrects dims and fill-property
        ff_props = vars[ff_conn_name]
        self.assertEqual(ff_props[_VAR_DIMS], ["Mesh2d_faces", "Mesh2d_face_N_faces"])
        self.assertIn("_FillValue", ff_props)
        self.assertEqual(ff_props["_FillValue"], -1)

        # Check that a 'normal' connectivity does *not* have a _FillValue
        fn_conn_name = mesh_props["face_node_connectivity"]
        fn_props = vars[fn_conn_name]
        self.assertNotIn("_FillValue", fn_props)

        # For what it's worth, *also* check the actual data array in the file
        ds = _thread_safe_nc.DatasetWrapper(tempfile_path)
        conn_var = ds.variables[ff_conn_name]
        data = conn_var[:]
        ds.close()
        self.assertIsInstance(data, np.ma.MaskedArray)
        self.assertEqual(data.fill_value, -1)
        # Compare raw values stored to indices, but with -1 at missing points
        raw_data = data.data
        filled_indices = indices.filled(-1)
        self.assertArrayEqual(raw_data, filled_indices)

    def test_one_dimensional(self):
        # Test a mesh with edges only.
        mesh = make_mesh(n_edges=5, n_faces=0, mesh_kwargs={"var_name": "Mesh1d"})

        # Save and snapshot the result
        tempfile_path = self.check_save_mesh(mesh)
        dims, vars = scan_dataset(tempfile_path)

        # there is a single mesh-var
        (mesh_name,) = vars_meshnames(vars)

        # the dims include edges but not faces
        self.assertEqual(
            list(dims.keys()),
            ["Mesh1d_node", "Mesh1d_edge", "Mesh1d_edge_N_nodes"],
        )
        self.assertEqual(vars_meshdim(vars, "node"), "Mesh1d_node")
        self.assertEqual(vars_meshdim(vars, "edge"), "Mesh1d_edge")

        # check suitable mesh properties
        self.assertEqual(mesh_name, "Mesh1d")
        mesh_props = vars[mesh_name]
        self.assertEqual(mesh_props["topology_dimension"], 1)
        self.assertIn("edge_node_connectivity", mesh_props)
        self.assertNotIn("face_node_connectivity", mesh_props)

    def test_location_coord_units(self):
        # Check that units on mesh locations are handled correctly.
        # NOTE: at present, the Mesh class cannot handle coordinates that are
        # not recognised by 'guess_coord_axis' == suitable standard names
        mesh = make_mesh(
            nodecoord_xyargs=(
                {
                    "standard_name": "projection_x_coordinate",
                    "var_name": "node_x",
                    "units": "degrees",  # should NOT convert to 'degrees_east'
                    "axis": "x",  # N.B. this is quietly dropped !!
                },
                {
                    "standard_name": "projection_y_coordinate",
                    "var_name": "node_y",
                    "units": "ms-1",
                    "axis": "y",  # N.B. this is quietly dropped !!
                },
            ),
            facecoord_xyargs=(
                {
                    "standard_name": "longitude",
                    "var_name": "face_x",
                    "units": "",  # SHOULD result in no units property
                },
                {
                    "standard_name": "latitude",
                    "var_name": "face_y",  # SHOULD convert to 'degrees_north'
                    "units": "degrees",
                },
            ),
        )

        # Save and snapshot the result
        tempfile_path = self.check_save_mesh(mesh)
        dims, vars = scan_dataset(tempfile_path)

        # there is a single mesh-var
        (mesh_name,) = vars_meshnames(vars)

        # find the node- and face-coordinate variables
        node_x = vars["node_x"]
        node_y = vars["node_y"]
        face_x = vars["face_x"]
        face_y = vars["face_y"]

        # Check that units are as expected.
        # 1. 'long/lat' degree units are converted to east/north
        # 2. non- (plain) lonlat are NOT converted
        # 3. other names remain as whatever was given
        # 4. no units on input --> none on output
        self.assertEqual(node_x["units"], "degrees")
        self.assertEqual(node_y["units"], "ms-1")
        self.assertNotIn("units", face_x)
        self.assertEqual(face_y["units"], "degrees_north")

        # Check also that we did not add 'axis' properties.
        # We should *only* do that for dim-coords.
        self.assertNotIn("axis", node_x)
        self.assertNotIn("axis", node_y)
        self.assertNotIn("axis", face_x)
        self.assertNotIn("axis", face_y)

    @staticmethod
    def _namestext(names):
        name_texts = [
            f'{title}="{name}"'
            for title, name in zip(("standard", "long", "var"), names)
        ]
        return f'({" ".join(name_texts)})'

    def test_mesh_names(self):
        # Check the selection of mesh-variables names.
        # N.B. this is basically centralised in Saver._get_mesh_variable_name,
        # but we test in an implementation-neutral way (as it's fairly easy).
        mesh_names_tests = [
            # no names : based on dimensionality
            (
                (None, None, None),
                (None, None, "Mesh_2d"),
            ),
            # var_name only
            (
                (None, None, "meshvar_x"),
                (None, None, "meshvar_x"),
            ),
            # standard_name only : does not apply to Mesh
            (
                ("air_temperature", None, None),
                ("air_temperature", None, "Mesh_2d"),
            ),
            # long_name only
            (
                (None, "my_long_name", None),
                (None, "my_long_name", "my_long_name"),
            ),
            # long_name that needs "fixing"
            (
                (None, "my long name&%!", None),
                (None, "my long name&%!", "my_long_name___"),
            ),
            # standard + long names
            (
                ("air_temperature", "this_long_name", None),
                ("air_temperature", "this_long_name", "this_long_name"),
            ),
            # long + var names
            (
                (None, "my_longname", "varname"),
                (None, "my_longname", "varname"),
            ),
            # all 3 names
            (
                ("air_temperature", "airtemp long name", "meshvar_varname_1"),
                ("air_temperature", "airtemp long name", "meshvar_varname_1"),
            ),
        ]
        for given_names, expected_names in mesh_names_tests:
            mesh_stdname, mesh_longname, mesh_varname = given_names
            mesh_name_kwargs = {
                "standard_name": mesh_stdname,
                "long_name": mesh_longname,
                "var_name": mesh_varname,
            }
            # Make a mesh, with the mesh names set for the testcase
            mesh = make_mesh(mesh_kwargs=mesh_name_kwargs)

            filepath = self.check_save_mesh(mesh)
            dims, vars = scan_dataset(filepath)

            (mesh_name,) = vars_meshnames(vars)
            mesh_props = vars[mesh_name]
            result_names = (
                mesh_props.get("standard_name", None),
                mesh_props.get("long_name", None),
                mesh_name,
            )
            fail_msg = (
                f"Unexpected resulting names {self._namestext(result_names)} "
                f"when saving mesh with {self._namestext(given_names)}"
            )
            self.assertEqual(expected_names, result_names, fail_msg)

    def test_location_coord_names(self):
        # Check the selection of mesh-element coordinate names.
        # Check the selection of mesh-variables names.
        # N.B. this is basically centralised in Saver._get_mesh_variable_name,
        # but we test in an implementation-neutral way (as it's fairly easy).

        # Options here are limited because the Mesh relies on guess_axis so,
        # for now anyway, coords *must* have a known X/Y-type standard-name
        coord_names_tests = [
            # standard_name only
            (
                ("longitude", None, None),
                ("longitude", None, "longitude"),
            ),
            # standard + long names --> standard
            (
                ("grid_longitude", "long name", None),
                ("grid_longitude", "long name", "grid_longitude"),
            ),
            # standard + var names
            (
                ("grid_longitude", None, "var_name"),
                ("grid_longitude", None, "var_name"),
            ),
            # all 3 names
            (
                ("projection_x_coordinate", "long name", "x_var_name"),
                ("projection_x_coordinate", "long name", "x_var_name"),
            ),
            # # no standard name ?
            # # not possible at present, as Mesh requires a recognisable
            # # standard_name to identify the axis of a location-coord.
            # # TODO: test this if+when Mesh usage is relaxed
            # (
            #     (None, None, 'node_x'),
            #     (None, None, "node_x"),
            # ),
        ]
        for given_names, expected_names in coord_names_tests:
            mesh_stdname, mesh_longname, mesh_varname = given_names

            mesh = make_mesh()
            # Apply the names to the node_x coord of the mesh
            coord = mesh.node_coords[0]
            for key, name in zip(
                ("standard_name", "long_name", "var_name"), given_names
            ):
                setattr(coord, key, name)

            filepath = self.check_save_mesh(mesh)
            dims, vars = scan_dataset(filepath)

            (mesh_name,) = vars_meshnames(vars)
            coord_varname = vars[mesh_name]["node_coordinates"].split(" ")[0]
            coord_props = vars[coord_varname]
            result_names = (
                coord_props.get("standard_name", None),
                coord_props.get("long_name", None),
                coord_varname,
            )
            fail_msg = (
                f"Unexpected resulting names {self._namestext(result_names)} "
                "when saving mesh coordinate "
                f"with {self._namestext(given_names)}"
            )
            self.assertEqual(expected_names, result_names, fail_msg)

    def test_mesh_dim_names(self):
        # Check the selection of dimension names from the mesh.

        dim_names_tests = [
            (None, "Mesh2d_face"),
            ("my_face_dimension", "my_face_dimension"),
            ("dim invalid-name &%!", "dim_invalid_name____"),
        ]
        for given_name, expected_name in dim_names_tests:
            mesh = make_mesh(mesh_kwargs={"face_dimension": given_name})

            filepath = self.check_save_mesh(mesh)
            dims, vars = scan_dataset(filepath)

            (mesh_name,) = vars_meshnames(vars)
            conn_varname = vars[mesh_name]["face_node_connectivity"]
            face_dim = vars[conn_varname][_VAR_DIMS][0]
            fail_msg = (
                f'Unexpected resulting dimension name "{face_dim}" '
                f'when saving mesh with dimension name of "{given_name}".'
            )
            self.assertEqual(expected_name, face_dim, fail_msg)

    def test_connectivity_names(self):
        # Check the selection of connectivity names.
        conn_names_tests = [
            # var_name only
            (
                (None, None, "meshvar_x"),
                (None, None, "meshvar_x"),
            ),
            # standard_name only
            (
                ("air_temperature", None, None),
                ("air_temperature", None, "air_temperature"),
            ),
            # long_name only
            (
                (None, "my_long_name", None),
                (None, "my_long_name", "my_long_name"),
            ),
            # standard + long names
            (
                ("air_temperature", "airtemp long name", None),
                ("air_temperature", "airtemp long name", "air_temperature"),
            ),
            # standard + var names
            (
                ("air_temperature", None, "my_var"),
                ("air_temperature", None, "my_var"),
            ),
            # all 3 names
            (
                ("air_temperature", "airtemp long name", "meshvar_varname_1"),
                ("air_temperature", "airtemp long name", "meshvar_varname_1"),
            ),
            # long name only, with invalid content
            # N.B. behaves *differently* from same in mesh/coord context
            (
                (None, "name with spaces", None),  # character validation
                (None, "name with spaces", "mesh2d_faces"),
            ),
        ]
        for given_names, expected_names in conn_names_tests:
            mesh_stdname, mesh_longname, mesh_varname = given_names

            # Make a mesh and afterwards set the names of one connectivity
            mesh = make_mesh()
            # Apply test names to the face-node connectivity
            conn = mesh.face_node_connectivity
            for key, name in zip(
                ("standard_name", "long_name", "var_name"), given_names
            ):
                setattr(conn, key, name)

            filepath = self.check_save_mesh(mesh)
            dims, vars = scan_dataset(filepath)

            (mesh_name,) = vars_meshnames(vars)
            mesh_props = vars[mesh_name]
            conn_name = mesh_props["face_node_connectivity"]
            conn_props = vars[conn_name]
            result_names = (
                conn_props.get("standard_name", None),
                conn_props.get("long_name", None),
                conn_name,
            )
            fail_msg = (
                f"Unexpected resulting names {self._namestext(result_names)} "
                "when saving connectivity "
                f"with {self._namestext(given_names)}"
            )
            self.assertEqual(expected_names, result_names, fail_msg)

    def test_multiple_equal_mesh(self):
        mesh1 = make_mesh()
        mesh2 = make_mesh()

        # Save and snapshot the result
        tempfile_path = self.check_save_mesh([mesh1, mesh2])
        dims, vars = scan_dataset(tempfile_path)

        # In this case there should be only *one* mesh.
        mesh_names = vars_meshnames(vars)
        self.assertEqual(1, len(mesh_names))

        # Check it has the correct number of coords + conns (no duplicates)
        # Should have 2 each X and Y coords (face+node): _no_ edge coords.
        coord_vars_x = vars_w_props(vars, standard_name="longitude")
        coord_vars_y = vars_w_props(vars, standard_name="latitude")
        self.assertEqual(2, len(coord_vars_x))
        self.assertEqual(2, len(coord_vars_y))

        # Check the connectivities are all present: _only_ 1 var of each type.
        for conn in mesh1.all_connectivities:
            if conn is not None:
                conn_vars = vars_w_props(vars, cf_role=conn.cf_role)
                self.assertEqual(1, len(conn_vars))

    def test_multiple_different_meshes(self):
        # Create 2 meshes with different faces, but same edges.
        # N.B. they should *not* then share an edge dimension !
        mesh1 = make_mesh(n_faces=3, n_edges=2)
        mesh2 = make_mesh(n_faces=4, n_edges=2)

        # Save and snapshot the result
        tempfile_path = self.check_save_mesh([mesh1, mesh2])
        dims, vars = scan_dataset(tempfile_path)

        # Check the dims are as expected
        self.assertEqual(dims["Mesh2d_faces"], 3)
        self.assertEqual(dims["Mesh2d_faces_0"], 4)
        # There are no 'second' edge and node dims
        self.assertEqual(dims["Mesh2d_nodes"], 5)
        self.assertEqual(dims["Mesh2d_edge"], 2)

        # Check there are two independent meshes in the file...

        # there are exactly 2 meshes in the file
        mesh_names = vars_meshnames(vars)
        self.assertEqual(sorted(mesh_names), ["Mesh2d", "Mesh2d_0"])

        # they use different dimensions
        # mesh1
        self.assertEqual(vars_meshdim(vars, "node", mesh_name="Mesh2d"), "Mesh2d_nodes")
        self.assertEqual(vars_meshdim(vars, "face", mesh_name="Mesh2d"), "Mesh2d_faces")
        if "edge_coordinates" in vars["Mesh2d"]:
            self.assertEqual(
                vars_meshdim(vars, "edge", mesh_name="Mesh2d"), "Mesh2d_edge"
            )

        # mesh2
        self.assertEqual(
            vars_meshdim(vars, "node", mesh_name="Mesh2d_0"), "Mesh2d_nodes"
        )
        self.assertEqual(
            vars_meshdim(vars, "face", mesh_name="Mesh2d_0"), "Mesh2d_faces_0"
        )
        if "edge_coordinates" in vars["Mesh2d_0"]:
            self.assertEqual(
                vars_meshdim(vars, "edge", mesh_name="Mesh2d_0"),
                "Mesh2d_edge",
            )

        # the relevant coords + connectivities are also distinct
        # mesh1
        self.assertEqual(vars["node_x"][_VAR_DIMS], ["Mesh2d_nodes"])
        self.assertEqual(vars["face_x"][_VAR_DIMS], ["Mesh2d_faces"])
        self.assertEqual(
            vars["mesh2d_faces"][_VAR_DIMS],
            ["Mesh2d_faces", "Mesh2d_face_N_nodes"],
        )
        if "edge_coordinates" in vars["Mesh2d"]:
            self.assertEqual(vars["longitude"][_VAR_DIMS], ["Mesh2d_edge"])
            self.assertEqual(
                vars["mesh2d_edge"][_VAR_DIMS],
                ["Mesh2d_edge", "Mesh2d_edge_N_nodes"],
            )

        # mesh2
        self.assertEqual(vars["node_x_0"][_VAR_DIMS], ["Mesh2d_nodes"])
        self.assertEqual(vars["face_x_0"][_VAR_DIMS], ["Mesh2d_faces_0"])
        self.assertEqual(
            vars["mesh2d_faces_0"][_VAR_DIMS],
            ["Mesh2d_faces_0", "Mesh2d_0_face_N_nodes"],
        )
        if "edge_coordinates" in vars["Mesh2d_0"]:
            self.assertEqual(vars["longitude_0"][_VAR_DIMS], ["Mesh2d_edge"])
            self.assertEqual(
                vars["mesh2d_edge_0"][_VAR_DIMS],
                ["Mesh2d_edge", "Mesh2d_0_edge_N_nodes"],
            )


# WHEN MODIFYING THIS MODULE, CHECK IF ANY CORRESPONDING CHANGES ARE NEEDED IN
# :mod:`iris.tests.unit.fileformats.netcdf.test_Saver__lazy.`


if __name__ == "__main__":
    tests.main()
