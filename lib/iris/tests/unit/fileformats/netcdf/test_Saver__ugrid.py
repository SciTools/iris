# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.fileformats.netcdf.Saver` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from pathlib import Path
import shutil
from subprocess import check_output
import tempfile

import netCDF4 as nc
import numpy as np

from iris import save
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList
from iris.experimental.ugrid import Connectivity, Mesh
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
    """
    Make a test mesh.

    Mesh has faces edges, face-coords and edge-coords, numbers of which can be controlled.

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

    # NB when creating coords, supply axis to make Mesh.to_AuxCoords work
    node_coords = [
        AuxCoord(np.arange(n_nodes), standard_name=name)
        for loc, name in zip(XY_LOCS, XY_NAMES)
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
            AuxCoord(np.arange(n_edges), standard_name=name)
            for loc, name in zip(XY_LOCS, XY_NAMES)
        ]
        apply_xyargs(edge_coords, edgecoord_xyargs)

    if n_faces:
        topology_dimension = 2
        connectivities["face_node_connectivity"] = Connectivity(
            np.zeros((n_faces, 4), np.int32), cf_role="face_node_connectivity"
        )
        face_coords = [
            AuxCoord(np.arange(n_faces), standard_name=name)
            for loc, name in zip(XY_LOCS, XY_NAMES)
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
                conn = Connectivity(
                    np.zeros(dims, dtype=np.int32), cf_role=role
                )
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
    if basic:
        # Use some helpful non-minimal settings as our 'basic' mesh.
        use_kwargs = dict(
            n_nodes=5,
            n_faces=2,
            nodecoord_xyargs=tuple(
                dict(var_name=f"node_{loc}") for loc in XY_LOCS
            ),
            facecoord_xyargs=tuple(
                dict(var_name=f"face_{loc}") for loc in XY_LOCS
            ),
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
    """Get the size of a location-dimension from a mesh."""
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
            result = conn.shape[conn.src_dim]
    return result


# Pre-create a simple "standard" test mesh for multiple uses
_DEFAULT_MESH = make_mesh()


def make_cube(mesh=_DEFAULT_MESH, location="face", **kwargs):
    dim = mesh_location_size(mesh, location)
    cube = Cube(np.zeros(dim, np.float32))
    for meshco in mesh.to_MeshCoords(location):
        cube.add_aux_coord(meshco, (0,))
    for key, val in kwargs.items():
        setattr(cube, key, val)
    return cube


def add_height_dim(cube):
    # Add an extra inital 'height' dimension onto a cube.
    cube = cube.copy()  # Avoid trashing the input cube.
    cube.add_aux_coord(AuxCoord([0.0], standard_name="height", units="m"))
    # Make three copies with different heights
    cubes = [cube.copy() for _ in range(3)]
    for i_cube, cube in enumerate(cubes):
        cube.coord("height").points = [i_cube]
    # Merge to create an additional 'height' dimension.
    cube = CubeList(cubes).merge_cube()
    return cube


def scan_dataset(filepath):
    """
    Snapshot a netcdf dataset (the key metadata).

    Returns:
        dimsdict, varsdict
        * dimsdict (dict):
            A map of dimension-name: length.
        * varsdict (dict):
            A map of each variable's properties, {var_name: propsdict}
            Each propsdict is {attribute-name: value} over the var's ncattrs().
            Each propsdict ALSO contains a ['_DIMS'] entry listing the
            variable's dims.

    """
    ds = nc.Dataset(filepath)
    # dims dict is {name: len}
    dimsdict = {name: dim.size for name, dim in ds.dimensions.items()}
    # vars dict is {name: {attr:val}}
    varsdict = {}
    for name, var in ds.variables.items():
        varsdict[name] = {prop: getattr(var, prop) for prop in var.ncattrs()}
        varsdict[name]["_DIMS"] = list(var.dimensions)
    ds.close()
    return dimsdict, varsdict


def vars_w_props(varsdict, **kwargs):
    """
    Subset a vars dict, {name:props}, returning only those where each
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
        name: attrs
        for name, attrs in varsdict.items()
        if check_attrs_match(attrs)
    }
    return varsdict


def vars_w_dims(varsdict, dim_names):
    """Subset a vars dict, returning all those which map all the specified dims."""
    varsdict = {
        name: propsdict
        for name, propsdict in varsdict.items()
        if all(dim in propsdict["_DIMS"] for dim in dim_names)
    }
    return varsdict


def vars_meshnames(vars):
    """Return the names of all the mesh variables (found by cf_role)."""
    return list(vars_w_props(vars, cf_role="mesh_topology").keys())


def vars_meshdim(vars, location, mesh_name=None):
    """ "
    Extract a dim-name for a given element location.

    Args:
        * vars (varsdict):
            file varsdict, as returned from 'snapshot_dataset'.
        * location (string):
            a mesh location : 'node' / 'edge' / 'face'
        * mesh_name (string or None):
            If given, identifies the mesh var.
            Otherwise, find a unique mesh var (i.e. there must be exactly 1).

    Returns:
        dim_name (string)
            The dim-name of the mesh dim for the given location.

    TODO: relies on the element having coordinates, which in future will not
        always be the case. This can be fixed

    """
    if mesh_name is None:
        # Find "the" meshvar -- assuming there is just one.
        (mesh_name,) = vars_meshnames(vars)
    mesh_props = vars[mesh_name]
    loc_coords = mesh_props[f"{location}_coordinates"].split(" ")
    (single_location_dim,) = vars[loc_coords[0]]["_DIMS"]
    return single_location_dim


class TestSaveUgrid__cube(tests.IrisTest):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.tempfile_path = str(cls.temp_dir / "tmp.nc")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def check_save(self, data):
        save(data, self.tempfile_path)
        text = check_output(f"ncdump -h {self.tempfile_path}", shell=True)
        text = text.decode()
        print(text)
        return scan_dataset(self.tempfile_path)

    def test_basic_mesh(self):
        # Save a small mesh example and check most aspects of the resultin file.
        data = make_cube()  # A simple face-mapped data example.

        # Save and snapshot the result
        dims, vars = self.check_save(data)

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
        self.assertTrue(
            all(vars[co]["_DIMS"] == [face_dim] for co in face_coords)
        )

        # The dims of the datavar also == [<faces-dim>]
        self.assertEqual(a_props["_DIMS"], [face_dim])

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
        self.assertEqual(
            len(vars_w_props(vars, cf_role="edge_node_connectivity")), 0
        )

        # The dims are precisely (nodes, faces, nodes-per-face), in that order.
        self.assertEqual(
            list(dims.keys()),
            ["Mesh2d_nodes", "Mesh2d_faces", "Mesh2d_face_N_nodes"],
        )

        # The variables are (mesh, 2*node-coords, 2*face-coords, face-nodes, data),
        # in that order
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

        # For definiteness, also check against a full CDL snapshot
        self.assertCDL(self.tempfile_path)

    def test_multi_cubes_common_mesh(self):
        cube1 = make_cube(var_name="a")
        cube2 = make_cube(var_name="b")

        # Save and snapshot the result
        dims, vars = self.check_save([cube1, cube2])

        # there is exactly 1 mesh in the file
        (mesh_name,) = vars_meshnames(vars)

        # both the main variables reference the same mesh, and 'face' location
        v_a, v_b = vars["a"], vars["b"]
        self.assertEqual(v_a["mesh"], mesh_name)
        self.assertEqual(v_a["location"], "face")
        self.assertEqual(v_b["mesh"], mesh_name)
        self.assertEqual(v_b["location"], "face")

    def test_multi_cubes_different_locations(self):
        cube1 = make_cube(var_name="a", location="face")
        cube2 = make_cube(var_name="b", location="node")

        # Save and snapshot the result
        dims, vars = self.check_save([cube1, cube2])

        # there is exactly 1 mesh in the file
        (mesh_name,) = vars_meshnames(vars)

        # the main variables reference the same mesh at different locations
        v_a, v_b = vars["a"], vars["b"]
        self.assertEqual(v_a["mesh"], mesh_name)
        self.assertEqual(v_a["location"], "face")
        self.assertEqual(v_b["mesh"], mesh_name)
        self.assertEqual(v_b["location"], "node")

        # the main variables map the face and node dimensions
        face_dim = vars_meshdim(vars, "face")
        node_dim = vars_meshdim(vars, "node")
        self.assertEqual(v_a["_DIMS"], [face_dim])
        self.assertEqual(v_b["_DIMS"], [node_dim])

    def test_multi_cubes_identical_meshes(self):
        # Make 2 identical meshes
        # NOTE: *can't* name these explicitly, as it stops them being identical.
        mesh1 = make_mesh()
        mesh2 = make_mesh()
        cube1 = make_cube(var_name="a", mesh=mesh1)
        cube2 = make_cube(var_name="b", mesh=mesh2)

        # Save and snapshot the result
        dims, vars = self.check_save([cube1, cube2])

        # there are exactly 2 meshes in the file
        mesh_names = vars_meshnames(vars)
        self.assertEqual(sorted(mesh_names), ["Mesh2d", "Mesh2d_0"])

        # they use different dimensions
        self.assertEqual(
            vars_meshdim(vars, "node", mesh_name="Mesh2d"), "Mesh2d_nodes"
        )
        self.assertEqual(
            vars_meshdim(vars, "face", mesh_name="Mesh2d"), "Mesh2d_faces"
        )
        self.assertEqual(
            vars_meshdim(vars, "node", mesh_name="Mesh2d_0"), "Mesh2d_nodes_0"
        )
        self.assertEqual(
            vars_meshdim(vars, "face", mesh_name="Mesh2d_0"), "Mesh2d_faces_0"
        )

        # there are exactly two data-variables with a 'mesh' property
        mesh_datavars = vars_w_props(vars, mesh="*")
        self.assertEqual(["a", "b"], list(mesh_datavars))

        # the data variables reference the two separate meshes
        a_props, b_props = vars["a"], vars["b"]
        self.assertEqual(a_props["mesh"], "Mesh2d")
        self.assertEqual(a_props["location"], "face")
        self.assertEqual(b_props["mesh"], "Mesh2d_0")
        self.assertEqual(b_props["location"], "face")

        # the data variables map the appropriate node dimensions
        self.assertEqual(a_props["_DIMS"], ["Mesh2d_faces"])
        self.assertEqual(b_props["_DIMS"], ["Mesh2d_faces_0"])

    def test_multi_cubes_different_mesh(self):
        # Check that we can correctly distinguish 2 different meshes.
        cube1 = make_cube(var_name="a")
        cube2 = make_cube(var_name="b", mesh=make_mesh(n_faces=4))

        # Save and snapshot the result
        dims, vars = self.check_save([cube1, cube2])

        # there are 2 meshes in the file
        mesh_names = vars_meshnames(vars)
        self.assertEqual(len(mesh_names), 2)

        # there are two (data)variables with a 'mesh' property
        mesh_datavars = vars_w_props(vars, mesh="*")
        self.assertEqual(2, len(mesh_datavars))
        self.assertEqual(["a", "b"], sorted(mesh_datavars.keys()))

        # the main variables reference the respective meshes, and 'face' location
        a_props, b_props = vars["a"], vars["b"]
        mesh_a, loc_a = a_props["mesh"], a_props["location"]
        mesh_b, loc_b = b_props["mesh"], b_props["location"]
        self.assertNotEqual(mesh_a, mesh_b)
        self.assertEqual(loc_a, "face")
        self.assertEqual(loc_b, "face")

    def test_nonmesh_dim(self):
        # Check where the data variable has a 'normal' dim and a mesh dim.
        cube = make_cube()
        cube = add_height_dim(cube)

        # Save and snapshot the result
        dims, vars = self.check_save(cube)

        # have just 1 mesh, including a face and node coordinates.
        (mesh_name,) = vars_meshnames(vars)
        face_dim = vars_meshdim(vars, "face", mesh_name)
        _ = vars_meshdim(vars, "node", mesh_name)

        # have just 1 data-variable
        ((data_name, data_props),) = vars_w_props(vars, mesh="*").items()

        # data maps to the height + mesh dims
        self.assertEqual(data_props["_DIMS"], ["height", face_dim])
        self.assertEqual(data_props["mesh"], mesh_name)
        self.assertEqual(data_props["location"], "face")

    def test_nonmesh_hybrid_dim(self):
        # Check a case with a hybrid non-mesh dimension
        cube = realistic_4d()
        # Strip off the time and longtude dims, to make it simpler.
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
        dims, vars = self.check_save(cube)

        # have just 1 mesh, including face and node coordinates.
        (mesh_name,) = vars_meshnames(vars)
        face_dim = vars_meshdim(vars, "face", mesh_name)
        _ = vars_meshdim(vars, "node", mesh_name)

        # have hybrid vertical dimension, with all the usual term variables.
        self.assertIn("model_level_number", dims)
        vert_vars = list(vars_w_dims(vars, ["model_level_number"]).keys())
        # The list of file variables mapping the vertical dimensio:
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
        self.assertEqual(data_props["_DIMS"], ["model_level_number", face_dim])
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
        dims, vars = self.check_save([cube_1, cube_2])

        # There is only 1 mesh
        (mesh_name,) = vars_meshnames(vars)

        # both variables reference the same mesh
        v_a, v_b = vars["a"], vars["b"]
        self.assertEqual(v_a["mesh"], mesh_name)
        self.assertEqual(v_a["location"], "face")
        self.assertEqual(v_b["mesh"], mesh_name)
        self.assertEqual(v_b["location"], "face")

        # Check the var dimensions
        self.assertEqual(v_a["_DIMS"], ["height", "Mesh2d_faces"])
        self.assertEqual(v_b["_DIMS"], ["Mesh2d_faces", "height"])

    def test_alternate_connectivity_dim_order(self):
        # A mesh with some connectivities in the 'other' order.
        # This should also create a property with the dimension name
        mesh = make_mesh(n_edges=7)
        # Get the face-node and edge-node connectivities
        face_nodes_conn = mesh.face_node_connectivity
        edge_nodes_conn = mesh.edge_node_connectivity
        # Transpose them : N.B. this sets src_dim=1, as it should be.
        nodesfirst_faces_conn = face_nodes_conn.transpose()
        nodesfirst_edges_conn = edge_nodes_conn.transpose()
        # Make a new mesh with both face and edge connectivities 'transposed'.
        mesh2 = Mesh(
            topology_dimension=mesh.topology_dimension,
            node_coords_and_axes=zip(mesh.node_coords, XY_LOCS),
            face_coords_and_axes=zip(mesh.face_coords, XY_LOCS),
            connectivities=[nodesfirst_faces_conn, nodesfirst_edges_conn],
        )

        # Build a cube on the modified mesh
        cube = make_cube(mesh=mesh2)

        # Save and snapshot the result
        dims, vars = self.check_save(cube)

        # Check shape and dimensions of the associated connectivity variables.
        (mesh_name,) = vars_meshnames(vars)
        mesh_props = vars[mesh_name]
        faceconn_name = mesh_props["face_node_connectivity"]
        edgeconn_name = mesh_props["edge_node_connectivity"]
        faceconn_props = vars[faceconn_name]
        edgeconn_props = vars[edgeconn_name]
        self.assertEqual(
            faceconn_props["_DIMS"], ["Mesh_2d_face_N_nodes", "Mesh2d_face"]
        )
        self.assertEqual(
            edgeconn_props["_DIMS"], ["Mesh_2d_edge_N_nodes", "Mesh2d_edge"]
        )

        # Check the dimension lengths are also as expected
        self.assertEqual(dims["Mesh2d_face"], 2)
        self.assertEqual(dims["Mesh_2d_face_N_nodes"], 4)
        self.assertEqual(dims["Mesh2d_edge"], 7)
        self.assertEqual(dims["Mesh_2d_edge_N_nodes"], 2)

        # the mesh has extra location-dimension properties
        self.assertEqual(mesh_props["face_dimension"], "Mesh2d_face")
        self.assertEqual(mesh_props["edge_dimension"], "Mesh2d_edge")


if __name__ == "__main__":
    tests.main()
