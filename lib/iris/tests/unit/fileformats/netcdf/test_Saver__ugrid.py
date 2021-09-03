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

import numpy as np

from iris import save
from iris.coords import AuxCoord
from iris.cube import Cube
from iris.experimental.ugrid import Connectivity, Mesh

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
    mesh._mesh_dims = mesh_dims

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


# Pre-create a simple "standard" test mesh for multiple uses
_DEFAULT_MESH = make_mesh()


def make_cube(mesh=_DEFAULT_MESH, location="face", **kwargs):
    dim = mesh._mesh_dims[location]
    cube = Cube(np.zeros(dim, np.float32))
    for meshco in mesh.to_MeshCoords(location):
        cube.add_aux_coord(meshco, (0,))
    for key, val in kwargs.items():
        setattr(cube, key, val)
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
            Each propsdict ALSO contains a ['_dims'] entry listing the
            variable's dims.

    """
    import netCDF4 as nc

    ds = nc.Dataset(filepath)
    # dims dict is {name: len}
    dimsdict = {name: dim.size for name, dim in ds.dimensions.items()}
    # vars dict is {name: {attr:val}}
    varsdict = {}
    for name, var in ds.variables.items():
        varsdict[name] = {prop: getattr(var, prop) for prop in var.ncattrs()}
        varsdict[name]["_dims"] = list(var.dimensions)
    ds.close()
    return dimsdict, varsdict


def vars_w_props(varsdict, **kwargs):
    """
    Subset a vars dict, {name:props}, returning only those where each
    <attribute>=<value>, defined by the given keywords.
    Except that '<key>="*"' means that an attribute '<key>' merely _exists_.

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
        if all(dim in propsdict["_dims"] for dim in dim_names)
    }
    return varsdict


def vars_meshvars(vars):
    """Subset a varsdict, returning those which are mesh variables (by cf_role)."""
    return vars_w_props(vars, cf_role="mesh_topology")


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

    """
    if mesh_name is None:
        # Find "the" meshvar -- assuming there is just one.
        (mesh_name,) = vars_meshvars(vars).keys()
    mesh_props = vars[mesh_name]
    loc_coords = mesh_props[f"{location}_coordinates"].split(" ")
    (single_location_dim,) = vars[loc_coords[0]]["_dims"]
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

    def test_basic_mesh_cdl(self):
        data = make_cube(mesh=_DEFAULT_MESH)
        dims, vars = self.check_save(data)

        # There is exactly 1 mesh var.
        ((mesh_name, mesh_props),) = vars_meshvars(vars).items()

        # There is exactly 1 mesh-linked (data)var
        data_vars = vars_w_props(vars, mesh="*")
        ((a_name, a_props),) = data_vars.items()

        # The mesh var links to the mesh, with location 'faces'
        self.assertEqual(a_name, "unknown")
        self.assertEqual(a_props["mesh"], mesh_name)
        self.assertEqual(a_props["location"], "face")

        # There are 2 face coords == those listed in the mesh
        face_coords = mesh_props["face_coordinates"].split(" ")
        self.assertEqual(len(face_coords), 2)
        # get face dim (actually, from the dims of the first face coord)
        face_dim = vars_meshdim(vars, "face")
        # The face coords should both map that single dim.
        self.assertTrue(
            all(vars[co]["_dims"] == [face_dim] for co in face_coords)
        )

        # The dims of the datavar also == [<faces-dim>]
        self.assertEqual(a_props["_dims"], [face_dim])

        # There are 2 node coordinates == those listed in the mesh.
        node_coords = mesh_props["node_coordinates"].split(" ")
        self.assertEqual(len(node_coords), 2)
        # These should be the only ones using the 'nodes' dimension.
        node_dim = vars_meshdim(vars, "node")
        self.assertEqual(
            sorted(node_coords), sorted(vars_w_dims(vars, [node_dim]).keys())
        )

        # There are no edges.
        self.assertEqual(
            len(vars_w_props(vars, cf_role="edge_node_connectivity")), 0
        )

    def test_multi_cubes_common_mesh(self):
        cube1 = make_cube(var_name="a")
        cube2 = make_cube(var_name="b")
        dims, vars = self.check_save([cube1, cube2])
        # there is exactly 1 mesh in the file
        ((mesh_name, _),) = vars_meshvars(vars).items()
        # both the main variables reference the same mesh, and 'face' location
        v_a, v_b = vars["a"], vars["b"]
        self.assertEqual(v_a["mesh"], mesh_name)
        self.assertEqual(v_a["location"], "face")
        self.assertEqual(v_b["mesh"], mesh_name)
        self.assertEqual(v_b["location"], "face")
        # self.assertCDL(self.tempfile_path)

    def test_multi_cubes_different_locations(self):
        cube1 = make_cube(var_name="a", location="face")
        cube2 = make_cube(var_name="b", location="node")
        dims, vars = self.check_save([cube1, cube2])

        # there is exactly 1 mesh in the file
        ((mesh_name, mesh_props),) = vars_meshvars(vars).items()

        # the main variables reference the same mesh at different locations
        v_a, v_b = vars["a"], vars["b"]
        self.assertEqual(v_a["mesh"], mesh_name)
        self.assertEqual(v_a["location"], "face")
        self.assertEqual(v_b["mesh"], mesh_name)
        self.assertEqual(v_b["location"], "node")

        # the main variables map the face and node dimensions
        face_dim = vars_meshdim(vars, "face")
        node_dim = vars_meshdim(vars, "node")
        self.assertEqual(v_a["_dims"], [face_dim])
        self.assertEqual(v_b["_dims"], [node_dim])

    def test_multi_cubes_identical_meshes(self):
        mesh1 = make_mesh()
        mesh2 = make_mesh()
        cube1 = make_cube(var_name="a", mesh=mesh1)
        cube2 = make_cube(var_name="b", mesh=mesh2)
        cube1 = make_cube(var_name="a")
        cube2 = make_cube(var_name="b")
        dims, vars = self.check_save([cube1, cube2])

        # there are 2 meshes in the file
        mesh_vars = vars_meshvars(vars)
        self.assertEqual(len(mesh_vars), 2)

        # there are two (data)variables with a 'mesh' property
        mesh_datavars = vars_w_props(vars, mesh="*")
        self.assertEqual(2, len(mesh_datavars))
        self.assertEqual(["a", "b"], sorted(mesh_datavars.keys()))

        # the data variables reference the two separate meshes
        a_props, b_props = vars["a"], vars["b"]
        mesh_a, mesh_b = a_props["mesh"], b_props["mesh"]
        self.assertEqual(sorted([mesh_a, mesh_b]), sorted(mesh_vars.keys()))

    def test_multi_cubes_different_mesh(self):
        cube1 = make_cube(var_name="a")
        cube2 = make_cube(var_name="b", mesh=make_mesh(n_faces=4))
        self.check_save([cube1, cube2])
        dims, vars = self.check_save([cube1, cube2])
        # there are 2 meshes in the file
        mesh_vars = vars_w_props(vars, cf_role="mesh_topology")
        self.assertEqual(len(mesh_vars), 2)
        # there are two (data)variables with a 'mesh' property
        mesh_datavars = vars_w_props(vars, mesh="*")
        self.assertEqual(2, len(mesh_datavars))
        self.assertEqual(["a", "b"], sorted(mesh_datavars.keys()))
        # the main variables reference the same mesh, and 'face' location
        # self.assertCDL(self.tempfile_path)


if __name__ == "__main__":
    tests.main()
