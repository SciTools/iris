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
    conn_role_kwargs={},  # role: kwargs
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


# Make simple "standard" test meshes for multiple uses
_DEFAULT_MESH = make_mesh()
# NB the 'minimal' mesh might have just nodes, but Mesh doesn't (yet) support it
_MINIMAL_MESH = make_mesh(basic=False, n_nodes=3, n_edges=2, n_faces=0)


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
    Snapshot a dataset (the key metadata).

    Returns:
        a tuple (dimsdict, varsdict)
        * dimsdict (dict):
            A mapping of dimension-name: length.
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
    Subset a vars dict, {name:props}, for those where given attributes=values.
    Except '<key>="*"' means the '<key>' attribute must only exist.

    """

    def check_attrs_match(attrs):
        result = True
        for key, val in kwargs.items():
            result = key in attrs
            if result and val != "*":
                # val='*'' for a simple existence check
                # Otherwise actual value must also match
                result = attrs[key] == val
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
    """Subset a vars dict elect, selecting only those with all the given dimensions."""
    varsdict = {
        name: propsdict
        for name, propsdict in varsdict.items()
        if all(dim in propsdict["_dims"] for dim in dim_names)
    }
    return varsdict


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

    def test_minimal_mesh_cdl(self):
        data = make_cube(mesh=_MINIMAL_MESH, location="edge")
        dims, vars = self.check_save(data)

        # There should be 1 mesh var.
        mesh_vars = vars_w_props(vars, cf_role="mesh_topology")
        self.assertEqual(1, len(mesh_vars))
        (mesh_name,) = mesh_vars.keys()

        # There should be 1 mesh-linked (data)var
        data_vars = vars_w_props(vars, mesh="*")
        self.assertEqual(1, len(data_vars))

        # The mesh var should link to the mesh, at 'edges'
        self.assertEqual(["unknown"], list(data_vars.keys()))
        (a_props,) = data_vars.values()
        self.assertEqual(mesh_name, a_props["mesh"])
        self.assertEqual("edge", a_props["location"])

        # get name of first edge coord
        edge_coord = vars[mesh_name]["edge_coordinates"].split(" ")[0]
        # get edge dim = first dim of edge coord
        (edge_dim,) = vars[edge_coord]["_dims"]

        # The dims of the datavar should == [edges]
        self.assertEqual([edge_dim], a_props["_dims"])

    def test_basic_mesh_cdl(self):
        data = make_cube(mesh=_DEFAULT_MESH)
        self.check_save(data)
        # self.assertCDL(self.tempfile_path)

    def test_multi_cubes_common_mesh(self):
        cube1 = make_cube(var_name="a")
        cube2 = make_cube(var_name="b")
        dims, vars = self.check_save([cube1, cube2])
        # there is only 1 mesh in the file
        mesh_vars = vars_w_props(vars, cf_role="mesh_topology")
        self.assertEqual(len(mesh_vars), 1)
        (mesh_name,) = mesh_vars.keys()
        # both the main variables reference the same mesh, and 'face' location
        v_a, v_b = vars["a"], vars["b"]
        self.assertEqual(mesh_name, v_a["mesh"])
        self.assertEqual("face", v_a["location"])
        self.assertEqual(mesh_name, v_b["mesh"])
        self.assertEqual("face", v_b["location"])
        # self.assertCDL(self.tempfile_path)

    def test_multi_cubes_identical_meshes(self):
        mesh1 = make_mesh()
        mesh2 = make_mesh()
        cube1 = make_cube(var_name="a", mesh=mesh1)
        cube2 = make_cube(var_name="b", mesh=mesh2)
        cube1 = make_cube(var_name="a")
        cube2 = make_cube(var_name="b")
        dims, vars = self.check_save([cube1, cube2])
        # there are 2 meshes in the file
        mesh_vars = vars_w_props(vars, cf_role="mesh_topology")
        self.assertEqual(len(mesh_vars), 2)
        # there are two (data)variables with a 'mesh' property
        mesh_datavars = vars_w_props(vars, mesh="*")
        self.assertEqual(2, len(mesh_datavars))
        self.assertEqual(["a", "b"], sorted(mesh_datavars.keys()))
        # the main variables reference the same mesh, and 'face' location
        v_a, v_b = vars["a"], vars["b"]
        mesh_a, mesh_b = v_a["mesh"], v_b["mesh"]
        self.assertNotEqual(mesh_a, mesh_b)
        self.assertEqual(sorted([mesh_a, mesh_b]), sorted(mesh_vars.keys()))
        # self.assertCDL(self.tempfile_path)

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

    def test_multi_cubes_different_locations(self):
        cube1 = make_cube(var_name="a", location="face")
        cube2 = make_cube(var_name="b", location="node")
        self.check_save([cube1, cube2])
        # self.assertCDL(self.tempfile_path)


if __name__ == "__main__":
    tests.main()
