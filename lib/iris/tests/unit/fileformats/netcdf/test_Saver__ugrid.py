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


from subprocess import check_output


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

    def test_minimal_mesh_cdl(self):
        data = make_cube(mesh=_MINIMAL_MESH, location="edge")
        self.check_save(data)
        # self.assertCDL(self.tempfile_path)

    def test_basic_mesh_cdl(self):
        data = make_cube(mesh=_DEFAULT_MESH)
        self.check_save(data)
        # self.assertCDL(self.tempfile_path)

    def test_multi_cubes_common_mesh(self):
        cube1 = make_cube(var_name="a")
        cube2 = make_cube(var_name="b")
        self.check_save([cube1, cube2])
        # self.assertCDL(self.tempfile_path)

    def test_multi_cubes_identical_mesh(self):
        mesh1 = make_mesh()
        mesh2 = make_mesh()
        cube1 = make_cube(var_name="a", mesh=mesh1)
        cube2 = make_cube(var_name="b", mesh=mesh2)
        self.check_save([cube1, cube2])
        # self.assertCDL(self.tempfile_path)

    def test_multi_cubes_different_mesh(self):
        cube1 = make_cube(var_name="a")
        cube2 = make_cube(var_name="b", mesh=make_mesh(n_faces=4))
        self.check_save([cube1, cube2])
        # self.assertCDL(self.tempfile_path)

    def test_multi_cubes_different_locations(self):
        cube1 = make_cube(var_name="a", location="face")
        cube2 = make_cube(var_name="b", location="node")
        self.check_save([cube1, cube2])
        # self.assertCDL(self.tempfile_path)


if __name__ == "__main__":
    tests.main()
