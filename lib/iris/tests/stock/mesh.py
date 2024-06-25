# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Helper functions making objects for unstructured mesh testing."""

import numpy as np

from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.experimental.ugrid.mesh import Connectivity, Mesh, MeshCoord

# Default creation controls for creating a test Mesh.
# Note: we're not creating any kind of sensible 'normal' mesh here, the numbers
# of nodes/faces/edges are quite arbitrary and the connectivities we generate
# are pretty random too.
_TEST_N_NODES = 15
_TEST_N_FACES = 3
_TEST_N_EDGES = 5
_TEST_N_BOUNDS = 4


def sample_mesh(
    n_nodes=None,
    n_faces=None,
    n_edges=None,
    lazy_values=False,
    nodes_per_face=None,
    masked_connecteds=False,
):
    """Make a test mesh.

    Mesh has nodes, plus faces and/or edges, with face-coords and edge-coords,
    numbers of which can be controlled.

    Parameters
    ----------
    n_nodes : int or None
        Number of nodes in mesh. Default is 15.  Cannot be 0.
    n_edges : int or None
        Number of edges in mesh. Default is 5.
        If not 0, edge coords and an 'edge_node_connectivity' are included.
    n_faces : int or None
        Number of faces in mesh. Default is 3.
        If not 0, face coords and a 'face_node_connectivity' are included.
    lazy_values : bool, default=False
        If True, all content values of coords and connectivities are lazy.
    nodes_per_face : int or None
        Number of nodes per face. Default is 4.
    masked_connecteds : bool, default=False
        If True, mask some of the connected spaces in the connectivity indices
        arrays.

    """
    if lazy_values:
        import dask.array as da

        arr = da
    else:
        arr = np

    if n_nodes is None:
        n_nodes = _TEST_N_NODES
    if n_faces is None:
        n_faces = _TEST_N_FACES
    if n_edges is None:
        n_edges = _TEST_N_EDGES
    if nodes_per_face is None:
        nodes_per_face = _TEST_N_BOUNDS
    node_x = AuxCoord(
        1100 + arr.arange(n_nodes),
        standard_name="longitude",
        units="degrees_east",
        long_name="long-name",
        var_name="var-name",
        attributes={
            # N.B. cast this so that a save-load roundtrip preserves it
            "a": np.int64(1),
            "b": "c",
        },
    )
    node_y = AuxCoord(1200 + arr.arange(n_nodes), standard_name="latitude")

    connectivities = []
    if n_edges == 0:
        edge_coords_and_axes = None
    else:
        # Define a rather arbitrary edge-nodes connectivity.
        # Some nodes are left out, because n_edges*2 < n_nodes.
        conns = arr.arange(n_edges * 2, dtype=int)
        # Missing nodes include #0-5, because we add 5.
        conns = ((conns + 5) % n_nodes).reshape((n_edges, 2))
        if masked_connecteds:
            conns[0, -1] = -1
            conns = arr.ma.masked_less(conns, 0)
        edge_nodes = Connectivity(conns, cf_role="edge_node_connectivity")
        connectivities.append(edge_nodes)

        edge_x = AuxCoord(2100 + arr.arange(n_edges), standard_name="longitude")
        edge_y = AuxCoord(2200 + arr.arange(n_edges), standard_name="latitude")
        edge_coords_and_axes = [(edge_x, "x"), (edge_y, "y")]

    if n_faces == 0:
        face_coords_and_axes = None
    else:
        # Define a rather arbitrary face-nodes connectivity.
        # Some nodes are left out, because n_faces*n_bounds < n_nodes.
        conns = arr.arange(n_faces * nodes_per_face, dtype=int)
        conns = (conns % n_nodes).reshape((n_faces, nodes_per_face))
        if masked_connecteds:
            conns[0, -1] = -1
            conns = arr.ma.masked_less(conns, 0)
        face_nodes = Connectivity(conns, cf_role="face_node_connectivity")
        connectivities.append(face_nodes)

        # Some numbers for the edge coordinates.
        face_x = AuxCoord(3100 + arr.arange(n_faces), standard_name="longitude")
        face_y = AuxCoord(3200 + arr.arange(n_faces), standard_name="latitude")
        face_coords_and_axes = [(face_x, "x"), (face_y, "y")]

    mesh = Mesh(
        topology_dimension=2,
        node_coords_and_axes=[(node_x, "x"), (node_y, "y")],
        connectivities=connectivities,
        edge_coords_and_axes=edge_coords_and_axes,
        face_coords_and_axes=face_coords_and_axes,
    )
    return mesh


def sample_meshcoord(mesh=None, location="face", axis="x", **extra_kwargs):
    """Create a test MeshCoord.

    The creation args are defaulted, including the mesh.
    If not provided as an arg, a new mesh is created with sample_mesh().

    """
    if mesh is None:
        mesh = sample_mesh()
    result = MeshCoord(mesh=mesh, location=location, axis=axis, **extra_kwargs)
    return result


def sample_mesh_cube(nomesh_faces=None, n_z=2, with_parts=False, **meshcoord_kwargs):
    """Create a 2d test cube with 1 'normal' and 1 unstructured dimension (with a Mesh).

    Result contains : dimcoords for both dims; an auxcoord on the unstructured
    dim; 2 mesh-coords.

    By default, the mesh is provided by :func:`sample_mesh`, so coordinates
    and connectivity  are not realistic.

    Parameters
    ----------
    nomesh_faces : int or None, optional, default=None
        If set, don't add MeshCoords, so dim 1 is just a plain anonymous dim.
        Set its length to the given value.
    n_z : int, optional, default=2
        Length of the 'normal' dim.  If 0, it is *omitted*.
    with_parts : bool, optional, default=False
        If set, return all the constituent component coords
    **meshcoord_kwargs : dict, optional
        Extra controls passed to :func:`sample_meshcoord` for MeshCoord
        creation, to allow user-specified location/mesh.  The 'axis' key is
        not available, as we always add both an 'x' and 'y' MeshCOord.

    Returns
    -------
    cube
        if with_parts not set
    (cube, parts)
        if with_parts is set
        'parts' is (mesh, dim0-dimcoord, dim1-dimcoord, dim1-auxcoord,
        x-meshcoord [or None], y-meshcoord [or None]).

    """
    nomesh = nomesh_faces is not None
    if nomesh:
        mesh = None
        n_faces = nomesh_faces
    else:
        mesh = meshcoord_kwargs.pop("mesh", None)
        if mesh is None:
            mesh = sample_mesh()
        meshx, meshy = (
            sample_meshcoord(axis=axis, mesh=mesh, **meshcoord_kwargs)
            for axis in ("x", "y")
        )
        n_faces = meshx.shape[0]

    mesh_dimco = DimCoord(np.arange(n_faces), long_name="i_mesh_face", units="1")

    auxco_x = AuxCoord(np.zeros(n_faces), long_name="mesh_face_aux", units="1")

    zco = DimCoord(np.arange(n_z), long_name="level", units=1)
    cube = Cube(np.zeros((n_z, n_faces)), long_name="mesh_phenom")
    cube.add_dim_coord(zco, 0)
    if nomesh:
        mesh_coords = []
    else:
        mesh_coords = [meshx, meshy]

    cube.add_dim_coord(mesh_dimco, 1)
    for co in mesh_coords + [auxco_x]:
        cube.add_aux_coord(co, 1)

    if not with_parts:
        result = cube
    else:
        if nomesh:
            meshx, meshy = None, None
        parts = (mesh, zco, mesh_dimco, auxco_x, meshx, meshy)
        result = (cube, parts)

    return result
