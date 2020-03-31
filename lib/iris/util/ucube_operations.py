# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Miscellaneous Functions for working with unstructured cubes.

That is, cubes where 'cube.ugrid' is not None, in which case it is a
:class:`~iris.fileformats.ugrid_cf_reader.CubeUgrid`, describing an
unstructured mesh.

"""
import math

from gridded.pyugrid.ugrid import UGrid
import numpy as np

from iris.fileformats.ugrid_cf_reader import CubeUgrid
from iris.cube import Cube
from iris.coords import DimCoord


def remap_element_numbers(elems, indices, n_old_elements=None):
    """
    Calculate the elements array resulting from an element selection operation.

    This means that the 'old element numbers' in the array are replaced with
    'new element numbers'.  The new elements are those selected by applying the
    'indices' to the original elements array.

    Where an 'old element number' is not in those selected by 'indices', it
    must be replaced by a -1 ("missing") in the output.

    Args:

    * elems (array of int):
        An array of element numbers.

    * indices (int, slice or sequence):
        Any valid 1-D array-indexing operation.

    * n_old_elements (int or None):
        The number of 'old' elements, i.e. maximum valid element index + 1.
        If not given, this defaults to max(np.max(elems), np.max(indices)) + 1.
        However, this would not work correctly if 'indices' is a slice
        operation involving negative indices, e.g. slice(0, -2).
        In such cases, an exception will be raised.

    Result:

    * new_elems (array of int):
        An array of the same size and dtype as the input, with 'old' element
        numbers replaced by 'new'.

    .. TODO:

        The missing value may in fact *not* always be -1.
        Just ignore that, for now.

    """
    if n_old_elements is None:
        if (
            isinstance(indices, slice)
            and indices.start < 0
            or indices.stop < 0
        ):
            msg = (
                '"indices" is {}, which uses negative indices. '
                'This is invalid when "n_old_elements" is not given.'
            )
            raise ValueError(msg.format(indices))
        n_old_elements = max(np.max(indices), np.max(elems)) + 1
    old_face_numbers = np.arange(n_old_elements)[indices]
    n_new_elems = old_face_numbers.shape[0]
    new_face_numbers = np.arange(n_new_elems, dtype=int)
    old_to_new_face_numbers = np.full((n_old_elements,), -1, dtype=int)
    # N.B. "-1" means a "missing" neighbour.
    old_to_new_face_numbers[indices] = new_face_numbers
    # Remap elems through this, so each face link gets its equivalent
    # 'new number' : N.B. some of which are now 'missing'.
    elems = old_to_new_face_numbers[elems]
    return elems


def ugrid_subset(grid, indices, mesh_location="face"):
    """
    Make a subset extraction of a grid object.

    Args:

    * grid (gridded.pyugrid.UGrid):
        input grid.

    * indices (1-dimensional array-like of int or bool, or slices):
        A numpy indexing key into a 1-dimensional array.
        Makes a pointwise selection from the selected element type.
        If boolean, must match the number of the relevant elements, otherwise
        a list of the indices to keep.

    * mesh_location (str):
        Which type of grid element to select on.
        One of 'face', 'edge', 'node' : 'volume' is not supported.

    returns:

        * new_grid (gridded.pyugrid.UGrid):
          A new grid object with only the selected elements.

          *  If mesh_location is 'nodes', the result has no edges or faces.

          * If mesh_location is 'faces', the result has the same nodes as the
            input, but no edges.

          * If mesh_location is 'edges', the result has the same nodes as the
            input, but no faces.

          Other non-essential information will be either similarly index-selected
          or discard

    """
    # All elements of UGRid:
    #     nodes,
    #     edges=None,
    #     boundaries=None,
    #     face_face_connectivity=None,
    #     face_edge_connectivity=None,
    #     edge_coordinates=None,
    #     face_coordinates=None,
    #     boundary_coordinates=None,

    if mesh_location == "node":
        # Just take the relevant node info.
        # Result has no edges, faces or boundaries.
        result = UGrid(nodes=grid.nodes[indices])
        # I *think* this is still valid in "gridded".
        # Although it only handles meshes with a nominal
        # "mesh.topology_dimension = 2"
        #   see : https://github.com/NOAA-ORR-ERD/gridded/blob/v0.2.5/gridded/pyugrid/ugrid.py#L974
        # However, it can+will *save* a grid with no faces.
        #   see : https://github.com/NOAA-ORR-ERD/gridded/blob/v0.2.5/gridded/pyugrid/ugrid.py#L1001
    elif mesh_location == "edge":
        # Take selected edges + all original nodes.
        # Result has no faces.  Boundaries unaffected.
        result = UGrid(nodes=grid.nodes, edges=grid.edges[indices])
        # Reattach other mesh info, indexing on edge dimension as needed.
        # NOT appropriate :
        #     faces
        #     face_face_connectivity
        #     face_edge_connectivity (*why* it's simplest to remove faces?)
        #     face_coordinates
        # IS appropriate :
        #     edges (new)
        #     edge_coordinates
        #     boundaries
        #     boundary_coordinates
        #
        # QUESTION : we don't *have to* discard all the face information,
        #   (because faces don't depend on edges at all)
        #   For now, it certainly seems the simplest approach.
        #   It's probably appropriate if the selected element is the one
        #   relevant to our cube data, in practice.
        #   It would also assist "minimising" (i.e. pruning unused
        #   sub-elements), which we might also want to do.
        #   SO... is this convenient, is it really best ??
        #   If we don't, we can retain 'face_edge_connectivity', and
        #   'edge_face_connectivity, but update all the edge numbers
        #   (as for 'f2f' below).
        #
        if grid.edge_coordinates is not None:
            result.edge_coordinates = grid.edge_coordinates[indices]
        #
        # QUESTION : it seems that UGrid doesn't have 'edge_edge_connectivity',
        #   (unlike face-face)
        #   If it did, we could do similar to the 'f2f' remap below..
        #
        result.boundaries = grid.boundaries
        result.boundary_coordinates = grid.boundary_coordinates
    elif mesh_location == "face":
        # Take relevant faces + copy original nodes.
        # Result has no edges.  Boundaries unaffected.
        result = UGrid(nodes=grid.nodes, faces=grid.faces[indices])
        # Reattach other mesh info, indexing on edge dimension as needed.
        # NOT appropriate :
        #     edges
        #     edge_coordinates
        #     face_edge_connectivity (*why* it's simplest to remove edges?)
        # IS appropriate:
        #     faces (new)
        #     face_coordinates
        #     face_face_connectivity  (tricky but logical: see below)
        #     boundaries
        #     boundary_coordinates
        #
        # QUESTION : as with 'edges' above, we don't *have to* discard all the
        #   edges here (because edges obviously don't depend on faces)
        #   If we don't, we can retain 'face_edge_connectivity', and
        #   'edge_face_connectivity, but update all the face numbers
        #   (as for 'f2f' below).
        #
        f2f = grid.face_face_connectivity
        if f2f is not None:
            # Reduce to only the wanted parts of the f2f array.
            f2f = f2f[indices]
            # Replace all face linkage numbers with the "new numbers",
            # -- which includes setting any lost faces to 'missing'.
            f2f = remap_element_numbers(
                f2f, indices, n_old_elements=grid.faces.shape[0]
            )
            # This is the result face-to-face array.
            result.face_face_connectivity = f2f
        if grid.face_coordinates is not None:
            result.face_coordinates = grid.face_coordinates[indices]
        result.boundaries = grid.boundaries
        result.boundary_coordinates = grid.boundary_coordinates
    else:
        raise ValueError("")

    return result


def ucube_subset(cube, indices):
    """
    Select points from an unstructured cube in the unstructured dimension.

    Args:

    * cube (iris.cube.Cube):
        input cube, which must have an unstructured dimension.

    * indices (1-dimensional array-like of int or bool):
        A pointwise selection from the unstructured dimension.
        If boolean, must match the number of the relevant elements, otherwise
        a list of the indices to keep.

    returns:

        * new_cube (iris.cube.Cube):
          A new cube on a reduced grid, containing only the selected elements.
          `new_cube.ugrid` is a reduced grid, as described for "ugrid_subset".

    """
    if cube.ugrid is None:
        raise ValueError("Cube is not unstructured : cannot ucube_subset.")

    # Get the unstructured dim.
    i_unstruct_dim = cube.ugrid.cube_dim

    # Apply the selection indices along that dim.
    inds = tuple(
        indices if i_dim == i_unstruct_dim else slice(None)
        for i_dim in range(cube.ndim)
    )
    result = cube[inds]

    # Re-attach a derived ugrid object.
    result.ugrid = CubeUgrid(
        cube_dim=i_unstruct_dim,
        grid=ugrid_subset(cube.ugrid.grid, indices, cube.ugrid.mesh_location),
        mesh_location=cube.ugrid.mesh_location,
        topology_dimension=cube.ugrid.topology_dimension,
        node_coordinates=cube.ugrid.node_coordinates,
    )

    return result


def identify_cubesphere(grid):
    """
    Determine the cubesphere structure in an unstructured grid.

    Uses connectivity information to check if it looks like a cubesphere, and
    if so returns an equivalent shape for viewing the data as a cubesphere
    structure.

    Args :

    * grid (gridded.pyugrid.UGrid):
        ugrid representation for the grid.

    Returns :

    * cubesphere_shape (tuple of int, or None):
        an array shape tuple, such that reshaping the unstructured dimension
        into this shape lets you index the data by a 'standard' cubesphere
        indexing scheme, as [i_face, face_iy, face_ix].

    .. Note:

        We already observed that this doesn't work for more complex LFRic
        output files (e.g. "aquaplanet" example).
        The current implementation in terms of the "my_neighbours" function
        is creaky anyway :  This probably could/should have been replaced by a
        check that each face has expected "rectangular connectivity".  That is
        in fact easier to write but, knowing what we do, let's now just not
        bother.

    """
    not_failed = grid.faces is not None
    if not_failed:
        not_failed = grid.num_vertices == 4
    if not_failed:
        size = grid.faces.shape[0]
        # Check length divides exactly by 6
        not_failed = size % 6 == 0
    if not_failed:
        # Check face-size is a square.
        size = size // 6
        side = round(math.sqrt(size * 1.0))
        if size != (side * side):
            not_failed = False
        shape = (6, side, side)
    if not_failed:
        # Check connectivity as expected within each face.
        if grid.face_face_connectivity is None:
            grid.build_face_face_connectivity()
        ff = grid.face_face_connectivity
        ff2 = my_neighbours(side)
        # Check that the 4 neighbours of each face are the expected
        # numbers, but not necessarily in the same order, because working
        # out that order was too hard (!)
        ff1 = ff.copy()  # Because sorting messes with the original.
        ff1 = ff1.reshape((6, side, side, 4))  # To match structured shape
        ff1.sort(axis=-1)
        ff2.sort(axis=-1)
        not_failed = np.all(ff1 == ff2)

    if not_failed:
        result = shape
    else:
        result = None
    return result


def my_neighbours(n_side):
    """
    Construct a 'standard' map of face neighbours for a cubesphere.

    We assume a given face numbering, which seems to match LFRic.
    We construct 4 neighbours for each in "some way".  This should match the
    UGRID face_face_connectivity array, except that the ordering of
    neighbours is a bit peculiar, so we don't replicate that.

    ( We will then be able to compare an actual grid face_face_connectivity
    array with this, up-to the neighbour ordering )

    .. Note:

        As remarked above, this approach is now considered obsolete.

    """
    shape = (6, n_side, n_side)
    # Produce an array of face numbers in a 'standard order'.
    face_nums = np.arange(np.prod(shape)).reshape(shape)

    # Make an adjacent-faces map, which has 1 extra cell all around each face.
    # Each point in the central n*n of each face of this corresponds to a cube
    # face :  The four adjacent cells to this then give us its four
    # neighbouring faces.
    # For this, we need to fill in the edges of the array with the face numbers
    # from the adjacent faces.
    # There is no particularly neat way of doing this,
    # as far as I know.
    af = np.zeros((6, n_side + 2, n_side + 2))

    # Pre-fill all with -1 so we can check we've done it all.
    af[...] = -1

    # Fill the 4 corners of each af 'face+' with -2.
    # We will never use these parts, as they are not 4-connected to the central
    # area.
    for ix in (0, -1):
        for iy in (0, -1):
            af[:, ix, iy] = -2

    # Short name for face_nums.
    fn = face_nums

    # Fill in the central region of each 'face+' with the numbers of the
    # matching face : AKA the easy bit !
    for i_face in range(6):
        af[i_face, 1:-1, 1:-1] = fn[i_face]

    # Around the equator, fill in all left- and right-hand margins.
    for i_face in range(4):
        i_lhs = (i_face - 1) % 4
        i_rhs = (i_face + 1) % 4
        af[i_face, 1:-1, 0] = fn[i_lhs, :, -1]
        af[i_face, 1:-1, -1] = fn[i_rhs, :, 0]

    # Now fill the edges of the array with face numbers from adjacent faces
    # There is no particularly neat way of doing this, as far as I know.
    # We follow Harold Dyson's famous C4 cubesphere connectivity diagram, and
    # use special asserts to check known values in that n_side=4 case.

    # Plug together edges adjacent to the NORTH face.

    # Faces 64..67 = fn[4,0,:] <-above/below-> Faces 51..48 = fn[3, 0, ::-1]
    if n_side == 4:
        assert np.all(fn[4, 0, :] == [64, 65, 66, 67])
        assert np.all(fn[3, 0, ::-1] == [51, 50, 49, 48])
    assert np.all(af[3, 0, 1:-1] == -1)
    af[3, 0, 1:-1] = fn[4, 0, ::-1]
    assert np.all(af[4, 0, 1:-1] == -1)
    af[4, 0, 1:-1] = fn[3, 0, ::-1]

    # Faces 64,68,72,76 = fn[4,:,0] <-above/left-> 0..3 = fn[0, 0, :]
    if n_side == 4:
        assert np.all(fn[4, :, 0] == [64, 68, 72, 76])
        assert np.all(fn[0, 0, :] == [0, 1, 2, 3])
    assert np.all(af[0, 0, 1:-1] == -1)
    af[0, 0, 1:-1] = fn[4, :, 0]
    assert np.all(af[4, 1:-1, 0] == -1)
    af[4, 1:-1, 0] = fn[0, 0, :]

    # Faces 67,71,75,79 = fn[4,:,-1] <-above/right-> 35,34,33,32 = fn[2,0,::-1]
    if n_side == 4:
        assert np.all(fn[4, :, -1] == [67, 71, 75, 79])
        assert np.all(fn[2, 0, ::-1] == [35, 34, 33, 32])
    assert np.all(af[2, 0, 1:-1] == -1)
    af[2, 0, 1:-1] = fn[4, ::-1, -1]
    assert np.all(af[4, 1:-1, -1] == -1)
    af[4, 1:-1, -1] = fn[2, 0, ::-1]

    # Faces 76..79 = fn[4,-1,:] <-above/below-> 16..19 = fn[1,0,:]
    if n_side == 4:
        assert np.all(fn[4, -1, :] == [76, 77, 78, 79])
        assert np.all(fn[1, 0, :] == [16, 17, 18, 19])
    assert np.all(af[1, 0, 1:-1] == -1)
    af[1, 0, 1:-1] = fn[4, -1, :]
    assert np.all(af[4, -1, 1:-1] == -1)
    af[4, -1, 1:-1] = fn[1, 0, :]

    # Plug together edges adjacent to the SOUTH face.

    # Faces 80..83 = fn[5, 0,:] <-above/below-> Faces 28..31 = fn[1, -1, :]
    if n_side == 4:
        assert np.all(fn[5, 0, :] == [80, 81, 82, 83])
        assert np.all(fn[1, -1, :] == [28, 29, 30, 31])
    assert np.all(af[1, -1, 1:-1] == -1)
    af[1, -1, 1:-1] = fn[5, 0, :]
    assert np.all(af[5, 0, 1:-1] == -1)
    af[5, 0, 1:-1] = fn[1, -1, :]

    # Faces 92..95 = fn[5,-1,:] <-above/above-> Faces 63,62,61,60 = fn[3, -1, ::-1]
    if n_side == 4:
        assert np.all(fn[5, -1, :] == [92, 93, 94, 95])
        assert np.all(fn[3, -1, ::-1] == [63, 62, 61, 60])
    assert np.all(af[3, -1, 1:-1] == -1)
    af[3, -1, 1:-1] = fn[5, -1, ::-1]
    assert np.all(af[5, -1, 1:-1] == -1)
    af[5, -1, 1:-1] = fn[3, -1, ::-1]

    # Faces 80,84,88,92 = fn[5, :, 0] <--> Faces 15,14,13,12 = fn[0, -1, ::-1]
    if n_side == 4:
        assert np.all(fn[5, :, 0] == [80, 84, 88, 92])
        assert np.all(fn[0, -1, ::-1] == [15, 14, 13, 12])
    assert np.all(af[0, -1, 1:-1] == -1)
    af[0, -1, 1:-1] = fn[5, ::-1, 0]
    assert np.all(af[5, 1:-1, 0] == -1)
    af[5, 1:-1, 0] = fn[0, -1, ::-1]

    # Faces 83,87,91,95 = fn[5, :, -1] <--> Faces 44,45,46,47 = fn[2, -1, :]
    if n_side == 4:
        assert np.all(fn[5, :, -1] == [83, 87, 91, 95])
        assert np.all(fn[2, -1, :] == [44, 45, 46, 47])
    assert np.all(af[2, -1, 1:-1] == -1)
    af[2, -1, 1:-1] = fn[5, :, -1]
    assert np.all(af[5, 1:-1, -1] == -1)
    af[5, 1:-1, -1] = fn[2, -1, :]

    # Just check that we still left all corners untouched, and all other points
    # look valid.
    assert np.all(af[:, [0, 0, -1, -1], [0, -1, 0, -1]] == -2)
    af[:, [0, 0, -1, -1], [0, -1, 0, -1]] = 0
    assert af.min() == 0
    # Put corners back as a guard against using them (they should not appear
    # in the output)
    af[:, [0, 0, -1, -1], [0, -1, 0, -1]] = -2

    # Extract the 4 neighbours of each face, and combine these into a
    # connectivity array = (6, n, n, 4).
    conns_left = af[:, 1:-1, 0:-2]
    conns_right = af[:, 1:-1, 2:]
    conns_down = af[:, 0:-2, 1:-1]
    conns_up = af[:, 2:, 1:-1]
    conns_4 = np.stack((conns_down, conns_right, conns_up, conns_left))
    conns_4 = conns_4.transpose((1, 2, 3, 0))
    assert conns_4.shape == (6, n_side, n_side, 4)

    # Check we didn't pick up any corner points my mistake.
    assert conns_4.min() == 0
    return conns_4


def pseudo_cube(cube, shape, new_dim_names=None):
    """
    Create a pseudo-cube from an unstructured cube, by replacing the
    unstructured dimension with a given multi-dimensional shape.

    Note: not very complete, for lack of a cube.transpose()
    Note: sadly, the result is no longer unstructured.  That would require
        some kind of extension to `gridded`, or special support within the
        unstructured cube.

    .. TODO:

        Should we ever need this to return "structured cubes", requires
        considerable re-think.

    """
    if cube.ugrid is None:
        raise ValueError("Cube is not unstructured : cannot make pseudo-cube.")

    # Get the unstructured dim.
    i_unstruct_dim = cube.ugrid.cube_dim

    # Default names for new dims.
    if new_dim_names is not None:
        if len(new_dim_names) != len(shape):
            msg = (
                "Number of dim names is len({}) = {}, "
                "does not match length of shape = len({}) = {}."
            )
            raise ValueError(
                msg.format(
                    new_dim_names, len(new_dim_names), shape, len(shape)
                )
            )
    else:
        new_dim_names = [
            "Dim_{:d}".format(i_dim) for i_dim in range(len(shape))
        ]

    n_shape_size = np.prod(shape)
    n_cube_unstruct = cube.shape[i_unstruct_dim]
    if n_cube_unstruct != n_shape_size:
        msg = (
            "Pseudo-shape {} is {} points, "
            "does not match Cube unstructured size = {}."
        )
        raise ValueError(msg.format(cube.shape, n_shape_size, n_cube_unstruct))

    # What we really want here is a cube.reshape(), but this is too much work
    # for now.
    # As an over-simplified placeholder, make a cube of the right shape and
    # re-attach any coords not mapping to the unstructured dimension.
    # Also, to be simpler, first move the unstructured dim to the front...

    # Make list of dims with unstructured moved to front.
    new_dims = [i_unstruct_dim] + [
        i_dim for i_dim in range(cube.ndim) if i_dim != i_unstruct_dim
    ]
    # Copy cube + transpose.
    cube = cube.copy()  # Because cube.transpose is in-place (!yuck, yuck!)
    cube.transpose(new_dims)

    # Create data with new dims by reshaping.
    data = cube.core_data()
    new_shape = list(shape) + list(data.shape)[1:]
    data = data.reshape(new_shape)
    result = Cube(data)
    result.metadata = cube.metadata

    # Attach duplicate dim-coords.
    i_dim_offset = len(shape) - 1
    derived_names = [fact.name() for fact in cube.aux_factories]
    for select_dim_coords in (True, False):
        coords = cube.coords(dim_coords=select_dim_coords)
        if not select_dim_coords:
            # Don't migrate any aux factories -- too tricky for first cut!
            # TODO: fix
            coords = [co for co in coords if co.name() not in derived_names]
        for coord in coords:
            coord_dims = cube.coord_dims(coord)
            if 0 in coord_dims:
                # Can't handle coords that map the unstructured dim.
                continue
            coord_dims = [i_dim + i_dim_offset for i_dim in list(coord_dims)]
            if select_dim_coords:
                result.add_dim_coord(coord.copy(), coord_dims)
            else:
                result.add_aux_coord(coord.copy(), coord_dims)

    # Add identifying DimCoords as labels for the new dimensions.
    for i_dim, (dim_name, dim_size) in enumerate(zip(new_dim_names, shape)):
        coord = DimCoord(np.arange(dim_size), long_name=dim_name)
        result.add_dim_coord(coord, (i_dim))

    return result
