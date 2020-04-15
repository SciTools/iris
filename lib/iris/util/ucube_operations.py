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
import matplotlib.pyplot as plt
import numpy as np

import cartopy.crs as ccrs
from iris.fileformats.ugrid_cf_reader import CubeUgrid
from iris.cube import Cube
from iris.coords import DimCoord


def ugrid_plot(cube, axes=None, set_global=True, show=True, crs_plot=None):
    """
    Plot unstructured cube.

    The last dimension must be unstructured.
    Any other dimensions are reduced by taking the first point.

    Args:

    * cube
        cube to draw.
    * axes (matplotlib.Axes or None):
        axes to draw on.  If None create one, with coastlines and gridlines.
    * set_global (bool):
        whether to call "axes.set_global()".
    * show (bool):
        whether to call "plt.show()".
    * crs_plot (cartopy.crs.Projection or None):
        If axes is None, create an axes with this projection.
        If None, a default Orthographic projection is used.

    """
    assert cube.ugrid is not None
    assert cube.ugrid.cube_dim == cube.ndim - 1

    # Select first point in any additional dimensions.
    while cube.ndim > 1:
        temp = cube[0]
        temp.ugrid = cube.ugrid
        # Note: cube indexing does not preserve grid : JUST HACK IT for now
        cube = temp

    if not axes:
        plt.figure(figsize=(12, 8))
        if crs_plot is None:
            crs_plot = ccrs.Orthographic(
                central_longitude=-27, central_latitude=27.0
            )
            # Force fine drawing of curved lines.
            crs_plot._threshold *= 0.01
        axes = plt.axes(projection=crs_plot)
        axes.coastlines()
        axes.gridlines()
    if set_global:
        axes.set_global()
    assert cube.ndim == 1
    assert cube.ugrid is not None
    ug = cube.ugrid.grid
    data = cube.data
    elem_type = cube.ugrid.mesh_location  # E.G. face
    crs_cube = ccrs.Geodetic()
    if elem_type == "node":
        xx = ug.node_lon
        yy = ug.node_lat
        plt.scatter(xx, yy, c=data, transform=crs_cube)
    elif elem_type == "edge":
        # Don't bother with this, for now.
        raise ValueError("No edge plots yet.")
    elif elem_type == "face":
        for i_face in range(cube.shape[0]):
            i_nodes = ug.faces[i_face]
            xx = ug.node_lon[i_nodes]
            yy = ug.node_lat[i_nodes]
            plt.fill(xx, yy, data[i_face], transform=crs_cube)

    if show:
        plt.show()


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

        The painful panel-to-panel connectivity stitching is based on the
        analysis of the "C4 cube" by @hdyson.
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


class PseudoshapedCubeIndexer:
    """
    Indexable object to provide a syntax for a "pseudocube slicing" operation.

    Wraps up a cube with a related 'structure shape'.
    When you index it, it returns a derived 'subset cube' with a subset mesh.

    This is an alternative to having a "pseudo-structured cube" with multiple
    dimensions in its mesh, as we haven't yet defined such a thing.
    See 'pseudo_cube' function above for something more like that, but which
    returns an "ordinary" (i.e. not unstructured) cube.

    .. for example:

        >>> print(cube)
        sample_data / (1)                   (*-- : 96)
             ugrid information:
                  topology.face                  x
                  topology_dimension: 2
                  node_coordinates: latitude longitude

        >>> cubesphere_shape = identify_cubesphere(cube.ugrid.grid)
        >>> print(cubesphere_shape)
        (6, 4, 4)

        >>> cs_indexer = PseudoshapedCubeIndexer(cube, cubesphere_shape)
        >>> face_cube = cs_indexer[0]
        >>> print(face_cube)
        sample_data / (1)                   (*-- : 16)
             ugrid information:
                  mesh.face                      x
                  topology_dimension: 2
                  node_coordinates: latitude longitude

    """

    def __init__(self, cube, shape):
        self.cube = cube
        self.shape = shape

    def __getitem__(self, keys):
        # Return a subset cube
        n_elems = np.prod(self.shape)
        all_elem_numbers = np.arange(n_elems).reshape(self.shape)
        reqd_elem_inds = list(all_elem_numbers[keys].flatten())
        return ucube_subset(self.cube, reqd_elem_inds)


def coords_within_regions(coords_array, regions_array_xy0_xy1):
    """
    Check that a given array of coordinates is within a given array of
    bounded regions.

    Args:
        * coords_array:
            The array of coordinate(s) to be checked.
        * regions_array_xy0_xy1:
            The array of region(s) the coordinates must lie within. Each region
            is formatted as a quad of bounds; x, y lower then x, y upper. If a
            bound value is None, the region is considered 'open-ended' in that
            direction (useful in defining a band for example).

    Returns:
        * result:
            A boolean array indicating which coordinates are within at least
            one of the regions.

    """

    def standardise_array(array, final_dim_len):
        """
        Ensure array is in correct format for checking coords are in regions.

        Args:
            * array:
                The array to be checked.
            * final_dim_len:
                The expected length of the array's final dimension e.g. an
                array of xy coordinates should == 2.

        Returns:
            * result (`numpy.ndarray`):
                A numpy array of exactly two dimensions, where the final
                dimension's length is equal to the final_dim_len argument.

        """
        array = np.array(array)
        assert array.ndim in (1, 2)
        assert array.shape[-1] == final_dim_len
        if array.ndim == 1:
            array = np.expand_dims(array, axis=0)

        return array

    coords_array = standardise_array(coords_array, 2)
    coords_list = []
    # Get x coords, normalise to -180..+180 .
    coords_list.append((coords_array[..., 0] + 360.0 + 180.0) % 360.0 - 180.0)
    # Get y coords
    coords_list.append(coords_array[..., 1])
    coord_mins = [min(c) for c in coords_list]
    coord_maxs = [max(c) for c in coords_list]

    regions_array_xy0_xy1 = standardise_array(regions_array_xy0_xy1, 4)
    # List arrays of: x_lowers, y_lowers, x_uppers, y_uppers
    bounds_list = list(np.swapaxes(regions_array_xy0_xy1, axis1=0, axis2=1))

    # Add axes help broadcasting in downstream checks.
    coords_list = [np.expand_dims(c, axis=0) for c in coords_list]
    bounds_list = [np.expand_dims(b, axis=1) for b in bounds_list]

    # Check that x and y are either higher or lower than the relevant bounds,
    # allowing for any of the bounds to be 'skipped' if set to None (e.g. if
    # the user wants to constrain on just one axis).
    checks_list = []
    for i in (0, 1):  # The lower bounds.
        bounds_array = bounds_list[i]
        # Replace None with a small number - guaranteeing True.
        bounds_array = np.where(
            np.equal(bounds_array, [None]), coord_mins[i], bounds_array
        )
        checks_list.append(coords_list[i] >= bounds_array)
    for i in (2, 3):  # The upper bounds.
        # coord list half length of bounds list so % 2 for index.
        coord_ix = i % 2
        bounds_array = bounds_list[i]
        # Replace None with a large number - guaranteeing True.
        bounds_array = np.where(
            np.equal(bounds_array, [None]), coord_maxs[coord_ix], bounds_array
        )
        checks_list.append(coords_list[coord_ix] <= bounds_array)

    # Collapse the individual bounds checks into sub-arrays of whether each
    # coord is in each region.
    coords_within_each = np.logical_and.reduce(checks_list)
    # Collapse the region checks to give an array of whether each coord is
    # in any of the regions.
    coords_within_any = np.logical_or.reduce(coords_within_each)

    return coords_within_any


def xy_region_extract(cube, regions_xy0_xy1, slice_type="enclose"):
    """
    Extract regions from a cube with an unstructured dimension. Based on the
    mesh's data locations (faces/edges/nodes).

    Args:

    * cube (`iris.cube.Cube`):
        The cube that provides the data locations to be regionally constrained.
    * regions_xy0_xy1
        A list/array of regions the data locations must lie within. See
        :meth: utils.ucube_operations.coords_within_regions for more detail.
    * slice_type
        * enclose
            data locations must fall entirely within at least one of the
            region(s). I.e. all of the associated nodes are within.
        * intersect
            data locations must fall at least partially within at least
            one of the region(s). I.e. >0 of the associated nodes are within.
        * centre
            the data locations' centres must fall within at least one of the
            region(s). Appropriate when data is located on nodes.

    Returns:

    * result (`iris.cube.Cube`):
        An unstructured cube with only data for locations that are within
        the input region(s).

    """
    ug = cube.ugrid.grid
    element_type = cube.ugrid.mesh_location
    if element_type not in ("node", "edge", "face"):
        msg = "Unsupported data location: {}"
        raise ValueError(msg.format(element_type))

    if slice_type == "centre":
        # Alternative behaviours for filtering the element coords by the
        # bounding region(s).
        if element_type == "node":
            coords_to_check = ug.nodes
        elif element_type == "edge":
            if ug.edge_coordinates is None:
                ug.build_edge_coordinates()
            coords_to_check = ug.edge_coordinates
        elif element_type == "face":
            if ug.face_coordinates is None:
                ug.build_face_coordinates()
            coords_to_check = ug.face_coordinates
        elements_wanted = coords_within_regions(
            coords_to_check, regions_xy0_xy1
        )

    elif slice_type in ("intersect", "enclose"):
        if element_type == "node":
            msg = "slice_type '{}' is inappropriate when data is located on nodes."
            raise ValueError(msg.format(slice_type))
        # Get the full list of nodes that are within the region(s).
        nodes_within = coords_within_regions(ug.nodes, regions_xy0_xy1)
        if element_type == "edge":
            element_nodes = ug.edges
        elif element_type == "face":
            element_nodes = ug.faces
        # Index those nodes within the region(s) to find which ones are part of
        # the elements.
        element_nodes_within = nodes_within[element_nodes]
        # Alternative behaviours for filtering the elements by the bounding
        # region(s).
        if slice_type == "intersect":
            agg_method = np.any
        else:
            agg_method = np.all
        elements_wanted = agg_method(element_nodes_within, axis=1)

    else:
        msg = "Unsupported slice_type: {}"
        raise ValueError(msg.format(slice_type))

    # Return a cube subset based on the selected elements.
    region_cube = ucube_subset(cube, elements_wanted)
    return region_cube
