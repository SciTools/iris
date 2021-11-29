# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

"""
Utility operations specific to unstructured data.

"""
from typing import AnyStr, Iterable, Union

import dask.array as da
import numpy as np

from iris.cube import Cube


def recombine_submeshes(
    mesh_cube: Cube,
    submesh_cubes: Union[Iterable[Cube], Cube],
    index_coord_name: AnyStr = "i_mesh_index",
) -> Cube:
    """
    Put data from sub-meshes back onto the original full mesh.

    The result is a cube like ``mesh_cube``, but with its data replaced by a
    combination of the data in the ``submesh_cubes``.

    Parameters
    ----------
    mesh_cube : Cube
        Describes the mesh and mesh-location onto which the all the
        ``submesh-cubes``' data are mapped, and acts as a template for the
        result.
        Must have a :class:`~iris.experimental.ugrid.mesh.Mesh`.

    submesh_cubes : iterable of Cube, or Cube
        Cubes, each with data on a _subset_ of the ``mesh_cube`` datapoints
        (within the mesh dimension).
        The submesh cubes do not need to have a mesh.
        There must be at least 1 of them, to determine the result phenomenon.
        Their metadata (names, units and attributes) must all be the same,
        _except_ that 'var_name' is ignored.
        Their dtypes must all be the same.
        Their shapes and dimension-coords must all match those of
        ``mesh_cube``, except in the mesh dimension, which can have different
        sizes between the submeshes, and from the ``mesh_cube``.
        The mesh dimension of each must have a 1-D coord named by
        ``index_coord_name``.  These "index coords" can vary in length, but
        they must all have matching metadata (units, attributes and names
        except 'var_name'), and must also match the coord of that name in
        ``mesh_cube``, if there is one.
        The ".points" values of the index coords specify, for each datapoint,
        its location in the original mesh -- i.e. they are indices into the
        relevant mesh-location dimension.

    index_coord_name : Cube
        Coord name of an index coord containing the mesh location indices, in
        every submesh cube.

    Returns
    -------
    result_cube
        A cube with the same mesh, location, and shape as ``mesh_cube``, but
        with its data replaced by that from the``submesh_cubes``.
        The result phenomeon identity is also that of the``submesh_cubes``,
        i.e. units, attributes and names (except 'var_name', which is None).

    Notes
    -----
    Where regions overlap, the result data comes from the submesh cube
    containing that location which appears _last_ in ``submesh_cubes``.

    Where no region contains a datapoint, it will be masked in the result.
    HINT: alternatively, values covered by no region can be set to the
    original 'full_mesh_cube' data value, if 'full_mesh_cube' is *also* passed
    as the first of the 'region_cubes'.

    The ``result_cube`` dtype is that of the ``submesh_cubes``.

    """
    if not submesh_cubes:
        raise ValueError("'submesh_cubes' must be non-empty.")

    mesh_dim = mesh_cube.mesh_dim()
    if mesh_dim is None:
        raise ValueError("'mesh_cube' has no \".mesh\".")

    #
    # Perform consistency checks on all the region-cubes.
    #
    if not isinstance(submesh_cubes, Iterable):
        # Treat a single submesh cube input as a list-of-1.
        submesh_cubes = [submesh_cubes]

    result_metadata = None
    result_dtype = None
    indexcoord_metadata = None
    for i_sub, cube in enumerate(submesh_cubes):
        sub_str = (
            f"Submesh cube #{i_sub + 1}/{len(submesh_cubes)}, "
            f'"{cube.name()}"'
        )

        # Check dimensionality.
        if cube.ndim != mesh_cube.ndim:
            err = (
                f"{sub_str} has {cube.ndim} dimensions, but "
                f"'mesh_cube' has {mesh_cube.ndim}."
            )
            raise ValueError(err)

        # Get cube metadata + dtype : must match, and will apply to the result
        dtype = cube.dtype
        metadata = cube.metadata._replace(var_name=None)
        if i_sub == 0:
            # Store the first-cube metadata + dtype as reference
            result_metadata = metadata
            result_dtype = dtype
        else:
            # Check subsequent region-cubes metadata + dtype against the first
            if metadata != result_metadata:
                err = (
                    f"{sub_str} has metadata {metadata}, "
                    "which does not match that of the other region_cubes, "
                    f"which is {result_metadata}."
                )
                raise ValueError(err)
            elif dtype != result_dtype:
                err = (
                    f"{sub_str} has a dtype of {dtype}, "
                    "which does not match that of the other region_cubes, "
                    f"which is {result_dtype}."
                )
                raise ValueError(err)

        # For each dim, check that coords match other regions, and full-cube
        for i_dim in range(mesh_cube.ndim):
            if i_dim == mesh_dim:
                # mesh dim : look for index coords (by name)
                full_coord = mesh_cube.coords(
                    name_or_coord=index_coord_name, dimensions=(i_dim,)
                )
                sub_coord = cube.coords(
                    name_or_coord=index_coord_name, dimensions=(i_dim,)
                )
            else:
                # non-mesh dims : look for dim-coords (only)
                full_coord = mesh_cube.coords(
                    dim_coords=True, dimensions=(i_dim,)
                )
                sub_coord = cube.coords(dim_coords=True, dimensions=(i_dim,))

            if full_coord:
                (full_coord,) = full_coord
                full_dimname = full_coord.name()
                full_metadata = full_coord.metadata._replace(var_name=None)
            if sub_coord:
                (sub_coord,) = sub_coord
                sub_dimname = sub_coord.name()
                sub_metadata = sub_coord.metadata._replace(var_name=None)

            err = None
            # N.B. checks for mesh- and non-mesh-dims are different
            if i_dim != mesh_dim:
                # i_dim == mesh_dim :  checks for non-mesh dims
                if full_coord and not sub_coord:
                    err = (
                        f"{sub_str} has no dim-coord for dimension "
                        f"{i_dim}, to match the 'mesh_cube' dimension "
                        f'"{full_dimname}".'
                    )
                elif sub_coord and not full_coord:
                    err = (
                        f'{sub_str} has a dim-coord "{sub_dimname}" for '
                        f"dimension {i_dim}, but 'mesh_cube' has none."
                    )
                elif sub_coord != full_coord:
                    err = (
                        f'{sub_str} has a dim-coord "{sub_dimname}" for '
                        f"dimension {i_dim}, which does not match that "
                        f"of 'mesh_cube', \"{full_dimname}\"."
                    )
            else:
                # i_dim == mesh_dim :  different rules for this one
                if not sub_coord:
                    # Must have an index coord on the mesh dimension
                    err = (
                        f'{sub_str} has no "{index_coord_name}" coord on '
                        f"the mesh dimension (dimension {mesh_dim})."
                    )
                elif full_coord and sub_metadata != full_metadata:
                    # May *not* have full-cube index, but if so it must match
                    err = (
                        f"{sub_str} has an index coord "
                        f'"{index_coord_name}" whose ".metadata" does not '
                        f"match that of the same name in 'mesh_cube' :  "
                        f"{sub_metadata} != {full_metadata}."
                    )
                else:
                    # At this point, we know we *have* an index coord, and it does
                    # not conflict with the one on 'mesh_cube' (if any).
                    # Now check for matches between the region cubes.
                    if indexcoord_metadata is None:
                        # Store first occurrence (from first region-cube)
                        indexcoord_metadata = sub_metadata
                    elif sub_metadata != indexcoord_metadata:
                        # Compare subsequent occurrences (from other region-cubes)
                        err = (
                            f"{sub_str} has an index coord "
                            f'"{index_coord_name}" whose ".metadata" does not '
                            f"match that of the other submesh-cubes :  "
                            f"{sub_metadata} != {indexcoord_metadata}."
                        )

            if err:
                raise ValueError(err)

    # Use the mesh_dim to transpose inputs + outputs, if required, as it is
    # simpler for all the array operations to always have the mesh dim *last*.
    if mesh_dim == mesh_cube.ndim - 1:
        # Mesh dim is already the last one : no tranpose required
        untranspose_dims = None
    else:
        dim_range = np.arange(mesh_cube.ndim, dtype=int)
        # Transpose all inputs to mesh-last order
        tranpose_dims = [i_dim for i_dim in dim_range if i_dim != mesh_dim] + [
            mesh_dim
        ]  # chop out mesh_dim + put it at the end

        def transposed_copy(cube, dim_order):
            cube = cube.copy()
            cube.transpose(dim_order)
            return cube

        mesh_cube = transposed_copy(mesh_cube, tranpose_dims)
        submesh_cubes = [
            transposed_copy(region_cube, tranpose_dims)
            for region_cube in submesh_cubes
        ]

        # Also prepare for transforming the output back to the original order
        untranspose_dims = dim_range.copy()
        # Neat trick to produce the reverse operation
        untranspose_dims[tranpose_dims] = dim_range

    #
    # Here's the core operation..
    #
    def fill_region(target, regiondata, regioninds):
        if not target.flags.writeable:
            # The initial input can be a section of a da.zeros(), which has no
            # real array "behind" it.  This means that real arrays created in
            # memory are only chunk-sized, but it also means that 'target' may
            # not be writeable.  So take a copy to fix that, where needed.
            target = target.copy()
        # N.B. Indices are basically 1D, but may have leading *1 dims for
        # alignment, to satisfy da.map_blocks
        assert all(size == 1 for size in regioninds.shape[:-1])
        inds = regioninds.flatten()
        # Assign blocks with indexing on the last dim only
        target[..., inds] = regiondata
        return target

    # Create an initially 'empty' (all-masked) dask array matching the input.
    # N.B. this does not use the mesh_cube.lazy_data() array, but only its
    # shape and dtype, since the data itself is not used in the calculation.
    # N.B. chunking matches the input cube, allowing performance control.
    input_data = mesh_cube.lazy_data()
    result_array = da.ma.masked_array(
        da.zeros(
            input_data.shape,
            dtype=result_dtype,
            chunks=input_data.chunksize,
        ),
        True,
    )

    # Wrap this repeatedly with a lazy operation to assign each region.
    # It is done this way because we couldn't get map_blocks to correctly wrap
    # a function which does all regions in a single operation.
    # TODO: replace with a single-stage solution: Probably better, if possible.
    # Notes on resultant calculation properties:
    # 1. map_blocks is chunk-mapped, so it is parallelisable and space-saving
    # 2. However, fetching less than a whole chunk is not efficient
    for cube in submesh_cubes:
        # Lazy data array from the region cube
        sub_data = cube.lazy_data()

        # Lazy indices from the mesh-dim coord
        mesh_dimcoord = cube.coord(
            name_or_coord=index_coord_name, dimensions=cube.ndim - 1
        )
        indarr = mesh_dimcoord.lazy_points()

        # Extend indarr dimensions to align it with the 'target' array dims
        assert indarr.ndim == 1
        shape = (1,) * (cube.ndim - 1) + indarr.shape
        indarr = indarr.reshape(shape)

        # Apply the operation to paste from one region into the target
        # N.B. replacing 'result_array' each time around the loop
        result_array = da.map_blocks(
            fill_region,
            result_array,
            sub_data,
            indarr,
            dtype=result_array.dtype,
            meta=np.ndarray,
        )

    # Construct the result cube
    result_cube = mesh_cube.copy()
    result_cube.data = result_array
    # Copy names, units + attributes from region data (N.B. but not var_name)
    result_cube.metadata = result_metadata
    if untranspose_dims is not None:
        # Re-order dims as in the original input
        result_cube.transpose(untranspose_dims)

    return result_cube
