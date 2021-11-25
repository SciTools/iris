# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

"""
Utility operations specific to unstructured data.

"""
from typing import AnyStr, Iterable

import dask.array as da
import numpy as np

from iris.cube import Cube


def recombine_regions(
    full_mesh_cube: Cube,
    region_cubes: Iterable[Cube],
    index_coord_name: AnyStr = "i_mesh_index",
) -> Cube:
    """
    Put data from regional sub-meshes back onto the original full mesh.

    The result is a region_cube identical to 'full_mesh_cube', but with its data
    replaced by a combination of data from the provided 'region_cubes'.
    The result metadata, including name and units, are also replaced by those
    of the 'region_cubes' (which must all be the same).

    Args:

    * full_mesh_cube
        Describes the full mesh and mesh-location to which the region data
        refers, and acts as a template for the result.
        Must have a :class:`~iris.experimental.ugrid.mesh.Mesh`.
        Its mesh dimension must have a dimension coordinate, containing a
        simple sequence of index values == "np.arange(n_mesh)".

    * region_cubes
        Contain data on a subset of the 'full_mesh_cube' mesh locations.
        The region cubes do not need to have a mesh.  There must be at least
        1 of them, to determine the result phenomenon.
        Their shapes and dimension-coords must all match those of
        'full_mesh_cube', except in the mesh dimension, which can have
        different sizes between the regions, and from the 'full_mesh_cube'.
        The mesh dimension of each region cube must have a 1-D coord named by
        'index_coord_name'.  Although these region index coords can vary in
        length, they must all have matching metadata (names, units and
        attributes), and must also match the coord of that name in the
        'full_mesh_cube', if there is one.
        The ".points" values of the region index coords specify, for each
        datapoint, its location in the original mesh -- i.e. they are indices
        into the relevant mesh-location dimension.

    * index_coord_name
        Coord name of the index coords in each region cubes, containing the
        mesh location indices.

    Result:

    * result_cube
        An unstructured region_cube identical to 'full_mesh_cube', and with the
        same mesh and location, but with its data replaced by that from the
        'region_cubes'.
        Where regions overlap, the result data comes from the last-listed of the
        original region cubes which contain that location.
        Where no region contains a datapoint, it will be masked in the result.
        HINT: alternatively, values covered by no region can be taken from the
        original 'full_mesh_cube' data, if 'full_mesh_cube' is *also* passed
        as the first of the 'region_cubes'.

    """
    if not region_cubes:
        raise ValueError("'region_cubes' must be non-empty.")

    mesh_dim = full_mesh_cube.mesh_dim()
    if mesh_dim is None:
        raise ValueError("'full_mesh_cube' has no \".mesh\".")

    # Check the basic required properties of the input.
    mesh_dim_coords = full_mesh_cube.coords(
        dim_coords=True, dimensions=(mesh_dim,)
    )
    if not mesh_dim_coords:
        err = (
            "'full_mesh_cube' has no dim-coord on the mesh dimension, "
            f"(dimension {mesh_dim})."
        )
        raise ValueError(err)

    #
    # Perform consistency checks on all the region-cubes.
    #

    def metadata_no_varname(cube_or_coord):
        # Get a metadata object but omit any var_name.
        metadata = cube_or_coord.metadata
        fields = metadata._asdict()
        fields["var_name"] = None
        result = metadata.__class__(**fields)
        return result

    n_regions = len(region_cubes)
    n_dims = full_mesh_cube.ndim
    regioncube_metadata = None
    indexcoord_metadata = None
    for i_region, region_cube in enumerate(region_cubes):
        reg_cube_str = (
            f'Region cube #{i_region}/{n_regions}, "{region_cube.name()}"'
        )
        reg_ndims = region_cube.ndim

        # Check dimensionality.
        if reg_ndims != n_dims:
            err = (
                f"{reg_cube_str} has {reg_ndims} dimensions, but "
                f"'full_mesh_cube' has {n_dims}."
            )
            raise ValueError(err)

        # Get region_cube metadata, which will apply to the result..
        region_cube_metadata = metadata_no_varname(region_cube)
        if regioncube_metadata is None:
            # Store the first region-cube metadata as a reference
            regioncube_metadata = region_cube_metadata
        elif region_cube_metadata != regioncube_metadata:
            # Check subsequent region-cubes metadata against the first.
            err = (
                f"{reg_cube_str} has metadata {region_cube_metadata}, "
                "which does not match that of the first region region_cube, "
                f'"{region_cubes[0].name()}", '
                f"which is {regioncube_metadata}."
            )
            raise ValueError(err)

        # For each dim, check that coords match other regions, and full-cube.
        for i_dim in range(full_mesh_cube.ndim):
            if i_dim == mesh_dim:
                # mesh dim : look for index coords (by name).
                fulldim = full_mesh_cube.coords(
                    name_or_coord=index_coord_name, dimensions=(i_dim,)
                )
                regdim = region_cube.coords(
                    name_or_coord=index_coord_name, dimensions=(i_dim,)
                )
            else:
                # non-mesh dims : look for dim-coords (only)
                fulldim = full_mesh_cube.coords(
                    dim_coords=True, dimensions=(i_dim,)
                )
                regdim = region_cube.coords(
                    dim_coords=True, dimensions=(i_dim,)
                )

            if fulldim:
                (fulldim,) = fulldim
                full_dimname = fulldim.name()
                fulldim_metadata = metadata_no_varname(fulldim)
            if regdim:
                (regdim,) = regdim
                reg_dimname = regdim.name()
                regdim_metadata = metadata_no_varname(regdim)

            err = None
            # N.B. checks for mesh- and non-mesh-dims are different.
            if i_dim != mesh_dim:
                # i_dim == mesh_dim :  checks for non-mesh dims.
                if fulldim and not regdim:
                    err = (
                        f"{reg_cube_str} has no dim-coord for dimension "
                        "{i_dim}, to match the 'full_mesh_cube' dimension "
                        f'"{full_dimname}".'
                    )
                elif regdim and not fulldim:
                    err = (
                        f'{reg_cube_str} has a dim-coord "{reg_dimname}" for '
                        f"dimension {i_dim}, but 'full_mesh_cube' has none."
                    )
                elif regdim != fulldim:
                    err = (
                        f'{reg_cube_str} has a dim-coord "{reg_dimname}" for '
                        f"dimension {i_dim}, which does not match that "
                        f"of 'full_mesh_cube', \"{full_dimname}\"."
                    )
            else:
                # i_dim == mesh_dim :  different rules for this one
                if not regdim:
                    # Must have an index coord on the mesh dimension
                    err = (
                        f'{reg_cube_str} has no "{index_coord_name}" coord on '
                        f"the mesh dimension (dimension {mesh_dim})."
                    )
                elif fulldim and regdim_metadata != fulldim_metadata:
                    # May *not* have full-cube index, but if so it must match
                    err = (
                        f"{reg_cube_str} has an index coord "
                        f'"{index_coord_name}" whose ".metadata" does not '
                        "match that on 'full_mesh_cube' :  "
                        f"{regdim_metadata} != {fulldim_metadata}."
                    )

                # At this point, we know we *have* an index coord, and it does not
                # conflict with the one on 'full_mesh_cube' (if any).
                # Now check for matches between the region cubes.
                if indexcoord_metadata is None:
                    # Store first occurrence (from first region-cube)
                    indexcoord_metadata = regdim_metadata
                elif regdim_metadata != indexcoord_metadata:
                    # Compare subsequent occurences (from other region-cubes)
                    err = (
                        f"{reg_cube_str} has an index coord "
                        f'"{index_coord_name}" whose ".metadata" does not '
                        f"match that of the first region-cube :  "
                        f"{regdim_metadata} != {indexcoord_metadata}."
                    )

        if err:
            raise ValueError(err)

    # Use the mesh_dim to transpose inputs + outputs, if required, as it is
    # simpler for all the array operations to always have the mesh dim *last*.
    if mesh_dim == full_mesh_cube.ndim - 1:
        # Mesh dim is already the last one : no tranposes required
        untranspose_dims = None
    else:
        dim_range = np.arange(full_mesh_cube.ndim, dtype=int)
        # Transpose all inputs to mesh-last order.
        tranpose_dims = [i_dim for i_dim in dim_range if i_dim != mesh_dim] + [
            mesh_dim
        ]  # chop out mesh_dim + put it at the end.

        def transposed_copy(cube, dim_order):
            cube = cube.copy()
            cube.transpose()
            return cube

        full_mesh_cube = transposed_copy(full_mesh_cube, tranpose_dims)
        region_cubes = [
            transposed_copy(region_cube, tranpose_dims)
            for region_cube in region_cubes
        ]

        # Also prepare for transforming the output back to the original order.
        untranspose_dims = dim_range.copy()
        # Neat trick to produce the reverse operation.
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
        # Assign blocks with indexing on the last dim only.
        target[..., inds] = regiondata
        return target

    # Create an initially 'empty' (all-masked) dask array matching the input.
    # N.B. this does not use the full_mesh_cube.lazy_data() array, but only its
    # shape and dtype, since the data itself is not used in the calculation.
    # N.B. chunking matches the input cube, allowing performance control.
    input_data = full_mesh_cube.lazy_data()
    result_array = da.ma.masked_array(
        da.zeros(
            input_data.shape,
            dtype=input_data.dtype,
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
    for region_cube in region_cubes:
        # Lazy data array from the region cube
        datarr = region_cube.lazy_data()

        # Lazy indices from the mesh-dim coord.
        mesh_dimcoord = region_cube.coord(
            name_or_coord=index_coord_name, dimensions=region_cube.ndim - 1
        )
        indarr = mesh_dimcoord.lazy_points()

        # Extend indarr dimensions to align it with the 'target' array dims.
        assert indarr.ndim == 1
        shape = (1,) * (region_cube.ndim - 1) + indarr.shape
        indarr = indarr.reshape(shape)

        # Apply the operation to paste from one region into the target.
        # N.B. replacing 'result_array' each time around the loop.
        result_array = da.map_blocks(
            fill_region,
            result_array,
            datarr,
            indarr,
            dtype=result_array.dtype,
            meta=np.ndarray,
        )

    # Construct the result cube.
    result_cube = full_mesh_cube.copy()
    result_cube.data = result_array
    # Copy names, units + attributes from region data (N.B. but not var_name)
    result_cube.metadata = regioncube_metadata
    if untranspose_dims:
        # Re-order dims as in the original input.
        result_cube.transpose(untranspose_dims)

    return result_cube
