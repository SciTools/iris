# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Routines for putting data on new strata (aka. isosurfaces), often in the
Z direction.

"""

from functools import partial

import numpy as np
import stratify

from iris.coords import AuxCoord, Coord, DimCoord
from iris.cube import Cube


def _copy_coords_without_z_dim(src, tgt, z_dim):
    """
    Helper function to copy across non z-dimenson coordinates between cubes.

    Parameters
    ----------
    src : :class:`~iris.cube.Cube`
        Incoming cube containing the coordinates to be copied from.

    tgt : :class:`~iris.cube.Cube`
        Outgoing cube for the coordinates to be copied to.

    z_dim : int
        Dimension within the `src` cube that is the z-dimension.
        This dimension will not be copied. For example, the incoming
        z-dimension cube has model level_height, whilst the outgoing
        z-dimension cube has pressure.

    """
    # Copy across non z-dimension coordinates.
    for coord in src.dim_coords:
        [dim] = src.coord_dims(coord)
        if dim != z_dim:
            tgt.add_dim_coord(coord.copy(), dim)

    for coord in src.aux_coords:
        dims = src.coord_dims(coord)
        if z_dim not in dims:
            tgt.add_aux_coord(coord.copy(), dims)

    for coord in src.derived_coords:
        dims = src.coord_dims(coord)
        if z_dim not in dims:
            tgt.add_aux_coord(coord.copy(), dims)


def relevel(cube, src_levels, tgt_levels, axis=None, interpolator=None):
    """
    Interpolate the cube onto the specified target levels, given the
    source levels of the cube.

    For example, suppose we have two datasets `P(i,j,k)` and `H(i,j,k)`
    and we want `P(i,j,H)`. We call :func:`relevel` with `cube=P`,
    `src_levels=H` and `tgt_levels` being an array of the values of `H`
    we would like.

    This routine is especially useful for computing isosurfaces of phenomenon
    that are generally monotonic in the direction of interpolation, such as
    height/pressure or salinity/depth.

    Args:

    cube : :class:`~iris.cube.Cube`
        The phenomenon data to be re-levelled.

    src_levels : :class:`~iris.cube.Cube`, :class:`~iris.coord.Coord` or string
        Describes the source levels of the `cube` that will be interpolated
        over. The `src_levels` must be in the same system as the `tgt_levels`.
        The dimensions of `src_levels` must be broadcastable to the dimensions
        of the `cube`.
        Note that, the coordinate name containing the source levels in the
        `cube` may be provided.

    tgt_levels : array-like
        Describes the target levels of the `cube` to be interpolated to. The
        `tgt_levels` must be in the same system as the `src_levels`. The
        dimensions of the `tgt_levels` must be broadcastable to the dimensions
        of the `cube`, except in the nominated axis of interpolation.

    axis : int, :class:`~iris.coords.Coord` or string
        The axis of interpolation. Defaults to the first dimension of the
        `cube`, which is typically the z-dimension. Note that, the coordinate
        name specifying the z-dimension of the `cube` may be provided.

    interpolator : callable or None
        The interpolator to use when computing the interpolation. The function
        will be passed the following positional arguments::

            (tgt-data, src-data, cube-data, axis-of-interpolation)

        If the interpolator is None, :func:`stratify.interpolate` will be used
        with linear interpolation and NaN extrapolation.

        An example of constructing an alternative interpolation scheme::

            from functools import partial
            interpolator = partial(stratify.interpolate,
                                   interpolation=stratify.INTERPOLATE_NEAREST,
                                   extrapolation=stratify.EXTRAPOLATE_LINEAR)

    """
    # Identify the z-coordinate within the phenomenon cube.
    if axis is None:
        axis = 0

    if isinstance(axis, (str, Coord)):
        [axis] = cube.coord_dims(axis)

    # Get the source level data.
    if isinstance(src_levels, str):
        src_data = cube.coord(src_levels).points
    elif isinstance(src_levels, Coord):
        src_data = src_levels.points
    else:
        src_data = src_levels.data

    # The dimensions of cube and src_data must be broadcastable.
    try:
        cube_data, src_data = np.broadcast_arrays(cube.data, src_data)
    except ValueError:
        emsg = (
            "Cannot broadcast the cube and src_levels with "
            "shapes {} and {}."
        )
        raise ValueError(emsg.format(cube.shape, src_data.shape))

    tgt_levels = np.asarray(tgt_levels)
    tgt_aux_dims = axis
    if tgt_levels.ndim != 1:
        # The dimensions of tgt_levels must be broadcastable to cube
        # in everything but the interpolation axis - otherwise raise
        # an exception.
        dim_delta = cube_data.ndim - tgt_levels.ndim
        # The axis is relative to the cube. Calculate the axis of
        # interplation relative to the tgt_levels.
        tgt_axis = axis - dim_delta
        # Calculate the cube shape without the axis of interpolation.
        data_shape = list(cube_data.shape)
        data_shape.pop(axis)
        # Calculate the tgt_levels shape without the axis of interpolation.
        target_shape = list(tgt_levels.shape)
        target_shape.pop(tgt_axis)
        # Now ensure that the shapes are broadcastable.
        try:
            np.broadcast_arrays(np.empty(data_shape), np.empty(target_shape))
        except ValueError:
            emsg = (
                "Cannot broadcast the cube and tgt_levels with "
                "shapes {} and {}, whilst ignoring axis of interpolation."
            )
            raise ValueError(emsg.format(cube_data.shape, tgt_levels.shape))
        # Calculate the dimensions over the cube that the tgt_levels span.
        tgt_aux_dims = list(range(cube_data.ndim))[dim_delta:]

    if interpolator is None:
        # Use the default stratify interpolator.
        interpolator = partial(
            stratify.interpolate, interpolation="linear", extrapolation="nan"
        )

    # Now perform the interpolation.
    new_data = interpolator(tgt_levels, src_data, cube_data, axis=axis)

    # Create a result cube with the correct shape and metadata.
    result = Cube(new_data, **cube.copy().metadata._asdict())

    # Copy across non z-dimension coordinates from the source cube
    # to the result cube.
    _copy_coords_without_z_dim(cube, result, axis)

    kwargs = dict(
        standard_name=src_levels.standard_name,
        long_name=src_levels.long_name,
        var_name=src_levels.var_name,
        units=src_levels.units,
        attributes=src_levels.attributes,
    )

    # Add our new interpolated coordinate to the result cube.
    try:
        coord = DimCoord(tgt_levels, **kwargs)
        result.add_dim_coord(coord, axis)
    except ValueError:
        # Attach the data to the trailing dimensions.
        coord = AuxCoord(tgt_levels, **kwargs)
        result.add_aux_coord(coord, tgt_aux_dims)

    return result
