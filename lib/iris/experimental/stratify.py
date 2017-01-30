# (C) British Crown Copyright 2017, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
Routines for putting data on new strata (aka. isosurfaces), often in the
Z direction.

"""
from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

from functools import partial

import numpy as np
import stratify

from iris.cube import Cube
from iris.coords import Coord, AuxCoord, DimCoord


def _copy_coords_without_dim(cube, result, z_dim):
    """
    Helper function to copy across non-z coordinates between cubes.

    Parameters
    ----------
    cube : iris.cube.Cube instance
        Incoming cube containing the coordinates to be copied.
    result : iris.cube.Cube instance
        Incoming cube for the coordinates to be copied to.
    z_dim : Dimension within the cube of the z-dimension
        The dimension will not be copied, as the incoming z-dimension
        is model level_height, whilst the outgoing z-dimension is pressure.

    Returns
    -------
    Cube
        A single cube with the non-z dimensions added to
        incoming cube.

    """
    # Copy across non-Z coordinates.
    for coord in cube.dim_coords:
        dim, = cube.coord_dims(coord)
        if dim != z_dim:
            result.add_dim_coord(coord.copy(), dim)
    for coord in cube.aux_coords:
        dims = cube.coord_dims(coord)
        if z_dim not in dims:
            result.add_aux_coord(coord.copy(), dims)
    for coord in cube.derived_coords:
        dims = cube.coord_dims(coord)
        if z_dim not in dims:
            result.add_aux_coord(coord.copy(), dims)


def relevel(phenom, level_data, target_levels, interp_direction=0,
            interpolator=None):
    """
    Interpolate the given phenomenon onto the given levels of level_data.

    For example, suppose we have two datasets `P(i,j,k)` and `H(i,j,k)`
    and we want ``P(i,j,H)`` we call relevel with
    ``phenom=P`` and ``level_data=H`` and ``target_levels`` being
    an array of the values of ``H`` we would like.

    This routine is especially useful for computing isosurfaces of
    phenomenon that are generally monotonic in the direction of interpolation
    (such as height/pressure and salinity/depth).

    Parameters
    ----------
    phenom : :class:`iris.cube.Cube`
        The data to be re-levelled.

    level_data : :class:`~iris.cube.Cube` or :class:`~iris.coord.Coord`
        The coordinate values (cube/coord) in the same system as
        ``target levels``.
        All dimensions of ``level_data`` must be broadcastable to ``phenom``

    target_levels : array-like
        The levels of ``level_data`` to pick from ``phenom``.
        For the default interpolator, this must be a 1d array.
        For a custom interpolator this may be multi-dimensional and must
        broadcast to phenom and level_data except in the phenomenon's
        ``interp_direction`` dimension.

    interp_direction : int or :class:`~iris.coords.Coord` instance
        The direction of interpolation. This is necessary because
        level_data is rarely 1D. This is often the z dimension coordinate
        of ``phenom``.

    interpolator : callable or None
        The interpolator to use when computing the interpolation. The function
        will be passed the following positional arguments::

            (target levels, level data, phenom data, interp axis)

        If the interpolator is None, :func:`stratify.interpolate` will be used
        with linear interpolation and NaN extrapolation.

        An example of constructing an alternative interpolation scheme:

            from functools import partial
            interpolator = partial(stratify.interpolate,
                                   interpolation=stratify.INTERPOLATE_NEAREST,
                                   extrapolation=stratify.EXTRAPOLATE_LINEAR)

    """
    # Identify the z-coordinate within the phenom cube
    if isinstance(interp_direction, Coord):
        [interp_dim] = phenom.coord_dims(interp_direction)
    else:
        interp_dim = interp_direction

    if isinstance(level_data, Coord):
        source_data = level_data.points
    else:
        source_data = level_data.data

    # phenom and level_data must be broadcastable.
    phenom_data, source_data = np.broadcast_arrays(phenom.data,
                                                   source_data)

    target_levels = np.array(target_levels)
    target_phenom_dim = interp_dim
    if target_levels.ndim != 1:
        # target levels must be broadcastable to phenom in everything
        # but the interpolation axis (if not, raise)
        t_dim_delta = phenom_data.ndim - target_levels.ndim
        tl_axis = interp_dim - t_dim_delta
        data_shape = list(phenom_data.shape)
        data_shape.pop(interp_dim)
        target_shape = list(target_levels.shape)
        target_shape.pop(tl_axis)
        np.broadcast_arrays(np.empty(data_shape), np.empty(target_shape))
        target_phenom_dim = list(range(phenom_data.ndim))[t_dim_delta:]

    if interpolator is None:
        interpolator = partial(stratify.interpolate,
                               interpolation='linear', extrapolation='nan')

    interpolated_phenom = interpolator(target_levels,
                                       source_data, phenom_data,
                                       axis=interp_dim)

    # Create a result Cube with the correct shape and metadata.
    result = Cube(interpolated_phenom, **phenom.copy().metadata._asdict())
    # Copy across non-Z coordinates.
    _copy_coords_without_dim(phenom, result, interp_dim)

    coord_kwargs = dict(standard_name=level_data.standard_name,
                        long_name=level_data.long_name,
                        var_name=level_data.var_name,
                        units=level_data.units,
                        attributes=level_data.attributes)
    # Add our new interp-dim coordinate.
    try:
        result.add_dim_coord(DimCoord(target_levels, **coord_kwargs),
                             interp_dim)
    except ValueError:
        # Attach the data to the last dimesions.
        result.add_aux_coord(
            AuxCoord(target_levels, **coord_kwargs),
            target_phenom_dim)
    return result
