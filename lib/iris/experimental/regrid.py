# (C) British Crown Copyright 2013 - 2014, Met Office
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
Regridding functions.

"""
import collections
import copy
import functools
import warnings

import numpy as np
import numpy.ma as ma
from scipy.sparse import csc_matrix

import iris.analysis.cartography
from iris.analysis._scipy_interpolate import _RegularGridInterpolator
import iris.analysis._interpolator
import iris.coord_systems
import iris.cube
import iris.unit


_Version = collections.namedtuple('Version',
                                  ('major', 'minor', 'micro'))
_NP_VERSION = _Version(*(int(val) for val in
                         np.version.version.split('.')))


def _snapshot_grid(cube):
    """
    Helper function that returns deep copies of lateral dimension coordinates
    from a cube.

    """
    x, y = _get_xy_dim_coords(cube)
    return x.copy(), y.copy()


def _get_xy_dim_coords(cube):
    """
    Return the x and y dimension coordinates from a cube.

    This function raises a ValueError if the cube does not contain one and
    only one set of x and y dimension coordinates. It also raises a ValueError
    if the identified x and y coordinates do not have coordinate systems that
    are equal.

    Args:

    * cube:
        An instance of :class:`iris.cube.Cube`.

    Returns:
        A tuple containing the cube's x and y dimension coordinates.

    """
    x_coords = cube.coords(axis='x', dim_coords=True)
    if len(x_coords) != 1:
        raise ValueError('Cube {!r} must contain a single 1D x '
                         'coordinate.'.format(cube.name()))
    x_coord = x_coords[0]

    y_coords = cube.coords(axis='y', dim_coords=True)
    if len(y_coords) != 1:
        raise ValueError('Cube {!r} must contain a single 1D y '
                         'coordinate.'.format(cube.name()))
    y_coord = y_coords[0]

    if x_coord.coord_system != y_coord.coord_system:
        raise ValueError("The cube's x ({!r}) and y ({!r}) "
                         "coordinates must have the same coordinate "
                         "system.".format(x_coord.name(), y_coord.name()))

    return x_coord, y_coord


def _get_xy_coords(cube):
    """
    Return the x and y coordinates from a cube.

    This function will preferentially return a pair of dimension
    coordinates (if there are more than one potential x or y dimension
    coordinates a ValueError will be raised). If the cube does not have
    a pair of x and y dimension coordinates it will return 1D auxiliary
    coordinates (including scalars). If there is not one and only one set
    of x and y auxiliary coordinates a ValueError will be raised.

    Having identified the x and y coordinates, the function checks that they
    have equal coordinate systems and that they do not occupy the same
    dimension on the cube.

    Args:

    * cube:
        An instance of :class:`iris.cube.Cube`.

    Returns:
        A tuple containing the cube's x and y coordinates.

    """
    # Look for a suitable dimension coords first.
    x_coords = cube.coords(axis='x', dim_coords=True)
    if not x_coords:
        # If there is no x coord in dim_coords look for scalars or
        # monotonic coords in aux_coords.
        x_coords = [coord for coord in cube.coords(axis='x', dim_coords=False)
                    if coord.ndim == 1 and coord.is_monotonic()]
    if len(x_coords) != 1:
        raise ValueError('Cube {!r} must contain a single 1D x '
                         'coordinate.'.format(cube.name()))
    x_coord = x_coords[0]

    # Look for a suitable dimension coords first.
    y_coords = cube.coords(axis='y', dim_coords=True)
    if not y_coords:
        # If there is no y coord in dim_coords look for scalars or
        # monotonic coords in aux_coords.
        y_coords = [coord for coord in cube.coords(axis='y', dim_coords=False)
                    if coord.ndim == 1 and coord.is_monotonic()]
    if len(y_coords) != 1:
        raise ValueError('Cube {!r} must contain a single 1D y '
                         'coordinate.'.format(cube.name()))
    y_coord = y_coords[0]

    if x_coord.coord_system != y_coord.coord_system:
        raise ValueError("The cube's x ({!r}) and y ({!r}) "
                         "coordinates must have the same coordinate "
                         "system.".format(x_coord.name(), y_coord.name()))

    # The x and y coordinates must describe different dimensions
    # or be scalar coords.
    x_dims = cube.coord_dims(x_coord)
    x_dim = None
    if x_dims:
        x_dim = x_dims[0]

    y_dims = cube.coord_dims(y_coord)
    y_dim = None
    if y_dims:
        y_dim = y_dims[0]

    if x_dim is not None and y_dim == x_dim:
        raise ValueError("The cube's x and y coords must not describe the "
                         "same data dimension.")

    return x_coord, y_coord


def _sample_grid(src_coord_system, grid_x_coord, grid_y_coord):
    """
    Convert the rectilinear grid coordinates to a curvilinear grid in
    the source coordinate system.

    The `grid_x_coord` and `grid_y_coord` must share a common coordinate
    system.

    Args:

    * src_coord_system:
        The :class:`iris.coord_system.CoordSystem` for the grid of the
        source Cube.
    * grid_x_coord:
        The :class:`iris.coords.DimCoord` for the X coordinate.
    * grid_y_coord:
        The :class:`iris.coords.DimCoord` for the Y coordinate.

    Returns:
        A tuple of the X and Y coordinate values as 2-dimensional
        arrays.

    """
    grid_x, grid_y = np.meshgrid(grid_x_coord.points, grid_y_coord.points)
    # Skip the CRS transform if we can to avoid precision problems.
    if src_coord_system == grid_x_coord.coord_system:
        sample_grid_x = grid_x
        sample_grid_y = grid_y
    else:
        src_crs = src_coord_system.as_cartopy_crs()
        grid_crs = grid_x_coord.coord_system.as_cartopy_crs()
        sample_xyz = src_crs.transform_points(grid_crs, grid_x, grid_y)
        sample_grid_x = sample_xyz[..., 0]
        sample_grid_y = sample_xyz[..., 1]
    return sample_grid_x, sample_grid_y


def _regrid_bilinear_array(src_data, x_dim, y_dim, src_x_coord, src_y_coord,
                           sample_grid_x, sample_grid_y,
                           extrapolation_mode='nanmask'):
    """
    Regrid the given data from the src grid to the sample grid.

    The result will be a MaskedArray if either/both of:
     - the source array is a MaskedArray,
     - the extrapolation_mode is 'mask' and the result requires
       extrapolation.

    If the result is a MaskedArray the mask for each element will be set
    if either/both of:
     - there is a non-zero contribution from masked items in the input data,
     - the element requires extrapolation and the extrapolation_mode
       dictates a masked value.

    Args:

    * src_data:
        An N-dimensional NumPy array or MaskedArray.
    * x_dim:
        The X dimension within `src_data`.
    * y_dim:
        The Y dimension within `src_data`.
    * src_x_coord:
        The X :class:`iris.coords.DimCoord`.
    * src_y_coord:
        The Y :class:`iris.coords.DimCoord`.
    * sample_grid_x:
        A 2-dimensional array of sample X values.
    * sample_grid_y:
        A 2-dimensional array of sample Y values.

    Kwargs:

    * extrapolation_mode:
        Must be one of the following strings:

          * 'linear' - The extrapolation points will be calculated by
            extending the gradient of the closest two points.
          * 'nan' - The extrapolation points will be be set to NaN.
          * 'error' - A ValueError exception will be raised, notifying an
            attempt to extrapolate.
          * 'mask' - The extrapolation points will always be masked, even
            if the source data is not a MaskedArray.
          * 'nanmask' - If the source data is a MaskedArray the
            extrapolation points will be masked. Otherwise they will be
            set to NaN.

        The default mode of extrapolation is 'nanmask'.

    Returns:
        The regridded data as an N-dimensional NumPy array. The lengths
        of the X and Y dimensions will now match those of the sample
        grid.

    """
    if sample_grid_x.shape != sample_grid_y.shape:
        raise ValueError('Inconsistent sample grid shapes.')
    if sample_grid_x.ndim != 2:
        raise ValueError('Sample grid must be 2-dimensional.')

    # Prepare the result data array
    shape = list(src_data.shape)
    assert shape[x_dim] == src_x_coord.shape[0]
    assert shape[y_dim] == src_y_coord.shape[0]

    shape[y_dim] = sample_grid_x.shape[0]
    shape[x_dim] = sample_grid_x.shape[1]

    # If we're given integer values, convert them to the smallest
    # possible float dtype that can accurately preserve the values.
    dtype = src_data.dtype
    if dtype.kind == 'i':
        dtype = np.promote_types(dtype, np.float16)

    if isinstance(src_data, ma.MaskedArray):
        data = ma.empty(shape, dtype=dtype)
        data.mask = np.zeros(data.shape, dtype=np.bool)
    else:
        data = np.empty(shape, dtype=dtype)

    # The interpolation class requires monotonically increasing coordinates,
    # so flip the coordinate(s) and data if the aren't.
    reverse_x = src_x_coord.points[0] > src_x_coord.points[1]
    reverse_y = src_y_coord.points[0] > src_y_coord.points[1]
    flip_index = [slice(None)] * src_data.ndim
    if reverse_x:
        src_x_coord = src_x_coord[::-1]
        flip_index[x_dim] = slice(None, None, -1)
    if reverse_y:
        src_y_coord = src_y_coord[::-1]
        flip_index[y_dim] = slice(None, None, -1)
    src_data = src_data[tuple(flip_index)]

    if src_x_coord.circular:
        extend = iris.analysis._interpolator._extend_circular_coord_and_data
        x_points, src_data = extend(src_x_coord, src_data, x_dim)
    else:
        x_points = src_x_coord.points

    # Slice out the first full 2D piece of data for construction of the
    # interpolator.
    index = [0] * src_data.ndim
    index[x_dim] = index[y_dim] = slice(None)
    initial_data = src_data[tuple(index)]
    if y_dim < x_dim:
        initial_data = initial_data.T

    # Construct the interpolator, we will fill in any values out of bounds
    # manually.
    interpolator = _RegularGridInterpolator([x_points,
                                             src_y_coord.points],
                                            initial_data, fill_value=None,
                                            bounds_error=False)
    # The constructor of the _RegularGridInterpolator class does
    # some unnecessary checks on these values, so we set them
    # afterwards instead. Sneaky. ;-)
    try:
        modes = iris.analysis._interpolator._LINEAR_EXTRAPOLATION_MODES
        mode = modes[extrapolation_mode]
    except KeyError:
        raise ValueError('Invalid extrapolation mode.')
    interpolator.bounds_error = mode.bounds_error
    interpolator.fill_value = mode.fill_value

    # Construct the target coordinate points array, suitable for passing to
    # the interpolator multiple times.
    interpolator_coords = [sample_grid_x.astype(np.float64)[..., np.newaxis],
                           sample_grid_y.astype(np.float64)[..., np.newaxis]]

    # Map all the requested values into the range of the source
    # data (centred over the centre of the source data to allow
    # extrapolation where required).
    min_x, max_x = x_points.min(), x_points.max()
    min_y, max_y = src_y_coord.points.min(), src_y_coord.points.max()
    if src_x_coord.units.modulus:
        modulus = src_x_coord.units.modulus
        offset = (max_x + min_x - modulus) * 0.5
        interpolator_coords[0] -= offset
        interpolator_coords[0] = (interpolator_coords[0] % modulus) + offset

    interpolator_coords = np.dstack(interpolator_coords)

    def interpolate(data):
        # Update the interpolator for this data slice.
        data = data.astype(interpolator.values.dtype)
        if y_dim < x_dim:
            data = data.T
        interpolator.values = data
        data = interpolator(interpolator_coords)
        if y_dim > x_dim:
            data = data.T
        return data

    # Build up a shape suitable for passing to ndindex, inside the loop we
    # will insert slice(None) on the data indices.
    iter_shape = list(shape)
    iter_shape[x_dim] = iter_shape[y_dim] = 1

    # Iterate through each 2d slice of the data, updating the interpolator
    # with the new data as we go.
    for index in np.ndindex(tuple(iter_shape)):
        index = list(index)
        index[x_dim] = index[y_dim] = slice(None)

        src_subset = src_data[tuple(index)]
        interpolator.fill_value = mode.fill_value
        data[tuple(index)] = interpolate(src_subset)

        if isinstance(data, ma.MaskedArray) or mode.force_mask:
            # NB. np.ma.getmaskarray returns an array of `False` if
            # `src_subset` is not a masked array.
            src_mask = np.ma.getmaskarray(src_subset)
            interpolator.fill_value = mode.mask_fill_value
            mask_fraction = interpolate(src_mask)
            new_mask = (mask_fraction > 0)

            if np.ma.isMaskedArray(data):
                data.mask[tuple(index)] = new_mask
            elif np.any(new_mask):
                # Set mask=False to ensure we have an expanded mask array.
                data = np.ma.MaskedArray(data, mask=False)
                data.mask[tuple(index)] = new_mask

    return data


def _regrid_reference_surface(src_surface_coord, surface_dims, x_dim, y_dim,
                              src_x_coord, src_y_coord,
                              sample_grid_x, sample_grid_y, regrid_callback):
    """
    Return a new reference surface coordinate appropriate to the sample
    grid.

    Args:

    * src_surface_coord:
        The :class:`iris.coords.Coord` containing the source reference
        surface.
    * surface_dims:
        The tuple of the data dimensions relevant to the source
        reference surface coordinate.
    * x_dim:
        The X dimension within the source Cube.
    * y_dim:
        The Y dimension within the source Cube.
    * src_x_coord:
        The X :class:`iris.coords.DimCoord`.
    * src_y_coord:
        The Y :class:`iris.coords.DimCoord`.
    * sample_grid_x:
        A 2-dimensional array of sample X values.
    * sample_grid_y:
        A 2-dimensional array of sample Y values.
    * regrid_callback:
        The routine that will be used to calculate the interpolated
        values for the new reference surface.

    Returns:
        The new reference surface coordinate.

    """
    # Determine which of the reference surface's dimensions span the X
    # and Y dimensions of the source cube.
    surface_x_dim = surface_dims.index(x_dim)
    surface_y_dim = surface_dims.index(y_dim)
    surface = regrid_callback(src_surface_coord.points,
                              surface_x_dim, surface_y_dim,
                              src_x_coord, src_y_coord,
                              sample_grid_x, sample_grid_y)
    surface_coord = src_surface_coord.copy(surface)
    return surface_coord


def _create_cube(data, src, x_dim, y_dim, src_x_coord, src_y_coord,
                 grid_x_coord, grid_y_coord, sample_grid_x, sample_grid_y,
                 regrid_callback):
    """
    Return a new Cube for the result of regridding the source Cube onto
    the new grid.

    All the metadata and coordinates of the result Cube are copied from
    the source Cube, with two exceptions:
        - Grid dimension coordinates are copied from the grid Cube.
        - Auxiliary coordinates which span the grid dimensions are
          ignored, except where they provide a reference surface for an
          :class:`iris.aux_factory.AuxCoordFactory`.

    Args:

    * data:
        The regridded data as an N-dimensional NumPy array.
    * src:
        The source Cube.
    * x_dim:
        The X dimension within the source Cube.
    * y_dim:
        The Y dimension within the source Cube.
    * src_x_coord:
        The X :class:`iris.coords.DimCoord`.
    * src_y_coord:
        The Y :class:`iris.coords.DimCoord`.
    * grid_x_coord:
        The :class:`iris.coords.DimCoord` for the new grid's X
        coordinate.
    * grid_y_coord:
        The :class:`iris.coords.DimCoord` for the new grid's Y
        coordinate.
    * sample_grid_x:
        A 2-dimensional array of sample X values.
    * sample_grid_y:
        A 2-dimensional array of sample Y values.
    * regrid_callback:
        The routine that will be used to calculate the interpolated
        values of any reference surfaces.

    Returns:
        The new, regridded Cube.

    """
    # Create a result cube with the appropriate metadata
    result = iris.cube.Cube(data)
    result.metadata = copy.deepcopy(src.metadata)

    # Copy across all the coordinates which don't span the grid.
    # Record a mapping from old coordinate IDs to new coordinates,
    # for subsequent use in creating updated aux_factories.
    coord_mapping = {}

    def copy_coords(src_coords, add_method):
        for coord in src_coords:
            dims = src.coord_dims(coord)
            if coord is src_x_coord:
                coord = grid_x_coord
            elif coord is src_y_coord:
                coord = grid_y_coord
            elif x_dim in dims or y_dim in dims:
                continue
            result_coord = coord.copy()
            add_method(result_coord, dims)
            coord_mapping[id(coord)] = result_coord

    copy_coords(src.dim_coords, result.add_dim_coord)
    copy_coords(src.aux_coords, result.add_aux_coord)

    # Copy across any AuxFactory instances, and regrid their reference
    # surfaces where required.
    for factory in src.aux_factories:
        for coord in factory.dependencies.itervalues():
            if coord is None:
                continue
            dims = src.coord_dims(coord)
            if x_dim in dims and y_dim in dims:
                result_coord = _regrid_reference_surface(
                    coord, dims, x_dim, y_dim, src_x_coord, src_y_coord,
                    sample_grid_x, sample_grid_y, regrid_callback)
                result.add_aux_coord(result_coord, dims)
                coord_mapping[id(coord)] = result_coord
        try:
            result.add_aux_factory(factory.updated(coord_mapping))
        except KeyError:
            msg = 'Cannot update aux_factory {!r} because of dropped' \
                  ' coordinates.'.format(factory.name())
            warnings.warn(msg)
    return result


def regrid_bilinear_rectilinear_src_and_grid(src, grid,
                                             extrapolation_mode='mask'):
    """
    Return a new Cube that is the result of regridding the source Cube
    onto the grid of the grid Cube using bilinear interpolation.

    Both the source and grid Cubes must be defined on rectilinear grids.

    Auxiliary coordinates which span the grid dimensions are ignored,
    except where they provide a reference surface for an
    :class:`iris.aux_factory.AuxCoordFactory`.

    Args:

    * src:
        The source :class:`iris.cube.Cube` providing the data.
    * grid:
        The :class:`iris.cube.Cube` which defines the new grid.

    Kwargs:

    * extrapolation_mode:
        Must be one of the following strings:

          * 'linear' - The extrapolation points will be calculated by
            extending the gradient of the closest two points.
          * 'nan' - The extrapolation points will be be set to NaN.
          * 'error' - A ValueError exception will be raised, notifying an
            attempt to extrapolate.
          * 'mask' - The extrapolation points will always be masked, even
            if the source data is not a MaskedArray.
          * 'nanmask' - If the source data is a MaskedArray the
            extrapolation points will be masked. Otherwise they will be
            set to NaN.

        The default mode of extrapolation is 'mask'.

    Returns:
        The :class:`iris.cube.Cube` resulting from regridding the source
        data onto the grid defined by the grid Cube.

    """
    # Validity checks
    if not isinstance(src, iris.cube.Cube):
        raise TypeError("'src' must be a Cube")
    if not isinstance(grid, iris.cube.Cube):
        raise TypeError("'grid' must be a Cube")
    src_x_coord, src_y_coord = _get_xy_dim_coords(src)
    grid_x_coord, grid_y_coord = _get_xy_dim_coords(grid)
    src_cs = src_x_coord.coord_system
    grid_cs = grid_x_coord.coord_system
    if src_cs is None and grid_cs is None:
        if not (src_x_coord.is_compatible(grid_x_coord) and
                src_y_coord.is_compatible(grid_y_coord)):
            raise ValueError("The rectilinear grid coordinates of the 'src' "
                             "and 'grid' Cubes have no coordinate system but "
                             "they do not have matching coordinate metadata.")
    elif src_cs is None or grid_cs is None:
        raise ValueError("The rectilinear grid coordinates of the 'src' and "
                         "'grid' Cubes must either both have coordinate "
                         "systems or both have no coordinate system but with "
                         "matching coordinate metadata.")

    def _check_units(coord):
        if coord.coord_system is None:
            # No restriction on units.
            pass
        elif isinstance(coord.coord_system,
                        (iris.coord_systems.GeogCS,
                         iris.coord_systems.RotatedGeogCS)):
            # Units for lat-lon or rotated pole must be 'degrees'. Note
            # that 'degrees_east' etc. are equal to 'degrees'.
            if coord.units != 'degrees':
                msg = "Unsupported units for coordinate system. " \
                      "Expected 'degrees' got {!r}.".format(coord.units)
                raise ValueError(msg)
        else:
            # Units for other coord systems must be equal to metres.
            if coord.units != 'm':
                msg = "Unsupported units for coordinate system. " \
                      "Expected 'metres' got {!r}.".format(coord.units)
                raise ValueError(msg)

    for coord in (src_x_coord, src_y_coord, grid_x_coord, grid_y_coord):
        _check_units(coord)

    modes = iris.analysis._interpolator._LINEAR_EXTRAPOLATION_MODES
    if extrapolation_mode not in modes:
        raise ValueError('Invalid extrapolation mode.')

    # Convert the grid to a 2D sample grid in the src CRS.
    sample_grid_x, sample_grid_y = _sample_grid(src_cs,
                                                grid_x_coord, grid_y_coord)

    # Compute the interpolated data values.
    x_dim = src.coord_dims(src_x_coord)[0]
    y_dim = src.coord_dims(src_y_coord)[0]
    data = _regrid_bilinear_array(src.data, x_dim, y_dim,
                                  src_x_coord, src_y_coord,
                                  sample_grid_x, sample_grid_y,
                                  extrapolation_mode)

    # Wrap up the data as a Cube.
    regrid_callback = functools.partial(_regrid_bilinear_array,
                                        extrapolation_mode='nan')
    result = _create_cube(data, src, x_dim, y_dim, src_x_coord, src_y_coord,
                          grid_x_coord, grid_y_coord,
                          sample_grid_x, sample_grid_y,
                          regrid_callback)
    return result


def _within_bounds(src_bounds, tgt_bounds, orderswap=False):
    """
    Determine which target bounds lie within the extremes of the source bounds.

    Args:

    * src_bounds (ndarray):
        An (n, 2) shaped array of monotonic contiguous source bounds.
    * tgt_bounds (ndarray):
        An (n, 2) shaped array corresponding to the target bounds.

    Kwargs:

    * orderswap (bool):
        A Boolean indicating whether the target bounds are in descending order
        (True). Defaults to False.

    Returns:
        Boolean ndarray, indicating whether each target bound is within the
        extremes of the source bounds.

    """
    min_bound = np.min(src_bounds)
    max_bound = np.max(src_bounds)

    # Swap upper-lower is necessary.
    if orderswap is True:
        upper, lower = tgt_bounds.T
    else:
        lower, upper = tgt_bounds.T

    return (((lower <= max_bound) * (lower >= min_bound)) *
            ((upper <= max_bound) * (upper >= min_bound)))


def _cropped_bounds(bounds, lower, upper):
    """
    Return a new bounds array and corresponding slice object (or indices) of
    the original data array, resulting from cropping the provided bounds
    between the specified lower and upper values. The bounds at the
    extremities will be truncated so that they start and end with lower and
    upper.

    This function will return an empty NumPy array and slice if there is no
    overlap between the region covered by bounds and the region from lower to
    upper.

    If lower > upper the resulting bounds may not be contiguous and the
    indices object will be a tuple of indices rather than a slice object.

    Args:

    * bounds:
        An (n, 2) shaped array of monotonic contiguous bounds.
    * lower:
        Lower bound at which to crop the bounds array.
    * upper:
        Upper bound at which to crop the bounds array.

    Returns:
        A tuple of the new bounds array and the corresponding slice object or
        indices from the zeroth axis of the original array.

    """
    reversed_flag = False
    # Ensure order is increasing.
    if bounds[0, 0] > bounds[-1, 0]:
        # Reverse bounds
        bounds = bounds[::-1, ::-1]
        reversed_flag = True

    # Number of bounds.
    n = bounds.shape[0]

    if lower <= upper:
        if lower > bounds[-1, 1] or upper < bounds[0, 0]:
            new_bounds = bounds[0:0]
            indices = slice(0, 0)
        else:
            # A single region lower->upper.
            if lower < bounds[0, 0]:
                # Region extends below bounds so use first lower bound.
                l = 0
                lower = bounds[0, 0]
            else:
                # Index of last lower bound less than or equal to lower.
                l = np.nonzero(bounds[:, 0] <= lower)[0][-1]
            if upper > bounds[-1, 1]:
                # Region extends above bounds so use last upper bound.
                u = n - 1
                upper = bounds[-1, 1]
            else:
                # Index of first upper bound greater than or equal to
                # upper.
                u = np.nonzero(bounds[:, 1] >= upper)[0][0]
            # Extract the bounds in our region defined by lower->upper.
            new_bounds = np.copy(bounds[l:(u + 1), :])
            # Replace first and last values with specified bounds.
            new_bounds[0, 0] = lower
            new_bounds[-1, 1] = upper
            if reversed_flag:
                indices = slice(n - (u + 1), n - l)
            else:
                indices = slice(l, u + 1)
    else:
        # Two regions [0]->upper, lower->[-1]
        # [0]->upper
        if upper < bounds[0, 0]:
            # Region outside src bounds.
            new_bounds_left = bounds[0:0]
            indices_left = tuple()
            slice_left = slice(0, 0)
        else:
            if upper > bounds[-1, 1]:
                # Whole of bounds.
                u = n - 1
                upper = bounds[-1, 1]
            else:
                # Index of first upper bound greater than or equal to upper.
                u = np.nonzero(bounds[:, 1] >= upper)[0][0]
            # Extract the bounds in our region defined by [0]->upper.
            new_bounds_left = np.copy(bounds[0:(u + 1), :])
            # Replace last value with specified bound.
            new_bounds_left[-1, 1] = upper
            if reversed_flag:
                indices_left = tuple(range(n - (u + 1), n))
                slice_left = slice(n - (u + 1), n)
            else:
                indices_left = tuple(range(0, u + 1))
                slice_left = slice(0, u + 1)
        # lower->[-1]
        if lower > bounds[-1, 1]:
            # Region is outside src bounds.
            new_bounds_right = bounds[0:0]
            indices_right = tuple()
            slice_right = slice(0, 0)
        else:
            if lower < bounds[0, 0]:
                # Whole of bounds.
                l = 0
                lower = bounds[0, 0]
            else:
                # Index of last lower bound less than or equal to lower.
                l = np.nonzero(bounds[:, 0] <= lower)[0][-1]
            # Extract the bounds in our region defined by lower->[-1].
            new_bounds_right = np.copy(bounds[l:, :])
            # Replace first value with specified bound.
            new_bounds_right[0, 0] = lower
            if reversed_flag:
                indices_right = tuple(range(0, n - l))
                slice_right = slice(0, n - l)
            else:
                indices_right = tuple(range(l, n))
                slice_right = slice(l, None)

        if reversed_flag:
            # Flip everything around.
            indices_left, indices_right = indices_right, indices_left
            slice_left, slice_right = slice_right, slice_left

        # Combine regions.
        new_bounds = np.concatenate((new_bounds_left, new_bounds_right))
        # Use slices if possible, but if we have two regions use indices.
        if indices_left and indices_right:
            indices = indices_left + indices_right
        elif indices_left:
            indices = slice_left
        elif indices_right:
            indices = slice_right
        else:
            indices = slice(0, 0)

    if reversed_flag:
        new_bounds = new_bounds[::-1, ::-1]

    return new_bounds, indices


def _cartesian_area(y_bounds, x_bounds):
    """
    Return an array of the areas of each cell given two arrays
    of cartesian bounds.

    Args:

    * y_bounds:
        An (n, 2) shaped NumPy array.
    * x_bounds:
        An (m, 2) shaped NumPy array.

    Returns:
        An (n, m) shaped Numpy array of areas.

    """
    heights = y_bounds[:, 1] - y_bounds[:, 0]
    widths = x_bounds[:, 1] - x_bounds[:, 0]
    return np.abs(np.outer(heights, widths))


def _spherical_area(y_bounds, x_bounds, radius=1.0):
    """
    Return an array of the areas of each cell on a sphere
    given two arrays of latitude and longitude bounds in radians.

    Args:

    * y_bounds:
        An (n, 2) shaped NumPy array of latitide bounds in radians.
    * x_bounds:
        An (m, 2) shaped NumPy array of longitude bounds in radians.
    * radius:
        Radius of the sphere. Default is 1.0.

    Returns:
        An (n, m) shaped Numpy array of areas.

    """
    return iris.analysis.cartography._quadrant_area(
        y_bounds + np.pi / 2.0, x_bounds, radius)


def _get_bounds_in_units(coord, units, dtype):
    """Return a copy of coord's bounds in the specified units and dtype."""
    # The bounds are cast to dtype before conversion to prevent issues when
    # mixing float32 and float64 types.
    return coord.units.convert(coord.bounds.astype(dtype), units).astype(dtype)


def _weighted_mean_with_mdtol(data, weights, axis=None, mdtol=0):
    """
    Return the weighted mean of an array over the specified axis
    using the provided weights (if any) and a permitted fraction of
    masked data.

    Args:

    * data (array-like):
        Data to be averaged.

    * weights (array-like):
        An array of the same shape as the data that specifies the contribution
        of each corresponding data element to the calculated mean.

    Kwargs:

    * axis (int or tuple of ints):
        Axis along which the mean is computed. The default is to compute
        the mean of the flattened array.

    * mdtol (float):
        Tolerance of missing data. The value returned in each element of the
        returned array will be masked if the fraction of masked data exceeds
        mdtol. This fraction is weighted by the `weights` array if one is
        provided. mdtol=0 means no missing data is tolerated
        while mdtol=1 will mean the resulting element will be masked if and
        only if all the contributing elements of data are masked.
        Defaults to 0.

    Returns:
        Numpy array (possibly masked) or scalar.

    """
    res = ma.average(data, weights=weights, axis=axis)
    if ma.isMaskedArray(data) and mdtol < 1:
        weights_total = weights.sum(axis=axis)
        masked_weights = weights.copy()
        masked_weights[~ma.getmaskarray(data)] = 0
        masked_weights_total = masked_weights.sum(axis=axis)
        frac_masked = np.true_divide(masked_weights_total, weights_total)
        mask_pt = frac_masked > mdtol
        if np.any(mask_pt):
            if np.isscalar(res):
                res = ma.masked
            elif ma.isMaskedArray(res):
                res.mask |= mask_pt
            else:
                res = ma.masked_array(res, mask=mask_pt)
    return res


def _regrid_area_weighted_array(src_data, x_dim, y_dim,
                                src_x_bounds, src_y_bounds,
                                grid_x_bounds, grid_y_bounds,
                                grid_x_decreasing, grid_y_decreasing,
                                area_func, circular=False, mdtol=0):
    """
    Regrid the given data from its source grid to a new grid using
    an area weighted mean to determine the resulting data values.

    .. note::

        Elements in the returned array that lie either partially
        or entirely outside of the extent of the source grid will
        be masked irrespective of the value of mdtol.

    Args:

    * src_data:
        An N-dimensional NumPy array.
    * x_dim:
        The X dimension within `src_data`.
    * y_dim:
        The Y dimension within `src_data`.
    * src_x_bounds:
        A NumPy array of bounds along the X axis defining the source grid.
    * src_y_bounds:
        A NumPy array of bounds along the Y axis defining the source grid.
    * grid_x_bounds:
        A NumPy array of bounds along the X axis defining the new grid.
    * grid_y_bounds:
        A NumPy array of bounds along the Y axis defining the new grid.
    * grid_x_decreasing:
        Boolean indicating whether the X coordinate of the new grid is
        in descending order.
    * grid_y_decreasing:
        Boolean indicating whether the Y coordinate of the new grid is
        in descending order.
    * area_func:
        A function that returns an (p, q) array of weights given an (p, 2)
        shaped array of Y bounds and an (q, 2) shaped array of X bounds.

    Kwargs:

    * circular:
        A boolean indicating whether the `src_x_bounds` are periodic. Default
        is False.

    * mdtol:
        Tolerance of missing data. The value returned in each element of the
        returned array will be masked if the fraction of missing data exceeds
        mdtol. This fraction is calculated based on the area of masked cells
        within each target cell. mdtol=0 means no missing data is tolerated
        while mdtol=1 will mean the resulting element will be masked if and
        only if all the overlapping elements of the source grid are masked.
        Defaults to 0.

    Returns:
        The regridded data as an N-dimensional NumPy array. The lengths
        of the X and Y dimensions will now match those of the target
        grid.

    """
    # Create empty data array to match the new grid.
    # Note that dtype is not preserved and that the array is
    # masked to allow for regions that do not overlap.
    new_shape = list(src_data.shape)
    if x_dim is not None:
        new_shape[x_dim] = grid_x_bounds.shape[0]
    if y_dim is not None:
        new_shape[y_dim] = grid_y_bounds.shape[0]

    # Flag to indicate whether the original data was a masked array.
    src_masked = ma.isMaskedArray(src_data)
    if src_masked:
        new_data = ma.zeros(new_shape, fill_value=src_data.fill_value)
    else:
        new_data = ma.zeros(new_shape)
    # Assign to mask to explode it, allowing indexed assignment.
    new_data.mask = False

    indices = [slice(None)] * new_data.ndim

    # Determine which grid bounds are within src extent.
    y_within_bounds = _within_bounds(src_y_bounds, grid_y_bounds,
                                     grid_y_decreasing)
    x_within_bounds = _within_bounds(src_x_bounds, grid_x_bounds,
                                     grid_x_decreasing)

    # Cache which src_bounds are within grid bounds
    cached_x_bounds = []
    cached_x_indices = []
    for (x_0, x_1) in grid_x_bounds:
        if grid_x_decreasing:
            x_0, x_1 = x_1, x_0
        x_bounds, x_indices = _cropped_bounds(src_x_bounds, x_0, x_1)
        cached_x_bounds.append(x_bounds)
        cached_x_indices.append(x_indices)

    # Simple for loop approach.
    for j, (y_0, y_1) in enumerate(grid_y_bounds):
        # Reverse lower and upper if dest grid is decreasing.
        if grid_y_decreasing:
            y_0, y_1 = y_1, y_0
        y_bounds, y_indices = _cropped_bounds(src_y_bounds, y_0, y_1)
        for i, (x_0, x_1) in enumerate(grid_x_bounds):
            # Reverse lower and upper if dest grid is decreasing.
            if grid_x_decreasing:
                x_0, x_1 = x_1, x_0
            x_bounds = cached_x_bounds[i]
            x_indices = cached_x_indices[i]

            # Determine whether to mask element i, j based on overlap with
            # src.
            # If x_0 > x_1 then we want [0]->x_1 and x_0->[0] + mod in the case
            # of wrapped longitudes. However if the src grid is not global
            # (i.e. circular) this new cell would include a region outside of
            # the extent of the src grid and should therefore be masked.
            outside_extent = x_0 > x_1 and not circular
            if (outside_extent or not y_within_bounds[j] or not
                    x_within_bounds[i]):
                # Mask out element(s) in new_data
                if x_dim is not None:
                    indices[x_dim] = i
                if y_dim is not None:
                    indices[y_dim] = j
                new_data[tuple(indices)] = ma.masked
            else:
                # Calculate weighted mean of data points.
                # Slice out relevant data (this may or may not be a view()
                # depending on x_indices being a slice or not).
                if x_dim is not None:
                    indices[x_dim] = x_indices
                if y_dim is not None:
                    indices[y_dim] = y_indices
                if isinstance(x_indices, tuple) and \
                        isinstance(y_indices, tuple):
                    raise RuntimeError('Cannot handle split bounds '
                                       'in both x and y.')
                data = src_data[tuple(indices)]

                # Calculate weights based on areas of cropped bounds.
                weights = area_func(y_bounds, x_bounds)

                # Numpy 1.7 allows the axis keyword arg to be a tuple.
                # If the version of NumPy is less than 1.7 manipulate the axes
                # of the data so the x and y dimensions can be flattened.
                if _NP_VERSION.minor < 7:
                    if y_dim is not None and x_dim is not None:
                        flattened_shape = list(data.shape)
                        if y_dim > x_dim:
                            data = np.rollaxis(data, y_dim, data.ndim)
                            data = np.rollaxis(data, x_dim, data.ndim)
                            del flattened_shape[y_dim]
                            del flattened_shape[x_dim]
                        else:
                            data = np.rollaxis(data, x_dim, data.ndim)
                            data = np.rollaxis(data, y_dim, data.ndim)
                            del flattened_shape[x_dim]
                            del flattened_shape[y_dim]
                            weights = weights.T
                        flattened_shape.append(-1)
                        data = data.reshape(*flattened_shape)
                    elif y_dim is not None:
                        flattened_shape = list(data.shape)
                        del flattened_shape[y_dim]
                        flattened_shape.append(-1)
                        data = data.swapaxes(y_dim, -1).reshape(
                            *flattened_shape)
                    elif x_dim is not None:
                        flattened_shape = list(data.shape)
                        del flattened_shape[x_dim]
                        flattened_shape.append(-1)
                        data = data.swapaxes(x_dim, -1).reshape(
                            *flattened_shape)
                    weights = weights.ravel()
                    axis = -1
                else:
                    # Transpose weights to match dim ordering in data.
                    weights_shape_y = weights.shape[0]
                    weights_shape_x = weights.shape[1]
                    if x_dim is not None and y_dim is not None and \
                            x_dim < y_dim:
                        weights = weights.T
                    # Broadcast the weights array to allow numpy's ma.average
                    # to be called.
                    weights_padded_shape = [1] * data.ndim
                    axes = []
                    if y_dim is not None:
                        weights_padded_shape[y_dim] = weights_shape_y
                        axes.append(y_dim)
                    if x_dim is not None:
                        weights_padded_shape[x_dim] = weights_shape_x
                        axes.append(x_dim)
                    # Assign new shape to raise error on copy.
                    weights.shape = weights_padded_shape
                    # Broadcast weights to match shape of data.
                    _, weights = np.broadcast_arrays(data, weights)
                    # Axes of data over which the weighted mean is calculated.
                    axis = tuple(axes)

                # Calculate weighted mean taking into account missing data.
                new_data_pt = _weighted_mean_with_mdtol(
                    data, weights=weights, axis=axis, mdtol=mdtol)

                # Insert data (and mask) values into new array.
                if x_dim is not None:
                    indices[x_dim] = i
                if y_dim is not None:
                    indices[y_dim] = j
                new_data[tuple(indices)] = new_data_pt

    # Remove new mask if original data was not masked
    # and no values in the new array are masked.
    if not src_masked and not new_data.mask.any():
        new_data = new_data.data

    return new_data


def regrid_area_weighted_rectilinear_src_and_grid(src_cube, grid_cube,
                                                  mdtol=0):
    """
    Return a new cube with data values calculated using the area weighted
    mean of data values from src_grid regridded onto the horizontal grid of
    grid_cube.

    This function requires that the horizontal grids of both cubes are
    rectilinear (i.e. expressed in terms of two orthogonal 1D coordinates)
    and that these grids are in the same coordinate system. This function
    also requires that the coordinates describing the horizontal grids
    all have bounds.

    .. note::

        Elements in data array of the returned cube that lie either partially
        or entirely outside of the horizontal extent of the src_cube will
        be masked irrespective of the value of mdtol.

    Args:

    * src_cube:
        An instance of :class:`iris.cube.Cube` that supplies the data,
        metadata and coordinates.
    * grid_cube:
        An instance of :class:`iris.cube.Cube` that supplies the desired
        horizontal grid definition.

    Kwargs:

    * mdtol:
        Tolerance of missing data. The value returned in each element of the
        returned cube's data array will be masked if the fraction of masked
        data in the overlapping cells of the source cube exceeds mdtol. This
        fraction is calculated based on the area of masked cells within each
        target cell. mdtol=0 means no missing data is tolerated while mdtol=1
        will mean the resulting element will be masked if and only if all the
        overlapping cells of the source cube are masked. Defaults to 0.

    Returns:
        A new :class:`iris.cube.Cube` instance.

    """
    # Get the 1d monotonic (or scalar) src and grid coordinates.
    src_x, src_y = _get_xy_coords(src_cube)
    grid_x, grid_y = _get_xy_coords(grid_cube)

    # Condition 1: All x and y coordinates must have contiguous bounds to
    # define areas.
    if not src_x.is_contiguous() or not src_y.is_contiguous() or \
            not grid_x.is_contiguous() or not grid_y.is_contiguous():
        raise ValueError("The horizontal grid coordinates of both the source "
                         "and grid cubes must have contiguous bounds.")

    # Condition 2: Everything must have the same coordinate system.
    src_cs = src_x.coord_system
    grid_cs = grid_x.coord_system
    if src_cs != grid_cs:
        raise ValueError("The horizontal grid coordinates of both the source "
                         "and grid cubes must have the same coordinate "
                         "system.")

    # Condition 3: cannot create vector coords from scalars.
    src_x_dims = src_cube.coord_dims(src_x)
    src_x_dim = None
    if src_x_dims:
        src_x_dim = src_x_dims[0]
    src_y_dims = src_cube.coord_dims(src_y)
    src_y_dim = None
    if src_y_dims:
        src_y_dim = src_y_dims[0]
    if src_x_dim is None and grid_x.shape[0] != 1 or \
            src_y_dim is None and grid_y.shape[0] != 1:
        raise ValueError('The horizontal grid coordinates of source cube '
                         'includes scalar coordinates, but the new grid does '
                         'not. The new grid must not require additional data '
                         'dimensions to be created.')

    # Determine whether to calculate flat or spherical areas.
    # Don't only rely on coord system as it may be None.
    spherical = (isinstance(src_cs, (iris.coord_systems.GeogCS,
                                     iris.coord_systems.RotatedGeogCS)) or
                 src_x.units == 'degrees' or src_x.units == 'radians')

    # Get src and grid bounds in the same units.
    x_units = iris.unit.Unit('radians') if spherical else src_x.units
    y_units = iris.unit.Unit('radians') if spherical else src_y.units

    # Operate in highest precision.
    src_dtype = np.promote_types(src_x.bounds.dtype, src_y.bounds.dtype)
    grid_dtype = np.promote_types(grid_x.bounds.dtype, grid_y.bounds.dtype)
    dtype = np.promote_types(src_dtype, grid_dtype)

    src_x_bounds = _get_bounds_in_units(src_x, x_units, dtype)
    src_y_bounds = _get_bounds_in_units(src_y, y_units, dtype)
    grid_x_bounds = _get_bounds_in_units(grid_x, x_units, dtype)
    grid_y_bounds = _get_bounds_in_units(grid_y, y_units, dtype)

    # Determine whether target grid bounds are decreasing. This must
    # be determined prior to wrap_lons being called.
    grid_x_decreasing = grid_x_bounds[-1, 0] < grid_x_bounds[0, 0]
    grid_y_decreasing = grid_y_bounds[-1, 0] < grid_y_bounds[0, 0]

    # Wrapping of longitudes.
    if spherical:
        base = np.min(src_x_bounds)
        modulus = x_units.modulus
        # Only wrap if necessary to avoid introducing floating
        # point errors.
        if np.min(grid_x_bounds) < base or \
                np.max(grid_x_bounds) > (base + modulus):
            grid_x_bounds = iris.analysis.cartography.wrap_lons(grid_x_bounds,
                                                                base, modulus)

    # Determine whether the src_x coord has periodic boundary conditions.
    circular = getattr(src_x, 'circular', False)

    # Use simple cartesian area function or one that takes into
    # account the curved surface if coord system is spherical.
    if spherical:
        area_func = _spherical_area
    else:
        area_func = _cartesian_area

    # Calculate new data array for regridded cube.
    new_data = _regrid_area_weighted_array(src_cube.data, src_x_dim, src_y_dim,
                                           src_x_bounds, src_y_bounds,
                                           grid_x_bounds, grid_y_bounds,
                                           grid_x_decreasing,
                                           grid_y_decreasing,
                                           area_func, circular, mdtol)

    # Wrap up the data as a Cube.
    # Create 2d meshgrids as required by _create_cube func.
    meshgrid_x, meshgrid_y = np.meshgrid(grid_x.points, grid_y.points)
    new_cube = _create_cube(new_data, src_cube, src_x_dim, src_y_dim,
                            src_x, src_y, grid_x, grid_y,
                            meshgrid_x, meshgrid_y,
                            _regrid_bilinear_array)

    # Slice out any length 1 dimensions.
    indices = [slice(None, None)] * new_data.ndim
    if src_x_dim is not None and new_cube.shape[src_x_dim] == 1:
        indices[src_x_dim] = 0
    if src_y_dim is not None and new_cube.shape[src_y_dim] == 1:
        indices[src_y_dim] = 0
    if 0 in indices:
        new_cube = new_cube[tuple(indices)]

    return new_cube


def regrid_weighted_curvilinear_to_rectilinear(src_cube, weights, grid_cube):
    """
    Return a new cube with the data values calculated using the weighted
    mean of data values from :data:`src_cube` and the weights from
    :data:`weights` regridded onto the horizontal grid of :data:`grid_cube`.

    This function requires that the :data:`src_cube` has a curvilinear
    horizontal grid and the target :data:`grid_cube` is rectilinear
    i.e. expressed in terms of two orthogonal 1D horizontal coordinates.
    Both grids must be in the same coordinate system, and the :data:`grid_cube`
    must have horizontal coordinates that are both bounded and contiguous.

    Note that, for any given target :data:`grid_cube` cell, only the points
    from the :data:`src_cube` that are bound by that cell will contribute to
    the cell result. The bounded extent of the :data:`src_cube` will not be
    considered here.

    A target :data:`grid_cube` cell result will be calculated as,
    :math:`\sum (src\_cube.data_{ij} * weights_{ij}) / \sum weights_{ij}`, for
    all :math:`ij` :data:`src_cube` points that are bound by that cell.

    .. warning::

        * Only 2D cubes are supported.
        * All coordinates that span the :data:`src_cube` that don't define
          the horizontal curvilinear grid will be ignored.
        * The :class:`iris.unit.Unit` of the horizontal grid coordinates
          must be either :data:`degrees` or :data:`radians`.

    Args:

    * src_cube:
        A :class:`iris.cube.Cube` instance that defines the source
        variable grid to be regridded.
    * weights:
        A :class:`numpy.ndarray` instance that defines the weights
        for the source variable grid cells. Must have the same shape
        as the :data:`src_cube.data`.
    * grid_cube:
        A :class:`iris.cube.Cube` instance that defines the target
        rectilinear grid.

    Returns:
        A :class:`iris.cube.Cube` instance.

    """
    if src_cube.shape != weights.shape:
        msg = 'The source cube and weights require the same data shape.'
        raise ValueError(msg)

    if src_cube.ndim != 2 or grid_cube.ndim != 2:
        msg = 'The source cube and target grid cube must reference 2D data.'
        raise ValueError(msg)

    if src_cube.aux_factories:
        msg = 'All source cube derived coordinates will be ignored.'
        warnings.warn(msg)

    # Get the source cube x and y 2D auxiliary coordinates.
    sx, sy = src_cube.coord(axis='x'), src_cube.coord(axis='y')
    # Get the target grid cube x and y dimension coordinates.
    tx, ty = _get_xy_dim_coords(grid_cube)

    if sx.units.modulus is None or sy.units.modulus is None or \
            sx.units != sy.units:
        msg = 'The source cube x ({!r}) and y ({!r}) coordinates must ' \
            'have units of degrees or radians.'
        raise ValueError(msg.format(sx.name(), sy.name()))

    if tx.units.modulus is None or ty.units.modulus is None or \
            tx.units != ty.units:
        msg = 'The target grid cube x ({!r}) and y ({!r}) coordinates must ' \
            'have units of degrees or radians.'
        raise ValueError(msg.format(tx.name(), ty.name()))

    if sx.units != tx.units:
        msg = 'The source cube and target grid cube must have x and y ' \
            'coordinates with the same units.'
        raise ValueError(msg)

    if sx.ndim != sy.ndim:
        msg = 'The source cube x ({!r}) and y ({!r}) coordinates must ' \
            'have the same dimensionality.'
        raise ValueError(msg.format(sx.name(), sy.name()))

    if sx.ndim != 2:
        msg = 'The source cube x ({!r}) and y ({!r}) coordinates must ' \
            'be 2D auxiliary coordinates.'
        raise ValueError(msg.format(sx.name(), sy.name()))

    if sx.coord_system != sy.coord_system:
        msg = 'The source cube x ({!r}) and y ({!r}) coordinates must ' \
            'have the same coordinate system.'
        raise ValueError(msg.format(sx.name(), sy.name()))

    if sx.coord_system != tx.coord_system and \
            sx.coord_system is not None and \
            tx.coord_system is not None:
        msg = 'The source cube and target grid cube must have the same ' \
            'coordinate system.'
        raise ValueError(msg)

    if not tx.has_bounds() or not tx.is_contiguous():
        msg = 'The target grid cube x ({!r})coordinate requires ' \
            'contiguous bounds.'
        raise ValueError(msg.format(tx.name()))

    if not ty.has_bounds() or not ty.is_contiguous():
        msg = 'The target grid cube y ({!r}) coordinate requires ' \
            'contiguous bounds.'
        raise ValueError(msg.format(ty.name()))

    def _src_align_and_flatten(coord):
        # Return a flattened, unmasked copy of a coordinate's points array that
        # will align with a flattened version of the source cube's data.
        points = coord.points
        if src_cube.coord_dims(coord) == (1, 0):
            points = points.T
        if points.shape != src_cube.shape:
            msg = 'The shape of the points array of !r is not compatible ' \
                'with the shape of !r.'.format(coord.name(), src_cube.name())
            raise ValueError(msg)
        return np.asarray(points.flatten())

    # Align and flatten the coordinate points of the source space.
    sx_points = _src_align_and_flatten(sx)
    sy_points = _src_align_and_flatten(sy)

    # Match the source cube x coordinate range to the target grid
    # cube x coordinate range.
    min_sx, min_tx = np.min(sx.points), np.min(tx.points)
    modulus = sx.units.modulus
    if min_sx < 0 and min_tx >= 0:
        indices = np.where(sx_points < 0)
        sx_points[indices] += modulus
    elif min_sx >= 0 and min_tx < 0:
        indices = np.where(sx_points > (modulus / 2))
        sx_points[indices] -= modulus

    # Create target grid cube x and y cell boundaries.
    tx_depth, ty_depth = tx.points.size, ty.points.size
    tx_dim, = grid_cube.coord_dims(tx)
    ty_dim, = grid_cube.coord_dims(ty)

    tx_cells = np.concatenate((tx.bounds[:, 0],
                               tx.bounds[-1, 1].reshape(1)))
    ty_cells = np.concatenate((ty.bounds[:, 0],
                               ty.bounds[-1, 1].reshape(1)))

    # Determine the target grid cube x and y cells that bound
    # the source cube x and y points.

    def _regrid_indices(cells, depth, points):
        # Calculate the minimum difference in cell extent.
        extent = np.min(np.diff(cells))
        if extent == 0:
            # Detected an dimension coordinate with an invalid
            # zero length cell extent.
            msg = 'The target grid cube {} ({!r}) coordinate contains ' \
                'a zero length cell extent.'
            axis, name = 'x', tx.name()
            if points is sy_points:
                axis, name = 'y', ty.name()
            raise ValueError(msg.format(axis, name))
        elif extent > 0:
            # The cells of the dimension coordinate are in ascending order.
            indices = np.searchsorted(cells, points, side='right') - 1
        else:
            # The cells of the dimension coordinate are in descending order.
            # np.searchsorted() requires ascending order, so we require to
            # account for this restriction.
            cells = cells[::-1]
            right = np.searchsorted(cells, points, side='right')
            left = np.searchsorted(cells, points, side='left')
            indices = depth - right
            # Only those points that exactly match the left-hand cell bound
            # will differ between 'left' and 'right'. Thus their appropriate
            # target cell location requires to be recalculated to give the
            # correct descending [upper, lower) interval cell, source to target
            # regrid behaviour.
            delta = np.where(left != right)[0]
            if delta.size:
                indices[delta] = depth - left[delta]
        return indices

    x_indices = _regrid_indices(tx_cells, tx_depth, sx_points)
    y_indices = _regrid_indices(ty_cells, ty_depth, sy_points)

    # Now construct a sparse M x N matix, where M is the flattened target
    # space, and N is the flattened source space. The sparse matrix will then
    # be populated with those source cube points that contribute to a specific
    # target cube cell.

    # Determine the valid indices and their offsets in M x N space.
    if ma.isMaskedArray(src_cube.data):
        # Calculate the valid M offsets, accounting for the source cube mask.
        mask = ~src_cube.data.mask.flatten()
        cols = np.where((y_indices >= 0) & (y_indices < ty_depth) &
                        (x_indices >= 0) & (x_indices < tx_depth) &
                        mask)[0]
    else:
        # Calculate the valid M offsets.
        cols = np.where((y_indices >= 0) & (y_indices < ty_depth) &
                        (x_indices >= 0) & (x_indices < tx_depth))[0]

    # Reduce the indices to only those that are valid.
    x_indices = x_indices[cols]
    y_indices = y_indices[cols]

    # Calculate the valid N offsets.
    if ty_dim < tx_dim:
        rows = y_indices * tx.points.size + x_indices
    else:
        rows = x_indices * ty.points.size + y_indices

    # Calculate the associated valid weights.
    weights_flat = weights.flatten()
    data = weights_flat[cols]

    # Build our sparse M x N matrix of weights.
    sparse_matrix = csc_matrix((data, (rows, cols)),
                               shape=(grid_cube.data.size, src_cube.data.size))

    # Performing a sparse sum to collapse the matrix to (M, 1).
    sum_weights = sparse_matrix.sum(axis=1).getA()

    # Determine the rows (flattened target indices) that have a
    # contribution from one or more source points.
    rows = np.nonzero(sum_weights)

    # Calculate the numerator of the weighted mean (M, 1).
    numerator = sparse_matrix * src_cube.data.reshape(-1, 1)

    # Calculate the weighted mean payload.
    weighted_mean = ma.masked_all(numerator.shape, dtype=numerator.dtype)
    weighted_mean[rows] = numerator[rows] / sum_weights[rows]

    # Construct the final regridded weighted mean cube.
    dim_coords_and_dims = zip((ty.copy(), tx.copy()), (ty_dim, tx_dim))
    cube = iris.cube.Cube(weighted_mean.reshape(grid_cube.shape),
                          dim_coords_and_dims=dim_coords_and_dims)
    cube.metadata = copy.deepcopy(src_cube.metadata)

    for coord in src_cube.coords(dimensions=()):
        cube.add_aux_coord(coord.copy())

    return cube
