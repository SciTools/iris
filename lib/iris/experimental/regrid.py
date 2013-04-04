# (C) British Crown Copyright 2013, Met Office
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
import warnings

import numpy as np
import numpy.ma as ma

import iris.analysis.cartography
import iris.coords
import iris.coord_systems
import iris.cube
import iris.unit


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
    src_crs = src_coord_system.as_cartopy_crs()
    grid_crs = grid_x_coord.coord_system.as_cartopy_crs()
    grid_x, grid_y = np.meshgrid(grid_x_coord.points, grid_y_coord.points)
    # Skip the CRS transform if we can to avoid precision problems.
    if src_crs == grid_crs:
        sample_grid_x = grid_x
        sample_grid_y = grid_y
    else:
        sample_xyz = src_crs.transform_points(grid_crs, grid_x, grid_y)

        # NB. Cartopy 0.7.0 contains a bug in transform_points which
        # mixes up the dimensions in the return value. We undo the damage
        # with a quick transpose.
        sample_xyz = sample_xyz.transpose(1, 0, 2)

        sample_grid_x = sample_xyz[..., 0]
        sample_grid_y = sample_xyz[..., 1]
    return sample_grid_x, sample_grid_y


def _regrid_bilinear_array(src_data, x_dim, y_dim, src_x_coord, src_y_coord,
                           sample_grid_x, sample_grid_y):
    """
    Regrid the given data from the src grid to the sample grid.

    Args:

    * src_data:
        An N-dimensional NumPy array.
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
    shape[y_dim] = sample_grid_x.shape[0]
    shape[x_dim] = sample_grid_x.shape[1]
    # If we're given integer values, convert them to the smallest
    # possible float dtype that can accurately preserve the values.
    dtype = src_data.dtype
    if dtype.kind == 'i':
        dtype = np.promote_types(dtype, np.float16)
    data = np.empty(shape, dtype=dtype)

    # TODO: Replace ... see later comment.
    # A crummy, temporary hack using iris.analysis.interpolate.linear()
    # The faults include, but are not limited to:
    # 1) It uses a nested `for` loop.
    # 2) It is doing lots of unncessary metadata faffing.
    # 3) It ends up performing two linear interpolations for each
    # column of results, and the first linear interpolation does a lot
    # more work than we'd ideally do, as most of the interpolated data
    # is irrelevant to the second interpolation.
    src = iris.cube.Cube(src_data)
    src.add_dim_coord(src_x_coord, x_dim)
    src.add_dim_coord(src_y_coord, y_dim)

    indices = [slice(None)] * data.ndim
    linear = iris.analysis.interpolate.linear
    for yx_index in np.ndindex(sample_grid_x.shape):
        x = sample_grid_x[yx_index]
        y = sample_grid_y[yx_index]
        column_pos = [(src_x_coord,  x), (src_y_coord, y)]
        column_data = linear(src, column_pos, 'nan').data
        indices[y_dim] = yx_index[0]
        indices[x_dim] = yx_index[1]
        data[tuple(indices)] = column_data

    # TODO:
    # Altenative:
    # Locate the four pairs of src x and y indices relevant to each grid
    # location:
    #   => x_indices.shape == (4, ny, nx); y_indices.shape == (4, ny, nx)
    # Calculate the relative weight of each corner:
    #   => weights.shape == (4, ny, nx)
    # Extract the src data relevant to the grid locations:
    #   => raw_data = src.data[..., y_indices, x_indices]
    #      NB. Can't rely on this index order in general.
    #   => raw_data.shape == (..., 4, ny, nx)
    # Weight it:
    #   => Reshape `weights` to broadcast against `raw_data`.
    #   => weighted_data = raw_data * weights
    #   => weighted_data.shape == (..., 4, ny, nx)
    # Sum over the `4` dimension:
    #   => data = weighted_data.sum(axis=sample_axis)
    #   => data.shape == (..., ny, nx)
    # Should be able to re-use the weights to calculate the interpolated
    # values for auxiliary coordinates as well.

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


def regrid_bilinear_rectilinear_src_and_grid(src, grid):
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
    if src_cs is None or grid_cs is None:
        raise ValueError("Both 'src' and 'grid' Cubes must have a"
                         " coordinate system for their rectilinear grid"
                         " coordinates.")

    def _valid_units(coord):
        if isinstance(coord.coord_system, (iris.coord_systems.GeogCS,
                                           iris.coord_systems.RotatedGeogCS)):
            valid_units = 'degrees'
        else:
            valid_units = 'm'
        return coord.units == valid_units
    if not all(_valid_units(coord) for coord in (src_x_coord, src_y_coord,
                                                 grid_x_coord, grid_y_coord)):
        raise ValueError("Unsupported units: must be 'degrees' or 'm'.")

    # Convert the grid to a 2D sample grid in the src CRS.
    sample_grid_x, sample_grid_y = _sample_grid(src_cs,
                                                grid_x_coord, grid_y_coord)

    # Compute the interpolated data values.
    x_dim = src.coord_dims(src_x_coord)[0]
    y_dim = src.coord_dims(src_y_coord)[0]
    src_data = src.data
    fill_value = None
    if np.ma.isMaskedArray(src_data):
        # Since we're using iris.analysis.interpolate.linear which
        # doesn't respect the mask, we replace masked values with NaN.
        fill_value = src_data.fill_value
        src_data = src_data.filled(np.nan)
    data = _regrid_bilinear_array(src_data, x_dim, y_dim,
                                  src_x_coord, src_y_coord,
                                  sample_grid_x, sample_grid_y)
    # Convert NaN based results to masked array where appropriate.
    mask = np.isnan(data)
    if np.any(mask):
        data = np.ma.MaskedArray(data, mask, fill_value=fill_value)

    # Wrap up the data as a Cube.
    result = _create_cube(data, src, x_dim, y_dim, src_x_coord, src_y_coord,
                          grid_x_coord, grid_y_coord,
                          sample_grid_x, sample_grid_y,
                          _regrid_bilinear_array)
    return result


def _within_bounds(bounds, lower, upper):
    """
    Return whether both lower and upper lie within the extremes
    of bounds.

    """
    min_bound = np.min(bounds)
    max_bound = np.max(bounds)

    return (min_bound <= lower <= max_bound) and \
        (min_bound <= upper <= max_bound)


def _cropped_bounds(bounds, lower, upper):
    """
    Return a new bounds array and corresponding slice object (or indices)
    that result from cropping the provided bounds between the specified lower
    and upper values. The bounds at the extremities will be truncated so that
    they start and end with lower and upper.

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
            #indices = tuple(range(l, u + 1))
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


def _get_bounds_in_units(coord, units):
    """Return a copy of coord's bounds in the specified units."""
    coord = coord.copy()
    coord.convert_units(units)
    return coord.bounds


def _regrid_area_weighted_array(src_data, x_dim, y_dim,
                                src_x_bounds, src_y_bounds,
                                grid_x_bounds, grid_y_bounds,
                                grid_x_decreasing, grid_y_decreasing,
                                area_func, circular=False):
    """
    Regrid the given data from its source grid to a new grid using
    an area weighted mean to determine the resulting data values.

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
    * circular:
        A boolean indicating whether the `src_x_bounds` are periodic. Default
        is False.

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
    masked = ma.isMaskedArray(src_data)
    if masked:
        new_data = ma.zeros(new_shape, fill_value=src_data.fill_value)
    else:
        new_data = ma.zeros(new_shape)
    # Assign to mask to explode it, allowing indexed assignment.
    new_data.mask = False

    # Simple for loop approach.
    indices = [slice(None)] * new_data.ndim
    for j, (y_0, y_1) in enumerate(grid_y_bounds):
        # Reverse lower and upper if dest grid is decreasing.
        if grid_y_decreasing:
            y_0, y_1 = y_1, y_0
        y_bounds, y_indices = _cropped_bounds(src_y_bounds, y_0, y_1)
        for i, (x_0, x_1) in enumerate(grid_x_bounds):
            # Reverse lower and upper if dest grid is decreasing.
            if grid_x_decreasing:
                x_0, x_1 = x_1, x_0
            x_bounds, x_indices = _cropped_bounds(src_x_bounds, x_0, x_1)

            # Determine whether to mask element i, j based on overlap with
            # src.
            # If x_0 > x_1 then we want [0]->x_1 and x_0->[-1] in the case of
            # wrapped longitudes. However if the src grid is not global (i.e.
            # circular) this new cell would include a region outside of the
            # extent of the src grid and should therefore be masked.
            outside_extent = x_0 > x_1 and not circular
            if (outside_extent or not _within_bounds(src_y_bounds, y_0, y_1) or
                    not _within_bounds(src_x_bounds, x_0, x_1)):
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
                Version = collections.namedtuple('Version',
                                                 ('major', 'minor', 'micro'))
                np_version = Version(*(int(val) for val in
                                       np.version.version.split('.')))
                if np_version.minor < 7:
                    warnings.warn('Rolling axes.')
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
                    # Axes of data over which the weighted mean is calculated.
                    axis = -1
                    new_data_pt = ma.average(data, weights=weights.ravel(),
                                             axis=axis)
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
                    _, broadcasted_weights = np.broadcast_arrays(data, weights)
                    # Axes of data over which the weighted mean is calculated.
                    axis = tuple(axes)
                    # Calculate weighted mean taking into account missing data.
                    new_data_pt = ma.average(data, weights=broadcasted_weights,
                                             axis=axis)

                # Determine suitable mask for data associated with cell.
                # Could use all() here.
                if masked:
                    # data.mask may be a bool, if not collapse via any().
                    if data.mask.ndim:
                        new_data_pt_mask = data.mask.any(axis=axis)
                    else:
                        new_data_pt_mask = data.mask

                # Insert data (and mask) values into new array.
                if x_dim is not None:
                    indices[x_dim] = i
                if y_dim is not None:
                    indices[y_dim] = j
                new_data[tuple(indices)] = new_data_pt
                if masked:
                    new_data.mask[tuple(indices)] = new_data_pt_mask

    # Remove new mask if original data was not masked
    # and no values in the new array are masked.
    if not masked and not new_data.mask.any():
        new_data = new_data.data

    # Special case to make 0-dimensional results take the same form as NumPy
    if new_data.shape == ():
        new_data = new_data.flat[0]

    return new_data


def regrid_area_weighted_rectilinear_src_and_grid(src_cube, grid_cube):
    """
    Return a new cube with data values calculated using the area weighted
    mean of data values from src_grid regridded onto the horizontal grid of
    grid_cube.

    Args:

    * src_cube:
        An instance of :class:`iris.cube.Cube` that supplies the data,
        metadata and coordinates.
    * grid_cube:
        An instance of :class:`iris.cube.Cube` that supplies the desired
        horizontal grid definition.

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

    # Condition 3: Check for compatible coords.
    if not src_x.is_compatible(grid_x) or not src_y.is_compatible(grid_y):
        raise ValueError('The new grid must be defined on the same coordinate '
                         'system, and have the same coordinate metadata, as '
                         'the source.')

    # Condition 4: cannot create vector coords from scalars.
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
        raise ValueError('The new grid must not require additional data '
                         'dimensions.')

    # Determine whether to calculate flat or spherical areas.
    # Don't only rely on coord system as it may be None.
    spherical = (isinstance(src_cs, (iris.coord_systems.GeogCS,
                                     iris.coord_systems.RotatedGeogCS)) or
                 src_x.units == 'degrees' or src_x.units == 'radians')

    # Get src and grid bounds in the same units.
    x_units = iris.unit.Unit('radians') if spherical else src_x.units
    y_units = iris.unit.Unit('radians') if spherical else src_y.units

    src_x_bounds = _get_bounds_in_units(src_x, x_units)
    src_y_bounds = _get_bounds_in_units(src_y, y_units)
    grid_x_bounds = _get_bounds_in_units(grid_x, x_units)
    grid_y_bounds = _get_bounds_in_units(grid_y, y_units)

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
                                           area_func, circular)

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
