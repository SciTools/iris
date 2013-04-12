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

import numpy as np

import iris.coord_systems
import iris.coords
import iris.cube


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
    result.metadata = src.metadata

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
