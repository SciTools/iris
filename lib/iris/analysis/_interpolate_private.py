# (C) British Crown Copyright 2010 - 2017, Met Office
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
This is the 'original' content of :mod:`iris.analysis.interpolate`, which has
now been deprecated.

A rename was essential to provide a deprecation warning on import of the
original name, while still providing this code for internal usage (for now)
without triggering the deprecation notice.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

import collections
import warnings

import numpy as np
import scipy
import scipy.spatial
from scipy.interpolate.interpolate import interp1d

from iris.analysis import Linear
import iris.cube
import iris.coord_systems
import iris.coords
from iris._deprecation import warn_deprecated
import iris.exceptions


def nearest_neighbour_indices(cube, sample_points):
    """
    Returns the indices to select the data value(s) closest to the given coordinate point values.

    The sample_points mapping does not have to include coordinate values corresponding to all data
    dimensions. Any dimensions unspecified will default to a full slice.

    For example:

        >>> cube = iris.load_cube(iris.sample_data_path('ostia_monthly.nc'))
        >>> iris.analysis.interpolate.nearest_neighbour_indices(cube, [('latitude', 0), ('longitude', 10)])
        (slice(None, None, None), 9, 12)
        >>> iris.analysis.interpolate.nearest_neighbour_indices(cube, [('latitude', 0)])
        (slice(None, None, None), 9, slice(None, None, None))

    Args:

    * cube:
        An :class:`iris.cube.Cube`.
    * sample_points
        A list of tuple pairs mapping coordinate instances or unique coordinate names in the cube to point values.

    Returns:
        The tuple of indices which will select the point in the cube closest to the supplied coordinate values.

    .. note::

        Nearest neighbour interpolation of multidimensional coordinates is not
        yet supported.

    .. deprecated:: 1.10

        The module :mod:`iris.analysis.interpolate` is deprecated.
        Please replace usage of
        :func:`iris.analysis.interpolate.nearest_neighbour_indices`
        with :meth:`iris.coords.Coord.nearest_neighbour_index`.

    """
    if isinstance(sample_points, dict):
        msg = ('Providing a dictionary to specify points is deprecated. '
               'Please provide a list of (coordinate, values) pairs.')
        warn_deprecated(msg)
        sample_points = list(sample_points.items())

    if sample_points:
        try:
            coord, values = sample_points[0]
        except ValueError:
            raise ValueError('Sample points must be a list of (coordinate, value) pairs. Got %r.' % sample_points)

    points = []
    for coord, values in sample_points:
        if isinstance(coord, six.string_types):
            coord = cube.coord(coord)
        else:
            coord = cube.coord(coord)
        points.append((coord, values))
    sample_points = points

    # Build up a list of indices to span the cube.
    indices = [slice(None, None)] * cube.ndim
    
    # Build up a dictionary which maps the cube's data dimensions to a list (which will later
    # be populated by coordinates in the sample points list)
    dim_to_coord_map = {}
    for i in range(cube.ndim):
        dim_to_coord_map[i] = []

    # Iterate over all of the specifications provided by sample_points
    for coord, point in sample_points:
        data_dim = cube.coord_dims(coord)

        # If no data dimension then we don't need to make any modifications to indices.
        if not data_dim:
            continue
        elif len(data_dim) > 1:
            raise iris.exceptions.CoordinateMultiDimError("Nearest neighbour interpolation of multidimensional "
                                                          "coordinates is not supported.")
        data_dim = data_dim[0]

        dim_to_coord_map[data_dim].append(coord)

        #calculate the nearest neighbour
        min_index = coord.nearest_neighbour_index(point)

        if getattr(coord, 'circular', False):
            warnings.warn("Nearest neighbour on a circular coordinate may not be picking the nearest point.", DeprecationWarning)

        # If the dimension has already been interpolated then assert that the index from this coordinate
        # agrees with the index already calculated, otherwise we have a contradicting specification
        if indices[data_dim] != slice(None, None) and min_index != indices[data_dim]:
            raise ValueError('The coordinates provided (%s) over specify dimension %s.' %
                                        (', '.join([coord.name() for coord in dim_to_coord_map[data_dim]]), data_dim))

        indices[data_dim] = min_index

    return tuple(indices)


def extract_nearest_neighbour(cube, sample_points):
    """
    Returns a new cube using data value(s) closest to the given coordinate point values.

    The sample_points mapping does not have to include coordinate values corresponding to all data
    dimensions. Any dimensions unspecified will default to a full slice.

    For example:

        >>> cube = iris.load_cube(iris.sample_data_path('ostia_monthly.nc'))
        >>> iris.analysis.interpolate.extract_nearest_neighbour(cube, [('latitude', 0), ('longitude', 10)])
        <iris 'Cube' of surface_temperature / (K) (time: 54)>
        >>> iris.analysis.interpolate.extract_nearest_neighbour(cube, [('latitude', 0)])
        <iris 'Cube' of surface_temperature / (K) (time: 54; longitude: 432)>

    Args:

    * cube:
        An :class:`iris.cube.Cube`.
    * sample_points
        A list of tuple pairs mapping coordinate instances or unique coordinate names in the cube to point values.

    Returns:
        A cube that represents uninterpolated data as near to the given points as possible.

    .. deprecated:: 1.10

        The module :mod:`iris.analysis.interpolate` is deprecated.
        Please replace usage of
        :func:`iris.analysis.interpolate.extract_nearest_neighbour`
        with :meth:`iris.cube.Cube.interpolate` using the scheme
        :class:`iris.analysis.Nearest`.

    """
    return cube[nearest_neighbour_indices(cube, sample_points)]


def regrid(source_cube, grid_cube, mode='bilinear', **kwargs):
    """
    Returns a new cube with values derived from the source_cube on the horizontal grid specified
    by the grid_cube.

    Fundamental input requirements:
        1) Both cubes must have a CoordSystem.
        2) The source 'x' and 'y' coordinates must not share data dimensions with any other coordinates.

    In addition, the algorithm currently used requires:
        3) Both CS instances must be compatible:
            i.e. of the same type, with the same attribute values, and with compatible coordinates.
        4) No new data dimensions can be created.
        5) Source cube coordinates to map to a single dimension.

    Args:

    * source_cube:
        An instance of :class:`iris.cube.Cube` which supplies the source data and metadata.
    * grid_cube:
        An instance of :class:`iris.cube.Cube` which supplies the horizontal grid definition.

    Kwargs:

    * mode (string):
        Regridding interpolation algorithm to be applied, which may be one of the following:

            * 'bilinear' for bi-linear interpolation (default), see :func:`iris.analysis.interpolate.linear`.
            * 'nearest' for nearest neighbour interpolation.

    Returns:
        A new :class:`iris.cube.Cube` instance.

    .. note::

        The masked status of values are currently ignored.  See :func:\
`~iris.experimental.regrid.regrid_bilinear_rectilinear_src_and_grid`
        for regrid support with mask awareness.

    .. deprecated:: 1.10

        Please use :meth:`iris.cube.Cube.regrid` instead, with an appropriate
        regridding scheme:

        *   For mode='bilinear', simply use the :class:`~iris.analysis.Linear`
            scheme.

        *   For mode='nearest', use the :class:`~iris.analysis.Nearest` scheme,
            with extrapolation_mode='extrapolate', but be aware of the
            following possible differences:

            *   Any missing result points, i.e. those which match source points
                which are masked or NaN, are returned as as NaN values by this
                routine.  The 'Nearest' scheme, however, represents missing
                results as masked points in a masked array.
                *Which* points are missing is unchanged.

            *   Longitude wrapping for this routine is controlled by the
                'circular' property of the x coordinate.
                The 'Nearest' scheme, however, *always* wraps any coords with
                modular units, such as (correctly formed) longitudes.
                Thus, behaviour can be different if "x_coord.circular" is
                False :  In that case, if the original non-longitude-wrapped
                operation is required, it can be replicated by converting all
                X and Y coordinates' units to '1' and removing their coordinate
                systems.

    """
    if mode == 'bilinear':
        scheme = Linear(**kwargs)
        return source_cube.regrid(grid_cube, scheme)

    # Condition 1
    source_cs = source_cube.coord_system(iris.coord_systems.CoordSystem)
    grid_cs = grid_cube.coord_system(iris.coord_systems.CoordSystem)
    if (source_cs is None) != (grid_cs is None):
        raise ValueError("The source and grid cubes must both have a CoordSystem or both have None.")

    # Condition 2: We can only have one x coordinate and one y coordinate with the source CoordSystem, and those coordinates
    # must be the only ones occupying their respective dimension
    source_x = source_cube.coord(axis='x', coord_system=source_cs)
    source_y = source_cube.coord(axis='y', coord_system=source_cs)

    source_x_dims = source_cube.coord_dims(source_x)
    source_y_dims = source_cube.coord_dims(source_y)

    source_x_dim = None
    if source_x_dims:
        if len(source_x_dims) > 1:
            raise ValueError('The source x coordinate may not describe more than one data dimension.')
        source_x_dim = source_x_dims[0]
        dim_sharers = ', '.join([coord.name() for coord in source_cube.coords(contains_dimension=source_x_dim) if coord is not source_x])
        if dim_sharers:
            raise ValueError('No coordinates may share a dimension (dimension %s) with the x '
                             'coordinate, but (%s) do.' % (source_x_dim, dim_sharers))

    source_y_dim = None
    if source_y_dims:
        if len(source_y_dims) > 1:
            raise ValueError('The source y coordinate may not describe more than one data dimension.')
        source_y_dim = source_y_dims[0]
        dim_sharers = ', '.join([coord.name() for coord in source_cube.coords(contains_dimension=source_y_dim) if coord is not source_y])
        if dim_sharers:
            raise ValueError('No coordinates may share a dimension (dimension %s) with the y '
                             'coordinate, but (%s) do.' % (source_y_dim, dim_sharers))

    if source_x_dim is not None and source_y_dim == source_x_dim:
        raise ValueError('The source x and y coords may not describe the same data dimension.')


    # Condition 3
    # Check for compatible horizontal CSs. Currently that means they're exactly the same except for the coordinate
    # values.
    # The same kind of CS ...
    compatible = (source_cs == grid_cs)
    if compatible:
        grid_x = grid_cube.coord(axis='x', coord_system=grid_cs)
        grid_y = grid_cube.coord(axis='y', coord_system=grid_cs)
        compatible = source_x.is_compatible(grid_x) and \
            source_y.is_compatible(grid_y)
    if not compatible:
        raise ValueError("The new grid must be defined on the same coordinate system, and have the same coordinate "
                         "metadata, as the source.")

    # Condition 4
    if grid_cube.coord_dims(grid_x) and not source_x_dims or \
            grid_cube.coord_dims(grid_y) and not source_y_dims:
        raise ValueError("The new grid must not require additional data dimensions.")

    x_coord = grid_x.copy()
    y_coord = grid_y.copy()


    #
    # Adjust the data array to match the new grid.
    #

    # get the new shape of the data
    new_shape = list(source_cube.shape)
    if source_x_dims:
        new_shape[source_x_dims[0]] = grid_x.shape[0]
    if source_y_dims:
        new_shape[source_y_dims[0]] = grid_y.shape[0]

    new_data = np.empty(new_shape, dtype=source_cube.data.dtype)

    # Prepare the index pattern which will be used to insert a single "column" of data.
    # NB. A "column" is a slice constrained to a single XY point, which therefore extends over *all* the other axes.
    # For an XYZ cube this means a column only extends over Z and corresponds to the normal definition of "column".
    indices = [slice(None, None)] * new_data.ndim

    if mode == 'bilinear':
        # Perform bilinear interpolation, passing through any keywords.
        points_dict = [(source_x, list(x_coord.points)), (source_y, list(y_coord.points))]
        new_data = linear(source_cube, points_dict, **kwargs).data
    else:
        # Perform nearest neighbour interpolation on each column in turn.
        for iy, y in enumerate(y_coord.points):
            for ix, x in enumerate(x_coord.points):
                column_pos = [(source_x,  x), (source_y, y)]
                column_data = extract_nearest_neighbour(source_cube, column_pos).data
                if source_y_dim is not None:
                    indices[source_y_dim] = iy
                if source_x_dim is not None:
                    indices[source_x_dim] = ix
                new_data[tuple(indices)] = column_data

    # Special case to make 0-dimensional results take the same form as NumPy
    if new_data.shape == ():
        new_data = new_data.flat[0]

    # Start with just the metadata and the re-sampled data...
    new_cube = iris.cube.Cube(new_data)
    new_cube.metadata = source_cube.metadata
    new_cube.fill_value = source_cube.fill_value

    # ... and then copy across all the unaffected coordinates.

    # Record a mapping from old coordinate IDs to new coordinates,
    # for subsequent use in creating updated aux_factories.
    coord_mapping = {}

    def copy_coords(source_coords, add_method):
        for coord in source_coords:
            if coord is source_x or coord is source_y:
                continue
            dims = source_cube.coord_dims(coord)
            new_coord = coord.copy()
            add_method(new_coord, dims)
            coord_mapping[id(coord)] = new_coord

    copy_coords(source_cube.dim_coords, new_cube.add_dim_coord)
    copy_coords(source_cube.aux_coords, new_cube.add_aux_coord)

    for factory in source_cube.aux_factories:
        new_cube.add_aux_factory(factory.updated(coord_mapping))

    # Add the new coords
    if source_x in source_cube.dim_coords:
        new_cube.add_dim_coord(x_coord, source_x_dim)
    else:
        new_cube.add_aux_coord(x_coord, source_x_dims)

    if source_y in source_cube.dim_coords:
        new_cube.add_dim_coord(y_coord, source_y_dim)
    else:
        new_cube.add_aux_coord(y_coord, source_y_dims)

    return new_cube


def linear(cube, sample_points, extrapolation_mode='linear'):
    """
    Return a cube of the linearly interpolated points given the desired
    sample points.

    Given a list of tuple pairs mapping coordinates (or coordinate names)
    to their desired values, return a cube with linearly interpolated values.
    If more than one coordinate is specified, the linear interpolation will be
    carried out in sequence, thus providing n-linear interpolation
    (bi-linear, tri-linear, etc.).

    If the input cube's data is masked, the result cube will have a data
    mask interpolated to the new sample points

    .. testsetup::

        import numpy as np

    For example:

        >>> cube = iris.load_cube(iris.sample_data_path('air_temp.pp'))
        >>> sample_points = [('latitude', np.linspace(-90, 90, 10)),
        ...                  ('longitude', np.linspace(-180, 180, 20))]
        >>> iris.analysis.interpolate.linear(cube, sample_points)
        <iris 'Cube' of air_temperature / (K) (latitude: 10; longitude: 20)>

    .. note::

        By definition, linear interpolation requires all coordinates to
        be 1-dimensional.

    .. note::

        If a specified coordinate is single valued its value will be
        extrapolated to the desired sample points by assuming a gradient of
        zero.

    Args:

    * cube
        The cube to be interpolated.

    * sample_points
        List of one or more tuple pairs mapping coordinate to desired
        points to interpolate. Points may be a scalar or a numpy array
        of values.  Multi-dimensional coordinates are not supported.

    Kwargs:

    * extrapolation_mode - string - one of 'linear', 'nan' or 'error'

        * If 'linear' the point will be calculated by extending the
          gradient of closest two points.
        * If 'nan' the extrapolation point will be put as a NaN.
        * If 'error' a value error will be raised notifying of the
          attempted extrapolation.

    .. note::

        If the source cube's data, or any of its resampled coordinates,
        have an integer data type they will be promoted to a floating
        point data type in the result.

    .. deprecated:: 1.10

        The module :mod:`iris.analysis.interpolate` is deprecated.
        Please replace usage of
        :func:`iris.analysis.interpolate.linear`
        with :meth:`iris.cube.Cube.interpolate` using the scheme
        :class:`iris.analysis.Linear`.

    """
    if isinstance(sample_points, dict):
        sample_points = list(sample_points.items())

    # catch the case where a user passes a single (coord/name, value) pair rather than a list of pairs
    if sample_points and not (isinstance(sample_points[0], collections.Container) and not isinstance(sample_points[0], six.string_types)):
        raise TypeError('Expecting the sample points to be a list of tuple pairs representing (coord, points), got a list of %s.' % type(sample_points[0]))

    scheme = Linear(extrapolation_mode)
    return cube.interpolate(sample_points, scheme)
