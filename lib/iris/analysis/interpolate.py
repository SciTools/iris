# (C) British Crown Copyright 2010 - 2012, Met Office
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
Interpolation and re-gridding routines.

See also: :mod:`NumPy <numpy>`, and :ref:`SciPy <scipy:modindex>`.

"""
import collections
import warnings
from copy import deepcopy

import numpy
import scipy
import scipy.spatial
from scipy.interpolate import interpolate

import iris.cube
import iris.coord_systems
import iris.coords
import iris.exceptions


def _ll_to_cart(lon, lat):
    # Based on cartopy.img_transform.ll_to_cart()
    x = numpy.sin(numpy.deg2rad(90 - lat)) * numpy.cos(numpy.deg2rad(lon))
    y = numpy.sin(numpy.deg2rad(90 - lat)) * numpy.sin(numpy.deg2rad(lon))
    z = numpy.cos(numpy.deg2rad(90 - lat))
    return (x, y, z)

def _cartesian_sample_points(sample_points, sample_point_coord_names):
    # Replace geographic latlon with cartesian xyz.
    # Generates coords suitable for nearest point calculations with scipy.spatial.cKDTree.
    #
    # Input:
    # sample_points[coord][datum] : list of sample_positions for each datum, formatted for fast use of _ll_to_cart()
    # sample_point_coord_names[coord] : list of n coord names
    #
    # Output:
    # list of [x,y,z,t,etc] positions, formatted for kdtree

    # Find lat and lon coord indices
    i_lat = i_lon = None
    i_non_latlon = range(len(sample_point_coord_names))
    for i, name in enumerate(sample_point_coord_names):
        if "latitude" in name:  
            i_lat = i
            i_non_latlon.remove(i_lat)
        if "longitude" in name:  
            i_lon = i
            i_non_latlon.remove(i_lon)

    if i_lat is None or i_lon is None:
        return sample_points.transpose()
    
    num_points = len(sample_points[0])
    cartesian_points = [None] * num_points

    # Get the point coordinates without the latlon
    for p in range(num_points):
        cartesian_points[p] = [sample_points[c][p] for c in i_non_latlon]

    # Add cartesian xyz coordinates from latlon
    x, y, z = _ll_to_cart(sample_points[i_lon], sample_points[i_lat])
    for p in range(num_points):
        cartesian_point = cartesian_points[p]
        cartesian_point.append(x[p])
        cartesian_point.append(y[p])
        cartesian_point.append(z[p])
        
    return cartesian_points


def nearest_neighbour_indices(cube, sample_points):
    """
    Returns the indices to select the data value(s) closest to the given coordinate point values.

    The sample_points mapping does not have to include coordinate values corresponding to all data
    dimensions. Any dimensions unspecified will default to a full slice.

    For example:
        >>> cube = iris.load_strict(iris.sample_data_path('PP', 'globClim1', 'theta.pp'))
        >>> iris.analysis.interpolate.nearest_neighbour_indices(cube, [('latitude', 0), ('longitude', 10)])
        (slice(None, None, None), 72, 5)
        >>> iris.analysis.interpolate.nearest_neighbour_indices(cube, [('latitude', 0)])
        (slice(None, None, None), 72, slice(None, None, None))
    
    Args:

    * cube:
        An :class:`iris.cube.Cube`.
    * sample_points
        A list of tuple pairs mapping coordinate instances or unique coordinate names in the cube to point values.

    Returns:
        The tuple of indices which will select the point in the cube closest to the supplied coordinate values.

    """
    if isinstance(sample_points, dict):
        warnings.warn('Providing a dictionary to specify points is deprecated. Please provide a list of (coordinate, values) pairs.')
        sample_points = sample_points.items()
    
    if sample_points:
        try:
            coord, values = sample_points[0]
        except ValueError:
            raise ValueError('Sample points must be a list of (coordinate, value) pairs. Got %r.' % sample_points)
    
    points = []
    for coord, values in sample_points:
        if isinstance(coord, basestring):
            coord = cube.coord(coord)
        else:
            coord = cube.coord(coord=coord)
        points.append((coord, values))
    sample_points = points
    
    # Build up a list of indices to span the cube.
    indices = [slice(None, None)] * cube.data.ndim
    
    # Build up a dictionary which maps the cube's data dimensions to a list (which will later
    # be populated by coordinates in the sample points list)
    dim_to_coord_map = {}
    for i in range(cube.data.ndim):
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


def _nearest_neighbour_indices_ndcoords(cube, sample_point, cache=None):
    """
    See documentation for :func:`iris.analysis.interpolate.nearest_neighbour_indices`.
    
    This function is adapted for points sampling a multi-dimensional coord,
    and can currently only do nearest neighbour interpolation.
    
    Because this function can be slow for multidimensional coordinates,
    a 'cache' dictionary can be provided by the calling code. 
    
    """
    
    # Developer notes:
    # A "sample space cube" is made which only has the coords and dims we are sampling on.
    # We get the nearest neighbour using this sample space cube.
    
    if isinstance(sample_point, dict):
        warnings.warn('Providing a dictionary to specify points is deprecated. Please provide a list of (coordinate, values) pairs.')
        sample_point = sample_point.items()
    
    if sample_point:
        try:
            coord, value = sample_point[0]
        except ValueError:
            raise ValueError('Sample points must be a list of (coordinate, value) pairs. Got %r.' % sample_point)
    
    # Convert names to coords in sample_point
    point = []
    ok_coord_ids = set(map(id, cube.dim_coords + cube.aux_coords))
    for coord, value in sample_point:
        if isinstance(coord, basestring):
            coord = cube.coord(coord)
        else:
            coord = cube.coord(coord=coord)
        if id(coord) not in ok_coord_ids:
            msg = ('Invalid sample coordinate {!r}: derived coordinates are'
                   ' not allowed.'.format(coord.name()))
            raise ValueError(msg)
        point.append((coord, value))
        
    # Reformat sample_point for use in _cartesian_sample_points(), below.
    sample_point = numpy.array([[value] for coord, value in point])
    sample_point_coords = [coord for coord, value in point]
    sample_point_coord_names = [coord.name() for coord, value in point]

    # Which dims are we sampling?
    sample_dims = set()
    for coord in sample_point_coords:
        for dim in cube.coord_dims(coord):
            sample_dims.add(dim)
    sample_dims = sorted(list(sample_dims))

    # Extract a sub cube that lives in just the sampling space.
    sample_space_slice = [0] * cube.data.ndim
    for sample_dim in sample_dims:
        sample_space_slice[sample_dim] = slice(None, None)
    sample_space_slice = tuple(sample_space_slice)
    sample_space_cube = cube[sample_space_slice]
    
    #...with just the sampling coords
    for coord in sample_space_cube.coords():
        if not coord.name() in sample_point_coord_names:
            sample_space_cube.remove_coord(coord)
            
    # Order the sample point coords according to the sample space cube coords
    sample_space_coord_names = [coord.name() for coord in sample_space_cube.coords()]
    new_order = [sample_space_coord_names.index(name) for name in sample_point_coord_names]
    sample_point = numpy.array([sample_point[i] for i in new_order])
    sample_point_coord_names = [sample_point_coord_names[i] for i in new_order]
    
    # Convert the sample point to cartesian coords.
    # If there is no latlon within the coordinate there will be no change.
    # Otherwise, geographic latlon is replaced with cartesian xyz.
    cartesian_sample_point = _cartesian_sample_points(sample_point, sample_point_coord_names)[0]

    sample_space_coords = sample_space_cube.dim_coords + sample_space_cube.aux_coords
    sample_space_coords_and_dims = [(coord, sample_space_cube.coord_dims(coord)) for coord in sample_space_coords]

    if cache is not None and cube in cache:
        kdtree = cache[cube]
    else:
        # Create a "sample space position" for each datum: sample_space_data_positions[coord_index][datum_index]
        sample_space_data_positions = numpy.empty((len(sample_space_coords_and_dims), sample_space_cube.data.size), dtype=float)
        for d, ndi in enumerate(numpy.ndindex(sample_space_cube.data.shape)):
            for c, (coord, coord_dims) in enumerate(sample_space_coords_and_dims):
                # Index of this datum along this coordinate (could be nD). 
                keys = tuple(ndi[ind] for ind in coord_dims) if coord_dims else slice(None, None)
                # Position of this datum along this coordinate.
                sample_space_data_positions[c][d] = coord.points[keys]

        # Convert to cartesian coordinates. Flatten for kdtree compatibility.
        cartesian_space_data_coords = _cartesian_sample_points(sample_space_data_positions, sample_point_coord_names)

        # Get the nearest datum index to the sample point. This is the goal of the function.
        kdtree = scipy.spatial.cKDTree(cartesian_space_data_coords)

    cartesian_distance, datum_index = kdtree.query(cartesian_sample_point)
    sample_space_ndi = numpy.unravel_index(datum_index, sample_space_cube.data.shape)

    # Turn sample_space_ndi into a main cube slice.
    # Map sample cube to main cube dims and leave the rest as a full slice.
    main_cube_slice = [slice(None, None)] * cube.data.ndim
    for sample_coord, sample_coord_dims in sample_space_coords_and_dims:
        # Find the coord in the main cube
        main_coord = cube.coord(sample_coord.name())
        main_coord_dims = cube.coord_dims(main_coord)
        # Mark the nearest data index/indices with respect to this coord
        for sample_i, main_i in zip(sample_coord_dims, main_coord_dims):
            main_cube_slice[main_i] = sample_space_ndi[sample_i]


    # Update cache
    if cache is not None:
        cache[cube] = kdtree

    return tuple(main_cube_slice)


def extract_nearest_neighbour(cube, sample_points):
    """
    Returns a new cube using data value(s) closest to the given coordinate point values.

    The sample_points mapping does not have to include coordinate values corresponding to all data
    dimensions. Any dimensions unspecified will default to a full slice.

    For example:
        >>> cube = iris.load_strict(iris.sample_data_path('PP', 'globClim1', 'theta.pp'))
        >>> iris.analysis.interpolate.extract_nearest_neighbour(cube, [('latitude', 0), ('longitude', 10)])
        <iris 'Cube' of air_potential_temperature (model_level_number: 38)>
        >>> iris.analysis.interpolate.extract_nearest_neighbour(cube, [('latitude', 0)])
        <iris 'Cube' of air_potential_temperature (model_level_number: 38; longitude: 192)>
    
    Args:

    * cube:
        An :class:`iris.cube.Cube`.
    * sample_points
        A list of tuple pairs mapping coordinate instances or unique coordinate names in the cube to point values.

    Returns:
        A cube that represents uninterpolated data as near to the given points as possible.

    """
    return cube[nearest_neighbour_indices(cube, sample_points)]


def nearest_neighbour_data_value(cube, sample_points):
    """
    Returns the data value closest to the given coordinate point values.

    The sample_points mapping must include coordinate values corresponding to all data
    dimensions.

    For example:
        >>> cube = iris.load_strict(iris.sample_data_path('PP', 'globClim1', 'theta.pp'))
        >>> iris.analysis.interpolate.nearest_neighbour_data_value(cube, [('latitude', 0), ('longitude', 10), ('model_level_number', 1)])
        299.35156
        >>> iris.analysis.interpolate.nearest_neighbour_data_value(cube, [('latitude', 0)])
        Traceback (most recent call last):
        ...
        ValueError: The sample points [('latitude', 0)] was not specific enough to return a single value from the cube.
    
    
    Args:

    * cube:
        An :class:`iris.cube.Cube`.
    * sample_points
        A list of tuple pairs mapping coordinate instances or unique coordinate names in the cube to point values.

    Returns:
        The tuple of indices which will select the point in the cube closest to the supplied coordinate values.

    """
    indices = nearest_neighbour_indices(cube, sample_points)
    for ind in indices:
        if isinstance(ind, slice):
            raise ValueError('The sample points given (%s) were not specific enough to return a '
                             'single value from the cube.' % sample_points)
    
    return cube.data[indices]


def regrid(source_cube, grid_cube, mode='bilinear', **kwargs):
    """
    Returns a new cube with values derived from the source_cube on the horizontal grid specified
    by the grid_cube.

    Fundamental input requirements:
        1) Both cubes must have a HorizontalCS.
        2) The source 'x' and 'y' coordinates must not share data dimensions with any other coordinates.
       
    In addition, the algorithm currently used requires:
        3) Both CS instances must be compatible:
            i.e. of the same type, with the same attribute values, and with compatible coordinates.
        4) No new data dimensions can be created.

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

    """
    # Condition 1
    source_cs = source_cube.coord_system(iris.coord_systems.HorizontalCS)
    grid_cs = grid_cube.coord_system(iris.coord_systems.HorizontalCS)
    if source_cs is None or grid_cs is None:
        raise ValueError("The source and grid cubes must contain a HorizontalCS.")

    # Condition 2: We can only have one x coordinate and one y coordinate with the source HorizontalCS, and those coordinates 
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
        compatible = (source_x._as_defn() == grid_x._as_defn() and
                      source_y._as_defn() == grid_y._as_defn())
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

    new_data = numpy.empty(new_shape, dtype=source_cube.data.dtype)

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


def regrid_to_max_resolution(cubes, **kwargs):
    """
    Returns all the cubes re-gridded to the highest horizontal resolution.
    
    Horizontal resolution is defined by the number of grid points/cells covering the horizontal plane.
    See :func:`iris.analysis.interpolation.regrid` regarding mode of interpolation. 

    Args:

    * cubes:
        An iterable of :class:`iris.cube.Cube` instances.

    Returns:
        A list of new :class:`iris.cube.Cube` instances.

    """
    # TODO: This could be significantly improved for readability and functionality.
    resolution = lambda cube_: (cube_.shape[cube_.coord_dims(cube_.coord(axis="x"))[0]]) * (cube_.shape[cube_.coord_dims(cube_.coord(axis="y"))[0]])
    grid_cube = max(cubes, key=resolution)
    return [cube.regridded(grid_cube, **kwargs) for cube in cubes]


def linear(cube, sample_points, extrapolation_mode='linear'):
    """
    Return a cube of the linearly interpolated points given the desired
    sample points.
    
    Given a list of tuple pairs mapping coordinates to their desired
    values, return a cube with linearly interpolated values. If more
    than one coordinate is specified, the linear interpolation will be
    carried out in sequence, thus providing n-linear interpolation
    (bi-linear, tri-linear, etc.).
    
    .. note::
        By definition, linear interpolation requires all coordinates to
        be 1-dimensional.
    
    Args:
    
    * cube
        The cube to be interpolated.
        
    * sample_points
        List of one or more tuple pairs mapping coordinate to desired
        points to interpolate. Points may be a scalar or a numpy array
        of values.
    
    Kwargs:
    
    * extrapolation_mode - string - one of 'linear', 'nan' or 'error'
    
        * If 'linear' the point will be calculated by extending the
          gradient of closest two points.
        * If 'nan' the extrapolation point will be put as a NAN.
        * If 'error' a value error will be raised notifying of the
          attempted extrapolation.
    
    .. note::
        The datatype of the resultant cube's data and coordinates will
        updated to the data type of the incoming cube.
     
    """
    if not isinstance(cube, iris.cube.Cube):
        raise ValueError('Expecting a cube instance, got %s' % type(cube))

    if isinstance(sample_points, dict):
        warnings.warn('Providing a dictionary to specify points is deprecated. Please provide a list of (coordinate, values) pairs.')
        sample_points = sample_points.items()

    # catch the case where a user passes a single (coord/name, value) pair rather than a list of pairs
    if sample_points and not (isinstance(sample_points[0], collections.Container) and not isinstance(sample_points[0], basestring)):
        raise TypeError('Expecting the sample points to be a list of tuple pairs representing (coord, points), got a list of %s.' % type(sample_points[0]))
    
    points = []
    for (coord, values) in sample_points:
        if isinstance(coord, basestring):
            coord = cube.coord(coord)
        else:
            coord = cube.coord(coord=coord)
        points.append((coord, values))
    sample_points = points

    if len(sample_points) == 0:
        raise ValueError('Expecting a non-empty list of coord value pairs, got %r.' % sample_points)

    if cube.data.dtype.kind == 'i':
        raise ValueError("Cannot linearly interpolate a cube which has integer type data. Consider casting the "
                         "cube's data to floating points in order to continue.")

    bounds_error = (extrapolation_mode == 'error')

    # Handle an over-specified points_dict or a specification which does not describe a data dimension
    data_dimensions_requested = []
    for coord, values in sample_points:
        if coord.ndim > 1:
            raise ValueError('Cannot linearly interpolate over %s as it is multi-dimensional.' % coord.name())
        data_dim = cube.coord_dims(coord)
        if not data_dim:
            raise ValueError('Requested a point over a coordinate which does not describe a dimension (%s).' % coord.name())
        else:
            data_dim = data_dim[0]
        if data_dim in data_dimensions_requested:
            raise ValueError('Requested a point which over specifies a dimension: (%s). ' % coord.name())
        data_dimensions_requested.append(data_dim)

    # Iterate over all of the requested keys in the given points_dict calling this routine repeatedly.
    if len(sample_points) > 1:
        result = cube
        for coord, cells in sample_points:
            result = linear(result, [(coord, cells)], extrapolation_mode=extrapolation_mode)
        return result
    
    else:
        # take the single coordinate name and associated cells from the dictionary
        coord, requested_points = sample_points[0]
        
        requested_points = numpy.array(requested_points, dtype=cube.data.dtype)
        
        # build up indices so that we can quickly subset the original cube to be of the desired size
        new_cube_slices = [slice(None, None)] * cube.data.ndim
        # get this coordinate's index position (which we have already tested is not None)
        data_dim = cube.coord_dims(coord)[0]
        
        if requested_points.ndim > 0:
            # we want the interested dimension to be of len(requested_points)
            new_cube_slices[data_dim] = tuple([0] * len(requested_points))
        else:
            new_cube_slices[data_dim] = 0
        
        # Subset the original cube to get an appropriately sized cube.
        # NB. This operation will convert any DimCoords on the dimension
        # being sliced into AuxCoords. This removes the value of their
        # `circular` flags, and there's nowhere left to put it.
        new_cube = cube[tuple(new_cube_slices)]

        # now that we have got a cube at the desired location, get the data.
        if getattr(coord, 'circular', False):
            coord_slice_in_cube = [slice(None, None)] * cube.data.ndim
            coord_slice_in_cube[data_dim] = slice(0, 1)
            points = numpy.append(coord.points, coord.points[0] + numpy.array(coord.units.modulus or 0, dtype=coord.dtype))
            data = numpy.append(cube.data, cube.data[tuple(coord_slice_in_cube)], axis=data_dim)
        else:
            points = coord.points
            data = cube.data
        
        if len(points) <= 1:
            raise ValueError('Cannot linearly interpolate a coordinate (%s) with one point.' % coord.name())
        
        monotonic, direction = iris.util.monotonic(points, return_direction=True)
        if not monotonic:
            raise ValueError('Unable to linearly interpolate this cube as the coordinate "%s" is not monotonic' % coord.name())
        
        # if the coord is monotonic decreasing, then we need to flip it as SciPy's interp1d is expecting monotonic increasing.
        if direction == -1:
            points = iris.util.reverse(points, axes=0)
            data = iris.util.reverse(data, axes=data_dim)
        
        # limit the datatype of the outcoming points to be the datatype of the cube's data
        # (otherwise, interp1d will up-cast an incoming pair. i.e. (int32, float32) -> float64)
        if points.dtype.num < data.dtype.num:
            points = points.astype(data.dtype)
        
        # Now that we have subsetted the original cube, we must update all coordinates on the data dimension.
        for shared_dim_coord in cube.coords(contains_dimension=data_dim):
            if shared_dim_coord.ndim != 1:
                raise iris.exceptions.NotYetImplementedError('Linear interpolation of multi-dimensional coordinates.')
            
            new_coord = new_cube.coord(coord=shared_dim_coord)
            new_coord.bounds = None
            
            if shared_dim_coord._as_defn() != coord._as_defn():
                shared_coord_points = shared_dim_coord.points
                if getattr(coord, 'circular', False):
                    mod_val = numpy.array(shared_dim_coord.units.modulus or 0, dtype=shared_coord_points.dtype)
                    shared_coord_points = numpy.append(shared_coord_points, shared_coord_points[0] + mod_val)
                
                # If the coordinate which we were interpolating over was monotonic decreasing,
                # we need to flip this coordinate's values
                if direction == -1:
                    shared_coord_points = iris.util.reverse(shared_coord_points, axes=0)
                
                coord_points = points
                
                if shared_coord_points.dtype.num < data.dtype.num:
                    shared_coord_points = shared_coord_points.astype(data.dtype)
                
                interpolator = interpolate.interp1d(coord_points, shared_coord_points,
                                                    kind='linear', bounds_error=bounds_error)
                
                if extrapolation_mode == 'linear':
                    interpolator = iris.util.Linear1dExtrapolator(interpolator)
                
                new_coord.points = interpolator(requested_points)
            else:
                new_coord.points = requested_points
                    
        # now we can go ahead and interpolate the data
        interpolator = interpolate.interp1d(points, data, axis=data_dim,
                                            kind='linear', copy=False,
                                            bounds_error=bounds_error)
        
        if extrapolation_mode == 'linear':
            interpolator = iris.util.Linear1dExtrapolator(interpolator)
        
        new_cube.data = interpolator(requested_points)
        
        return new_cube
