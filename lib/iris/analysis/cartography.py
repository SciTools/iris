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
Various utilities and numeric transformations relevant to cartography.

"""
import itertools
import math
import warnings

import cartopy
import cartopy.img_transform
from mpl_toolkits.basemap import pyproj
import numpy

import iris.analysis
import iris.coords
import iris.coord_systems
import iris.exceptions
import iris.unit


# This value is used as a fall-back if the cube does not define the earth 
DEFAULT_SPHERICAL_EARTH_RADIUS = 6367470
# TODO: This should not be necessary, as CF is always in meters
DEFAULT_SPHERICAL_EARTH_RADIUS_UNIT = iris.unit.Unit('m')


def wrap_lons(lons, base, period):
    """
    Returns given longitudes in the range between base and base + period.
    
    ::
        wrapped_lon = wrap_lons(185, -180, 360)
         
    """
    return ((lons - base + period * 2) % period) + base


def _proj4(pole_lon, pole_lat):
    proj4_params = {'proj': 'ob_tran', 'o_proj': 'latlon', 'o_lon_p': 0,
        'o_lat_p': pole_lat, 'lon_0': 180 + pole_lon,
        'to_meter': math.degrees(1)}
    from mpl_toolkits.basemap import pyproj
    proj = pyproj.Proj(proj4_params)
    return proj


def unrotate_pole(rotated_lons, rotated_lats, pole_lon, pole_lat):
    """
    Given an array of lons and lats with a rotated pole, convert to unrotated lons and lats.
     
    .. note:: Uses proj.4 to perform the conversion.
    
    """
    proj4_wrapper = _proj4(pole_lon, pole_lat)
    # NB. pyproj modifies the proj.4 init string and adds unit=meter
    # which breaks our to_meter=57...
    # So we have to do the radian-degree correction explicitly
    d = math.degrees(1)
    std_lons, std_lats = proj4_wrapper(rotated_lons / d, rotated_lats / d, inverse=True)
    return std_lons, std_lats


def rotate_pole(lons, lats, pole_lon, pole_lat):
    """
    Given an array of lons and lats, convert to lons and lats with a rotated pole.
        
    .. note:: Uses proj.4 to perform the conversion.
    
    """
    proj4_wrapper = _proj4(pole_lon, pole_lat)
    rlons, rlats = proj4_wrapper(lons, lats)
    # NB. pyproj modifies the proj.4 init string and adds unit=meter
    # which breaks our to_meter=57...
    # So we have to do the radian-degree correction explicitly
    d = math.degrees(1)
    rlons *= d
    rlats *= d
    return rlons, rlats


def _get_lat_lon_coords(cube):
    lat_coords = filter(lambda coord: "latitude" in coord.name(), cube.coords())
    lon_coords = filter(lambda coord: "longitude" in coord.name(), cube.coords())
    if len(lat_coords) > 1 or len(lon_coords) > 1:
        raise ValueError("Calling _get_lat_lon_coords() with multiple lat or lon coords is currently disallowed")
    lat_coord = lat_coords[0]
    lon_coord = lon_coords[0]
    return (lat_coord, lon_coord)


def xy_range(cube, mode=None, projection=None):
    """
    Return the x & y range of this Cube.
    
    Args:
    
        * cube - The cube for which to calculate xy extents.
        
    Kwargs:
    
        * mode - If the coordinate has bounds, use the mode keyword to specify the
                 min/max calculation (iris.coords.POINT_MODE or iris.coords.BOUND_MODE).

        * projection - Calculate the xy range in an alternative projection. 
            
    """
    # Helpful error if we have an inappropriate CoordSystem
    cs = cube.coord_system("CoordSystem")
    if cs is not None and not isinstance(cs, (iris.coord_systems.GeogCS, iris.coord_systems.RotatedGeogCS)):
        raise ValueError("Latlon coords cannot be found with {0}.".format(type(cs)))
    
    x_coord, y_coord = cube.coord(axis="X"), cube.coord(axis="Y")
    cs = cube.coord_system('CoordSystem')
    
    if x_coord.has_bounds() != x_coord.has_bounds():
        raise ValueError('Cannot get the range of the x and y coordinates if they do '
                         'not have the same presence of bounds.')
    
    if x_coord.has_bounds():
        if mode not in [iris.coords.POINT_MODE, iris.coords.BOUND_MODE]:
            raise ValueError('When the coordinate has bounds, please specify "mode".')
        _mode = mode
    else:
        _mode = iris.coords.POINT_MODE
    
    # Get the x and y grids
    if isinstance(cs, iris.coord_systems.RotatedGeogCS):
        if _mode == iris.coords.POINT_MODE:
            x, y = get_xy_grids(cube)
        else:
            x, y = get_xy_contiguous_bounded_grids(cube)
    else:
        if _mode == iris.coords.POINT_MODE:
            x = x_coord.points
            y = y_coord.points
        else:
            x = x_coord.bounds
            y = y_coord.bounds
            
    if projection:
        # source projection
        source_cs = cube.coord_system("CoordSystem")
        if source_cs is not None:
            source_proj = source_cs.as_cartopy_projection()
        else:
            #source_proj = cartopy.crs.PlateCarree()
            raise Exception('Unknown source coordinate system')
            
        if source_proj != projection:
            # TODO: Ensure there is a test for this
            x, y = projection.transform_points(x=x, y=y, src_crs=source_proj)
    
    # Get the x and y range
    if getattr(x_coord, 'circular', False):
        x_range = (numpy.min(x), numpy.min(x) + x_coord.units.modulus)
    else: 
        x_range = (numpy.min(x), numpy.max(x))

    y_range = (numpy.min(y), numpy.max(y))

    return (x_range, y_range)


def get_xy_grids(cube):
    """
    Return 2d x and y points in the native coordinate system.
    ::
    
        x, y = get_xy_grids(cube)
    
    """
    x_coord, y_coord = cube.coord(axis="X"), cube.coord(axis="Y")
    cs = cube.coord_system('CoordSystem')
    
    x = x_coord.points
    y = y_coord.points
    
    # Convert to 2 x 2d grid of data
    x, y = numpy.meshgrid(x, y)

    return (x, y)


def get_xy_contiguous_bounded_grids(cube):
    """
    Return 2d lat and lon bounds.
    
    Returns array of shape (n+1, m+1).
    ::
    
        lats, lons = cs.get_lat_lon_bounded_grids()
    
    """
    x_coord, y_coord = cube.coord(axis="X"), cube.coord(axis="Y")
    cs = cube.coord_system('CoordSystem')
    
    x = x_coord.contiguous_bounds()
    y = y_coord.contiguous_bounds()
    x, y = numpy.meshgrid(x, y)

    return (x, y)


def _quadrant_area(radian_colat_bounds, radian_lon_bounds, radius_of_earth):
    """Calculate spherical segment areas.

    - radian_colat_bounds    -- [n,2] array of colatitude bounds (radians)
    - radian_lon_bounds      -- [n,2] array of longitude bounds (radians)
    - radius_of_earth        -- radius of the earth, (currently assumed spherical)

    Area weights are calulated for each lat/lon cell as:

        .. math::
        
            r^2 (lon_1 - lon_0) ( cos(colat_0) - cos(colat_1))
    
    The resulting array will have a shape of *(radian_colat_bounds.shape[0], radian_lon_bounds.shape[0])*

    """
    #ensure pairs of bounds
    if radian_colat_bounds.shape[-1] != 2 or radian_lon_bounds.shape[-1] != 2 or \
       radian_colat_bounds.ndim != 2 or radian_lon_bounds.ndim != 2:
        raise ValueError("Bounds must be [n,2] array")

    #fill in a new array of areas
    radius_sqr = radius_of_earth ** 2
    areas = numpy.ndarray((radian_colat_bounds.shape[0], radian_lon_bounds.shape[0]))
    # we use abs because backwards bounds (min > max) give negative areas.
    for j in range(radian_colat_bounds.shape[0]):
        areas[j, :] = [(radius_sqr * math.cos(radian_colat_bounds[j, 0]) * (radian_lon_bounds[i, 1] - radian_lon_bounds[i, 0])) - \
                      (radius_sqr * math.cos(radian_colat_bounds[j, 1]) * (radian_lon_bounds[i, 1] - radian_lon_bounds[i, 0]))   \
                      for i in range(radian_lon_bounds.shape[0])] 
        
    return numpy.abs(areas)


def area_weights(cube):
    """
    Returns an array of area weights, with the same dimensions as the cube.
    
    This is a 2D lat/lon area weights array, repeated over the non lat/lon dimensions.
    
    The cube must have coordinates 'latitide' and 'longitude' with contiguous bounds. 
    
    Area weights are calculated for each lat/lon cell as:

        .. math::
        
            r^2 cos(lat_0) (lon_1 - lon_0) - r^2 cos(lat_1) (lon_1 - lon_0)

    Currently, only supports a spherical datum.
    Uses earth radius from the cube, if present and spherical.
    Defaults to iris.analysis.cartography.DEFAULT_SPHERICAL_EARTH_RADIUS.    
    
    """
    # Get the radius of the earth
    cs = cube.coord_system("CoordSystem")
    if isinstance(cs, iris.coord_systems.GeogCS):
        if cs.inverse_flattening != 0.0:
            warnings.warn("Assuming spherical earth from ellipsoid.")
        radius_of_earth = cs.semi_major_axis
    elif isinstance(cs, iris.coord_systems.RotatedGeogCS) and cs.ellipsoid is not None:
        if cs.ellipsoid.inverse_flattening != 0.0:
            warnings.warn("Assuming spherical earth from ellipsoid.")
        radius_of_earth = cs.ellipsoid.semi_major_axis
    else:
        warnings.warn("Using DEFAULT_SPHERICAL_EARTH_RADIUS.")
        radius_of_earth = DEFAULT_SPHERICAL_EARTH_RADIUS
        
    # Get the lon and lat coords and axes
    lat, lon = _get_lat_lon_coords(cube)
    
    if lat.ndim > 1:
        raise iris.exceptions.CoordinateMultiDimError(lat)
    if lon.ndim > 1:
        raise iris.exceptions.CoordinateMultiDimError(lon)
    
    lat_dim = cube.coord_dims(lat)
    lat_dim = lat_dim[0] if lat_dim else None
    
    lon_dim = cube.coord_dims(lon)
    lon_dim = lon_dim[0] if lon_dim else None
    
    # Ensure they have contiguous bounds
    if (not lat.is_contiguous()) or (not lon.is_contiguous()):
        raise ValueError("Currently need contiguous bounds to calculate area weights")
    
    # Convert from degrees to radians
    lat = lat.unit_converted('radians')
    lon = lon.unit_converted('radians')
    
    # Create 2D weights from bounds
    if lat.has_bounds() and lon.has_bounds():
        # Use the geographical area as the weight for each cell
        # Convert latitudes to co-latitude. I.e from -90 --> +90  to  0 --> pi
        ll_weights = _quadrant_area(lat.bounds+numpy.pi/2., lon.bounds, radius_of_earth)
    
    # Create 2D weights from points
    else:
        raise iris.exceptions.NotYetImplementedError("Point-based weighting algorithm not yet identified")

    # Do we need to transpose?
    # Quadrant_area always returns shape (y,x)
    # Does the cube have them the other way round?
    if lon_dim < lat_dim:
        ll_weights = ll_weights.transpose()
    
    # Now we create an array of weights for each cell.
        
    # First, get the non latlon shape
    other_dims_shape = numpy.array(cube.shape)
    other_dims_shape[lat_dim] = 1
    other_dims_shape[lon_dim] = 1
    other_dims_array = numpy.ones(other_dims_shape)
     
    # Create the broadcast object from the weights array and the 'other dims',
    # to match the shape of the cube.
    broad_weights = ll_weights * other_dims_array

    return broad_weights


def project(cube, target_proj, nx=None, ny=None):
    """
    Return a new cube that is the result of projecting a cube from its
    coordinate system into a specified projection e.g. Robinson or Polar
    Stereographic. This function is intended to be used in cases where the
    cube's coordinates prevent one from directly visualising the data, e.g.
    when the longitude and latitude are two dimensional and do not make up
    a regular grid.

    Args:
        * cube
            An instance of :class:`iris.cube.Cube`.
        * target_proj
            An instance of the Cartopy Projection class, or an instance of
            :class:`iris.coord_systems.CoordSystem` from which a projection
            will be obtained.
    Kwargs:
        * nx
            Desired number of sample points in the x direction.
        * ny
            Desired number of sample points in the y direction.

    Returns:
        An instance of :class:`iris.cube.Cube` and a list describing the
        extent of the projection.

    .. note::
        This function uses a nearest neighbour approach rather than any form
        of linear/non-linear interpolation to determine the data value of each
        cell in the resulting cube. Consequently it may have an adverse effect
        on the statistics of the data e.g. the mean and standard deviation
        will not be preserved.

    .. note::
        This function assumes global data and will if necessary extrapolate
        beyond the geographical extent of the source cube using a nearest
        neighbour approach.

    """
    try:
        lat_coord, lon_coord = _get_lat_lon_coords(cube)
    except IndexError:
        raise ValueError('Cannot get latitude/longitude ' \
                         'coordinates from cube {!r}.'.format(cube.name()))

    if lat_coord.coord_system != lon_coord.coord_system:
        raise ValueError('latitude and longitude coords appear to have' \
                         'different coordinates systems.')

    if lon_coord.units != 'degrees':
        lon_coord = lon_coord.unit_converted('degrees')
    if lat_coord.units != 'degrees':
        lat_coord = lat_coord.unit_converted('degrees')

    # Determine source coordinate system
    if lat_coord.coord_system is None:
        # Assume WGS84 latlon if unspecified
        warnings.warn('Coordinate system of latitude and longitude '\
                      'coordinates is not specified. Assuming WGS84 Geodetic.')
        orig_cs = iris.coord_systems.GeogCS(semi_major_axis=6378137.0,
                                            inverse_flattening=298.257223563)
    else:
        orig_cs = lat_coord.coord_system

    # Convert to cartopy crs
    source_cs = orig_cs.as_cartopy_crs()

    # Obtain coordinate arrays (ignoring bounds) and convert to 2d
    # if not already.
    source_x = lon_coord.points
    source_y = lat_coord.points
    if source_x.ndim != 2 or source_y.ndim !=2:
        source_x, source_y = numpy.meshgrid(source_x, source_y)

    # Calculate target grid
    if isinstance(target_proj, iris.coord_systems.CoordSystem):
        target_proj = target_proj.as_cartopy_projection()

    # Resolution of new grid
    if nx == None:
        nx = source_x.shape[1]
    if ny == None:
        ny = source_x.shape[0]

    target_x, target_y, extent = cartopy.img_transform.mesh_projection(target_proj,
                                                                       nx, ny)

    # Determine dimension mappings - expect either 1d or 2d
    if lat_coord.ndim != lon_coord.ndim:
        raise ValueError("The latitude and longitude coordinates have "
                         "different dimensionality.")

    latlon_ndim = lat_coord.ndim
    lon_dims = cube.coord_dims(lon_coord)
    lat_dims = cube.coord_dims(lat_coord)

    if latlon_ndim == 1:
        xdim = lon_dims[0]
        ydim = lat_dims[0]
    elif latlon_ndim == 2:
        if lon_dims != lat_dims:
            raise ValueError("The 2d latitude and longitude coordinates "
                             "correspond to different dimensions.")
        # If coords are 2d assume that grid is ordered such that x corresponds
        # to the last dimension (shortest stride).
        xdim = lon_dims[1]
        ydim = lon_dims[0]
    else:
        raise ValueError('Expected the latitude and longitude coordinates '\
                         'to have 1 or 2 dimensions, got {} and '\
                         '{}.'.format(lat_coord.ndim, lon_coord.ndim))

    # Create array to store regridded data
    new_shape = list(cube.shape)
    new_shape[xdim] = nx
    new_shape[ydim] = ny
    new_data = numpy.ma.zeros(new_shape, cube.data.dtype)

    # Create iterators to step through cube data in lat long slices
    new_shape[xdim] = 1
    new_shape[ydim] = 1
    index_it = numpy.ndindex(*new_shape)
    if lat_coord.ndim == 1 and lon_coord.ndim ==1:
        slice_it = cube.slices([lat_coord, lon_coord])
    elif lat_coord.ndim == 2 and lon_coord.ndim == 2:
        slice_it = cube.slices(lat_coord)
    else:
        raise ValueError('Expected the latitude and longitude coordinates '\
                         'to have 1 or 2 dimensions, got {} and '\
                         '{}.'.format(lat_coord.ndim, lon_coord.ndim))

    ## Mask out points outside of extent in source_cs - disabled until
    ## a way to specify global/limited extent is agreed upon and code
    ## is generalised to handle -180 to +180, 0 to 360 and >360 longitudes.
    #source_desired_xy = source_cs.transform_points(target_proj,
    #                                               target_x.flatten(),
    #                                               target_y.flatten())
    #if numpy.any(source_x < 0.0) and numpy.any(source_x > 180.0):
    #    raise ValueError('Unable to handle range of longitude.')
    ## This does not work in all cases e.g. lon > 360
    #if numpy.any(source_x > 180.0):
    #    source_desired_x = (source_desired_xy[:, 0].reshape(ny, nx) + 360.0) % 360.0
    #else:
    #    source_desired_x = source_desired_xy[:, 0].reshape(ny, nx)
    #source_desired_y = source_desired_xy[:, 1].reshape(ny, nx)
    #outof_extent_points = ((source_desired_x < source_x.min()) |
    #                       (source_desired_x > source_x.max()) |
    #                       (source_desired_y < source_y.min()) |
    #                       (source_desired_y > source_y.max()))
    ## Make array a mask by default (rather than a single bool) to allow mask to be
    ## assigned to slices.
    #new_data.mask = numpy.zeros(new_shape)

    # Step through cube data, regrid onto desired projection and insert results
    # in new_data array
    for index, ll_slice in itertools.izip(index_it, slice_it):
        # Regrid source data onto target grid
        index = list(index)
        index[xdim] = slice(None, None)
        index[ydim] = slice(None, None)
        new_data[index] = cartopy.img_transform.regrid(ll_slice.data,
                                                       source_x, source_y,
                                                       source_cs, target_proj,
                                                       target_x, target_y)

        ## Mask out points beyond extent
        #new_data[index].mask[outof_extent_points] = True

    # Remove mask if it is unnecessary
    if not numpy.any(new_data.mask):
        new_data = new_data.data

    # Create new cube
    new_cube = iris.cube.Cube(new_data)

    # Add new grid coords
    x_coord = iris.coords.DimCoord(target_x[0, :], 'projection_x_coordinate')
    y_coord = iris.coords.DimCoord(target_y[:, 0], 'projection_y_coordinate')
    new_cube.add_dim_coord(x_coord, xdim)
    new_cube.add_dim_coord(y_coord, ydim)

    # Add resampled lat/lon in original coord system
    source_desired_xy = source_cs.transform_points(target_proj,
                                                   target_x.flatten(),
                                                   target_y.flatten())
    new_lon_points = source_desired_xy[:, 0].reshape(ny, nx)
    new_lat_points = source_desired_xy[:, 1].reshape(ny, nx)
    new_lon_coord = iris.coords.AuxCoord(new_lon_points,
                                         standard_name='longitude',
                                         units='degrees',
                                         coord_system=orig_cs)
    new_lat_coord = iris.coords.AuxCoord(new_lat_points,
                                         standard_name='latitude',
                                         units='degrees',
                                         coord_system=orig_cs)
    new_cube.add_aux_coord(new_lon_coord, [ydim, xdim])
    new_cube.add_aux_coord(new_lat_coord, [ydim, xdim])

    coords_to_ignore = set()
    coords_to_ignore.update(cube.coords(contains_dimension=xdim))
    coords_to_ignore.update(cube.coords(contains_dimension=ydim))
    for coord in cube.dim_coords:
        if coord not in coords_to_ignore:
            new_cube.add_dim_coord(coord.copy(), cube.coord_dims(coord))
    for coord in cube.aux_coords:
        if coord not in coords_to_ignore:
            new_cube.add_aux_coord(coord.copy(), cube.coord_dims(coord))
    discarded_coords = coords_to_ignore.difference([lat_coord, lon_coord])
    if discarded_coords:
        warnings.warn('Discarding coordinates that share dimensions with ' \
                      '{} and {}: {}'.format(lat_coord.name(),
                                             lon_coord.name(),
                                             [coord.name() for
                                              coord in discarded_coords]))

    # TODO handle derived coords/aux_factories

    # Copy metadata across
    new_cube.metadata = cube.metadata

    # Record transform in cube's history
    new_cube.add_history('Converted from {} to {}'.format(type(source_cs).__name__,
                                                          type(target_proj).__name__))

    return new_cube, extent
