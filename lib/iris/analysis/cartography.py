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

import math
import warnings

import pyproj
import numpy
import cartopy.crs

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
    
    # Points or bounds?
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
            
    # Project?
    if projection:
        # source projection
        source_cs = cube.coord_system("CoordSystem")
        if source_cs:
            source_proj = source_cs.cartopy_map()
        else:
            source_proj = cartopy.crs.PlateCarree()
            
        if source_proj != projection:
            # TODO: Ensure there is a test for this
            x, y = projection.transform_points(x=x, y=y, src_crs=source_projsource_proj)
    
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
