# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Various utilities and numeric transformations relevant to cartography."""

from collections import namedtuple
import copy
import warnings

import cartopy.crs as ccrs
import cartopy.img_transform
import cf_units
import dask.array as da
import numpy as np
import numpy.ma as ma

import iris.coord_systems
import iris.coords
import iris.exceptions
from iris.util import _meshgrid

from ._grid_angles import gridcell_angles, rotate_grid_vectors

# List of contents to control Sphinx autodocs.
# Unfortunately essential to get docs for the grid_angles functions.
__all__ = [
    "DistanceDifferential",
    "PartialDifferential",
    "area_weights",
    "cosine_latitude_weights",
    "get_xy_contiguous_bounded_grids",
    "get_xy_grids",
    "gridcell_angles",
    "project",
    "rotate_grid_vectors",
    "rotate_pole",
    "rotate_winds",
    "unrotate_pole",
    "wrap_lons",
]

# This value is used as a fall-back if the cube does not define the earth
DEFAULT_SPHERICAL_EARTH_RADIUS = 6367470
# TODO: This should not be necessary, as CF is always in meters
DEFAULT_SPHERICAL_EARTH_RADIUS_UNIT = cf_units.Unit("m")
# Distance differentials for coordinate systems at specified locations
DistanceDifferential = namedtuple("DistanceDifferential", "dx1 dy1 dx2 dy2")
# Partial differentials between coordinate systems
PartialDifferential = namedtuple("PartialDifferential", "dx1 dy1")


def wrap_lons(lons, base, period):
    """Wrap longitude values into the range between base and base+period.

    Parameters
    ----------
    lons :
    base :
    period :

    Examples
    --------
    .. testsetup::

        import numpy as np
        from iris.analysis.cartography import wrap_lons

    ::

        >>> print(wrap_lons(np.array([185, 30, -200, 75]), -180, 360))
        [-175.   30.  160.   75.]

    Notes
    -----
    This function maintains laziness when called; it does not realise data.
    See more at :doc:`/userguide/real_and_lazy_data`.
    """
    # It is important to use 64bit floating precision when changing a floats
    # numbers range.
    lons = lons.astype(np.float64)
    return ((lons - base) % period) + base


def unrotate_pole(rotated_lons, rotated_lats, pole_lon, pole_lat):
    """Convert arrays of rotated-pole longitudes and latitudes to unrotated
    arrays of longitudes and latitudes. The values of ``pole_lon`` and
    ``pole_lat`` should describe the location of the rotated pole that
    describes the arrays of rotated-pole longitudes and latitudes.

    As the arrays of rotated-pole longitudes and latitudes must describe a
    rectilinear grid, the arrays of rotated-pole longitudes and latitudes must
    be of the same shape as each other.

    .. note:: Uses proj.4 to perform the conversion.

    Parameters
    ----------
    rotated_lons :
        An array of rotated-pole longitude values.
    rotated_lats :
        An array of rotated-pole latitude values.
    pole_lon :
        The longitude of the rotated pole that describes the arrays of
        rotated-pole longitudes and latitudes.
    pole_lat :
        The latitude of the rotated pole that describes the arrays of
        rotated-pole longitudes and latitudes.

    Returns
    -------
    An array of unrotated longitudes and an array of unrotated latitudes.

    Examples
    --------
    ::

        lons, lats = unrotate_pole(rotated_lons, rotated_lats, pole_lon, pole_lat)

    """
    src_proj = ccrs.RotatedGeodetic(pole_longitude=pole_lon, pole_latitude=pole_lat)
    target_proj = ccrs.Geodetic()
    res = target_proj.transform_points(x=rotated_lons, y=rotated_lats, src_crs=src_proj)
    unrotated_lon = res[..., 0]
    unrotated_lat = res[..., 1]

    return unrotated_lon, unrotated_lat


def rotate_pole(lons, lats, pole_lon, pole_lat):
    """Convert arrays of longitudes and latitudes to arrays of rotated-pole
    longitudes and latitudes. The values of ``pole_lon`` and ``pole_lat``
    should describe the rotated pole that the arrays of longitudes and
    latitudes are to be rotated onto.

    As the arrays of longitudes and latitudes must describe a rectilinear grid,
    the arrays of rotated-pole longitudes and latitudes must be of the same
    shape as each other.

    .. note:: Uses proj.4 to perform the conversion.

    Parameters
    ----------
    lons :
        An array of longitude values.
    lats :
        An array of latitude values.
    pole_lon :
        The longitude of the rotated pole that the arrays of longitudes and
        latitudes are to be rotated onto.
    pole_lat :
        The latitude of the rotated pole that the arrays of longitudes and
        latitudes are to be rotated onto.

    Returns
    -------
    An array of rotated-pole longitudes and an array of rotated-pole latitudes.

    Examples
    --------
    ::

        rotated_lons, rotated_lats = rotate_pole(lons, lats, pole_lon, pole_lat)

    """
    src_proj = ccrs.Geodetic()
    target_proj = ccrs.RotatedGeodetic(pole_longitude=pole_lon, pole_latitude=pole_lat)
    res = target_proj.transform_points(x=lons, y=lats, src_crs=src_proj)
    rotated_lon = res[..., 0]
    rotated_lat = res[..., 1]

    return rotated_lon, rotated_lat


def _get_lon_lat_coords(cube):
    def search_for_coord(coord_iterable, coord_name):
        return [coord for coord in coord_iterable if coord_name in coord.name()]

    lat_coords = search_for_coord(cube.dim_coords, "latitude") or search_for_coord(
        cube.coords(), "latitude"
    )
    lon_coords = search_for_coord(cube.dim_coords, "longitude") or search_for_coord(
        cube.coords(), "longitude"
    )
    if len(lat_coords) > 1 or len(lon_coords) > 1:
        raise ValueError(
            "Calling `_get_lon_lat_coords` with multiple same-type (i.e. dim/aux) lat or lon coords"
            " is currently disallowed"
        )
    lat_coord = lat_coords[0]
    lon_coord = lon_coords[0]
    return lon_coord, lat_coord


def _xy_range(cube, mode=None):
    """Return the x & y range of this Cube.

    Parameters
    ----------
    cube :
        The cube for which to calculate xy extents.
    mode : optional, default=None
        If the coordinate has bounds, set this to specify the
        min/max calculation.
        Set to iris.coords.POINT_MODE or iris.coords.BOUND_MODE.

    """
    # Helpful error if we have an inappropriate CoordSystem
    cs = cube.coord_system("CoordSystem")
    cs_valid_types = (
        iris.coord_systems.GeogCS,
        iris.coord_systems.RotatedGeogCS,
    )
    if (cs is not None) and not isinstance(cs, cs_valid_types):
        raise ValueError("Latlon coords cannot be found with {0}.".format(type(cs)))

    x_coord, y_coord = cube.coord(axis="X"), cube.coord(axis="Y")
    cs = cube.coord_system("CoordSystem")

    if x_coord.has_bounds() != y_coord.has_bounds():
        raise ValueError(
            "Cannot get the range of the x and y coordinates if they do "
            "not have the same presence of bounds."
        )

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

    # Get the x and y range
    if getattr(x_coord, "circular", False):
        x_range = (np.min(x), np.min(x) + x_coord.units.modulus)
    else:
        x_range = (np.min(x), np.max(x))

    y_range = (np.min(y), np.max(y))

    return (x_range, y_range)


def get_xy_grids(cube):
    """Return 2D X and Y points for a given cube.

    Parameters
    ----------
    cube :
        The cube for which to generate 2D X and Y points.

    Examples
    --------
    ::

        x, y = get_xy_grids(cube)

    Notes
    -----
    This function maintains laziness when called; it does not realise data.
    See more at :doc:`/userguide/real_and_lazy_data`.

    """
    x_coord, y_coord = cube.coord(axis="X"), cube.coord(axis="Y")

    x = x_coord.points
    y = y_coord.points

    if x.ndim == y.ndim == 1:
        # Convert to 2D.
        x, y = _meshgrid(x, y)
    elif x.ndim == y.ndim == 2:
        # They are already in the correct shape.
        pass
    else:
        raise ValueError("Expected 1D or 2D XY coords")

    return (x, y)


def get_xy_contiguous_bounded_grids(cube):
    """Return 2d arrays for x and y bounds.

    Returns array of shape (n+1, m+1).

    Parameters
    ----------
    cube : :class:`iris.cube.Cube`

    Examples
    --------
    ::

        xs, ys = get_xy_contiguous_bounded_grids(cube)

    Notes
    -----
    This function maintains laziness when called; it does not realise data.
    See more at :doc:`/userguide/real_and_lazy_data`.

    """
    x_coord, y_coord = cube.coord(axis="X"), cube.coord(axis="Y")

    x = x_coord.contiguous_bounds()
    y = y_coord.contiguous_bounds()
    x, y = _meshgrid(x, y)

    return (x, y)


def _quadrant_area(radian_lat_bounds, radian_lon_bounds, radius_of_earth):
    """Calculate spherical segment areas.

    Area weights are calculated for each lat/lon cell as:

    .. math::

        r^2 (lon_1 - lon_0) ( sin(lat_1) - sin(lat_0))

    The resulting array will have a shape of
    *(radian_lat_bounds.shape[0], radian_lon_bounds.shape[0])*

    The calculations are done at 64 bit precision and the returned array
    will be of type numpy.float64.

    Parameters
    ----------
    radian_lat_bounds :
        [n,2] array of latitude bounds (radians)
    radian_lon_bounds :
        [n,2] array of longitude bounds (radians)
    radius_of_earth :
        radius of the earth (currently assumed spherical)

    """
    # ensure pairs of bounds
    if (
        radian_lat_bounds.shape[-1] != 2
        or radian_lon_bounds.shape[-1] != 2
        or radian_lat_bounds.ndim != 2
        or radian_lon_bounds.ndim != 2
    ):
        raise ValueError("Bounds must be [n,2] array")

    # fill in a new array of areas
    radius_sqr = radius_of_earth**2
    radian_lat_64 = radian_lat_bounds.astype(np.float64)
    radian_lon_64 = radian_lon_bounds.astype(np.float64)

    ylen = np.sin(radian_lat_64[:, 1]) - np.sin(radian_lat_64[:, 0])
    xlen = radian_lon_64[:, 1] - radian_lon_64[:, 0]
    areas = radius_sqr * np.outer(ylen, xlen)

    # we use abs because backwards bounds (min > max) give negative areas.
    return np.abs(areas)


def area_weights(cube, normalize=False):
    r"""Returns an array of area weights, with the same dimensions as the cube.

    This is a 2D lat/lon area weights array, repeated over the non lat/lon
    dimensions.

    The cube must have coordinates 'latitude' and 'longitude' with bounds.

    Area weights are calculated for each lat/lon cell as:

    .. math::

        r^2 (lon_1 - lon_0) (\sin(lat_1) - \sin(lat_0))

    Currently, only supports a spherical datum.
    Uses earth radius from the cube, if present and spherical.
    Defaults to iris.analysis.cartography.DEFAULT_SPHERICAL_EARTH_RADIUS.

    Parameters
    ----------
    cube : :class:`iris.cube.Cube`
        The cube to calculate area weights for.
    normalize : bool, optional, default=False
        If False, weights are grid cell areas. If True, weights are grid
        cell areas divided by the total grid area.

    Returns
    -------
    broad_weights :

    """
    # Get the radius of the earth
    cs = cube.coord_system("CoordSystem")
    if isinstance(cs, iris.coord_systems.GeogCS):
        if cs.inverse_flattening != 0.0:
            warnings.warn(
                "Assuming spherical earth from ellipsoid.",
                category=iris.exceptions.IrisDefaultingWarning,
            )
        radius_of_earth = cs.semi_major_axis
    elif isinstance(cs, iris.coord_systems.RotatedGeogCS) and (
        cs.ellipsoid is not None
    ):
        if cs.ellipsoid.inverse_flattening != 0.0:
            warnings.warn(
                "Assuming spherical earth from ellipsoid.",
                category=iris.exceptions.IrisDefaultingWarning,
            )
        radius_of_earth = cs.ellipsoid.semi_major_axis
    else:
        warnings.warn(
            "Using DEFAULT_SPHERICAL_EARTH_RADIUS.",
            category=iris.exceptions.IrisDefaultingWarning,
        )
        radius_of_earth = DEFAULT_SPHERICAL_EARTH_RADIUS

    # Get the lon and lat coords and axes
    try:
        lon, lat = _get_lon_lat_coords(cube)
    except IndexError:
        raise ValueError(
            "Cannot get latitude/longitude coordinates from cube {!r}.".format(
                cube.name()
            )
        )

    if lat.ndim > 1:
        raise iris.exceptions.CoordinateMultiDimError(lat)
    if lon.ndim > 1:
        raise iris.exceptions.CoordinateMultiDimError(lon)

    lat_dim = cube.coord_dims(lat)
    lat_dim = lat_dim[0] if lat_dim else None

    lon_dim = cube.coord_dims(lon)
    lon_dim = lon_dim[0] if lon_dim else None

    if not (lat.has_bounds() and lon.has_bounds()):
        msg = (
            "Coordinates {!r} and {!r} must have bounds to determine "
            "the area weights.".format(lat.name(), lon.name())
        )
        raise ValueError(msg)

    # Convert from degrees to radians
    lat = lat.copy()
    lon = lon.copy()

    for coord in (lat, lon):
        if coord.units in (cf_units.Unit("degrees"), cf_units.Unit("radians")):
            coord.convert_units("radians")
        else:
            msg = (
                "Units of degrees or radians required, coordinate "
                "{!r} has units: {!r}".format(coord.name(), coord.units.name)
            )
            raise ValueError(msg)

    # Create 2D weights from bounds.
    # Use the geographical area as the weight for each cell
    ll_weights = _quadrant_area(lat.bounds, lon.bounds, radius_of_earth)

    # Normalize the weights if necessary.
    if normalize:
        ll_weights /= ll_weights.sum()

    # Now we create an array of weights for each cell. This process will
    # handle adding the required extra dimensions and also take care of
    # the order of dimensions.
    broadcast_dims = [x for x in (lat_dim, lon_dim) if x is not None]
    wshape = []
    for idim, dim in zip((0, 1), (lat_dim, lon_dim)):
        if dim is not None:
            wshape.append(ll_weights.shape[idim])
    ll_weights = ll_weights.reshape(wshape)
    broad_weights = iris.util.broadcast_to_shape(ll_weights, cube.shape, broadcast_dims)

    return broad_weights


def cosine_latitude_weights(cube):
    r"""Returns an array of latitude weights, with the same dimensions as
    the cube. The weights are the cosine of latitude.

    These are n-dimensional latitude weights repeated over the dimensions
    not covered by the latitude coordinate.

    The cube must have a coordinate with 'latitude' in the name. Out of
    range values (greater than 90 degrees or less than -90 degrees) will
    be clipped to the valid range.

    Weights are calculated for each latitude as:

    .. math::

        w_l = \cos \phi_l

    Parameters
    ----------
    cube : :class:`iris.cube.Cube`

    Examples
    --------
    Compute weights suitable for averaging type operations::

        from iris.analysis.cartography import cosine_latitude_weights
        cube = iris.load_cube(iris.sample_data_path('air_temp.pp'))
        weights = cosine_latitude_weights(cube)

    Compute weights suitable for EOF analysis (or other covariance type
    analyses)::

        import numpy as np
        from iris.analysis.cartography import cosine_latitude_weights
        cube = iris.load_cube(iris.sample_data_path('air_temp.pp'))
        weights = np.sqrt(cosine_latitude_weights(cube))

    Notes
    -----
    This function maintains laziness when called; it does not realise data.
    See more at :doc:`/userguide/real_and_lazy_data`.
    """
    # Find all latitude coordinates, we want one and only one.
    lat_coords = [coord for coord in cube.coords() if "latitude" in coord.name()]
    if len(lat_coords) > 1:
        raise ValueError("Multiple latitude coords are currently disallowed.")
    try:
        lat = lat_coords[0]
    except IndexError:
        raise ValueError(
            "Cannot get latitude coordinate from cube {!r}.".format(cube.name())
        )

    # Get the dimension position(s) of the latitude coordinate.
    lat_dims = cube.coord_dims(lat)

    # Convert to radians.
    lat = lat.copy()
    lat.convert_units("radians")

    # Compute the weights as the cosine of latitude. In some cases,
    # particularly when working in 32-bit precision, the latitude values can
    # extend beyond the allowable range of [-pi/2, pi/2] due to numerical
    # precision. We first check for genuinely out of range values, and issue a
    # warning if these are found. Then the cosine is computed and clipped to
    # the valid range [0, 1].
    threshold = np.deg2rad(0.001)  # small value for grid resolution
    if np.any(lat.points < -np.pi / 2.0 - threshold) or np.any(
        lat.points > np.pi / 2.0 + threshold
    ):
        warnings.warn(
            "Out of range latitude values will be clipped to the valid range.",
            category=iris.exceptions.IrisDefaultingWarning,
        )
    points = lat.points
    l_weights = np.cos(points).clip(0.0, 1.0)

    # Create weights for each grid point. This operation handles adding extra
    # dimensions and also the order of the dimensions.
    broadcast_dims = [x for x in lat_dims if x is not None]
    wshape = []
    for idim, dim in enumerate(lat_dims):
        if dim is not None:
            wshape.append(l_weights.shape[idim])
    l_weights = l_weights.reshape(wshape)
    broad_weights = iris.util.broadcast_to_shape(l_weights, cube.shape, broadcast_dims)

    return broad_weights


def project(cube, target_proj, nx=None, ny=None):
    """Nearest neighbour regrid to a specified target projection.

    Return a new cube that is the result of projecting a cube with 1 or 2
    dimensional latitude-longitude coordinates from its coordinate system into
    a specified projection e.g. Robinson or Polar Stereographic.
    This function is intended to be used in cases where the cube's coordinates
    prevent one from directly visualising the data, e.g. when the longitude
    and latitude are two dimensional and do not make up a regular grid.

    Parameters
    ----------
    cube : :class:`iris.cube.Cube`
        An instance of :class:`iris.cube.Cube`.
    target_proj : :class:`iris.coord_systems.CoordSystem`
        An instance of the Cartopy Projection class, or an instance of
        :class:`iris.coord_systems.CoordSystem` from which a projection
        will be obtained.
    nx : optional, default=None
        Desired number of sample points in the x direction for a domain
        covering the globe.
    ny : optional, default=None
        Desired number of sample points in the y direction for a domain
        covering the globe.

    Returns
    -------
    :class:`iris.cube.Cube`
        An instance of :class:`iris.cube.Cube` and a list describing the
        extent of the projection.

    Notes
    -----
    .. note::

        If there are both dim and aux latitude-longitude coordinates, only
        the dim coordinates will be used.

    .. note::

        This function assumes global data and will if necessary extrapolate
        beyond the geographical extent of the source cube using a nearest
        neighbour approach. nx and ny then include those points which are
        outside of the target projection.

    .. note::

        Masked arrays are handled by passing their masked status to the
        resulting nearest neighbour values.  If masked, the value in the
        resulting cube is set to 0.

    .. note::

        This function does not maintain laziness when called; it realises data.
        See more at :doc:`/userguide/real_and_lazy_data`.

    .. warning::

        This function uses a nearest neighbour approach rather than any form
        of linear/non-linear interpolation to determine the data value of each
        cell in the resulting cube. Consequently it may have an adverse effect
        on the statistics of the data e.g. the mean and standard deviation
        will not be preserved.

    .. warning::

        If the target projection is non-rectangular, e.g. Robinson, the target
        grid may include points outside the boundary of the projection. The
        latitude/longitude of such points may be unpredictable.

    """
    try:
        lon_coord, lat_coord = _get_lon_lat_coords(cube)
    except IndexError:
        raise ValueError(
            "Cannot get latitude/longitude coordinates from cube {!r}.".format(
                cube.name()
            )
        )

    if lat_coord.coord_system != lon_coord.coord_system:
        raise ValueError(
            "latitude and longitude coords appear to have "
            "different coordinates systems."
        )

    if lon_coord.units != "degrees":
        lon_coord = lon_coord.copy()
        lon_coord.convert_units("degrees")
    if lat_coord.units != "degrees":
        lat_coord = lat_coord.copy()
        lat_coord.convert_units("degrees")

    # Determine source coordinate system
    if lat_coord.coord_system is None:
        # Assume WGS84 latlon if unspecified
        warnings.warn(
            "Coordinate system of latitude and longitude "
            "coordinates is not specified. Assuming WGS84 Geodetic.",
            category=iris.exceptions.IrisDefaultingWarning,
        )
        orig_cs = iris.coord_systems.GeogCS(
            semi_major_axis=6378137.0, inverse_flattening=298.257223563
        )
    else:
        orig_cs = lat_coord.coord_system

    # Convert to cartopy crs
    source_cs = orig_cs.as_cartopy_crs()

    # Obtain coordinate arrays (ignoring bounds) and convert to 2d
    # if not already.
    source_x = lon_coord.points
    source_y = lat_coord.points
    if source_x.ndim != 2 or source_y.ndim != 2:
        source_x, source_y = _meshgrid(source_x, source_y)

    # Calculate target grid
    target_cs = None
    if isinstance(target_proj, iris.coord_systems.CoordSystem):
        target_cs = target_proj
        target_proj = target_proj.as_cartopy_projection()

    # Resolution of new grid
    if nx is None:
        nx = source_x.shape[1]
    if ny is None:
        ny = source_x.shape[0]

    target_x, target_y, extent = cartopy.img_transform.mesh_projection(
        target_proj, nx, ny
    )

    # Determine dimension mappings - expect either 1d or 2d
    if lat_coord.ndim != lon_coord.ndim:
        raise ValueError(
            "The latitude and longitude coordinates have different dimensionality."
        )

    latlon_ndim = lat_coord.ndim
    lon_dims = cube.coord_dims(lon_coord)
    lat_dims = cube.coord_dims(lat_coord)

    if latlon_ndim == 1:
        xdim = lon_dims[0]
        ydim = lat_dims[0]
    elif latlon_ndim == 2:
        if lon_dims != lat_dims:
            raise ValueError(
                "The 2d latitude and longitude coordinates "
                "correspond to different dimensions."
            )
        # If coords are 2d assume that grid is ordered such that x corresponds
        # to the last dimension (shortest stride).
        xdim = lon_dims[1]
        ydim = lon_dims[0]
    else:
        raise ValueError(
            "Expected the latitude and longitude coordinates "
            "to have 1 or 2 dimensions, got {} and "
            "{}.".format(lat_coord.ndim, lon_coord.ndim)
        )

    # Create array to store regridded data
    new_shape = list(cube.shape)
    new_shape[xdim] = nx
    new_shape[ydim] = ny
    new_data = ma.zeros(new_shape, cube.data.dtype)

    # Create iterators to step through cube data in lat long slices
    new_shape[xdim] = 1
    new_shape[ydim] = 1
    index_it = np.ndindex(*new_shape)
    if lat_coord.ndim == 1 and lon_coord.ndim == 1:
        slice_it = cube.slices([lat_coord, lon_coord])
    elif lat_coord.ndim == 2 and lon_coord.ndim == 2:
        slice_it = cube.slices(lat_coord)
    else:
        raise ValueError(
            "Expected the latitude and longitude coordinates "
            "to have 1 or 2 dimensions, got {} and "
            "{}.".format(lat_coord.ndim, lon_coord.ndim)
        )

    #    # Mask out points outside of extent in source_cs - disabled until
    #    # a way to specify global/limited extent is agreed upon and code
    #    # is generalised to handle -180 to +180, 0 to 360 and >360 longitudes.
    #    source_desired_xy = source_cs.transform_points(target_proj,
    #                                                   target_x.flatten(),
    #                                                   target_y.flatten())
    #    if np.any(source_x < 0.0) and np.any(source_x > 180.0):
    #        raise ValueError('Unable to handle range of longitude.')
    #    # This does not work in all cases e.g. lon > 360
    #    if np.any(source_x > 180.0):
    #        source_desired_x = (source_desired_xy[:, 0].reshape(ny, nx) +
    #                            360.0) % 360.0
    #    else:
    #        source_desired_x = source_desired_xy[:, 0].reshape(ny, nx)
    #    source_desired_y = source_desired_xy[:, 1].reshape(ny, nx)
    #    outof_extent_points = ((source_desired_x < source_x.min()) |
    #                           (source_desired_x > source_x.max()) |
    #                           (source_desired_y < source_y.min()) |
    #                           (source_desired_y > source_y.max()))
    #    # Make array a mask by default (rather than a single bool) to allow mask
    #    # to be assigned to slices.
    #    new_data.mask = np.zeros(new_shape)

    # Step through cube data, regrid onto desired projection and insert results
    # in new_data array
    for index, ll_slice in zip(index_it, slice_it):
        # Regrid source data onto target grid
        index = list(index)
        index[xdim] = slice(None, None)
        index[ydim] = slice(None, None)
        index = tuple(index)  # Numpy>=1.16 : index with tuple, *not* list.
        new_data[index] = cartopy.img_transform.regrid(
            ll_slice.data,
            source_x,
            source_y,
            source_cs,
            target_proj,
            target_x,
            target_y,
        )

    #    # Mask out points beyond extent
    #    new_data[index].mask[outof_extent_points] = True

    # Remove mask if it is unnecessary
    if not np.any(new_data.mask):
        new_data = new_data.data

    # Create new cube
    new_cube = iris.cube.Cube(new_data)

    # Add new grid coords
    x_coord = iris.coords.DimCoord(
        target_x[0, :],
        "projection_x_coordinate",
        units="m",
        coord_system=copy.copy(target_cs),
    )
    y_coord = iris.coords.DimCoord(
        target_y[:, 0],
        "projection_y_coordinate",
        units="m",
        coord_system=copy.copy(target_cs),
    )

    new_cube.add_dim_coord(x_coord, xdim)
    new_cube.add_dim_coord(y_coord, ydim)

    # Add resampled lat/lon in original coord system
    source_desired_xy = source_cs.transform_points(
        target_proj, target_x.flatten(), target_y.flatten()
    )
    new_lon_points = source_desired_xy[:, 0].reshape(ny, nx)
    new_lat_points = source_desired_xy[:, 1].reshape(ny, nx)
    new_lon_coord = iris.coords.AuxCoord(
        new_lon_points,
        standard_name="longitude",
        units="degrees",
        coord_system=orig_cs,
    )
    new_lat_coord = iris.coords.AuxCoord(
        new_lat_points,
        standard_name="latitude",
        units="degrees",
        coord_system=orig_cs,
    )
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
        warnings.warn(
            "Discarding coordinates that share dimensions with {} and {}: {}".format(
                lat_coord.name(),
                lon_coord.name(),
                [coord.name() for coord in discarded_coords],
            ),
            category=iris.exceptions.IrisIgnoringWarning,
        )

    # TODO handle derived coords/aux_factories

    # Copy metadata across
    new_cube.metadata = cube.metadata

    return new_cube, extent


def _transform_xy(crs_from, x, y, crs_to):
    """Shorthand function to transform 2d points between coordinate
    reference systems.

    Parameters
    ----------
    crs_from, crs_to : :class:`cartopy.crs.Projection`
        The coordinate reference systems.
    x, y : array
        point locations defined in 'crs_from'.

    Returns
    -------
    x, y
        Arrays of locations defined in 'crs_to'.

    """
    pts = crs_to.transform_points(crs_from, x, y)
    return pts[..., 0], pts[..., 1]


def _inter_crs_differentials(crs1, x, y, crs2):
    """Calculate coordinate partial differentials from crs1 to crs2.

    Returns dx2/dx1, dy2/dx1, dx2/dy1 and dy2/dy1, at given locations.

    Parameters
    ----------
    crs1, crs2 : :class:`cartopy.crs.Projection`
        The coordinate systems, "from" and "to".
    x, y : array
        Point locations defined in 'crs1'.

    Returns
    -------
    arrays
        (dx2/dx1, dy2/dx1, dx2/dy1, dy2/dy1) at given locations. Each
        element of this tuple will be the same shape as the 'x' and 'y'
        arrays and will be the partial differentials between the two systems.

    """
    # Get locations in target crs.
    crs2_x, crs2_y = _transform_xy(crs1, x, y, crs2)

    # Define small x-deltas in the source crs.
    VECTOR_DELTAS_FACTOR = 360000.0  # Empirical factor to obtain small delta.
    delta_x = (crs1.x_limits[1] - crs1.x_limits[0]) / VECTOR_DELTAS_FACTOR
    delta_x = delta_x * np.ones(x.shape)
    eps = 1e-9
    # Reverse deltas where we would otherwise step outside the valid range.
    invalid_dx = x + delta_x > crs1.x_limits[1] - eps
    delta_x[invalid_dx] = -delta_x[invalid_dx]
    # Calculate the transformed point with x = x + dx.
    crs2_x2, crs2_y2 = _transform_xy(crs1, x + delta_x, y, crs2)
    # Form differentials wrt dx.
    dx2_dx = (crs2_x2 - crs2_x) / delta_x
    dy2_dx = (crs2_y2 - crs2_y) / delta_x

    # Define small y-deltas in the source crs.
    delta_y = (crs1.y_limits[1] - crs1.y_limits[0]) / VECTOR_DELTAS_FACTOR
    delta_y = delta_y * np.ones(y.shape)
    # Reverse deltas where we would otherwise step outside the valid range.
    invalid_dy = y + delta_y > crs1.y_limits[1] - eps
    delta_y[invalid_dy] = -delta_y[invalid_dy]
    # Calculate the transformed point with y = y + dy.
    crs2_x2, crs2_y2 = _transform_xy(crs1, x, y + delta_y, crs2)
    # Form differentials wrt dy.
    dx2_dy = (crs2_x2 - crs2_x) / delta_y
    dy2_dy = (crs2_y2 - crs2_y) / delta_y

    return dx2_dx, dy2_dx, dx2_dy, dy2_dy


def _crs_distance_differentials(crs, x, y):
    """Calculate d(distance) / d(x) and ... / d(y).

    Calculate d(distance) / d(x) and ... / d(y) for a coordinate
    reference system at specified locations.

    Parameters
    ----------
    crs : :class:`cartopy.crs.Projection`
        The coordinate reference system.
    x, y : array
        Locations at which to calculate the differentials,
        defined in 'crs' coordinate reference system.

    Returns
    -------
    (abs(ds/dx), abs(ds/dy))
        Numerically approximated partial differentials,
        i.e. scaling factors between changes in distance and changes in
        coordinate values.

    """
    # Make a true-latlon coordinate system for distance calculations.
    crs_latlon = ccrs.Geodetic(globe=crs.globe)
    # Transform points to true-latlon (just to get the true latitudes).
    _, true_lat = _transform_xy(crs, x, y, crs_latlon)
    # Get coordinate differentials w.r.t. true-latlon.
    dlon_dx, dlat_dx, dlon_dy, dlat_dy = _inter_crs_differentials(crs, x, y, crs_latlon)
    # Calculate effective scalings of X and Y coordinates.
    lat_factor = np.cos(np.deg2rad(true_lat)) ** 2
    ds_dx = np.sqrt(dlat_dx * dlat_dx + dlon_dx * dlon_dx * lat_factor)
    ds_dy = np.sqrt(dlat_dy * dlat_dy + dlon_dy * dlon_dy * lat_factor)
    return ds_dx, ds_dy


def _transform_distance_vectors(u_dist, v_dist, ds, dx2, dy2):
    """Transform distance vectors from one coordinate reference system to
    another, preserving magnitude and physical direction.

    Parameters
    ----------
    u_dist, v_dist : array
        Components of each vector along the x and y directions of the source
        crs at each location.
    ds : `DistanceDifferential`
        Distance differentials for the source and the target crs at specified
        locations.
    dx2, dy2 : `PartialDifferential`
        Partial differentials from the source to the target crs.

    Returns
    -------
    tuple
        (ut_dist, vt_dist): Tuple of arrays containing the vector components
        along the x and y directions of the target crs at each location.

    """
    # Scale input distance vectors --> source-coordinate differentials.
    u1, v1 = u_dist / ds.dx1, v_dist / ds.dy1
    # Transform vectors into the target system.
    u2 = dx2.dx1 * u1 + dx2.dy1 * v1
    v2 = dy2.dx1 * u1 + dy2.dy1 * v1
    # Rescale output coordinate vectors --> target distance vectors.
    u2_dist, v2_dist = u2 * ds.dx2, v2 * ds.dy2

    return u2_dist, v2_dist


def _transform_distance_vectors_tolerance_mask(src_crs, x, y, tgt_crs, ds, dx2, dy2):
    """Return a mask that can be applied to data array to mask elements
    where the magnitude of vectors are not preserved due to numerical
    errors introduced by the transformation between coordinate systems.

    Parameters
    ----------
    src_crs : `cartopy.crs.Projection`
        The source coordinate reference systems.
    x, y : array
        Locations of each vector defined in 'src_crs'.
    tgt_crs : `cartopy.crs.Projection`
        The target coordinate reference systems.
    ds : `DistanceDifferential`
        Distance differentials for src_crs and tgt_crs at specified locations
    dx2, dy2 : `PartialDifferential`
        Partial differentials from src_crs to tgt_crs.

    Returns
    -------
    2d boolean array that is the same shape as x and y.

    """
    if x.shape != y.shape:
        raise ValueError(
            "Arrays do not have matching shapes. "
            "x.shape is {}, y.shape is {}.".format(x.shape, y.shape)
        )
    ones = np.ones(x.shape)
    zeros = np.zeros(x.shape)
    u_one_t, v_zero_t = _transform_distance_vectors(ones, zeros, ds, dx2, dy2)
    u_zero_t, v_one_t = _transform_distance_vectors(zeros, ones, ds, dx2, dy2)
    # Squared magnitudes should be equal to one within acceptable tolerance.
    # A value of atol=2e-3 is used, which masks any magnitude changes >0.5%
    #  (approx percentage - based on experimenting).
    sqmag_1_0 = u_one_t**2 + v_zero_t**2
    sqmag_0_1 = u_zero_t**2 + v_one_t**2
    mask = np.logical_not(
        np.logical_and(
            np.isclose(sqmag_1_0, ones, atol=2e-3),
            np.isclose(sqmag_0_1, ones, atol=2e-3),
        )
    )
    return mask


def rotate_winds(u_cube, v_cube, target_cs):
    r"""Transform wind vectors to a different coordinate system.

    The input cubes contain U and V components parallel to the local X and Y
    directions of the input grid at each point.

    The output cubes contain the same winds, at the same locations, but
    relative to the grid directions of a different coordinate system.
    Thus in vector terms, the magnitudes will always be the same, but the
    angles can be different.

    The outputs retain the original horizontal dimension coordinates, but
    also have two 2-dimensional auxiliary coordinates containing the X and
    Y locations in the target coordinate system.

    Parameters
    ----------
    u_cube :
        An instance of :class:`iris.cube.Cube` that contains the x-component
        of the vector.
    v_cube :
        An instance of :class:`iris.cube.Cube` that contains the y-component
        of the vector.
    target_cs :
        An instance of :class:`iris.coord_systems.CoordSystem` that specifies
        the new grid directions.

    Returns
    -------
    (u', v') tuple of :class:`iris.cube.Cube`
        A (u', v') tuple of :class:`iris.cube.Cube` instances that are the u
        and v components in the requested target coordinate system.
        The units are the same as the inputs.

    Notes
    -----
    .. note::

        The U and V values relate to distance, with units such as 'm s-1'.
        These are not the same as coordinate vectors, which transform in a
        different manner.

    .. note::

        The names of the output cubes are those of the inputs, prefixed with
        'transformed\_' (e.g. 'transformed_x_wind').

    .. note::

            This function does not maintain laziness when called; it realises data.
            See more at :doc:`/userguide/real_and_lazy_data`.

    .. warning::

        Conversion between rotated-pole and non-rotated systems can be
        expressed analytically.  However, this function always uses a numerical
        approach. In locations where this numerical approach does not preserve
        magnitude to an accuracy of 0.1%, the corresponding elements of the
        returned cubes will be masked.

    """
    # Check u_cube and v_cube have the same shape. We iterate through
    # the u and v cube slices which relies on the shapes matching.
    if u_cube.shape != v_cube.shape:
        msg = (
            "Expected u and v cubes to have the same shape. "
            "u cube has shape {}, v cube has shape {}."
        )
        raise ValueError(msg.format(u_cube.shape, v_cube.shape))

    # Check the u_cube and v_cube have the same x and y coords.
    msg = (
        "Coordinates differ between u and v cubes. Coordinate {!r} from "
        "u cube does not equal coordinate {!r} from v cube."
    )
    if u_cube.coord(axis="x") != v_cube.coord(axis="x"):
        raise ValueError(
            msg.format(u_cube.coord(axis="x").name(), v_cube.coord(axis="x").name())
        )
    if u_cube.coord(axis="y") != v_cube.coord(axis="y"):
        raise ValueError(
            msg.format(u_cube.coord(axis="y").name(), v_cube.coord(axis="y").name())
        )

    # Check x and y coords have the same coordinate system.
    x_coord = u_cube.coord(axis="x")
    y_coord = u_cube.coord(axis="y")
    if x_coord.coord_system != y_coord.coord_system:
        msg = (
            "Coordinate systems of x and y coordinates differ. "
            "Coordinate {!r} has a coord system of {!r}, but coordinate "
            "{!r} has a coord system of {!r}."
        )
        raise ValueError(
            msg.format(
                x_coord.name(),
                x_coord.coord_system,
                y_coord.name(),
                y_coord.coord_system,
            )
        )

    # Convert from iris coord systems to cartopy CRSs to access
    # transform functionality. Use projection as cartopy
    # transform_vectors relies on x_limits and y_limits.
    if x_coord.coord_system is not None:
        src_crs = x_coord.coord_system.as_cartopy_projection()
    else:
        # Default to Geodetic (but actually use PlateCarree as a
        # projection is needed).
        src_crs = ccrs.PlateCarree()
    target_crs = target_cs.as_cartopy_projection()

    # Check the number of dimensions of the x and y coords is the same.
    # Subsequent logic assumes either both 1d or both 2d.
    x = x_coord.points
    y = y_coord.points
    if x.ndim != y.ndim or x.ndim > 2 or y.ndim > 2:
        msg = (
            "x and y coordinates must have the same number of dimensions "
            "and be either 1D or 2D. The number of dimensions are {} and "
            "{}, respectively.".format(x.ndim, y.ndim)
        )
        raise ValueError(msg)

    # Check the dimension mappings match between u_cube and v_cube.
    if u_cube.coord_dims(x_coord) != v_cube.coord_dims(x_coord):
        raise ValueError(
            "Dimension mapping of x coordinate differs between u and v cubes."
        )
    if u_cube.coord_dims(y_coord) != v_cube.coord_dims(y_coord):
        raise ValueError(
            "Dimension mapping of y coordinate differs between u and v cubes."
        )
    x_dims = u_cube.coord_dims(x_coord)
    y_dims = u_cube.coord_dims(y_coord)

    # Convert points to 2D, if not already, and determine dims.
    if x.ndim == y.ndim == 1:
        x, y = _meshgrid(x, y)
        dims = (y_dims[0], x_dims[0])
    else:
        dims = x_dims

    # Transpose x, y 2d arrays to match the order in cube's data
    # array so that x, y and the sliced data all line up.
    if dims[0] > dims[1]:
        x = x.transpose()
        y = y.transpose()

    # Create resulting cubes - produce lazy output data if at least
    # one input cube has lazy data
    lazy_output = u_cube.has_lazy_data() or v_cube.has_lazy_data()
    if lazy_output:
        ut_cube = u_cube.copy(data=da.empty_like(u_cube.lazy_data()))
        vt_cube = v_cube.copy(data=da.empty_like(v_cube.lazy_data()))
    else:
        ut_cube = u_cube.copy()
        vt_cube = v_cube.copy()
    ut_cube.rename("transformed_{}".format(u_cube.name()))
    vt_cube.rename("transformed_{}".format(v_cube.name()))

    # Get distance scalings for source crs.
    ds_dx1, ds_dy1 = _crs_distance_differentials(src_crs, x, y)

    # Get distance scalings for target crs.
    x2, y2 = _transform_xy(src_crs, x, y, target_crs)
    ds_dx2, ds_dy2 = _crs_distance_differentials(target_crs, x2, y2)

    ds = DistanceDifferential(ds_dx1, ds_dy1, ds_dx2, ds_dy2)

    # Calculate coordinate partial differentials from source crs to target crs.
    dx2_dx1, dy2_dx1, dx2_dy1, dy2_dy1 = _inter_crs_differentials(
        src_crs, x, y, target_crs
    )

    dx2 = PartialDifferential(dx2_dx1, dx2_dy1)
    dy2 = PartialDifferential(dy2_dx1, dy2_dy1)

    # Calculate mask based on preservation of magnitude.
    mask = _transform_distance_vectors_tolerance_mask(
        src_crs, x, y, target_crs, ds, dx2, dy2
    )
    apply_mask = mask.any()
    if apply_mask:
        # Make masked arrays to accept masking.
        if lazy_output:
            ut_cube = ut_cube.copy(data=da.ma.empty_like(ut_cube.core_data()))
            vt_cube = vt_cube.copy(data=da.ma.empty_like(vt_cube.core_data()))
        else:
            ut_cube.data = ma.asanyarray(ut_cube.data)
            vt_cube.data = ma.asanyarray(vt_cube.data)

    # Project vectors with u, v components one horiz slice at a time and
    # insert into the resulting cubes.
    shape = list(u_cube.shape)
    for dim in dims:
        shape[dim] = 1
    ndindex = np.ndindex(*shape)
    for index in ndindex:
        index = list(index)
        for dim in dims:
            index[dim] = slice(None, None)
        index = tuple(index)
        u = u_cube.core_data()[index]
        v = v_cube.core_data()[index]
        ut, vt = _transform_distance_vectors(u, v, ds, dx2, dy2)
        if apply_mask:
            if lazy_output:
                ut = da.ma.masked_array(ut, mask=mask)
                vt = da.ma.masked_array(vt, mask=mask)
            else:
                ut = ma.asanyarray(ut)
                ut[mask] = ma.masked
                vt = ma.asanyarray(vt)
                vt[mask] = ma.masked
        ut_cube.core_data()[index] = ut
        vt_cube.core_data()[index] = vt

    # Calculate new coords of locations in target coordinate system.
    xyz_tran = target_crs.transform_points(src_crs, x, y)
    xt = xyz_tran[..., 0].reshape(x.shape)
    yt = xyz_tran[..., 1].reshape(y.shape)

    # Transpose xt, yt 2d arrays to match the dim order
    # of the original x an y arrays - i.e. undo the earlier
    # transpose (if applied).
    if dims[0] > dims[1]:
        xt = xt.transpose()
        yt = yt.transpose()

    xt_coord = iris.coords.AuxCoord(
        xt, standard_name="projection_x_coordinate", coord_system=target_cs
    )
    yt_coord = iris.coords.AuxCoord(
        yt, standard_name="projection_y_coordinate", coord_system=target_cs
    )
    # Set units based on coord_system.
    if isinstance(
        target_cs,
        (iris.coord_systems.GeogCS, iris.coord_systems.RotatedGeogCS),
    ):
        xt_coord.units = yt_coord.units = "degrees"
    else:
        xt_coord.units = yt_coord.units = "m"

    ut_cube.add_aux_coord(xt_coord, dims)
    ut_cube.add_aux_coord(yt_coord, dims)
    vt_cube.add_aux_coord(xt_coord.copy(), dims)
    vt_cube.add_aux_coord(yt_coord.copy(), dims)

    return ut_cube, vt_cube
