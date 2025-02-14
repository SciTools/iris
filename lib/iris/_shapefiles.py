# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

from __future__ import annotations

import importlib
from itertools import product
import sys
from typing import overload
import warnings

from affine import Affine
import numpy as np
from pyproj import CRS, Transformer
import rasterio.features as rfeatures
import rasterio.transform as rtransform
import rasterio.warp as rwarp
import shapely
import shapely.errors
import shapely.geometry as sgeom
import shapely.ops as sops

import iris
from iris.warnings import IrisDefaultingWarning, IrisUserWarning

if "iris.cube" in sys.modules:
    import iris.cube
if "iris.analysis.cartography" in sys.modules:
    import iris.analysis.cartography


@overload
def create_shapefile_mask(
    geometry: shapely.Geometry,
    geometry_crs: cartopy.crs | CRS,
    cube: iris.cube.Cube,
    all_touched: bool = False,
    invert: bool = False,
) -> np.array: ...
def create_shapefile_mask(
    geometry: shapely.Geometry,
    geometry_crs: cartopy.crs | CRS,
    cube: iris.cube.Cube,
    **kwargs,
) -> np.array:
    """Make a mask for a cube from a shape.

    Get the mask of the intersection between the
    given shapely geometry and cube with x/y DimCoords.
    Can take a minimum weight and evaluate area overlaps instead.

    Transforming is performed by GDAL warp.

    Parameters
    ----------
    geometry : :class:`shapely.Geometry`
    geometry_crs : :class:`cartopy.crs`, optional
        A :class:`~iris.coord_systems` object describing
        the coord_system of the shapefile. Defaults to None,
        in which case the geometry_crs is assumed to be the
        same as the :class:`iris.cube.Cube`.
    cube : :class:`iris.cube.Cube`
        A :class:`~iris.cube.Cube` which has 1d x and y coordinates.
    **kwargs :
        Additional keyword arguments to pass to `rasterio.features.geometry_mask`.
        Valid keyword arguments are:
        all_touched : bool, optional
        invert : bool, optional

    Returns
    -------
    :class:`np.array`
        An array of the shape of the x & y coordinates of the cube, with points
        to mask equal to True.

    Raises
    ------
    TypeError
        If the cube is not a :class:`~iris.cube.Cube`.
    ValueError
        If the :class:`~iris.cube.Cube` has a semi-structured model grid.

    Notes
    -----
    For best masking results, both the :class:`iris.cube.Cube` _and_ masking geometry should have a
    coordinate reference system (CRS) defined. Masking results will be most reliable
    when the :class:`iris.cube.Cube` and masking geometry have the same CRS.

    Where the :class:`iris.cube.Cube` CRS and the geometry CRS differ, the geometry will be
    transformed to the cube CRS using the pyproj library. This is a best-effort
    transformation and may not be perfect, especially for complex geometries and
    non-standard coordinate referece systems. Consult the `pyproj documentation`_ for
    more information.

    If the :class:`iris.cube.Cube` has no :class:`~iris.coord_systems`, the default GeogCS is used where
    the coordinate units are degrees. For any other coordinate units,
    the cube **must** have a :class:`~iris.coord_systems` defined.

    If a CRS is not provided for the the masking geometry, the CRS of the :class:`iris.cube.Cube` is assumed.

    Warning
    -------
    Because shape vectors are inherently Cartesian in nature, they contain no inherent
    understanding of the spherical geometry underpinning geographic coordinate systems.
    For this reason, shapefiles or shape vectors that cross the antimeridian or poles
    are not supported by this function to avoid unexpected masking behaviour.

    Shape geometries can be checked prior to masking using the :func:`is_geometry_valid`.

    See Also
    --------
    :func:`is_geometry_valid`

    .. _`pyproj documentation`: https://pyproj4.github.io/pyproj/stable/api/transformer.html#pyproj-transformer
    """
    # Check validity of geometry CRS
    is_geometry_valid(geometry, geometry_crs)

    # Check cube is a Cube
    if not isinstance(cube, iris.cube.Cube):
        if isinstance(cube, iris.cube.CubeList):
            msg = "Received CubeList object rather than Cube - \
            to mask a CubeList iterate over each Cube"
            raise TypeError(msg)
        else:
            msg = "Received non-Cube object where a Cube is expected"
            raise TypeError(msg)

    # Get cube coordinates
    y_name, x_name = _cube_primary_xy_coord_names(cube)
    # Check if cube lons units are in degrees, and if so do they exist in [0, 360] or [-180, 180]
    if (cube.coord(x_name).units.origin == "degrees") and (
        cube.coord(x_name).points.max() > 180
    ):
        # Convert to [-180, 180] domain
        cube = cube.intersection(iris.coords.CoordExtent(x_name, -180, 180))

    # Check for CRS equality and transform if necessary
    cube_crs = cube.coord_system().as_cartopy_projection()
    if not geometry_crs.equals(cube_crs):
        transform_warning_msg = "Geometry CRS does not match cube CRS. Iris will attempt to transform the geometry onto the cube CRS..."
        warnings.warn(transform_warning_msg, category=iris.warnings.IrisUserWarning)
        # Set-up transform via pyproj
        t = Transformer.from_crs(
            crs_from=geometry_crs, crs_to=cube_crs, always_xy=True
        ).transform
        # Transform geometry
        geometry = shapely.ops.transform(t, geometry)
        # Recheck geometry validity
        if not shapely.is_valid(geometry):
            msg = f"Shape geometry is invalid (not well formed): {shapely.is_valid_reason(geometry)}."
            raise TypeError(msg)

    x_points = cube.coord(x_name).points
    y_points = cube.coord(y_name).points
    dx = iris.util.regular_step(cube.coord(x_name))
    dy = iris.util.regular_step(cube.coord(y_name))
    w = len(x_points)
    h = len(y_points)

    # Define raster transform based on cube
    # This maps the geometry domain onto the cube domain
    trans = Affine.translation(x_points[0] - dx / 2, y_points[0] - dy / 2)
    scale = Affine.scale(dx, dy)
    tr = trans * scale

    # Generate mask from geometry
    mask_template = rfeatures.geometry_mask(
        geometries=shapely.get_parts(geometry),
        out_shape=(h, w),
        transform=tr,
        **kwargs,
    )

    # If cube was on circular domain, then the transformed
    # mask template needs shifting to match the cube domain
    if cube.coord(x_name).circular:
        mask_template = np.roll(mask_template, w // 2, axis=1)

    return mask_template


def is_geometry_valid(
    geometry: shapely.Geometry,
    geometry_crs: cartopy.crs,
) -> None:
    """Check the validity of the shape geometry.

    This function checks that:
    1) The geometry is a valid shapely geometry.
    2) The geometry falls within bounds equivalent to
       lon = [-180, 180] and lat = [-90, 90].
    3) The geometry does not cross the antimeridian,
        based on the assumption that the shape will
        cross the antimeridian if the difference between
        the shape bound longitudes is greater than 180.
    4) The geometry does not cross the poles.

    Parameters
    ----------
    geometry : :class:`shapely.geometry.base.BaseGeometry`
        The geometry to check.
    geometry_crs : :class:`cartopy.crs`
        The coordinate reference system of the geometry.

    Returns
    -------
    None if the geometry is valid.OSTN15_NTv2OSGBtoETRS.gsb

    Raises
    ------
    TypeError
        If the geometry is not a valid shapely geometry.
    ValueError
        If the geometry is not valid for the given coordinate system.
        This most likely occurs when the geometry coordinates are not within the bounds of the
        geometry coordinates reference system.
    ValueError
        If the geometry crosses the antimeridian.
    ValueError
        If the geometry crosses the poles.

    Examples
    --------
    >>> from shapely.geometry import box
    >>> from pyproj import CRS
    >>> from iris._shapefiles import is_geometry_valid

    Create a valid geometry covering Canada, and check
    its validity for the WGS84 coordinate system:

    >>> canada = box(-143.5,42.6,-37.8,84.0)
    >>> wgs84 = CRS.from_epsg(4326)
    >>> is_geometry_valid(canada, wgs84)

    The function returns silently as the geometry is valid.

    Now create an invalid geometry covering the Bering Sea,
    and check its validity for the WGS84 coordinate system.

    >>> bering_sea = box(148.42,49.1,-138.74,73.12)
    >>> wgs84 = CRS.from_epsg(4326)
    >>> is_geometry_valid(bering_sea, wgs84) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last)
    ValueError: Geometry crossing the 180th meridian is not supported.
    """
    WGS84_crs = CRS.from_epsg(4326)

    # Check geometry is valid shapely geometry
    if not shapely.is_valid(geometry):
        msg = f"Shape geometry is invalid (not well formed): {shapely.is_valid_reason(geometry)}."
        raise TypeError(msg)

    # Check that the geometry is within the bounds of the coordinate system
    # If the geometry is not in WGS84, transform the geometry to WGS84
    # This is more reliable than transforming the lon_lat_bounds to the geometry CRS
    lon_lat_bounds = shapely.geometry.Polygon.from_bounds(
        xmin=-180.0, ymin=-90.0, xmax=180.0, ymax=90.0
    )
    if not geometry_crs.equals(WGS84_crs):
        # Make pyproj transformer
        # Transforms the input geometry to the WGS84 coordinate system
        t = Transformer.from_crs(geometry_crs, WGS84_crs, always_xy=True).transform
        geometry = shapely.ops.transform(t, geometry)

    geom_valid = lon_lat_bounds.contains(shapely.get_parts(geometry))
    if not geom_valid.all():
        msg = f"Geometry {shapely.get_parts(geometry)[~geom_valid]} is not valid for the given coordinate system {geometry_crs.to_string()}. \nCheck that your coordinates are correctly specified."
        raise ValueError(msg)

    # Check if shape crosses the 180th meridian (or equivalent)
    if bool(abs(geometry.bounds[2] - geometry.bounds[0]) > 180.0):
        msg = "Geometry crossing the antimeridian is not supported."
        raise ValueError(msg)

    # Check if the geometry crosses the poles
    npole = sgeom.Point(0, 90)
    spole = sgeom.Point(0, -90)
    if geometry.intersects(npole) or geometry.intersects(spole):
        msg = "Geometry crossing the poles is not supported."
        raise ValueError(msg)

    return


def _cube_primary_xy_coord_names(cube: iris.cube.Cube) -> tuple[str, str]:
    """Return the primary latitude and longitude coordinate names, or long names, from a cube.

    Parameters
    ----------
    cube : :class:`iris.cube.Cube`

    Returns
    -------
    tuple of str
        The names of the primary latitude and longitude coordinates.

    """
    latc = (
        cube.coords(axis="y", dim_coords=True)[0]
        if cube.coords(axis="y", dim_coords=True)
        else -1
    )
    lonc = (
        cube.coords(axis="x", dim_coords=True)[0]
        if cube.coords(axis="x", dim_coords=True)
        else -1
    )

    if -1 in (latc, lonc):
        msg = "Error retrieving 1d xy coordinates in cube: {!r}"
        raise ValueError(msg.format(cube))

    latitude = latc.name()
    longitude = lonc.name()
    return latitude, longitude
