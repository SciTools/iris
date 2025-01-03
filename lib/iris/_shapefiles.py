# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

# Much of this code is originally based off the ASCEND library, developed in
# the Met Office by Chris Kent, Emilie Vanvyve, David Bentley, Joana Mendes
# many thanks to them. Converted to iris by Alex Chamberlain-Clay

from __future__ import annotations

import importlib
from itertools import product
import sys
import warnings

import numpy as np
from pyproj import CRS
import rasterio.features as rfeatures
import rasterio.transform as rtransform
import rasterio.warp as rwarp
import shapely
import shapely.errors
import shapely.geometry as sgeom
import shapely.ops

import iris
from iris.warnings import IrisDefaultingWarning, IrisUserWarning

if "iris.cube" in sys.modules:
    import iris.cube
if "iris.analysis.cartography" in sys.modules:
    import iris.analysis.cartography


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
    cube : :class:`iris.cube.Cube`
        A :class:`~iris.cube.Cube` which has 1d x and y coordinates.
    minimum_weight : float, default 0.0
        A float between 0 and 1 determining what % of a cell
        a shape must cover for the cell to remain unmasked.
        eg: 0.1 means that at least 10% of the shape overlaps the cell
        to be unmasked.
        Requires geometry to be a Polygon or MultiPolygon
        Defaults to 0.0 (eg only test intersection).
    geometry_crs : :class:`cartopy.crs`, optional
        A :class:`~iris.coord_systems` object describing
        the coord_system of the shapefile. Defaults to None,
        in which case the geometry_crs is assumed to be the
        same as the `cube`.
    **kwargs
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

    Notes
    -----
    For best masking results, both the cube _and_ masking geometry should have a
    coordinate reference system (CRS) defined. Masking results will be most reliable
    when the cube and masking geometry have the same CRS.

    If the cube has no coord_system, the default GeogCS is used where
    the coordinate units are degrees. For any other coordinate units,
    the cube **must** have a coord_system defined.

    If a CRS is not provided for the the masking geometry, the CRS of the cube is assumed.

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

    # Check for CRS equality and transform if necessary
    dst_crs = cube.coord_system().as_cartopy_projection()
    if not geometry_crs.equals(dst_crs):
        trans_geometry = rwarp.transform_geom(
            geom=geometry, src_crs=geometry_crs, dst_crs=dst_crs
        )
        geometry = sgeom.shape(trans_geometry)

    # Get cube coordinates
    y_name, x_name = _cube_primary_xy_coord_names(cube)
    # Check if cube lons exist in [0, 360] or [-180, 180]
    if cube.coord(x_name).circular:
        cube = cube.intersection(longitude=(-180, 180))
    x_points = cube.coord(x_name).points
    y_points = cube.coord(y_name).points
    w = len(x_points)
    h = len(y_points)

    # Define raster transform based on cube
    # This maps the geometry domain onto the cube domain
    # using an Affine transformation
    tr, w, h = rwarp.calculate_default_transform(
        src_crs=dst_crs,
        dst_crs=dst_crs,
        width=w,
        height=h,
        dst_width=w,
        dst_height=h,
        src_geoloc_array=(
            np.meshgrid(
                x_points,
                y_points,
                indexing="xy",
            )
        ),
    )
    # Generate mask from geometry
    mask_template = rfeatures.geometry_mask(
        geometries=shapely.get_parts(geometry), out_shape=(h, w), transform=tr
    )

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
    None if the geometry is valid.

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
        msg = "Geometry is not a valid Shapely object"
        raise TypeError(msg)

    # Check that the geometry is within the bounds of the coordinate system
    # If the geometry is not in WGS84, transform the validation bounds
    # to the geometry's CR
    lon_lat_bounds = shapely.geometry.Polygon.from_bounds(
        xmin=-180.0, ymin=-90.0, xmax=180.0, ymax=90.0
    )
    if not geometry_crs.equals(WGS84_crs):
        lon_lat_bounds = rwarp.transform_geom(
            src_crs=WGS84_crs, dst_crs=geometry_crs, geom=lon_lat_bounds
        )
        lon_lat_bounds = sgeom.shape(lon_lat_bounds)
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
    if not geometry_crs.equals(WGS84_crs):
        npole = rwarp.transform_geom(
            src_crs=WGS84_crs, dst_crs=geometry_crs, geom=npole
        )
        spole = rwarp.transform_geom(
            src_crs=WGS84_crs, dst_crs=geometry_crs, geom=spole
        )
        npole = sgeom.shape(npole)
        spole = sgeom.shape(spole)
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
