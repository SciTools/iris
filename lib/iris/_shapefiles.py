# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

from __future__ import annotations

from itertools import product
import sys
import warnings

from affine import Affine
import cartopy.crs as ccrs
import numpy as np
from pyproj import CRS, Transformer
import rasterio.features as rfeatures
import shapely
import shapely.geometry as sgeom
import shapely.ops

import iris
from iris.exceptions import IrisError
from iris.warnings import IrisUserWarning
# from iris.util import mutually_exclusive_keywords

if "iris.cube" in sys.modules:
    import iris.cube
if "iris.analysis.cartography" in sys.modules:
    import iris.analysis.cartography


# @mutually_exclusive_keywords("minimum_weight", "all_touched")
def create_shapefile_mask(
    geometry: shapely.Geometry,
    geometry_crs: ccrs.CRS | CRS,
    cube: iris.cube.Cube,
    minimum_weight: float = 0.0,
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
    minimum_weight : float, optional
        The minimum weight of the geometry to be included in the mask.
        If the weight is less than this value, the geometry will not be
        included in the mask. Defaults to 0.0.

    Other Parameters
    ----------------
    all_touched : bool, optional
        If True, all pixels touched by the geometry will be included in the mask.
        If False, only pixels fully covered by the geometry will be included in the mask.
        Defaults to True.
    invert : bool, optional
        If True, the mask will be inverted, so that pixels not covered by the geometry
        will be included in the mask. Defaults to False.

    Returns
    -------
    :class:`np.array`
        An array of the shape of the x & y coordinates of the cube, with points
        to mask equal to True.

    Raises
    ------
    TypeError
        If the cube is not a :class:`~iris.cube.Cube`.
    IrisError
        If the cube does not have a coordinate reference system defined.
    TypeError
        If the geometry is not a valid shapely geometry.
    ValueError
        If the :class:`~iris.cube.Cube` has a semi-structured model grid.
    ValueError
        If the minimum_weight is not between 0.0 and 1.0.
    ValueError
        If the minimum_weight is greater than 0.0 and all_touched is True.

    Warns
    -----
    IrisUserWarning
        If the geometry CRS does not match the cube CRS, and the geometry is transformed
        to the cube CRS using pyproj.

    Notes
    -----
    For best masking results, both the :class:`iris.cube.Cube` _and_ masking geometry should have a
    coordinate reference system (CRS) defined. Masking results will be most reliable
    when the :class:`iris.cube.Cube` and masking geometry have the same CRS.

    Where the :class:`iris.cube.Cube` CRS and the geometry CRS differ, the geometry will be
    transformed to the cube CRS using the pyproj library. This is a best-effort
    transformation and may not be perfect, especially for complex geometries and
    non-standard coordinate reference systems. Consult the `pyproj documentation`_ for
    more information.

    If a CRS is not provided for the the masking geometry, the CRS of the :class:`iris.cube.Cube` is assumed.

    Note that `minimum_weight` and `all_touched` are mutually exclusive options: an error will be raised if
    a `minimum_weight` > 0 *and* `all_touched` is set to `True`. This is because
    `all_touched=True` is equivalent to `minimum_weight=0`.

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

    # Check cube coordinate system
    if not cube.coord_system():
        err_msg = "Cube does not have a coordinate references system defined. For reliable results we recommend you add a coordinate system to your cube."
        raise IrisError(err_msg)

    # Check minimum_weight is within range
    if (minimum_weight < 0.0) or (minimum_weight > 1.0):
        msg = "Minimum weight must be between 0.0 and 1.0"
        raise ValueError(msg)

    # Check compatibility of function arguments
    # all_touched and minimum_weight are mutually exclusive
    if (minimum_weight > 0) and (kwargs.get("all_touched") is True):
        msg = "Cannot use minimum_weight > 0.0 with all_touched=True."
        raise ValueError(msg)

    # Get cube coordinates
    x_coord = cube.coord(axis="x", dim_coords=True)
    y_coord = cube.coord(axis="y", dim_coords=True)
    # Check if cube lons units are in degrees, and if so do they exist in [0, 360] or [-180, 180]
    if (x_coord.units.origin == "degrees") and (x_coord.points.max() > 180):
        # Convert to [-180, 180] domain
        cube = cube.intersection(iris.coords.CoordExtent(x_coord.name(), -180, 180))

    if geometry_crs is None:
        # If no geometry CRS is provided, assume it is the same as the cube CRS
        geometry_crs = cube.coord_system().as_cartopy_projection()

    # Check for CRS equality and transform if necessary
    cube_crs = cube.coord_system().as_cartopy_projection()
    if not geometry_crs.equals(cube_crs):
        transform_warning_msg = "Geometry CRS does not match cube CRS. Iris will attempt to transform the geometry onto the cube CRS..."
        warnings.warn(transform_warning_msg, category=IrisUserWarning)
        geometry = _transform_geometry(
            geometry=geometry,
            geometry_crs=geometry_crs,
            cube_crs=cube_crs,
        )

    w = len(x_coord.points)
    h = len(y_coord.points)
    # Mask by weight if minimum_weight > 0.0
    if minimum_weight > 0:
        mask_template = _get_weighted_mask(
            geometry=geometry,
            cube=cube,
            minimum_weight=minimum_weight,
        )
    else:
        if (minimum_weight == 0) and (kwargs.get("all_touched") is None):
            # For speed, if minimum_weight is 0, then
            # we can use the geometry_mask function directly
            # This is equivalent to all_touched=True
            all_touched = True
        elif (minimum_weight == 0) and (kwargs.get("all_touched") is not None):
            if not isinstance(kwargs.get("all_touched"), bool):
                msg = "`all_touched` kwarg must be True or False."
                raise TypeError(msg)
            all_touched = kwargs.get("all_touched")
        # Define raster transform based on cube
        # This maps the geometry domain onto the cube domain
        tr = _make_raster_cube_transform(cube)
        # Generate mask from geometry
        mask_template = rfeatures.geometry_mask(
            geometries=shapely.get_parts(geometry),
            out_shape=(h, w),
            transform=tr,
            all_touched=all_touched,
        )

    # If cube was on circular domain, then the transformed
    # mask template needs shifting to match the cube domain
    if x_coord.circular:
        mask_template = np.roll(mask_template, w // 2, axis=1)

    if kwargs.get("invert"):
        if not isinstance(kwargs.get("invert"), bool):
                msg = "`invert` kwarg must be True or False."
                raise TypeError(msg)
        # Invert the mask
        mask_template = np.logical_not(mask_template)

    return mask_template


def is_geometry_valid(
    geometry: shapely.Geometry,
    geometry_crs: ccrs.CRS | CRS,
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

    The function returns silently if the geometry is valid.

    Now create an invalid geometry covering the Bering Sea,
    and check its validity for the WGS84 coordinate system.

    >>> bering_sea = box(148.42,49.1,-138.74,73.12)
    >>> wgs84 = CRS.from_epsg(4326)
    >>> is_geometry_valid(bering_sea, wgs84) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last)
    ValueError: Geometry crossing the 180th meridian is not supported.
    """
    WGS84_crs = CRS.from_epsg(4326)

    # Check crs is valid type
    if not isinstance(geometry_crs, (ccrs.CRS | CRS)):
        msg = f"Geometry CRS must be a cartopy.crs or pyproj.CRS object, not {type(geometry_crs)}."
        raise TypeError(msg)

    # Check geometry is valid shapely geometry
    if not shapely.is_valid_input(geometry):
        msg = "Shape geometry is not a valid shape (not well formed)."
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
    # Exception for MultiPoint geometries where sequential points
    # may be separated by more than 180 degrees
    if not isinstance(geometry, sgeom.MultiPoint):
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


def _transform_geometry(
    geometry: shapely.Geometry, geometry_crs: ccrs.CRS | CRS, cube_crs: ccrs.CRS
) -> shapely.Geometry:
    """Transform a geometry to the cube CRS using pyproj.

    Parameters
    ----------
    geometry : :class:`shapely.Geometry`
        The geometry to transform.
    geometry_crs : :class:`cartopy.crs`, :class:`pyproj.CRS`
        The coordinate reference system of the geometry.
    cube_crs : :class:`cartopy.crs`, :class:`pyproj.CRS`
        The coordinate reference system of the cube.

    Returns
    -------
    :class:`shapely.Geometry`
        The transformed geometry.
    """
    # Set-up transform via pyproj
    t = Transformer.from_crs(
        crs_from=geometry_crs, crs_to=cube_crs, always_xy=True
    ).transform
    # Transform geometry
    return shapely.ops.transform(t, geometry)


def _get_weighted_mask(
    cube: iris.cube.Cube,
    geometry: shapely.Geometry,
    minimum_weight: float,
) -> np.array:
    """Get a mask based on the geometry and minimum weight.

    Parameters
    ----------
    cube : :class:`iris.cube.Cube`
        The cube to mask.
    geometry : :class:`shapely.Geometry`
        The geometry to use for masking.
    minimum_weight : float
        The minimum weight of the geometry to be included in the mask.
        If the weight is less than this value, the geometry will not be
        included in the mask.

    Returns
    -------
    :class:`np.array`
        An array of the shape of the x & y coordinates of the cube, with points
        to mask equal to True.
    """
    if not cube.coord(axis="x", dim_coords=True).has_bounds():
        cube.coord(axis="x", dim_coords=True).guess_bounds()
    if not cube.coord(axis="y", dim_coords=True).has_bounds():
        cube.coord(axis="y", dim_coords=True).guess_bounds()
    # Get the shape of the cube
    w = len(cube.coord(axis="x", dim_coords=True).points)
    h = len(cube.coord(axis="y", dim_coords=True).points)
    # Get the bounds of the cube
    # x_bounds = _get_mod_rebased_coord_bounds(cube.coord(x_name))
    # y_bounds = _get_mod_rebased_coord_bounds(cube.coord(y_name))
    x_bounds = cube.coord(axis="x", dim_coords=True).bounds
    y_bounds = cube.coord(axis="y", dim_coords=True).bounds
    # Generate Sort-Tile-Recursive (STR) packed R-tree of bounding boxes
    # https://shapely.readthedocs.io/en/stable/strtree.html
    grid_boxes = [
        sgeom.box(x[0], y[0], x[1], y[1]) for y, x in product(y_bounds, x_bounds)
    ]
    grid_tree = shapely.STRtree(grid_boxes)
    # Find grid boxes indexes that intersect with the geometry
    idxs = grid_tree.query(geometry, predicate="intersects")
    # Get grid box indexes that intersect with the minimum weight criteria
    mask_idxs_bool = [
        grid_boxes[idx].intersection(geometry).area / grid_boxes[idx].area
        >= minimum_weight
        for idx in idxs
    ]
    mask_idxs = idxs[mask_idxs_bool]
    mask_xy = [list(product(range(h), range(w)))[i] for i in mask_idxs]
    # Create mask from grid box indexes

    weighted_mask_template = np.ones((h, w), dtype=bool)
    # Set mask = True for grid box indexes identified above
    for xy in mask_xy:
        weighted_mask_template[xy] = False
    return weighted_mask_template


def _make_raster_cube_transform(cube: iris.cube.Cube) -> Affine:
    """Create a rasterio transform for the cube.

    Raises
    ------
    CoordinateNotRegularError
        If the cube dimension coordinates are not regular,
        such that :func:`iris.util.regular_step` returns an error.

    Returns
    -------
    :class:`affine.Affine`
        An affine transform object that maps the geometry domain onto the cube domain.
    """
    x_points = cube.coord(axis="x", dim_coords=True).points
    y_points = cube.coord(axis="y", dim_coords=True).points
    dx = iris.util.regular_step(cube.coord(axis="x", dim_coords=True))
    dy = iris.util.regular_step(cube.coord(axis="y", dim_coords=True))
    # Create a rasterio transform based on the cube
    # This maps the geometry domain onto the cube domain
    trans = Affine.translation(x_points[0] - dx / 2, y_points[0] - dy / 2)
    scale = Affine.scale(dx, dy)
    return trans * scale
