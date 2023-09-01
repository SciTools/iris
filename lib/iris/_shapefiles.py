# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

import warnings

import numpy as np
import shapely
import shapely.errors
import shapely.geometry as sgeom
import shapely.ops
import shapely.prepared as prepped

import iris
import iris.coord_systems
from iris.cube import Cube, CubeList
import iris.fileformats.pp

# ------------------------------------------------------------------------------
# GLOBAL VARIABLES
# ------------------------------------------------------------------------------

DEFAULT_CS = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)


def create_shapefile_mask(
    geometry, cube, minimum_weight=0.0, shape_coord_system=None
):
    """Get the mask array of the intersection between the given shapely geometry and
    cube.

    Arguments:
        geometry        : A :class:`shapely.Geometry` object

        cube            : A :class:`iris.cube.Cube`
                            with x and y coordinates
        minimum_weight  : A float between 0 and 1 determining what % of a cell
                            a shape must cover for the cell to remain unmasked.
                            eg: 0.1 means that at least 10% of the shape overlaps the cell
                            to be unmasked

    Returns:
        A numpy array of the shape of the x & y coordinates of the cube, with points to mask equal to True
    """
    if not isinstance(cube, Cube):
        if isinstance(cube, CubeList):
            msg = "Received CubeList object rather than Cube - to mask a CubeList iterate over each Cube"
            raise TypeError(msg)
        else:
            msg = "Received non-Cube object where a Cube is expected"
            raise TypeError(msg)

    if not minimum_weight or minimum_weight == 0.0:
        weights = False
    elif isinstance(
        geometry,
        (
            sgeom.Point,
            sgeom.LineString,
            sgeom.LinearRing,
            sgeom.MultiPoint,
            sgeom.MultiLineString,
        ),
    ):
        weights = False
        warnings.warn(
            "Shape is of invalid type for minimum weight masking, must use a Polygon rather than Line shape.\n Masking based off intersection instead. "
        )
    else:
        weights = True

    # prepare shape
    cube_cs = cube.coord_system()
    if not cube_cs:
        warnings.warn("Cube has no coord_system; using default GeogCS lat/lon")
        cube_cs = DEFAULT_CS
    trans_geo = transform_coord_system(geometry, cube, shape_coord_system)
    prepped.prep(trans_geo)

    # prepare 2D cube
    y_name, x_name = cube_primary_xy_coord_names(cube)
    cube_2d = cube.slices([y_name, x_name]).next()
    cube_xy_guessbounds(cube_2d)
    xmod = cube.coord(x_name).units.modulus
    ymod = cube.coord(y_name).units.modulus
    cube.coord("longitude")
    mask_template = np.zeros(cube_2d.shape, dtype=np.float64)

    # perform the masking
    for count, idx in enumerate(np.ndindex(cube_2d.shape)):
        # get the bounds of the grid cell
        xi = idx[cube_2d.coord_dims(x_name)[0]]
        yi = idx[cube_2d.coord_dims(y_name)[0]]
        x0, x1 = cube_2d.coord(x_name).bounds[xi]
        y0, y1 = cube_2d.coord(y_name).bounds[yi]

        if xmod:
            x0, x1 = rebase_values_to_modulus((x0, x1), xmod)
        if ymod:
            y0, y1 = rebase_values_to_modulus((y0, y1), ymod)

        # create a new polygon of the grid cell and check intersection
        cell_box = sgeom.box(x0, y0, x1, y1)
        intersect_bool = trans_geo.intersects(cell_box)
        # mask all points without a intersection
        if intersect_bool is False:
            mask_template[idx] = True
        # if weights method used, mask intersections below required weight
        if intersect_bool is True and weights is True:
            if (
                shapely.covers(trans_geo, cell_box) is False
            ):  # box is not 100% inside shape
                intersect_area = trans_geo.intersection(cell_box).area
                if (intersect_area / cell_box.area) <= minimum_weight:
                    mask_template[idx] = True

    return mask_template


def transform_coord_system(geometry, cube, geometry_system=None):
    """Project the shape onto another coordinate system.

    Arguments:
        target: The target :class:`iris.coord_systems.CoordSystem`
                or a :class:`iris.cube.Cube` object defining the coordinate
                system to which the shape should be transformed

    Returns:
        A transformed shape (copy)
    """
    y_name, x_name = cube_primary_xy_coord_names(cube)
    target_system = cube.coord_system()
    if not target_system:
        warnings.warn("Cube has no coord_system; using default lat/lon")
        target_system = DEFAULT_CS
    if geometry_system is None:
        geometry_system = DEFAULT_CS
    target_proj = target_system.as_cartopy_projection()
    source_crs = geometry_system.as_cartopy_projection()

    trans_geometry = target_proj.project_geometry(geometry, source_crs)
    if target_system == DEFAULT_CS and cube.coord(x_name).points[-1] > 180:
        trans_geometry = shapely.transform(trans_geometry, trans_func)

    return trans_geometry


def trans_func(geometry):
    for point in geometry:
        if point[0] < 0:
            point[0] = 360 - np.abs(point[0])
    return geometry


def cube_bbox_shape_intersection(cube, a_shape):
    """Determines the intersection of a shape with the bounding
    box of a cube.

    Arguments:
        cube: An :class:`iris,.cube.Cube`, for which the bounds of
              the horizontal coordinates are to be determined
        a_shape: a Shape object, whose intersection with the cube bounding
               box is to be determined
    """

    # create a geometry from the cube bounds
    cube_bounds = get_cube_bounds(cube)
    cube_bbox = sgeom.box(*cube_bounds)

    # project bounding box to coord system of shape
    cube_cs = get_cube_coord_system(cube)
    if not isinstance(DEFAULT_CS, type(cube_cs)):
        shape_crs = DEFAULT_CS.as_cartopy_projection()
        cube_crs = cube_cs.as_cartopy_crs()
        trans_bbox = shape_crs.project_geometry(cube_bbox, cube_crs)
    else:
        trans_bbox = cube_bbox

    # calculate the intersection of the shape and the
    # transformed bounding box

    bbox_intersect = shapely.intersection(trans_bbox, a_shape)
    return bbox_intersect


def rotate_cube_longitude(x_origin, cube):
    """Rotate a cube's data **in place** along the longitude axis to the new
    longitude origin.

    Arguments:
        x_origin (float): The new x-coord origin
        cube            : The :class:`iris.cube.Cube` cube to rotate
    """
    latitude, longitude = cube_primary_xy_coord_names(cube)
    xcord = cube.coord(longitude)
    dx = xcord.points[1] - xcord.points[0]
    lon_shift = xcord.points[0] - x_origin
    number_moves = np.int64(np.rint(lon_shift / dx))
    new_coord = xcord.points - (number_moves * dx)
    cube.data = np.roll(
        cube.data, -number_moves, axis=cube.coord_dims(longitude)[0]
    )
    xcord.points = new_coord
    if xcord.has_bounds():
        xcord.bounds = xcord.bounds - (number_moves * dx)
    else:
        xcord.guess_bounds()


def cube_primary_xy_coord_names(cube):
    """Return the primary latitude and longitude coordinate standard names, or
    long names, from a cube.

    Arguments:
        cube (:class:`iris.cube.Cube`): An Iris cube

    Returns:
        The names of the primary latitude and longitude coordinates
    """
    latc = cube.coords(axis="y")[0] if cube.coords(axis="y") else -1
    lonc = cube.coords(axis="x")[0] if cube.coords(axis="x") else -1

    if -1 in (latc, lonc):
        msg = "Error retrieving xy dimensions in cube: {!r}"
        raise ValueError(msg.format(cube))

    latitude = latc.standard_name if latc.standard_name else latc.long_name
    longitude = lonc.standard_name if lonc.standard_name else lonc.long_name
    return latitude, longitude


def cube_xy_guessbounds(cube):
    """Guess latitude/longitude bounds of the cube and add them (**in place**)
    if not present.

    Arguments:
        cube (:class:`iris.cube.Cube`): An Iris cube

    Warning:
        This function modifies the passed `cube` in place, adding bounds to the
        latitude and longitude coordinates.
    """
    for coord in cube_primary_xy_coord_names(cube):
        if not cube.coord(coord).has_bounds():
            cube.coord(coord).guess_bounds()


def get_cube_bounds(cube, expansion=1.0):
    """Determines the bounds of the horizontal coordinates of
    the given cube.

    Arguments:
        cube: An :class:`iris,.cube.Cube`, for which the bounds of
              the horizontal coordinates are to be determined
        expansion: a value by which to expand the bounds of a cube,
                   so that both end points of a grid are captured.
    """
    # get the primary coordinate names
    yname, xname = cube_primary_xy_coord_names(cube)
    # ensure coordinates have bounds
    for coord in (xname, yname):
        if not cube.coord(coord).has_bounds():
            cube.coord(coord).guess_bounds()
    # get the extents of the coordinates
    x_min = cube.coord(xname).bounds.min()
    x_max = cube.coord(xname).bounds.max()
    y_min = cube.coord(yname).bounds.min()
    y_max = cube.coord(yname).bounds.max()
    return np.array(
        [
            x_min - expansion,
            y_min - expansion,
            x_max + expansion,
            y_max + expansion,
        ]
    )


def get_cube_coord_system(
    cube,
    default_cs=iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS),
):
    """Get a cube's coordinate system.

    Arguments:
        cube      : An :class:`iris.cube.Cube` cube
        default_cs: :class:`iris.coord_systems.CoordSystem` coordinate system
                    to be used if missing from cube

    Returns:
        The coordinate system (:class:`iris.coord_systems.CoordSystem`) of
        `cube`
    """
    cube_cs = cube.coord_system()
    if not cube_cs:
        warnings.warn("Cube has no coord_system; using default lat/lon")
        cube_cs = default_cs
    return cube_cs


def rebase_values_to_modulus(values, modulus):
    """Rebase values to a modulus value.

    Arguments:
        values (list, tuple): The values to be re-based
        modulus             : The value to re-base to

    Returns:
        A list of re-based values.
    """
    rebased = []
    for val in values:
        nv = np.abs(val) % modulus
        if val < 0.0:
            nv *= -1.0
        if np.abs(nv - modulus) < 1e-10:
            nv = 0.0
        rebased.append(nv)
    return rebased


def transform_geometry_coord_system(geometry, source, target):
    """Transform a geometry into a new coordinate system.

    Arguments:
        geometry: A :class:`shapely.geometry.base.BaseGeometry` *Polygon* or
                  *MultiPolygon* object
        source  : A geometry's :class:`iris.coord_systems.CoordSystem`
                  coordinate system
        target  : The geometry's new :class:`iris.coord_systems.CoordSystem`
                  coordinate system

    Returns:
        A transformed *Polygon*/*Multipolygon* instance

    Warning:
        This operation may result in geometries which are not valid. Please
        check that the transformed geometry is as you expect.
    """
    if source == target:
        return geometry

    if hasattr(geometry, "geoms"):
        trans_geoms = []
        for poly in geometry.geoms:
            trans_geoms.append(transform_geometry_points(poly, source, target))
        geo_type = type(geometry)
        trans_geometry = geo_type(trans_geoms)
    else:
        trans_geometry = transform_geometry_points(geometry, source, target)
    return trans_geometry


def transform_geometry_points(geometry, source, target):
    """Transform geometry points into a new coordinate system.

    Arguments:
        geometry: A :class:`shapely.geometry.base.BaseGeometry` geometry
        source  : A shape's :class:`iris.coord_systems.CoordSystem` coordinate
                  system
        target  : The new :class:`iris.coord_systems.CoordSystem` coordinate
                  system

    Returns:
        The points of the `geometry` in the `target` coordinate system.

    Note:
        Multi- and collection types are not supported (see
        :func:`transform_geometry_coord_system`).

    Warning:
        This operation may result in geometries which are not valid. Please
        check that the transformed geometry is as you expect.
    """
    target_proj = target.as_cartopy_projection()
    source_proj = source.as_cartopy_projection()
    fn = lambda xs, ys: target_proj.transform_points(
        source_proj, np.array(xs, copy=False), np.array(ys, copy=False)
    )[:, 0:2].T
    return shapely.ops.transform(fn, geometry)
