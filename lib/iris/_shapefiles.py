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

# ------------------------------------------------------------------------------
# GLOBAL VARIABLES
# ------------------------------------------------------------------------------


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
    # verification
    from iris.cube import Cube, CubeList

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
    trans_geo = _transform_coord_system(geometry, cube, shape_coord_system)
    prepped.prep(trans_geo)

    # prepare 2D cube
    y_name, x_name = _cube_primary_xy_coord_names(cube)
    cube_2d = cube.slices([y_name, x_name]).next()
    _cube_xy_guessbounds(cube_2d)
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
            x0, x1 = _rebase_values_to_modulus((x0, x1), xmod)
        if ymod:
            y0, y1 = _rebase_values_to_modulus((y0, y1), ymod)

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


def _transform_coord_system(geometry, cube, geometry_system=None):
    """Project the shape onto another coordinate system.

    Arguments:
        target: The target :class:`iris.coord_systems.CoordSystem`
                or a :class:`iris.cube.Cube` object defining the coordinate
                system to which the shape should be transformed

    Returns:
        A transformed shape (copy)
    """
    y_name, x_name = _cube_primary_xy_coord_names(cube)
    DEFAULT_CS = _get_global_cs()
    target_system = cube.coord_system()
    if not target_system:
        warnings.warn("Cube has no coord_system; using default GeogCS lat/lon")
        target_system = DEFAULT_CS
    if geometry_system is None:
        geometry_system = DEFAULT_CS
    target_proj = target_system.as_cartopy_projection()
    source_crs = geometry_system.as_cartopy_projection()

    trans_geometry = target_proj.project_geometry(geometry, source_crs)
    if target_system == DEFAULT_CS and cube.coord(x_name).points[-1] > 180:
        trans_geometry = shapely.transform(trans_geometry, _trans_func)

    return trans_geometry


def _trans_func(geometry):
    """pocket function for transforming the x coord of a geometry from -180 to 180 to 0-360"""
    for point in geometry:
        if point[0] < 0:
            point[0] = 360 - np.abs(point[0])
    return geometry


def _get_global_cs():
    """pocket function for returning the iris default coord system to avoid circular imports"""
    import iris.fileformats.pp

    DEFAULT_CS = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
    return DEFAULT_CS


def _cube_primary_xy_coord_names(cube):
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


def _cube_xy_guessbounds(cube):
    """Guess latitude/longitude bounds of the cube and add them (**in place**)
    if not present.

    Arguments:
        cube (:class:`iris.cube.Cube`): An Iris cube

    Warning:
        This function modifies the passed `cube` in place, adding bounds to the
        latitude and longitude coordinates.
    """
    for coord in _cube_primary_xy_coord_names(cube):
        if not cube.coord(coord).has_bounds():
            cube.coord(coord).guess_bounds()


def _rebase_values_to_modulus(values, modulus):
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
