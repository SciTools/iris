# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

# Much of this code is originally based off the ASCEND library, developed in
# the Met Office by Chris Kent, Emilie Vanvyve, David Bentley, Joana Mendes
# many thanks to them. Converted to iris by Alex Chamberlain-Clay


from itertools import product
import warnings

import dask.array
import numpy as np
import shapely
import shapely.errors
import shapely.geometry as sgeom
import shapely.ops

from iris.exceptions import IrisDefaultingWarning


def create_shapefile_mask(
    geometry,
    cube,
    minimum_weight=0.0,
):
    """Makes a mask for a cube from a shape

    Get the mask of the intersection between the
    given shapely geometry and cube with x/y DimCoords.
    Can take a minimum weight and evaluate area overlaps instead


    Parameters
    ----------
        geometry        : A :class:`shapely.Geometry` object

        cube            : A :class:`iris.cube.Cube`
                            which has 1d x and y coordinates
        minimum_weight  : A float between 0 and 1 determining what % of a cell
                            a shape must cover for the cell to remain unmasked.
                            eg: 0.1 means that at least 10% of the shape overlaps the cell
                            to be unmasked
                            Requires geometry to be a Polygon or MultiPolygon
                            Defaults to 0.0 (eg only test intersection)

    Returns
    -------
        A :class:`np.array` of the shape of the x & y coordinates of the cube, with points to mask equal to True

    """
    from iris.cube import Cube, CubeList

    try:
        msg = "Geometry is not a valid Shapely object"
        if shapely.is_valid(geometry) is False:
            raise TypeError(msg)
    except Exception:
        raise TypeError(msg)
    if not isinstance(cube, Cube):
        if isinstance(cube, CubeList):
            msg = "Received CubeList object rather than Cube - \
            to mask a CubeList iterate over each Cube"
            raise TypeError(msg)
        else:
            msg = "Received non-Cube object where a Cube is expected"
            raise TypeError(msg)
    if minimum_weight > 0.0 and isinstance(
        geometry,
        (
            sgeom.Point,
            sgeom.LineString,
            sgeom.LinearRing,
            sgeom.MultiPoint,
            sgeom.MultiLineString,
        ),
    ):
        minimum_weight = 0.0
        warnings.warn(
            """Shape is of invalid type for minimum weight masking,
            must use a Polygon rather than Line shape.\n
              Masking based off intersection instead. """,
            IrisDefaultingWarning,
        )

    # prepare 2D cube
    y_name, x_name = _cube_primary_xy_coord_names(cube)
    cube_2d = cube.slices([y_name, x_name]).next()
    for coord in cube_2d.dim_coords:
        if not coord.has_bounds():
            coord.guess_bounds()
    trans_geo = _transform_coord_system(geometry, cube_2d)

    y_coord, x_coord = [cube_2d.coord(n) for n in (y_name, x_name)]
    x_bounds = _get_mod_rebased_coord_bounds(x_coord)
    y_bounds = _get_mod_rebased_coord_bounds(y_coord)
    # prepare array for dark
    bounds_array = np.asarray(list(product(x_bounds, y_bounds)))
    box_template = _template_func(bounds_array)
    # shapely can do lazy evaluation of intersections if it's given a list of grid box shapes
    # delayed lets us do it in parallel
    intersect_template = dask.delayed(shapely.intersects(trans_geo, box_template))
    # we want areas not under shapefile to be True (to mask)
    intersect_template = np.invert(intersect_template).compute()
    # now calc area overlaps if doing weighted comparison
    if minimum_weight > 0.0:
        for count, bol in enumerate(intersect_template):
            if not bol:
                intersect_area = trans_geo.intersection(box_template[count]).area
                if (intersect_area / box_template[count].area) <= minimum_weight:
                    intersect_template[count] = True
                else:
                    intersect_template[count] = False
    mask_template = np.reshape(intersect_template, cube_2d.shape[::-1]).T
    return mask_template


def _template_func(bounds_array):
    """Take array of 2x2 bounds of cells, and returns list of Shapely Polygons of cell bounds"""
    template = list(range(bounds_array.shape[0]))
    for count, idx in enumerate(bounds_array):
        # get the bounds of the grid cell
        x0, x1 = idx[0]
        y0, y1 = idx[1]
        # create a new polygon of the grid cell a
        template[count] = sgeom.box(x0, y0, x1, y1)

    return template


def _transform_coord_system(geometry, cube, geometry_system=None):
    """Project the shape onto another coordinate system.

    Parameters
    ----------
                geometry: A :class:`shapely.Geometry` object

                    cube: A :class:`iris.cube.Cube` with the coord_system to
                        be projected to and a x coordinate

         geometry_system: A :class:`iris.coord_systems` object describing
                        the coord_system of the shapefile. Defaults to None,
                        which is treated as GeogCS

    Returns
    -------
        A transformed copy of the provided :class:`shapely.Geometry`
    """
    y_name, x_name = _cube_primary_xy_coord_names(cube)
    import iris.analysis.cartography

    DEFAULT_CS = iris.coord_systems.GeogCS(
        iris.analysis.cartography.DEFAULT_SPHERICAL_EARTH_RADIUS
    )
    target_system = cube.coord_system()
    if not target_system:
        warnings.warn(
            "Cube has no coord_system; using default GeogCS lat/lon",
            IrisDefaultingWarning,
        )
        target_system = DEFAULT_CS
    if geometry_system is None:
        geometry_system = DEFAULT_CS
    target_proj = target_system.as_cartopy_projection()
    source_proj = geometry_system.as_cartopy_projection()

    trans_geometry = target_proj.project_geometry(geometry, source_proj)
    # A default coord system in iris can be either -180 to 180 or 0 to 360
    if target_system == DEFAULT_CS and cube.coord(x_name).points[-1] > 180:
        trans_geometry = shapely.transform(trans_geometry, _trans_func)

    return trans_geometry


def _trans_func(geometry):
    """Pocket function for transforming the x coord of a geometry from -180 to 180 to 0-360"""
    for point in geometry:
        if point[0] < 0:
            point[0] = 360 - np.abs(point[0])
    return geometry


def _cube_primary_xy_coord_names(cube):
    """Return the primary latitude and longitude coordinate standard names, or
    long names, from a cube.

    Arguments:
        cube (:class:`iris.cube.Cube`): An Iris cube

    Returns
    -------
        The names of the primary latitude and longitude coordinates
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


def _get_mod_rebased_coord_bounds(coord):
    """Takes in a coord and returns a array of
    the bounds of that coord rebased to the modulus
    Arguments:
        coord (:class:`iris.coords.Coord`): An Iris coordinate
        with a modulus

    Returns
    -------
    A 1d Numpy array of [start,end] pairs for bounds of the coord
    """
    modulus = coord.units.modulus
    # Force realisation (rather than core_bounds) - more efficient for the
    #  repeated indexing happening downstream.
    result = np.array(coord.bounds)
    if modulus:
        result[result < 0.0] = (np.abs(result[result < 0.0]) % modulus) * -1
        result[np.isclose(result, modulus, 1e-10)] = 0.0
    return result
