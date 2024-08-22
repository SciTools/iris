# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

# Much of this code is originally based off the ASCEND library, developed in
# the Met Office by Chris Kent, Emilie Vanvyve, David Bentley, Joana Mendes
# many thanks to them. Converted to iris by Alex Chamberlain-Clay


from itertools import product
import warnings

import numpy as np
import shapely
import shapely.errors
import shapely.geometry as sgeom
import shapely.ops

import iris.analysis.cartography
from iris.warnings import IrisDefaultingWarning, IrisUserWarning


def create_shapefile_mask(
    geometry: shapely.Geometry,
    cube: iris.cube.Cube,
    minimum_weight: float = 0.0,
    geometry_crs: cartopy.crs = None,
) -> np.array:
    """Make a mask for a cube from a shape.

    Get the mask of the intersection between the
    given shapely geometry and cube with x/y DimCoords.
    Can take a minimum weight and evaluate area overlaps instead

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

    Returns
    -------
    :class:`np.array`
        An array of the shape of the x & y coordinates of the cube, with points
        to mask equal to True.

    Notes
    -----
    For best masking results, both the cube _and_ masking geometry should have a
    coordinate reference system (CRS) defined. Masking results will be most reliable
    when the cube and masking geometry have the same CRS.

    If the cube has no coord_system, the default GeogCS is used where
    the coordinate units are degrees. For any other coordinate units,
    the cube **must** have a coord_system defined.

    If a CRS is not provided for the the masking geometry, the CRS of the cube is assumed.
    """
    from iris.cube import Cube, CubeList

    try:
        msg = "Geometry is not a valid Shapely object"
        if not shapely.is_valid(geometry):
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
            category=IrisDefaultingWarning,
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
    box_template = [
        sgeom.box(x[0], y[0], x[1], y[1]) for x, y in product(x_bounds, y_bounds)
    ]
    # shapely can do lazy evaluation of intersections if it's given a list of grid box shapes
    # delayed lets us do it in parallel
    intersect_template = shapely.intersects(trans_geo, box_template)
    # we want areas not under shapefile to be True (to mask)
    intersect_template = np.invert(intersect_template)
    # now calc area overlaps if doing weights and adjust mask
    if minimum_weight > 0.0:
        intersections = np.array(box_template)[~intersect_template]
        intersect_template[~intersect_template] = [
            trans_geo.intersection(box).area / box.area <= minimum_weight
            for box in intersections
        ]
    mask_template = np.reshape(intersect_template, cube_2d.shape[::-1]).T
    return mask_template


def _transform_coord_system(
    geometry: shapely.Geometry, 
    cube: iris.cube.Cube, 
    geometry_crs: cartopy.crs = None
) -> shapely.Geometry:
    """Project the shape onto another coordinate system.

    Parameters
    ----------
    geometry : :class:`shapely.Geometry`
    cube : :class:`iris.cube.Cube`
        :class:`~iris.cube.Cube` with the coord_system to be projected to and
        a x coordinate.
    geometry_crs : :class:`cartopy.crs`, optional
        A :class:`cartopy.crs` object describing
        the coord_system of the shapefile. Defaults to None,
        in which case the geometry_crs is assumed to be the
        same as the `cube`.

    Returns
    -------
    :class:`shapely.Geometry`
        A transformed copy of the provided :class:`shapely.Geometry`.

    """
    _ , x_name = _cube_primary_xy_coord_names(cube)

    target_system = cube.coord_system().as_cartopy_projection()
    if not target_system:
        # If no cube coord_system do our best to guess...
        if (
            cube.coord(axis="x").units == "degrees"
            and cube.coord(axis="y").units == "degrees"
        ):
            # If units of degrees assume GeogCS
            target_system = iris.coord_systems.GeogCS(
                iris.analysis.cartography.DEFAULT_SPHERICAL_EARTH_RADIUS
            )
            warnings.warn(
                "Cube has no coord_system; using default GeogCS lat/lon.",
                category=IrisDefaultingWarning,
            )
        else:
            # For any other units, don't guess and raise an error
            target_system = iris.coord_systems.GeogCS(
                iris.analysis.cartography.DEFAULT_SPHERICAL_EARTH_RADIUS
            )
            raise ValueError("Cube has no coord_system; cannot guess coord_system!")

    if not geometry_crs:
        # If no geometry_crs assume it has the cube coord_system
        geometry_crs = target_system
        warnings.warn(
            "No geometry coordinate reference system supplied; using cube coord_system instead.",
            category=IrisDefaultingWarning,
        )

    trans_geometry = target_proj.project_geometry(geometry, source_proj)

    # A GeogCS in iris can be either -180 to 180 or 0 to 360. If cube is 0-360, shift geom to match
    if (
        isinstance(target_system, iris.coord_systems.GeogCS)
        and cube.coord(x_name).points[-1] > 180
    ):
        # chop geom at 0 degree line very finely then transform
        prime_meridian_line = shapely.LineString([(0, 90), (0, -90)])
        trans_geometry = trans_geometry.difference(prime_meridian_line.buffer(0.00001))
        trans_geometry = shapely.transform(trans_geometry, _trans_func)

    return trans_geometry


def _trans_func(geometry):
    """Pocket function for transforming the x coord of a geometry from -180 to 180 to 0-360."""
    for point in geometry:
        if point[0] < 0:
            point[0] = 360 - np.abs(point[0])
    return geometry


def _cube_primary_xy_coord_names(cube):
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


def _get_mod_rebased_coord_bounds(coord):
    """Take in a coord and returns a array of the bounds of that coord rebased to the modulus.

    Parameters
    ----------
    coord : :class:`iris.coords.Coord`
        An Iris coordinate with a modulus.

    Returns
    -------
    :class:`np.array`
        A 1d Numpy array of [start,end] pairs for bounds of the coord.

    """
    modulus = coord.units.modulus
    # Force realisation (rather than core_bounds) - more efficient for the
    #  repeated indexing happening downstream.
    result = np.array(coord.bounds)
    if modulus:
        result[result < 0.0] = (np.abs(result[result < 0.0]) % modulus) * -1
        result[np.isclose(result, modulus, 1e-10)] = 0.0
    return result
