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

    from iris.cube import Cube, CubeList

    try:
        msg = TypeError("Geometry is not a valid Shapely object")
        if geometry.is_valid is False:
            raise msg
    except Exception:
        raise msg
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

    # prepare shape
    trans_geo = _transform_coord_system(geometry, cube, shape_coord_system)

    # prepare 2D cube
    for coord in cube.dim_coords:
        if not coord.has_bounds():
            coord.guess_bounds()
    y_name, x_name = _cube_primary_xy_coord_names(cube)
    cube_2d = cube.slices([y_name, x_name]).next()

    y_coord, x_coord = [cube_2d.coord(n) for n in (y_name, x_name)]
    xmod = x_coord.units.modulus
    ymod = y_coord.units.modulus
    # prepare array for dark
    bounds_array = np.asarray(list(product(x_coord.bounds, y_coord.bounds)))
    da_bounds_array = dask.array.from_array(bounds_array, chunks="auto")
    dask_template = dask.array.map_blocks(
        map_blocks_func,
        da_bounds_array,
        xmod,
        ymod,
        trans_geo,
        minimum_weight,
        drop_axis=[1, 2],
        dtype=bool,
        meta=np.array(False),
    )
    mask_template = np.reshape(dask_template.compute(), cube_2d.shape[::-1]).T

    return mask_template


def map_blocks_func(bounds_array, xmod, ymod, shapefile, minimum_weight):
    dask_template = np.empty(bounds_array.shape[0], dtype=bool)
    for count, idx in enumerate(bounds_array):
        # get the bounds of the grid cell
        x0, x1 = idx[0]
        y0, y1 = idx[1]
        if xmod:
            x0, x1 = _rebase_values_to_modulus((x0, x1), xmod)
        if ymod:
            y0, y1 = _rebase_values_to_modulus((y0, y1), ymod)
        # create a new polygon of the grid cell and check intersection
        cell_box = sgeom.box(x0, y0, x1, y1)
        intersect_bool = shapefile.intersects(cell_box)
        # mask all points without a intersection
        if intersect_bool is False:
            dask_template[count] = True
        # if weights method used, mask intersections below required weight
        elif intersect_bool is True and minimum_weight > 0.0:
            intersect_area = shapefile.intersection(cell_box).area
            if (intersect_area / cell_box.area) <= minimum_weight:
                dask_template[count] = True
            else:
                dask_template[count] = False
        else:
            dask_template[count] = False

    return dask_template


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
    """pocket function for transforming the x coord of a geometry from -180 to 180 to 0-360"""
    for point in geometry:
        if point[0] < 0:
            point[0] = 360 - np.abs(point[0])
    return geometry


def _cube_primary_xy_coord_names(cube):
    """Return the primary latitude and longitude coordinate standard names, or
    long names, from a cube.

    Arguments:
        cube (:class:`iris.cube.Cube`): An Iris cube

    Returns:
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
