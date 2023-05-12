# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Extra stock routines for making and manipulating cubes with 2d coordinates,
to mimic ocean grid data.

"""

import numpy as np
import numpy.ma as ma

from iris.analysis.cartography import unrotate_pole
from iris.coord_systems import RotatedGeogCS
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube


def expand_1d_x_and_y_bounds_to_2d_xy(x_bounds_1d, y_bounds_1d):
    """
    Convert bounds for separate 1-D X and Y coords into bounds for the
    equivalent 2D coordinates.

    The output arrays have 4 points per cell, for 4 'corners' of a gridcell,
    in the usual anticlockwise order
    (bottom-left, bottom-right, top-right, top-left).

    If 1-dimensional X and Y coords have shapes nx and ny, then their bounds
    have shapes  (nx, 2) and (ny, 2).
    The equivalent 2d coordinates would have values which are a "meshgrid" of
    the original 1-D points, both having the shape (ny, ny).
    The outputs are 2d bounds arrays suitable for such 2d coordinates.

    Args:

    * x_bounds_1d, y_bounds_1d : (array)
        Coordinate bounds arrays, with shapes (nx, 2) and (ny, 2).

    Result:

    * x_bounds_2d, y_bounds_2d : (array)
        Expanded 2d bounds arrays, both of shape (ny, nx, 4).

    """
    shapes = [bds.shape for bds in (x_bounds_1d, y_bounds_1d)]
    for shape in shapes:
        if len(shape) != 2 or shape[1] != 2:
            msg = (
                "One-dimensional bounds arrays must have shapes (ny, 2) "
                "and (nx, 2).  Got {} and {}."
            )
            raise ValueError(msg.format(*shapes))

    # Construct output arrays, which are both (ny, nx, 4).
    nx, ny = [shape[0] for shape in shapes]
    bds_2d_x = np.zeros((ny, nx, 4))
    bds_2d_y = bds_2d_x.copy()

    # Expand x bounds to 2D array (ny, nx, 4) : the same over all 'Y'.
    # Bottom left+right corners are the original 1-D x bounds.
    bds_2d_x[:, :, 0] = x_bounds_1d[:, 0].reshape((1, nx))
    bds_2d_x[:, :, 1] = x_bounds_1d[:, 1].reshape((1, nx))
    # Top left+right corners are the same as bottom left+right.
    bds_2d_x[:, :, 2] = bds_2d_x[:, :, 1].copy()
    bds_2d_x[:, :, 3] = bds_2d_x[:, :, 0].copy()

    # Expand y bounds to 2D array (ny, nx, 4) : the same over all 'X'.
    # Left-hand lower+upper corners are the original 1-D y bounds.
    bds_2d_y[:, :, 0] = y_bounds_1d[:, 0].reshape((ny, 1))
    bds_2d_y[:, :, 3] = y_bounds_1d[:, 1].reshape((ny, 1))
    # Right-hand lower+upper corners are the same as the left-hand ones.
    bds_2d_y[:, :, 1] = bds_2d_y[:, :, 0].copy()
    bds_2d_y[:, :, 2] = bds_2d_y[:, :, 3].copy()

    return bds_2d_x, bds_2d_y


def grid_coords_2d_from_1d(x_coord_1d, y_coord_1d):
    """
    Calculate a pair of 2d X+Y coordinates from 1d ones, in a "meshgrid" style.
    If the inputs are bounded, the outputs have 4 points per bounds in the
    usual way, i.e. points 0,1,2,3 are the gridcell corners anticlockwise from
    the bottom left.

    """
    for coord in (x_coord_1d, y_coord_1d):
        if coord.ndim != 1:
            msg = (
                "Input coords must be one-dimensional. "
                'Coordinate "{}" has shape {}.'
            )
            raise ValueError(msg.format(coord.name(), coord.shape))

    # Calculate centre-points as a mesh of the 2 inputs.
    pts_2d_x, pts_2d_y = np.meshgrid(x_coord_1d.points, y_coord_1d.points)
    if not x_coord_1d.has_bounds() or not y_coord_1d.has_bounds():
        bds_2d_x = None
        bds_2d_y = None
    else:
        bds_2d_x, bds_2d_y = expand_1d_x_and_y_bounds_to_2d_xy(
            x_coord_1d.bounds, y_coord_1d.bounds
        )

    # Make two new coords + return them.
    result = []
    for pts, bds, name in zip(
        (pts_2d_x, pts_2d_y), (bds_2d_x, bds_2d_y), ("longitude", "latitude")
    ):
        coord = AuxCoord(pts, bounds=bds, standard_name=name, units="degrees")
        result.append(coord)

    return result


def sample_2d_latlons(regional=False, rotated=False, transformed=False):
    """
    Construct small 2d cubes with 2d X and Y coordinates.

    This makes cubes with 'expanded' coordinates (4 bounds per cell), analogous
    to ORCA data.
    The coordinates are always geographical, so either it has a coord system
    or they are "true" lats + lons.
    ( At present, they are always latitudes and longitudes, but maybe in a
    rotated system. )
    The results always have fully contiguous bounds.

    Kwargs:
    * regional (bool):
        If False (default), results cover the whole globe, and there is
        implicit connectivity between rhs + lhs of the array.
        If True, coverage is regional and edges do not connect.
    * rotated (bool):
        If False, X and Y coordinates are true-latitudes and longitudes, with
        an implicit coordinate system (i.e. None).
        If True, the X and Y coordinates are lats+lons in a selected
        rotated-latlon coordinate system.
    * transformed (bool):
        Build coords from rotated coords as for 'rotated', but then replace
        their values with the equivalent "true" lats + lons, and no
        coord-system (defaults to true-latlon).
        In this case, the X and Y coords are no longer 'meshgrid' style,
        i.e. the points + bounds values vary in *both* dimensions.

    .. note::

        'transformed' is an alternative to 'rotated' :  when 'transformed' is
        set, then 'rotated' has no effect.

    .. Some sample results printouts ::

        >>> print(sample_2d_latlons())
        test_data / (unknown)               (-- : 5; -- : 6)
             Auxiliary coordinates:
                  latitude                      x       x
                  longitude                     x       x
        >>>
        >>> print(sample_2d_latlons().coord(axis='x')[0, :2])
        AuxCoord(array([ 37.5 ,  93.75]),
                 bounds=array([[   0.   ,   65.625,   65.625,    0.   ],
                               [  65.625,  121.875,  121.875,   65.625]]),
                 standard_name='longitude', units=Unit('degrees'))
        >>> print(np.round(sample_2d_latlons().coord(axis='x').points, 3))
        [[  37.5    93.75  150.    206.25  262.5   318.75]
         [  37.5    93.75  150.    206.25  262.5   318.75]
         [  37.5    93.75  150.    206.25  262.5   318.75]
         [  37.5    93.75  150.    206.25  262.5   318.75]
         [  37.5    93.75  150.    206.25  262.5   318.75]]
        >>> print(np.round(sample_2d_latlons().coord(axis='y').points, 3))
        [[-85.  -85.  -85.  -85.  -85.  -85. ]
         [-47.5 -47.5 -47.5 -47.5 -47.5 -47.5]
         [-10.  -10.  -10.  -10.  -10.  -10. ]
         [ 27.5  27.5  27.5  27.5  27.5  27.5]
         [ 65.   65.   65.   65.   65.   65. ]]


        >>> print(np.round(
            sample_2d_latlons(rotated=True).coord(axis='x').points, 3))
        [[  37.5    93.75  150.    206.25  262.5   318.75]
         [  37.5    93.75  150.    206.25  262.5   318.75]
         [  37.5    93.75  150.    206.25  262.5   318.75]
         [  37.5    93.75  150.    206.25  262.5   318.75]
         [  37.5    93.75  150.    206.25  262.5   318.75]]
        >>> print(sample_2d_latlons(rotated=True).coord(axis='y').coord_system)
        RotatedGeogCS(75.0, 120.0)


        >>> print(
            sample_2d_latlons(transformed=True).coord(axis='y').coord_system)
        None
        >>> print(np.round(
            sample_2d_latlons(transformed=True).coord(axis='x').points, 3))
        [[ -50.718  -40.983  -46.74   -71.938  -79.293  -70.146]
         [ -29.867   17.606   77.936  157.145 -141.037  -93.172]
         [ -23.139   31.007   87.699  148.322 -154.639 -100.505]
         [ -16.054   41.218   92.761  143.837 -164.738 -108.105]
         [  10.86    61.78   100.236  137.285  175.511 -135.446]]
        >>> print(np.round(
            sample_2d_latlons(transformed=True).coord(axis='y').points, 3))
        [[-70.796 -74.52  -79.048 -79.26  -74.839 -70.96 ]
         [-34.99  -46.352 -59.721 -60.34  -47.305 -35.499]
         [  1.976 -10.626 -22.859 -23.349 -11.595   1.37 ]
         [ 38.914  25.531  14.312  13.893  24.585  38.215]
         [ 74.197  60.258  51.325  51.016  59.446  73.268]]
        >>>

    """

    def sample_cube(xargs, yargs):
        # Make a test cube with given latitude + longitude coordinates.
        # xargs/yargs are args for np.linspace (start, stop, N), to make the X
        # and Y coordinate points.
        x0, x1, nx = xargs
        y0, y1, ny = yargs
        # Data has cycling values, staggered a bit in successive rows.
        data = np.zeros((ny, nx))
        data.flat[:] = np.arange(ny * nx) % (nx + 2)
        # Build a 2d cube with longitude + latitude coordinates.
        cube = Cube(data, long_name="test_data")
        x_pts = np.linspace(x0, x1, nx, endpoint=True)
        y_pts = np.linspace(y0, y1, ny, endpoint=True)
        co_x = DimCoord(x_pts, standard_name="longitude", units="degrees")
        co_y = DimCoord(y_pts, standard_name="latitude", units="degrees")
        cube.add_dim_coord(co_y, 0)
        cube.add_dim_coord(co_x, 1)
        return cube

    # Start by making a "normal" cube with separate 1-D X and Y coords.
    if regional:
        # Make a small regional cube.
        cube = sample_cube(xargs=(150.0, 243.75, 6), yargs=(-10.0, 40.0, 5))

        # Add contiguous bounds.
        for ax in ("x", "y"):
            cube.coord(axis=ax).guess_bounds()
    else:
        # Global data, but at a drastically reduced resolution.
        cube = sample_cube(xargs=(37.5, 318.75, 6), yargs=(-85.0, 65.0, 5))

        # Make contiguous bounds and adjust outer edges to ensure it is global.
        for name in ("longitude", "latitude"):
            coord = cube.coord(name)
            coord.guess_bounds()
            bds = coord.bounds.copy()
            # Make bounds global, by fixing lowest and uppermost values.
            if name == "longitude":
                bds[0, 0] = 0.0
                bds[-1, 1] = 360.0
            else:
                bds[0, 0] = -90.0
                bds[-1, 1] = 90.0
            coord.bounds = bds

    # Now convert the 1-d coords to 2-d equivalents.
    # Get original 1-d coords.
    co_1d_x, co_1d_y = [cube.coord(axis=ax).copy() for ax in ("x", "y")]

    # Calculate 2-d equivalents.
    co_2d_x, co_2d_y = grid_coords_2d_from_1d(co_1d_x, co_1d_y)

    # Remove the old grid coords.
    for coord in (co_1d_x, co_1d_y):
        cube.remove_coord(coord)

    # Add the new grid coords.
    for coord in (co_2d_x, co_2d_y):
        cube.add_aux_coord(coord, (0, 1))

    if transformed or rotated:
        # Put the cube locations into a rotated coord system.
        pole_lat, pole_lon = 75.0, 120.0
        if transformed:
            # Reproject coordinate values from rotated to true lat-lons.
            co_x, co_y = [cube.coord(axis=ax) for ax in ("x", "y")]
            # Unrotate points.
            lons, lats = co_x.points, co_y.points
            lons, lats = unrotate_pole(lons, lats, pole_lon, pole_lat)
            co_x.points, co_y.points = lons, lats
            # Unrotate bounds.
            lons, lats = co_x.bounds, co_y.bounds
            # Note: save the shape, flatten + then re-apply the shape, because
            # "unrotate_pole" uses "cartopy.crs.CRS.transform_points", which
            # only works on arrays of 1 or 2 dimensions.
            shape = lons.shape
            lons, lats = unrotate_pole(
                lons.flatten(), lats.flatten(), pole_lon, pole_lat
            )
            co_x.bounds, co_y.bounds = lons.reshape(shape), lats.reshape(shape)
        else:
            # "Just" rotate operation : add a coord-system to each coord.
            cs = RotatedGeogCS(pole_lat, pole_lon)
            for coord in cube.coords():
                coord.coord_system = cs

    return cube


def make_bounds_discontiguous_at_point(
    cube, at_iy, at_ix, in_y=False, upper=True
):
    """
    Meddle with the XY grid bounds of a 2D cube to make the grid discontiguous.

    Changes the points and bounds of a single gridcell, so that it becomes
    discontinuous with an adjacent gridcell : either the one to its right, or
    the one above it (if 'in_y' is True).

    Also masks the cube data at the given point.

    The cube must be 2-dimensional and have bounded 2d 'x' and 'y' coordinates.

    """
    x_coord = cube.coord(axis="x")
    y_coord = cube.coord(axis="y")
    assert x_coord.shape == y_coord.shape
    assert (
        coord.bounds.ndim == 3 and coord.shape[-1] == 4
        for coord in (x_coord, y_coord)
    )

    # For both X and Y coord, move points + bounds to create a discontinuity.
    def adjust_coord(coord):
        pts, bds = coord.points, coord.bounds
        # Fetch the 4 bounds (bottom-left, bottom-right, top-right, top-left)
        bds_bl, bds_br, bds_tr, bds_tl = bds[at_iy, at_ix]
        if not in_y:
            # Make a discontinuity "at" (iy, ix), by moving the right-hand edge
            # of the cell to the midpoint of the existing left+right bounds.
            new_bds_b = 0.5 * (bds_bl + bds_br)
            new_bds_t = 0.5 * (bds_tl + bds_tr)
            if upper:
                bds_br, bds_tr = new_bds_b, new_bds_t
            else:
                bds_bl, bds_tl = new_bds_b, new_bds_t
        else:
            # Same but in the 'grid y direction' :
            # Make a discontinuity "at" (iy, ix), by moving the **top** edge of
            # the cell to the midpoint of the existing **top+bottom** bounds.
            new_bds_l = 0.5 * (bds_bl + bds_tl)
            new_bds_r = 0.5 * (bds_br + bds_tr)
            if upper:
                bds_tl, bds_tr = new_bds_l, new_bds_r
            else:
                bds_bl, bds_br = new_bds_l, new_bds_r

        # Write in the new bounds (all 4 corners).
        bds[at_iy, at_ix] = [bds_bl, bds_br, bds_tr, bds_tl]
        # Also reset the cell midpoint to the middle of the 4 new corners,
        # in case having a midpoint outside the corners might cause a problem.
        new_pt = 0.25 * sum([bds_bl, bds_br, bds_tr, bds_tl])
        pts[at_iy, at_ix] = new_pt
        # Write back the coord points+bounds (can only assign whole arrays).
        coord.points, coord.bounds = pts, bds

    adjust_coord(x_coord)
    adjust_coord(y_coord)

    # Check which dimensions are spanned by each coordinate.
    for coord in (x_coord, y_coord):
        span = set(cube.coord_dims(coord))
        if not span:
            msg = "The coordinate {!r} doesn't span a data dimension."
            raise ValueError(msg.format(coord.name()))

    masked_data = ma.masked_array(cube.data)

    # Mask all points which would be found discontiguous.
    # Note that find_discontiguities finds all instances where a cell is
    # discontiguous with a neighbouring cell to its *right* or *above*
    # that cell.
    masked_data[at_iy, at_ix] = ma.masked
    if in_y or not upper:
        masked_data[at_iy, at_ix - 1] = ma.masked
    if not in_y or not upper:
        masked_data[at_iy - 1, at_ix] = ma.masked

    cube.data = masked_data
