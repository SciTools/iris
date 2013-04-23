# (C) British Crown Copyright 2013 Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
Support for conservative regridding via ESMPy.
"""

import numpy as np

import ESMF

import cartopy.crs as ccrs
import iris
import iris.experimental.regrid as i_regrid


def _get_coord_crs(coord):
    """ Get the Cartopy crs relevant to a coordinate (or None). """
    cs = coord.coord_system
    if cs is None:
        return None
    return cs.as_cartopy_crs()


_crs_truelatlon = ccrs.Geodetic()
""" A static Cartopy Geodetic() instance for transforming to true-lat-lons. """


def _convert_latlons(crs, x_array, y_array):
    """
    Convert x+y coords in a given crs to (x,y) values in true-lat-lons.

    ..note:
        Uses a plain Cartopy Geodetic to convert to true-lat-lons.  This makes
        no allowance for a non-spherical earth.  But then, neither does ESMF.

    """
    ll_values = _crs_truelatlon.transform_points(crs, x_array, y_array)
    return ll_values[..., 0], ll_values[..., 1]


def _make_esmpy_field_from_coords(x_coord, y_coord, ref_name='field',
                                  data=None, mask=None):
    """
    Create an ESMPy ESMF.Field on given coordinates.

    Create a ESMF.Grid from the coordinates, defining corners and centre
    positions as lats+lons.
    Add a grid mask if provided.
    Create and return a Field mapped on this Grid, setting data if provided.

    Args:

    * x_coord, y_coord (:class:`iris.coords.Coord`):
        One-dimensional coordinates of shape (nx,) and (ny,).
        Their contiguous bounds define an ESMF.Grid of shape (nx, ny).

    Kwargs:

    * data (:class:`numpy.ndarray`, shape (nx,ny)):
        Set the Field data content.
    * mask (:class:`numpy.ndarray`, boolean, shape (nx,ny)):
        Add a mask item to the grid, assigning it 0/1 where mask=False/True.

    """
    # Create a Grid object describing the coordinate cells.
    dims = [len(coord.points) for coord in (x_coord, y_coord)]
    dims = np.array(dims, dtype=np.int32)  # specific type required by ESMF.
    grid = ESMF.Grid(dims)

    # Get all cell corner coordinates as true-lat-lons
    x_bounds, y_bounds = np.meshgrid(x_coord.contiguous_bounds(),
                                     y_coord.contiguous_bounds())
    grid_crs = _get_coord_crs(x_coord)
    lon_bounds, lat_bounds = _convert_latlons(grid_crs, x_bounds, y_bounds)

    # Add grid 'coord' element for corners, and fill with corner values.
    grid.add_coords(staggerlocs=[ESMF.StaggerLoc.CORNER])
    grid_corners_x = grid.get_coords(0, ESMF.StaggerLoc.CORNER)
    grid_corners_x[:] = lon_bounds
    grid_corners_y = grid.get_coords(1, ESMF.StaggerLoc.CORNER)
    grid_corners_y[:] = lat_bounds

    # calculate the cell centre-points
    # NOTE: we don't care about Iris' idea of where the points 'really' are
    # *but* ESMF requires the data in the CENTER for conservative regrid,
    # according to the documentation :
    #  - http://www.earthsystemmodeling.org/
    #        esmf_releases/public/last/ESMF_refdoc.pdf
    #  - section  22.2.3 : ESMF_REGRIDMETHOD
    #
    # We are currently determining cell centres in native coords, then
    # converting these into true-lat-lons.
    # It is confirmed by experiment that moving these centre location *does*
    # changes the regrid results.
    # TODO: work out why this is needed, and whether these centres are 'right'.

    # Average cell corners in native coordinates, then translate to lats+lons
    # (more costly, but presumably 'more correct' than averaging lats+lons).
    x_centres = x_coord.contiguous_bounds()
    x_centres = 0.5 * (x_centres[:-1] + x_centres[1:])
    y_centres = y_coord.contiguous_bounds()
    y_centres = 0.5 * (y_centres[:-1] + y_centres[1:])
    x_points, y_points = np.meshgrid(x_centres, y_centres)
    lon_points, lat_points = _convert_latlons(grid_crs, x_points, y_points)

    # Add grid 'coord' element for centres + fill with centre-points values.
    grid.add_coords(staggerlocs=[ESMF.StaggerLoc.CENTER])
    grid_centers_x = grid.get_coords(0, ESMF.StaggerLoc.CENTER)
    grid_centers_x[:] = lon_points
    grid_centers_y = grid.get_coords(1, ESMF.StaggerLoc.CENTER)
    grid_centers_y[:] = lat_points

    # Add a mask item, if requested
    if mask is not None:
        grid.add_item(ESMF.GridItem.MASK,
                      [ESMF.StaggerLoc.CENTER])
        grid_mask = grid.get_item(ESMF.GridItem.MASK)
        grid_mask[:] = np.where(mask, 1, 0)

    # create a Field based on this grid
    field = ESMF.Field(grid, ref_name)

    # assign data content, if provided
    if data is not None:
        field.data[:] = data

    return field


def regrid_conservative_via_esmpy(source_cube, grid_cube_or_coords):
    """
    Perform a conservative regridding with ESMPy.

    Regrids the data of a source cube onto a new grid defined by a destination
    cube or coordinates.

    Args:

    * source_cube (:class:`iris.cube.Cube`):
        Source data.  Must be two-dimensional, with two identifiable horizontal
        dimension coordinates.
    * grid_cube_or_coords :
        Either a :class:`iris.cube.Cube`, or a pair of
        :class:`iris.coords.Coord`, defining the target horizontal grid.
        If a cube, *only* the horizontal dimension coordinates are used.

    Returns:
        A new cube derived from source_cube, regridded onto the specified
        horizontal grid.  Any additional coordinates mapped onto horizontal
        spatial axes are lost, while all other metadata is retained.

    ..note:
        Both source and destination cubes must have two dimension coordinates
        identified with axes 'x' and 'y', and having the same, defined
        coord_system.
        The grids are defined by `iris.coords.Coord.contiguous_bounds`() of
        these.

    ..note:
        Initialises the ESMF Manager, if it was not already called.
        This implements default Manager operations (e.g. logging).
        To alter these, use an earlier ESMF.Manager() call.

    """
    # Process parameters to get input+output horizontal coordinates.
    src_coords = i_regrid._get_xy_dim_coords(source_cube)
    if isinstance(grid_cube_or_coords, iris.cube.Cube):
        dst_coords = i_regrid._get_xy_dim_coords(grid_cube_or_coords)
    else:
        dst_coords = grid_cube_or_coords

    # Check source+target coordinates are suitable.
    # NOTE: '_get_xy_dim_coords' ensures the coords exist; are unique; and have
    # same coord_system.  We also need them to have a _valid_ coord_system.
    if _get_coord_crs(src_coords[0]) is None:
        raise ValueError('Source X+Y coordinates have no coord_system.')
    if _get_coord_crs(dst_coords[0]) is None:
        raise ValueError('Destination X+Y coordinates have no coord_system.')

    # FOR NOW: 2d only
    if source_cube.ndim != 2:
        raise ValueError('Source cube must be 2-dimensional.')

    # Initialise the ESMF manager in case it was not already done.
    ESMF.Manager()
        # NOTE: Implements default settings.  If you don't like these, call it
        # yourself first (then this call does nothing).

    # Get the source data, reformed into the right dimension order, (x,y).
    src_data = source_cube.data
    src_dims_xy = [source_cube.coord_dims(coord)[0] for coord in src_coords]
    src_data = src_data.transpose(src_dims_xy)

    # Work out whether we have missing data to define a source grid mask.
    srcdata_mask = np.ma.getmask(src_data)
    if not np.any(srcdata_mask):
        srcdata_mask = None

    # Construct ESMF Field objects on source and destination grids.
    src_field = _make_esmpy_field_from_coords(src_coords[0], src_coords[1],
                                              data=src_data,
                                              mask=srcdata_mask)
    dst_field = _make_esmpy_field_from_coords(dst_coords[0], dst_coords[1])

    # Make extra Field for destination coverage fraction (for missing data).
    coverage_field = ESMF.Field(dst_field.grid, 'validmask_dst')

    # Do the actual regrid with ESMF.
    regrid_method = ESMF.Regrid(src_field, dst_field,
                                src_mask_values=np.array([1], dtype=np.int32),
                                dst_mask_values=np.array([1], dtype=np.int32),
                                regrid_method=ESMF.RegridMethod.CONSERVE,
                                unmapped_action=ESMF.UnmappedAction.IGNORE,
                                dst_frac_field=coverage_field)
    regrid_method(src_field, dst_field)
    data = dst_field.data

    # Convert destination 'coverage fraction' into a simple missing-data mask.
    # = part of cell is outside the source grid, or over a masked datapoint.
    dst_mask = coverage_field.data < (1.0 - 1e-8)
    # Mask the data field if any points are masked.
    if np.any(dst_mask):
        data = np.ma.array(data, mask=dst_mask)

    # Transpose ESMF result dims (X,Y) back to the order of the source
    inverse_dims = np.zeros(data.ndim)
    inverse_dims[src_dims_xy] = np.arange(data.ndim)
    data = data.transpose(inverse_dims)

    # Return result as a new cube based on the source.
    # TODO: please tidy this interface !!!
    return i_regrid._create_cube(
        data,
        src=source_cube,
        x_dim=src_dims_xy[0],
        y_dim=src_dims_xy[1],
        src_x_coord=src_coords[0],
        src_y_coord=src_coords[1],
        grid_x_coord=dst_coords[0],
        grid_y_coord=dst_coords[1],
        sample_grid_x=dst_coords[0].points,
        sample_grid_y=dst_coords[1].points,
        regrid_callback=i_regrid._regrid_bilinear_array)
