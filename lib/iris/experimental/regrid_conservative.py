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
Support for conservative regridding with ESMPy.

"""

import numpy as np

import ESMF

import cartopy.crs as ccrs
import iris
import iris.experimental.regrid as i_regrid


def _get_coord_crs(co):
    cs = co.coord_system
    if cs is None:
        return None
    return cs.as_cartopy_crs()


# A private, plain Geodetic crs used for transforming points to true-lat-lon
_crs_truelatlon = ccrs.Geodetic()
_convert_latlons = _crs_truelatlon.transform_points


def _make_esmpy_field_from_coords(x_coord, y_coord, ref_name='field'):
    """ Create an ESMPy Grid for the coordinates and return a Field on it."""
    # create a Grid object describing point positions
    xy_coords = (x_coord, y_coord)
    dims = [len(coord.points) for coord in xy_coords]
    dims = np.array(dims, dtype=np.int32)
    grid = ESMF.Grid(dims)

    # calculate cell corners and transform to true-lat-lons
    x_bounds, y_bounds = np.meshgrid(x_coord.contiguous_bounds(),
                                     y_coord.contiguous_bounds())
    grid_crs = _get_coord_crs(x_coord)
    ll_bounds = _convert_latlons(grid_crs,
                                 x_bounds,
                                 y_bounds)

    lon_bounds, lat_bounds = ll_bounds[:, :, 0], ll_bounds[:, :, 1]

    # get refs to Grid X/Y corner arrays + fill with bounds values
    grid.add_coords(staggerlocs=[ESMF.StaggerLoc.CORNER])
    grid_corners_x = grid.get_coords(0, ESMF.StaggerLoc.CORNER)
    grid_corners_x[:] = lon_bounds
    grid_corners_y = grid.get_coords(1, ESMF.StaggerLoc.CORNER)
    grid_corners_y[:] = lat_bounds

    # also define the centre points
    # NOTE: we don't care about Iris' idea of where the points 'really' are
    # ESMF requires the data in the CENTER for conservative regrid, according
    # to the documentation :
    #  - http://www.earthsystemmodeling.org/
    #        esmf_releases/public/last/ESMF_refdoc.pdf
    #  - section  22.2.3 : ESMF_REGRIDMETHOD
    #
    # NOTE: Still not sure that the documentation states that this is necessary
    # (meaning not totally clear).
    # Have confirmed that changing these values *does* affect results.
    #

    # calculate cell centres as lat/lon values
    # NOTE: calculate in source coords + transformed to lat/lon.
    # - could do in lat/lon with little error, but guess this is more 'correct'
    x_centres = x_coord.contiguous_bounds()
    x_centres = 0.5 * (x_centres[:-1] + x_centres[1:])
    y_centres = y_coord.contiguous_bounds()
    y_centres = 0.5 * (y_centres[:-1] + y_centres[1:])
    x_points, y_points = np.meshgrid(x_centres, y_centres)
    ll_points = _convert_latlons(grid_crs,
                                 x_points,
                                 y_points)
    lon_points, lat_points = ll_points[:, :, 0], ll_points[:, :, 1]

    # get the coordinate pointers and setup the coordinate arrays
    grid.add_coords(staggerlocs=[ESMF.StaggerLoc.CENTER])
    grid_centers_x = grid.get_coords(0, ESMF.StaggerLoc.CENTER)
    grid_centers_x[:] = lon_points
    grid_centers_y = grid.get_coords(1, ESMF.StaggerLoc.CENTER)
    grid_centers_y[:] = lat_points

    # TODO: do we possibly need to define MASKING here ??

    # create + return a Field based on this grid
    field = ESMF.Field(grid, ref_name)
    return field


def _get_esmf_regrid_coverage_mask(src_grid, dst_grid):
    """
    Calculate where esmf regrid destination grid cells have _no_ overlap with
    the source grid.

    Args:
    * src_grid, dst_grid
        ESMF.Grid objects

    Returns:
        numpy boolean array of same shape as dst_grid
    """
    src_coverage = ESMF.Field(src_grid, 'src_coverage')
    dst_coverage = ESMF.Field(dst_grid, 'dst_coverage')
    src_coverage.data[:] = 1.0
    regrid_method = ESMF.Regrid(src_coverage, dst_coverage,
                                regrid_method=ESMF.RegridMethod.CONSERVE,
                                unmapped_action=ESMF.UnmappedAction.IGNORE)
    regrid_method(src_coverage, dst_coverage)
    return dst_coverage.data > 0.0


def regrid_conservative_via_esmpy(source_cube, grid_cube_or_coords):
    """
    Perform a conservative regridding with ESMPy.

    Regrids the data of a source cube onto a new grid defined by a destination
    cube.

    Args:
    * source_cube, grid_cube : ```???```cube
        source data cube.  It must have two identifiable independent horizontal
        dimension coordinates, with the same coord_sytem.

    * grid_cube_or_coords :
        *Either* a ```???```cube, or a pair of ```???```Coordinates, defining
        the target horizontal grid.
        If a cube, only its horizontal dimension coordinates are used.
        There must be two identifiable horizontal coordinates (X and Y), with
        the same coord_system.

    Returns:
        A new cube derived from source_cube, regridded onto the specified
        horizontal grid.  Any additional coordinates mapped onto horizontal
        spatial axes are lost, while all other metadata is retained.

    ..note:
        ??? history contains added note

    ..note:
        ***Limited*** to grids (source and target) where ...
        *   exactly two horizontal coordinates are identifiable as X and Y.
        *   both horizontal coordinates are bounded, and have the same
            coordinate system, which is geographical
            (?? in some cases a coordinate system may be inferred ??)
    """
    # process parameters to get input+output horizontal coordinates
    # NOTE: _get_xy_dim_coords tests exist, unique, and coor_systems match
    # (but allows coord_system == None)
    src_coords = i_regrid._get_xy_dim_coords(source_cube)
    if isinstance(grid_cube_or_coords, iris.cube.Cube):
        dst_coords = i_regrid._get_xy_dim_coords(grid_cube_or_coords)
    else:
        dst_coords = grid_cube_or_coords

    # check source+target coordinates are suitable + convert to Cartopy crs
    def check_xy_coords(name, coords):
        if _get_coord_crs(coords[0]) is None:
            raise ValueError(name + ' X+Y coordinates do not have a '
                             'coord_system.')
        if not coords[0].has_bounds() or not coords[1].has_bounds():
            raise ValueError(name + ' X+Y coordinates are not both bounded.')

    check_xy_coords('Source', src_coords)
    check_xy_coords('Target', dst_coords)

    # initialise the ESMF manager in case not already done
    ESMF.Manager()
        # Implements default settings.  If you don't like these, just call it
        # first yourself (then this call does nothing).

    # construct ESMF field objects on the  source and destination grids.
    src_field = _make_esmpy_field_from_coords(*src_coords)
    dst_field = _make_esmpy_field_from_coords(*dst_coords)

    # assign the source data, reformed into the right dimension order (=x,y)
    src_data = source_cube.data
    # FOR NOW: 2d only
    assert src_data.ndim == 2

    # Transpose source data into fixed (x,y) order for esmf
    src_dims_xy = [source_cube.coord_dims(coord)[0] for coord in src_coords]
    src_data = src_data.transpose(src_dims_xy)
    src_field.data[:] = src_data
    dst_field.data[:] = np.NaN  # FOR NOW: in case of arithmetic gremlins

    # perform the actual regridding
    regrid_method = ESMF.Regrid(src_field, dst_field,
                                src_mask_values=np.array([1], dtype=np.int32),
                                dst_mask_values=np.array([1], dtype=np.int32),
                                regrid_method=ESMF.RegridMethod.CONSERVE,
                                unmapped_action=ESMF.UnmappedAction.IGNORE)
    regrid_method(src_field, dst_field)

    # check for masking due to source MDI or cells outside original grid ...
    validmask_src = ESMF.Field(src_field.grid, 'validmask_src')
    validmask_dst = ESMF.Field(dst_field.grid, 'validmask_dst')
    if hasattr(src_data, 'mask') and np.any(src_data.mask):
        validmask_src.data[:] = np.logical_not(src_data.mask)
    else:
        validmask_src.data[:] = 1.0
    validmask_dst.data[:] = 0.0  # TODO: almost certainly not needed ...
    regrid_method(validmask_src, validmask_dst)

    # add a mask to the data field, if not all good
    data = dst_field.data
    dst_mask = validmask_dst.data < (1.0 - 1e-8)
    if np.any(dst_mask):
        data = np.ma.array(data, mask=dst_mask)

    # Transpose esmf result dims (X,Y) back to the order of the source
    # TODO: really need to "invert" ordering - but for 2D it is just the same!
    data = data.transpose(src_dims_xy)

    # Return result as a new cube based on the source.
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
