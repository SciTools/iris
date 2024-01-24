# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Support for conservative regridding via ESMPy.

.. note::

    .. deprecated:: 3.2.0

    This package will be removed in a future release.
    Please use
    `iris-esmf-regrid <https://github.com/SciTools-incubator/iris-esmf-regrid>`_
    instead.

"""

import functools

import cartopy.crs as ccrs
import numpy as np

import iris
from iris._deprecation import warn_deprecated
from iris.analysis._interpolation import get_xy_dim_coords
from iris.analysis._regrid import RectilinearRegridder, _create_cube
from iris.util import _meshgrid

wmsg = (
    "The 'iris.experimental.regrid_conservative' package is deprecated since "
    "version 3.2, and will be removed in a future release.  Please use "
    "iris-emsf-regrid instead. "
    "See https://github.com/SciTools-incubator/iris-esmf-regrid."
)
warn_deprecated(wmsg)


#: A static Cartopy Geodetic() instance for transforming to true-lat-lons.
_CRS_TRUELATLON = ccrs.Geodetic()


def _convert_latlons(crs, x_array, y_array):
    """Convert x+y coords in a given crs to (x,y) values in true-lat-lons.

    .. note::

        Uses a plain Cartopy Geodetic to convert to true-lat-lons.  This makes
        no allowance for a non-spherical earth.  But then, neither does ESMF.

    """
    ll_values = _CRS_TRUELATLON.transform_points(crs, x_array, y_array)
    return ll_values[..., 0], ll_values[..., 1]


def _make_esmpy_field(x_coord, y_coord, ref_name="field", data=None, mask=None):
    """Create an ESMPy ESMF.Field on given coordinates.

    Create a ESMF.Grid from the coordinates, defining corners and centre
    positions as lats+lons.
    Add a grid mask if provided.
    Create and return a Field mapped on this Grid, setting data if provided.

    Parameters
    ----------
    x_coord, y_coord : :class:`iris.coords.Coord`
        One-dimensional coordinates of shape (nx,) and (ny,).
        Their contiguous bounds define an ESMF.Grid of shape (nx, ny).
    data : :class:`numpy.ndarray`, shape (nx,ny), optional
        Set the Field data content.
    mask : :class:`numpy.ndarray`, bool, shape (nx,ny), optional
        Add a mask item to the grid, assigning it 0/1 where mask=False/True.

    """
    # Lazy import so we can build the docs with no ESMF.
    import ESMF

    # Create a Grid object describing the coordinate cells.
    dims = [len(coord.points) for coord in (x_coord, y_coord)]
    dims = np.array(dims, dtype=np.int32)  # specific type required by ESMF.
    grid = ESMF.Grid(dims)

    # Get all cell corner coordinates as true-lat-lons
    x_bounds, y_bounds = _meshgrid(
        x_coord.contiguous_bounds(), y_coord.contiguous_bounds()
    )
    grid_crs = x_coord.coord_system.as_cartopy_crs()
    lon_bounds, lat_bounds = _convert_latlons(grid_crs, x_bounds, y_bounds)

    # Add grid 'coord' element for corners, and fill with corner values.
    grid.add_coords(staggerloc=ESMF.StaggerLoc.CORNER)
    grid_corners_x = grid.get_coords(0, ESMF.StaggerLoc.CORNER)
    grid_corners_x[:] = lon_bounds.T
    grid_corners_y = grid.get_coords(1, ESMF.StaggerLoc.CORNER)
    grid_corners_y[:] = lat_bounds.T

    # calculate the cell centre-points
    # NOTE: we don't care about Iris' idea of where the points 'really' are
    # *but* ESMF requires the data in the CENTER for conservative regrid,
    # according to the documentation :
    #  - https://www.earthsystemmodeling.org/
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
    x_points, y_points = _meshgrid(x_centres, y_centres)
    lon_points, lat_points = _convert_latlons(grid_crs, x_points, y_points)

    # Add grid 'coord' element for centres + fill with centre-points values.
    grid.add_coords(staggerloc=ESMF.StaggerLoc.CENTER)
    grid_centers_x = grid.get_coords(0, ESMF.StaggerLoc.CENTER)
    grid_centers_x[:] = lon_points.T
    grid_centers_y = grid.get_coords(1, ESMF.StaggerLoc.CENTER)
    grid_centers_y[:] = lat_points.T

    # Add a mask item, if requested
    if mask is not None:
        grid.add_item(ESMF.GridItem.MASK, [ESMF.StaggerLoc.CENTER])
        grid_mask = grid.get_item(ESMF.GridItem.MASK)
        grid_mask[:] = np.where(mask, 1, 0)

    # create a Field based on this grid
    field = ESMF.Field(grid, ref_name)

    # assign data content, if provided
    if data is not None:
        field.data[:] = data

    return field


def regrid_conservative_via_esmpy(source_cube, grid_cube):
    """Perform a conservative regridding with ESMPy.

    .. note ::

        .. deprecated:: 3.2.0

        This function is scheduled to be removed in a future release.
        Please use
        `iris-esmf-regrid <https://github.com/SciTools-incubator/iris-esmf-regrid>`_
        instead.

        For example :

        .. code::

            from emsf_regrid.schemes import ESMFAreaWeighted
            result = src_cube.regrid(grid_cube, ESMFAreaWeighted())

    Regrids the data of a source cube onto a new grid defined by a destination
    cube.

    Parameters
    ----------
    source_cube : :class:`iris.cube.Cube`
        Source data.  Must have two identifiable horizontal dimension
        coordinates.
    grid_cube : :class:`iris.cube.Cube`
        Define the target horizontal grid:  Only the horizontal dimension
        coordinates are actually used.

    Returns
    -------
    :class:`iris.cube.Cube`
        A new cube derived from source_cube, regridded onto the specified
        horizontal grid.

    Notes
    -----
    Any additional coordinates which map onto the horizontal dimensions are
    removed, while all other metadata is retained.
    If there are coordinate factories with 2d horizontal reference surfaces,
    the reference surfaces are also regridded, using ordinary bilinear
    interpolation.

    .. note::

        Both source and destination cubes must have two dimension coordinates
        identified with axes 'X' and 'Y' which share a coord_system with a
        Cartopy CRS.
        The grids are defined by :meth:`iris.coords.Coord.contiguous_bounds` of
        these.

    .. note::

        Initialises the ESMF Manager, if it was not already called.
        This implements default Manager operations (e.g. logging).

        To alter this, make a prior call to ESMF.Manager().

    """
    wmsg = (
        "The function "
        "'iris.experimental.regrid_conservative."
        "regrid_weighted_curvilinear_to_rectilinear' "
        "has been deprecated, and will be removed in a future release.  "
        "Please consult the docstring for details."
    )
    warn_deprecated(wmsg)

    # Lazy import so we can build the docs with no ESMF.
    import ESMF

    # Get source + target XY coordinate pairs and check they are suitable.
    src_coords = get_xy_dim_coords(source_cube)
    dst_coords = get_xy_dim_coords(grid_cube)
    src_cs = src_coords[0].coord_system
    grid_cs = dst_coords[0].coord_system
    if src_cs is None or grid_cs is None:
        raise ValueError(
            "Both 'src' and 'grid' Cubes must have a"
            " coordinate system for their rectilinear grid"
            " coordinates."
        )

    if src_cs.as_cartopy_crs() is None or grid_cs.as_cartopy_crs() is None:
        raise ValueError(
            "Both 'src' and 'grid' Cubes coord_systems must have "
            "a valid associated Cartopy CRS."
        )

    def _valid_units(coord):
        if isinstance(
            coord.coord_system,
            (iris.coord_systems.GeogCS, iris.coord_systems.RotatedGeogCS),
        ):
            valid_units = "degrees"
        else:
            valid_units = "m"
        return coord.units == valid_units

    if not all(_valid_units(coord) for coord in src_coords + dst_coords):
        raise ValueError("Unsupported units: must be 'degrees' or 'm'.")

    # Initialise the ESMF manager in case it was not already done.
    ESMF.Manager()

    # Create a data array for the output cube.
    src_dims_xy = [source_cube.coord_dims(coord)[0] for coord in src_coords]
    # Size matches source, except for X+Y dimensions
    dst_shape = np.array(source_cube.shape)
    dst_shape[src_dims_xy] = [coord.shape[0] for coord in dst_coords]
    # NOTE: result array is masked -- fix this afterward if all unmasked
    fullcube_data = np.ma.zeros(dst_shape)

    # Iterate 2d slices over all possible indices of the 'other' dimensions
    all_other_dims = [
        i_dim for i_dim in range(source_cube.ndim) if i_dim not in src_dims_xy
    ]
    all_combinations_of_other_inds = np.ndindex(*dst_shape[all_other_dims])
    for other_indices in all_combinations_of_other_inds:
        # Construct a tuple of slices to address the 2d xy field
        slice_indices_array = np.array([slice(None)] * source_cube.ndim)
        slice_indices_array[all_other_dims] = other_indices
        slice_indices_tuple = tuple(slice_indices_array)

        # Get the source data, reformed into the right dimension order, (x,y).
        src_data_2d = source_cube.data[slice_indices_tuple]
        if src_dims_xy[0] > src_dims_xy[1]:
            src_data_2d = src_data_2d.transpose()

        # Work out whether we have missing data to define a source grid mask.
        if np.ma.is_masked(src_data_2d):
            srcdata_mask = np.ma.getmask(src_data_2d)
        else:
            srcdata_mask = None

        # Construct ESMF Field objects on source and destination grids.
        src_field = _make_esmpy_field(
            src_coords[0], src_coords[1], data=src_data_2d, mask=srcdata_mask
        )
        dst_field = _make_esmpy_field(dst_coords[0], dst_coords[1])

        # Make Field for destination coverage fraction (for missing data calc).
        coverage_field = ESMF.Field(dst_field.grid, "validmask_dst")

        # Do the actual regrid with ESMF.
        mask_flag_values = np.array([1], dtype=np.int32)
        regrid_method = ESMF.Regrid(
            src_field,
            dst_field,
            src_mask_values=mask_flag_values,
            regrid_method=ESMF.RegridMethod.CONSERVE,
            unmapped_action=ESMF.UnmappedAction.IGNORE,
            dst_frac_field=coverage_field,
        )
        regrid_method(src_field, dst_field)
        data = np.ma.masked_array(dst_field.data)

        # Convert destination 'coverage fraction' into a missing-data mask.
        # Set = wherever part of cell goes outside source grid, or overlaps a
        # masked source cell.
        coverage_tolerance_threshold = 1.0 - 1.0e-8
        data.mask = coverage_field.data < coverage_tolerance_threshold

        # Transpose ESMF result dims (X,Y) back to the order of the source
        if src_dims_xy[0] > src_dims_xy[1]:
            data = data.transpose()

        # Paste regridded slice back into parent array
        fullcube_data[slice_indices_tuple] = data

    # Remove the data mask if completely unused.
    if not np.ma.is_masked(fullcube_data):
        fullcube_data = np.array(fullcube_data)

    # Generate a full 2d sample grid, as required for regridding orography
    # NOTE: as seen in "regrid_bilinear_rectilinear_src_and_grid"
    # TODO: can this not also be wound into the _create_cube method ?
    src_cs = src_coords[0].coord_system
    sample_grid_x, sample_grid_y = RectilinearRegridder._sample_grid(
        src_cs, dst_coords[0], dst_coords[1]
    )

    # Return result as a new cube based on the source.
    # TODO: please tidy this interface !!!
    _regrid_callback = functools.partial(
        RectilinearRegridder._regrid,
        src_x_coord=src_coords[0],
        src_y_coord=src_coords[1],
        sample_grid_x=sample_grid_x,
        sample_grid_y=sample_grid_y,
    )

    def regrid_callback(*args, **kwargs):
        _data, dims = args
        return _regrid_callback(_data, *dims, **kwargs)

    return _create_cube(
        fullcube_data,
        source_cube,
        [src_dims_xy[0], src_dims_xy[1]],
        [dst_coords[0], dst_coords[1]],
        2,
        regrid_callback,
    )
