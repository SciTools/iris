# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
import functools

import cf_units
import numpy as np
import numpy.ma as ma
from scipy.sparse import csr_array

from iris._lazy_data import map_complete_blocks
from iris.analysis._interpolation import get_xy_dim_coords, snapshot_grid
from iris.analysis._regrid import RectilinearRegridder, _create_cube
import iris.analysis.cartography
import iris.coord_systems
from iris.util import _meshgrid


class AreaWeightedRegridder:
    """
    This class provides support for performing area-weighted regridding.

    """

    def __init__(self, src_grid_cube, target_grid_cube, mdtol=1):
        """
        Create an area-weighted regridder for conversions between the source
        and target grids.

        Args:

        * src_grid_cube:
            The :class:`~iris.cube.Cube` providing the source grid.
        * target_grid_cube:
            The :class:`~iris.cube.Cube` providing the target grid.

        Kwargs:

        * mdtol (float):
            Tolerance of missing data. The value returned in each element of
            the returned array will be masked if the fraction of masked data
            exceeds mdtol. mdtol=0 means no missing data is tolerated while
            mdtol=1 will mean the resulting element will be masked if and only
            if all the contributing elements of data are masked.
            Defaults to 1.

        .. Note::

            Both source and target cubes must have an XY grid defined by
            separate X and Y dimensions with dimension coordinates.
            All of the XY dimension coordinates must also be bounded, and have
            the same coordinate system.

        """
        # Snapshot the state of the source cube to ensure that the regridder is
        # impervious to external changes to the original cubes.
        self._src_grid = snapshot_grid(src_grid_cube)

        # Missing data tolerance.
        if not (0 <= mdtol <= 1):
            msg = "Value for mdtol must be in range 0 - 1, got {}."
            raise ValueError(msg.format(mdtol))
        self._mdtol = mdtol

        # Store regridding information
        _regrid_info = _regrid_area_weighted_rectilinear_src_and_grid__prepare(
            src_grid_cube, target_grid_cube
        )
        (
            src_x,
            src_y,
            src_x_dim,
            src_y_dim,
            self.grid_x,
            self.grid_y,
            self.meshgrid_x,
            self.meshgrid_y,
            self.weights,
        ) = _regrid_info

    def __call__(self, cube):
        """
        Regrid this :class:`~iris.cube.Cube` onto the target grid of
        this :class:`AreaWeightedRegridder`.

        The given cube must be defined with the same grid as the source
        grid used to create this :class:`AreaWeightedRegridder`.

        If the source cube has lazy data, the returned cube will also
        have lazy data.

        Args:

        * cube:
            A :class:`~iris.cube.Cube` to be regridded.

        Returns:
            A cube defined with the horizontal dimensions of the target
            and the other dimensions from this cube. The data values of
            this cube will be converted to values on the new grid using
            area-weighted regridding.

        .. note::

            If the source cube has lazy data,
            `chunks <https://docs.dask.org/en/latest/array-chunks.html>`__
            in the horizontal dimensions will be combined before regridding.

        """
        src_x, src_y = get_xy_dim_coords(cube)
        if (src_x, src_y) != self._src_grid:
            raise ValueError(
                "The given cube is not defined on the same "
                "source grid as this regridder."
            )
        src_x_dim = cube.coord_dims(src_x)[0]
        src_y_dim = cube.coord_dims(src_y)[0]
        _regrid_info = (
            src_x,
            src_y,
            src_x_dim,
            src_y_dim,
            self.grid_x,
            self.grid_y,
            self.meshgrid_x,
            self.meshgrid_y,
            self.weights,
        )
        return _regrid_area_weighted_rectilinear_src_and_grid__perform(
            cube, _regrid_info, mdtol=self._mdtol
        )


#
# Support routines, all originally in iris.experimental.regrid
#


def _get_xy_coords(cube):
    """
    Return the x and y coordinates from a cube.

    This function will preferentially return a pair of dimension
    coordinates (if there are more than one potential x or y dimension
    coordinates a ValueError will be raised). If the cube does not have
    a pair of x and y dimension coordinates it will return 1D auxiliary
    coordinates (including scalars). If there is not one and only one set
    of x and y auxiliary coordinates a ValueError will be raised.

    Having identified the x and y coordinates, the function checks that they
    have equal coordinate systems and that they do not occupy the same
    dimension on the cube.

    Args:

    * cube:
        An instance of :class:`iris.cube.Cube`.

    Returns:
        A tuple containing the cube's x and y coordinates.

    """
    # Look for a suitable dimension coords first.
    x_coords = cube.coords(axis="x", dim_coords=True)
    if not x_coords:
        # If there is no x coord in dim_coords look for scalars or
        # monotonic coords in aux_coords.
        x_coords = [
            coord
            for coord in cube.coords(axis="x", dim_coords=False)
            if coord.ndim == 1 and coord.is_monotonic()
        ]
    if len(x_coords) != 1:
        raise ValueError(
            "Cube {!r} must contain a single 1D x "
            "coordinate.".format(cube.name())
        )
    x_coord = x_coords[0]

    # Look for a suitable dimension coords first.
    y_coords = cube.coords(axis="y", dim_coords=True)
    if not y_coords:
        # If there is no y coord in dim_coords look for scalars or
        # monotonic coords in aux_coords.
        y_coords = [
            coord
            for coord in cube.coords(axis="y", dim_coords=False)
            if coord.ndim == 1 and coord.is_monotonic()
        ]
    if len(y_coords) != 1:
        raise ValueError(
            "Cube {!r} must contain a single 1D y "
            "coordinate.".format(cube.name())
        )
    y_coord = y_coords[0]

    if x_coord.coord_system != y_coord.coord_system:
        raise ValueError(
            "The cube's x ({!r}) and y ({!r}) "
            "coordinates must have the same coordinate "
            "system.".format(x_coord.name(), y_coord.name())
        )

    # The x and y coordinates must describe different dimensions
    # or be scalar coords.
    x_dims = cube.coord_dims(x_coord)
    x_dim = None
    if x_dims:
        x_dim = x_dims[0]

    y_dims = cube.coord_dims(y_coord)
    y_dim = None
    if y_dims:
        y_dim = y_dims[0]

    if x_dim is not None and y_dim == x_dim:
        raise ValueError(
            "The cube's x and y coords must not describe the "
            "same data dimension."
        )

    return x_coord, y_coord


def _within_bounds(src_bounds, tgt_bounds, orderswap=False):
    """
    Determine which target bounds lie within the extremes of the source bounds.

    Args:

    * src_bounds (ndarray):
        An (n, 2) shaped array of monotonic contiguous source bounds.
    * tgt_bounds (ndarray):
        An (n, 2) shaped array corresponding to the target bounds.

    Kwargs:

    * orderswap (bool):
        A Boolean indicating whether the target bounds are in descending order
        (True). Defaults to False.

    Returns:
        Boolean ndarray, indicating whether each target bound is within the
        extremes of the source bounds.

    """
    min_bound = np.min(src_bounds) - 1e-14
    max_bound = np.max(src_bounds) + 1e-14

    # Swap upper-lower is necessary.
    if orderswap is True:
        upper, lower = tgt_bounds.T
    else:
        lower, upper = tgt_bounds.T

    return ((lower <= max_bound) * (lower >= min_bound)) * (
        (upper <= max_bound) * (upper >= min_bound)
    )


def _cropped_bounds(bounds, lower, upper):
    """
    Return a new bounds array and corresponding slice object (or indices) of
    the original data array, resulting from cropping the provided bounds
    between the specified lower and upper values. The bounds at the
    extremities will be truncated so that they start and end with lower and
    upper.

    This function will return an empty NumPy array and slice if there is no
    overlap between the region covered by bounds and the region from lower to
    upper.

    If lower > upper the resulting bounds may not be contiguous and the
    indices object will be a tuple of indices rather than a slice object.

    Args:

    * bounds:
        An (n, 2) shaped array of monotonic contiguous bounds.
    * lower:
        Lower bound at which to crop the bounds array.
    * upper:
        Upper bound at which to crop the bounds array.

    Returns:
        A tuple of the new bounds array and the corresponding slice object or
        indices from the zeroth axis of the original array.

    """
    reversed_flag = False
    # Ensure order is increasing.
    if bounds[0, 0] > bounds[-1, 0]:
        # Reverse bounds
        bounds = bounds[::-1, ::-1]
        reversed_flag = True

    # Number of bounds.
    n = bounds.shape[0]

    if lower <= upper:
        if lower > bounds[-1, 1] or upper < bounds[0, 0]:
            new_bounds = bounds[0:0]
            indices = slice(0, 0)
        else:
            # A single region lower->upper.
            if lower < bounds[0, 0]:
                # Region extends below bounds so use first lower bound.
                lindex = 0
                lower = bounds[0, 0]
            else:
                # Index of last lower bound less than or equal to lower.
                lindex = np.nonzero(bounds[:, 0] <= lower)[0][-1]
            if upper > bounds[-1, 1]:
                # Region extends above bounds so use last upper bound.
                uindex = n - 1
                upper = bounds[-1, 1]
            else:
                # Index of first upper bound greater than or equal to
                # upper.
                uindex = np.nonzero(bounds[:, 1] >= upper)[0][0]
            # Extract the bounds in our region defined by lower->upper.
            new_bounds = np.copy(bounds[lindex : (uindex + 1), :])
            # Replace first and last values with specified bounds.
            new_bounds[0, 0] = lower
            new_bounds[-1, 1] = upper
            if reversed_flag:
                indices = slice(n - (uindex + 1), n - lindex)
            else:
                indices = slice(lindex, uindex + 1)
    else:
        # Two regions [0]->upper, lower->[-1]
        # [0]->upper
        if upper < bounds[0, 0]:
            # Region outside src bounds.
            new_bounds_left = bounds[0:0]
            indices_left = tuple()
            slice_left = slice(0, 0)
        else:
            if upper > bounds[-1, 1]:
                # Whole of bounds.
                uindex = n - 1
                upper = bounds[-1, 1]
            else:
                # Index of first upper bound greater than or equal to upper.
                uindex = np.nonzero(bounds[:, 1] >= upper)[0][0]
            # Extract the bounds in our region defined by [0]->upper.
            new_bounds_left = np.copy(bounds[0 : (uindex + 1), :])
            # Replace last value with specified bound.
            new_bounds_left[-1, 1] = upper
            if reversed_flag:
                indices_left = tuple(range(n - (uindex + 1), n))
                slice_left = slice(n - (uindex + 1), n)
            else:
                indices_left = tuple(range(0, uindex + 1))
                slice_left = slice(0, uindex + 1)
        # lower->[-1]
        if lower > bounds[-1, 1]:
            # Region is outside src bounds.
            new_bounds_right = bounds[0:0]
            indices_right = tuple()
            slice_right = slice(0, 0)
        else:
            if lower < bounds[0, 0]:
                # Whole of bounds.
                lindex = 0
                lower = bounds[0, 0]
            else:
                # Index of last lower bound less than or equal to lower.
                lindex = np.nonzero(bounds[:, 0] <= lower)[0][-1]
            # Extract the bounds in our region defined by lower->[-1].
            new_bounds_right = np.copy(bounds[lindex:, :])
            # Replace first value with specified bound.
            new_bounds_right[0, 0] = lower
            if reversed_flag:
                indices_right = tuple(range(0, n - lindex))
                slice_right = slice(0, n - lindex)
            else:
                indices_right = tuple(range(lindex, n))
                slice_right = slice(lindex, None)

        if reversed_flag:
            # Flip everything around.
            indices_left, indices_right = indices_right, indices_left
            slice_left, slice_right = slice_right, slice_left

        # Combine regions.
        new_bounds = np.concatenate((new_bounds_left, new_bounds_right))
        # Use slices if possible, but if we have two regions use indices.
        if indices_left and indices_right:
            indices = indices_left + indices_right
        elif indices_left:
            indices = slice_left
        elif indices_right:
            indices = slice_right
        else:
            indices = slice(0, 0)

    if reversed_flag:
        new_bounds = new_bounds[::-1, ::-1]

    return new_bounds, indices


def _get_bounds_in_units(coord, units, dtype):
    """Return a copy of coord's bounds in the specified units and dtype."""
    # The bounds are cast to dtype before conversion to prevent issues when
    # mixing float32 and float64 types.
    return coord.units.convert(
        coord.contiguous_bounds().astype(dtype), units
    ).astype(dtype)


def _regrid_area_weighted_rectilinear_src_and_grid__prepare(
    src_cube, grid_cube
):
    """
    First (setup) part of 'regrid_area_weighted_rectilinear_src_and_grid'.

    Check inputs and calculate related info. The 'regrid info' returned
    can be re-used over many 2d slices.

    """
    # Get the 1d monotonic (or scalar) src and grid coordinates.
    src_x, src_y = _get_xy_coords(src_cube)
    grid_x, grid_y = _get_xy_coords(grid_cube)

    # Condition 1: All x and y coordinates must have contiguous bounds to
    # define areas.
    if (
        not src_x.is_contiguous()
        or not src_y.is_contiguous()
        or not grid_x.is_contiguous()
        or not grid_y.is_contiguous()
    ):
        raise ValueError(
            "The horizontal grid coordinates of both the source "
            "and grid cubes must have contiguous bounds."
        )

    # Condition 2: Everything must have the same coordinate system.
    src_cs = src_x.coord_system
    grid_cs = grid_x.coord_system
    if src_cs != grid_cs:
        raise ValueError(
            "The horizontal grid coordinates of both the source "
            "and grid cubes must have the same coordinate "
            "system."
        )

    # Condition 3: cannot create vector coords from scalars.
    src_x_dims = src_cube.coord_dims(src_x)
    src_x_dim = None
    if src_x_dims:
        src_x_dim = src_x_dims[0]
    src_y_dims = src_cube.coord_dims(src_y)
    src_y_dim = None
    if src_y_dims:
        src_y_dim = src_y_dims[0]
    if (
        src_x_dim is None
        and grid_x.shape[0] != 1
        or src_y_dim is None
        and grid_y.shape[0] != 1
    ):
        raise ValueError(
            "The horizontal grid coordinates of source cube "
            "includes scalar coordinates, but the new grid does "
            "not. The new grid must not require additional data "
            "dimensions to be created."
        )

    # Determine whether to calculate flat or spherical areas.
    # Don't only rely on coord system as it may be None.
    spherical = (
        isinstance(
            src_cs,
            (iris.coord_systems.GeogCS, iris.coord_systems.RotatedGeogCS),
        )
        or src_x.units == "degrees"
        or src_x.units == "radians"
    )

    # Get src and grid bounds in the same units.
    x_units = cf_units.Unit("radians") if spherical else src_x.units
    y_units = cf_units.Unit("radians") if spherical else src_y.units

    # Operate in highest precision.
    src_dtype = np.promote_types(src_x.bounds.dtype, src_y.bounds.dtype)
    grid_dtype = np.promote_types(grid_x.bounds.dtype, grid_y.bounds.dtype)
    dtype = np.promote_types(src_dtype, grid_dtype)

    src_x_bounds = _get_bounds_in_units(src_x, x_units, dtype)
    src_y_bounds = _get_bounds_in_units(src_y, y_units, dtype)
    grid_x_bounds = _get_bounds_in_units(grid_x, x_units, dtype)
    grid_y_bounds = _get_bounds_in_units(grid_y, y_units, dtype)

    # TODO: consider removing this.
    # Create 2d meshgrids as required by _create_cube func.
    meshgrid_x, meshgrid_y = _meshgrid(grid_x.points, grid_y.points)

    # Wrapping of longitudes.
    if spherical:
        modulus = x_units.modulus
    else:
        modulus = None

    def _calculate_regrid_area_weighted_weights(
        src_x_bounds,
        src_y_bounds,
        grid_x_bounds,
        grid_y_bounds,
        spherical,
        modulus=None,
    ):
        src_shape = (len(src_x_bounds) - 1, len(src_y_bounds) - 1)
        tgt_shape = (len(grid_x_bounds) - 1, len(grid_y_bounds) - 1)

        if spherical:
            # Changing the dtype here replicates old regridding behaviour.
            dtype = np.float64
            src_x_bounds = src_x_bounds.astype(dtype)
            src_y_bounds = src_y_bounds.astype(dtype)
            grid_x_bounds = grid_x_bounds.astype(dtype)
            grid_y_bounds = grid_y_bounds.astype(dtype)

            src_y_bounds = np.sin(src_y_bounds)
            grid_y_bounds = np.sin(grid_y_bounds)
        x_info = _get_coord_to_coord_matrix(
            src_x_bounds, grid_x_bounds, circular=spherical, mod=modulus
        )
        y_info = _get_coord_to_coord_matrix(src_y_bounds, grid_y_bounds)
        weights_matrix = _combine_xy_weights(
            x_info, y_info, src_shape, tgt_shape
        )
        return weights_matrix

    weights = _calculate_regrid_area_weighted_weights(
        src_x_bounds,
        src_y_bounds,
        grid_x_bounds,
        grid_y_bounds,
        spherical,
        modulus,
    )
    return (
        src_x,
        src_y,
        src_x_dim,
        src_y_dim,
        grid_x,
        grid_y,
        meshgrid_x,
        meshgrid_y,
        weights,
    )


def _regrid_area_weighted_rectilinear_src_and_grid__perform(
    src_cube, regrid_info, mdtol
):
    """
    Second (regrid) part of 'regrid_area_weighted_rectilinear_src_and_grid'.

    Perform the prepared regrid calculation on a single 2d cube.

    """
    (
        src_x,
        src_y,
        src_x_dim,
        src_y_dim,
        grid_x,
        grid_y,
        meshgrid_x,
        meshgrid_y,
        weights,
    ) = regrid_info

    tgt_shape = (len(grid_x.points), len(grid_y.points))

    # Calculate new data array for regridded cube.
    regrid = functools.partial(
        _regrid_along_dims,
        x_dim=src_x_dim,
        y_dim=src_y_dim,
        weights=weights,
        tgt_shape=tgt_shape,
        mdtol=mdtol,
    )

    new_data = map_complete_blocks(
        src_cube, regrid, (src_y_dim, src_x_dim), meshgrid_x.shape
    )

    # Wrap up the data as a Cube.

    _regrid_callback = functools.partial(
        RectilinearRegridder._regrid,
        src_x_coord=src_x,
        src_y_coord=src_y,
        sample_grid_x=meshgrid_x,
        sample_grid_y=meshgrid_y,
    )
    # TODO: investigate if an area weighted callback would be more appropriate.
    # _regrid_callback = functools.partial(
    #     _regrid_along_dims,
    #     weights=weights,
    #     tgt_shape=tgt_shape,
    #     mdtol=mdtol,
    # )

    def regrid_callback(*args, **kwargs):
        _data, dims = args
        return _regrid_callback(_data, *dims, **kwargs)

    new_cube = _create_cube(
        new_data,
        src_cube,
        [src_x_dim, src_y_dim],
        [grid_x, grid_y],
        2,
        regrid_callback,
    )

    # Slice out any length 1 dimensions.
    indices = [slice(None, None)] * new_data.ndim
    if src_x_dim is not None and new_cube.shape[src_x_dim] == 1:
        indices[src_x_dim] = 0
    if src_y_dim is not None and new_cube.shape[src_y_dim] == 1:
        indices[src_y_dim] = 0
    if 0 in indices:
        new_cube = new_cube[tuple(indices)]

    return new_cube


def _get_coord_to_coord_matrix(
    src_bounds, tgt_bounds, circular=False, mod=None
):
    m = len(tgt_bounds) - 1
    n = len(src_bounds) - 1

    src_decreasing = src_bounds[0] > src_bounds[1]
    tgt_decreasing = tgt_bounds[0] > tgt_bounds[1]
    if src_decreasing:
        src_bounds = src_bounds[::-1]
    if tgt_decreasing:
        tgt_bounds = tgt_bounds[::-1]

    if circular:
        adjust = (tgt_bounds.min() - src_bounds.min()) // mod
        src_bounds = src_bounds + (mod * adjust)
        src_bounds = np.append(src_bounds, src_bounds + mod)
        nn = (2 * n) + 1
    else:
        nn = n

    i = max(np.searchsorted(tgt_bounds, src_bounds[0], side="right") - 1, 0)
    j = max(np.searchsorted(src_bounds, tgt_bounds[0], side="right") - 1, 0)

    data = []
    rows = []
    cols = []

    floor = max(tgt_bounds[i], src_bounds[j])
    while i < m and j < nn:
        rows.append(i)
        cols.append(j)
        if tgt_bounds[i + 1] < src_bounds[j + 1]:
            weight = (tgt_bounds[i + 1] - floor) / (
                tgt_bounds[i + 1] - tgt_bounds[i]
            )
            floor = tgt_bounds[i + 1]
            i += 1
        elif tgt_bounds[i + 1] < src_bounds[j + 1]:
            weight = (tgt_bounds[i + 1] - floor) / (
                tgt_bounds[i + 1] - tgt_bounds[i]
            )
            floor = tgt_bounds[i + 1]
            i += 1
            j += 1
        else:
            weight = (src_bounds[j + 1] - floor) / (
                tgt_bounds[i + 1] - tgt_bounds[i]
            )
            floor = src_bounds[j + 1]
            j += 1
        data.append(weight)

    data = np.array(data)
    rows = np.array(rows)
    cols = np.array(cols)

    if circular:
        # remove out of bounds points
        oob = np.where(cols == n)
        data = np.delete(data, oob)
        rows = np.delete(rows, oob)
        cols = np.delete(cols, oob)
        # wrap indices
        cols = cols % (n + 1)

    if src_decreasing:
        cols = n - cols - 1
    if tgt_decreasing:
        rows = m - rows - 1
    return data, rows, cols


def _combine_xy_weights(x_info, y_info, src_shape, tgt_shape):
    x_src, y_src = src_shape
    x_tgt, y_tgt = tgt_shape
    src_size = x_src * y_src
    tgt_size = x_tgt * y_tgt
    x_weight, x_rows, x_cols = x_info
    y_weight, y_rows, y_cols = y_info

    xy_weight = x_weight[:, np.newaxis] * y_weight[np.newaxis, :]
    xy_weight = xy_weight.flatten()

    xy_rows = (x_rows[:, np.newaxis] * y_tgt) + y_rows[np.newaxis, :]
    xy_rows = xy_rows.flatten()

    xy_cols = (x_cols[:, np.newaxis] * y_src) + y_cols[np.newaxis, :]
    xy_cols = xy_cols.flatten()

    combined_weights = csr_array(
        (xy_weight, (xy_rows, xy_cols)), shape=(tgt_size, src_size)
    )
    return combined_weights


def _regrid_no_masks(data, weights, tgt_shape):
    extra_dims = len(data.shape) > 2
    if extra_dims:
        extra_shape = data.shape[2:]
        result = data.reshape(np.prod(data.shape[:2]), -1)
    else:
        result = data.flatten()
    result = weights @ result
    if extra_dims:
        result = result.reshape(*(tgt_shape + extra_shape))
    else:
        result = result.reshape(*tgt_shape)
    return result


def _standard_regrid(data, weights, tgt_shape, mdtol, oob_invalid=True):
    unmasked = ~ma.getmaskarray(data)
    weight_sums = _regrid_no_masks(unmasked, weights, tgt_shape)
    mdtol = max(mdtol, 1e-8)
    tgt_mask = weight_sums > 1 - mdtol
    # TODO: make this more efficient
    if oob_invalid:
        inbound_sums = _regrid_no_masks(np.ones_like(data), weights, tgt_shape)
        oob_mask = inbound_sums > 1 - 1e-8
        tgt_mask = tgt_mask * oob_mask
    masked_weight_sums = weight_sums * tgt_mask
    normalisations = np.ones_like(weight_sums)
    normalisations[tgt_mask] /= masked_weight_sums[tgt_mask]

    if ma.isMaskedArray(data):
        fill_value = data.fill_value
        normalisations = ma.array(
            normalisations, mask=~tgt_mask, fill_value=fill_value
        )
    elif np.any(~tgt_mask):
        normalisations = ma.array(normalisations, mask=~tgt_mask)

    # Use input cube dtype or convert values to the smallest possible float
    # dtype when necessary.
    dtype = np.promote_types(data.dtype, np.float16)

    result = _regrid_no_masks(ma.getdata(data), weights, tgt_shape)
    result = result * normalisations
    result = result.astype(dtype)
    return result


def _regrid_along_dims(data, x_dim, y_dim, weights, tgt_shape, mdtol):
    # TODO: check that this is equivalent to the reordering in curvilinear regridding!
    if x_dim is None:
        x_none = True
        data = np.expand_dims(data, 0)
        x_dim = 0
        if y_dim is not None:
            y_dim += 1
    else:
        x_none = False
    if y_dim is None:
        y_none = True
        data = np.expand_dims(data, 0)
        y_dim = 0
        x_dim += 1
    else:
        y_none = False
    data = np.moveaxis(data, [x_dim, y_dim], [0, 1])
    result = _standard_regrid(data, weights, tgt_shape, mdtol)
    # Ensure consistent ordering with old behaviour
    result = ma.array(result, order="F")
    result = np.moveaxis(result, [0, 1], [x_dim, y_dim])
    if y_none:
        result = result[0]
    if x_none:
        result = result[0]
    return result
