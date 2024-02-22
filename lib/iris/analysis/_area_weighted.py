# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
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
    """Provide support for performing area-weighted regridding."""

    def __init__(self, src_grid_cube, target_grid_cube, mdtol=1):
        """Create an area-weighted regridder for conversions between the source and target grids.

        Parameters
        ----------
        src_grid_cube : :class:`~iris.cube.Cube`
            The :class:`~iris.cube.Cube` providing the source grid.
        target_grid_cube : :class:`~iris.cube.Cube`
            The :class:`~iris.cube.Cube` providing the target grid.
        mdtol : float, default=1
            Tolerance of missing data. The value returned in each element of
            the returned array will be masked if the fraction of masked data
            exceeds mdtol. mdtol=0 means no missing data is tolerated while
            mdtol=1 will mean the resulting element will be masked if and only
            if all the contributing elements of data are masked.
            Defaults to 1.

        Notes
        -----
        .. note::

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
        """Regrid :class:`~iris.cube.Cube` onto target grid :class:`AreaWeightedRegridder`.

        The given cube must be defined with the same grid as the source
        grid used to create this :class:`AreaWeightedRegridder`.

        If the source cube has lazy data, the returned cube will also
        have lazy data.

        Parameters
        ----------
        cube : :class:`~iris.cube.Cube`
            A :class:`~iris.cube.Cube` to be regridded.

        Returns
        -------
        :class:`~iris.cube.Cube`
            A cube defined with the horizontal dimensions of the target
            and the other dimensions from this cube. The data values of
            this cube will be converted to values on the new grid using
            area-weighted regridding.

        Notes
        -----
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
    """Return the x and y coordinates from a cube.

    This function will preferentially return a pair of dimension
    coordinates (if there are more than one potential x or y dimension
    coordinates a ValueError will be raised). If the cube does not have
    a pair of x and y dimension coordinates it will return 1D auxiliary
    coordinates (including scalars). If there is not one and only one set
    of x and y auxiliary coordinates a ValueError will be raised.

    Having identified the x and y coordinates, the function checks that they
    have equal coordinate systems and that they do not occupy the same
    dimension on the cube.

    Parameters
    ----------
    cube : :class:`iris.cube.Cube`
        An instance of :class:`iris.cube.Cube`.

    Returns
    -------
    tuple
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
            "Cube {!r} must contain a single 1D x coordinate.".format(cube.name())
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
            "Cube {!r} must contain a single 1D y coordinate.".format(cube.name())
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
            "The cube's x and y coords must not describe the same data dimension."
        )

    return x_coord, y_coord


def _get_bounds_in_units(coord, units, dtype):
    """Return a copy of coord's bounds in the specified units and dtype.

    Return as contiguous bounds.

    """
    # The bounds are cast to dtype before conversion to prevent issues when
    # mixing float32 and float64 types.
    return coord.units.convert(coord.contiguous_bounds().astype(dtype), units).astype(
        dtype
    )


def _regrid_area_weighted_rectilinear_src_and_grid__prepare(src_cube, grid_cube):
    """First (setup) part of 'regrid_area_weighted_rectilinear_src_and_grid'.

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
        """Return weights matrix to be used in regridding."""
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
        x_info = _get_coord_to_coord_matrix_info(
            src_x_bounds, grid_x_bounds, circular=spherical, mod=modulus
        )
        y_info = _get_coord_to_coord_matrix_info(src_y_bounds, grid_y_bounds)
        weights_matrix = _combine_xy_weights(x_info, y_info, src_shape, tgt_shape)
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
    """Second (regrid) part of 'regrid_area_weighted_rectilinear_src_and_grid'.

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

    tgt_shape = (len(grid_y.points), len(grid_x.points))

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


def _get_coord_to_coord_matrix_info(src_bounds, tgt_bounds, circular=False, mod=None):
    """First part of weight calculation.

    Calculate the weights contribution from a single pair of
    coordinate bounds. Search for pairs of overlapping source and
    target bounds and associate weights with them.

    Note: this assumes that the bounds are monotonic.
    """
    # Calculate the number of cells represented by the bounds.
    m = len(tgt_bounds) - 1
    n = len(src_bounds) - 1

    # Ensure bounds are strictly increasing.
    src_decreasing = src_bounds[0] > src_bounds[1]
    tgt_decreasing = tgt_bounds[0] > tgt_bounds[1]
    if src_decreasing:
        src_bounds = src_bounds[::-1]
    if tgt_decreasing:
        tgt_bounds = tgt_bounds[::-1]

    if circular:
        # For circular coordinates (e.g. longitude) account for source and
        # target bounds which span different ranges (e.g. (-180, 180) vs
        # (0, 360)). We ensure that all possible overlaps between source and
        # target bounds are accounted for by including two copies of the
        # source bounds, shifted appropriately by the modulus.
        adjust = (tgt_bounds.min() - src_bounds.min()) // mod
        src_bounds = src_bounds + (mod * adjust)
        src_bounds = np.append(src_bounds, src_bounds + mod)
        nn = (2 * n) + 1
    else:
        nn = n

    # Before iterating through pairs of overlapping bounds, find an
    # appropriate place to start iteration. Note that this assumes that
    # the bounds are increasing.
    i = max(np.searchsorted(tgt_bounds, src_bounds[0], side="right") - 1, 0)
    j = max(np.searchsorted(src_bounds, tgt_bounds[0], side="right") - 1, 0)

    data = []
    rows = []
    cols = []

    # Iterate through overlapping cells in the source and target bounds.
    # For the sake of calculations, we keep track of the minimum value of
    # the intersection of each cell.
    floor = max(tgt_bounds[i], src_bounds[j])
    while i < m and j < nn:
        # Record the current indices.
        rows.append(i)
        cols.append(j)

        # Determine the next indices and floor.
        if tgt_bounds[i + 1] < src_bounds[j + 1]:
            next_floor = tgt_bounds[i + 1]
            next_i = i + 1
        elif tgt_bounds[i + 1] == src_bounds[j + 1]:
            next_floor = tgt_bounds[i + 1]
            next_i = i + 1
            j += 1
        else:
            next_floor = src_bounds[j + 1]
            next_i = i
            j += 1

        # Calculate and record the weight for the current overlapping cells.
        weight = (next_floor - floor) / (tgt_bounds[i + 1] - tgt_bounds[i])
        data.append(weight)

        # Update indices and floor
        i = next_i
        floor = next_floor

    data = np.array(data)
    rows = np.array(rows)
    cols = np.array(cols)

    if circular:
        # Remove out of bounds points. When the source bounds were duplicated
        # an "out of bounds" cell was introduced between the two copies.
        oob = np.where(cols == n)
        data = np.delete(data, oob)
        rows = np.delete(rows, oob)
        cols = np.delete(cols, oob)

        # Wrap indices. Since we duplicated the source bounds there may be
        # indices which are greater than n which will need to be corrected.
        cols = cols % (n + 1)

    # Correct indices which were flipped due to reversing decreasing bounds.
    if src_decreasing:
        cols = n - cols - 1
    if tgt_decreasing:
        rows = m - rows - 1

    return data, rows, cols


def _combine_xy_weights(x_info, y_info, src_shape, tgt_shape):
    """Second part of weight calculation.

    Combine the weights contributions from both pairs of coordinate
    bounds (i.e. the source/target pairs for the x and y coords).
    Return the result as a sparse array.
    """
    x_src, y_src = src_shape
    x_tgt, y_tgt = tgt_shape
    src_size = x_src * y_src
    tgt_size = x_tgt * y_tgt
    x_weight, x_rows, x_cols = x_info
    y_weight, y_rows, y_cols = y_info

    # Regridding weights will be applied to a flattened (y, x) array.
    # Weights and indices are constructed in a way to account for this.
    # Weights of the combined matrix are constructed by broadcasting
    # the x_weights and y_weights. The resulting array contains every
    # combination of x weight and y weight. Then we flatten this array.
    xy_weight = y_weight[:, np.newaxis] * x_weight[np.newaxis, :]
    xy_weight = xy_weight.flatten()

    # Given the x index and y index associated with a weight, calculate
    # the equivalent index in the flattened (y, x) array.
    xy_rows = (y_rows[:, np.newaxis] * x_tgt) + x_rows[np.newaxis, :]
    xy_rows = xy_rows.flatten()
    xy_cols = (y_cols[:, np.newaxis] * x_src) + x_cols[np.newaxis, :]
    xy_cols = xy_cols.flatten()

    # Create a sparse matrix for efficient weight application.
    combined_weights = csr_array(
        (xy_weight, (xy_rows, xy_cols)), shape=(tgt_size, src_size)
    )
    return combined_weights


def _standard_regrid_no_masks(data, weights, tgt_shape):
    """Regrid unmasked data to an unmasked result.

    Assumes that the first two dimensions are the x-y grid.
    """
    # Reshape data to a form suitable for matrix multiplication.
    extra_shape = data.shape[:-2]
    data = data.reshape(-1, np.prod(data.shape[-2:]))

    # Apply regridding weights.
    # The order of matrix multiplication is chosen to be consistent
    # with existing regridding code.
    result = data @ weights.T

    # Reshape result to a suitable form.
    result = result.reshape(*(extra_shape + tgt_shape))
    return result


def _standard_regrid(data, weights, tgt_shape, mdtol):
    """Regrid data and handle masks.

    Assumes that the first two dimensions are the x-y grid.
    """
    # This is set to keep consistent with legacy behaviour.
    # This is likely to become switchable in the future, see:
    # https://github.com/SciTools/iris/issues/5461
    oob_invalid = True

    data_shape = data.shape
    if ma.is_masked(data):
        unmasked = ~ma.getmaskarray(data)
        # Calculate contribution from unmasked sources to each target point.
        weight_sums = _standard_regrid_no_masks(unmasked, weights, tgt_shape)
    else:
        # If there are no masked points then all contributions will be
        # from unmasked sources, so we can skip this calculation
        weight_sums = np.ones(data_shape[:-2] + tgt_shape)
    mdtol = max(mdtol, 1e-8)
    tgt_mask = weight_sums > 1 - mdtol
    # If out of bounds sources are treated the same as masked sources this
    # will already have been calculated above, so we can skip this calculation.
    if oob_invalid or not ma.is_masked(data):
        # Calculate the proportion of each target cell which is covered by the
        # source. For the sake of efficiency, this is calculated for a 2D slice
        # which is then broadcast.
        inbound_sums = _standard_regrid_no_masks(
            np.ones(data_shape[-2:]), weights, tgt_shape
        )
        if oob_invalid:
            # Legacy behaviour, if the full area of a target cell does not lie
            # in bounds it will be masked.
            oob_mask = inbound_sums > 1 - 1e-8
        else:
            # Note: this code is currently inaccessible. This code exists to lay
            # the groundwork for future work which will make out of bounds
            # behaviour switchable.
            oob_mask = inbound_sums > 1 - mdtol
        # Broadcast the mask to the shape of the full array
        oob_slice = ((np.newaxis,) * len(data.shape[:-2])) + np.s_[:, :]
        tgt_mask = tgt_mask * oob_mask[oob_slice]

    # Calculate normalisations.
    normalisations = tgt_mask.astype(weight_sums.dtype)
    normalisations[tgt_mask] /= weight_sums[tgt_mask]

    # Mask points in the result.
    if ma.isMaskedArray(data):
        # If the source is masked, the result should have a similar mask.
        fill_value = data.fill_value
        normalisations = ma.array(normalisations, mask=~tgt_mask, fill_value=fill_value)
    elif np.any(~tgt_mask):
        normalisations = ma.array(normalisations, mask=~tgt_mask)

    # Use input cube dtype or convert values to the smallest possible float
    # dtype when necessary.
    dtype = np.promote_types(data.dtype, np.float16)

    # Perform regridding on unmasked data.
    result = _standard_regrid_no_masks(ma.filled(data, 0.0), weights, tgt_shape)
    # Apply normalisations and masks to the regridded data.
    result = result * normalisations
    result = result.astype(dtype)
    return result


def _regrid_along_dims(data, x_dim, y_dim, weights, tgt_shape, mdtol):
    """Regrid data, handling masks and dimensions."""
    # Handle scalar coordinates.
    # Note: scalar source coordinates are only handled when their
    # corresponding target coordinate is also scalar.
    num_scalar_dims = 0
    if x_dim is None:
        num_scalar_dims += 1
        data = np.expand_dims(data, -1)
        x_dim = -1
    if y_dim is None:
        num_scalar_dims += 1
        data = np.expand_dims(data, -1)
        y_dim = -1
        if num_scalar_dims == 2:
            y_dim = -2

    # Standard regridding expects the last two dimensions to belong
    # to the y and x coordinate and will output as such.
    # Axes are moved to account for an arbitrary dimension ordering.
    data = np.moveaxis(data, [y_dim, x_dim], [-2, -1])
    result = _standard_regrid(data, weights, tgt_shape, mdtol)
    result = np.moveaxis(result, [-2, -1], [y_dim, x_dim])

    for _ in range(num_scalar_dims):
        result = np.squeeze(result, axis=-1)
    return result
