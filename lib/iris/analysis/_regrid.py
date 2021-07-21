# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

import copy
import functools
import warnings

import numpy as np
import numpy.ma as ma
from scipy.sparse import csc_matrix
from scipy.sparse import diags as sparse_diags

from iris._lazy_data import map_complete_blocks
from iris.analysis._interpolation import (
    EXTRAPOLATION_MODES,
    extend_circular_coord_and_data,
    get_xy_dim_coords,
    snapshot_grid,
)
from iris.analysis._scipy_interpolate import _RegularGridInterpolator
from iris.util import _meshgrid


def _transform_xy_arrays(crs_from, x, y, crs_to):
    """
    Transform 2d points between cartopy coordinate reference systems.

    NOTE: copied private function from iris.analysis.cartography.

    Args:

    * crs_from, crs_to (:class:`cartopy.crs.Projection`):
        The coordinate reference systems.
    * x, y (arrays):
        point locations defined in 'crs_from'.

    Returns:
        x, y :  Arrays of locations defined in 'crs_to'.

    """
    pts = crs_to.transform_points(crs_from, x, y)
    return pts[..., 0], pts[..., 1]


def _regrid_weighted_curvilinear_to_rectilinear__prepare(
    src_cube, weights, grid_cube
):
    """
    First (setup) part of 'regrid_weighted_curvilinear_to_rectilinear'.

    Check inputs and calculate the sparse regrid matrix and related info.
    The 'regrid info' returned can be re-used over many 2d slices.

    """
    if src_cube.aux_factories:
        msg = "All source cube derived coordinates will be ignored."
        warnings.warn(msg)

    # Get the source cube x and y 2D auxiliary coordinates.
    sx, sy = src_cube.coord(axis="x"), src_cube.coord(axis="y")
    # Get the target grid cube x and y dimension coordinates.
    tx, ty = get_xy_dim_coords(grid_cube)

    if sx.units != sy.units:
        msg = (
            "The source cube x ({!r}) and y ({!r}) coordinates must "
            "have the same units."
        )
        raise ValueError(msg.format(sx.name(), sy.name()))

    if src_cube.coord_dims(sx) != src_cube.coord_dims(sy):
        msg = (
            "The source cube x ({!r}) and y ({!r}) coordinates must "
            "map onto the same cube dimensions."
        )
        raise ValueError(msg.format(sx.name(), sy.name()))

    if sx.coord_system != sy.coord_system:
        msg = (
            "The source cube x ({!r}) and y ({!r}) coordinates must "
            "have the same coordinate system."
        )
        raise ValueError(msg.format(sx.name(), sy.name()))

    if sx.coord_system is None:
        msg = (
            "The source X and Y coordinates must have a defined "
            "coordinate system."
        )
        raise ValueError(msg)

    if tx.units != ty.units:
        msg = (
            "The target grid cube x ({!r}) and y ({!r}) coordinates must "
            "have the same units."
        )
        raise ValueError(msg.format(tx.name(), ty.name()))

    if tx.coord_system is None:
        msg = (
            "The target X and Y coordinates must have a defined "
            "coordinate system."
        )
        raise ValueError(msg)

    if tx.coord_system != ty.coord_system:
        msg = (
            "The target grid cube x ({!r}) and y ({!r}) coordinates must "
            "have the same coordinate system."
        )
        raise ValueError(msg.format(tx.name(), ty.name()))

    if weights is None:
        weights = np.ones(sx.shape)
    if weights.shape != sx.shape:
        msg = (
            "Provided weights must have the same shape as the X and Y "
            "coordinates."
        )
        raise ValueError(msg)

    if not tx.has_bounds() or not tx.is_contiguous():
        msg = (
            "The target grid cube x ({!r})coordinate requires "
            "contiguous bounds."
        )
        raise ValueError(msg.format(tx.name()))

    if not ty.has_bounds() or not ty.is_contiguous():
        msg = (
            "The target grid cube y ({!r}) coordinate requires "
            "contiguous bounds."
        )
        raise ValueError(msg.format(ty.name()))

    def _src_align_and_flatten(coord):
        # Return a flattened, unmasked copy of a coordinate's points array that
        # will align with a flattened version of the source cube's data.
        #
        # PP-TODO: Should work with any cube dimensions for X and Y coords.
        #  Probably needs fixing anyway?
        #
        points = coord.points
        if src_cube.coord_dims(coord) == (1, 0):
            points = points.T
        if points.shape != src_cube.shape:
            msg = (
                "The shape of the points array of {!r} is not compatible "
                "with the shape of {!r}."
            )
            raise ValueError(msg.format(coord.name(), src_cube.name()))
        return np.asarray(points.flatten())

    # Align and flatten the coordinate points of the source space.
    sx_points = _src_align_and_flatten(sx)
    sy_points = _src_align_and_flatten(sy)

    # Transform source X and Y points into the target coord-system, if needed.
    if sx.coord_system != tx.coord_system:
        src_crs = sx.coord_system.as_cartopy_projection()
        tgt_crs = tx.coord_system.as_cartopy_projection()
        sx_points, sy_points = _transform_xy_arrays(
            src_crs, sx_points, sy_points, tgt_crs
        )
    #
    # TODO: how does this work with scaled units ??
    #  e.g. if crs is latlon, units could be degrees OR radians ?
    #

    # Wrap modular values (e.g. longitudes) if required.
    modulus = sx.units.modulus
    if modulus is not None:
        # Match the source cube x coordinate range to the target grid
        # cube x coordinate range.
        min_sx, min_tx = np.min(sx.points), np.min(tx.points)
        if min_sx < 0 and min_tx >= 0:
            indices = np.where(sx_points < 0)
            # Ensure += doesn't raise a TypeError
            if not np.can_cast(modulus, sx_points.dtype):
                sx_points = sx_points.astype(type(modulus), casting="safe")
            sx_points[indices] += modulus
        elif min_sx >= 0 and min_tx < 0:
            indices = np.where(sx_points > (modulus / 2))
            # Ensure -= doesn't raise a TypeError
            if not np.can_cast(modulus, sx_points.dtype):
                sx_points = sx_points.astype(type(modulus), casting="safe")
            sx_points[indices] -= modulus

    # Create target grid cube x and y cell boundaries.
    tx_depth, ty_depth = tx.points.size, ty.points.size
    (tx_dim,) = grid_cube.coord_dims(tx)
    (ty_dim,) = grid_cube.coord_dims(ty)

    tx_cells = np.concatenate((tx.bounds[:, 0], tx.bounds[-1, 1].reshape(1)))
    ty_cells = np.concatenate((ty.bounds[:, 0], ty.bounds[-1, 1].reshape(1)))

    # Determine the target grid cube x and y cells that bound
    # the source cube x and y points.

    def _regrid_indices(cells, depth, points):
        # Calculate the minimum difference in cell extent.
        extent = np.min(np.diff(cells))
        if extent == 0:
            # Detected an dimension coordinate with an invalid
            # zero length cell extent.
            msg = (
                "The target grid cube {} ({!r}) coordinate contains "
                "a zero length cell extent."
            )
            axis, name = "x", tx.name()
            if points is sy_points:
                axis, name = "y", ty.name()
            raise ValueError(msg.format(axis, name))
        elif extent > 0:
            # The cells of the dimension coordinate are in ascending order.
            indices = np.searchsorted(cells, points, side="right") - 1
        else:
            # The cells of the dimension coordinate are in descending order.
            # np.searchsorted() requires ascending order, so we require to
            # account for this restriction.
            cells = cells[::-1]
            right = np.searchsorted(cells, points, side="right")
            left = np.searchsorted(cells, points, side="left")
            indices = depth - right
            # Only those points that exactly match the left-hand cell bound
            # will differ between 'left' and 'right'. Thus their appropriate
            # target cell location requires to be recalculated to give the
            # correct descending [upper, lower) interval cell, source to target
            # regrid behaviour.
            delta = np.where(left != right)[0]
            if delta.size:
                indices[delta] = depth - left[delta]
        return indices

    x_indices = _regrid_indices(tx_cells, tx_depth, sx_points)
    y_indices = _regrid_indices(ty_cells, ty_depth, sy_points)

    # Now construct a sparse M x N matix, where M is the flattened target
    # space, and N is the flattened source space. The sparse matrix will then
    # be populated with those source cube points that contribute to a specific
    # target cube cell.

    # Determine the valid indices and their offsets in M x N space.
    # Calculate the valid M offsets.
    cols = np.where(
        (y_indices >= 0)
        & (y_indices < ty_depth)
        & (x_indices >= 0)
        & (x_indices < tx_depth)
    )[0]

    # Reduce the indices to only those that are valid.
    x_indices = x_indices[cols]
    y_indices = y_indices[cols]

    # Calculate the valid N offsets.
    if ty_dim < tx_dim:
        rows = y_indices * tx.points.size + x_indices
    else:
        rows = x_indices * ty.points.size + y_indices

    # Calculate the associated valid weights.
    weights_flat = weights.flatten()
    data = weights_flat[cols]

    # Build our sparse M x N matrix of weights.
    sparse_matrix = csc_matrix(
        (data, (rows, cols)), shape=(grid_cube.data.size, src_cube.data.size)
    )

    # Performing a sparse sum to collapse the matrix to (M, 1).
    sum_weights = sparse_matrix.sum(axis=1).getA()

    # Determine the rows (flattened target indices) that have a
    # contribution from one or more source points.
    rows = np.nonzero(sum_weights)

    # NOTE: when source points are masked, this 'sum_weights' is possibly
    # incorrect and needs re-calculating.  Likewise 'rows' may cover target
    # cells which happen to get no data.  This is dealt with by adjusting as
    # required in the '__perform' function, below.

    regrid_info = (sparse_matrix, sum_weights, rows, grid_cube)
    return regrid_info


def _regrid_weighted_curvilinear_to_rectilinear__perform(
    src_cube, regrid_info
):
    """
    Second (regrid) part of 'regrid_weighted_curvilinear_to_rectilinear'.

    Perform the prepared regrid calculation on a single 2d cube.

    """
    from iris.cube import Cube

    sparse_matrix, sum_weights, rows, grid_cube = regrid_info

    # Calculate the numerator of the weighted mean (M, 1).
    is_masked = ma.isMaskedArray(src_cube.data)
    if not is_masked:
        data = src_cube.data
    else:
        # Use raw data array
        data = src_cube.data.data
        # Check if there are any masked source points to take account of.
        is_masked = np.ma.is_masked(src_cube.data)
        if is_masked:
            # Zero any masked source points so they add nothing in output sums.
            mask = src_cube.data.mask
            data[mask] = 0.0
            # Calculate a new 'sum_weights' to allow for missing source points.
            # N.B. it is more efficient to use the original once-calculated
            # sparse matrix, but in this case we can't.
            # Hopefully, this post-multiplying by the validities is less costly
            # than repeating the whole sparse calculation.
            valid_src_cells = ~mask.flat[:]
            src_cell_validity_factors = sparse_diags(
                np.array(valid_src_cells, dtype=int), 0
            )
            valid_weights = sparse_matrix * src_cell_validity_factors
            sum_weights = valid_weights.sum(axis=1).getA()
            # Work out where output cells are missing all contributions.
            # This allows for where 'rows' contains output cells that have no
            # data because of missing input points.
            zero_sums = sum_weights == 0.0
            # Make sure we can still divide by sum_weights[rows].
            sum_weights[zero_sums] = 1.0

    # Calculate sum in each target cell, over contributions from each source
    # cell.
    numerator = sparse_matrix * data.reshape(-1, 1)

    # Create a template for the weighted mean result.
    weighted_mean = ma.masked_all(numerator.shape, dtype=numerator.dtype)

    # Calculate final results in all relevant places.
    weighted_mean[rows] = numerator[rows] / sum_weights[rows]
    if is_masked:
        # Ensure masked points where relevant source cells were all missing.
        if np.any(zero_sums):
            # Make masked if it wasn't.
            weighted_mean = np.ma.asarray(weighted_mean)
            # Mask where contributing sums were zero.
            weighted_mean[zero_sums] = np.ma.masked

    # Construct the final regridded weighted mean cube.
    tx = grid_cube.coord(axis="x", dim_coords=True)
    ty = grid_cube.coord(axis="y", dim_coords=True)
    (tx_dim,) = grid_cube.coord_dims(tx)
    (ty_dim,) = grid_cube.coord_dims(ty)
    dim_coords_and_dims = list(zip((ty.copy(), tx.copy()), (ty_dim, tx_dim)))
    cube = Cube(
        weighted_mean.reshape(grid_cube.shape),
        dim_coords_and_dims=dim_coords_and_dims,
    )
    cube.metadata = copy.deepcopy(src_cube.metadata)

    for coord in src_cube.coords(dimensions=()):
        cube.add_aux_coord(coord.copy())

    return cube


class CurvilinearRegridder:
    """
    This class provides support for performing point-in-cell regridding
    between a curvilinear source grid and a rectilinear target grid.

    """

    def __init__(self, src_grid_cube, target_grid_cube, weights=None):
        """
        Create a regridder for conversions between the source
        and target grids.

        Args:

        * src_grid_cube:
            The :class:`~iris.cube.Cube` providing the source grid.
        * tgt_grid_cube:
            The :class:`~iris.cube.Cube` providing the target grid.

        Optional Args:

        * weights:
            A :class:`numpy.ndarray` instance that defines the weights
            for the grid cells of the source grid. Must have the same shape
            as the data of the source grid.
            If unspecified, equal weighting is assumed.

        """
        from iris.cube import Cube

        # Validity checks.
        if not isinstance(src_grid_cube, Cube):
            raise TypeError("'src_grid_cube' must be a Cube")
        if not isinstance(target_grid_cube, Cube):
            raise TypeError("'target_grid_cube' must be a Cube")
        # Snapshot the state of the cubes to ensure that the regridder
        # is impervious to external changes to the original source cubes.
        self._src_cube = src_grid_cube.copy()
        self._target_cube = target_grid_cube.copy()
        self.weights = weights
        self._regrid_info = None

    @staticmethod
    def _get_horizontal_coord(cube, axis):
        """
        Gets the horizontal coordinate on the supplied cube along the
        specified axis.

        Args:

        * cube:
            An instance of :class:`iris.cube.Cube`.
        * axis:
            Locate coordinates on `cube` along this axis.

        Returns:
            The horizontal coordinate on the specified axis of the supplied
            cube.

        """
        coords = cube.coords(axis=axis, dim_coords=False)
        if len(coords) != 1:
            raise ValueError(
                "Cube {!r} must contain a single 1D {} "
                "coordinate.".format(cube.name(), axis)
            )
        return coords[0]

    def __call__(self, src):
        """
        Regrid the supplied :class:`~iris.cube.Cube` on to the target grid of
        this :class:`_CurvilinearRegridder`.

        The given cube must be defined with the same grid as the source
        grid used to create this :class:`_CurvilinearRegridder`.

        If the source cube has lazy data, it will be realized before
        regridding and the returned cube will also have realized data.

        Args:

        * src:
            A :class:`~iris.cube.Cube` to be regridded.

        Returns:
            A cube defined with the horizontal dimensions of the target
            and the other dimensions from this cube. The data values of
            this cube will be converted to values on the new grid using
            point-in-cell regridding.

        """
        from iris.cube import Cube, CubeList

        # Validity checks.
        if not isinstance(src, Cube):
            raise TypeError("'src' must be a Cube")

        gx = self._get_horizontal_coord(self._src_cube, "x")
        gy = self._get_horizontal_coord(self._src_cube, "y")
        src_grid = (gx.copy(), gy.copy())
        sx = self._get_horizontal_coord(src, "x")
        sy = self._get_horizontal_coord(src, "y")
        if (sx, sy) != src_grid:
            raise ValueError(
                "The given cube is not defined on the same "
                "source grid as this regridder."
            )

        # Call the regridder function.
        # This includes repeating over any non-XY dimensions, because the
        # underlying routine does not support this.
        # FOR NOW: we will use cube.slices and merge to achieve this,
        # though that is not a terribly efficient method ...
        # TODO: create a template result cube and paste data slices into it,
        # which would be more efficient.
        result_slices = CubeList([])
        for slice_cube in src.slices(sx):
            if self._regrid_info is None:
                # Calculate the basic regrid info just once.
                self._regrid_info = (
                    _regrid_weighted_curvilinear_to_rectilinear__prepare(
                        slice_cube, self.weights, self._target_cube
                    )
                )
            slice_result = (
                _regrid_weighted_curvilinear_to_rectilinear__perform(
                    slice_cube, self._regrid_info
                )
            )
            result_slices.append(slice_result)
        result = result_slices.merge_cube()
        return result


class RectilinearRegridder:
    """
    This class provides support for performing nearest-neighbour or
    linear regridding between source and target grids.

    """

    def __init__(
        self, src_grid_cube, tgt_grid_cube, method, extrapolation_mode
    ):
        """
        Create a regridder for conversions between the source
        and target grids.

        Args:

        * src_grid_cube:
            The :class:`~iris.cube.Cube` providing the source grid.
        * tgt_grid_cube:
            The :class:`~iris.cube.Cube` providing the target grid.
        * method:
            Either 'linear' or 'nearest'.
        * extrapolation_mode:
            Must be one of the following strings:

              * 'extrapolate' - The extrapolation points will be
                calculated by extending the gradient of the closest two
                points.
              * 'nan' - The extrapolation points will be be set to NaN.
              * 'error' - An exception will be raised, notifying an
                attempt to extrapolate.
              * 'mask' - The extrapolation points will always be masked, even
                if the source data is not a MaskedArray.
              * 'nanmask' - If the source data is a MaskedArray the
                extrapolation points will be masked. Otherwise they will be
                set to NaN.

        """
        from iris.cube import Cube

        # Validity checks.
        if not isinstance(src_grid_cube, Cube):
            raise TypeError("'src_grid_cube' must be a Cube")
        if not isinstance(tgt_grid_cube, Cube):
            raise TypeError("'tgt_grid_cube' must be a Cube")
        # Snapshot the state of the cubes to ensure that the regridder
        # is impervious to external changes to the original source cubes.
        self._src_grid = snapshot_grid(src_grid_cube)
        self._tgt_grid = snapshot_grid(tgt_grid_cube)
        # Check the target grid units.
        for coord in self._tgt_grid:
            self._check_units(coord)
        # Whether to use linear or nearest-neighbour interpolation.
        if method not in ("linear", "nearest"):
            msg = "Regridding method {!r} not supported.".format(method)
            raise ValueError(msg)
        self._method = method
        # The extrapolation mode.
        if extrapolation_mode not in EXTRAPOLATION_MODES:
            msg = "Invalid extrapolation mode {!r}"
            raise ValueError(msg.format(extrapolation_mode))
        self._extrapolation_mode = extrapolation_mode

    @property
    def method(self):
        return self._method

    @property
    def extrapolation_mode(self):
        return self._extrapolation_mode

    @staticmethod
    def _sample_grid(src_coord_system, grid_x_coord, grid_y_coord):
        """
        Convert the rectilinear grid coordinates to a curvilinear grid in
        the source coordinate system.

        The `grid_x_coord` and `grid_y_coord` must share a common coordinate
        system.

        Args:

        * src_coord_system:
            The :class:`iris.coord_system.CoordSystem` for the grid of the
            source Cube.
        * grid_x_coord:
            The :class:`iris.coords.DimCoord` for the X coordinate.
        * grid_y_coord:
            The :class:`iris.coords.DimCoord` for the Y coordinate.

        Returns:
            A tuple of the X and Y coordinate values as 2-dimensional
            arrays.

        """
        grid_x, grid_y = _meshgrid(grid_x_coord.points, grid_y_coord.points)
        # Skip the CRS transform if we can to avoid precision problems.
        if src_coord_system == grid_x_coord.coord_system:
            sample_grid_x = grid_x
            sample_grid_y = grid_y
        else:
            src_crs = src_coord_system.as_cartopy_crs()
            grid_crs = grid_x_coord.coord_system.as_cartopy_crs()
            sample_xyz = src_crs.transform_points(grid_crs, grid_x, grid_y)
            sample_grid_x = sample_xyz[..., 0]
            sample_grid_y = sample_xyz[..., 1]
        return sample_grid_x, sample_grid_y

    @staticmethod
    def _regrid(
        src_data,
        x_dim,
        y_dim,
        src_x_coord,
        src_y_coord,
        sample_grid_x,
        sample_grid_y,
        method="linear",
        extrapolation_mode="nanmask",
    ):
        """
        Regrid the given data from the src grid to the sample grid.

        The result will be a MaskedArray if either/both of:
         - the source array is a MaskedArray,
         - the extrapolation_mode is 'mask' and the result requires
           extrapolation.

        If the result is a MaskedArray the mask for each element will be set
        if either/both of:
         - there is a non-zero contribution from masked items in the input data
         - the element requires extrapolation and the extrapolation_mode
           dictates a masked value.

        Args:

        * src_data:
            An N-dimensional NumPy array or MaskedArray.
        * x_dim:
            The X dimension within `src_data`.
        * y_dim:
            The Y dimension within `src_data`.
        * src_x_coord:
            The X :class:`iris.coords.DimCoord`.
        * src_y_coord:
            The Y :class:`iris.coords.DimCoord`.
        * sample_grid_x:
            A 2-dimensional array of sample X values.
        * sample_grid_y:
            A 2-dimensional array of sample Y values.

        Kwargs:

        * method:
            Either 'linear' or 'nearest'. The default method is 'linear'.
        * extrapolation_mode:
            Must be one of the following strings:

              * 'linear' - The extrapolation points will be calculated by
                extending the gradient of the closest two points.
              * 'nan' - The extrapolation points will be be set to NaN.
              * 'error' - A ValueError exception will be raised, notifying an
                attempt to extrapolate.
              * 'mask' - The extrapolation points will always be masked, even
                if the source data is not a MaskedArray.
              * 'nanmask' - If the source data is a MaskedArray the
                extrapolation points will be masked. Otherwise they will be
                set to NaN.

            The default mode of extrapolation is 'nanmask'.

        Returns:
            The regridded data as an N-dimensional NumPy array. The lengths
            of the X and Y dimensions will now match those of the sample
            grid.

        """
        #
        # XXX: At the moment requires to be a static method as used by
        # experimental regrid_area_weighted_rectilinear_src_and_grid
        #
        if sample_grid_x.shape != sample_grid_y.shape:
            raise ValueError("Inconsistent sample grid shapes.")
        if sample_grid_x.ndim != 2:
            raise ValueError("Sample grid must be 2-dimensional.")

        # Prepare the result data array
        shape = list(src_data.shape)
        assert shape[x_dim] == src_x_coord.shape[0]
        assert shape[y_dim] == src_y_coord.shape[0]

        shape[y_dim] = sample_grid_x.shape[0]
        shape[x_dim] = sample_grid_x.shape[1]

        dtype = src_data.dtype
        if method == "linear":
            # If we're given integer values, convert them to the smallest
            # possible float dtype that can accurately preserve the values.
            if dtype.kind == "i":
                dtype = np.promote_types(dtype, np.float16)

        if ma.isMaskedArray(src_data):
            data = ma.empty(shape, dtype=dtype)
            data.mask = np.zeros(data.shape, dtype=np.bool_)
        else:
            data = np.empty(shape, dtype=dtype)

        # The interpolation class requires monotonically increasing
        # coordinates, so flip the coordinate(s) and data if they aren't.
        reverse_x = (
            src_x_coord.points[0] > src_x_coord.points[1]
            if src_x_coord.points.size > 1
            else False
        )
        reverse_y = src_y_coord.points[0] > src_y_coord.points[1]
        flip_index = [slice(None)] * src_data.ndim
        if reverse_x:
            src_x_coord = src_x_coord[::-1]
            flip_index[x_dim] = slice(None, None, -1)
        if reverse_y:
            src_y_coord = src_y_coord[::-1]
            flip_index[y_dim] = slice(None, None, -1)
        src_data = src_data[tuple(flip_index)]

        if src_x_coord.circular:
            x_points, src_data = extend_circular_coord_and_data(
                src_x_coord, src_data, x_dim
            )
        else:
            x_points = src_x_coord.points

        # Slice out the first full 2D piece of data for construction of the
        # interpolator.
        index = [0] * src_data.ndim
        index[x_dim] = index[y_dim] = slice(None)
        initial_data = src_data[tuple(index)]
        if y_dim < x_dim:
            initial_data = initial_data.T

        # Construct the interpolator, we will fill in any values out of bounds
        # manually.
        interpolator = _RegularGridInterpolator(
            [x_points, src_y_coord.points],
            initial_data,
            method=method,
            bounds_error=False,
            fill_value=None,
        )
        # The constructor of the _RegularGridInterpolator class does
        # some unnecessary checks on these values, so we set them
        # afterwards instead. Sneaky. ;-)
        try:
            mode = EXTRAPOLATION_MODES[extrapolation_mode]
        except KeyError:
            raise ValueError("Invalid extrapolation mode.")
        interpolator.bounds_error = mode.bounds_error
        interpolator.fill_value = mode.fill_value

        # Construct the target coordinate points array, suitable for passing to
        # the interpolator multiple times.
        interp_coords = [
            sample_grid_x.astype(np.float64)[..., np.newaxis],
            sample_grid_y.astype(np.float64)[..., np.newaxis],
        ]

        # Map all the requested values into the range of the source
        # data (centred over the centre of the source data to allow
        # extrapolation where required).
        min_x, max_x = x_points.min(), x_points.max()
        if src_x_coord.units.modulus:
            modulus = src_x_coord.units.modulus
            offset = (max_x + min_x - modulus) * 0.5
            interp_coords[0] -= offset
            interp_coords[0] = (interp_coords[0] % modulus) + offset

        interp_coords = np.dstack(interp_coords)

        weights = interpolator.compute_interp_weights(interp_coords)

        def interpolate(data):
            # Update the interpolator for this data slice.
            data = data.astype(interpolator.values.dtype)
            if y_dim < x_dim:
                data = data.T
            interpolator.values = data
            data = interpolator.interp_using_pre_computed_weights(weights)
            if y_dim > x_dim:
                data = data.T
            return data

        # Build up a shape suitable for passing to ndindex, inside the loop we
        # will insert slice(None) on the data indices.
        iter_shape = list(shape)
        iter_shape[x_dim] = iter_shape[y_dim] = 1

        # Iterate through each 2d slice of the data, updating the interpolator
        # with the new data as we go.
        for index in np.ndindex(tuple(iter_shape)):
            index = list(index)
            index[x_dim] = index[y_dim] = slice(None)

            src_subset = src_data[tuple(index)]
            interpolator.fill_value = mode.fill_value
            data[tuple(index)] = interpolate(src_subset)

            if ma.isMaskedArray(data) or mode.force_mask:
                # NB. np.ma.getmaskarray returns an array of `False` if
                # `src_subset` is not a masked array.
                src_mask = np.ma.getmaskarray(src_subset)
                interpolator.fill_value = mode.mask_fill_value
                mask_fraction = interpolate(src_mask)
                new_mask = mask_fraction > 0

                if np.ma.isMaskedArray(data):
                    data.mask[tuple(index)] = new_mask
                elif np.any(new_mask):
                    # Set mask=False to ensure we have an expanded mask array.
                    data = np.ma.MaskedArray(data, mask=False)
                    data.mask[tuple(index)] = new_mask

        return data

    @staticmethod
    def _create_cube(
        data,
        src,
        x_dim,
        y_dim,
        src_x_coord,
        src_y_coord,
        grid_x_coord,
        grid_y_coord,
        sample_grid_x,
        sample_grid_y,
        regrid_callback,
    ):
        """
        Return a new Cube for the result of regridding the source Cube onto
        the new grid.

        All the metadata and coordinates of the result Cube are copied from
        the source Cube, with two exceptions:
            - Grid dimension coordinates are copied from the grid Cube.
            - Auxiliary coordinates which span the grid dimensions are
              ignored, except where they provide a reference surface for an
              :class:`iris.aux_factory.AuxCoordFactory`.

        Args:

        * data:
            The regridded data as an N-dimensional NumPy array.
        * src:
            The source Cube.
        * x_dim:
            The X dimension within the source Cube.
        * y_dim:
            The Y dimension within the source Cube.
        * src_x_coord:
            The X :class:`iris.coords.DimCoord`.
        * src_y_coord:
            The Y :class:`iris.coords.DimCoord`.
        * grid_x_coord:
            The :class:`iris.coords.DimCoord` for the new grid's X
            coordinate.
        * grid_y_coord:
            The :class:`iris.coords.DimCoord` for the new grid's Y
            coordinate.
        * sample_grid_x:
            A 2-dimensional array of sample X values.
        * sample_grid_y:
            A 2-dimensional array of sample Y values.
        * regrid_callback:
            The routine that will be used to calculate the interpolated
            values of any reference surfaces.

        Returns:
            The new, regridded Cube.

        """
        from iris.cube import Cube

        #
        # XXX: At the moment requires to be a static method as used by
        # experimental regrid_area_weighted_rectilinear_src_and_grid
        #
        # Create a result cube with the appropriate metadata
        result = Cube(data)
        result.metadata = copy.deepcopy(src.metadata)

        # Copy across all the coordinates which don't span the grid.
        # Record a mapping from old coordinate IDs to new coordinates,
        # for subsequent use in creating updated aux_factories.
        coord_mapping = {}

        def copy_coords(src_coords, add_method):
            for coord in src_coords:
                dims = src.coord_dims(coord)
                if coord == src_x_coord:
                    coord = grid_x_coord
                elif coord == src_y_coord:
                    coord = grid_y_coord
                elif x_dim in dims or y_dim in dims:
                    continue
                result_coord = coord.copy()
                add_method(result_coord, dims)
                coord_mapping[id(coord)] = result_coord

        copy_coords(src.dim_coords, result.add_dim_coord)
        copy_coords(src.aux_coords, result.add_aux_coord)

        def regrid_reference_surface(
            src_surface_coord,
            surface_dims,
            x_dim,
            y_dim,
            src_x_coord,
            src_y_coord,
            sample_grid_x,
            sample_grid_y,
            regrid_callback,
        ):
            # Determine which of the reference surface's dimensions span the X
            # and Y dimensions of the source cube.
            surface_x_dim = surface_dims.index(x_dim)
            surface_y_dim = surface_dims.index(y_dim)
            surface = regrid_callback(
                src_surface_coord.points,
                surface_x_dim,
                surface_y_dim,
                src_x_coord,
                src_y_coord,
                sample_grid_x,
                sample_grid_y,
            )
            surface_coord = src_surface_coord.copy(surface)
            return surface_coord

        # Copy across any AuxFactory instances, and regrid their reference
        # surfaces where required.
        for factory in src.aux_factories:
            for coord in factory.dependencies.values():
                if coord is None:
                    continue
                dims = src.coord_dims(coord)
                if x_dim in dims and y_dim in dims:
                    result_coord = regrid_reference_surface(
                        coord,
                        dims,
                        x_dim,
                        y_dim,
                        src_x_coord,
                        src_y_coord,
                        sample_grid_x,
                        sample_grid_y,
                        regrid_callback,
                    )
                    result.add_aux_coord(result_coord, dims)
                    coord_mapping[id(coord)] = result_coord
            try:
                result.add_aux_factory(factory.updated(coord_mapping))
            except KeyError:
                msg = (
                    "Cannot update aux_factory {!r} because of dropped"
                    " coordinates.".format(factory.name())
                )
                warnings.warn(msg)
        return result

    def _check_units(self, coord):
        from iris.coord_systems import GeogCS, RotatedGeogCS

        if coord.coord_system is None:
            # No restriction on units.
            pass
        elif isinstance(
            coord.coord_system,
            (GeogCS, RotatedGeogCS),
        ):
            # Units for lat-lon or rotated pole must be 'degrees'. Note
            # that 'degrees_east' etc. are equal to 'degrees'.
            if coord.units != "degrees":
                msg = (
                    "Unsupported units for coordinate system. "
                    "Expected 'degrees' got {!r}.".format(coord.units)
                )
                raise ValueError(msg)
        else:
            # Units for other coord systems must be equal to metres.
            if coord.units != "m":
                msg = (
                    "Unsupported units for coordinate system. "
                    "Expected 'metres' got {!r}.".format(coord.units)
                )
                raise ValueError(msg)

    def __call__(self, src):
        """
        Regrid this :class:`~iris.cube.Cube` on to the target grid of
        this :class:`RectilinearRegridder`.

        The given cube must be defined with the same grid as the source
        grid used to create this :class:`RectilinearRegridder`.

        If the source cube has lazy data, the returned cube will also
        have lazy data.

        Args:

        * src:
            A :class:`~iris.cube.Cube` to be regridded.

        Returns:
            A cube defined with the horizontal dimensions of the target
            and the other dimensions from this cube. The data values of
            this cube will be converted to values on the new grid using
            either nearest-neighbour or linear interpolation.

        .. note::

            If the source cube has lazy data,
            `chunks <https://docs.dask.org/en/latest/array-chunks.html>`__
            in the horizontal dimensions will be combined before regridding.

        """
        from iris.cube import Cube

        # Validity checks.
        if not isinstance(src, Cube):
            raise TypeError("'src' must be a Cube")
        if get_xy_dim_coords(src) != self._src_grid:
            raise ValueError(
                "The given cube is not defined on the same "
                "source grid as this regridder."
            )

        src_x_coord, src_y_coord = get_xy_dim_coords(src)
        grid_x_coord, grid_y_coord = self._tgt_grid
        src_cs = src_x_coord.coord_system
        grid_cs = grid_x_coord.coord_system

        if src_cs is None and grid_cs is None:
            if not (
                src_x_coord.is_compatible(grid_x_coord)
                and src_y_coord.is_compatible(grid_y_coord)
            ):
                raise ValueError(
                    "The rectilinear grid coordinates of the "
                    "given cube and target grid have no "
                    "coordinate system but they do not have "
                    "matching coordinate metadata."
                )
        elif src_cs is None or grid_cs is None:
            raise ValueError(
                "The rectilinear grid coordinates of the given "
                "cube and target grid must either both have "
                "coordinate systems or both have no coordinate "
                "system but with matching coordinate metadata."
            )

        # Check the source grid units.
        for coord in (src_x_coord, src_y_coord):
            self._check_units(coord)

        # Convert the grid to a 2D sample grid in the src CRS.
        sample_grid = self._sample_grid(src_cs, grid_x_coord, grid_y_coord)
        sample_grid_x, sample_grid_y = sample_grid

        # Compute the interpolated data values.
        x_dim = src.coord_dims(src_x_coord)[0]
        y_dim = src.coord_dims(src_y_coord)[0]

        # Define regrid function
        regrid = functools.partial(
            self._regrid,
            x_dim=x_dim,
            y_dim=y_dim,
            src_x_coord=src_x_coord,
            src_y_coord=src_y_coord,
            sample_grid_x=sample_grid_x,
            sample_grid_y=sample_grid_y,
            method=self._method,
            extrapolation_mode=self._extrapolation_mode,
        )

        data = map_complete_blocks(
            src, regrid, (y_dim, x_dim), sample_grid_x.shape
        )

        # Wrap up the data as a Cube.
        regrid_callback = functools.partial(
            self._regrid, method=self._method, extrapolation_mode="nan"
        )
        result = self._create_cube(
            data,
            src,
            x_dim,
            y_dim,
            src_x_coord,
            src_y_coord,
            grid_x_coord,
            grid_y_coord,
            sample_grid_x,
            sample_grid_y,
            regrid_callback,
        )
        return result
