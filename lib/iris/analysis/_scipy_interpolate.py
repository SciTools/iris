import itertools

import numpy as np
from scipy.sparse import csr_matrix

# ============================================================================
# |                        Copyright SciPy                                   |
# | Code from this point unto the termination banner is copyright SciPy.     |
# | License details can be found at scipy.org/scipylib/license.html          |
# ============================================================================

# Source: https://github.com/scipy/scipy/blob/b94a5d5ccc08dddbc88453477ff2625\
# 9aeaafb32/scipy/interpolate/interpnd.pyx#L167


def _ndim_coords_from_arrays(points, ndim=None):
    """
    Convert a tuple of coordinate arrays to a (..., ndim)-shaped array.

    """
    if isinstance(points, tuple) and len(points) == 1:
        # handle argument tuple
        points = points[0]
    if isinstance(points, tuple):
        p = np.broadcast_arrays(*points)
        for j in range(1, len(p)):
            if p[j].shape != p[0].shape:
                raise ValueError(
                    "coordinate arrays do not have the same shape"
                )
        points = np.empty(p[0].shape + (len(points),), dtype=float)
        for j, item in enumerate(p):
            points[..., j] = item
    else:
        points = np.asanyarray(points)
        # XXX Feed back to scipy.
        if points.ndim <= 1:
            if ndim is None:
                points = points.reshape(-1, 1)
            else:
                points = points.reshape(-1, ndim)
    return points


# source: https://github.com/scipy/scipy/blob/b94a5d5ccc08dddbc88453477ff2625\
# 9aeaafb32/scipy/interpolate/interpolate.py#L1400
class _RegularGridInterpolator:

    """
    Interpolation on a regular grid in arbitrary dimensions

    The data must be defined on a regular grid; the grid spacing however may be
    uneven.  Linear and nearest-neighbour interpolation are supported. After
    setting up the interpolator object, the interpolation method (*linear* or
    *nearest*) may be chosen at each evaluation.

    .. versionadded:: 0.14

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions.

    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.

    method : str
        The method of interpolation to perform. Supported are "linear" and
        "nearest". This parameter will become the default for the object's
        ``__call__`` method.

    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.

    fill_value : number, optional
        If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated.

    Methods
    -------
    __call__

    Notes
    -----
    Contrary to LinearNDInterpolator and NearestNDInterpolator, this class
    avoids expensive triangulation of the input data by taking advantage of the
    regular grid structure.

    """

    # this class is based on code originally programmed by Johannes Buchner,
    # see https://github.com/JohannesBuchner/regulargrid

    def __init__(
        self,
        points,
        values,
        method="linear",
        bounds_error=True,
        fill_value=np.nan,
    ):
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)
        self.method = method
        self.bounds_error = bounds_error

        if not hasattr(values, "ndim"):
            # allow reasonable duck-typed values
            values = np.asarray(values)

        if len(points) > values.ndim:
            raise ValueError(
                "There are %d point arrays, but values has %d "
                "dimensions" % (len(points), values.ndim)
            )

        if hasattr(values, "dtype") and hasattr(values, "astype"):
            if not np.issubdtype(values.dtype, np.inexact):
                values = values.astype(float)

        self.fill_value = fill_value
        if fill_value is not None:
            if hasattr(values, "dtype") and not np.can_cast(
                fill_value, values.dtype
            ):
                raise ValueError(
                    "fill_value must be either 'None' or "
                    "of a type compatible with values"
                )

        for i, p in enumerate(points):
            if not np.all(np.diff(p) > 0.0):
                raise ValueError(
                    "The points in dimension %d must be strictly "
                    "ascending" % i
                )
            if not np.asarray(p).ndim == 1:
                raise ValueError(
                    "The points in dimension %d must be " "1-dimensional" % i
                )
            if not values.shape[i] == len(p):
                raise ValueError(
                    "There are %d points and %d values in "
                    "dimension %d" % (len(p), values.shape[i], i)
                )
        self.grid = tuple([np.asarray(p) for p in points])
        self.values = values

    def __call__(self, xi, method=None):
        """
        Interpolation at coordinates

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at.

        method : str
            The method of interpolation to perform. Supported are "linear" and
            "nearest".

        """
        # Note: No functionality should live in this method. It should all be
        # decomposed into the two interfaces (compute weights + use weights).
        weights = self.compute_interp_weights(xi, method)
        return self.interp_using_pre_computed_weights(weights)

    def compute_interp_weights(self, xi, method=None):
        """
        Prepare the interpolator for interpolation to the given sample points.

        .. note::
            This interface provides the ability to reuse weights on multiple
            data sources, such as in the case of regridding. For normal
            interpolation, simply call the interpolator with the sample points.

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at.

        Returns
        -------
        A tuple of the items necessary for passing to
        :meth:`interp_using_pre_computed_weights`. The contents of this return
        value are not guaranteed to be consistent across Iris versions, and
        should only be used for passing to
        :meth:`interp_using_pre_computed_weights`.

        Example
        -------
        >>> coords = np.array([[[50.7, -3.5],
                                [50.6, -3.5]],
                               [[50.7, -3.1],
                                [50.6, -3.1]]])
        >>> compute_interp_weights(coords)

        """
        ndim = len(self.grid)
        xi = _ndim_coords_from_arrays(xi, ndim=ndim)
        if xi.shape[-1] != ndim:
            raise ValueError(
                "The requested sample points xi have dimension "
                "%d, but this RegularGridInterpolator has "
                "dimension %d" % (xi.shape[1], ndim)
            )

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if not np.logical_and(
                    np.all(self.grid[i][0] <= p), np.all(p <= self.grid[i][-1])
                ):
                    raise ValueError(
                        "One of the requested xi is out of "
                        "bounds in dimension %d" % i
                    )

        method = self.method if method is None else method
        prepared = (xi_shape, method) + self._find_indices(xi.T)

        if method == "linear":

            xi_shape, method, indices, norm_distances, out_of_bounds = prepared

            # Allocate arrays for describing the sparse matrix.
            n_src_values_per_result_value = 2**ndim
            n_result_values = len(indices[0])
            n_non_zero = n_result_values * n_src_values_per_result_value
            weights = np.ones(n_non_zero, dtype=norm_distances[0].dtype)
            col_indices = np.empty(n_non_zero)
            row_ptrs = np.arange(
                0,
                n_non_zero + n_src_values_per_result_value,
                n_src_values_per_result_value,
            )

            corners = itertools.product(
                *[
                    [(i, 1 - n), (i + 1, n)]
                    for i, n in zip(indices, norm_distances)
                ]
            )
            shape = self.values.shape[:ndim]

            for i, corner in enumerate(corners):
                corner_indices = [ci for ci, cw in corner]
                n_indices = np.ravel_multi_index(
                    corner_indices, shape, mode="wrap"
                )
                col_indices[i::n_src_values_per_result_value] = n_indices
                for ci, cw in corner:
                    weights[i::n_src_values_per_result_value] *= cw

            n_src_values = np.prod(list(map(len, self.grid)))
            sparse_matrix = csr_matrix(
                (weights, col_indices, row_ptrs),
                shape=(n_result_values, n_src_values),
            )

            prepared = (xi_shape, method, sparse_matrix, None, out_of_bounds)

        return prepared

    def interp_using_pre_computed_weights(self, computed_weights):
        """
        Perform the interpolation using pre-computed interpolation weights.

        .. note::
            This interface provides the ability to reuse weights on multiple
            data sources, such as in the case of regridding. For normal
            interpolation, simply call the interpolator with the sample points,
            rather using this decomposed interface.

        Parameters
        ----------
        computed_weights : *intentionally undefined interface*
            The pre-computed interpolation weights which come from calling
            :meth:`compute_interp_weights`.

        """
        [
            xi_shape,
            method,
            indices,
            norm_distances,
            out_of_bounds,
        ] = computed_weights

        method = self.method if method is None else method
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)

        ndim = len(self.grid)

        if method == "linear":
            result = self._evaluate_linear_sparse(indices)
        elif method == "nearest":
            result = self._evaluate_nearest(
                indices, norm_distances, out_of_bounds
            )
        if not self.bounds_error and self.fill_value is not None:
            result[out_of_bounds] = self.fill_value

        return result.reshape(xi_shape[:-1] + self.values.shape[ndim:])

    def _evaluate_linear_sparse(self, sparse_matrix):
        ndim = len(self.grid)
        if ndim == self.values.ndim:
            result = sparse_matrix * self.values.reshape(-1)
        else:
            shape = (sparse_matrix.shape[1], -1)
            result = sparse_matrix * self.values.reshape(shape)

        return result

    def _evaluate_nearest(self, indices, norm_distances, out_of_bounds):
        idx_res = []
        for i, yi in zip(indices, norm_distances):
            idx_res.append(np.where(yi <= 0.5, i, i + 1))
        return self.values[tuple(idx_res)]

    def _find_indices(self, xi):
        # find relevant edges between which xi are situated
        indices = []
        # compute distance to lower edge in unity units
        norm_distances = []
        # check for out of bounds xi
        out_of_bounds = np.zeros((xi.shape[1]), dtype=bool)
        # iterate through dimensions
        for x, grid in zip(xi, self.grid):
            i = np.searchsorted(grid, x) - 1
            i[i < 0] = 0
            i[i > grid.size - 2] = grid.size - 2
            indices.append(i)
            # TODO: Add this to scipy's version.
            if grid.size == 1:
                norm_distances.append(x - grid[i])
            else:
                norm_distances.append((x - grid[i]) / (grid[i + 1] - grid[i]))
            if not self.bounds_error:
                out_of_bounds += x < grid[0]
                out_of_bounds += x > grid[-1]
        return indices, norm_distances, out_of_bounds


# ============================================================================
# |                        END SciPy copyright                               |
# ============================================================================
