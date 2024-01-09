# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Routines for lazy data handling.

To avoid replicating implementation-dependent test and conversion code.

"""

from functools import lru_cache, wraps

import dask
import dask.array as da
import dask.config
import dask.utils
import numpy as np
import numpy.ma as ma


def non_lazy(func):
    """Turn a lazy function into a function that returns a result immediately."""

    @wraps(func)
    def inner(*args, **kwargs):
        """Immediately return the results of a lazy function."""
        result = func(*args, **kwargs)
        return dask.compute(result)[0]

    return inner


def is_lazy_data(data):
    """Return whether the argument is an Iris 'lazy' data array.

    At present, this means simply a :class:`dask.array.Array`.
    We determine this by checking for a "compute" property.

    """
    result = hasattr(data, "compute")
    return result


def is_lazy_masked_data(data):
    """Return True if the argument is both an Iris 'lazy' data array and the
    underlying array is of masked type.  Otherwise return False.

    """
    return is_lazy_data(data) and ma.isMA(da.utils.meta_from_array(data))


@lru_cache
def _optimum_chunksize_internals(
    chunks,
    shape,
    limit=None,
    dtype=np.dtype("f4"),
    dims_fixed=None,
    dask_array_chunksize=dask.config.get("array.chunk-size"),
):
    """Reduce or increase an initial chunk shap.

    Reduce or increase an initial chunk shape to get close to a chosen ideal
    size, while prioritising the splitting of the earlier (outer) dimensions
    and keeping intact the later (inner) ones.

    Parameters
    ----------
    chunks : tuple of int
        Pre-existing chunk shape of the target data.
    shape : tuple of int
        The full array shape of the target data.
    limit : int
        The 'ideal' target chunk size, in bytes.  Default from
        :mod:`dask.config`.
    dtype : np.dtype
        Numpy dtype of target data.
    dims_fixed : list of bool
        If set, a list of values equal in length to 'chunks' or 'shape'.
        'True' values indicate a dimension that can not be changed, i.e. that
        element of the result must equal the corresponding value in 'chunks' or
        data.shape.

    Returns
    -------
    tuple of int
        The proposed shape of one full chunk.

    Notes
    -----
    The purpose of this is very similar to
    :func:`dask.array.core.normalize_chunks`, when called as
    `(chunks='auto', shape, dtype=dtype, previous_chunks=chunks, ...)`.
    Except, the operation here is optimised specifically for a 'c-like'
    dimension order, i.e. outer dimensions first, as for netcdf variables.
    So if, in future, this policy can be implemented in dask, then we would
    prefer to replace this function with a call to that one.
    Accordingly, the arguments roughly match 'normalize_chunks', except
    that we don't support the alternative argument forms of that routine.
    The return value, however, is a single 'full chunk', rather than a
    complete chunking scheme : so an equivalent code usage could be
    "chunks = [c[0] for c in normalise_chunks('auto', ...)]".

    """
    # Set the chunksize limit.
    if limit is None:
        # Fetch the default 'optimal' chunksize from the dask config.
        limit = dask_array_chunksize
        # Convert to bytes
        limit = dask.utils.parse_bytes(limit)

    point_size_limit = limit / dtype.itemsize

    if dims_fixed is not None:
        if not np.any(dims_fixed):
            dims_fixed = None

    if dims_fixed is None:
        # Get initial result chunks, starting with a copy of the input.
        working = list(chunks)
    else:
        # Adjust the operation to ignore the 'fixed' dims.
        # (We reconstruct the original later, before return).
        chunks = np.array(chunks)
        dims_fixed_arr = np.array(dims_fixed)
        # Reduce the target size by the fixed size of all the 'fixed' dims.
        point_size_limit = point_size_limit // np.prod(chunks[dims_fixed_arr])
        # Work on only the 'free' dims.
        original_shape = tuple(shape)
        shape = tuple(np.array(shape)[~dims_fixed_arr])
        working = list(chunks[~dims_fixed_arr])

    if len(working) >= 1:
        if np.prod(working) < point_size_limit:
            # If size is less than maximum, expand the chunks, multiplying
            # later (i.e. inner) dims first.
            i_expand = len(shape) - 1
            while np.prod(working) < point_size_limit and i_expand >= 0:
                factor = np.floor(point_size_limit * 1.0 / np.prod(working))
                new_dim = working[i_expand] * int(factor)
                if new_dim >= shape[i_expand]:
                    # Clip to dim size : must not exceed the full shape.
                    new_dim = shape[i_expand]
                else:
                    # 'new_dim' is less than the relevant dim of 'shape' -- but
                    # it is also the largest possible multiple of the
                    # input-chunks, within the size limit.
                    # So : 'i_expand' is the outer (last) dimension over which
                    # we will multiply the input chunks, and 'new_dim' is a
                    # value giving the fewest possible chunks within that dim.

                    # Now replace 'new_dim' with the value **closest to
                    # equal-size chunks**, for the same (minimum) number of
                    # chunks.  More-equal chunks are practically better.
                    # E.G. : "divide 8 into multiples of 2, with a limit of 7",
                    # produces new_dim=6, meaning chunks of sizes (6, 2).
                    # But (4, 4) is clearly better for memory and time cost.

                    # Calculate how many (expanded) chunks fit in this dim.
                    dim_chunks = np.ceil(shape[i_expand] * 1.0 / new_dim)
                    # Get "ideal" (equal) size for that many chunks.
                    ideal_equal_chunk_size = shape[i_expand] / dim_chunks
                    # Use the nearest whole multiple of input chunks >= ideal.
                    new_dim = int(
                        working[i_expand]
                        * np.ceil(ideal_equal_chunk_size / working[i_expand])
                    )

                working[i_expand] = new_dim
                i_expand -= 1
        else:
            # Similarly, reduce if too big, reducing earlier (outer) dims first.
            i_reduce = 0
            while np.prod(working) > point_size_limit:
                factor = np.ceil(np.prod(working) / point_size_limit)
                new_dim = int(working[i_reduce] / factor)
                if new_dim < 1:
                    new_dim = 1
                working[i_reduce] = new_dim
                i_reduce += 1

    working = tuple(working)

    if dims_fixed is None:
        result = working
    else:
        # Reconstruct the original form
        result = []
        for i_dim in range(len(original_shape)):
            if dims_fixed[i_dim]:
                dim = chunks[i_dim]
            else:
                dim = working[0]
                working = working[1:]
            result.append(dim)

    return result


@wraps(_optimum_chunksize_internals)
def _optimum_chunksize(
    chunks,
    shape,
    limit=None,
    dtype=np.dtype("f4"),
    dims_fixed=None,
):
    # By providing dask_array_chunksize as an argument, we make it so that the
    # output of _optimum_chunksize_internals depends only on its arguments (and
    # thus we can use lru_cache)
    return _optimum_chunksize_internals(
        tuple(chunks),
        tuple(shape),
        limit=limit,
        dtype=dtype,
        dims_fixed=dims_fixed,
        dask_array_chunksize=dask.config.get("array.chunk-size"),
    )


def as_lazy_data(
    data, chunks=None, asarray=False, dims_fixed=None, dask_chunking=False
):
    """Convert the input array `data` to a :class:`dask.array.Array`.

    Parameters
    ----------
    data : array-like
        An indexable object with 'shape', 'dtype' and 'ndim' properties.
        This will be converted to a :class:`dask.array.Array`.
    chunks : list of int, optional
        If present, a source chunk shape, e.g. for a chunked netcdf variable.
    asarray : bool, optional
        If True, then chunks will be converted to instances of `ndarray`.
        Set to False (default) to pass passed chunks through unchanged.
    dims_fixed : list of bool, optional
        If set, a list of values equal in length to 'chunks' or data.ndim.
        'True' values indicate a dimension which can not be changed, i.e. the
        result for that index must equal the value in 'chunks' or data.shape.
    dask_chunking : bool, optional
        If True, Iris chunking optimisation will be bypassed, and dask's default
        chunking will be used instead. Including a value for chunks while dask_chunking
        is set to True will result in a failure.

    Returns
    -------
    :class:`dask.array.Array`
        The input array converted to a :class:`dask.array.Array`.

    Notes
    -----
    The result chunk size is a multiple of 'chunks', if given, up to the
    dask default chunksize, i.e. `dask.config.get('array.chunk-size'),
    or the full data shape if that is smaller.
    If 'chunks' is not given, the result has chunks of the full data shape,
    but reduced by a factor if that exceeds the dask default chunksize.

    """
    if dask_chunking:
        if chunks is not None:
            raise ValueError(
                f"Dask chunking chosen, but chunks already assigned value {chunks}"
            )
        lazy_params = {"asarray": asarray, "meta": np.ndarray}
    else:
        if chunks is None:
            # No existing chunks : Make a chunk the shape of the entire input array
            # (but we will subdivide it if too big).
            chunks = list(data.shape)

        # Adjust chunk size for better dask performance,
        # NOTE: but only if no shape dimension is zero, so that we can handle the
        # PPDataProxy of "raw" landsea-masked fields, which have a shape of (0, 0).
        if all(elem > 0 for elem in data.shape):
            # Expand or reduce the basic chunk shape to an optimum size.
            chunks = _optimum_chunksize(
                chunks,
                shape=data.shape,
                dtype=data.dtype,
                dims_fixed=dims_fixed,
            )
        lazy_params = {
            "chunks": chunks,
            "asarray": asarray,
            "meta": np.ndarray,
        }
    if isinstance(data, ma.core.MaskedConstant):
        data = ma.masked_array(data.data, mask=data.mask)
    if not is_lazy_data(data):
        data = da.from_array(data, **lazy_params)
    return data


def _co_realise_lazy_arrays(arrays):
    """Compute multiple lazy arrays and return a list of real values.

    All the arrays are computed together, so they can share results for common
    graph elements.

    Casts all results with `np.asanyarray`, and converts any MaskedConstants
    appearing into masked arrays, to ensure that all return values are
    writeable NumPy array objects.

    Any non-lazy arrays are passed through, as they are by `da.compute`.
    They undergo the same result standardisation.

    """
    computed_arrays = da.compute(*arrays)
    results = []
    for lazy_in, real_out in zip(arrays, computed_arrays):
        # Ensure we always have arrays.
        # Note : in some cases dask (and numpy) will return a scalar
        # numpy.int/numpy.float object rather than an ndarray.
        # Recorded in https://github.com/dask/dask/issues/2111.
        real_out = np.asanyarray(real_out)
        if isinstance(real_out, ma.core.MaskedConstant):
            # Convert any masked constants into NumPy masked arrays.
            # NOTE: in this case, also apply the original lazy-array dtype, as
            # masked constants *always* have dtype float64.
            real_out = ma.masked_array(
                real_out.data, mask=real_out.mask, dtype=lazy_in.dtype
            )
        results.append(real_out)
    return results


def as_concrete_data(data):
    """Return the actual content of a lazy array, as a numpy array.
    If the input data is a NumPy `ndarray` or masked array, return it
    unchanged.

    If the input data is lazy, return the realised result.

    Parameters
    ----------
    data :
        A dask array, NumPy `ndarray` or masked array

    Returns
    -------
    NumPy `ndarray` or masked array.

    """
    if is_lazy_data(data):
        (data,) = _co_realise_lazy_arrays([data])

    return data


def multidim_lazy_stack(stack):
    """Recursively build a multidimensional stacked dask array.

    This is needed because :meth:`dask.array.Array.stack` only accepts a
    1-dimensional list.

    Parameters
    ----------
    stack :
        An ndarray of :class:`dask.array.Array`.

    Returns
    -------
    The input array converted to a lazy :class:`dask.array.Array`.

    """
    if stack.ndim == 0:
        # A 0-d array cannot be stacked.
        result = stack.item()
    elif stack.ndim == 1:
        # Another base case : simple 1-d goes direct in dask.
        result = da.stack(list(stack))
    else:
        # Recurse because dask.stack does not do multi-dimensional.
        result = da.stack([multidim_lazy_stack(subarray) for subarray in stack])
    return result


def co_realise_cubes(*cubes):
    """Fetch 'real' data for multiple cubes, in a shared calculation.

    This computes any lazy data, equivalent to accessing each `cube.data`.
    However, lazy calculations and data fetches can be shared between the
    computations, improving performance.

    Parameters
    ----------
    cubes : list of :class:`~iris.cube.Cube`
        Arguments, each of which is a cube to be realised.

    Examples
    --------
    ::

        # Form stats.
        a_std = cube_a.collapsed(['x', 'y'], iris.analysis.STD_DEV)
        b_std = cube_b.collapsed(['x', 'y'], iris.analysis.STD_DEV)
        ab_mean_diff = (cube_b - cube_a).collapsed(['x', 'y'],
                                                   iris.analysis.MEAN)
        std_err = (a_std * a_std + b_std * b_std) ** 0.5

        # Compute stats together (to avoid multiple data passes).
        co_realise_cubes(a_std, b_std, ab_mean_diff, std_err)


        .. note::

            Cubes with non-lazy data may also be passed, with no ill effect.

    """
    results = _co_realise_lazy_arrays([cube.core_data() for cube in cubes])
    for cube, result in zip(cubes, results):
        cube.data = result


def lazy_elementwise(lazy_array, elementwise_op):
    """Apply a (numpy-style) elementwise array operation to a lazy array.

    Elementwise means that it performs a independent calculation at each point
    of the input, producing a result array of the same shape.

    Parameters
    ----------
    lazy_array :
        The lazy array object to operate on.
    elementwise_op :
        The elementwise operation, a function operating on numpy arrays.

    Notes
    -----
    .. note:

        A single-point "dummy" call is made to the operation function, to
        determine dtype of the result.
        This return dtype must be stable in actual operation (!)

    """
    # This is just a wrapper to provide an Iris-specific abstraction for a
    # lazy operation in Dask (map_blocks).

    # Explicitly determine the return type with a dummy call.
    # This makes good practical sense for unit conversions, as a Unit.convert
    # call may cast to float, or not, depending on unit equality : Thus, it's
    # much safer to get udunits to decide that for us.
    dtype = elementwise_op(np.zeros(1, lazy_array.dtype)).dtype

    return da.map_blocks(elementwise_op, lazy_array, dtype=dtype)


def map_complete_blocks(src, func, dims, out_sizes):
    """Apply a function to complete blocks.

    Complete means that the data is not chunked along the chosen dimensions.

    Parameters
    ----------
    src : :class:`~iris.cube.Cube` or array-like
        Source cube that function is applied to.
    func :
        Function to apply.
    dims : tuple of int
        Dimensions that cannot be chunked.
    out_sizes : tuple of int
        Output size of dimensions that cannot be chunked.

    """
    if is_lazy_data(src):
        data = src
    elif not hasattr(src, "has_lazy_data"):
        # Not a lazy array and not a cube.  So treat as ordinary numpy array.
        return func(src)
    elif not src.has_lazy_data():
        return func(src.data)
    else:
        data = src.lazy_data()

    # Ensure dims are not chunked
    in_chunks = list(data.chunks)
    for dim in dims:
        in_chunks[dim] = src.shape[dim]
    data = data.rechunk(in_chunks)

    # Determine output chunks
    out_chunks = list(data.chunks)
    for dim, size in zip(dims, out_sizes):
        out_chunks[dim] = size

    return data.map_blocks(func, chunks=out_chunks, dtype=src.dtype)
