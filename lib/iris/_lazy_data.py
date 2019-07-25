# (C) British Crown Copyright 2017 - 2019, Met Office
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
Routines for lazy data handling.

To avoid replicating implementation-dependent test and conversion code.

"""
from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

from collections import Iterable
from functools import wraps

import dask
import dask.array as da
import dask.array.core
import dask.config

import numpy as np
import numpy.ma as ma


def non_lazy(func):
    """
    Turn a lazy function into a function that returns a result immediately.

    """
    @wraps(func)
    def inner(*args, **kwargs):
        """Immediately return the results of a lazy function."""
        result = func(*args, **kwargs)
        return dask.compute(result)[0]
    return inner


def is_lazy_data(data):
    """
    Return whether the argument is an Iris 'lazy' data array.

    At present, this means simply a Dask array.
    We determine this by checking for a "compute" property.

    """
    result = hasattr(data, 'compute')
    return result


# A magic value, chosen to minimise chunk creation time and chunk processing
# time within dask.
_MAX_CHUNK_SIZE = 8 * 1024 * 1024 * 2

# _MAX_CHUNK_SIZE = dask.config.get('array', {}).get('chunk-size', None)
# if _MAX_CHUNK_SIZE is not None:
#     # Convert to bytes
#     _MAX_CHUNK_SIZE = da.core.parse_bytes(_MAX_CHUNK_SIZE)
# else:
#     # Fall back on our own "magic" value.
#     _MAX_CHUNK_SIZE = 8 * 1024 * 1024 * 2


def _optimise_chunksize(chunks, shape,
                        limit=None,
                        dtype=np.dtype('f4')):
    """
    Reduce or increase an initial chunk shape to get close to a chosen ideal
    size, while prioritising the splitting of the earlier (outer) dimensions
    and keeping intact the later (inner) ones.

    Args:

    * chunks (tuple of int, or None):
        Pre-existing chunk shape of the target data : None if unknown.
    * shape (tuple of int):
        The full array shape of the target data.
    * limit (int):
        The 'ideal' target chunk size, in bytes.
    * dtype (np.dtype):
        Numpy dtype of target data.

    Returns:
    * chunk (tuple of int):
        The proposed shape of one full chunk.

    .. note::
        The use of this is very similar to `dask.array.core.normalize_chunks`,
        when called as
        `(chunks='auto', shape, dtype=dtype, previous_chunks=chunks, ...)`.
        Except : that the logic here is optimised for a specific to a 'c-like'
        dimension order, i.e. outer dimensions first, as in netcdf variables.
        So if, in future, this policy can be implemented in dask, then we would
        prefer to replace this function with a call to that one.
        Accordingly, the arguments roughly match 'normalize_chunks', except
        that we don't support the alternative argument forms of that routine.
        This routine also returns a single 'full chunk', rather
        than a complete chunking scheme : so equivalent code usage would be
        "chunks = [c[0] for c in normalise_chunks(chunks, ...)]".

    """
    # Return chunks unchanged, for types of invocation we don't comprehend.
    if (any(elem <= 0 for elem in shape) or
            not isinstance(chunks, Iterable) or
            len(chunks) != len(shape)):
        # Don't modify chunks for special values like -1, (0,), 'auto',
        # or if shape contains 0 or -1 (like raw landsea-mask data proxies).
        return chunks

    # Calculate default chunksize limit.
    if limit is None:
        limit = _MAX_CHUNK_SIZE * 4
    point_size_limit = limit / dtype.itemsize

    # Create result chunks, starting with a copy of the input.
    result = list(chunks)
    if shape is None:
        shape = result[:]

    if np.prod(result) < point_size_limit:
        # If size is less than maximum, expand the chunks, multiplying later
        # (i.e. inner) dims first.
        i_expand = len(shape) - 1
        while np.prod(result) < point_size_limit and i_expand >= 0:
            factor = np.floor(point_size_limit * 1.0 / np.prod(result))
            new_dim = result[i_expand] * int(factor)
            # Clip to dim size : N.B. means it cannot exceed the original dims.
            if new_dim > shape[i_expand]:
                new_dim = shape[i_expand]
            result[i_expand] = new_dim
            i_expand -= 1
    else:
        # Similarly, reduce if too big, reducing earlier (outer) dims first.
        i_reduce = 0
        while np.prod(result) > point_size_limit:
            factor = np.ceil(np.prod(result) / point_size_limit)
            new_dim = int(result[i_reduce] / factor)
            if new_dim < 1:
                new_dim = 1
            result[i_reduce] = new_dim
            i_reduce += 1

    return tuple(result)


def as_lazy_data(data, chunks=None, asarray=False):
    """
    Convert the input array `data` to a dask array.

    Args:

    * data:
        An array. This will be converted to a dask array.

    Kwargs:

    * chunks:
        Describes how the created dask array should be split up. Defaults to a
        value first defined in biggus (being `8 * 1024 * 1024 * 2`).
        For more information see
        http://dask.pydata.org/en/latest/array-creation.html#chunks.

    * asarray:
        If True, then chunks will be converted to instances of `ndarray`.
        Set to False (default) to pass passed chunks through unchanged.

    Returns:
        The input array converted to a dask array.

    """
    if chunks is None:
        # No existing chunks : Make a chunk the shape of the entire input array
        # (but we will subdivide it if too big).
        chunks = list(data.shape)

    # Expand or reduce the basic chunk shape to an optimum size.
    chunks = _optimise_chunksize(chunks, shape=data.shape)

    if isinstance(data, ma.core.MaskedConstant):
        data = ma.masked_array(data.data, mask=data.mask)
    if not is_lazy_data(data):
        data = da.from_array(data, chunks=chunks, asarray=asarray)
    return data


def _co_realise_lazy_arrays(arrays):
    """
    Compute multiple lazy arrays and return a list of real values.

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
            real_out = ma.masked_array(real_out.data, mask=real_out.mask,
                                       dtype=lazy_in.dtype)
        results.append(real_out)
    return results


def as_concrete_data(data):
    """
    Return the actual content of a lazy array, as a numpy array.
    If the input data is a NumPy `ndarray` or masked array, return it
    unchanged.

    If the input data is lazy, return the realised result.

    Args:

    * data:
        A dask array, NumPy `ndarray` or masked array

    Returns:
        A NumPy `ndarray` or masked array.

    """
    if is_lazy_data(data):
        data, = _co_realise_lazy_arrays([data])

    return data


def multidim_lazy_stack(stack):
    """
    Recursively build a multidimensional stacked dask array.

    This is needed because dask.array.stack only accepts a 1-dimensional list.

    Args:

    * stack:
        An ndarray of dask arrays.

    Returns:
        The input array converted to a lazy dask array.

    """
    if stack.ndim == 0:
        # A 0-d array cannot be stacked.
        result = stack.item()
    elif stack.ndim == 1:
        # Another base case : simple 1-d goes direct in dask.
        result = da.stack(list(stack))
    else:
        # Recurse because dask.stack does not do multi-dimensional.
        result = da.stack([multidim_lazy_stack(subarray)
                           for subarray in stack])
    return result


def co_realise_cubes(*cubes):
    """
    Fetch 'real' data for multiple cubes, in a shared calculation.

    This computes any lazy data, equivalent to accessing each `cube.data`.
    However, lazy calculations and data fetches can be shared between the
    computations, improving performance.

    Args:

    * cubes (list of :class:`~iris.cube.Cube`):
        Arguments, each of which is a cube to be realised.

    For example::

        # Form stats.
        a_std = cube_a.collapsed(['x', 'y'], iris.analysis.STD_DEV)
        b_std = cube_b.collapsed(['x', 'y'], iris.analysis.STD_DEV)
        ab_mean_diff = (cube_b - cube_a).collapsed(['x', 'y'],
                                                   iris.analysis.MEAN)
        std_err = (a_std * a_std + b_std * b_std) ** 0.5

        # Compute stats together (to avoid multiple data passes).
        co_realise_cubes(a_std, b_std, ab_mean_diff, std_err)


    .. Note::

        Cubes with non-lazy data may also be passed, with no ill effect.

    """
    results = _co_realise_lazy_arrays([cube.core_data() for cube in cubes])
    for cube, result in zip(cubes, results):
        cube.data = result


def lazy_elementwise(lazy_array, elementwise_op):
    """
    Apply a (numpy-style) elementwise array operation to a lazy array.

    Elementwise means that it performs a independent calculation at each point
    of the input, producing a result array of the same shape.

    Args:

    * lazy_array:
        The lazy array object to operate on.
    * elementwise_op:
        The elementwise operation, a function operating on numpy arrays.

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
