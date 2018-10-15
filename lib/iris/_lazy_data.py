# (C) British Crown Copyright 2017 - 2018, Met Office
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

from functools import wraps

import dask
import dask.array as da
import dask.context
from dask.local import get_sync as dget_sync
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


def _limited_shape(shape):
    # Reduce a shape to less than a default overall number-of-points, reducing
    # earlier dimensions preferentially.
    # Note: this is only a heuristic, assuming that earlier dimensions are
    # 'outer' storage dimensions -- not *always* true, even for NetCDF data.
    shape = list(shape)
    i_reduce = 0
    while np.prod(shape) > _MAX_CHUNK_SIZE:
        factor = np.ceil(np.prod(shape) / _MAX_CHUNK_SIZE)
        new_dim = int(shape[i_reduce] / factor)
        if new_dim < 1:
            new_dim = 1
        shape[i_reduce] = new_dim
        i_reduce += 1
    return tuple(shape)


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
        # Default to the shape of the wrapped array-like,
        # but reduce it if larger than a default maximum size.
        chunks = _limited_shape(data.shape)

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
