# (C) British Crown Copyright 2017, Met Office
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

import dask
import dask.array as da
import dask.context
from dask.local import get_sync as dget_sync
import numpy as np
import numpy.ma as ma


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
        # Realise dask array, ensuring the data result is always a NumPy array.
        # In some cases dask may return a scalar numpy.int/numpy.float object
        # rather than a numpy.ndarray object.
        # Recorded in https://github.com/dask/dask/issues/2111.
        dtype = data.dtype
        data = np.asanyarray(data.compute())
        if isinstance(data, ma.core.MaskedConstant):
            data = ma.masked_array(data.data, dtype=dtype, mask=data.mask)

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
