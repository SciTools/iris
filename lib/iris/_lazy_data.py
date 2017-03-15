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

import dask.array as da
import numpy as np
import numpy.ma as ma


def is_lazy_data(data):
    """
    Return whether the argument is an Iris 'lazy' data array.

    At present, this means simply a Dask array.
    We determine this by checking for a "compute" property.
    NOTE: ***for now only*** accept Biggus arrays also.

    """
    result = hasattr(data, 'compute')
    return result


# A magic value, borrowed from biggus
_MAX_CHUNK_SIZE = 8 * 1024 * 1024 * 2


def as_lazy_data(data, chunks=_MAX_CHUNK_SIZE):
    """
    Convert the input array `data` to a lazy dask array.

    Args:

    * data:
        An array. This will be converted to a lazy dask array.

    Kwargs:

    * chunks:
        Describes how the created dask array should be split up. Defaults to a
        value first defined in biggus (being `8 * 1024 * 1024 * 2`).
        For more information see
        http://dask.pydata.org/en/latest/array-creation.html#chunks.

    Returns:
        The input array converted to a lazy dask array.

    """
    if not is_lazy_data(data):
        if isinstance(data, ma.MaskedArray):
            data = array_masked_to_nans(data)
        data = da.from_array(data, chunks=chunks)
    return data


def array_masked_to_nans(array):
    """
    Convert a masked array to a NumPy `ndarray` filled with NaN values. Input
    NumPy arrays with no mask are returned unchanged.
    This is used for dask integration, as dask does not support masked arrays.

    Args:

    * array:
        A NumPy `ndarray` or masked array.

    Returns:
        A NumPy `ndarray`. This is the input array if unmasked, or an array
        of floating-point values with NaN values where the mask was `True` if
        the input array is masked.

    .. note::
        The fill value and mask of the input masked array will be lost.

    .. note::
        Integer masked arrays are cast to 8-byte floats because NaN is a
        floating-point value.

    """
    if not ma.isMaskedArray(array):
        result = array
    else:
        if ma.is_masked(array):
            if array.dtype.kind == 'i':
                array = array.astype(np.dtype('f8'))
            mask = array.mask
            array[mask] = np.nan
        result = array.data
    return result


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


def convert_nans_array(array, nans=None, result_dtype=None):
    """
    Convert an array into a masked array, by masking any NaN points.

    Optional data type casting is also performed to the specified dtype.

    The resultant array may also be filled with the fill_value, if requested.

    """
    if not ma.isMaskedArray(array) and array.dtype.kind == 'f':
        # First, calculate the mask.
        mask = np.isnan(array)
        # Now, cast the dtype, if required.
        if result_dtype is not None:
            result_dtype = np.dtype(result_dtype)
            if array.dtype != result_dtype:
                array = array.astype(result_dtype)
        # Finally, mask or fill the data, as required or raise an exception
        # if we detect there are NaNs present and we didn't expect any.
        if np.any(mask):
            if nans is None:
                emsg = 'Array contains unexpected NaNs.'
                raise ValueError(emsg)
            elif nans is ma.masked:
                # Mask the array with the default fill_value.
                array = ma.masked_array(array, mask=mask)
            else:
                # Fill the array.
                array[mask] = nans
    return array
