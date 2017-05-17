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
import numpy as np
import numpy.ma as ma


def _iris_dask_defaults():
    """
    Set dask defaults for Iris. The current default dask operation mode for
    Iris is running single-threaded using `dask.async.get_sync`. This default
    ensures that running Iris under "normal" conditions will not use up all
    available computational resource.

    Otherwise, by default, `dask` will use a multi-threaded scheduler that uses
    all available CPUs.

    .. note::
        We only want Iris to set dask options in the case where doing so will
        not change user-specified options that have already been set.

    """
    if 'pool' not in dask.context._globals and \
            'get' not in dask.context._globals:
        dask.set_options(get=dask.async.get_sync)


# Run this at import time to set dask options for Iris.
_iris_dask_defaults()


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

    Returns:
        The input array converted to a dask array.

    """
    if not is_lazy_data(data):
        if ma.isMaskedArray(data):
            data = array_masked_to_nans(data)
        data = da.from_array(data, chunks=chunks)
    return data


def as_concrete_data(data, **kwargs):
    """
    Return the actual content of a lazy array, as a numpy array.
    If the input data is a NumPy `ndarray` or masked array, return it
    unchanged.

    If the input data is lazy, return the realised result.

    Where lazy data contains NaNs these are translated by filling or converting
    to masked data, using the :func:`~iris._lazy_data.convert_nans_array`
    function.

    Args:

    * data:
        A dask array, NumPy `ndarray` or masked array

    Kwargs are passed through to :func:`~iris._lazy_data.convert_nans_array`.

    Returns:
        A NumPy `ndarray` or masked array.

    """
    if is_lazy_data(data):
        # Realise dask array, ensuring the data result is always a NumPy array.
        # In some cases dask may return a scalar numpy.int/numpy.float object
        # rather than a numpy.ndarray object.
        # Recorded in https://github.com/dask/dask/issues/2111.
        data = np.asanyarray(data.compute())
        # Convert any missing data as requested.
        data = convert_nans_array(data, **kwargs)

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
            mask = array.mask
            if array.dtype.kind in 'biu':
                result = array.data.astype(np.dtype('f8'))
            else:
                result = array.data.copy()
            result[mask] = np.nan
        else:
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


def lazy_concat(data, axis):
    """
    Concatenate dask arrays along a given axis.

    Args:

    * data:
        An iterable of dask arrays.

    * axis:
        The data dimension along which to perform the concatenation.

    Returns:
        The input data concatenated along the specified axis.s

    """
    return da.concatenate(data, axis)


def convert_nans_array(array, nans_replacement=None, result_dtype=None):
    """
    Convert a :class:`~numpy.ndarray` that may contain one or more NaN values
    to either a :class:`~numpy.ma.core.MaskedArray` or a
    :class:`~numpy.ndarray` with the NaN values filled.

    Args:

    * array:
        The :class:`~numpy.ndarray` to be converted.

    Kwargs:

    * nans_replacement:
        If `nans_replacement` is None, then raise an exception if the `array`
        contains any NaN values (default behaviour).
        If `nans_replacement` is `numpy.ma.masked`, then convert the `array`
        to a :class:`~numpy.ma.core.MaskedArray`.
        Otherwise, use the specified `nans_replacement` value as the `array`
        fill value.

    * result_dtype:
        Cast the resultant array to this target :class:`~numpy.dtype`.

    Returns:
        An :class:`numpy.ndarray`.

    .. note::
        An input array that is either a :class:`~numpy.ma.core.MaskedArray`
        or has an integral dtype will be returned unaltered.

    .. note::
        In some cases, the input array is modified in-place.

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
            if nans_replacement is None:
                emsg = 'Array contains unexpected NaNs.'
                raise ValueError(emsg)
            elif nans_replacement is ma.masked:
                # Mask the array with the default fill_value.
                array = ma.masked_array(array, mask=mask)
            else:
                # Check the fill value is appropriate for the
                # result array dtype.
                try:
                    [fill_value] = np.asarray([nans_replacement],
                                              dtype=array.dtype)
                except OverflowError:
                    emsg = 'Fill value of {!r} invalid for array result {!r}.'
                    raise ValueError(emsg.format(nans_replacement,
                                                 array.dtype))
                # Fill the array.
                array[mask] = fill_value
    return array
