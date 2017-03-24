# (C) British Crown Copyright 2010 - 2017, Met Office
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
Management of common state and behaviour for cube and coordinate data.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

import copy
import numpy as np
import numpy.ma as ma

from iris._lazy_data import as_concrete_data, as_lazy_data, is_lazy_data


# TBD:
#   - unit test coverage.
#   - pep8 


class DataManager(object):
    def __init__(self, data, dtype=None):
        self._lazy_array = None
        self._real_array = None
        self.data = data

        self._dtype = None
        self._dtype_setter(dtype)

        # Enforce the manager contract.
        self._assert_axiom()

    def __copy__(self):
        name = self.__class__.__name__
        emsg = ('Shallow-copy of {!r} is not permitted. Use '
                'copy.deepcopy() or {}.copy() instead.')
        raise copy.Error(emsg.format(name, name))

    def __deepcopy__(self, memo):
        return self._deepcopy(memo)

    def _assert_axiom(self):
        """
        Definition of the manager state, that should never be violated.

        """
        # Ensure there is a valid data state.
        is_lazy = bool(self._lazy_array is None)
        is_real = bool(self._real_array is None)
        emsg = 'Unexpected data state, got {}lazy and {}real data.'
        state = is_lazy ^ is_real
        assert state, emsg.format('' if is_lazy else 'no ',
                                  '' if is_real else 'no ')
        # Ensure intended dtype validity.
        state = self._dtype is None or self._dtype in 'biu'
        emsg = 'Unexpected dtype state, got {!r}'
        assert state, emsg.format(self._dtype)

        # Ensure lazy data with intended dtype validity.
        state = not (self.has_real_data() and self._dtype is not None)
        emsg = ('Unexpected dtype with real data state, got '
                'real data and intended {!r}.')
        assert state, emsg.format(self._dtype)

    def _deepcopy(self, memo, data=None, dtype=None):
        ...
        # what's best/possible - copy or deepcopy a dask array ?

    def _dtype_setter(self, dtype):
        """
        Set the intended dtype of the realised lazy data. This is to support
        the case of lazy masked integral and boolean data in dask.

        Args:

        * dtype:
            A numpy :class:`~numpy.dtype`, array-protocol type string,
            or built-in scalar type.

        """
        if dtype is None:
            self._dtype = None
        else:
            dtype = np.dtype(dtype)
            if self.has_real_data():
                emsg = 'Cannot set dtype, no lazy data is available.'
                raise ValueError(emgs)
            if dtype.kind not in 'biu':
                emsg = ('Can only cast lazy data to an integer or boolean '
                        'dtype, got {!r}.')
                raise ValueError(emsg.format(dtype))
            self._dtype = dtype
        # check this logic for replace usage and copy usage!

    @property
    def core_data(self):
        """
        If real data is being managed, then return the :class:`~numpy.ndarray`
        or :class:`numpy.ma.core.MaskedArray`. Otherwise, return the lazy
        :class:`~dask.array.core.Array`.

        Returns:
            The real or lazy data.

        """
        if self.has_real_data():
            result = self._real_array
        else:
            result = self._lazy_array
        return result

    @property
    def data(self):
        """
        Returns the real :class:`~numpy.ndarray` or
        :class:`numpy.ma.core.MaskedArray`.

        Returns:
            The real data.

        .. note::
            Any lazy data being managed will be realised.

        """
        if self.has_lazy_data():
            try:
                # Realise the lazy data.
                result = as_concrete_data(self._lazy_array,
                                          nans_replacement=ma.masked,
                                          result_dtype=self.dtype)
                # Assign the realised result.
                self._real_array = result
                # Reset the lazy data and intended dtype of the realised
                # lazy data.
                self._lazy_array = None
                self._dtype = None
            except MemoryError:
                emsg = ('Failed to realise the lazy data as there was not '
                        'enough memory available.\n'
                        'The data shape would have been {!r} with {!r}.\n '
                        'Consider freeing up variables or indexing the data '
                        'before trying again.')
                raise MemoryError(emsg.format(self.shape, self.dtype))

            # Check the manager contract, as the managed data has changed.
            self._assert_axiom()

        return self._real_array

    @data.setter
    def data(self, data):
        # Ensure we have numpy-like data.
        if not (hasattr(data, 'shape') and hasattr(data, 'dtype')):
            data = np.asanyarray(data)

        # Determine whether the __init__ has completed.
        loaded = self._lazy_array is not None or self._real_array is not None

        if loaded and self.shape != data.shape:
            # The _ONLY_ data reshape permitted is converting a 0-dimensional
            # array i.e. self.shape == () into a 1-dimensional array of length
            # one i.e. data.shape == (1,)
            if self.shape or data.shape != (1,):
                emsg = 'Require data with shape {!r}, got {!r}.'
                raise ValueError(emsg.format(self.shape, data.shape))

        # Set lazy or real data, and reset the other.
        if is_lazy_data(data):
            self._lazy_array = data
            self._real_array = None
        else:
            if not ma.isMaskedArray(data):
                # Coerce input data to ndarray (including ndarray subclasses).
                data = np.asarray(data)
            self._lazy_array = None
            self._real_array = data

        # Always reset the intended dtype of the realised lazy data, as the
        # managed data has changed.
        self._dtype = None

        # Check the manager contract, as the managed data has changed.
        self._assert_axiom()

    @property
    def dtype(self):
        """
        The dtype of the realised lazy data or the dtype of the real data.

        """
        if self._dtype is not None:
            result = self._dtype
        else:
            result = self.core_data.dtype
        return result

    @property
    def ndim(self):
        """
        The number of dimensions covered by the data being managed.

        """
        return len(self.shape)

    @shape
    def shape(self):
        """
        The shape of the data being managed.

        """
        return self.core_data.shape

    def copy(self, data=None, dtype=None):
        """
        Returns a deep copy of this :class:`DataManager`.

        Kwargs:

        * data:
            Replace the data of the copy with this data.

        * dtype:
            Replace the intended dtype of the lazy data
            in the copy with this :class:`~numpy.dtype`.

        Returns:
            A copy :class:`DataManager`.
            
        """
        return self._deepcopy({}, data, dtype)

    def has_lazy_data(self):
        """
        Determine whether lazy data is being managed.

        Returns:
            Boolean.

        """
        return self._lazy_array is not None

    def has_real_data(self):
        """
        Determine whether real data is being managed.

        Returns:
            Boolean.

        """
        return self._real_array is not None

    def lazy_data(self):
        """
        Return the lazy representation of the managed data.

        If only real data is being managed, then return a lazy
        representation of that real data.

        Returns:
            :class:`~dask.array.core.Array`

        .. note::
            This method will never realise any lazy data.

        """
        if self.has_real_data():
            result = as_lazy_data(self._real_array)
        else:
            result = self._lazy_array
        return result

    def replace(self, data, dtype=None):
        """
        Perform an in-place replacement of the managed data.

        Args:

        * data:
            Replace the managed data with either the :class:`~numpy.ndarray`
            or :class:`~numpy.ma.core.MaskedArray` real data, or lazy
            :class:`dask.array.core.Array`

        Kwargs:

        * dtype:
            The intended dtype of the specified lazy data.

        """
        # Capture the currently managed data.
        cache = self.core_data
        # Perform in-place assignment.
        self.data = data
        try:
            self._dtype_setter(dtype)
        except ValueError as exception:
            # Reinstate the original (cached) managed data.
            self.data = cache
            raise exception
         

