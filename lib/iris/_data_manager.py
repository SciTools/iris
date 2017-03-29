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
from iris.util import array_equal


class DataManager(object):
    """
    Provides a well defined API for management of real or lazy data.

    """

    def __init__(self, data, realised_dtype=None):
        """
        Create a data manager for the specified data.

        Args:

        * data:
            The :class:`~numpy.ndarray` or :class:`~numpy.ma.core.MaskedArray`
            real data, or :class:`~dask.array.core.Array` lazy data to be
            managed.

        Kwargs:

        * realised_dtype:
            The intended dtype of the specified lazy data.

        """
        self._lazy_array = None
        self._real_array = None
        self.data = data

        self._realised_dtype = None
        self._realised_dtype_setter(realised_dtype)

        # Enforce the manager contract.
        self._assert_axioms()

    def __copy__(self):
        """
        Forbid :class:`~iris._data_manager.DataManager` instance
        shallow-copy support.

        """
        name = type(self).__name__
        emsg = ('Shallow-copy of {!r} is not permitted. Use '
                'copy.deepcopy() or {}.copy() instead.')
        raise copy.Error(emsg.format(name, name))

    def __deepcopy__(self, memo):
        """
        Allow :class:`~iris._data_manager.DataManager` instance
        deepcopy support.

        Args:

        * memo:
            :class:`copy` memo dictionary.

        """
        return self._deepcopy(memo)

    def __eq__(self, other):
        """
        Perform :class:`~iris._data_manager.DataManager` instance equality.
        Note that, this is explicitly not a lazy operation and will load any
        lazy payload to determine the equality result.

        Comparison is strict with regards to lazy or real managed payload,
        the realised_dtype, the dtype of the payload and the payload content.

        Args:

        * other:
            The :class:`~iris._data_manager.DataManager` instance to
            compare with.

        Returns:
            Boolean.

        """
        result = NotImplemented

        if isinstance(other, type(self)):
            result = False
            same_lazy = self.has_lazy_data() == other.has_lazy_data()
            same_realised_dtype = self._realised_dtype == other._realised_dtype
            same_dtype = self.dtype == other.dtype
            if same_lazy and same_realised_dtype and same_dtype:
                result = array_equal(self.core_data, other.core_data)

        return result

    def __ne__(self, other):
        """
        Perform :class:`~iris._data_manager.DataManager` instance inequality.
        Note that, this is explicitly not a lazy operation and will load any
        lazy payload to determine the inequality result.

        Args:

        * other:
            The :class:`~iris._data_manager.DataManager` instance to
            compare with.

        Returns:
            Boolean.

        """
        result = self.__eq__(other)

        if result is not NotImplemented:
            result = not result

        return result

    def __repr__(self):
        """
        Returns an string representation of the instance.

        """
        fmt = '{cls}({self.core_data!r}{dtype})'
        dtype = ''

        if self._realised_dtype is not None:
            dtype = ', realised_dtype={!r}'.format(self._realised_dtype)

        result = fmt.format(self=self, cls=type(self).__name__, dtype=dtype)

        return result

    def _assert_axioms(self):
        """
        Definition of the manager state, that should never be violated.

        """
        # Ensure there is a valid data state.
        is_lazy = self._lazy_array is not None
        is_real = self._real_array is not None
        emsg = 'Unexpected data state, got {}lazy and {}real data.'
        state = is_lazy ^ is_real
        assert state, emsg.format('' if is_lazy else 'no ',
                                  '' if is_real else 'no ')
        # Ensure validity of realised dtype.
        state = (self._realised_dtype is None or
                 self._realised_dtype.kind in 'biu')
        emsg = 'Unexpected realised dtype state, got {!r}'
        assert state, emsg.format(self._realised_dtype)

        # Ensure validity of lazy data with realised dtype.
        state = not (self.has_real_data() and self._realised_dtype is not None)
        emsg = ('Unexpected real data with realised dtype, got '
                'real data and realised {!r}.')
        assert state, emsg.format(self._realised_dtype)

    def _deepcopy(self, memo, data=None, realised_dtype=None):
        """
        Perform a deepcopy of the :class:`~iris._data_manager.DataManager`
        instance.

        Args:

        * memo:
            :class:`copy` memo dictionary.

        Kwargs:

        * data:
            Replacement data to substitute the currently managed
            data with.

        * realised_dtype:
            Replacement for the intended dtype of the realised lazy data.

        Returns:
            :class:`~iris._data_manager.DataManager` instance.

        """
        try:
            if data is None:
                # Copy the managed data.
                if self.has_lazy_data():
                    data = copy.deepcopy(self._lazy_array, memo)
                else:
                    data = self._real_array.copy()
            else:
                # Check that the replacement data is valid relative to
                # the currently managed data.
                DataManager(self.core_data).replace(data)
                # If the replacement data is valid, then use it but
                # without copying it.

            result = DataManager(data, realised_dtype=realised_dtype)
        except ValueError as error:
            emsg = 'Cannot copy {!r} - {}'
            raise ValueError(emsg.format(type(self).__name__, error))

        return result

    def _realised_dtype_setter(self, realised_dtype):
        """
        Set the intended dtype of the realised lazy data. This is to support
        the case of lazy masked integral and boolean data in dask.

        Args:

        * realised_dtype:
            A numpy :class:`~numpy.dtype`, array-protocol type string,
            or built-in scalar type.

        """
        if realised_dtype is None:
            self._realised_dtype = None
        else:
            if self.has_real_data():
                emsg = 'Cannot set realised dtype, no lazy data is available.'
                raise ValueError(emsg)
            realised_dtype = np.dtype(realised_dtype)
            if realised_dtype.kind not in 'biu':
                emsg = ('Can only cast lazy data to an integer or boolean '
                        'dtype, got {!r}.')
                raise ValueError(emsg.format(realised_dtype))
            self._realised_dtype = realised_dtype

            # Check the manager contract, as the managed dtype has changed.
            self._assert_axioms()

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
        Returns the real data. Any lazy data being managed will be realised.

        Returns:
            :class:`~numpy.ndarray` or :class:`numpy.ma.core.MaskedArray`.

        """
        if self.has_lazy_data():
            try:
                # Realise the lazy data.
                result = as_concrete_data(self._lazy_array,
                                          nans_replacement=ma.masked,
                                          result_dtype=self.dtype)
                # Assign the realised result.
                self._real_array = result
                # Reset the lazy data and the realised dtype.
                self._lazy_array = None
                self._realised_dtype = None
            except MemoryError:
                emsg = ('Failed to realise the lazy data as there was not '
                        'enough memory available.\n'
                        'The data shape would have been {!r} with {!r}.\n '
                        'Consider freeing up variables or indexing the data '
                        'before trying again.')
                raise MemoryError(emsg.format(self.shape, self.dtype))

            # Check the manager contract, as the managed data has changed.
            self._assert_axioms()

        return self._real_array

    @data.setter
    def data(self, data):
        """
        Replaces the currently managed data with the specified data, which must
        be of an equivalent shape.

        Note that, the only shape promotion permitted is for 0-dimensional
        scalar data to be replaced with a single item 1-dimensional data.

        Args:

        * data:
            The :class:`~numpy.ndarray` or :class:`~numpy.ma.core.MaskedArray`
            real data, or :class:`~dask.array.core.Array` lazy data to be
            managed.

        """
        # Ensure we have numpy-like data.
        if not (hasattr(data, 'shape') and hasattr(data, 'dtype')):
            data = np.asanyarray(data)

        # Determine whether the class instance has been created,
        # as this method is called from within the __init__.
        init_done = (self._lazy_array is not None or
                     self._real_array is not None)

        if init_done and self.shape != data.shape:
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

        # Always reset the realised dtype, as the managed data has changed.
        self._realised_dtype = None

        # Check the manager contract, as the managed data has changed.
        self._assert_axioms()

    @property
    def dtype(self):
        """
        The dtype of the realised lazy data or the dtype of the real data.

        """
        if self._realised_dtype is not None:
            result = self._realised_dtype
        else:
            result = self.core_data.dtype
        return result

    @property
    def ndim(self):
        """
        The number of dimensions covered by the data being managed.

        """
        return self.core_data.ndim

    @property
    def shape(self):
        """
        The shape of the data being managed.

        """
        return self.core_data.shape

    def copy(self, data=None, realised_dtype=None):
        """
        Returns a deep copy of this :class:`~iris._data_manager.DataManager`
        instance.

        Kwargs:

        * data:
            Replace the data of the copy with this data.

        * realised_dtype:
            Replace the intended dtype of the lazy data
            in the copy with this :class:`~numpy.dtype`.

        Returns:
            A copy :class:`~iris._data_manager.DataManager` instance.

        """
        memo = {}
        return self._deepcopy(memo, data=data, realised_dtype=realised_dtype)

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

    def replace(self, data, realised_dtype=None):
        """
        Perform an in-place replacement of the managed data.

        Args:

        * data:
            Replace the managed data with either the :class:`~numpy.ndarray`
            or :class:`~numpy.ma.core.MaskedArray` real data, or lazy
            :class:`dask.array.core.Array`

        Kwargs:

        * realised_dtype:
            The intended dtype of the specified lazy data.

        .. note::
            Data replacement alone will clear the intended dtype
            of the realised lazy data.

        """
        # Snapshot the currently managed data.
        original_data = self.core_data
        # Perform in-place data assignment.
        self.data = data
        try:
            self._realised_dtype_setter(realised_dtype)
        except ValueError as error:
            # Backout the data replacement, and reinstate the cached
            # original managed data.
            self.data = original_data
            raise error
