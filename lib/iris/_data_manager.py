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

    def __init__(self, data, fill_value=None, realised_dtype=None):
        """
        Create a data manager for the specified data.

        Args:

        * data:
            The :class:`~numpy.ndarray` or :class:`~numpy.ma.core.MaskedArray`
            real data, or :class:`~dask.array.core.Array` lazy data to be
            managed.

        Kwargs:

        * fill_value:
            The intended fill-value of :class:`~iris._data_manager.DataManager`
            masked data. Note that, the fill-value is cast relative to the
            dtype of the :class:`~iris._data_manager.DataManager`.

        * realised_dtype:
            The intended dtype of the specified lazy data, which must be
            either integer or boolean. This is to handle the case of lazy
            integer or boolean masked data.

        """
        # Initialise the instance.
        self._fill_value = None
        self._lazy_array = None
        self._real_array = None
        self._realised_dtype = None

        # Assign the data payload to be managed.
        self.data = data

        # Set the lazy data realised dtype, if appropriate.
        self._realised_dtype_setter(realised_dtype)

        # Set the fill-value, must be set after the realised dtype.
        if ma.isMaskedArray(data) and fill_value is None:
            self._propogate_masked_data_fill_value()
        else:
            self.fill_value = fill_value

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
        the realised_dtype, the dtype of the payload, the fill-value and the
        payload content.

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
            same_fill_value = self.fill_value == other.fill_value
            same_realised_dtype = self._realised_dtype == other._realised_dtype
            same_dtype = self.dtype == other.dtype
            if same_lazy and same_fill_value and same_realised_dtype \
                    and same_dtype:
                result = array_equal(self.core_data(), other.core_data())

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
        fmt = '{cls}({data!r}{fill_value}{dtype})'
        fill_value = ''
        dtype = ''

        if self.fill_value is not None:
            fill_value = ', fill_value={!r}'.format(self.fill_value)

        if self._realised_dtype is not None:
            dtype = ', realised_dtype={!r}'.format(self._realised_dtype)

        result = fmt.format(data=self.core_data(), cls=type(self).__name__,
                            fill_value=fill_value, dtype=dtype)

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
        state = not (not self.has_lazy_data() and
                     self._realised_dtype is not None)
        emsg = ('Unexpected real data with realised dtype, got '
                'real data and realised {!r}.')
        assert state, emsg.format(self._realised_dtype)

        state = not (self.has_lazy_data() and
                     self._lazy_array.dtype.kind != 'f' and
                     self._realised_dtype is not None)
        emsg = ('Unexpected lazy data dtype with realised dtype, got '
                'lazy data {!r} and realised {!r}.')
        assert state, emsg.format(self._lazy_array.dtype, self._realised_dtype)

    def _deepcopy(self, memo, data=None, fill_value='none',
                  realised_dtype='none'):
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

        * fill_value:
            Replacement fill-value.

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
                DataManager(self.core_data()).replace(data)
                # If the replacement data is valid, then use it but
                # without copying it.

            if isinstance(fill_value, six.string_types) and \
                    fill_value == 'none':
                fill_value = self.fill_value

            if isinstance(realised_dtype, six.string_types) and \
                    realised_dtype == 'none':
                realised_dtype = self._realised_dtype

            result = DataManager(data, fill_value=fill_value,
                                 realised_dtype=realised_dtype)
        except ValueError as error:
            emsg = 'Cannot copy {!r} - {}'
            raise ValueError(emsg.format(type(self).__name__, error))

        return result

    def _propogate_masked_data_fill_value(self):
        """
        Align the data manager fill-value with the real masked array
        fill-value.

        """
        data = self._real_array
        if ma.isMaskedArray(data):
            # Determine the default numpy fill-value.
            np_fill_value = ma.masked_array(0, dtype=data.dtype).fill_value
            if data.fill_value == np_fill_value:
                # Never store the numpy default fill-value, rather
                # represent this by clearing the data manager fill-value.
                self.fill_value = None
            else:
                # Propogate the masked array fill-value to the data manager.
                self.fill_value = data.fill_value
                # Catch the case where numpy has a fill-value (default, or
                # otherwise) that is invalid for the underlying dtype.
                if self.fill_value != data.fill_value:
                    self.fill_value = None

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
            realised_dtype = np.dtype(realised_dtype)
            if realised_dtype != self.dtype:
                if not self.has_lazy_data():
                    emsg = ('Cannot set realised dtype, no lazy data '
                            'is available.')
                    raise ValueError(emsg)
                if self._lazy_array.dtype.kind != 'f':
                    emsg = ('Cannot set realised dtype for lazy data '
                            'with {!r}.')
                    raise ValueError(emsg.format(self._lazy_array.dtype))
                if realised_dtype.kind not in 'biu':
                    emsg = ('Can only cast lazy data to an integer or boolean '
                            'dtype, got {!r}.')
                    raise ValueError(emsg.format(realised_dtype))
                self._realised_dtype = realised_dtype

                # Check the manager contract, as the managed dtype has changed.
                self._assert_axioms()

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

        if ma.isMaskedArray(self._real_array):
            # Align the numpy fill-value with the data manager fill-value.
            self._real_array.fill_value = self.fill_value

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
            if isinstance(data, ma.core.MaskedConstant):
                # Promote to a masked array so that the fill-value is
                # writeable to the data owner.
                data = ma.array(data.data, mask=data.mask, dtype=data.dtype)
            self._lazy_array = None
            self._real_array = data

        # Always reset the realised dtype, as the managed data has changed.
        self._realised_dtype = None

        # Reset the fill-value appropriately.
        if ma.isMaskedArray(data):
            # Align the data manager fill-value with the numpy fill-value.
            self._propogate_masked_data_fill_value()
        else:
            # Clear the data manager fill-value.
            self.fill_value = None

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
            result = self.core_data().dtype

        return result

    @property
    def fill_value(self):
        return self._fill_value

    @fill_value.setter
    def fill_value(self, fill_value):
        if fill_value is not None:
            # Convert the given value to the dtype of the data manager.
            fill_value = np.asarray([fill_value])[0]
            target_dtype = self.dtype
            if fill_value.dtype.kind == 'f' and target_dtype.kind in 'biu':
                # Perform rounding when converting floats to ints.
                fill_value = np.rint(fill_value)
            try:
                [fill_value] = np.asarray([fill_value], dtype=target_dtype)
            except OverflowError:
                emsg = 'Fill value of {!r} invalid for {!r}.'
                raise ValueError(emsg.format(fill_value, self.dtype))
        self._fill_value = fill_value

    @property
    def ndim(self):
        """
        The number of dimensions covered by the data being managed.

        """
        return self.core_data().ndim

    @property
    def shape(self):
        """
        The shape of the data being managed.

        """
        return self.core_data().shape

    def copy(self, data=None, fill_value='none', realised_dtype='none'):
        """
        Returns a deep copy of this :class:`~iris._data_manager.DataManager`
        instance.

        Kwargs:

        * data:
            Replace the data of the copy with this data.

        * fill_value:
            Replacement fill-value.

        * realised_dtype:
            Replace the intended dtype of the lazy data
            in the copy with this :class:`~numpy.dtype`.

        Returns:
            A copy :class:`~iris._data_manager.DataManager` instance.

        """
        memo = {}
        return self._deepcopy(memo, data=data, fill_value=fill_value,
                              realised_dtype=realised_dtype)

    def core_data(self):
        """
        If real data is being managed, then return the :class:`~numpy.ndarray`
        or :class:`numpy.ma.core.MaskedArray`. Otherwise, return the lazy
        :class:`~dask.array.core.Array`.

        Returns:
            The real or lazy data.

        """
        if self.has_lazy_data():
            result = self._lazy_array
        else:
            result = self._real_array

        return result

    def has_lazy_data(self):
        """
        Determine whether lazy data is being managed.

        Returns:
            Boolean.

        """
        return self._lazy_array is not None

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
        if self.has_lazy_data():
            result = self._lazy_array
        else:
            result = as_lazy_data(self._real_array)

        return result

    def replace(self, data, fill_value=None, realised_dtype=None):
        """
        Perform an in-place replacement of the managed data.

        Args:

        * data:
            Replace the managed data with either the :class:`~numpy.ndarray`
            or :class:`~numpy.ma.core.MaskedArray` real data, or lazy
            :class:`dask.array.core.Array`

        Kwargs:

        * fill_value:
            Replacement for the :class:`~iris._data_manager.DataManager`
            fill-value.

        * realised_dtype:
            The intended dtype of the specified lazy data.

        .. note::
            Data replacement alone will clear the intended dtype
            of the realised lazy data, and the fill-value.

        """
        # Snapshot the currently managed data.
        original_data = self.core_data()
        # Perform in-place data assignment.
        self.data = data
        try:
            self._realised_dtype_setter(realised_dtype)
            self.fill_value = fill_value
        except ValueError as error:
            # Backout the data replacement, and reinstate the cached
            # original managed data.
            self._lazy_array = self._real_array = None
            if is_lazy_data(original_data):
                self._lazy_array = original_data
            else:
                self._real_array = original_data
            raise error
