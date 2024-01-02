# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Management of common state and behaviour for cube and coordinate data.

"""

import copy

import numpy as np
import numpy.ma as ma

from iris._lazy_data import as_concrete_data, as_lazy_data, is_lazy_data


class DataManager:
    """Provides a well defined API for management of real or lazy data."""

    def __init__(self, data):
        """Create a data manager for the specified data.

        Parameters
        ----------
        data :
            The :class:`~numpy.ndarray` or :class:`~numpy.ma.core.MaskedArray`
            real data, or :class:`~dask.array.core.Array` lazy data to be
            managed.

        """
        # Initialise the instance.
        self._lazy_array = None
        self._real_array = None

        # Assign the data payload to be managed.
        self.data = data

        # Enforce the manager contract.
        self._assert_axioms()

    def __copy__(self):
        """Forbid :class:`~iris._data_manager.DataManager` instance
        shallow-copy support.

        """
        name = type(self).__name__
        emsg = (
            "Shallow-copy of {!r} is not permitted. Use "
            "copy.deepcopy() or {}.copy() instead."
        )
        raise copy.Error(emsg.format(name, name))

    def __deepcopy__(self, memo):
        """Allow :class:`~iris._data_manager.DataManager` instance
        deepcopy support.

        Parameters
        ----------
        memo : :func:`copy`
            :func:`copy` memo dictionary.

        """
        return self._deepcopy(memo)

    def __eq__(self, other):
        """Perform :class:`~iris._data_manager.DataManager` instance equality.
        Note that, this is explicitly not a lazy operation and will load any
        lazy payload to determine the equality result.

        Comparison is strict with regards to lazy or real managed payload,
        the realised_dtype, the dtype of the payload, the fill-value and the
        payload content.

        Parameters
        ----------
        other : :class:`~iris._data_manager.DataManager`
            The :class:`~iris._data_manager.DataManager` instance to
            compare with.

        Returns
        -------
        bool

        """
        from iris.util import array_equal

        result = NotImplemented

        if isinstance(other, type(self)):
            result = False
            same_lazy = self.has_lazy_data() == other.has_lazy_data()
            same_dtype = self.dtype == other.dtype
            if same_lazy and same_dtype:
                result = array_equal(self.core_data(), other.core_data())

        return result

    def __ne__(self, other):
        """Perform :class:`~iris._data_manager.DataManager` instance inequality.
        Note that, this is explicitly not a lazy operation and will load any
        lazy payload to determine the inequality result.

        Parameters
        ----------
        other : :class:`~iris._data_manager.DataManager`
            The :class:`~iris._data_manager.DataManager` instance to
            compare with.

        Returns
        -------
        bool

        """
        result = self.__eq__(other)

        if result is not NotImplemented:
            result = not result

        return result

    def __repr__(self):
        """Returns an string representation of the instance."""
        fmt = "{cls}({data!r})"
        result = fmt.format(data=self.core_data(), cls=type(self).__name__)

        return result

    def _assert_axioms(self):
        """Definition of the manager state, that should never be violated."""
        # Ensure there is a valid data state.
        is_lazy = self._lazy_array is not None
        is_real = self._real_array is not None
        emsg = "Unexpected data state, got {}lazy and {}real data."
        state = is_lazy ^ is_real
        assert state, emsg.format("" if is_lazy else "no ", "" if is_real else "no ")

    def _deepcopy(self, memo, data=None):
        """Perform a deepcopy of the :class:`~iris._data_manager.DataManager`
        instance.

        Parameters
        ----------
        memo : :func:`copy`
            :func:`copy` memo dictionary.
        data : optional
            Replacement data to substitute the currently managed
            data with.

        Returns
        -------
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
                dm_check = DataManager(self.core_data())
                dm_check.data = data
                # If the replacement data is valid, then use it but
                # without copying it.
            result = DataManager(data)
        except ValueError as error:
            emsg = "Cannot copy {!r} - {}"
            raise ValueError(emsg.format(type(self).__name__, error))

        return result

    @property
    def data(self):
        """Returns the real data. Any lazy data being managed will be realised.

        Returns
        -------
        :class:`~numpy.ndarray` or :class:`numpy.ma.core.MaskedArray`.

        """
        if self.has_lazy_data():
            try:
                # Realise the lazy data.
                result = as_concrete_data(self._lazy_array)
                # Assign the realised result.
                self._real_array = result
                # Reset the lazy data and the realised dtype.
                self._lazy_array = None
            except MemoryError:
                emsg = (
                    "Failed to realise the lazy data as there was not "
                    "enough memory available.\n"
                    "The data shape would have been {!r} with {!r}.\n "
                    "Consider freeing up variables or indexing the data "
                    "before trying again."
                )
                raise MemoryError(emsg.format(self.shape, self.dtype))

        # Check the manager contract, as the managed data has changed.
        self._assert_axioms()

        return self._real_array

    @data.setter
    def data(self, data):
        """Replaces the currently managed data with the specified data, which must
        be of an equivalent shape.

        Note that, the only shape promotion permitted is for 0-dimensional
        scalar data to be replaced with a single item 1-dimensional data.

        Parameters
        ----------
        data :
            The :class:`~numpy.ndarray` or :class:`~numpy.ma.core.MaskedArray`
            real data, or :class:`~dask.array.core.Array` lazy data to be
            managed.

        """
        # Ensure we have numpy-like data.
        if not (hasattr(data, "shape") and hasattr(data, "dtype")):
            data = np.asanyarray(data)

        # Determine whether the class instance has been created,
        # as this method is called from within the __init__.
        init_done = self._lazy_array is not None or self._real_array is not None

        if init_done and self.shape != data.shape:
            # The _ONLY_ data reshape permitted is converting a 0-dimensional
            # array i.e. self.shape == () into a 1-dimensional array of length
            # one i.e. data.shape == (1,)
            if self.shape or data.shape != (1,):
                emsg = "Require data with shape {!r}, got {!r}."
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

        # Check the manager contract, as the managed data has changed.
        self._assert_axioms()

    @property
    def dtype(self):
        """The dtype of the realised lazy data or the dtype of the real data."""
        return self.core_data().dtype

    @property
    def ndim(self):
        """The number of dimensions covered by the data being managed."""
        return self.core_data().ndim

    @property
    def shape(self):
        """The shape of the data being managed."""
        return self.core_data().shape

    def copy(self, data=None):
        """Returns a deep copy of this :class:`~iris._data_manager.DataManager`
        instance.

        Parameters
        ----------
        data :
            Replace the data of the copy with this data.

        Returns
        -------
        A copy :class:`~iris._data_manager.DataManager` instance.

        """
        memo = {}
        return self._deepcopy(memo, data=data)

    def core_data(self):
        """If real data is being managed, then return the :class:`~numpy.ndarray`
        or :class:`numpy.ma.core.MaskedArray`. Otherwise, return the lazy
        :class:`~dask.array.core.Array`.

        Returns
        -------
        The real or lazy data.

        """
        if self.has_lazy_data():
            result = self._lazy_array
        else:
            result = self._real_array

        return result

    def has_lazy_data(self):
        """Determine whether lazy data is being managed.

        Returns
        -------
        bool

        """
        return self._lazy_array is not None

    def lazy_data(self):
        """Return the lazy representation of the managed data.

        If only real data is being managed, then return a lazy
        representation of that real data.

        Returns
        -------
        :class:`~dask.array.core.Array`

        .. note::
            This method will never realise any lazy data.

        """
        if self.has_lazy_data():
            result = self._lazy_array
        else:
            result = as_lazy_data(self._real_array)

        return result
