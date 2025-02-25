# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Management of common state and behaviour for cube and coordinate data."""

import copy

import numpy as np
import numpy.ma as ma

from iris._lazy_data import as_concrete_data, as_lazy_data, is_lazy_data
import iris.exceptions
import iris.warnings


class DataManager:
    """Provides a well defined API for management of real or lazy data."""

    def __init__(self, data, shape=None):
        """Create a data manager for the specified data.

        Parameters
        ----------
        data : ArrayLike, optional
            The :class:`~numpy.ndarray` or :class:`~numpy.ma.core.MaskedArray`
            real data, or :class:`~dask.array.core.Array` lazy data to be
            managed. If a value of ``None`` is given, the data manager will be
            considered dataless.

        shape : tuple, optional
            A tuple, representing the shape of the data manager. This can only
            be used in the case of ``data=None``, and will render the data manager
            dataless.

        """
        if (shape is None) and (data is None):
            msg = 'one of "shape" or "data" should be provided; both are None'
            raise ValueError(msg)
        elif (shape is not None) and (data is not None):
            msg = '"shape" should only be provided if "data" is None'
            raise ValueError(msg)

        # Initialise the instance.
        self._shape = shape
        self._lazy_array = None
        self._real_array = None

        # Assign the data payload to be managed.
        self.data = data

    def __copy__(self):
        """Forbid :class:`~iris._data_manager.DataManager` instance shallow-copy support."""
        name = type(self).__name__
        emsg = (
            "Shallow-copy of {!r} is not permitted. Use "
            "copy.deepcopy() or {}.copy() instead."
        )
        raise copy.Error(emsg.format(name, name))

    def __deepcopy__(self, memo):
        """Allow :class:`~iris._data_manager.DataManager` instance deepcopy support.

        Parameters
        ----------
        memo : :func:`copy`
            :func:`copy` memo dictionary.

        """
        return self._deepcopy(memo)

    def __eq__(self, other):
        """Perform :class:`~iris._data_manager.DataManager` instance equality.

        Perform :class:`~iris._data_manager.DataManager` instance equality.
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
            if self.is_dataless() and other.is_dataless():
                result = self.shape == other.shape
            else:
                result = False
                same_lazy = self.has_lazy_data() == other.has_lazy_data()
                same_dtype = self.dtype == other.dtype
                if same_lazy and same_dtype:
                    result = array_equal(self.core_data(), other.core_data())
        return result

    def __ne__(self, other):
        """Perform :class:`~iris._data_manager.DataManager` instance inequality.

        Perform :class:`~iris._data_manager.DataManager` instance inequality.
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
        """Return an string representation of the instance."""
        fmt = "{cls}({data!r})"
        result = fmt.format(data=self.core_data(), cls=type(self).__name__)
        if self.is_dataless():
            result = f"{result}, shape={self.shape}"

        return result

    def _assert_axioms(self):
        """Definition of the manager state, that should never be violated."""
        # Ensure there is a valid data state.
        is_lazy = self._lazy_array is not None
        is_real = self._real_array is not None

        if not (is_lazy ^ is_real):
            if is_lazy and is_real:
                msg = "Unexpected data state, got both lazy and real data."
                raise ValueError(msg)
            elif self._shape is None:
                msg = "Unexpected data state, got no lazy or real data, and no shape."
                raise ValueError(msg)

    def _deepcopy(self, memo, data=None):
        """Perform a deepcopy of the :class:`~iris._data_manager.DataManager` instance.

        Parameters
        ----------
        memo : :func:`copy`
            :func:`copy` memo dictionary.
        data : ArrayLike, optional
            Replacement data to substitute the currently managed
            data with.

        Returns
        -------
        :class:`~iris._data_manager.DataManager` instance.

        """
        shape = None
        try:
            if data is None:
                # Copy the managed data.
                if self.has_lazy_data():
                    data = copy.deepcopy(self._lazy_array, memo)
                elif self._real_array is not None:
                    data = self._real_array.copy()
                else:
                    shape = self._shape
            elif type(data) is str and data == iris.DATALESS:
                shape = self.shape
                data = None
            else:
                # Check that the replacement data is valid relative to
                # the currently managed data.
                dm_check = DataManager(self.core_data())
                dm_check.data = data
                # If the replacement data is valid, then use it but
                # without copying it.
            result = DataManager(data=data, shape=shape)
        except ValueError as error:
            emsg = "Cannot copy {!r} - {}"
            raise ValueError(emsg.format(type(self).__name__, error))
        return result

    @property
    def data(self):
        """Return the real data. Any lazy data being managed will be realised.

        Returns
        -------
        :class:`~numpy.ndarray` or :class:`numpy.ma.core.MaskedArray` or ``None``.

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
        """Replace the currently managed data with the specified data.

        Replace the currently managed data with the specified data, which must
        be of an equivalent shape.

        Note that, the only shape promotion permitted is for 0-dimensional
        scalar data to be replaced with a single item 1-dimensional data.

        Parameters
        ----------
        data :
            The :class:`~numpy.ndarray` or :class:`~numpy.ma.core.MaskedArray`
            real data, or :class:`~dask.array.core.Array` lazy data to be
            managed. If data is ``None``, the current shape will be maintained.

        """
        if data is None:
            self._shape = self.shape
            self._lazy_array = None
            self._real_array = None

        # Ensure we have numpy-like data.
        else:
            if not (hasattr(data, "shape") and hasattr(data, "dtype")):
                data = np.asanyarray(data)

            # Determine whether the class already has a defined shape,
            # as this method is called from __init__.
            has_shape = self._shape is not None
            if has_shape and self.shape != data.shape:
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
            # sets ``self._shape`` if it is None, or if it is being converted from
            # ( ) to (1, )
            if not has_shape or (self._shape == () and data.shape == (1,)):
                self._shape = self.core_data().shape

        # Check the manager contract, as the managed data has changed.
        self._assert_axioms()

    @property
    def dtype(self):
        """The dtype of the realised lazy data or the dtype of the real data."""
        return self.core_data().dtype if not self.is_dataless() else None

    @property
    def ndim(self):
        """The number of dimensions covered by the data being managed."""
        return len(self.shape)

    @property
    def shape(self):
        """The shape of the data being managed."""
        return self._shape

    def is_dataless(self) -> bool:
        """Determine whether the cube has no data.

        Returns
        -------
        bool

        """
        return self.core_data() is None

    def copy(self, data=None):
        """Return a deep copy of this :class:`~iris._data_manager.DataManager` instance.

        Parameters
        ----------
        data : optional
            Replace the data of the copy with this data.

        Returns
        -------
        A copy :class:`~iris._data_manager.DataManager` instance.

        """
        memo = {}
        return self._deepcopy(memo, data=data)

    def core_data(self):
        """Provide real data or lazy data.

        If real data is being managed, then return the :class:`~numpy.ndarray`
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
        if self.is_dataless():
            result = None
        elif self.has_lazy_data():
            result = self._lazy_array
        else:
            result = as_lazy_data(self._real_array)

        return result
