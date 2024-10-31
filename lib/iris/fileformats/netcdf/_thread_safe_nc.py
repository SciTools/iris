# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to ensure all calls to the netCDF4 library are thread-safe.

Intention is that no other Iris module should import the netCDF4 module.

"""

from abc import ABC
from threading import Lock
import typing

import netCDF4
import numpy as np

_GLOBAL_NETCDF4_LOCK = Lock()

# Doesn't need thread protection, but this allows all netCDF4 refs to be
#  replaced with thread_safe refs.
default_fillvals = netCDF4.default_fillvals


class _ThreadSafeWrapper(ABC):
    """Contains a netCDF4 class instance, ensuring wrapping all API calls.

    Contains a netCDF4 class instance, ensuring wrapping all API calls within
    _GLOBAL_NETCDF4_LOCK.

    Designed to 'gate keep' all the instance's API calls, but allowing the
    same API as if working directly with the instance itself.

    Using a contained object instead of inheritance, as we cannot successfully
    subclass or monkeypatch netCDF4 classes, because they are only wrappers for
    the C-layer.
    """

    # Note: this is only used to create a "contained" from passed args.
    CONTAINED_CLASS = NotImplemented
    # Note: this defines how we identify/check that a contained is of the expected type
    # (in a duck-type way).
    _DUCKTYPE_CHECK_PROPERTIES: typing.List[str] = [NotImplemented]

    # Allows easy type checking, avoiding difficulties with isinstance and mocking.
    THREAD_SAFE_FLAG = True

    @classmethod
    def is_contained_type(cls, instance):
        return all(hasattr(instance, attr) for attr in cls._DUCKTYPE_CHECK_PROPERTIES)

    @classmethod
    def from_existing(cls, instance):
        """Routine to pass an existing instance to __init__, where it is contained."""
        assert cls.is_contained_type(instance)
        return cls(instance)

    def __init__(self, *args, **kwargs):
        """Contain an existing instance, or generate a new one from arguments."""
        if len(args) == 1 and self.is_contained_type(args[0]):
            # Passed a contained-type object : Wrap ourself around that.
            instance = args[0]
            # We should never find ourselves "wrapping a wrapper".
            assert not hasattr(instance, "THREAD_SAFE_FLAG")
        else:
            # Create a contained object of the intended type from passed args.
            with _GLOBAL_NETCDF4_LOCK:
                instance = self.CONTAINED_CLASS(*args, **kwargs)

        self._contained_instance = instance

    def __getattr__(self, item):
        if item == "_contained_instance":
            # Special behaviour when accessing the _contained_instance itself.
            return object.__getattribute__(self, item)
        else:
            with _GLOBAL_NETCDF4_LOCK:
                return getattr(self._contained_instance, item)

    def __setattr__(self, key, value):
        if key == "_contained_instance":
            # Special behaviour when accessing the _contained_instance itself.
            object.__setattr__(self, key, value)
        else:
            with _GLOBAL_NETCDF4_LOCK:
                return setattr(self._contained_instance, key, value)

    def __getitem__(self, item):
        with _GLOBAL_NETCDF4_LOCK:
            return self._contained_instance.__getitem__(item)

    def __setitem__(self, key, value):
        with _GLOBAL_NETCDF4_LOCK:
            return self._contained_instance.__setitem__(key, value)


class DimensionWrapper(_ThreadSafeWrapper):
    """Accessor for a netCDF4.Dimension, always acquiring _GLOBAL_NETCDF4_LOCK.

    All API calls should be identical to those for netCDF4.Dimension.
    """

    CONTAINED_CLASS = netCDF4.Dimension
    _DUCKTYPE_CHECK_PROPERTIES = ["isunlimited"]


class VariableWrapper(_ThreadSafeWrapper):
    """Accessor for a netCDF4.Variable, always acquiring _GLOBAL_NETCDF4_LOCK.

    All API calls should be identical to those for netCDF4.Variable.
    """

    CONTAINED_CLASS = netCDF4.Variable
    _DUCKTYPE_CHECK_PROPERTIES = ["dimensions", "dtype"]

    def setncattr(self, *args, **kwargs) -> None:
        """Call netCDF4.Variable.setncattr within _GLOBAL_NETCDF4_LOCK.

        Only defined explicitly in order to get some mocks to work.
        """
        with _GLOBAL_NETCDF4_LOCK:
            return self._contained_instance.setncattr(*args, **kwargs)

    @property
    def dimensions(self) -> typing.List[str]:
        """Call netCDF4.Variable.dimensions within _GLOBAL_NETCDF4_LOCK.

        Only defined explicitly in order to get some mocks to work.
        """
        with _GLOBAL_NETCDF4_LOCK:
            # Return value is a list of strings so no need for
            #  DimensionWrapper, unlike self.get_dims().
            return self._contained_instance.dimensions

    # All Variable API that returns Dimension(s) is wrapped to instead return
    #  DimensionWrapper(s).

    def get_dims(self, *args, **kwargs) -> typing.Tuple[DimensionWrapper]:
        """Call netCDF4.Variable.get_dims() within _GLOBAL_NETCDF4_LOCK.

        Call netCDF4.Variable.get_dims() within _GLOBAL_NETCDF4_LOCK,
        returning DimensionWrappers.  The original returned netCDF4.Dimensions
        are simply replaced with their respective DimensionWrappers, ensuring
        that downstream calls are also performed within _GLOBAL_NETCDF4_LOCK.
        """
        with _GLOBAL_NETCDF4_LOCK:
            dimensions_ = list(self._contained_instance.get_dims(*args, **kwargs))
        return tuple([DimensionWrapper.from_existing(d) for d in dimensions_])


class GroupWrapper(_ThreadSafeWrapper):
    """Accessor for a netCDF4.Group, always acquiring _GLOBAL_NETCDF4_LOCK.

    All API calls should be identical to those for netCDF4.Group.
    """

    CONTAINED_CLASS = netCDF4.Group
    # Note: will also accept a whole Dataset object, but that is OK.
    _DUCKTYPE_CHECK_PROPERTIES = ["createVariable"]

    # All Group API that returns Dimension(s) is wrapped to instead return
    #  DimensionWrapper(s).

    @property
    def dimensions(self) -> typing.Dict[str, DimensionWrapper]:
        """Call dimensions of netCDF4.Group/Dataset within _GLOBAL_NETCDF4_LOCK.

        Calls dimensions of netCDF4.Group/Dataset within _GLOBAL_NETCDF4_LOCK,
        returning DimensionWrappers.  The original returned netCDF4.Dimensions
        are simply replaced with their respective DimensionWrappers, ensuring
        that downstream calls are also performed within _GLOBAL_NETCDF4_LOCK.
        """
        with _GLOBAL_NETCDF4_LOCK:
            dimensions_ = self._contained_instance.dimensions
        return {k: DimensionWrapper.from_existing(v) for k, v in dimensions_.items()}

    def createDimension(self, *args, **kwargs) -> DimensionWrapper:
        """Call createDimension() from netCDF4.Group/Dataset within _GLOBAL_NETCDF4_LOCK.

        Call createDimension() from netCDF4.Group/Dataset within
        _GLOBAL_NETCDF4_LOCK, returning DimensionWrapper. The original returned
        netCDF4.Dimension is simply replaced with its respective
        DimensionWrapper, ensuring that downstream calls are also performed
        within _GLOBAL_NETCDF4_LOCK.
        """
        with _GLOBAL_NETCDF4_LOCK:
            new_dimension = self._contained_instance.createDimension(*args, **kwargs)
        return DimensionWrapper.from_existing(new_dimension)

    # All Group API that returns Variable(s) is wrapped to instead return
    #  VariableWrapper(s).

    @property
    def variables(self) -> typing.Dict[str, VariableWrapper]:
        """Call variables of netCDF4.Group/Dataset within _GLOBAL_NETCDF4_LOCK.

        Calls variables of netCDF4.Group/Dataset within _GLOBAL_NETCDF4_LOCK,
        returning VariableWrappers.  The original returned netCDF4.Variables
        are simply replaced with their respective VariableWrappers, ensuring
        that downstream calls are also performed within _GLOBAL_NETCDF4_LOCK.
        """
        with _GLOBAL_NETCDF4_LOCK:
            variables_ = self._contained_instance.variables
        return {k: VariableWrapper.from_existing(v) for k, v in variables_.items()}

    def createVariable(self, *args, **kwargs) -> VariableWrapper:
        """Call createVariable() from netCDF4.Group/Dataset within _GLOBAL_NETCDF4_LOCK.

        Call createVariable() from netCDF4.Group/Dataset within
        _GLOBAL_NETCDF4_LOCK, returning VariableWrapper.  The original
        returned netCDF4.Variable is simply replaced with its respective
        VariableWrapper, ensuring that downstream calls are also performed
        within _GLOBAL_NETCDF4_LOCK.
        """
        with _GLOBAL_NETCDF4_LOCK:
            new_variable = self._contained_instance.createVariable(*args, **kwargs)
        return VariableWrapper.from_existing(new_variable)

    def get_variables_by_attributes(
        self, *args, **kwargs
    ) -> typing.List[VariableWrapper]:
        """Call get_variables_by_attributes() from netCDF4.Group/Dataset.

        Call get_variables_by_attributes() from netCDF4.Group/Dataset
        within_GLOBAL_NETCDF4_LOCK, returning VariableWrappers.

        The original returned netCDF4.Variables are simply replaced with their
        respective VariableWrappers, ensuring that downstream calls are
        also performed within _GLOBAL_NETCDF4_LOCK.
        """
        with _GLOBAL_NETCDF4_LOCK:
            variables_ = list(
                self._contained_instance.get_variables_by_attributes(*args, **kwargs)
            )
        return [VariableWrapper.from_existing(v) for v in variables_]

    # All Group API that returns Group(s) is wrapped to instead return
    #  GroupWrapper(s).

    @property
    def groups(self):
        """Call groups of netCDF4.Group/Dataset within _GLOBAL_NETCDF4_LOCK.

        Calls groups of netCDF4.Group/Dataset within _GLOBAL_NETCDF4_LOCK,
        returning GroupWrappers.

        The original returned netCDF4.Groups are simply replaced with their
        respective GroupWrappers, ensuring that downstream calls are
        also performed within _GLOBAL_NETCDF4_LOCK.
        """
        with _GLOBAL_NETCDF4_LOCK:
            groups_ = self._contained_instance.groups
        return {k: GroupWrapper.from_existing(v) for k, v in groups_.items()}

    @property
    def parent(self):
        """Call parent of netCDF4.Group/Dataset within _GLOBAL_NETCDF4_LOCK.

        Calls parent of netCDF4.Group/Dataset within _GLOBAL_NETCDF4_LOCK,
        returning a GroupWrapper.

        The original returned netCDF4.Group is simply replaced with its
        respective GroupWrapper, ensuring that downstream calls are
        also performed within _GLOBAL_NETCDF4_LOCK.

        """
        with _GLOBAL_NETCDF4_LOCK:
            parent_ = self._contained_instance.parent
        return GroupWrapper.from_existing(parent_)

    def createGroup(self, *args, **kwargs):
        """Call createGroup() from netCDF4.Group/Dataset.

        Call createGroup() from netCDF4.Group/Dataset within
        _GLOBAL_NETCDF4_LOCK, returning GroupWrapper.  The original returned
        netCDF4.Group is simply replaced with its respective GroupWrapper,
        ensuring that downstream calls are also performed within
        _GLOBAL_NETCDF4_LOCK.
        """
        with _GLOBAL_NETCDF4_LOCK:
            new_group = self._contained_instance.createGroup(*args, **kwargs)
        return GroupWrapper.from_existing(new_group)


class DatasetWrapper(GroupWrapper):
    """Accessor for a netCDF4.Dataset, always acquiring _GLOBAL_NETCDF4_LOCK.

    All API calls should be identical to those for netCDF4.Dataset.
    """

    CONTAINED_CLASS = netCDF4.Dataset
    # Note: 'close' exists on Dataset but not Group (though a rather weak distinction).
    _DUCKTYPE_CHECK_PROPERTIES = ["createVariable", "close"]

    @classmethod
    def fromcdl(cls, *args, **kwargs):
        """Call netCDF4.Dataset.fromcdl() within _GLOBAL_NETCDF4_LOCK.

        Call netCDF4.Dataset.fromcdl() within _GLOBAL_NETCDF4_LOCK,
        returning a DatasetWrapper.  The original returned netCDF4.Dataset is
        simply replaced with its respective DatasetWrapper, ensuring that
        downstream calls are also performed within _GLOBAL_NETCDF4_LOCK.
        """
        with _GLOBAL_NETCDF4_LOCK:
            instance = cls.CONTAINED_CLASS.fromcdl(*args, **kwargs)
        return cls.from_existing(instance)


class NetCDFDataProxy:
    """A reference to the data payload of a single NetCDF file variable."""

    __slots__ = ("shape", "dtype", "path", "variable_name", "fill_value")

    def __init__(self, shape, dtype, path, variable_name, fill_value):
        self.shape = shape
        self.dtype = dtype
        self.path = path
        self.variable_name = variable_name
        self.fill_value = fill_value

    @property
    def ndim(self):
        # noqa: D102
        return len(self.shape)

    @property
    def dask_meta(self):
        return np.ma.array(np.empty((0,) * self.ndim, dtype=self.dtype), mask=True)

    def __getitem__(self, keys):
        # Using a DatasetWrapper causes problems with invalid ID's and the
        # netCDF4 library, presumably because __getitem__ gets called so many
        # times by Dask. Use _GLOBAL_NETCDF4_LOCK directly instead.
        with _GLOBAL_NETCDF4_LOCK:
            dataset = netCDF4.Dataset(self.path)
            try:
                variable = dataset.variables[self.variable_name]
                # Get the NetCDF variable data and slice.
                var = variable[keys]
            finally:
                dataset.close()
        return np.asanyarray(var)

    def __repr__(self):
        fmt = (
            "<{self.__class__.__name__} shape={self.shape}"
            " dtype={self.dtype!r} path={self.path!r}"
            " variable_name={self.variable_name!r}>"
        )
        return fmt.format(self=self)

    def __getstate__(self):
        return {attr: getattr(self, attr) for attr in self.__slots__}

    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)


class NetCDFWriteProxy:
    """An object mimicking the data access of a netCDF4.Variable.

    The "opposite" of a NetCDFDataProxy : An object mimicking the data access
    of a netCDF4.Variable, but where the data is to be ***written to***.

    It encapsulates the netcdf file and variable which are actually to be
    written to.  This opens the file each time, to enable writing the data
    chunk, then closes it.
    TODO: could be improved with a caching scheme, but this just about works.
    """

    def __init__(self, filepath, cf_var, file_write_lock):
        self.path = filepath
        self.varname = cf_var.name
        self.lock = file_write_lock

    def __setitem__(self, keys, array_data):
        # Write to the variable.
        # First acquire a file-specific lock for all workers writing to this file.
        self.lock.acquire()
        # Open the file for writing + write to the specific file variable.
        # Exactly as above, in NetCDFDataProxy : a DatasetWrapper causes problems with
        # invalid ID's and the netCDF4 library, for so-far unknown reasons.
        # Instead, use _GLOBAL_NETCDF4_LOCK, and netCDF4 _directly_.
        with _GLOBAL_NETCDF4_LOCK:
            dataset = None
            try:
                dataset = netCDF4.Dataset(self.path, "r+")
                var = dataset.variables[self.varname]
                var[keys] = array_data
            finally:
                try:
                    if dataset:
                        dataset.close()
                finally:
                    # *ALWAYS* let go !
                    self.lock.release()

    def __repr__(self):
        return f"<{self.__class__.__name__} path={self.path!r} var={self.varname!r}>"
