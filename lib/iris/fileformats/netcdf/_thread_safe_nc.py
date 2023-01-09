# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Module to ensure all calls to the netCDF4 library are thread-safe.

Intention is that no other Iris module should import the netCDF module.

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


class _ThreadSafeAggregator(ABC):
    """
    Contains a netCDF4 class instance, ensuring wrapping all API calls within _GLOBAL_NETCDF4_LOCK.

    Designed to 'gate keep' all the instance's API calls, but allowing the
    same API as if working directly with the instance itself.

    Using an aggregator because we cannot successfully subclass or monkeypatch
    netCDF4 classes, as they are only wrappers for the C-layer.
    """

    CONTAINED_CLASS = NotImplemented

    # Allows easy assertions, without difficulties with isinstance and mocking.
    THREAD_SAFE_FLAG = True

    @classmethod
    def _from_existing(cls, instance):
        """Pass an existing instance to __init__, where it is contained."""
        assert isinstance(instance, cls.CONTAINED_CLASS)
        return cls(instance)

    def __init__(self, *args, **kwargs):
        """Contain an existing instance, or generate a new one from arguments."""
        if isinstance(args[0], self.CONTAINED_CLASS):
            instance = args[0]
        else:
            with _GLOBAL_NETCDF4_LOCK:
                instance = self.CONTAINED_CLASS(*args, **kwargs)

        self.__contained = instance

    def __getattr__(self, item):
        if item[-11:] == "__contained":
            # Special behaviour when accessing the __contained instance itself.
            return ABC.__getattribute__(
                self, f"{_ThreadSafeAggregator.__name__}__contained"
            )
        else:
            with _GLOBAL_NETCDF4_LOCK:
                return getattr(self.__contained, item)

    def __setattr__(self, key, value):
        if key[-11:] == "__contained":
            # Special behaviour when accessing the __contained instance itself.
            ABC.__setattr__(self, key, value)
        else:
            with _GLOBAL_NETCDF4_LOCK:
                return setattr(self.__contained, key, value)

    def __getitem__(self, item):
        with _GLOBAL_NETCDF4_LOCK:
            return self.__contained.__getitem__(item)

    def __setitem__(self, key, value):
        with _GLOBAL_NETCDF4_LOCK:
            return self.__contained.__setitem__(key, value)


class DimensionContainer(_ThreadSafeAggregator):
    """
    Accessor for a netCDF4.Dimension, always acquiring _GLOBAL_NETCDF4_LOCK.

    All API calls should be identical to those for netCDF4.Dimension.
    """

    CONTAINED_CLASS = netCDF4.Dimension


class VariableContainer(_ThreadSafeAggregator):
    """
    Accessor for a netCDF4.Variable, always acquiring _GLOBAL_NETCDF4_LOCK.

    All API calls should be identical to those for netCDF4.Variable.
    """

    CONTAINED_CLASS = netCDF4.Variable

    def setncattr(self, *args, **kwargs) -> None:
        """
        Calls netCDF4.Variable.setncattr within _GLOBAL_NETCDF4_LOCK.

        Only defined explicitly in order to get some mocks to work.
        """
        with _GLOBAL_NETCDF4_LOCK:
            return self.__contained.setncattr(*args, **kwargs)

    @property
    def dimensions(self) -> typing.List[str]:
        """
        Calls netCDF4.Variable.dimensions within _GLOBAL_NETCDF4_LOCK.

        Only defined explicitly in order to get some mocks to work.
        """
        with _GLOBAL_NETCDF4_LOCK:
            # Return value is a list of strings so no need for
            #  DimensionContainer, unlike self.get_dims().
            return self.__contained.dimensions

    # All Variable API that returns Dimension(s) is wrapped to instead return
    #  DimensionContainer(s).

    def get_dims(self, *args, **kwargs) -> typing.Tuple[DimensionContainer]:
        """
        Calls netCDF4.Variable.get_dims() within _GLOBAL_NETCDF4_LOCK, returning DimensionContainers.

        The original returned netCDF4.Dimensions are simply replaced with their
        respective DimensionContainers, ensuring that downstream calls are
        also performed within _GLOBAL_NETCDF4_LOCK.
        """
        with _GLOBAL_NETCDF4_LOCK:
            dimensions_ = self.__contained.get_dims(*args, **kwargs)
        return tuple(
            [DimensionContainer._from_existing(d) for d in dimensions_]
        )


class GroupContainer(_ThreadSafeAggregator):
    """
    Accessor for a netCDF4.Group, always acquiring _GLOBAL_NETCDF4_LOCK.

    All API calls should be identical to those for netCDF4.Group.
    """

    CONTAINED_CLASS = netCDF4.Group

    # All Group API that returns Dimension(s) is wrapped to instead return
    #  DimensionContainer(s).

    @property
    def dimensions(self) -> typing.Dict[str, DimensionContainer]:
        """
        Calls dimensions of netCDF4.Group/Dataset within _GLOBAL_NETCDF4_LOCK, returning DimensionContainers.

        The original returned netCDF4.Dimensions are simply replaced with their
        respective DimensionContainers, ensuring that downstream calls are
        also performed within _GLOBAL_NETCDF4_LOCK.
        """
        with _GLOBAL_NETCDF4_LOCK:
            dimensions_ = self.__contained.dimensions
        return {
            k: DimensionContainer._from_existing(v)
            for k, v in dimensions_.items()
        }

    def createDimension(self, *args, **kwargs) -> DimensionContainer:
        """
        Calls createDimension() from netCDF4.Group/Dataset within _GLOBAL_NETCDF4_LOCK, returning DimensionContainer.

        The original returned netCDF4.Dimension is simply replaced with its
        respective DimensionContainer, ensuring that downstream calls are
        also performed within _GLOBAL_NETCDF4_LOCK.
        """
        with _GLOBAL_NETCDF4_LOCK:
            new_dimension = self.__contained.createDimension(*args, **kwargs)
        return DimensionContainer._from_existing(new_dimension)

    # All Group API that returns Variable(s) is wrapped to instead return
    #  VariableContainer(s).

    @property
    def variables(self) -> typing.Dict[str, VariableContainer]:
        """
        Calls variables of netCDF4.Group/Dataset within _GLOBAL_NETCDF4_LOCK, returning VariableContainers.

        The original returned netCDF4.Variables are simply replaced with their
        respective VariableContainers, ensuring that downstream calls are
        also performed within _GLOBAL_NETCDF4_LOCK.
        """
        with _GLOBAL_NETCDF4_LOCK:
            variables_ = self.__contained.variables
        return {
            k: VariableContainer._from_existing(v)
            for k, v in variables_.items()
        }

    def createVariable(self, *args, **kwargs) -> VariableContainer:
        """
        Calls createVariable() from netCDF4.Group/Dataset within _GLOBAL_NETCDF4_LOCK, returning VariableContainer.

        The original returned netCDF4.Variable is simply replaced with its
        respective VariableContainer, ensuring that downstream calls are
        also performed within _GLOBAL_NETCDF4_LOCK.
        """
        with _GLOBAL_NETCDF4_LOCK:
            new_variable = self.__contained.createVariable(*args, **kwargs)
        return VariableContainer._from_existing(new_variable)

    def get_variables_by_attributes(
        self, *args, **kwargs
    ) -> typing.List[VariableContainer]:
        """
        Calls get_variables_by_attributes() from netCDF4.Group/Dataset within _GLOBAL_NETCDF4_LOCK, returning VariableContainers.

        The original returned netCDF4.Variables are simply replaced with their
        respective VariableContainers, ensuring that downstream calls are
        also performed within _GLOBAL_NETCDF4_LOCK.
        """
        with _GLOBAL_NETCDF4_LOCK:
            variables_ = self.__contained.get_variables_by_attributes(
                *args, **kwargs
            )
        return [VariableContainer._from_existing(v) for v in variables_]

    # All Group API that returns Group(s) is wrapped to instead return
    #  GroupContainer(s).

    @property
    def groups(self):
        """
        Calls groups of netCDF4.Group/Dataset within _GLOBAL_NETCDF4_LOCK, returning GroupContainers.

        The original returned netCDF4.Groups are simply replaced with their
        respective GroupContainers, ensuring that downstream calls are
        also performed within _GLOBAL_NETCDF4_LOCK.
        """
        with _GLOBAL_NETCDF4_LOCK:
            groups_ = self.__contained.groups
        return {
            k: GroupContainer._from_existing(v) for k, v in groups_.items()
        }

    @property
    def parent(self):
        """
        Calls parent of netCDF4.Group/Dataset within _GLOBAL_NETCDF4_LOCK, returning a GroupContainer.

        The original returned netCDF4.Group is simply replaced with its
        respective GrpupContainer, ensuring that downstream calls are
        also performed within _GLOBAL_NETCDF4_LOCK.
        """
        with _GLOBAL_NETCDF4_LOCK:
            parent_ = self.__contained.parent
        return GroupContainer._from_existing(parent_)

    def createGroup(self, *args, **kwargs):
        """
        Calls createGroup() from netCDF4.Group/Dataset within _GLOBAL_NETCDF4_LOCK, returning GroupContainer.

        The original returned netCDF4.Group is simply replaced with its
        respective GroupContainer, ensuring that downstream calls are
        also performed within _GLOBAL_NETCDF4_LOCK.
        """
        with _GLOBAL_NETCDF4_LOCK:
            new_group = self.__contained.createGroup(*args, **kwargs)
        return GroupContainer._from_existing(new_group)


class DatasetContainer(GroupContainer):
    """
    Accessor for a netCDF4.Dataset, always acquiring _GLOBAL_NETCDF4_LOCK.

    All API calls should be identical to those for netCDF4.Dataset.
    """

    CONTAINED_CLASS = netCDF4.Dataset

    @classmethod
    def fromcdl(cls, *args, **kwargs):
        """
        Calls netCDF4.Dataset.fromcdl() within _GLOBAL_NETCDF4_LOCK, returning a DatasetContainer.

        The original returned netCDF4.Dataset is simply replaced with its
        respective DatasetContainer, ensuring that downstream calls are
        also performed within _GLOBAL_NETCDF4_LOCK.
        """
        with _GLOBAL_NETCDF4_LOCK:
            instance = cls.CONTAINED_CLASS.fromcdl(*args, **kwargs)
        return cls._from_existing(instance)


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
        return len(self.shape)

    def __getitem__(self, keys):
        # Using a DatasetContainer causes problems with invalid ID's and the
        #  netCDF4 library, presumably because __getitem__ gets called so many
        #  times by Dask. Use _GLOBAL_NETCDF4_LOCK directly instead.
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
