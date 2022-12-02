# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Module to ensure all calls to the netCDF4 library are thread-safe.

Intention is that no other Iris module should import the netCDF module.

"""
from functools import wraps
import threading
import typing

import netCDF4

GLOBAL_NETCDF_ACCESS_LOCK = threading.Lock()

# Doesn't need thread protection, but this allows all netCDF4 refs to be
#  replaced with thread_safe refs.
default_fillvals = netCDF4.default_fillvals


class ThreadSafeAggregator(object):
    """
    Contains a netCDF4 instance, ensuring always operates with GLOBAL_NETCDF_ACCESS_LOCK.

    Designed to 'gate keep' all operations with the instance, but allowing the
    same API as if working directly with the instance itself.

    Using an aggregator because we cannot successfully subclass or monkeypatch
    netCDF4 classes as they are only wrappers for the C-layer.
    """

    CONTAINED_CLASS = NotImplemented

    # Allows easy assertions, without difficulties with isinstance and mocking.
    THREAD_SAFE_FLAG = True

    @classmethod
    def from_existing(cls, instance):
        assert isinstance(instance, cls.CONTAINED_CLASS)
        return cls(instance)

    def __init__(self, *args, **kwargs):
        if isinstance(args[0], self.CONTAINED_CLASS):
            instance = args[0]
        else:
            with GLOBAL_NETCDF_ACCESS_LOCK:
                instance = self.CONTAINED_CLASS(*args, **kwargs)

        self.__contained = instance

    def __getattr__(self, item):
        if item == f"_{self.__class__.__name__}__contained":
            # Special behaviour when accessing the __contained instance itself.
            return object.__getattribute__(
                self, f"_{ThreadSafeAggregator.__name__}__contained"
            )
        else:
            with GLOBAL_NETCDF_ACCESS_LOCK:
                return getattr(self.__contained, item)

    def __setattr__(self, key, value):
        if key == f"_{ThreadSafeAggregator.__name__}__contained":
            # Special behaviour when accessing the __contained instance itself.
            object.__setattr__(self, key, value)
        else:
            with GLOBAL_NETCDF_ACCESS_LOCK:
                return setattr(self.__contained, key, value)

    def __getitem__(self, item):
        with GLOBAL_NETCDF_ACCESS_LOCK:
            return self.__contained.__getitem__(item)

    def __setitem__(self, key, value):
        with GLOBAL_NETCDF_ACCESS_LOCK:
            return self.__contained.__setitem__(key, value)


class DimensionContainer(ThreadSafeAggregator):
    """Accessor for a netCDF4.Dimension, always acquiring GLOBAL_NETCDF_ACCESS_LOCK."""

    CONTAINED_CLASS = netCDF4.Dimension

    @wraps(CONTAINED_CLASS.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class VariableContainer(ThreadSafeAggregator):
    """Accessor for a netCDF4.Variable, always acquiring GLOBAL_NETCDF_ACCESS_LOCK."""

    CONTAINED_CLASS = netCDF4.Variable

    @wraps(CONTAINED_CLASS.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @wraps(CONTAINED_CLASS.setncattr)
    def setncattr(self, *args, **kwargs) -> None:
        # Needed explicitly to get some mocks to work.
        with GLOBAL_NETCDF_ACCESS_LOCK:
            return self.__contained.setncattr(*args, **kwargs)

    @property
    def dimensions(self) -> typing.List[str]:
        # Needed explicitly to get some mocks to work.
        #  Only returns a list of strings so no DimensionContainer is needed.
        with GLOBAL_NETCDF_ACCESS_LOCK:
            return self.__contained.dimensions

    # All Variable API that returns Dimension(s) is wrapped to instead return
    #  DimensionContainer(s).

    @wraps(CONTAINED_CLASS.get_dims)
    def get_dims(self, *args, **kwargs) -> typing.Tuple[DimensionContainer]:
        with GLOBAL_NETCDF_ACCESS_LOCK:
            dimensions_ = self.__contained.get_dims(*args, **kwargs)
        return tuple(
            [DimensionContainer.from_existing(d) for d in dimensions_]
        )


# Docstrings that aren't covered by @wraps:
VariableContainer.dimensions.__doc__ = (
    VariableContainer.CONTAINED_CLASS.dimensions.__doc__
)


class DatasetContainer(ThreadSafeAggregator):
    """Accessor for a netCDF4.Dataset, always acquiring GLOBAL_NETCDF_ACCESS_LOCK."""

    CONTAINED_CLASS = netCDF4.Dataset

    @classmethod
    def fromcdl(cls, *args, **kwargs):
        with GLOBAL_NETCDF_ACCESS_LOCK:
            instance = cls.CONTAINED_CLASS.fromcdl(*args, **kwargs)
        return cls.from_existing(instance)

    @wraps(CONTAINED_CLASS.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # All Dataset API that returns Dimension(s) is wrapped to instead return
    #  DimensionContainer(s).

    @property
    def dimensions(self) -> typing.Dict[str, DimensionContainer]:
        with GLOBAL_NETCDF_ACCESS_LOCK:
            dimensions_ = self.__contained.dimensions
        return {
            k: DimensionContainer.from_existing(v)
            for k, v in dimensions_.items()
        }

    @wraps(CONTAINED_CLASS.createDimension)
    def createDimension(self, *args, **kwargs) -> DimensionContainer:
        with GLOBAL_NETCDF_ACCESS_LOCK:
            new_dimension = self.__contained.createDimension(*args, **kwargs)
        result = DimensionContainer.from_existing(new_dimension)
        return result

    # All Dataset API that returns Variable(s) is wrapped to instead return
    #  VariableContainer(s).

    @property
    def variables(self) -> typing.Dict[str, VariableContainer]:
        with GLOBAL_NETCDF_ACCESS_LOCK:
            variables_ = self.__contained.variables
        return {
            k: VariableContainer.from_existing(v)
            for k, v in variables_.items()
        }

    @wraps(CONTAINED_CLASS.createVariable)
    def createVariable(self, *args, **kwargs) -> VariableContainer:
        with GLOBAL_NETCDF_ACCESS_LOCK:
            new_variable = self.__contained.createVariable(*args, **kwargs)
        result = VariableContainer.from_existing(new_variable)
        return result

    @wraps(CONTAINED_CLASS.get_variables_by_attributes)
    def get_variables_by_attributes(
        self, *args, **kwargs
    ) -> typing.List[VariableContainer]:
        with GLOBAL_NETCDF_ACCESS_LOCK:
            variables_ = self.__contained.get_variables_by_attributes(
                *args, **kwargs
            )
        return [VariableContainer.from_existing(v) for v in variables_]


# Docstrings that aren't covered by @wraps:
DatasetContainer.dimensions.__doc__ = (
    DatasetContainer.CONTAINED_CLASS.dimensions.__doc__
)
DatasetContainer.variables.__doc__ = (
    DatasetContainer.CONTAINED_CLASS.variables.__doc__
)
