# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Module to ensure all calls to the netCDF4 library are thread-safe.

Intention is that no other Iris module should import the netCDF module.

"""
import threading

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

    # Allows easy assertions, without difficulties with isinstance and mocking.
    THREAD_SAFE_FLAG = True

    def __init__(self, contained):
        self.__contained = contained

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


class VariableContainer(ThreadSafeAggregator):
    """Accessor for a netCDF4.Variable, always acquiring GLOBAL_NETCDF_ACCESS_LOCK."""

    @property
    def dimensions(self):
        # Needed explicitly to get some mocks to work.
        with GLOBAL_NETCDF_ACCESS_LOCK:
            return self.__contained.dimensions

    def setncattr(self, *args, **kwargs):
        # Needed explicitly to get some mocks to work.
        with GLOBAL_NETCDF_ACCESS_LOCK:
            return self.__contained.setncattr(*args, **kwargs)


class DatasetContainer(ThreadSafeAggregator):
    """Accessor for a netCDF4.Dataset, always acquiring GLOBAL_NETCDF_ACCESS_LOCK."""

    def __init__(self, *args, **kwargs):
        with GLOBAL_NETCDF_ACCESS_LOCK:
            ds = netCDF4.Dataset(*args, **kwargs)
        super().__init__(ds)

    # All Dataset API that returns Variable(s) is wrapper to instead return
    #  VariableContainer(s).

    @property
    def variables(self):
        with GLOBAL_NETCDF_ACCESS_LOCK:
            variables_ = self.__contained.variables
        return {k: VariableContainer(v) for k, v in variables_.items()}

    def createVariable(self, *args, **kwargs):
        with GLOBAL_NETCDF_ACCESS_LOCK:
            new_variable = self.__contained.createVariable(*args, **kwargs)
        result = VariableContainer(new_variable)
        return result

    def get_variables_by_attributes(self, *args, **kwargs):
        with GLOBAL_NETCDF_ACCESS_LOCK:
            variables_ = self.__contained.get_variables_by_attributes(
                *args, **kwargs
            )
        return [VariableContainer(v) for v in variables_]
