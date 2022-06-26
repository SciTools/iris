# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
A wrapper for an xarray.Dataset that simulates a netCDF4.Dataset.
This enables code to read/write xarray data as if it were a netcdf file.

NOTE: readonly, for now.
TODO: add modify/save functions later.

NOTE: this code is effectively independent of Iris, and does not really belong.
However, this is a convenient place to test, for now.

"""
from collections import OrderedDict


class XrMimic:
    """
    An netcdf dataset mimic wrapped around an xarray object,
    dim, var or dataset.

    These all contain an underlying xarray object type, and all potentially
    have a name + group (though dataset name is unused).
    N.B. name is provided separately, as not all xr types have it.

    We also support object equality checks.

    """

    def __init__(self, xr, name=None, group=None):
        """
        Create a mimic object wrapping a :class:`nco.Ncobj` component.
        Note: not all the underlying objects have a name, so provide that
        separately.

        """
        self._xr = xr
        self._name = name
        self._group = group

    @property
    def name(self):
        return self._name

    def group(self):
        return self._group

    def __eq__(self, other):
        return self._xr == other._xr

    def __ne__(self, other):
        return not self == other


class DimensionMimic(XrMimic):
    """
    A Dimension object mimic wrapper.

    Dimension additional properties: length, unlimited

    """

    def __init__(self, name, len, isunlimited=False, group=None):
        # Note that there *is* no underlying xarray object.
        # So we make up something, to support equality checks.
        id_placeholder = (name, len, isunlimited)
        super().__init__(xr=id_placeholder, name=name, group=group)
        self._len = len  # A private version, for now, in case needs change.
        self._unlimited = isunlimited

    @property
    def size(self):
        return 0 if self.isunlimited() else self.len

    def __len__(self):
        return self._len

    def isunlimited(self):
        return self._unlimited


class Nc4AttrsMimic(XrMimic):
    """
    A class mixin for a Mimic with attribute access.

    I.E. shared by variables and datasets.

    """

    def ncattrs(self):
        return self._xr.attrs.keys()  # Probably do *not* need/expect a list ?

    def getncattr(self, attr_name):
        if attr_name in self._xr.attrs:
            result = self._xr.attrs[attr_name]
        else:
            raise AttributeError()
        return result

    def __getattr__(self, attr_name):
        return self.getncattr(attr_name)


class VariableMimic(Nc4AttrsMimic):
    """
    A Variable object mimic wrapper.

    Variable additional properties:
        dimensions, dtype, data (+ attributes, parent-group)
        shape, size, ndim

    """

    @property
    def dtype(self):
        return self._xr.dtype

    def chunking(self):
        return None

    @property
    def datatype(self):
        return self.dtype

    @property
    def dimensions(self):
        return self._xr.dims

    def __getitem__(self, keys):
        if self.ndim == 0:
            return self._xr.data
        else:
            return self._xr[keys].data

    @property
    def shape(self):
        return self._xr.shape

    @property
    def ndim(self):
        return self._xr.ndim

    @property
    def size(self):
        return self._xr.size


class DatasetMimic(Nc4AttrsMimic):
    """
    A Group object mimic wrapper.

    Group properties:
        name, dimensions, variables, (sub)groups (+ attributes, parent-group)

    """

    def __init__(self, xr):
        super().__init__(xr)

        self._sourcepath = self._xr.encoding.get("source", "")

        unlim_dims = self._xr.encoding["unlimited_dims"]
        self.dimensions = OrderedDict(
            [
                (name, DimensionMimic(name, len, name in unlim_dims))
                for name, len in self._xr.dims.items()
            ]
        )

        self.variables = OrderedDict(
            [
                (name, VariableMimic(var))
                for name, var in self._xr.variables.items()
            ]
        )

    def filepath(self):
        return self._sourcepath

    def groups(self):
        return None

    def close(self):
        # ?should we not be doing "something" here ??
        pass


def fake_nc4python_dataset(xr_group):
    """
    Make a wrapper around an :class:`ncobj.Group` object to emulate a
    :class:`netCDF4.Dataset'.

    The resulting :class:`GroupMimic` supports the essential properties of a
    read-mode :class:`netCDF4.Dataset', enabling an arbitrary netcdf data
    structure in memory to be "read" as if it were a file
    (i.e. without writing it to disk).

    In particular, variable data access is delegated to the original,
    underlying :class:`ncobj.Group` object :  This provides deferred, sectional
    data access on request, in the usual way, avoiding the need to read in all
    the variable data.

    """
    return DatasetMimic(xr_group)
