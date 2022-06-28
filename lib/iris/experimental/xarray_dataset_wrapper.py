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
from typing import Optional

import netCDF4 as nc
import numpy as np
import xarray
import xarray as xr


class XrMimic:
    """
    An netcdf object "mimic" wrapped around an xarray object, which will be
    either a dim, var or dataset.

    These (mostly) contain an underlying xarray object, and all potentially
    have a name + group (though dataset name is unused).
    N.B. name is provided separately, as xr types do not "know" their own names
    - e.g. an xr.Variable has no 'name' property.

    We also support object equality checks.

    NOTE: a DimensionMimic, uniquely, does *NOT* in fact contain an xarray
    object, so its self._xr == None.   See  DimensionMimic docstring.

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

    NOTE: a DimensionMimic does *NOT* contain an xarray object representing the
    dimension, because xarray doesn't have such objects.
    So, in xarray, you can't rename or modify an existing Dataset dimension.
    But you can re-order, add, and remove ones that no variable uses.

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

    #
    # writing
    #
    def setncattr(self, attr_name, value):
        if isinstance(value, bytes):
            value = value.decode()
        self._xr.attrs[attr_name] = value

    # NOTE: not currently supporting ".my_attribute = value" type access.
    # def __setattr__(self, attr_name, value):
    #     self.setncattr(attr_name, value)


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

    #
    # writing
    #
    def __setitem__(self, keys, data):
        self._xr[keys] = data


class DatasetMimic(Nc4AttrsMimic):
    """
    An object mimicking an netCDF4.Dataset, wrapping an xarray.Dataset.

    """

    def __init__(self, xrds: Optional[xarray.Dataset] = None):
        if xrds is None:
            # Initialise empty dataset if not passed in.
            xrds = xr.Dataset()
        super().__init__(xrds)

        # Capture original filepath, if known.
        self._sourcepath = self._xr.encoding.get("source", "")

        # Keep track of variables which were renamed on creation to prevent
        # them being made into coords (which are not writable).
        self._output_renames = {}

        # Capture existing dimensions in input
        unlim_dims = self._xr.encoding.get("unlimited_dims", set())
        self.dimensions = OrderedDict()
        for name, len in self._xr.dims.items():
            is_unlim = name in unlim_dims
            dim = DimensionMimic(name, len, isunlimited=is_unlim)
            self.dimensions[name] = dim

        # Capture existing variables in input
        self.variables = OrderedDict()
        for name, var in self._xr.variables.items():
            var_mimic = VariableMimic(var, name=name)
            self.variables[name] = var_mimic

    def filepath(self) -> str:
        return self._sourcepath

    def to_xarray_dataset(self) -> xr.Dataset:
        """Get an xarray.Dataset representing the simulated netCDF4.Dataset."""
        ds = self._xr
        # Drop the 'extra' coordinate variables which were required to make
        # indexing constructions work.
        ds = ds.drop_vars(self.dimensions.keys())
        # Rename original dimension coords back to their dimension name.
        ds = ds.rename_vars(self._output_renames)
        # Apply "nofill" encoding to all the output vars which did do not
        # actually provide a '_FillVAlue' attribute.
        # TODO: check that a provided fill-value behaves as expected
        for varname, var in ds.variables.items():
            # if 'missing_value' in var.attrs:
            #     print(varname)
            #     del var.attrs['missing_value']
            if "_FillValue" not in var.attrs:
                var.encoding["_FillValue"] = None
        return ds

    def groups(self):
        # Xarray does not support groups :-(
        return None

    def sync(self):
        pass

    def close(self):
        pass

    @staticmethod
    def _dimcoord_adjusted_name(dimname):
        return f"_{dimname}_XRDS_RENAMED_"

    #
    # modify/write support
    #
    def createDimension(
        self, dimname, size=None, actual_length=0
    ) -> DimensionMimic:
        """
        Simulate netCDF4 call.

        N.B. the extra 'actual_length' keyword can be used in conjunction with
        size=0, to create an unlimited dimension of known 'current length'.

        """
        # NOTE: this does not work in-place, but forces us to replace the
        # original dataset.  Therefore caller can't use a ref to the original.
        # This *could* also mean that DimensionMimics don't work, but in fact
        # it is okay since xarray doesn't use dimension objects, and netCDF4
        # anyway requires us to create all the dims *first*.
        # TODO: check that 'unlimited' works -- suspect that present code can't
        #  cope with setting the 'current length' ?
        self._xr = self._xr.expand_dims({dimname: size}, -1)
        size = size or 0
        is_unlim = size == 0
        actual_length = actual_length or size
        if is_unlim:
            unlim_dims = self._xr.encoding.setdefault(
                "unlimited_dimensions", set()
            )
            unlim_dims.add(dimname)
        dim = DimensionMimic(dimname, actual_length, is_unlim)
        self.dimensions[dimname] = dim
        if actual_length > 0:
            # NOTE: for now, we are adding an extra index variable on each
            # dimension, since this avoids much problems with variables being
            # automatically converted to IndexVariables.
            # These extra coord variables do *NOT* appear in self.variables,
            # and are absent from the dataset produced by 'to_xarray_dataset'.
            data = np.arange(actual_length, dtype=int)
            self._xr[dimname] = data
        return dim

    # Expected default controls in createVariable call,
    # from iris.fileformats.netcdf.Saver
    _netcdf_saver_defaults = {
        "zlib": False,
        "complevel": 4,
        "shuffle": True,
        "fletcher32": False,
        "contiguous": False,
        "chunksizes": None,
        "endian": "native",
        "least_significant_digit": None,
        "packing": None,
    }

    def createVariable(
        self, varname, datatype, dimensions=(), fill_value=None, **kwargs
    ) -> VariableMimic:
        # TODO: kwargs should probably translate into 'encoding' on ds or vars
        # FOR NOW: simply check we have no "active" kwargs requesting
        # non-default operation.  Unfortunately, that involves some
        # detailed knowledge of the netCDF4.createVariable interface.
        for kwarg, val in kwargs.items():
            assert kwarg in self._netcdf_saver_defaults
            assert val == self._netcdf_saver_defaults[kwarg]

        datatype = np.dtype(datatype)
        shape = tuple(self._xr.dims[dimname] for dimname in dimensions)

        # Note: initially create with all-missing data.  This can subsequently
        # be assigned different values, and even support partial writes.
        # TODO: would really like to support Dask arrays here.
        if fill_value is not None:
            attrs = {"_FillValue": fill_value}
            use_fill = fill_value
        else:
            attrs = {}
            dt_code = f"{datatype.kind}{datatype.itemsize}"
            use_fill = nc.default_fillvals[dt_code]
        data = np.full(shape, fill_value=use_fill, dtype=datatype)

        xr_var = xr.Variable(dims=dimensions, data=data, attrs=attrs)
        if varname in self._xr.dims:
            # We need to avoid creating vars as coords, for which we currently
            # use a nasty trick :  Insert with a modified name, and rename back
            # on output (see 'to_xarray_dataset').
            # TODO: see if xarray provides a cleaner way to get what we want.
            alt_varname = f"XDRS_RENAMED_{varname}_"
            self._output_renames[alt_varname] = varname
            varname = alt_varname

        # Install the var, and immediately re-fetch it, since the internal
        # object is *not* generally the same as the one we put in.
        self._xr[varname] = xr_var
        xr_var = self._xr.variables[varname]
        # Create a mimic for interfacing to the xarray.Variable.
        var_mimic = VariableMimic(xr_var)
        self.variables[varname] = var_mimic
        return var_mimic


def fake_nc4python_dataset(xr_group: Optional[xr.Dataset] = None):
    """
    Make a wrapper around an xarray Dataset which emulates a
    :class:`netCDF4.Dataset'.

    The resulting :class:`DatasetMimic` supports essential properties of a
    read-mode :class:`netCDF4.Dataset', enabling an arbitrary netcdf data
    structure in memory to be "read" as if it were a file
    (i.e. without writing it to disk).
    It likewise supports write operations, which translates netCDF4 writes
    into xarray operations on the internal dataset.

    Only a limited netCDF4 API is currently supported : enough to allow Iris to
    read and write xarray datasets in place of netcdf files.

    In addition to the netCDF4 read API, a version of the contents as a viable
    xarray.Dataset can be obtained at any point, by calling
    :meth:`DatasetMimic.to_xarray_dataset`.

    Parameters
    ----------
    xr_group : xarray.Dataset, optional
        If given, return a DatasetMimic wrapped around this data.
        If absent, return an *empty* (but writeable) DatasetMimic.

    Returns
    -------
        dataset : DatasetMimic

    """
    return DatasetMimic(xr_group)
