# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""The netCDF implementation of the :mod:`iris.fileformats.cf.dataset` interface.

This is the only module that knows both the CF interface and the netCDF4 API.
Everything netCDF-specific that the CF layer used to reach through a
:class:`~iris.fileformats.cf.CFVariable` - ``ncattrs``, ``getncattr``,
``setncattr``, ``chunking()``, ``file_format``, ``createVariable``,
``createDimension``, ``filepath()``, ``isopen()`` - either has a named member
on the interface or lives here as a netCDF-only member.

See section 4.5 of ``docs/superpowers/specs/2026-09-21-zarr-io-design.md``.

"""

from collections.abc import Iterator, MutableMapping
from typing import Any

import numpy as np

from iris.fileformats.cf.dataset import CFDataset, CFDatasetVariable

from . import _bytecoding_datasets, _thread_safe_nc

#: What ``netCDF4.Variable.chunking()`` answers for an unchunked variable.
#: ``None`` means the same thing, and arrives from non-version-4 files.
_CONTIGUOUS = "contiguous"

#: The member an emulating variable carries instead of file storage. See
#: https://github.com/SciTools/iris/issues/4994 "Xarray bridge".
_EMULATED_DATA_ARRAY = "_data_array"


class _NetCDFAttributes(MutableMapping):
    """A netCDF object's attributes as a mapping: read once, written through.

    Values are read once, at construction, because the interface promises a
    mapping whose ``keys`` and ``items`` cost nothing - attribute tracking asks
    "which of these went unread?" on every variable of every loaded file, and
    a lazily-fetching mapping would make that question expensive.

    """

    def __init__(self, target):
        """Materialise the attributes of ``target``, a netCDF variable or dataset."""
        self._target = target
        self._values: dict[str, Any] = {}
        for name in target.ncattrs():
            try:
                value = target.getncattr(name)
            except AttributeError:
                # ncattrs() can list a name that getncattr then refuses. The
                # netCDF4 library does this for some malformed files, and
                # cf/_reader.py's _getncattr tolerated it with this default.
                value = ""
            self._values[name] = value

    def __getitem__(self, key: str) -> Any:
        """Return ``key``'s value."""
        return self._values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set ``key``'s value, here and in the netCDF object."""
        self._target.setncattr(key, value)
        self._values[key] = value

    def __delitem__(self, key: str) -> None:
        """Remove ``key``, here and from the netCDF object."""
        self._target.delncattr(key)
        del self._values[key]

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over the attribute names, in file order."""
        return iter(self._values)

    def __len__(self) -> int:
        """Return the number of attributes."""
        return len(self._values)

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"{self.__class__.__name__}({self._values!r})"


class NetCDFDatasetVariable(CFDatasetVariable):
    """One variable of a netCDF file, presented through the CF interface."""

    def __init__(self, variable, location: str, *, write_lock=None):
        """Wrap ``variable``, a thread-safe netCDF variable wrapper.

        Parameters
        ----------
        variable : :class:`~iris.fileformats.netcdf._thread_safe_nc.VariableWrapper`
            The wrapped netCDF variable. May be an
            :class:`~iris.fileformats.netcdf._bytecoding_datasets.EncodedVariable`,
            whose shape, dimensions and dtype differ from the file's own for
            character data; those are passed through, not reached around.
        location : str
            The path or URL of the dataset this variable belongs to.
        write_lock : optional
            The lock shared by every variable of one dataset, used by
            :meth:`write_handle`. Supplied by :class:`NetCDFDataset`.

        """
        self._variable = variable
        self._location = location
        self._write_lock = write_lock
        self._attributes = _NetCDFAttributes(variable)

    @property
    def name(self) -> str:
        """The variable's name within its dataset."""
        return self._variable.name

    @property
    def location(self) -> str:
        """The path or URL of the dataset holding this variable."""
        return self._location

    @property
    def dimensions(self) -> tuple:
        """The names of the dimensions this variable spans, in order."""
        return tuple(self._variable.dimensions)

    @property
    def shape(self) -> tuple:
        """The variable's shape."""
        return tuple(self._variable.shape)

    @property
    def dtype(self):
        """The variable's stored data type, or ``str`` for a VLEN string type."""
        return self._variable.dtype

    @property
    def size(self) -> int:
        """The total number of elements in the variable."""
        return self._variable.size

    @property
    def fill_value(self) -> Any:
        """The variable's ``_FillValue`` attribute, or ``None`` if it has none."""
        return self._attributes.get("_FillValue")

    @property
    def chunking(self) -> tuple | None:
        """The variable's storage chunk shape, or ``None`` when unchunked.

        netCDF answers ``None`` for a non-version-4 file and the string
        ``"contiguous"`` for an unchunked version-4 variable. Both mean the
        same thing to a caller choosing a Dask chunking, so both become
        ``None``.

        """
        chunks = self._variable.chunking()
        if chunks is None or chunks == _CONTIGUOUS:
            return None
        return tuple(chunks)

    @property
    def attributes(self) -> MutableMapping:
        """The variable's CF attributes."""
        return self._attributes

    def __getitem__(self, keys) -> np.ndarray:
        """Return the indexed portion of the variable's data."""
        return self._variable[keys]

    def __setitem__(self, keys, values) -> None:
        """Write ``values`` into the indexed portion of the variable."""
        self._variable[keys] = values

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"{self.__class__.__name__}({self.name!r}, {self.location!r})"

    # netCDF-only members below: named here rather than reached for through
    # getattr, so that a Zarr caller fails to import them rather than failing
    # at run time with an AttributeError from somewhere unrelated.

    @property
    def variable(self):
        """The backing thread-safe netCDF variable wrapper."""
        return self._variable

    @property
    def is_variable_length(self) -> bool:
        """Whether this is a netCDF variable-length (VLEN) type.

        Such a variable's total size cannot be known without reading it - see
        https://github.com/Unidata/netcdf-c/issues/1893 - so the loader has to
        guess whether it is worth making lazy.

        """
        datatype = getattr(self._variable, "datatype", None)
        return isinstance(datatype, _thread_safe_nc.VLType)

    @property
    def is_emulated(self) -> bool:
        """Whether an emulating object supplies this variable's data directly.

        The Xarray bridge, https://github.com/SciTools/iris/issues/4994: the
        "file" is an emulator and its variables carry their own arrays.

        """
        return hasattr(self._variable, _EMULATED_DATA_ARRAY)

    @property
    def emulated_data_array(self):
        """The array an emulating variable carries instead of file storage."""
        if not self.is_emulated:
            # A plain netCDF4 variable's __getattr__ looks up ncattrs for an
            # unknown name and raises its own, unrelated message; raise the
            # one callers actually need to recognise.
            raise AttributeError(_EMULATED_DATA_ARRAY)
        return getattr(self._variable, _EMULATED_DATA_ARRAY)

    @emulated_data_array.setter
    def emulated_data_array(self, value) -> None:
        setattr(self._variable, _EMULATED_DATA_ARRAY, value)

    def deprecated_netcdf_member(self, name: str) -> Any:
        """Return a member of the backing netCDF variable wrapper."""
        return getattr(self._variable, name)

    def write_handle(self) -> Any:
        """Return a picklable object supporting ``__setitem__``, for Dask stores."""
        raise NotImplementedError
