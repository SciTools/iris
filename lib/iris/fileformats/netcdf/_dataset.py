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

from collections.abc import Iterator, Mapping, MutableMapping
from typing import Any
import warnings

import numpy as np

from iris.fileformats.cf.dataset import CFDataset, CFDatasetVariable
import iris.warnings

from . import _bytecoding_datasets, _dask_locks, _thread_safe_nc

#: What ``netCDF4.Variable.chunking()`` answers for an unchunked variable.
#: ``None`` means the same thing, and arrives from non-version-4 files.
_CONTIGUOUS = "contiguous"

#: The member an emulating variable carries instead of file storage. See
#: https://github.com/SciTools/iris/issues/4994 "Xarray bridge".
_EMULATED_DATA_ARRAY = "_data_array"


def _bytes_if_ascii(value):
    """Return an ASCII string as bytes, and anything else unchanged.

    netCDF4 stores a bytes attribute as NC_CHAR. Coercing on the way in is
    what keeps Iris's string attributes to that type across file formats,
    rather than leaving it to netCDF4's own str handling.

    """
    if isinstance(value, str):
        try:
            return value.encode(encoding="ascii")
        except (AttributeError, UnicodeEncodeError):
            pass
    return value


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
        # Coerce on the way out only.  Caching the coerced value would make a
        # just-written attribute read back as bytes, where the same attribute
        # read from a file reads back as str.  See finding F12.
        self._target.setncattr(key, _bytes_if_ascii(value))
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
        """Return a picklable object supporting ``__setitem__``, for Dask stores.

        It carries the file path and variable name rather than the open file,
        reopening on each write, so that a worker can use it after the saver
        that created it has closed its own handle.

        """
        proxy_class: type[_thread_safe_nc.NetCDFWriteProxy]
        if isinstance(self._variable, _bytecoding_datasets.EncodedVariable):
            proxy_class = _bytecoding_datasets.EncodedNetCDFWriteProxy
        else:
            # Only reachable for a dataset opened without string encoding;
            # the saver always encodes.
            proxy_class = _thread_safe_nc.NetCDFWriteProxy
        return proxy_class(self._location, self._variable, self._write_lock)


#: The formats that make CF loading slow, and that the user can convert away
#: from with "nccopy".
_LEGACY_FORMATS = ("NETCDF3_CLASSIC", "NETCDF3_64BIT")


class NetCDFDataset(CFDataset):
    """A netCDF file, presented through the CF interface."""

    def __init__(
        self,
        location,
        mode: str = "r",
        *,
        netcdf_format=None,
        warn_legacy_format: bool = False,
    ):
        """Open a netCDF file.

        Parameters
        ----------
        location : str or :class:`pathlib.Path`
            The file's path or URL.
        mode : str, default="r"
            The netCDF4 open mode.
        netcdf_format : str, optional
            The netCDF format to create, when writing.
        warn_legacy_format : bool, default=False
            Whether to warn that a netCDF3 file would load faster if
            converted. Only loading asks for this; the loader is where the
            user can act on it.

        """
        # Set first, so that __del__ and close() are safe if opening fails.
        self._dataset: _thread_safe_nc.DatasetWrapper | None = None
        self._owned = True
        self._closed = False
        self._variables: dict[str, NetCDFDatasetVariable] | None = None
        self._attributes: _NetCDFAttributes | None = None
        self._write_lock = None

        self._location = str(location)
        self._mode = mode

        if mode == "r" and not _bytecoding_datasets.DECODE_TO_STRINGS_ON_READ:
            # The user has turned string decoding off for reads. Writing has
            # no such switch: the saver always encodes.
            dataset_class = _thread_safe_nc.DatasetWrapper
        else:
            dataset_class = _bytecoding_datasets.EncodedDataset

        # netCDF4.Dataset validates "format" against a fixed list and rejects
        # None, so omit it entirely rather than passing the default through.
        extra = {} if netcdf_format is None else {"format": netcdf_format}
        self._dataset = dataset_class(location, mode=mode, **extra)

        if warn_legacy_format and self._dataset.file_format in _LEGACY_FORMATS:
            warnings.warn(
                "Optimise CF-netCDF loading by converting data from NetCDF3 "
                'to NetCDF4 file format using the "nccopy" command.',
                category=iris.warnings.IrisLoadWarning,
            )

        # Turn off *any* automatic decoding by netCDF4 itself. Iris decodes
        # byte data on its own terms, in _bytecoding_datasets. Inert on an
        # EncodedDataset, which blocks the call; real on a DatasetWrapper.
        self._dataset.set_auto_chartostring(False)

    @classmethod
    def from_existing(cls, dataset) -> "NetCDFDataset":
        """Wrap an already-open netCDF dataset, without taking ownership of it.

        ``dataset`` may be a thread-safe wrapper, a bare
        :class:`netCDF4.Dataset`, or any object emulating one - the Xarray
        bridge passes the last of these. :meth:`close` will not release it,
        because whoever opened it is still responsible for it.

        """
        instance = cls.__new__(cls)
        instance._owned = False
        instance._closed = False
        instance._variables = None
        instance._attributes = None
        instance._write_lock = None
        # netCDF4 exposes no public attribute recording a dataset's open
        # mode, so a borrowed dataset's true mode is unobservable: "r+" is a
        # stipulation, not a reading. It is also the correct one for this
        # PR's only caller, Saver (Task 12), a write path, and it is
        # consistent with the wrapping just below - EncodedDataset is what
        # __init__ picks for every mode except plain "r".
        instance._mode = "r+"

        if not hasattr(dataset, "THREAD_SAFE_FLAG"):
            # The wrappers forbid re-wrapping, so only wrap what is not one.
            dataset = _bytecoding_datasets.EncodedDataset.from_existing(dataset)
        instance._dataset = dataset
        instance._dataset.set_auto_chartostring(False)
        instance._location = str(dataset.filepath())
        return instance

    @property
    def location(self) -> str:
        """The file's path or URL."""
        return self._location

    @property
    def mode(self) -> str:
        """The mode the file was opened in."""
        return self._mode

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has already released the file."""
        return self._closed

    @property
    def variables(self) -> Mapping:
        """The file's variables, by name."""
        if self._variables is None:
            # Only unset while __init__ or from_existing is still running.
            assert self._dataset is not None
            self._variables = {
                name: NetCDFDatasetVariable(
                    variable, self._location, write_lock=self.write_lock
                )
                for name, variable in self._dataset.variables.items()
            }
        return self._variables

    @property
    def dimensions(self) -> Mapping:
        """The file's dimension lengths, by name.

        An unlimited dimension reports the number of records written so far,
        which is what ``len()`` of a netCDF4 dimension gives and what every
        caller in Iris - all of them membership tests - needs.

        ``.size`` rather than ``len()``: :class:`~iris.fileformats.netcdf.
        _thread_safe_nc.DimensionWrapper` is a composition wrapper whose
        ``__getattr__`` forwards ordinary attribute lookups but is never
        consulted for implicit dunder-protocol calls, so ``len(dimension)``
        raises ``TypeError`` where ``dimension.size`` reaches the same value
        through a normal attribute.

        """
        # Only unset while __init__ or from_existing is still running.
        assert self._dataset is not None
        return {
            name: dimension.size for name, dimension in self._dataset.dimensions.items()
        }

    @property
    def attributes(self) -> MutableMapping:
        """The file's global attributes."""
        if self._attributes is None:
            self._attributes = _NetCDFAttributes(self._dataset)
        return self._attributes

    def create_dimension(self, name: str, size: int | None) -> None:
        """Declare a dimension of the given length, or unlimited for ``None``."""
        # Only unset while __init__ or from_existing is still running.
        assert self._dataset is not None
        self._dataset.createDimension(name, size)

    def create_variable(
        self, name: str, dtype, dimensions=(), *, fill_value=None, **encoding
    ) -> NetCDFDatasetVariable:
        """Create and return a new variable.

        ``**encoding`` reaches :meth:`netCDF4.Dataset.createVariable`
        unaltered: ``compression``, ``zlib``, ``complevel``, ``shuffle``,
        ``chunksizes``, ``least_significant_digit`` and the rest.

        """
        # Only unset while __init__ or from_existing is still running.
        assert self._dataset is not None
        variable = self._dataset.createVariable(
            name, dtype, tuple(dimensions), fill_value=fill_value, **encoding
        )
        wrapped = NetCDFDatasetVariable(
            variable, self._location, write_lock=self.write_lock
        )
        if self._variables is not None:
            # Keep an already-materialised mapping in step, rather than
            # leaving it stale for the rest of the dataset's life.
            self._variables[name] = wrapped
        return wrapped

    def sync(self) -> None:
        """Flush buffered writes to the file."""
        # Only unset while __init__ or from_existing is still running.
        assert self._dataset is not None
        self._dataset.sync()

    def finalise(self) -> None:
        """Do nothing: a netCDF file needs no completion step.

        Kept so that callers can be written once. Zarr's consolidated metadata
        is what this exists for - see finding F6.

        """

    def close(self) -> None:
        """Close the file, if this object opened it."""
        if self._owned and self._dataset is not None and not self._closed:
            self._dataset.close()
        self._closed = True

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"{self.__class__.__name__}({self.location!r}, mode={self.mode!r})"

    # netCDF-only members below.

    @property
    def dataset(self):
        """The backing thread-safe netCDF dataset wrapper."""
        return self._dataset

    @property
    def write_lock(self):
        """The lock every worker writing to this file must hold.

        One per dataset, because under the threaded scheduler
        :func:`iris.fileformats.netcdf._dask_locks.get_worker_lock` returns a
        new :class:`threading.Lock` on each call, and two such locks exclude
        nothing.

        """
        if self._write_lock is None:
            self._write_lock = _dask_locks.get_worker_lock(self._location)
        return self._write_lock
