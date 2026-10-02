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

Reading and writing share one class, and the read path must not touch
:attr:`NetCDFDataset.write_lock`: making the lock raises for Dask schedulers
that only the saver cares about, so it is made on first use rather than at
construction. :meth:`NetCDFDataset.from_existing` serves both paths and
cannot tell them apart, so it stipulates an open mode rather than observing
one, and always wraps for byte encoding.

See section 4.5 of ``docs/superpowers/specs/2026-09-21-zarr-io-design.md``.

"""

from collections.abc import Iterator, Mapping, MutableMapping
from typing import Any
import warnings

import numpy as np

from iris._deprecation import warn_deprecated
from iris._lazy_data import as_lazy_data
from iris.fileformats.cf.dataset import CFDataset, CFDatasetVariable
import iris.warnings

from . import _bytecoding_datasets, _dask_locks, _thread_safe_nc

#: What ``netCDF4.Variable.chunking()`` answers for an unchunked variable.
#: ``None`` means the same thing, and arrives from non-version-4 files.
_CONTIGUOUS = "contiguous"

#: The member an ncdata emulating variable carries instead of file storage.
_EMULATED_DATA_ARRAY = "_data_array"

# A stab in the dark at the mean length of the "ragged dimension" for netCDF
# "variable length arrays" (`NetCDF.VLType` type). Total array size is unknown
# until the variable is read in. Making this number bigger makes it more likely
# an array will be loaded lazily.
_MEAN_VL_ARRAY_LEN = 10


def _bytes_if_ascii(value):
    """Return an ASCII string as bytes, and anything else unchanged.

    netCDF4 stores a bytes value as NC_CHAR and a str value as NC_STRING.
    Iris writes its string attributes as NC_CHAR, so they are encoded here
    before being set.

    """
    if isinstance(value, str):
        try:
            return value.encode(encoding="ascii")
        except (AttributeError, UnicodeEncodeError):
            pass
    return value


class _NetCDFAttributes(MutableMapping):
    """A netCDF object's attributes as a mapping: read once, written through."""

    def __init__(self, target):
        """Materialise the attributes of ``target``, a netCDF variable or dataset."""
        self._target = target
        self._values: dict[str, Any] = {}
        for name in target.ncattrs():
            try:
                value = target.getncattr(name)
            except AttributeError:
                # For some malformed files, netCDF4 lists an attribute name
                # that it then refuses to return.
                value = ""
            self._values[name] = value

    def __getitem__(self, key: str) -> Any:
        """Return ``key``'s value."""
        return self._values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set ``key``'s value, here and in the netCDF object."""
        # Cache the original, not the coerced value: a just-written attribute
        # must read back as str, exactly as one read from a file does.
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

    def __init__(self, variable, location: str, *, write_lock_factory=None):
        """Wrap ``variable``, a thread-safe netCDF variable wrapper.

        Parameters
        ----------
        variable : :class:`~iris.fileformats.netcdf._thread_safe_nc.VariableWrapper`
            The wrapped netCDF variable. For character data this may be an
            :class:`~iris.fileformats.netcdf._bytecoding_datasets.EncodedVariable`,
            which reports a different shape, dimensions and dtype from the
            file's own. This class reports whatever the wrapper reports.
        location : str
            The path or URL of the dataset this variable belongs to.
        write_lock_factory : callable, optional
            Returns the lock shared by every variable of one dataset. Called
            by :meth:`write_handle`. A callable rather than a lock, so that a
            read never makes one.

        """
        self._variable = variable
        self._location = location
        self._write_lock_factory = write_lock_factory
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
        """The variable's storage chunk shape, or ``None`` when unchunked."""
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

    # netCDF-only members below.

    @property
    def variable(self):
        """The backing thread-safe netCDF variable wrapper."""
        return self._variable

    @property
    def unencoded_variable(self):
        """The backing variable, from beneath any byte-encoding wrapper.

        An :class:`~iris.fileformats.netcdf._bytecoding_datasets.EncodedVariable`
        hides the on-disk ``dtype`` and character dimension; callers needing
        those want this rather than :attr:`variable`. An unwrapped variable is
        returned unchanged.

        """
        if isinstance(self._variable, _bytecoding_datasets.EncodedVariable):
            return self._variable._contained_instance
        return self._variable

    @property
    def is_variable_length(self) -> bool:
        """Whether this is a netCDF variable-length (VLEN) type.

        Such a variable's total size cannot be known without reading it.

        """
        datatype = getattr(self._variable, "datatype", None)
        return isinstance(datatype, _thread_safe_nc.VLType)

    @property
    def is_emulated(self) -> bool:
        """Whether an emulating object supplies this variable's data directly.

        ncdata, which bridges Iris and Xarray, passes a netCDF4 emulator
        whose variables carry their own arrays instead of file storage.

        """
        return hasattr(self._variable, _EMULATED_DATA_ARRAY)

    @property
    def emulated_data_array(self):
        """The array an emulating variable carries instead of file storage."""
        if not self.is_emulated:
            raise AttributeError(_EMULATED_DATA_ARRAY)
        return getattr(self._variable, _EMULATED_DATA_ARRAY)

    @emulated_data_array.setter
    def emulated_data_array(self, value) -> None:
        if not self.is_emulated:
            raise AttributeError(_EMULATED_DATA_ARRAY)
        setattr(self._variable, _EMULATED_DATA_ARRAY, value)

    def deprecated_netcdf_member(self, name: str) -> Any:
        """Return a member of the backing netCDF variable wrapper, with a warning."""
        # Fetch before warning: if getattr raises, no warning is issued. A
        # failed hasattr() probe has not used the deprecated member.
        value = getattr(self._variable, name)
        warn_deprecated(
            f"Reaching netCDF variable member {name!r} through a CFVariable is "
            "deprecated and will be removed in a future release. Use the CF "
            "dataset interface instead - iris.fileformats.cf.dataset - or, for "
            "a netCDF-only need, cf_var.cf_data.variable."
        )
        return value

    def write_handle(self) -> Any:
        """Return a picklable object supporting ``__setitem__``, for Dask stores.

        It carries the file path and variable name rather than the open file,
        and always encodes string data, whatever wrapper this variable arrived
        in.

        """
        write_lock = None
        if self._write_lock_factory is not None:
            write_lock = self._write_lock_factory()
        return _bytecoding_datasets.EncodedNetCDFWriteProxy(
            self._location, self._variable, write_lock
        )

    def read_data(self, chunking_policy):
        """Return this variable's data, lazily where a lazy array is worth it.

        Parameters
        ----------
        chunking_policy : callable
            Called with no arguments, returning the ``(chunks, dims_fixed)``
            pair to build a lazy array with. It is called only once this method
            has decided the result will be lazy: under
            :meth:`~iris.fileformats.cf.loader.ChunkControl.from_file` the
            policy raises for a variable the store has not chunked, and a
            variable small enough to read whole must not raise.

        Returns
        -------
        numpy.ndarray or dask.array.Array
            Real data for a small variable, lazy data otherwise.

        """
        # Deferred, and the module rather than its members: iris.fileformats.cf
        # reaches this module through iris.fileformats.cf._reader, so a
        # module-level import would be a cycle; and the tests patch
        # _LAZYVAR_MIN_BYTES on the module object.
        from iris.fileformats.cf import loader

        if self.is_emulated:
            # The variable is not an actual netCDF4 file variable, but an
            # emulating object with an attached data array (either numpy or
            # dask), which can be returned immediately as-is. This is the hook
            # ncdata (https://github.com/SciTools/ncdata) uses to translate data
            # to and from other packages' netCDF data containers.
            # See https://github.com/SciTools/iris/issues/4994.
            result = self.emulated_data_array
            if result.dtype.kind == "S":
                # We must also perform any byte-to-string decoding since, in
                #  ncdata, the emulating objects don't do this, and also don't
                #  support a 'set_auto_chartostring(True)'.
                #  Therefore, do here what an EncodedVariable.__getitem__ would
                #  do : ..
                # .. get details based on the file (type 'char') variable  ..
                encoder = _bytecoding_datasets.VariableEncoder.from_var(
                    self.unencoded_variable
                )
                # .. convert byte array to strings.
                result = encoder.decode_bytes_to_stringarray(result)
            return result

        # Determine size of data; however can't do this for variable length (VLEN)
        # netCDF arrays as the size of the array can only be known by reading the
        # data; see https://github.com/Unidata/netcdf-c/issues/1893.
        # Note: "Variable length" netCDF types have a datatype of `nc.VLType`.
        if self.is_variable_length:
            msg = (
                f"NetCDF variable `{self.name}` is a variable length type of kind "
                f"{self.dtype} "
                "thus the total data size cannot be known in advance. This may affect "
                "the lazy loading of the data."
            )
            warnings.warn(msg, category=iris.warnings.IrisLoadWarning)

            # Give user the chance to pass a hint of the average variable length array
            # size via the chunk control context manager. This allows for better
            # decisions to be made on whether the data should be lazy-loaded or not.
            mean_vl_array_len = _MEAN_VL_ARRAY_LEN
            chunk_control = loader.CHUNK_CONTROL
            if chunk_control.mode is not chunk_control.Modes.AS_DASK:
                if chunks := chunk_control.var_dim_chunksizes.get(self.name):
                    if vl_chunk_hint := chunks.get("_vl_hint"):
                        mean_vl_array_len = vl_chunk_hint

            # Special handling for strings (`str` type) as these don't have an
            # itemsize attribute; assume 4 bytes which is sufficient for unicode
            # character storage
            itemsize = 4 if self.dtype is str else self.dtype.itemsize

            # For `VLType` self.size will just return the known dimension size.
            total_bytes = self.size * mean_vl_array_len * itemsize
        else:
            # Normal NCVariable type:
            total_bytes = self.size * self.dtype.itemsize

        if total_bytes < loader._LAZYVAR_MIN_BYTES:
            # Don't make a lazy array, as it will cost more memory AND more time
            # to access.
            result = self[:]

            # Special handling of masked scalar value; this will be returned as
            # an `np.ma.masked` instance which will lose the original dtype.
            # Workaround for this it return a 1-element masked array of the
            # correct dtype. Note: this is not an issue for masked arrays,
            # only masked scalar values.
            if result is np.ma.masked:
                result = np.ma.masked_all(1, dtype=self.dtype)
            return result

        # Get lazy chunked data out of a cf variable.
        # Creates Dask wrappers around data arrays for any cube components which
        # can have lazy values, e.g. Cube, Coord, CellMeasure, AuxiliaryVariable.
        dtype = loader._get_actual_dtype(self)

        # Make a data-proxy that mimics array access and can fetch from the file.
        # Note: Special handling needed for "variable length string" types which
        # return a dtype of `str`, rather than a numpy type; use `S1` in this case.
        if getattr(self.dtype, "kind", None) == "U":
            # Special handling for "string variables".
            fill_value = ""
        else:
            fill_dtype = "S1" if self.dtype is str else self.dtype.str[1:]
            fill_value = self.attributes.get(
                "_FillValue", _thread_safe_nc.default_fillvals[fill_dtype]
            )

        # Switch type of proxy, based on type of variable.
        # It is done this way, instead of using an instance variable, because the
        #  limited nature of the wrappers makes a stateful choice awkward,
        #  e.g. especially, "variable.group()" is *not* the parent DatasetWrapper.
        if isinstance(self.variable, _bytecoding_datasets.EncodedVariable):
            proxy_class = _bytecoding_datasets.EncodedNetCDFDataProxy
        else:
            proxy_class = _thread_safe_nc.NetCDFDataProxy

        proxy = proxy_class(self.variable, dtype, self.location, fill_value)

        chunks, dims_fixed = chunking_policy()
        return as_lazy_data(
            proxy,
            meta=proxy.dask_meta,
            chunks=chunks,
            dims_fixed=dims_fixed,
            cache_key=repr(proxy),
        )


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

        if warn_legacy_format:
            self._warn_if_legacy_format()

        # Turn off *any* automatic decoding by netCDF4 itself. Iris decodes
        # byte data on its own terms, in _bytecoding_datasets. Inert on an
        # EncodedDataset, which blocks the call; real on a DatasetWrapper.
        self._dataset.set_auto_chartostring(False)

    def _warn_if_legacy_format(self) -> None:
        """Warn that this file would load faster in netCDF4 format, if it would."""
        assert self._dataset is not None
        if self._dataset.file_format in _LEGACY_FORMATS:
            warnings.warn(
                "Optimise CF-netCDF loading by converting data from NetCDF3 "
                'to NetCDF4 file format using the "nccopy" command.',
                category=iris.warnings.IrisLoadWarning,
            )

    @classmethod
    def from_existing(
        cls, dataset, *, warn_legacy_format: bool = False
    ) -> "NetCDFDataset":
        """Wrap an already-open netCDF dataset, without taking ownership of it.

        ``dataset`` may be a thread-safe wrapper, a bare
        :class:`netCDF4.Dataset`, or an emulator of one, as ncdata passes.
        :meth:`close` will not release it, because whoever opened it is still
        responsible for it.

        Parameters
        ----------
        warn_legacy_format : bool, default=False
            Whether to warn that a netCDF3 file would load faster if
            converted. Only loading asks for this.

        """
        instance = cls.__new__(cls)
        instance._owned = False
        instance._closed = False
        instance._variables = None
        instance._attributes = None
        instance._write_lock = None
        # netCDF4 records no open mode, so this is stipulated, not observed.
        # Only __repr__ reads it.
        instance._mode = "r+"

        if not hasattr(dataset, "THREAD_SAFE_FLAG"):
            # The wrappers forbid re-wrapping, so only wrap what is not one.
            # Always encoded: DECODE_TO_STRINGS_ON_READ is not consulted here,
            # because from_existing cannot tell a read from a write.
            dataset = _bytecoding_datasets.EncodedDataset.from_existing(dataset)
        instance._dataset = dataset
        instance._dataset.set_auto_chartostring(False)
        instance._location = str(dataset.filepath())

        if warn_legacy_format:
            instance._warn_if_legacy_format()

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
        """Whether the file has been released, by this object or by its owner.

        A borrowed dataset can be closed by its owner without this object
        knowing, so the backing object is asked too. An emulator need not
        implement ``isopen()``; one that does not is taken to be open.

        """
        if self._closed:
            return True
        isopen = getattr(self._dataset, "isopen", None)
        if isopen is None:
            return False
        return not isopen()

    def _materialised_variables(self) -> dict[str, "NetCDFDatasetVariable"]:
        """Return the wrapper mapping, building it from the file if need be."""
        if self._variables is None:
            assert self._dataset is not None
            self._variables = {
                name: NetCDFDatasetVariable(
                    variable,
                    self._location,
                    write_lock_factory=self._write_lock_factory,
                )
                for name, variable in self._dataset.variables.items()
            }
        return self._variables

    @property
    def variables(self) -> Mapping:
        """The file's variables, by name."""
        return self._materialised_variables()

    @property
    def dimensions(self) -> Mapping:
        """The file's dimension lengths, by name.

        An unlimited dimension reports the number of records written so far.

        """
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
        assert self._dataset is not None
        variable = self._dataset.createVariable(
            name, dtype, tuple(dimensions), fill_value=fill_value, **encoding
        )
        wrapped = NetCDFDatasetVariable(
            variable, self._location, write_lock_factory=self._write_lock_factory
        )
        # Register this wrapper, so that a later lookup by name returns it
        # rather than a second wrapper with its own attribute cache.
        self._materialised_variables()[name] = wrapped
        return wrapped

    def sync(self) -> None:
        """Flush buffered writes to the file."""
        assert self._dataset is not None
        self._dataset.sync()

    def finalise(self) -> None:
        """Do nothing: a netCDF file needs no completion step.

        Exists for stores that do, such as Zarr's consolidated metadata.

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

        One per dataset. Made on first use, never at construction.

        """
        if self._write_lock is None:
            self._write_lock = _dask_locks.get_worker_lock(self._location)
        return self._write_lock

    def _write_lock_factory(self):
        """Return :attr:`write_lock`, making it on the first call.

        Handed to every :class:`NetCDFDatasetVariable` so that each can find
        the one shared lock at the moment it builds a write handle.

        """
        return self.write_lock
