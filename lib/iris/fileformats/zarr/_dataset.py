# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""The Zarr implementation of the :mod:`iris.fileformats.cf.dataset` interface.

This is the only module that knows both the CF interface and the zarr-python
API. Everything Zarr-specific the CF layer could want - where dimension names
are recorded, what a chunk is, what the store's own fill value is - either has
a named member on the interface or lives here as a Zarr-only member.

Three things about Zarr shape this module, and each is a trap if forgotten.

**A dimension name guarantees less than a netCDF one.** The version 3 core
specification makes ``dimension_names`` optional, allows a null entry, only
*recommends* that names are distinct within an array, and places no
restriction at all on reusing a name across arrays. All of that is conforming
Zarr, so none of it may raise. A dimension with no name is **anonymous**, which
Iris has always had: a cube dimension carrying no coordinate.

**``Array.fill_value`` is not ``_FillValue``.** It says what unwritten storage
reads back as; it does not say the value is missing. zarr-python defaults it to
zero, so reading it as a missing-value sentinel masks every valid zero in an
array that never declared one. :attr:`ZarrDatasetVariable.fill_value` reports
it as the storage fact it is, and the masking decision is made in
:mod:`iris.fileformats.zarr._decode` from the CF attributes.

**The read unit is the inner chunk, even when the array is sharded.** Zarr's
sharding codec is indexed, so a reader fetches only the inner chunks it needs.
:attr:`ZarrDatasetVariable.chunking` therefore returns ``Array.chunks``, never
``Array.shards``.

Chunking is sized by :func:`iris._lazy_data._optimum_chunksize`, the same way
the netCDF path sizes it, and is not then adjusted to the store's own
chunking. Iris warns when a stored chunk is larger than Dask's target, from
:func:`iris.fileformats.cf.loader._chunks_from_chunk_control`, and leaves the
choice to ``CHUNK_CONTROL``.

See sections 4.2, 4.4 and 4.6 of
``docs/superpowers/specs/2026-09-21-zarr-io-design.md``.

.. todo:: Move the maintainer prose above into a ``maintainer-notes`` block.

"""

from collections.abc import Callable, Iterator, Mapping, MutableMapping
from functools import partial
import json
from types import MappingProxyType
from typing import Any
import warnings

import dask
import dask.array as da
from dask.array.core import normalize_chunks
import dask.config
import dask.utils
import numpy as np
import zarr

from iris._lazy_data import _optimum_chunksize, as_lazy_data
from iris.exceptions import TranslationError
from iris.fileformats.cf import loader as cf_loader
from iris.fileformats.cf.dataset import CFDataset, CFDatasetVariable
import iris.warnings

from . import _decode

#: What :attr:`CFDataset.format` answers, by the store's Zarr version.
_FORMAT_BY_ZARR_VERSION = {2: "ZARR_V2", 3: "ZARR_V3"}

#: The version 2 attribute carrying dimension names, as xarray and NCZarr
#: write it. Storage, not a CF attribute, so it is withheld from `attributes`.
_ARRAY_DIMENSIONS = "_ARRAY_DIMENSIONS"

#: netCDF-c's own bookkeeping attribute, stamped onto the root group of every
#: store it writes (confirmed on NCZarr output in both its `nczarr` and
#: `xarray` URL modes, and never on a subgroup or an array). netCDF4-python
#: hides it from `Dataset.ncattrs()` by convention, so it has never reached a
#: cube loaded through netCDF; Zarr has no equivalent convention, so it must
#: be withheld here for the same reason as `_ARRAY_DIMENSIONS` above -
#: provenance the writer left behind, not CF metadata the user wrote.
_NC_PROPERTIES = "_NCProperties"


def _dimension_names(array) -> tuple:
    """Return one name or None per dimension of `array`.

    Parameters
    ----------
    array : :class:`zarr.Array`
        The array to read names from. Version 3 carries them in its metadata;
        version 2 carries them in the ``_ARRAY_DIMENSIONS`` attribute, which
        is a de-facto convention rather than part of the Zarr specification.

    Returns
    -------
    tuple
        Of length ``array.ndim``. ``None`` where the store names no dimension.

    Raises
    ------
    iris.exceptions.TranslationError
        If `array` declares any Zarr version but 2 or 3. zarr-python parses
        ``zarr_format`` as a ``Literal`` and rejects such an array before
        Iris sees it, so this guards against a future widening there.

    Warns
    -----
    IrisCfLoadWarning
        If the store names a number of dimensions other than ``array.ndim``.
        The version 3 specification says the lengths "must" match, so such a
        store is malformed; its names are ignored and the array loads with
        anonymous dimensions.

    """
    anonymous = (None,) * array.ndim
    if not array.ndim:
        return ()

    zarr_format = array.metadata.zarr_format
    if zarr_format == 3:
        names = array.metadata.dimension_names
    elif zarr_format == 2:
        names = array.attrs.get(_ARRAY_DIMENSIONS)
    else:
        message = (
            "Iris reads Zarr version 2 and version 3 stores. This array "
            f"declares Zarr version {zarr_format!r}: {array.basename}"
        )
        raise TranslationError(message)

    if names is None:
        return anonymous

    if len(names) != array.ndim:
        message = (
            "Ignoring malformed dimension_names, which names "
            f"{len(names)} dimensions of a {array.ndim}-dimensional array: "
            f"{array.basename}"
        )
        warnings.warn(message, category=iris.warnings.IrisCfLoadWarning)
        return anonymous

    return tuple(names)


def _assemble_dimensions(variables) -> dict:
    """Return the dimension sizes the store's arrays agree on.

    Parameters
    ----------
    variables : mapping of str to :class:`ZarrDatasetVariable`
        Every array in one group.

    Returns
    -------
    dict
        Name to size, for every name used at exactly one size. Anonymous
        dimensions have no name and so are absent.

    Warns
    -----
    IrisCfLoadWarning
        For each name used at more than one size. Zarr places no restriction
        on reusing a name across arrays, so this is conforming and must not
        raise; merging the two would produce a wrong cube, so the name is
        omitted instead and the arrays that use it keep it in their own
        `dimensions`.

    """
    sizes: dict[str, int] = {}
    conflicts: dict[str, set[int]] = {}
    for variable in variables.values():
        for name, size in zip(variable.dimensions, variable.shape):
            if name is None:
                continue
            if name not in sizes:
                sizes[name] = size
            elif sizes[name] != size:
                conflicts.setdefault(name, {sizes[name]}).add(size)

    for name, found in conflicts.items():
        message = (
            "Ignoring a dimension name used at more than one size, "
            f"{sorted(found)}: {name}"
        )
        warnings.warn(message, category=iris.warnings.IrisCfLoadWarning)
        del sizes[name]

    return sizes


class _ZarrAttributes(MutableMapping):
    """A Zarr node's attributes as a mapping, less the ones `withhold` names.

    Which attributes those are is the caller's decision, taken where the
    mapping is built. Both the constants named there - `_ARRAY_DIMENSIONS`
    and `_NC_PROPERTIES` - carry the reason with them.
    """

    def __init__(
        self,
        node: "zarr.Group | zarr.Array",
        *,
        withhold: frozenset = frozenset(),
    ):
        """Materialise the attributes of `node`, a zarr array or group.

        Writing through this mapping writes to `node`, which is the store.
        """
        self._node = node
        self._withhold = frozenset(withhold)
        self._values = {
            key: value for key, value in node.attrs.items() if key not in self._withhold
        }

    def __getitem__(self, key: str) -> Any:
        """Return `key`'s value."""
        return self._values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set `key`'s value, here and in the store."""
        if key in self._withhold:
            msg = f"Cannot set a storage attribute through `attributes`: {key}"
            raise KeyError(msg)
        self._node.attrs[key] = value
        self._values[key] = value

    def __delitem__(self, key: str) -> None:
        """Remove `key`, here and from the store."""
        del self._node.attrs[key]
        del self._values[key]

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over the attribute names, in store order."""
        return iter(self._values)

    def __len__(self) -> int:
        """Return the number of attributes."""
        return len(self._values)

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"{self.__class__.__name__}({self._values!r})"


class ZarrDatasetVariable(CFDatasetVariable):
    """One array of a Zarr store, presented through the CF interface."""

    def __init__(self, array, location: str, format: str):
        """Wrap `array`.

        Parameters
        ----------
        array : :class:`zarr.Array`
            The array this variable presents.
        location : str
            The path or URL of the dataset this array belongs to.
        format : str
            ``ZARR_V2`` or ``ZARR_V3``, as its dataset reports.

        """
        self._array = array
        self._location = location
        self._format = format
        self._dimensions = _dimension_names(array)
        self._attributes = _ZarrAttributes(
            array, withhold=frozenset({_ARRAY_DIMENSIONS})
        )

    @property
    def name(self) -> str:
        """The array's name within its group."""
        return self._array.basename

    @property
    def location(self) -> str:
        """The path or URL of the dataset holding this variable."""
        return self._location

    @property
    def format(self) -> str:
        """``ZARR_V2`` or ``ZARR_V3``."""
        return self._format

    @property
    def dimensions(self) -> tuple:
        """One name per dimension, ``None`` where the store names none."""
        return self._dimensions

    @property
    def shape(self) -> tuple:
        """The array's shape."""
        return tuple(self._array.shape)

    @property
    def dtype(self) -> np.dtype:
        """The array's stored data type."""
        return self._array.dtype

    @property
    def size(self) -> int:
        """The total number of elements in the array."""
        return int(self._array.size)

    @property
    def fill_value(self) -> Any:
        """What unwritten storage reads back as. **Not** a missing-value flag.

        Version 3 requires this field, and zarr-python defaults it to zero, so
        it says nothing about whether a value is missing. The CF
        ``_FillValue`` and ``missing_value`` attributes say that, and
        :mod:`iris.fileformats.zarr._decode` reads them.

        """
        return self._array.fill_value

    @property
    def chunking(self) -> tuple | None:
        """The array's inner chunk shape, never its shard shape."""
        chunks = self._array.chunks
        return tuple(chunks) if chunks else None

    @property
    def storage_settings(self) -> Mapping:
        """The Zarr storage settings this array holds."""
        return MappingProxyType(
            {
                "chunks": self._array.chunks,
                "shards": self._array.shards,
                "codecs": self._array.metadata.to_dict().get("codecs"),
                "zarr_format": self._array.metadata.zarr_format,
            }
        )

    @property
    def attributes(self) -> MutableMapping:
        """The array's CF attributes, less the ones that are really storage."""
        return self._attributes

    def __getitem__(self, keys) -> np.ndarray:
        """Return the indexed portion of the array's data."""
        return self._array[keys]

    def __setitem__(self, keys, values) -> None:
        """Write `values` into the indexed portion of the array."""
        self._array[keys] = values

    def read_data(self, chunking_policy: Callable[[], tuple]) -> Any:
        """Return this array's data, lazily where a lazy array is worth it.

        Parameters
        ----------
        chunking_policy : callable
            Called with no arguments, returning the ``(chunks, dims_fixed)``
            pair to build a lazy array with. Called only once the result is
            known to be lazy.

        Returns
        -------
        numpy.ndarray, numpy.ma.MaskedArray or dask.array.Array
            Real data for a small array, lazy data otherwise. Masked whenever
            the variable declares anything to decode.

        Notes
        -----
        There is no data proxy. ``zarr.Array`` is picklable, a chunk is an
        independent object in a key-value store so concurrent reads need no
        lock, and a store is a namespace rather than an open file handle. CF
        decoding is a ``map_blocks`` layer over the raw array rather than
        something hidden inside ``__getitem__``, so it is visible in the
        graph and testable without a store.

        """
        parameters = _decode.decode_parameters(
            self._attributes,
            dtype=self.dtype,
            storage_fill_value=self.fill_value,
            zarr_format=self.zarr_format,
            variable_name=self.name,
        )
        decoding = _decode.decoding_required(parameters)

        if self.size * self.dtype.itemsize < cf_loader._LAZYVAR_MIN_BYTES:
            # A lazy array would cost more memory AND more time to access.
            result = np.asanyarray(self._array[...])
            if decoding:
                result = _decode.mask_and_scale(result, **parameters)
        else:
            chunks, dims_fixed = chunking_policy()
            # Chunks are sized on the dtype the values will have once decoded,
            # not on the one the store holds: a packed int8 that unpacks to
            # float64 costs eight times its stored size the moment the decode
            # layer runs, and Dask's target chunk size is a target for what one
            # task holds. The netCDF path arrives at the same place from the
            # other end - its data proxy declares the decoded dtype, and
            # `as_lazy_data` sizes on the array it is given. `_unpacked_dtype`
            # returns the stored dtype for a variable with no packing, so
            # nothing moves in the usual case.
            decoded_dtype = parameters["dtype"]

            if chunks == "auto":
                # CHUNK_CONTROL.as_dask(): the user asked for Dask's own
                # policy, so the store's layout is deliberately not consulted.
                # The policy is still Dask's - its algorithm, its configured
                # target - but it is called here rather than left to
                # `from_array`, which is the only way to hand it the decoded
                # dtype. `from_array` would size on the array it is given, and
                # the array it is given is the stored one.
                chunks = tuple(
                    sizes[0]
                    for sizes in normalize_chunks(
                        "auto", shape=self.shape, dtype=decoded_dtype
                    )
                )
            else:
                if chunks is None:
                    # What `as_lazy_data` makes of None, done here because this
                    # path reaches `_optimum_chunksize` first and that does not
                    # accept it: one chunk spanning the array, divided down
                    # below.
                    chunks = self.shape
                chunks = _optimum_chunksize(
                    chunks, shape=self.shape, dtype=decoded_dtype, dims_fixed=dims_fixed
                )

            # Fixed in every dimension, because the sizing above is the final
            # word whichever branch produced it: `as_lazy_data` would otherwise
            # re-run its own sizing on the stored dtype and undo it.
            result = as_lazy_data(
                self._array,
                chunks=chunks,
                meta=np.empty((0,) * self.ndim, dtype=self.dtype),
                dims_fixed=(True,) * self.ndim,
                cache_key=self.cache_key,
            )

            if decoding:
                # `partial` rather than `**parameters` on the `map_blocks`
                # call: `dtype` is a keyword `map_blocks` consumes for itself
                # rather than forwarding, and `mask_and_scale` needs it too.
                # Binding it once here also makes the dtype Dask declares and
                # the dtype the layer produces one value rather than two
                # statements that could disagree.
                result = da.map_blocks(
                    partial(_decode.mask_and_scale, **parameters),
                    result,
                    dtype=decoded_dtype,
                    meta=np.ma.masked_array(np.empty((0,) * self.ndim, decoded_dtype)),
                )

        return result

    def write_handle(self) -> "ZarrDatasetVariable":
        """Return self: a Zarr write target needs no proxy.

        ``zarr.Array`` is picklable and a chunk is an independent object in a
        key-value store, so there is no open file handle to avoid holding and
        no lock to acquire.

        """
        return self

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"{self.__class__.__name__}({self.name!r}, {self.location!r})"

    # Zarr-only members below.

    @property
    def array(self):
        """The backing :class:`zarr.Array`. Nothing generic may read this."""
        return self._array

    @property
    def zarr_format(self) -> int:
        """2 or 3. The one Zarr fact that changes how a value is read."""
        return self._array.metadata.zarr_format

    @property
    def cache_key(self) -> str:
        """An identity for this array that goes stale when the array does.

        `iris._lazy_data.CACHE` is process-wide, so an address alone is not
        enough: a store rewritten in place keeps its address and changes its
        contents. The metadata document covers shape, chunks, shards, dtype,
        fill value, codecs, dimension names and attributes in one value, and
        is treated as opaque - listing the parts Iris cares about would go
        stale, silently, the first time zarr-python added a field that
        mattered.
        """
        return json.dumps(
            [self._location, self._array.path, self._array.metadata.to_dict()],
            sort_keys=True,
        )


#: What `mode` a dataset may be opened in. Writing is not implemented.
_READ_MODES = ("r",)

#: The one message every unimplemented write member raises.
_NO_SAVING = "Saving to Zarr is not supported."


class ZarrDataset(CFDataset):
    """A Zarr store, or one group of one, presented through the CF interface."""

    def __init__(self, location, mode: str = "r", *, group: str | None = None):
        """Open a Zarr store.

        Parameters
        ----------
        location : str or :class:`pathlib.Path`
            The store's path or URL. A URL is handed to zarr-python
            unaltered; Iris adds no store library and handles no credentials,
            so an unreachable scheme fails with zarr-python's own message. A
            store object - such as a :class:`zarr.storage.MemoryStore`, which
            has no path - is also accepted and handed through unaltered;
            `location` then becomes zarr-python's own rendering of it, the
            best name available for a store with no address.
        mode : str, default="r"
            Only ``"r"`` is supported.
        group : str, optional
            A ``/``-separated path to a group within the store. The root group
            by default. Iris reads one group per call and does not walk
            nested groups.

        Raises
        ------
        NotImplementedError
            For any `mode` other than ``"r"``.
        FileNotFoundError
            If `location` does not exist.
        zarr.errors.GroupNotFoundError
            If `location` is not a Zarr store, or holds no such `group`.

        """
        # Set first, so that close() is safe if opening fails.
        self._group: zarr.Group | None = None
        self._closed = False
        self._variables: dict[str, ZarrDatasetVariable] | None = None
        self._attributes: _ZarrAttributes | None = None
        self._dimensions: dict[str, int] | None = None

        if mode not in _READ_MODES:
            raise NotImplementedError(f"{_NO_SAVING} Cannot open with mode={mode!r}.")

        self._mode = mode
        target = str(location)
        self._location = target if group is None else f"{target.rstrip('/')}/{group}"

        # `location` itself, not its stringified `target`, is handed to
        # zarr-python: stringifying a store object such as a `MemoryStore`
        # loses the store, leaving only a name zarr-python cannot reopen.
        self._group = zarr.open_group(store=location, mode=mode, path=group)
        self._format = _FORMAT_BY_ZARR_VERSION[self._group.metadata.zarr_format]

        # Assembled here rather than on first access, because assembling is
        # what warns about a dimension name used at two different lengths and
        # a load never asks for the mapping: the CF layer reads each
        # variable's own `dimensions`. Left to first access, that warning
        # reached the unit tests and no user. It costs no extra reads -
        # `variables` is materialised by everything that opens a store.
        self._dimensions = _assemble_dimensions(self.variables)

    @property
    def location(self) -> str:
        """The store's path or URL, with the group path when there is one."""
        return self._location

    @property
    def format(self) -> str:
        """``ZARR_V2`` or ``ZARR_V3``."""
        return self._format

    @property
    def mode(self) -> str:
        """The mode the store was opened in."""
        return self._mode

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has been called.

        A Zarr store is a namespace rather than an open file handle, so there
        is nothing to release and nothing else to ask.

        """
        return self._closed

    @property
    def variables(self) -> Mapping:
        """The group's arrays, by name. Arrays in subgroups are not included."""
        if self._variables is None:
            assert self._group is not None
            self._variables = {
                name: ZarrDatasetVariable(array, self._location, self._format)
                for name, array in self._group.arrays()
            }
        return self._variables

    @property
    def dimensions(self) -> Mapping:
        """The dimension sizes this group's arrays agree on.

        Assembled when the store is opened, not on first access here.
        """
        assert self._dimensions is not None
        return self._dimensions

    @property
    def storage_settings(self) -> Mapping:
        """The store-wide Zarr settings."""
        assert self._group is not None
        consolidated = self._group.metadata.consolidated_metadata
        return MappingProxyType(
            {
                "zarr_format": self._group.metadata.zarr_format,
                "consolidated": consolidated is not None,
            }
        )

    @property
    def attributes(self) -> MutableMapping:
        """The group's attributes, less netCDF-c's own bookkeeping stamp."""
        if self._attributes is None:
            assert self._group is not None
            self._attributes = _ZarrAttributes(
                self._group, withhold=frozenset({_NC_PROPERTIES})
            )
        return self._attributes

    def create_dimension(self, name: str, size: int | None) -> None:
        """Raise: saving to Zarr is not supported."""
        raise NotImplementedError(_NO_SAVING)

    def create_variable(
        self, name: str, dtype, dimensions=(), *, fill_value=None, **storage_settings
    ):
        """Raise: saving to Zarr is not supported."""
        raise NotImplementedError(_NO_SAVING)

    def supports_dtype(self, dtype) -> bool:
        """Whether this store can hold a variable of this dtype.

        Always True. Zarr's data types cover the whole of NumPy's fixed-width
        numeric range, in both versions, so there is no dtype a caller could
        reach here with that the store would have to narrow. Contrast
        :meth:`~iris.fileformats.netcdf._dataset.NetCDFDataset.supports_dtype`,
        which answers False for the integer types the netCDF 3 formats have no
        type for.

        Parameters
        ----------
        dtype : :class:`numpy.dtype`
            The dtype a variable would be created with.

        Returns
        -------
        bool

        """
        return True

    def sync(self) -> None:
        """Do nothing: a Zarr write is not buffered."""

    def finalise(self) -> None:
        """Raise: saving to Zarr is not supported.

        A Zarr store's completion step is consolidating its metadata, which
        only a writer performs.

        """
        raise NotImplementedError(_NO_SAVING)

    def close(self) -> None:
        """Record that this dataset is finished with.

        A store is a namespace, not an open handle, so nothing is released.
        The flag exists because callers - :class:`iris.fileformats.cf.CFReader`
        among them - must be able to ask.

        """
        self._closed = True

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"{self.__class__.__name__}({self.location!r}, mode={self.mode!r})"

    # Zarr-only members below.

    @property
    def subgroup_names(self) -> tuple:
        """The names of this group's subgroups.

        What the loader names when a group holds no data variables but the
        store plainly holds data elsewhere.

        """
        assert self._group is not None
        return tuple(self._group.group_keys())
