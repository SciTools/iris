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

from . import _decode, _encode

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


def _selection_shape(shape: tuple, keys) -> tuple | None:
    """Return the shape `keys` selects out of an array of `shape`.

    Measured with numpy's own indexing rules, which are the rules
    zarr-python implements for a basic selection, against a zero-strided
    dummy so that measuring allocates nothing.

    Parameters
    ----------
    shape : tuple of int
        The array's full shape.
    keys : Any
        The selection, as it reached ``__setitem__``.

    Returns
    -------
    tuple of int or None
        ``None`` for anything but a basic selection - a boolean mask or an
        index array, which zarr accepts and Iris never writes, and which
        would make a copy to measure. A caller treats that as "nothing to
        check".

    """
    parts = keys if isinstance(keys, tuple) else (keys,)
    if not all(
        isinstance(part, (slice, int, np.integer)) or part is Ellipsis or part is None
        for part in parts
    ):
        return None
    dummy = np.broadcast_to(np.empty((), dtype=np.int8), shape)
    try:
        return dummy[keys].shape
    except IndexError:
        # Too many indices for the rank, say. zarr-python raises for these
        # too, with its own message, so there is nothing to add here.
        return None


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
        dtype=None,
    ):
        """Materialise the attributes of `node`, a zarr array or group.

        Writing through this mapping writes to `node`, which is the store.

        Parameters
        ----------
        node : zarr.Array or zarr.Group
            Whose attributes these are.
        withhold : frozenset, optional
            Names that are storage rather than CF, hidden here and refused
            on assignment.
        dtype : numpy.dtype, optional
            The array's dtype, when `node` is an array. What decides the
            encoding of ``_FillValue``; a group has neither.


        """
        self._node = node
        self._withhold = frozenset(withhold)
        self._dtype = dtype
        self._values = {
            key: value for key, value in node.attrs.items() if key not in self._withhold
        }

    def __getitem__(self, key: str) -> Any:
        """Return `key`'s value."""
        return self._values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set `key`'s value, here and in the store.

        The value is converted to something JSON can hold *before* zarr sees
        it. A value zarr's encoder rejects raises **and stays in the
        mapping**, so converting on failure would mean converting a store
        that is already wrong.

        """
        if key in self._withhold:
            msg = f"Cannot set a storage attribute through `attributes`: {key}"
            raise KeyError(msg)
        encoded = self._encode(key, value)
        self._node.attrs[key] = encoded
        self._values[key] = encoded

    def _encode(self, key: str, value: Any) -> Any:
        """Return `value` in the form this attribute is stored in."""
        if self._dtype is not None:
            if key == "_FillValue":
                return _encode.encode_fill_value(value, self._dtype)
            if key == "missing_value":
                return _encode.encode_missing_values(value, self._dtype)
        return _encode.json_native(value, name=key)

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
            array, withhold=frozenset({_ARRAY_DIMENSIONS}), dtype=array.dtype
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
        """Write `values` into the indexed portion of the array.

        Packed first, filled second. `values` arrives in physical units;
        :func:`~iris.fileformats.zarr._encode.pack` applies this array's own
        ``scale_factor``/``add_offset`` attributes, the transform
        netCDF4-python would otherwise have applied automatically on
        assignment, which a Zarr array has no equivalent of. Masked values are
        filled with the array's own fill value afterwards, not before: the
        fill lives in *stored* (packed) space, so filling first would send an
        already-filled sentinel through the same transform as real data and
        lose it. A Zarr array has no mask of its own either way - spec §4.5
        settles that the fill goes in the data - and handing numpy.ma
        straight to zarr raises inside the bytes codec, naming neither the
        variable nor the mask.

        A value whose shape does not match the selection is refused first.
        zarr-python truncates an oversized assignment without a word, so a
        caller that sized the array wrongly loses data in silence; too few
        values raise, but from inside numpy, naming neither this variable
        nor the array it belongs to.

        Raises
        ------
        ValueError
            If `values` neither matches the selection's shape nor is a
            scalar to broadcast over it.

        """
        if self._array.ndim == 0 and keys == slice(None):
            # `_CFSaver._lazy_stream_data` writes a whole variable as
            # `cf_var[:] = data`, which is `slice(None)` - one axis too many
            # for a 0-d array, which zarr indexes with `...` or `()` instead.
            # A scalar cube is ordinary, and this is otherwise the only
            # store that cannot hold one.
            keys = Ellipsis

        selected = _selection_shape(self.shape, keys)
        given = np.shape(values)
        if selected is not None and given not in ((), selected):
            message = (
                f"Cannot write values of shape {given} into {self.name!r}, "
                f"whose selected region has shape {selected}. Zarr neither "
                "resizes an array nor reports an oversized write, so this "
                "would otherwise lose values in silence."
            )
            raise ValueError(message)

        packed = _encode.pack(
            values,
            self._attributes.get("scale_factor"),
            self._attributes.get("add_offset"),
            self._array.dtype,
        )
        self._array[keys] = _encode.filled(packed, self._array.fill_value)

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
    def write_grid(self) -> tuple:
        """The unit a concurrent writer must own whole cells of.

        The shards when the array is sharded, the chunks when it is not. A
        shard is one stored object holding many inner chunks, so two tasks
        writing different chunks of one shard read-modify-write the same
        object and one update is lost. Stating the invariant over the chunk
        grid alone is what xarray's ``safe_chunks`` had to be corrected to;
        see pydata/xarray#10831.

        """
        return tuple(self._array.shards or self._array.chunks)

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


#: The modes that only read.
_READ_MODES = ("r",)

#: The modes that may write. ``"r+"`` opens an existing store for update and
#: creates nothing, which is what the region write of a future multi-process
#: saver needs. See §4.5 of docs/superpowers/specs/2026-09-21-zarr-io-design.md.
_WRITE_MODES = ("w-", "w", "a", "r+")

#: Every mode, in the order the error message lists them.
_MODES = _READ_MODES + _WRITE_MODES

#: The Zarr versions Iris can write. This constrains writing only; reading
#: handles either version. Adding version 2 here also means implementing the
#: three version-specific points of §4.5 of
#: docs/superpowers/specs/2026-09-21-zarr-io-design.md.
_WRITABLE_ZARR_FORMATS = (3,)

#: The version a write creates when the caller names none. Separate from
#: `_WRITABLE_ZARR_FORMATS` because the order of that tuple says nothing
#: about which member is the default.
_DEFAULT_ZARR_FORMAT = 3

#: The dtype kinds whose fill value must survive conversion unchanged.
#: Integers and booleans have no rounding to spend: `numpy.int16(1.5)` is
#: `1`, which would leave the array's own fill saying one number and its
#: `_FillValue` attribute saying another. The floating kinds are left out
#: because their conversion loses precision by definition - `float32` cannot
#: hold the `float64` the caller wrote either way - and a reader narrows the
#: declared sentinel to the stored dtype before comparing, which is what
#: makes that case safe.
_EXACT_FILL_KINDS = "biu"

#: The dtypes every zarr-python Iris supports can store, by NumPy kind and
#: item size in bytes. Read by `supports_dtype`, which is the only thing that
#: should.
#:
#: Stated as what Zarr *has*, not as what it lacks. A list of exclusions has
#: to name every dtype NumPy can spell, including the ones it gains later and
#: the ones that are only a container for another - a structured dtype with
#: an object field, a zero-length void - and a list that misses one answers
#: True for a dtype `create_variable` then refuses.
#:
#: It is also the intersection across versions rather than what the newest
#: can hold. zarr-python 3.1 replaced a fixed enum of fifteen data types with
#: an extensible registry, so 3.4 stores a structured dtype and a
#: `datetime64` that 3.0.8 - the oldest Iris accepts - has no data type for
#: **[verified]**. Answering from the newest would be wrong on the oldest,
#: which is the harder failure to notice.
#:
#: Byte order is deliberately not part of the answer, although 3.0.8's table
#: is little-endian only: a PP or GRIB load yields big-endian arrays, and a
#: False here would send a big-endian `int64` into the saver's narrowing path
#: on every version, including the ones that would have stored it.
_STORABLE_ITEMSIZES_BY_KIND = {
    "b": frozenset({1}),
    "i": frozenset({1, 2, 4, 8}),
    "u": frozenset({1, 2, 4, 8}),
    "f": frozenset({2, 4, 8}),
    "c": frozenset({8, 16}),
}

#: The text kinds, which Zarr stores as its `string` and `bytes` types. Their
#: item size is the string length, so there is no set of sizes to check -
#: only that it is not zero, which is not a string Zarr can hold.
_STORABLE_TEXT_KINDS = "US"

#: The storage keywords `create_variable` passes to zarr-python, which are
#: zarr-python's own. Iris does not invent names for them and does not
#: translate netCDF's: see `_KEYWORD_ADVICE` below, and §4.5 of the design.
#:
#: `order` is **not** here, although zarr-python's own signature takes it:
#: it has no effect on a version 3 array, which is all Iris writes, and
#: zarr-python merely warns and ignores it. A keyword Iris accepts and
#: silently discards is worse than one it refuses with a reason.
_ZARR_ENCODING_KEYWORDS = frozenset(
    {"chunks", "shards", "compressors", "filters", "serializer"}
)

#: What to say about a keyword someone tried here that this store does not
#: take. Mostly netCDF's, where the advice is the Zarr name if there is one
#: and a plain "no equivalent" if there is not - a compression model that
#: differs is worth saying out loud, and silently mapping `zlib=True` onto a
#: Blosc codec would be a false equivalence. The rest are keywords that read
#: as though they ought to work here, and are worth more than silence.
_KEYWORD_ADVICE = {
    "zlib": "use compressors, e.g. compressors=[zarr.codecs.GzipCodec()]",
    "complevel": "use compressors, which carry their own level",
    "shuffle": "use compressors, e.g. the shuffle option of BloscCodec",
    "fletcher32": "no equivalent; Zarr checksums with a Crc32c codec",
    "contiguous": "no equivalent; a Zarr array is always chunked",
    "chunksizes": "use chunks",
    "endian": "no equivalent; a Zarr codec decides byte order",
    "least_significant_digit": "no equivalent",
    "unlimited_dimensions": "no equivalent; a Zarr array has a fixed shape",
    "order": (
        "zarr-python takes order for a Zarr version 2 array, and Iris writes "
        "version 3, where it has no effect"
    ),
    "append": (
        "there is no append; use mode='a' to add new arrays to a store that "
        "already exists"
    ),
}


def check_encoding_keywords(encoding) -> None:
    """Raise for any storage keyword a Zarr store does not have.

    Parameters
    ----------
    encoding : dict
        The storage keywords a caller passed.

    Raises
    ------
    TypeError
        Naming each unusable keyword, and what to do instead where there is
        something to say.

    """
    unknown = set(encoding) - _ZARR_ENCODING_KEYWORDS
    if not unknown:
        return
    lines = []
    for keyword in sorted(unknown):
        advice = _KEYWORD_ADVICE.get(keyword)
        if advice is None:
            lines.append(f"  {keyword}: not a Zarr storage keyword")
        else:
            lines.append(f"  {keyword}: {advice}")
    message = (
        "Unusable Zarr storage keyword(s). A Zarr store takes zarr-python's "
        "own keywords, and Iris does not translate netCDF's onto them:\n"
        + "\n".join(lines)
    )
    raise TypeError(message)


def existing_name_message(name: str, location: str) -> str:
    """Return the message for a name a Zarr group already holds.

    One sentence, shared by :meth:`ZarrDataset.create_variable` and
    :class:`iris.fileformats.zarr.saver.Saver`'s own earlier check of the
    same fact, so the two call sites cannot drift into saying it two ways.

    Parameters
    ----------
    name : str
        The array name that already exists.
    location : str
        The group's path or URL, as :attr:`ZarrDataset.location` reports it.

    Returns
    -------
    str

    """
    return (
        f"Cannot create {name!r} in the Zarr group {location!r}: it "
        "already holds something of that name, and Iris neither "
        "overwrites nor merges into one."
    )


def _storage_fill_value(fill_value, dtype, name: str):
    """Return `fill_value` as a `dtype` scalar, or raise if it cannot be one.

    An array's own fill value and its ``_FillValue`` attribute must be the
    same number. The attribute is what declares a value missing, the fill is
    what unwritten storage reads back as, and a reader that trusts one and
    meets the other masks the wrong points or none. Converting quietly is
    what would make them differ: ``numpy.int16(1.5)`` is ``1``, while the
    attribute would still say ``1.5``, so a reader comparing against the
    attribute - xarray does - finds nothing to mask.

    Parameters
    ----------
    fill_value :
        The sentinel the caller gave.
    dtype : numpy.dtype or type
        The array's dtype. ``str`` itself is a variable-length string array,
        whose sentinel is text and has nothing to round.
    name : str
        The array's name, for the message.

    Returns
    -------
    The sentinel as a scalar of `dtype`.

    Raises
    ------
    ValueError
        If `dtype` cannot hold `fill_value`.

    """
    if dtype is str:
        return np.dtype(str).type(fill_value)

    dtype = np.dtype(dtype)
    refusal = (
        f"Cannot use fill_value={fill_value!r} for the Zarr array {name!r}: "
        f"{dtype} cannot hold it, so the array's own fill value and its "
        "'_FillValue' attribute would be two different numbers. Pass a "
        "sentinel of the array's own type."
    )
    try:
        with np.errstate(over="raise"):
            stored = dtype.type(fill_value)
    except (OverflowError, ValueError, TypeError, FloatingPointError) as exception:
        raise ValueError(refusal) from exception
    if dtype.kind in _EXACT_FILL_KINDS and stored != fill_value:
        raise ValueError(refusal)
    return stored


class ZarrDataset(CFDataset):
    """A Zarr store, or one group of one, presented through the CF interface."""

    def __init__(
        self,
        location,
        mode: str = "r",
        *,
        group: str | None = None,
        zarr_format: int | None = None,
    ):
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
            One of ``"r"``, ``"r+"``, ``"a"``, ``"w"`` or ``"w-"``, with
            zarr-python's meanings. ``"w-"`` creates and refuses an existing
            target; ``"w"`` replaces one; ``"a"`` adds to one; ``"r+"`` adds
            to one and refuses to create it.
        group : str, optional
            A ``/``-separated path to a group within the store. The root group
            by default. Iris reads one group per call and does not walk
            nested groups.
        zarr_format : int, optional
            The version to create, when writing. Version 3 when not given,
            and version 3 is the only one Iris writes. Reading handles either
            version and ignores this: zarr-python reads the version from the
            store, and a version named here that the store disagrees with
            would be reported as a missing group.

        Raises
        ------
        ValueError
            For an unknown `mode`.
        NotImplementedError
            For a write `mode` with a `zarr_format` Iris does not write.
        FileExistsError
            For ``mode="w-"`` onto a target that already exists.
        FileNotFoundError
            If `location` does not exist and `mode` does not create it.
        zarr.errors.GroupNotFoundError
            If `location` is not a Zarr store, or holds no such `group`.

        """
        # Set first, so that close() is safe if opening fails.
        self._group: zarr.Group | None = None
        self._closed = False
        self._variables: dict[str, ZarrDatasetVariable] | None = None
        self._attributes: _ZarrAttributes | None = None
        self._dimensions: dict[str, int] | None = None

        #: A record of dimension names and sizes.
        #: Zarr stores no dimension object of its own, so this is the only
        #: record of a declared size, and the only way to catch two arrays
        #: declaring one name at two sizes.
        self._declared_dimensions: dict[str, int] = {}

        if mode not in _MODES:
            message = (
                f"Unknown mode {mode!r} for a Zarr store. Expected one of {_MODES}."
            )
            raise ValueError(message)

        writing = mode in _WRITE_MODES
        if not writing:
            # Reading detects the version from the store. Naming one here
            # would constrain that detection, and a store of the other
            # version would then be reported as a missing group.
            zarr_format = None
        elif zarr_format is None:
            zarr_format = _DEFAULT_ZARR_FORMAT
        elif zarr_format not in _WRITABLE_ZARR_FORMATS:
            message = (
                f"zarr_format={zarr_format} is not currently supported for "
                "writing in Iris. Currently supported formats are: "
                f"{_WRITABLE_ZARR_FORMATS}. "
            )
            raise NotImplementedError(message)

        self._mode = mode
        target = str(location)
        self._location = target if group is None else f"{target.rstrip('/')}/{group}"

        # `location` itself, not its stringified `target`, is handed to
        # zarr-python: stringifying a store object such as a `MemoryStore`
        # loses the store, leaving only a name zarr-python cannot reopen.
        try:
            self._group = zarr.open_group(
                store=location,
                mode=mode,
                path=group,
                zarr_format=zarr_format,
                # A write mode must see what the store actually holds, not a
                # frozen consolidated snapshot: `mode="a"` onto a store this
                # saver already consolidated would otherwise list only the
                # arrays present at the last consolidation, and never the one
                # this call is about to add. A read mode keeps the
                # consolidated fast path, which is what it is for.
                use_consolidated=False if writing else None,
            )
        except FileExistsError as exception:
            message = (
                f"Cannot create the Zarr store {self._location!r}: it already "
                f"exists, and mode={mode!r} will not overwrite it. Pass "
                "mode='w' to replace it, or mode='a' to add to it."
            )
            raise FileExistsError(message) from exception
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
        """The dimension sizes this group's arrays agree on, and any declared.

        Assembled when the store is opened, and again after a write has
        changed what the answer would be. A dimension declared by
        :meth:`create_dimension` appears here before any array carries it,
        because that is what a caller choosing a new name needs to see.

        """
        if self._dimensions is None:
            self._dimensions = {
                **_assemble_dimensions(self.variables),
                **self._declared_dimensions,
            }
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
        """Declare a dimension of the given length.

        Zarr has no dimension object: a dimension is a name an array carries
        in its own metadata. So this records the name and size, and
        :meth:`create_variable` writes the names onto each array. The record
        is what lets a size conflict be caught here rather than becoming two
        arrays that disagree.

        A name an array in this group already carries counts as taken, at
        that array's length. ``mode="a"`` onto an existing store is the case
        that needs it: the arrays already there cannot be resized, so a new
        declaration that disagrees with one of them has no outcome that is
        not a lie - the new array would be created at the new size while
        :attr:`dimensions` reports one length for a name two arrays hold at
        two. The saver avoids the collision by renaming its own dimension;
        this is what makes the collision visible to it.

        Parameters
        ----------
        name : str
            The dimension's name.
        size : int or None
            Its length. ``None`` requests an unlimited dimension, which Zarr
            has no form for.

        Raises
        ------
        NotImplementedError
            For ``size=None``.
        ValueError
            If `name` is already in use at a different size, or if this
            dataset was opened read-only.

        """
        self._check_writable("create_dimension")
        if size is None:
            message = (
                f"Cannot create the unlimited dimension {name!r}: a Zarr array "
                "has a fixed shape, so there is no unlimited dimension to "
                "create."
            )
            raise NotImplementedError(message)

        declared = self._declared_dimensions.get(name)
        if declared is not None and declared != size:
            message = (
                f"Cannot declare dimension {name!r} with size {size}: it is "
                f"already declared with size {declared}."
            )
            raise ValueError(message)
        dimensions = dict(self.dimensions)
        existing = dimensions.get(name)
        if existing is not None and existing != size:
            message = (
                f"Cannot declare dimension {name!r} with size {size}: an array "
                f"in the Zarr group {self._location!r} already uses that name "
                f"with size {existing}, and an existing array is never resized."
            )
            raise ValueError(message)
        self._declared_dimensions[name] = size
        # Updated rather than dropped: the mapping was just read, and
        # dropping it would reassemble it from every array in the group on
        # the next call - once per declaration, over a store that may be
        # remote.
        dimensions[name] = size
        self._dimensions = dimensions

    def _check_writable(self, action: str) -> None:
        """Raise unless this dataset was opened in a mode that writes."""
        if self._mode not in _WRITE_MODES:
            message = (
                f"Cannot {action} on a Zarr store opened with mode={self._mode!r}. "
                f"Open it with one of {_WRITE_MODES} to write."
            )
            raise ValueError(message)

    def create_variable(
        self, name: str, dtype, dimensions=(), *, fill_value=None, **encoding
    ) -> "ZarrDatasetVariable":
        """Create and return a new array.

        Parameters
        ----------
        name : str
            The array's name within this group.
        dtype : numpy.dtype or str or type
            The array's dtype. ``str`` itself asks for a variable-length
            string array, which is how this store holds text.
        dimensions : iterable of str, optional
            The names of its dimensions, each already declared by
            :meth:`create_dimension`. Empty for a scalar.
        fill_value : optional
            The CF missing-value sentinel. When given it is written as a
            ``_FillValue`` attribute as well as the array's own fill, because
            on version 3 the attribute is the masking declaration and the
            storage fill is not. When not given, the array takes the netCDF
            default fill for its dtype, and declares nothing: the saver
            declares one if it finds the data is masked.
        **encoding
            Zarr's own storage keywords: ``chunks``, ``shards``,
            ``compressors``, ``filters``, ``serializer``.

        Returns
        -------
        :class:`ZarrDatasetVariable`

        Raises
        ------
        ValueError
            If the dataset is read-only, a dimension is undeclared, `name`
            already exists in this group, Zarr has no data type for `dtype`,
            or `dtype` cannot hold `fill_value`.
        TypeError
            For a storage keyword this store does not have.

        """
        self._check_writable("create_variable")
        check_encoding_keywords(encoding)
        assert self._group is not None

        dtype = dtype if dtype is str else np.dtype(dtype)
        dimensions = tuple(dimensions)
        shape = tuple(self._dimension_size(dimension) for dimension in dimensions)

        if dtype is not str and not self.supports_dtype(dtype):
            # Asked here so that the refusal and `supports_dtype` cannot
            # disagree: a caller told the store holds a dtype must then be
            # able to create it, and the one that answers False must say so
            # in Iris's words rather than zarr-python's.
            message = (
                f"Cannot create the Zarr array {name!r} with dtype {dtype}: "
                "Zarr has no data type for it."
            )
            raise ValueError(message)

        if name in self._group:
            raise ValueError(existing_name_message(name, self._location))

        if fill_value is None:
            # None here means zarr-python's own default for the dtype
            # applies, and the array declares no sentinel of its own. That
            # covers a variable-length string array, which the netCDF
            # default-fill table has no entry for, and every other dtype it
            # has no entry for either - a Zarr store holds plenty netCDF
            # does not. Whether a *missing-value* sentinel is needed is a
            # question about the data, settled by the saver, which is the
            # only thing that sees it.
            stored_fill = (
                None if dtype is str else _encode.default_fill_value(np.dtype(dtype))
            )
        else:
            stored_fill = _storage_fill_value(fill_value, dtype, name)

        create_kwargs = dict(encoding)
        if stored_fill is not None:
            create_kwargs["fill_value"] = stored_fill

        array = self._group.create_array(
            name,
            shape=shape,
            dtype=dtype,
            dimension_names=dimensions or None,
            **create_kwargs,
        )
        variable = ZarrDatasetVariable(array, self._location, self._format)
        if fill_value is not None:
            variable.attributes["_FillValue"] = fill_value

        # Both caches describe a group that has just changed.
        self._variables = None
        self._dimensions = None
        return variable

    def _dimension_size(self, name: str) -> int:
        """Return a declared dimension's size, or raise naming it."""
        size = self.dimensions.get(name)
        if size is None:
            message = (
                f"Cannot create an array over dimension {name!r}: no dimension "
                "of that name has been declared in this group."
            )
            raise ValueError(message)
        return size

    def supports_dtype(self, dtype) -> bool:
        """Whether this store can hold a variable of this dtype.

        True for the whole of NumPy's fixed-width numeric range and its
        fixed-length text, which is everything the Iris savers can reach here
        with: there is no integer or float a saver must narrow to get it into
        a Zarr store. Contrast
        :meth:`~iris.fileformats.netcdf._dataset.NetCDFDataset.supports_dtype`,
        which answers False for the integer types the netCDF 3 formats have no
        type for.

        Read from `_STORABLE_ITEMSIZES_BY_KIND`, which is what Zarr has
        rather than what it lacks, and is the intersection across the
        zarr-python versions Iris accepts rather than what the newest can
        hold. So it answers False for a few dtypes the installed zarr-python
        might store - a structured dtype, a `datetime64` - and that is the
        safe direction of inexact: :meth:`create_variable` asks this before
        it creates anything, so a False is a refusal in Iris's words, while a
        True this class then went back on would be a capability question with
        two answers.

        Parameters
        ----------
        dtype : :class:`numpy.dtype`
            The dtype a variable would be created with.

        Returns
        -------
        bool

        """
        dtype = np.dtype(dtype)
        if dtype.kind in _STORABLE_TEXT_KINDS:
            return dtype.itemsize > 0
        sizes = _STORABLE_ITEMSIZES_BY_KIND.get(dtype.kind)
        return sizes is not None and dtype.itemsize in sizes

    def sync(self) -> None:
        """Do nothing: a Zarr write is not buffered."""

    def finalise(self) -> None:
        """Write consolidated metadata for the whole store.

        One document at the root listing every array and group, which is what
        makes a remote store usable: a reader fetches one object instead of
        one per node. Separate from :meth:`close` because it is a write, and
        a one-shot one - a worker that fills part of a store must not do it,
        since several writers of one metadata object race.

        Always consolidates from the root, even when this dataset is one
        group of the store. A consolidated document that does not mention the
        group just written is what a default reader would believe.

        Raises
        ------
        ValueError
            If this dataset was opened read-only.
        RuntimeError
            If the store cannot be consolidated. Raised rather than returned
            quietly: a store left with stale consolidated metadata hides the
            data just written from every default reader.

        """
        self._check_writable("finalise")
        assert self._group is not None
        with warnings.catch_warnings():
            # zarr-python warns that consolidated metadata is not part of the
            # version 3 specification. Iris asks for it deliberately, and the
            # user cannot act on it.
            warnings.filterwarnings(
                "ignore",
                message="Consolidated metadata is currently not part",
                category=UserWarning,
            )
            try:
                zarr.consolidate_metadata(self._group.store)
            except Exception as exception:
                message = (
                    f"Could not consolidate the metadata of the Zarr store "
                    f"{self._location!r}. The data written is in the store, but "
                    "a reader that uses consolidated metadata may not see it."
                )
                raise RuntimeError(message) from exception

    def drop_consolidated_metadata(self) -> None:
        """Remove the store's consolidated metadata document, if it has one.

        The other half of :meth:`finalise`, for a save that declines
        consolidation. A document an earlier save wrote does not mention what
        this one has just written, and a default reader believes the
        document: the data would be in the store and invisible to anything
        that did not know to look past it. Reading an unconsolidated store is
        slower; reading a stale document is wrong.

        Works from the whole store's root, like :meth:`finalise`, because
        that is where the document lives however deep the group written was.
        Writing an array at the root happens to rewrite the root's own
        metadata without the document, which is why this is only reachable
        when the group written was not the root - but it is not conditional
        on that, because relying on a side effect to keep a promise is how
        the promise comes to be kept only sometimes.

        Does nothing when the store has no document, including when it has no
        root group at all: a store whose only group is the one just written
        has nothing stale in it.

        Raises
        ------
        ValueError
            If this dataset was opened read-only.
        RuntimeError
            If the document cannot be removed, for the same reason
            :meth:`finalise` raises: a store left carrying it hides the data
            just written.

        """
        self._check_writable("drop consolidated metadata")
        assert self._group is not None
        store = self._group.store
        try:
            consolidated = zarr.open_group(store=store, mode="r+")
        except (FileNotFoundError, KeyError):
            # No root group - zarr-python's `GroupNotFoundError` is a
            # `FileNotFoundError` - so there is no document at the root to go
            # stale. A store whose groups are all implicit is conforming Zarr
            # and must not acquire a root group here.
            return
        if consolidated.metadata.consolidated_metadata is None:
            return

        try:
            # Reopened ignoring the document, so that the metadata written
            # back is the group's own and carries no consolidated section.
            # Rewriting the attributes is what writes it: zarr-python has no
            # call that removes a consolidated document on its own.
            root = zarr.open_group(store=store, mode="r+", use_consolidated=False)
            root.update_attributes(dict(root.attrs))
        except Exception as exception:
            message = (
                f"Could not remove the consolidated metadata of the Zarr store "
                f"{self._location!r}. The data written is in the store, but the "
                "store's consolidated metadata predates it, so a reader that "
                "uses it may not see what was just written."
            )
            raise RuntimeError(message) from exception

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
