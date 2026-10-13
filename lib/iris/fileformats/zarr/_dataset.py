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
``Array.shards``, and :func:`_align_chunks` aggregates upward into whole
multiples of it.

See sections 4.2, 4.4 and 4.6 of
``docs/superpowers/specs/2026-09-21-zarr-io-design.md``.

"""

from collections.abc import Callable, Iterator, Mapping, MutableMapping
from types import MappingProxyType
from typing import Any
import warnings

import numpy as np
import zarr

from iris.fileformats.cf.dataset import CFDataset, CFDatasetVariable
import iris.warnings

#: What :attr:`CFDataset.format` answers, by the store's Zarr version.
_FORMAT_BY_ZARR_VERSION = {2: "ZARR_V2", 3: "ZARR_V3"}

#: The version 2 attribute carrying dimension names, as xarray and NCZarr
#: write it. Storage, not a CF attribute, so it is withheld from `attributes`.
_ARRAY_DIMENSIONS = "_ARRAY_DIMENSIONS"


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

    if array.metadata.zarr_format == 3:
        names = array.metadata.dimension_names
    else:
        names = array.attrs.get(_ARRAY_DIMENSIONS)

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


class _ZarrAttributes(MutableMapping):
    """A Zarr node's attributes as a mapping, less the ones that are storage.

    ``_ARRAY_DIMENSIONS`` is how a version 2 store records dimension names.
    Under the interface's split that is a property, reported by
    :attr:`ZarrDatasetVariable.dimensions`, not a CF attribute - so it is
    withheld here. Leaving it in would put it on every cube loaded from a
    version 2 store, since every attribute the loading rules do not consume
    reaches ``cube.attributes``.
    """

    def __init__(self, node, *, withhold: frozenset = frozenset()):
        """Materialise the attributes of `node`, a zarr array or group."""
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
    def encoding(self) -> Mapping:
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
        """Not yet implemented.

        A later change adds this, following §4.4 of
        ``docs/superpowers/specs/2026-09-21-zarr-io-design.md``.

        """
        raise NotImplementedError

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
