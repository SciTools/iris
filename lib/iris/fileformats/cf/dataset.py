# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""The format-agnostic description of a CF-conforming array store.

:class:`CFDataset` and :class:`CFDatasetVariable` are what the rest of the CF
layer is written against. They are deliberately small: only what the
:class:`~iris.fileformats.cf.CFVariable` classes, the CF loader and the CF
saver actually need, with one implementation per storage format
(:mod:`iris.fileformats.netcdf._dataset` today, Zarr next).

The split that matters is ``attributes`` versus everything else. A netCDF
variable presents ``units`` - a CF attribute read from the file - and
``dimensions`` - a property of the storage - through the same attribute syntax,
and a caller cannot tell which is which. Here, CF attributes are only ever
reached through ``attributes``, and storage properties are named, typed
members.

``attributes`` is an open-ended set of data keys read from a file, with no
schema to enumerate, which is why :class:`TrackedAttributes` exists and why
:meth:`~iris.fileformats.cf.CFVariable.__getattr__` is allowed to forward to
it. Tracking is the load-bearing part: Iris decides which file attributes
survive onto a loaded cube by asking which ones the loading rules did *not*
read, so a read that goes unrecorded silently leaks a CF-reserved attribute
onto the cube.

See sections 4.2, 4.3 and 4.5 of
``docs/superpowers/specs/2026-09-21-zarr-io-design.md``.

"""

from abc import ABC, abstractmethod
from collections.abc import (
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    MutableMapping,
    ValuesView,
)
from types import MappingProxyType
from typing import Any

import numpy as np


class TrackedAttributes(MutableMapping):
    """A variable's attributes, recording which of them have been looked up.

    Single-key lookups record: :meth:`__getitem__`, :meth:`get` and
    :meth:`__contains__` - the last because ``hasattr(cf_var, name)`` marks an
    attribute used today, by reaching ``__getattr__``. Bulk access does not:
    iteration, :meth:`keys`, :meth:`values`, :meth:`items` and :func:`len`
    leave the record alone, and :attr:`untracked` is the explicit escape hatch
    for a single-key lookup that must not count.

    """

    def __init__(self, source: MutableMapping, *, ignored: Iterable[str] = ()):
        """Wrap ``source``, treating any ``ignored`` name it holds as already read."""
        self._source = source
        self._ignored = frozenset(ignored)
        self._read: set[str] = set()
        self.reset()

    def __getitem__(self, key: str) -> Any:
        """Return ``key``'s value, recording it as read."""
        # Index first: a KeyError must escape without recording anything.
        value = self._source[key]
        self._read.add(key)
        return value

    def __contains__(self, key: object) -> bool:
        """Return whether ``key`` is present, recording a hit as read."""
        result = key in self._source
        if result:
            self._read.add(key)  # type: ignore[arg-type]
        return result

    def __setitem__(self, key: str, value: Any) -> None:
        """Set ``key``'s value, recording nothing."""
        self._source[key] = value

    def __delitem__(self, key: str) -> None:
        """Remove ``key``, and any record of it having been read."""
        del self._source[key]
        self._read.discard(key)

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over the attribute names, recording nothing."""
        return iter(self._source)

    def __len__(self) -> int:
        """Return the number of attributes, recording nothing."""
        return len(self._source)

    def keys(self) -> KeysView:
        """Return a view of the attribute names, recording nothing."""
        return self._source.keys()

    def values(self) -> ValuesView:
        """Return a view of the attribute values, recording nothing."""
        return self._source.values()

    def items(self) -> ItemsView:
        """Return a view of the attribute pairs, recording nothing."""
        return self._source.items()

    def __repr__(self) -> str:
        """Return a string representation, recording nothing."""
        return f"{self.__class__.__name__}({dict(self._source)!r})"

    @property
    def untracked(self) -> Mapping:
        """A read-only view of the same attributes that records nothing."""
        return MappingProxyType(dict(self._source))

    @property
    def read(self) -> frozenset:
        """The names looked up since construction or the last :meth:`reset`."""
        return frozenset(self._read)

    @property
    def unread(self) -> frozenset:
        """The names present but not looked up since the last :meth:`reset`."""
        return frozenset(self._source) - self._read

    def reset(self) -> None:
        """Forget every recorded lookup, re-seeding from the ignored names."""
        self._read = set(self._ignored) & set(self._source)


class CFDatasetVariable(ABC):
    """One named array in a CF-conforming dataset."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The variable's name within its dataset."""

    @property
    @abstractmethod
    def location(self) -> str:
        """The path or URL of the dataset holding this variable.

        Repeated from :attr:`CFDataset.location` with the same meaning, so that
        a variable can label a message or a data proxy without a reference back
        to its dataset.

        """

    @property
    @abstractmethod
    def dimensions(self) -> tuple:
        """The names of the dimensions this variable spans, in order."""

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """The variable's shape."""

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """The variable's stored data type.

        Usually a :class:`numpy.dtype`. netCDF's variable-length string type
        reports the builtin :class:`str` instead, and
        :func:`iris.fileformats.netcdf.loader._get_cf_var_data` branches on
        that, so it is passed through rather than normalised.

        """

    @property
    @abstractmethod
    def size(self) -> int:
        """The total number of elements in the variable."""

    @property
    @abstractmethod
    def fill_value(self) -> Any:
        """The store's own fill value, or ``None`` if it has none."""

    @property
    @abstractmethod
    def chunking(self) -> tuple | None:
        """The variable's storage chunk shape, or ``None`` when unchunked."""

    @property
    @abstractmethod
    def attributes(self) -> MutableMapping:
        """The variable's CF attributes, materialised once at construction."""

    @abstractmethod
    def __getitem__(self, keys) -> np.ndarray:
        """Return the indexed portion of the variable's data."""

    @abstractmethod
    def __setitem__(self, keys, values) -> None:
        """Write ``values`` into the indexed portion of the variable."""

    @abstractmethod
    def write_handle(self) -> Any:
        """Return a picklable object supporting ``__setitem__`` on this variable.

        This is what a Dask worker receives as a ``da.store`` target, so it must
        survive pickling and must remain usable after the dataset it came from
        has been closed.

        """

    @property
    def ndim(self) -> int:
        """The number of dimensions the variable spans."""
        return len(self.shape)

    def __len__(self) -> int:
        """Return the length of the variable's leading dimension."""
        if not self.shape:
            # netCDF4.Variable.__len__ raises exactly this for a scalar, and
            # CFVariable.__len__ forwards to it today.
            msg = "len() of unsized object"
            raise TypeError(msg)
        return self.shape[0]

    def deprecated_netcdf_member(self, name: str) -> Any:
        """Return a netCDF4-only member of the object backing this variable.

        The one-release compatibility route for code that reached netCDF4 API
        through a :class:`~iris.fileformats.cf.CFVariable`. A store with no
        backing netCDF4 object has nothing to offer and raises, which is the
        correct answer rather than a special case.

        """
        raise AttributeError(name)


class CFDataset(ABC):
    """A CF-conforming array store, open for reading or writing."""

    @property
    @abstractmethod
    def location(self) -> str:
        """The store's path or URL, for messages and data proxies."""

    @property
    @abstractmethod
    def mode(self) -> str:
        """The mode the store was opened in: ``r``, ``r+``, ``a``, ``w`` or ``w-``."""

    @property
    @abstractmethod
    def closed(self) -> bool:
        """Whether :meth:`close` has already released the store."""

    @property
    @abstractmethod
    def variables(self) -> Mapping:
        """The store's variables, by name."""

    @property
    @abstractmethod
    def dimensions(self) -> Mapping:
        """The store's dimension lengths, by name."""

    @property
    @abstractmethod
    def attributes(self) -> MutableMapping:
        """The store's global attributes."""

    @abstractmethod
    def create_dimension(self, name: str, size: int | None) -> None:
        """Declare a dimension of the given length.

        ``size=None`` requests an unlimited dimension. A store with no such
        concept raises :class:`NotImplementedError`.

        """

    @abstractmethod
    def create_variable(
        self, name: str, dtype, dimensions=(), *, fill_value=None, **encoding
    ) -> "CFDatasetVariable":
        """Create and return a new variable.

        ``**encoding`` is storage-specific by nature - ``zlib``, ``complevel``
        and ``chunksizes`` for netCDF; ``compressors``, ``chunks`` and
        ``shards`` for Zarr - so each implementation documents the keys it
        accepts, and generic CF code never constructs them.

        """

    @abstractmethod
    def sync(self) -> None:
        """Flush buffered writes to the store."""

    @abstractmethod
    def finalise(self) -> None:
        """Perform the store's one-shot completion step, after every write.

        Deliberately **not** part of :meth:`close`: a worker that writes one
        slab of a store must close its handle without performing a completion
        step that only one process may perform.

        """

    @abstractmethod
    def close(self) -> None:
        """Release the store's resources."""

    def __enter__(self) -> "CFDataset":
        """Return this dataset, for use as a context manager."""
        return self

    def __exit__(self, *exc_info) -> None:
        """Close this dataset on leaving the context, whatever happened in it."""
        self.close()
