# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Machinery shared by cube merge and cube concatenate.

This module is the substrate beneath :mod:`iris._merge` and
:mod:`iris._concatenate`.  It deliberately imports nothing from Iris at
runtime, so that either engine may import it without any risk of a circular
import.  The only Iris names it requires are annotations, which are guarded
by :data:`typing.TYPE_CHECKING`.

"""

from __future__ import annotations

from collections import namedtuple
import itertools
from typing import TYPE_CHECKING, Any

import dask
import dask.array as da
import numpy as np
from xxhash import xxh3_64

if TYPE_CHECKING:
    from collections.abc import Mapping

    from iris.coords import AncillaryVariable, AuxCoord, CellMeasure, DimCoord

# Restrict the names imported from this namespace.
__all__ = ["ArrayHash", "array_id", "compute_hashes", "hash_array"]


def hash_ndarray(a: np.ndarray) -> np.ndarray:
    """Compute a hash from a numpy array.

    Calculates a 64-bit non-cryptographic hash of the provided array, using
    the fast ``xxhash`` hashing algorithm.

    Parameters
    ----------
    a :
        The array to hash.

    Returns
    -------
    numpy.ndarray :
        An array of shape (1,) containing the hash value.

    """
    # Include the array dtype as it is not preserved by `ndarray.tobytes()`.
    hash = xxh3_64(f"dtype={a.dtype}".encode("utf-8"))

    # Hash the bytes representing the array data.
    hash.update(b"data=")
    if np.ma.is_masked(a):
        # Hash only the unmasked data
        hash.update(a.compressed().tobytes())
        # Hash the mask
        hash.update(b"mask=")
        hash.update(a.mask.tobytes())
    else:
        hash.update(a.tobytes())
    return np.frombuffer(hash.digest(), dtype=np.int64)


def hash_chunk(
    x_chunk: np.ndarray,
    axis: tuple[int] | None,
    keepdims: bool,
) -> np.ndarray:
    """Compute a hash from a numpy array.

    This function can be applied to each chunk or intermediate chunk in
    :func:`~dask.array.reduction`. It preserves the number of input dimensions
    to facilitate combining intermediate results into intermediate chunks.

    Parameters
    ----------
    x_chunk :
        The array to hash.
    axis :
        Unused but required by :func:`~dask.array.reduction`.
    keepdims :
        Unused but required by :func:`~dask.array.reduction`.

    Returns
    -------
    numpy.ndarray :
        An array containing the hash value.

    """
    return hash_ndarray(x_chunk).reshape((1,) * x_chunk.ndim)


def hash_aggregate(
    x_chunk: np.ndarray,
    axis: tuple[int] | None,
    keepdims: bool,
) -> np.int64:
    """Compute a hash from a numpy array.

    This function can be applied as the final step in :func:`~dask.array.reduction`.

    Parameters
    ----------
    x_chunk :
        The array to hash.
    axis :
        Unused but required by :func:`~dask.array.reduction`.
    keepdims :
        Unused but required by :func:`~dask.array.reduction`.

    Returns
    -------
    np.int64 :
        The hash value.

    """
    (result,) = hash_ndarray(x_chunk)
    return result


def hash_array(a: da.Array | np.ndarray) -> np.int64:
    """Calculate a hash representation of the provided array.

    Calculates a 64-bit non-cryptographic hash of the provided array, using
    the fast ``xxhash`` hashing algorithm.

    Note that the computed hash depends on how the array is chunked.

    Parameters
    ----------
    a :
        The array that requires to have its hexdigest calculated.

    Returns
    -------
    np.int64
        The array's hash.

    """
    if isinstance(a, da.Array):
        # Use :func:`~dask.array.reduction` to compute a hash from a Dask array.
        #
        # A hash of each input chunk will be computed by the `chunk` function
        # and those hashes will be combined into one or more intermediate chunks.
        # If there are multiple intermediate chunks, a hash for each intermediate
        # chunk will be computed by the `combine` function and the
        # results will be combined into a new layer of intermediate chunks. This
        # will be repeated until only a single intermediate chunk remains.
        # Finally, a single hash value will be computed from the last
        # intermediate chunk by the `aggregate` function.
        result = da.reduction(
            a,
            chunk=hash_chunk,
            combine=hash_chunk,
            aggregate=hash_aggregate,
            keepdims=False,
            meta=np.empty(tuple(), dtype=np.int64),
            dtype=np.int64,
        )
    else:
        result = hash_aggregate(a, None, False)
    return result


class ArrayHash(namedtuple("ArrayHash", ["value", "chunks"])):
    """Container for a hash value and the chunks used when computing it.

    Parameters
    ----------
    value : :class:`np.int64`
        The hash value.
    chunks : tuple
        The chunks the array had when the hash was computed.
    """

    __slots__ = ()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            raise TypeError(f"Unable to compare {repr(self)} to {repr(other)}")

        def shape(chunks):
            return tuple(sum(c) for c in chunks)

        if shape(self.chunks) == shape(other.chunks):
            if self.chunks != other.chunks:
                raise ValueError(
                    "Unable to compare arrays with different chunks: "
                    f"{self.chunks} != {other.chunks}"
                )
            result = self.value == other.value
        else:
            result = False
        return result


def array_id(
    coord: DimCoord | AuxCoord | AncillaryVariable | CellMeasure,
    bound: bool,
) -> str:
    """Get a unique key for looking up arrays associated with coordinates."""
    return f"{id(coord)}{bound}"


def compute_hashes(
    arrays: Mapping[str, np.ndarray | da.Array],
) -> dict[str, ArrayHash]:
    """Compute hashes for the arrays that will be compared.

    Two arrays are considered equal if each unmasked element compares equal
    and the masks are equal. However, hashes depend on chunking and dtype.
    Therefore, arrays with the same shape are rechunked so they have the same
    chunks and arrays with numerical dtypes are cast up to the same dtype before
    computing the hashes.

    Parameters
    ----------
    arrays :
        A mapping with key-array pairs.

    Returns
    -------
    dict[str, ArrayHash] :
        An dictionary of hashes.

    """
    hashes = {}

    def is_numerical(dtype):
        return np.issubdtype(dtype, np.bool_) or np.issubdtype(dtype, np.number)

    def group_key(item):
        _, a = item
        if is_numerical(a.dtype):
            dtype = "numerical"
        else:
            dtype = str(a.dtype)
        return a.shape, dtype

    sorted_arrays = sorted(arrays.items(), key=group_key)
    for _, group_iter in itertools.groupby(sorted_arrays, key=group_key):
        array_ids, group = zip(*group_iter)
        # Unify dtype for numerical arrays, as the hash depends on it
        if is_numerical(group[0].dtype):
            dtype = np.result_type(*group)
            same_dtype_arrays = tuple(a.astype(dtype) for a in group)
        else:
            same_dtype_arrays = group
        if any(isinstance(a, da.Array) for a in same_dtype_arrays):
            # Unify chunks as the hash depends on the chunks.
            indices = tuple(range(group[0].ndim))
            # Because all arrays in a group have the same shape, `indices`
            # are the same for all of them. Providing `indices` as a tuple
            # instead of letters is easier to do programmatically.
            argpairs = [(a, indices) for a in same_dtype_arrays]
            __, rechunked_arrays = da.core.unify_chunks(*itertools.chain(*argpairs))
        else:
            rechunked_arrays = same_dtype_arrays
        for key, rechunked in zip(array_ids, rechunked_arrays):
            if isinstance(rechunked, da.Array):
                chunks = rechunked.chunks
            else:
                chunks = tuple((i,) for i in rechunked.shape)
            hashes[key] = (hash_array(rechunked), chunks)

    (hashes,) = dask.compute(hashes)
    return {k: ArrayHash(*v) for k, v in hashes.items()}
