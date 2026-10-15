# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Value conversions a Zarr store needs on the way in.

The mirror of :mod:`iris.fileformats.zarr._decode`: pure functions of their
arguments, with no zarr-python call and no Iris object between them, so that
every rule can be tested on a plain value.

**Attribute values are converted before zarr sees them, never after.** A value
zarr's JSON encoder rejects raises, *and stays in the attributes mapping*, so a
caller that converts on failure has already corrupted the store's metadata.

**``_FillValue`` and ``missing_value`` are encoded differently, and the
difference is not Iris's.** On a floating array, released xarray reads
``_FillValue`` through zarr's own fill-value parser, which wants the base64 of
a little-endian ``float64`` and fails the *whole dataset open* on a plain JSON
number. It reads ``missing_value`` as an ordinary attribute, where base64 is a
string that masks nothing. So the first is packed and the second is not. See
§4.5 of the native Zarr I/O design,
``docs/superpowers/specs/2026-09-21-zarr-io-design.md``.

**Packing is applied here, not assumed from the backend.** netCDF4-python
applies ``scale_factor``/``add_offset`` packing itself, on assignment, once
those attributes are set on the variable - which is why the generic CF saver
only ever writes the attributes and streams the unpacked values through
unchanged. Zarr has no such behaviour, so :func:`pack` does the arithmetic
that would otherwise silently not happen. It is the mirror of
:func:`iris.fileformats.zarr._decode.unpack`, called from
:meth:`iris.fileformats.zarr._dataset.ZarrDatasetVariable.__setitem__` before
the fill described above, because the fill value lives in *packed* space and
the data arrives in *unpacked* space - packing an already-filled sentinel
would put it through the same transform as real data and lose it.

"""

import base64
import json
import struct
from typing import Any

import dask.array as da
import numpy as np

from iris._lazy_data import is_lazy_data

#: The format of a base64 ``_FillValue``: one little-endian float64, whatever
#: the array's own floating width.
_PACKED_DOUBLE = "<d"

#: The fill value each dtype takes when a masked source declares none. These
#: are netCDF's own defaults, so that one cube saved to both formats declares
#: the same missing value. Keyed as ``numpy.dtype.str`` without its byte-order
#: character. Not read from netCDF4: this package may not import it.
DEFAULT_FILL_VALUES: dict[str, Any] = {
    "i1": -127,
    "u1": 255,
    "i2": -32767,
    "u2": 65535,
    "i4": -2147483647,
    "u4": 4294967295,
    "i8": -9223372036854775806,
    "u8": 18446744073709551614,
    "f4": 9.9692099683868690e36,
    "f8": 9.9692099683868690e36,
}


def json_native(value: Any, *, name: str) -> Any:
    """Return `value` as something zarr's JSON encoder accepts.

    Parameters
    ----------
    value : Any
        The attribute value, as the cube holds it.
    name : str
        The attribute's name, for the error message. A failure here is a
        value the user wrote, so the message has to say which one.

    Returns
    -------
    Any
        A JSON-serialisable value: NumPy scalars become Python ones, NumPy
        arrays become lists, ``bytes`` become ``str`` via ASCII, a ``list``
        or ``tuple`` becomes a list of converted elements, and everything
        else is returned unchanged.

    Raises
    ------
    TypeError
        If the value cannot be stored. It is **not** converted to its string
        form: a string that cannot be read back is a worse answer than a
        refusal, and a warning would not stop the save.

    """
    converted: Any
    if isinstance(value, bytes):
        try:
            converted = value.decode("ascii")
        except UnicodeDecodeError as exception:
            message = (
                f"Cannot save attribute {name!r}: its bytes are not ascii, "
                "and a Zarr attribute is text."
            )
            raise TypeError(message) from exception
    elif isinstance(value, np.generic):
        converted = value.item()
    elif isinstance(value, np.ndarray):
        converted = value.tolist()
    elif isinstance(value, (list, tuple)):
        # Element by element, because a sequence of NumPy scalars is an
        # ordinary thing to build - `[data.min(), data.max()]`, say - and
        # passing it through unexamined leaves the NumPy inside it for
        # json.dumps to reject. A tuple comes back as a list because JSON
        # has no tuple, which the rule above already settles for an array.
        converted = [json_native(item, name=name) for item in value]
    else:
        converted = value

    try:
        json.dumps(converted)
    except TypeError as exception:
        message = (
            f"Cannot save attribute {name!r}: a value of type "
            f"{type(value).__name__} has no JSON form. Convert it to a number, "
            "a string, or a list of either."
        )
        raise TypeError(message) from exception

    return converted


def encode_fill_value(value: Any, dtype: np.dtype) -> Any:
    """Return a ``_FillValue`` attribute in the encoding readers expect.

    Parameters
    ----------
    value : Any
        The sentinel, in any numeric form.
    dtype : numpy.dtype
        The array's dtype, which decides the encoding.

    Returns
    -------
    str or int or float
        For a floating array, the base64 of the little-endian ``float64``
        form - including for ``float32``, which is what released xarray
        reads. For anything else, a plain JSON value.

    """
    if dtype.kind == "f":
        packed = struct.pack(_PACKED_DOUBLE, float(value))
        return base64.b64encode(packed).decode("ascii")
    return json_native(value, name="_FillValue")


def encode_missing_values(value: Any, dtype: np.dtype) -> Any:
    """Return a ``missing_value`` attribute as plain JSON numbers.

    Never base64, whatever the dtype: xarray reads this attribute as an
    ordinary value, so a packed string there opens cleanly and masks nothing.
    CF section 2.5.1 allows a vector here, so a sequence stays a sequence.

    Parameters
    ----------
    value : Any
        One sentinel, or a sequence of them.
    dtype : numpy.dtype
        The array's dtype. Unused, and named so that the two encoders have
        one signature: a caller choosing between them should not have to
        learn which one wants the dtype.

    Returns
    -------
    int or float or list

    """
    del dtype
    return json_native(value, name="missing_value")


def default_fill_value(dtype: np.dtype) -> np.generic | None:
    """Return the sentinel to use for a masked source that declares none.

    Parameters
    ----------
    dtype : numpy.dtype
        The array's dtype.

    Returns
    -------
    numpy.generic or None
        The netCDF default fill for that dtype, as a value of it, or
        ``None`` for a dtype the table has no entry for.

    Notes
    -----
    ``None`` rather than an error, because the table is netCDF's and a Zarr
    store holds a good deal netCDF does not - ``float16``, ``complex64``, a
    fixed-width byte string. Nothing about such a dtype stops it being
    stored; what it lacks is a *missing-value* sentinel, which only matters
    once the data turns out to be masked. That is known in
    :meth:`iris.fileformats.zarr.saver.Saver._backend_write`, and refused
    there, where the message can say so truthfully.

    """
    key = dtype.str[1:]
    if key not in DEFAULT_FILL_VALUES:
        return None
    return dtype.type(DEFAULT_FILL_VALUES[key])


def filled(data, fill_value):
    """Return `data` with any masked points replaced by `fill_value`.

    A Zarr array stores the fill in the data, as spec §4.5 settles: no
    separate mask array. Writing a masked array to one raises inside numpy,
    so this is not optional.

    Parameters
    ----------
    data : numpy.ndarray or dask.array.Array
        Real or lazy, masked or not.
    fill_value : Any
        What replaces the masked points.

    Returns
    -------
    numpy.ndarray or dask.array.Array
        Lazy if `data` was lazy. An unmasked input is returned unchanged,
        not copied.

    """
    if is_lazy_data(data):
        return da.ma.filled(data, fill_value)
    if np.ma.isMaskedArray(data):
        return data.filled(fill_value)
    return data


def pack(data, scale_factor, add_offset, dtype: np.dtype):
    """Return `data` packed by CF's scale and offset, as `dtype`.

    The inverse of :func:`iris.fileformats.zarr._decode.unpack`:
    ``packed = round((unpacked - add_offset) / scale_factor)``. Rounding
    applies only when `dtype` is an integer type - exactly netCDF4-python's
    own transform, not a rounding version of it applied regardless of target -
    so a non-integer `dtype` is merely narrowed, the same as :func:`unpack`
    narrows an already-unpacked value. CF packing is round-to-nearest, not
    truncation - ``int()`` would truncate toward zero and bias every negative
    value by up to one count.

    The arithmetic runs in float64 regardless of `dtype`: `scale_factor` and
    `add_offset` reach here as plain Python floats, decoded from JSON, which
    has no float32. A result can therefore differ from netCDF4's float32
    computation of the same packing by one count - the float64 answer is the
    more accurate one, but it is not byte-identical to netCDF4's.

    Parameters
    ----------
    data : numpy.ndarray, numpy.ma.MaskedArray or dask.array.Array
        The values to write, in physical units. A mask, if any, is carried
        through the arithmetic unchanged and is still present in the result.
    scale_factor, add_offset : optional
        CF's packing parameters, read from the variable's own attributes.
        Either absent behaves as its identity - no offset, unit scale.
    dtype : numpy.dtype
        The array's stored dtype, which the result is narrowed to.

    Returns
    -------
    numpy.ndarray, numpy.ma.MaskedArray or dask.array.Array
        `data` itself, unchanged, when neither parameter is given - a
        variable with no packing attributes keeps its own dtype untouched,
        rather than being rounded and recast to a dtype it already has.

    Raises
    ------
    ValueError
        If `scale_factor` is zero, naming it: every value would divide by
        zero, silently becoming infinite or NaN rather than raising on its
        own.

    """
    if scale_factor == 0:
        message = "Cannot pack with scale_factor=0: every value would divide by zero."
        raise ValueError(message)
    if scale_factor is None and add_offset is None:
        return data
    result = data
    if add_offset is not None:
        result = result - add_offset
    if scale_factor is not None:
        result = result / scale_factor

    def _round_and_narrow(block):
        """Round (if `dtype` is an integer type) and cast one block."""
        if dtype.kind in "iu":
            block = np.round(block)
        if np.ma.isMaskedArray(block):
            # Not `block.astype(dtype)`: a masked array carries its own
            # `fill_value`, inherited from `block`'s own (floating) dtype -
            # typically a value like ``1e+20`` - and `MaskedArray.astype`
            # casts that too, into a dtype it was never meant to fit.  numpy
            # warns "invalid value encountered in cast" for that, about a
            # value this function never writes anywhere; reconstructing from
            # the underlying data and mask lets the new dtype's own default
            # apply instead, and casts only the real values.
            return np.ma.masked_array(
                block.data.astype(dtype, copy=False), mask=block.mask
            )
        return block.astype(dtype, copy=False)

    if is_lazy_data(result):
        return result.map_blocks(_round_and_narrow, dtype=dtype)
    return _round_and_narrow(result)
