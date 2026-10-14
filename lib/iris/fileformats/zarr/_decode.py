# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""The CF decoding netCDF4 does for free and Zarr does not.

netCDF4 applies ``_FillValue``/``missing_value`` masking,
``valid_min``/``valid_max``/``valid_range`` masking and
``scale_factor``/``add_offset`` unpacking on every read. Zarr returns a plain
``numpy.ndarray`` and applies none of it, so these rules are implemented here.
They live on the Zarr side rather than in the shared CF layer because the
netCDF path gets the behaviour from the C library and has nothing to reuse.

Everything here is a pure function of values, with no store and no Dask in
sight. :func:`mask_and_scale` is what the loader lays over a raw array as a
``map_blocks`` layer - visible in the graph, fusable with its neighbours, and
testable on a plain ndarray.

Two traps, both from the Zarr specification rather than from CF.

**A floating ``_FillValue`` is base64 on version 3.** Released xarray encodes
it as base64 of a little-endian ``float64`` for *every* floating dtype,
including ``float32``, and its reader requires that form. It is the de-facto
encoding, not a producer's mistake, so Iris reads both it and a plain JSON
number. CF section 2.5.1 asks for "the scalar attribute with the name
``_FillValue`` and of the same type as its variable"; this is a deliberate,
narrow deviation, taken because interoperability with the other Zarr readers
is the reason for native Zarr support.

**``Array.fill_value`` is not a missing-value declaration.** It says what
unwritten storage reads back as, and zarr-python defaults it to zero. On
version 3 it is never a mask source: an ``int16`` array of ``[0, 1, 2]`` with
no CF attributes would otherwise load with its valid zero masked. On version 2
it is one, but only when no CF attribute is present, because version 2 stores
predate the convention - which is what released xarray's
``use_zarr_fill_value_as_mask`` does. zarr-python 3 defaults the field to zero
for a version 2 store too, so a store written today can lose a genuine zero
this way; :func:`decode_parameters` warns for exactly that case - the storage
fill value is zero - and not for a non-zero one, which is the writer's own
deliberate, pre-convention sentinel and not something to warn about.

See section 4.4 of ``docs/superpowers/specs/2026-09-21-zarr-io-design.md``.

"""

import base64
import binascii
import struct
from typing import Any
import warnings

import numpy as np

import iris.warnings

#: The width of the base64 `_FillValue` encoding: one little-endian float64.
_PACKED_DOUBLE = "<d"
_PACKED_DOUBLE_BYTES = 8

#: The Zarr version whose stores predate the CF `_FillValue` convention, and
#: whose `Array.fill_value` is therefore read as a missing-value sentinel.
_FILL_VALUE_IS_A_MASK_BEFORE = 3


def decode_fill_value(value: Any, dtype: np.dtype, variable_name: str) -> Any:
    """Return a CF missing-value sentinel as a value of `dtype`.

    Parameters
    ----------
    value : Any
        The attribute's value, as the store holds it: a JSON number, or - for
        a floating variable - the base64 of a little-endian ``float64``.
    dtype : numpy.dtype
        The variable's own dtype. The result is narrowed to it, because a
        sentinel of a different width never equals anything in the array.
    variable_name : str
        Named in the warning, so the user can find the variable.

    Returns
    -------
    numpy.generic or None
        ``None`` when `value` is neither form, or does not fit `dtype` -
        both are producer errors.

    Warns
    -----
    IrisCfLoadWarning
        When `value` cannot be read as a sentinel of `dtype`.

    """
    if isinstance(value, (bool, np.bool_)):
        # A JSON true/false is a number to Python and nothing to CF.
        decoded = None
    elif isinstance(value, (int, float, np.number)):
        decoded = value
    elif isinstance(value, str) and dtype.kind == "f":
        decoded = _decode_packed_double(value)
    else:
        decoded = None

    if decoded is None:
        message = (
            "Ignoring a missing-value attribute that is neither a number nor "
            f"the base64 of a float64, on variable {variable_name}: {value!r}"
        )
        warnings.warn(message, category=iris.warnings.IrisCfLoadWarning)
        return None

    try:
        return dtype.type(decoded)
    except (OverflowError, ValueError):
        # Out of range for dtype: a producer error, not a reason to fail
        # the whole load. A float dtype never raises here - an overflowing
        # float narrows to +/-inf instead, which is ordinary numpy
        # behaviour and not something to guard against.
        message = (
            f"Ignoring a missing-value attribute that does not fit {dtype}, "
            f"on variable {variable_name}: {value!r}"
        )
        warnings.warn(message, category=iris.warnings.IrisCfLoadWarning)
        return None


def decode_missing_values(value: Any, dtype: np.dtype, variable_name: str) -> list:
    """Return every sentinel a ``missing_value`` attribute declares.

    CF section 2.5.1 makes ``missing_value`` "a scalar or vector containing
    values indicating missing data", so a JSON array there is as valid as a
    number and every element of it masks - which is what netCDF4 does with a
    vector ``missing_value``, and what Iris has always done through it.
    ``_FillValue`` is not read this way: section 2.5.1 makes that one a
    scalar.

    Parameters
    ----------
    value : Any
        The attribute's value: one sentinel, or a sequence of them.
    dtype : numpy.dtype
        The variable's own dtype, which every sentinel is narrowed to.
    variable_name : str
        Named in any warning.

    Returns
    -------
    list
        One entry per element :func:`decode_fill_value` could read. An
        element it could not is warned about and dropped, leaving the rest:
        one bad sentinel in a vector is no reason to ignore the good ones.

    """
    if isinstance(value, (list, tuple)) or (
        isinstance(value, np.ndarray) and value.ndim > 0
    ):
        elements = list(value)
    else:
        elements = [value]
    decoded = (decode_fill_value(element, dtype, variable_name) for element in elements)
    return [sentinel for sentinel in decoded if sentinel is not None]


def _decode_packed_double(value: str) -> float | None:
    """Return the float in a base64 little-endian float64, or None."""
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None
    if len(raw) != _PACKED_DOUBLE_BYTES:
        return None
    return struct.unpack(_PACKED_DOUBLE, raw)[0]


def decode_parameters(
    attributes,
    *,
    dtype: np.dtype,
    storage_fill_value: Any,
    zarr_format: int,
    variable_name: str,
) -> dict:
    """Return the CF decode parameters for one variable.

    Parameters
    ----------
    attributes : mapping
        The variable's CF attributes.
    dtype : numpy.dtype
        The variable's stored dtype, before any unpacking.
    storage_fill_value : Any
        ``zarr.Array.fill_value`` - what unwritten storage reads back as. A
        mask source only on version 2, and only when the variable declares no
        CF missing-value attribute.
    zarr_format : int
        2 or 3. The one place the version changes what is decoded.
    variable_name : str
        Named in any warning.

    Returns
    -------
    dict
        The keyword arguments :func:`mask_and_scale` takes, always with the
        same keys. :func:`decoding_required` says whether they do anything.

    Warns
    -----
    IrisCfLoadWarning
        When a version 2 variable declares no CF missing-value attribute, and
        the storage fill value about to be used as a sentinel is zero. Zero
        is zarr-python's own default for a storage fill value a writer never
        set, so it is the case design document section 4.4 calls "the
        ordinary case, not the exotic one" when it rules the field out as a
        mask source on version 3 - a plausible real value, not a sentinel a
        writer chose. A non-zero storage fill value - the classic netCDF
        sentinel ``9.96921e+36``, say - is the case section 4.4 keeps this
        rule for, and does not warn: a writer old enough to predate the CF
        convention chose it on purpose, which is the rule's own premise.

    """
    # The two attributes are read differently because CF section 2.5.1
    # defines them differently: `_FillValue` is a scalar, `missing_value` is
    # "a scalar or vector".
    sentinels = []
    if "_FillValue" in attributes:
        decoded = decode_fill_value(attributes["_FillValue"], dtype, variable_name)
        if decoded is not None:
            sentinels.append(decoded)
    if "missing_value" in attributes:
        sentinels.extend(
            decode_missing_values(attributes["missing_value"], dtype, variable_name)
        )

    if (
        not sentinels
        and zarr_format < _FILL_VALUE_IS_A_MASK_BEFORE
        and storage_fill_value is not None
    ):
        # Warn only when the sentinel about to be adopted is zero - the
        # value this field defaults to when a writer never set it (design
        # document section 4.4). A non-zero storage fill value is a writer's
        # deliberate choice, old enough to predate `_FillValue`, which is
        # exactly the case this rule exists for; masking on it is
        # unremarkable and does not warn. Masking itself is unchanged either
        # way - only whether this warns.
        if storage_fill_value == 0:
            message = (
                f"Variable {variable_name} declares no `_FillValue` or "
                f"`missing_value` on a version 2 store, so its storage fill "
                f"value {storage_fill_value!r} will be read as missing: values "
                "equal to it are masked. Set a `_FillValue` attribute to "
                "control this."
            )
            warnings.warn(message, category=iris.warnings.IrisCfLoadWarning)
        sentinels.append(storage_fill_value)

    valid_min = attributes.get("valid_min")
    valid_max = attributes.get("valid_max")
    valid_range = attributes.get("valid_range")
    if valid_range is not None and len(valid_range) == 2:
        valid_min, valid_max = valid_range

    scale_factor = attributes.get("scale_factor")
    add_offset = attributes.get("add_offset")

    return {
        "sentinels": tuple(sentinels),
        "valid_min": valid_min,
        "valid_max": valid_max,
        "scale_factor": scale_factor,
        "add_offset": add_offset,
        "dtype": _unpacked_dtype(dtype, scale_factor, add_offset),
    }


def _unpacked_dtype(dtype, scale_factor, add_offset) -> np.dtype:
    """Return the dtype the unpacked values will have.

    The same calculation as
    :func:`iris.fileformats.cf.loader._get_actual_dtype`, done here on plain
    values so that the dtype the Dask graph declares and the dtype the decode
    layer produces are one statement rather than two.
    """
    probe = np.zeros(1, dtype=dtype)
    if scale_factor is not None:
        probe = scale_factor * probe
    if add_offset is not None:
        probe = add_offset + probe
    return probe.dtype


def decoding_required(parameters: dict) -> bool:
    """Whether these parameters would change anything.

    A variable with nothing to decode gets no ``map_blocks`` layer and an
    unmasked Dask meta. Adding a masked meta without a layer beneath it would
    be a meta that lies about its own graph.
    """
    if parameters["sentinels"]:
        return True
    return any(
        parameters[name] is not None
        for name in ("valid_min", "valid_max", "scale_factor", "add_offset")
    )


def mask_by_sentinels(block: np.ndarray, sentinels: tuple) -> np.ndarray:
    """Return a boolean mask of the elements equal to any sentinel.

    A NaN sentinel is matched with :func:`numpy.isnan`, because NaN does not
    equal itself and NaN is what nearly every version 3 store's base64
    ``_FillValue`` decodes to.
    """
    mask = np.zeros(block.shape, dtype=bool)
    for sentinel in sentinels:
        if isinstance(sentinel, (float, np.floating)) and np.isnan(sentinel):
            mask |= np.isnan(block)
        else:
            mask |= block == sentinel
    return mask


def mask_by_valid_range(block: np.ndarray, valid_min, valid_max) -> np.ndarray:
    """Return a boolean mask of the elements outside the valid range."""
    mask = np.zeros(block.shape, dtype=bool)
    if valid_min is not None:
        mask |= block < valid_min
    if valid_max is not None:
        mask |= block > valid_max
    return mask


def unpack(block: np.ndarray, scale_factor, add_offset, dtype: np.dtype) -> np.ndarray:
    """Return `block` unpacked by CF's scale and offset, as `dtype`."""
    result = block
    if scale_factor is not None:
        result = scale_factor * result
    if add_offset is not None:
        result = add_offset + result
    return result.astype(dtype, copy=False)


def mask_and_scale(
    block: np.ndarray,
    *,
    sentinels: tuple,
    valid_min,
    valid_max,
    scale_factor,
    add_offset,
    dtype: np.dtype,
) -> np.ma.MaskedArray:
    """Apply the CF decode rules to one block, in CF's order.

    Masking first, on the **packed** values, because CF declares a sentinel to
    be "of the same type as its variable"; unpacking second. The result is
    always a masked array, so that every block matches the Dask meta the
    loader declares for the layer.

    Parameters
    ----------
    block : numpy.ndarray
        One Dask block of the raw stored values.
    sentinels : tuple
        Values meaning "missing", already narrowed to the stored dtype.
    valid_min, valid_max : optional
        CF's valid range. Values outside it are masked.
    scale_factor, add_offset : optional
        CF's packing. ``value * scale_factor + add_offset``.
    dtype : numpy.dtype
        The dtype of the result, which the Dask graph has already declared.

    Returns
    -------
    numpy.ma.MaskedArray

    """
    mask = mask_by_sentinels(block, sentinels)
    mask |= mask_by_valid_range(block, valid_min, valid_max)
    return np.ma.masked_array(unpack(block, scale_factor, add_offset, dtype), mask=mask)
