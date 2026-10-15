# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :mod:`iris.fileformats.zarr._encode`."""

import base64
import struct

import dask.array as da
import numpy as np
import pytest

from iris.fileformats.zarr._encode import (
    DEFAULT_FILL_VALUES,
    default_fill_value,
    encode_fill_value,
    encode_missing_values,
    filled,
    json_native,
)


def unpack(encoded):
    """Return the float in a base64 little-endian float64."""
    return struct.unpack("<d", base64.b64decode(encoded))[0]


class TestJsonNative:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (np.float32(1.5), 1.5),
            (np.int16(3), 3),
            (np.bool_(True), True),
            (np.array([1, 2], dtype=np.int32), [1, 2]),
            (np.array([[1.0], [2.0]], dtype=np.float64), [[1.0], [2.0]]),
            (b"bytes", "bytes"),
            ("text", "text"),
            (1, 1),
            (1.5, 1.5),
            ([1, "two"], [1, "two"]),
            (None, None),
        ],
    )
    def test_it_converts(self, value, expected):
        result = json_native(value, name="an_attribute")
        assert result == expected
        # Every result must survive the JSON encoder zarr will hand it to.
        import json

        json.dumps(result)

    def test_a_numpy_scalar_becomes_a_python_scalar(self):
        # Not merely equal: zarr's JSON encoder rejects numpy types outright.
        assert type(json_native(np.float32(1.5), name="a")) is float

    def test_a_non_ascii_string_is_kept(self):
        # JSON is text, so unlike netCDF there is nothing to coerce. A degree
        # sign in a title is ordinary.
        assert json_native("°C", name="units") == "°C"

    def test_non_ascii_bytes_raise(self):
        with pytest.raises(TypeError, match="ascii"):
            json_native("é".encode(), name="an_attribute")

    def test_an_unserialisable_value_raises_naming_the_attribute(self):
        class Opaque:
            pass

        with pytest.raises(TypeError, match="an_attribute"):
            json_native(Opaque(), name="an_attribute")

    def test_it_does_not_stringify(self):
        # Spec §4.5: "An attribute the format cannot hold is an error, and is
        # not quietly turned into a string."
        class Stringy:
            def __str__(self):
                return "looks fine"

        with pytest.raises(TypeError):
            json_native(Stringy(), name="an_attribute")


class TestEncodeFillValue:
    def test_float32_is_base64_of_a_float64(self):
        encoded = encode_fill_value(np.float32(-99.0), np.dtype("f4"))
        assert isinstance(encoded, str)
        assert unpack(encoded) == -99.0

    def test_float64_is_base64_too(self):
        encoded = encode_fill_value(np.float64(-99.0), np.dtype("f8"))
        assert unpack(encoded) == -99.0

    def test_nan_survives(self):
        assert np.isnan(unpack(encode_fill_value(np.nan, np.dtype("f4"))))

    def test_an_integer_is_a_plain_integer(self):
        encoded = encode_fill_value(np.int16(-9), np.dtype("i2"))
        assert encoded == -9
        assert type(encoded) is int

    def test_an_unsigned_integer_is_a_plain_integer(self):
        assert encode_fill_value(np.uint8(255), np.dtype("u1")) == 255

    def test_a_string_fill_is_itself(self):
        assert encode_fill_value("", np.dtype("<U4")) == ""


class TestEncodeMissingValues:
    # F10: base64 here would open in xarray and silently not mask.
    def test_a_float_scalar_stays_a_number(self):
        assert encode_missing_values(np.float32(-99.0), np.dtype("f4")) == -99.0

    def test_a_float_scalar_is_not_base64(self):
        assert not isinstance(
            encode_missing_values(np.float32(-99.0), np.dtype("f4")), str
        )

    def test_a_vector_stays_a_vector(self):
        result = encode_missing_values(
            np.array([-99.0, -98.0], dtype="f4"), np.dtype("f4")
        )
        assert result == [-99.0, -98.0]

    def test_a_list_stays_a_list(self):
        assert encode_missing_values([-9, -8], np.dtype("i2")) == [-9, -8]


class TestDefaultFillValue:
    @pytest.mark.parametrize(
        ("dtype", "expected"),
        [
            ("f4", 9.9692099683868690e36),
            ("f8", 9.9692099683868690e36),
            ("i1", -127),
            ("i2", -32767),
            ("i4", -2147483647),
            ("i8", -9223372036854775806),
            ("u1", 255),
            ("u2", 65535),
            ("u4", 4294967295),
        ],
    )
    def test_the_table_is_netcdfs(self, dtype, expected):
        # One cube saved to both formats declares the same missing value.
        result = default_fill_value(np.dtype(dtype))
        assert result == np.dtype(dtype).type(expected)
        assert result.dtype == np.dtype(dtype)

    def test_an_unknown_dtype_raises_naming_it(self):
        with pytest.raises(ValueError, match="complex"):
            default_fill_value(np.dtype("c8"))

    def test_every_entry_fits_its_dtype(self):
        for key, value in DEFAULT_FILL_VALUES.items():
            dtype = np.dtype(key)
            assert dtype.type(value) == dtype.type(value)  # no overflow to inf


class TestFilled:
    def test_a_masked_array_is_filled(self):
        data = np.ma.masked_array([1.0, 2.0, 3.0], mask=[False, True, False])
        result = filled(data, -99.0)
        assert not np.ma.isMaskedArray(result)
        np.testing.assert_array_equal(result, [1.0, -99.0, 3.0])

    def test_an_unmasked_array_passes_through(self):
        data = np.array([1.0, 2.0])
        assert filled(data, -99.0) is data

    def test_a_lazy_masked_array_stays_lazy(self):
        # Realising here would defeat the whole deferred save.
        data = da.ma.masked_greater(da.arange(4, chunks=2, dtype="f4"), 2)
        result = filled(data, -99.0)
        assert isinstance(result, da.Array)
        np.testing.assert_array_equal(result.compute(), [0.0, 1.0, 2.0, -99.0])

    def test_a_fully_masked_array_is_all_fill(self):
        data = np.ma.masked_all(3, dtype="f4")
        np.testing.assert_array_equal(filled(data, -99.0), [-99.0] * 3)
