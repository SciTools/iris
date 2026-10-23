# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :mod:`iris.fileformats.zarr._decode`."""

import base64
import struct

import numpy as np
import pytest

from iris.fileformats.zarr import _decode
from iris.warnings import IrisCfLoadWarning


def base64_double(value):
    """Return `value` in the encoding released xarray writes for a float."""
    return base64.b64encode(struct.pack("<d", value)).decode("ascii")


def parameters(attributes, *, dtype="i2", storage_fill_value=0, zarr_format=3):
    return _decode.decode_parameters(
        attributes,
        dtype=np.dtype(dtype),
        storage_fill_value=storage_fill_value,
        zarr_format=zarr_format,
        variable_name="v",
    )


class Test__decode_fill_value:
    def test_fill_value_from_a_json_number(self):
        result = _decode._decode_fill_value(-999, np.dtype("i2"), "v")
        assert result == -999
        assert result.dtype == np.dtype("i2")

    def test_fill_value_from_base64_nan(self):
        """NOAA GFS writes 'AAAAAAAA+H8=' and it is not a CF violation.

        base64(struct.pack("<d", float("nan"))) is that literal string, and
        released xarray writes it for every floating dtype.
        """
        result = _decode._decode_fill_value(
            "AAAAAAAA+H8=", np.dtype("f4"), "temperature_2m"
        )
        assert result.dtype == np.dtype("f4")
        assert np.isnan(result)

    def test_fill_value_from_base64_finite_sentinel(self):
        # Not only NaN: a finite sentinel must survive the narrowing too.
        encoded = base64_double(-999.0)
        result = _decode._decode_fill_value(encoded, np.dtype("f4"), "v")
        assert result == np.float32(-999.0)

    def test_fill_value_from_a_json_number_on_a_float(self):
        result = _decode._decode_fill_value(-999.0, np.dtype("f4"), "v")
        assert result == np.float32(-999.0)

    def test_an_undecodable_fill_value_warns_and_is_ignored(self):
        with pytest.warns(IrisCfLoadWarning, match="banana"):
            result = _decode._decode_fill_value("banana", np.dtype("f4"), "v")
        assert result is None

    def test_a_base64_string_of_the_wrong_width_warns(self):
        encoded = base64.b64encode(b"abc").decode("ascii")
        with pytest.warns(IrisCfLoadWarning, match="v"):
            assert _decode._decode_fill_value(encoded, np.dtype("f4"), "v") is None

    def test_a_base64_string_on_an_integer_variable_warns(self):
        # The convention is for floating dtypes. An integer variable's sentinel
        # is a JSON number, and a string there is a producer error.
        with pytest.warns(IrisCfLoadWarning, match="v"):
            result = _decode._decode_fill_value(base64_double(1.0), np.dtype("i2"), "v")
        assert result is None

    def test_an_out_of_range_integer_warns_and_is_ignored(self):
        # 70000 does not fit an int16, however a producer came to write it.
        with pytest.warns(IrisCfLoadWarning, match="v"):
            result = _decode._decode_fill_value(70000, np.dtype("i2"), "v")
        assert result is None

    def test_a_negative_out_of_range_integer_warns_and_is_ignored(self):
        with pytest.warns(IrisCfLoadWarning, match="v"):
            result = _decode._decode_fill_value(-70000, np.dtype("i2"), "v")
        assert result is None

    def test_an_out_of_range_unsigned_integer_warns_and_is_ignored(self):
        with pytest.warns(IrisCfLoadWarning, match="v"):
            result = _decode._decode_fill_value(300, np.dtype("u1"), "v")
        assert result is None


class Test_decode_parameters:
    """Includes `_decode_missing_values`, which only this calls."""

    def test_version_3_ignores_the_storage_fill_value(self):
        """An int16 [0, 1, 2] with no CF attributes loads with nothing masked.

        zarr-python defaults `fill_value` to zero, so reading it as a sentinel
        masks valid data in the ordinary case, not the exotic one.
        """
        assert parameters({}, storage_fill_value=0)["sentinels"] == ()

    def test_version_3_zeros_survive_masking_and_scaling_unchanged(self):
        """The property the rule above exists for, demonstrated end to end.

        An int16 array of zeros, no CF attributes, storage fill value zero:
        pushed through the actual masking function with the actual parameters
        `decode_parameters` produces, not hand-written keyword arguments.
        """
        params = parameters({}, storage_fill_value=0, zarr_format=3)
        block = np.array([0, 0, 1], dtype="i2")
        result = _decode.mask_and_scale(block, **params)
        assert not result.mask.any()
        assert result.tolist() == [0, 0, 1]

    def test_version_2_uses_the_storage_fill_value_when_there_is_no_attribute(self):
        with pytest.warns(IrisCfLoadWarning, match="v"):
            result = parameters({}, storage_fill_value=0, zarr_format=2)
        assert result["sentinels"] == (0,)

    def test_a_cf_attribute_beats_the_storage_fill_value_on_version_2(self):
        result = parameters({"_FillValue": -999}, storage_fill_value=0, zarr_format=2)
        assert result["sentinels"] == (np.int16(-999),)

    def test_both_cf_missing_value_attributes_are_sentinels(self):
        result = parameters({"_FillValue": -999, "missing_value": -888})
        assert set(result["sentinels"]) == {np.int16(-999), np.int16(-888)}

    def test_a_vector_missing_value_declares_every_sentinel(self):
        """CF section 2.5.1: "a scalar or vector containing values".

        netCDF4 masks on every element of a vector `missing_value`, so a store
        and a file carrying the same attribute have to mask the same values.
        """
        result = parameters({"missing_value": [-999, -888]})
        assert set(result["sentinels"]) == {np.int16(-999), np.int16(-888)}

    def test_a_vector_missing_value_masks_every_sentinel(self):
        params = parameters({"missing_value": [-999, -888]})
        block = np.array([-999, 1, -888, 2], dtype="i2")
        result = _decode.mask_and_scale(block, **params)
        assert result.mask.tolist() == [True, False, True, False]

    def test_a_vector_missing_value_keeps_the_version_2_fill_value_out(self):
        """The second failure the scalar-only reading caused, and the worse one.

        A rejected `missing_value` left a version 2 variable looking as though it
        declared no CF attribute at all, so the storage fill value became the
        sentinel instead - masking every valid zero in the array.
        """
        result = parameters(
            {"missing_value": [-999, -888]}, storage_fill_value=0, zarr_format=2
        )
        assert 0 not in result["sentinels"]

    def test_one_unreadable_element_does_not_discard_the_others(self):
        with pytest.warns(IrisCfLoadWarning, match="v"):
            result = parameters({"missing_value": [-999, "nonsense"]})
        assert result["sentinels"] == (np.int16(-999),)

    def test_a_vector_fill_value_is_not_read_as_sentinels(self):
        """CF section 2.5.1 makes `_FillValue` a scalar, unlike `missing_value`.

        A producer writing a vector there has written something the convention
        does not define, and a warning naming the variable is more use than a
        guess at what was meant.
        """
        with pytest.warns(IrisCfLoadWarning, match="v"):
            result = parameters({"_FillValue": [-999, -888]})
        assert result["sentinels"] == ()

    def test_valid_range_sets_both_bounds(self):
        result = parameters({"valid_range": [0, 10]})
        assert result["valid_min"] == 0
        assert result["valid_max"] == 10

    def test_valid_min_and_valid_max_are_read_separately(self):
        result = parameters({"valid_min": 1, "valid_max": 9})
        assert (result["valid_min"], result["valid_max"]) == (1, 9)

    def test_packing_parameters_and_the_resulting_dtype(self):
        result = parameters({"scale_factor": 0.0001, "add_offset": 1.0}, dtype="u2")
        assert result["scale_factor"] == 0.0001
        assert result["add_offset"] == 1.0
        assert result["dtype"] == np.dtype("f8")


class Test_decoding_required:
    def test_decoding_required_is_false_for_a_plain_array(self):
        assert _decode.decoding_required(parameters({})) is False

    def test_decoding_required_is_true_for_each_rule(self):
        assert _decode.decoding_required(parameters({"_FillValue": -999})) is True
        assert _decode.decoding_required(parameters({"valid_min": 1})) is True
        assert _decode.decoding_required(parameters({"scale_factor": 2})) is True


class Test__mask_by_sentinels:
    def test_a_sentinel_is_masked(self):
        block = np.array([0, 1, -999], dtype="i2")
        result = _decode._mask_by_sentinels(block, (np.int16(-999),))
        assert result.tolist() == [False, False, True]

    def test_mask_by_a_nan_sentinel(self):
        """NaN never equals itself, so a NaN sentinel needs `isnan`, not `==`.

        The base64 `_FillValue` nearly every xarray-written store carries is NaN,
        so this is the common case rather than the exotic one.
        """
        block = np.array([1.0, np.nan], dtype="f4")
        result = _decode._mask_by_sentinels(block, (np.float32("nan"),))
        assert result.tolist() == [False, True]


class Test__mask_by_valid_range:
    def test_values_outside_the_range_are_masked(self):
        block = np.array([-1, 5, 11], dtype="i2")
        result = _decode._mask_by_valid_range(block, 0, 10)
        assert result.tolist() == [True, False, True]


class Test__unpack:
    def test_scale_and_offset_are_applied(self):
        block = np.array([1, 2], dtype="u2")
        result = _decode._unpack(block, 0.5, 1.0, np.dtype("f8"))
        np.testing.assert_allclose(result, [1.5, 2.0])
        assert result.dtype == np.dtype("f8")


class Test_mask_and_scale:
    def test_mask_and_scale_masks_the_packed_values(self):
        """Masking is applied before unpacking, as netCDF4 does it.

        CF section 2.5.1: `_FillValue` is "of the same type as its variable", so
        it is a packed value and comparing it with an unpacked one matches
        nothing.
        """
        block = np.array([-999, 1, 2], dtype="i2")
        result = _decode.mask_and_scale(
            block,
            sentinels=(np.int16(-999),),
            valid_min=None,
            valid_max=None,
            scale_factor=0.5,
            add_offset=0.0,
            dtype=np.dtype("f8"),
        )
        assert result.mask.tolist() == [True, False, False]
        np.testing.assert_allclose(result.compressed(), [0.5, 1.0])

    def test_mask_and_scale_always_returns_a_masked_array(self):
        # The Dask meta says masked, so every block must be, mask or no mask.
        block = np.array([1, 2], dtype="i2")
        result = _decode.mask_and_scale(
            block,
            sentinels=(np.int16(-999),),
            valid_min=None,
            valid_max=None,
            scale_factor=None,
            add_offset=None,
            dtype=np.dtype("i2"),
        )
        assert isinstance(result, np.ma.MaskedArray)
        assert not result.mask.any()
