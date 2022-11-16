# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the `iris.fileformats.pp._data_bytes_to_shaped_array` function.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import io
from unittest import mock

import numpy as np
import numpy.ma as ma
import pytest

import iris.fileformats.pp as pp


@pytest.mark.parametrize("data_shape", [(2, 3)])
@pytest.mark.parametrize(
    "expected_shape", [(2, 3), (3, 2), (1, 3), (2, 2), (3, 3), (2, 4)]
)
@pytest.mark.parametrize(
    "data_type", [np.float32, np.int32, np.int16, np.int8]
)
def test_data_padding__no_compression(data_shape, expected_shape, data_type):
    data = np.empty(data_shape, dtype=data_type)

    # create the field data buffer
    buffer = io.BytesIO()
    buffer.write(data)
    buffer.seek(0)
    data_bytes = buffer.read()

    lbpack = pp.SplittableInt(0, dict(n1=0, n2=1))
    boundary_packing = None
    mdi = -1
    args = (
        data_bytes,
        lbpack,
        boundary_packing,
        expected_shape,
        data_type,
        mdi,
    )
    data_length, expected_length = np.prod(data_shape), np.prod(expected_shape)

    if expected_length <= data_length:
        result = pp._data_bytes_to_shaped_array(*args)
        assert result.shape == expected_shape
    else:
        emsg = r"data containing \d+ words does not match expected length"
        with pytest.raises(ValueError, match=emsg):
            _ = pp._data_bytes_to_shaped_array(*args)


class Test__data_bytes_to_shaped_array__lateral_boundary_compression(
    tests.IrisTest
):
    def setUp(self):
        self.data_shape = 30, 40
        y_halo, x_halo, rim = 2, 3, 4

        data_len = np.prod(self.data_shape)
        decompressed = np.arange(data_len).reshape(*self.data_shape)
        decompressed *= np.arange(self.data_shape[1]) % 3 + 1

        decompressed_mask = np.zeros(self.data_shape, np.bool_)
        decompressed_mask[
            y_halo + rim : -(y_halo + rim), x_halo + rim : -(x_halo + rim)
        ] = True

        self.decompressed = ma.masked_array(
            decompressed, mask=decompressed_mask
        )

        self.north = decompressed[-(y_halo + rim) :, :]
        self.east = decompressed[
            y_halo + rim : -(y_halo + rim), -(x_halo + rim) :
        ]
        self.south = decompressed[: y_halo + rim, :]
        self.west = decompressed[
            y_halo + rim : -(y_halo + rim), : x_halo + rim
        ]

        # Get the bytes of the north, east, south, west arrays combined.
        buf = io.BytesIO()
        buf.write(self.north.copy())
        buf.write(self.east.copy())
        buf.write(self.south.copy())
        buf.write(self.west.copy())
        buf.seek(0)
        self.data_payload_bytes = buf.read()

    def test_boundary_decompression(self):
        boundary_packing = mock.Mock(rim_width=4, x_halo=3, y_halo=2)
        lbpack = mock.Mock(n1=0)
        r = pp._data_bytes_to_shaped_array(
            self.data_payload_bytes,
            lbpack,
            boundary_packing,
            self.data_shape,
            self.decompressed.dtype,
            -9223372036854775808,
        )
        r = ma.masked_array(r, np.isnan(r), fill_value=-9223372036854775808)
        self.assertMaskedArrayEqual(r, self.decompressed)


class Test__data_bytes_to_shaped_array__land_packed(tests.IrisTest):
    def setUp(self):
        # Sets up some useful arrays for use with the land/sea mask
        # decompression.
        self.land = np.array(
            [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64
        )
        sea = ~self.land.astype(np.bool_)
        self.land_masked_data = np.array([1, 3, 4.5])
        self.sea_masked_data = np.array([1, 3, 4.5, -4, 5, 0, 1, 2, 3])

        # Compute the decompressed land mask data.
        self.decomp_land_data = ma.masked_array(
            [[0, 1, 0, 0], [3, 0, 0, 0], [0, 0, 0, 4.5]],
            mask=sea,
            dtype=np.float64,
        )
        # Compute the decompressed sea mask data.
        self.decomp_sea_data = ma.masked_array(
            [[1, -10, 3, 4.5], [-10, -4, 5, 0], [1, 2, 3, -10]],
            mask=self.land,
            dtype=np.float64,
        )

        self.land_mask = mock.Mock(
            data=self.land, lbrow=self.land.shape[0], lbnpt=self.land.shape[1]
        )

    def create_lbpack(self, value):
        name_mapping = dict(n5=slice(4, None), n4=3, n3=2, n2=1, n1=0)
        return pp.SplittableInt(value, name_mapping)

    def test_no_land_mask(self):
        # Check that without a mask, it returns the raw (compressed) data.
        with mock.patch("numpy.frombuffer", return_value=np.arange(3)):
            result = pp._data_bytes_to_shaped_array(
                mock.Mock(),
                self.create_lbpack(120),
                None,
                (3, 4),
                np.dtype(">f4"),
                -999,
                mask=None,
            )
            self.assertArrayAllClose(result, np.arange(3))

    def test_land_mask(self):
        # Check basic land unpacking.
        field_data = self.land_masked_data
        result = self.check_read_data(field_data, 120, self.land_mask)
        self.assertMaskedArrayEqual(result, self.decomp_land_data)

    def test_land_masked_data_too_long(self):
        # Check land unpacking with field data that is larger than the mask.
        field_data = np.tile(self.land_masked_data, 2)
        result = self.check_read_data(field_data, 120, self.land_mask)
        self.assertMaskedArrayEqual(result, self.decomp_land_data)

    def test_sea_mask(self):
        # Check basic land unpacking.
        field_data = self.sea_masked_data
        result = self.check_read_data(field_data, 220, self.land_mask)
        self.assertMaskedArrayEqual(result, self.decomp_sea_data)

    def test_sea_masked_data_too_long(self):
        # Check sea unpacking with field data that is larger than the mask.
        field_data = np.tile(self.sea_masked_data, 2)
        result = self.check_read_data(field_data, 220, self.land_mask)
        self.assertMaskedArrayEqual(result, self.decomp_sea_data)

    def test_bad_lbpack(self):
        # Check basic land unpacking.
        field_data = self.sea_masked_data
        with self.assertRaises(ValueError):
            self.check_read_data(field_data, 320, self.land_mask)

    def check_read_data(self, field_data, lbpack, mask):
        # Calls pp._data_bytes_to_shaped_array with the necessary mocked
        # items, an lbpack instance, the correct data shape and mask instance.
        with mock.patch("numpy.frombuffer", return_value=field_data):
            data = pp._data_bytes_to_shaped_array(
                mock.Mock(),
                self.create_lbpack(lbpack),
                None,
                mask.shape,
                np.dtype(">f4"),
                -999,
                mask=mask,
            )
        return ma.masked_array(data, np.isnan(data), fill_value=-999)


if __name__ == "__main__":
    tests.main()
