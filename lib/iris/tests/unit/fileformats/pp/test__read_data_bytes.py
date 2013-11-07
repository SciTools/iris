# (C) British Crown Copyright 2013, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""Unit tests for the `iris.fileformats.pp._read_data_bytes` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import numpy as np

import iris.fileformats.pp as pp


class Test_read_data__land_packed(tests.IrisTest):
    def setUp(self):
        # Sets up some useful arrays for use with the land/sea mask
        # decompression.
        self.land = np.array([[0, 1, 0, 0],
                              [1, 0, 0, 0],
                              [0, 0, 0, 1]], dtype=np.float64)
        sea = ~self.land.astype(np.bool)
        self.land_masked_data = np.array([1, 3, 4.5])
        self.sea_masked_data = np.array([1, 3, 4.5, -4, 5, 0, 1, 2, 3])

        # Compute the decompressed land mask data.
        self.decomp_land_data = np.ma.masked_array([[0, 1, 0, 0],
                                                    [3, 0, 0, 0],
                                                    [0, 0, 0, 4.5]],
                                                   mask=sea,
                                                   dtype=np.float64)
        # Compute the decompressed sea mask data.
        self.decomp_sea_data = np.ma.masked_array([[1, -10, 3, 4.5],
                                                   [-10, -4, 5, 0],
                                                   [1, 2, 3, -10]],
                                                  mask=self.land,
                                                  dtype=np.float64)

        self.land_mask = mock.Mock(data=self.land,
                                   lbrow=self.land.shape[0],
                                   lbnpt=self.land.shape[1])

    def create_lbpack(self, value):
        name_mapping = dict(n5=slice(4, None), n4=3, n3=2, n2=1, n1=0)
        return pp.SplittableInt(value, name_mapping)

    def test_no_land_mask(self):
        with mock.patch('numpy.frombuffer',
                        return_value=np.arange(3)):
            with self.assertRaises(ValueError):
                pp._read_data_bytes(mock.Mock(), self.create_lbpack(120),
                                    (3, 4), np.dtype('>f4'),
                                    -999, mask=None)

    def test_land_mask(self):
        # Check basic land unpacking.
        field_data = self.land_masked_data
        result = self.check_read_data(field_data, 120, self.land_mask)
        # XXX Needn't be almost...
        self.assertMaskedArrayAlmostEqual(result, self.decomp_land_data)

    def test_land_masked_data_too_long(self):
        # Check land unpacking with field data that is larger than the mask.
        field_data = np.tile(self.land_masked_data, 2)
        result = self.check_read_data(field_data, 120, self.land_mask)
        # XXX Needn't be almost...
        self.assertMaskedArrayAlmostEqual(result, self.decomp_land_data)

    def test_sea_mask(self):
        # Check basic land unpacking.
        field_data = self.sea_masked_data
        result = self.check_read_data(field_data, 220, self.land_mask)
        # XXX Needn't be almost...
        self.assertMaskedArrayAlmostEqual(result, self.decomp_sea_data)

    def test_sea_masked_data_too_long(self):
        # Check sea unpacking with field data that is larger than the mask.
        field_data = np.tile(self.sea_masked_data, 2)
        result = self.check_read_data(field_data, 220, self.land_mask)
        # XXX Needn't be almost...
        self.assertMaskedArrayAlmostEqual(result, self.decomp_sea_data)

    def test_bad_lbpack(self):
        # Check basic land unpacking.
        field_data = self.sea_masked_data
        with self.assertRaises(ValueError):
            self.check_read_data(field_data, 320, self.land_mask)

    def check_read_data(self, field_data, lbpack, mask):
        # Calls pp._read_data_bytes with the necessary mocked items, an lbpack
        # instance, the correct data shape and mask instance.
        with mock.patch('numpy.frombuffer', return_value=field_data):
            return pp._read_data_bytes(mock.Mock(), self.create_lbpack(lbpack),
                                       mask.shape, np.dtype('>f4'),
                                       -999, mask=mask)


if __name__ == "__main__":
    tests.main()
