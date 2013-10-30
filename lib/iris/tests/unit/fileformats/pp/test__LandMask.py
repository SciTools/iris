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
"""Unit tests for the `iris.fileformats.pp._LandMask` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import numpy as np

import iris.fileformats.pp as pp


class Test_land_mask_field(tests.IrisTest):
    def test_mutibility_of_land_mask_field(self):
        self.lm = pp._LandMask(None)
        mock_field = mock.Mock(data=np.empty((3, 4, 2)))

        # Check we can change the field, and that it is used by the shape
        # property.
        self.lm.land_mask_field = mock_field
        self.assertEqual(self.lm.shape, (3, 4, 2))


class Test_shape(tests.IrisTest):
    def test_shape(self):
        lm = pp._LandMask(mock.Mock(data=np.empty((3, 4, 2))))
        self.assertEqual(lm.shape, (3, 4, 2))


class Test_equality(tests.IrisTest):
    def test_different_field_same_data(self):
        lm1 = pp._LandMask(mock.Mock(data=np.ones((3, 4, 2))))
        lm2 = pp._LandMask(mock.Mock(data=np.ones((3, 4, 2))))
        self.assertNotEqual(lm1, lm2)

    def test_same_field(self):
        mask = mock.Mock(np.ones((3, 4, 2)))
        lm1 = pp._LandMask(mask)
        lm2 = pp._LandMask(mask)
        self.assertEqual(lm1, lm2)

    def test_different_types(self):
        mask = mock.Mock(np.ones((3, 4, 2)))
        lm1 = pp._LandMask(mask)
        self.assertEqual(lm1.__eq__(None), NotImplemented)
        self.assertNotEqual(lm1, None)


class Test_land_mask(tests.IrisTest):
    def test_land_mask(self):
        lm = pp._LandMask(mock.Mock(data=np.identity(5, dtype=np.float32)))
        self.assertArrayEqual(lm.land_mask, np.identity(5, dtype=np.float32))


class Test_sea_mask(tests.IrisTest):
    def test_sea_mask(self):
        lm = pp._LandMask(mock.Mock(data=np.identity(5, dtype=np.float32)))
        self.assertArrayEqual(
            lm.sea_mask, np.logical_not(np.identity(5, dtype=np.float32)))


if __name__ == "__main__":
    tests.main()
