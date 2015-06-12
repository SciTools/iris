# (C) British Crown Copyright 2014 - 2015, Met Office
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
"""
Unit tests for `iris.fileformats.grib._save_rules.fixup_float32_as_int32`.

"""

from __future__ import (absolute_import, division, print_function)

import six

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris.fileformats.grib._save_rules import fixup_float32_as_int32


class Test(tests.IrisTest):
    def test_positive_zero(self):
        result = fixup_float32_as_int32(0.0)
        self.assertEqual(result, 0)

    def test_negative_zero(self):
        result = fixup_float32_as_int32(-0.0)
        self.assertEqual(result, 0)

    def test_high_bit_clear_1(self):
        # Start with the float32 value for the bit pattern 0x00000001.
        result = fixup_float32_as_int32(1.401298464324817e-45)
        self.assertEqual(result, 1)

    def test_high_bit_clear_2(self):
        # Start with the float32 value for the bit pattern 0x00000002.
        result = fixup_float32_as_int32(2.802596928649634e-45)
        self.assertEqual(result, 2)

    def test_high_bit_set_1(self):
        # Start with the float32 value for the bit pattern 0x80000001.
        result = fixup_float32_as_int32(-1.401298464324817e-45)
        self.assertEqual(result, -1)

    def test_high_bit_set_2(self):
        # Start with the float32 value for the bit pattern 0x80000002.
        result = fixup_float32_as_int32(-2.802596928649634e-45)
        self.assertEqual(result, -2)


if __name__ == '__main__':
    tests.main()
