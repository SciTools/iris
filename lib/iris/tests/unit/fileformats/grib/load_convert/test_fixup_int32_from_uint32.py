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
Unit tests for `iris.fileformats.grib._load_convert.fixup_int32_from_uint32`.

"""

from __future__ import (absolute_import, division, print_function)

import six

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris.fileformats.grib._load_convert import fixup_int32_from_uint32


class Test(tests.IrisTest):
    def test_negative(self):
        result = fixup_int32_from_uint32(0x80000005)
        self.assertEqual(result, -5)

    def test_negative_zero(self):
        result = fixup_int32_from_uint32(0x80000000)
        self.assertEqual(result, 0)

    def test_zero(self):
        result = fixup_int32_from_uint32(0)
        self.assertEqual(result, 0)

    def test_positive(self):
        result = fixup_int32_from_uint32(200000)
        self.assertEqual(result, 200000)

    def test_already_negative(self):
        # If we *already* have a negative value the fixup routine should
        # leave it alone.
        result = fixup_int32_from_uint32(-7)
        self.assertEqual(result, -7)


if __name__ == '__main__':
    tests.main()
