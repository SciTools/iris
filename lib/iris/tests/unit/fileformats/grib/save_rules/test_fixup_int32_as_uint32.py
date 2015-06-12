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
Unit tests for `iris.fileformats.grib._save_rules.fixup_int32_as_uint32`.

"""

from __future__ import (absolute_import, division, print_function)

import six

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris.fileformats.grib._save_rules import fixup_int32_as_uint32


class Test(tests.IrisTest):
    def test_very_negative(self):
        with self.assertRaises(ValueError):
            fixup_int32_as_uint32(-0x80000000)

    def test_negative(self):
        result = fixup_int32_as_uint32(-3)
        self.assertEqual(result, 0x80000003)

    def test_zero(self):
        result = fixup_int32_as_uint32(0)
        self.assertEqual(result, 0)

    def test_positive(self):
        result = fixup_int32_as_uint32(5)
        self.assertEqual(result, 5)

    def test_very_positive(self):
        with self.assertRaises(ValueError):
            fixup_int32_as_uint32(0x80000000)


if __name__ == '__main__':
    tests.main()
