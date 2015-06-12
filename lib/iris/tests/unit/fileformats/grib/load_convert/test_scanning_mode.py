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
Test function :func:`iris.fileformats.grib._load_convert.scanning_mode.

"""

from __future__ import (absolute_import, division, print_function)

import six

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from iris.exceptions import TranslationError
from iris.fileformats.grib._load_convert import scanning_mode, ScanningMode


class Test(tests.IrisTest):
    def test_unset(self):
        expected = ScanningMode(False, False, False, False)
        self.assertEqual(scanning_mode(0x0), expected)

    def test_i_negative(self):
        expected = ScanningMode(i_negative=True, j_positive=False,
                                j_consecutive=False, i_alternative=False)
        self.assertEqual(scanning_mode(0x80), expected)

    def test_j_positive(self):
        expected = ScanningMode(i_negative=False, j_positive=True,
                                j_consecutive=False, i_alternative=False)
        self.assertEqual(scanning_mode(0x40), expected)

    def test_j_consecutive(self):
        expected = ScanningMode(i_negative=False, j_positive=False,
                                j_consecutive=True, i_alternative=False)
        self.assertEqual(scanning_mode(0x20), expected)

    def test_i_alternative(self):
        with self.assertRaises(TranslationError):
            scanning_mode(0x10)


if __name__ == '__main__':
    tests.main()
