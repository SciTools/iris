# (C) British Crown Copyright 2014, Met Office
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
Unit tests for the `iris.fileformats.grib.message._DataProxy` class.

"""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np
from numpy.random import randint

from iris.exceptions import TranslationError
from iris.fileformats.grib._message import _DataProxy
from iris.tests.unit.fileformats.grib import _make_test_message


class Test__bitmap(tests.IrisTest):
    def setUp(self):
        self.bitmap = randint(2, size=(12))
        self.values = np.arange(12)

    def test_no_bitmap(self):
        message = _make_test_message({6: {'bitMapIndicator': 255,
                                          'bitmap': None},
                                      7: {'codedValues': self.values}})
        data_proxy = _DataProxy((12,), '', '', message, '')
        expected = None
        result = data_proxy._bitmap(message.sections[6])
        self.assertEqual(expected, result)

    def test_bitmap_present(self):
        message = _make_test_message({6: {'bitMapIndicator': 0,
                                          'bitmap': self.bitmap},
                                      7: {'codedValues': self.values}})
        data_proxy = _DataProxy((12,), '', '', message, '')
        result = data_proxy._bitmap(message.sections[6])
        self.assertArrayEqual(self.bitmap, result)

    def test_bitmap__invalid_indicator(self):
        message = _make_test_message({6: {'bitMapIndicator': 100,
                                          'bitmap': None},
                                      7: {'codedValues': self.values}})
        data_proxy = _DataProxy((12,), '', '', message, '')
        with self.assertRaisesRegexp(TranslationError, 'unsupported bitmap'):
            data_proxy._bitmap(message.sections[6])


if __name__ == '__main__':
    tests.main()
