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
Unit tests for the `iris.fileformats.grib.message._DataProxy` class.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np
from numpy.random import randint

from iris.exceptions import TranslationError
from iris.fileformats.grib._message import _DataProxy


class Test__bitmap(tests.IrisTest):
    def test_no_bitmap(self):
        section_6 = {'bitMapIndicator': 255, 'bitmap': None}
        data_proxy = _DataProxy(0, 0, 0, 0, 2)
        result = data_proxy._bitmap(section_6)
        self.assertIsNone(result)

    def test_bitmap_present(self):
        bitmap = randint(2, size=(12))
        section_6 = {'bitMapIndicator': 0, 'bitmap': bitmap}
        data_proxy = _DataProxy(0, 0, 0, 0, 2)
        result = data_proxy._bitmap(section_6)
        self.assertArrayEqual(bitmap, result)

    def test_bitmap__invalid_indicator(self):
        section_6 = {'bitMapIndicator': 100, 'bitmap': None}
        data_proxy = _DataProxy(0, 0, 0, 0, 2)
        with self.assertRaisesRegexp(TranslationError, 'unsupported bitmap'):
            data_proxy._bitmap(section_6)


if __name__ == '__main__':
    tests.main()
