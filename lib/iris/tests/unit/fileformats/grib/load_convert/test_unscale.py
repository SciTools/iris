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
Test function :func:`iris.fileformats.grib._load_convert.unscale.

"""

from __future__ import (absolute_import, division, print_function)

import six

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import numpy as np
import numpy.ma as ma

from iris.fileformats.grib._load_convert import unscale, _MDI as MDI

# Reference GRIB2 Regulation 92.1.12.


class Test(tests.IrisTest):
    def test_single(self):
        self.assertEqual(unscale(123, 1), 12.3)
        self.assertEqual(unscale(123, -1), 1230.0)
        self.assertEqual(unscale(123, 2), 1.23)
        self.assertEqual(unscale(123, -2), 12300.0)

    def test_single_mdi(self):
        self.assertIs(unscale(10, MDI), ma.masked)
        self.assertIs(unscale(MDI, 1), ma.masked)

    def test_array(self):
        items = [[1, [0.1, 1.2, 12.3, 123.4]],
                 [-1, [10.0, 120.0, 1230.0, 12340.0]],
                 [2, [0.01, 0.12, 1.23, 12.34]],
                 [-2, [100.0, 1200.0, 12300.0, 123400.0]]]
        values = np.array([1, 12, 123, 1234])
        for factor, expected in items:
            result = unscale(values, [factor] * values.size)
            self.assertFalse(ma.isMaskedArray(result))
            np.testing.assert_array_equal(result, np.array(expected))

    def test_array_mdi(self):
        result = unscale([1, MDI, 100, 1000], [1, 1, 1, MDI])
        self.assertTrue(ma.isMaskedArray(result))
        expected = ma.masked_values([0.1, MDI, 10.0, MDI], MDI)
        np.testing.assert_array_almost_equal(result, expected)


if __name__ == '__main__':
    tests.main()
