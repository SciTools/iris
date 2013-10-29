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
"""Unit tests for the :class:`iris.analysis.MEAN` class instance."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np
import numpy.ma as ma

from iris.analysis import MEAN


class Test_aggregate(tests.IrisTest):
    def setUp(self):
        self.array = ma.array([[1, 2, 3],
                               [3, 4, 5]],
                              mask=[[False, True, False],
                                    [True, False, False]],
                              dtype=np.float64)
        self.expected_mean_axis0 = ma.array([1., 4., 4.], mask=None)
        self.expected_mean_axis1 = ma.array([2., 4.5], mask=None)

    def _test(self, mdtol=None, result_axis_a=None, result_axis_b=None):

        if result_axis_a is None:
            result_axis_a = self.expected_mean_axis0
        if result_axis_b is None:
            result_axis_b = self.expected_mean_axis1

        mean = MEAN.aggregate(self.array, 0, mdtol=mdtol)
        self.assertMaskedArrayAlmostEqual(mean, result_axis_a)

        mean = MEAN.aggregate(self.array, 1, mdtol=mdtol)
        self.assertMaskedArrayAlmostEqual(mean, result_axis_b)

    def test_masked_notol(self):
        self._test()

    def test_masked_above_tol(self):
        self._test(mdtol=0.55)

    def test_masked_below_tol(self):
        self._test(
            mdtol=0.45,
            result_axis_a=ma.array([1., 4., 4.], mask=[True, True, False]))

    def test_masked_below_tol_alt(self):
        self._test(
            mdtol=0.1,
            result_axis_a=ma.array([1., 4., 4.], mask=[True, True, False]),
            result_axis_b=ma.array([2., 4.5], mask=[True, True]))

    def test_unmasked_with_mdtol(self):
        mean = MEAN.aggregate(self.array.data, 0, mdtol=0.5)
        self.assertArrayAlmostEqual(mean, np.array([2., 3., 4.]))

        mean = MEAN.aggregate(self.array.data, 1, mdtol=0.5)
        self.assertArrayAlmostEqual(mean, np.array([2., 4.]))

    def test_unmasked_no_mdtol(self):
        mean = MEAN.aggregate(self.array.data, 0)
        self.assertArrayAlmostEqual(mean, np.array([2., 3., 4.]))

        mean = MEAN.aggregate(self.array.data, 1)
        self.assertArrayAlmostEqual(mean, np.array([2., 4.]))


if __name__ == "__main__":
    tests.main()
