# (C) British Crown Copyright 2013 - 2014, Met Office
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
from matplotlib.mlab import ma
"""Unit tests for the `iris.tests.IrisTest` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np
from numpy.ma import array as marr


from iris.tests import IrisTest


class _Test_assertMaskedArrayAllClose(IrisTest):
    def _check_ok(self, value_a, value_b, **kwargs):
        self.assertMaskedArrayAllClose(value_a, value_b, **kwargs)

    def _check_fail(self, value_a, value_b, **kwargs):
        with self.assertRaises(AssertionError):
            self.assertMaskedArrayAllClose(value_a, value_b, **kwargs)


class Test_assertMaskedArrayAllClose__arrays(_Test_assertMaskedArrayAllClose):
    def test_simple(self):
        self._check_ok(marr([1, 2, 3, 4]),
                       marr([1, 2, 3, 4]))

    def test_bad_different(self):
        self._check_fail(marr([1, 2, 3, 4]),
                         marr([1, 2, 3, 777]))

    def test_multdim(self):
        self._check_ok(marr([[1, 2, 3], [4, 5, 6]]),
                       marr([[1, 2, 3], [4, 5, 6]]))

    def test_diff_types(self):
        self._check_ok(marr([1, 2, 3, 4]),
                       marr([1.0, 2.0, 3.0, 4.0]))

    def test_bad_size(self):
        self._check_fail(marr([1, 2, 3, 4]),
                         marr([1, 2, 3]))

    def test_bad_shape(self):
        self._check_fail(marr([1, 2, 3, 4]),
                         marr([[1, 2], [3, 4]]))

    def test_masked_same(self):
        self._check_ok(
            marr([1, 2, 3, 4], mask=[False, True, False, False]),
            marr([1, 2, 3, 4], mask=[False, True, False, False]))

    def test_masked_different(self):
        self._check_ok(
            marr([1, 77, 3, 4], mask=[False, True, False, False]),
            marr([1, 2, 3, 4], mask=[False, True, False, False]))

    def test_bad_masked_different(self):
        self._check_fail(
            marr([1, 77, 3, 999], mask=[False, True, False, False]),
            marr([1, 2, 3, 4], mask=[False, True, False, False]))

    def test_bad_masks_different(self):
        self._check_fail(
            marr([1, 2, 3, 4], mask=[False, True, False, False]),
            marr([1, 2, 3, 4], mask=[False, False, False, False]))

    def test_mask_specs_false(self):
        self._check_ok(
            marr([1, 2, 3, 4], mask=[False, False, False, False]),
            marr([1, 2, 3, 4], mask=False))

    def test_mask_specs_true(self):
        self._check_ok(
            marr([1, 2, 3, 4], mask=[True, True, True, True]),
            marr([1, 2, 3, 4], mask=True))

    def test_empty(self):
        self._check_ok(
            marr([]),
            marr([]))

    def test_bad_empty_nonempty(self):
        self._check_fail(marr([]),
                         marr([1]))

    def test_bad_empty_nonempty_masked(self):
        self._check_fail(marr([]),
                         marr([1], mask=[True]))


class Test_assertMaskedArrayAllClose__scalars(_Test_assertMaskedArrayAllClose):
    def test_simple(self):
        self._check_ok(marr(1), marr(1))

    def test_bad_different(self):
        self._check_fail(marr(1), marr(2))

    def test_diff_types(self):
        self._check_ok(marr(1), marr(1.0))

    def test_masked_same(self):
        self._check_ok(marr(1, mask=True),
                       marr(1, mask=True))

    def test_masked_difference(self):
        self._check_ok(marr(1, mask=True),
                       marr(2, mask=True))

    def test_bad_masked_unmasked(self):
        self._check_fail(marr(1, mask=True),
                         marr(1, mask=False))


class Test_assertMaskedArrayAllClose__values(_Test_assertMaskedArrayAllClose):
    def test_close(self):
        a, b = 1.0, 1.0 + 1.0e-7
        self.assertNotEqual(a, b)
        self._check_ok(marr(a), marr(b))

    def test_bad_close(self):
        a, b = 1.0, 1.0 + 1.0e-5
        self._check_fail(marr(a), marr(b))

    def test_close_atol(self):
        self._check_ok(marr(1.0), marr(1.001), atol=0.002)

    def test_bad_atol(self):
        self._check_fail(marr(1.0), marr(1.001), atol=0.0008)

    def test_close_rtol(self):
        self._check_ok(marr(100.0), marr(101.0), rtol=0.02)

    def test_bad_rtol(self):
        self._check_fail(marr(100.0), marr(101.0), rtol=0.008)

    def test_close_atol_rtol(self):
        self._check_ok(marr(100.0), marr(101.0), rtol=0.005, atol=0.51)

    def test_bad_atol_rtol(self):
        self._check_fail(marr(100.0), marr(101.0), rtol=0.005, atol=0.49)

    def test_msg_verbose(self):
        with self.assertRaises(AssertionError) as err_context:
            self._check_ok(marr(1.74), marr(2.13),
                           err_msg='Custom warning', verbose=True)
        msg = err_context.exception.message
        self.assertTrue('Not equal to tolerance' in msg)
        self.assertTrue('Custom warning' in msg)
        self.assertTrue('array(1.74)' in msg)
        self.assertTrue('array(2.13)' in msg)
        with self.assertRaises(AssertionError) as err_context:
            self._check_ok(marr(1.74), marr(2.13),
                           err_msg='Custom warning', verbose=False)
        msg = err_context.exception.message
        self.assertTrue('Not equal to tolerance' in msg)
        self.assertTrue('Custom warning' in msg)
        self.assertFalse('array(1.74)' in msg)
        self.assertFalse('array(2.13)' in msg)


if __name__ == "__main__":
    tests.main()
