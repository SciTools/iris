# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the function :func:`iris.analysis.maths._inplace_common_checks`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np

from iris.analysis.maths import _inplace_common_checks
from iris.cube import Cube


class Test(tests.IrisTest):
    # `_inplace_common_checks` is a pass-through function that does not return
    # anything but will fail iff `cube` and `other` have integer dtype. Thus in
    # a sense we only want to test the failing cases. Doing so, however, leaves
    # us open to the case where currently known good cases fail silently.
    # To avoid this all the known good cases are also tested by relying on the
    # fact that functions with no return value implicitly return `None`. If
    # these currently known good cases ever changed these tests would start
    # failing and indicate something was wrong.
    def setUp(self):
        self.scalar_int = 5
        self.scalar_float = 5.5

        self.float_data = np.array([8, 9], dtype=np.float64)
        self.int_data = np.array([9, 8], dtype=np.int64)
        self.uint_data = np.array([9, 8], dtype=np.uint64)

        self.float_cube = Cube(self.float_data)
        self.int_cube = Cube(self.int_data)
        self.uint_cube = Cube(self.uint_data)

        self.op = "addition"
        self.emsg = "Cannot perform inplace {}".format(self.op)

    def test_float_cubes(self):
        result = _inplace_common_checks(
            self.float_cube, self.float_cube, self.op
        )
        self.assertIsNone(result)

    def test_int_cubes(self):
        result = _inplace_common_checks(self.int_cube, self.int_cube, self.op)
        self.assertIsNone(result)

    def test_uint_cubes(self):
        result = _inplace_common_checks(
            self.uint_cube, self.uint_cube, self.op
        )
        self.assertIsNone(result)

    def test_float_cube_int_cube(self):
        result = _inplace_common_checks(
            self.float_cube, self.int_cube, self.op
        )
        self.assertIsNone(result)

    def test_float_cube_uint_cube(self):
        result = _inplace_common_checks(
            self.float_cube, self.uint_cube, self.op
        )
        self.assertIsNone(result)

    def test_int_cube_float_cube(self):
        with self.assertRaisesRegex(ArithmeticError, self.emsg):
            _inplace_common_checks(self.int_cube, self.float_cube, self.op)

    def test_uint_cube_float_cube(self):
        with self.assertRaisesRegex(ArithmeticError, self.emsg):
            _inplace_common_checks(self.uint_cube, self.float_cube, self.op)

    def test_float_cube__scalar_int(self):
        result = _inplace_common_checks(
            self.float_cube, self.scalar_int, self.op
        )
        self.assertIsNone(result)

    def test_float_cube__scalar_float(self):
        result = _inplace_common_checks(
            self.float_cube, self.scalar_float, self.op
        )
        self.assertIsNone(result)

    def test_float_cube__int_array(self):
        result = _inplace_common_checks(
            self.float_cube, self.int_data, self.op
        )
        self.assertIsNone(result)

    def test_float_cube__float_array(self):
        result = _inplace_common_checks(
            self.float_cube, self.float_data, self.op
        )
        self.assertIsNone(result)

    def test_int_cube__scalar_int(self):
        result = _inplace_common_checks(
            self.int_cube, self.scalar_int, self.op
        )
        self.assertIsNone(result)

    def test_int_cube_uint_cube(self):
        result = _inplace_common_checks(self.int_cube, self.uint_cube, self.op)
        self.assertIsNone(result)

    def test_uint_cube_uint_cube(self):
        result = _inplace_common_checks(
            self.uint_cube, self.uint_cube, self.op
        )
        self.assertIsNone(result)

    def test_uint_cube_int_cube(self):
        with self.assertRaisesRegex(ArithmeticError, self.emsg):
            _inplace_common_checks(self.uint_cube, self.int_cube, self.op)

    def test_int_cube__scalar_float(self):
        with self.assertRaisesRegex(ArithmeticError, self.emsg):
            _inplace_common_checks(self.int_cube, self.scalar_float, self.op)

    def test_int_cube__int_array(self):
        result = _inplace_common_checks(self.int_cube, self.int_cube, self.op)
        self.assertIsNone(result)

    def test_int_cube__float_array(self):
        with self.assertRaisesRegex(ArithmeticError, self.emsg):
            _inplace_common_checks(self.int_cube, self.float_data, self.op)

    def test_uint_cube__scalar_float(self):
        with self.assertRaisesRegex(ArithmeticError, self.emsg):
            _inplace_common_checks(self.uint_cube, self.scalar_float, self.op)

    def test_uint_cube__int_array(self):
        with self.assertRaisesRegex(ArithmeticError, self.emsg):
            _inplace_common_checks(self.uint_cube, self.int_cube, self.op)

    def test_uint_cube__float_array(self):
        with self.assertRaisesRegex(ArithmeticError, self.emsg):
            _inplace_common_checks(self.uint_cube, self.float_data, self.op)


if __name__ == "__main__":
    tests.main()
