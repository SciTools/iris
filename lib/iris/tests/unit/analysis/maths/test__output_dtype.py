# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the function :func:`iris.analysis.maths._output_dtype`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from itertools import product
import operator

import numpy as np

from iris.analysis.maths import _output_dtype


class Test(tests.IrisTest):
    def setUp(self):
        # Operators which result in a value of the same dtype as their
        # arguments when the arguments' dtypes are the same.
        self.same_result_ops = [
            operator.add,
            operator.sub,
            operator.mul,
            operator.pow,
            operator.floordiv,
            np.add,
            np.subtract,
            np.multiply,
            np.power,
            np.floor_divide,
        ]

        self.unary_same_result_ops = [np.abs]

        # Operators which always result in a float.
        self.float_ops = [operator.truediv, np.true_divide]

        self.unary_float_ops = [np.log, np.log2, np.log10, np.exp]

        self.all_binary_ops = self.same_result_ops + self.float_ops
        self.all_unary_ops = self.unary_same_result_ops + self.unary_float_ops

        self.dtypes = [
            np.dtype("i2"),
            np.dtype("i4"),
            np.dtype("i8"),
            np.dtype("f2"),
            np.dtype("f4"),
            np.dtype("f8"),
        ]

    def _binary_error_message(
        self,
        op,
        first_dtype,
        second_dtype,
        expected_dtype,
        result_dtype,
        in_place=False,
    ):
        msg = (
            "Output for {op.__class__.__name__} {op.__name__!r} and "
            "arguments ({dt1!r}, {dt2!r}, in_place={in_place}) "
            "was {res!r}. Expected {exp!r}."
        )
        return msg.format(
            op=op,
            dt1=first_dtype,
            dt2=second_dtype,
            exp=expected_dtype,
            res=result_dtype,
            in_place=in_place,
        )

    def _unary_error_message(
        self, op, dtype, expected_dtype, result_dtype, in_place=False
    ):
        msg = (
            "Output for {op.__class__.__name__} {op.__name__!r} and "
            "arguments ({dt!r}, in_place={in_place}) was {res!r}. "
            "Expected {exp!r}."
        )
        return msg.format(
            op=op,
            dt=dtype,
            exp=expected_dtype,
            res=result_dtype,
            in_place=in_place,
        )

    def test_same_result(self):
        # Check that the result dtype is the same as the input dtypes for
        # relevant operators.
        for dtype in self.dtypes:
            for op in self.same_result_ops:
                result_dtype = _output_dtype(op, dtype, dtype)
                self.assertEqual(
                    dtype,
                    result_dtype,
                    self._binary_error_message(op, dtype, dtype, dtype, result_dtype),
                )
            for op in self.unary_same_result_ops:
                result_dtype = _output_dtype(op, dtype)
                self.assertEqual(
                    dtype,
                    result_dtype,
                    self._unary_error_message(op, dtype, dtype, result_dtype),
                )

    def test_binary_float(self):
        # Check that the result dtype is a float for relevant operators.
        # Perform checks for a selection of cases.
        cases = [
            (np.dtype("i2"), np.dtype("i2"), np.dtype("f8")),
            (np.dtype("i2"), np.dtype("i4"), np.dtype("f8")),
            (np.dtype("i4"), np.dtype("i4"), np.dtype("f8")),
            (np.dtype("i2"), np.dtype("f2"), np.dtype("f4")),
            (np.dtype("i2"), np.dtype("f4"), np.dtype("f4")),
            (np.dtype("i8"), np.dtype("f2"), np.dtype("f8")),
            (np.dtype("f2"), np.dtype("f2"), np.dtype("f2")),
            (np.dtype("f4"), np.dtype("f4"), np.dtype("f4")),
            (np.dtype("f2"), np.dtype("f4"), np.dtype("f4")),
        ]
        for dtype1, dtype2, expected_dtype in cases:
            for op in self.float_ops:
                result_dtype = _output_dtype(op, dtype1, dtype2)
                self.assertEqual(
                    expected_dtype,
                    result_dtype,
                    self._binary_error_message(
                        op, dtype1, dtype2, expected_dtype, result_dtype
                    ),
                )

    def test_unary_float(self):
        cases = [
            (np.dtype("i2"), np.dtype("f4")),
            (np.dtype("i4"), np.dtype("f8")),
            (np.dtype("i8"), np.dtype("f8")),
            (np.dtype("f2"), np.dtype("f2")),
            (np.dtype("f4"), np.dtype("f4")),
            (np.dtype("f8"), np.dtype("f8")),
        ]
        for dtype, expected_dtype in cases:
            for op in self.unary_float_ops:
                result_dtype = _output_dtype(op, dtype)
                self.assertEqual(
                    expected_dtype,
                    result_dtype,
                    self._unary_error_message(op, dtype, expected_dtype, result_dtype),
                )

    def test_binary_float_argument(self):
        # Check that when one argument is a float dtype, a float dtype results
        # Unary operators are covered by other tests.
        dtypes = [
            np.dtype("i2"),
            np.dtype("i4"),
            np.dtype("i8"),
            np.dtype("f2"),
            np.dtype("f4"),
            np.dtype("f8"),
        ]
        expected_dtypes = [
            np.dtype("f4"),
            np.dtype("f8"),
            np.dtype("f8"),
            np.dtype("f2"),
            np.dtype("f4"),
            np.dtype("f8"),
        ]
        for op in self.all_binary_ops:
            for dtype, expected_dtype in zip(dtypes, expected_dtypes):
                result_dtype = _output_dtype(op, dtype, np.dtype("f2"))
                self.assertEqual(
                    expected_dtype,
                    result_dtype,
                    self._binary_error_message(
                        op, dtype, np.dtype("f2"), expected_dtype, result_dtype
                    ),
                )

    def test_in_place(self):
        # Check that when the in_place argument is True, the result is always
        # the same as first operand.
        for dtype1, dtype2 in product(self.dtypes, self.dtypes):
            for op in self.all_binary_ops:
                result_dtype = _output_dtype(op, dtype1, dtype2, in_place=True)
                self.assertEqual(
                    result_dtype,
                    dtype1,
                    self._binary_error_message(
                        op, dtype1, dtype2, dtype1, result_dtype, in_place=True
                    ),
                )
        for dtype in self.dtypes:
            for op in self.all_unary_ops:
                result_dtype = _output_dtype(op, dtype, in_place=True)
                self.assertEqual(
                    result_dtype,
                    dtype,
                    self._unary_error_message(
                        op, dtype, dtype, result_dtype, in_place=True
                    ),
                )

    def test_commuative(self):
        # Check that the operation is commutative if in_place is not specified.
        for dtype1, dtype2 in product(self.dtypes, self.dtypes):
            for op in self.all_binary_ops:
                result_dtype1 = _output_dtype(op, dtype1, dtype2)
                result_dtype2 = _output_dtype(op, dtype2, dtype1)
                self.assertEqual(
                    result_dtype1,
                    result_dtype2,
                    "_output_dtype is not commutative with arguments "
                    "{!r} and {!r}: {!r} != {!r}".format(
                        dtype1, dtype2, result_dtype1, result_dtype2
                    ),
                )


if __name__ == "__main__":
    tests.main()
