# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the function
:func:`iris.fileformats.um._optimal_array_structuring.optimal_array_structure`.

"""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np

from iris.fileformats.um._optimal_array_structuring import (
    optimal_array_structure,
)


class Test__optimal_dimensioning_structure:
    pass


class Test_optimal_array_structure(tests.IrisTest):
    def _check_arrays_and_dims(self, result, spec):
        self.assertEqual(set(result.keys()), set(spec.keys()))
        for keyname in spec.keys():
            result_array, result_dims = result[keyname]
            spec_array, spec_dims = spec[keyname]
            self.assertEqual(
                result_dims,
                spec_dims,
                'element dims differ for "{}": '
                "result={!r}, expected {!r}".format(
                    keyname, result_dims, spec_dims
                ),
            )
            self.assertArrayEqual(
                result_array,
                spec_array,
                'element arrays differ for "{}": '
                "result={!r}, expected {!r}".format(
                    keyname, result_array, spec_array
                ),
            )

    def test_none(self):
        with self.assertRaises(IndexError):
            _ = optimal_array_structure([], [])

    def test_one(self):
        # A single value does not make a dimension (no length-1 dims).
        elements = [("a", np.array([1]))]
        shape, primaries, elems_and_dims = optimal_array_structure(elements)
        self.assertEqual(shape, ())
        self.assertEqual(primaries, set())
        self.assertEqual(elems_and_dims, {})

    def test_1d(self):
        elements = [("a", np.array([1, 2, 4]))]
        shape, primaries, elems_and_dims = optimal_array_structure(elements)
        self.assertEqual(shape, (3,))
        self.assertEqual(primaries, set("a"))
        self._check_arrays_and_dims(
            elems_and_dims, {"a": (np.array([1, 2, 4]), (0,))}
        )

    def test_1d_actuals(self):
        # Test use of alternate element values for array construction.
        elements = [("a", np.array([1, 2, 4]))]
        actual_values = [("a", np.array([7, 3, 9]))]
        shape, primaries, elems_and_dims = optimal_array_structure(
            elements, actual_values
        )
        self.assertEqual(shape, (3,))
        self.assertEqual(primaries, set("a"))
        self._check_arrays_and_dims(
            elems_and_dims, {"a": (np.array([7, 3, 9]), (0,))}
        )

    def test_actuals_mismatch_fail(self):
        elements = [("a", np.array([1, 2, 4]))]
        actual_values = [("b", np.array([7, 3, 9]))]
        with self.assertRaisesRegex(ValueError, "Names.* do not match.*"):
            shape, primaries, elems_and_dims = optimal_array_structure(
                elements, actual_values
            )

    def test_2d(self):
        elements = [
            ("a", np.array([2, 2, 2, 3, 3, 3])),
            ("b", np.array([7, 8, 9, 7, 8, 9])),
        ]
        shape, primaries, elems_and_dims = optimal_array_structure(elements)
        self.assertEqual(shape, (2, 3))
        self.assertEqual(primaries, set(["a", "b"]))
        self._check_arrays_and_dims(
            elems_and_dims,
            {"a": (np.array([2, 3]), (0,)), "b": (np.array([7, 8, 9]), (1,))},
        )

    def test_2d_with_element_values(self):
        # Confirm that elements values are used in the output when supplied.
        elements = [
            ("a", np.array([2, 2, 2, 3, 3, 3])),
            ("b", np.array([7, 8, 9, 7, 8, 9])),
        ]
        elements_values = [
            ("a", np.array([6, 6, 6, 8, 8, 8])),
            ("b", np.array([3, 4, 5, 3, 4, 5])),
        ]
        shape, primaries, elems_and_dims = optimal_array_structure(
            elements, elements_values
        )
        self.assertEqual(shape, (2, 3))
        self.assertEqual(primaries, set(["a", "b"]))
        self._check_arrays_and_dims(
            elems_and_dims,
            {"a": (np.array([6, 8]), (0,)), "b": (np.array([3, 4, 5]), (1,))},
        )

    def test_non_2d(self):
        # An incomplete 2d expansion just becomes 1d
        elements = [
            ("a", np.array([2, 2, 2, 3, 3])),
            ("b", np.array([7, 8, 9, 7, 8])),
        ]
        shape, primaries, elems_and_dims = optimal_array_structure(elements)
        self.assertEqual(shape, (5,))
        self.assertEqual(primaries, set())
        self._check_arrays_and_dims(
            elems_and_dims,
            {
                "a": (np.array([2, 2, 2, 3, 3]), (0,)),
                "b": (np.array([7, 8, 9, 7, 8]), (0,)),
            },
        )

    def test_degenerate(self):
        # A all-same vector does not appear in the output.
        elements = [("a", np.array([1, 2, 3])), ("b", np.array([4, 4, 4]))]
        shape, primaries, elems_and_dims = optimal_array_structure(elements)
        self.assertEqual(shape, (3,))
        self.assertEqual(primaries, set(["a"]))
        self._check_arrays_and_dims(
            elems_and_dims, {"a": (np.array([1, 2, 3]), (0,))}
        )

    def test_1d_duplicates(self):
        # When two have the same structure, the first is 'the dimension'.
        elements = [("a", np.array([1, 3, 4])), ("b", np.array([6, 7, 9]))]
        shape, primaries, elems_and_dims = optimal_array_structure(elements)
        self.assertEqual(shape, (3,))
        self.assertEqual(primaries, set("a"))
        self._check_arrays_and_dims(
            elems_and_dims,
            {
                "a": (np.array([1, 3, 4]), (0,)),
                "b": (np.array([6, 7, 9]), (0,)),
            },
        )

    def test_1d_duplicates_order(self):
        # Same as previous but reverse passed order of elements 'a' and 'b'.
        elements = [("b", np.array([6, 7, 9])), ("a", np.array([1, 3, 4]))]
        shape, primaries, elems_and_dims = optimal_array_structure(elements)
        self.assertEqual(shape, (3,))
        # The only difference is the one chosen as 'principal'
        self.assertEqual(primaries, set("b"))
        self._check_arrays_and_dims(
            elems_and_dims,
            {
                "a": (np.array([1, 3, 4]), (0,)),
                "b": (np.array([6, 7, 9]), (0,)),
            },
        )

    def test_3_way(self):
        elements = [
            ("t1", np.array([2, 3, 4])),
            ("t2", np.array([4, 5, 6])),
            ("period", np.array([9, 8, 7])),
        ]
        shape, primaries, elems_and_dims = optimal_array_structure(elements)
        self.assertEqual(shape, (3,))
        self.assertEqual(primaries, set(["t1"]))
        self._check_arrays_and_dims(
            elems_and_dims,
            {
                "t1": (np.array([2, 3, 4]), (0,)),
                "t2": (np.array([4, 5, 6]), (0,)),
                "period": (np.array([9, 8, 7]), (0,)),
            },
        )

    def test_mixed_dims(self):
        elements = [
            ("t1", np.array([1, 1, 11, 11])),
            ("t2", np.array([15, 16, 25, 26])),
            ("ft", np.array([15, 16, 15, 16])),
        ]
        shape, primaries, elems_and_dims = optimal_array_structure(elements)
        self.assertEqual(shape, (2, 2))
        self.assertEqual(primaries, set(["t1", "ft"]))
        self._check_arrays_and_dims(
            elems_and_dims,
            {
                "t1": (np.array([1, 11]), (0,)),
                "t2": (np.array([[15, 16], [25, 26]]), (0, 1)),
                "ft": (np.array([15, 16]), (1,)),
            },
        )

    def test_missing_dim(self):
        # Case with no dimension element for dimension 1.
        elements = [
            ("t1", np.array([1, 1, 11, 11])),
            ("t2", np.array([15, 16, 25, 26])),
        ]
        shape, primaries, elems_and_dims = optimal_array_structure(elements)
        self.assertEqual(shape, (4,))
        # The potential 2d nature can not be recognised.
        # 't1' is auxiliary, as it has duplicate values over the dimension.
        self.assertEqual(primaries, set(["t2"]))
        self._check_arrays_and_dims(
            elems_and_dims,
            {
                "t1": (np.array([1, 1, 11, 11]), (0,)),
                "t2": (np.array([15, 16, 25, 26]), (0,)),
            },
        )

    def test_optimal_structure_decision(self):
        # Checks the optimal structure decision logic is working correctly:
        # given the arrays we have here we would expect 'a' to be the primary
        # dimension, as it has higher priority for being supplied first.
        elements = [
            ("a", np.array([1, 1, 1, 2, 2, 2])),
            ("b", np.array([0, 1, 2, 0, 1, 2])),
            ("c", np.array([11, 11, 11, 14, 14, 14])),
            ("d", np.array([10, 10, 10, 10, 10, 10])),
        ]
        shape, primaries, elems_and_dims = optimal_array_structure(elements)
        self.assertEqual(shape, (2, 3))
        self.assertEqual(primaries, set(["a", "b"]))
        self._check_arrays_and_dims(
            elems_and_dims,
            {
                "a": (np.array([1, 2]), (0,)),
                "c": (np.array([11, 14]), (0,)),
                "b": (np.array([0, 1, 2]), (1,)),
            },
        )


if __name__ == "__main__":
    tests.main()
