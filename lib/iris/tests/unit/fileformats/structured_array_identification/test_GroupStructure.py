# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the
:mod:`iris.fileformats._structured_array_identification.GroupStructure` class.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np

from iris.fileformats._structured_array_identification import (
    ArrayStructure,
    GroupStructure,
)


def regular_array_structures(shape, names="abcdefg"):
    # Construct column major appropriate ArrayStructures for the given
    # shape.
    running_product = 1
    array_structures = {}
    for name, dim_len in zip(names, shape):
        array_structures[name] = ArrayStructure(
            running_product, np.arange(dim_len)
        )
        running_product *= dim_len
    return array_structures


class TestGroupStructure_from_component_arrays(tests.IrisTest):
    def test_different_sizes(self):
        arrays = {"a": np.arange(6), "b": np.arange(5)}
        msg = "All array elements must have the same size."
        with self.assertRaisesRegex(ValueError, msg):
            GroupStructure.from_component_arrays(arrays)

    def test_structure_creation(self):
        # Test that the appropriate dictionary containing ArrayStructures is
        # computed when constructing a GroupStructure from_component_arrays.
        array = np.arange(6)
        expected_structure = {"a": ArrayStructure.from_array(array)}

        grp = GroupStructure.from_component_arrays({"a": array})

        self.assertEqual(grp.length, 6)
        self.assertEqual(grp._cmpt_structure, expected_structure)


class TestGroupStructure_possible_structures(tests.IrisTest):
    def test_simple_3d_structure(self):
        # Construct a structure representing a (3, 2, 4) group and assert
        # that the result is of the expected form.
        array_structures = {
            "a": ArrayStructure(1, [1, -1, 2]),
            "b": ArrayStructure(3, [1, -1]),
            "c": ArrayStructure(6, [1, -1, 2, 3]),
        }
        structure = GroupStructure(24, array_structures, array_order="f")
        expected = (
            [
                ("a", array_structures["a"]),
                ("b", array_structures["b"]),
                ("c", array_structures["c"]),
            ],
        )
        self.assertEqual(structure.possible_structures(), expected)

    def assert_potentials(self, length, array_structures, expected):
        structure = GroupStructure(length, array_structures, array_order="f")
        allowed = structure.possible_structures()
        names = [
            [name for (name, _) in allowed_structure]
            for allowed_structure in allowed
        ]
        self.assertEqual(names, expected)

    def test_multiple_potentials(self):
        # More than one potential dimension for dim 1.
        array_structures = regular_array_structures((4, 2, 3))
        array_structures["shared b"] = ArrayStructure(4, [-10, 4])
        self.assert_potentials(
            24, array_structures, [["a", "b", "c"], ["a", "shared b", "c"]]
        )

    def test_alternate_potentials(self):
        # More than one potential dimension for dim 1.
        array_structures = regular_array_structures((4, 2, 3))
        array_structures.update(regular_array_structures((6, 4), names="xy"))
        self.assert_potentials(
            24, array_structures, [["x", "y"], ["a", "b", "c"]]
        )

    def test_shared_first_dimension(self):
        # One 2d potential as well as one 3d, using the same first dimension.
        array_structures = regular_array_structures((4, 2, 3))
        array_structures["bc combined"] = ArrayStructure(4, np.arange(6))
        self.assert_potentials(
            24, array_structures, [["a", "bc combined"], ["a", "b", "c"]]
        )

    def test_non_viable_element(self):
        # One 2d potential as well as one 3d, using the same first dimension.
        array_structures = regular_array_structures((4, 2, 3))
        array_structures.pop("c")
        array_structures["strange_length"] = ArrayStructure(4, np.arange(5))
        self.assert_potentials(24, array_structures, [])

    def test_completely_unstructured_element(self):
        # One of the arrays is entirely unstructured.
        array_structures = regular_array_structures((4, 2, 3))
        array_structures["unstructured"] = None
        self.assert_potentials(24, array_structures, [["a", "b", "c"]])


class TestGroupStructure_build_arrays(tests.IrisTest):
    def assert_built_array(self, name, result, expected):
        ex_arr, ex_dims = expected
        re_arr, re_dims = result[name]
        self.assertEqual(ex_dims, re_dims)
        self.assertArrayEqual(ex_arr, re_arr)

    def test_build_arrays_regular_f_order(self):
        # Construct simple orthogonal 1d array structures, adding a trailing
        # dimension to the second, and assert the result of build_arrays
        # produces the required result.
        elements = regular_array_structures((2, 3))

        a = elements["a"].construct_array(6)
        b = elements["b"].construct_array(6)
        # Make b 2 dimensional.
        b = np.vstack([b, b + 100]).T

        grp = GroupStructure(6, elements, array_order="f")

        result = grp.build_arrays((2, 3), {"a": a, "b": b})
        self.assert_built_array("a", result, ([0, 1], (0,)))
        self.assert_built_array(
            "b", result, ([[0, 100], [1, 101], [2, 102]], (1,))
        )

    def test_build_arrays_unstructured(self):
        # Check that an unstructured array gets reshaped appropriately.
        grp = GroupStructure(6, {"a": None}, array_order="c")
        orig = np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3)
        r = grp.build_arrays((2, 3), {"a": orig.flatten(order="c")})
        self.assert_built_array("a", r, (orig, (0, 1)))

    def test_build_arrays_unstructured_ndim_f_order(self):
        # Passing an unstructured array to build_arrays, should result in the
        # appropriately shaped array, plus any trailing dimensions.
        grp = GroupStructure(6, {"a": None}, array_order="f")
        orig = np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3)
        orig = np.dstack([orig, orig + 10])
        r = grp.build_arrays((2, 3), {"a": orig.reshape((-1, 2), order="f")})
        self.assert_built_array("a", r, (orig, (0, 1)))

    def test_build_arrays_unstructured_ndim_c_order(self):
        # Passing an unstructured array to build_arrays, should result in the
        # appropriately shaped array, plus any trailing dimensions.
        grp = GroupStructure(6, {"a": None}, array_order="c")
        orig = np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3)
        orig = np.dstack([orig, orig + 10])
        r = grp.build_arrays((2, 3), {"a": orig.reshape((-1, 2), order="c")})
        self.assert_built_array("a", r, (orig, (0, 1)))

    def test_structured_array_not_applicable(self):
        # Just because an array has a possible structure, does not mean it
        # gets used. Check that 'd' which would make a good 1D array, doesn't
        # get used in a specific shape.
        elements = regular_array_structures((2, 2, 3))
        elements["d"] = ArrayStructure(3, np.arange(4))
        grp = GroupStructure(12, elements, array_order="f")

        d = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]).reshape(
            (3, 4), order="f"
        )
        expected = np.array([[[0, 1, 2], [0, 2, 3]], [[0, 1, 3], [1, 2, 3]]])
        r = grp.build_arrays(
            (2, 2, 3),
            {
                "a": np.arange(12),
                "b": np.arange(12),
                "c": np.arange(12),
                "d": d.flatten(order="f"),
            },
        )
        self.assert_built_array("d", r, (expected, (0, 1, 2)))


if __name__ == "__main__":
    tests.main()
