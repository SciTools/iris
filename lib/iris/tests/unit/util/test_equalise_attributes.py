# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :func:`iris.util.equalise_attributes` function."""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np

from iris.coords import AuxCoord
from iris.cube import Cube, CubeAttrsDict
import iris.tests.stock
from iris.tests.unit.common.metadata.test_CubeMetadata import (
    _TEST_ATTRNAME,
    make_attrsdict,
)
from iris.util import equalise_attributes


class TestEqualiseAttributes(tests.IrisTest):
    def setUp(self):
        empty = Cube([])

        self.cube_no_attrs = empty.copy()

        self.cube_a1 = empty.copy()
        self.cube_a1.attributes.update({"a": 1})

        self.cube_a2 = empty.copy()
        self.cube_a2.attributes.update({"a": 2})

        self.cube_a1b5 = empty.copy()
        self.cube_a1b5.attributes.update({"a": 1, "b": 5})

        self.cube_a1b6 = empty.copy()
        self.cube_a1b6.attributes.update({"a": 1, "b": 6})

        self.cube_a2b6 = empty.copy()
        self.cube_a2b6.attributes.update({"a": 2, "b": 6})

        self.cube_b5 = empty.copy()
        self.cube_b5.attributes.update({"b": 5})

        # Array attribute values
        v1 = np.array([11, 12, 13])
        v2 = np.array([11, 9999, 13])
        self.v1 = v1
        self.v2 = v2

        self.cube_a1b5v1 = empty.copy()
        self.cube_a1b5v1.attributes.update({"a": 1, "b": 5, "v": v1})

        self.cube_a1b6v1 = empty.copy()
        self.cube_a1b6v1.attributes.update({"a": 1, "b": 6, "v": v1})

        self.cube_a1b6v2 = empty.copy()
        self.cube_a1b6v2.attributes.update({"a": 1, "b": 6, "v": v2})

    def _test(self, cubes, expect_attributes, expect_removed):
        """Test."""
        working_cubes = [cube.copy() for cube in cubes]
        original_working_list = [cube for cube in working_cubes]
        # Exercise basic operation
        actual_removed = equalise_attributes(working_cubes)
        # Check they are the same cubes
        self.assertEqual(working_cubes, original_working_list)
        # Check resulting attributes all match the expected set
        for cube in working_cubes:
            self.assertEqual(cube.attributes, expect_attributes)
        # Check removed attributes all match as expected
        self.assertEqual(len(actual_removed), len(expect_removed))
        for actual, expect in zip(actual_removed, expect_removed):
            self.assertEqual(actual, expect)
        # Check everything else remains the same
        for new_cube, old_cube in zip(working_cubes, cubes):
            cube_before_noatts = old_cube.copy()
            cube_before_noatts.attributes.clear()
            cube_after_noatts = new_cube.copy()
            cube_after_noatts.attributes.clear()
            self.assertEqual(cube_after_noatts, cube_before_noatts)

    def test_no_attrs(self):
        cubes = [self.cube_no_attrs]
        self._test(cubes, {}, [{}])

    def test_single(self):
        cubes = [self.cube_a1]
        self._test(cubes, {"a": 1}, [{}])

    def test_identical(self):
        cubes = [self.cube_a1, self.cube_a1.copy()]
        self._test(cubes, {"a": 1}, [{}, {}])

    def test_one_extra(self):
        cubes = [self.cube_a1, self.cube_a1b5.copy()]
        self._test(cubes, {"a": 1}, [{}, {"b": 5}])

    def test_one_different(self):
        cubes = [self.cube_a1b5, self.cube_a1b6]
        self._test(cubes, {"a": 1}, [{"b": 5}, {"b": 6}])

    def test_common_no_diffs(self):
        cubes = [self.cube_a1b5, self.cube_a1b5.copy()]
        self._test(cubes, {"a": 1, "b": 5}, [{}, {}])

    def test_common_all_diffs(self):
        cubes = [self.cube_a1b5, self.cube_a2b6]
        self._test(cubes, {}, [{"a": 1, "b": 5}, {"a": 2, "b": 6}])

    def test_none_common(self):
        cubes = [self.cube_a1, self.cube_b5]
        self._test(cubes, {}, [{"a": 1}, {"b": 5}])

    def test_array_extra(self):
        cubes = [self.cube_a1b6, self.cube_a1b6v1]
        self._test(cubes, {"a": 1, "b": 6}, [{}, {"v": self.v1}])

    def test_array_different(self):
        cubes = [self.cube_a1b5v1, self.cube_a1b6v2]
        self._test(cubes, {"a": 1}, [{"b": 5, "v": self.v1}, {"b": 6, "v": self.v2}])

    def test_array_same(self):
        cubes = [self.cube_a1b5v1, self.cube_a1b6v1]
        self._test(cubes, {"a": 1, "v": self.v1}, [{"b": 5}, {"b": 6}])

    @tests.skip_data
    def test_complex_nonecommon(self):
        # Example with cell methods and factories, but no common attributes.
        cubes = [
            iris.tests.stock.global_pp(),
            iris.tests.stock.hybrid_height(),
        ]
        removed = cubes[0].attributes.copy()
        self._test(cubes, {}, [removed, {}])

    @tests.skip_data
    def test_complex_somecommon(self):
        # Example with cell methods and factories, plus some common attributes.
        cubes = [iris.tests.stock.global_pp(), iris.tests.stock.simple_pp()]
        self._test(
            cubes,
            {
                "STASH": iris.fileformats.pp.STASH(model=1, section=16, item=203),
                "source": "Data from Met Office Unified Model",
            },
            [{}, {}],
        )


class TestSplitattributes:
    """Extra testing for cases where attributes differ specifically by type.

    That is, where there is a new possibility of 'mismatch' due to the newer "typing"
    of attributes as global or local.

    Specifically, it is now possible that although
    "cube1.attributes.keys() == cube2.attributes.keys()",
    AND "cube1.attributes[k] == cube2.attributes[k]" for all keys,
    YET STILL (possibly) "cube1.attributes != cube2.attributes"
    """

    @staticmethod
    def _sample_splitattrs_cube(attr_global_local):
        attrs = CubeAttrsDict(
            globals=make_attrsdict(attr_global_local[0]),
            locals=make_attrsdict(attr_global_local[1]),
        )
        return Cube([0], attributes=attrs)

    @staticmethod
    def check_equalised_result(cube1, cube2):
        equalise_attributes([cube1, cube2])
        # Note: "X" represents a missing attribute, as in test_CubeMetadata
        return [
            (
                cube1.attributes.globals.get(_TEST_ATTRNAME, "X")
                + cube1.attributes.locals.get(_TEST_ATTRNAME, "X")
            ),
            (
                cube2.attributes.globals.get(_TEST_ATTRNAME, "X")
                + cube2.attributes.locals.get(_TEST_ATTRNAME, "X")
            ),
        ]

    def test__global_and_local__bothsame(self):
        # A trivial case showing that the original globals+locals are both preserved.
        cube1 = self._sample_splitattrs_cube("AB")
        cube2 = self._sample_splitattrs_cube("AB")
        result = self.check_equalised_result(cube1, cube2)
        assert result == ["AB", "AB"]

    def test__globals_different(self):
        cube1 = self._sample_splitattrs_cube("AX")
        cube2 = self._sample_splitattrs_cube("BX")
        result = self.check_equalised_result(cube1, cube2)
        assert result == ["XX", "XX"]

    def test__locals_different(self):
        cube1 = self._sample_splitattrs_cube("XA")
        cube2 = self._sample_splitattrs_cube("XB")
        result = self.check_equalised_result(cube1, cube2)
        assert result == ["XX", "XX"]

    def test__oneglobal_onelocal__different(self):
        cube1 = self._sample_splitattrs_cube("AX")
        cube2 = self._sample_splitattrs_cube("XB")
        result = self.check_equalised_result(cube1, cube2)
        assert result == ["XX", "XX"]

    # This case fails without the split-attributes fix.
    def test__oneglobal_onelocal__same(self):
        cube1 = self._sample_splitattrs_cube("AX")
        cube2 = self._sample_splitattrs_cube("XA")
        result = self.check_equalised_result(cube1, cube2)
        assert result == ["XX", "XX"]

    def test__sameglobals_onelocal__different(self):
        cube1 = self._sample_splitattrs_cube("AB")
        cube2 = self._sample_splitattrs_cube("AX")
        result = self.check_equalised_result(cube1, cube2)
        assert result == ["XX", "XX"]

    # This case fails without the split-attributes fix.
    def test__sameglobals_onelocal__same(self):
        cube1 = self._sample_splitattrs_cube("AA")
        cube2 = self._sample_splitattrs_cube("AX")
        result = self.check_equalised_result(cube1, cube2)
        assert result == ["XX", "XX"]

    # This case fails without the split-attributes fix.
    def test__differentglobals_samelocals(self):
        cube1 = self._sample_splitattrs_cube("AC")
        cube2 = self._sample_splitattrs_cube("BC")
        result = self.check_equalised_result(cube1, cube2)
        assert result == ["XX", "XX"]


class TestNonCube:
    # Just to assert that we can do operations on non-cube components (like Coords),
    # in fact effectively, anything with a ".attributes".
    # Even though the docstring does not admit this, we test it because we put in
    # special code to preserve it when adding the split-attribute handling.
    def test(self):
        attrs = [1, 1, 2]
        coords = [
            AuxCoord([0], attributes={"a": attr, "b": "all_the_same"}) for attr in attrs
        ]
        equalise_attributes(coords)
        assert all(coord.attributes == {"b": "all_the_same"} for coord in coords)


if __name__ == "__main__":
    tests.main()
