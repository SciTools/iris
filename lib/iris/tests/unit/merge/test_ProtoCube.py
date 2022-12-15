# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris._merge.ProtoCube` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from abc import ABCMeta, abstractmethod
from unittest import mock

import numpy as np
import numpy.ma as ma

import iris
from iris._merge import ProtoCube
from iris.aux_factory import HybridHeightFactory, HybridPressureFactory
from iris.coords import AuxCoord, DimCoord
from iris.exceptions import MergeError


def example_cube():
    return iris.cube.Cube(
        np.array([1, 2, 3], dtype="i4"),
        standard_name="air_temperature",
        long_name="screen_air_temp",
        var_name="airtemp",
        units="K",
        attributes={"mint": "thin"},
    )


class Mixin_register(metaclass=ABCMeta):
    @property
    def cube1(self):
        return example_cube()

    @property
    @abstractmethod
    def cube2(self):
        pass

    @property
    @abstractmethod
    def fragments(self):
        pass

    def test_default(self):
        # Test what happens when we call:
        #   ProtoCube.register(cube)
        proto_cube = ProtoCube(self.cube1)
        result = proto_cube.register(self.cube2)
        self.assertEqual(result, not self.fragments)

    def test_no_error(self):
        # Test what happens when we call:
        #   ProtoCube.register(cube, error_on_mismatch=False)
        proto_cube = ProtoCube(self.cube1)
        result = proto_cube.register(self.cube2, error_on_mismatch=False)
        self.assertEqual(result, not self.fragments)

    def test_error(self):
        # Test what happens when we call:
        #   ProtoCube.register(cube, error_on_mismatch=True)
        proto_cube = ProtoCube(self.cube1)
        if self.fragments:
            with self.assertRaises(iris.exceptions.MergeError) as cm:
                proto_cube.register(self.cube2, error_on_mismatch=True)
            error_message = str(cm.exception)
            for substr in self.fragments:
                self.assertIn(substr, error_message)
        else:
            result = proto_cube.register(self.cube2, error_on_mismatch=True)
            self.assertTrue(result)


class Test_register__match(Mixin_register, tests.IrisTest):
    @property
    def fragments(self):
        return []

    @property
    def cube2(self):
        return example_cube()


class Test_register__standard_name(Mixin_register, tests.IrisTest):
    @property
    def fragments(self):
        return ["cube.standard_name", "air_temperature", "air_density"]

    @property
    def cube2(self):
        cube = example_cube()
        cube.standard_name = "air_density"
        return cube


class Test_register__long_name(Mixin_register, tests.IrisTest):
    @property
    def fragments(self):
        return ["cube.long_name", "screen_air_temp", "Belling"]

    @property
    def cube2(self):
        cube = example_cube()
        cube.long_name = "Belling"
        return cube


class Test_register__var_name(Mixin_register, tests.IrisTest):
    @property
    def fragments(self):
        return ["cube.var_name", "'airtemp'", "'airtemp2'"]

    @property
    def cube2(self):
        cube = example_cube()
        cube.var_name = "airtemp2"
        return cube


class Test_register__units(Mixin_register, tests.IrisTest):
    @property
    def fragments(self):
        return ["cube.units", "'K'", "'C'"]

    @property
    def cube2(self):
        cube = example_cube()
        cube.units = "C"
        return cube


class Test_register__attributes_unequal(Mixin_register, tests.IrisTest):
    @property
    def fragments(self):
        return ["cube.attributes", "'mint'"]

    @property
    def cube2(self):
        cube = example_cube()
        cube.attributes["mint"] = "waffer-thin"
        return cube


class Test_register__attributes_unequal_array(Mixin_register, tests.IrisTest):
    @property
    def fragments(self):
        return ["cube.attributes", "'mint'"]

    @property
    def cube1(self):
        cube = example_cube()
        cube.attributes["mint"] = np.arange(3)
        return cube

    @property
    def cube2(self):
        cube = example_cube()
        cube.attributes["mint"] = np.arange(3) + 1
        return cube


class Test_register__attributes_superset(Mixin_register, tests.IrisTest):
    @property
    def fragments(self):
        return ["cube.attributes", "'stuffed'"]

    @property
    def cube2(self):
        cube = example_cube()
        cube.attributes["stuffed"] = "yes"
        return cube


class Test_register__attributes_multi_diff(Mixin_register, tests.IrisTest):
    @property
    def fragments(self):
        return ["cube.attributes", "'sam'", "'mint'"]

    @property
    def cube1(self):
        cube = example_cube()
        cube.attributes["ralph"] = 1
        cube.attributes["sam"] = 2
        cube.attributes["tom"] = 3
        return cube

    @property
    def cube2(self):
        cube = example_cube()
        cube.attributes["ralph"] = 1
        cube.attributes["sam"] = "mug"
        cube.attributes["tom"] = 3
        cube.attributes["mint"] = "humbug"
        return cube


class Test_register__cell_method(Mixin_register, tests.IrisTest):
    @property
    def fragments(self):
        return ["cube.cell_methods"]

    @property
    def cube2(self):
        cube = example_cube()
        cube.add_cell_method(iris.coords.CellMethod("monty", ("python",)))
        return cube


class Test_register__data_shape(Mixin_register, tests.IrisTest):
    @property
    def fragments(self):
        return ["cube.shape", "(2,)", "(3,)"]

    @property
    def cube2(self):
        cube = example_cube()
        cube = cube[1:]
        return cube


class Test_register__data_dtype(Mixin_register, tests.IrisTest):
    @property
    def fragments(self):
        return ["cube data dtype", "int32", "int8"]

    @property
    def cube2(self):
        cube = example_cube()
        cube.data = cube.data.astype(np.int8)
        return cube


class _MergeTest:
    # A mixin test class for common test methods implementation.

    # used by check routine: inheritors must implement it
    _mergetest_type = NotImplementedError

    def check_merge_fails_with_message(self):
        proto_cube = iris._merge.ProtoCube(self.cube1)
        with self.assertRaises(MergeError) as arc:
            proto_cube.register(self.cube2, error_on_mismatch=True)
        return str(arc.exception)

    def check_fail(self, *substrs):
        if isinstance(substrs, str):
            substrs = [substrs]
        msg = self.check_merge_fails_with_message()
        for substr in substrs:
            self.assertIn(substr, msg)


class Test_register__CubeSig(_MergeTest, tests.IrisTest):
    # Test potential registration failures.

    _mergetest_type = "cube"

    def setUp(self):
        self.cube1 = iris.cube.Cube(
            [1, 2, 3],
            standard_name="air_temperature",
            units="K",
            attributes={"mint": "thin"},
        )
        self.cube2 = self.cube1.copy()

    def test_noise(self):
        # Test a massive set of all defn diffs to make sure it's not noise.
        self.cube1.var_name = "Arthur"
        cube2 = self.cube1[1:]
        cube2.data = cube2.data.astype(np.int8)
        cube2.data = ma.array(cube2.data)
        cube2.standard_name = "air_pressure"
        cube2.var_name = "Nudge"
        cube2.attributes["stuffed"] = "yes"
        cube2.attributes["mint"] = "waffer-thin"
        cube2.add_cell_method(iris.coords.CellMethod("monty", ("python",)))

        # Check the actual message, so we've got a readable reference text.
        self.cube2 = cube2
        msg = self.check_merge_fails_with_message()
        self.assertString(msg, self.result_path(ext="txt"))


class Test_register__CoordSig_general(_MergeTest, tests.IrisTest):

    _mergetest_type = "coord"

    def setUp(self):
        self.cube1 = iris.cube.Cube(np.zeros((3, 3, 3)))
        self.cube2 = self.cube1.copy()

    def test_scalar_defns_one_extra(self):
        self.cube2.add_aux_coord(DimCoord([1], standard_name="latitude"))
        self.check_fail("aux_coords (scalar)", "latitude")

    def test_scalar_defns_both_extra(self):
        self.cube2.add_aux_coord(DimCoord([1], standard_name="latitude"))
        self.cube1.add_aux_coord(DimCoord([1], standard_name="longitude"))
        self.check_fail("aux_coords (scalar)", "latitude", "longitude")

    def test_vector_dim_coords_and_dims_one_extra(self):
        self.cube2.add_dim_coord(
            DimCoord([1, 2, 3], standard_name="latitude"), 0
        )
        self.check_fail("dim_coords", "latitude")

    def test_vector_dim_coords_and_dims_both_extra(self):
        self.cube2.add_dim_coord(
            DimCoord([1, 2, 3], standard_name="latitude"), 0
        )
        self.cube1.add_dim_coord(
            DimCoord([1, 2, 3], standard_name="longitude"), 0
        )
        self.check_fail("dim_coords", "latitude", "longitude")

    def test_vector_aux_coords_and_dims_one_extra(self):
        self.cube2.add_aux_coord(
            DimCoord([1, 2, 3], standard_name="latitude"), 0
        )
        self.check_fail("aux_coords (non-scalar)", "latitude")

    def test_vector_aux_coords_and_dims_both_extra(self):
        self.cube2.add_aux_coord(
            DimCoord([1, 2, 3], standard_name="latitude"), 0
        )
        self.cube1.add_aux_coord(
            DimCoord([1, 2, 3], standard_name="longitude"), 0
        )
        self.check_fail("aux_coords (non-scalar)", "latitude", "longitude")

    def test_factory_defns_one_extra(self):
        self.cube2.add_aux_factory(mock.MagicMock(spec=HybridHeightFactory))
        self.check_fail("cube.aux_factories", "differ")

    def test_factory_defns_both_extra(self):
        self.cube2.add_aux_factory(mock.MagicMock(spec=HybridHeightFactory))
        self.cube1.add_aux_factory(mock.MagicMock(spec=HybridPressureFactory))
        self.check_fail("cube.aux_factories", "differ")

    def test_factory_defns_one_missing_term(self):
        self.cube1.add_aux_factory(mock.MagicMock(spec=HybridPressureFactory))
        no_delta_factory = mock.MagicMock(spec=HybridPressureFactory)
        no_delta_factory.delta = None
        self.cube2.add_aux_factory(no_delta_factory)

        self.check_fail("cube.aux_factories", "differ")

    def test_noise(self):
        cube2 = self.cube2

        # scalar
        cube2.add_aux_coord(DimCoord([1], long_name="liff"))
        cube2.add_aux_coord(DimCoord([1], long_name="life"))
        cube2.add_aux_coord(DimCoord([1], long_name="like"))

        self.cube1.add_aux_coord(DimCoord([1], var_name="ming"))
        self.cube1.add_aux_coord(DimCoord([1], var_name="mong"))
        self.cube1.add_aux_coord(DimCoord([1], var_name="moog"))

        # aux
        cube2.add_dim_coord(DimCoord([1, 2, 3], standard_name="latitude"), 0)
        cube2.add_dim_coord(DimCoord([1, 2, 3], standard_name="longitude"), 1)
        cube2.add_dim_coord(DimCoord([1, 2, 3], standard_name="altitude"), 2)

        self.cube1.add_dim_coord(
            DimCoord([1, 2, 3], long_name="equinimity"), 0
        )
        self.cube1.add_dim_coord(
            DimCoord([1, 2, 3], long_name="equinomity"), 1
        )
        self.cube1.add_dim_coord(
            DimCoord([1, 2, 3], long_name="equinumity"), 2
        )

        # dim
        cube2.add_aux_coord(DimCoord([1, 2, 3], var_name="one"), 0)
        cube2.add_aux_coord(DimCoord([1, 2, 3], var_name="two"), 1)
        cube2.add_aux_coord(DimCoord([1, 2, 3], var_name="three"), 2)

        self.cube1.add_aux_coord(DimCoord([1, 2, 3], long_name="ay"), 0)
        self.cube1.add_aux_coord(DimCoord([1, 2, 3], long_name="bee"), 1)
        self.cube1.add_aux_coord(DimCoord([1, 2, 3], long_name="cee"), 2)

        # factory
        cube2.add_aux_factory(mock.MagicMock(spec=HybridHeightFactory))
        self.cube1.add_aux_factory(mock.MagicMock(spec=HybridPressureFactory))

        # Check the actual message, so we've got a readable reference text.
        self.cube2 = cube2
        msg = self.check_merge_fails_with_message()
        self.assertString(msg, self.result_path(ext="txt"))


class _MergeTest_coordprops(_MergeTest):
    # A mixin test class for common coordinate properties tests.

    # This must be implemented by inheritors.
    _mergetest_type = NotImplementedError

    def test_nochange(self):
        # This should simply succeed.
        proto_cube = iris._merge.ProtoCube(self.cube1)
        proto_cube.register(self.cube2, error_on_mismatch=True)

    def _props_fail(self, *terms):
        self.check_fail(
            self._mergetest_type, self.coord_to_change.name(), *terms
        )

    def test_standard_name(self):
        self.coord_to_change.standard_name = "soil_temperature"
        self._props_fail("air_temperature", "soil_temperature")

    def test_long_name(self):
        self.coord_to_change.long_name = "alternate_name"
        self._props_fail("air_temperature")

    def test_var_name(self):
        self.coord_to_change.var_name = "alternate_name"
        self._props_fail("air_temperature")

    def test_units(self):
        self.coord_to_change.units = "m"
        self._props_fail("air_temperature")

    def test_attrs_unequal(self):
        self.coord_to_change.attributes["att_a"] = 99
        self._props_fail("air_temperature")

    def test_attrs_set(self):
        self.coord_to_change.attributes["att_extra"] = 101
        self._props_fail("air_temperature")

    def test_coord_system(self):
        self.coord_to_change.coord_system = mock.Mock()
        self._props_fail("air_temperature")


class Test_register__CoordSig_scalar(_MergeTest_coordprops, tests.IrisTest):

    _mergetest_type = "aux_coords (scalar)"

    def setUp(self):
        self.cube1 = iris.cube.Cube(np.zeros((3, 3, 3)))
        self.cube1.add_aux_coord(
            DimCoord(
                [1],
                standard_name="air_temperature",
                long_name="eg_scalar",
                var_name="t1",
                units="K",
                attributes={"att_a": 1, "att_b": 2},
                coord_system=None,
            )
        )
        self.coord_to_change = self.cube1.coord("air_temperature")
        self.cube2 = self.cube1.copy()


class _MergeTest_coordprops_vect(_MergeTest_coordprops):
    # A derived mixin test class.
    # Adds extra props test for aux+dim coords (test points, bounds + dims)
    _mergetest_type = NotImplementedError
    _coord_typename = NotImplementedError

    def test_points(self):
        self.coord_to_change.points = self.coord_to_change.points + 1.0
        self.check_fail(self._mergetest_type, "air_temperature")

    def test_bounds(self):
        self.coord_to_change.bounds = self.coord_to_change.bounds + 1.0
        self.check_fail(self._mergetest_type, "air_temperature")

    def test_dims(self):
        self.cube2.remove_coord(self.coord_to_change)
        cube2_add_method = getattr(self.cube2, "add_" + self._coord_typename)
        cube2_add_method(self.coord_to_change, (1,))
        self.check_fail(self._mergetest_type, "mapping")


class Test_register__CoordSig_dim(_MergeTest_coordprops_vect, tests.IrisTest):

    _mergetest_type = "dim_coords"
    _coord_typename = "dim_coord"

    def setUp(self):
        self.cube1 = iris.cube.Cube(np.zeros((3, 3)))
        self.cube1.add_dim_coord(
            DimCoord(
                [15, 25, 35],
                bounds=[[10, 20], [20, 30], [30, 40]],
                standard_name="air_temperature",
                long_name="eg_scalar",
                var_name="t1",
                units="K",
                attributes={"att_a": 1, "att_b": 2},
                coord_system=None,
            ),
            (0,),
        )
        self.coord_to_change = self.cube1.coord("air_temperature")
        self.cube2 = self.cube1.copy()

    def test_circular(self):
        # Extra failure mode that only applies to dim coords
        self.coord_to_change.circular = True
        self.check_fail(self._mergetest_type, "air_temperature")


class Test_register__CoordSig_aux(_MergeTest_coordprops_vect, tests.IrisTest):

    _mergetest_type = "aux_coords (non-scalar)"
    _coord_typename = "aux_coord"

    def setUp(self):
        self.cube1 = iris.cube.Cube(np.zeros((3, 3)))
        self.cube1.add_aux_coord(
            AuxCoord(
                [65, 45, 85],
                bounds=[[60, 70], [40, 50], [80, 90]],
                standard_name="air_temperature",
                long_name="eg_scalar",
                var_name="t1",
                units="K",
                attributes={"att_a": 1, "att_b": 2},
                coord_system=None,
            ),
            (0,),
        )
        self.coord_to_change = self.cube1.coord("air_temperature")
        self.cube2 = self.cube1.copy()


if __name__ == "__main__":
    tests.main()
