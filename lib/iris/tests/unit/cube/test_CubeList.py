# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.cube.CubeList` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import collections

import iris.tests as tests
import iris.tests.stock

from unittest import mock

from cf_units import Unit
import numpy as np

from iris import Constraint
from iris.cube import Cube, CubeList
from iris.coords import AuxCoord, DimCoord
import iris.coord_systems
import iris.exceptions
from iris.fileformats.pp import STASH


class Test_concatenate_cube(tests.IrisTest):
    def setUp(self):
        self.units = Unit(
            "days since 1970-01-01 00:00:00", calendar="gregorian"
        )
        self.cube1 = Cube([1, 2, 3], "air_temperature", units="K")
        self.cube1.add_dim_coord(
            DimCoord([0, 1, 2], "time", units=self.units), 0
        )

    def test_pass(self):
        self.cube2 = Cube([1, 2, 3], "air_temperature", units="K")
        self.cube2.add_dim_coord(
            DimCoord([3, 4, 5], "time", units=self.units), 0
        )
        result = CubeList([self.cube1, self.cube2]).concatenate_cube()
        self.assertIsInstance(result, Cube)

    def test_fail(self):
        units = Unit("days since 1970-01-02 00:00:00", calendar="gregorian")
        cube2 = Cube([1, 2, 3], "air_temperature", units="K")
        cube2.add_dim_coord(DimCoord([0, 1, 2], "time", units=units), 0)
        with self.assertRaises(iris.exceptions.ConcatenateError):
            CubeList([self.cube1, cube2]).concatenate_cube()

    def test_empty(self):
        exc_regexp = "can't concatenate an empty CubeList"
        with self.assertRaisesRegex(ValueError, exc_regexp):
            CubeList([]).concatenate_cube()


class Test_extract_overlapping(tests.IrisTest):
    def setUp(self):
        shape = (6, 14, 19)
        n_time, n_lat, n_lon = shape
        n_data = n_time * n_lat * n_lon
        cube = Cube(np.arange(n_data, dtype=np.int32).reshape(shape))
        coord = iris.coords.DimCoord(
            points=np.arange(n_time),
            standard_name="time",
            units="hours since epoch",
        )
        cube.add_dim_coord(coord, 0)
        cs = iris.coord_systems.GeogCS(6371229)
        coord = iris.coords.DimCoord(
            points=np.linspace(-90, 90, n_lat),
            standard_name="latitude",
            units="degrees",
            coord_system=cs,
        )
        cube.add_dim_coord(coord, 1)
        coord = iris.coords.DimCoord(
            points=np.linspace(-180, 180, n_lon),
            standard_name="longitude",
            units="degrees",
            coord_system=cs,
        )
        cube.add_dim_coord(coord, 2)
        self.cube = cube

    def test_extract_one_str_dim(self):
        cubes = iris.cube.CubeList([self.cube[2:], self.cube[:4]])
        a, b = cubes.extract_overlapping("time")
        self.assertEqual(a.coord("time"), self.cube.coord("time")[2:4])
        self.assertEqual(b.coord("time"), self.cube.coord("time")[2:4])

    def test_extract_one_list_dim(self):
        cubes = iris.cube.CubeList([self.cube[2:], self.cube[:4]])
        a, b = cubes.extract_overlapping(["time"])
        self.assertEqual(a.coord("time"), self.cube.coord("time")[2:4])
        self.assertEqual(b.coord("time"), self.cube.coord("time")[2:4])

    def test_extract_two_dims(self):
        cubes = iris.cube.CubeList([self.cube[2:, 5:], self.cube[:4, :10]])
        a, b = cubes.extract_overlapping(["time", "latitude"])
        self.assertEqual(a.coord("time"), self.cube.coord("time")[2:4])
        self.assertEqual(
            a.coord("latitude"), self.cube.coord("latitude")[5:10]
        )
        self.assertEqual(b.coord("time"), self.cube.coord("time")[2:4])
        self.assertEqual(
            b.coord("latitude"), self.cube.coord("latitude")[5:10]
        )

    def test_different_orders(self):
        cubes = iris.cube.CubeList([self.cube[::-1][:4], self.cube[:4]])
        a, b = cubes.extract_overlapping("time")
        self.assertEqual(a.coord("time"), self.cube[::-1].coord("time")[2:4])
        self.assertEqual(b.coord("time"), self.cube.coord("time")[2:4])


class Test_merge_cube(tests.IrisTest):
    def setUp(self):
        self.cube1 = Cube([1, 2, 3], "air_temperature", units="K")
        self.cube1.add_aux_coord(AuxCoord([0], "height", units="m"))

    def test_pass(self):
        cube2 = self.cube1.copy()
        cube2.coord("height").points = [1]
        result = CubeList([self.cube1, cube2]).merge_cube()
        self.assertIsInstance(result, Cube)

    def test_fail(self):
        cube2 = self.cube1.copy()
        cube2.rename("not air temperature")
        with self.assertRaises(iris.exceptions.MergeError):
            CubeList([self.cube1, cube2]).merge_cube()

    def test_empty(self):
        with self.assertRaises(ValueError):
            CubeList([]).merge_cube()

    def test_single_cube(self):
        result = CubeList([self.cube1]).merge_cube()
        self.assertEqual(result, self.cube1)
        self.assertIsNot(result, self.cube1)

    def test_repeated_cube(self):
        with self.assertRaises(iris.exceptions.MergeError):
            CubeList([self.cube1, self.cube1]).merge_cube()


class Test_merge__time_triple(tests.IrisTest):
    @staticmethod
    def _make_cube(fp, rt, t, realization=None):
        cube = Cube(np.arange(20).reshape(4, 5))
        cube.add_dim_coord(DimCoord(np.arange(5), long_name="x", units="1"), 1)
        cube.add_dim_coord(DimCoord(np.arange(4), long_name="y", units="1"), 0)
        cube.add_aux_coord(
            DimCoord(fp, standard_name="forecast_period", units="1")
        )
        cube.add_aux_coord(
            DimCoord(rt, standard_name="forecast_reference_time", units="1")
        )
        cube.add_aux_coord(DimCoord(t, standard_name="time", units="1"))
        if realization is not None:
            cube.add_aux_coord(
                DimCoord(realization, standard_name="realization", units="1")
            )
        return cube

    def test_orthogonal_with_realization(self):
        # => fp: 2; rt: 2; t: 2; realization: 2
        triples = (
            (0, 10, 1),
            (0, 10, 2),
            (0, 11, 1),
            (0, 11, 2),
            (1, 10, 1),
            (1, 10, 2),
            (1, 11, 1),
            (1, 11, 2),
        )
        en1_cubes = [
            self._make_cube(*triple, realization=1) for triple in triples
        ]
        en2_cubes = [
            self._make_cube(*triple, realization=2) for triple in triples
        ]
        cubes = CubeList(en1_cubes) + CubeList(en2_cubes)
        (cube,) = cubes.merge()
        self.assertCML(cube, checksum=False)

    def test_combination_with_realization(self):
        # => fp, rt, t: 8; realization: 2
        triples = (
            (0, 10, 1),
            (0, 10, 2),
            (0, 11, 1),
            (0, 11, 3),  # This '3' breaks the pattern.
            (1, 10, 1),
            (1, 10, 2),
            (1, 11, 1),
            (1, 11, 2),
        )
        en1_cubes = [
            self._make_cube(*triple, realization=1) for triple in triples
        ]
        en2_cubes = [
            self._make_cube(*triple, realization=2) for triple in triples
        ]
        cubes = CubeList(en1_cubes) + CubeList(en2_cubes)
        (cube,) = cubes.merge()
        self.assertCML(cube, checksum=False)

    def test_combination_with_extra_realization(self):
        # => fp, rt, t, realization: 17
        triples = (
            (0, 10, 1),
            (0, 10, 2),
            (0, 11, 1),
            (0, 11, 2),
            (1, 10, 1),
            (1, 10, 2),
            (1, 11, 1),
            (1, 11, 2),
        )
        en1_cubes = [
            self._make_cube(*triple, realization=1) for triple in triples
        ]
        en2_cubes = [
            self._make_cube(*triple, realization=2) for triple in triples
        ]
        # Add extra that is a duplicate of one of the time triples
        # but with a different realisation.
        en3_cubes = [self._make_cube(0, 10, 2, realization=3)]
        cubes = CubeList(en1_cubes) + CubeList(en2_cubes) + CubeList(en3_cubes)
        (cube,) = cubes.merge()
        self.assertCML(cube, checksum=False)

    def test_combination_with_extra_triple(self):
        # => fp, rt, t, realization: 17
        triples = (
            (0, 10, 1),
            (0, 10, 2),
            (0, 11, 1),
            (0, 11, 2),
            (1, 10, 1),
            (1, 10, 2),
            (1, 11, 1),
            (1, 11, 2),
        )
        en1_cubes = [
            self._make_cube(*triple, realization=1) for triple in triples
        ]
        # Add extra time triple on the end.
        en2_cubes = [
            self._make_cube(*triple, realization=2)
            for triple in triples + ((1, 11, 3),)
        ]
        cubes = CubeList(en1_cubes) + CubeList(en2_cubes)
        (cube,) = cubes.merge()
        self.assertCML(cube, checksum=False)


class Test_xml(tests.IrisTest):
    def setUp(self):
        self.cubes = CubeList([Cube(np.arange(3)), Cube(np.arange(3))])

    def test_byteorder_default(self):
        self.assertIn("byteorder", self.cubes.xml())

    def test_byteorder_false(self):
        self.assertNotIn("byteorder", self.cubes.xml(byteorder=False))

    def test_byteorder_true(self):
        self.assertIn("byteorder", self.cubes.xml(byteorder=True))


class Test_extract(tests.IrisTest):
    def setUp(self):
        self.scalar_cubes = CubeList()
        for i in range(5):
            for letter in "abcd":
                self.scalar_cubes.append(Cube(i, long_name=letter))

    def test_scalar_cube_name_constraint(self):
        # Test the name based extraction of a CubeList containing scalar cubes.
        res = self.scalar_cubes.extract("a")
        expected = CubeList([Cube(i, long_name="a") for i in range(5)])
        self.assertEqual(res, expected)

    def test_scalar_cube_data_constraint(self):
        # Test the extraction of a CubeList containing scalar cubes
        # when using a cube_func.
        val = 2
        constraint = iris.Constraint(cube_func=lambda c: c.data == val)
        res = self.scalar_cubes.extract(constraint)
        expected = CubeList([Cube(val, long_name=letter) for letter in "abcd"])
        self.assertEqual(res, expected)


class ExtractMixin:
    # Choose "which" extract method to test.
    # Effectively "abstract" -- inheritor must define this property :
    #   method_name = 'extract_cube' / 'extract_cubes'

    def setUp(self):
        self.cube_x = Cube(0, long_name="x")
        self.cube_y = Cube(0, long_name="y")
        self.cons_x = Constraint("x")
        self.cons_y = Constraint("y")
        self.cons_any = Constraint(cube_func=lambda cube: True)
        self.cons_none = Constraint(cube_func=lambda cube: False)

    def check_extract(self, cubes, constraints, expected):
        # Check that extracting a cubelist with the given arguments has the
        # expected result.
        # 'expected' and the operation results can be:
        #  * None
        #  * a single cube
        #  * a list of cubes --> cubelist (with cubes matching)
        #  * string --> a ConstraintMatchException matching the string
        cubelist = CubeList(cubes)
        method = getattr(cubelist, self.method_name)
        if isinstance(expected, str):
            with self.assertRaisesRegex(
                iris.exceptions.ConstraintMismatchError, expected
            ):
                method(constraints)
        else:
            result = method(constraints)
            if expected is None:
                self.assertIsNone(result)
            elif isinstance(expected, Cube):
                self.assertIsInstance(result, Cube)
                self.assertEqual(result, expected)
            elif isinstance(expected, list):
                self.assertIsInstance(result, CubeList)
                self.assertEqual(result, expected)
            else:
                msg = (
                    'Unhandled usage in "check_extract" call: '
                    '"expected" arg has type {}, value {}.'
                )
                raise ValueError(msg.format(type(expected), expected))


class Test_extract_cube(ExtractMixin, tests.IrisTest):
    method_name = "extract_cube"

    def test_empty(self):
        self.check_extract([], self.cons_x, "Got 0 cubes .* expecting 1")

    def test_single_cube_ok(self):
        self.check_extract([self.cube_x], self.cons_x, self.cube_x)

    def test_single_cube_fail__too_few(self):
        self.check_extract(
            [self.cube_x], self.cons_y, "Got 0 cubes .* expecting 1"
        )

    def test_single_cube_fail__too_many(self):
        self.check_extract(
            [self.cube_x, self.cube_y],
            self.cons_any,
            "Got 2 cubes .* expecting 1",
        )

    def test_string_as_constraint(self):
        # Check that we can use a string, that converts to a constraint
        # ( via "as_constraint" ).
        self.check_extract([self.cube_x], "x", self.cube_x)

    def test_none_as_constraint(self):
        # Check that we can use a None, that converts to a constraint
        # ( via "as_constraint" ).
        self.check_extract([self.cube_x], None, self.cube_x)

    def test_constraint_in_list__fail(self):
        # Check that we *cannot* use [constraint]
        msg = "cannot be cast to a constraint"
        with self.assertRaisesRegex(TypeError, msg):
            self.check_extract([], [self.cons_x], [])

    def test_multi_cube_ok(self):
        self.check_extract(
            [self.cube_x, self.cube_y], self.cons_x, self.cube_x
        )  # NOTE: returns a cube

    def test_multi_cube_fail__too_few(self):
        self.check_extract(
            [self.cube_x, self.cube_y],
            self.cons_none,
            "Got 0 cubes .* expecting 1",
        )

    def test_multi_cube_fail__too_many(self):
        self.check_extract(
            [self.cube_x, self.cube_y],
            self.cons_any,
            "Got 2 cubes .* expecting 1",
        )


class ExtractCubesMixin(ExtractMixin):
    method_name = "extract_cubes"


class Test_extract_cubes__noconstraint(ExtractCubesMixin, tests.IrisTest):
    """Test with an empty list of constraints."""

    def test_empty(self):
        self.check_extract([], [], [])

    def test_single_cube(self):
        self.check_extract([self.cube_x], [], [])

    def test_multi_cubes(self):
        self.check_extract([self.cube_x, self.cube_y], [], [])


class ExtractCubesSingleConstraintMixin(ExtractCubesMixin):
    """
    Common code for testing extract_cubes with a single constraint.
    Generalised, so that we can do the same tests for a "bare" constraint,
    and a list containing a single [constraint].

    """

    # Effectively "abstract" -- inheritor must define this property :
    #   wrap_test_constraint_as_list_of_one = True / False

    def check_extract(self, cubes, constraint, result):
        # Overload standard test operation.
        if self.wrap_test_constraint_as_list_of_one:
            constraint = [constraint]
        super().check_extract(cubes, constraint, result)

    def test_empty(self):
        self.check_extract([], self.cons_x, "Got 0 cubes .* expecting 1")

    def test_single_cube_ok(self):
        self.check_extract(
            [self.cube_x], self.cons_x, [self.cube_x]
        )  # NOTE: always returns list NOT cube

    def test_single_cube__fail_mismatch(self):
        self.check_extract(
            [self.cube_x], self.cons_y, "Got 0 cubes .* expecting 1"
        )

    def test_multi_cube_ok(self):
        self.check_extract(
            [self.cube_x, self.cube_y], self.cons_x, [self.cube_x]
        )  # NOTE: always returns list NOT cube

    def test_multi_cube__fail_too_few(self):
        self.check_extract(
            [self.cube_x, self.cube_y],
            self.cons_none,
            "Got 0 cubes .* expecting 1",
        )

    def test_multi_cube__fail_too_many(self):
        self.check_extract(
            [self.cube_x, self.cube_y],
            self.cons_any,
            "Got 2 cubes .* expecting 1",
        )


class Test_extract_cubes__bare_single_constraint(
    ExtractCubesSingleConstraintMixin, tests.IrisTest
):
    """Testing with a single constraint as the argument."""

    wrap_test_constraint_as_list_of_one = False


class Test_extract_cubes__list_single_constraint(
    ExtractCubesSingleConstraintMixin, tests.IrisTest
):
    """Testing with a list of one constraint as the argument."""

    wrap_test_constraint_as_list_of_one = True


class Test_extract_cubes__multi_constraints(ExtractCubesMixin, tests.IrisTest):
    """
    Testing when the 'constraints' arg is a list of multiple constraints.
    """

    def test_empty(self):
        # Always fails.
        self.check_extract(
            [], [self.cons_x, self.cons_any], "Got 0 cubes .* expecting 1"
        )

    def test_single_cube_ok(self):
        # Possible if the one cube matches all the constraints.
        self.check_extract(
            [self.cube_x],
            [self.cons_x, self.cons_any],
            [self.cube_x, self.cube_x],
        )

    def test_single_cube__fail_too_few(self):
        self.check_extract(
            [self.cube_x],
            [self.cons_x, self.cons_y],
            "Got 0 cubes .* expecting 1",
        )

    def test_multi_cube_ok(self):
        self.check_extract(
            [self.cube_x, self.cube_y],
            [self.cons_y, self.cons_x],  # N.B. reverse order !
            [self.cube_y, self.cube_x],
        )

    def test_multi_cube_castable_constraint_args(self):
        # Check with args that *aren't* constraints, but can be converted
        # ( via "as_constraint" ).
        self.check_extract(
            [self.cube_x, self.cube_y],
            ["y", "x", self.cons_y],
            [self.cube_y, self.cube_x, self.cube_y],
        )

    # NOTE: not bothering to check we can cast a 'None', as it will anyway
    # fail with multiple input cubes.

    def test_multi_cube__fail_too_few(self):
        self.check_extract(
            [self.cube_x, self.cube_y],
            [self.cons_x, self.cons_y, self.cons_none],
            "Got 0 cubes .* expecting 1",
        )

    def test_multi_cube__fail_too_many(self):
        self.check_extract(
            [self.cube_x, self.cube_y],
            [self.cons_x, self.cons_y, self.cons_any],
            "Got 2 cubes .* expecting 1",
        )


class Test_iteration(tests.IrisTest):
    def setUp(self):
        self.scalar_cubes = CubeList()
        for i in range(5):
            for letter in "abcd":
                self.scalar_cubes.append(Cube(i, long_name=letter))

    def test_iterable(self):
        self.assertTrue(isinstance(self.scalar_cubes, collections.Iterable))

    def test_iteration(self):
        letters = "abcd" * 5
        for i, cube in enumerate(self.scalar_cubes):
            self.assertEqual(cube.long_name, letters[i])


class TestPrint(tests.IrisTest):
    def setUp(self):
        self.cubes = CubeList([iris.tests.stock.lat_lon_cube()])

    def test_summary(self):
        expected = (
            "0: unknown / (unknown)       "
            "          (latitude: 3; longitude: 4)"
        )
        self.assertEqual(str(self.cubes), expected)

    def test_summary_name_unit(self):
        self.cubes[0].long_name = "aname"
        self.cubes[0].units = "1"
        expected = (
            "0: aname / (1)       "
            "                  (latitude: 3; longitude: 4)"
        )
        self.assertEqual(str(self.cubes), expected)

    def test_summary_stash(self):
        self.cubes[0].attributes["STASH"] = STASH.from_msi("m01s00i004")
        expected = (
            "0: m01s00i004 / (unknown)       "
            "       (latitude: 3; longitude: 4)"
        )
        self.assertEqual(str(self.cubes), expected)


class TestRealiseData(tests.IrisTest):
    def test_realise_data(self):
        # Simply check that calling CubeList.realise_data is calling
        # _lazy_data.co_realise_cubes.
        mock_cubes_list = [mock.Mock(ident=count) for count in range(3)]
        test_cubelist = CubeList(mock_cubes_list)
        call_patch = self.patch("iris._lazy_data.co_realise_cubes")
        test_cubelist.realise_data()
        # Check it was called once, passing cubes as *args.
        self.assertEqual(
            call_patch.call_args_list, [mock.call(*mock_cubes_list)]
        )


if __name__ == "__main__":
    tests.main()
