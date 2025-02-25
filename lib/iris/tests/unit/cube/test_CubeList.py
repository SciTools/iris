# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.cube.CubeList` class."""

import collections
import copy
from unittest import mock

from cf_units import Unit
import numpy as np
import pytest

from iris import Constraint
import iris.coord_systems
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList
import iris.exceptions
from iris.fileformats.pp import STASH
from iris.tests import _shared_utils
import iris.tests.stock

NOT_CUBE_MSG = "cannot be put in a cubelist, as it is not a Cube."
NON_ITERABLE_MSG = "object is not iterable"


class Test_append:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cubelist = iris.cube.CubeList()
        self.cube1 = iris.cube.Cube(1, long_name="foo")
        self.cube2 = iris.cube.Cube(1, long_name="bar")

    def test_pass(self):
        self.cubelist.append(self.cube1)
        assert self.cubelist[-1] == self.cube1
        self.cubelist.append(self.cube2)
        assert self.cubelist[-1] == self.cube2

    def test_fail(self):
        with pytest.raises(ValueError, match=NOT_CUBE_MSG):
            self.cubelist.append(None)


class Test_concatenate_cube:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.units = Unit("days since 1970-01-01 00:00:00", calendar="standard")
        self.cube1 = Cube([1, 2, 3], "air_temperature", units="K")
        self.cube1.add_dim_coord(DimCoord([0, 1, 2], "time", units=self.units), 0)

    def test_pass(self):
        self.cube2 = Cube([1, 2, 3], "air_temperature", units="K")
        self.cube2.add_dim_coord(DimCoord([3, 4, 5], "time", units=self.units), 0)
        result = CubeList([self.cube1, self.cube2]).concatenate_cube()
        assert isinstance(result, Cube)

    def test_fail(self):
        units = Unit("days since 1970-01-02 00:00:00", calendar="standard")
        cube2 = Cube([1, 2, 3], "air_temperature", units="K")
        cube2.add_dim_coord(DimCoord([0, 1, 2], "time", units=units), 0)
        with pytest.raises(iris.exceptions.ConcatenateError):
            CubeList([self.cube1, cube2]).concatenate_cube()

    def test_names_differ_fail(self):
        self.cube2 = Cube([1, 2, 3], "air_temperature", units="K")
        self.cube2.add_dim_coord(DimCoord([3, 4, 5], "time", units=self.units), 0)
        self.cube3 = Cube([1, 2, 3], "air_pressure", units="Pa")
        self.cube3.add_dim_coord(DimCoord([3, 4, 5], "time", units=self.units), 0)
        exc_regexp = "Cube names differ: air_temperature != air_pressure"
        with pytest.raises(iris.exceptions.ConcatenateError, match=exc_regexp):
            CubeList([self.cube1, self.cube2, self.cube3]).concatenate_cube()

    def test_empty(self):
        exc_regexp = "can't concatenate an empty CubeList"
        with pytest.raises(ValueError, match=exc_regexp):
            CubeList([]).concatenate_cube()


class Test_extend:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube1 = iris.cube.Cube(1, long_name="foo")
        self.cube2 = iris.cube.Cube(1, long_name="bar")
        self.cubelist1 = iris.cube.CubeList([self.cube1])
        self.cubelist2 = iris.cube.CubeList([self.cube2])

    def test_pass(self):
        cubelist = copy.copy(self.cubelist1)
        cubelist.extend(self.cubelist2)
        assert cubelist == self.cubelist1 + self.cubelist2
        cubelist.extend([self.cube2])
        assert cubelist[-1] == self.cube2

    def test_fail(self):
        with pytest.raises(TypeError, match=NON_ITERABLE_MSG):
            self.cubelist1.extend(self.cube1)
        with pytest.raises(TypeError, match=NON_ITERABLE_MSG):
            self.cubelist1.extend(None)
        with pytest.raises(ValueError, match=NOT_CUBE_MSG):
            self.cubelist1.extend(range(3))


class Test_extract_overlapping:
    @pytest.fixture(autouse=True)
    def _setup(self):
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
        assert a.coord("time") == self.cube.coord("time")[2:4]
        assert b.coord("time") == self.cube.coord("time")[2:4]

    def test_extract_one_list_dim(self):
        cubes = iris.cube.CubeList([self.cube[2:], self.cube[:4]])
        a, b = cubes.extract_overlapping(["time"])
        assert a.coord("time") == self.cube.coord("time")[2:4]
        assert b.coord("time") == self.cube.coord("time")[2:4]

    def test_extract_two_dims(self):
        cubes = iris.cube.CubeList([self.cube[2:, 5:], self.cube[:4, :10]])
        a, b = cubes.extract_overlapping(["time", "latitude"])
        assert a.coord("time") == self.cube.coord("time")[2:4]
        assert a.coord("latitude") == self.cube.coord("latitude")[5:10]
        assert b.coord("time") == self.cube.coord("time")[2:4]
        assert b.coord("latitude") == self.cube.coord("latitude")[5:10]

    def test_different_orders(self):
        cubes = iris.cube.CubeList([self.cube[::-1][:4], self.cube[:4]])
        a, b = cubes.extract_overlapping("time")
        assert a.coord("time") == self.cube[::-1].coord("time")[2:4]
        assert b.coord("time") == self.cube.coord("time")[2:4]


class Test_iadd:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube1 = iris.cube.Cube(1, long_name="foo")
        self.cube2 = iris.cube.Cube(1, long_name="bar")
        self.cubelist1 = iris.cube.CubeList([self.cube1])
        self.cubelist2 = iris.cube.CubeList([self.cube2])

    def test_pass(self):
        cubelist = copy.copy(self.cubelist1)
        cubelist += self.cubelist2
        assert cubelist == self.cubelist1 + self.cubelist2
        cubelist += [self.cube2]
        assert cubelist[-1] == self.cube2

    def test_fail(self):
        with pytest.raises(TypeError, match=NON_ITERABLE_MSG):
            self.cubelist1 += self.cube1
        with pytest.raises(TypeError, match=NON_ITERABLE_MSG):
            self.cubelist1 += 1.0
        with pytest.raises(ValueError, match=NOT_CUBE_MSG):
            self.cubelist1 += range(3)


class Test_insert:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube1 = iris.cube.Cube(1, long_name="foo")
        self.cube2 = iris.cube.Cube(1, long_name="bar")
        self.cubelist = iris.cube.CubeList([self.cube1] * 3)

    def test_pass(self):
        self.cubelist.insert(1, self.cube2)
        assert self.cubelist[1] == self.cube2

    def test_fail(self):
        with pytest.raises(ValueError, match=NOT_CUBE_MSG):
            self.cubelist.insert(0, None)


class Test_merge_cube:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube1 = Cube([1, 2, 3], "air_temperature", units="K")
        self.cube1.add_aux_coord(AuxCoord([0], "height", units="m"))

    def test_pass(self):
        cube2 = self.cube1.copy()
        cube2.coord("height").points = [1]
        result = CubeList([self.cube1, cube2]).merge_cube()
        assert isinstance(result, Cube)

    def test_fail(self):
        cube2 = self.cube1.copy()
        cube2.rename("not air temperature")
        with pytest.raises(iris.exceptions.MergeError):
            CubeList([self.cube1, cube2]).merge_cube()

    def test_empty(self):
        with pytest.raises(ValueError):
            CubeList([]).merge_cube()

    def test_single_cube(self):
        result = CubeList([self.cube1]).merge_cube()
        assert result == self.cube1
        assert result is not self.cube1

    def test_repeated_cube(self):
        with pytest.raises(iris.exceptions.MergeError):
            CubeList([self.cube1, self.cube1]).merge_cube()


class Test_merge__time_triple:
    @pytest.fixture(autouse=True)
    def _setup(self, request):
        self.request = request

    @staticmethod
    def _make_cube(fp, rt, t, realization=None):
        cube = Cube(np.arange(20).reshape(4, 5))
        cube.add_dim_coord(DimCoord(np.arange(5), long_name="x", units="1"), 1)
        cube.add_dim_coord(DimCoord(np.arange(4), long_name="y", units="1"), 0)
        cube.add_aux_coord(DimCoord(fp, standard_name="forecast_period", units="1"))
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
        en1_cubes = [self._make_cube(*triple, realization=1) for triple in triples]
        en2_cubes = [self._make_cube(*triple, realization=2) for triple in triples]
        cubes = CubeList(en1_cubes) + CubeList(en2_cubes)
        (cube,) = cubes.merge()
        _shared_utils.assert_CML(self.request, cube, checksum=False)

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
        en1_cubes = [self._make_cube(*triple, realization=1) for triple in triples]
        en2_cubes = [self._make_cube(*triple, realization=2) for triple in triples]
        cubes = CubeList(en1_cubes) + CubeList(en2_cubes)
        (cube,) = cubes.merge()
        _shared_utils.assert_CML(self.request, cube, checksum=False)

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
        en1_cubes = [self._make_cube(*triple, realization=1) for triple in triples]
        en2_cubes = [self._make_cube(*triple, realization=2) for triple in triples]
        # Add extra that is a duplicate of one of the time triples
        # but with a different realisation.
        en3_cubes = [self._make_cube(0, 10, 2, realization=3)]
        cubes = CubeList(en1_cubes) + CubeList(en2_cubes) + CubeList(en3_cubes)
        (cube,) = cubes.merge()
        _shared_utils.assert_CML(self.request, cube, checksum=False)

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
        en1_cubes = [self._make_cube(*triple, realization=1) for triple in triples]
        # Add extra time triple on the end.
        en2_cubes = [
            self._make_cube(*triple, realization=2)
            for triple in triples + ((1, 11, 3),)
        ]
        cubes = CubeList(en1_cubes) + CubeList(en2_cubes)
        (cube,) = cubes.merge()
        _shared_utils.assert_CML(self.request, cube, checksum=False)


class Test_setitem:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube1 = iris.cube.Cube(1, long_name="foo")
        self.cube2 = iris.cube.Cube(1, long_name="bar")
        self.cube3 = iris.cube.Cube(1, long_name="boo")
        self.cubelist = iris.cube.CubeList([self.cube1] * 3)

    def test_pass(self):
        self.cubelist[1] = self.cube2
        assert self.cubelist[1] == self.cube2
        self.cubelist[:2] = (self.cube2, self.cube3)
        assert self.cubelist == iris.cube.CubeList([self.cube2, self.cube3, self.cube1])

    def test_fail(self):
        with pytest.raises(ValueError, match=NOT_CUBE_MSG):
            self.cubelist[0] = None
        with pytest.raises(ValueError, match=NOT_CUBE_MSG):
            self.cubelist[0:2] = [self.cube3, None]

        with pytest.raises(TypeError, match=NON_ITERABLE_MSG):
            self.cubelist[:1] = 2.5
        with pytest.raises(TypeError, match=NON_ITERABLE_MSG):
            self.cubelist[:1] = self.cube1


class Test_xml:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cubes = CubeList([Cube(np.arange(3)), Cube(np.arange(3))])

    def test_byteorder_default(self):
        assert "byteorder" in self.cubes.xml()

    def test_byteorder_false(self):
        assert "byteorder" not in self.cubes.xml(byteorder=False)

    def test_byteorder_true(self):
        assert "byteorder" in self.cubes.xml(byteorder=True)


class Test_extract:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.scalar_cubes = CubeList()
        for i in range(5):
            for letter in "abcd":
                self.scalar_cubes.append(Cube(i, long_name=letter))

    def test_scalar_cube_name_constraint(self):
        # Test the name based extraction of a CubeList containing scalar cubes.
        res = self.scalar_cubes.extract("a")
        expected = CubeList([Cube(i, long_name="a") for i in range(5)])
        assert res == expected

    def test_scalar_cube_data_constraint(self):
        # Test the extraction of a CubeList containing scalar cubes
        # when using a cube_func.
        val = 2
        constraint = iris.Constraint(cube_func=lambda c: c.data == val)
        res = self.scalar_cubes.extract(constraint)
        expected = CubeList([Cube(val, long_name=letter) for letter in "abcd"])
        assert res == expected


class ExtractMixin:
    # Choose "which" extract method to test.
    # Effectively "abstract" -- inheritor must define this property :
    #   method_name = 'extract_cube' / 'extract_cubes'

    @pytest.fixture(autouse=True)
    def _setup(self):
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
            with pytest.raises(iris.exceptions.ConstraintMismatchError, match=expected):
                method(constraints)
        else:
            result = method(constraints)
            if expected is None:
                assert result is None
            elif isinstance(expected, Cube):
                assert isinstance(result, Cube)
                assert result == expected
            elif isinstance(expected, list):
                assert isinstance(result, CubeList)
                assert result == expected
            else:
                msg = (
                    'Unhandled usage in "check_extract" call: '
                    '"expected" arg has type {}, value {}.'
                )
                raise ValueError(msg.format(type(expected), expected))


class Test_extract_cube(ExtractMixin):
    method_name = "extract_cube"

    def test_empty(self):
        self.check_extract([], self.cons_x, "Got 0 cubes .* expecting 1")

    def test_single_cube_ok(self):
        self.check_extract([self.cube_x], self.cons_x, self.cube_x)

    def test_single_cube_fail__too_few(self):
        self.check_extract([self.cube_x], self.cons_y, "Got 0 cubes .* expecting 1")

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
        with pytest.raises(TypeError, match=msg):
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


class Test_extract_cubes__noconstraint(ExtractCubesMixin):
    """Test with an empty list of constraints."""

    def test_empty(self):
        self.check_extract([], [], [])

    def test_single_cube(self):
        self.check_extract([self.cube_x], [], [])

    def test_multi_cubes(self):
        self.check_extract([self.cube_x, self.cube_y], [], [])


class ExtractCubesSingleConstraintMixin(ExtractCubesMixin):
    """Common code for testing extract_cubes with a single constraint.
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
        self.check_extract([self.cube_x], self.cons_y, "Got 0 cubes .* expecting 1")

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


class Test_extract_cubes__bare_single_constraint(ExtractCubesSingleConstraintMixin):
    """Testing with a single constraint as the argument."""

    wrap_test_constraint_as_list_of_one = False


class Test_extract_cubes__list_single_constraint(ExtractCubesSingleConstraintMixin):
    """Testing with a list of one constraint as the argument."""

    wrap_test_constraint_as_list_of_one = True


class Test_extract_cubes__multi_constraints(ExtractCubesMixin):
    """Testing when the 'constraints' arg is a list of multiple constraints."""

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


class Test_iteration:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.scalar_cubes = CubeList()
        for i in range(5):
            for letter in "abcd":
                self.scalar_cubes.append(Cube(i, long_name=letter))

    def test_iterable(self):
        assert isinstance(self.scalar_cubes, collections.abc.Iterable)

    def test_iteration(self):
        letters = "abcd" * 5
        for i, cube in enumerate(self.scalar_cubes):
            assert cube.long_name == letters[i]


class TestPrint:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cubes = CubeList([iris.tests.stock.lat_lon_cube()])

    def test_summary(self):
        expected = "0: unknown / (unknown)                 (latitude: 3; longitude: 4)"
        assert str(self.cubes) == expected

    def test_summary_name_unit(self):
        self.cubes[0].long_name = "aname"
        self.cubes[0].units = "1"
        expected = "0: aname / (1)                         (latitude: 3; longitude: 4)"
        assert str(self.cubes) == expected

    def test_summary_stash(self):
        self.cubes[0].attributes["STASH"] = STASH.from_msi("m01s00i004")
        expected = "0: m01s00i004 / (unknown)              (latitude: 3; longitude: 4)"
        assert str(self.cubes) == expected


class TestRealiseData:
    def test_realise_data(self, mocker):
        # Simply check that calling CubeList.realise_data is calling
        # _lazy_data.co_realise_cubes.
        mock_cubes_list = [mock.Mock(ident=count) for count in range(3)]
        test_cubelist = CubeList(mock_cubes_list)
        call_patch = mocker.patch("iris._lazy_data.co_realise_cubes")
        test_cubelist.realise_data()
        # Check it was called once, passing cubes as *args.
        assert call_patch.call_args_list == [mock.call(*mock_cubes_list)]


class Test_CubeList_copy:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube_list = iris.cube.CubeList()
        self.copied_cube_list = self.cube_list.copy()

    def test_copy(self):
        assert isinstance(self.copied_cube_list, iris.cube.CubeList)


class TestHtmlRepr:
    """Confirm that Cubelist._repr_html_() creates a fresh
    :class:`iris.experimental.representation.CubeListRepresentation` object, and uses
    it in the expected way.

    Notes
    -----
    This only tests code connectivity.  The functionality is tested elsewhere, at
    `iris.tests.unit.experimental.representation.test_CubeListRepresentation`
    """

    @staticmethod
    def test__repr_html_(mocker):
        test_cubelist = CubeList([])

        target = "iris.experimental.representation.CubeListRepresentation"
        class_mock = mocker.patch(target)
        # Exercise the function-under-test.
        test_cubelist._repr_html_()

        assert class_mock.call_args_list == [
            # "CubeListRepresentation()" was called exactly once, with the cubelist as arg
            mock.call(test_cubelist)
        ]
        assert class_mock.return_value.repr_html.call_args_list == [
            # "CubeListRepresentation(cubelist).repr_html()" was called exactly once, with no args
            mock.call()
        ]
