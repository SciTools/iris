# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test the constrained cube loading mechanism."""

import datetime

import pytest

import iris
from iris import AttributeConstraint, NameConstraint
from iris.tests import _shared_utils
import iris.tests.stock as stock

SN_AIR_POTENTIAL_TEMPERATURE = "air_potential_temperature"
SN_SPECIFIC_HUMIDITY = "specific_humidity"


# TODO: Workaround, pending #1262
def workaround_pending_1262(cubes):
    """Reverse the cube if sigma was chosen as a dim_coord."""
    for i, cube in enumerate(cubes):
        ml = cube.coord("model_level_number").points
        if ml[0] > ml[1]:
            cubes[i] = cube[::-1]


@_shared_utils.skip_data
class TestSimple:
    @pytest.fixture(autouse=True)
    def _setup(self):
        names = ["grid_latitude", "grid_longitude"]
        self.slices = iris.cube.CubeList(stock.realistic_4d().slices(names))

    def test_constraints(self):
        constraint = iris.Constraint(model_level_number=10)
        sub_list = self.slices.extract(constraint)
        assert len(sub_list) == 6

        constraint = iris.Constraint(model_level_number=[10, 22])
        sub_list = self.slices.extract(constraint)
        assert len(sub_list) == 2 * 6

        constraint = iris.Constraint(model_level_number=lambda c: (c > 30) | (c <= 3))
        sub_list = self.slices.extract(constraint)
        assert len(sub_list) == 43 * 6

        constraint = iris.Constraint(
            coord_values={"model_level_number": lambda c: c > 1000}
        )
        sub_list = self.slices.extract(constraint)
        assert len(sub_list) == 0

        constraint = iris.Constraint(model_level_number=10) & iris.Constraint(
            time=datetime.datetime(2009, 9, 9, 18, 0)
        )
        sub_list = self.slices.extract(constraint)
        assert len(sub_list) == 1

        constraint = iris.Constraint(SN_AIR_POTENTIAL_TEMPERATURE)
        sub_list = self.slices.extract(constraint)
        assert len(sub_list) == 70 * 6

    def test_coord_availability(self):
        # "model_level_number" coordinate available
        constraint = iris.Constraint(model_level_number=lambda x: True)
        result = self.slices.extract(constraint)
        assert result

        # "wibble" coordinate is not available
        constraint = iris.Constraint(wibble=lambda x: False)
        result = self.slices.extract(constraint)
        assert not result

        # "wibble" coordinate is not available
        constraint = iris.Constraint(wibble=lambda x: True)
        result = self.slices.extract(constraint)
        assert not result

        # "lambda x: False" always (confusingly) throws away the cube
        constraint = iris.Constraint(model_level_number=lambda x: False)
        result = self.slices.extract(constraint)
        assert not result

    def test_mismatched_type(self):
        constraint = iris.Constraint(model_level_number="aardvark")
        sub_list = self.slices.extract(constraint)
        assert len(sub_list) == 0

    def test_cell(self):
        cell = iris.coords.Cell(10)
        constraint = iris.Constraint(model_level_number=cell)
        sub_list = self.slices.extract(constraint)
        assert len(sub_list) == 6

    def test_cell_equal_bounds(self):
        cell = self.slices[0].coord("level_height").cell(0)
        constraint = iris.Constraint(level_height=cell)
        sub_list = self.slices.extract(constraint)
        assert len(sub_list) == 6

    def test_cell_different_bounds(self):
        cell = iris.coords.Cell(10, bound=(9, 11))
        constraint = iris.Constraint(model_level_number=cell)
        sub_list = self.slices.extract(constraint)
        assert len(sub_list) == 0


class ConstraintMixin:
    """Mix-in class for attributes & utilities common to the "normal" and "strict" test cases."""

    @pytest.fixture(autouse=True)
    def _setup_mixin(self):
        self.dec_path = _shared_utils.get_data_path(
            ["PP", "globClim1", "dec_subset.pp"]
        )
        self.theta_path = _shared_utils.get_data_path(["PP", "globClim1", "theta.pp"])

        self.humidity = iris.Constraint(SN_SPECIFIC_HUMIDITY)
        self.theta = iris.Constraint(SN_AIR_POTENTIAL_TEMPERATURE)

        # Coord based constraints
        self.level_10 = iris.Constraint(model_level_number=10)
        self.level_22 = iris.Constraint(model_level_number=22)

        # Value based coord constraint
        self.level_30 = iris.Constraint(model_level_number=30)
        self.level_gt_30_le_3 = iris.Constraint(
            model_level_number=lambda c: (c > 30) | (c <= 3)
        )
        self.invalid_inequality = iris.Constraint(
            coord_values={"model_level_number": lambda c: c > 1000}
        )

        # bound based coord constraint
        self.level_height_of_model_level_number_10 = iris.Constraint(level_height=1900)
        self.model_level_number_10_22 = iris.Constraint(model_level_number=[10, 22])

        # Invalid constraints
        self.pressure_950 = iris.Constraint(model_level_number=950)

        self.lat_30 = iris.Constraint(latitude=30)
        self.lat_gt_45 = iris.Constraint(latitude=lambda c: c > 45)


class RelaxedConstraintMixin(ConstraintMixin):
    suffix: str

    @staticmethod
    def fixup_sigma_to_be_aux(cubes):
        # XXX Fix the cubes such that the sigma coordinate is always an AuxCoord. Pending gh issue #18
        if isinstance(cubes, iris.cube.Cube):
            cubes = [cubes]

        for cube in cubes:
            sigma = cube.coord("sigma")
            sigma = iris.coords.AuxCoord.from_coord(sigma)
            cube.replace_coord(sigma)

    def assert_constraint_cml(self, request: pytest.FixtureRequest, cubes, filename):
        filename = f"{filename}_{self.suffix}.cml"
        _shared_utils.assert_CML(request, cubes, ("constrained_load", filename))

    def load_match(self, files, constraints):
        raise NotImplementedError()  # defined in subclasses

    def test_single_atomic_constraint(self, request):
        cubes = self.load_match(self.dec_path, self.level_10)
        self.fixup_sigma_to_be_aux(cubes)
        self.assert_constraint_cml(request, cubes, "all_10")

        cubes = self.load_match(self.dec_path, self.theta)
        self.assert_constraint_cml(request, cubes, "theta")

        cubes = self.load_match(self.dec_path, self.model_level_number_10_22)
        self.fixup_sigma_to_be_aux(cubes)
        workaround_pending_1262(cubes)
        self.assert_constraint_cml(request, cubes, "all_ml_10_22")

        # Check that it didn't matter that we provided sets & tuples to the model_level
        for constraint in [
            iris.Constraint(model_level_number=set([10, 22])),
            iris.Constraint(model_level_number=tuple([10, 22])),
        ]:
            cubes = self.load_match(self.dec_path, constraint)
            self.fixup_sigma_to_be_aux(cubes)
            workaround_pending_1262(cubes)
            self.assert_constraint_cml(request, cubes, "all_ml_10_22")

    def test_string_standard_name(self, request):
        cubes = self.load_match(self.dec_path, SN_AIR_POTENTIAL_TEMPERATURE)
        self.assert_constraint_cml(request, cubes, "theta")

        cubes = self.load_match(self.dec_path, [SN_AIR_POTENTIAL_TEMPERATURE])
        self.assert_constraint_cml(request, cubes, "theta")

        cubes = self.load_match(
            self.dec_path, iris.Constraint(SN_AIR_POTENTIAL_TEMPERATURE)
        )
        self.assert_constraint_cml(request, cubes, "theta")

        cubes = self.load_match(
            self.dec_path,
            iris.Constraint(SN_AIR_POTENTIAL_TEMPERATURE, model_level_number=10),
        )
        self.fixup_sigma_to_be_aux(cubes)
        self.assert_constraint_cml(request, cubes, "theta_10")

    def test_latitude_constraint(self, request):
        cubes = self.load_match(self.theta_path, self.lat_30)
        self.assert_constraint_cml(request, cubes, "theta_lat_30")

        cubes = self.load_match(self.theta_path, self.lat_gt_45)
        self.assert_constraint_cml(request, cubes, "theta_lat_gt_30")

    def test_single_expression_constraint(self, request):
        cubes = self.load_match(self.theta_path, self.theta & self.level_10)
        self.fixup_sigma_to_be_aux(cubes)
        self.assert_constraint_cml(request, cubes, "theta_10")

        cubes = self.load_match(self.theta_path, self.level_10 & self.theta)
        self.fixup_sigma_to_be_aux(cubes)
        self.assert_constraint_cml(request, cubes, "theta_10")

    def test_dual_atomic_constraint(self, request):
        cubes = self.load_match(self.dec_path, [self.theta, self.level_10])
        self.fixup_sigma_to_be_aux(cubes)
        self.assert_constraint_cml(request, cubes, "theta_and_all_10")

    def test_dual_repeated_constraint(self, request):
        cubes = self.load_match(self.dec_path, [self.theta, self.theta])
        self.fixup_sigma_to_be_aux(cubes)
        self.assert_constraint_cml(request, cubes, "theta_and_theta")

    def test_dual_expression_constraint(self, request):
        cubes = self.load_match(
            self.dec_path,
            [self.theta & self.level_10, self.level_gt_30_le_3 & self.theta],
        )
        self.fixup_sigma_to_be_aux(cubes)
        self.assert_constraint_cml(
            request, cubes, "theta_10_and_theta_level_gt_30_le_3"
        )

    def test_invalid_constraint(self, request):
        cubes = self.load_match(self.theta_path, self.pressure_950)
        self.assert_constraint_cml(request, cubes, "pressure_950")

        cubes = self.load_match(self.theta_path, self.invalid_inequality)
        self.assert_constraint_cml(request, cubes, "invalid_inequality")

    def test_inequality_constraint(self, request):
        cubes = self.load_match(self.theta_path, self.level_gt_30_le_3)
        self.assert_constraint_cml(request, cubes, "theta_gt_30_le_3")


class StrictConstraintMixin(RelaxedConstraintMixin):
    def test_single_atomic_constraint(self, request):
        cubes = self.load_match(self.theta_path, self.theta)
        self.assert_constraint_cml(request, cubes, "theta")

        cubes = self.load_match(self.theta_path, self.level_10)
        self.fixup_sigma_to_be_aux(cubes)
        self.assert_constraint_cml(request, cubes, "theta_10")

    def test_invalid_constraint(self):
        with pytest.raises(iris.exceptions.ConstraintMismatchError):
            self.load_match(self.theta_path, self.pressure_950)

    def test_dual_atomic_constraint(self, request):
        cubes = self.load_match(self.dec_path, [self.theta, self.level_10 & self.theta])
        self.fixup_sigma_to_be_aux(cubes)
        self.assert_constraint_cml(request, cubes, "theta_and_theta_10")


@_shared_utils.skip_data
class TestCubeLoadConstraint(RelaxedConstraintMixin):
    suffix = "load_match"

    def load_match(self, files, constraints):
        cubes = iris.load(files, constraints)
        if not isinstance(cubes, iris.cube.CubeList):
            raise Exception("NOT A CUBE LIST! " + str(type(cubes)))
        return cubes


@_shared_utils.skip_data
class TestCubeListConstraint(RelaxedConstraintMixin):
    suffix = "load_match"

    def load_match(self, files, constraints):
        cubes = iris.load(files).extract(constraints)
        if not isinstance(cubes, iris.cube.CubeList):
            raise Exception("NOT A CUBE LIST! " + str(type(cubes)))
        return cubes


@_shared_utils.skip_data
class TestCubeListStrictConstraint(StrictConstraintMixin):
    suffix = "load_strict"

    def load_match(self, files, constraints):
        cubes = iris.load(files).extract_cubes(constraints)
        return cubes


@_shared_utils.skip_data
class TestCubeExtract__names(ConstraintMixin):
    @pytest.fixture(autouse=True)
    def _setup(self, _setup_mixin):
        fname = iris.sample_data_path("atlantic_profiles.nc")
        self.cubes = iris.load(fname)
        cube = iris.load_cube(self.theta_path)
        # Expected names...
        self.standard_name = "air_potential_temperature"
        self.long_name = "AIR POTENTIAL TEMPERATURE"
        self.var_name = "apt"
        self.stash = "m01s00i004"
        # Configure missing names...
        cube.long_name = self.long_name
        cube.var_name = self.var_name
        # Add this cube to the mix...
        self.cubes.append(cube)
        self.index = len(self.cubes) - 1

    def test_standard_name(self):
        constraint = iris.Constraint(self.standard_name)
        result = self.cubes.extract_cube(constraint)
        assert result is not None
        assert result.standard_name == self.standard_name

    def test_long_name(self):
        constraint = iris.Constraint(self.long_name)
        result = self.cubes.extract_cube(constraint)
        assert result is not None
        assert result.long_name == self.long_name

    def test_var_name(self):
        constraint = iris.Constraint(self.var_name)
        result = self.cubes.extract_cube(constraint)
        assert result is not None
        assert result.var_name == self.var_name

    def test_stash(self):
        constraint = iris.Constraint(self.stash)
        result = self.cubes.extract_cube(constraint)
        assert result is not None
        assert str(result.attributes["STASH"]) == self.stash

    def test_unknown(self):
        cube = self.cubes[self.index]
        # Clear the cube metadata.
        cube.standard_name = None
        cube.long_name = None
        cube.var_name = None
        cube.attributes = None
        # Extract the unknown cube.
        constraint = iris.Constraint("unknown")
        result = self.cubes.extract_cube(constraint)
        assert result is not None
        assert result.name() == "unknown"


@_shared_utils.skip_data
class TestCubeExtract__name_constraint(ConstraintMixin):
    @pytest.fixture(autouse=True)
    def _setup(self, _setup_mixin):
        fname = iris.sample_data_path("atlantic_profiles.nc")
        self.cubes = iris.load(fname)
        cube = iris.load_cube(self.theta_path)
        # Expected names...
        self.standard_name = "air_potential_temperature"
        self.long_name = "air potential temperature"
        self.var_name = "apt"
        self.stash = "m01s00i004"
        # Configure missing names...
        cube.long_name = self.long_name
        cube.var_name = self.var_name
        # Add this cube to the mix...
        self.cubes.append(cube)
        self.index = len(self.cubes) - 1

    def test_standard_name(self):
        # No match.
        constraint = NameConstraint(standard_name="wibble")
        result = self.cubes.extract(constraint)
        assert not result

        # Match.
        constraint = NameConstraint(standard_name=self.standard_name)
        result = self.cubes.extract_cube(constraint)
        assert result is not None
        assert result.standard_name == self.standard_name

        # Match - callable.
        kwargs = dict(standard_name=lambda item: item.startswith("air_pot"))
        constraint = NameConstraint(**kwargs)
        result = self.cubes.extract_cube(constraint)
        assert result is not None
        assert result.standard_name == self.standard_name

    def test_standard_name__none(self):
        cube = self.cubes[self.index]
        cube.standard_name = None
        constraint = NameConstraint(standard_name=None, long_name=self.long_name)
        result = self.cubes.extract_cube(constraint)
        assert result is not None
        assert result.standard_name is None
        assert result.long_name == self.long_name

    def test_long_name(self):
        # No match.
        constraint = NameConstraint(long_name="wibble")
        result = self.cubes.extract(constraint)
        assert not result

        # Match.
        constraint = NameConstraint(long_name=self.long_name)
        result = self.cubes.extract_cube(constraint)
        assert result is not None
        assert result.long_name == self.long_name

        # Match - callable.
        kwargs = dict(
            long_name=lambda item: item is not None and item.startswith("air pot")
        )
        constraint = NameConstraint(**kwargs)
        result = self.cubes.extract_cube(constraint)
        assert result is not None
        assert result.long_name == self.long_name

    def test_long_name__none(self):
        cube = self.cubes[self.index]
        cube.long_name = None
        constraint = NameConstraint(standard_name=self.standard_name, long_name=None)
        result = self.cubes.extract_cube(constraint)
        assert result is not None
        assert result.standard_name == self.standard_name
        assert result.long_name is None

    def test_var_name(self):
        # No match.
        constraint = NameConstraint(var_name="wibble")
        result = self.cubes.extract(constraint)
        assert not result

        # Match.
        constraint = NameConstraint(var_name=self.var_name)
        result = self.cubes.extract_cube(constraint)
        assert result is not None
        assert result.var_name == self.var_name

        # Match - callable.
        kwargs = dict(var_name=lambda item: item.startswith("ap"))
        constraint = NameConstraint(**kwargs)
        result = self.cubes.extract_cube(constraint)
        assert result is not None
        assert result.var_name == self.var_name

    def test_var_name__none(self):
        cube = self.cubes[self.index]
        cube.var_name = None
        constraint = NameConstraint(standard_name=self.standard_name, var_name=None)
        result = self.cubes.extract_cube(constraint)
        assert result is not None
        assert result.standard_name == self.standard_name
        assert result.var_name is None

    def test_stash(self):
        # No match.
        constraint = NameConstraint(STASH="m01s00i444")
        result = self.cubes.extract(constraint)
        assert not result

        # Match.
        constraint = NameConstraint(STASH=self.stash)
        result = self.cubes.extract_cube(constraint)
        assert result is not None
        assert str(result.attributes["STASH"]) == self.stash

        # Match - callable.
        kwargs = dict(STASH=lambda stash: stash.item == 4)
        constraint = NameConstraint(**kwargs)
        result = self.cubes.extract_cube(constraint)
        assert result is not None

    def test_stash__none(self):
        cube = self.cubes[self.index]
        del cube.attributes["STASH"]
        constraint = NameConstraint(standard_name=self.standard_name, STASH=None)
        result = self.cubes.extract_cube(constraint)
        assert result is not None
        assert result.standard_name == self.standard_name
        assert result.attributes.get("STASH") is None

    def test_compound(self):
        # Match.
        constraint = NameConstraint(
            standard_name=self.standard_name, long_name=self.long_name
        )
        result = self.cubes.extract_cube(constraint)
        assert result is not None
        assert result.standard_name == self.standard_name

        # No match - var_name.
        constraint = NameConstraint(
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name="wibble",
        )
        result = self.cubes.extract(constraint)
        assert not result

        # Match.
        constraint = NameConstraint(
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
        )
        result = self.cubes.extract_cube(constraint)
        assert result is not None
        assert result.standard_name == self.standard_name
        assert result.long_name == self.long_name
        assert result.var_name == self.var_name

        # No match - STASH.
        constraint = NameConstraint(
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
            STASH="m01s00i444",
        )
        result = self.cubes.extract(constraint)
        assert not result

        # Match.
        constraint = NameConstraint(
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
            STASH=self.stash,
        )
        result = self.cubes.extract_cube(constraint)
        assert result is not None
        assert result.standard_name == self.standard_name
        assert result.long_name == self.long_name
        assert result.var_name == self.var_name
        assert result.var_name == self.var_name

        # No match - standard_name.
        constraint = NameConstraint(
            standard_name="wibble",
            long_name=self.long_name,
            var_name=self.var_name,
            STASH=self.stash,
        )
        result = self.cubes.extract(constraint)
        assert not result

    def test_unknown(self):
        # No match.
        constraint = NameConstraint(None, None, None, None)
        result = self.cubes.extract(constraint)
        assert not result

        # Match.
        cube = self.cubes[self.index]
        cube.standard_name = None
        cube.long_name = None
        cube.var_name = None
        cube.attributes = None
        constraint = NameConstraint(None, None, None, None)
        result = self.cubes.extract_cube(constraint)
        assert result is not None
        assert result.standard_name is None
        assert result.long_name is None
        assert result.var_name is None
        assert result.attributes.get("STASH") is None


@_shared_utils.skip_data
class TestCubeExtract(ConstraintMixin):
    @pytest.fixture(autouse=True)
    def _setup(self, _setup_mixin):
        self.cube = iris.load_cube(self.theta_path)

    def test_attribute_constraint(self, request):
        # There is no my_attribute on the cube, so ensure it returns None.
        constraint = AttributeConstraint(my_attribute="foobar")
        cube = self.cube.extract(constraint)
        assert cube is None

        orig_cube = self.cube
        # add an attribute to the cubes
        orig_cube.attributes["my_attribute"] = "foobar"

        constraint = AttributeConstraint(my_attribute="foobar")
        cube = orig_cube.extract(constraint)
        _shared_utils.assert_CML(
            request, cube, ("constrained_load", "attribute_constraint.cml")
        )

        constraint = AttributeConstraint(my_attribute="not me")
        cube = orig_cube.extract(constraint)
        assert cube is None

        kwargs = dict(my_attribute=lambda val: val.startswith("foo"))
        constraint = AttributeConstraint(**kwargs)
        cube = orig_cube.extract(constraint)
        _shared_utils.assert_CML(
            request, cube, ("constrained_load", "attribute_constraint.cml")
        )

        kwargs = dict(my_attribute=lambda val: not val.startswith("foo"))
        constraint = AttributeConstraint(**kwargs)
        cube = orig_cube.extract(constraint)
        assert cube is None

        kwargs = dict(my_non_existant_attribute="hello world")
        constraint = AttributeConstraint(**kwargs)
        cube = orig_cube.extract(constraint)
        assert cube is None

    def test_standard_name(self):
        r = iris.Constraint(SN_AIR_POTENTIAL_TEMPERATURE)
        assert self.cube.extract(r).standard_name, SN_AIR_POTENTIAL_TEMPERATURE

        r = iris.Constraint("wibble")
        assert self.cube.extract(r) is None

    def test_empty_data(self):
        # Ensure that the process of WHERE does not load data if there
        # was empty data to start with...
        cube = self.cube
        assert cube.has_lazy_data()
        cube = self.cube.extract(self.level_10)
        assert cube.has_lazy_data()
        cube = self.cube.extract(self.level_10).extract(self.level_10)
        assert cube.has_lazy_data()

    def test_non_existent_coordinate(self):
        # Check the behaviour when a constraint is given for a coordinate which does not exist/span a dimension
        assert self.cube[0, :, :].extract(self.level_10) is None

        assert self.cube.extract(iris.Constraint(wibble=10)) is None


@_shared_utils.skip_data
class TestConstraints(ConstraintMixin):
    def test_constraint_expressions(self):
        rt = repr(self.theta)
        rl10 = repr(self.level_10)

        rt_l10 = repr(self.theta & self.level_10)
        expr = "ConstraintCombination(%s, %s, <built-in function %s>)" % (
            rt,
            rl10,
            "and_",
        )
        assert expr == rt_l10

    def test_string_repr(self):
        rt = repr(iris.Constraint(SN_AIR_POTENTIAL_TEMPERATURE))
        assert rt == "Constraint(name='%s')" % SN_AIR_POTENTIAL_TEMPERATURE

        rt = repr(iris.Constraint(SN_AIR_POTENTIAL_TEMPERATURE, model_level_number=10))
        assert (
            rt
            == "Constraint(name='%s', coord_values={'model_level_number': 10})"
            % SN_AIR_POTENTIAL_TEMPERATURE
        )

    def test_number_of_raw_cubes(self):
        # Test the constraints generate the correct number of raw cubes.
        raw_cubes = iris.load_raw(self.theta_path)
        assert len(raw_cubes) == 38

        raw_cubes = iris.load_raw(self.theta_path, [self.level_10])
        assert len(raw_cubes) == 1

        raw_cubes = iris.load_raw(self.theta_path, [self.theta])
        assert len(raw_cubes) == 38

        raw_cubes = iris.load_raw(self.dec_path, [self.level_30])
        assert len(raw_cubes) == 4

        raw_cubes = iris.load_raw(self.dec_path, [self.theta])
        assert len(raw_cubes) == 38


class TestBetween:
    def run_test(self, function, numbers, results):
        for number, result in zip(numbers, results):
            assert function(number) == result

    def test_le_ge(self):
        function = iris.util.between(2, 4)
        numbers = [1, 2, 3, 4, 5]
        results = [False, True, True, True, False]
        self.run_test(function, numbers, results)

    def test_lt_gt(self):
        function = iris.util.between(2, 4, rh_inclusive=False, lh_inclusive=False)
        numbers = [1, 2, 3, 4, 5]
        results = [False, False, True, False, False]
        self.run_test(function, numbers, results)

    def test_le_gt(self):
        function = iris.util.between(2, 4, rh_inclusive=False)
        numbers = [1, 2, 3, 4, 5]
        results = [False, True, True, False, False]
        self.run_test(function, numbers, results)

    def test_lt_ge(self):
        function = iris.util.between(2, 4, lh_inclusive=False)
        numbers = [1, 2, 3, 4, 5]
        results = [False, False, True, True, False]
        self.run_test(function, numbers, results)
