# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for
:func:`iris.fileformats.netcdf._translate_constraints_to_var_callback`.

"""

import pytest

import iris
from iris.fileformats.cf import CFDataVariable
from iris.fileformats.netcdf.loader import _translate_constraints_to_var_callback
from iris.tests import _shared_utils


class Test:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.data_variables = [
            CFDataVariable("var1", mocker.MagicMock(standard_name="x_wind")),
            CFDataVariable("var2", mocker.MagicMock(standard_name="y_wind")),
            CFDataVariable("var1", mocker.MagicMock(long_name="x component of wind")),
            CFDataVariable(
                "var1",
                mocker.MagicMock(
                    standard_name="x_wind", long_name="x component of wind"
                ),
            ),
            CFDataVariable("var1", mocker.MagicMock()),
        ]

    def test_multiple_constraints(self):
        constrs = [
            iris.NameConstraint(standard_name="x_wind"),
            iris.NameConstraint(var_name="var2"),
        ]
        callback = _translate_constraints_to_var_callback(constrs)
        result = [callback(var) for var in self.data_variables]
        _shared_utils.assert_array_equal(result, [True, True, False, True, False])

    def test_multiple_constraints_invalid(self):
        constrs = [
            iris.NameConstraint(standard_name="x_wind"),
            iris.NameConstraint(var_name="var1", STASH="m01s00i024"),
        ]
        result = _translate_constraints_to_var_callback(constrs)
        assert result is None

    def test_multiple_constraints__multiname(self, mocker):
        # Modify the first constraint to require BOTH var-name and std-name match
        constrs = [
            iris.NameConstraint(standard_name="x_wind", var_name="var1"),
            iris.NameConstraint(var_name="var2"),
        ]
        callback = _translate_constraints_to_var_callback(constrs)
        # Add 2 extra vars: one passes both name checks, and the other does not
        vars = self.data_variables + [
            CFDataVariable("var1", mocker.MagicMock(standard_name="x_wind")),
            CFDataVariable("var1", mocker.MagicMock(standard_name="air_pressure")),
        ]
        result = [callback(var) for var in vars]
        _shared_utils.assert_array_equal(
            result, [True, True, False, True, False, True, False]
        )

    def test_non_name_constraint(self):
        constr = iris.AttributeConstraint(STASH="m01s00i002")
        result = _translate_constraints_to_var_callback(constr)
        assert result is None

    def test_str_constraint(self):
        result = _translate_constraints_to_var_callback("x_wind")
        assert result is None

    def test_constaint_with_name(self):
        constr = iris.Constraint(name="x_wind")
        result = _translate_constraints_to_var_callback(constr)
        assert result is None

    def test_name_constraint_standard_name(self):
        constr = iris.NameConstraint(standard_name="x_wind")
        callback = _translate_constraints_to_var_callback(constr)
        result = [callback(var) for var in self.data_variables]
        _shared_utils.assert_array_equal(result, [True, False, False, True, False])

    def test_name_constraint_long_name(self):
        constr = iris.NameConstraint(long_name="x component of wind")
        callback = _translate_constraints_to_var_callback(constr)
        result = [callback(var) for var in self.data_variables]
        _shared_utils.assert_array_equal(result, [False, False, True, True, False])

    def test_name_constraint_var_name(self):
        constr = iris.NameConstraint(var_name="var1")
        callback = _translate_constraints_to_var_callback(constr)
        result = [callback(var) for var in self.data_variables]
        _shared_utils.assert_array_equal(result, [True, False, True, True, True])

    def test_name_constraint_standard_name_var_name(self):
        constr = iris.NameConstraint(standard_name="x_wind", var_name="var1")
        callback = _translate_constraints_to_var_callback(constr)
        result = [callback(var) for var in self.data_variables]
        _shared_utils.assert_array_equal(result, [True, False, False, True, False])

    def test_name_constraint_standard_name_long_name_var_name(self):
        constr = iris.NameConstraint(
            standard_name="x_wind",
            long_name="x component of wind",
            var_name="var1",
        )
        callback = _translate_constraints_to_var_callback(constr)
        result = [callback(var) for var in self.data_variables]
        _shared_utils.assert_array_equal(result, [False, False, False, True, False])

    def test_name_constraint_with_stash(self):
        constr = iris.NameConstraint(standard_name="x_wind", STASH="m01s00i024")
        result = _translate_constraints_to_var_callback(constr)
        assert result is None

    def test_no_constraints(self):
        constrs = []
        result = _translate_constraints_to_var_callback(constrs)
        assert result is None
