# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for
:func:`iris.fileformats.netcdf.translate_constraints_to_var_callback`.

"""

from unittest.mock import Mock

import iris
from iris.fileformats.netcdf import translate_constraints_to_var_callback

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests


class Test(tests.IrisTest):
    data_variables = [
        Mock(standard_name="x_wind", cf_name="var1"),
        Mock(standard_name="y_wind", cf_name="var2"),
        Mock(long_name="x component of wind", cf_name="var1"),
        Mock(
            standard_name="x_wind",
            long_name="x component of wind",
            cf_name="var1",
        ),
    ]

    def test_multiple_constraints(self):
        constrs = [
            iris.NameConstraint(standard_name="x_wind"),
            iris.NameConstraint(var_name="var1"),
        ]
        result = translate_constraints_to_var_callback(constrs)
        self.assertIsNone(result)

    def test_non_NameConstraint(self):
        constr = iris.AttributeConstraint(STASH="m01s00i002")
        result = translate_constraints_to_var_callback(constr)
        self.assertIsNone(result)

    def test_str_constraint(self):
        result = translate_constraints_to_var_callback("x_wind")
        self.assertIsNone(result)

    def test_Constaint_with_name(self):
        constr = iris.Constraint(name="x_wind")
        result = translate_constraints_to_var_callback(constr)
        self.assertIsNone(result)

    def test_NameConstraint_standard_name(self):
        constr = iris.NameConstraint(standard_name="x_wind")
        callback = translate_constraints_to_var_callback(constr)
        result = [callback(var) for var in self.data_variables]
        self.assertArrayEqual(result, [True, False, False, True])

    def test_NameConstraint_long_name(self):
        constr = iris.NameConstraint(long_name="x component of wind")
        callback = translate_constraints_to_var_callback(constr)
        result = [callback(var) for var in self.data_variables]
        self.assertArrayEqual(result, [False, False, True, True])

    def test_NameConstraint_var_name(self):
        constr = iris.NameConstraint(var_name="var1")
        callback = translate_constraints_to_var_callback(constr)
        result = [callback(var) for var in self.data_variables]
        self.assertArrayEqual(result, [True, False, True, True])

    def test_NameConstraint_standard_name_var_name(self):
        constr = iris.NameConstraint(standard_name="x_wind", var_name="var1")
        callback = translate_constraints_to_var_callback(constr)
        result = [callback(var) for var in self.data_variables]
        self.assertArrayEqual(result, [True, False, False, True])

    def test_NameConstraint_standard_name_long_name_var_name(self):
        constr = iris.NameConstraint(
            standard_name="x_wind",
            long_name="x component of wind",
            var_name="var1",
        )
        callback = translate_constraints_to_var_callback(constr)
        result = [callback(var) for var in self.data_variables]
        self.assertArrayEqual(result, [False, False, False, True])

    def test_NameConstraint_with_STASH(self):
        constr = iris.NameConstraint(
            standard_name="x_wind", STASH="m01s00i024"
        )
        result = translate_constraints_to_var_callback(constr)
        self.assertIsNone(result)


if __name__ == "__main__":
    tests.main()
