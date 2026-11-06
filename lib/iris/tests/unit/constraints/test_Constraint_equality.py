# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for equality testing of different constraint types."""

from iris._constraints import AttributeConstraint, Constraint, NameConstraint


class Test_Constraint__hash__:
    def test_empty(self):
        c1 = Constraint()
        c2 = Constraint()
        assert hash(c1) == hash(c1)
        assert hash(c1) != hash(c2)


class Test_Constraint__eq__:
    def test_empty_same(self):
        c1 = Constraint()
        c2 = Constraint()
        assert c1 == c2
        assert c1 is not c2

    def test_emptyname_same(self):
        c1 = Constraint("")
        c2 = Constraint("")
        assert c1 == c2

    def test_empty_emptyname_differ(self):
        c1 = Constraint()
        c2 = Constraint("")
        assert c1 != c2

    def test_names_same(self):
        c1 = Constraint("a")
        c2 = Constraint("a")
        assert c1 == c2

    def test_names_differ(self):
        c1 = Constraint("a")
        c2 = Constraint("b")
        assert c1 != c2

    def test_funcs_same(self):
        # *Same* functions match
        def func(cube):
            return False

        c1 = Constraint(cube_func=func)
        c2 = Constraint(cube_func=func)
        assert c1 == c2

    def test_funcs_differ(self):
        # Identical but different funcs do not match.
        c1 = Constraint(cube_func=lambda c: False)
        c2 = Constraint(cube_func=lambda c: False)
        assert c1 != c2

    def test_coord_names_same(self):
        c1 = Constraint(some_coordname=3)
        c2 = Constraint(some_coordname=3)
        assert c1 == c2

    def test_coord_names_differ(self):
        c1 = Constraint(some_coordname_A=3)
        c2 = Constraint(some_coordname_B=3)
        assert c1 != c2

    def test_coord_values_differ(self):
        c1 = Constraint(some_coordname=3)
        c2 = Constraint(some_coordname=4)
        assert c1 != c2

    def test_coord_orders_differ(self):
        # We *could* maybe ignore Coordinate order, but at present we don't.
        c1 = Constraint(coordname_1=1, coordname_2=2)
        c2 = Constraint(coordname_2=2, coordname_1=1)
        assert c1 != c2

    def test_coord_values_functions_same(self):
        def func(coord):
            return False

        c1 = Constraint(some_coordname=func)
        c2 = Constraint(some_coordname=func)
        assert c1 == c2

    def test_coord_values_functions_differ(self):
        # Identical functions are not the same.
        c1 = Constraint(some_coordname=lambda c: True)
        c2 = Constraint(some_coordname=lambda c: True)
        assert c1 != c2

    def test_coord_values_and_keys_same(self):
        # **kwargs and 'coord_values=' are combined without distinction.
        c1 = Constraint(coord_values={"a": [2, 3]})
        c2 = Constraint(a=[2, 3])
        assert c1 == c2


class Test_AttributeConstraint__hash__:
    def test_empty(self):
        c1 = AttributeConstraint()
        c2 = AttributeConstraint()
        assert hash(c1) == hash(c1)
        assert hash(c1) != hash(c2)


class Test_AttributeConstraint__eq__:
    def test_empty_same(self):
        c1 = AttributeConstraint()
        c2 = AttributeConstraint()
        assert c1 == c2
        assert c1 is not c2

    def test_attribute_plain_empty_diff(self):
        c1 = AttributeConstraint()
        c2 = Constraint()
        assert c1 != c2

    def test_names_same(self):
        c1 = AttributeConstraint(a=1)
        c2 = AttributeConstraint(a=1)
        assert c1 == c2

    def test_names_diff(self):
        c1 = AttributeConstraint(a=1)
        c2 = AttributeConstraint(a=1, b=1)
        assert c1 != c2

    def test_values_diff(self):
        c1 = AttributeConstraint(a=1, b=1)
        c2 = AttributeConstraint(a=1, b=2)
        assert c1 != c2

    def test_func_same(self):
        def func(attrs):
            return False

        c1 = AttributeConstraint(a=func)
        c2 = AttributeConstraint(a=func)
        assert c1 == c2

    def test_func_diff(self):
        c1 = AttributeConstraint(a=lambda a: False)
        c2 = AttributeConstraint(a=lambda a: False)
        assert c1 != c2


class Test_NameConstraint__hash__:
    def test_empty(self):
        c1 = NameConstraint()
        c2 = NameConstraint()
        assert hash(c1) == hash(c1)
        assert hash(c1) != hash(c2)


class Test_NameConstraint__eq__:
    def test_empty_same(self):
        c1 = NameConstraint()
        c2 = NameConstraint()
        assert c1 == c2
        assert c1 is not c2

    def test_attribute_plain_empty_diff(self):
        c1 = NameConstraint()
        c2 = Constraint()
        assert c1 != c2

    def test_names_same(self):
        c1 = NameConstraint(standard_name="air_temperature")
        c2 = NameConstraint(standard_name="air_temperature")
        assert c1 == c2

    def test_full_same(self):
        c1 = NameConstraint(
            standard_name="air_temperature",
            long_name="temp",
            var_name="tair",
            STASH="m01s02i003",
        )
        c2 = NameConstraint(
            standard_name="air_temperature",
            long_name="temp",
            var_name="tair",
            STASH="m01s02i003",
        )
        assert c1 == c2

    def test_missing_diff(self):
        c1 = NameConstraint(standard_name="air_temperature", var_name="tair")
        c2 = NameConstraint(standard_name="air_temperature")
        assert c1 != c2

    def test_standard_name_diff(self):
        c1 = NameConstraint(standard_name="air_temperature")
        c2 = NameConstraint(standard_name="height")
        assert c1 != c2

    def test_long_name_diff(self):
        c1 = NameConstraint(long_name="temp")
        c2 = NameConstraint(long_name="t3")
        assert c1 != c2

    def test_var_name_diff(self):
        c1 = NameConstraint(var_name="tair")
        c2 = NameConstraint(var_name="xxx")
        assert c1 != c2

    def test_stash_diff(self):
        c1 = NameConstraint(STASH="m01s02i003")
        c2 = NameConstraint(STASH="m01s02i777")
        assert c1 != c2

    def test_func_same(self):
        def func(name):
            return True

        c1 = NameConstraint(STASH="m01s02i003", long_name=func)
        c2 = NameConstraint(STASH="m01s02i003", long_name=func)
        assert c1 == c2

    def test_func_diff(self):
        c1 = NameConstraint(STASH="m01s02i003", long_name=lambda n: True)
        c2 = NameConstraint(STASH="m01s02i003", long_name=lambda n: True)
        assert c1 != c2


class Test_ConstraintCombination__hash__:
    def test_empty(self):
        c1 = Constraint() & Constraint()
        c2 = Constraint() & Constraint()
        assert hash(c1) == hash(c1)
        assert hash(c1) != hash(c2)

    def test_identical_construction(self):
        c1, c2 = Constraint(a=1), Constraint(b=1)
        cc1 = c1 & c2
        cc2 = c1 & c2
        assert hash(cc1) != hash(cc2)


class Test_ConstraintCombination__eq__:
    def test_empty_same(self):
        c1 = Constraint() & Constraint()
        c2 = Constraint() & Constraint()
        assert c1 == c2
        assert c1 is not c2

    def test_multi_components_same(self):
        c1 = Constraint("a") & Constraint(b=1)
        c2 = Constraint("a") & Constraint(b=1)
        assert c1 == c2

    def test_multi_components_diff(self):
        c1 = Constraint("a") & Constraint(b=1, c=2)
        c2 = Constraint("a") & Constraint(b=1)
        assert c1 != c2

    def test_different_component_order(self):
        c1, c2 = Constraint("a"), Constraint(b=1)
        cc1 = c1 & c2
        cc2 = c2 & c1
        assert cc1 != cc2
