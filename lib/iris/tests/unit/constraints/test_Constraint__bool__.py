# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :meth:`iris._constraints.Constraint.__bool__`."""

import operator

import pytest

from iris._constraints import (
    AttributeConstraint,
    Constraint,
    ConstraintCombination,
    NameConstraint,
)


class Test_Constraint__bool__:
    # Using a Constraint in a boolean context (e.g. with the ``and``/``or``/
    # ``not`` keywords) used to silently return one of the operands, quietly
    # discarding the other. It should instead raise an informative TypeError
    # so the user is directed towards the ``&`` operator (see #4337).

    _match = "truth value of a Constraint is ambiguous"

    def test_bool(self):
        with pytest.raises(TypeError, match=self._match):
            bool(Constraint("air_temperature"))

    def test_keyword_or(self):
        c1 = Constraint("air_temperature")
        c2 = Constraint("time")
        with pytest.raises(TypeError, match=self._match):
            c1 or c2

    def test_keyword_and(self):
        c1 = Constraint("air_temperature")
        c2 = Constraint("time")
        with pytest.raises(TypeError, match=self._match):
            c1 and c2

    def test_keyword_not(self):
        with pytest.raises(TypeError, match=self._match):
            not Constraint("air_temperature")

    def test_constraint_combination(self):
        combination = ConstraintCombination(
            Constraint("air_temperature"),
            Constraint("time"),
            operator.__and__,
        )
        with pytest.raises(TypeError, match=self._match):
            bool(combination)

    def test_attribute_constraint(self):
        with pytest.raises(TypeError, match=self._match):
            bool(AttributeConstraint(STASH="m01s00i024"))

    def test_name_constraint(self):
        with pytest.raises(TypeError, match=self._match):
            bool(NameConstraint(standard_name="air_temperature"))

    def test_and_operator_still_works(self):
        # The supported way of combining constraints must be unaffected.
        c1 = Constraint("air_temperature")
        c2 = Constraint("time")
        assert isinstance(c1 & c2, ConstraintCombination)
