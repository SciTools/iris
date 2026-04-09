# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the object :mod:`iris.experimental.units.USE_CFPINT` ."""

import pytest

from iris.common.mixin import CfpintUnit, CfUnit
from iris.cube import Cube
from iris.experimental.units import USE_CFPINT


def test_without_context():
    cube = Cube(1, units="m")
    assert isinstance(cube.units, CfUnit)


def test_with_context():
    with USE_CFPINT.context():
        cube = Cube(1, units="m")
    assert isinstance(cube.units, CfpintUnit)


def test_explicit_context():
    with USE_CFPINT.context(False):
        cube_false_context = Cube(1, units="m")
    assert isinstance(cube_false_context.units, CfUnit)
    with USE_CFPINT.context(pint_units=True):
        cube_true_context = Cube(1, units="m")
    assert isinstance(cube_true_context.units, CfpintUnit)


def test_error_no_context():
    with pytest.raises(TypeError):
        _ = Cube(1, units=["m"])


def test_error_with_context():
    with pytest.raises(TypeError):
        with USE_CFPINT.context():
            _ = Cube(1, units=["m"])
