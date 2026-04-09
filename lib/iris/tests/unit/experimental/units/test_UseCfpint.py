# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the object :mod:`iris.experimental.units.USE_CFPINT` ."""

from cf_units import Unit as cf_unit
from pint import Unit as pint_unit
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


def test_from_pint():
    unit = pint_unit("m")
    cube_no_context = Cube(1, units=unit)
    with USE_CFPINT.context():
        cube_with_context = Cube(1, units=unit)
    assert isinstance(cube_no_context.units, CfpintUnit)
    assert isinstance(cube_with_context.units, CfpintUnit)


def test_from_cf_units():
    unit = cf_unit("m")
    cube_no_context = Cube(1, units=unit)
    with USE_CFPINT.context():
        cube_with_context = Cube(1, units=unit)
    assert isinstance(cube_no_context.units, CfUnit)
    assert isinstance(cube_with_context.units, CfUnit)


def test_error_no_context():
    with pytest.raises(TypeError):
        _ = Cube(1, units=["m"])


def test_error_with_context():
    with pytest.raises(TypeError):
        with USE_CFPINT.context():
            _ = Cube(1, units=["m"])
