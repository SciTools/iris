# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the object :mod:`iris.experimental.units.USE_CFPINT` ."""

from cf_units import Unit as cf_unit

from iris.cube import Cube
from iris.common.mixin import CfpintUnit
from iris.experimental.units import USE_CFPINT

def test_without_context():
    cube = Cube(1, units="m")
    assert isinstance(cube.units, cf_unit)

def test_with_context():
    with USE_CFPINT.context():
        cube = Cube(1, units="m")
    assert isinstance(cube.units, CfpintUnit)
