# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :func:`iris.common.mixin.make_unit`."""

from cf_units import Unit as cf_unit
import numpy as np
from pint import Unit as pint_unit

from iris.common.mixin import CfpintUnit, CfUnit, make_unit
from iris.experimental.units import USE_CFPINT


def test_from_pint():
    unit = pint_unit("m")
    no_context_unit = make_unit(unit)
    with USE_CFPINT.context():
        context_unit = make_unit(unit)
    assert isinstance(no_context_unit, CfpintUnit)
    assert isinstance(context_unit, CfpintUnit)


def test_from_cf_units():
    unit = cf_unit("m")
    no_context_unit = make_unit(unit)
    with USE_CFPINT.context():
        context_unit = make_unit(unit)
    assert isinstance(no_context_unit, CfUnit)
    assert isinstance(context_unit, CfUnit)


def test_from_number():
    int_unit = make_unit(1)
    float_unit = make_unit(1.0)
    numpy_unit = make_unit(np.float64(1))
    assert isinstance(int_unit, CfUnit)
    assert isinstance(float_unit, CfUnit)
    assert isinstance(numpy_unit, CfUnit)
    with USE_CFPINT.context():
        int_unit = make_unit(1)
        float_unit = make_unit(1.0)
        numpy_unit = make_unit(np.float64(1))
        assert isinstance(int_unit, CfpintUnit)
        assert isinstance(float_unit, CfpintUnit)
        assert isinstance(numpy_unit, CfpintUnit)
