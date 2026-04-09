# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Generic definition of units as used in Iris."""

from typing import Any

try:
    import cf_units

    from ._cf_units import CfUnit
except ImportError:
    cf_units = None

try:
    import cfpint
    import pint

    from ._pint import PintUnit
except ImportError:
    cfpint = None
    pint = None


if not cf_units and not cfpint:
    raise ImportError("Either 'cfpint' and or 'cf_units' must be installed.")


def _default_units_class():
    from iris.experimental.units import USE_CFPINT

    if USE_CFPINT:
        result = PintUnit
    else:
        result = CfUnit
    return result


def make_unit(arg: cf_units.Unit | pint.Unit | Any) -> CfUnit | PintUnit:
    """Convert input into an Iris unit.

    Converts strings to units, and pint/cf_units Units to the Iris specialised
    derived unit types.

    The type returned is either :class:`iris.common.units.CfUnit` or
    :class:`iris.common.units.PintUnit`.  If the argument is a non-unit object, such as
    a string or number, the resulting type is determined by the
    :data:`iris.common.units.USE_CFPINT` control.
    """
    if cf_units is not None and isinstance(arg, cf_units.Unit):
        unit_class = CfUnit
    elif pint is not None and isinstance(arg, pint.Unit):
        unit_class = PintUnit
    else:
        unit_class = _default_units_class()
    return unit_class.from_unit(arg)
