# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.fileformats._nc_load_rules.helpers._normalise_bounds_units`."""

from typing import Any

import numpy as np
import pytest

from iris.fileformats._nc_load_rules.helpers import (
    _normalise_bounds_units,
    _WarnComboIgnoringCfLoad,
)
from iris.tests import _shared_utils
from iris.tests.unit.fileformats.nc_load_rules.helpers import (
    CFVariableDouble,
    MockerMixin,
)
from iris.warnings import IrisCfLoadWarning

CF_NAME = "dummy_bnds"


class Test(MockerMixin):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.bounds = self.mocker.sentinel.bounds

    def _make_cf_bounds_var(
        self, units: str | None = None, unitless: bool = False
    ) -> CFVariableDouble:
        """Construct a double CF bounds variable.

        Deliberately no ``flag_values``/``flag_masks``/``flag_meanings``: their
        absence from ``.attributes`` is what tells ``helpers.get_attr_units``
        this is not a flag variable.
        """
        if units is None:
            units = "days since 1970-01-01"

        attrs: dict[str, Any] = {"calendar": None}
        if not unitless:
            attrs["units"] = units

        cf_var = CFVariableDouble(**attrs)
        # cf_name/dtype are structural members of a real CFVariable, not CF
        # attributes, so CFVariableDouble doesn't declare them - mypy can't
        # see that setting them here is exactly what the docstring prescribes.
        cf_var.cf_name = CF_NAME  # type: ignore[attr-defined]
        cf_var.dtype = float  # type: ignore[attr-defined]
        return cf_var

    def test_unitless(self) -> None:
        """Test bounds variable with no units."""
        cf_bounds_var = self._make_cf_bounds_var(unitless=True)
        # cf_bounds_var is deliberately a CFVariableDouble, not a real
        # CFBoundaryVariable - see the class docstring. Repeated below at
        # every other call for the same reason.
        result = _normalise_bounds_units(
            None,
            cf_bounds_var,  # type: ignore[arg-type]
            self.bounds,
        )
        assert result == self.bounds

    def test_invalid_units__pass_through(self) -> None:
        """Test bounds variable with invalid units."""
        units = "invalid"
        cf_bounds_var = self._make_cf_bounds_var(units=units)
        wmsg = f"Ignoring invalid units {units!r} on netCDF variable {CF_NAME!r}"
        with pytest.warns(_WarnComboIgnoringCfLoad, match=wmsg):
            result = _normalise_bounds_units(
                None,
                cf_bounds_var,  # type: ignore[arg-type]
                self.bounds,
            )
        assert result == self.bounds

    @pytest.mark.parametrize("units", ["unknown", "no_unit", "1", "kelvin"])
    def test_ignore_bounds(self, units) -> None:
        """Test bounds variable with incompatible units compared to points."""
        points_units = "km"
        cf_bounds_var = self._make_cf_bounds_var(units=units)
        wmsg = (
            f"Ignoring bounds on NetCDF variable {CF_NAME!r}. "
            f"Expected units compatible with {points_units!r}"
        )
        with pytest.warns(IrisCfLoadWarning, match=wmsg):
            result = _normalise_bounds_units(
                points_units,
                cf_bounds_var,  # type: ignore[arg-type]
                self.bounds,
            )
        assert result is None

    def test_compatible(self) -> None:
        """Test bounds variable with compatible units requiring conversion."""
        points_units, bounds_units = "days since 1970-01-01", "hours since 1970-01-01"
        cf_bounds_var = self._make_cf_bounds_var(units=bounds_units)
        bounds = np.arange(10, dtype=float) * 24
        result = _normalise_bounds_units(
            points_units,
            cf_bounds_var,  # type: ignore[arg-type]
            bounds,
        )
        expected = bounds / 24
        _shared_utils.assert_array_equal(result, expected)

    def test_same_units(self) -> None:
        """Test bounds variable with same units as points."""
        units = "days since 1970-01-01"
        cf_bounds_var = self._make_cf_bounds_var(units=units)
        bounds = np.arange(10, dtype=float)
        result = _normalise_bounds_units(
            units,
            cf_bounds_var,  # type: ignore[arg-type]
            bounds,
        )
        _shared_utils.assert_array_equal(result, bounds)
        assert result is bounds
