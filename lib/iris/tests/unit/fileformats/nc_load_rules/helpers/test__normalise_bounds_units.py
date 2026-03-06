# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.fileformats._nc_load_rules.helpers._normalise_bounds_units`."""

from typing import Optional

import numpy as np
import pytest
from pytest_mock import MockType

from iris.fileformats._nc_load_rules.helpers import (
    _normalise_bounds_units,
    _WarnComboIgnoringCfLoad,
)
from iris.tests import _shared_utils
from iris.tests.unit.fileformats.nc_load_rules.helpers import MockerMixin
from iris.warnings import IrisCfLoadWarning

CF_NAME = "dummy_bnds"


class Test(MockerMixin):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.bounds = self.mocker.sentinel.bounds

    def _make_cf_bounds_var(
        self,
        units: Optional[str] = None,
        unitless: bool = False,
    ) -> MockType:
        """Construct a mock CF bounds variable."""
        if units is None:
            units = "days since 1970-01-01"

        cf_data = self.mocker.Mock(spec=[])
        # we want to mock the absence of flag attributes to helpers.get_attr_units
        # see https://docs.python.org/3/library/unittest.mock.html#deleting-attributes
        del cf_data.flag_values
        del cf_data.flag_masks
        del cf_data.flag_meanings

        cf_var = self.mocker.MagicMock(
            cf_name=CF_NAME,
            cf_data=cf_data,
            units=units,
            calendar=None,
            dtype=float,
        )

        if unitless:
            del cf_var.units

        return cf_var

    def test_unitless(self) -> None:
        """Test bounds variable with no units."""
        cf_bounds_var = self._make_cf_bounds_var(unitless=True)
        result = _normalise_bounds_units(None, cf_bounds_var, self.bounds)
        assert result == self.bounds

    def test_invalid_units__pass_through(self) -> None:
        """Test bounds variable with invalid units."""
        units = "invalid"
        cf_bounds_var = self._make_cf_bounds_var(units=units)
        wmsg = f"Ignoring invalid units {units!r} on netCDF variable {CF_NAME!r}"
        with pytest.warns(_WarnComboIgnoringCfLoad, match=wmsg):
            result = _normalise_bounds_units(None, cf_bounds_var, self.bounds)
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
            result = _normalise_bounds_units(points_units, cf_bounds_var, self.bounds)
        assert result is None

    def test_compatible(self) -> None:
        """Test bounds variable with compatible units requiring conversion."""
        points_units, bounds_units = "days since 1970-01-01", "hours since 1970-01-01"
        cf_bounds_var = self._make_cf_bounds_var(units=bounds_units)
        bounds = np.arange(10, dtype=float) * 24
        result = _normalise_bounds_units(points_units, cf_bounds_var, bounds)
        expected = bounds / 24
        _shared_utils.assert_array_equal(result, expected)

    def test_same_units(self) -> None:
        """Test bounds variable with same units as points."""
        units = "days since 1970-01-01"
        cf_bounds_var = self._make_cf_bounds_var(units=units)
        bounds = np.arange(10, dtype=float)
        result = _normalise_bounds_units(units, cf_bounds_var, bounds)
        _shared_utils.assert_array_equal(result, bounds)
        assert result is bounds
