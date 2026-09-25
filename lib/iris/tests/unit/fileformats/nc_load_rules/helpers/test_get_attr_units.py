# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.fileformats._nc_load_rules.helpers.\
get_attr_units`.

"""

import cf_units
import numpy as np
import pytest

from iris.fileformats._nc_load_rules.helpers import get_attr_units
from iris.loading import LOAD_PROBLEMS
from iris.tests import _shared_utils
from iris.tests.unit.fileformats.nc_load_rules.helpers import (
    CFVariableDouble,
    MockerMixin,
)
from iris.warnings import IrisCfLoadWarning


class TestGetAttrUnits(MockerMixin):
    def _make_cf_var(self, global_attributes=None):
        if global_attributes is None:
            global_attributes = {}

        cf_group = self.mocker.Mock(global_attributes=global_attributes)

        cf_var = CFVariableDouble(
            standard_name=None,
            long_name=None,
            units="\u266b",
            cell_methods=None,
        )
        cf_var.cf_name = "sound_frequency"
        cf_var.filename = "DUMMY"
        cf_var.dtype = np.float64
        cf_var.cf_group = cf_group
        return cf_var

    def test_unicode_character(self):
        attributes = {}
        expected_attributes = {"invalid_units": "\u266b"}
        cf_var = self._make_cf_var()
        attr_units = get_attr_units(cf_var, attributes)
        assert attr_units == "?"
        assert attributes == expected_attributes

    def test_warn(self):
        attributes = {}
        expected_attributes = {"invalid_units": "\u266b"}
        cf_var = self._make_cf_var()
        with pytest.warns(IrisCfLoadWarning, match="Ignoring invalid units"):
            attr_units = get_attr_units(cf_var, attributes)
        assert attr_units == "?"
        assert attributes == expected_attributes

    def test_capture(self):
        attributes = {}
        expected_attributes = {"invalid_units": "\u266b"}
        cf_var = self._make_cf_var()
        with _shared_utils.assert_no_warnings_regexp("Ignoring invalid units"):
            attr_units = get_attr_units(cf_var, attributes, capture_invalid=True)
        assert attr_units == "?"
        assert attributes == expected_attributes

        load_problem = LOAD_PROBLEMS.problems[-1]
        assert load_problem.loaded == {"units": "\u266b"}

    def test_flag_values_untracked(self):
        """Presence of flag_values must force NO_UNIT_STRING without recording a read.

        This is the load-bearing assertion for the untracked probe in
        `get_attr_units` (`name in cf_var.attributes.untracked`): if that
        probe is ever simplified to `name in cf_var.attributes` (a tracked
        read), `flag_values`/`flag_masks`/`flag_meanings` would stop being
        copied onto loaded cubes by `_add_unused_attributes`, silently. The
        second assertion below is the one that catches that regression - the
        first alone would still pass.
        """
        cf_var = CFVariableDouble(units="1", flag_values="1, 2, 3")
        cf_var.cf_name = "flag_var"

        attr_units = get_attr_units(cf_var, {})

        assert attr_units == cf_units._NO_UNIT_STRING
        assert "flag_values" not in cf_var.attributes.read
