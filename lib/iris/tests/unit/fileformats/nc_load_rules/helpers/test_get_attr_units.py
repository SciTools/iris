# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.fileformats._nc_load_rules.helpers.\
get_attr_units`.

"""

import numpy as np
import pytest

from iris.fileformats._nc_load_rules.helpers import get_attr_units
from iris.fileformats.cf import CFDataVariable
from iris.loading import LOAD_PROBLEMS
from iris.tests import _shared_utils
from iris.tests.unit.fileformats.nc_load_rules.helpers import MockerMixin
from iris.warnings import IrisCfLoadWarning


class TestGetAttrUnits(MockerMixin):
    def _make_cf_var(self, global_attributes=None):
        if global_attributes is None:
            global_attributes = {}

        cf_group = self.mocker.Mock(global_attributes=global_attributes)

        cf_var = self.mocker.MagicMock(
            spec=CFDataVariable,
            cf_name="sound_frequency",
            cf_data=self.mocker.Mock(spec=[]),
            filename="DUMMY",
            standard_name=None,
            long_name=None,
            units="\u266b",
            dtype=np.float64,
            cell_methods=None,
            cf_group=cf_group,
        )
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
