# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.fileformats._nc_load_rules.helpers.\
get_attr_units`.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

from unittest import mock

import numpy as np

from iris.fileformats._nc_load_rules.helpers import get_attr_units
from iris.fileformats.cf import CFDataVariable
from iris.loading import LOAD_PROBLEMS
from iris.warnings import IrisCfLoadWarning


class TestGetAttrUnits(tests.IrisTest):
    @staticmethod
    def _make_cf_var(global_attributes=None):
        if global_attributes is None:
            global_attributes = {}

        cf_group = mock.Mock(global_attributes=global_attributes)

        cf_var = mock.MagicMock(
            spec=CFDataVariable,
            cf_name="sound_frequency",
            cf_data=mock.Mock(spec=[]),
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
        self.assertEqual(attr_units, "?")
        self.assertEqual(attributes, expected_attributes)

    def test_warn(self):
        attributes = {}
        expected_attributes = {"invalid_units": "\u266b"}
        cf_var = self._make_cf_var()
        with self.assertWarns(IrisCfLoadWarning, msg="Ignoring invalid units"):
            attr_units = get_attr_units(cf_var, attributes)
        self.assertEqual(attr_units, "?")
        self.assertEqual(attributes, expected_attributes)

    def test_capture(self):
        attributes = {}
        expected_attributes = {"invalid_units": "\u266b"}
        cf_var = self._make_cf_var()
        with self.assertNoWarningsRegexp("Ignoring invalid units"):
            attr_units = get_attr_units(cf_var, attributes, capture_invalid=True)
        self.assertEqual(attr_units, "?")
        self.assertEqual(attributes, expected_attributes)

        load_problem = LOAD_PROBLEMS.problems[-1]
        self.assertEqual(load_problem.loaded, {"units": "\u266b"})


if __name__ == "__main__":
    tests.main()
