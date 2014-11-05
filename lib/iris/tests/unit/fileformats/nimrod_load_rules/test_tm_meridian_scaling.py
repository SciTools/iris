# (C) British Crown Copyright 2014, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
Unit tests for the `iris.fileformats.nimrod_load_rules.tm_meridian_scaling`
function.

"""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

from iris.fileformats.nimrod_load_rules import (tm_meridian_scaling,
                                                NIMROD_DEFAULT,
                                                MERIDIAN_SCALING_BNG)
from iris.fileformats.nimrod import NimrodField


class Test(tests.IrisTest):
    def setUp(self):
        self.field = mock.Mock(tm_meridian_scaling=NIMROD_DEFAULT,
                               spec=NimrodField,
                               float32_mdi=-123)
        self.cube = mock.Mock()

    def _call_tm_meridian_scaling(self, scaling_value):
        self.field.tm_meridian_scaling = scaling_value
        tm_meridian_scaling(self.cube, self.field)

    def test_unhandled(self):
        with mock.patch('warnings.warn') as warn:
            self._call_tm_meridian_scaling(1)
        warn.assert_called_once()

    @tests.no_warnings
    def test_british_national_grid(self):
        # A value is not returned in this rule currently.
        self.assertEqual(None,
                         self._call_tm_meridian_scaling(MERIDIAN_SCALING_BNG))

    def test_null(self):
        with mock.patch('warnings.warn') as warn:
            self._call_tm_meridian_scaling(NIMROD_DEFAULT)
            self._call_tm_meridian_scaling(self.field.float32_mdi)
        self.assertEqual(warn.call_count, 0)


if __name__ == "__main__":
    tests.main()
