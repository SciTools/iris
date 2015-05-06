# (C) British Crown Copyright 2014 - 2015, Met Office
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
Tests for function :func:`iris.fileformats.grib._load_convert.data_cutoff`.

"""

from __future__ import (absolute_import, division, print_function)

import six

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import mock

from iris.fileformats.grib._load_convert import _MDI as MDI

from iris.fileformats.grib._load_convert import data_cutoff


class TestDataCutoff(tests.IrisTest):
    def _check(self, hours, minutes, request_warning, expect_warning=False):
        # Setup the environment.
        patch_target = 'iris.fileformats.grib._load_convert.options'
        with mock.patch(patch_target) as options:
            options.warn_on_unsupported = request_warning
            with mock.patch('warnings.warn') as warn:
                # The call being tested.
                data_cutoff(hours, minutes)
        # Check the result.
        if expect_warning:
            self.assertEqual(len(warn.mock_calls), 1)
            args, kwargs = warn.call_args
            self.assertIn('data cutoff', args[0])
        else:
            self.assertEqual(len(warn.mock_calls), 0)

    def test_neither(self):
        self._check(MDI, MDI, False)

    def test_hours(self):
        self._check(3, MDI, False)

    def test_minutes(self):
        self._check(MDI, 20, False)

    def test_hours_and_minutes(self):
        self._check(30, 40, False)

    def test_neither_warning(self):
        self._check(MDI, MDI, True, False)

    def test_hours_warning(self):
        self._check(3, MDI, True, True)

    def test_minutes_warning(self):
        self._check(MDI, 20, True, True)

    def test_hours_and_minutes_warning(self):
        self._check(30, 40, True, True)


if __name__ == '__main__':
    tests.main()
