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
Tests for function :func:`iris.fileformats.grib._load_convert._hindcast_fix`.

"""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from collections import namedtuple

from iris.fileformats.grib._load_convert import _hindcast_fix as hindcast_fix


class TestHindcastFix(tests.IrisTest):
    # setup tests : provided value, fix-applies, expected-fixed
    FixTest = namedtuple('FixTest', ('given', 'fixable', 'fixed'))
    test_values = [
        FixTest(0, False, None),
        FixTest(100, False, None),
        FixTest(2 * 2**30 - 1, False, None),
        FixTest(2 * 2**30, False, None),
        FixTest(2 * 2**30 + 1, True, -1),
        FixTest(2 * 2**30 + 2, True, -2),
        FixTest(3 * 2**30 - 1, True, -(2**30 - 1)),
        FixTest(3 * 2**30, False, None)]

    def setUp(self):
        self.patch_warn = self.patch('warnings.warn')

    def test_fix(self):
        # Check hindcast fixing.
        for given, fixable, fixed in self.test_values:
            result = hindcast_fix(given)
            expected = fixed if fixable else given
            self.assertEqual(result, expected)

    def test_fix_warning(self):
        # Check warning appears when enabled.
        self.patch('iris.fileformats.grib._load_convert.options'
                   '.warn_on_unsupported', True)
        hindcast_fix(2 * 2**30 + 5)
        self.assertEqual(self.patch_warn.call_count, 1)
        self.assertIn('Re-interpreting large grib forecastTime',
                      self.patch_warn.call_args[0][0])

    def test_fix_warning_disabled(self):
        # Default is no warning.
        hindcast_fix(2 * 2**30 + 5)
        self.assertEqual(self.patch_warn.call_count, 0)


if __name__ == '__main__':
    tests.main()
