# (C) British Crown Copyright 2019, Met Office
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
Unit tests for the :func:`iris._cube_coord_common.get_valid_standard_name`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris._cube_coord_common import get_valid_standard_name


class Test(tests.IrisTest):
    def setUp(self):
        self.emsg = "'{}' is not a valid standard_name"

    def test_valid_standard_name(self):
        name = 'air_temperature'
        self.assertEqual(get_valid_standard_name(name), name)

    def test_invalid_standard_name(self):
        name = 'not_a_standard_name'
        with self.assertRaisesRegexp(ValueError, self.emsg.format(name)):
            get_valid_standard_name(name)

    def test_valid_standard_name_valid_modifier(self):
        name = 'air_temperature standard_error'
        self.assertEqual(get_valid_standard_name(name), name)

    def test_valid_standard_name_valid_modifier_extra_spaces(self):
        name = 'air_temperature      standard_error'
        self.assertEqual(get_valid_standard_name(name), name)

    def test_invalid_standard_name_valid_modifier(self):
        name = 'not_a_standard_name standard_error'
        with self.assertRaisesRegexp(ValueError, self.emsg.format(name)):
            get_valid_standard_name(name)

    def test_valid_standard_invalid_name_modifier(self):
        name = 'air_temperature extra_names standard_error'
        with self.assertRaisesRegexp(ValueError, self.emsg.format(name)):
            get_valid_standard_name(name)

    def test_valid_standard_valid_name_modifier_extra_names(self):
        name = 'air_temperature standard_error extra words'
        with self.assertRaisesRegexp(ValueError, self.emsg.format(name)):
            get_valid_standard_name(name)


if __name__ == '__main__':
    tests.main()
