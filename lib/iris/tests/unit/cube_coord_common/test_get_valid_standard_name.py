# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :func:`iris._cube_coord_common.get_valid_standard_name`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris._cube_coord_common import get_valid_standard_name


class Test(tests.IrisTest):
    def setUp(self):
        self.emsg = "'{}' is not a valid standard_name"

    def test_none_is_valid(self):
        name = None
        self.assertEqual(get_valid_standard_name(name), name)

    def test_empty_is_not_valid(self):
        name = ''
        self.assertEqual(get_valid_standard_name(name), name)

    def test_only_whitespace_is_not_valid(self):
        name = '       '
        self.assertEqual(get_valid_standard_name(name), name)

    def test_none_is_valid(self):
        name = None
        self.assertEqual(get_valid_standard_name(name), name)

    def test_valid_standard_name(self):
        name = "air_temperature"
        self.assertEqual(get_valid_standard_name(name), name)

    def test_standard_name_alias(self):
        name = "atmosphere_optical_thickness_due_to_pm1_ambient_aerosol"
        self.assertEqual(get_valid_standard_name(name), name)

    def test_invalid_standard_name(self):
        name = "not_a_standard_name"
        with self.assertRaisesRegex(ValueError, self.emsg.format(name)):
            get_valid_standard_name(name)

    def test_valid_standard_name_valid_modifier(self):
        name = "air_temperature standard_error"
        self.assertEqual(get_valid_standard_name(name), name)

    def test_valid_standard_name_valid_modifier_extra_spaces(self):
        name = "air_temperature      standard_error"
        self.assertEqual(get_valid_standard_name(name), name)

    def test_invalid_standard_name_valid_modifier(self):
        name = "not_a_standard_name standard_error"
        with self.assertRaisesRegex(ValueError, self.emsg.format(name)):
            get_valid_standard_name(name)

    def test_valid_standard_invalid_name_modifier(self):
        name = "air_temperature extra_names standard_error"
        with self.assertRaisesRegex(ValueError, self.emsg.format(name)):
            get_valid_standard_name(name)

    def test_valid_standard_valid_name_modifier_extra_names(self):
        name = "air_temperature standard_error extra words"
        with self.assertRaisesRegex(ValueError, self.emsg.format(name)):
            get_valid_standard_name(name)


if __name__ == "__main__":
    tests.main()
