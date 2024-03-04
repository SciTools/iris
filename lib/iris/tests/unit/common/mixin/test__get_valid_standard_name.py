# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :func:`iris.common.mixin._get_valid_standard_name`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from iris.common.mixin import _get_valid_standard_name


class Test(tests.IrisTest):
    def setUp(self):
        self.emsg = "'{}' is not a valid standard_name"

    def test_pass_thru_none(self):
        name = None
        self.assertEqual(_get_valid_standard_name(name), name)

    def test_pass_thru_empty(self):
        name = ""
        self.assertEqual(_get_valid_standard_name(name), name)

    def test_pass_thru_whitespace(self):
        name = "       "
        self.assertEqual(_get_valid_standard_name(name), name)

    def test_valid_standard_name(self):
        name = "air_temperature"
        self.assertEqual(_get_valid_standard_name(name), name)

    def test_standard_name_alias(self):
        name = "atmosphere_optical_thickness_due_to_pm1_ambient_aerosol"
        self.assertEqual(_get_valid_standard_name(name), name)

    def test_invalid_standard_name(self):
        name = "not_a_standard_name"
        with self.assertRaisesRegex(ValueError, self.emsg.format(name)):
            _get_valid_standard_name(name)

    def test_valid_standard_name_valid_modifier(self):
        name = "air_temperature standard_error"
        self.assertEqual(_get_valid_standard_name(name), name)

    def test_valid_standard_name_valid_modifier_extra_spaces(self):
        name = "air_temperature      standard_error"
        self.assertEqual(_get_valid_standard_name(name), name)

    def test_invalid_standard_name_valid_modifier(self):
        name = "not_a_standard_name standard_error"
        with self.assertRaisesRegex(ValueError, self.emsg.format(name)):
            _get_valid_standard_name(name)

    def test_valid_standard_invalid_name_modifier(self):
        name = "air_temperature extra_names standard_error"
        with self.assertRaisesRegex(ValueError, self.emsg.format(name)):
            _get_valid_standard_name(name)

    def test_valid_standard_valid_name_modifier_extra_names(self):
        name = "air_temperature standard_error extra words"
        with self.assertRaisesRegex(ValueError, self.emsg.format(name)):
            _get_valid_standard_name(name)


if __name__ == "__main__":
    tests.main()
