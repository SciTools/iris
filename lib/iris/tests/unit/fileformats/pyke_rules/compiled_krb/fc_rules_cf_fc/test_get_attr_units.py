# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test function :func:`iris.fileformats._pyke_rules.compiled_krb.\
fc_rules_cf_fc.build_cube_metadata`.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

from unittest import mock

import numpy as np

from iris.fileformats._pyke_rules.compiled_krb.fc_rules_cf_fc import \
    get_attr_units


class TestGetAttrUnits(tests.IrisTest):
    @staticmethod
    def _make_cf_var(global_attributes=None):
        if global_attributes is None:
            global_attributes = {}

        cf_group = mock.Mock(global_attributes=global_attributes)

        cf_var = mock.Mock(
            cf_name='sound_frequency',
            standard_name=None,
            long_name=None,
            units=u'\u266b',
            dtype=np.float64,
            cell_methods=None,
            cf_group=cf_group)
        return cf_var

    def test_unicode_character(self):
        attributes = {}
        expected_attributes = {'invalid_units': u'\u266b'}
        cf_var = self._make_cf_var()
        attr_units = get_attr_units(cf_var, attributes)
        self.assertEqual(attr_units, '?')
        self.assertEqual(attributes, expected_attributes)


if __name__ == "__main__":
    tests.main()
