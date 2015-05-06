# (C) British Crown Copyright 2015, Met Office
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
Test function :func:`iris.fileformats._pyke_rules.compiled_krb.\
fc_rules_cf_fc.build_cube_metadata`.

"""

from __future__ import (absolute_import, division, print_function)

import six

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy as np
import mock

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
        self.assertEqual(attr_units, 'unknown')
        self.assertEqual(attributes, expected_attributes)


if __name__ == "__main__":
    tests.main()
