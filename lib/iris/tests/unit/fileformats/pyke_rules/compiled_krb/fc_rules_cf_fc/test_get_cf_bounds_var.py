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
Test function :func:`iris.fileformats._pyke_rules.compiled_krb.\
fc_rules_cf_fc.get_cf_bounds_var`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests


from iris.fileformats._pyke_rules.compiled_krb.fc_rules_cf_fc import \
    get_cf_bounds_var, CF_ATTR_BOUNDS, CF_ATTR_CLIMATOLOGY

from iris.tests import mock


class TestGetCFBoundsVar(tests.IrisTest):
    # Tests to check that get_cf_bounds_var will return the bounds_var and
    # the correct climatological flag.
    def _generic_test(self, test_climatological_bounds=False):
        cf_coord_var = mock.MagicMock()

        cf_group_dict = {'TEST': mock.sentinel.bounds_var}
        if test_climatological_bounds:
            cf_coord_var.cf_group.climatology = cf_group_dict
            test_attr = CF_ATTR_CLIMATOLOGY
        else:
            cf_coord_var.cf_group.bounds = cf_group_dict
            test_attr = CF_ATTR_BOUNDS

        for attr in (CF_ATTR_BOUNDS, CF_ATTR_CLIMATOLOGY):
            attr_val = 'TEST' if attr == test_attr else None
            setattr(cf_coord_var, attr, attr_val)

        bounds_var, climatological = get_cf_bounds_var(cf_coord_var)
        self.assertIs(bounds_var, mock.sentinel.bounds_var)
        self.assertEqual(climatological, test_climatological_bounds)

    def test_bounds_normal(self):
        self._generic_test(test_climatological_bounds=False)

    def test_bounds_climatological(self):
        self._generic_test(test_climatological_bounds=True)


if __name__ == '__main__':
    tests.main()
