# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test function :func:`iris.fileformats._nc_load_rules.helpers.\
get_cf_bounds_var`.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

from unittest import mock

from iris.fileformats._nc_load_rules.helpers import (
    CF_ATTR_BOUNDS,
    CF_ATTR_CLIMATOLOGY,
    get_cf_bounds_var,
)


class TestGetCFBoundsVar(tests.IrisTest):
    # Tests to check that get_cf_bounds_var will return the bounds_var and
    # the correct climatological flag.
    def _generic_test(self, test_climatological_bounds=False):
        cf_coord_var = mock.MagicMock()

        cf_group_dict = {"TEST": mock.sentinel.bounds_var}
        if test_climatological_bounds:
            cf_coord_var.cf_group.climatology = cf_group_dict
            test_attr = CF_ATTR_CLIMATOLOGY
        else:
            cf_coord_var.cf_group.bounds = cf_group_dict
            test_attr = CF_ATTR_BOUNDS

        for attr in (CF_ATTR_BOUNDS, CF_ATTR_CLIMATOLOGY):
            attr_val = "TEST" if attr == test_attr else None
            setattr(cf_coord_var, attr, attr_val)

        bounds_var, climatological = get_cf_bounds_var(cf_coord_var)
        self.assertIs(bounds_var, mock.sentinel.bounds_var)
        self.assertEqual(climatological, test_climatological_bounds)

    def test_bounds_normal(self):
        self._generic_test(test_climatological_bounds=False)

    def test_bounds_climatological(self):
        self._generic_test(test_climatological_bounds=True)


if __name__ == "__main__":
    tests.main()
