# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.fileformats._nc_load_rules.helpers.\
get_cf_bounds_var`.

"""

from iris.fileformats._nc_load_rules.helpers import (
    CF_ATTR_BOUNDS,
    CF_ATTR_CLIMATOLOGY,
    get_cf_bounds_var,
)
from iris.tests.unit.fileformats.nc_load_rules.helpers import MockerMixin


class TestGetCFBoundsVar(MockerMixin):
    # Tests to check that get_cf_bounds_var will return the bounds_var and
    # the correct climatological flag.
    def _generic_test(self, test_climatological_bounds=False):
        cf_coord_var = self.mocker.MagicMock()

        cf_group_dict = {"TEST": self.mocker.sentinel.bounds_var}
        if test_climatological_bounds:
            cf_coord_var.cf_group.climatology = cf_group_dict
            test_attr = CF_ATTR_CLIMATOLOGY
        else:
            cf_coord_var.cf_group.bounds = cf_group_dict
            test_attr = CF_ATTR_BOUNDS

        cf_coord_var.attributes = {test_attr: "TEST"}

        bounds_var, climatological = get_cf_bounds_var(cf_coord_var)
        assert bounds_var is self.mocker.sentinel.bounds_var
        assert climatological == test_climatological_bounds

    def test_bounds_normal(self):
        self._generic_test(test_climatological_bounds=False)

    def test_bounds_climatological(self):
        self._generic_test(test_climatological_bounds=True)

    def test_bounds_wins_over_climatology(self):
        cf_coord_var = self.mocker.MagicMock()
        cf_coord_var.cf_group.bounds = {"B": self.mocker.sentinel.bounds_var}
        cf_coord_var.cf_group.climatology = {"C": self.mocker.sentinel.clim_var}
        cf_coord_var.attributes = {
            CF_ATTR_BOUNDS: "B",
            CF_ATTR_CLIMATOLOGY: "C",
        }

        bounds_var, climatological = get_cf_bounds_var(cf_coord_var)

        assert bounds_var is self.mocker.sentinel.bounds_var
        assert climatological is False
