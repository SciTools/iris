# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test function :func:`iris.fileformats._nc_load_rules.helpers.\
get_names`.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

from unittest import mock

import numpy as np

from iris.fileformats._nc_load_rules.helpers import get_names


class TestGetNames(tests.IrisTest):
    """
    The tests included in this class cover all the variations of possible
    combinations of the following inputs:
    * standard_name = [None, 'projection_y_coordinate', 'latitude_coordinate']
    * long_name = [None, 'lat_long_name']
    * var_name = ['grid_latitude', 'lat_var_name']
    * coord_name = [None, 'latitude']

    standard_name, var_name and coord_name each contain a different valid CF
    standard name so that it is clear which is being used to set the resulting
    standard_name.

    """

    @staticmethod
    def _make_cf_var(standard_name, long_name, cf_name):
        cf_var = mock.Mock(
            cf_name=cf_name,
            standard_name=standard_name,
            long_name=long_name,
            units="degrees",
            dtype=np.float64,
            cell_methods=None,
            cf_group=mock.Mock(global_attributes={}),
        )
        return cf_var

    def check_names(self, inputs, expected):
        # Inputs - attributes on the fake CF Variable. Note: coord_name is
        # optionally set in some pyke rules.
        standard_name, long_name, var_name, coord_name = inputs
        # Expected - The expected names and attributes.
        exp_std_name, exp_long_name, exp_var_name, exp_attributes = expected

        cf_var = self._make_cf_var(
            standard_name=standard_name, long_name=long_name, cf_name=var_name
        )
        attributes = {}
        res_standard_name, res_long_name, res_var_name = get_names(
            cf_var, coord_name, attributes
        )

        # Check the names and attributes are as expected.
        self.assertEqual(res_standard_name, exp_std_name)
        self.assertEqual(res_long_name, exp_long_name)
        self.assertEqual(res_var_name, exp_var_name)
        self.assertEqual(attributes, exp_attributes)

    def test_var_name_valid(self):
        # Only var_name is set and it is set to a valid standard name.
        inp = (None, None, "grid_latitude", None)
        exp = ("grid_latitude", None, "grid_latitude", {})
        self.check_names(inp, exp)

    def test_var_name_valid_coord_name_set(self):
        # var_name is a valid standard name, coord_name is also set.
        inp = (None, None, "grid_latitude", "latitude")
        exp = ("latitude", None, "grid_latitude", {})
        self.check_names(inp, exp)

    def test_var_name_invalid(self):
        # Only var_name is set but it is not a valid standard name.
        inp = (None, None, "lat_var_name", None)
        exp = (None, None, "lat_var_name", {})
        self.check_names(inp, exp)

    def test_var_name_invalid_coord_name_set(self):
        # var_name is not a valid standard name, the coord_name is also set.
        inp = (None, None, "lat_var_name", "latitude")
        exp = ("latitude", None, "lat_var_name", {})
        self.check_names(inp, exp)

    def test_long_name_set_var_name_valid(self):
        # long_name is not None, var_name is set to a valid standard name.
        inp = (None, "lat_long_name", "grid_latitude", None)
        exp = ("grid_latitude", "lat_long_name", "grid_latitude", {})
        self.check_names(inp, exp)

    def test_long_name_set_var_name_valid_coord_name_set(self):
        # long_name is not None, var_name is set to a valid standard name, and
        # coord_name is set.
        inp = (None, "lat_long_name", "grid_latitude", "latitude")
        exp = ("latitude", "lat_long_name", "grid_latitude", {})
        self.check_names(inp, exp)

    def test_long_name_set_var_name_invalid(self):
        # long_name is not None, var_name is not set to a valid standard name.
        inp = (None, "lat_long_name", "lat_var_name", None)
        exp = (None, "lat_long_name", "lat_var_name", {})
        self.check_names(inp, exp)

    def test_long_name_set_var_name_invalid_coord_name_set(self):
        # long_name is not None, var_name is not set to a valid standard name,
        # and coord_name is set.
        inp = (None, "lat_long_name", "lat_var_name", "latitude")
        exp = ("latitude", "lat_long_name", "lat_var_name", {})
        self.check_names(inp, exp)

    def test_std_name_valid_var_name_valid(self):
        # standard_name is a valid standard name, var_name is a valid standard
        # name.
        inp = ("projection_y_coordinate", None, "grid_latitude", None)
        exp = ("projection_y_coordinate", None, "grid_latitude", {})
        self.check_names(inp, exp)

    def test_std_name_valid_var_name_valid_coord_name_set(self):
        # standard_name is a valid standard name, var_name is a valid standard
        # name, coord_name is set.
        inp = ("projection_y_coordinate", None, "grid_latitude", "latitude")
        exp = ("projection_y_coordinate", None, "grid_latitude", {})
        self.check_names(inp, exp)

    def test_std_name_valid_var_name_invalid(self):
        # standard_name is a valid standard name, var_name is not a valid
        # standard name.
        inp = ("projection_y_coordinate", None, "lat_var_name", None)
        exp = ("projection_y_coordinate", None, "lat_var_name", {})
        self.check_names(inp, exp)

    def test_std_name_valid_var_name_invalid_coord_name_set(self):
        # standard_name is a valid standard name, var_name is not a valid
        # standard name, coord_name is set.
        inp = ("projection_y_coordinate", None, "lat_var_name", "latitude")
        exp = ("projection_y_coordinate", None, "lat_var_name", {})
        self.check_names(inp, exp)

    def test_std_name_valid_long_name_set_var_name_valid(self):
        # standard_name is a valid standard name, long_name is not None,
        # var_name is a valid standard name.
        inp = (
            "projection_y_coordinate",
            "lat_long_name",
            "grid_latitude",
            None,
        )
        exp = ("projection_y_coordinate", "lat_long_name", "grid_latitude", {})
        self.check_names(inp, exp)

    def test_std_name_valid_long_name_set_var_name_valid_coord_name_set(self):
        # standard_name is a valid standard name, long_name is not None,
        # var_name is a valid standard name, coord_name is set.
        inp = (
            "projection_y_coordinate",
            "lat_long_name",
            "grid_latitude",
            "latitude",
        )
        exp = ("projection_y_coordinate", "lat_long_name", "grid_latitude", {})
        self.check_names(inp, exp)

    def test_std_name_valid_long_name_set_var_name_invalid(self):
        # standard_name is a valid standard name, long_name is not None,
        # var_name is not a valid standard name.
        inp = (
            "projection_y_coordinate",
            "lat_long_name",
            "lat_var_name",
            None,
        )
        exp = ("projection_y_coordinate", "lat_long_name", "lat_var_name", {})
        self.check_names(inp, exp)

    def test_std_name_valid_long_name_set_var_name_invalid_coord_name_set(
        self,
    ):
        # standard_name is a valid standard name, long_name is not None,
        # var_name is not a valid standard name, coord_name is set.
        inp = (
            "projection_y_coordinate",
            "lat_long_name",
            "lat_var_name",
            "latitude",
        )
        exp = ("projection_y_coordinate", "lat_long_name", "lat_var_name", {})
        self.check_names(inp, exp)

    def test_std_name_invalid_var_name_valid(self):
        # standard_name is not a valid standard name, var_name is a valid
        # standard name.
        inp = ("latitude_coord", None, "grid_latitude", None)
        exp = ("grid_latitude", None, "grid_latitude", {})
        self.check_names(inp, exp)

    def test_std_name_invalid_var_name_valid_coord_name_set(self):
        # standard_name is not a valid standard name, var_name is a valid
        # standard name, coord_name is set.
        inp = ("latitude_coord", None, "grid_latitude", "latitude")
        exp = (
            "latitude",
            None,
            "grid_latitude",
            {"invalid_standard_name": "latitude_coord"},
        )
        self.check_names(inp, exp)

    def test_std_name_invalid_var_name_invalid(self):
        # standard_name is not a valid standard name, var_name is not a valid
        # standard name.
        inp = ("latitude_coord", None, "lat_var_name", None)
        exp = (None, None, "lat_var_name", {})
        self.check_names(inp, exp)

    def test_std_name_invalid_var_name_invalid_coord_name_set(self):
        # standard_name is not a valid standard name, var_name is not a valid
        # standard name, coord_name is set.
        inp = ("latitude_coord", None, "lat_var_name", "latitude")
        exp = (
            "latitude",
            None,
            "lat_var_name",
            {"invalid_standard_name": "latitude_coord"},
        )
        self.check_names(inp, exp)

    def test_std_name_invalid_long_name_set_var_name_valid(self):
        # standard_name is not a valid standard name, long_name is not None
        # var_name is a valid standard name.
        inp = ("latitude_coord", "lat_long_name", "grid_latitude", None)
        exp = (
            "grid_latitude",
            "lat_long_name",
            "grid_latitude",
            {"invalid_standard_name": "latitude_coord"},
        )
        self.check_names(inp, exp)

    def test_std_name_invalid_long_name_set_var_name_valid_coord_name_set(
        self,
    ):
        # standard_name is not a valid standard name, long_name is not None,
        # var_name is a valid standard name, coord_name is set.
        inp = ("latitude_coord", "lat_long_name", "grid_latitude", "latitude")
        exp = (
            "latitude",
            "lat_long_name",
            "grid_latitude",
            {"invalid_standard_name": "latitude_coord"},
        )
        self.check_names(inp, exp)

    def test_std_name_invalid_long_name_set_var_name_invalid(self):
        # standard_name is not a valid standard name, long_name is not None
        # var_name is not a valid standard name.
        inp = ("latitude_coord", "lat_long_name", "lat_var_name", None)
        exp = (
            None,
            "lat_long_name",
            "lat_var_name",
            {"invalid_standard_name": "latitude_coord"},
        )
        self.check_names(inp, exp)

    def test_std_name_invalid_long_name_set_var_name_invalid_coord_name_set(
        self,
    ):
        # standard_name is not a valid standard name, long_name is not None,
        # var_name is not a valid standard name, coord_name is set.
        inp = ("latitude_coord", "lat_long_name", "lat_var_name", "latitude")
        exp = (
            "latitude",
            "lat_long_name",
            "lat_var_name",
            {"invalid_standard_name": "latitude_coord"},
        )
        self.check_names(inp, exp)


if __name__ == "__main__":
    tests.main()
