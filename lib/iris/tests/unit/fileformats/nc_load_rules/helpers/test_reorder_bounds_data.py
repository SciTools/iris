# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test function :func:`iris.fileformats._nc_load_rules.helpers.\
reorder_bounds_data`.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

from unittest import mock

import numpy as np

from iris.fileformats._nc_load_rules.helpers import reorder_bounds_data


class Test(tests.IrisTest):
    def test_fastest_varying(self):
        bounds_data = np.arange(24).reshape(2, 3, 4)
        cf_bounds_var = mock.Mock(
            dimensions=("foo", "bar", "nv"), cf_name="wibble_bnds"
        )
        cf_coord_var = mock.Mock(dimensions=("foo", "bar"))

        res = reorder_bounds_data(bounds_data, cf_bounds_var, cf_coord_var)
        # Vertex dimension (nv) is already at the end.
        self.assertArrayEqual(res, bounds_data)

    def test_slowest_varying(self):
        bounds_data = np.arange(24).reshape(4, 2, 3)
        cf_bounds_var = mock.Mock(dimensions=("nv", "foo", "bar"))
        cf_coord_var = mock.Mock(dimensions=("foo", "bar"))

        res = reorder_bounds_data(bounds_data, cf_bounds_var, cf_coord_var)
        # Move zeroth dimension (nv) to the end.
        expected = np.rollaxis(bounds_data, 0, bounds_data.ndim)
        self.assertArrayEqual(res, expected)

    def test_different_dim_names(self):
        bounds_data = np.arange(24).reshape(2, 3, 4)
        cf_bounds_var = mock.Mock(
            dimensions=("foo", "bar", "nv"), cf_name="wibble_bnds"
        )
        cf_coord_var = mock.Mock(dimensions=("x", "y"), cf_name="wibble")
        with self.assertRaisesRegex(ValueError, "dimension names"):
            reorder_bounds_data(bounds_data, cf_bounds_var, cf_coord_var)


if __name__ == "__main__":
    tests.main()
