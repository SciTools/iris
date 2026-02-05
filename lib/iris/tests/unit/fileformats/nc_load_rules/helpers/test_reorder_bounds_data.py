# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.fileformats._nc_load_rules.helpers.\
reorder_bounds_data`.

"""

import numpy as np
import pytest

from iris.fileformats._nc_load_rules.helpers import reorder_bounds_data
from iris.tests import _shared_utils


class Test:
    def test_fastest_varying(self, mocker):
        bounds_data = np.arange(24).reshape(2, 3, 4)
        cf_bounds_var = mocker.Mock(
            dimensions=("foo", "bar", "nv"), cf_name="wibble_bnds"
        )
        cf_coord_var = mocker.Mock(dimensions=("foo", "bar"))

        res = reorder_bounds_data(bounds_data, cf_bounds_var, cf_coord_var)
        # Vertex dimension (nv) is already at the end.
        _shared_utils.assert_array_equal(res, bounds_data)

    def test_slowest_varying(self, mocker):
        bounds_data = np.arange(24).reshape(4, 2, 3)
        cf_bounds_var = mocker.Mock(dimensions=("nv", "foo", "bar"))
        cf_coord_var = mocker.Mock(dimensions=("foo", "bar"))

        res = reorder_bounds_data(bounds_data, cf_bounds_var, cf_coord_var)
        # Move zeroth dimension (nv) to the end.
        expected = np.rollaxis(bounds_data, 0, bounds_data.ndim)
        _shared_utils.assert_array_equal(res, expected)

    def test_different_dim_names(self, mocker):
        bounds_data = np.arange(24).reshape(2, 3, 4)
        cf_bounds_var = mocker.Mock(
            dimensions=("foo", "bar", "nv"), cf_name="wibble_bnds"
        )
        cf_coord_var = mocker.Mock(dimensions=("x", "y"), cf_name="wibble")
        with pytest.raises(ValueError, match="dimension names"):
            reorder_bounds_data(bounds_data, cf_bounds_var, cf_coord_var)
