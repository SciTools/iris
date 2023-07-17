# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Integration tests for :class:`iris.cube.Cube`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

import numpy as np

import iris
from iris._lazy_data import as_lazy_data, is_lazy_data
from iris.analysis import MEAN
from iris.cube import Cube


class Test_aggregated_by(tests.IrisTest):
    @tests.skip_data
    def test_agg_by_aux_coord(self):
        problem_test_file = tests.get_data_path(
            ("NetCDF", "testing", "small_theta_colpex.nc")
        )
        # While loading, "turn off" loading small variables as real data.
        with mock.patch(
            "iris.fileformats.netcdf.loader._LAZYVAR_MIN_BYTES", 0
        ):
            cube = iris.load_cube(
                problem_test_file, "air_potential_temperature"
            )

        # Test aggregating by aux coord, notably the `forecast_period` aux
        # coord on `cube`, whose `_points` attribute is a lazy array.
        # This test then ensures that aggregating using `points` instead is
        # successful.

        # First confirm we've got a lazy array.
        # NB. This checks the merge process in `load_cube()` hasn't
        # triggered the load of the coordinate's data.
        forecast_period_coord = cube.coord("forecast_period")

        self.assertTrue(is_lazy_data(forecast_period_coord.core_points()))

        # Now confirm we can aggregate along this coord.
        res_cube = cube.aggregated_by("forecast_period", MEAN)
        res_cell_methods = res_cube.cell_methods[0]
        self.assertEqual(res_cell_methods.coord_names, ("forecast_period",))
        self.assertEqual(res_cell_methods.method, "mean")


class TestDataFillValue(tests.IrisTest):
    def test_real(self):
        data = np.ma.masked_array([1, 2, 3], [0, 1, 0], fill_value=10)
        cube = Cube(data)
        cube.data.fill_value = 20
        self.assertEqual(cube.data.fill_value, 20)

    def test_lazy(self):
        data = np.ma.masked_array([1, 2, 3], [0, 1, 0], fill_value=10)
        data = as_lazy_data(data)
        cube = Cube(data)
        cube.data.fill_value = 20
        self.assertEqual(cube.data.fill_value, 20)


if __name__ == "__main__":
    tests.main()
