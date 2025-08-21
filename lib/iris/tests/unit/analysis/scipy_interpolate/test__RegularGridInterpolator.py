# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the
:func:`iris.analysis._scipy_interpolate._RegularGridInterpolator` class.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from iris.analysis._scipy_interpolate import _RegularGridInterpolator
from iris.tests import _shared_utils
import iris.tests.stock as stock


class Test:
    @pytest.fixture(autouse=True)
    def _setup(self):
        # Load a source cube, then generate an interpolator instance, calculate
        # the interpolation weights and set up a target grid.
        self.cube = stock.simple_2d()
        x_points = self.cube.coord("bar").points
        y_points = self.cube.coord("foo").points
        self.interpolator = _RegularGridInterpolator(
            [x_points, y_points],
            self.cube.data,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        newx = x_points + 0.7
        newy = y_points + 0.7

        d_0 = self.cube.data[0, 0]
        d_1 = self.cube.data[0, 1]
        d_2 = self.cube.data[1, 0]
        d_3 = self.cube.data[1, 1]
        px_0, px_1 = x_points[0], x_points[1]
        py_0, py_1 = y_points[0], y_points[1]
        px_t = px_0 + 0.7
        py_t = py_0 + 0.7
        dyt_0 = self._interpolate_point(py_t, py_0, py_1, d_0, d_1)
        dyt_1 = self._interpolate_point(py_t, py_0, py_1, d_2, d_3)
        self.test_increment = self._interpolate_point(px_t, px_0, px_1, dyt_0, dyt_1)

        xv, yv = np.meshgrid(newy, newx)
        self.tgrid = np.dstack((yv, xv))
        self.weights = self.interpolator.compute_interp_weights(self.tgrid)

    @staticmethod
    def _interpolate_point(p_t, p_0, p_1, d_0, d_1):
        return d_0 + (d_1 - d_0) * ((p_t - p_0) / (p_1 - p_0))

    def test_compute_interp_weights(self):
        weights = self.weights
        assert isinstance(weights, tuple)
        assert len(weights) == 5
        assert weights[0] == self.tgrid.shape
        assert weights[1] == "linear"
        assert isinstance(weights[2], csr_matrix)

    def test__evaluate_linear_sparse(self):
        interpolator = self.interpolator
        weights = self.weights
        output_data = interpolator._evaluate_linear_sparse(weights[2])
        test_data = self.cube.data.reshape(-1) + self.test_increment
        _shared_utils.assert_array_almost_equal(output_data, test_data)

    def test_interp_using_pre_computed_weights(self):
        interpolator = self.interpolator
        weights = self.weights
        output_data = interpolator.interp_using_pre_computed_weights(weights)
        test_data = self.cube.data + self.test_increment
        assert output_data.shape == self.cube.data.shape
        _shared_utils.assert_array_almost_equal(output_data, test_data)
