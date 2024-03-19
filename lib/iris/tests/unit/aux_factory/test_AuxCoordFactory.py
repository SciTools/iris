# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for `iris.aux_factory.AuxCoordFactory`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import numpy as np
import pytest

import iris
from iris._lazy_data import as_lazy_data, is_lazy_data
from iris.aux_factory import AuxCoordFactory
from iris.coords import AuxCoord
from iris.tests._shared_utils import assert_array_equal, get_data_path, skip_data


class Test__nd_points:
    def test_numpy_scalar_coord__zero_ndim(self):
        points = np.array(1)
        coord = AuxCoord(points)
        result = AuxCoordFactory._nd_points(coord, (), 0)
        expected = np.array([1])
        assert_array_equal(result, expected)

    def test_numpy_scalar_coord(self):
        value = 1
        points = np.array(value)
        coord = AuxCoord(points)
        result = AuxCoordFactory._nd_points(coord, (), 2)
        expected = np.array(value).reshape(1, 1)
        assert_array_equal(result, expected)

    def test_numpy_simple(self):
        points = np.arange(12).reshape(4, 3)
        coord = AuxCoord(points)
        result = AuxCoordFactory._nd_points(coord, (0, 1), 2)
        expected = points
        assert_array_equal(result, expected)

    def test_numpy_complex(self):
        points = np.arange(12).reshape(4, 3)
        coord = AuxCoord(points)
        result = AuxCoordFactory._nd_points(coord, (3, 2), 5)
        expected = points.T[np.newaxis, np.newaxis, ..., np.newaxis]
        assert_array_equal(result, expected)

    def test_lazy_simple(self):
        raw_points = np.arange(12).reshape(4, 3)
        points = as_lazy_data(raw_points, raw_points.shape)
        coord = AuxCoord(points)
        assert is_lazy_data(coord.core_points())
        result = AuxCoordFactory._nd_points(coord, (0, 1), 2)
        # Check we haven't triggered the loading of the coordinate values.
        assert is_lazy_data(coord.core_points())
        assert is_lazy_data(result)
        expected = raw_points
        assert_array_equal(result, expected)

    def test_lazy_complex(self):
        raw_points = np.arange(12).reshape(4, 3)
        points = as_lazy_data(raw_points, raw_points.shape)
        coord = AuxCoord(points)
        assert is_lazy_data(coord.core_points())
        result = AuxCoordFactory._nd_points(coord, (3, 2), 5)
        # Check we haven't triggered the loading of the coordinate values.
        assert is_lazy_data(coord.core_points())
        assert is_lazy_data(result)
        expected = raw_points.T[np.newaxis, np.newaxis, ..., np.newaxis]
        assert_array_equal(result, expected)


class Test__nd_bounds:
    def test_numpy_scalar_coord__zero_ndim(self):
        points = np.array(0.5)
        bounds = np.arange(2)
        coord = AuxCoord(points, bounds=bounds)
        result = AuxCoordFactory._nd_bounds(coord, (), 0)
        expected = bounds
        assert_array_equal(result, expected)

    def test_numpy_scalar_coord(self):
        points = np.array(0.5)
        bounds = np.arange(2).reshape(1, 2)
        coord = AuxCoord(points, bounds=bounds)
        result = AuxCoordFactory._nd_bounds(coord, (), 2)
        expected = bounds[np.newaxis]
        assert_array_equal(result, expected)

    def test_numpy_simple(self):
        points = np.arange(12).reshape(4, 3)
        bounds = np.arange(24).reshape(4, 3, 2)
        coord = AuxCoord(points, bounds=bounds)
        result = AuxCoordFactory._nd_bounds(coord, (0, 1), 2)
        expected = bounds
        assert_array_equal(result, expected)

    def test_numpy_complex(self):
        points = np.arange(12).reshape(4, 3)
        bounds = np.arange(24).reshape(4, 3, 2)
        coord = AuxCoord(points, bounds=bounds)
        result = AuxCoordFactory._nd_bounds(coord, (3, 2), 5)
        expected = bounds.transpose((1, 0, 2)).reshape(1, 1, 3, 4, 1, 2)
        assert_array_equal(result, expected)

    def test_lazy_simple(self):
        raw_points = np.arange(12).reshape(4, 3)
        points = as_lazy_data(raw_points, raw_points.shape)
        raw_bounds = np.arange(24).reshape(4, 3, 2)
        bounds = as_lazy_data(raw_bounds, raw_bounds.shape)
        coord = AuxCoord(points, bounds=bounds)
        assert is_lazy_data(coord.core_bounds())
        result = AuxCoordFactory._nd_bounds(coord, (0, 1), 2)
        # Check we haven't triggered the loading of the coordinate values.
        assert is_lazy_data(coord.core_bounds())
        assert is_lazy_data(result)
        expected = raw_bounds
        assert_array_equal(result, expected)

    def test_lazy_complex(self):
        raw_points = np.arange(12).reshape(4, 3)
        points = as_lazy_data(raw_points, raw_points.shape)
        raw_bounds = np.arange(24).reshape(4, 3, 2)
        bounds = as_lazy_data(raw_bounds, raw_bounds.shape)
        coord = AuxCoord(points, bounds=bounds)
        assert is_lazy_data(coord.core_bounds())
        result = AuxCoordFactory._nd_bounds(coord, (3, 2), 5)
        # Check we haven't triggered the loading of the coordinate values.
        assert is_lazy_data(coord.core_bounds())
        assert is_lazy_data(result)
        expected = raw_bounds.transpose((1, 0, 2)).reshape(1, 1, 3, 4, 1, 2)
        assert_array_equal(result, expected)


@skip_data
class Test_lazy_aux_coords:
    @pytest.fixture()
    def sample_cube(self, mocker):
        path = get_data_path(["NetCDF", "testing", "small_theta_colpex.nc"])
        # While loading, "turn off" loading small variables as real data.
        mocker.patch("iris.fileformats.netcdf.loader._LAZYVAR_MIN_BYTES", 0)
        cube = iris.load_cube(path, "air_potential_temperature")
        return cube

    def _check_lazy(self, cube):
        coords = cube.aux_coords + cube.derived_coords
        for coord in coords:
            assert coord.has_lazy_points()
            if coord.has_bounds():
                assert coord.has_lazy_bounds()

    def test_lazy_coord_loading(self, sample_cube):
        # Test that points and bounds arrays stay lazy upon cube loading.
        self._check_lazy(sample_cube)

    def test_lazy_coord_printing(self, sample_cube):
        # Test that points and bounds arrays stay lazy after cube printing.
        _ = str(sample_cube)
        self._check_lazy(sample_cube)
