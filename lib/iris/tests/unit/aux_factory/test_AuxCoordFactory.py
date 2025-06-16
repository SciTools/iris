# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for `iris.aux_factory.AuxCoordFactory`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import dask.array as da
import dask.config
import dask.utils
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


class Test_rechunk:
    class TestAuxFact(AuxCoordFactory):
        # A minimal AuxCoordFactory that enables us to test the re-chunking logic.
        def __init__(self, nx, ny, nz):
            def make_co(name, dims):
                dims = tuple(dims)
                pts = da.ones(dims, dtype=np.int32, chunks=dims)
                bds = np.stack([pts-0.5, pts+0.5], axis=-1)
                co = AuxCoord(pts, bounds=bds, long_name=name)
                return co
            self.x = make_co("x", (nx, 1, 1))
            self.y = make_co("y", (1, ny, 1))
            self.z = make_co("z", (1, 1, nz))

        @property
        def dependencies(self):
            return {'x': self.x, "y": self.y, "z": self.z}

        def _calculate_array(self, *dep_arrays, **other_args):
            x, y, z = dep_arrays
            return x * y * z

        def make_coord(self, coord_dims_func):
            # N.B. don't bother with dim remapping, we know it is not needed.
            points = self._derive_array(
                *(getattr(self, name).core_points() for name in ("x", "y", "z"))
            )
            bounds = self._derive_array(
                *(getattr(self, name).core_bounds() for name in ("x", "y", "z"))
            )
            result = AuxCoord(
                points,
                bounds=bounds,
                long_name="testco",
            )
            return result

    @pytest.mark.parametrize('nz', [10, 100, 1000])
    def test_rechunk(self, nz):
        # Test calculation which forms (NX, 1, 1) * (1, NY, 1) * (1, 1, NZ)
        #  at different NZ sizes eventually needing to rechunk on both Y and X
        nx, ny = 10, 10
        chunksize = 4000 * 4  # *4 for np.int32 element size
        # Summary  of expectation:
        # (10, 10, 10) = 1,000: ok
        # (10, 10, 100) = 10,000 --> (3, 10, 100) = 3000 --> rechunk, dividing X by 3
        # (10, 10, 1000) = 100,1000 --> (1, 3, 1000) --> rechunk both X and Y
        aux_co = self.TestAuxFact(nx, ny, nz)
        daskformat_chunksize = f"{chunksize}b"

        with dask.config.set({"array.chunk-size": daskformat_chunksize}):
            result = aux_co.make_coord(None)

        assert result.has_lazy_points()
        assert result.has_lazy_bounds()
        chunksize_points_bounds = (
            result.core_points().chunksize,
            result.core_bounds().chunksize,
        )
        expect_chunks_points_bounds = {
            10: ((10, 10, 10), (10, 10, 10, 1)),  # no rechunk
            100: ((3, 10, 100), (2, 10, 100, 1)),  # divide x by 3 (bounds: 5)
            1000: ((1, 3, 1000), (1, 2, 1000, 1)),  # divide x,y by 10,3 (bounds: 10,5)
        }[nz]
        assert chunksize_points_bounds == expect_chunks_points_bounds
