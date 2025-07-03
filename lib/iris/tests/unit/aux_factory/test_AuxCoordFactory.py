# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for `iris.aux_factory.AuxCoordFactory`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
from typing import Iterable, Tuple

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


def make_dimco(name, dims):
    dims = tuple(dims)
    # Create simple points + bounds arrays
    pts = np.ones(dims, dtype=np.int32)
    bds = np.stack([pts - 0.5, pts + 0.5], axis=-1)
    # Make both points and bounds lazy, with a single chunk.
    pts, bds = (da.from_array(x, chunks=-1) for x in (pts, bds))
    co = AuxCoord(pts, bounds=bds, long_name=name)
    return co


_Chunkspec = Iterable[int | Iterable[int]]
_Chunks = Tuple[Tuple[int, ...], ...]


def chunkspecs(points: _Chunkspec, bounds: _Chunkspec) -> Tuple[_Chunks, _Chunks]:
    """Convert chunks to a standard form for comparison.

    Because Python literals for the Dask chunks "tuple of tuples" form are rather hard
    to read, especially when they are mostly in the form ((n,), (m,), (p,)),
    i.e. a single chunk per dim.

    This function takes specifically two chunk inputs, for points + bounds.

    Parameters
    ----------
    points, bounds : _Chunkspec
        A pair (points, bounds) of chunk specs.
        Each is a sequence of dimension specs, where each dim is either an int or an
        iterable of int.

    Returns
    -------
    points, bounds : _Chunks
        a pair of array chunk descriptors, in the form of Dask chunks arguments.

    """
    pts_spec, bds_spec = [
        tuple(
            (dimspec,) if isinstance(dimspec, int) else tuple(int(x) for x in dimspec)
            for dimspec in chunks
        )
        for chunks in (points, bounds)
    ]
    return pts_spec, bds_spec


class Test_rechunk:
    class TestAuxFact(AuxCoordFactory):
        """A minimal AuxCoordFactory that enables us to test the re-chunking logic."""

        def __init__(self, nx, ny, nz):
            # NOTE: In a *real* factory coordinate, the dependencies are references to
            # cube coordinates.  The "make_coord" function needs to broadcast/transpose
            # their points and bounds to align with the result dimensions before passing
            # them to "_derive_array" / "_calculate_array".
            # We store our dependencies pre-aligned with result dims so that we *don't*
            # need to bother with dim mapping, and our "make_coord" is much simpler.
            self.x = make_dimco("x", (nx, 1, 1))
            self.y = make_dimco("y", (1, ny, 1))
            self.z = make_dimco("z", (1, 1, nz))

        @property
        def dependencies(self):
            return {"x": self.x, "y": self.y, "z": self.z}

        def _calculate_array(self, *dep_arrays, **other_args):
            # Do this a slightly clunky way, because it generalises nicely to 'N' args.
            # N.B. from experiment, this gets the same chunks as a one-line a+b+c+...
            result = 0
            for arg in dep_arrays:
                result += arg
            return result

        def make_coord(self, coord_dims_func):
            # N.B. no dim re-mapping needed, as dep arrays are all pre-aligned.
            points = self._derive_array(
                *(getattr(self, name).core_points() for name in self.dependencies)
            )
            bounds = self._derive_array(
                *(getattr(self, name).core_bounds() for name in self.dependencies)
            )
            result = AuxCoord(
                points,
                bounds=bounds,
                long_name="testco",
            )
            return result

    # Test the calculation which forms (NX, 1, 1) * (1, NY, 1) * (1, 1, NZ)
    #  - for different NZ sizes, eventually needing to rechunk on both Y and X.
    @pytest.mark.parametrize("nz", [10, 100, 1000])
    # Test with  mixtures of real and lazy dependencies.
    @pytest.mark.parametrize("deptypes", ["all_lazy", "mixed_real_lazy", "all_real"])
    def test_rechunk(self, nz, deptypes):
        """Test hybrid coordinate chunk handling, to avoid excessive memory costs."""
        nx, ny = 10, 10
        chunksize = 9000 * 4  # *4 for np.int32 element size
        # Rough summary  of expectation with different nz (detail below):
        #   (10, 10, 10) = 1,000: ok
        #   (10, 10, 100) = 10,000 --> (5, 10, 100) = 5000 --> rechunk, dividing X by 2
        #   (10, 10, 1000) = 100,000 --> (1, 5, 1000) --> rechunk both X and Y
        aux_co = self.TestAuxFact(nx, ny, nz)

        if deptypes != "all_lazy":
            # Touch all dependencies to realise
            names = ["x", "y", "z"] if deptypes == "all_real" else ["y"]
            for name in names:
                co = getattr(aux_co, name)
                co.points = co.points
                co.bounds = co.bounds

        daskformat_chunksize = f"{chunksize}b"
        with dask.config.set({"array.chunk-size": daskformat_chunksize}):
            result = aux_co.make_coord(None)

        # Results should *always* be lazy, even when dependencies are all real.
        assert result.has_lazy_points()
        assert result.has_lazy_bounds()

        # Check the expected chunking of the result.
        result_pts_bds_chunks = chunkspecs(
            result.core_points().chunks,
            result.core_bounds().chunks,
        )
        expected_pts_bds_chunks = {
            10: chunkspecs(points=[10, 10, 10], bounds=[10, 10, 10, 2]),
            100: chunkspecs(points=[[5, 5], 10, 100], bounds=[5 * [2], 10, 100, 2]),
            1000: chunkspecs(
                points=[10 * [1], [5, 5], 1000], bounds=[10 * [1], 5 * [2], 1000, 2]
            ),
        }[nz]
        assert result_pts_bds_chunks == expected_pts_bds_chunks

    class MultiDimTestFactory(TestAuxFact):
        """An extended test factory with an added multidimensional term."""

        # Use fixed test dimensions, for simplicity.
        _MULTIDIM_TEST_DIMS = {"nz": 7, "ny": 4, "nx": 5}

        def __init__(self):
            nz, ny, nx = (self._MULTIDIM_TEST_DIMS["n" + name] for name in "zyx")
            self.z = make_dimco("x", (nz, 1, 1))
            self.y = make_dimco("y", (1, ny, 1))
            self.x = make_dimco("z", (1, 1, nx))
            mm_data = da.from_array(np.ones((nz, ny, nx)))
            mm_bounds = da.stack([mm_data - 0.5, mm_data + 0.5], axis=-1)
            self.mm = AuxCoord(mm_data, bounds=mm_bounds, long_name="mm")

        @property
        def dependencies(self):
            return {name: getattr(self, name) for name in ("z", "y", "x", "mm")}

    # More-or-less duplicates "test_rechunk", but with a multidimensional factory.
    # Apply 2 different chunksize limits to check the rechunking behaviour.
    @pytest.mark.parametrize("rechunk", ["norechunk", "withrechunk"])
    # Chunk multidim coordinate in 2 ways: single-chunk, and irregular multi-chunk.
    @pytest.mark.parametrize("chunktype", ["plainchunks", "fancychunks"])
    def test_multidim(self, rechunk, chunktype):
        """Test chunk handling with multidimensional terms."""
        aux_co = self.MultiDimTestFactory()

        # When forcing a rechunk, choose a set chunksize which causes a partial
        #  rechunking, on outer dimensions only.
        rechunk_chunksize = 25 * 4  # *4 for np.int32 element size

        if chunktype == "fancychunks":
            # Apply an irregular chunking to the multidimensional coord, to demonstrate
            #  (a) that this is passed through when not rechunking, and
            #  (b) that it is correctly, partially, re-chunked when rechunking occurs.
            mpts_chunks = ((2, 5), (1, 3), (2, 3))
            aux_co.mm.points = aux_co.mm.core_points().rechunk(mpts_chunks)
            aux_co.mm.bounds = aux_co.mm.core_bounds().rechunk(mpts_chunks + (2,))

            # Also re-chunk the other deps similarly on those dims...
            # N.B. because : an irregular chunking can only survive rechunking if chunks
            #  of other deps all "agree" with it in each dim: Otherwise, chunks get
            #  broken up when they arithmetically combine with other, unaligned ones.
            aux_co.y.points = aux_co.y.core_points().rechunk((1, (1, 3), 1))
            aux_co.y.bounds = aux_co.y.core_bounds().rechunk((1, (1, 3), 1, 2))
            aux_co.x.points = aux_co.x.core_points().rechunk((1, 1, (2, 3)))
            aux_co.x.bounds = aux_co.x.core_bounds().rechunk((1, 1, (2, 3), 2))

            # These are the expected results...
            norechunk_pts_bds_chunks = chunkspecs(
                points=[[2, 5], [1, 3], [2, 3]], bounds=[[2, 5], [1, 3], [2, 3], 2]
            )
            rechunked_pts_bds_chunks = chunkspecs(
                # Points dim #0 is rechunked, rest retain original chunking.
                points=[7 * [1], [1, 3], [2, 3]],
                # Bounds dim #0 is rechunked, and #1 as well.
                bounds=[7 * [1], 4 * [1], [2, 3], 2],
            )

        else:
            # Expected results of the simpler single-chunk multidimensional test.
            norechunk_pts_bds_chunks = chunkspecs(
                points=[7, 4, 5], bounds=[7, 4, 5, [1, 1]]
            )
            rechunked_pts_bds_chunks = chunkspecs(
                # dim#0 split into individual indices, and dim#1 divided in 2.
                points=[7 * [1], [2, 2], 5],
                bounds=[7 * [1], [2, 2], 5, [1, 1]],
            )

        do_rechunk = rechunk == "withrechunk"
        if do_rechunk:
            chunksize = rechunk_chunksize
            expected_pts_bds_chunks = rechunked_pts_bds_chunks
        else:
            chunksize = 9999 * 4  # *4 for np.int32 element size
            expected_pts_bds_chunks = norechunk_pts_bds_chunks

        daskformat_chunksize = f"{chunksize}b"
        with dask.config.set({"array.chunk-size": daskformat_chunksize}):
            result = aux_co.make_coord(None)

        # Check the expected chunking of the result.
        result_pts_bds_chunks = chunkspecs(
            result.core_points().chunks,
            result.core_bounds().chunks,
        )
        assert result_pts_bds_chunks == expected_pts_bds_chunks
