# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test the function :func:`iris._lazy data.as_lazy_data`."""

from types import SimpleNamespace
from unittest import mock

import dask.array as da
import dask.config
import numpy as np
import numpy.ma as ma
import pytest

from iris._lazy_data import CACHE, _optimum_chunksize, as_lazy_data
from iris.fileformats.netcdf._thread_safe_nc import NetCDFDataProxy


class Test_as_lazy_data:
    def test_lazy(self):
        data = da.from_array(np.arange(24).reshape((2, 3, 4)), chunks="auto")
        result = as_lazy_data(data)
        assert isinstance(result, da.core.Array)

    def test_real(self):
        data = np.arange(24).reshape((2, 3, 4))
        result = as_lazy_data(data)
        assert isinstance(result, da.core.Array)

    def test_masked(self):
        data = np.ma.masked_greater(np.arange(24), 10)
        result = as_lazy_data(data)
        assert isinstance(result, da.core.Array)

    def test_non_default_chunks(self):
        data = np.arange(24)
        chunks = (12,)
        lazy_data = as_lazy_data(data, chunks=chunks)
        (result,) = np.unique(lazy_data.chunks)
        assert result == 24

    def test_dask_chunking(self, mocker):
        data = np.arange(24)
        chunks = (12,)
        optimum = mocker.patch("iris._lazy_data._optimum_chunksize")
        optimum.return_value = chunks
        _ = as_lazy_data(data, chunks="auto")
        assert not optimum.called

    def test_with_masked_constant(self):
        masked_data = ma.masked_array([8], mask=True)
        masked_constant = masked_data[0]
        result = as_lazy_data(masked_constant)
        assert isinstance(result, da.core.Array)

    def test_missing_meta(self):
        class MyProxy:
            pass

        data = MyProxy()

        with pytest.raises(
            ValueError,
            match=r"For performance reasons, `meta` cannot be `None` if `data` is anything other than a Numpy "
            r"or Dask array.",
        ):
            as_lazy_data(data)


class Test_as_lazy_data__cache_key:
    @staticmethod
    def _proxy(shape=(2, 3, 4), dtype="f4"):
        # NetCDFDataProxy reads only `shape` and `name` from what it is given,
        # so it can be built without a file as long as it is never indexed.
        return NetCDFDataProxy(
            SimpleNamespace(shape=shape, name="air_temperature"),
            np.dtype(dtype),
            "/does/not/exist.nc",
            -1.0,
        )

    def test_key_is_the_caller_s_string(self):
        proxy = self._proxy()
        CACHE._cache.clear()
        as_lazy_data(proxy, meta=proxy.dask_meta, cache_key=repr(proxy))
        (key,) = CACHE._cache
        assert key == (
            repr(proxy),
            (2, 3, 4),
            False,
            proxy.dask_meta.dtype,
            type(proxy.dask_meta),
        )

    def test_no_key_means_no_caching(self):
        proxy = self._proxy()
        CACHE._cache.clear()
        as_lazy_data(proxy, meta=proxy.dask_meta)
        assert len(CACHE._cache) == 0

    def test_second_call_reuses_the_cached_array(self, mocker):
        proxy = self._proxy()
        CACHE._cache.clear()
        from_array = mocker.patch("iris._lazy_data.da.from_array", wraps=da.from_array)
        as_lazy_data(proxy, meta=proxy.dask_meta, cache_key=repr(proxy))
        as_lazy_data(proxy, meta=proxy.dask_meta, cache_key=repr(proxy))
        assert from_array.call_count == 1

    def test_key_without_meta_is_refused(self):
        data = np.arange(24).reshape((2, 3, 4))
        with pytest.raises(ValueError, match="cache_key"):
            as_lazy_data(data, cache_key="anything")


def test_lazy_data_imports_no_file_format():
    """Pin spec section 4.4: `_lazy_data` knows about no file format at all."""
    import inspect
    import re

    import iris._lazy_data

    source = inspect.getsource(iris._lazy_data)
    offenders = [
        line
        for line in source.splitlines()
        if re.match(r"\s*(from|import)\s", line) and "iris.fileformats" in line
    ]
    assert offenders == []


class Test__optimised_chunks:
    # Stable, known chunksize for testing.
    FIXED_CHUNKSIZE_LIMIT = 1024 * 1024 * 64

    @staticmethod
    def _dummydata(shape):
        return mock.Mock(spec=da.core.Array, dtype=np.dtype("f4"), shape=shape)

    def test_chunk_size_limiting(self):
        # Check default chunksizes for large data (with a known size limit).
        given_shapes_and_resulting_chunks = [
            ((16, 1024, 1024), (16, 1024, 1024)),  # largest unmodified
            ((17, 1011, 1022), (8, 1011, 1022)),
            ((16, 1024, 1025), (8, 1024, 1025)),
            ((1, 17, 1011, 1022), (1, 8, 1011, 1022)),
            ((17, 1, 1011, 1022), (8, 1, 1011, 1022)),
            ((11, 2, 1011, 1022), (5, 2, 1011, 1022)),
        ]
        err_fmt = "Result of optimising chunks {} was {}, expected {}"
        for shape, expected in given_shapes_and_resulting_chunks:
            chunks = _optimum_chunksize(shape, shape, limit=self.FIXED_CHUNKSIZE_LIMIT)
            msg = err_fmt.format(shape, chunks, expected)
            assert chunks == expected, msg

    def test_chunk_size_expanding(self):
        # Check the expansion of small chunks, (with a known size limit).
        given_shapes_and_resulting_chunks = [
            ((1, 100, 100), (16, 100, 100), (16, 100, 100)),
            ((1, 100, 100), (5000, 100, 100), (1667, 100, 100)),
            ((3, 300, 200), (10000, 3000, 2000), (3, 1500, 2000)),
            ((3, 300, 200), (10000, 300, 2000), (27, 300, 2000)),
            ((3, 300, 200), (8, 300, 2000), (8, 300, 2000)),
            ((3, 300, 200), (117, 300, 1000), (39, 300, 1000)),
        ]
        err_fmt = "Result of optimising shape={};chunks={} was {}, expected {}"
        for shape, fullshape, expected in given_shapes_and_resulting_chunks:
            chunks = _optimum_chunksize(
                chunks=shape, shape=fullshape, limit=self.FIXED_CHUNKSIZE_LIMIT
            )
            msg = err_fmt.format(fullshape, shape, chunks, expected)
            assert chunks == expected, msg

    def test_chunk_expanding_equal_division(self):
        # Check that expansion chooses equal chunk sizes as far as possible.

        # Table of test cases:
        #   (input-chunkshape, full-shape, size-limit, result-chunkshape)
        testcases_chunksin_fullshape_limit_result = [
            ((4,), (12,), 15, (12,)),  # gives a single chunk, of size 12
            ((4,), (13,), 15, (8,)),  # chooses chunks of 8+5, better than 12+1
            ((4,), (16,), 15, (8,)),  # 8+8 is better than 12+4; 16 is too big.
            ((4,), (96,), 15, (12,)),  # 12 is largest 'allowed'
            ((4,), (96,), 31, (24,)),  # 28 doesn't divide 96 so neatly,
            # A multi-dimensional case, where trailing dims are 'filled'.
            ((4, 5, 100), (25, 10, 200), 16 * 2000, (16, 10, 200)),
            # Equivalent case with additional initial dimensions.
            (
                (1, 1, 4, 5, 100),
                (3, 5, 25, 10, 200),
                16 * 2000,
                (1, 1, 16, 10, 200),
            ),  # effectively the same as the previous.
        ]
        err_fmt_main = (
            "Main chunks result of optimising "
            "chunks={},shape={},limit={} "
            "was {}, expected {}"
        )
        for (
            chunks,
            shape,
            limit,
            expected_result,
        ) in testcases_chunksin_fullshape_limit_result:
            result = _optimum_chunksize(
                chunks=chunks, shape=shape, limit=limit, dtype=np.dtype("b1")
            )
            msg = err_fmt_main.format(chunks, shape, limit, result, expected_result)
            assert result == expected_result, msg

    def test_default_chunksize(self):
        # Check that the "ideal" chunksize is taken from the dask config.
        with dask.config.set({"array.chunk-size": "20b"}):
            chunks = _optimum_chunksize((1, 8), shape=(400, 20), dtype=np.dtype("f4"))
            assert chunks == (1, 4)

    def test_default_chunks_limiting(self, mocker):
        # Check that chunking is still controlled when no specific 'chunks'
        # is passed.
        limitcall_patch = mocker.patch("iris._lazy_data._optimum_chunksize")
        test_shape = (3, 2, 4)
        data = self._dummydata(test_shape)
        as_lazy_data(data)
        assert limitcall_patch.call_args_list == [
            mock.call(
                tuple(test_shape),
                shape=test_shape,
                dtype=np.dtype("f4"),
                dims_fixed=None,
            )
        ]

    def test_shapeless_data(self, mocker):
        # Check that chunk optimisation is skipped if shape contains a zero.
        limitcall_patch = mocker.patch("iris._lazy_data._optimum_chunksize")
        test_shape = (2, 1, 0, 2)
        data = self._dummydata(test_shape)
        as_lazy_data(data, chunks=test_shape)
        assert not limitcall_patch.called
