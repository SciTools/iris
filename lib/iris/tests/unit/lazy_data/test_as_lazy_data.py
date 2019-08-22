# (C) British Crown Copyright 2017 - 2019, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""Test the function :func:`iris._lazy data.as_lazy_data`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import dask.array as da
import dask.config
import numpy as np
import numpy.ma as ma

from iris._lazy_data import as_lazy_data, _optimum_chunksize
from iris.tests import mock


class Test_as_lazy_data(tests.IrisTest):
    def test_lazy(self):
        data = da.from_array(np.arange(24).reshape((2, 3, 4)), chunks='auto')
        result = as_lazy_data(data)
        self.assertIsInstance(result, da.core.Array)

    def test_real(self):
        data = np.arange(24).reshape((2, 3, 4))
        result = as_lazy_data(data)
        self.assertIsInstance(result, da.core.Array)

    def test_masked(self):
        data = np.ma.masked_greater(np.arange(24), 10)
        result = as_lazy_data(data)
        self.assertIsInstance(result, da.core.Array)

    def test_non_default_chunks(self):
        data = np.arange(24)
        chunks = (12,)
        lazy_data = as_lazy_data(data, chunks=chunks)
        result, = np.unique(lazy_data.chunks)
        self.assertEqual(result, 24)

    def test_with_masked_constant(self):
        masked_data = ma.masked_array([8], mask=True)
        masked_constant = masked_data[0]
        result = as_lazy_data(masked_constant)
        self.assertIsInstance(result, da.core.Array)


class Test__optimised_chunks(tests.IrisTest):
    # Stable, known chunksize for testing.
    FIXED_CHUNKSIZE_LIMIT = 1024 * 1024 * 64

    @staticmethod
    def _dummydata(shape):
        return mock.Mock(spec=da.core.Array,
                         dtype=np.dtype('f4'),
                         shape=shape)

    def test_chunk_size_limiting(self):
        # Check default chunksizes for large data (with a known size limit).
        given_shapes_and_resulting_chunks = [
            ((16, 1024, 1024), (16, 1024, 1024)),  # largest unmodified
            ((17, 1011, 1022), (8, 1011, 1022)),
            ((16, 1024, 1025), (8, 1024, 1025)),
            ((1, 17, 1011, 1022), (1, 8, 1011, 1022)),
            ((17, 1, 1011, 1022), (8, 1, 1011, 1022)),
            ((11, 2, 1011, 1022), (5, 2, 1011, 1022))
        ]
        err_fmt = 'Result of optimising chunks {} was {}, expected {}'
        for (shape, expected) in given_shapes_and_resulting_chunks:
            chunks = _optimum_chunksize(shape, shape,
                                        limit=self.FIXED_CHUNKSIZE_LIMIT)
            msg = err_fmt.format(shape, chunks, expected)
            self.assertEqual(chunks, expected, msg)

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
        err_fmt = 'Result of optimising shape={};chunks={} was {}, expected {}'
        for (shape, fullshape, expected) in given_shapes_and_resulting_chunks:
            chunks = _optimum_chunksize(chunks=shape, shape=fullshape,
                                        limit=self.FIXED_CHUNKSIZE_LIMIT)
            msg = err_fmt.format(fullshape, shape, chunks, expected)
            self.assertEqual(chunks, expected, msg)

    def test_chunk_expanding_equal_division(self):
        # Check that expansion chooses equal chunk sizes as far as possible.

        # Table of test cases:
        # (input-chunks, full-shape, size-limit, result, division)
        # Note : "division" is the resulting sizes of the chunks in the
        #     outermost chunked dimension (see code below).
        testcases_chunksin_shape_limit_chunksout_division = [
            # Simple 1-D cases : chunk multiples with increasing target shape
            ((4,), (5,), 15, (5,), [5]),
            ((4,), (12,), 15, (12,), [12]),
            ((4,), (13,), 15, (8,), [8, 5]),
            ((4,), (15,), 15, (8,), [8, 7]),
            ((4,), (16,), 15, (8,), [8, 8]),
            ((4,), (17,), 15, (12,), [12, 5]),
            ((4,), (23,), 15, (12,), [12, 11]),
            ((4,), (24,), 15, (12,), [12, 12]),
            ((4,), (25,), 15, (12,), [12, 12, 1]),
            ((4,), (96,), 15, (12,), [12, 12, 12, 12, 12, 12, 12, 12]),
            ((4,), (96,), 16, (16,), [16, 16, 16, 16, 16, 16]),
            ((4,), (96,), 21, (20,), [20, 20, 20, 20, 16]),
            ((4,), (96,), 24, (24,), [24, 24, 24, 24]),
            ((4,), (96,), 28, (24,), [24, 24, 24, 24]),
            ((4,), (97,), 28, (28,), [28, 28, 28, 13]),
            ((4,), (96,), 32, (32,), [32, 32, 32]),
            # multi-dimensional cases, similar but trailing dims are 'full'.
            ((4, 10, 100), (12, 10, 200), 16*2000, (12, 10, 200), [12]),
            ((4, 10, 100), (12, 10*2, 200/2), 16*2000, (12, 20, 100), [12]),
            ((4, 10, 100), (12, 10/2, 200*2), 16*2000, (12, 5, 400), [12]),
            ((4, 10, 100), (15, 10, 200), 16*2000, (15, 10, 200), [15]),
            ((4, 10, 100), (16, 10, 200), 16*2000, (16, 10, 200), [16]),
            ((4, 10, 100), (17, 10, 200), 16*2000, (12, 10, 200), [12, 5]),
            ((4, 10, 100), (23, 10, 200), 16*2000, (12, 10, 200), [12, 11]),
            ((4, 10, 100), (24, 10, 200), 16*2000, (12, 10, 200), [12, 12]),
            ((4, 10, 100), (25, 10, 200), 16*2000, (16, 10, 200), [16, 9]),
            # an equivalent testcase with extra initial dimensions (undivided).
            ((1, 1, 4, 10, 100), (3, 5, 25, 10, 200), 16*2000,
                (1, 1, 16, 10, 200), [16, 9]),
            # some further 'ordinary' multidimensional cases.
            ((4, 10, 100), (31, 10, 200), 16*2000, (16, 10, 200), [16, 15]),
            ((4, 10, 100), (32, 10, 200), 16*2000, (16, 10, 200), [16, 16]),
            ((4, 10, 100), (81, 10, 200), 16*2000, (16, 10, 200),
                [16, 16, 16, 16, 16, 1]),
        ]
        err_fmt_main = ('Main chunks result of optimising '
                        'chunks={},shape={},limit={} '
                        'was {}, expected {}')
        err_fmt_division = ('\nDivision result from optimising '
                            'chunks={},shape={},limit={} : '
                            ' was {}, expected {}')
        for (chunks, shape, limit, expect_chunks, expect_division) in \
                testcases_chunksin_shape_limit_chunksout_division:
            result = _optimum_chunksize(chunks=chunks,
                                        shape=shape,
                                        limit=limit,
                                        dtype=np.dtype('b1'))
            msg = err_fmt_main.format(chunks, shape, limit,
                                      result, expect_chunks)
            self.assertEqual(result, expect_chunks, msg)

            # From result, make a list of chunk sizes in the outer chunked dim.
            i_chunked_dim = [ind for ind, dim in enumerate(result)
                             if dim > 1][0]
            chunksize = result[i_chunked_dim]
            fullsize = shape[i_chunked_dim]
            n_full_chunks = int(np.floor(fullsize / chunksize))
            division = [chunksize] * n_full_chunks
            n_rest = int(fullsize - n_full_chunks * chunksize)
            if n_rest > 0:
                # Chunksize is not an exact fit, so add a final, partial chunk.
                division += [n_rest]
            # Check the calculated division, too.
            msg = err_fmt_division.format(chunks, shape, limit, division,
                                          expect_division)
            self.assertEqual(division, expect_division, msg)

    def test_default_chunksize(self):
        # Check that the "ideal" chunksize is taken from the dask config.
        with dask.config.set({'array.chunk-size': '20b'}):
            chunks = _optimum_chunksize((1, 8),
                                        shape=(400, 20),
                                        dtype=np.dtype('f4'))
            self.assertEqual(chunks, (1, 4))

    def test_default_chunks_limiting(self):
        # Check that chunking is still controlled when no specific 'chunks'
        # is passed.
        limitcall_patch = self.patch('iris._lazy_data._optimum_chunksize')
        test_shape = (3, 2, 4)
        data = self._dummydata(test_shape)
        as_lazy_data(data)
        self.assertEqual(limitcall_patch.call_args_list,
                         [mock.call(list(test_shape),
                                    shape=test_shape,
                                    dtype=np.dtype('f4'))])

    def test_shapeless_data(self):
        # Check that chunk optimisation is skipped if shape contains a zero.
        limitcall_patch = self.patch('iris._lazy_data._optimum_chunksize')
        test_shape = (2, 1, 0, 2)
        data = self._dummydata(test_shape)
        as_lazy_data(data, chunks=test_shape)
        self.assertFalse(limitcall_patch.called)


if __name__ == '__main__':
    tests.main()
