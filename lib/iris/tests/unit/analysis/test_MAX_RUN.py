# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :data:`iris.analysis.MAX_RUN` aggregator."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import dask.array as da
import numpy as np
import numpy.ma as ma

from iris._lazy_data import as_concrete_data, is_lazy_data
from iris.analysis import MAX_RUN


def bool_func(x):
    return x == 1


class UnmaskedTest(tests.IrisTest):
    def setUp(self):
        """Set up 1d and 2d unmasked data arrays for max run testing.

        Uses 1 and 3 rather than 1 and 0 to check that lambda is being applied.
        """
        self.data_1ds = [
            (np.array([3, 1, 1, 3, 3, 3]), 2),  # One run
            (np.array([3, 1, 1, 3, 1, 3]), 2),  # Two runs
            (np.array([3, 3, 3, 3, 3, 3]), 0),  # No run
            (np.array([3, 3, 1, 3, 3, 3]), 1),  # Max run of 1
            (np.array([1, 1, 1, 3, 1, 3]), 3),  # Run to start
            (np.array([3, 1, 3, 1, 1, 1]), 3),  # Run to end
            (np.array([1, 1, 1, 1, 1, 1]), 6),  # All run
        ]

        self.data_2d_axis0 = np.array(
            [
                [3, 1, 1, 3, 3, 3],  # One run
                [3, 1, 1, 3, 1, 3],  # Two runs
                [3, 3, 3, 3, 3, 3],  # No run
                [3, 3, 1, 3, 3, 3],  # Max run of 1
                [1, 1, 1, 3, 1, 3],  # Run to start
                [3, 1, 3, 1, 1, 1],  # Run to end
                [1, 1, 1, 1, 1, 1],  # All run
            ]
        ).T
        self.expected_2d_axis0 = np.array([2, 2, 0, 1, 3, 3, 6])

        self.data_2d_axis1 = self.data_2d_axis0.T
        self.expected_2d_axis1 = self.expected_2d_axis0


class MaskedTest(tests.IrisTest):
    def setUp(self):
        """Set up 1d and 2d unmasked data arrays for max run testing.

        Uses 1 and 3 rather than 1 and 0 to check that lambda is being applied.
        """
        self.data_1ds = [
            (
                ma.masked_array(
                    np.array([1, 1, 1, 3, 1, 3]), np.array([0, 0, 0, 0, 0, 0])
                ),
                3,
            ),  # No mask
            (
                ma.masked_array(
                    np.array([1, 1, 1, 3, 1, 3]), np.array([0, 0, 0, 0, 1, 1])
                ),
                3,
            ),  # Mask misses run
            (
                ma.masked_array(
                    np.array([1, 1, 1, 3, 1, 3]), np.array([1, 1, 1, 0, 0, 0])
                ),
                1,
            ),  # Mask max run
            (
                ma.masked_array(
                    np.array([1, 1, 1, 3, 1, 3]), np.array([0, 0, 1, 0, 0, 0])
                ),
                2,
            ),  # Partially mask run
            (
                ma.masked_array(
                    np.array([3, 1, 1, 1, 1, 3]), np.array([0, 0, 1, 0, 0, 0])
                ),
                2,
            ),  # Mask interrupts run
            (
                ma.masked_array(
                    np.array([1, 1, 1, 3, 1, 3]), np.array([1, 1, 1, 1, 1, 1])
                ),
                0,
            ),  # All mask
            (
                ma.masked_array(
                    np.array([1, 1, 1, 3, 1, 3]), np.array([1, 1, 1, 1, 0, 1])
                ),
                1,
            ),  # All mask or run
        ]

        self.data_2d_axis0 = ma.masked_array(
            np.array(
                [
                    [1, 1, 1, 3, 1, 3],
                    [1, 1, 1, 3, 1, 3],
                    [1, 1, 1, 3, 1, 3],
                    [1, 1, 1, 3, 1, 3],
                    [1, 1, 1, 3, 1, 3],
                    [1, 1, 1, 3, 1, 3],
                ]
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],  # No mask
                    [0, 0, 0, 0, 1, 1],  # Mask misses run
                    [1, 1, 1, 0, 0, 0],  # Mask max run
                    [0, 0, 1, 0, 0, 0],  # Partially mask run
                    [1, 1, 1, 1, 1, 1],  # All mask
                    [1, 1, 1, 1, 0, 1],  # All mask or run
                ]
            ),
        ).T

        self.expected_2d_axis0 = np.array([3, 3, 1, 2, 0, 1])

        self.data_2d_axis1 = self.data_2d_axis0.T
        self.expected_2d_axis1 = self.expected_2d_axis0


class RealMixin:
    def run_func(self, *args, **kwargs):
        return MAX_RUN.call_func(*args, **kwargs)

    def check_array(self, result, expected):
        self.assertArrayEqual(result, expected)


class LazyMixin:
    def run_func(self, *args, **kwargs):
        return MAX_RUN.lazy_func(*args, **kwargs)

    def check_array(self, result, expected, expected_chunks):
        self.assertTrue(is_lazy_data(result))
        self.assertTupleEqual(result.chunks, expected_chunks)
        result = as_concrete_data(result)
        self.assertArrayEqual(result, expected)


class TestBasic(UnmaskedTest, RealMixin):
    def test_1d(self):
        for data, expected in self.data_1ds:
            result = self.run_func(
                data,
                axis=0,
                function=bool_func,
            )
            self.check_array(result, expected)

    def test_2d_axis0(self):
        result = self.run_func(
            self.data_2d_axis0,
            axis=0,
            function=bool_func,
        )
        self.check_array(result, self.expected_2d_axis0)

    def test_2d_axis1(self):
        result = self.run_func(
            self.data_2d_axis1,
            axis=1,
            function=bool_func,
        )
        self.check_array(result, self.expected_2d_axis1)


class TestLazy(UnmaskedTest, LazyMixin):
    def test_1d(self):
        for data, expected in self.data_1ds:
            data = da.from_array(data)
            result = self.run_func(
                data,
                axis=0,
                function=bool_func,
            )
            self.check_array(result, expected, ())

    def test_2d_axis0(self):
        data = da.from_array(self.data_2d_axis0)
        result = self.run_func(
            data,
            axis=0,
            function=bool_func,
        )
        self.check_array(
            result, self.expected_2d_axis0, ((len(self.expected_2d_axis0),),)
        )

    def test_2d_axis1(self):
        data = da.from_array(self.data_2d_axis1)
        result = self.run_func(
            data,
            axis=1,
            function=bool_func,
        )
        self.check_array(
            result, self.expected_2d_axis1, ((len(self.expected_2d_axis1),),)
        )


class TestLazyChunked(UnmaskedTest, LazyMixin):
    def test_1d(self):
        for data, expected in self.data_1ds:
            data = da.from_array(data, chunks=(1,))
            result = self.run_func(
                data,
                axis=0,
                function=bool_func,
            )
            self.check_array(result, expected, ())

    def test_2d_axis0_chunk0(self):
        data = da.from_array(self.data_2d_axis0, chunks=(1, -1))
        result = self.run_func(
            data,
            axis=0,
            function=bool_func,
        )
        self.check_array(
            result, self.expected_2d_axis0, ((len(self.expected_2d_axis0),),)
        )

    def test_2d_axis0_chunk1(self):
        data = da.from_array(self.data_2d_axis0, chunks=(-1, 1))
        result = self.run_func(
            data,
            axis=0,
            function=bool_func,
        )
        expected_chunks = (tuple([1] * len(self.expected_2d_axis0)),)
        self.check_array(result, self.expected_2d_axis0, expected_chunks)

    def test_2d_axis1_chunk0(self):
        data = da.from_array(self.data_2d_axis1, chunks=(1, -1))
        result = self.run_func(
            data,
            axis=1,
            function=bool_func,
        )
        expected_chunks = (tuple([1] * len(self.expected_2d_axis1)),)
        self.check_array(result, self.expected_2d_axis1, expected_chunks)

    def test_2d_axis1_chunk1(self):
        data = da.from_array(self.data_2d_axis1, chunks=(-1, 1))
        result = self.run_func(
            data,
            axis=1,
            function=bool_func,
        )
        self.check_array(
            result, self.expected_2d_axis1, ((len(self.expected_2d_axis1),),)
        )


class TestMasked(MaskedTest, RealMixin):
    def test_1d(self):
        for data, expected in self.data_1ds:
            result = self.run_func(
                data,
                axis=0,
                function=bool_func,
            )
            self.check_array(result, expected)

    def test_2d_axis0(self):
        result = self.run_func(
            self.data_2d_axis0,
            axis=0,
            function=bool_func,
        )
        self.check_array(result, self.expected_2d_axis0)

    def test_2d_axis1(self):
        result = self.run_func(
            self.data_2d_axis1,
            axis=1,
            function=bool_func,
        )
        self.check_array(result, self.expected_2d_axis1)


class Test_name(tests.IrisTest):
    def test(self):
        self.assertEqual(MAX_RUN.name(), "max_run")


class Test_cell_method(tests.IrisTest):
    def test(self):
        self.assertIsNone(MAX_RUN.cell_method)


if __name__ == "__main__":
    tests.main()
