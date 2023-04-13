# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :data:`iris.analysis.RMS` aggregator."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np
import numpy.ma as ma

from iris._lazy_data import as_lazy_data
from iris.analysis import RMS


class Test_aggregate(tests.IrisTest):
    def test_1d(self):
        # 1-dimensional input
        data = np.array([5, 2, 6, 4], dtype=np.float64)
        rms = RMS.aggregate(data, 0)
        expected_rms = 4.5
        self.assertAlmostEqual(rms, expected_rms)

    def test_2d(self):
        # 2-dimensional input
        data = np.array([[5, 2, 6, 4], [12, 4, 10, 8]], dtype=np.float64)
        expected_rms = np.array([4.5, 9.0], dtype=np.float64)
        rms = RMS.aggregate(data, 1)
        self.assertArrayAlmostEqual(rms, expected_rms)

    def test_1d_weighted(self):
        # 1-dimensional input with weights
        data = np.array([4, 7, 10, 8], dtype=np.float64)
        weights = np.array([1, 4, 3, 2], dtype=np.float64)
        expected_rms = 8.0
        rms = RMS.aggregate(data, 0, weights=weights)
        self.assertAlmostEqual(rms, expected_rms)

    def test_2d_weighted(self):
        # 2-dimensional input with weights
        data = np.array([[4, 7, 10, 8], [14, 16, 20, 8]], dtype=np.float64)
        weights = np.array([[1, 4, 3, 2], [2, 1, 1.5, 0.5]], dtype=np.float64)
        expected_rms = np.array([8.0, 16.0], dtype=np.float64)
        rms = RMS.aggregate(data, 1, weights=weights)
        self.assertArrayAlmostEqual(rms, expected_rms)

    def test_unit_weighted(self):
        # unit weights should be the same as no weights
        data = np.array([5, 2, 6, 4], dtype=np.float64)
        weights = np.ones_like(data)
        rms = RMS.aggregate(data, 0, weights=weights)
        expected_rms = 4.5
        self.assertAlmostEqual(rms, expected_rms)

    def test_masked(self):
        # masked entries should be completely ignored
        data = ma.array(
            [5, 10, 2, 11, 6, 4],
            mask=[False, True, False, True, False, False],
            dtype=np.float64,
        )
        expected_rms = 4.5
        rms = RMS.aggregate(data, 0)
        self.assertAlmostEqual(rms, expected_rms)

    def test_masked_weighted(self):
        # weights should work properly with masked arrays
        data = ma.array(
            [4, 7, 18, 10, 11, 8],
            mask=[False, False, True, False, True, False],
            dtype=np.float64,
        )
        weights = np.array([1, 4, 5, 3, 8, 2], dtype=np.float64)
        expected_rms = 8.0
        rms = RMS.aggregate(data, 0, weights=weights)
        self.assertAlmostEqual(rms, expected_rms)


class Test_lazy_aggregate(tests.IrisTest):
    def test_1d(self):
        # 1-dimensional input.
        data = as_lazy_data(np.array([5, 2, 6, 4], dtype=np.float64))
        rms = RMS.lazy_aggregate(data, 0)
        expected_rms = 4.5
        self.assertAlmostEqual(rms, expected_rms)

    def test_2d(self):
        # 2-dimensional input.
        data = as_lazy_data(
            np.array([[5, 2, 6, 4], [12, 4, 10, 8]], dtype=np.float64)
        )
        expected_rms = np.array([4.5, 9.0], dtype=np.float64)
        rms = RMS.lazy_aggregate(data, 1)
        self.assertArrayAlmostEqual(rms, expected_rms)

    def test_1d_weighted(self):
        # 1-dimensional input with weights.
        data = as_lazy_data(np.array([4, 7, 10, 8], dtype=np.float64))
        weights = np.array([1, 4, 3, 2], dtype=np.float64)
        expected_rms = 8.0
        rms = RMS.lazy_aggregate(data, 0, weights=weights)
        self.assertAlmostEqual(rms, expected_rms)

    def test_1d_lazy_weighted(self):
        # 1-dimensional input with lazy weights.
        data = as_lazy_data(np.array([4, 7, 10, 8], dtype=np.float64))
        weights = as_lazy_data(np.array([1, 4, 3, 2], dtype=np.float64))
        expected_rms = 8.0
        rms = RMS.lazy_aggregate(data, 0, weights=weights)
        self.assertAlmostEqual(rms, expected_rms)

    def test_2d_weighted(self):
        # 2-dimensional input with weights.
        data = as_lazy_data(
            np.array([[4, 7, 10, 8], [14, 16, 20, 8]], dtype=np.float64)
        )
        weights = np.array([[1, 4, 3, 2], [2, 1, 1.5, 0.5]], dtype=np.float64)
        expected_rms = np.array([8.0, 16.0], dtype=np.float64)
        rms = RMS.lazy_aggregate(data, 1, weights=weights)
        self.assertArrayAlmostEqual(rms, expected_rms)

    def test_unit_weighted(self):
        # Unit weights should be the same as no weights.
        data = as_lazy_data(np.array([5, 2, 6, 4], dtype=np.float64))
        weights = np.ones_like(data)
        expected_rms = 4.5
        rms = RMS.lazy_aggregate(data, 0, weights=weights)
        self.assertAlmostEqual(rms, expected_rms)

    def test_masked(self):
        # Masked entries should be completely ignored.
        data = as_lazy_data(
            ma.array(
                [5, 10, 2, 11, 6, 4],
                mask=[False, True, False, True, False, False],
                dtype=np.float64,
            )
        )
        expected_rms = 4.5
        rms = RMS.lazy_aggregate(data, 0)
        self.assertAlmostEqual(rms, expected_rms)

    def test_masked_weighted(self):
        data = as_lazy_data(
            ma.array(
                [4, 7, 18, 10, 11, 8],
                mask=[False, False, True, False, True, False],
                dtype=np.float64,
            )
        )
        weights = np.array([1, 4, 5, 3, 8, 2])
        expected_rms = 8.0
        rms = RMS.lazy_aggregate(data, 0, weights=weights)
        self.assertAlmostEqual(rms, expected_rms)


class Test_name(tests.IrisTest):
    def test(self):
        self.assertEqual(RMS.name(), "root_mean_square")


class Test_aggregate_shape(tests.IrisTest):
    def test(self):
        shape = ()
        kwargs = dict()
        self.assertTupleEqual(RMS.aggregate_shape(**kwargs), shape)
        kwargs = dict(tom="jerry", calvin="hobbes")
        self.assertTupleEqual(RMS.aggregate_shape(**kwargs), shape)


if __name__ == "__main__":
    tests.main()
