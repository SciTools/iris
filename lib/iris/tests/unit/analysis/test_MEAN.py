# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :data:`iris.analysis.MEAN` aggregator."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np
import numpy.ma as ma

from iris._lazy_data import as_concrete_data, as_lazy_data
from iris.analysis import MEAN


class Test_lazy_aggregate(tests.IrisTest):
    def setUp(self):
        self.data = ma.arange(12).reshape(3, 4)
        self.data.mask = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1]]
        # --> fractions of masked-points in columns = [0, 1/3, 2/3, 1]
        self.array = as_lazy_data(self.data)
        self.axis = 0
        self.expected_masked = ma.mean(self.data, axis=self.axis)

    def test_mdtol_default(self):
        # Default operation is "mdtol=1" --> unmasked if *any* valid points.
        # --> output column masks = [0, 0, 0, 1]
        agg = MEAN.lazy_aggregate(self.array, axis=self.axis)
        masked_result = as_concrete_data(agg)
        self.assertMaskedArrayAlmostEqual(masked_result, self.expected_masked)

    def test_mdtol_belowall(self):
        # Mdtol=0.25 --> masked columns = [0, 1, 1, 1]
        agg = MEAN.lazy_aggregate(self.array, axis=self.axis, mdtol=0.25)
        masked_result = as_concrete_data(agg)
        expected_masked = self.expected_masked
        expected_masked.mask = [False, True, True, True]
        self.assertMaskedArrayAlmostEqual(masked_result, expected_masked)

    def test_mdtol_intermediate(self):
        # mdtol=0.5 --> masked columns = [0, 0, 1, 1]
        agg = MEAN.lazy_aggregate(self.array, axis=self.axis, mdtol=0.5)
        masked_result = as_concrete_data(agg)
        expected_masked = self.expected_masked
        expected_masked.mask = [False, False, True, True]
        self.assertMaskedArrayAlmostEqual(masked_result, expected_masked)

    def test_mdtol_aboveall(self):
        # mdtol=0.75 --> masked columns = [0, 0, 0, 1]
        # In this case, effectively the same as mdtol=None.
        agg = MEAN.lazy_aggregate(self.array, axis=self.axis, mdtol=0.75)
        masked_result = as_concrete_data(agg)
        self.assertMaskedArrayAlmostEqual(masked_result, self.expected_masked)

    def test_multi_axis(self):
        data = np.arange(24.0).reshape((2, 3, 4))
        collapse_axes = (0, 2)
        lazy_data = as_lazy_data(data)
        agg = MEAN.lazy_aggregate(lazy_data, axis=collapse_axes)
        result = as_concrete_data(agg)
        expected = np.mean(data, axis=collapse_axes)
        self.assertArrayAllClose(result, expected)

    def test_last_axis(self):
        # From setUp:
        # self.data.mask = [[0, 0, 0, 1],
        #                   [0, 0, 1, 1],
        #                   [0, 1, 1, 1]]
        # --> fractions of masked-points in ROWS = [1/4, 1/2, 3/4]
        axis = -1
        agg = MEAN.lazy_aggregate(self.array, axis=axis, mdtol=0.51)
        expected_masked = ma.mean(self.data, axis=-1)
        expected_masked = np.ma.masked_array(expected_masked, [0, 0, 1])
        masked_result = as_concrete_data(agg)
        self.assertMaskedArrayAlmostEqual(masked_result, expected_masked)

    def test_all_axes_belowtol(self):
        agg = MEAN.lazy_aggregate(self.array, axis=None, mdtol=0.75)
        expected_masked = ma.mean(self.data)
        masked_result = as_concrete_data(agg)
        self.assertMaskedArrayAlmostEqual(masked_result, expected_masked)

    def test_all_axes_abovetol(self):
        agg = MEAN.lazy_aggregate(self.array, axis=None, mdtol=0.45)
        expected_masked = ma.masked_less([0.0], 1)
        masked_result = as_concrete_data(agg)
        self.assertMaskedArrayAlmostEqual(masked_result, expected_masked)


class Test_name(tests.IrisTest):
    def test(self):
        self.assertEqual(MEAN.name(), "mean")


class Test_aggregate_shape(tests.IrisTest):
    def test(self):
        shape = ()
        kwargs = dict()
        self.assertTupleEqual(MEAN.aggregate_shape(**kwargs), shape)
        kwargs = dict(one=1, two=2)
        self.assertTupleEqual(MEAN.aggregate_shape(**kwargs), shape)


if __name__ == "__main__":
    tests.main()
