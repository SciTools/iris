# (C) British Crown Copyright 2015, Met Office
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
"""Unit tests for the :data:`iris.analysis.PERCENTILE` aggregator."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np
import numpy.ma as ma

from iris.analysis import PERCENTILE


class Test_aggregate(tests.IrisTest):
    def test_missing_mandatory_kwarg(self):
        emsg = "percentile aggregator requires .* keyword argument 'percent'"
        with self.assertRaisesRegexp(ValueError, emsg):
            PERCENTILE.aggregate('dummy', axis=0)

    def test_1d_single(self):
        data = np.arange(11)
        actual = PERCENTILE.aggregate(data, axis=0, percent=50)
        expected = 5
        self.assertTupleEqual(actual.shape, ())
        self.assertEqual(actual, expected)

    def test_masked_1d_single(self):
        data = ma.arange(11)
        data[3:7] = ma.masked
        actual = PERCENTILE.aggregate(data, axis=0, percent=50)
        expected = 7
        self.assertTupleEqual(actual.shape, ())
        self.assertEqual(actual, expected)

    def test_1d_multi(self):
        data = np.arange(11)
        percent = np.array([20, 50, 90])
        actual = PERCENTILE.aggregate(data, axis=0, percent=percent)
        expected = [2, 5, 9]
        self.assertTupleEqual(actual.shape, percent.shape)
        self.assertArrayEqual(actual, expected)

    def test_masked_1d_multi(self):
        data = ma.arange(11)
        data[3:9] = ma.masked
        percent = np.array([25, 50, 75])
        actual = PERCENTILE.aggregate(data, axis=0, percent=percent)
        expected = [1, 2, 9]
        self.assertTupleEqual(actual.shape, percent.shape)
        self.assertArrayEqual(actual, expected)

    def test_2d_single(self):
        shape = (2, 11)
        data = np.arange(np.prod(shape)).reshape(shape)
        actual = PERCENTILE.aggregate(data, axis=0, percent=50)
        self.assertTupleEqual(actual.shape, shape[-1:])
        expected = np.arange(shape[-1]) + 5.5
        self.assertArrayEqual(actual, expected)

    def test_masked_2d_single(self):
        shape = (2, 11)
        data = ma.arange(np.prod(shape)).reshape(shape)
        data[0, ::2] = ma.masked
        data[1, 1::2] = ma.masked
        actual = PERCENTILE.aggregate(data, axis=0, percent=50)
        self.assertTupleEqual(actual.shape, shape[-1:])
        expected = np.empty(shape[-1:])
        expected[1::2] = data[0, 1::2]
        expected[::2] = data[1, ::2]
        self.assertArrayEqual(actual, expected)

    def test_2d_multi(self):
        shape = (2, 10)
        data = np.arange(np.prod(shape)).reshape(shape)
        percent = np.array([10, 50, 90, 100])
        actual = PERCENTILE.aggregate(data, axis=0, percent=percent)
        self.assertTupleEqual(actual.shape, (shape[-1], percent.size))
        expected = np.array(range(shape[-1]) * percent.size)
        expected = expected.reshape(percent.size, shape[-1]).T + 1
        expected = expected + (percent / 10 - 1)
        self.assertArrayAlmostEqual(actual, expected)

    def test_masked_2d_multi(self):
        shape = (3, 10)
        data = ma.arange(np.prod(shape)).reshape(shape)
        data[1] = ma.masked
        percent = np.array([10, 50, 70, 80])
        actual = PERCENTILE.aggregate(data, axis=0, percent=percent)
        self.assertTupleEqual(actual.shape, (shape[-1], percent.size))
        expected = np.array(range(shape[-1]) * percent.size)
        expected = expected.reshape(percent.size, shape[-1]).T
        expected = expected + (percent / 10 * 2)
        self.assertArrayAlmostEqual(actual, expected)


class Test_name(tests.IrisTest):
    def test(self):
        self.assertEqual(PERCENTILE.name(), 'percentile')


class Test_aggregate_shape(tests.IrisTest):
    def test_missing_mandatory_kwarg(self):
        emsg = "percentile aggregator requires .* keyword argument 'percent'"
        with self.assertRaisesRegexp(ValueError, emsg):
            PERCENTILE.aggregate_shape()
        with self.assertRaisesRegexp(ValueError, emsg):
            kwargs = dict()
            PERCENTILE.aggregate_shape(**kwargs)
        with self.assertRaisesRegexp(ValueError, emsg):
            kwargs = dict(point=10)
            PERCENTILE.aggregate_shape(**kwargs)

    def test_mandatory_kwarg_no_shape(self):
        kwargs = dict(percent=50)
        self.assertTupleEqual(PERCENTILE.aggregate_shape(**kwargs), ())
        kwargs = dict(percent=[50])
        self.assertTupleEqual(PERCENTILE.aggregate_shape(**kwargs), ())

    def test_mandatory_kwarg_shape(self):
        kwargs = dict(percent=(10, 20))
        self.assertTupleEqual(PERCENTILE.aggregate_shape(**kwargs), (2,))
        kwargs = dict(percent=range(13))
        self.assertTupleEqual(PERCENTILE.aggregate_shape(**kwargs), (13,))


class Test_cell_method(tests.IrisTest):
    def test(self):
        self.assertIsNone(PERCENTILE.cell_method)


if __name__ == "__main__":
    tests.main()
