# (C) British Crown Copyright 2014 - 2015, Met Office
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
"""Unit tests for the :data:`iris.analysis.MEAN` aggregator."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import dask.array as da
import numpy as np
import numpy.ma as ma

from iris.analysis import MEAN


class Test_lazy_aggregate(tests.IrisTest):
    def setUp(self):
        self.data = np.arange(12.0).reshape(3, 4)
        self.data[2, 1:] = np.nan
        self.array = da.from_array(self.data, chunks=self.data.shape)
        masked_data = ma.masked_array(self.data,
                                      mask=np.isnan(self.data))
        self.axis = 0
        self.expected_masked = ma.mean(masked_data, axis=self.axis)

    def test_mdtol_default(self):
        agg = MEAN.lazy_aggregate(self.array, axis=self.axis)
        result = agg.compute()
        masked_result = ma.masked_array(result, mask=np.isnan(result))
        self.assertMaskedArrayAlmostEqual(masked_result,
                                          self.expected_masked)

    def test_mdtol_below(self):
        agg = MEAN.lazy_aggregate(self.array, axis=self.axis, mdtol=0.3)
        result = agg.compute()
        masked_result = ma.masked_array(result, mask=np.isnan(result))
        expected_masked = self.expected_masked
        expected_masked.mask = [False, True, True, True]
        self.assertMaskedArrayAlmostEqual(masked_result,
                                          expected_masked)

    def test_mdtol_above(self):
        agg = MEAN.lazy_aggregate(self.array, axis=self.axis, mdtol=0.4)
        result = agg.compute()
        masked_result = ma.masked_array(result, mask=np.isnan(result))
        self.assertMaskedArrayAlmostEqual(masked_result,
                                          self.expected_masked)

    def test_multi_axis(self):
        data = np.arange(24.0).reshape((2, 3, 4))
        collapse_axes = (0, 2)
        lazy_data = da.from_array(data, chunks=1e6)
        agg = MEAN.lazy_aggregate(lazy_data, axis=collapse_axes)
        expected = np.mean(data, axis=collapse_axes)
        self.assertArrayAllClose(agg.compute(), expected)


class Test_name(tests.IrisTest):
    def test(self):
        self.assertEqual(MEAN.name(), 'mean')


class Test_aggregate_shape(tests.IrisTest):
    def test(self):
        shape = ()
        kwargs = dict()
        self.assertTupleEqual(MEAN.aggregate_shape(**kwargs), shape)
        kwargs = dict(one=1, two=2)
        self.assertTupleEqual(MEAN.aggregate_shape(**kwargs), shape)


if __name__ == "__main__":
    tests.main()
