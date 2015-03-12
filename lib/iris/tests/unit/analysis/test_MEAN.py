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

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import biggus
import numpy.ma as ma

from iris.analysis import MEAN


class Test_lazy_aggregate(tests.IrisTest):
    def setUp(self):
        self.data = ma.arange(12).reshape(3, 4)
        self.data[2, 1:] = ma.masked
        self.array = biggus.NumpyArrayAdapter(self.data)
        self.axis = 0

    def test_mdtol_default(self):
        agg = MEAN.lazy_aggregate(self.array, axis=self.axis)
        result = agg.masked_array()
        expected = ma.mean(self.data, axis=self.axis)
        self.assertArrayAlmostEqual(result, expected)

    def test_mdtol_below(self):
        agg = MEAN.lazy_aggregate(self.array, axis=self.axis, mdtol=0.3)
        result = agg.masked_array()
        expected = ma.mean(self.data, axis=self.axis)
        expected.mask = [False, True, True, True]
        self.assertMaskedArrayAlmostEqual(result, expected)

    def test_mdtol_above(self):
        agg = MEAN.lazy_aggregate(self.array, axis=self.axis, mdtol=0.4)
        result = agg.masked_array()
        expected = ma.mean(self.data, axis=self.axis)
        self.assertMaskedArrayAlmostEqual(result, expected)


class Test_required(tests.IrisTest):
    def test(self):
        self.assertIsNone(MEAN.required)


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
