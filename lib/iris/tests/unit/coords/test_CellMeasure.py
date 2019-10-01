# (C) British Crown Copyright 2015 - 2019, Met Office
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
"""Unit tests for the :class:`iris.coords.CellMeasure` class."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.coords import CellMeasure
from iris._lazy_data import as_lazy_data


class Tests(tests.IrisTest):
    def setUp(self):
        self.values = np.array((10., 12., 16., 9.))
        self.measure = CellMeasure(self.values, units='m^2',
                                   standard_name='cell_area',
                                   long_name='measured_area',
                                   var_name='area',
                                   attributes={'notes': '1m accuracy'},
                                   measure='area')

    def test_invalid_measure(self):
        msg = "measure must be 'area' or 'volume', not length"
        with self.assertRaisesRegexp(ValueError, msg):
            self.measure.measure = 'length'

    def test_set_measure(self):
        v = 'volume'
        self.measure.measure = v
        self.assertEqual(self.measure.measure, v)

    def test_data(self):
        self.assertArrayEqual(self.measure.data, self.values)

    def test_set_data(self):
        new_vals = np.array((1., 2., 3., 4.))
        self.measure.data = new_vals
        self.assertArrayEqual(self.measure.data, new_vals)

    def test_set_data__int(self):
        new_vals = np.array((1, 2, 3, 4), dtype=np.int32)
        self.measure.data = new_vals
        self.assertArrayEqual(self.measure.data, new_vals)

    def test_set_data__uint(self):
        new_vals = np.array((1, 2, 3, 4), dtype=np.uint32)
        self.measure.data = new_vals
        self.assertArrayEqual(self.measure.data, new_vals)

    def test_set_data__lazy(self):
        new_vals = as_lazy_data(np.array((1., 2., 3., 4.)))
        self.measure.data = new_vals
        self.assertArrayEqual(self.measure.data, new_vals)

    def test_data_different_shape(self):
        new_vals = np.array((1., 2., 3.))
        msg = 'New data shape must match existing data shape.'
        with self.assertRaisesRegexp(ValueError, msg):
            self.measure.data = new_vals

    def test_shape(self):
        self.assertEqual(self.measure.shape, (4,))

    def test_ndim(self):
        self.assertEqual(self.measure.ndim, 1)

    def test___getitem__(self):
        sub_measure = self.measure[2]
        self.assertArrayEqual(self.values[2], sub_measure.data)

    def test___getitem__data_copy(self):
        # Check that a sliced cell measure has independent data.
        sub_measure = self.measure[1:3]
        old_values = sub_measure.data.copy()
        # Change the original one.
        self.measure.data[:] = 0.0
        # Check the new one has not changed.
        self.assertArrayEqual(sub_measure.data, old_values)

    def test_copy(self):
        new_vals = np.array((7., 8.))
        copy_measure = self.measure.copy(new_vals)
        self.assertArrayEqual(copy_measure.data, new_vals)

    def test_repr_other_metadata(self):
        expected = (", long_name='measured_area', "
                    "var_name='area', attributes={'notes': '1m accuracy'}")
        self.assertEqual(self.measure._repr_other_metadata(), expected)

    def test__str__(self):
        expected = ("CellMeasure(array([10., 12., 16.,  9.]), "
                    "measure=area, standard_name='cell_area', "
                    "units=Unit('m^2'), long_name='measured_area', "
                    "var_name='area', attributes={'notes': '1m accuracy'})")
        self.assertEqual(self.measure.__str__(), expected)

    def test__repr__(self):
        expected = ("CellMeasure(array([10., 12., 16.,  9.]), "
                    "measure=area, standard_name='cell_area', "
                    "units=Unit('m^2'), long_name='measured_area', "
                    "var_name='area', attributes={'notes': '1m accuracy'})")
        self.assertEqual(self.measure.__repr__(), expected)

    def test__eq__(self):
        self.assertEqual(self.measure, self.measure)

if __name__ == '__main__':
    tests.main()
