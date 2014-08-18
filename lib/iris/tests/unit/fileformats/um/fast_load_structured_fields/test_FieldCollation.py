# (C) British Crown Copyright 2014, Met Office
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
"""
Unit tests for the class
:class:`iris.fileformats.um._fast_load_structured_fields.FieldCollation`.

"""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from collections import namedtuple
from copy import deepcopy

import mock
from netCDF4 import netcdftime
import numpy as np

from iris.fileformats.um._fast_load_structured_fields import \
    FieldCollation

# Required field elements and their default values
_dummy_field_elements_and_defaults = {
    # 't1'
    'lbyr': 2007,
    'lbmon': 1,
    'lbdat': 1,
    'lbhr': 0,
    'lbmin': 0,
    'lbsec': 0,
    # 't2'
    'lbyrd': 2007,
    'lbmond': 1,
    'lbdatd': 1,
    'lbhrd': 0,
    'lbmind': 0,
    'lbsecd': 0,
    # others
    'lbft': 0,
    'lbrsvd': [0, 0, 0, 0],
    'lbuser': [0, 0, 0, 0, 0, 0, 0],
    'blev': 1,
    'lbproc': 0,
    '_i_field': None
}


class DummyField(object):
    """A testing object mocking relevant parts of a PPField3."""
    def __init__(self, date1=None, date2=None, ensemble=None,
                 pseudo_level=None, **property_kwargs):
        """Create a dummy PPField3 with the specified element values."""
        # Setup content defaults
        elems = deepcopy(_dummy_field_elements_and_defaults)
        # Alias date keys values
        if date1 is not None:
            elems['lbdat'] = date1
        if date2 is not None:
            elems['lbdatd'] = date2
        # Implement settings from arbitrary passed keywords
        elems.update(**property_kwargs)
        for el_name, el_value in elems.iteritems():
            setattr(self, el_name, el_value)
        # Implement special keywords for specific array-property elements
        if ensemble is not None:
            self.lbrsvd[3] = ensemble
        if pseudo_level is not None:
            self.lbuser[4] = pseudo_level

    def __setattr__(self, key, value):
        # Only allow creation of the intended field properties
        assert key in _dummy_field_elements_and_defaults
        object.__setattr__(self, key, value)


def _dt(date_arg):
    # Construct datetimes (or arrays of) from ints with the testing defaults.
    int_vals = np.array(date_arg)
    dt_vals = [netcdftime.datetime(2007, 1, date)
               for date in int_vals.flat]
    return np.array(dt_vals).reshape(int_vals.shape)


class Test_FieldCollation(tests.IrisTest):
    def _dummy_fields(self, **value_lists_kwargs):
        """
        Make a group of test fields with given values.

        All kwargs are either length 1 or 'N', for some specific N (i.e.
        lengths > 1 are all the same).  (Length 1 can also be scalars).
        Returns a list of 'N' fields with the given values.

        """
        # Find vectors length (default 1).
        lengths = [len(val)
                   for val in value_lists_kwargs.values()
                   if hasattr(val, '__len__')]
        length = max(lengths + [1])

        # Expand all kwargs to the full vector length.
        vector_kwargs = {}
        for keyname, value in value_lists_kwargs.iteritems():
            if not hasattr(value, '__len__'):
                value = [value] * length
            elif len(value) == 1:
                value = value * length
            vector_kwargs[keyname] = value

        # Iterate to produce all our test fields.
        self.test_fields = []
        for i_field in range(length):
            field_kwargs = {key: value[i_field]
                            for key, value in vector_kwargs.iteritems()}
            field_kwargs['_i_field'] = i_field + 1001
            self.test_fields.append(DummyField(**field_kwargs))

        return self.test_fields

    def _collate_result(self, fields):
        # Invoke the testee, but make it think mock fields are type PPField3.
        with mock.patch('iris.fileformats.um._fast_load_structured_fields.'
                        'PPField3',
                        new=DummyField):
            result = FieldCollation(fields)
        return result

    def _test_fields(self, item):
        # Convert nested tuples/lists of field-numbers into fields.
        if isinstance(item, int):
            result = self.test_fields[item - 1001]
        else:
            result = type(item)(self._test_fields(el) for el in item)
        return result

    def _check_arrays_and_dims(self, result, spec):
        result = result.element_arrays_and_dims
        self.assertEqual(set(result.keys()), set(spec.keys()))
        for keyname in spec.keys():
            result_array, result_dims = result[keyname]
            spec_array, spec_dims = spec[keyname]
            self.assertEqual(result_dims, spec_dims,
                             'element dims differ for "{}": '
                             'result={!r}, expected {!r}'.format(
                                 keyname, result_dims, spec_dims))
            self.assertArrayEqual(result_array, spec_array,
                                  'element arrays differ for "{}": '
                                  'result={!r}, expected {!r}'.format(
                                      keyname, result_array, spec_array))

    def test_none(self):
        with self.assertRaises(AssertionError):
            result = self._collate_result([])

    def test_one(self):
        # A single field does not make a dimension (no length-1 dims).
        fields = self._dummy_fields(date1=17)
        result = self._collate_result(fields)
        self.assertIsInstance(result, FieldCollation)
        self.assertEqual(result.fields, tuple(self.test_fields))
        self.assertEqual(result.primary_dimension_elements, set())
        self.assertEqual(result.vector_dims_shape, (1,))
        self._check_arrays_and_dims(result, {})

    def test_1d_time(self):
        fields = self._dummy_fields(date1=[7, 3, 29])
        result = self._collate_result(fields)
        self.assertIsInstance(result, FieldCollation)
        self.assertEqual(result.fields, self._test_fields((1001, 1002, 1003)))
        self.assertEqual(result.primary_dimension_elements, set(['t1']))
        self.assertEqual(result.vector_dims_shape, (3,))
        self._check_arrays_and_dims(result, {'t1': (_dt([7, 3, 29]), (0,))})

    def test_ensemble(self):
        fields = self._dummy_fields(ensemble=[71, 43])
        result = self._collate_result(fields)
        self.assertIsInstance(result, FieldCollation)
        self.assertEqual(result.fields, self._test_fields((1001, 1002)))
        self.assertEqual(result.primary_dimension_elements, set(['lbrsvd4']))
        self.assertEqual(result.vector_dims_shape, (2,))
        self._check_arrays_and_dims(result, {'lbrsvd4': ([71, 43], (0,))})

    def test_pseudo_level(self):
        fields = self._dummy_fields(pseudo_level=[105, 101])
        result = self._collate_result(fields)
        self.assertIsInstance(result, FieldCollation)
        self.assertEqual(result.fields, self._test_fields((1001, 1002)))
        self.assertEqual(result.primary_dimension_elements, set(['lbuser5']))
        self.assertEqual(result.vector_dims_shape, (2,))
        self._check_arrays_and_dims(result, {'lbuser5': ([105, 101], (0,))})

    def test_height(self):
        fields = self._dummy_fields(blev=[5, 7])
        result = self._collate_result(fields)
        self.assertIsInstance(result, FieldCollation)
        self.assertEqual(result.fields, self._test_fields((1001, 1002)))
        self.assertEqual(result.primary_dimension_elements, set(['blev']))
        self.assertEqual(result.vector_dims_shape, (2,))
        self._check_arrays_and_dims(result, {'blev': ([5, 7], (0,))})

    def test_2d(self):
        fields = self._dummy_fields(ensemble=[2, 2, 2, 3, 3, 3],
                                    pseudo_level=[7, 8, 9, 7, 8, 9])
        result = self._collate_result(fields)
        self.assertIsInstance(result, FieldCollation)
        self.assertEqual(result.fields,
                         self._test_fields(tuple(range(1001, 1007))))
        self.assertEqual(result.primary_dimension_elements,
                         set(['lbrsvd4', 'lbuser5']))
        self.assertEqual(result.vector_dims_shape, (2, 3))
        self._check_arrays_and_dims(result,
                                    {'lbrsvd4': ([2, 3], (0,)),
                                     'lbuser5': ([7, 8, 9], (1,))})

    def test_time_tracking_t1_t2(self):
        fields = self._dummy_fields(date1=[4, 5, 6],
                                    date2=[6, 7, 8])
        result = self._collate_result(fields)
        self.assertIsInstance(result, FieldCollation)
        self.assertEqual(result.fields,
                         self._test_fields((1001, 1002, 1003)))
        self.assertEqual(result.vector_dims_shape, (3,))
        self.assertEqual(result.primary_dimension_elements, set(['t1']))
        self._check_arrays_and_dims(result, {'t1': (_dt([4, 5, 6]), (0,)),
                                             't2': (_dt([6, 7, 8]), (0,))})

    def test_time_forecast_3way(self):
        # An example like forecast data.
        fields = self._dummy_fields(date2=[1, 1, 11, 11],
                                    date1=[15, 16, 25, 26],
                                    lbft=[6, 9, 6, 9])
        result = self._collate_result(fields)
        self.assertIsInstance(result, FieldCollation)
        self.assertEqual(result.fields,
                         self._test_fields(tuple(range(1001, 1005))))
        self.assertEqual(result.vector_dims_shape, (2, 2))
        self.assertEqual(result.primary_dimension_elements,
                         set(['t2', 'lbft']))
        self._check_arrays_and_dims(result,
                                    {'t2': (_dt([1, 11]), (0,)),
                                     't1': (_dt([[15, 16], [25, 26]]), (0, 1)),
                                     'lbft': ([6, 9], (1,))})


class Test__time_comparable_int(tests.IrisTest):

    dummy_collation = FieldCollation([DummyField()])
    time_convert_function = dummy_collation._time_comparable_int

    def test(self):
        # Define a list of date-time tuples, which should remain both all
        # distinct and in ascending order when converted...
        test_date_tuples = [
            # Increment each component in turn to check that all are handled.
            (2004, 1, 1, 0, 0, 0),
            (2004, 1, 1, 0, 0, 1),
            (2004, 1, 1, 0, 1, 0),
            (2004, 1, 1, 1, 0, 0),
            (2004, 1, 2, 0, 0, 0),
            (2004, 2, 1, 0, 0, 0),
            # Go across 2004-02-29 leap-day, and on to "Feb 31 .. Mar 1".
            (2004, 2, 27, 0, 0, 0),
            (2004, 2, 28, 0, 0, 0),
            (2004, 2, 29, 0, 0, 0),
            (2004, 2, 30, 0, 0, 0),
            (2004, 2, 31, 0, 0, 0),
            (2004, 3, 1, 0, 0, 0),
            (2005, 1, 1, 0, 0, 0)]

        test_date_ints = [self.time_convert_function(*test_tuple)
                          for test_tuple in test_date_tuples]
        # Check all values are distinct.
        self.assertEqual(len(test_date_ints), len(set(test_date_ints), ))
        # Check all values are in order.
        self.assertEqual(test_date_ints, sorted(test_date_ints))


if __name__ == "__main__":
    tests.main()
