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
Unit tests for the function
:func:`iris.fileformats.um._optimal_array_structuring.optimal_array_structure`.

"""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import numpy as np

from iris.fileformats.um._optimal_array_structuring import \
    optimal_array_structure


class Test(tests.IrisTest):
    def _check_arrays_and_dims(self, result, spec):
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
        with self.assertRaises(IndexError):
            result = optimal_array_structure([], [])

    def test_one(self):
        # A single value does not make a dimension (no length-1 dims).
        elements = [('a', np.array([1]))]
        dims, primaries, elems_and_dims = optimal_array_structure(elements)
        self.assertEqual(dims, (1,))
        self.assertEqual(primaries, set())
        self.assertEqual(elems_and_dims, {})

    def test_1d(self):
        elements = [('a', np.array([1, 2, 4]))]
        dims, primaries, elems_and_dims = optimal_array_structure(elements)
        self.assertEqual(dims, (3,))
        self.assertEqual(primaries, set('a'))
        self._check_arrays_and_dims(elems_and_dims,
                                    {'a': (np.array([1, 2, 4]), (0,))})

    def test_1d_actuals(self):
        # Test use of alternate element values for array construction.
        elements = [('a', np.array([1, 2, 4]))]
        actual_values = [('a', np.array([7, 3, 9]))]
        dims, primaries, elems_and_dims = optimal_array_structure(
            elements, actual_values)
        self.assertEqual(dims, (3,))
        self.assertEqual(primaries, set('a'))
        self._check_arrays_and_dims(elems_and_dims,
                                    {'a': (np.array([7, 3, 9]), (0,))})

    def test_actuals_mismatch_fail(self):
        elements = [('a', np.array([1, 2, 4]))]
        actual_values = [('b', np.array([7, 3, 9]))]
        with self.assertRaisesRegexp(ValueError, 'Names.* do not match.*'):
            dims, primaries, elems_and_dims = optimal_array_structure(
                elements, actual_values)

    def test_2d(self):
        elements = [('a', np.array([2, 2, 2, 3, 3, 3])),
                    ('b', np.array([7, 8, 9, 7, 8, 9]))]
        dims, primaries, elems_and_dims = optimal_array_structure(elements)
        self.assertEqual(dims, (2, 3,))
        self.assertEqual(primaries, set(['a', 'b']))
        self._check_arrays_and_dims(elems_and_dims,
                                    {'a': (np.array([2, 3]), (0,)),
                                     'b': (np.array([7, 8, 9]), (1,))})

    def test_non_2d(self):
        # An incomplete 2d expansion just becomes 1d
        elements = [('a', np.array([2, 2, 2, 3, 3])),
                    ('b', np.array([7, 8, 9, 7, 8]))]
        dims, primaries, elems_and_dims = optimal_array_structure(elements)
        self.assertEqual(dims, (5,))
        self.assertEqual(primaries, set())
        self._check_arrays_and_dims(elems_and_dims,
                                    {'a': (np.array([2, 2, 2, 3, 3]), (0,)),
                                     'b': (np.array([7, 8, 9, 7, 8]), (0,))})

    def test_degenerate(self):
        # A all-same vector does not appear in the output.
        elements = [('a', np.array([1, 2, 3])),
                    ('b', np.array([4, 4, 4]))]
        dims, primaries, elems_and_dims = optimal_array_structure(elements)
        self.assertEqual(dims, (3,))
        self.assertEqual(primaries, set(['a']))
        self._check_arrays_and_dims(elems_and_dims,
                                    {'a': (np.array([1, 2, 3]), (0,))})

    def test_1d_duplicates(self):
        # When two have the same structure, the first is 'the dimension'.
        elements = [('a', np.array([1, 3, 4])),
                    ('b', np.array([6, 7, 9]))]
        dims, primaries, elems_and_dims = optimal_array_structure(elements)
        self.assertEqual(dims, (3,))
        self.assertEqual(primaries, set('a'))
        self._check_arrays_and_dims(elems_and_dims,
                                    {'a': (np.array([1, 3, 4]), (0,)),
                                     'b': (np.array([6, 7, 9]), (0,))})

    def test_1d_duplicates_order(self):
        # Same as previous but reverse passed order of elements 'a' and 'b'.
        elements = [('b', np.array([6, 7, 9])),
                    ('a', np.array([1, 3, 4]))]
        dims, primaries, elems_and_dims = optimal_array_structure(elements)
        self.assertEqual(dims, (3,))
        # The only difference is the one chosen as 'principal'
        self.assertEqual(primaries, set('b'))
        self._check_arrays_and_dims(elems_and_dims,
                                    {'a': (np.array([1, 3, 4]), (0,)),
                                     'b': (np.array([6, 7, 9]), (0,))})

    def test_3_way(self):
        elements = [('t1', np.array([2, 3, 4])),
                    ('t2', np.array([4, 5, 6])),
                    ('period', np.array([9, 8, 7]))]
        dims, primaries, elems_and_dims = optimal_array_structure(elements)
        self.assertEqual(dims, (3,))
        self.assertEqual(primaries, set(['t1']))
        self._check_arrays_and_dims(elems_and_dims,
                                    {'t1': (np.array([2, 3, 4]), (0,)),
                                     't2': (np.array([4, 5, 6]), (0,)),
                                     'period': (np.array([9, 8, 7]), (0,))})

    def test_mixed_dims(self):
        elements = [('t1', np.array([1, 1, 11, 11])),
                    ('t2', np.array([15, 16, 25, 26])),
                    ('ft', np.array([15, 16, 15, 16]))]
        dims, primaries, elems_and_dims = optimal_array_structure(elements)
        self.assertEqual(dims, (2, 2))
        self.assertEqual(primaries, set(['t1', 'ft']))
        self._check_arrays_and_dims(
            elems_and_dims,
            {'t1': (np.array([1, 11]), (0,)),
             't2': (np.array([[15, 16], [25, 26]]), (0, 1)),
             'ft': (np.array([15, 16]), (1,))})

    def test_missing_dim(self):
        # Case with no dimension element for dimension 1.
        elements = [('t1', np.array([1, 1, 11, 11])),
                    ('t2', np.array([15, 16, 25, 26]))]
        dims, primaries, elems_and_dims = optimal_array_structure(elements)
        self.assertEqual(dims, (4,))
        # The potential 2d nature can not be recognised.
        # 't1' is auxiliary, as it has duplicate values over the dimension.
        self.assertEqual(primaries, set(['t2']))
        self._check_arrays_and_dims(
            elems_and_dims,
            {'t1': (np.array([1, 1, 11, 11]), (0,)),
             't2': (np.array([15, 16, 25, 26]), (0,))})


if __name__ == "__main__":
    tests.main()
