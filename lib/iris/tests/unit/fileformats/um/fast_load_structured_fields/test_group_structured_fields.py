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
:func:`iris.fileformats.um._fast_load_structured_fields.group_structured_fields`.

"""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import mock
import numpy as np

import iris

from iris.fileformats.um._fast_load_structured_fields import \
    group_structured_fields


def _tovec(value, length, default):
    if value is None:
        value = default
    try:
        own_length = len(value)
        assert own_length == length
    except:
        value = [value] * length
    return value


class Test__grouping(tests.IrisTest):
    def _dummy_fields_iter(self, stashes=None, models=None, lbprocs=None):
        # Make a group of test fields, and return an iterator over it.
        a_vec = [vec for vec in (stashes, models, lbprocs) if vec is not None]
        number = len(a_vec[0])
        stashes = _tovec(stashes, number, default=31)
        models = _tovec(models, number, default=71)
        lbprocs = _tovec(lbprocs, number, default=91)
        self.test_fields = [
            mock.MagicMock(lbuser=[0, 0, 0, x_stash, 0, 0, x_model],
                           lbproc=x_lbproc,
                           i_field=ind + 1001)
            for ind, x_stash, x_model, x_lbproc in zip(
                range(number), stashes, models, lbprocs)]
        return (fld for fld in self.test_fields)

    def _group_result(self, fields):
        # Run the testee, but returning just the groups (not FieldCollations).
        with mock.patch('iris.fileformats.um._fast_load_structured_fields.'
                        'FieldCollation', new=lambda args: args):
            result = list(group_structured_fields(fields))
        return result

    def _test_fields(self, item):
        # Convert nested tuples/lists of field-numbers into fields.
        if isinstance(item, int):
            result = self.test_fields[item - 1001]
        else:
            result = type(item)(self._test_fields(el) for el in item)
        return result

    def test_none(self):
        null_iter = (x for x in [])
        result = self._group_result(null_iter)
        self.assertEqual(result, [])

    def test_one(self):
        fields_iter = self._dummy_fields_iter(stashes=[1])
        result = self._group_result(fields_iter)
        self.assertEqual(result, self._test_fields([(1001,)]))

    def test_allsame(self):
        fields_iter = self._dummy_fields_iter(stashes=[1, 1, 1])
        result = self._group_result(fields_iter)
        self.assertEqual(result, self._test_fields([(1001, 1002, 1003)]))

    def test_stashes_different(self):
        fields_iter = self._dummy_fields_iter(stashes=[1, 1, 22, 1, 22, 333])
        result = self._group_result(fields_iter)
        self.assertEqual(result, self._test_fields([(1001, 1002, 1004),
                                                    (1003, 1005),
                                                    (1006,)]))

    def test_models_different(self):
        fields_iter = self._dummy_fields_iter(models=[10, 21, 10])
        result = self._group_result(fields_iter)
        self.assertEqual(result, self._test_fields([(1001, 1003), (1002,)]))

    def test_lbprocs_different(self):
        fields_iter = self._dummy_fields_iter(lbprocs=[991, 995, 991])
        result = self._group_result(fields_iter)
        self.assertEqual(result, self._test_fields([(1001, 1003), (1002,)]))

    def test_2d_combines(self):
        fields_iter = self._dummy_fields_iter(
            stashes=[11, 11, 15, 11],
            lbprocs=[31, 42, 31, 42])
        result = self._group_result(fields_iter)
        self.assertEqual(result, self._test_fields([(1001,),
                                                    (1002, 1004),
                                                    (1003,)]))

    def test_sortorder(self):
        fields_iter = self._dummy_fields_iter(stashes=[11, 7, 12])
        result = self._group_result(fields_iter)
        self.assertEqual(result, self._test_fields([(1002,),
                                                    (1001,),
                                                    (1003,)]))

    def test_sortorder_2d(self):
        fields_iter = self._dummy_fields_iter(
            stashes=[11, 11, 12],
            lbprocs=[31,  9,  1])
        result = self._group_result(fields_iter)
        self.assertEqual(result, self._test_fields([(1002,),
                                                    (1001,),
                                                    (1003,)]))


if __name__ == "__main__":
    tests.main()
