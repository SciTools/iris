# (C) British Crown Copyright 2013, Met Office
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
"""Unit tests for the `iris.pdatetime.PartialDateTime` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import operator

from datetime import datetime

from iris.pdatetime import PartialDateTime
import iris.tests.unit as unit


class Test___init__(tests.IrisTest):
    def test_positional(self):
        # Test that we can define PartialDateTimes with positional arguments.
        pd = PartialDateTime(1066, None, 10)
        self.assertEqual(pd.year, 1066)
        self.assertEqual(pd.month, None)
        self.assertEqual(pd.day, 10)

    def test_keyword_args(self):
        # Test that we can define PartialDateTimes with keyword arguments.
        pd = PartialDateTime(microsecond=10)
        self.assertEqual(pd.year, None)
        self.assertEqual(pd.microsecond, 10)


class Test_timetuple(tests.IrisTest):
    def test_time_tuple(self):
        # Check that the PartialDateTime class implements a timetuple (needed
        # because of http://bugs.python.org/issue8005).
        pd = PartialDateTime(*range(7))
        self.assertEqual(pd.timetuple, tuple(range(7)))

    def test_datetime_python(self):
        # Ensure that 'bug' is still present with datetime object comparison.
        # If NotImplemented is raised, then the timetuple workaround can be
        # removed.
        NoTuple = type('NoTuple', PartialDateTime.__bases__,
                       dict(PartialDateTime.__dict__))
        del NoTuple.timetuple

        with self.assertRaises(TypeError):
            datetime(1, 1, 1) <= NoTuple(1, 1, 1)


class Test__compare(tests.IrisTest):
    def mocked_partial_datetime(
            self, year=None, month=None, day=None, hour=None, minute=None,
            second=None, microsecond=None):
        """
        Construct a mocked PartialDateTime, with a _fields attribute and
        such that it is an iterator of the given "iter_values".

        """
        partial = mock.Mock(spec=PartialDateTime,
                            _fields=PartialDateTime._fields,
                            year=year, month=month, day=day, hour=hour,
                            minute=minute, second=second,
                            microsecond=microsecond)
        iter_values = [getattr(partial, field_name)
                       for field_name in partial._fields]
        partial.__iter__ = mock.Mock(return_value=iter(iter_values))
        return partial

    def test_type_not_implemented(self):
        # Check that NotImplemented is returned for bad types (an int in
        # this case).
        pd = PartialDateTime(*range(7))
        self.assertEqual(pd._compare(1), NotImplemented)

    def test_not_implemented_types(self):
        # Check that "known_time_implementations" is used to check whether a
        # type can be compared with a PartialDateTime.
        pd = PartialDateTime(*range(7))
        with unit.patched_isinstance(return_value=False) as new_isinstance:
            self.assertEqual(pd._compare(1), NotImplemented)
        new_isinstance.assert_called_once_with(
            1, PartialDateTime.known_time_implementations)


class _Test_operator(object):
    def _other(self):
        self.other = mock.Mock(name='datetime_cmp', spec=datetime, year=3,
                               month=3, day=3, second=3)

    def test_no_difference(self):
        res = self.op(PartialDateTime(year=3, month=3, day=3, second=3),
                      self.other)
        self.assertIs(self.return_value['no_difference'], res)

    def test_item1_lo(self):
        res = self.op(PartialDateTime(year=1, month=3, second=3),
                      self.other)
        self.assertIs(self.return_value['item1_lo'], res)

    def test_item1_hi(self):
        res = self.op(PartialDateTime(year=5, month=3, day=4),
                      self.other)
        self.assertIs(self.return_value['item1_hi'], res)

    def test_item2_lo(self):
        res = self.op(PartialDateTime(year=3, month=1, second=3),
                      self.other)
        self.assertIs(self.return_value['item2_lo'], res)

    def test_item2_hi(self):
        res = self.op(PartialDateTime(year=3, month=5, day=4),
                      self.other)
        self.assertIs(self.return_value['item2_hi'], res)

    def test_item3_lo(self):
        res = self.op(PartialDateTime(year=3, month=3, second=1),
                      self.other)
        self.assertIs(self.return_value['item3_lo'], res)

    def test_item3_hi(self):
        res = self.op(PartialDateTime(year=3, month=3, day=5),
                      self.other)
        self.assertIs(self.return_value['item3_hi'], res)

    def test_mix_hi_lo(self):
        res = self.op(PartialDateTime(year=5, month=1, day=5),
                      self.other)
        self.assertIs(self.return_value['mix_hi_lo'], res)

    def test_mix_lo_hi(self):
        res = self.op(PartialDateTime(year=1, month=5, day=5),
                      self.other)
        self.assertIs(self.return_value['mix_lo_hi'], res)


class Test___eq__(tests.IrisTest, _Test_operator):
    def setUp(self):
        self.op = operator.eq
        self._other()
        self.return_value = {
            'no_difference': True, 'item1_lo': False, 'item1_hi': False,
            'item2_lo': False, 'item2_hi': False, 'item3_lo': False,
            'item3_hi': False, 'mix_hi_lo': False, 'mix_lo_hi': False}


class Test___ne__(tests.IrisTest, _Test_operator):
    def setUp(self):
        self.op = operator.ne
        self._other()
        self.return_value = {
            'no_difference': False, 'item1_lo': True, 'item1_hi': True,
            'item2_lo': True, 'item2_hi': True, 'item3_lo': True,
            'item3_hi': True, 'mix_hi_lo': True, 'mix_lo_hi': True}


class Test___gt__(tests.IrisTest, _Test_operator):
    def setUp(self):
        self.op = operator.gt
        self._other()
        self.return_value = {
            'no_difference': False, 'item1_lo': False, 'item1_hi': True,
            'item2_lo': False, 'item2_hi': True, 'item3_lo': False,
            'item3_hi': True, 'mix_hi_lo': True, 'mix_lo_hi': False}


class Test___lt__(tests.IrisTest, _Test_operator):
    def setUp(self):
        self.op = operator.lt
        self._other()
        self.return_value = {
            'no_difference': False, 'item1_lo': True, 'item1_hi': False,
            'item2_lo': True, 'item2_hi': False, 'item3_lo': True,
            'item3_hi': False, 'mix_hi_lo': False, 'mix_lo_hi': True}


class Test___ge__(tests.IrisTest, _Test_operator):
    def setUp(self):
        self.op = operator.ge
        self._other()
        self.return_value = {
            'no_difference': True, 'item1_lo': False, 'item1_hi': True,
            'item2_lo': False, 'item2_hi': True, 'item3_lo': False,
            'item3_hi': True, 'mix_hi_lo': True, 'mix_lo_hi': False}


class Test___le__(tests.IrisTest, _Test_operator):
    def setUp(self):
        self.op = operator.le
        self._other()
        self.return_value = {
            'no_difference': True, 'item1_lo': True, 'item1_hi': False,
            'item2_lo': True, 'item2_hi': False, 'item3_lo': True,
            'item3_hi': False, 'mix_hi_lo': False, 'mix_lo_hi': True}


class Test__comparison_x(tests.IrisTest):
    def setUp(self):
        self.other = mock.Mock(name='datetime_cmp', spec=datetime, year=3,
                               month=3, day=3, hour=3)
        self.pd = mock.Mock(name='dummy_pdt', spec=PartialDateTime,
                            _fields=['year', 'month', 'day'])
        self.op = mock.Mock(name='operator')

    def test_skip_missing_attribute(self):
        # Test that each attribute is compared between self and other, other
        # than self attributes of None.
        self.pd.__iter__ = mock.Mock(return_value=iter([0, None, 1]))
        PartialDateTime._comparison_x(self.pd, self.op, self.other)

        self.assertEqual(self.op.mock_calls,
                         [mock.call(0, self.other.year),
                          mock.call(1, self.other.day)])

    def test_return_false_with_no_attrib(self):
        # Ensure that false is returned when no attribute is compared against.
        self.pd.__iter__ = mock.Mock(return_value=iter([None, None, None]))
        res = PartialDateTime._comparison_x(self.pd, self.op, self.other)

        self.assertFalse(self.op.called)
        self.assertIs(res, False)


class Test__comparison_xe(tests.IrisTest):
    def setUp(self):
        self.other = mock.Mock(name='datetime_cmp', spec=datetime, year=3,
                               month=3, day=3, hour=3)
        self.pd = mock.Mock(name='dummy_pdt', spec=PartialDateTime,
                            _fields=['year', 'month', 'day'])
        self.op = mock.Mock(name='operator')

    def test_skip_missing_attribute(self):
        # Test that each attribute is compared between self and other, other
        # than self attributes of None.
        self.pd.__iter__ = mock.Mock(return_value=iter([0, None, 1]))
        PartialDateTime._comparison_xe(self.pd, self.op, self.other)

        self.assertEqual(self.op.mock_calls,
                         [mock.call(0, self.other.year),
                          mock.call(1, self.other.day)])

    def test_return_false_with_no_attrib(self):
        # Ensure that false is returned when no attribute is compared against.
        self.pd.__iter__ = mock.Mock(return_value=iter([None, None, None]))
        res = PartialDateTime._comparison_xe(self.pd, self.op, self.other)

        self.assertFalse(self.op.called)
        self.assertIs(res, False)


class Test__comparison_eq(tests.IrisTest):
    def setUp(self):
        self.other = mock.Mock(name='datetime_cmp', spec=datetime, year=3,
                               month=3, day=3, hour=3)
        self.pd = mock.Mock(name='dummy_pdt', spec=PartialDateTime,
                            _fields=['year', 'month', 'day'])
        self.op = mock.Mock(name='operator')

    def test_skip_missing_attribute(self):
        # Test that each attribute is compared between self and other, other
        # than self attributes of None.
        self.pd.__iter__ = mock.Mock(return_value=iter([0, None, 1]))
        PartialDateTime._comparison_eq(self.pd, self.op, self.other)

        self.assertEqual(self.op.mock_calls,
                         [mock.call(0, self.other.year),
                          mock.call(1, self.other.day)])

    def test_return_false_with_no_attrib(self):
        # Ensure that false is returned when no attribute is compared against.
        self.pd.__iter__ = mock.Mock(return_value=iter([None, None, None]))
        res = PartialDateTime._comparison_eq(self.pd, self.op, self.other)

        self.assertFalse(self.op.called)
        self.assertIs(res, False)


if __name__ == "__main__":
    tests.main()
