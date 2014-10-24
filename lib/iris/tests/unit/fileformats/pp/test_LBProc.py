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
"""Unit tests for :class:`iris.fileformats.pp.LBProc`."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

from iris.fileformats.pp import LBProc


class Test___init__(tests.IrisTest):
    def test_int(self):
        LBProc(42)

    def test_str(self):
        LBProc('245')

    def test_negative(self):
        with self.assertRaises(ValueError):
            LBProc(-1)

    def test_invalid_str(self):
        with self.assertRaises(ValueError):
            LBProc('asdf')


class Test_flag1(tests.IrisTest):
    def test_true(self):
        self.assertTrue(LBProc(1).flag1)

    def test_false(self):
        self.assertFalse(LBProc(2).flag1)

    def test_many(self):
        for i in xrange(100):
            self.assertEqual(LBProc(i).flag1, i & 1)


class Test_flag2(tests.IrisTest):
    def test_true(self):
        self.assertTrue(LBProc(3).flag2)

    def test_false(self):
        self.assertFalse(LBProc(1).flag2)

    def test_many(self):
        for i in xrange(100):
            self.assertEqual(LBProc(i).flag2, bool(i & 2))


class Test_flag4(tests.IrisTest):
    def test_true(self):
        self.assertTrue(LBProc(6).flag4)

    def test_false(self):
        self.assertFalse(LBProc(8).flag2)

    def test_many(self):
        for i in xrange(100):
            self.assertEqual(LBProc(i).flag4, bool(i & 4))


class Test_flag131072(tests.IrisTest):
    def test_true(self):
        self.assertTrue(LBProc(135448).flag131072)

    def test_false(self):
        self.assertFalse(LBProc(4376).flag131072)

    def test_many(self):
        for i in xrange(0, 260000, 1000):
            self.assertEqual(LBProc(i).flag131072, bool(i & 131072))


class Test_flag3(tests.IrisTest):
    def test_invalid(self):
        lbproc = LBProc(9)
        with self.assertRaises(AttributeError):
            lbproc.flag3


class Test_flag262144(tests.IrisTest):
    def test_invalid(self):
        lbproc = LBProc(9)
        with self.assertRaises(AttributeError):
            lbproc.flag262144


class Test___int__(tests.IrisTest):
    def test(self):
        self.assertEqual(int(LBProc(99)), 99)


class Test___eq__(tests.IrisTest):
    def test_equal(self):
        self.assertTrue(LBProc(17).__eq__(LBProc(17)))

    def test_equal_int(self):
        self.assertTrue(LBProc(17).__eq__(17))

    def test_not_equal(self):
        self.assertFalse(LBProc(17).__eq__(LBProc(18)))

    def test_not_equal_int(self):
        self.assertFalse(LBProc(17).__eq__(16))


class Test___ne__(tests.IrisTest):
    def test_equal(self):
        self.assertFalse(LBProc(7).__ne__(LBProc(7)))

    def test_equal_int(self):
        self.assertFalse(LBProc(8).__ne__(8))

    def test_not_equal(self):
        self.assertTrue(LBProc(9).__ne__(LBProc(14)))

    def test_not_equal_int(self):
        self.assertTrue(LBProc(10).__ne__(15))


class Test___iadd__(tests.IrisTest):
    def test(self):
        lbproc = LBProc(12)
        lbproc += 8
        self.assertEqual(int(lbproc), 20)


class Test___iand__(tests.IrisTest):
    def test(self):
        lbproc = LBProc(12)
        lbproc &= 8
        self.assertEqual(int(lbproc), 8)


class Test___ior__(tests.IrisTest):
    def test(self):
        lbproc = LBProc(12)
        lbproc |= 1
        self.assertEqual(int(lbproc), 13)


class Test_flags(tests.IrisTest):
    def test(self):
        lbproc = LBProc(26)
        self.assertEqual(lbproc.flags, (2, 8, 16))


class Test___str__(tests.IrisTest):
    def test(self):
        lbproc = LBProc(8641)
        self.assertEqual(str(lbproc), '8641')


class Test___len__(tests.IrisTest):
    def test_zero(self):
        lbproc = LBProc(0)
        with mock.patch('warnings.warn') as warn:
            length = len(lbproc)
        warn.assert_called_once()
        self.assertEqual(length, 1)

    def test_positive(self):
        lbproc = LBProc(24)
        with mock.patch('warnings.warn') as warn:
            length = len(lbproc)
        warn.assert_called_once()
        self.assertEqual(length, 2)


class Test___getitem__(tests.IrisTest):
    def test(self):
        lbproc = LBProc(1234)
        with mock.patch('warnings.warn') as warn:
            digit = lbproc[1]
        warn.assert_called_once()
        self.assertEqual(digit, 3)


class Test___setitem__(tests.IrisTest):
    def test_ok(self):
        lbproc = LBProc(1234)
        with mock.patch('warnings.warn') as warn:
            lbproc[1] = 9
        warn.assert_called_once()
        self.assertEqual(int(lbproc), 1294)

    def test_invalid(self):
        lbproc = LBProc(1234)
        with mock.patch('warnings.warn') as warn:
            with self.assertRaises(ValueError):
                lbproc[1] = 81
        warn.assert_called_once()


if __name__ == '__main__':
    tests.main()
