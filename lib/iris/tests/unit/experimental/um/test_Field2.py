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
Unit tests for :class:`iris.experimental.um.Field2`.

"""

from __future__ import (absolute_import, division, print_function)

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy as np

from iris.experimental.um import Field2


def make_field():
    headers = (np.arange(64) + 1) * 10
    return Field2(headers[:45], headers[45:], None)


class Test_lbyr(tests.IrisTest):
    def test(self):
        field = make_field()
        self.assertEqual(field.lbyr, 10)


class Test_lbmon(tests.IrisTest):
    def test(self):
        field = make_field()
        self.assertEqual(field.lbmon, 20)


class Test_lbday(tests.IrisTest):
    def test(self):
        field = make_field()
        self.assertEqual(field.lbday, 60)


class Test_lbrsvd1(tests.IrisTest):
    def test(self):
        field = make_field()
        self.assertEqual(field.lbrsvd1, 340)


class Test_lbrsvd4(tests.IrisTest):
    def test(self):
        field = make_field()
        self.assertEqual(field.lbrsvd4, 370)


class Test_lbuser7(tests.IrisTest):
    def test(self):
        field = make_field()
        self.assertEqual(field.lbuser7, 450)


class Test_bdx(tests.IrisTest):
    def test(self):
        field = make_field()
        self.assertEqual(field.bdx, 620)


if __name__ == '__main__':
    tests.main()
