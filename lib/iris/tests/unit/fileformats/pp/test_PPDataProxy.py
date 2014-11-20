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
"""Unit tests for the `iris.fileformats.pp.PPDataProxy` class."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

from iris.fileformats.pp import PPDataProxy, SplittableInt


class Test_lbpack(tests.IrisTest):
    def test_lbpack_SplittableInt(self):
        lbpack = mock.Mock(spec_set=SplittableInt)
        proxy = PPDataProxy(None, None, None, None,
                            None, lbpack, None, None, None)
        self.assertEqual(proxy.lbpack, lbpack)
        self.assertIs(proxy.lbpack, lbpack)

    def test_lnpack_raw(self):
        lbpack = 4321
        proxy = PPDataProxy(None, None, None, None,
                            None, lbpack, None, None, None)
        self.assertEqual(proxy.lbpack, lbpack)
        self.assertIsNot(proxy.lbpack, lbpack)
        self.assertIsInstance(proxy.lbpack, SplittableInt)
        self.assertEqual(proxy.lbpack.n1, lbpack % 10)
        self.assertEqual(proxy.lbpack.n2, lbpack // 10 % 10)
        self.assertEqual(proxy.lbpack.n3, lbpack // 100 % 10)
        self.assertEqual(proxy.lbpack.n4, lbpack // 1000 % 10)


if __name__ == '__main__':
    tests.main()
