# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.fileformats.pp.PPDataProxy` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

from iris.fileformats.pp import PPDataProxy, SplittableInt


class Test_lbpack(tests.IrisTest):
    def test_lbpack_SplittableInt(self):
        lbpack = mock.Mock(spec_set=SplittableInt)
        proxy = PPDataProxy(None, None, None, None, None, lbpack, None, None)
        self.assertEqual(proxy.lbpack, lbpack)
        self.assertIs(proxy.lbpack, lbpack)

    def test_lbpack_raw(self):
        lbpack = 4321
        proxy = PPDataProxy(None, None, None, None, None, lbpack, None, None)
        self.assertEqual(proxy.lbpack, lbpack)
        self.assertIsNot(proxy.lbpack, lbpack)
        self.assertIsInstance(proxy.lbpack, SplittableInt)
        self.assertEqual(proxy.lbpack.n1, lbpack % 10)
        self.assertEqual(proxy.lbpack.n2, lbpack // 10 % 10)
        self.assertEqual(proxy.lbpack.n3, lbpack // 100 % 10)
        self.assertEqual(proxy.lbpack.n4, lbpack // 1000 % 10)


if __name__ == "__main__":
    tests.main()
