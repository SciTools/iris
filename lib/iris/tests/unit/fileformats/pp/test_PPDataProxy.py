# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.fileformats.pp.PPDataProxy` class."""

from iris.fileformats.pp import PPDataProxy, SplittableInt


class Test_lbpack:
    def test_lbpack_splittable_int(self, mocker):
        lbpack = mocker.Mock(spec_set=SplittableInt)
        proxy = PPDataProxy(None, None, None, None, None, lbpack, None, None)
        assert proxy.lbpack == lbpack
        assert proxy.lbpack is lbpack

    def test_lbpack_raw(self):
        lbpack = 4321
        proxy = PPDataProxy(None, None, None, None, None, lbpack, None, None)
        assert proxy.lbpack == lbpack
        assert proxy.lbpack is not lbpack
        assert isinstance(proxy.lbpack, SplittableInt)
        assert proxy.lbpack.n1 == lbpack % 10
        assert proxy.lbpack.n2 == lbpack // 10 % 10
        assert proxy.lbpack.n3 == lbpack // 100 % 10
        assert proxy.lbpack.n4 == lbpack // 1000 % 10
