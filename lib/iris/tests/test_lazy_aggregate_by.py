# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
import unittest

from iris._lazy_data import as_lazy_data
from iris.tests import test_aggregate_by


# Simply redo the tests of test_aggregate_by.py with lazy data
class TestLazyAggregateBy(test_aggregate_by.TestAggregateBy):
    def setUp(self):
        super().setUp()

        self.cube_single.data = as_lazy_data(self.cube_single.data)
        self.cube_multi.data = as_lazy_data(self.cube_multi.data)
        self.cube_single_masked.data = as_lazy_data(
            self.cube_single_masked.data
        )
        self.cube_multi_masked.data = as_lazy_data(self.cube_multi_masked.data)
        self.cube_easy.data = as_lazy_data(self.cube_easy.data)
        self.cube_easy_weighted.data = as_lazy_data(
            self.cube_easy_weighted.data
        )

        assert self.cube_single.has_lazy_data()
        assert self.cube_multi.has_lazy_data()
        assert self.cube_single_masked.has_lazy_data()
        assert self.cube_multi_masked.has_lazy_data()
        assert self.cube_easy.has_lazy_data()
        assert self.cube_easy_weighted.has_lazy_data()

    def tearDown(self):
        super().tearDown()

        # Note: weighted easy cube is not expected to have lazy data since
        # WPERCENTILE is not lazy.
        assert self.cube_single.has_lazy_data()
        assert self.cube_multi.has_lazy_data()
        assert self.cube_single_masked.has_lazy_data()
        assert self.cube_multi_masked.has_lazy_data()
        assert self.cube_easy.has_lazy_data()


if __name__ == "__main__":
    unittest.main()
