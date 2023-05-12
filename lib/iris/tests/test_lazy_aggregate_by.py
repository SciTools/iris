# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
import unittest

from iris._lazy_data import as_lazy_data
from iris.analysis import SUM
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


class TestLazyAggregateByWeightedByCube(TestLazyAggregateBy):
    def setUp(self):
        super().setUp()

        self.weights_single = self.cube_single[:, 0, 0].copy(
            self.weights_single
        )
        self.weights_single.units = "m2"
        self.weights_multi = self.cube_multi[:, 0, 0].copy(self.weights_multi)
        self.weights_multi.units = "m2"

    def test_str_aggregation_weighted_sum_single(self):
        aggregateby_cube = self.cube_single.aggregated_by(
            "height",
            SUM,
            weights=self.weights_single,
        )
        self.assertEqual(aggregateby_cube.units, "kelvin m2")

    def test_coord_aggregation_weighted_sum_single(self):
        aggregateby_cube = self.cube_single.aggregated_by(
            self.coord_z_single,
            SUM,
            weights=self.weights_single,
        )
        self.assertEqual(aggregateby_cube.units, "kelvin m2")

    def test_str_aggregation_weighted_sum_multi(self):
        aggregateby_cube = self.cube_multi.aggregated_by(
            ["height", "level"],
            SUM,
            weights=self.weights_multi,
        )
        self.assertEqual(aggregateby_cube.units, "kelvin m2")

    def test_str_aggregation_rev_order_weighted_sum_multi(self):
        aggregateby_cube = self.cube_multi.aggregated_by(
            ["level", "height"],
            SUM,
            weights=self.weights_multi,
        )
        self.assertEqual(aggregateby_cube.units, "kelvin m2")

    def test_coord_aggregation_weighted_sum_multi(self):
        aggregateby_cube = self.cube_multi.aggregated_by(
            [self.coord_z1_multi, self.coord_z2_multi],
            SUM,
            weights=self.weights_multi,
        )
        self.assertEqual(aggregateby_cube.units, "kelvin m2")

    def test_coord_aggregation_rev_order_weighted_sum_multi(self):
        aggregateby_cube = self.cube_multi.aggregated_by(
            [self.coord_z2_multi, self.coord_z1_multi],
            SUM,
            weights=self.weights_multi,
        )
        self.assertEqual(aggregateby_cube.units, "kelvin m2")


if __name__ == "__main__":
    unittest.main()
