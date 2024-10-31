# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
import pytest

from iris._lazy_data import as_lazy_data
from iris.analysis import SUM
from iris.cube import Cube
from iris.tests import test_aggregate_by


# Simply redo the tests of test_aggregate_by.py with lazy data
class TestLazyAggregateBy(test_aggregate_by.TestAggregateBy):
    @pytest.fixture(autouse=True)
    def _setup_subclass(self, _setup):
        # Requests _setup to ensure this fixture runs AFTER _setup.
        self.cube_single.data = as_lazy_data(self.cube_single.data)
        self.cube_multi.data = as_lazy_data(self.cube_multi.data)
        self.cube_single_masked.data = as_lazy_data(self.cube_single_masked.data)
        self.cube_multi_masked.data = as_lazy_data(self.cube_multi_masked.data)
        self.cube_easy.data = as_lazy_data(self.cube_easy.data)
        self.cube_easy_weighted.data = as_lazy_data(self.cube_easy_weighted.data)

    @pytest.fixture(autouse=True)
    def _lazy_checks(self, _setup_subclass):
        # Requests _setup_subclass to ensure this fixture runs AFTER _setup_subclass.
        # TODO: ASSERTS IN FIXTURES ARE AN ANTIPATTERN, find an alternative.
        #  https://github.com/m-burst/flake8-pytest-style/issues/31
        #  (have given this a few hours without success, something to revisit).
        def _checker(cubes: list[Cube]):
            for cube in cubes:
                assert cube.has_lazy_data()

        _checker(
            [
                self.cube_single,
                self.cube_multi,
                self.cube_single_masked,
                self.cube_multi_masked,
                self.cube_easy,
                self.cube_easy_weighted,
            ]
        )

        yield

        _checker(
            [
                self.cube_single,
                self.cube_multi,
                self.cube_single_masked,
                self.cube_multi_masked,
                self.cube_easy,
                # Note: weighted easy cube is not expected to have lazy data since
                #  WPERCENTILE is not lazy.
            ]
        )


class TestLazyAggregateByWeightedByCube(TestLazyAggregateBy):
    @pytest.fixture(autouse=True)
    def _setup_sub2(self, _setup_subclass):
        # Requests _setup_subclass to ensure this fixture runs AFTER _setup_subclass.

        self.weights_single = self.cube_single[:, 0, 0].copy(self.weights_single)
        self.weights_single.units = "m2"
        self.weights_multi = self.cube_multi[:, 0, 0].copy(self.weights_multi)
        self.weights_multi.units = "m2"

    def test_str_aggregation_weighted_sum_single(self):
        aggregateby_cube = self.cube_single.aggregated_by(
            "height",
            SUM,
            weights=self.weights_single,
        )
        assert aggregateby_cube.units == "kelvin m2"

    def test_coord_aggregation_weighted_sum_single(self):
        aggregateby_cube = self.cube_single.aggregated_by(
            self.coord_z_single,
            SUM,
            weights=self.weights_single,
        )
        assert aggregateby_cube.units == "kelvin m2"

    def test_str_aggregation_weighted_sum_multi(self):
        aggregateby_cube = self.cube_multi.aggregated_by(
            ["height", "level"],
            SUM,
            weights=self.weights_multi,
        )
        assert aggregateby_cube.units == "kelvin m2"

    def test_str_aggregation_rev_order_weighted_sum_multi(self):
        aggregateby_cube = self.cube_multi.aggregated_by(
            ["level", "height"],
            SUM,
            weights=self.weights_multi,
        )
        assert aggregateby_cube.units == "kelvin m2"

    def test_coord_aggregation_weighted_sum_multi(self):
        aggregateby_cube = self.cube_multi.aggregated_by(
            [self.coord_z1_multi, self.coord_z2_multi],
            SUM,
            weights=self.weights_multi,
        )
        assert aggregateby_cube.units == "kelvin m2"

    def test_coord_aggregation_rev_order_weighted_sum_multi(self):
        aggregateby_cube = self.cube_multi.aggregated_by(
            [self.coord_z2_multi, self.coord_z1_multi],
            SUM,
            weights=self.weights_multi,
        )
        assert aggregateby_cube.units == "kelvin m2"
