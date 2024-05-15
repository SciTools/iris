# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Benchmarks relating to :meth:`iris.cube.CubeList.merge` and ``concatenate``."""

import numpy as np

from iris import analysis, coords, cube

from .generate_data.stock import realistic_4d_w_everything


class AggregationMixin:
    params = [[False, True]]
    param_names = ["Lazy operations"]

    def setup(self, lazy_run: bool):
        # 4d cube instead of merge, or replicate if break
        cube = realistic_4d_w_everything(lazy=lazy_run)
        self.lazy_run = lazy_run

        agg_mln_data = np.arange(0, 80, 10)
        agg_mln_repeat = np.repeat(agg_mln_data, 10)

        agg_mln_coord = coords.AuxCoord(points=agg_mln_repeat, long_name="aggregatable")

        if lazy_run:
            agg_mln_coord.points = agg_mln_coord.lazy_points()
        cube.add(agg_mln_coord, 1)
        self.cube = cube


class Aggregation(AggregationMixin):
    def time_aggregated_by_MEAN(self):
        self.cube.aggregated_by("aggregatable", analysis.MEAN)

    def time_aggregated_by_COUNT(self):
        self.cube.aggregated_by(
            "aggregatable", analysis.COUNT, function=lambda values: 340 > values > 280
        )

    def time_aggregated_by_GMEAN(self):
        self.cube.aggregated_by("aggregatable", analysis.GMEAN)

    def time_aggregated_by_HMEAN(self):
        self.cube.aggregated_by("aggregatable", analysis.HMEAN)

    def time_aggregated_by_MAX_RUN(self):
        self.cube.aggregated_by(
            "aggregatable", analysis.MAX_RUN, function=lambda values: 340 > values > 280
        )

    def time_aggregated_by_MAX(self):
        self.cube.aggregated_by("aggregatable", analysis.MAX)

    def time_aggregated_by_MEDIAN(self):
        self.cube.aggregated_by("aggregatable", analysis.MEDIAN)

    def time_aggregated_by_MIN(self):
        self.cube.aggregated_by("aggregatable", analysis.MIN)

    def time_aggregated_by_PEAK(self):
        self.cube.aggregated_by("aggregatable", analysis.PEAK)

    def time_aggregated_by_PERCENTILE(self):
        self.cube.aggregated_by(
            "aggregatable", analysis.PERCENTILE, percent=[10, 50, 90]
        )

    def time_aggregated_by_FAST_PERCENTILE(self):
        self.cube.aggregated_by(
            "aggregatable",
            analysis.PERCENTILE,
            percent=[10, 50, 90],
            fast_percentile_method=True,
        )

    def time_aggregated_by_PROPORTION(self):
        self.cube.aggregated_by(
            "aggregatable",
            analysis.PROPORTION,
            function=lambda values: 340 > values > 280,
        )

    def time_aggregated_by_STD_DEV(self):
        self.cube.aggregated_by("aggregatable", analysis.STD_DEV)

    def time_aggregated_by_VARIANCE(self):
        self.cube.aggregated_by("aggregatable", analysis.VARIANCE)

    def time_aggregated_by_RMS(self):
        self.cube.aggregated_by("aggregatable", analysis.RMS)

    def time_collapsed_by_MEAN(self):
        self.cube.collapsed("latitude", analysis.MEAN)

    def time_collapsed_by_COUNT(self):
        self.cube.collapsed(
            "latitude", analysis.COUNT, function=lambda values: 340 > values > 280
        )

    def time_collapsed_by_GMEAN(self):
        self.cube.collapsed("latitude", analysis.GMEAN)

    def time_collapsed_by_HMEAN(self):
        self.cube.collapsed("latitude", analysis.HMEAN)

    def time_collapsed_by_MAX_RUN(self):
        self.cube.collapsed(
            "latitude", analysis.MAX_RUN, function=lambda values: 340 > values > 280
        )

    def time_collapsed_by_MAX(self):
        self.cube.collapsed("latitude", analysis.MAX)

    def time_collapsed_by_MEDIAN(self):
        self.cube.collapsed("latitude", analysis.MEDIAN)

    def time_collapsed_by_MIN(self):
        self.cube.collapsed("latitude", analysis.MIN)

    def time_collapsed_by_PEAK(self):
        self.cube.collapsed("latitude", analysis.PEAK)

    def time_collapsed_by_PERCENTILE(self):
        self.cube.collapsed("latitude", analysis.PERCENTILE, percent=[10, 50, 90])

    def time_collapsed_by_FAST_PERCENTILE(self):
        self.cube.collapsed(
            "latitude",
            analysis.PERCENTILE,
            percent=[10, 50, 90],
            fast_percentile_method=True,
        )

    def time_collapsed_by_PROPORTION(self):
        self.cube.collapsed(
            "latitude", analysis.PROPORTION, function=lambda values: 340 > values > 280
        )

    def time_collapsed_by_STD_DEV(self):
        self.cube.collapsed("latitude", analysis.STD_DEV)

    def time_collapsed_by_VARIANCE(self):
        self.cube.collapsed("latitude", analysis.VARIANCE)

    def time_collapsed_by_RMS(self):
        self.cube.collapsed("latitude", analysis.RMS)


class WeightedAggregation(AggregationMixin):
    def setup(self, lazy_run):
        super().setup(lazy_run)
        self.weights = np.linspace(0, 1, 100)

    def time_w_aggregated_by_WPERCENTILE(self):
        self.cube.aggregated_by(
            "aggregatable", analysis.WPERCENTILE, self.weights, percent=[10, 50, 90]
        )

    def time_w_aggregated_by_SUM(self):
        self.cube.aggregated_by("aggregatable", analysis.SUM, self.weights)

    def time_w_aggregated_by_RMS(self):
        self.cube.aggregated_by("aggregatable", analysis.RMS, self.weights)

    def time_w_aggregated_by_MEAN(self):
        self.cube.aggregated_by("aggregatable", analysis.MEAN, self.weights)

    def time_w_collapsed_by_WPERCENTILE(self):
        self.cube.collapsed(
            "latitude", analysis.WPERCENTILE, self.weights, percent=[10, 50, 90]
        )

    def time_w_collapsed_by_SUM(self):
        self.cube.collapsed("latitude", analysis.SUM, self.weights)

    def time_w_collapsed_by_RMS(self):
        self.cube.collapsed("latitude", analysis.RMS, self.weights)

    def time_w_collapsed_by_MEAN(self):
        self.cube.collapsed("latitude", analysis.MEAN, self.weights)
