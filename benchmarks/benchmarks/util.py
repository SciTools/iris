# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Benchmarks for :mod:`iris.util`.

"""

import numpy as np

from iris.util import lift_empty_masks

from . import on_demand_benchmark


@on_demand_benchmark
class LiftEmptyMasks:
    """
    Demonstrates the pros and cons of using @lift_empty_masks in different scenarios.
    """

    params = [[False, True], [False, True]]
    param_names = ["lifting enabled", "scalar false mask"]

    @staticmethod
    def toy_mean_func(array: np.ndarray) -> np.ndarray:
        return array.mean(axis=0)

    @staticmethod
    def toy_access_func(array: np.ndarray) -> None:
        for i in range(array.shape[0]):
            _ = array[i]

    def setup(self, lifting_enabled, scalar_false_mask) -> None:
        if lifting_enabled:
            self.toy_mean_func = lift_empty_masks(self.toy_mean_func)
            self.toy_access_func = lift_empty_masks(self.toy_access_func)
        input_data = np.random.rand(1000, 1000)
        if scalar_false_mask:
            input_mask = np.ma.nomask
        else:
            input_mask = np.full(input_data.shape, False)
        self.input_array = np.ma.masked_array(data=input_data, mask=input_mask)

    def time_mean(self, _, __) -> None:
        _ = self.toy_mean_func(self.input_array)

    def time_many_access(self, _, __) -> None:
        self.toy_access_func(self.input_array)
