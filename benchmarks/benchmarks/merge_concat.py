# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Benchmarks relating to :meth:`iris.cube.CubeList.merge` and ``concatenate``."""

import numpy as np

from iris.cube import CubeList

from . import TrackAddedMemoryAllocation
from .generate_data.stock import realistic_4d_w_everything


class Merge:
    # TODO: Improve coverage.

    cube_list: CubeList

    def setup(self):
        source_cube = realistic_4d_w_everything()

        # Merge does not yet fully support cell measures and ancillary variables.
        for cm in source_cube.cell_measures():
            source_cube.remove_cell_measure(cm)
        for av in source_cube.ancillary_variables():
            source_cube.remove_ancillary_variable(av)

        second_cube = source_cube.copy()
        scalar_coord = second_cube.coords(dimensions=[])[0]
        scalar_coord.points = scalar_coord.points + 1
        self.cube_list = CubeList([source_cube, second_cube])

    def time_merge(self):
        _ = self.cube_list.merge_cube()

    @TrackAddedMemoryAllocation.decorator_repeating()
    def track_mem_merge(self):
        _ = self.cube_list.merge_cube()


class Concatenate:
    # TODO: Improve coverage.

    cube_list: CubeList

    def setup(self):
        source_cube = realistic_4d_w_everything()
        second_cube = source_cube.copy()
        first_dim_coord = second_cube.coord(dimensions=0, dim_coords=True)
        first_dim_coord.points = (
            first_dim_coord.points + np.ptp(first_dim_coord.points) + 1
        )
        self.cube_list = CubeList([source_cube, second_cube])

    def time_concatenate(self):
        _ = self.cube_list.concatenate_cube()

    @TrackAddedMemoryAllocation.decorator_repeating()
    def track_mem_merge(self):
        _ = self.cube_list.concatenate_cube()
