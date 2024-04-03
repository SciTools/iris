# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Iterate benchmark tests."""

import numpy as np

from iris import coords, cube, iterate

from . import ARTIFICIAL_DIM_SIZE


def setup():
    """General variables needed by multiple benchmark classes."""
    global data_1d
    global data_2d
    global general_cube

    data_2d = np.zeros((ARTIFICIAL_DIM_SIZE,) * 2)
    data_1d = data_2d[0]
    general_cube = cube.Cube(data_2d)


class IZip:
    def setup(self):
        local_cube = general_cube.copy()
        coord_a = coords.AuxCoord(points=data_1d, long_name="a")
        coord_b = coords.AuxCoord(points=data_1d, long_name="b")
        self.coord_names = (coord.long_name for coord in (coord_a, coord_b))

        local_cube.add_aux_coord(coord_a, 0)
        local_cube.add_aux_coord(coord_b, 1)
        self.cube = local_cube

    def time_izip(self):
        iterate.izip(self.cube, coords=self.coord_names)
