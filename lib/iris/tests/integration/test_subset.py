# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Integration tests for subset."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np

from iris.coords import DimCoord
from iris.cube import Cube


def _make_test_cube():
    data = np.zeros((4, 4, 1))
    lats, longs = [0, 10, 20, 30], [5, 15, 25, 35]
    lat_coord = DimCoord(lats, standard_name="latitude", units="degrees")
    lon_coord = DimCoord(longs, standard_name="longitude", units="degrees")
    vrt_coord = DimCoord([850], long_name="pressure", units="hPa")
    return Cube(
        data,
        long_name="test_cube",
        units="1",
        attributes=None,
        dim_coords_and_dims=[(lat_coord, 0), (lon_coord, 1)],
        aux_coords_and_dims=[(vrt_coord, None)],
    )


class TestSubset(tests.IrisTest):
    def setUp(self):
        self.cube = _make_test_cube()

    def test_coordinate_subset(self):
        coord = self.cube.coord("pressure")
        subsetted = self.cube.subset(coord)
        self.assertEqual(self.cube, subsetted)


if __name__ == "__main__":
    tests.main()
