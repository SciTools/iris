# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :mod:`iris._representation` module."""

import numpy as np
import iris.tests as tests
import iris._representation
from iris.cube import Cube
from iris.coords import (
    DimCoord,
    AuxCoord,
    CellMeasure,
    AncillaryVariable,
    CellMethod,
)


def example_cube():
    cube = Cube(
        np.arange(6).reshape([3, 2]),
        standard_name="air_temperature",
        long_name="screen_air_temp",
        var_name="airtemp",
        units="K",
    )
    lat = DimCoord([0, 1, 2], standard_name="latitude", units="degrees")
    cube.add_dim_coord(lat, 0)
    return cube


class Test_CubeSummary(tests.IrisTest):
    def setUp(self):
        self.cube = example_cube()

    def test_header(self):
        rep = iris._representation.CubeSummary(self.cube)
        header_left = rep.header.nameunit
        header_right = rep.header.dimension_header.contents

        self.assertEqual(header_left, "air_temperature / (K)")
        self.assertEqual(header_right, ["latitude: 3", "-- : 2"])

    def test_coord(self):
        rep = iris._representation.CubeSummary(self.cube)
        dim_section = rep.dim_coord_section

        self.assertEqual(len(dim_section.contents), 1)

        dim_summary = dim_section.contents[0]

        name = dim_summary.name
        dim_chars = dim_summary.dim_chars
        extra = dim_summary.extra

        self.assertEqual(name, "latitude")
        self.assertEqual(dim_chars, ["x", "-"])
        self.assertEqual(extra, "")


if __name__ == "__main__":
    tests.main()
