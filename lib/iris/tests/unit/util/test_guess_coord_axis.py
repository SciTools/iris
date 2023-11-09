# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Test function :func:`iris.util.guess_coord_axis"""

import pytest

import iris.coords
from iris.util import guess_coord_axis


@pytest.fixture
def coord():
    coord = iris.coords.DimCoord(points=(1, 2, 3, 4, 5))
    return coord


@pytest.mark.parametrize("coordinate, axis",
                          [("longitude", "X"),
                           ("grid_longitude", "X"),
                           ("projection_x_coordinate", "X"),
                           ("latitude", "Y"),
                           ("grid_latitude", "Y"),
                           ("projection_y_coordinate", "Y")]
                          )
def testcoord(coordinate, axis, coord):
    coord.standard_name = coordinate
    assert guess_coord_axis(coord) == axis


class TestCoords:
    def test_pressure_units(self, coord):
        coord.units = "hPa"

        assert guess_coord_axis(coord) == "Z"

    def test_time_units(self, coord):
        coord.units = "days since 1970-01-01 00:00:00"

        assert guess_coord_axis(coord) == "T"

    def test_ignore_axis(self, coord):
        coord.standard_name = "longitude"
        coord.ignore_axis = True

        assert guess_coord_axis(coord) is None
