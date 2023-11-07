# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Test function :func:`iris.util.guess_coord_axis"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np
import iris.coords
import pytest

from iris.util import guess_coord_axis


@pytest.fixture
def coord():
    coord = iris.coords.DimCoord(
        points=(1, 2, 3, 4, 5)
    )
    return coord


class TestCoords:

    def test_longitude(self, coord):
        coord.standard_name = "longitude"

        assert guess_coord_axis(coord) == "X"

    def test_latitude(self, coord):
        coord.standard_name = "latitude"

        assert guess_coord_axis(coord) == "Y"

    def test_pressure_units(self, coord):
        coord.units = "hPa"

        assert guess_coord_axis(coord) == "Z"

    def test_time_units(self, coord):
        coord.units = "days since 1970-01-01 00:00:00"

        assert guess_coord_axis(coord) == "T"

    def test_guess_coord(self, coord):
        coord.standard_name = "longitude"
        coord.guess_coord = False

        assert guess_coord_axis(coord) is None
