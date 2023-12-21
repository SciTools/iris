# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.util.guess_coord_axis`."""

import pytest

from iris.util import guess_coord_axis


class TestGuessCoord:
    @pytest.mark.parametrize(
        "coordinate, axis",
        [
            ("longitude", "X"),
            ("grid_longitude", "X"),
            ("projection_x_coordinate", "X"),
            ("latitude", "Y"),
            ("grid_latitude", "Y"),
            ("projection_y_coordinate", "Y"),
        ],
    )
    def test_coord(self, coordinate, axis, sample_coord):
        sample_coord.standard_name = coordinate
        assert guess_coord_axis(sample_coord) == axis

    @pytest.mark.parametrize(
        "units, axis",
        [
            ("hPa", "Z"),
            ("days since 1970-01-01 00:00:00", "T"),
        ],
    )
    def test_units(self, units, axis, sample_coord):
        sample_coord.units = units
        assert guess_coord_axis(sample_coord) == axis

    @pytest.mark.parametrize(
        "ignore_axis, result",
        [
            (True, None),
            (False, "X"),
        ],
    )
    def test_ignore_axis(self, ignore_axis, result, sample_coord):
        sample_coord.standard_name = "longitude"
        sample_coord.ignore_axis = ignore_axis

        assert guess_coord_axis(sample_coord) == result
