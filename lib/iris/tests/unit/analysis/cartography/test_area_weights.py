# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""Unit tests for the `iris.analysis.cartography.area_weights` function."""

import pytest

import iris.analysis.cartography
import iris.tests.stock as stock


class TestInvalidUnits:
    def test_latitude_no_units(self):
        cube = stock.lat_lon_cube()
        cube.coord("longitude").guess_bounds()
        cube.coord("latitude").guess_bounds()
        cube.coord("latitude").units = None
        with pytest.raises(ValueError, match="Units of degrees or radians required"):
            iris.analysis.cartography.area_weights(cube)

    def test_longitude_no_units(self):
        cube = stock.lat_lon_cube()
        cube.coord("latitude").guess_bounds()
        cube.coord("longitude").guess_bounds()
        cube.coord("longitude").units = None
        with pytest.raises(ValueError, match="Units of degrees or radians required"):
            iris.analysis.cartography.area_weights(cube)
