# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""Unit tests for :func:`iris.analysis.cartography._xy_range`."""

import pytest

from iris.analysis.cartography import _xy_range
from iris.tests import _shared_utils
import iris.tests.stock as stock


class Test:
    def test_bounds_mismatch(self):
        cube = stock.realistic_3d()
        cube.coord("grid_longitude").guess_bounds()

        with pytest.raises(ValueError, match="bounds"):
            _ = _xy_range(cube)

    def test_non_circular(self):
        cube = stock.realistic_3d()
        assert not cube.coord("grid_longitude").circular

        result_non_circ = _xy_range(cube)
        assert result_non_circ == ((-5.0, 5.0), (-4.0, 4.0))

    @_shared_utils.skip_data
    def test_geog_cs_circular(self):
        cube = stock.global_pp()
        assert cube.coord("longitude").circular

        result = _xy_range(cube)
        _shared_utils.assert_array_almost_equal(
            result, ((0, 360), (-90, 90)), decimal=0
        )

    @_shared_utils.skip_data
    def test_geog_cs_regional(self):
        cube = stock.global_pp()
        cube = cube[10:20, 20:30]
        assert not cube.coord("longitude").circular

        result = _xy_range(cube)
        _shared_utils.assert_array_almost_equal(
            result, ((75, 108.75), (42.5, 65)), decimal=0
        )
