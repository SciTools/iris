# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

"""Unit tests for :func:`iris.analysis.cartography._xy_range`"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.

import iris.tests as tests  # isort:skip
import numpy as np

from iris.analysis.cartography import _xy_range
import iris.tests.stock as stock


class Test(tests.IrisTest):
    def test_bounds_mismatch(self):
        cube = stock.realistic_3d()
        cube.coord("grid_longitude").guess_bounds()

        with self.assertRaisesRegex(ValueError, "bounds"):
            _ = _xy_range(cube)

    def test_non_circular(self):
        cube = stock.realistic_3d()
        assert not cube.coord("grid_longitude").circular

        result_non_circ = _xy_range(cube)
        self.assertEqual(result_non_circ, ((-5.0, 5.0), (-4.0, 4.0)))

    @tests.skip_data
    def test_geog_cs_circular(self):
        cube = stock.global_pp()
        assert cube.coord("longitude").circular

        result = _xy_range(cube)
        np.testing.assert_array_almost_equal(
            result, ((0, 360), (-90, 90)), decimal=0
        )

    @tests.skip_data
    def test_geog_cs_regional(self):
        cube = stock.global_pp()
        cube = cube[10:20, 20:30]
        assert not cube.coord("longitude").circular

        result = _xy_range(cube)
        np.testing.assert_array_almost_equal(
            result, ((75, 108.75), (42.5, 65)), decimal=0
        )


if __name__ == "__main__":
    tests.main()
