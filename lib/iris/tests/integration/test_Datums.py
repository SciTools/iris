# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Integration tests for :class:`iris.coord_systems` datum suppport."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import cartopy.crs as ccrs
import numpy as np

from iris.coord_systems import GeogCS, LambertConformal


class TestDatumTransformation(tests.IrisTest):
    def setUp(self):
        self.x_points = np.array([-1.5])
        self.y_points = np.array([50.5])

        self.start_crs = ccrs.OSGB(False)

    def test_transform_points_datum(self):
        # Iris version
        wgs84 = GeogCS.from_datum("WGS84")
        iris_cs = LambertConformal(
            central_lat=54,
            central_lon=-4,
            secant_latitudes=[52, 56],
            ellipsoid=wgs84,
        )
        iris_cs_as_cartopy = iris_cs.as_cartopy_crs()

        # Cartopy equivalent
        cartopy_cs = ccrs.LambertConformal(
            central_latitude=54,
            central_longitude=-4,
            standard_parallels=[52, 56],
            globe=ccrs.Globe("WGS84"),
        )

        expected = cartopy_cs.transform_points(
            self.start_crs, self.x_points, self.y_points
        )

        actual = iris_cs_as_cartopy.transform_points(
            self.start_crs, self.x_points, self.y_points
        )

        self.assertArrayEqual(expected, actual)
