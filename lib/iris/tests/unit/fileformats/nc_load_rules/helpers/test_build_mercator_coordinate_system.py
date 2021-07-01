# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test function :func:`iris.fileformats._nc_load_rules.helpers.\
build_mercator_coordinate_system`.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

from unittest import mock

import iris
from iris.coord_systems import Mercator
from iris.fileformats._nc_load_rules.helpers import (
    build_mercator_coordinate_system,
)


class TestBuildMercatorCoordinateSystem(tests.IrisTest):
    def test_valid(self):
        cf_grid_var = mock.Mock(
            spec=[],
            longitude_of_projection_origin=-90,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )

        cs = build_mercator_coordinate_system(None, cf_grid_var)

        expected = Mercator(
            longitude_of_projection_origin=(
                cf_grid_var.longitude_of_projection_origin
            ),
            ellipsoid=iris.coord_systems.GeogCS(
                cf_grid_var.semi_major_axis, cf_grid_var.semi_minor_axis
            ),
        )
        self.assertEqual(cs, expected)

    def test_inverse_flattening(self):
        cf_grid_var = mock.Mock(
            spec=[],
            longitude_of_projection_origin=-90,
            semi_major_axis=6377563.396,
            inverse_flattening=299.3249646,
        )

        cs = build_mercator_coordinate_system(None, cf_grid_var)

        expected = Mercator(
            longitude_of_projection_origin=(
                cf_grid_var.longitude_of_projection_origin
            ),
            ellipsoid=iris.coord_systems.GeogCS(
                cf_grid_var.semi_major_axis,
                inverse_flattening=cf_grid_var.inverse_flattening,
            ),
        )
        self.assertEqual(cs, expected)

    def test_longitude_missing(self):
        cf_grid_var = mock.Mock(
            spec=[],
            semi_major_axis=6377563.396,
            inverse_flattening=299.3249646,
        )

        cs = build_mercator_coordinate_system(None, cf_grid_var)

        expected = Mercator(
            ellipsoid=iris.coord_systems.GeogCS(
                cf_grid_var.semi_major_axis,
                inverse_flattening=cf_grid_var.inverse_flattening,
            )
        )
        self.assertEqual(cs, expected)


if __name__ == "__main__":
    tests.main()
