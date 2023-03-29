# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test function :func:`iris.fileformats._nc_load_rules.helpers.\
build_transverse_mercator_coordinate_system`.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

from unittest import mock

import iris
from iris.coord_systems import TransverseMercator
from iris.fileformats._nc_load_rules.helpers import (
    build_transverse_mercator_coordinate_system,
)


class TestBuildTransverseMercatorCoordinateSystem(tests.IrisTest):
    def _test(self, inverse_flattening=False, no_options=False):
        test_easting = -100
        test_northing = 200
        test_scale_factor = 1.234
        gridvar_props = dict(
            latitude_of_projection_origin=35.3,
            longitude_of_central_meridian=-75,
            false_easting=test_easting,
            false_northing=test_northing,
            scale_factor_at_central_meridian=test_scale_factor,
            semi_major_axis=6377563.396,
        )

        if inverse_flattening:
            gridvar_props["inverse_flattening"] = 299.3249646
            expected_ellipsoid = iris.coord_systems.GeogCS(
                6377563.396, inverse_flattening=299.3249646
            )
        else:
            gridvar_props["semi_minor_axis"] = 6356256.909
            expected_ellipsoid = iris.coord_systems.GeogCS(
                6377563.396, 6356256.909
            )

        if no_options:
            del gridvar_props["false_easting"]
            del gridvar_props["false_northing"]
            del gridvar_props["scale_factor_at_central_meridian"]
            test_easting = 0
            test_northing = 0
            test_scale_factor = 1.0

        cf_grid_var = mock.Mock(spec=[], **gridvar_props)

        cs = build_transverse_mercator_coordinate_system(None, cf_grid_var)

        expected = TransverseMercator(
            latitude_of_projection_origin=(
                cf_grid_var.latitude_of_projection_origin
            ),
            longitude_of_central_meridian=(
                cf_grid_var.longitude_of_central_meridian
            ),
            false_easting=test_easting,
            false_northing=test_northing,
            scale_factor_at_central_meridian=test_scale_factor,
            ellipsoid=expected_ellipsoid,
        )

        self.assertEqual(cs, expected)

    def test_basic(self):
        self._test()

    def test_inverse_flattening(self):
        self._test(inverse_flattening=True)

    def test_missing_optionals(self):
        self._test(no_options=True)


if __name__ == "__main__":
    tests.main()
