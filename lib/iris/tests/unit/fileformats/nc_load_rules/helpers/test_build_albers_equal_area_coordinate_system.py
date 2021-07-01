# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test function :func:`iris.fileformats._nc_load_rules.helpers.\
build_albers_equal_area_coordinate_system`.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

from unittest import mock

import iris
from iris.coord_systems import AlbersEqualArea
from iris.fileformats._nc_load_rules.helpers import (
    build_albers_equal_area_coordinate_system,
)


class TestBuildAlbersEqualAreaCoordinateSystem(tests.IrisTest):
    def _test(self, inverse_flattening=False, no_optionals=False):
        if no_optionals:
            # Most properties are optional for this system.
            gridvar_props = {}
            # Setup all the expected default values
            test_lat = 0
            test_lon = 0
            test_easting = 0
            test_northing = 0
            test_parallels = (20, 50)
        else:
            # Choose test values and setup corresponding named properties.
            test_lat = -35
            test_lon = 175
            test_easting = -100
            test_northing = 200
            test_parallels = (-27, 3)
            gridvar_props = dict(
                latitude_of_projection_origin=test_lat,
                longitude_of_central_meridian=test_lon,
                false_easting=test_easting,
                false_northing=test_northing,
                standard_parallel=test_parallels,
            )

        # Add ellipsoid args.
        gridvar_props["semi_major_axis"] = 6377563.396
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

        cf_grid_var = mock.Mock(spec=[], **gridvar_props)

        cs = build_albers_equal_area_coordinate_system(None, cf_grid_var)

        expected = AlbersEqualArea(
            latitude_of_projection_origin=test_lat,
            longitude_of_central_meridian=test_lon,
            false_easting=test_easting,
            false_northing=test_northing,
            standard_parallels=test_parallels,
            ellipsoid=expected_ellipsoid,
        )

        self.assertEqual(cs, expected)

    def test_basic(self):
        self._test()

    def test_inverse_flattening(self):
        # Check when inverse_flattening is provided instead of semi_minor_axis.
        self._test(inverse_flattening=True)

    def test_no_optionals(self):
        # Check defaults, when all optional attributes are absent.
        self._test(no_optionals=True)


if __name__ == "__main__":
    tests.main()
