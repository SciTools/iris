# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test function :func:`iris.fileformats._nc_load_rules.helpers.\
build_vertical_perspective_coordinate_system`.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

from unittest import mock

import iris
from iris.coord_systems import VerticalPerspective
from iris.fileformats._nc_load_rules.helpers import (
    build_vertical_perspective_coordinate_system,
)


class TestBuildVerticalPerspectiveCoordinateSystem(tests.IrisTest):
    def _test(self, inverse_flattening=False, no_offsets=False):
        """
        Generic test that can check vertical perspective validity with or
        without inverse flattening, and false_east/northing-s.
        """
        test_easting = 100.0
        test_northing = 200.0
        cf_grid_var_kwargs = {
            "spec": [],
            "latitude_of_projection_origin": 1.0,
            "longitude_of_projection_origin": 2.0,
            "perspective_point_height": 2000000.0,
            "false_easting": test_easting,
            "false_northing": test_northing,
            "semi_major_axis": 6377563.396,
        }

        ellipsoid_kwargs = {"semi_major_axis": 6377563.396}
        if inverse_flattening:
            ellipsoid_kwargs["inverse_flattening"] = 299.3249646
        else:
            ellipsoid_kwargs["semi_minor_axis"] = 6356256.909
        cf_grid_var_kwargs.update(ellipsoid_kwargs)

        if no_offsets:
            del cf_grid_var_kwargs["false_easting"]
            del cf_grid_var_kwargs["false_northing"]
            test_easting = 0
            test_northing = 0

        cf_grid_var = mock.Mock(**cf_grid_var_kwargs)
        ellipsoid = iris.coord_systems.GeogCS(**ellipsoid_kwargs)

        cs = build_vertical_perspective_coordinate_system(None, cf_grid_var)
        expected = VerticalPerspective(
            latitude_of_projection_origin=cf_grid_var.latitude_of_projection_origin,
            longitude_of_projection_origin=cf_grid_var.longitude_of_projection_origin,
            perspective_point_height=cf_grid_var.perspective_point_height,
            false_easting=test_easting,
            false_northing=test_northing,
            ellipsoid=ellipsoid,
        )

        self.assertEqual(cs, expected)

    def test_valid(self):
        self._test(inverse_flattening=False)

    def test_inverse_flattening(self):
        # Check when inverse_flattening is provided instead of semi_minor_axis.
        self._test(inverse_flattening=True)

    def test_no_offsets(self):
        # Check when false_easting/northing attributes are absent.
        self._test(no_offsets=True)


if __name__ == "__main__":
    tests.main()
