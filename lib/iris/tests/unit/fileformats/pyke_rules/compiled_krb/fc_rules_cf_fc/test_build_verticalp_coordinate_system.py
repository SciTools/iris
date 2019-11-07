# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test function :func:`iris.fileformats._pyke_rules.compiled_krb.\
fc_rules_cf_fc.build_vertical_perspective_coordinate_system`.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

from unittest import mock

import iris
from iris.coord_systems import VerticalPerspective
from iris.fileformats._pyke_rules.compiled_krb.fc_rules_cf_fc import \
    build_vertical_perspective_coordinate_system


class TestBuildVerticalPerspectiveCoordinateSystem(tests.IrisTest):
    def _test(self, inverse_flattening=False):
        """
        Generic test that can check vertical perspective validity with or
        without inverse flattening.
        """
        cf_grid_var_kwargs = {
            'spec': [],
            'latitude_of_projection_origin': 1.0,
            'longitude_of_projection_origin': 2.0,
            'perspective_point_height': 2000000.0,
            'false_easting': 100.0,
            'false_northing': 200.0,
            'semi_major_axis': 6377563.396}

        ellipsoid_kwargs = {'semi_major_axis': 6377563.396}
        if inverse_flattening:
            ellipsoid_kwargs['inverse_flattening'] = 299.3249646
        else:
            ellipsoid_kwargs['semi_minor_axis'] = 6356256.909
        cf_grid_var_kwargs.update(ellipsoid_kwargs)

        cf_grid_var = mock.Mock(**cf_grid_var_kwargs)
        ellipsoid = iris.coord_systems.GeogCS(**ellipsoid_kwargs)

        cs = build_vertical_perspective_coordinate_system(None, cf_grid_var)
        expected = VerticalPerspective(
            latitude_of_projection_origin=cf_grid_var.
            latitude_of_projection_origin,
            longitude_of_projection_origin=cf_grid_var.
            longitude_of_projection_origin,
            perspective_point_height=cf_grid_var.perspective_point_height,
            false_easting=cf_grid_var.false_easting,
            false_northing=cf_grid_var.false_northing,
            ellipsoid=ellipsoid)

        self.assertEqual(cs, expected)

    def test_valid(self):
        self._test(inverse_flattening=False)

    def test_inverse_flattening(self):
        self._test(inverse_flattening=True)
