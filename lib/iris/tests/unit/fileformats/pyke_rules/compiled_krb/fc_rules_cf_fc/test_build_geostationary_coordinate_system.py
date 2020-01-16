# (C) British Crown Copyright 2020, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
Test function :func:`iris.fileformats._pyke_rules.compiled_krb.\
fc_rules_cf_fc.build_geostationary_coordinate_system`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import iris
from iris.coord_systems import Geostationary
from iris.fileformats._pyke_rules.compiled_krb.fc_rules_cf_fc import \
    build_geostationary_coordinate_system
from iris.tests import mock


class TestBuildGeostationaryCoordinateSystem(tests.IrisTest):
    def _test(self, inverse_flattening=False, replace_props=None,
              remove_props=None):
        """
        Generic test that can check vertical perspective validity with or
        without inverse flattening.
        """
        # Make a dictionary of the non-ellipsoid properties to be added to
        # both a test coord-system, and a test grid-mapping cf_var.
        non_ellipsoid_kwargs = {
            'latitude_of_projection_origin': 0.0,
            'longitude_of_projection_origin': 2.0,
            'perspective_point_height': 2000000.0,
            'sweep_angle_axis': 'x',
            'false_easting': 100.0,
            'false_northing': 200.0}

        # Make specified adjustments to the non-ellipsoid properties.
        if remove_props:
            for key in remove_props:
                non_ellipsoid_kwargs.pop(key, None)
        if replace_props:
            for key, value in replace_props.items():
                non_ellipsoid_kwargs[key] = value

        # Make a dictionary of ellipsoid properties, to be added to both a test
        # ellipsoid and the grid-mapping cf_var.
        ellipsoid_kwargs = {'semi_major_axis': 6377563.396}
        if inverse_flattening:
            ellipsoid_kwargs['inverse_flattening'] = 299.3249646
        else:
            ellipsoid_kwargs['semi_minor_axis'] = 6356256.909

        cf_grid_var_kwargs = non_ellipsoid_kwargs.copy()
        cf_grid_var_kwargs.update(ellipsoid_kwargs)
        cf_grid_var = mock.Mock(spec=[], **cf_grid_var_kwargs)
        cs = build_geostationary_coordinate_system(None, cf_grid_var)
        ellipsoid = iris.coord_systems.GeogCS(**ellipsoid_kwargs)
        expected = Geostationary(ellipsoid=ellipsoid, **non_ellipsoid_kwargs)
        self.assertEqual(cs, expected)

    def test_valid(self):
        self._test(inverse_flattening=False)

    def test_inverse_flattening(self):
        self._test(inverse_flattening=True)

    def test_false_offsets_missing(self):
        self._test(remove_props=['false_easting', 'false_northing'])

    def test_false_offsets_none(self):
        self._test(replace_props={'false_easting': None,
                                  'false_northing': None})


if __name__ == "__main__":
    tests.main()
