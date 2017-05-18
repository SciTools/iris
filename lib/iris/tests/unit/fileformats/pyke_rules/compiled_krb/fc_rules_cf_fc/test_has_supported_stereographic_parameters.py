# (C) British Crown Copyright 2016 - 2017, Met Office
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
fc_rules_cf_fc.has_supported_stereographic_parameters`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import warnings

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy as np
import six

from iris.coord_systems import Stereographic
from iris.fileformats._pyke_rules.compiled_krb.fc_rules_cf_fc import \
    has_supported_stereographic_parameters
from iris.tests import mock


def _engine(cf_grid_var, cf_name):
    cf_group = {cf_name: cf_grid_var}
    cf_var = mock.Mock(cf_group=cf_group)
    return mock.Mock(cf_var=cf_var)


class TestHasSupportedStereographicParameters(tests.IrisTest):
    def test_valid(self):
        cf_name = 'stereographic'
        cf_grid_var = mock.Mock(
            spec=[],
            latitude_of_projection_origin=0,
            longitude_of_projection_origin=0,
            false_easting=-100,
            false_northing=200,
            scale_factor_at_projection_origin=1,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909)
        engine = _engine(cf_grid_var, cf_name)

        is_valid = has_supported_stereographic_parameters(engine, cf_name)

        self.assertTrue(is_valid)

    def test_invalid_scale_factor(self):
        # Iris does not yet support scale factors other than one for
        # stereographic projections
        cf_name = 'stereographic'
        cf_grid_var = mock.Mock(
            spec=[],
            latitude_of_projection_origin=0,
            longitude_of_projection_origin=0,
            false_easting=-100,
            false_northing=200,
            scale_factor_at_projection_origin=0.9,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909)
        engine = _engine(cf_grid_var, cf_name)

        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            is_valid = has_supported_stereographic_parameters(engine, cf_name)

        self.assertFalse(is_valid)
        self.assertEqual(len(warns), 1)
        six.assertRegex(self, str(warns[0]), 'Scale factor')


if __name__ == "__main__":
    tests.main()
