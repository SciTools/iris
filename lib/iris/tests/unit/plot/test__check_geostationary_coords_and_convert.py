# (C) British Crown Copyright 2019, Met Office
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
"""Unit tests for the `iris.plot._check_geostationary_coords_and_convert
function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np
from mock import Mock

from cartopy.crs import Geostationary, NearsidePerspective
from iris.plot import _check_geostationary_coords_and_convert


class Test__check_geostationary_coords_and_convert(tests.IrisTest):
    def setUp(self):
        geostationary_altitude = 35785831.0
        # proj4_params is the one attribute of the Geostationary class that
        # is needed for the function.
        self.proj4_params = {'h': geostationary_altitude}

        # Simulate the maximum-dimension array that could be processed.
        a = np.linspace(0, 2, 6)
        b = np.linspace(2, 3, 5)
        self.x_original, self.y_original = np.meshgrid(a, b)

        # Expected arrays if conversion takes place.
        self.x_converted, self.y_converted = (
            i * geostationary_altitude for i in (self.x_original,
                                                 self.y_original))

    def _test(self, geostationary=True):
        # Re-usable test for when Geostationary is present OR absent.
        if geostationary:
            # A Geostationary projection WILL be processed.
            projection_spec = Geostationary
            target_tuple = (self.x_converted, self.y_converted)
        else:
            # A non-Geostationary projection WILL NOT be processed.
            projection_spec = NearsidePerspective
            target_tuple = (self.x_original, self.y_original)

        projection = Mock(spec=projection_spec)
        projection.proj4_params = self.proj4_params
        # Projection is looked for within a dictionary called kwargs.
        kwargs = {'transform': projection}

        x, y = _check_geostationary_coords_and_convert(self.x_original,
                                                       self.y_original,
                                                       kwargs)
        self.assertArrayEqual((x, y), target_tuple)

    def test_geostationary_present(self):
        self._test(geostationary=True)

    def test_geostationary_absent(self):
        self._test(geostationary=False)
