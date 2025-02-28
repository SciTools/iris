# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.plot._check_geostationary_coords_and_convert
function.
"""

from cartopy.crs import Geostationary, NearsidePerspective
import numpy as np
import pytest

from iris.plot import _check_geostationary_coords_and_convert
from iris.tests import _shared_utils


class Test__check_geostationary_coords_and_convert:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.mocker = mocker

        geostationary_altitude = 35785831.0
        # proj4_params is the one attribute of the Geostationary class that
        # is needed for the function.
        self.proj4_params = {"h": geostationary_altitude}

        # Simulate the maximum-dimension array that could be processed.
        a = np.linspace(0, 2, 6)
        b = np.linspace(2, 3, 5)
        self.x_original, self.y_original = np.meshgrid(a, b)

        # Expected arrays if conversion takes place.
        self.x_converted, self.y_converted = (
            i * geostationary_altitude for i in (self.x_original, self.y_original)
        )

    def _test(self, geostationary=True):
        # Reusable test for when Geostationary is present OR absent.
        if geostationary:
            # A Geostationary projection WILL be processed.
            projection_spec = Geostationary
            target_tuple = (self.x_converted, self.y_converted)
        else:
            # A non-Geostationary projection WILL NOT be processed.
            projection_spec = NearsidePerspective
            target_tuple = (self.x_original, self.y_original)

        projection = self.mocker.Mock(spec=projection_spec)
        projection.proj4_params = self.proj4_params
        # Projection is looked for within a dictionary called kwargs.
        kwargs = {"transform": projection}

        x, y = _check_geostationary_coords_and_convert(
            self.x_original, self.y_original, kwargs
        )
        _shared_utils.assert_array_equal((x, y), target_tuple)

    def test_geostationary_present(self):
        self._test(geostationary=True)

    def test_geostationary_absent(self):
        self._test(geostationary=False)
