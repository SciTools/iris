# (C) British Crown Copyright 2013 - 2014, Met Office
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
"""Unit tests for the :mod:`iris.fileformats.grib.grib_save_rules` module."""

from __future__ import (absolute_import, division, print_function)

import iris.tests as tests

import iris
from iris.fileformats.pp import EARTH_RADIUS as PP_DEFAULT_EARTH_RADIUS
import mock
import numpy as np


class GdtTestMixin(object):
    """Some handy common test capabilities for grib grid-definition tests."""
    TARGET_MODULE = 'iris.fileformats.grib._save_rules'

    def setUp(self):
        # Patch the gribapi of the tested module.
        self.mock_gribapi = self.patch(self.TARGET_MODULE + '.gribapi')

        # Fix the mock gribapi to record key assignments.
        def grib_set_trap(grib, name, value):
            # Record a key setting on the mock passed as the 'grib message id'.
            grib.keys[name] = value

        self.mock_gribapi.grib_set_long = grib_set_trap
        self.mock_gribapi.grib_set_float = grib_set_trap
        self.mock_gribapi.grib_set_double = grib_set_trap
        self.mock_gribapi.grib_set_long_array = grib_set_trap

        # Create a mock 'grib message id', with a 'keys' dict for settings.
        self.mock_grib = mock.Mock(keys={})

        # Initialise the test cube and its coords to something barely usable.
        self.test_cube = self._make_test_cube()

    def _default_coord_system(self):
        return iris.coord_systems.GeogCS(PP_DEFAULT_EARTH_RADIUS)

    def _default_x_points(self):
        # Define simple, regular coordinate points.
        return [1.0, 2.0, 3.0]

    def _default_y_points(self):
        return [7.0, 8.0]  # N.B. is_regular will *fail* on length-1 coords.

    def _make_test_cube(self, cs=None, x_points=None, y_points=None):
        # Create a cube with given properties, or minimal defaults.
        if cs is None:
            cs = self._default_coord_system()
        if x_points is None:
            x_points = self._default_x_points()
        if y_points is None:
            y_points = self._default_y_points()

        x_coord = iris.coords.DimCoord(x_points, standard_name='longitude',
                                       units='degrees',
                                       coord_system=cs)
        y_coord = iris.coords.DimCoord(y_points, standard_name='latitude',
                                       units='degrees',
                                       coord_system=cs)
        test_cube = iris.cube.Cube(np.zeros((len(y_points), len(x_points))))
        test_cube.add_dim_coord(y_coord, 0)
        test_cube.add_dim_coord(x_coord, 1)
        return test_cube

    def _check_key(self, name, value):
        # Test that a specific grib key assignment occurred.
        msg_fmt = 'Expected grib setting "{}" = {}, got {}'
        found = self.mock_grib.keys.get(name)
        if found is None:
            self.assertEqual(0, 1, msg_fmt.format(name, value, '((UNSET))'))
        else:
            self.assertArrayEqual(found, value,
                                  msg_fmt.format(name, value, found))
