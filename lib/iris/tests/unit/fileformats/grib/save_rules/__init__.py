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

import mock
import numpy as np


class GdtTestMixin(object):
    """Some handy common test capabilities for grib grid-definition tests."""
    def setUp(self, target_module):
        # Create mock x and y coords.
        x_coord = mock.MagicMock()
        y_coord = mock.MagicMock()
        self.mock_x_coord = x_coord
        self.mock_y_coord = y_coord

        # Create a mock cube object that returns the x and y coords.
        def mock_coord_call(dimensions=None):
            # Accept a single dim 0/1 and return the mock X or Y coord.
            return [y_coord, x_coord][dimensions[0]]

        self.mock_cube = mock.Mock(coord=mock_coord_call)

        # Patch the gribapi of the tested module.
        self.mock_gribapi = self.patch(target_module + '.gribapi')

        # Fix the mock gribapi to record key assignments.
        def grib_set_trap(grib, name, value):
            # Record a key setting on the mock passed as the 'grib message id'.
            grib.keys[name] = value

        self.mock_gribapi.grib_set_long = grib_set_trap
        self.mock_gribapi.grib_set_float = grib_set_trap
        self.mock_gribapi.grib_set_double = grib_set_trap

        # Create a mock 'grib message id', with a 'keys' dict for settings.
        self.mock_grib = mock.Mock(keys={})

        # Initialise the cube coords to something barely usable.
        self._set_coords()

    def _default_coord_system(self):
        return mock.Mock()

    def _default_x_points(self):
        # Define simple, regular coordinate points.
        return [1.0, 2.0, 3.0]

    def _default_y_points(self):
        return [7.0, 8.0]  # N.B. is_regular will *fail* on length-1 coords.

    def _set_coords(self, cs=None, x_points=None, y_points=None):
        # Set mock x+y coords with given properties, or minimal defaults.
        if cs is None:
            cs = self._default_coord_system()
        if x_points is None:
            x_points = self._default_x_points()
        if y_points is None:
            y_points = self._default_y_points()

        for coord, points in zip([self.mock_x_coord, self.mock_y_coord],
                                 [x_points, y_points]):
            # Fake the coordinate coord-system and points.
            coord.coord_system = cs
            points = np.array(points)
            coord.points = points
            # Fake the coordinate array-like properties
            coord.shape = points.shape
            coord.ndim = points.ndim
            # Avoid 'ignoring bounds' warnings.
            coord.has_bounds = lambda: False

    def _check_key(self, name, value):
        # Test that a specific grib key assignment occurred.
        msg_fmt = 'Expected grib setting "{}" = {}, got {}'
        found = self.mock_grib.keys.get(name)
        if found is None:
            self.assertEqual(0, 1, msg_fmt.format(name, value, '((UNSET))'))
        else:
            self.assertArrayEqual(found, value,
                                  msg_fmt.format(name, value, found))
