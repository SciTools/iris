# (C) British Crown Copyright 2014 - 2015, Met Office
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
"""Unit tests for the :func:`iris.analysis.cartography.project` function."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

import iris
from iris.coords import DimCoord
from iris.analysis.cartography import project


class TestAll(tests.IrisTest):
    def setUp(self):
        cs = iris.coord_systems.GeogCS(654321)
        self.cube = iris.cube.Cube(np.zeros(25).reshape(5, 5))
        self.cube.add_dim_coord(
            DimCoord(np.arange(5), standard_name="latitude", units='degrees',
                     coord_system=cs), 0)
        self.cube.add_dim_coord(
            DimCoord(np.arange(5), standard_name="longitude", units='degrees',
                     coord_system=cs), 1)

        self.tcs = iris.coord_systems.GeogCS(600000)

    def test_is_iris_coord_system(self):
        res, _ = project(self.cube, self.tcs)
        self.assertEqual(res.coord('projection_y_coordinate').coord_system,
                         self.tcs)
        self.assertEqual(res.coord('projection_x_coordinate').coord_system,
                         self.tcs)

        self.assertIsNot(res.coord('projection_y_coordinate').coord_system,
                         self.tcs)
        self.assertIsNot(res.coord('projection_x_coordinate').coord_system,
                         self.tcs)


if __name__ == "__main__":
    tests.main()
