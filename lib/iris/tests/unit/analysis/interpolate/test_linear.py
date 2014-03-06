# (C) British Crown Copyright 2014, Met Office
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
"""Unit tests for the :func:`iris.analysis.interpolate.linear` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np
import numpy.ma as ma

from iris.analysis.interpolate import linear
import iris.coords
import iris.tests.stock as stock


class Test_masks(tests.IrisTest):
    def test_mask_retention(self):
        cube = stock.realistic_4d_w_missing_data()
        interp_cube = linear(cube, [('pressure', [850, 950])])
        self.assertIsInstance(interp_cube.data, ma.MaskedArray)
        # this value is masked in the input
        self.assertTrue(cube.data.mask[0, 2, 2, 0])
        # and is still masked in the output
        self.assertTrue(interp_cube.data.mask[0, 1, 2, 0])


class TestNDCoords(tests.IrisTest):
    def setUp(self):
        cube = stock.simple_3d_w_multidim_coords()
        cube.add_dim_coord(iris.coords.DimCoord(range(3), 'longitude'), 1)
        cube.add_dim_coord(iris.coords.DimCoord(range(4), 'latitude'), 2)
        cube.data = cube.data.astype(np.float32)
        self.cube = cube

    def test_first(self):
        cube = self.cube
        
        with self.assertRaisesRegexp(ValueError, "Interpolation coords must be 1-d for rectilinear interpolation."):
            interp_cube = linear(cube, {'foo': 15, 'bar': 10})
        
        interp_cube = linear(cube, {'latitude': 1.5, 'longitude': 1.5})
        self.assertCMLApproxData(interp_cube, ('experimental', 'analysis', 'interpolate', 'linear_nd_2_coords.cml'))
        
        interp_cube = linear(cube, {'wibble': np.float32(1.5)})
        self.assertCMLApproxData(interp_cube, ('experimental', 'analysis', 'interpolate', 'linear_nd_with_extrapolation.cml'))
        
        interp_cube = linear(cube, {'wibble': 20})
        self.assertArrayEqual(np.mean(cube.data, axis=0), interp_cube.data)
        self.assertCMLApproxData(interp_cube, ('experimental', 'analysis', 'interpolate', 'linear_nd.cml'))


if __name__ == "__main__":
    tests.main()
