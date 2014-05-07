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
"""
Test function
:func:`iris.experimental.regrid.regrid_area_weighted_rectilinear_src_and_grid`.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy as np
import numpy.ma as ma

from iris.coords import DimCoord
from iris.coord_systems import GeogCS
from iris.cube import Cube
from iris.experimental.regrid \
    import regrid_area_weighted_rectilinear_src_and_grid as regrid
from iris.tests.experimental.regrid.\
    test_regrid_area_weighted_rectilinear_src_and_grid import \
    _resampled_grid
import iris.tests.stock


class TestMdtol(tests.IrisTest):
    def setUp(self):
        cube = Cube(np.arange(24, dtype=np.int32).reshape((3, 2, 4)))
        cs = GeogCS(6371229)
        coord = DimCoord(points=np.array([-1, 0, 1], dtype=np.int32),
                         standard_name='latitude',
                         units='degrees',
                         coord_system=cs)
        cube.add_dim_coord(coord, 0)
        coord = DimCoord(points=np.array([-1, 0, 1, 2], dtype=np.int32),
                         standard_name='longitude',
                         units='degrees',
                         coord_system=cs)
        cube.add_dim_coord(coord, 2)
        cube.coord('latitude').guess_bounds()
        cube.coord('longitude').guess_bounds()
        # A simple (3, 2, 4) cube.
        self.simple_cube = cube
        # A simple (3, 2, 4) cube with a masked element.
        self.simple_masked_cube = cube.copy(
            data=ma.masked_array(self.simple_cube.data))
        self.simple_masked_cube.data[1, 1, 2] = ma.masked

    def test_default(self):
        src = self.simple_masked_cube
        dest = _resampled_grid(self.simple_masked_cube, 2.3, 2.4)
        res = regrid(src, dest)
        expected_mask = np.zeros((7, 2, 9), bool)
        expected_mask[slice(2, 5), 1, slice(4, 7)] = True
        self.assertArrayEqual(res.data.mask, expected_mask)

    def test_zero(self):
        src = self.simple_masked_cube
        dest = _resampled_grid(self.simple_masked_cube, 2.3, 2.4)
        res = regrid(src, dest, mdtol=0)
        expected_mask = np.zeros((7, 2, 9), bool)
        expected_mask[slice(2, 5), 1, slice(4, 7)] = True
        self.assertArrayEqual(res.data.mask, expected_mask)

    def test_one(self):
        src = self.simple_masked_cube
        dest = _resampled_grid(self.simple_masked_cube, 2.3, 2.4)
        res = regrid(src, dest, mdtol=1)
        expected_mask = np.zeros((7, 2, 9), bool)
        # Only a single cell has all contributing cells masked.
        expected_mask[3, 1, 5] = True
        self.assertArrayEqual(res.data.mask, expected_mask)

    def test_fraction(self):
        src = self.simple_masked_cube
        dest = _resampled_grid(self.simple_masked_cube, 2.3, 2.4)
        res = regrid(src, dest, mdtol=0.4)
        expected_mask = np.zeros((7, 2, 9), bool)
        # Corners of overlapping cells have one of four
        # masked cells so should not be masked. The rest have
        # half or all that are masked.
        expected_mask[3, 1, slice(4, 7)] = True
        expected_mask[slice(2, 5), 1, 5] = True
        self.assertArrayEqual(res.data.mask, expected_mask)

    def test_equal_missing(self):
        # Test the behaviour when mdtol equals the fraction
        # of missing data.
        src = self.simple_masked_cube
        dest = _resampled_grid(self.simple_masked_cube, 2.3, 2.4)
        res = regrid(src, dest, mdtol=0.25)
        expected_mask = np.zeros((7, 2, 9), bool)
        # Corners of overlapping cells have one of four
        # masked cells so should not be masked. The rest have
        # half or all that are masked.
        expected_mask[3, 1, slice(4, 7)] = True
        expected_mask[slice(2, 5), 1, 5] = True
        self.assertArrayEqual(res.data.mask, expected_mask)

    def test_equal_missing_and_non_missing(self):
        # Test the behaviour when mdtol equals the fraction
        # of missing data and non missing data.
        src = self.simple_masked_cube
        dest = _resampled_grid(self.simple_masked_cube, 2.3, 2.4)
        res = regrid(src, dest, mdtol=0.5)
        expected_mask = np.zeros((7, 2, 9), bool)
        # Corners of overlapping cells have one of four
        # masked cells so should not be masked. The edge cells
        # have half that are masked so should also not be masked. The
        # centre cell is entirely masked.
        expected_mask[3, 1, 5] = True
        self.assertArrayEqual(res.data.mask, expected_mask)


if __name__ == '__main__':
    tests.main()
