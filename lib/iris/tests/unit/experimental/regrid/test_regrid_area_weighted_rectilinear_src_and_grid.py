# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test function
:func:`iris.experimental.regrid.regrid_area_weighted_rectilinear_src_and_grid`.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import numpy as np
import numpy.ma as ma

from iris.coord_systems import GeogCS
from iris.coords import DimCoord
from iris.cube import Cube
from iris.experimental.regrid import (
    regrid_area_weighted_rectilinear_src_and_grid as regrid,
)
from iris.tests.experimental.regrid.test_regrid_area_weighted_rectilinear_src_and_grid import (
    _resampled_grid,
)


class TestMdtol(tests.IrisTest):
    # Tests to check the masking behaviour controlled by mdtol kwarg.
    def setUp(self):
        # A (3, 2, 4) cube with a masked element.
        cube = Cube(np.ma.arange(24, dtype=np.int32).reshape((3, 2, 4)))
        cs = GeogCS(6371229)
        coord = DimCoord(
            points=np.array([-1, 0, 1], dtype=np.int32),
            standard_name="latitude",
            units="degrees",
            coord_system=cs,
        )
        cube.add_dim_coord(coord, 0)
        coord = DimCoord(
            points=np.array([-1, 0, 1, 2], dtype=np.int32),
            standard_name="longitude",
            units="degrees",
            coord_system=cs,
        )
        cube.add_dim_coord(coord, 2)
        cube.coord("latitude").guess_bounds()
        cube.coord("longitude").guess_bounds()
        cube.data[1, 1, 2] = ma.masked
        self.src_cube = cube
        # Create (7, 2, 9) grid cube.
        self.grid_cube = _resampled_grid(cube, 2.3, 2.4)

    def test_default(self):
        res = regrid(self.src_cube, self.grid_cube)
        expected_mask = np.zeros((7, 2, 9), bool)
        expected_mask[2:5, 1, 4:7] = True
        self.assertArrayEqual(res.data.mask, expected_mask)

    def test_zero(self):
        res = regrid(self.src_cube, self.grid_cube, mdtol=0)
        expected_mask = np.zeros((7, 2, 9), bool)
        expected_mask[2:5, 1, 4:7] = True
        self.assertArrayEqual(res.data.mask, expected_mask)

    def test_one(self):
        res = regrid(self.src_cube, self.grid_cube, mdtol=1)
        expected_mask = np.zeros((7, 2, 9), bool)
        # Only a single cell has all contributing cells masked.
        expected_mask[3, 1, 5] = True
        self.assertArrayEqual(res.data.mask, expected_mask)

    def test_fraction_below_min(self):
        # Cells in target grid that overlap with the masked src cell
        # have the following fractions (approx. due to spherical area).
        #   4      5      6      7
        # 2 ----------------------
        #   | 0.33 | 0.66 | 0.50 |
        # 3 ----------------------
        #   | 0.33 | 1.00 | 0.75 |
        # 4 ----------------------
        #   | 0.33 | 0.66 | 0.50 |
        # 5 ----------------------
        #

        # Threshold less than minimum fraction.
        mdtol = 0.2
        res = regrid(self.src_cube, self.grid_cube, mdtol=mdtol)
        expected_mask = np.zeros((7, 2, 9), bool)
        expected_mask[2:5, 1, 4:7] = True
        self.assertArrayEqual(res.data.mask, expected_mask)

    def test_fraction_between_min_and_max(self):
        # Threshold between min and max fraction. See
        # test_fraction_below_min() comment for picture showing
        # the fractions of masked data.
        mdtol = 0.6
        res = regrid(self.src_cube, self.grid_cube, mdtol=mdtol)
        expected_mask = np.zeros((7, 2, 9), bool)
        expected_mask[2:5, 1, 5] = True
        expected_mask[3, 1, 6] = True
        self.assertArrayEqual(res.data.mask, expected_mask)

    def test_src_not_masked_array(self):
        self.src_cube.data = self.src_cube.data.filled(1.0)
        res = regrid(self.src_cube, self.grid_cube, mdtol=0.9)
        self.assertFalse(ma.isMaskedArray(res.data))

    def test_boolean_mask(self):
        self.src_cube.data = np.ma.arange(24).reshape(3, 2, 4)
        res = regrid(self.src_cube, self.grid_cube, mdtol=0.9)
        self.assertEqual(ma.count_masked(res.data), 0)

    def test_scalar_no_overlap(self):
        # Slice src so result collapses to a scalar.
        src_cube = self.src_cube[:, 1, :]
        # Regrid to a single cell with no overlap with masked src cells.
        grid_cube = self.grid_cube[2, 1, 3]
        res = regrid(src_cube, grid_cube, mdtol=0.8)
        self.assertFalse(ma.isMaskedArray(res.data))

    def test_scalar_with_overlap_below_mdtol(self):
        # Slice src so result collapses to a scalar.
        src_cube = self.src_cube[:, 1, :]
        # Regrid to a single cell with 50% overlap with masked src cells.
        grid_cube = self.grid_cube[3, 1, 4]
        # Set threshold (mdtol) to greater than 0.5 (50%).
        res = regrid(src_cube, grid_cube, mdtol=0.6)
        self.assertEqual(ma.count_masked(res.data), 0)

    def test_scalar_with_overlap_above_mdtol(self):
        # Slice src so result collapses to a scalar.
        src_cube = self.src_cube[:, 1, :]
        # Regrid to a single cell with 50% overlap with masked src cells.
        grid_cube = self.grid_cube[3, 1, 4]
        # Set threshold (mdtol) to less than 0.5 (50%).
        res = regrid(src_cube, grid_cube, mdtol=0.4)
        self.assertEqual(ma.count_masked(res.data), 1)


class TestWrapAround(tests.IrisTest):
    def test_float_tolerant_equality(self):
        # Ensure that floating point numbers are treated appropriately when
        # introducing precision difference from wrap_around.
        source = Cube([[1]])
        cs = GeogCS(6371229)

        bounds = np.array([[-91, 0]], dtype="float")
        points = bounds.mean(axis=1)
        lon_coord = DimCoord(
            points,
            bounds=bounds,
            standard_name="longitude",
            units="degrees",
            coord_system=cs,
        )
        source.add_aux_coord(lon_coord, 1)

        bounds = np.array([[-90, 90]], dtype="float")
        points = bounds.mean(axis=1)
        lat_coord = DimCoord(
            points,
            bounds=bounds,
            standard_name="latitude",
            units="degrees",
            coord_system=cs,
        )
        source.add_aux_coord(lat_coord, 0)

        grid = Cube([[0]])
        bounds = np.array([[270, 360]], dtype="float")
        points = bounds.mean(axis=1)
        lon_coord = DimCoord(
            points,
            bounds=bounds,
            standard_name="longitude",
            units="degrees",
            coord_system=cs,
        )
        grid.add_aux_coord(lon_coord, 1)
        grid.add_aux_coord(lat_coord, 0)

        res = regrid(source, grid)
        # The result should be equal to the source data and NOT be masked.
        self.assertArrayEqual(res.data, np.array([1.0]))


if __name__ == "__main__":
    tests.main()
