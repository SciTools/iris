# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris.util.mask_cube_from_shapefile`."""

import numpy as np
import pytest
import shapely

from iris.coord_systems import RotatedGeogCS
from iris.coords import DimCoord
import iris.cube
import iris.tests as tests
from iris.util import mask_cube_from_shapefile
from iris.warnings import IrisUserWarning


class TestBasicCubeMasking(tests.IrisTest):
    """Unit tests for mask_cube_from_shapefile function."""

    def setUp(self):
        basic_data = np.array([[1, 2, 3], [4, 8, 12]])
        self.basic_cube = iris.cube.Cube(basic_data)
        coord = DimCoord(
            np.array([0, 1.0]),
            standard_name="projection_y_coordinate",
            bounds=[[0, 0.5], [0.5, 1]],
            units="1",
        )
        self.basic_cube.add_dim_coord(coord, 0)
        coord = DimCoord(
            np.array([0, 1.0, 1.5]),
            standard_name="projection_x_coordinate",
            bounds=[[0, 0.5], [0.5, 1], [1, 1.5]],
            units="1",
        )
        self.basic_cube.add_dim_coord(coord, 1)

    def test_basic_cube_intersect(self):
        shape = shapely.geometry.box(0.6, 0.6, 0.9, 0.9)
        masked_cube = mask_cube_from_shapefile(self.basic_cube, shape)
        assert (
            np.sum(masked_cube.data) == 8
        ), f"basic cube masking failed test - expected 8 got {np.sum(masked_cube.data)}"

    def test_basic_cube_intersect_in_place(self):
        shape = shapely.geometry.box(0.6, 0.6, 0.9, 0.9)
        cube = self.basic_cube.copy()
        mask_cube_from_shapefile(cube, shape, in_place=True)
        assert (
            np.sum(cube.data) == 8
        ), f"basic cube masking failed test - expected 8 got {np.sum(cube.data)}"

    def test_basic_cube_intersect_low_weight(self):
        shape = shapely.geometry.box(0.1, 0.6, 1, 1)
        masked_cube = mask_cube_from_shapefile(
            self.basic_cube, shape, minimum_weight=0.2
        )
        assert (
            np.sum(masked_cube.data) == 12
        ), f"basic cube masking weighting failed test - expected 12 got {np.sum(masked_cube.data)}"

    def test_basic_cube_intersect_high_weight(self):
        shape = shapely.geometry.box(0.1, 0.6, 1, 1)
        masked_cube = mask_cube_from_shapefile(
            self.basic_cube, shape, minimum_weight=0.7
        )
        assert (
            np.sum(masked_cube.data) == 8
        ), f"basic cube masking weighting failed test- expected 8 got {np.sum(masked_cube.data)}"

    def test_cube_list_error(self):
        cubelist = iris.cube.CubeList([self.basic_cube])
        shape = shapely.geometry.box(1, 1, 2, 2)
        with pytest.raises(TypeError, match="CubeList object rather than Cube"):
            mask_cube_from_shapefile(cubelist, shape)

    def test_non_cube_error(self):
        fake = None
        shape = shapely.geometry.box(1, 1, 2, 2)
        with pytest.raises(TypeError, match="Received non-Cube object"):
            mask_cube_from_shapefile(fake, shape)

    def test_line_shape_warning(self):
        shape = shapely.geometry.LineString([(0, 0.75), (2, 0.75)])
        with pytest.warns(IrisUserWarning, match="invalid type"):
            masked_cube = mask_cube_from_shapefile(
                self.basic_cube, shape, minimum_weight=0.1
            )
        assert (
            np.sum(masked_cube.data) == 24
        ), f"basic cube masking against line failed test - expected 24 got {np.sum(masked_cube.data)}"

    def test_cube_coord_mismatch_warning(self):
        shape = shapely.geometry.box(0.6, 0.6, 0.9, 0.9)
        cube = self.basic_cube
        cube.coord("projection_x_coordinate").points = [180, 360, 540]
        cube.coord("projection_x_coordinate").coord_system = RotatedGeogCS(30, 30)
        with pytest.warns(IrisUserWarning, match="masking"):
            mask_cube_from_shapefile(
                cube,
                shape,
            )

    def test_missing_xy_coord(self):
        shape = shapely.geometry.box(0.6, 0.6, 0.9, 0.9)
        cube = self.basic_cube
        cube.remove_coord("projection_x_coordinate")
        with pytest.raises(ValueError, match="1d xy coordinates"):
            mask_cube_from_shapefile(cube, shape)

    def test_shape_not_shape(self):
        shape = [5, 6, 7, 8]  # random array
        with pytest.raises(TypeError, match="valid Shapely"):
            mask_cube_from_shapefile(self.basic_cube, shape)

    def test_shape_invalid(self):
        shape = shapely.box(0, 1, 1, 1)
        with pytest.raises(TypeError, match="valid Shapely"):
            mask_cube_from_shapefile(self.basic_cube, shape)
