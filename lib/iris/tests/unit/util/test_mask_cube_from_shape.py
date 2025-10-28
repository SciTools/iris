# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris.util.mask_cube_from_shapefile`."""

import numpy as np
from pyproj import CRS
import pytest
from shapely.geometry import box

from iris.coord_systems import GeogCS
from iris.coords import DimCoord
from iris.cube import Cube
from iris.util import is_masked, mask_cube_from_shape


@pytest.fixture
def mock_cube():
    """Create a mock 10x10 Iris cube for testing."""
    x_points = np.linspace(0, 9, 10)
    y_points = np.linspace(0, 9, 10)
    x_coord = DimCoord(
        x_points,
        standard_name="longitude",
        units="degrees",
        coord_system=GeogCS(6371229),
    )
    y_coord = DimCoord(
        y_points,
        standard_name="latitude",
        units="degrees",
        coord_system=GeogCS(6371229),
    )
    data = np.ones((len(y_points), len(x_points)))
    cube = Cube(data, dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])
    return cube


def test_mask_cube_from_shape_inplace(
    mock_cube,
):
    shape = box(5, 5, 10, 10)
    masked_cube = mask_cube_from_shape(
        mock_cube, shape, shape_crs=CRS.from_epsg(4326), in_place=True
    )
    assert masked_cube is None
    assert is_masked(mock_cube.data)


def test_mask_cube_from_shape_not_inplace(mock_cube):
    shape = box(5, 5, 10, 10)
    masked_cube = mask_cube_from_shape(
        mock_cube, shape, shape_crs=CRS.from_epsg(4326), in_place=False
    )
    assert masked_cube is not None
    assert is_masked(masked_cube.data)
    # Original cube should remain unmasked
    assert not is_masked(mock_cube.data)
