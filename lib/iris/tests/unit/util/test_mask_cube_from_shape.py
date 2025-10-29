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
from iris.util import array_equal, is_masked, mask_cube_from_shape


@pytest.fixture
def square_polygon():
    # Create a roughly 3x3 square polygon
    return box(2.4, 2.4, 6.4, 6.4)


@pytest.fixture
def mock_cube():
    """Create a mock 9x9 Iris cube for testing."""
    x_points = np.linspace(1, 9, 9) - 0.5  # Specify cube cell midpoints
    y_points = np.linspace(1, 9, 9) - 0.5
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


def test_mask_cube_from_shape_inplace(mock_cube, square_polygon):
    masked_cube = mask_cube_from_shape(
        cube=mock_cube,
        shape=square_polygon,
        shape_crs=CRS.from_epsg(4326),
        in_place=True,
    )
    assert masked_cube is None
    assert is_masked(mock_cube.data)


def test_mask_cube_from_shape_not_inplace(mock_cube, square_polygon):
    masked_cube = mask_cube_from_shape(
        cube=mock_cube,
        shape=square_polygon,
        shape_crs=CRS.from_epsg(4326),
        in_place=False,
    )
    assert masked_cube is not None
    assert is_masked(masked_cube.data)
    # Original cube should remain unmasked
    assert not is_masked(mock_cube.data)


@pytest.mark.parametrize(
    "minimum_weight, expected_output",
    [
        (
            0.0,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ),
        ),
        (
            0.5,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ),
        ),
        (
            1.0,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ),
        ),
    ],
)
def test_basic_mask_cube_from_shape(
    mock_cube, square_polygon, minimum_weight, expected_output
):
    """Test the create_shape_mask function with different minimum weights."""
    expected_cube = mock_cube.copy(
        data=np.ma.array(
            expected_output, dtype=float, mask=np.logical_not(expected_output)
        )
    )
    # Create a mask using the square polygon
    mask = mask_cube_from_shape(
        cube=mock_cube,
        shape=square_polygon,
        shape_crs=None,
        minimum_weight=minimum_weight,
    )

    assert array_equal(mask.data, expected_cube.data)


def test_mask_cube_from_shape_invert(mock_cube, square_polygon):
    """Test the create_shape_mask function with different minimum weights."""
    expected_output = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
    )

    expected_cube = mock_cube.copy(
        data=np.ma.array(
            np.logical_not(expected_output), dtype=float, mask=expected_output
        )
    )
    # Create a mask using the square polygon
    mask = mask_cube_from_shape(
        cube=mock_cube,
        shape=square_polygon,
        shape_crs=None,
        minimum_weight=0,
        invert=True,
    )

    assert array_equal(mask.data, expected_cube.data)


def test_mask_cube_from_shape_all_touched(mock_cube, square_polygon):
    """Test the create_shape_mask function with different minimum weights."""
    expected_output = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
    )

    expected_cube = mock_cube.copy(
        data=np.ma.array(
            expected_output, dtype=float, mask=np.logical_not(expected_output)
        )
    )
    # Create a mask using the square polygon
    mask = mask_cube_from_shape(
        cube=mock_cube,
        shape=square_polygon,
        shape_crs=None,
        all_touched=True,
    )

    assert array_equal(mask.data, expected_cube.data)
