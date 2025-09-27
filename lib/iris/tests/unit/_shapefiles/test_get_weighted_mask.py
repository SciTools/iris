# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris._shapefiles._get_weighted_mask`."""

import numpy as np
import pytest
from shapely.geometry import box

from iris._shapefiles import _get_weighted_mask
from iris.coord_systems import GeogCS
from iris.coords import DimCoord
from iris.cube import Cube


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


@pytest.mark.parametrize(
    "minimum_weight, expected_mask",
    [
        (
            0.0,
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
                dtype=bool,
            ),
        ),
        (
            0.5,
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 0, 0, 1, 1, 1],
                    [1, 1, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
                dtype=bool,
            ),
        ),
        (
            1.0,
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
                dtype=bool,
            ),
        ),
    ],
)
def test_basic_weighted_mask(mock_cube, square_polygon, minimum_weight, expected_mask):
    """Test the create_shape_mask function with different minimum weights."""
    # Create a mask using the square polygon
    mask = _get_weighted_mask(mock_cube, square_polygon, minimum_weight=minimum_weight)

    # Check that the masked area corresponds to the square polygon
    assert np.array_equal(mask, expected_mask)
