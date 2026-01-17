# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris._shapefiles._make_raster_cube_transform`."""

from affine import Affine
import numpy as np
import pytest

from iris._shapefiles import _make_raster_cube_transform
from iris.coords import DimCoord
from iris.cube import Cube
from iris.exceptions import CoordinateNotRegularError
from iris.util import regular_step


@pytest.fixture
def mock_cube():
    """Create a mock Iris cube for testing."""
    x_points = np.array([0.0, 1.0, 2.0, 3.0])
    y_points = np.array([0.0, 1.0, 2.0, 3.0])
    x_coord = DimCoord(x_points, standard_name="longitude", units="degrees")
    y_coord = DimCoord(y_points, standard_name="latitude", units="degrees")
    data = np.zeros((len(y_points), len(x_points)))
    cube = Cube(data, dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])
    return cube


@pytest.fixture
def mock_nonregular_cube():
    """Create a mock Iris cube for testing."""
    x_points = np.array(
        [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.5, 10.0]
    )
    y_points = np.array(
        [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.5, 10.0]
    )
    x_coord = DimCoord(x_points, standard_name="longitude", units="degrees")
    y_coord = DimCoord(y_points, standard_name="latitude", units="degrees")
    data = np.zeros((len(y_points), len(x_points)))
    cube = Cube(data, dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])
    return cube


def test_make_raster_cube_transform(mock_cube):
    """Test the `_make_raster_cube__transform` function."""
    x_name = "longitude"
    y_name = "latitude"
    x_coord, y_coord = [mock_cube.coord(a) for a in (x_name, y_name)]

    # Call the function
    transform = _make_raster_cube_transform(x_coord, y_coord)

    # Validate the result
    dx = regular_step(x_coord)
    dy = regular_step(y_coord)
    expected_transform = Affine.translation(-dx / 2, -dy / 2) * Affine.scale(dx, dy)

    assert isinstance(transform, Affine)
    assert transform == expected_transform


def test_invalid_cube(mock_nonregular_cube):
    x_coord, y_coord = [
        mock_nonregular_cube.coord(a) for a in ("longitude", "latitude")
    ]
    # Assert that all invalid geometries raise the expected error
    errormessage = "Coord longitude is not regular"
    with pytest.raises(CoordinateNotRegularError, match=errormessage):
        _make_raster_cube_transform(x_coord, y_coord)
