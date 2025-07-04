# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris._shapefiles._make_raster_cube_transform`."""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

from affine import Affine
import numpy as np
import pytest

import iris
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

    # Call the function
    transform = _make_raster_cube_transform(mock_cube)

    # Validate the result
    dx = regular_step(mock_cube.coord(x_name))
    dy = regular_step(mock_cube.coord(y_name))
    expected_transform = Affine.translation(-dx / 2, -dy / 2) * Affine.scale(dx, dy)

    assert isinstance(transform, Affine)
    assert transform == expected_transform


def test_invalid_cube(mock_nonregular_cube):
    # Assert that all invalid geometries raise the expected error
    errormessage = "Coord longitude is not regular"
    with pytest.raises(CoordinateNotRegularError, match=errormessage):
        _make_raster_cube_transform(mock_nonregular_cube)
