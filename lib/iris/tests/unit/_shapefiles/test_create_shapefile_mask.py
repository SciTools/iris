# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris._shapefiles.create_shapefile_mask`."""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import numpy as np
from pyproj import CRS
import pytest
from shapely.geometry import Point, Polygon

from iris._shapefiles import create_shapefile_mask
from iris.coord_systems import GeogCS
from iris.coords import DimCoord
from iris.cube import Cube, CubeList
from iris.warnings import IrisUserWarning


@pytest.fixture(scope="session")
def wgs84_crs():
    return CRS.from_epsg(4326)


@pytest.fixture
def square_polygon():
    # Create a a 3x3 square polygon from (3,3) to (6,6) using shapely
    return Polygon([(3, 3), (6, 3), (6, 6), (3, 6)])


@pytest.fixture
def circle_polygon():
    # Create a a circular polygon centred on (5,5) with radius (2,) using shapely
    return Point(5, 5).buffer(2)


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


def test_basic_create_shapefile_mask(square_polygon, wgs84_crs, mock_cube):
    """Test the create_shapefile_mask function."""
    # Create a mask using the square polygon
    mask = create_shapefile_mask(square_polygon, wgs84_crs, mock_cube)

    # Check that the mask is a boolean array with the same shape as the cube data
    assert mask.shape == mock_cube.data.shape
    assert mask.dtype == np.bool_

    # Check that the masked area corresponds to the square polygon
    expected_mask = np.ones_like(mock_cube.data, dtype=bool)
    expected_mask[3:6, 3:6] = False  # The square polygon covers this area
    assert np.array_equal(mask, expected_mask)


def test_invert_create_shapefile_mask(square_polygon, wgs84_crs, mock_cube):
    """Test the create_shapefile_mask function."""
    # Create a mask using the square polygon
    mask = create_shapefile_mask(square_polygon, wgs84_crs, mock_cube, invert=True)

    # Check that the mask is a boolean array with the same shape as the cube data
    assert mask.shape == mock_cube.data.shape
    assert mask.dtype == np.bool_

    # Check that the masked area corresponds to the square polygon
    expected_mask = np.zeros_like(mock_cube.data, dtype=bool)
    expected_mask[3:6, 3:6] = True  # The square polygon covers this area
    assert np.array_equal(mask, expected_mask)


def test_all_touched_true_create_shapefile_mask(circle_polygon, wgs84_crs, mock_cube):
    """Test the create_shapefile_mask function."""
    # Create a mask using the square polygon
    mask = create_shapefile_mask(circle_polygon, wgs84_crs, mock_cube, all_touched=True)

    # Check that the mask is a boolean array with the same shape as the cube data
    assert mask.shape == mock_cube.data.shape
    assert mask.dtype == np.bool_

    # Check that the masked area corresponds to the circular polygon
    expected_mask = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=bool,
    )
    assert np.array_equal(mask, expected_mask)


def test_all_touched_false_create_shapefile_mask(circle_polygon, wgs84_crs, mock_cube):
    """Test the create_shapefile_mask function."""
    # Create a mask using the square polygon
    mask = create_shapefile_mask(
        circle_polygon, wgs84_crs, mock_cube, all_touched=False
    )

    # Check that the mask is a boolean array with the same shape as the cube data
    assert mask.shape == mock_cube.data.shape
    assert mask.dtype == np.bool_

    # Check that the masked area corresponds to the circular polygon
    expected_mask = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=bool,
    )
    assert np.array_equal(mask, expected_mask)


def test_create_shapefile_mask_(square_polygon, wgs84_crs, mock_cube):
    """Test the create_shapefile_mask function."""
    # Create a mask using the square polygon
    mask = create_shapefile_mask(square_polygon, wgs84_crs, mock_cube, invert=True)

    # Check that the mask is a boolean array with the same shape as the cube data
    assert mask.shape == mock_cube.data.shape
    assert mask.dtype == np.bool_

    # Check that the masked area corresponds to the square polygon
    expected_mask = np.zeros_like(mock_cube.data, dtype=bool)
    expected_mask[3:6, 3:6] = True  # The square polygon covers this area
    assert np.array_equal(mask, expected_mask)


class TestCreateShapefileMaskErrors:
    def test_invalid_polygon_type(self, wgs84_crs, mock_cube):
        # Pass an invalid geometry type (e.g., a string)
        with pytest.raises(TypeError):
            create_shapefile_mask("not_a_polygon", wgs84_crs, mock_cube)

    def test_invalid_crs_type(self, square_polygon, mock_cube):
        # Pass an invalid CRS type (e.g., a string)
        with pytest.raises(TypeError):
            create_shapefile_mask(square_polygon, "not_a_crs", mock_cube)

    def test_invalid_cube_type(self, square_polygon, wgs84_crs):
        # Pass an invalid cube type (e.g., a string or CubeList)
        with pytest.raises(TypeError):
            create_shapefile_mask(square_polygon, wgs84_crs, "not_a_cube")
        with pytest.raises(TypeError):
            create_shapefile_mask(square_polygon, wgs84_crs, CubeList())

    def test_invalid_minimum_weight(self, square_polygon, wgs84_crs):
        # Pass invalid minimum_weight values
        with pytest.raises(TypeError):
            create_shapefile_mask(
                square_polygon, wgs84_crs, mock_cube, minimum_weight="not_a_number"
            )
        with pytest.raises(TypeError):
            create_shapefile_mask(
                square_polygon, wgs84_crs, mock_cube, minimum_weight=-1
            )
        with pytest.raises(TypeError):
            create_shapefile_mask(
                square_polygon, wgs84_crs, mock_cube, minimum_weight=2
            )

    def test_invalid_args(self, square_polygon, wgs84_crs, mock_cube):
        # Pass invalid minimum_weight values
        with pytest.raises(ValueError):
            create_shapefile_mask(
                square_polygon,
                wgs84_crs,
                mock_cube,
                minimum_weight=0.5,
                all_touched=True,
            )

    def test_incompatible_crs_warning(self, square_polygon, mock_cube):
        # Pass a CRS that does not match the cube's CRS
        crs = CRS.from_epsg(3857)  # Web Mercator, different from WGS84
        warn_message = "Geometry CRS does not match cube CRS. Iris will attempt to transform the geometry onto the cube CRS..."
        with pytest.warns(IrisUserWarning, match=warn_message):
            create_shapefile_mask(square_polygon, crs, mock_cube)


# Note: `minimum_weight` keyword argument is tested under its' own unit test
#       `test_mask_cube_from_shapefile.py` in the `lib/iris/tests/unit/_shapefiles/` directory.
