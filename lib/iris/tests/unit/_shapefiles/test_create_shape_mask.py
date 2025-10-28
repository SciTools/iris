# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris._shapefiles.create_shape_mask`."""

import numpy as np
from pyproj import CRS
import pytest
from shapely.geometry import Point, Polygon

from iris._shapefiles import create_shape_mask
from iris.coord_systems import GeogCS
from iris.coords import DimCoord
from iris.cube import Cube, CubeList
from iris.exceptions import IrisError
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


def test_basic_create_shape_mask(square_polygon, wgs84_crs, mock_cube):
    """Test the create_shape_mask function."""
    # Create a mask using the square polygon
    mask = create_shape_mask(
        geometry=square_polygon, geometry_crs=wgs84_crs, cube=mock_cube
    )

    # Check that the mask is a boolean array with the same shape as the cube data
    assert mask.shape == mock_cube.data.shape
    assert mask.dtype == np.bool_

    # Check that the masked area corresponds to the square polygon
    expected_mask = np.ones_like(mock_cube.data, dtype=bool)
    expected_mask[3:6, 3:6] = False  # The square polygon covers this area
    assert np.array_equal(mask, expected_mask)


def test_basic_create_shape_mask_with_None_crs(square_polygon, mock_cube):
    """Test the create_shape_mask function."""
    # Create a mask using the square polygon
    # Here we pass None for geometry_crs to test default behaviour
    # which assumes the geometry is in the same CRS as the cube
    mask = create_shape_mask(geometry=square_polygon, geometry_crs=None, cube=mock_cube)

    # Check that the mask is a boolean array with the same shape as the cube data
    assert mask.shape == mock_cube.data.shape
    assert mask.dtype == np.bool_

    # Check that the masked area corresponds to the square polygon
    expected_mask = np.ones_like(mock_cube.data, dtype=bool)
    expected_mask[3:6, 3:6] = False  # The square polygon covers this area
    assert np.array_equal(mask, expected_mask)


def test_basic_create_shape_mask_radians(square_polygon, wgs84_crs, mock_cube):
    """Test the create_shape_mask function."""
    # Convert mock cube coordinates to radians
    mock_cube.coord("longitude").convert_units("radians")
    mock_cube.coord("latitude").convert_units("radians")

    # Create a mask using the square polygon
    mask = create_shape_mask(
        geometry=square_polygon, geometry_crs=wgs84_crs, cube=mock_cube
    )

    # Check that the mask is a boolean array with the same shape as the cube data
    assert mask.shape == mock_cube.data.shape
    assert mask.dtype == np.bool_

    # Check that the masked area corresponds to the square polygon
    expected_mask = np.ones_like(mock_cube.data, dtype=bool)
    expected_mask[3:6, 3:6] = False  # The square polygon covers this area
    assert np.array_equal(mask, expected_mask)


def test_invert_create_shape_mask(square_polygon, wgs84_crs, mock_cube):
    """Test the create_shape_mask function."""
    # Create a mask using the square polygon
    mask = create_shape_mask(
        geometry=square_polygon, geometry_crs=wgs84_crs, cube=mock_cube, invert=True
    )

    # Check that the mask is a boolean array with the same shape as the cube data
    assert mask.shape == mock_cube.data.shape
    assert mask.dtype == np.bool_

    # Check that the masked area corresponds to the square polygon
    expected_mask = np.zeros_like(mock_cube.data, dtype=bool)
    expected_mask[3:6, 3:6] = True  # The square polygon covers this area
    assert np.array_equal(mask, expected_mask)


def test_all_touched_true_create_shape_mask(circle_polygon, wgs84_crs, mock_cube):
    """Test the create_shape_mask function."""
    # Create a mask using the circular polygon
    mask = create_shape_mask(
        geometry=circle_polygon,
        geometry_crs=wgs84_crs,
        cube=mock_cube,
        all_touched=True,
    )

    # Check that the mask is a boolean array with the same shape as the cube data
    assert mask.shape == mock_cube.data.shape
    assert mask.dtype == np.bool_

    # Check that the masked area corresponds to the circular polygon
    expected_mask = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=bool,
    )
    assert np.array_equal(mask, expected_mask)


def test_all_touched_false_create_shape_mask(circle_polygon, wgs84_crs, mock_cube):
    """Test the create_shape_mask function."""
    # Create a mask using the circular polygon
    mask = create_shape_mask(
        geometry=circle_polygon,
        geometry_crs=wgs84_crs,
        cube=mock_cube,
        all_touched=False,
    )

    # Check that the mask is a boolean array with the same shape as the cube data
    assert mask.shape == mock_cube.data.shape
    assert mask.dtype == np.bool_

    # Check that the masked area corresponds to the circular polygon
    expected_mask = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=bool,
    )
    assert np.array_equal(mask, expected_mask)


class TestCreateShapeMaskErrors:
    def test_invalid_cube_type(self, square_polygon, wgs84_crs):
        # Pass an invalid cube type (e.g., a string or CubeList)
        err_message = "Received non-Cube object where a Cube is expected"
        with pytest.raises(TypeError, match=err_message):
            create_shape_mask(
                geometry=square_polygon, geometry_crs=wgs84_crs, cube="not_a_cube"
            )

    def test_invalid_cubelist_type(self, square_polygon, wgs84_crs):
        err_message = (
            "Received CubeList object rather than Cube - "
            "to mask a CubeList iterate over each Cube"
        )
        with pytest.raises(TypeError, match=err_message):
            create_shape_mask(
                geometry=square_polygon, geometry_crs=wgs84_crs, cube=CubeList()
            )

    def test_invalid_cube_crs(self, square_polygon, wgs84_crs):
        # Pass a cube without a coordinate system
        cube = Cube(np.ones((10, 10)), dim_coords_and_dims=[])
        err_message = (
            "Cube coordinates do not have a coordinate references system \\(CRS\\) "
            "defined. A CRS must be defined to ensure reliable results."
        )
        with pytest.raises(IrisError, match=err_message):
            create_shape_mask(
                geometry=square_polygon, geometry_crs=wgs84_crs, cube=cube
            )

    @pytest.mark.parametrize(
        "minimum_weight, error_type",
        [(-1, ValueError), (2, ValueError)],
    )
    def test_invalid_minimum_weight(
        self, square_polygon, wgs84_crs, mock_cube, minimum_weight, error_type
    ):
        # Pass invalid minimum_weight values
        err_message = "Minimum weight must be between 0.0 and 1.0"
        with pytest.raises(error_type, match=err_message):
            create_shape_mask(
                geometry=square_polygon,
                geometry_crs=wgs84_crs,
                cube=mock_cube,
                minimum_weight=minimum_weight,
                all_touched=None,
            )

    @pytest.mark.parametrize(
        "minimum_weight, error_type",
        [(-1, ValueError), (2, ValueError)],
    )
    def test_invalid_minimum_weight_with_all_touched(
        self, square_polygon, wgs84_crs, mock_cube, minimum_weight, error_type
    ):
        # Pass invalid minimum_weight values
        err_message = "Minimum weight must be between 0.0 and 1.0"
        with pytest.raises(error_type, match=err_message):
            create_shape_mask(
                geometry=square_polygon,
                geometry_crs=wgs84_crs,
                cube=mock_cube,
                minimum_weight=minimum_weight,
                all_touched=False,
            )

    def test_invalid_args(self, square_polygon, wgs84_crs, mock_cube):
        # Pass invalid minimum_weight values
        err_message = "Cannot use minimum_weight > 0.0 with all_touched=True."
        with pytest.raises(ValueError, match=err_message):
            create_shape_mask(
                geometry=square_polygon,
                geometry_crs=wgs84_crs,
                cube=mock_cube,
                minimum_weight=0.5,
                all_touched=True,
            )

    def test_incompatible_crs_warning(self, square_polygon, mock_cube):
        # Pass a CRS that does not match the cube's CRS
        crs = CRS.from_epsg(3857)  # Web Mercator, different from WGS84
        warn_message = "Geometry CRS does not match cube CRS. Iris will attempt to transform the geometry onto the cube CRS..."
        with pytest.warns(IrisUserWarning, match=warn_message):
            create_shape_mask(geometry=square_polygon, geometry_crs=crs, cube=mock_cube)
