# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris._shapefiles.is_geometry_valid`."""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

from pyproj import CRS
import pytest
from shapely.geometry import box

from iris._shapefiles import is_geometry_valid


# Shareable shape fixtures used in:
# - util/test_mask_cube_from_shapefile.py
# - _shapefiles/test_is_geometry_valid.py
@pytest.fixture(scope="session")
def wgs84_crs():
    return CRS.from_epsg(4326)


@pytest.fixture(scope="session")
def osgb_crs():
    return CRS.from_epsg(27700)


@pytest.fixture(scope="session")
def basic_circular_geometry():
    # Define geometry of a basic circle with center at (0, 0) and radius 10
    center = (0, 0)
    radius = 10

    circle = Point(center).buffer(radius)

    return circle


@pytest.fixture(scope="session")
def basic_rectangular_geometry():
    # Define the coordinates of a basic rectangle
    min_lon = -90
    min_lat = -45
    max_lon = 90
    max_lat = 45

    # Create the rectangular geometry
    return box(min_lon, min_lat, max_lon, max_lat)


@pytest.fixture(scope="session")
def basic_point_geometry():
    # Define the coordinates of a basic point (lon, lat)
    return Point((-3.476204, 50.727059))


@pytest.fixture(scope="session")
def basic_line_geometry():
    # Define the coordinates of a basic line
    return LineString([(0, 0), (10, 10)])


@pytest.fixture(scope="session")
def basic_point_collection():
    # Define the coordinates of a basic collection of points
    # as (lon, lat) tuples, assuming a WGS84 projection.
    points = MultiPoint(
        [
            (0, 0),
            (10, 10),
            (-10, -10),
            (-3.476204, 50.727059),
            (174.761067, -36.846211),
            (-77.032801, 38.892717),
        ]
    )

    return points


@pytest.fixture(scope="session")
def canada_geometry():
    # Define the coordinates of a rectangle that covers Canada
    return box(-143.5, 42.6, -37.8, 84.0)


@pytest.fixture(scope="session")
def bering_sea_geometry():
    # Define the coordinates of a rectangle that covers the Bering Sea
    return box(148.42, 49.1, -138.74, 73.12)


@pytest.fixture(scope="session")
def uk_geometry():
    # Define the coordinates of a rectangle that covers the UK
    return box(-10, 49, 2, 61)


@pytest.fixture(scope="session")
def invalid_geometry_poles():
    # Define the coordinates of a rectangle that crosses the poles
    return box(-10, -90, 10, 90)


@pytest.fixture(scope="session")
def invalid_geometry_bounds():
    # Define the coordinates of a rectangle that is outside the bounds of the coordinate system
    return box(-200, -100, 200, 100)


class TestGeometry(tests.IrisTest):
    # Test validity of different geometries
    @pytest.mark.parametrize(
        "test_input",
        [
            "basic_circular_geometry",
            "basic_rectangular_geometry",
            "basic_point_geometry",
            "basic_line_geometry",
            "basic_point_collection",
            "canada_geometry",
        ],
    )
    def test_valid_geometry(test_input, expected):
        # Assert that all valid geometries are return None
        assert is_geometry_valid(request.getfixturevalue(test_input), wgs84) is None

    # Fixtures retrieved from conftest.py
    @pytest.mark.parametrize(
        "test_input, errortype, errormessage",
        [
            (
                "bering_sea_geometry",
                ValueError,
                "Geometry crossing the antimeridian is not supported.",
            ),
            (
                "invalid_geometry_poles",
                ValueError,
                "Geometry crossing the poles is not supported.",
            ),
            (
                "invalid_geometry_bounds",
                ValueError,
                "Geometry [<POLYGON ((200 -100, 200 100, -200 100, -200 -100, 200 -100))>] is not valid for the given coordinate system EPSG:4326. Check that your coordinates are correctly specified.",
            ),
            (
                "not a valid geometry",
                TypeError,
                "Geometry is not a valid Shapely object",
            ),
        ],
    )
    def test_invalid_geometry(test_input, errortype, errormessage):
        # Assert that all invalid geometries raise the expected error
        with pytest.raises(errortype, match=errormessage):
            is_geometry_valid(request.getfixturevalue(test_input), wgs84)
