# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris._shapefiles.is_geometry_valid`."""

from pyproj import CRS
import pytest
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    box,
)

from iris._shapefiles import is_geometry_valid
from iris.warnings import IrisUserWarning


# Shareable shape fixtures used in:
# - util/test_mask_cube_from_shapefile.py
# - _shapefiles/test_is_geometry_valid.py
@pytest.fixture()
def wgs84_crs():
    return CRS.from_epsg(4326)


@pytest.fixture()
def osgb_crs():
    return CRS.from_epsg(27700)


@pytest.fixture()
def basic_polygon_geometry():
    # Define the coordinates of a basic rectangle
    min_lon = -90
    min_lat = -45
    max_lon = 90
    max_lat = 45

    # Create the rectangular geometry
    return box(min_lon, min_lat, max_lon, max_lat)


@pytest.fixture()
def basic_wide_polygon_geometry():
    # Define the coordinates of a basic rectangle
    min_lon = -170
    min_lat = -45
    max_lon = 170
    max_lat = 45

    # Create the rectangular geometry
    return box(min_lon, min_lat, max_lon, max_lat)


@pytest.fixture()
def basic_multipolygon_geometry():
    # Define the coordinates of a basic rectangle
    min_lon = 0
    min_lat = 0
    max_lon = 8
    max_lat = 8

    # Create the rectangular geometry
    return MultiPolygon(
        [
            box(min_lon, min_lat, max_lon, max_lat),
            box(min_lon + 10, min_lat + 10, max_lon + 10, max_lat + 10),
        ]
    )


@pytest.fixture()
def basic_point_geometry():
    # Define the coordinates of a basic point (lon, lat)
    return Point((-3.476204, 50.727059))


@pytest.fixture()
def basic_line_geometry():
    # Define the coordinates of a basic line
    return LineString([(0, 0), (10, 10)])


@pytest.fixture()
def basic_multiline_geometry():
    # Define the coordinates of a basic line
    return MultiLineString([[(0, 0), (10, 10)], [(20, 20), (30, 30)]])


@pytest.fixture()
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


@pytest.fixture()
def canada_geometry():
    # Define the coordinates of a rectangle that covers Canada
    return box(-143.5, 42.6, -37.8, 84.0)


@pytest.fixture()
def bering_sea_geometry():
    # Define the coordinates of a rectangle that covers the Bering Sea
    return box(148.42, 49.1, -138.74, 73.12)


@pytest.fixture()
def uk_geometry():
    # Define the coordinates of a rectangle that covers the UK
    return box(-10, 49, 2, 61)


@pytest.fixture()
def invalid_geometry_poles():
    # Define the coordinates of a rectangle that crosses the poles
    return box(-10, -90, 10, 90)


@pytest.fixture()
def invalid_geometry_bounds():
    # Define the coordinates of a rectangle that is outside the bounds of the coordinate system
    return box(-200, -100, 200, 100)


@pytest.fixture()
def not_a_valid_geometry():
    # Return an invalid geometry type
    # This is not a valid geometry, e.g., a string
    return "This is not a valid geometry"


# Test validity of different geometries
@pytest.mark.parametrize(
    "test_input",
    [
        "basic_polygon_geometry",
        "basic_multipolygon_geometry",
        "basic_point_geometry",
        "basic_point_collection",
        "basic_line_geometry",
        "basic_multiline_geometry",
        "canada_geometry",
    ],
)
def test_valid_geometry(test_input, request, wgs84_crs):
    # Assert that all valid geometries are return None
    assert is_geometry_valid(request.getfixturevalue(test_input), wgs84_crs) is None


# Fixtures retrieved from conftest.py
# N.B. error message comparison is done with regex so
# any parentheses in the error message must be escaped (\)
@pytest.mark.parametrize(
    "test_input, errortype, err_message",
    [
        (
            "invalid_geometry_poles",
            ValueError,
            "Geometry crossing the poles is not supported.",
        ),
        (
            "invalid_geometry_bounds",
            ValueError,
            r"Geometry \[<POLYGON \(\(200 -100, 200 100, -200 100, -200 -100, 200 -100\)\)>\] is not valid for the given coordinate system EPSG:4326.\nCheck that your coordinates are correctly specified.",
        ),
        (
            "not_a_valid_geometry",
            TypeError,
            r"Shape geometry is not a valid shape \(not well formed\).",
        ),
    ],
)
def test_invalid_geometry(test_input, errortype, err_message, request, wgs84_crs):
    # Assert that all invalid geometries raise the expected error
    with pytest.raises(errortype, match=err_message):
        is_geometry_valid(request.getfixturevalue(test_input), wgs84_crs)


@pytest.mark.parametrize(
    "test_input",
    (
        "basic_wide_polygon_geometry",
        "bering_sea_geometry",
    ),
)
def test_warning_geometry(test_input, request, wgs84_crs):
    # Assert that all invalid geometries raise the expected error
    warn_message = (
        "Geometry crossing the antimeridian is not supported. "
        "Cannot verify non-crossing given current geometry bounds."
    )
    with pytest.warns(IrisUserWarning, match=warn_message):
        is_geometry_valid(request.getfixturevalue(test_input), wgs84_crs)


def test_invalid_crs(basic_polygon_geometry):
    # Assert that an invalid crs raise the expected error
    err_message = (
        "Geometry CRS must be a cartopy.crs or pyproj.CRS object, not <class 'str'>."
    )
    # Using a string as an invalid CRS type
    with pytest.raises(TypeError, match=err_message):
        is_geometry_valid(basic_polygon_geometry, "invalid_crs")
