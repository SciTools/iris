# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests fixture infra-structure."""

from pyproj import CRS
import pytest

import iris


@pytest.fixture
def sample_coord():
    sample_coord = iris.coords.DimCoord(points=(1, 2, 3, 4, 5))
    return sample_coord


# Shareable shape fixtures used in:
# - util/test_mask_cube_from_shapefile.py
# - _shapefiles/test_is_geometry_valid.py
@pytest.fixture(scope="session")
def wgs84_crs():
    return CRS.from_epsg(4326)

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
def invalid_geometry_poles():
    # Define the coordinates of a rectangle that crosses the poles
    return box(-10, -90, 10, 90)

@pytest.fixture(scope="session")
def invalid_geometry_bounds():
    # Define the coordinates of a rectangle that is outside the bounds of the coordinate system
    return box(-200, -100, 200, 100)