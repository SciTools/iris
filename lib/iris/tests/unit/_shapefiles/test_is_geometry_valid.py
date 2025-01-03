# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris._shapefiles.is_geometry_valid`."""

from pyproj import CRS
import pytest
from shapely.geometry import box

from iris._shapefiles import is_geometry_valid


# Fixtures retrieved from conftest.py
@pytest.mark.parametrize(
    "test_input",
    [
        basic_circular_geometry(),
        basic_rectangular_geometry(),
        basic_point_geometry(),
        basic_line_geometry(),
        basic_point_collection(),
        canada_geometry(),
    ],
)
def test_valid_geometry(test_input, expected):
    # Assert that all valid geometries are return None
    assert is_geometry_valid(test_input, wgs84) is None


# Fixtures retrieved from conftest.py
@pytest.mark.parametrize(
    "test_input, errortype, errormessage",
    [
        (
            bering_sea_geometry(),
            ValueError,
            "Geometry crossing the antimeridian is not supported.",
        ),
        (
            invalid_geometry_poles(),
            ValueError,
            "Geometry crossing the poles is not supported.",
        ),
        (
            invalid_geometry_bounds(),
            ValueError,
            "Geometry [<POLYGON ((200 -100, 200 100, -200 100, -200 -100, 200 -100))>] is not valid for the given coordinate system EPSG:4326. Check that your coordinates are correctly specified.",
        ),
        ("not a valid geometry", TypeError, "Geometry is not a valid Shapely object"),
    ],
)
def test_invalid_geometry(test_input, errortype, errormessage):
    # Assert that all invalid geometries raise the expected error
    with pytest.raises(errortype, match=errormessage):
        is_geometry_valid(test_input, wgs84)
