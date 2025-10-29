# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris._shapefiles._transform_geometry`."""

import numpy as np
import pyproj
from pyproj import CRS
from pyproj import exceptions as pyproj_exceptions
import pytest
import shapely
from shapely.geometry import LineString, Point, Polygon

import iris
from iris._shapefiles import _transform_geometry
from iris.tests import _shared_utils, stock


@pytest.fixture
def wgs84_crs():
    return CRS.from_epsg(4326)  # WGS84 coordinate system


@pytest.mark.parametrize(
    "input_geometry, wgs84_crs, input_cube_crs, output_expected_geometry",
    [
        (  # Basic geometry in WGS84, no transformation needed
            shapely.geometry.box(-10, 50, 2, 60),
            "wgs84_crs",
            stock.simple_pp().coord_system()._crs,
            shapely.geometry.box(-10, 50, 2, 60),
        ),
        (  # Basic geometry in WGS84, transformed to OSGB
            shapely.geometry.box(-10, 50, 2, 60),
            "wgs84_crs",
            iris.load_cube(
                _shared_utils.get_data_path(
                    ("NetCDF", "transverse_mercator", "tmean_1910_1910.nc")
                )
            )
            .coord_system()
            .as_cartopy_projection(),
            Polygon(  # Known Good Output
                [
                    (686600.5247600826, 18834.835866007765),
                    (622998.2965261642, 1130592.5248690117),
                    (-45450.06302316813, 1150844.967615187),
                    (-172954.59474739246, 41898.60193228102),
                    (686600.5247600826, 18834.835866007765),
                ]
            ),
        ),
        (  # Basic geometry in WGS84, no transformation needed
            LineString([(-10, 50), (2, 60)]),
            "wgs84_crs",
            stock.simple_pp().coord_system()._crs,
            LineString([(-10, 50), (2, 60)]),
        ),
        (  # Basic geometry in WGS84, no transformation needed
            Point((-10, 50)),
            "wgs84_crs",
            stock.simple_pp().coord_system()._crs,
            Point((-10, 50)),
        ),
    ],
    indirect=["wgs84_crs"],
)
def test_transform_geometry(
    input_geometry,
    wgs84_crs,
    input_cube_crs,
    output_expected_geometry,
):
    # Check PROJ network settings and disable network access for the test
    default_pyproj_network = pyproj.network.is_network_enabled()
    if default_pyproj_network:
        pyproj.network.set_network_enabled(active=False)

    out_geometry = _transform_geometry(
        geometry=input_geometry, geometry_crs=wgs84_crs, cube_crs=input_cube_crs
    )
    assert isinstance(out_geometry, shapely.geometry.base.BaseGeometry)
    assert output_expected_geometry == out_geometry

    # Reset PROJ network settings to default state
    if default_pyproj_network:
        pyproj.network.set_network_enabled(active=True)


# Assert that an invalid inputs raise the expected errors
@pytest.mark.parametrize(
    "input_geometry, input_geometry_crs, input_cube_crs, expected_error",
    [
        (  # Basic geometry in WGS84, no transformation needed
            "bad_input_geometry",
            "wgs84_crs",
            stock.simple_pp().coord_system()._crs,
            AttributeError,
        ),
        (  # Basic geometry in WGS84, no transformation needed
            shapely.geometry.box(-10, 50, 2, 60),
            "bad_input_crs",
            stock.simple_pp().coord_system()._crs,
            pyproj_exceptions.CRSError,
        ),
        (  # Basic geometry in WGS84, no transformation needed
            shapely.geometry.box(-10, 50, 2, 60),
            wgs84_crs,
            "bad_input_cube_crs",
            pyproj_exceptions.CRSError,
        ),
    ],
)
def test_transform_geometry_invalid_input(
    input_geometry, input_geometry_crs, input_cube_crs, expected_error
):
    with pytest.raises(expected_error):
        _transform_geometry(input_geometry, input_geometry_crs, input_cube_crs)


@pytest.mark.parametrize(
    "input_geometry, wgs84_crs, input_cube_crs",
    [
        (  # Basic geometry in WGS84, transformed to OSGB
            shapely.geometry.box(np.inf, np.inf, np.inf, np.inf),
            "wgs84_crs",
            iris.load_cube(
                _shared_utils.get_data_path(
                    ("NetCDF", "transverse_mercator", "tmean_1910_1910.nc")
                )
            )
            .coord_system()
            .as_cartopy_projection(),
        )
    ],
    indirect=["wgs84_crs"],
)
def test_transform_geometry_pyproj_network_edgecase(
    input_geometry, wgs84_crs, input_cube_crs
):
    # Simulate an edge case where PROJ network access is required to perform
    # a transformation but fails due to network related issues.
    # In this case pyproj will return a transformed geometry with Inf coordinates.
    # Check that this rasises an error with an appropriate message.
    # Note that in this case we supply an input geometry with Inf coordinates directly
    # to simulate the failed transformation as we cannot reliably simulate network
    # issues in a test environment.

    err_message = (
        "Error transforming geometry: geometry contains Inf coordinates.  This is likely due to a failed CRS transformation."
        "\nFailed transforms are often caused by network issues, often due to incorrectly configured SSL certificate paths."
    )
    with pytest.raises(ValueError, match=err_message):
        _transform_geometry(
            geometry=input_geometry, geometry_crs=wgs84_crs, cube_crs=input_cube_crs
        )
