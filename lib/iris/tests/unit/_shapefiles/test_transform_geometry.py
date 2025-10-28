# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris._shapefiles._transform_geometry`."""

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
    # Assert that all invalid geometries raise the expected error
    out_geometry = _transform_geometry(
        geometry=input_geometry, geometry_crs=wgs84_crs, cube_crs=input_cube_crs
    )
    assert isinstance(out_geometry, shapely.geometry.base.BaseGeometry)
    assert output_expected_geometry == out_geometry


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
