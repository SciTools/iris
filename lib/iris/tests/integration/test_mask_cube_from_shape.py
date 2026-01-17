# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for :func:`iris.util.mask_cube_from_shape`."""

import cartopy.io.shapereader as shpreader
import numpy as np
from pyproj import CRS
import pytest
from pytest import approx
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point

import iris
from iris._deprecation import IrisDeprecation
from iris.coord_systems import GeogCS
import iris.tests as tests
from iris.util import mask_cube_from_shape, mask_cube_from_shapefile
from iris.warnings import IrisUserWarning


@pytest.fixture
def wgs84_crs():
    return CRS.from_epsg(4326)


@pytest.fixture
def shp_reader():
    ne_countries = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )
    return shpreader.Reader(ne_countries)


@pytest.mark.parametrize(
    ("minimum_weight", "all_touched", "invert", "expected_sum"),
    [
        (0.0, None, None, 10522684.77),  # Minimum weight == 0
        (0.0, None, False, 10522684.77),  # Minimum weight == 0
        (0.0, True, False, 10522684.77),  # All touched == True
        (0.5, None, False, 8965584.47),  # Minimum weight == 0.5
        (1.0, None, False, 7504361.29),  # Minimum weight == 1
        (0.0, False, False, 8953582.05),  # All touched == False
        (0.0, True, True, 605637718.12),  # All touched == True, Invert == True
    ],
)
def test_global_proj_china(
    minimum_weight, all_touched, invert, expected_sum, shp_reader, wgs84_crs
):
    """Test masking with a shape for China with various parameter combinations."""
    path = tests.get_data_path(["NetCDF", "global", "xyt", "SMALL_total_column_co2.nc"])
    test_global = iris.load_cube(path)
    test_global.coord("latitude").coord_system = GeogCS(6371229)
    test_global.coord("longitude").coord_system = GeogCS(6371229)

    ne_china = [
        country.geometry
        for country in shp_reader.records()
        if "China" in country.attributes["NAME_LONG"]
    ][0]
    masked_test = mask_cube_from_shape(
        test_global,
        ne_china,
        shape_crs=wgs84_crs,
        minimum_weight=minimum_weight,
        all_touched=all_touched,
        invert=invert,
    )
    assert masked_test.ndim == 3
    assert approx(np.sum(masked_test.data), rel=0.001) == expected_sum


def test_global_proj_russia(shp_reader, wgs84_crs):
    """Test masking with a shape that crosses the antimeridian."""
    path = tests.get_data_path(["NetCDF", "global", "xyt", "SMALL_total_column_co2.nc"])
    test_global = iris.load_cube(path)
    test_global.coord("latitude").coord_system = GeogCS(6371229)
    test_global.coord("longitude").coord_system = GeogCS(6371229)
    ne_russia = [
        country.geometry
        for country in shp_reader.records()
        if "Russia" in country.attributes["NAME_LONG"]
    ][0]

    with pytest.warns(
        IrisUserWarning,
        match=(
            "Geometry crossing the antimeridian is not supported. "
            "Cannot verify non-crossing given current geometry bounds."
        ),
    ):
        _ = mask_cube_from_shape(test_global, ne_russia, shape_crs=wgs84_crs)


def test_rotated_pole_proj_uk(shp_reader, wgs84_crs):
    """Test masking a rotated pole projection cube for the UK with lat/lon shape."""
    path = tests.get_data_path(
        ["NetCDF", "rotated", "xy", "rotPole_landAreaFraction.nc"]
    )
    test_rotated = iris.load_cube(path)
    ne_uk = [
        country.geometry
        for country in shp_reader.records()
        if "United Kingdom" in country.attributes["NAME_LONG"]
    ][0]
    masked_test = mask_cube_from_shape(test_rotated, ne_uk, shape_crs=wgs84_crs)
    assert masked_test.ndim == 2
    assert approx(np.sum(masked_test.data), rel=0.001) == 102.77


def test_transverse_mercator_proj_uk(shp_reader, wgs84_crs):
    """Test masking a transverse mercator projection cube for the UK with lat/lon shape."""
    path = tests.get_data_path(["NetCDF", "transverse_mercator", "tmean_1910_1910.nc"])
    test_transverse = iris.load_cube(path)
    ne_uk = [
        country.geometry
        for country in shp_reader.records()
        if "United Kingdom" in country.attributes["NAME_LONG"]
    ][0]
    masked_test = mask_cube_from_shape(test_transverse, ne_uk, shape_crs=wgs84_crs)
    assert masked_test.ndim == 3
    assert approx(np.sum(masked_test.data), rel=0.001) == 90740.25


def test_rotated_pole_proj_germany_weighted_area(shp_reader, wgs84_crs):
    """Test masking a rotated pole projection cube for Germany with weighted area."""
    path = tests.get_data_path(
        ["NetCDF", "rotated", "xy", "rotPole_landAreaFraction.nc"]
    )
    test_rotated = iris.load_cube(path)
    ne_germany = [
        country.geometry
        for country in shp_reader.records()
        if "Germany" in country.attributes["NAME_LONG"]
    ][0]
    masked_test = mask_cube_from_shape(
        test_rotated, ne_germany, shape_crs=wgs84_crs, minimum_weight=0.9
    )
    assert masked_test.ndim == 2
    assert approx(np.sum(masked_test.data), rel=0.001) == 125.60199


def test_4d_global_proj_brazil(shp_reader, wgs84_crs):
    """Test masking a 4D global projection cube for Brazil with lat/lon shape."""
    path = tests.get_data_path(["NetCDF", "global", "xyz_t", "GEMS_CO2_Apr2006.nc"])
    test_4d_brazil = iris.load_cube(path, "Carbon Dioxide")
    test_4d_brazil.coord("latitude").coord_system = GeogCS(6371229)
    test_4d_brazil.coord("longitude").coord_system = GeogCS(6371229)
    ne_brazil = [
        country.geometry
        for country in shp_reader.records()
        if "Brazil" in country.attributes["NAME_LONG"]
    ][0]
    masked_test = mask_cube_from_shape(
        test_4d_brazil, ne_brazil, shape_crs=wgs84_crs, all_touched=True
    )
    assert masked_test.ndim == 4
    assert approx(np.sum(masked_test.data), rel=0.001) == 18616921.2


@pytest.mark.parametrize(
    # `expected_value` entries are pre-calculated Known Good Outputs.
    ("shape", "expected_value"),
    [
        (Point(-3.475446894622651, 50.72770791320487), 12061.74),  # (x,y)
        (
            LineString(
                [
                    (-5.712431305030631, 50.06590599588483),
                    (-3.0704940433528947, 58.644091639685456),
                ]
            ),
            120530.41,
        ),  # (x,y) to (x,y)
        (
            MultiPoint(
                [
                    (-5.712431305030631, 50.06590599588483),
                    (-3.0704940433528947, 58.644091639685456),
                ]
            ),
            24097.47,
        ),
        (
            MultiLineString(
                [
                    [
                        (-5.206405826948041, 49.95891620303525),
                        (-3.376975634580173, 58.67197421392852),
                    ],
                    [
                        (-6.2276386132877475, 56.71561805509071),
                        (1.7626540441873777, 52.48118683241357),
                    ],
                ]
            ),
            253248.44,
        ),
    ],
)
def test_global_proj_uk_shapes(shape, expected_value, wgs84_crs):
    """Test masking with a variety of shape types."""
    path = tests.get_data_path(["NetCDF", "global", "xyt", "SMALL_total_column_co2.nc"])
    test_global = iris.load_cube(path)
    test_global.coord("latitude").coord_system = GeogCS(6371229)
    test_global.coord("longitude").coord_system = GeogCS(6371229)
    masked_test = mask_cube_from_shape(
        test_global,
        shape,
        shape_crs=wgs84_crs,
    )
    assert masked_test.ndim == 3
    assert approx(np.sum(masked_test.data), rel=0.001) == expected_value


def test_mask_cube_from_shapefile_depreciation(shp_reader):
    """Test that the mask_cube_from_shapefile function raises a deprecation warning."""
    path = tests.get_data_path(["NetCDF", "global", "xyt", "SMALL_total_column_co2.nc"])
    test_global = iris.load_cube(path)
    test_global.coord("latitude").coord_system = GeogCS(6371229)
    test_global.coord("longitude").coord_system = GeogCS(6371229)
    ne_china = [
        country.geometry
        for country in shp_reader.records()
        if "China" in country.attributes["NAME_LONG"]
    ][0]

    with pytest.warns(
        IrisDeprecation,
        match=(
            "iris.util.mask_cube_from_shapefile has been deprecated, and will be removed in a "
            "future release. Please use iris.util.mask_cube_from_shape instead."
        ),
    ):
        mask_cube_from_shapefile(test_global, ne_china)
