# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for :func:`iris.util.mask_cube_from_shapefile`."""

import math

import cartopy.io.shapereader as shpreader
import numpy as np
from pyproj import CRS
from pytest import approx

import iris
import iris.tests as tests
from iris.util import mask_cube_from_shapefile


@tests.skip_data
class TestCubeMasking(tests.IrisTest):
    """integration tests of mask_cube_from_shapefile
    using different projections in iris_test_data -
    values are the KGO calculated using ASCEND.
    """

    def setUp(self):
        ne_countries = shpreader.natural_earth(
            resolution="10m", category="cultural", name="admin_0_countries"
        )
        self.reader = shpreader.Reader(ne_countries)
        self.wgs84 = CRS.from_epsg(4326)

    def test_global_proj_russia(self):
        path = tests.get_data_path(
            ["NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc"]
        )
        test_global = iris.load_cube(path)
        ne_russia = [
            country.geometry
            for country in self.reader.records()
            if "Russia" in country.attributes["NAME_LONG"]
        ][0]
        masked_test = mask_cube_from_shapefile(
            test_global, ne_russia, shape_crs=self.wgs84
        )
        print(np.sum(masked_test.data))
        assert 76845.37 == approx(np.sum(masked_test.data), rel=0.001)

    def test_rotated_pole_proj_germany(self):
        path = tests.get_data_path(
            ["NetCDF", "rotated", "xy", "rotPole_landAreaFraction.nc"]
        )
        test_rotated = iris.load_cube(path)
        ne_germany = [
            country.geometry
            for country in self.reader.records()
            if "Germany" in country.attributes["NAME_LONG"]
        ][0]
        masked_test = mask_cube_from_shapefile(
            test_rotated, ne_germany, shape_crs=self.wgs84
        )
        assert 179.46872 == approx(np.sum(masked_test.data), rel=0.001)

    def test_transverse_mercator_proj_uk(self):
        path = tests.get_data_path(
            ["NetCDF", "transverse_mercator", "tmean_1910_1910.nc"]
        )
        test_transverse = iris.load_cube(path)
        ne_uk = [
            country.geometry
            for country in self.reader.records()
            if "United Kingdom" in country.attributes["NAME_LONG"]
        ][0]
        masked_test = mask_cube_from_shapefile(
            test_transverse, ne_uk, shape_crs=self.wgs84
        )
        assert 90740.25 == approx(np.sum(masked_test.data), rel=0.001)

    def test_rotated_pole_proj_germany_weighted_area(self):
        path = tests.get_data_path(
            ["NetCDF", "rotated", "xy", "rotPole_landAreaFraction.nc"]
        )
        test_rotated = iris.load_cube(path)
        ne_germany = [
            country.geometry
            for country in self.reader.records()
            if "Germany" in country.attributes["NAME_LONG"]
        ][0]
        masked_test = mask_cube_from_shapefile(
            test_rotated, ne_germany, shape_crs=self.wgs84, minimum_weight=0.9
        )
        assert 125.60199 == approx(np.sum(masked_test.data), rel=0.001)

    def test_4d_global_proj_brazil(self):
        path = tests.get_data_path(["NetCDF", "global", "xyz_t", "GEMS_CO2_Apr2006.nc"])
        test_4d_brazil = iris.load_cube(path, "Carbon Dioxide")
        ne_brazil = [
            country.geometry
            for country in self.reader.records()
            if "Brazil" in country.attributes["NAME_LONG"]
        ][0]
        masked_test = mask_cube_from_shapefile(
            test_4d_brazil, ne_brazil, shape_crs=self.wgs84
        )
        assert 18616921.2 == approx(np.sum(masked_test.data), rel=0.001)
