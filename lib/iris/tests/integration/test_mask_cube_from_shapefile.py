# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for :func:`iris.util.mask_cube_from_shapefile`."""

import math

import cartopy.io.shapereader as shpreader
import numpy as np

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
        masked_test = mask_cube_from_shapefile(test_global, ne_russia)
        print(np.sum(masked_test.data))
        assert math.isclose(
            np.sum(masked_test.data), 76845.37, rel_tol=0.001
        ), "Global data with Russia mask failed test"

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
        masked_test = mask_cube_from_shapefile(test_rotated, ne_germany)
        assert math.isclose(
            np.sum(masked_test.data), 179.46872, rel_tol=0.001
        ), "rotated europe data with German mask failed test"

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
        masked_test = mask_cube_from_shapefile(test_transverse, ne_uk)
        assert math.isclose(
            np.sum(masked_test.data), 90740.25, rel_tol=0.001
        ), "transverse mercator UK data with UK mask failed test"

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
            test_rotated, ne_germany, minimum_weight=0.9
        )
        assert math.isclose(
            np.sum(masked_test.data), 125.60199, rel_tol=0.001
        ), "rotated europe data with 0.9 weight germany mask failed test"

    def test_4d_global_proj_brazil(self):
        path = tests.get_data_path(["NetCDF", "global", "xyz_t", "GEMS_CO2_Apr2006.nc"])
        test_4d_brazil = iris.load_cube(path, "Carbon Dioxide")
        ne_brazil = [
            country.geometry
            for country in self.reader.records()
            if "Brazil" in country.attributes["NAME_LONG"]
        ][0]
        masked_test = mask_cube_from_shapefile(
            test_4d_brazil,
            ne_brazil,
        )
        print(np.sum(masked_test.data))
        # breakpoint()
        assert math.isclose(
            np.sum(masked_test.data), 18616921.2, rel_tol=0.001
        ), "4d data with brazil mask failed test"
