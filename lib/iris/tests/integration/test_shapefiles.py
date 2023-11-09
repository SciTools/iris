import math

import cartopy.io.shapereader as shpreader
import numpy as np

import iris
import iris.tests as tests
from iris.util import apply_shapefile


@tests.skip_data
class TestCubeMasking(tests.IrisTest):
    ne_countries = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )
    reader = shpreader.Reader(ne_countries)

    def testGlobal(self):
        path = tests.get_data_path(
            ["NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc"]
        )
        test_global = iris.load_cube(path)
        ne_russia = [
            country.geometry
            for country in self.reader.records()
            if "Russia" in country.attributes["NAME_LONG"]
        ][0]
        masked_test = apply_shapefile(ne_russia, test_global)
        print(np.sum(masked_test.data))
        assert math.isclose(
            np.sum(masked_test.data), 76845.37, rel_tol=0.00001
        ), "Global data with Russia mask failed test"

    def testRotated(self):
        path = tests.get_data_path(
            ["NetCDF", "rotated", "xy", "rotPole_landAreaFraction.nc"]
        )
        test_rotated = iris.load_cube(path)
        ne_germany = [
            country.geometry
            for country in self.reader.records()
            if "Germany" in country.attributes["NAME_LONG"]
        ][0]
        masked_test = apply_shapefile(ne_germany, test_rotated)
        assert math.isclose(
            np.sum(masked_test.data), 179.46872, rel_tol=0.00001
        ), "rotated europe data with German mask failed test"

    def testTransverseMercator(self):
        path = tests.get_data_path(
            ["NetCDF", "transverse_mercator", "tmean_1910_1910.nc"]
        )
        test_transverse = iris.load_cube(path)
        ne_uk = [
            country.geometry
            for country in self.reader.records()
            if "United Kingdom" in country.attributes["NAME_LONG"]
        ][0]
        masked_test = apply_shapefile(ne_uk, test_transverse)
        assert math.isclose(
            np.sum(masked_test.data), 90740.25, rel_tol=0.00001
        ), "transverse mercator UK data with UK mask failed test"

    def testRotatedWeighted(self):
        path = tests.get_data_path(
            ["NetCDF", "rotated", "xy", "rotPole_landAreaFraction.nc"]
        )
        test_rotated = iris.load_cube(path)
        ne_germany = [
            country.geometry
            for country in self.reader.records()
            if "Germany" in country.attributes["NAME_LONG"]
        ][0]
        masked_test = apply_shapefile(
            ne_germany, test_rotated, minimum_weight=0.9
        )
        assert math.isclose(
            np.sum(masked_test.data), 125.60199, rel_tol=0.00001
        ), "rotated europe data with 0.9 weight germany mask failed test"
