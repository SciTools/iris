import math

import cartopy.io.shapereader as shpreader
import numpy as np

import iris._shapefiles
import iris.tests as tests
from iris.util import apply_shapefile


class TestCubeMasking(tests.IrisTest):
    ne_countries = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )
    reader = shpreader.Reader(ne_countries)

    def testGlobal(self):
        test_global = iris.load_cube(
            "/net/home/h05/achamber/git/iris/lib/iris/test_data/iris-test-data/test_data/NetCDF/global/xyt/SMALL_hires_wind_u_for_ipcc4.nc"
        )
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
        test_rotated = iris.load_cube(
            "/net/home/h05/achamber/git/iris/lib/iris/test_data/iris-test-data/test_data/NetCDF/rotated/xy/rotPole_landAreaFraction.nc"
        )
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
        test_transverse = iris.load_cube(
            "/net/home/h05/achamber/git/iris/lib/iris/test_data/iris-test-data/test_data/NetCDF/transverse_mercator/tmean_1910_1910.nc"
        )
        ne_uk = [
            country.geometry
            for country in self.reader.records()
            if "United Kingdom" in country.attributes["NAME_LONG"]
        ][0]
        masked_test = apply_shapefile(ne_uk, test_transverse)
        assert math.isclose(
            np.sum(masked_test.data), 90740.25, rel_tol=0.00001
        ), "transverse mercator UK data with UK mask failed test"
