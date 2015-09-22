# (C) British Crown Copyright 2013 - 2015, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
'''
Created on Apr 26, 2013

@author: itpp
'''

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import iris.unit

if tests.GRIB_AVAILABLE:
    import gribapi
    import iris.fileformats.grib.grib_phenom_translation as gptx


@tests.skip_grib
class TestGribLookupTableType(tests.IrisTest):
    def test_lookuptable_type(self):
        ll = gptx.LookupTable([('a', 1), ('b', 2)])
        assert ll['a'] == 1
        assert ll['q'] is None
        ll['q'] = 15
        assert ll['q'] == 15
        ll['q'] = 15
        assert ll['q'] == 15
        with self.assertRaises(KeyError):
            ll['q'] = 7
        del ll['q']
        ll['q'] = 7
        assert ll['q'] == 7


@tests.skip_grib
class TestGribPhenomenonLookup(tests.IrisTest):
    def test_grib1_cf_lookup(self):
        def check_grib1_cf(param,
                           standard_name, long_name, units,
                           height=None,
                           t2version=128, centre=98, expect_none=False):
            iris_units = iris.unit.Unit(units)
            cfdata = gptx.grib1_phenom_to_cf_info(param_number=param,
                                                  table2_version=t2version,
                                                  centre_number=centre)
            if expect_none:
                self.assertIsNone(cfdata)
            else:
                self.assertEqual(cfdata.standard_name, standard_name)
                self.assertEqual(cfdata.long_name, long_name)
                self.assertEqual(cfdata.units, iris_units)
                if height is None:
                    self.assertIsNone(cfdata.set_height)
                else:
                    self.assertEqual(cfdata.set_height, float(height))

        check_grib1_cf(165, 'x_wind', None, 'm s-1', 10.0)
        check_grib1_cf(168, 'dew_point_temperature', None, 'K', 2)
        check_grib1_cf(130, 'air_temperature', None, 'K')
        check_grib1_cf(235, None, "grib_skin_temperature", "K")
        check_grib1_cf(235, None, "grib_skin_temperature", "K",
                       t2version=9999, expect_none=True)
        check_grib1_cf(235, None, "grib_skin_temperature", "K",
                       centre=9999, expect_none=True)
        check_grib1_cf(9999, None, "grib_skin_temperature", "K",
                       expect_none=True)

    def test_grib2_cf_lookup(self):
        def check_grib2_cf(discipline, category, number,
                           standard_name, long_name, units,
                           expect_none=False):
            iris_units = iris.unit.Unit(units)
            cfdata = gptx.grib2_phenom_to_cf_info(param_discipline=discipline,
                                                  param_category=category,
                                                  param_number=number)
            if expect_none:
                self.assertIsNone(cfdata)
            else:
                self.assertEqual(cfdata.standard_name, standard_name)
                self.assertEqual(cfdata.long_name, long_name)
                self.assertEqual(cfdata.units, iris_units)

        # These should work
        check_grib2_cf(0, 0, 2, "air_potential_temperature", None, "K")
        check_grib2_cf(0, 19, 1, None, "grib_physical_atmosphere_albedo", "%")
        check_grib2_cf(2, 0, 2, "soil_temperature", None, "K")
        check_grib2_cf(10, 2, 0, "sea_ice_area_fraction", None, 1)
        check_grib2_cf(2, 0, 0, "land_area_fraction", None, 1)
        check_grib2_cf(0, 19, 1, None, "grib_physical_atmosphere_albedo", "%")
        check_grib2_cf(0, 1, 64,
                       "atmosphere_mass_content_of_water_vapor", None,
                       "kg m-2")
        check_grib2_cf(2, 0, 7, "surface_altitude", None, "m")

        # These should fail
        check_grib2_cf(9999, 2, 0, "sea_ice_area_fraction", None, 1,
                       expect_none=True)
        check_grib2_cf(10, 9999, 0, "sea_ice_area_fraction", None, 1,
                       expect_none=True)
        check_grib2_cf(10, 2, 9999, "sea_ice_area_fraction", None, 1,
                       expect_none=True)

    def test_cf_grib2_lookup(self):
        def check_cf_grib2(standard_name, long_name,
                           discipline, category, number, units,
                           expect_none=False):
            iris_units = iris.unit.Unit(units)
            gribdata = gptx.cf_phenom_to_grib2_info(standard_name, long_name)
            if expect_none:
                self.assertIsNone(gribdata)
            else:
                self.assertEqual(gribdata.discipline, discipline)
                self.assertEqual(gribdata.category, category)
                self.assertEqual(gribdata.number, number)
                self.assertEqual(gribdata.units, iris_units)

        # These should work
        check_cf_grib2("sea_surface_temperature", None,
                       10, 3, 0, 'K')
        check_cf_grib2("air_temperature", None,
                       0, 0, 0, 'K')
        check_cf_grib2("soil_temperature", None,
                       2, 0, 2, "K")
        check_cf_grib2("land_area_fraction", None,
                       2, 0, 0, '1')
        check_cf_grib2("land_binary_mask", None,
                       2, 0, 0, '1')
        check_cf_grib2("atmosphere_mass_content_of_water_vapor", None,
                       0, 1, 64, "kg m-2")
        check_cf_grib2("surface_altitude", None,
                       2, 0, 7, "m")

        # These should fail
        check_cf_grib2("air_temperature", "user_long_UNRECOGNISED",
                       0, 0, 0, 'K')
        check_cf_grib2("air_temperature_UNRECOGNISED", None,
                       0, 0, 0, 'K',
                       expect_none=True)
        check_cf_grib2(None, "user_long_UNRECOGNISED",
                       0, 0, 0, 'K',
                       expect_none=True)
        check_cf_grib2(None, "precipitable_water",
                       0, 1, 3, 'kg m-2')
        check_cf_grib2("invalid_unknown", "precipitable_water",
                       0, 1, 3, 'kg m-2',
                       expect_none=True)
        check_cf_grib2(None, None, 0, 0, 0, '',
                       expect_none=True)


if __name__ == '__main__':
    tests.main()
