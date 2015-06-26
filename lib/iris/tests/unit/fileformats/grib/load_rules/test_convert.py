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
"""Unit tests for :func:`iris.fileformats.grib.load_rules.convert`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import gribapi

import iris
from iris.fileformats.rules import Reference
from iris.tests import mock
from iris.tests.test_grib_load import TestGribSimple
from iris.tests.unit.fileformats import TestField
import iris.unit

from iris.fileformats.grib import GribWrapper
from iris.fileformats.grib.load_rules import convert


class Test_GribLevels_Mock(TestGribSimple):
    # Unit test levels with mocking.
    def test_grib2_height(self):
        grib = self.mock_grib()
        grib.edition = 2
        grib.typeOfFirstFixedSurface = 103
        grib.scaledValueOfFirstFixedSurface = 12345
        grib.scaleFactorOfFirstFixedSurface = 0
        grib.typeOfSecondFixedSurface = 255
        cube = self.cube_from_message(grib)
        self.assertEqual(
            cube.coord('height'),
            iris.coords.DimCoord(12345, standard_name="height", units="m"))

    def test_grib2_bounded_height(self):
        grib = self.mock_grib()
        grib.edition = 2
        grib.typeOfFirstFixedSurface = 103
        grib.scaledValueOfFirstFixedSurface = 12345
        grib.scaleFactorOfFirstFixedSurface = 0
        grib.typeOfSecondFixedSurface = 103
        grib.scaledValueOfSecondFixedSurface = 54321
        grib.scaleFactorOfSecondFixedSurface = 0
        cube = self.cube_from_message(grib)
        self.assertEqual(
            cube.coord('height'),
            iris.coords.DimCoord(33333, standard_name="height", units="m",
                                 bounds=[[12345, 54321]]))

    def test_grib2_diff_bound_types(self):
        grib = self.mock_grib()
        grib.edition = 2
        grib.typeOfFirstFixedSurface = 103
        grib.scaledValueOfFirstFixedSurface = 12345
        grib.scaleFactorOfFirstFixedSurface = 0
        grib.typeOfSecondFixedSurface = 102
        grib.scaledValueOfSecondFixedSurface = 54321
        grib.scaleFactorOfSecondFixedSurface = 0
        with mock.patch('warnings.warn') as warn:
            cube = self.cube_from_message(grib)
        warn.assert_called_with(
            "Different vertical bound types not yet handled.")


class TestBoundedTime(TestField):
    @staticmethod
    def is_forecast_period(coord):
        return (coord.standard_name == 'forecast_period' and
                coord.units == 'hours')

    @staticmethod
    def is_time(coord):
        return (coord.standard_name == 'time' and
                coord.units == 'hours since epoch')

    def assert_bounded_message(self, **kwargs):
        attributes = {'productDefinitionTemplateNumber': 0,
                      'edition': 1, '_forecastTime': 15,
                      '_forecastTimeUnit': 'hours',
                      'phenomenon_bounds': lambda u: (80, 120),
                      '_phenomenonDateTime': -1,
                      'table2Version': 9999}
        attributes.update(kwargs)
        message = mock.Mock(**attributes)
        self._test_for_coord(message, convert, self.is_forecast_period,
                             expected_points=[35],
                             expected_bounds=[[15, 55]])
        self._test_for_coord(message, convert, self.is_time,
                             expected_points=[100],
                             expected_bounds=[[80, 120]])

    def test_time_range_indicator_2(self):
        self.assert_bounded_message(timeRangeIndicator=2)

    def test_time_range_indicator_3(self):
        self.assert_bounded_message(timeRangeIndicator=3)

    def test_time_range_indicator_4(self):
        self.assert_bounded_message(timeRangeIndicator=4)

    def test_time_range_indicator_5(self):
        self.assert_bounded_message(timeRangeIndicator=5)

    def test_time_range_indicator_51(self):
        self.assert_bounded_message(timeRangeIndicator=51)

    def test_time_range_indicator_113(self):
        self.assert_bounded_message(timeRangeIndicator=113)

    def test_time_range_indicator_114(self):
        self.assert_bounded_message(timeRangeIndicator=114)

    def test_time_range_indicator_115(self):
        self.assert_bounded_message(timeRangeIndicator=115)

    def test_time_range_indicator_116(self):
        self.assert_bounded_message(timeRangeIndicator=116)

    def test_time_range_indicator_117(self):
        self.assert_bounded_message(timeRangeIndicator=117)

    def test_time_range_indicator_118(self):
        self.assert_bounded_message(timeRangeIndicator=118)

    def test_time_range_indicator_123(self):
        self.assert_bounded_message(timeRangeIndicator=123)

    def test_time_range_indicator_124(self):
        self.assert_bounded_message(timeRangeIndicator=124)

    def test_time_range_indicator_125(self):
        self.assert_bounded_message(timeRangeIndicator=125)

    def test_product_template_8(self):
        self.assert_bounded_message(edition=2,
                                    productDefinitionTemplateNumber=8)

    def test_product_template_9(self):
        self.assert_bounded_message(edition=2,
                                    productDefinitionTemplateNumber=9)


class Test_GribLevels(tests.IrisTest):
    def test_grib1_hybrid_height(self):
        gm = gribapi.grib_new_from_samples('regular_gg_ml_grib1')
        gw = GribWrapper(gm)
        results = convert(gw)

        factory, = results[0]
        self.assertEqual(factory.factory_class,
                         iris.aux_factory.HybridPressureFactory)
        delta, sigma, ref = factory.args
        self.assertEqual(delta, {'long_name': 'level_pressure'})
        self.assertEqual(sigma, {'long_name': 'sigma'})
        self.assertEqual(ref, Reference(name='surface_pressure'))

        ml_ref = iris.coords.CoordDefn('model_level_number', None, None,
                                       iris.unit.Unit('1'),
                                       {'positive': 'up'}, None)
        lp_ref = iris.coords.CoordDefn(None, 'level_pressure', None,
                                       iris.unit.Unit('Pa'),
                                       {}, None)
        s_ref = iris.coords.CoordDefn(None, 'sigma', None,
                                      iris.unit.Unit('1'),
                                      {}, None)

        aux_coord_defns = [coord._as_defn() for coord, dim in results[8]]
        self.assertIn(ml_ref, aux_coord_defns)
        self.assertIn(lp_ref, aux_coord_defns)
        self.assertIn(s_ref, aux_coord_defns)


if __name__ == "__main__":
    tests.main()
