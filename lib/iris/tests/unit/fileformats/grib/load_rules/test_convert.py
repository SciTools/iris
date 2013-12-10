# (C) British Crown Copyright 2013, Met Office
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

# Import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import gribapi
import mock

from iris.aux_factory import HybridPressureFactory
from iris.coords import DimCoord, CoordDefn
import iris.fileformats.grib
import iris.fileformats.grib.load_rules
from iris.fileformats.rules import Reference
from iris.tests.test_grib_load import TestGribSimple
from iris.unit import Unit


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
            DimCoord(12345, standard_name="height", units="m"))

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
            DimCoord(33333, standard_name="height", units="m",
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


class Test_GribLevels(tests.IrisTest):
    def test_grib1_hybrid_height(self):
        gm = gribapi.grib_new_from_samples('regular_gg_ml_grib1')
        gw = iris.fileformats.grib.GribWrapper(gm)
        results = iris.fileformats.grib.load_rules.convert(gw)

        factories = results[0]
        self.assertEqual(factories[0].factory_class, HybridPressureFactory)
        self.assertIn({'long_name': 'level_pressure'}, factories[0].args)
        self.assertIn({'long_name': 'sigma'}, factories[0].args)
        self.assertIn(Reference(name='surface_pressure'), factories[0].args)

        ml_ref = CoordDefn('model_level_number', None, None, Unit('1'),
                        {'positive': 'up'}, None)
        lp_ref = CoordDefn(None, 'level_pressure', None, Unit('Pa'), {}, None)
        s_ref = CoordDefn(None, 'sigma', None, Unit('1'), {}, None)

        aux_coord_defns = [coord._as_defn() for coord, dim in results[8]]
        self.assertIn(ml_ref, aux_coord_defns)
        self.assertIn(lp_ref, aux_coord_defns)
        self.assertIn(s_ref, aux_coord_defns)
        

if __name__ == "__main__":
    tests.main()
