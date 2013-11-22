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

import mock

from iris.coords import DimCoord
from iris.tests.test_grib_load import TestGribSimple


class Test_GribLevels(TestGribSimple):
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


if __name__ == "__main__":
    tests.main()
