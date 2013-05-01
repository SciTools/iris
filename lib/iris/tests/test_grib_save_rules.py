# (C) British Crown Copyright 2010 - 2013, Met Office
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
"""Unit tests for iris.fileformats.grib_save_rules"""

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import gribapi
import numpy as np
import numpy.ma as ma
import mock

import iris.cube
import iris.coords
import iris.fileformats.grib_save_rules as grib_save_rules


class Test_non_hybrid_surfaces(tests.IrisTest):
    # Test grib_save_rules.non_hybrid_surfaces()
    
    @mock.patch.object(gribapi, "grib_set_long")
    def test_altitude_point(self, mock_set_long):
        grib = None
        cube = iris.cube.Cube([1,2,3,4,5]) 
        cube.add_aux_coord(iris.coords.AuxCoord([12345], "altitude", units="m"))

        grib_save_rules.non_hybrid_surfaces(cube, grib)

        mock_set_long.assert_any_call(grib, "typeOfFirstFixedSurface", 102)
        mock_set_long.assert_any_call(grib, "scaleFactorOfFirstFixedSurface", 0)
        mock_set_long.assert_any_call(grib, "scaledValueOfFirstFixedSurface", 12345)
        mock_set_long.assert_any_call(grib, "typeOfSecondFixedSurface", -1)
        mock_set_long.assert_any_call(grib, "scaleFactorOfSecondFixedSurface", 255)
        mock_set_long.assert_any_call(grib, "scaledValueOfSecondFixedSurface", -1)        
        
    @mock.patch.object(gribapi, "grib_set_long")
    def test_height_point(self, mock_set_long):
        grib = None
        cube = iris.cube.Cube([1,2,3,4,5]) 
        cube.add_aux_coord(iris.coords.AuxCoord([12345], "height", units="m"))

        grib_save_rules.non_hybrid_surfaces(cube, grib)

        mock_set_long.assert_any_call(grib, "typeOfFirstFixedSurface", 103)
        mock_set_long.assert_any_call(grib, "scaleFactorOfFirstFixedSurface", 0)
        mock_set_long.assert_any_call(grib, "scaledValueOfFirstFixedSurface", 12345)
        mock_set_long.assert_any_call(grib, "typeOfSecondFixedSurface", -1)
        mock_set_long.assert_any_call(grib, "scaleFactorOfSecondFixedSurface", 255)
        mock_set_long.assert_any_call(grib, "scaledValueOfSecondFixedSurface", -1)        


class Test_data(tests.IrisTest):
    # Test grib_save_rules.data()
    
    @mock.patch.object(gribapi, "grib_set_double_array")
    @mock.patch.object(gribapi, "grib_set_double")
    def test_masked_array(self, mock_set_double, grib_set_double_array):
        grib = None
        cube = iris.cube.Cube(ma.MaskedArray([1,2,3,4,5], fill_value=54321)) 

        grib_save_rules.data(cube, grib)

        mock_set_double.assert_any_call(grib, "missingValue", float(54321))

    @mock.patch.object(gribapi, "grib_set_double_array")
    @mock.patch.object(gribapi, "grib_set_double")
    def test_numpy_array(self, mock_set_double, grib_set_double_array):
        grib = None
        cube = iris.cube.Cube(np.array([1,2,3,4,5])) 

        grib_save_rules.data(cube, grib)

        mock_set_double.assert_any_call(grib, "missingValue", float(-1e9))


if __name__ == "__main__":
    tests.main()
