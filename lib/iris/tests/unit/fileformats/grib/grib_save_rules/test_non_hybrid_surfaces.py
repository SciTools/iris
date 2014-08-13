# (C) British Crown Copyright 2013 - 2014, Met Office
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
"""Unit tests for module-level functions."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import gribapi
import numpy as np

import iris
import iris.cube
import iris.coords
import iris.fileformats.grib.grib_save_rules as grib_save_rules


class Test_non_hybrid_surfaces(tests.IrisTest):
    def test_bounded_altitude_feet(self):
        cube = iris.cube.Cube([0])
        cube.add_aux_coord(iris.coords.AuxCoord(
            1500.0, long_name='altitude', units='ft',
            bounds=np.array([1000.0, 2000.0])))
        grib = gribapi.grib_new_from_samples("GRIB2")
        grib_save_rules.non_hybrid_surfaces(cube, grib)
        self.assertEqual(
            gribapi.grib_get_long(grib, "typeOfFirstFixedSurface"),
            102)
        self.assertEqual(
            gribapi.grib_get_long(grib, "scaleFactorOfFirstFixedSurface"),
            1)
        self.assertEqual(
            gribapi.grib_get_long(grib, "scaledValueOfFirstFixedSurface"),
            3048)
        self.assertEqual(
            gribapi.grib_get_long(grib, "typeOfSecondFixedSurface"),
            102)
        self.assertEqual(
            gribapi.grib_get_long(grib, "scaleFactorOfSecondFixedSurface"),
            1)
        self.assertEqual(
            gribapi.grib_get_long(grib, "scaledValueOfSecondFixedSurface"),
            6096)

    def test_unbounded_height(self):
        cube = iris.cube.Cube([0])
        cube.add_aux_coord(iris.coords.AuxCoord([1.5], standard_name='height',
                                                units='m'))
        grib = gribapi.grib_new_from_samples("GRIB2")
        grib_save_rules.non_hybrid_surfaces(cube, grib)
        self.assertEqual(
            gribapi.grib_get_long(grib, "typeOfFirstFixedSurface"),
            103)
        self.assertEqual(
            gribapi.grib_get_long(grib, "scaleFactorOfFirstFixedSurface"),
            1)
        self.assertEqual(
            gribapi.grib_get_long(grib, "scaledValueOfFirstFixedSurface"),
            15)
        self.assertEqual(
            gribapi.grib_get_long(grib, "typeOfSecondFixedSurface"),
            0xff)
        self.assertEqual(
            gribapi.grib_get_long(grib, "scaleFactorOfSecondFixedSurface"),
            0xffffffff)
        self.assertEqual(
            gribapi.grib_get_long(grib, "scaledValueOfSecondFixedSurface"),
            0xffffffff)


if __name__ == "__main__":
    tests.main()
