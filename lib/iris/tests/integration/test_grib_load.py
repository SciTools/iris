# (C) British Crown Copyright 2010 - 2020, Met Office
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
"""
Integration tests for grib2 file loading.

This code used to be part of 'tests/test_grib_load.py', but these integration-
style tests have been split out of there.

The remainder of the old 'tests/test_grib_load.py' is now renamed as
'tests/test_grib_load_translations.py'.  Those tests are implementation-
specific, and target the module 'iris_grib'.

"""
from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import iris
import iris.exceptions
import iris.tests.stock
import iris.util
from unittest import skipIf

# Skip out some tests that fail now that grib edition 2 files no longer use
# the GribWrapper.
# TODO: either fix these problems, or remove the tests.
skip_irisgrib_fails = skipIf(True,
                             'Test(s) are not currently usable with the new '
                             'grib 2 loader.')


@tests.skip_data
@tests.skip_grib
class TestBasicLoad(tests.GraphicsTest):
    def test_load_rotated(self):
        cubes = iris.load(tests.get_data_path(('GRIB', 'rotated_uk',
                                               "uk_wrongparam.grib1")))
        self.assertCML(cubes, ("grib_load", "rotated.cml"))

    def test_load_time_bound(self):
        cubes = iris.load(tests.get_data_path(('GRIB', "time_processed",
                                               "time_bound.grib1")))
        self.assertCML(cubes, ("grib_load", "time_bound_grib1.cml"))

    def test_load_time_processed(self):
        cubes = iris.load(tests.get_data_path(('GRIB', "time_processed",
                                               "time_bound.grib2")))
        self.assertCML(cubes, ("grib_load", "time_bound_grib2.cml"))

    def test_load_3_layer(self):
        cubes = iris.load(tests.get_data_path(('GRIB', "3_layer_viz",
                                               "3_layer.grib2")))
        cubes = iris.cube.CubeList([cubes[1], cubes[0], cubes[2]])
        self.assertCML(cubes, ("grib_load", "3_layer.cml"))

    def test_load_masked(self):
        gribfile = tests.get_data_path(
            ('GRIB', 'missing_values', 'missing_values.grib2'))
        cubes = iris.load(gribfile)
        self.assertCML(cubes, ('grib_load', 'missing_values_grib2.cml'))

    @skip_irisgrib_fails
    def test_y_fastest(self):
        cubes = iris.load(tests.get_data_path(("GRIB", "y_fastest",
                                               "y_fast.grib2")))
        self.assertCML(cubes, ("grib_load", "y_fastest.cml"))

    def test_polar_stereo_grib1(self):
        cube = iris.load_cube(tests.get_data_path(
            ("GRIB", "polar_stereo", "ST4.2013052210.01h")))
        self.assertCML(cube, ("grib_load", "polar_stereo_grib1.cml"))

    def test_polar_stereo_grib2_grid_definition(self):
        cube = iris.load_cube(tests.get_data_path(
            ("GRIB", "polar_stereo",
             "CMC_glb_TMP_ISBL_1015_ps30km_2013052000_P006.grib2")))
        self.assertEqual(cube.shape, (200, 247))
        pxc = cube.coord('projection_x_coordinate')
        self.assertAlmostEqual(pxc.points.max(), 4769905.5125, places=4)
        self.assertAlmostEqual(pxc.points.min(), -2610094.4875, places=4)
        pyc = cube.coord('projection_y_coordinate')
        self.assertAlmostEqual(pyc.points.max(), -216.1459, places=4)
        self.assertAlmostEqual(pyc.points.min(), -5970216.1459, places=4)
        self.assertEqual(pyc.coord_system, pxc.coord_system)
        self.assertEqual(pyc.coord_system.grid_mapping_name, 'stereographic')
        self.assertEqual(pyc.coord_system.central_lat, 90.0)
        self.assertEqual(pyc.coord_system.central_lon, 249.0)
        self.assertEqual(pyc.coord_system.false_easting, 0.0)
        self.assertEqual(pyc.coord_system.false_northing, 0.0)
        self.assertEqual(pyc.coord_system.true_scale_lat, 60.0)

    def test_lambert_grib1(self):
        cube = iris.load_cube(tests.get_data_path(
            ("GRIB", "lambert", "lambert.grib1")))
        self.assertCML(cube, ("grib_load", "lambert_grib1.cml"))

    def test_lambert_grib2(self):
        cube = iris.load_cube(tests.get_data_path(
            ("GRIB", "lambert", "lambert.grib2")))
        self.assertCML(cube, ("grib_load", "lambert_grib2.cml"))

    def test_regular_gg_grib1(self):
        cube = iris.load_cube(tests.get_data_path(
            ('GRIB', 'gaussian', 'regular_gg.grib1')))
        self.assertCML(cube, ('grib_load', 'regular_gg_grib1.cml'))

    def test_regular_gg_grib2(self):
        cube = iris.load_cube(tests.get_data_path(
            ('GRIB', 'gaussian', 'regular_gg.grib2')))
        self.assertCML(cube, ('grib_load', 'regular_gg_grib2.cml'))

    def test_reduced_ll(self):
        cube = iris.load_cube(tests.get_data_path(
            ("GRIB", "reduced", "reduced_ll.grib1")))
        self.assertCML(cube, ("grib_load", "reduced_ll_grib1.cml"))

    def test_reduced_gg(self):
        cube = iris.load_cube(tests.get_data_path(
            ("GRIB", "reduced", "reduced_gg.grib2")))
        self.assertCML(cube, ("grib_load", "reduced_gg_grib2.cml"))


@tests.skip_data
@tests.skip_grib
class TestIjDirections(tests.GraphicsTest):
    @staticmethod
    def _old_compat_load(name):
        cube = iris.load(tests.get_data_path(('GRIB', 'ij_directions',
                                              name)))[0]
        return [cube]

    def test_ij_directions_ipos_jpos(self):
        cubes = self._old_compat_load("ipos_jpos.grib2")
        self.assertCML(cubes, ("grib_load", "ipos_jpos.cml"))

    def test_ij_directions_ipos_jneg(self):
        cubes = self._old_compat_load("ipos_jneg.grib2")
        self.assertCML(cubes, ("grib_load", "ipos_jneg.cml"))

    def test_ij_directions_ineg_jneg(self):
        cubes = self._old_compat_load("ineg_jneg.grib2")
        self.assertCML(cubes, ("grib_load", "ineg_jneg.cml"))

    def test_ij_directions_ineg_jpos(self):
        cubes = self._old_compat_load("ineg_jpos.grib2")
        self.assertCML(cubes, ("grib_load", "ineg_jpos.cml"))


@tests.skip_data
@tests.skip_grib
class TestShapeOfEarth(tests.GraphicsTest):
    @staticmethod
    def _old_compat_load(name):
        cube = iris.load(tests.get_data_path(('GRIB', 'shape_of_earth',
                                              name)))[0]
        return cube

    def test_shape_of_earth_basic(self):
        # pre-defined sphere
        cube = self._old_compat_load("0.grib2")
        self.assertCML(cube, ("grib_load", "earth_shape_0.cml"))

    def test_shape_of_earth_custom_1(self):
        # custom sphere
        cube = self._old_compat_load("1.grib2")
        self.assertCML(cube, ("grib_load", "earth_shape_1.cml"))

    def test_shape_of_earth_IAU65(self):
        # IAU65 oblate sphere
        cube = self._old_compat_load("2.grib2")
        self.assertCML(cube, ("grib_load", "earth_shape_2.cml"))

    def test_shape_of_earth_custom_3(self):
        # custom oblate spheroid (km)
        cube = self._old_compat_load("3.grib2")
        self.assertCML(cube, ("grib_load", "earth_shape_3.cml"))

    def test_shape_of_earth_IAG_GRS80(self):
        # IAG-GRS80 oblate spheroid
        cube = self._old_compat_load("4.grib2")
        self.assertCML(cube, ("grib_load", "earth_shape_4.cml"))

    def test_shape_of_earth_WGS84(self):
        # WGS84
        cube = self._old_compat_load("5.grib2")
        self.assertCML(cube, ("grib_load", "earth_shape_5.cml"))

    def test_shape_of_earth_pre_6(self):
        # pre-defined sphere
        cube = self._old_compat_load("6.grib2")
        self.assertCML(cube, ("grib_load", "earth_shape_6.cml"))

    def test_shape_of_earth_custom_7(self):
        # custom oblate spheroid (m)
        cube = self._old_compat_load("7.grib2")
        self.assertCML(cube, ("grib_load", "earth_shape_7.cml"))

    def test_shape_of_earth_grib1(self):
        # grib1 - same as grib2 shape 6, above
        cube = self._old_compat_load("global.grib1")
        self.assertCML(cube, ("grib_load", "earth_shape_grib1.cml"))


if __name__ == "__main__":
    tests.main()
