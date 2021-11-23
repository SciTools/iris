# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Integration tests for area weighted regridding."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import iris
from iris.analysis import AreaWeighted


@tests.skip_data
class AreaWeightedTests(tests.IrisTest):
    def setUp(self):
        # Prepare a cube and a template

        cube_file_path = tests.get_data_path(
            ["NetCDF", "regrid", "regrid_xyt.nc"]
        )
        self.cube = iris.load_cube(cube_file_path)

        template_file_path = tests.get_data_path(
            ["NetCDF", "regrid", "regrid_template_global_latlon.nc"]
        )
        self.template_cube = iris.load_cube(template_file_path)

    def test_regrid_area_w_lazy(self):
        # Regrid the cube onto the template.
        out = self.cube.regrid(self.template_cube, AreaWeighted())
        # Check data is still lazy
        self.assertTrue(self.cube.has_lazy_data())
        self.assertTrue(out.has_lazy_data())
        # Save the data
        with self.temp_filename(suffix=".nc") as fname:
            iris.save(out, fname)

    def test_regrid_area_w_lazy_chunked(self):
        # Chunked data makes the regridder run repeatedly
        self.cube.data = self.cube.lazy_data().rechunk((1, -1, -1))
        # Regrid the cube onto the template.
        out = self.cube.regrid(self.template_cube, AreaWeighted())
        # Check data is still lazy
        self.assertTrue(self.cube.has_lazy_data())
        self.assertTrue(out.has_lazy_data())
        # Save the data
        with self.temp_filename(suffix=".nc") as fname:
            iris.save(out, fname)

    def test_regrid_area_w_real_save(self):
        real_cube = self.cube.copy()
        real_cube.data
        # Regrid the cube onto the template.
        out = real_cube.regrid(self.template_cube, AreaWeighted())
        # Realise the data
        out.data
        # Save the data
        with self.temp_filename(suffix=".nc") as fname:
            iris.save(out, fname)

    def test_regrid_area_w_real_start(self):
        real_cube = self.cube.copy()
        real_cube.data
        # Regrid the cube onto the template.
        out = real_cube.regrid(self.template_cube, AreaWeighted())
        # Save the data
        with self.temp_filename(suffix=".nc") as fname:
            iris.save(out, fname)


if __name__ == "__main__":
    tests.main()
