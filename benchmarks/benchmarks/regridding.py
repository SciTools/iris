# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Regridding benchmark test

"""

# import iris tests first so that some things can be initialised before
# importing anything else.
from iris import tests  # isort:skip

import iris
from iris.analysis import AreaWeighted


@tests.skip_data
class RegriddingTests:
    def setup(self):
        # Prepare a cube and a template

        cube_file_path = tests.get_data_path(
            ["NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc"]
        )
        # cube_file_path = tests.get_data_path(
        #     ["NetCDF", "GloSea", "glosea_bm.nc"]
        # )
        self.cube = iris.load_cube(cube_file_path)

        template_file_path = tests.get_data_path(
            ["NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc"]
        )
        # template_file_path = tests.get_data_path(
        #     ["NetCDF", "GloSea", "template_regrid.nc"]
        # )
        self.template_cube = iris.load_cube(template_file_path)

        # Chunked data makes the regridder run repeatedly
        self.cube.data = self.cube.lazy_data().rechunk((1, -1, -1))

    def time_regrid_area_w(self):
        # Regrid the cube onto the template.
        out = self.cube.regrid(self.template_cube, AreaWeighted())
        # Realise the data
        out.data
