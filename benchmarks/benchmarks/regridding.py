# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Regridding benchmark test

"""

# import iris tests first so that some things can be initialised before
# importing anything else
from iris import tests  # isort:skip

import iris
from iris.analysis import AreaWeighted


class HorizontalChunkedRegridding:
    def setup(self) -> None:
        # Prepare a cube and a template

        cube_file_path = tests.get_data_path(
            ["NetCDF", "regrid", "regrid_xyt.nc"]
        )
        self.cube = iris.load_cube(cube_file_path)

        template_file_path = tests.get_data_path(
            ["NetCDF", "regrid", "regrid_template_global_latlon.nc"]
        )
        self.template_cube = iris.load_cube(template_file_path)

        # Chunked data makes the regridder run repeatedly
        self.cube.data = self.cube.lazy_data().rechunk((1, -1, -1))

    def time_regrid_area_w(self) -> None:
        # Regrid the cube onto the template.
        out = self.cube.regrid(self.template_cube, AreaWeighted())
        # Realise the data
        out.data
