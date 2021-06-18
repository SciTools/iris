# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

"""
This file currently exists as a simple demonstration of Airspeed Velocity
performance testing. it is expected to be expanded/replaced in future.

"""

import iris

# import iris tests first so that some things can be initialised before
# importing anything else.
from iris import tests
from iris.analysis import AreaWeighted


@tests.skip_data
class RegriddingTests:
    def setup(self):
        # Prepare a cube and a regridding scheme.
        file_path = tests.get_data_path(
            ["NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc"]
        )
        self.cube = iris.load_cube(file_path)
        self.scheme_area_w = AreaWeighted()

    def time_regrid_area_w(self):
        # Regrid the cube onto itself.
        self.cube.regrid(self.cube, self.scheme_area_w)
