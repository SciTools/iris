# import iris tests first so that some things can be initialised before
# importing anything else.
from iris import tests

import iris

from iris import analysis


@tests.skip_data
class RegriddingTests:
    def setup(self):
        # Prepare a cube and a regridding scheme.
        file_path = tests.get_data_path(
            ["NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc"]
        )
        self.cube = iris.load(file_path)[0]
        self.scheme_area_w = analysis.AreaWeighted()

    def time_regrid_basic(self):
        # Regrid the cube onto itself.
        cube_rotated = self.cube.regrid(self.cube, self.scheme_area_w)
