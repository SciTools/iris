# import iris tests first so that some things can be initialised before
# importing anything else.
from iris import tests

import iris

from iris import analysis


@tests.skip_data
class RegriddingTests:
    def setup(self):
        file_path_start = tests.get_data_path(
            ["NetCDF", "global", "xyz_t", "GEMS_CO2_Apr2006.nc"]
        )
        self.cube_start = iris.load(file_path_start)[0]

        file_path_end = tests.get_data_path(
            ["NetCDF", "rotated", "xy", "rotPole_landAreaFraction.nc"]
        )
        self.cube_end = iris.load(file_path_end)[0]

        self.scheme_linear = analysis.Linear()

    def time_regrid_basic(self):
        cube_rotated = self.cube_start.regrid(
            self.cube_end, self.scheme_linear
        )
