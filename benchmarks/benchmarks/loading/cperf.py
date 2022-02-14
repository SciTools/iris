# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Loading benchmarks for the CPerf scheme of the UK Met Office's NG-VAT project.

CPerf = comparing performance working with data in UM versus LFRic formats.
"""
import numpy as np

from iris import load_cube, save

# TODO: remove uses of PARSE_UGRID_ON_LOAD once UGRID parsing is core behaviour.
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD

from .. import BENCHMARK_DATA, on_demand_benchmark
from ..generate_data.ugrid import make_cubesphere_testfile

# Files available from the UK Met Office:
#  moo ls moose:/adhoc/projects/avd/asv/data_for_nightly_tests/

# The data of the core test UM files has dtype=np.float32 shape=(1920, 2560)
_UM_DIMS_YX = (1920, 2560)
# The closest cubesphere size in terms of datapoints is sqrt(1920*2560 / 6)
#  This gives ~= 905, i.e. "C905"
_N_CUBESPHERE_UM_EQUIVALENT = int(np.sqrt(np.prod(_UM_DIMS_YX) / 6))


@on_demand_benchmark
class SingleDiagnostic:
    # The larger files take a long time to realise.
    timeout = 600.0

    params = [
        ["LFRic", "UM", "UM_lbpack0", "UM_netcdf"],
        [False, True],
        [False, True],
    ]
    param_names = ["file type", "height dim (len 71)", "time dim (len 3)"]

    def setup(self, file_type, three_d, three_times):
        if file_type == "LFRic":
            # Generate an appropriate synthetic LFRic file.
            if three_times:
                n_times = 3
            else:
                n_times = 1

            # Use a cubesphere size ~equivalent to our UM test data.
            cells_per_panel_edge = _N_CUBESPHERE_UM_EQUIVALENT
            create_kwargs = dict(c_size=cells_per_panel_edge, n_times=n_times)

            if three_d:
                create_kwargs["n_levels"] = 71

            # Will re-use a file if already present.
            file_path = make_cubesphere_testfile(**create_kwargs)

        else:
            # Locate the appropriate UM file.
            if three_times:
                # pa/pb003 files
                numeric = "003"
            else:
                # pa/pb000 files
                numeric = "000"

            if three_d:
                # theta diagnostic, N1280 file w/ 71 levels (1920, 2560, 71)
                file_name = f"umglaa_pb{numeric}-theta"
            else:
                # surface_temp diagnostic, N1280 file (1920, 2560)
                file_name = f"umglaa_pa{numeric}-surfacetemp"

            file_suffices = {
                "UM": "",  # packed FF (WGDOS lbpack = 1)
                "UM_lbpack0": ".uncompressed",  # unpacked FF (lbpack = 0)
                "UM_netcdf": ".nc",  # UM file -> Iris -> NetCDF file
            }
            suffix = file_suffices[file_type]

            file_path = (BENCHMARK_DATA / file_name).with_suffix(suffix)
            if not file_path.exists():
                message = "\n".join(
                    [
                        f"Expected local file not found: {file_path}",
                        "Available from the UK Met Office.",
                    ]
                )
                raise FileNotFoundError(message)

        self.file_path = file_path
        self.file_type = file_type

        # To setup time_realise().
        self.loaded_cube = self.load()

    def load(self, realise_coords: bool = False):
        with PARSE_UGRID_ON_LOAD.context():
            cube = load_cube(str(self.file_path))

        assert cube.has_lazy_data()

        # UM files load lon/lat as DimCoords, which are always realised.
        expecting_lazy_coords = self.file_type == "LFRic"
        for coord_name in "longitude", "latitude":
            coord = cube.coord(coord_name)
            assert coord.has_lazy_points() == expecting_lazy_coords
            assert coord.has_lazy_bounds() == expecting_lazy_coords
            if realise_coords:
                # Don't touch actual points/bounds objects - permanent
                #  realisation plays badly with ASV's re-run strategy.
                if coord.has_lazy_points():
                    coord.core_points().compute()
                if coord.has_lazy_bounds():
                    coord.core_bounds().compute()

        return cube

    def time_load(self, _, __, ___):
        """
        The 'real world comparison'
          * UM coords are always realised (DimCoords).
          * LFRic coords are not realised by default (MeshCoords).

        """
        _ = self.load()

    def time_load_w_realised_coords(self, _, __, ___):
        """A valuable extra comparison where both UM and LFRic coords are realised."""
        _ = self.load(realise_coords=True)

    def time_realise(self, _, __, ___):
        # Don't touch loaded_cube.data - permanent realisation plays badly with
        #  ASV's re-run strategy.
        assert self.loaded_cube.has_lazy_data()
        self.loaded_cube.core_data().compute()
