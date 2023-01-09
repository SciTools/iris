# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests  # isort:skip

import os
import tempfile

import iris
import iris.coords
from iris.fileformats.netcdf import _thread_safe_nc
from iris.fileformats.pp import STASH
import iris.tests.pp as pp
import iris.util


def callback_000003000000_16_202_000128_1860_09_01_00_00_b_pp(
    cube, field, filename
):
    cube.attributes["STASH"] = STASH(1, 16, 202)
    cube.standard_name = "geopotential_height"
    cube.units = "m"


def callback_HadCM2_ts_SAT_ann_18602100_b_pp(cube, field, filename):
    def reset_pole(coord_name):
        coord = cube.coord(coord_name)
        coord.rename(coord.name().replace("grid_", ""))
        coord.coord_system = coord.coord_system.ellipsoid

    reset_pole("grid_latitude")
    reset_pole("grid_longitude")
    cube.standard_name = "air_temperature"
    cube.units = "Celsius"
    cube.attributes["STASH"] = STASH(1, 3, 236)
    # Force the height to 1.5m
    if cube.coords("height"):
        cube.remove_coord("height")
    height_coord = iris.coords.DimCoord(1.5, standard_name="height", units="m")
    cube.add_aux_coord(height_coord)


def callback_model_b_pp(cube, field, filename):
    cube.standard_name = "air_temperature"
    cube.units = "K"
    cube.attributes["STASH"] = STASH(1, 16, 203)


def callback_integer_b_pp(cube, field, filename):
    cube.standard_name = "land_binary_mask"
    cube.units = "1"
    del cube.attributes["STASH"]


def callback_001000000000_00_000_000000_1860_01_01_00_00_f_b_pp(
    cube, field, filename
):
    cube.standard_name = "sea_surface_height_above_geoid"
    cube.units = "m"


def callback_aaxzc_n10r13xy_b_pp(cube, field, filename):
    height_coord = iris.coords.DimCoord(1.5, long_name="height", units="m")
    cube.add_aux_coord(height_coord)


@tests.skip_data
class TestAll(tests.IrisTest, pp.PPTest):
    _ref_dir = ("usecases", "pp_to_cf_conversion")

    def _test_file(self, name):
        """This is the main test routine that is called for each of the files listed below."""
        pp_path = self._src_pp_path(name)

        # 1) Load the PP and check the Cube
        callback_name = "callback_" + name.replace(".", "_")
        callback = globals().get(callback_name)
        cubes = iris.load(pp_path, callback=callback)

        if name.endswith(".pp"):
            fname_name = name[:-3]
        else:
            fname_name = name

        self.assertCML(cubes, self._ref_dir + ("from_pp", fname_name + ".cml"))

        # 2) Save the Cube and check the netCDF
        nc_filenames = []

        for index, cube in enumerate(cubes):
            # Explicitly set a fill-value as a workaround for
            # https://github.com/Unidata/netcdf4-python/issues/725
            fill_value = _thread_safe_nc.default_fillvals[cube.dtype.str[1:]]

            file_nc = tempfile.NamedTemporaryFile(
                suffix=".nc", delete=False
            ).name
            iris.save(
                cube,
                file_nc,
                netcdf_format="NETCDF3_CLASSIC",
                fill_value=fill_value,
            )

            # Check the netCDF file against CDL expected output.
            self.assertCDL(
                file_nc,
                self._ref_dir
                + ("to_netcdf", "%s_%d.cdl" % (fname_name, index)),
            )
            nc_filenames.append(file_nc)

        # 3) Load the netCDF and check the Cube
        for index, nc_filename in enumerate(nc_filenames):
            # Read netCDF to Cube.
            cube = iris.load_cube(nc_filename)
            self.assertCML(
                cube,
                self._ref_dir
                + ("from_netcdf", "%s_%d.cml" % (fname_name, index)),
            )
            os.remove(nc_filename)

        # 4) Save the Cube and check the PP
        # Only the first four files pass their tests at the moment.

        if name in self.files_to_check[:4]:
            self._test_pp_save(cubes, name)

    def _src_pp_path(self, name):
        return tests.get_data_path(("PP", "cf_processing", name))

    def _test_pp_save(self, cubes, name):
        # If there's no existing reference file then make it from the *source* data
        reference_txt_path = tests.get_result_path(
            self._ref_dir + ("to_pp", name + ".txt")
        )
        reference_pp_path = self._src_pp_path(name)
        with self.cube_save_test(
            reference_txt_path, reference_pp_path=reference_pp_path
        ) as temp_pp_path:
            iris.save(cubes, temp_pp_path)

    files_to_check = [
        "000003000000.03.236.000128.1990.12.01.00.00.b.pp",
        "000003000000.03.236.004224.1990.12.01.00.00.b.pp",
        "000003000000.03.236.008320.1990.12.01.00.00.b.pp",
        "000003000000.16.202.000128.1860.09.01.00.00.b.pp",
        "001000000000.00.000.000000.1860.01.01.00.00.f.b.pp",
        "002000000000.44.101.131200.1920.09.01.00.00.b.pp",
        "008000000000.44.101.000128.1890.09.01.00.00.b.pp",
        "HadCM2_ts_SAT_ann_18602100.b.pp",
        "aaxzc_level_lat_orig.b.pp",
        "aaxzc_lon_lat_press_orig.b.pp",
        "abcza_pa19591997_daily_29.b.pp",
        "12187.b.pp",
        "ocean_xsect.b.pp",
        "model.b.pp",
        "integer.b.pp",
        "aaxzc_lon_lat_several.b.pp",
        "aaxzc_n10r13xy.b.pp",
        "aaxzc_time_press.b.pp",
        "aaxzc_tseries.b.pp",
        "abxpa_press_lat.b.pp",
        "st30211.b.pp",
        "st0fc942.b.pp",
        "st0fc699.b.pp",
    ]


def make_test_function(func_name, file_name):
    """Builds a function which can be later turned into a bound method."""
    scope = {}
    exec(
        """def %s(self):
                name = %r
                self._test_file(name)
    """
        % (func_name, file_name),
        scope,
        scope,
    )
    # return the newly created function
    return scope[func_name]


def attach_tests():
    # attach a test method on TestAll for each file to test
    for file_name in TestAll.files_to_check:
        func_name = "test_{}".format(file_name.replace(".", "_"))
        test_func = make_test_function(func_name, file_name)
        setattr(TestAll, func_name, test_func)


attach_tests()


if __name__ == "__main__":
    tests.main()
