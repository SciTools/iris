# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Tests for :mod:`iris.experimental.xarray_dataset_wrapper`.

Just very basic integration tests, for now.

"""
from pathlib import Path
import subprocess

import xarray as xr

import iris
from iris.experimental.xarray_dataset_wrapper import fake_nc4python_dataset
import iris.tests as itsts
from iris.tests.stock.netcdf import env_bin_path


def check_cubelists_equal(expected, result):
    expected = sorted(expected, key=lambda c: c.name())
    result = sorted(result, key=lambda c: c.name())
    assert result == expected


class TestLoad:
    def check_load_equality(self, filespec):
        # Simple integration equivalence test.
        filename = itsts.get_data_path(filespec)
        ds = xr.open_dataset(
            filename,
            decode_cf=False,
            decode_coords=False,
            decode_times=False,
        )
        # print(ds)
        nc_faked_xr = fake_nc4python_dataset(ds)

        # phenom_id = "temp_dmax_tmean_abs"
        # expected = iris.load_cube(filename, phenom_id)
        # result = iris.load_cube(nc_faked_xr, phenom_id)
        expected = iris.load(filename)
        result = iris.load(nc_faked_xr)
        # print('\n')
        # print(result)
        # print('---')
        # for i_cube, cube in enumerate(result):
        #     print(f'cube #{i_cube}')
        #     print(cube)
        #     print('\n')
        # print('\n')
        check_cubelists_equal(expected, result)

    def test_equality_eg1(self):
        filespec = [
            "NetCDF",
            "label_and_climate",
            "A1B-99999a-river-sep-2070-2099.nc",
        ]
        self.check_load_equality(filespec)

    def test_equality_eg2(self):
        filespec = ["NetCDF", "global", "xyz_t", "GEMS_CO2_Apr2006.nc"]
        self.check_load_equality(filespec)

    #
    # Sample code for modifying an input file
    #
    # def test_fix_mechanism(self):
    #     # Simple integration test.
    #     filespec = [
    #         "NetCDF", "label_and_climate", "A1B-99999a-river-sep-2070-2099.nc"]
    #
    #     filename = itsts.get_data_path(filespec)
    #     ds = xr.open_dataset(
    #         filename,
    #         decode_cf=False,
    #         decode_coords=False,
    #         decode_times=False,
    #     )
    #     # print(ds)
    #     nc_faked_xr = fake_nc4python_dataset(ds)
    #
    #     # Fix ds so the 'realization_weights' are recognised as an auxcoord
    #     weights_varname = 'weights'
    #     wvar = ds.variables[weights_varname]
    #     assert 'long_name' not in wvar.attrs
    #     ds.variables[weights_varname].attrs['long_name'] = \
    #         ds.variables[weights_varname].attrs['standard_name']
    #     del ds.variables[weights_varname].attrs['standard_name']
    #     for name, var in ds.variables.items():
    #         if 'temp_dmax_tmean_abs' in name:
    #             var.attrs['coordinates'] = weights_varname
    #
    #     result = iris.load(nc_faked_xr)
    #     print('\n')
    #     print(result)
    #     print('---')
    #     for i_cube, cube in enumerate(result):
    #         print(f'cube #{i_cube}')
    #         print(cube)
    #         print('\n')
    #     print('\n')


class TestSave:
    def test_1(self):
        filespec = ["NetCDF", "global", "xyz_t", "GEMS_CO2_Apr2006.nc"]
        filename = itsts.get_data_path(filespec)
        cubes = iris.load(filename)

        nc_faked_xr = fake_nc4python_dataset()
        iris.save(cubes, nc_faked_xr, saver="nc")
        ds = nc_faked_xr.to_xarray_dataset()

        xr_outpath = str(Path("tmp_xr.nc").absolute())
        ds.to_netcdf(xr_outpath)

        iris_outpath = str(Path("tmp_iris.nc").absolute())
        iris.save(cubes, "tmp_iris.nc")

        def capture_dump_lines(filepath_str):
            ncdump_path = str(env_bin_path("ncdump"))
            args = [ncdump_path, "-h", filepath_str]
            process_obj = subprocess.run(args, check=True, capture_output=True)
            lines = process_obj.stdout.decode().split("\n")
            return lines

        lines_xr_save = capture_dump_lines(xr_outpath)
        lines_iris_save = capture_dump_lines(iris_outpath)

        # Debug printout
        # print('\nIRIS OUTPUT:')
        # print('\n'.join(lines_iris_save))
        # print('\nIRIS-XARRAY OUTPUT:')
        # print('\n'.join(lines_xr_save))
        # print('')

        # Show that ncdump output is the same, whether created by normal Iris
        # save, or via iris.save --> xarray.Dataset --> xarray.Data.to_netcdf()
        # Compare, omitting the first line with the filename
        assert lines_xr_save[1:] == lines_iris_save[1:]
