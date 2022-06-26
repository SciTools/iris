import xarray as xr

import iris
from iris.experimental.xarray_dataset_wrapper import fake_nc4python_dataset
import iris.tests as itsts


def check_load_equality(filespec):
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
    expected = sorted(expected, key=lambda c: c.name())
    result = sorted(result, key=lambda c: c.name())
    assert result == expected


def test_equality_eg1():
    filespec = [
        "NetCDF",
        "label_and_climate",
        "A1B-99999a-river-sep-2070-2099.nc",
    ]
    check_load_equality(filespec)


def test_equality_eg2():
    filespec = ["NetCDF", "global", "xyz_t", "GEMS_CO2_Apr2006.nc"]
    check_load_equality(filespec)


# def test_fix_mechanism():
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
