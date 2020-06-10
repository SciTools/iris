# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test CF-NetCDF file loading and saving.

"""

# Import iris tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import os
import os.path
import shutil
import stat
from subprocess import check_call
import tempfile
from unittest import mock

import netCDF4 as nc
import numpy as np
import numpy.ma as ma

import iris
import iris.analysis.trajectory
import iris.fileformats._pyke_rules.compiled_krb.fc_rules_cf_fc as pyke_rules
import iris.fileformats.netcdf
from iris.fileformats.netcdf import load_cubes as nc_load_cubes
import iris.std_names
import iris.util
from iris.coords import AncillaryVariable, CellMeasure
import iris.coord_systems as icoord_systems
import iris.tests.stock as stock
from iris._lazy_data import is_lazy_data


@tests.skip_data
class TestNetCDFLoad(tests.IrisTest):
    def setUp(self):
        self.tmpdir = None

    def tearDown(self):
        if self.tmpdir is not None:
            shutil.rmtree(self.tmpdir)

    def test_monotonic(self):
        cubes = iris.load(
            tests.get_data_path(
                ("NetCDF", "testing", "test_monotonic_coordinate.nc")
            )
        )
        cubes = sorted(cubes, key=lambda cube: cube.var_name)
        self.assertCML(cubes, ("netcdf", "netcdf_monotonic.cml"))

    def test_load_global_xyt_total(self):
        # Test loading single xyt CF-netCDF file.
        cube = iris.load_cube(
            tests.get_data_path(
                ("NetCDF", "global", "xyt", "SMALL_total_column_co2.nc")
            )
        )
        self.assertCML(cube, ("netcdf", "netcdf_global_xyt_total.cml"))

    def test_load_global_xyt_hires(self):
        # Test loading another single xyt CF-netCDF file.
        cube = iris.load_cube(
            tests.get_data_path(
                ("NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc")
            )
        )
        self.assertCML(cube, ("netcdf", "netcdf_global_xyt_hires.cml"))

    def test_missing_time_bounds(self):
        # Check we can cope with a missing bounds variable.
        with self.temp_filename(suffix="nc") as filename:
            # Tweak a copy of the test data file to rename (we can't delete)
            # the time bounds variable.
            src = tests.get_data_path(
                ("NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc")
            )
            shutil.copyfile(src, filename)
            dataset = nc.Dataset(filename, mode="a")
            dataset.renameVariable("time_bnds", "foo")
            dataset.close()
            _ = iris.load_cube(filename, "eastward_wind")

    def test_load_global_xyzt_gems(self):
        # Test loading single xyzt CF-netCDF file (multi-cube).
        cubes = iris.load(
            tests.get_data_path(
                ("NetCDF", "global", "xyz_t", "GEMS_CO2_Apr2006.nc")
            )
        )
        cubes = sorted(cubes, key=lambda cube: cube.name())
        self.assertCML(cubes, ("netcdf", "netcdf_global_xyzt_gems.cml"))

        # Check the masked array fill value is propogated through the data
        # manager loading.
        lnsp = cubes[1]
        self.assertTrue(ma.isMaskedArray(lnsp.data))
        self.assertEqual(-32767.0, lnsp.data.fill_value)

    def test_load_global_xyzt_gems_iter(self):
        # Test loading stepped single xyzt CF-netCDF file (multi-cube).
        for i, cube in enumerate(
            sorted(
                iris.load(
                    tests.get_data_path(
                        ("NetCDF", "global", "xyz_t", "GEMS_CO2_Apr2006.nc")
                    )
                ),
                key=lambda cube: cube.name(),
            )
        ):
            self.assertCML(
                cube, ("netcdf", "netcdf_global_xyzt_gems_iter_%d.cml" % i)
            )

    # -------------------------------------------------------------------------
    # It is not considered necessary to have integration tests for
    # loading EVERY coordinate system. A subset are tested below.
    # -------------------------------------------------------------------------

    def test_load_rotated_xy_land(self):
        # Test loading single xy rotated pole CF-netCDF file.
        cube = iris.load_cube(
            tests.get_data_path(
                ("NetCDF", "rotated", "xy", "rotPole_landAreaFraction.nc")
            )
        )
        # Make sure the AuxCoords have lazy data.
        self.assertTrue(is_lazy_data(cube.coord("latitude").core_points()))
        self.assertCML(cube, ("netcdf", "netcdf_rotated_xy_land.cml"))

    def test_load_rotated_xyt_precipitation(self):
        # Test loading single xyt rotated pole CF-netCDF file.
        cube = iris.load_cube(
            tests.get_data_path(
                ("NetCDF", "rotated", "xyt", "small_rotPole_precipitation.nc")
            )
        )
        self.assertCML(
            cube, ("netcdf", "netcdf_rotated_xyt_precipitation.cml")
        )

    def test_load_tmerc_grid_and_clim_bounds(self):
        # Test loading a single CF-netCDF file with a transverse Mercator
        # grid_mapping and a time variable with climatology.
        cube = iris.load_cube(
            tests.get_data_path(
                ("NetCDF", "transverse_mercator", "tmean_1910_1910.nc")
            )
        )
        self.assertCML(cube, ("netcdf", "netcdf_tmerc_and_climatology.cml"))

    def test_load_tmerc_grid_with_projection_origin(self):
        # Test loading a single CF-netCDF file with a transverse Mercator
        # grid_mapping that uses longitude_of_projection_origin and
        # scale_factor_at_projection_origin instead of
        # longitude_of_central_meridian and scale_factor_at_central_meridian.
        cube = iris.load_cube(
            tests.get_data_path(
                (
                    "NetCDF",
                    "transverse_mercator",
                    "projection_origin_attributes.nc",
                )
            )
        )

        expected = icoord_systems.TransverseMercator(
            latitude_of_projection_origin=49.0,
            longitude_of_central_meridian=-2.0,
            false_easting=400000.0,
            false_northing=-100000.0,
            scale_factor_at_central_meridian=0.9996012717,
            ellipsoid=icoord_systems.GeogCS(
                semi_major_axis=6377563.396, semi_minor_axis=6356256.91
            ),
        )
        self.assertEqual(
            cube.coord("projection_x_coordinate").coord_system, expected
        )
        self.assertEqual(
            cube.coord("projection_y_coordinate").coord_system, expected
        )

    def test_load_lcc_grid(self):
        # Test loading a single CF-netCDF file with Lambert conformal conic
        # grid mapping.
        cube = iris.load_cube(
            tests.get_data_path(("NetCDF", "lambert_conformal", "test_lcc.nc"))
        )
        self.assertCML(cube, ("netcdf", "netcdf_lcc.cml"))

    def test_missing_climatology(self):
        # Check we can cope with a missing climatology variable.
        with self.temp_filename(suffix="nc") as filename:
            # Tweak a copy of the test data file to rename (we can't delete)
            # the climatology variable.
            src = tests.get_data_path(
                ("NetCDF", "transverse_mercator", "tmean_1910_1910.nc")
            )
            shutil.copyfile(src, filename)
            dataset = nc.Dataset(filename, mode="a")
            dataset.renameVariable("climatology_bounds", "foo")
            dataset.close()
            _ = iris.load_cube(filename, "Mean temperature")

    def test_load_merc_grid(self):
        # Test loading a single CF-netCDF file with a Mercator grid_mapping
        cube = iris.load_cube(
            tests.get_data_path(
                ("NetCDF", "mercator", "toa_brightness_temperature.nc")
            )
        )
        self.assertCML(cube, ("netcdf", "netcdf_merc.cml"))

    def test_load_stereographic_grid(self):
        # Test loading a single CF-netCDF file with a stereographic
        # grid_mapping.
        cube = iris.load_cube(
            tests.get_data_path(
                ("NetCDF", "stereographic", "toa_brightness_temperature.nc")
            )
        )
        self.assertCML(cube, ("netcdf", "netcdf_stereo.cml"))

    def test_cell_methods(self):
        # Test exercising CF-netCDF cell method parsing.
        cubes = iris.load(
            tests.get_data_path(("NetCDF", "testing", "cell_methods.nc"))
        )

        # TEST_COMPAT mod - new cube merge doesn't sort in the same way - test
        # can pass by manual sorting...
        cubes = iris.cube.CubeList(sorted(cubes, key=lambda cube: cube.name()))

        # TEST_COMPAT mod - different versions of the Python module
        # `netCDF4` give different data arrays: MaskedArray vs ndarray
        # Since we're not interested in the data we can just normalise
        # to MaskedArray (to minimise the change).
        for cube in cubes:
            # Force the fill value to be the default netCDF fill value
            # to ensure it matches the previous behaviour.
            cube.data = ma.masked_equal(cube.data, -2147483647)

        self.assertCML(cubes, ("netcdf", "netcdf_cell_methods.cml"))

    def test_ancillary_variables(self):
        # Note: using a CDL string as a test data reference, rather than a binary file.
        ref_cdl = """
            netcdf cm_attr {
            dimensions:
                axv = 3 ;
            variables:
                int64 qqv(axv) ;
                    qqv:long_name = "qq" ;
                    qqv:units = "1" ;
                    qqv:ancillary_variables = "my_av" ;
                int64 axv(axv) ;
                    axv:units = "1" ;
                    axv:long_name = "x" ;
                double my_av(axv) ;
                    my_av:units = "1" ;
                    my_av:long_name = "refs" ;
                    my_av:custom = "extra-attribute";
            data:
                axv = 1, 2, 3;
                my_av = 11., 12., 13.;
            }
            """
        self.tmpdir = tempfile.mkdtemp()
        cdl_path = os.path.join(self.tmpdir, "tst.cdl")
        nc_path = os.path.join(self.tmpdir, "tst.nc")
        # Write CDL string into a temporary CDL file.
        with open(cdl_path, "w") as f_out:
            f_out.write(ref_cdl)
        # Use ncgen to convert this into an actual (temporary) netCDF file.
        command = "ncgen -o {} {}".format(nc_path, cdl_path)
        check_call(command, shell=True)
        # Load with iris.fileformats.netcdf.load_cubes, and check expected content.
        cubes = list(nc_load_cubes(nc_path))
        self.assertEqual(len(cubes), 1)
        avs = cubes[0].ancillary_variables()
        self.assertEqual(len(avs), 1)
        expected = AncillaryVariable(
            np.ma.array([11.0, 12.0, 13.0]),
            long_name="refs",
            var_name="my_av",
            units="1",
            attributes={"custom": "extra-attribute"},
        )
        self.assertEqual(avs[0], expected)

    def test_status_flags(self):
        # Note: using a CDL string as a test data reference, rather than a binary file.
        ref_cdl = """
            netcdf cm_attr {
            dimensions:
                axv = 3 ;
            variables:
                int64 qqv(axv) ;
                    qqv:long_name = "qq" ;
                    qqv:units = "1" ;
                    qqv:ancillary_variables = "my_av" ;
                int64 axv(axv) ;
                    axv:units = "1" ;
                    axv:long_name = "x" ;
                byte my_av(axv) ;
                    my_av:long_name = "qq status_flag" ;
                    my_av:flag_values = 1b, 2b ;
                    my_av:flag_meanings = "a b" ;
            data:
                axv = 11, 21, 31;
                my_av = 1b, 1b, 2b;
            }
            """
        self.tmpdir = tempfile.mkdtemp()
        cdl_path = os.path.join(self.tmpdir, "tst.cdl")
        nc_path = os.path.join(self.tmpdir, "tst.nc")
        # Write CDL string into a temporary CDL file.
        with open(cdl_path, "w") as f_out:
            f_out.write(ref_cdl)
        # Use ncgen to convert this into an actual (temporary) netCDF file.
        command = "ncgen -o {} {}".format(nc_path, cdl_path)
        check_call(command, shell=True)
        # Load with iris.fileformats.netcdf.load_cubes, and check expected content.
        cubes = list(nc_load_cubes(nc_path))
        self.assertEqual(len(cubes), 1)
        avs = cubes[0].ancillary_variables()
        self.assertEqual(len(avs), 1)
        expected = AncillaryVariable(
            np.ma.array([1, 1, 2], dtype=np.int8),
            long_name="qq status_flag",
            var_name="my_av",
            units="no_unit",
            attributes={
                "flag_values": np.array([1, 2], dtype=np.int8),
                "flag_meanings": "a b",
            },
        )
        self.assertEqual(avs[0], expected)

    def test_cell_measures(self):
        # Note: using a CDL string as a test data reference, rather than a binary file.
        ref_cdl = """
            netcdf cm_attr {
            dimensions:
                axv = 3 ;
                ayv = 2 ;
            variables:
                int64 qqv(ayv, axv) ;
                    qqv:long_name = "qq" ;
                    qqv:units = "1" ;
                    qqv:cell_measures = "area: my_areas" ;
                int64 ayv(ayv) ;
                    ayv:units = "1" ;
                    ayv:long_name = "y" ;
                int64 axv(axv) ;
                    axv:units = "1" ;
                    axv:long_name = "x" ;
                double my_areas(ayv, axv) ;
                    my_areas:units = "m2" ;
                    my_areas:long_name = "standardised cell areas" ;
                    my_areas:custom = "extra-attribute";
            data:
                axv = 11, 12, 13;
                ayv = 21, 22;
                my_areas = 110., 120., 130., 221., 231., 241.;
            }
            """
        self.tmpdir = tempfile.mkdtemp()
        cdl_path = os.path.join(self.tmpdir, "tst.cdl")
        nc_path = os.path.join(self.tmpdir, "tst.nc")
        # Write CDL string into a temporary CDL file.
        with open(cdl_path, "w") as f_out:
            f_out.write(ref_cdl)
        # Use ncgen to convert this into an actual (temporary) netCDF file.
        command = "ncgen -o {} {}".format(nc_path, cdl_path)
        check_call(command, shell=True)
        # Load with iris.fileformats.netcdf.load_cubes, and check expected content.
        cubes = list(nc_load_cubes(nc_path))
        self.assertEqual(len(cubes), 1)
        cms = cubes[0].cell_measures()
        self.assertEqual(len(cms), 1)
        expected = CellMeasure(
            np.ma.array([[110.0, 120.0, 130.0], [221.0, 231.0, 241.0]]),
            measure="area",
            var_name="my_areas",
            long_name="standardised cell areas",
            units="m2",
            attributes={"custom": "extra-attribute"},
        )
        self.assertEqual(cms[0], expected)

    def test_deferred_loading(self):
        # Test exercising CF-netCDF deferred loading and deferred slicing.
        # shape (31, 161, 320)
        cube = iris.load_cube(
            tests.get_data_path(
                ("NetCDF", "global", "xyt", "SMALL_total_column_co2.nc")
            )
        )

        # Consecutive index on same dimension.
        self.assertCML(cube[0], ("netcdf", "netcdf_deferred_index_0.cml"))
        self.assertCML(cube[0][0], ("netcdf", "netcdf_deferred_index_1.cml"))
        self.assertCML(
            cube[0][0][0], ("netcdf", "netcdf_deferred_index_2.cml")
        )

        # Consecutive slice on same dimension.
        self.assertCML(cube[0:20], ("netcdf", "netcdf_deferred_slice_0.cml"))
        self.assertCML(
            cube[0:20][0:10], ("netcdf", "netcdf_deferred_slice_1.cml")
        )
        self.assertCML(
            cube[0:20][0:10][0:5], ("netcdf", "netcdf_deferred_slice_2.cml")
        )

        # Consecutive tuple index on same dimension.
        self.assertCML(
            cube[(0, 8, 4, 2, 14, 12),],
            ("netcdf", "netcdf_deferred_tuple_0.cml"),
        )
        self.assertCML(
            cube[(0, 8, 4, 2, 14, 12),][(0, 2, 4, 1),],
            ("netcdf", "netcdf_deferred_tuple_1.cml"),
        )
        subcube = cube[(0, 8, 4, 2, 14, 12),][(0, 2, 4, 1),][
            (1, 3),
        ]
        self.assertCML(subcube, ("netcdf", "netcdf_deferred_tuple_2.cml"))

        # Consecutive mixture on same dimension.
        self.assertCML(
            cube[0:20:2][(9, 5, 8, 0),][3],
            ("netcdf", "netcdf_deferred_mix_0.cml"),
        )
        self.assertCML(
            cube[(2, 7, 3, 4, 5, 0, 9, 10),][2:6][3],
            ("netcdf", "netcdf_deferred_mix_0.cml"),
        )
        self.assertCML(
            cube[0][(0, 2), (1, 3)], ("netcdf", "netcdf_deferred_mix_1.cml")
        )

    def test_units(self):
        # Test exercising graceful cube and coordinate units loading.
        cube0, cube1 = sorted(
            iris.load(tests.get_data_path(("NetCDF", "testing", "units.nc"))),
            key=lambda cube: cube.var_name,
        )

        self.assertCML(cube0, ("netcdf", "netcdf_units_0.cml"))
        self.assertCML(cube1, ("netcdf", "netcdf_units_1.cml"))


class TestNetCDFCRS(tests.IrisTest):
    def setUp(self):
        class Var:
            pass

        self.grid = Var()

    def test_lat_lon_major_minor(self):
        major = 63781370
        minor = 63567523
        self.grid.semi_major_axis = major
        self.grid.semi_minor_axis = minor
        crs = pyke_rules.build_coordinate_system(self.grid)
        self.assertEqual(crs, icoord_systems.GeogCS(major, minor))

    def test_lat_lon_earth_radius(self):
        earth_radius = 63700000
        self.grid.earth_radius = earth_radius
        crs = pyke_rules.build_coordinate_system(self.grid)
        self.assertEqual(crs, icoord_systems.GeogCS(earth_radius))


class SaverPermissions(tests.IrisTest):
    def test_noexist_directory(self):
        # Test capture of suitable exception raised on writing to a
        # non-existent directory.
        dir_name = os.path.join(tempfile.gettempdir(), "non_existent_dir")
        fnme = os.path.join(dir_name, "tmp.nc")
        with self.assertRaises(IOError):
            with iris.fileformats.netcdf.Saver(fnme, "NETCDF4"):
                pass

    def test_bad_permissions(self):
        # Non-exhaustive check that wrong permissions results in a suitable
        # exception being raised.
        dir_name = tempfile.mkdtemp()
        fnme = os.path.join(dir_name, "tmp.nc")
        try:
            os.chmod(dir_name, stat.S_IREAD)
            with self.assertRaises(IOError):
                iris.fileformats.netcdf.Saver(fnme, "NETCDF4")
            self.assertFalse(os.path.exists(fnme))
        finally:
            os.rmdir(dir_name)


@tests.skip_data
class TestSave(tests.IrisTest):
    def test_hybrid(self):
        cube = stock.realistic_4d()

        # Write Cube to netCDF file.
        with self.temp_filename(suffix=".nc") as file_out:
            iris.save(cube, file_out, netcdf_format="NETCDF3_CLASSIC")

            # Check the netCDF file against CDL expected output.
            self.assertCDL(
                file_out, ("netcdf", "netcdf_save_realistic_4d.cdl")
            )

    def test_no_hybrid(self):
        cube = stock.realistic_4d()
        cube.remove_aux_factory(cube.aux_factories[0])

        # Write Cube to netCDF file.
        with self.temp_filename(suffix=".nc") as file_out:
            iris.save(cube, file_out, netcdf_format="NETCDF3_CLASSIC")

            # Check the netCDF file against CDL expected output.
            self.assertCDL(
                file_out, ("netcdf", "netcdf_save_realistic_4d_no_hybrid.cdl")
            )

    def test_scalar_cube(self):
        cube = stock.realistic_4d()[0, 0, 0, 0]

        with self.temp_filename(suffix=".nc") as filename:
            iris.save(cube, filename, netcdf_format="NETCDF3_CLASSIC")
            self.assertCDL(
                filename, ("netcdf", "netcdf_save_realistic_0d.cdl")
            )

    def test_no_name_cube(self):
        # Cube with no names.
        cube = iris.cube.Cube(np.arange(20, dtype=np.float64).reshape((4, 5)))
        dim0 = iris.coords.DimCoord(np.arange(4, dtype=np.float64))
        dim1 = iris.coords.DimCoord(np.arange(5, dtype=np.float64), units="m")
        other = iris.coords.AuxCoord("foobar", units="no_unit")
        cube.add_dim_coord(dim0, 0)
        cube.add_dim_coord(dim1, 1)
        cube.add_aux_coord(other)
        with self.temp_filename(suffix=".nc") as filename:
            iris.save(cube, filename, netcdf_format="NETCDF3_CLASSIC")
            self.assertCDL(filename, ("netcdf", "netcdf_save_no_name.cdl"))


class TestNetCDFSave(tests.IrisTest):
    def setUp(self):
        self.cubell = iris.cube.Cube(
            np.arange(4).reshape(2, 2), "air_temperature"
        )
        self.cube = iris.cube.Cube(
            np.zeros([2, 2]),
            standard_name="surface_temperature",
            long_name=None,
            var_name="temp",
            units="K",
        )
        self.cube2 = iris.cube.Cube(
            np.ones([1, 2, 2]),
            standard_name=None,
            long_name="Something Random",
            var_name="temp2",
            units="K",
        )
        self.cube3 = iris.cube.Cube(
            np.ones([2, 2, 2]),
            standard_name=None,
            long_name="Something Random",
            var_name="temp3",
            units="K",
        )
        self.cube4 = iris.cube.Cube(
            np.zeros([10]),
            standard_name="air_temperature",
            long_name=None,
            var_name="temp",
            units="K",
        )
        self.cube5 = iris.cube.Cube(
            np.ones([20]),
            standard_name=None,
            long_name="air_temperature",
            var_name="temp2",
            units="K",
        )
        self.cube6 = iris.cube.Cube(
            np.ones([10]),
            standard_name=None,
            long_name="air_temperature",
            var_name="temp3",
            units="K",
        )

    @tests.skip_data
    def test_netcdf_save_format(self):
        # Read netCDF input file.
        file_in = tests.get_data_path(
            ("NetCDF", "global", "xyt", "SMALL_total_column_co2.nc")
        )
        cube = iris.load_cube(file_in)

        with self.temp_filename(suffix=".nc") as file_out:
            # Test default NETCDF4 file format saving.
            iris.save(cube, file_out)
            ds = nc.Dataset(file_out)
            self.assertEqual(
                ds.file_format, "NETCDF4", "Failed to save as NETCDF4 format"
            )
            ds.close()

            # Test NETCDF4_CLASSIC file format saving.
            iris.save(cube, file_out, netcdf_format="NETCDF4_CLASSIC")
            ds = nc.Dataset(file_out)
            self.assertEqual(
                ds.file_format,
                "NETCDF4_CLASSIC",
                "Failed to save as NETCDF4_CLASSIC format",
            )
            ds.close()

            # Test NETCDF3_CLASSIC file format saving.
            iris.save(cube, file_out, netcdf_format="NETCDF3_CLASSIC")
            ds = nc.Dataset(file_out)
            self.assertEqual(
                ds.file_format,
                "NETCDF3_CLASSIC",
                "Failed to save as NETCDF3_CLASSIC format",
            )
            ds.close()

            # Test NETCDF4_64BIT file format saving.
            iris.save(cube, file_out, netcdf_format="NETCDF3_64BIT")
            ds = nc.Dataset(file_out)
            self.assertTrue(
                ds.file_format in ["NETCDF3_64BIT", "NETCDF3_64BIT_OFFSET"],
                "Failed to save as NETCDF3_64BIT format",
            )
            ds.close()

            # Test invalid file format saving.
            with self.assertRaises(ValueError):
                iris.save(cube, file_out, netcdf_format="WIBBLE")

    @tests.skip_data
    def test_netcdf_save_single(self):
        # Test saving a single CF-netCDF file.
        # Read PP input file.
        file_in = tests.get_data_path(
            (
                "PP",
                "cf_processing",
                "000003000000.03.236.000128.1990.12.01.00.00.b.pp",
            )
        )
        cube = iris.load_cube(file_in)

        # Write Cube to netCDF file.
        with self.temp_filename(suffix=".nc") as file_out:
            iris.save(cube, file_out)

            # Check the netCDF file against CDL expected output.
            self.assertCDL(file_out, ("netcdf", "netcdf_save_single.cdl"))

    # TODO investigate why merge now make time an AuxCoord rather than a
    # DimCoord and why forecast_period is 'preferred'.
    @tests.skip_data
    def test_netcdf_save_multi2multi(self):
        # Test saving multiple CF-netCDF files.
        # Read PP input file.
        file_in = tests.get_data_path(
            ("PP", "cf_processing", "abcza_pa19591997_daily_29.b.pp")
        )
        cubes = iris.load(file_in)

        # Save multiple cubes to multiple files.
        for index, cube in enumerate(cubes):
            # Write Cube to netCDF file.
            with self.temp_filename(suffix=".nc") as file_out:
                iris.save(cube, file_out)

                # Check the netCDF file against CDL expected output.
                self.assertCDL(
                    file_out, ("netcdf", "netcdf_save_multi_%d.cdl" % index)
                )

    @tests.skip_data
    def test_netcdf_save_multi2single(self):
        # Test saving multiple cubes to a single CF-netCDF file.
        # Read PP input file.
        file_in = tests.get_data_path(
            ("PP", "cf_processing", "abcza_pa19591997_daily_29.b.pp")
        )
        cubes = iris.load(file_in)

        # Write Cube to netCDF file.
        with self.temp_filename(suffix=".nc") as file_out:
            # Check that it is the same on loading
            iris.save(cubes, file_out)

            # Check the netCDF file against CDL expected output.
            self.assertCDL(file_out, ("netcdf", "netcdf_save_multiple.cdl"))

    def test_netcdf_multi_nocoord(self):
        # Testing the saving of a cublist with no coords.
        cubes = iris.cube.CubeList([self.cube, self.cube2, self.cube3])
        with self.temp_filename(suffix=".nc") as file_out:
            iris.save(cubes, file_out)

            # Check the netCDF file against CDL expected output.
            self.assertCDL(file_out, ("netcdf", "netcdf_save_nocoord.cdl"))

    def test_netcdf_multi_samevarnme(self):
        # Testing the saving of a cublist with cubes of the same var_name.
        self.cube2.var_name = self.cube.var_name
        cubes = iris.cube.CubeList([self.cube, self.cube2])
        with self.temp_filename(suffix=".nc") as file_out:
            iris.save(cubes, file_out)

            # Check the netCDF file against CDL expected output.
            self.assertCDL(file_out, ("netcdf", "netcdf_save_samevar.cdl"))

    def test_netcdf_multi_with_coords(self):
        # Testing the saving of a cublist with coordinates.
        lat = iris.coords.DimCoord(
            np.arange(2), long_name=None, var_name="lat", units="degree_north"
        )
        lon = iris.coords.DimCoord(
            np.arange(2),
            standard_name="longitude",
            long_name=None,
            var_name="lon",
            units="degree_east",
        )
        rcoord = iris.coords.DimCoord(
            np.arange(1),
            standard_name=None,
            long_name="Rnd Coordinate",
            units=None,
        )
        self.cube.add_dim_coord(lon, 0)
        self.cube.add_dim_coord(lat, 1)
        self.cube2.add_dim_coord(lon, 1)
        self.cube2.add_dim_coord(lat, 2)
        self.cube2.add_dim_coord(rcoord, 0)

        cubes = iris.cube.CubeList([self.cube, self.cube2])
        with self.temp_filename(suffix=".nc") as file_out:
            iris.save(cubes, file_out)

            # Check the netCDF file against CDL expected output.
            self.assertCDL(file_out, ("netcdf", "netcdf_save_wcoord.cdl"))

    def test_netcdf_multi_wtih_samedimcoord(self):
        time1 = iris.coords.DimCoord(
            np.arange(10), standard_name="time", var_name="time"
        )
        time2 = iris.coords.DimCoord(
            np.arange(20), standard_name="time", var_name="time"
        )

        self.cube4.add_dim_coord(time1, 0)
        self.cube5.add_dim_coord(time2, 0)
        self.cube6.add_dim_coord(time1, 0)

        cubes = iris.cube.CubeList([self.cube4, self.cube5, self.cube6])
        with self.temp_filename(suffix=".nc") as file_out:
            iris.save(cubes, file_out)

            # Check the netCDF file against CDL expected output.
            self.assertCDL(
                file_out, ("netcdf", "netcdf_save_samedimcoord.cdl")
            )

    def test_netcdf_multi_conflict_name_dup_coord(self):
        # Duplicate coordinates with modified variable names lookup.
        latitude1 = iris.coords.DimCoord(
            np.arange(10), standard_name="latitude"
        )
        time2 = iris.coords.DimCoord(np.arange(2), standard_name="time")
        latitude2 = iris.coords.DimCoord(
            np.arange(2), standard_name="latitude"
        )

        self.cube6.add_dim_coord(latitude1, 0)
        self.cube.add_dim_coord(latitude2[:], 1)
        self.cube.add_dim_coord(time2[:], 0)

        cubes = iris.cube.CubeList([self.cube, self.cube6, self.cube6.copy()])
        with self.temp_filename(suffix=".nc") as file_out:
            iris.save(cubes, file_out)

            # Check the netCDF file against CDL expected output.
            self.assertCDL(
                file_out, ("netcdf", "multi_dim_coord_slightly_different.cdl")
            )

    @tests.skip_data
    def test_netcdf_hybrid_height(self):
        # Test saving a CF-netCDF file which contains a hybrid height
        # (i.e. dimensionless vertical) coordinate.
        # Read PP input file.
        names = ["air_potential_temperature", "surface_altitude"]
        file_in = tests.get_data_path(
            ("PP", "COLPEX", "small_colpex_theta_p_alt.pp")
        )
        cube = iris.load_cube(file_in, names[0])

        # Write Cube to netCDF file.
        with self.temp_filename(suffix=".nc") as file_out:
            iris.save(cube, file_out)

            # Check the netCDF file against CDL expected output.
            self.assertCDL(
                file_out, ("netcdf", "netcdf_save_hybrid_height.cdl")
            )

            # Read netCDF file.
            cubes = iris.load(file_out)
            cubes_names = [c.name() for c in cubes]
            self.assertEqual(cubes_names, names)

            # Check the PP read, netCDF write, netCDF read mechanism.
            self.assertCML(
                cubes.extract(names[0])[0],
                ("netcdf", "netcdf_save_load_hybrid_height.cml"),
            )

    @tests.skip_data
    def test_netcdf_save_ndim_auxiliary(self):
        # Test saving CF-netCDF with multi-dimensional auxiliary coordinates.
        # Read netCDF input file.
        file_in = tests.get_data_path(
            ("NetCDF", "rotated", "xyt", "small_rotPole_precipitation.nc")
        )
        cube = iris.load_cube(file_in)

        # Write Cube to nerCDF file.
        with self.temp_filename(suffix=".nc") as file_out:
            iris.save(cube, file_out)

            # Check the netCDF file against CDL expected output.
            self.assertCDL(
                file_out, ("netcdf", "netcdf_save_ndim_auxiliary.cdl")
            )

            # Read the netCDF file.
            cube = iris.load_cube(file_out)

            # Check the netCDF read, write, read mechanism.
            self.assertCML(
                cube, ("netcdf", "netcdf_save_load_ndim_auxiliary.cml")
            )

    def test_netcdf_save_conflicting_aux(self):
        # Test saving CF-netCDF with multi-dimensional auxiliary coordinates,
        # with conflicts.
        self.cube4.add_aux_coord(
            iris.coords.AuxCoord(np.arange(10), "time"), 0
        )
        self.cube6.add_aux_coord(
            iris.coords.AuxCoord(np.arange(10, 20), "time"), 0
        )

        cubes = iris.cube.CubeList([self.cube4, self.cube6])
        with self.temp_filename(suffix=".nc") as file_out:
            iris.save(cubes, file_out)

            # Check the netCDF file against CDL expected output.
            self.assertCDL(file_out, ("netcdf", "netcdf_save_conf_aux.cdl"))

    def test_netcdf_save_gridmapping(self):
        # Test saving CF-netCDF from a cubelist with various grid mappings.

        c1 = self.cubell
        c2 = self.cubell.copy()
        c3 = self.cubell.copy()

        coord_system = icoord_systems.GeogCS(6371229)
        coord_system2 = icoord_systems.GeogCS(6371228)
        coord_system3 = icoord_systems.RotatedGeogCS(30, 30)

        c1.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(1, 3),
                "latitude",
                long_name="1",
                units="degrees",
                coord_system=coord_system,
            ),
            1,
        )
        c1.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(1, 3),
                "longitude",
                long_name="1",
                units="degrees",
                coord_system=coord_system,
            ),
            0,
        )

        c2.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(1, 3),
                "latitude",
                long_name="2",
                units="degrees",
                coord_system=coord_system2,
            ),
            1,
        )
        c2.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(1, 3),
                "longitude",
                long_name="2",
                units="degrees",
                coord_system=coord_system2,
            ),
            0,
        )

        c3.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(1, 3),
                "grid_latitude",
                long_name="3",
                units="degrees",
                coord_system=coord_system3,
            ),
            1,
        )
        c3.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(1, 3),
                "grid_longitude",
                long_name="3",
                units="degrees",
                coord_system=coord_system3,
            ),
            0,
        )

        cubes = iris.cube.CubeList([c1, c2, c3])
        with self.temp_filename(suffix=".nc") as file_out:
            iris.save(cubes, file_out)

            # Check the netCDF file against CDL expected output.
            self.assertCDL(
                file_out, ("netcdf", "netcdf_save_gridmapmulti.cdl")
            )

    def test_netcdf_save_conflicting_names(self):
        # Test saving CF-netCDF with a dimension name corresponding to
        # an existing variable name (conflict).
        self.cube4.add_dim_coord(
            iris.coords.DimCoord(np.arange(10), "time"), 0
        )
        self.cube6.add_aux_coord(iris.coords.AuxCoord(1, "time"), None)

        cubes = iris.cube.CubeList([self.cube4, self.cube6])
        with self.temp_filename(suffix=".nc") as file_out:
            iris.save(cubes, file_out)

            # Check the netCDF file against CDL expected output.
            self.assertCDL(file_out, ("netcdf", "netcdf_save_conf_name.cdl"))

    @tests.skip_data
    def test_trajectory(self):
        file_in = tests.get_data_path(("PP", "aPPglob1", "global.pp"))
        cube = iris.load_cube(file_in)

        # extract a trajectory
        xpoint = cube.coord("longitude").points[:10]
        ypoint = cube.coord("latitude").points[:10]
        sample_points = [("latitude", xpoint), ("longitude", ypoint)]
        traj = iris.analysis.trajectory.interpolate(cube, sample_points)

        # save, reload and check
        with self.temp_filename(suffix=".nc") as temp_filename:
            iris.save(traj, temp_filename)
            reloaded = iris.load_cube(temp_filename)
            self.assertCML(
                reloaded, ("netcdf", "save_load_traj.cml"), checksum=False
            )
            self.assertArrayEqual(traj.data, reloaded.data)

    def test_attributes(self):
        # Should be global attributes.
        aglobals = {
            "history": "A long time ago...",
            "title": "Attribute test",
            "foo": "bar",
        }
        for k, v in aglobals.items():
            self.cube.attributes[k] = v
        # Should be overriden.
        aover = {"Conventions": "TEST"}
        for k, v in aover.items():
            self.cube.attributes[k] = v
        # Should be data varible attributes.
        avars = {
            "standard_error_multiplier": 23,
            "flag_masks": "a",
            "flag_meanings": "b",
            "flag_values": "c",
            "missing_value": 1.0e20,
            "STASH": iris.fileformats.pp.STASH(1, 2, 3),
        }
        for k, v in avars.items():
            self.cube.attributes[k] = v
        with self.temp_filename(suffix=".nc") as filename:
            iris.save(self.cube, filename)
            # Load the dataset.
            ds = nc.Dataset(filename, "r")
            exceptions = []
            # Should be global attributes.
            for gkey in aglobals:
                if getattr(ds, gkey) != aglobals.get(gkey):
                    exceptions.append(
                        "{} != {}".format(
                            getattr(ds, gkey), aglobals.get(gkey)
                        )
                    )
            # Should be overriden.
            for okey in aover:
                if getattr(ds, okey) == aover.get(okey):
                    exceptions.append(
                        "{} != {}".format(getattr(ds, okey), avars.get(okey))
                    )
            dv = ds["temp"]
            # Should be data varible attributes;
            # except STASH -> um_stash_source.
            for vkey in avars:
                if vkey != "STASH" and (getattr(dv, vkey) != avars.get(vkey)):
                    exceptions.append(
                        "{} != {}".format(getattr(dv, vkey), avars.get(vkey))
                    )
            if getattr(dv, "um_stash_source") != avars.get("STASH"):
                exc = "{} != {}".format(
                    getattr(dv, "um_stash_source"), avars.get(vkey)
                )
                exceptions.append(exc)
        self.assertEqual(exceptions, [])

    def test_conflicting_attributes(self):
        # Should be data variable attributes.
        self.cube.attributes["foo"] = "bar"
        self.cube2.attributes["foo"] = "orange"
        with self.temp_filename(suffix=".nc") as filename:
            iris.save([self.cube, self.cube2], filename)
            self.assertCDL(filename, ("netcdf", "netcdf_save_confl_attr.cdl"))

    def test_conflicting_global_attributes(self):
        # Should be data variable attributes, but raise a warning.
        attr_name = "history"
        self.cube.attributes[attr_name] = "Team A won."
        self.cube2.attributes[attr_name] = "Team B won."
        expected_msg = (
            "{attr_name!r} is being added as CF data variable "
            "attribute, but {attr_name!r} should only be a CF "
            "global attribute.".format(attr_name=attr_name)
        )
        with self.temp_filename(suffix=".nc") as filename:
            with mock.patch("warnings.warn") as warn:
                iris.save([self.cube, self.cube2], filename)
                warn.assert_called_with(expected_msg)
                self.assertCDL(
                    filename, ("netcdf", "netcdf_save_confl_global_attr.cdl")
                )

    def test_no_global_attributes(self):
        # Should all be data variable attributes.
        # Different keys.
        self.cube.attributes["a"] = "a"
        self.cube2.attributes["b"] = "a"
        self.cube3.attributes["c"] = "a"
        self.cube4.attributes["d"] = "a"
        self.cube5.attributes["e"] = "a"
        self.cube6.attributes["f"] = "a"
        # Different values.
        self.cube.attributes["g"] = "p"
        self.cube2.attributes["g"] = "q"
        self.cube3.attributes["g"] = "r"
        self.cube4.attributes["g"] = "s"
        self.cube5.attributes["g"] = "t"
        self.cube6.attributes["g"] = "u"
        # One different value.
        self.cube.attributes["h"] = "v"
        self.cube2.attributes["h"] = "v"
        self.cube3.attributes["h"] = "v"
        self.cube4.attributes["h"] = "w"
        self.cube5.attributes["h"] = "v"
        self.cube6.attributes["h"] = "v"
        cubes = [
            self.cube,
            self.cube2,
            self.cube3,
            self.cube4,
            self.cube5,
            self.cube6,
        ]
        with self.temp_filename(suffix=".nc") as filename:
            iris.save(cubes, filename)
            self.assertCDL(
                filename, ("netcdf", "netcdf_save_no_global_attr.cdl")
            )


class TestNetCDFSave__ancillaries(tests.IrisTest):
    """Test for saving data with ancillary variables."""

    def test_fulldims(self):
        testcube = stock.realistic_3d()
        ancil = iris.coords.AncillaryVariable(
            np.zeros(testcube.shape),
            long_name="ancil_data",
            units=1,
            attributes={"attr_1": 7, "attr_2": "chat"},
        )
        testcube.add_ancillary_variable(ancil, (0, 1, 2))
        with self.temp_filename(suffix=".nc") as filename:
            iris.save(testcube, filename)
            self.assertCDL(filename)

    def test_partialdims(self):
        # Test saving ancillary data which maps only dims 0 and 2.
        testcube = stock.realistic_3d()
        ancil = iris.coords.AncillaryVariable(
            np.zeros(testcube[:, 0, :].shape),
            long_name="time_lon_values",
            units="m",
        )
        testcube.add_ancillary_variable(ancil, (0, 2))
        with self.temp_filename(suffix=".nc") as filename:
            iris.save(testcube, filename)
            self.assertCDL(filename)

    def test_multiple(self):
        # Test saving with multiple ancillary variables.
        testcube = stock.realistic_3d()
        ancil1_time_lat_lon = iris.coords.AncillaryVariable(
            np.zeros(testcube.shape), long_name="data_values", units=1
        )
        testcube.add_ancillary_variable(ancil1_time_lat_lon, (0, 1, 2))
        ancil2_time = iris.coords.AncillaryVariable(
            np.zeros(testcube[:, 0, 0].shape),
            long_name="time_values",
            units="s",
        )
        testcube.add_ancillary_variable(ancil2_time, 0)
        ancil3_lon = iris.coords.AncillaryVariable(
            np.zeros(testcube[0, 0, :].shape),
            long_name="lon_values",
            units="m",
        )
        testcube.add_ancillary_variable(ancil3_lon, 2)
        with self.temp_filename(suffix=".nc") as filename:
            iris.save(testcube, filename)
            self.assertCDL(filename)

    def test_shared(self):
        # Check that saving cubes with matching ancillaries maps to a shared
        # file variable.
        testcube_1 = stock.realistic_3d()
        ancil = iris.coords.AncillaryVariable(
            np.zeros(testcube_1[0].shape), long_name="latlon_refs", units="s"
        )
        testcube_1.add_ancillary_variable(ancil, (1, 2))

        testcube_2 = testcube_1.copy()
        testcube_2.units = "m"
        testcube_2.rename("alternate_data")
        with self.temp_filename(suffix=".nc") as filename:
            iris.save([testcube_1, testcube_2], filename)
            self.assertCDL(filename)

            # Also check that only one, shared ancillary variable was written.
            ds = nc.Dataset(filename)
            self.assertIn("air_potential_temperature", ds.variables)
            self.assertIn("alternate_data", ds.variables)
            self.assertEqual(
                ds.variables["air_potential_temperature"].ancillary_variables,
                "latlon_refs",
            )
            self.assertEqual(
                ds.variables["alternate_data"].ancillary_variables,
                "latlon_refs",
            )

    def test_aliases(self):
        # Check that saving cubes with *differing* ancillaries of the same name
        # is correctly resolved.
        testcube_1 = stock.realistic_3d()
        testcube_2 = testcube_1.copy()
        testcube_2.units = "m"
        testcube_2.rename("alternate_data")

        ancil1 = iris.coords.AncillaryVariable(
            np.zeros(testcube_1[0].shape), long_name="latlon_refs", units="s"
        )
        testcube_1.add_ancillary_variable(ancil1, (1, 2))

        ancil2 = ancil1.copy()
        ancil2.data[0, 0] += 1.0
        testcube_2.add_ancillary_variable(ancil2, (1, 2))
        with self.temp_filename(suffix=".nc") as filename:
            iris.save([testcube_1, testcube_2], filename)
            self.assertCDL(filename)


class TestNetCDF3SaveInteger(tests.IrisTest):
    def setUp(self):
        self.cube = iris.cube.Cube(
            np.zeros((2, 2), dtype=np.float64),
            standard_name="surface_temperature",
            long_name=None,
            var_name="temp",
            units="K",
        )

    def test_int64_dimension_coord_netcdf3(self):
        coord = iris.coords.DimCoord(
            np.array([1, 2], dtype=np.int64), long_name="x"
        )
        self.cube.add_dim_coord(coord, 0)
        with self.temp_filename(suffix=".nc") as filename:
            iris.save(self.cube, filename, netcdf_format="NETCDF3_CLASSIC")
            reloaded = iris.load_cube(filename)
            self.assertCML(
                reloaded,
                ("netcdf", "int64_dimension_coord_netcdf3.cml"),
                checksum=False,
            )

    def test_int64_auxiliary_coord_netcdf3(self):
        coord = iris.coords.AuxCoord(
            np.array([1, 2], dtype=np.int64), long_name="x"
        )
        self.cube.add_aux_coord(coord, 0)
        with self.temp_filename(suffix=".nc") as filename:
            iris.save(self.cube, filename, netcdf_format="NETCDF3_CLASSIC")
            reloaded = iris.load_cube(filename)
            self.assertCML(
                reloaded,
                ("netcdf", "int64_auxiliary_coord_netcdf3.cml"),
                checksum=False,
            )

    def test_int64_data_netcdf3(self):
        self.cube.data = self.cube.data.astype(np.int64)
        with self.temp_filename(suffix=".nc") as filename:
            iris.save(self.cube, filename, netcdf_format="NETCDF3_CLASSIC")
            reloaded = iris.load_cube(filename)
            self.assertCML(reloaded, ("netcdf", "int64_data_netcdf3.cml"))

    def test_uint32_dimension_coord_netcdf3(self):
        coord = iris.coords.DimCoord(
            np.array([1, 2], dtype=np.uint32), long_name="x"
        )
        self.cube.add_dim_coord(coord, 0)
        with self.temp_filename(suffix=".nc") as filename:
            iris.save(self.cube, filename, netcdf_format="NETCDF3_CLASSIC")
            reloaded = iris.load_cube(filename)
            self.assertCML(
                reloaded,
                ("netcdf", "uint32_dimension_coord_netcdf3.cml"),
                checksum=False,
            )

    def test_uint32_auxiliary_coord_netcdf3(self):
        coord = iris.coords.AuxCoord(
            np.array([1, 2], dtype=np.uint32), long_name="x"
        )
        self.cube.add_aux_coord(coord, 0)
        with self.temp_filename(suffix=".nc") as filename:
            iris.save(self.cube, filename, netcdf_format="NETCDF3_CLASSIC")
            reloaded = iris.load_cube(filename)
            self.assertCML(
                reloaded,
                ("netcdf", "uint32_auxiliary_coord_netcdf3.cml"),
                checksum=False,
            )

    def test_uint32_data_netcdf3(self):
        self.cube.data = self.cube.data.astype(np.uint32)
        with self.temp_filename(suffix=".nc") as filename:
            iris.save(self.cube, filename, netcdf_format="NETCDF3_CLASSIC")
            reloaded = iris.load_cube(filename)
            self.assertCML(reloaded, ("netcdf", "uint32_data_netcdf3.cml"))

    def test_uint64_dimension_coord_netcdf3(self):
        # Points that cannot be safely cast to int32.
        coord = iris.coords.DimCoord(
            np.array([0, 18446744073709551615], dtype=np.uint64), long_name="x"
        )
        self.cube.add_dim_coord(coord, 0)
        with self.temp_filename(suffix=".nc") as filename:
            with self.assertRaises(ValueError):
                iris.save(self.cube, filename, netcdf_format="NETCDF3_CLASSIC")

    def test_uint64_auxiliary_coord_netcdf3(self):
        # Points that cannot be safely cast to int32.
        coord = iris.coords.AuxCoord(
            np.array([0, 18446744073709551615], dtype=np.uint64), long_name="x"
        )
        self.cube.add_aux_coord(coord, 0)
        with self.temp_filename(suffix=".nc") as filename:
            with self.assertRaises(ValueError):
                iris.save(self.cube, filename, netcdf_format="NETCDF3_CLASSIC")

    def test_uint64_data_netcdf3(self):
        # Data that cannot be safely cast to int32.
        self.cube.data = self.cube.data.astype(np.uint64)
        self.cube.data[0, 1] = 18446744073709551615
        with self.temp_filename(suffix=".nc") as filename:
            with self.assertRaises(ValueError):
                iris.save(self.cube, filename, netcdf_format="NETCDF3_CLASSIC")


class TestCFStandardName(tests.IrisTest):
    def setUp(self):
        pass

    def test_std_name_lookup_pass(self):
        # Test performing a CF standard name look-up hit.
        self.assertTrue("time" in iris.std_names.STD_NAMES)

    def test_std_name_lookup_fail(self):
        # Test performing a CF standard name look-up miss.
        self.assertFalse("phenomenon_time" in iris.std_names.STD_NAMES)


@tests.skip_data
class TestNetCDFUKmoProcessFlags(tests.IrisTest):
    def test_process_flags(self):
        # Test single process flags
        for _, process_desc in iris.fileformats.pp.LBPROC_PAIRS[1:]:
            # Get basic cube and set process flag manually
            ll_cube = stock.lat_lon_cube()
            ll_cube.attributes["ukmo__process_flags"] = (process_desc,)

            # Save cube to netCDF
            with self.temp_filename(suffix=".nc") as temp_filename:
                iris.save(ll_cube, temp_filename)

                # Reload cube
                cube = iris.load_cube(temp_filename)

                # Check correct number and type of flags
                self.assertTrue(
                    len(cube.attributes["ukmo__process_flags"]) == 1,
                    "Mismatch in number of process flags.",
                )
                process_flag = cube.attributes["ukmo__process_flags"][0]
                self.assertEqual(process_flag, process_desc)

        # Test mutiple process flags
        multiple_bit_values = ((128, 64), (4096, 1024), (8192, 1024))

        # Maps lbproc value to the process flags that should be created
        multiple_map = {
            bits: [iris.fileformats.pp.lbproc_map[bit] for bit in bits]
            for bits in multiple_bit_values
        }

        for bits, descriptions in multiple_map.items():

            ll_cube = stock.lat_lon_cube()
            ll_cube.attributes["ukmo__process_flags"] = descriptions

            # Save cube to netCDF
            with self.temp_filename(suffix=".nc") as temp_filename:
                iris.save(ll_cube, temp_filename)

                # Reload cube
                cube = iris.load_cube(temp_filename)

                # Check correct number and type of flags
                process_flags = cube.attributes["ukmo__process_flags"]
                self.assertTrue(
                    len(process_flags) == len(bits),
                    "Mismatch in " "number of process flags.",
                )
                self.assertEqual(set(process_flags), set(descriptions))


if __name__ == "__main__":
    tests.main()
