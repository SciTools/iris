# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test CF-NetCDF file loading and saving."""

import os
import os.path
import shutil
import stat

import numpy as np
import numpy.ma as ma
import pytest

import iris
from iris._lazy_data import is_lazy_data
import iris.analysis.trajectory
import iris.coord_systems as icoord_systems
from iris.fileformats._nc_load_rules import helpers as ncload_helpers
import iris.fileformats.netcdf
from iris.fileformats.netcdf import _thread_safe_nc
from iris.fileformats.netcdf import load_cubes as nc_load_cubes
import iris.std_names
from iris.tests import _shared_utils
import iris.tests.stock as stock
from iris.tests.stock.netcdf import ncgen_from_cdl
import iris.util
from iris.warnings import IrisCfSaveWarning


@_shared_utils.skip_data
class TestNetCDFLoad:
    def test_monotonic(self, request):
        cubes = iris.load(
            _shared_utils.get_data_path(
                ("NetCDF", "testing", "test_monotonic_coordinate.nc")
            )
        )
        cubes = sorted(cubes, key=lambda cube: cube.var_name)
        _shared_utils.assert_CML(request, cubes, ("netcdf", "netcdf_monotonic.cml"))

    def test_load_global_xyt_total(self, request):
        # Test loading single xyt CF-netCDF file.
        cube = iris.load_cube(
            _shared_utils.get_data_path(
                ("NetCDF", "global", "xyt", "SMALL_total_column_co2.nc")
            )
        )
        _shared_utils.assert_CML(
            request, cube, ("netcdf", "netcdf_global_xyt_total.cml")
        )

    def test_load_global_xyt_hires(self, request):
        # Test loading another single xyt CF-netCDF file.
        cube = iris.load_cube(
            _shared_utils.get_data_path(
                ("NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc")
            )
        )
        _shared_utils.assert_CML(
            request, cube, ("netcdf", "netcdf_global_xyt_hires.cml")
        )

    def test_missing_time_bounds(self, tmp_path):
        # Check we can cope with a missing bounds variable.
        filename = tmp_path / "tmp.nc"
        # Tweak a copy of the test data file to rename (we can't delete)
        # the time bounds variable.
        src = _shared_utils.get_data_path(
            ("NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc")
        )
        shutil.copyfile(src, filename)
        dataset = _thread_safe_nc.DatasetWrapper(filename, mode="a")
        dataset.renameVariable("time_bnds", "foo")
        dataset.close()
        _ = iris.load_cube(filename, "eastward_wind")

    def test_load_global_xyzt_gems(self, request):
        # Test loading single xyzt CF-netCDF file (multi-cube).
        cubes = iris.load(
            _shared_utils.get_data_path(
                ("NetCDF", "global", "xyz_t", "GEMS_CO2_Apr2006.nc")
            )
        )
        cubes = sorted(cubes, key=lambda cube: cube.name())
        _shared_utils.assert_CML(
            request, cubes, ("netcdf", "netcdf_global_xyzt_gems.cml")
        )

        # Check the masked array fill value is propagated through the data
        # manager loading.
        lnsp = cubes[1]
        assert ma.isMaskedArray(lnsp.data)
        assert -32767.0 == lnsp.data.fill_value

    def test_load_global_xyzt_gems_iter(self, request):
        # Test loading stepped single xyzt CF-netCDF file (multi-cube).
        for i, cube in enumerate(
            sorted(
                iris.load(
                    _shared_utils.get_data_path(
                        ("NetCDF", "global", "xyz_t", "GEMS_CO2_Apr2006.nc")
                    )
                ),
                key=lambda cube: cube.name(),
            )
        ):
            _shared_utils.assert_CML(
                request, cube, ("netcdf", "netcdf_global_xyzt_gems_iter_%d.cml" % i)
            )

    # -------------------------------------------------------------------------
    # It is not considered necessary to have integration tests for
    # loading EVERY coordinate system. A subset are tested below.
    # -------------------------------------------------------------------------

    def test_load_rotated_xy_land(self, request):
        # Test loading single xy rotated pole CF-netCDF file.
        cube = iris.load_cube(
            _shared_utils.get_data_path(
                ("NetCDF", "rotated", "xy", "rotPole_landAreaFraction.nc")
            )
        )
        # Make sure the AuxCoords have lazy data.
        assert is_lazy_data(cube.coord("latitude").core_points())
        _shared_utils.assert_CML(
            request, cube, ("netcdf", "netcdf_rotated_xy_land.cml")
        )

    def test_load_rotated_xyt_precipitation(self, request):
        # Test loading single xyt rotated pole CF-netCDF file.
        cube = iris.load_cube(
            _shared_utils.get_data_path(
                ("NetCDF", "rotated", "xyt", "small_rotPole_precipitation.nc")
            )
        )
        _shared_utils.assert_CML(
            request, cube, ("netcdf", "netcdf_rotated_xyt_precipitation.cml")
        )

    def test_load_tmerc_grid_and_clim_bounds(self, request):
        # Test loading a single CF-netCDF file with a transverse Mercator
        # grid_mapping and a time variable with climatology.
        cube = iris.load_cube(
            _shared_utils.get_data_path(
                ("NetCDF", "transverse_mercator", "tmean_1910_1910.nc")
            )
        )
        _shared_utils.assert_CML(
            request, cube, ("netcdf", "netcdf_tmerc_and_climatology.cml")
        )

    def test_load_tmerc_grid_with_projection_origin(self):
        # Test loading a single CF-netCDF file with a transverse Mercator
        # grid_mapping that uses longitude_of_projection_origin and
        # scale_factor_at_projection_origin instead of
        # longitude_of_central_meridian and scale_factor_at_central_meridian.
        cube = iris.load_cube(
            _shared_utils.get_data_path(
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
        assert cube.coord("projection_x_coordinate").coord_system == expected
        assert cube.coord("projection_y_coordinate").coord_system == expected

    def test_load_lcc_grid(self, request):
        # Test loading a single CF-netCDF file with Lambert conformal conic
        # grid mapping.
        cube = iris.load_cube(
            _shared_utils.get_data_path(("NetCDF", "lambert_conformal", "test_lcc.nc"))
        )
        _shared_utils.assert_CML(request, cube, ("netcdf", "netcdf_lcc.cml"))

    def test_missing_climatology(self, tmp_path):
        # Check we can cope with a missing climatology variable.
        filename = tmp_path / "tmp.nc"
        # Tweak a copy of the test data file to rename (we can't delete)
        # the climatology variable.
        src = _shared_utils.get_data_path(
            ("NetCDF", "transverse_mercator", "tmean_1910_1910.nc")
        )
        shutil.copyfile(src, filename)
        dataset = _thread_safe_nc.DatasetWrapper(filename, mode="a")
        dataset.renameVariable("climatology_bounds", "foo")
        dataset.close()
        _ = iris.load_cube(filename, "Mean temperature")

    def test_load_merc_grid(self, request):
        # Test loading a single CF-netCDF file with a Mercator grid_mapping
        cube = iris.load_cube(
            _shared_utils.get_data_path(
                ("NetCDF", "mercator", "toa_brightness_temperature.nc")
            )
        )
        _shared_utils.assert_CML(request, cube, ("netcdf", "netcdf_merc.cml"))

    def test_load_complex_merc_grid(self, request):
        # Test loading a single CF-netCDF file with a Mercator grid_mapping that
        # includes false easting and northing and a standard parallel
        cube = iris.load_cube(
            _shared_utils.get_data_path(
                ("NetCDF", "mercator", "false_east_north_merc.nc")
            )
        )
        _shared_utils.assert_CML(request, cube, ("netcdf", "netcdf_merc_false.cml"))

    def test_load_merc_grid_non_unit_scale_factor(self, request):
        # Test loading a single CF-netCDF file with a Mercator grid_mapping that
        # includes a non-unit scale factor at projection origin
        cube = iris.load_cube(
            _shared_utils.get_data_path(
                ("NetCDF", "mercator", "non_unit_scale_factor_merc.nc")
            )
        )
        _shared_utils.assert_CML(
            request, cube, ("netcdf", "netcdf_merc_scale_factor.cml")
        )

    def test_load_stereographic_grid(self, request):
        # Test loading a single CF-netCDF file with a stereographic
        # grid_mapping.
        cube = iris.load_cube(
            _shared_utils.get_data_path(
                ("NetCDF", "stereographic", "toa_brightness_temperature.nc")
            )
        )
        _shared_utils.assert_CML(request, cube, ("netcdf", "netcdf_stereo.cml"))

    def test_load_polar_stereographic_grid(self, request):
        # Test loading a single CF-netCDF file with a polar stereographic
        # grid_mapping.
        cube = iris.load_cube(
            _shared_utils.get_data_path(
                ("NetCDF", "polar", "toa_brightness_temperature.nc")
            )
        )
        _shared_utils.assert_CML(request, cube, ("netcdf", "netcdf_polar.cml"))

    def test_cell_methods(self, request):
        # Test exercising CF-netCDF cell method parsing.
        cubes = iris.load(
            _shared_utils.get_data_path(("NetCDF", "testing", "cell_methods.nc"))
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

        _shared_utils.assert_CML(request, cubes, ("netcdf", "netcdf_cell_methods.cml"))

    def test_deferred_loading(self, request):
        # Test exercising CF-netCDF deferred loading and deferred slicing.
        # shape (31, 161, 320)
        cube = iris.load_cube(
            _shared_utils.get_data_path(
                ("NetCDF", "global", "xyt", "SMALL_total_column_co2.nc")
            )
        )

        # Consecutive index on same dimension.
        _shared_utils.assert_CML(
            request, cube[0], ("netcdf", "netcdf_deferred_index_0.cml")
        )
        _shared_utils.assert_CML(
            request, cube[0][0], ("netcdf", "netcdf_deferred_index_1.cml")
        )
        _shared_utils.assert_CML(
            request, cube[0][0][0], ("netcdf", "netcdf_deferred_index_2.cml")
        )

        # Consecutive slice on same dimension.
        _shared_utils.assert_CML(
            request, cube[0:20], ("netcdf", "netcdf_deferred_slice_0.cml")
        )
        _shared_utils.assert_CML(
            request, cube[0:20][0:10], ("netcdf", "netcdf_deferred_slice_1.cml")
        )
        _shared_utils.assert_CML(
            request, cube[0:20][0:10][0:5], ("netcdf", "netcdf_deferred_slice_2.cml")
        )

        # Consecutive tuple index on same dimension.
        _shared_utils.assert_CML(
            request,
            cube[((0, 8, 4, 2, 14, 12),)],
            ("netcdf", "netcdf_deferred_tuple_0.cml"),
        )
        _shared_utils.assert_CML(
            request,
            cube[((0, 8, 4, 2, 14, 12),)][((0, 2, 4, 1),)],
            ("netcdf", "netcdf_deferred_tuple_1.cml"),
        )
        subcube = cube[((0, 8, 4, 2, 14, 12),)][((0, 2, 4, 1),)][(1, 3),]
        _shared_utils.assert_CML(
            request, subcube, ("netcdf", "netcdf_deferred_tuple_2.cml")
        )

        # Consecutive mixture on same dimension.
        _shared_utils.assert_CML(
            request,
            cube[0:20:2][((9, 5, 8, 0),)][3],
            ("netcdf", "netcdf_deferred_mix_0.cml"),
        )
        _shared_utils.assert_CML(
            request,
            cube[((2, 7, 3, 4, 5, 0, 9, 10),)][2:6][3],
            ("netcdf", "netcdf_deferred_mix_0.cml"),
        )
        _shared_utils.assert_CML(
            request, cube[0][(0, 2), (1, 3)], ("netcdf", "netcdf_deferred_mix_1.cml")
        )

    def test_um_stash_source(self, tmp_path):
        """Test that um_stash_source is converted into a STASH code."""
        # Note: using a CDL string as a test data reference, rather than a binary file.
        ref_cdl = """
            netcdf cm_attr {
            dimensions:
                axv = 3 ;
                ayv = 2 ;
            variables:
                int64 qqv(ayv, axv) ;
                    qqv:long_name = "qq" ;
                    qqv:ancillary_variables = "my_av" ;
                    qqv:cell_measures = "area: my_areas" ;
                    qqv:um_stash_source = "m01s02i003" ;
                int64 ayv(ayv) ;
                    ayv:long_name = "y" ;
                int64 axv(axv) ;
                    axv:units = "1" ;
                    axv:long_name = "x" ;
                double my_av(axv) ;
                    my_av:long_name = "refs" ;
                double my_areas(ayv, axv) ;
                    my_areas:long_name = "areas" ;
            data:
                axv = 11, 12, 13;
                ayv = 21, 22;
                my_areas = 110., 120., 130., 221., 231., 241.;
            }
            """
        cdl_path = str(tmp_path / "tst.cdl")
        nc_path = str(tmp_path / "tst.nc")
        # Create a temporary netcdf file from the CDL string.
        ncgen_from_cdl(ref_cdl, cdl_path, nc_path)
        # Load with iris.fileformats.netcdf.load_cubes, and check expected content.
        cubes = list(nc_load_cubes(nc_path))
        assert len(cubes) == 1
        assert cubes[0].attributes["STASH"] == iris.fileformats.pp.STASH(1, 2, 3)

    def test_ukmo__um_stash_source_priority(self, tmp_path):
        """Test that ukmo__um_stash_source is converted into a STASH code with a
        higher priority than um_stash_source.
        """
        # Note: using a CDL string as a test data reference, rather than a binary file.
        ref_cdl = """
            netcdf cm_attr {
            dimensions:
                axv = 3 ;
                ayv = 2 ;
            variables:
                int64 qqv(ayv, axv) ;
                    qqv:long_name = "qq" ;
                    qqv:ancillary_variables = "my_av" ;
                    qqv:cell_measures = "area: my_areas" ;
                    qqv:um_stash_source = "m01s02i003" ;
                    qqv:ukmo__um_stash_source = "m09s08i007" ;
                int64 ayv(ayv) ;
                    ayv:long_name = "y" ;
                int64 axv(axv) ;
                    axv:units = "1" ;
                    axv:long_name = "x" ;
                double my_av(axv) ;
                    my_av:long_name = "refs" ;
                double my_areas(ayv, axv) ;
                    my_areas:long_name = "areas" ;
            data:
                axv = 11, 12, 13;
                ayv = 21, 22;
                my_areas = 110., 120., 130., 221., 231., 241.;
            }
            """
        cdl_path = str(tmp_path / "tst.cdl")
        nc_path = str(tmp_path / "tst.nc")
        # Create a temporary netcdf file from the CDL string.
        ncgen_from_cdl(ref_cdl, cdl_path, nc_path)
        # Load with iris.fileformats.netcdf.load_cubes, and check expected content.
        cubes = list(nc_load_cubes(nc_path))
        assert len(cubes) == 1
        assert cubes[0].attributes["STASH"] == iris.fileformats.pp.STASH(9, 8, 7)

    def test_bad_um_stash_source(self, tmp_path):
        """Test that um_stash_source not in strict MSI form is kept."""
        # Note: using a CDL string as a test data reference, rather than a binary file.
        ref_cdl = """
            netcdf cm_attr {
            dimensions:
                axv = 3 ;
                ayv = 2 ;
            variables:
                int64 qqv(ayv, axv) ;
                    qqv:long_name = "qq" ;
                    qqv:ancillary_variables = "my_av" ;
                    qqv:cell_measures = "area: my_areas" ;
                    qqv:um_stash_source = "10*m01s02i003" ;
                int64 ayv(ayv) ;
                    ayv:long_name = "y" ;
                int64 axv(axv) ;
                    axv:units = "1" ;
                    axv:long_name = "x" ;
                double my_av(axv) ;
                    my_av:long_name = "refs" ;
                double my_areas(ayv, axv) ;
                    my_areas:long_name = "areas" ;
            data:
                axv = 11, 12, 13;
                ayv = 21, 22;
                my_areas = 110., 120., 130., 221., 231., 241.;
            }
            """
        cdl_path = str(tmp_path / "tst.cdl")
        nc_path = str(tmp_path / "tst.nc")
        # Create a temporary netcdf file from the CDL string.
        ncgen_from_cdl(ref_cdl, cdl_path, nc_path)
        # Load with iris.fileformats.netcdf.load_cubes, and check expected content.
        cubes = list(nc_load_cubes(nc_path))
        assert len(cubes) == 1
        assert not hasattr(cubes[0].attributes, "STASH")
        assert cubes[0].attributes["um_stash_source"] == "10*m01s02i003"

    def test_units(self, request):
        # Test exercising graceful cube and coordinate units loading.
        cube0, cube1 = sorted(
            iris.load(_shared_utils.get_data_path(("NetCDF", "testing", "units.nc"))),
            key=lambda cube: cube.var_name,
        )

        _shared_utils.assert_CML(request, cube0, ("netcdf", "netcdf_units_0.cml"))
        _shared_utils.assert_CML(request, cube1, ("netcdf", "netcdf_units_1.cml"))


class TestNetCDFCRS:
    @pytest.fixture(autouse=True)
    def _setup(self):
        class Var:
            pass

        self.grid = Var()

    def test_lat_lon_major_minor(self):
        major = 63781370
        minor = 63567523
        self.grid.semi_major_axis = major
        self.grid.semi_minor_axis = minor
        # NB 'build_coordinate_system' has an extra (unused) 'engine' arg, just
        # so that it has the same signature as other coord builder routines.
        engine = None
        crs = ncload_helpers.build_coordinate_system(engine, self.grid)
        assert crs == icoord_systems.GeogCS(major, minor)

    def test_lat_lon_earth_radius(self):
        earth_radius = 63700000
        self.grid.earth_radius = earth_radius
        # NB 'build_coordinate_system' has an extra (unused) 'engine' arg, just
        # so that it has the same signature as other coord builder routines.
        engine = None
        crs = ncload_helpers.build_coordinate_system(engine, self.grid)
        assert crs == icoord_systems.GeogCS(earth_radius)


class TestSaverPermissions:
    def test_noexist_directory(self, tmp_path):
        # Test capture of suitable exception raised on writing to a
        # non-existent directory.
        fnme = str(tmp_path / "non_existent_dir" / "tmp.nc")
        with pytest.raises(IOError, match="Permission denied"):
            with iris.fileformats.netcdf.Saver(fnme, "NETCDF4"):
                pass

    @pytest.fixture
    def tmp_path_no_write(self, tmp_path):
        dir = tmp_path / "no_write_dir"
        dir.mkdir()
        original_perms = dir.stat().st_mode
        dir.chmod(stat.S_IREAD | stat.S_IEXEC)
        yield dir
        dir.chmod(original_perms)

    def test_bad_permissions(self, tmp_path_no_write):
        # Skip this test for the root user. This is applicable to
        # running within a Docker container and/or CIaaS hosted testing.
        if os.getuid():
            # Non-exhaustive check that wrong permissions results in a suitable
            # exception being raised.
            fname = tmp_path_no_write / "tmp.nc"
            with pytest.raises(PermissionError):
                iris.fileformats.netcdf.Saver(str(fname), "NETCDF4")

            assert not fname.exists()


@_shared_utils.skip_data
class TestSave:
    def test_hybrid(self, tmp_path, request):
        cube = stock.realistic_4d()

        # Write Cube to netCDF file.
        file_out = tmp_path / "tmp.nc"
        iris.save(cube, file_out, netcdf_format="NETCDF3_CLASSIC")

        # Check the netCDF file against CDL expected output.
        _shared_utils.assert_CDL(
            request, file_out, ("netcdf", "netcdf_save_realistic_4d.cdl")
        )

    def test_no_hybrid(self, tmp_path, request):
        cube = stock.realistic_4d()
        cube.remove_aux_factory(cube.aux_factories[0])

        # Write Cube to netCDF file.
        file_out = tmp_path / "tmp.nc"
        iris.save(cube, file_out, netcdf_format="NETCDF3_CLASSIC")

        # Check the netCDF file against CDL expected output.
        _shared_utils.assert_CDL(
            request, file_out, ("netcdf", "netcdf_save_realistic_4d_no_hybrid.cdl")
        )

    def test_scalar_cube(self, tmp_path, request):
        cube = stock.realistic_4d()[0, 0, 0, 0]

        filename = tmp_path / "tmp.nc"
        iris.save(cube, filename, netcdf_format="NETCDF3_CLASSIC")
        _shared_utils.assert_CDL(
            request, filename, ("netcdf", "netcdf_save_realistic_0d.cdl")
        )

    def test_no_name_cube(self, tmp_path, request):
        # Cube with no names.
        cube = iris.cube.Cube(np.arange(20, dtype=np.float64).reshape((4, 5)))
        dim0 = iris.coords.DimCoord(np.arange(4, dtype=np.float64))
        dim1 = iris.coords.DimCoord(np.arange(5, dtype=np.float64), units="m")
        other = iris.coords.AuxCoord("foobar", units="no_unit")
        cube.add_dim_coord(dim0, 0)
        cube.add_dim_coord(dim1, 1)
        cube.add_aux_coord(other)
        filename = tmp_path / "tmp.nc"
        iris.save(cube, filename, netcdf_format="NETCDF3_CLASSIC")
        _shared_utils.assert_CDL(
            request, filename, ("netcdf", "netcdf_save_no_name.cdl")
        )


class TestNetCDFSave:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cubell = iris.cube.Cube(np.arange(4).reshape(2, 2), "air_temperature")
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

    @_shared_utils.skip_data
    def test_netcdf_save_format(self, tmp_path):
        # Read netCDF input file.
        file_in = _shared_utils.get_data_path(
            ("NetCDF", "global", "xyt", "SMALL_total_column_co2.nc")
        )
        cube = iris.load_cube(file_in)

        file_out = tmp_path / "tmp.nc"
        # Test default NETCDF4 file format saving.
        iris.save(cube, file_out)
        ds = _thread_safe_nc.DatasetWrapper(file_out)
        assert ds.file_format == "NETCDF4", "Failed to save as NETCDF4 format"
        ds.close()

        # Test NETCDF4_CLASSIC file format saving.
        iris.save(cube, file_out, netcdf_format="NETCDF4_CLASSIC")
        ds = _thread_safe_nc.DatasetWrapper(file_out)
        assert ds.file_format == "NETCDF4_CLASSIC", (
            "Failed to save as NETCDF4_CLASSIC format"
        )
        ds.close()

        # Test NETCDF3_CLASSIC file format saving.
        iris.save(cube, file_out, netcdf_format="NETCDF3_CLASSIC")
        ds = _thread_safe_nc.DatasetWrapper(file_out)
        assert ds.file_format == "NETCDF3_CLASSIC", (
            "Failed to save as NETCDF3_CLASSIC format"
        )
        ds.close()

        # Test NETCDF4_64BIT file format saving.
        iris.save(cube, file_out, netcdf_format="NETCDF3_64BIT")
        ds = _thread_safe_nc.DatasetWrapper(file_out)
        assert ds.file_format in ["NETCDF3_64BIT", "NETCDF3_64BIT_OFFSET"], (
            "Failed to save as NETCDF3_64BIT format"
        )
        ds.close()

        # Test invalid file format saving.
        with pytest.raises(ValueError, match="Unknown netCDF file format"):
            iris.save(cube, file_out, netcdf_format="WIBBLE")

    @_shared_utils.skip_data
    def test_netcdf_save_single(self, tmp_path, request):
        # Test saving a single CF-netCDF file.
        # Read PP input file.
        file_in = _shared_utils.get_data_path(
            (
                "PP",
                "cf_processing",
                "000003000000.03.236.000128.1990.12.01.00.00.b.pp",
            )
        )
        cube = iris.load_cube(file_in)

        # Write Cube to netCDF file.
        file_out = tmp_path / "tmp.nc"
        iris.save(cube, file_out)

        # Check the netCDF file against CDL expected output.
        _shared_utils.assert_CDL(
            request, file_out, ("netcdf", "netcdf_save_single.cdl")
        )

    # TODO investigate why merge now make time an AuxCoord rather than a
    # DimCoord and why forecast_period is 'preferred'.
    @_shared_utils.skip_data
    def test_netcdf_save_multi2multi(self, tmp_path, request):
        # Test saving multiple CF-netCDF files.
        # Read PP input file.
        file_in = _shared_utils.get_data_path(
            ("PP", "cf_processing", "abcza_pa19591997_daily_29.b.pp")
        )
        cubes = iris.load(file_in)

        # Save multiple cubes to multiple files.
        for index, cube in enumerate(cubes):
            # Write Cube to netCDF file.
            file_out = tmp_path / f"tmp_{index}.nc"
            iris.save(cube, file_out)

            # Check the netCDF file against CDL expected output.
            _shared_utils.assert_CDL(
                request, file_out, ("netcdf", "netcdf_save_multi_%d.cdl" % index)
            )

    @_shared_utils.skip_data
    def test_netcdf_save_multi2single(self, tmp_path, request):
        # Test saving multiple cubes to a single CF-netCDF file.
        # Read PP input file.
        file_in = _shared_utils.get_data_path(
            ("PP", "cf_processing", "abcza_pa19591997_daily_29.b.pp")
        )
        cubes = iris.load(file_in)

        # Write Cube to netCDF file.
        file_out = tmp_path / "tmp.nc"
        # Check that it is the same on loading
        iris.save(cubes, file_out)

        # Check the netCDF file against CDL expected output.
        _shared_utils.assert_CDL(
            request, file_out, ("netcdf", "netcdf_save_multiple.cdl")
        )

    def test_netcdf_multi_nocoord(self, tmp_path, request):
        # Testing the saving of a cublist with no coords.
        cubes = iris.cube.CubeList([self.cube, self.cube2, self.cube3])
        file_out = tmp_path / "tmp.nc"
        iris.save(cubes, file_out)

        # Check the netCDF file against CDL expected output.
        _shared_utils.assert_CDL(
            request, file_out, ("netcdf", "netcdf_save_nocoord.cdl")
        )

    def test_netcdf_multi_samevarnme(self, tmp_path, request):
        # Testing the saving of a cublist with cubes of the same var_name.
        self.cube2.var_name = self.cube.var_name
        cubes = iris.cube.CubeList([self.cube, self.cube2])
        file_out = tmp_path / "tmp.nc"
        iris.save(cubes, file_out)

        # Check the netCDF file against CDL expected output.
        _shared_utils.assert_CDL(
            request, file_out, ("netcdf", "netcdf_save_samevar.cdl")
        )

    def test_netcdf_multi_with_coords(self, tmp_path, request):
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
        file_out = tmp_path / "tmp.nc"
        iris.save(cubes, file_out)

        # Check the netCDF file against CDL expected output.
        _shared_utils.assert_CDL(
            request, file_out, ("netcdf", "netcdf_save_wcoord.cdl")
        )

    def test_netcdf_multi_wtih_samedimcoord(self, tmp_path, request):
        time1 = iris.coords.DimCoord(
            np.arange(10), standard_name="time", var_name="time", units="1"
        )
        time2 = iris.coords.DimCoord(
            np.arange(20), standard_name="time", var_name="time", units="1"
        )

        self.cube4.add_dim_coord(time1, 0)
        self.cube5.add_dim_coord(time2, 0)
        self.cube6.add_dim_coord(time1, 0)

        cubes = iris.cube.CubeList([self.cube4, self.cube5, self.cube6])
        file_out = tmp_path / "tmp.nc"
        iris.save(cubes, file_out)

        # Check the netCDF file against CDL expected output.
        _shared_utils.assert_CDL(
            request, file_out, ("netcdf", "netcdf_save_samedimcoord.cdl")
        )

    def test_netcdf_multi_conflict_name_dup_coord(self, tmp_path, request):
        # Duplicate coordinates with modified variable names lookup.
        latitude1 = iris.coords.DimCoord(
            np.arange(10), standard_name="latitude", units="1"
        )
        time2 = iris.coords.DimCoord(np.arange(2), standard_name="time", units="1")
        latitude2 = iris.coords.DimCoord(
            np.arange(2), standard_name="latitude", units="1"
        )

        self.cube6.add_dim_coord(latitude1, 0)
        self.cube.add_dim_coord(latitude2[:], 1)
        self.cube.add_dim_coord(time2[:], 0)

        cubes = iris.cube.CubeList([self.cube, self.cube6, self.cube6.copy()])
        file_out = tmp_path / "tmp.nc"
        iris.save(cubes, file_out)

        # Check the netCDF file against CDL expected output.
        _shared_utils.assert_CDL(
            request, file_out, ("netcdf", "multi_dim_coord_slightly_different.cdl")
        )

    @_shared_utils.skip_data
    def test_netcdf_hybrid_height(self, tmp_path, request):
        # Test saving a CF-netCDF file which contains a hybrid height
        # (i.e. dimensionless vertical) coordinate.
        # Read PP input file.
        names = ["air_potential_temperature", "surface_altitude"]
        file_in = _shared_utils.get_data_path(
            ("PP", "COLPEX", "small_colpex_theta_p_alt.pp")
        )
        cube = iris.load_cube(file_in, names[0])

        # Write Cube to netCDF file.
        file_out = tmp_path / "tmp.nc"
        iris.save(cube, file_out)

        # Check the netCDF file against CDL expected output.
        _shared_utils.assert_CDL(
            request, file_out, ("netcdf", "netcdf_save_hybrid_height.cdl")
        )

        # Read netCDF file.
        cubes = iris.load(file_out)
        cubes_names = [c.name() for c in cubes]
        assert cubes_names == names

        # Check the PP read, netCDF write, netCDF read mechanism.
        _shared_utils.assert_CML(
            request,
            cubes.extract(names[0])[0],
            ("netcdf", "netcdf_save_load_hybrid_height.cml"),
        )

    @_shared_utils.skip_data
    def test_netcdf_save_ndim_auxiliary(self, tmp_path, request):
        # Test saving CF-netCDF with multi-dimensional auxiliary coordinates.
        # Read netCDF input file.
        file_in = _shared_utils.get_data_path(
            ("NetCDF", "rotated", "xyt", "small_rotPole_precipitation.nc")
        )
        cube = iris.load_cube(file_in)

        # Write Cube to nerCDF file.
        file_out = tmp_path / "tmp.nc"
        iris.save(cube, file_out)

        # Check the netCDF file against CDL expected output.
        _shared_utils.assert_CDL(
            request, file_out, ("netcdf", "netcdf_save_ndim_auxiliary.cdl")
        )

        # Read the netCDF file.
        cube = iris.load_cube(file_out)

        # Check the netCDF read, write, read mechanism.
        _shared_utils.assert_CML(
            request, cube, ("netcdf", "netcdf_save_load_ndim_auxiliary.cml")
        )

    def test_netcdf_save_conflicting_aux(self, tmp_path, request):
        # Test saving CF-netCDF with multi-dimensional auxiliary coordinates,
        # with conflicts.
        self.cube4.add_aux_coord(
            iris.coords.AuxCoord(np.arange(10), "time", units="1"), 0
        )
        self.cube6.add_aux_coord(
            iris.coords.AuxCoord(np.arange(10, 20), "time", units="1"), 0
        )

        cubes = iris.cube.CubeList([self.cube4, self.cube6])
        file_out = tmp_path / "tmp.nc"
        iris.save(cubes, file_out)

        # Check the netCDF file against CDL expected output.
        _shared_utils.assert_CDL(
            request, file_out, ("netcdf", "netcdf_save_conf_aux.cdl")
        )

    def test_netcdf_save_gridmapping(self, tmp_path, request):
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
        file_out = tmp_path / "tmp.nc"
        iris.save(cubes, file_out)

        # Check the netCDF file against CDL expected output.
        _shared_utils.assert_CDL(
            request, file_out, ("netcdf", "netcdf_save_gridmapmulti.cdl")
        )

    def test_netcdf_save_conflicting_names(self, tmp_path, request):
        # Test saving CF-netCDF with a dimension name corresponding to
        # an existing variable name (conflict).
        self.cube4.add_dim_coord(
            iris.coords.DimCoord(np.arange(10), "time", units="1"), 0
        )
        self.cube6.add_aux_coord(iris.coords.AuxCoord(1, "time", units="1"), None)

        cubes = iris.cube.CubeList([self.cube4, self.cube6])
        file_out = tmp_path / "tmp.nc"
        iris.save(cubes, file_out)

        # Check the netCDF file against CDL expected output.
        _shared_utils.assert_CDL(
            request, file_out, ("netcdf", "netcdf_save_conf_name.cdl")
        )

    @_shared_utils.skip_data
    def test_trajectory(self, tmp_path, request):
        file_in = _shared_utils.get_data_path(("PP", "aPPglob1", "global.pp"))
        cube = iris.load_cube(file_in)

        # extract a trajectory
        xpoint = cube.coord("longitude").points[:10]
        ypoint = cube.coord("latitude").points[:10]
        sample_points = [("latitude", xpoint), ("longitude", ypoint)]
        traj = iris.analysis.trajectory.interpolate(cube, sample_points)

        # save, reload and check
        temp_filename = tmp_path / "tmp.nc"
        iris.save(traj, temp_filename)
        reloaded = iris.load_cube(temp_filename)
        _shared_utils.assert_CML(
            request, reloaded, ("netcdf", "save_load_traj.cml"), checksum=False
        )
        _shared_utils.assert_array_equal(traj.data, reloaded.data)

    def test_attributes(self, tmp_path):
        # Should be global attributes.
        aglobals = {
            "history": "A long time ago...",
            "title": "Attribute test",
            "foo": "bar",
        }
        for k, v in aglobals.items():
            self.cube.attributes[k] = v
        # Should be overridden.
        aover = {"Conventions": "TEST"}
        for k, v in aover.items():
            self.cube.attributes[k] = v
        # Should be data variable attributes.
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

        filename = tmp_path / "tmp.nc"
        iris.save(self.cube, filename)
        # Load the dataset.
        ds = _thread_safe_nc.DatasetWrapper(filename, "r")
        exceptions = []
        # Should be global attributes.
        for gkey in aglobals:
            if getattr(ds, gkey) != aglobals.get(gkey):
                exceptions.append(
                    "{} != {}".format(getattr(ds, gkey), aglobals.get(gkey))
                )
        # Should be overridden.
        for okey in aover:
            if getattr(ds, okey) == aover.get(okey):
                exceptions.append("{} != {}".format(getattr(ds, okey), avars.get(okey)))
        dv = ds["temp"]
        # Should be data variable attributes;
        # except STASH -> um_stash_source.
        for vkey in avars:
            if vkey != "STASH" and (getattr(dv, vkey) != avars.get(vkey)):
                exceptions.append("{} != {}".format(getattr(dv, vkey), avars.get(vkey)))
        if getattr(dv, "um_stash_source") != avars.get("STASH"):
            exc = "{} != {}".format(getattr(dv, "um_stash_source"), avars.get(vkey))
            exceptions.append(exc)

        assert exceptions == []

    def test_conflicting_attributes(self, tmp_path, request):
        # Should be data variable attributes.
        self.cube.attributes["foo"] = "bar"
        self.cube2.attributes["foo"] = "orange"
        filename = tmp_path / "tmp.nc"
        iris.save([self.cube, self.cube2], filename)
        _shared_utils.assert_CDL(
            request, filename, ("netcdf", "netcdf_save_confl_attr.cdl")
        )

    def test_conflicting_global_attributes(self, tmp_path, request):
        # Should be data variable attributes, but raise a warning.
        attr_name = "history"
        self.cube.attributes[attr_name] = "Team A won."
        self.cube2.attributes[attr_name] = "Team B won."
        expected_msg = (
            "{attr_name!r} is being added as CF data variable "
            "attribute, but {attr_name!r} should only be a CF "
            "global attribute.".format(attr_name=attr_name)
        )
        filename = tmp_path / "tmp.nc"
        with pytest.warns(IrisCfSaveWarning, match=expected_msg):
            iris.save([self.cube, self.cube2], filename)
        _shared_utils.assert_CDL(
            request, filename, ("netcdf", "netcdf_save_confl_global_attr.cdl")
        )

    def test_no_global_attributes(self, tmp_path, request):
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

        filename = tmp_path / "tmp.nc"
        iris.save(cubes, filename)
        _shared_utils.assert_CDL(
            request, filename, ("netcdf", "netcdf_save_no_global_attr.cdl")
        )


class TestNetCDFSave__ancillaries:
    """Test for saving data with ancillary variables."""

    def test_fulldims(self, tmp_path, request):
        testcube = stock.realistic_3d()
        ancil = iris.coords.AncillaryVariable(
            np.zeros(testcube.shape),
            long_name="ancil_data",
            units=1,
            attributes={"attr_1": 7, "attr_2": "chat"},
        )
        testcube.add_ancillary_variable(ancil, (0, 1, 2))
        filename = tmp_path / "tmp.nc"
        iris.save(testcube, filename)
        _shared_utils.assert_CDL(request, filename)

    def test_partialdims(self, tmp_path, request):
        # Test saving ancillary data which maps only dims 0 and 2.
        testcube = stock.realistic_3d()
        ancil = iris.coords.AncillaryVariable(
            np.zeros(testcube[:, 0, :].shape),
            long_name="time_lon_values",
            units="m",
        )
        testcube.add_ancillary_variable(ancil, (0, 2))
        filename = tmp_path / "tmp.nc"
        iris.save(testcube, filename)
        _shared_utils.assert_CDL(request, filename)

    def test_multiple(self, tmp_path, request):
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
        filename = tmp_path / "tmp.nc"
        iris.save(testcube, filename)
        _shared_utils.assert_CDL(request, filename)

    def test_shared(self, tmp_path, request):
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

        filename = tmp_path / "tmp.nc"
        iris.save([testcube_1, testcube_2], filename)
        _shared_utils.assert_CDL(request, filename)

        # Also check that only one, shared ancillary variable was written.
        ds = _thread_safe_nc.DatasetWrapper(filename)
        assert "air_potential_temperature" in ds.variables
        assert "alternate_data" in ds.variables
        assert (
            ds.variables["air_potential_temperature"].ancillary_variables
            == "latlon_refs"
        )
        assert ds.variables["alternate_data"].ancillary_variables == "latlon_refs"

    def test_aliases(self, tmp_path, request):
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
        filename = tmp_path / "tmp.nc"
        iris.save([testcube_1, testcube_2], filename)
        _shared_utils.assert_CDL(request, filename)

    def test_flag(self, tmp_path, request):
        testcube = stock.realistic_3d()
        flag = iris.coords.AncillaryVariable(
            np.ones(testcube.shape, dtype=np.int8),
            long_name="quality_flag",
            attributes={
                "flag_meanings": "PASS FAIL MISSING",
                "flag_values": np.array([1, 2, 9], dtype=np.int8),
            },
        )
        testcube.add_ancillary_variable(flag, (0, 1, 2))
        filename = tmp_path / "tmp.nc"
        iris.save(testcube, filename)
        _shared_utils.assert_CDL(request, filename)


class TestNetCDF3SaveInteger:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = iris.cube.Cube(
            np.zeros((2, 2), dtype=np.float64),
            standard_name="surface_temperature",
            long_name=None,
            var_name="temp",
            units="K",
        )

    def test_int64_dimension_coord_netcdf3(self, tmp_path, request):
        coord = iris.coords.DimCoord(np.array([1, 2], dtype=np.int64), long_name="x")
        self.cube.add_dim_coord(coord, 0)
        filename = tmp_path / "tmp.nc"
        iris.save(self.cube, filename, netcdf_format="NETCDF3_CLASSIC")
        reloaded = iris.load_cube(filename)
        _shared_utils.assert_CML(
            request,
            reloaded,
            ("netcdf", "int64_dimension_coord_netcdf3.cml"),
            checksum=False,
        )

    def test_int64_auxiliary_coord_netcdf3(self, tmp_path, request):
        coord = iris.coords.AuxCoord(np.array([1, 2], dtype=np.int64), long_name="x")
        self.cube.add_aux_coord(coord, 0)
        filename = tmp_path / "tmp.nc"
        iris.save(self.cube, filename, netcdf_format="NETCDF3_CLASSIC")
        reloaded = iris.load_cube(filename)
        _shared_utils.assert_CML(
            request,
            reloaded,
            ("netcdf", "int64_auxiliary_coord_netcdf3.cml"),
            checksum=False,
        )

    def test_int64_data_netcdf3(self, tmp_path, request):
        self.cube.data = self.cube.data.astype(np.int64)
        filename = tmp_path / "tmp.nc"
        iris.save(self.cube, filename, netcdf_format="NETCDF3_CLASSIC")
        reloaded = iris.load_cube(filename)
        _shared_utils.assert_CML(
            request, reloaded, ("netcdf", "int64_data_netcdf3.cml")
        )

    def test_uint32_dimension_coord_netcdf3(self, tmp_path, request):
        coord = iris.coords.DimCoord(np.array([1, 2], dtype=np.uint32), long_name="x")
        self.cube.add_dim_coord(coord, 0)
        filename = tmp_path / "tmp.nc"
        iris.save(self.cube, filename, netcdf_format="NETCDF3_CLASSIC")
        reloaded = iris.load_cube(filename)
        _shared_utils.assert_CML(
            request,
            reloaded,
            ("netcdf", "uint32_dimension_coord_netcdf3.cml"),
            checksum=False,
        )

    def test_uint32_auxiliary_coord_netcdf3(self, tmp_path, request):
        coord = iris.coords.AuxCoord(np.array([1, 2], dtype=np.uint32), long_name="x")
        self.cube.add_aux_coord(coord, 0)
        filename = tmp_path / "tmp.nc"
        iris.save(self.cube, filename, netcdf_format="NETCDF3_CLASSIC")
        reloaded = iris.load_cube(filename)
        _shared_utils.assert_CML(
            request,
            reloaded,
            ("netcdf", "uint32_auxiliary_coord_netcdf3.cml"),
            checksum=False,
        )

    def test_uint32_data_netcdf3(self, tmp_path, request):
        self.cube.data = self.cube.data.astype(np.uint32)
        filename = tmp_path / "tmp.nc"
        iris.save(self.cube, filename, netcdf_format="NETCDF3_CLASSIC")
        reloaded = iris.load_cube(filename)
        _shared_utils.assert_CML(
            request, reloaded, ("netcdf", "uint32_data_netcdf3.cml")
        )

    def test_uint64_dimension_coord_netcdf3(self, tmp_path):
        # Points that cannot be safely cast to int32.
        coord = iris.coords.DimCoord(
            np.array([0, 18446744073709551615], dtype=np.uint64), long_name="x"
        )
        self.cube.add_dim_coord(coord, 0)
        filename = tmp_path / "tmp.nc"
        with pytest.raises(ValueError, match="not supported by NETCDF3_CLASSIC"):
            iris.save(self.cube, filename, netcdf_format="NETCDF3_CLASSIC")

    def test_uint64_auxiliary_coord_netcdf3(self, tmp_path):
        # Points that cannot be safely cast to int32.
        coord = iris.coords.AuxCoord(
            np.array([0, 18446744073709551615], dtype=np.uint64), long_name="x"
        )
        self.cube.add_aux_coord(coord, 0)
        filename = tmp_path / "tmp.nc"
        with pytest.raises(ValueError, match="not supported by NETCDF3_CLASSIC"):
            iris.save(self.cube, filename, netcdf_format="NETCDF3_CLASSIC")

    def test_uint64_data_netcdf3(self, tmp_path):
        # Data that cannot be safely cast to int32.
        self.cube.data = self.cube.data.astype(np.uint64)
        self.cube.data[0, 1] = 18446744073709551615
        filename = tmp_path / "tmp.nc"
        with pytest.raises(ValueError, match="not supported by NETCDF3_CLASSIC"):
            iris.save(self.cube, filename, netcdf_format="NETCDF3_CLASSIC")


class TestCFStandardName:
    def test_std_name_lookup_pass(self):
        # Test performing a CF standard name look-up hit.
        assert "time" in iris.std_names.STD_NAMES

    def test_std_name_lookup_fail(self):
        # Test performing a CF standard name look-up miss.
        assert "phenomenon_time" not in iris.std_names.STD_NAMES


@_shared_utils.skip_data
class TestNetCDFUKmoProcessFlags:
    def test_process_flags(self, tmp_path):
        # Test single process flags
        for _, process_desc in iris.fileformats.pp.LBPROC_PAIRS[1:]:
            # Get basic cube and set process flag manually
            ll_cube = stock.lat_lon_cube()
            ll_cube.attributes["ukmo__process_flags"] = (process_desc,)

            # Save cube to netCDF
            temp_filename = tmp_path / "tmp.nc"
            iris.save(ll_cube, temp_filename)

            # Reload cube
            cube = iris.load_cube(temp_filename)

            # Check correct number and type of flags
            assert len(cube.attributes["ukmo__process_flags"]) == 1, (
                "Mismatch in number of process flags."
            )
            process_flag = cube.attributes["ukmo__process_flags"][0]
            assert process_flag == process_desc

        # Test multiple process flags
        multiple_bit_values = ((128, 64), (4096, 1024), (8192, 1024))

        # Maps lbproc value to the process flags that should be created
        multiple_map = {
            bits: tuple([iris.fileformats.pp.lbproc_map[bit] for bit in bits])
            for bits in multiple_bit_values
        }

        for bits, descriptions in multiple_map.items():
            ll_cube = stock.lat_lon_cube()
            ll_cube.attributes["ukmo__process_flags"] = descriptions

            # Save cube to netCDF
            temp_filename = tmp_path / "tmp.nc"
            iris.save(ll_cube, temp_filename)

            # Reload cube
            cube = iris.load_cube(temp_filename)

            # Check correct number and type of flags
            process_flags = cube.attributes["ukmo__process_flags"]
            assert len(process_flags) == len(bits), (
                "Mismatch in number of process flags."
            )
            assert set(process_flags) == set(descriptions)
