# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Integration tests for loading and saving netcdf files."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from contextlib import contextmanager
from itertools import repeat
import os.path
from os.path import join as path_join
import shutil
import tempfile
from unittest import mock
import warnings

import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import pytest

import iris
import iris.coord_systems
from iris.coords import CellMethod, DimCoord
from iris.cube import Cube, CubeList
import iris.exceptions
from iris.fileformats.netcdf import (
    CF_CONVENTIONS_VERSION,
    Saver,
    UnknownCellMethodWarning,
)
import iris.tests.stock as stock
from iris.tests.stock.netcdf import ncgen_from_cdl
import iris.tests.unit.fileformats.netcdf.test_load_cubes as tlc


@tests.skip_data
class TestAtmosphereSigma(tests.IrisTest):
    def setUp(self):
        # Modify stock cube so it is suitable to have a atmosphere sigma
        # factory added to it.
        cube = stock.realistic_4d_no_derived()
        cube.coord("surface_altitude").rename("surface_air_pressure")
        cube.coord("surface_air_pressure").units = "Pa"
        cube.coord("sigma").units = "1"
        ptop_coord = iris.coords.AuxCoord(1000.0, var_name="ptop", units="Pa")
        cube.add_aux_coord(ptop_coord, ())
        cube.remove_coord("level_height")
        # Construct and add atmosphere sigma factory.
        factory = iris.aux_factory.AtmosphereSigmaFactory(
            cube.coord("ptop"),
            cube.coord("sigma"),
            cube.coord("surface_air_pressure"),
        )
        cube.add_aux_factory(factory)
        self.cube = cube

    def test_save(self):
        with self.temp_filename(suffix=".nc") as filename:
            iris.save(self.cube, filename)
            self.assertCDL(filename)

    def test_save_load_loop(self):
        # Ensure that the AtmosphereSigmaFactory is automatically loaded
        # when loading the file.
        with self.temp_filename(suffix=".nc") as filename:
            iris.save(self.cube, filename)
            cube = iris.load_cube(filename, "air_potential_temperature")
            assert cube.coords("air_pressure")


@tests.skip_data
class TestHybridPressure(tests.IrisTest):
    def setUp(self):
        # Modify stock cube so it is suitable to have a
        # hybrid pressure factory added to it.
        cube = stock.realistic_4d_no_derived()
        cube.coord("surface_altitude").rename("surface_air_pressure")
        cube.coord("surface_air_pressure").units = "Pa"
        cube.coord("level_height").rename("level_pressure")
        cube.coord("level_pressure").units = "Pa"
        # Construct and add hybrid pressure factory.
        factory = iris.aux_factory.HybridPressureFactory(
            cube.coord("level_pressure"),
            cube.coord("sigma"),
            cube.coord("surface_air_pressure"),
        )
        cube.add_aux_factory(factory)
        self.cube = cube

    def test_save(self):
        with self.temp_filename(suffix=".nc") as filename:
            iris.save(self.cube, filename)
            self.assertCDL(filename)

    def test_save_load_loop(self):
        # Tests an issue where the variable names in the formula
        # terms changed to the standard_names instead of the variable names
        # when loading a previously saved cube.
        with self.temp_filename(suffix=".nc") as filename, self.temp_filename(
            suffix=".nc"
        ) as other_filename:
            iris.save(self.cube, filename)
            cube = iris.load_cube(filename, "air_potential_temperature")
            iris.save(cube, other_filename)
            other_cube = iris.load_cube(
                other_filename, "air_potential_temperature"
            )
            self.assertEqual(cube, other_cube)


@tests.skip_data
class TestSaveMultipleAuxFactories(tests.IrisTest):
    def test_hybrid_height_and_pressure(self):
        cube = stock.realistic_4d()
        cube.add_aux_coord(
            iris.coords.DimCoord(
                1200.0, long_name="level_pressure", units="hPa"
            )
        )
        cube.add_aux_coord(
            iris.coords.DimCoord(0.5, long_name="other sigma", units="1")
        )
        cube.add_aux_coord(
            iris.coords.DimCoord(
                1000.0, long_name="surface_air_pressure", units="hPa"
            )
        )
        factory = iris.aux_factory.HybridPressureFactory(
            cube.coord("level_pressure"),
            cube.coord("other sigma"),
            cube.coord("surface_air_pressure"),
        )
        cube.add_aux_factory(factory)
        with self.temp_filename(suffix=".nc") as filename:
            iris.save(cube, filename)
            self.assertCDL(filename)

    def test_shared_primary(self):
        cube = stock.realistic_4d()
        factory = iris.aux_factory.HybridHeightFactory(
            cube.coord("level_height"),
            cube.coord("sigma"),
            cube.coord("surface_altitude"),
        )
        factory.rename("another altitude")
        cube.add_aux_factory(factory)
        with self.temp_filename(
            suffix=".nc"
        ) as filename, self.assertRaisesRegex(
            ValueError, "multiple aux factories"
        ):
            iris.save(cube, filename)

    def test_hybrid_height_cubes(self):
        hh1 = stock.simple_4d_with_hybrid_height()
        hh1.attributes["cube"] = "hh1"
        hh2 = stock.simple_4d_with_hybrid_height()
        hh2.attributes["cube"] = "hh2"
        sa = hh2.coord("surface_altitude")
        sa.points = sa.points * 10
        with self.temp_filename(".nc") as fname:
            iris.save([hh1, hh2], fname)
            cubes = iris.load(fname, "air_temperature")
            cubes = sorted(cubes, key=lambda cube: cube.attributes["cube"])
            self.assertCML(cubes)

    def test_hybrid_height_cubes_on_dimension_coordinate(self):
        hh1 = stock.hybrid_height()
        hh2 = stock.hybrid_height()
        sa = hh2.coord("surface_altitude")
        sa.points = sa.points * 10
        emsg = "Unable to create dimensonless vertical coordinate."
        with self.temp_filename(".nc") as fname, self.assertRaisesRegex(
            ValueError, emsg
        ):
            iris.save([hh1, hh2], fname)


class TestUmVersionAttribute(tests.IrisTest):
    def test_single_saves_as_global(self):
        cube = Cube(
            [1.0],
            standard_name="air_temperature",
            units="K",
            attributes={"um_version": "4.3"},
        )
        with self.temp_filename(".nc") as nc_path:
            iris.save(cube, nc_path)
            self.assertCDL(nc_path)

    def test_multiple_same_saves_as_global(self):
        cube_a = Cube(
            [1.0],
            standard_name="air_temperature",
            units="K",
            attributes={"um_version": "4.3"},
        )
        cube_b = Cube(
            [1.0],
            standard_name="air_pressure",
            units="hPa",
            attributes={"um_version": "4.3"},
        )
        with self.temp_filename(".nc") as nc_path:
            iris.save(CubeList([cube_a, cube_b]), nc_path)
            self.assertCDL(nc_path)

    def test_multiple_different_saves_on_variables(self):
        cube_a = Cube(
            [1.0],
            standard_name="air_temperature",
            units="K",
            attributes={"um_version": "4.3"},
        )
        cube_b = Cube(
            [1.0],
            standard_name="air_pressure",
            units="hPa",
            attributes={"um_version": "4.4"},
        )
        with self.temp_filename(".nc") as nc_path:
            iris.save(CubeList([cube_a, cube_b]), nc_path)
            self.assertCDL(nc_path)


@contextmanager
def _patch_site_configuration():
    def cf_patch_conventions(conventions):
        return ", ".join([conventions, "convention1, convention2"])

    def update(config):
        config["cf_profile"] = mock.Mock(name="cf_profile")
        config["cf_patch"] = mock.Mock(name="cf_patch")
        config["cf_patch_conventions"] = cf_patch_conventions

    orig_site_config = iris.site_configuration.copy()
    update(iris.site_configuration)
    yield
    iris.site_configuration = orig_site_config


class TestConventionsAttributes(tests.IrisTest):
    def test_patching_conventions_attribute(self):
        # Ensure that user defined conventions are wiped and those which are
        # saved patched through site_config can be loaded without an exception
        # being raised.
        cube = Cube(
            [1.0],
            standard_name="air_temperature",
            units="K",
            attributes={"Conventions": "some user defined conventions"},
        )

        # Patch the site configuration dictionary.
        with _patch_site_configuration(), self.temp_filename(".nc") as nc_path:
            iris.save(cube, nc_path)
            res = iris.load_cube(nc_path)

        self.assertEqual(
            res.attributes["Conventions"],
            "{}, {}, {}".format(
                CF_CONVENTIONS_VERSION, "convention1", "convention2"
            ),
        )


class TestLazySave(tests.IrisTest):
    @tests.skip_data
    def test_lazy_preserved_save(self):
        fpath = tests.get_data_path(
            ("NetCDF", "label_and_climate", "small_FC_167_mon_19601101.nc")
        )
        acube = iris.load_cube(fpath, "air_temperature")
        self.assertTrue(acube.has_lazy_data())
        # Also check a coord with lazy points + bounds.
        self.assertTrue(acube.coord("forecast_period").has_lazy_points())
        self.assertTrue(acube.coord("forecast_period").has_lazy_bounds())
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(acube)
        # Check that cube data is not realised, also coord points + bounds.
        self.assertTrue(acube.has_lazy_data())
        self.assertTrue(acube.coord("forecast_period").has_lazy_points())
        self.assertTrue(acube.coord("forecast_period").has_lazy_bounds())


@tests.skip_data
class TestCellMeasures(tests.IrisTest):
    def setUp(self):
        self.fname = tests.get_data_path(("NetCDF", "ORCA2", "votemper.nc"))

    def test_load_raw(self):
        (cube,) = iris.load_raw(self.fname)
        self.assertEqual(len(cube.cell_measures()), 1)
        self.assertEqual(cube.cell_measures()[0].measure, "area")

    def test_load(self):
        cube = iris.load_cube(self.fname)
        self.assertEqual(len(cube.cell_measures()), 1)
        self.assertEqual(cube.cell_measures()[0].measure, "area")

    def test_merge_cell_measure_aware(self):
        (cube1,) = iris.load_raw(self.fname)
        (cube2,) = iris.load_raw(self.fname)
        cube2._cell_measures_and_dims[0][0].var_name = "not_areat"
        cubes = CubeList([cube1, cube2]).merge()
        self.assertEqual(len(cubes), 2)

    def test_concatenate_cell_measure_aware(self):
        (cube1,) = iris.load_raw(self.fname)
        cube1 = cube1[:, :, 0, 0]
        cm_and_dims = cube1._cell_measures_and_dims
        (cube2,) = iris.load_raw(self.fname)
        cube2 = cube2[:, :, 0, 0]
        cube2._cell_measures_and_dims[0][0].var_name = "not_areat"
        cube2.coord("time").points = cube2.coord("time").points + 1
        cubes = CubeList([cube1, cube2]).concatenate()
        self.assertEqual(cubes[0]._cell_measures_and_dims, cm_and_dims)
        self.assertEqual(len(cubes), 2)

    def test_concatenate_cell_measure_match(self):
        (cube1,) = iris.load_raw(self.fname)
        cube1 = cube1[:, :, 0, 0]
        cm_and_dims = cube1._cell_measures_and_dims
        (cube2,) = iris.load_raw(self.fname)
        cube2 = cube2[:, :, 0, 0]
        cube2.coord("time").points = cube2.coord("time").points + 1
        cubes = CubeList([cube1, cube2]).concatenate()
        self.assertEqual(cubes[0]._cell_measures_and_dims, cm_and_dims)
        self.assertEqual(len(cubes), 1)

    def test_round_trip(self):
        (cube,) = iris.load(self.fname)
        with self.temp_filename(suffix=".nc") as filename:
            iris.save(cube, filename, unlimited_dimensions=[])
            (round_cube,) = iris.load_raw(filename)
            self.assertEqual(len(round_cube.cell_measures()), 1)
            self.assertEqual(round_cube.cell_measures()[0].measure, "area")

    def test_print(self):
        cube = iris.load_cube(self.fname)
        printed = cube.__str__()
        self.assertIn(
            (
                "Cell measures:\n"
                "        cell_area                             -         -    "
                "    x         x"
            ),
            printed,
        )


@tests.skip_data
class TestCMIP6VolcelloLoad(tests.IrisTest):
    def setUp(self):
        self.fname = tests.get_data_path(
            (
                "NetCDF",
                "volcello",
                "volcello_Ofx_CESM2_deforest-globe_r1i1p1f1_gn.nc",
            )
        )

    def test_cmip6_volcello_load_issue_3367(self):
        # Ensure that reading a file which references itself in
        # `cell_measures` can be read. At the same time, ensure that we
        # still receive a warning about other variables mentioned in
        # `cell_measures` i.e. a warning should be raised about missing
        # areacello.
        areacello_str = "areacello"
        volcello_str = "volcello"
        expected_msg = (
            "Missing CF-netCDF measure variable %r, "
            "referenced by netCDF variable %r" % (areacello_str, volcello_str)
        )

        with mock.patch("warnings.warn") as warn:
            # ensure file loads without failure
            cube = iris.load_cube(self.fname)
            warn.assert_has_calls([mock.call(expected_msg)])

        # extra check to ensure correct variable was found
        assert cube.standard_name == "ocean_volume"


class TestSelfReferencingVarLoad(tests.IrisTest):
    def setUp(self):
        self.temp_dir_path = os.path.join(
            tempfile.mkdtemp(), "issue_3367_volcello_test_file.nc"
        )
        dataset = nc.Dataset(self.temp_dir_path, "w")

        dataset.createDimension("lat", 4)
        dataset.createDimension("lon", 5)
        dataset.createDimension("lev", 3)

        latitudes = dataset.createVariable("lat", np.float64, ("lat",))
        longitudes = dataset.createVariable("lon", np.float64, ("lon",))
        levels = dataset.createVariable("lev", np.float64, ("lev",))
        volcello = dataset.createVariable(
            "volcello", np.float32, ("lat", "lon", "lev")
        )

        latitudes.standard_name = "latitude"
        latitudes.units = "degrees_north"
        latitudes.axis = "Y"
        latitudes[:] = np.linspace(-90, 90, 4)

        longitudes.standard_name = "longitude"
        longitudes.units = "degrees_east"
        longitudes.axis = "X"
        longitudes[:] = np.linspace(0, 360, 5)

        levels.standard_name = "olevel"
        levels.units = "centimeters"
        levels.positive = "down"
        levels.axis = "Z"
        levels[:] = np.linspace(0, 10**5, 3)

        volcello.id = "volcello"
        volcello.out_name = "volcello"
        volcello.standard_name = "ocean_volume"
        volcello.units = "m3"
        volcello.realm = "ocean"
        volcello.frequency = "fx"
        volcello.cell_measures = "area: areacello volume: volcello"
        volcello = np.arange(4 * 5 * 3).reshape((4, 5, 3))

        dataset.close()

    def test_self_referencing_load_issue_3367(self):
        # Ensure that reading a file which references itself in
        # `cell_measures` can be read. At the same time, ensure that we
        # still receive a warning about other variables mentioned in
        # `cell_measures` i.e. a warning should be raised about missing
        # areacello.
        areacello_str = "areacello"
        volcello_str = "volcello"
        expected_msg = (
            "Missing CF-netCDF measure variable %r, "
            "referenced by netCDF variable %r" % (areacello_str, volcello_str)
        )

        with mock.patch("warnings.warn") as warn:
            # ensure file loads without failure
            cube = iris.load_cube(self.temp_dir_path)
            warn.assert_called_with(expected_msg)

        # extra check to ensure correct variable was found
        assert cube.standard_name == "ocean_volume"

    def tearDown(self):
        os.remove(self.temp_dir_path)


class TestCellMethod_unknown(tests.IrisTest):
    def test_unknown_method(self):
        cube = Cube([1, 2], long_name="odd_phenomenon")
        cube.add_cell_method(CellMethod(method="oddity", coords=("x",)))
        temp_dirpath = tempfile.mkdtemp()
        try:
            temp_filepath = os.path.join(temp_dirpath, "tmp.nc")
            iris.save(cube, temp_filepath)
            with warnings.catch_warnings(record=True) as warning_records:
                iris.load(temp_filepath)
            # Filter to get the warning we are interested in.
            warning_messages = [record.message for record in warning_records]
            warning_messages = [
                warn
                for warn in warning_messages
                if isinstance(warn, UnknownCellMethodWarning)
            ]
            self.assertEqual(len(warning_messages), 1)
            message = warning_messages[0].args[0]
            msg = (
                "NetCDF variable 'odd_phenomenon' contains unknown cell "
                "method 'oddity'"
            )
            self.assertIn(msg, message)
        finally:
            shutil.rmtree(temp_dirpath)


@tests.skip_data
class TestCoordSystem(tests.IrisTest):
    def setUp(self):
        tlc.setUpModule()

    def tearDown(self):
        tlc.tearDownModule()

    def test_load_laea_grid(self):
        cube = iris.load_cube(
            tests.get_data_path(
                ("NetCDF", "lambert_azimuthal_equal_area", "euro_air_temp.nc")
            )
        )
        self.assertCML(cube, ("netcdf", "netcdf_laea.cml"))

    datum_cf_var_cdl = """
        netcdf output {
        dimensions:
            y = 4 ;
            x = 3 ;
        variables:
            float data(y, x) ;
                data :standard_name = "toa_brightness_temperature" ;
                data :units = "K" ;
                data :grid_mapping = "mercator" ;
            int mercator ;
                mercator:grid_mapping_name = "mercator" ;
                mercator:longitude_of_prime_meridian = 0. ;
                mercator:earth_radius = 6378169. ;
                mercator:horizontal_datum_name = "OSGB36" ;
            float y(y) ;
                y:axis = "Y" ;
                y:units = "m" ;
                y:standard_name = "projection_y_coordinate" ;
            float x(x) ;
                x:axis = "X" ;
                x:units = "m" ;
                x:standard_name = "projection_x_coordinate" ;

        // global attributes:
                :Conventions = "CF-1.7" ;
                :standard_name_vocabulary = "CF Standard Name Table v27" ;

        data:

        data =
        0, 1, 2,
        3, 4, 5,
        6, 7, 8,
        9, 10, 11 ;

        mercator = _ ;

        y = 1, 2, 3, 5 ;

        x = -6, -4, -2 ;

        }
    """

    datum_wkt_cdl = """
netcdf output5 {
dimensions:
    y = 4 ;
    x = 3 ;
variables:
    float data(y, x) ;
        data :standard_name = "toa_brightness_temperature" ;
        data :units = "K" ;
        data :grid_mapping = "mercator" ;
    int mercator ;
        mercator:grid_mapping_name = "mercator" ;
        mercator:longitude_of_prime_meridian = 0. ;
        mercator:earth_radius = 6378169. ;
        mercator:longitude_of_projection_origin = 0. ;
        mercator:false_easting = 0. ;
        mercator:false_northing = 0. ;
        mercator:scale_factor_at_projection_origin = 1. ;
        mercator:crs_wkt = "PROJCRS[\\"unknown\\",BASEGEOGCRS[\\"unknown\\",DATUM[\\"OSGB36\\",ELLIPSOID[\\"unknown\\",6378169,0,LENGTHUNIT[\\"metre\\",1,ID[\\"EPSG\\",9001]]]],PRIMEM[\\"Greenwich\\",0,ANGLEUNIT[\\"degree\\",0.0174532925199433],ID[\\"EPSG\\",8901]]],CONVERSION[\\"unknown\\",METHOD[\\"Mercator (variant B)\\",ID[\\"EPSG\\",9805]],PARAMETER[\\"Latitude of 1st standard parallel\\",0,ANGLEUNIT[\\"degree\\",0.0174532925199433],ID[\\"EPSG\\",8823]],PARAMETER[\\"Longitude of natural origin\\",0,ANGLEUNIT[\\"degree\\",0.0174532925199433],ID[\\"EPSG\\",8802]],PARAMETER[\\"False easting\\",0,LENGTHUNIT[\\"metre\\",1],ID[\\"EPSG\\",8806]],PARAMETER[\\"False northing\\",0,LENGTHUNIT[\\"metre\\",1],ID[\\"EPSG\\",8807]]],CS[Cartesian,2],AXIS[\\"(E)\\",east,ORDER[1],LENGTHUNIT[\\"metre\\",1,ID[\\"EPSG\\",9001]]],AXIS[\\"(N)\\",north,ORDER[2],LENGTHUNIT[\\"metre\\",1,ID[\\"EPSG\\",9001]]]]" ;
    float y(y) ;
        y:axis = "Y" ;
        y:units = "m" ;
        y:standard_name = "projection_y_coordinate" ;
    float x(x) ;
        x:axis = "X" ;
        x:units = "m" ;
        x:standard_name = "projection_x_coordinate" ;

// global attributes:
        :standard_name_vocabulary = "CF Standard Name Table v27" ;
        :Conventions = "CF-1.7" ;
data:

 data =
  0, 1, 2,
  3, 4, 5,
  6, 7, 8,
  9, 10, 11 ;

 mercator = _ ;

 y = 1, 2, 3, 5 ;

 x = -6, -4, -2 ;
}
    """

    def test_load_datum_wkt(self):
        expected = "OSGB 1936"
        nc_path = tlc.cdl_to_nc(self.datum_wkt_cdl)
        with iris.FUTURE.context(datum_support=True):
            cube = iris.load_cube(nc_path)
        test_crs = cube.coord("projection_y_coordinate").coord_system
        actual = str(test_crs.as_cartopy_crs().datum)
        self.assertMultiLineEqual(expected, actual)

    def test_no_load_datum_wkt(self):
        nc_path = tlc.cdl_to_nc(self.datum_wkt_cdl)
        with self.assertWarnsRegex(FutureWarning, "iris.FUTURE.datum_support"):
            cube = iris.load_cube(nc_path)
        test_crs = cube.coord("projection_y_coordinate").coord_system
        actual = str(test_crs.as_cartopy_crs().datum)
        self.assertMultiLineEqual(actual, "unknown")

    def test_load_datum_cf_var(self):
        expected = "OSGB 1936"
        nc_path = tlc.cdl_to_nc(self.datum_cf_var_cdl)
        with iris.FUTURE.context(datum_support=True):
            cube = iris.load_cube(nc_path)
        test_crs = cube.coord("projection_y_coordinate").coord_system
        actual = str(test_crs.as_cartopy_crs().datum)
        self.assertMultiLineEqual(expected, actual)

    def test_no_load_datum_cf_var(self):
        nc_path = tlc.cdl_to_nc(self.datum_cf_var_cdl)
        with self.assertWarnsRegex(FutureWarning, "iris.FUTURE.datum_support"):
            cube = iris.load_cube(nc_path)
        test_crs = cube.coord("projection_y_coordinate").coord_system
        actual = str(test_crs.as_cartopy_crs().datum)
        self.assertMultiLineEqual(actual, "unknown")

    def test_save_datum(self):
        expected = "OSGB 1936"
        saved_crs = iris.coord_systems.Mercator(
            ellipsoid=iris.coord_systems.GeogCS.from_datum("OSGB36")
        )

        base_cube = stock.realistic_3d()
        base_lat_coord = base_cube.coord("grid_latitude")
        test_lat_coord = DimCoord(
            base_lat_coord.points,
            standard_name="projection_y_coordinate",
            coord_system=saved_crs,
        )
        base_lon_coord = base_cube.coord("grid_longitude")
        test_lon_coord = DimCoord(
            base_lon_coord.points,
            standard_name="projection_x_coordinate",
            coord_system=saved_crs,
        )
        test_cube = Cube(
            base_cube.data,
            standard_name=base_cube.standard_name,
            units=base_cube.units,
            dim_coords_and_dims=(
                (base_cube.coord("time"), 0),
                (test_lat_coord, 1),
                (test_lon_coord, 2),
            ),
        )

        with self.temp_filename(suffix=".nc") as filename:
            iris.save(test_cube, filename)
            with iris.FUTURE.context(datum_support=True):
                cube = iris.load_cube(filename)

        test_crs = cube.coord("projection_y_coordinate").coord_system
        actual = str(test_crs.as_cartopy_crs().datum)
        self.assertMultiLineEqual(expected, actual)


def _get_scale_factor_add_offset(cube, datatype):
    """Utility function used by netCDF data packing tests."""
    if isinstance(datatype, dict):
        dt = np.dtype(datatype["dtype"])
    else:
        dt = np.dtype(datatype)
    cmax = cube.data.max()
    cmin = cube.data.min()
    n = dt.itemsize * 8
    if ma.isMaskedArray(cube.data):
        masked = True
    else:
        masked = False
    if masked:
        scale_factor = (cmax - cmin) / (2**n - 2)
    else:
        scale_factor = (cmax - cmin) / (2**n - 1)
    if dt.kind == "u":
        add_offset = cmin
    elif dt.kind == "i":
        if masked:
            add_offset = (cmax + cmin) / 2
        else:
            add_offset = cmin + 2 ** (n - 1) * scale_factor
    return (scale_factor, add_offset)


@tests.skip_data
class TestPackedData(tests.IrisTest):
    def _single_test(self, datatype, CDLfilename, manual=False):
        # Read PP input file.
        file_in = tests.get_data_path(
            (
                "PP",
                "cf_processing",
                "000003000000.03.236.000128.1990.12.01.00.00.b.pp",
            )
        )
        cube = iris.load_cube(file_in)
        scale_factor, offset = _get_scale_factor_add_offset(cube, datatype)
        if manual:
            packspec = dict(
                dtype=datatype, scale_factor=scale_factor, add_offset=offset
            )
        else:
            packspec = datatype
        # Write Cube to netCDF file.
        with self.temp_filename(suffix=".nc") as file_out:
            iris.save(cube, file_out, packing=packspec)
            decimal = int(-np.log10(scale_factor))
            packedcube = iris.load_cube(file_out)
            # Check that packed cube is accurate to expected precision
            self.assertArrayAlmostEqual(
                cube.data, packedcube.data, decimal=decimal
            )
            # Check the netCDF file against CDL expected output.
            self.assertCDL(
                file_out,
                ("integration", "netcdf", "TestPackedData", CDLfilename),
            )

    def test_single_packed_signed(self):
        """Test saving a single CF-netCDF file with packing."""
        self._single_test("i2", "single_packed_signed.cdl")

    def test_single_packed_unsigned(self):
        """Test saving a single CF-netCDF file with packing into unsigned."""
        self._single_test("u1", "single_packed_unsigned.cdl")

    def test_single_packed_manual_scale(self):
        """Test saving a single CF-netCDF file with packing with scale
        factor and add_offset set manually."""
        self._single_test("i2", "single_packed_manual.cdl", manual=True)

    def _multi_test(self, CDLfilename, multi_dtype=False):
        """Test saving multiple packed cubes with pack_dtype list."""
        # Read PP input file.
        file_in = tests.get_data_path(
            ("PP", "cf_processing", "abcza_pa19591997_daily_29.b.pp")
        )
        cubes = iris.load(file_in)
        # ensure cube order is the same:
        cubes.sort(key=lambda cube: cube.cell_methods[0].method)
        datatype = "i2"
        scale_factor, offset = _get_scale_factor_add_offset(cubes[0], datatype)
        if multi_dtype:
            packdict = dict(
                dtype=datatype, scale_factor=scale_factor, add_offset=offset
            )
            packspec = [packdict, None, "u2"]
            dtypes = packspec
        else:
            packspec = datatype
            dtypes = repeat(packspec)

        # Write Cube to netCDF file.
        with self.temp_filename(suffix=".nc") as file_out:
            iris.save(cubes, file_out, packing=packspec)
            # Check the netCDF file against CDL expected output.
            self.assertCDL(
                file_out,
                ("integration", "netcdf", "TestPackedData", CDLfilename),
            )
            packedcubes = iris.load(file_out)
            packedcubes.sort(key=lambda cube: cube.cell_methods[0].method)
            for cube, packedcube, dtype in zip(cubes, packedcubes, dtypes):
                if dtype:
                    sf, ao = _get_scale_factor_add_offset(cube, dtype)
                    decimal = int(-np.log10(sf))
                    # Check that packed cube is accurate to expected precision
                    self.assertArrayAlmostEqual(
                        cube.data, packedcube.data, decimal=decimal
                    )
                else:
                    self.assertArrayEqual(cube.data, packedcube.data)

    def test_multi_packed_single_dtype(self):
        """Test saving multiple packed cubes with the same pack_dtype."""
        # Read PP input file.
        self._multi_test("multi_packed_single_dtype.cdl")

    def test_multi_packed_multi_dtype(self):
        """Test saving multiple packed cubes with pack_dtype list."""
        # Read PP input file.
        self._multi_test("multi_packed_multi_dtype.cdl", multi_dtype=True)


class TestScalarCube(tests.IrisTest):
    def test_scalar_cube_save_load(self):
        cube = iris.cube.Cube(1, long_name="scalar_cube")
        with self.temp_filename(suffix=".nc") as fout:
            iris.save(cube, fout)
            scalar_cube = iris.load_cube(fout)
            self.assertEqual(scalar_cube.name(), "scalar_cube")


class TestStandardName(tests.IrisTest):
    def test_standard_name_roundtrip(self):
        standard_name = "air_temperature detection_minimum"
        cube = iris.cube.Cube(1, standard_name=standard_name)
        with self.temp_filename(suffix=".nc") as fout:
            iris.save(cube, fout)
            detection_limit_cube = iris.load_cube(fout)
            self.assertEqual(detection_limit_cube.standard_name, standard_name)


class TestLoadMinimalGeostationary(tests.IrisTest):
    """
    Check we can load data with a geostationary grid-mapping, even when the
    'false-easting' and 'false_northing' properties are missing.

    """

    _geostationary_problem_cdl = """
netcdf geostationary_problem_case {
dimensions:
    y = 2 ;
    x = 3 ;
variables:
    short radiance(y, x) ;
        radiance:standard_name = "toa_outgoing_radiance_per_unit_wavelength" ;
        radiance:units = "W m-2 sr-1 um-1" ;
        radiance:coordinates = "y x" ;
        radiance:grid_mapping = "imager_grid_mapping" ;
    short y(y) ;
        y:units = "rad" ;
        y:axis = "Y" ;
        y:long_name = "fixed grid projection y-coordinate" ;
        y:standard_name = "projection_y_coordinate" ;
    short x(x) ;
        x:units = "rad" ;
        x:axis = "X" ;
        x:long_name = "fixed grid projection x-coordinate" ;
        x:standard_name = "projection_x_coordinate" ;
    int imager_grid_mapping ;
        imager_grid_mapping:grid_mapping_name = "geostationary" ;
        imager_grid_mapping:perspective_point_height = 35786023. ;
        imager_grid_mapping:semi_major_axis = 6378137. ;
        imager_grid_mapping:semi_minor_axis = 6356752.31414 ;
        imager_grid_mapping:latitude_of_projection_origin = 0. ;
        imager_grid_mapping:longitude_of_projection_origin = -75. ;
        imager_grid_mapping:sweep_angle_axis = "x" ;

data:

 // coord values, just so these can be dim-coords
 y = 0, 1 ;
 x = 0, 1, 2 ;

}
"""

    @classmethod
    def setUpClass(cls):
        # Create a temp directory for transient test files.
        cls.temp_dir = tempfile.mkdtemp()
        cls.path_test_cdl = path_join(cls.temp_dir, "geos_problem.cdl")
        cls.path_test_nc = path_join(cls.temp_dir, "geos_problem.nc")
        # Create reference CDL and netcdf files from the CDL text.
        ncgen_from_cdl(
            cdl_str=cls._geostationary_problem_cdl,
            cdl_path=cls.path_test_cdl,
            nc_path=cls.path_test_nc,
        )

    @classmethod
    def tearDownClass(cls):
        # Destroy the temp directory.
        shutil.rmtree(cls.temp_dir)

    def test_geostationary_no_false_offsets(self):
        # Check we can load the test data and coordinate system properties are correct.
        cube = iris.load_cube(self.path_test_nc)
        # Check the coordinate system properties has the correct default properties.
        cs = cube.coord_system()
        self.assertIsInstance(cs, iris.coord_systems.Geostationary)
        self.assertEqual(cs.false_easting, 0.0)
        self.assertEqual(cs.false_northing, 0.0)


@tests.skip_data
class TestConstrainedLoad(tests.IrisTest):
    filename = tests.get_data_path(
        ("NetCDF", "label_and_climate", "A1B-99999a-river-sep-2070-2099.nc")
    )

    def test_netcdf_with_NameConstraint(self):
        constr = iris.NameConstraint(var_name="cdf_temp_dmax_tmean_abs")
        cubes = iris.load(self.filename, constr)
        self.assertEqual(len(cubes), 1)
        self.assertEqual(cubes[0].var_name, "cdf_temp_dmax_tmean_abs")

    def test_netcdf_with_no_constraint(self):
        cubes = iris.load(self.filename)
        self.assertEqual(len(cubes), 3)


class TestSkippedCoord:
    # If a coord/cell measure/etcetera cannot be added to the loaded Cube, a
    #  Warning is raised and the coord is skipped.
    # This 'catching' is generic to all CannotAddErrors, but currently the only
    #  such problem that can exist in a NetCDF file is a mismatch of dimensions
    #  between phenomenon and coord.

    cdl_core = """
dimensions:
    length_scale = 1 ;
    lat = 3 ;
variables:
    float lat(lat) ;
        lat:standard_name = "latitude" ;
        lat:units = "degrees_north" ;
    short lst_unc_sys(length_scale) ;
        lst_unc_sys:long_name = "uncertainty from large-scale systematic
        errors" ;
        lst_unc_sys:units = "kelvin" ;
        lst_unc_sys:coordinates = "lat" ;

data:
    lat = 0, 1, 2;
    """

    @pytest.fixture(autouse=True)
    def create_nc_file(self, tmp_path):
        file_name = "dim_mismatch"
        cdl = f"netcdf {file_name}" + "{\n" + self.cdl_core + "\n}"
        self.nc_path = (tmp_path / file_name).with_suffix(".nc")
        ncgen_from_cdl(
            cdl_str=cdl,
            cdl_path=None,
            nc_path=str(self.nc_path),
        )
        yield
        self.nc_path.unlink()

    def test_lat_not_loaded(self):
        # iris#5068 includes discussion of possible retention of the skipped
        #  coords in the future.
        with pytest.warns(
            match="Missing data dimensions for multi-valued DimCoord"
        ):
            cube = iris.load_cube(self.nc_path)
        with pytest.raises(iris.exceptions.CoordinateNotFoundError):
            _ = cube.coord("lat")


if __name__ == "__main__":
    tests.main()
