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
import inspect
from itertools import repeat
import os.path
from os.path import join as path_join
import shutil
import tempfile
from typing import Iterable, Optional, Union
from unittest import mock
import warnings

import netCDF4
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


# Attributes to test, which should be 'global' type by default
_GLOBAL_TEST_ATTRS = set(iris.fileformats.netcdf._CF_GLOBAL_ATTRS)
# Remove this one, which has peculiar behaviour + is tested separately
# N.B. this is not the same as 'Conventions', but is caught in the crossfire when that
# one is processed.
_GLOBAL_TEST_ATTRS -= set(["conventions"])


# Define a fixture to parametrise tests over the 'global-style' test attributes.
# This just provides a more concise way of writing parametrised tests.
@pytest.fixture(params=_GLOBAL_TEST_ATTRS)
def global_attr(request):
    # N.B. "request" is a standard PyTest fixture
    return request.param  # Return the name of the attribute to test.


# Attributes to test, which should be 'data' type by default
_DATA_TEST__ATTRS = (
    iris.fileformats.netcdf._CF_DATA_ATTRS
    + iris.fileformats.netcdf._UKMO_DATA_ATTRS
)


# Define a fixture to parametrise over the 'data-style' test attributes.
# This just provides a more concise way of writing parametrised tests.
@pytest.fixture(params=_DATA_TEST__ATTRS)
def data_attr(request):
    # N.B. "request" is a standard PyTest fixture
    return request.param  # Return the name of the attribute to test.


class TestLoadSaveAttributes:  # (tests.IrisTest):
    @staticmethod
    def _calling_testname():
        """
        Search up the callstack for a function named "test_*", and return the name for
        use as a test identifier.

        Returns
        -------
        test_name : str
            Returns a string, with the initial "test_" removed.
        """
        test_name = None
        stack = inspect.stack()
        for frame in stack[1:]:
            full_name = frame[3]
            if full_name.startswith("test_"):
                # Return the name with the inital "test_" removed.
                test_name = full_name.replace("test_", "")
                break
        # Search should not fail, unless we were called from an inappropriate place?
        assert test_name is not None
        return test_name

    def _testfile_path(self, basename: str) -> str:
        # Make a filepath in the temporary directory, based on the name of the calling
        # test method, and the "self.attrname" it sets up.
        testname = self._calling_testname()
        # Turn that into a suitable temporary filename
        ext_name = getattr(self, "testname_extension", "")
        if ext_name:
            basename = basename + "_" + ext_name
        path_str = (
            f"{self.tmpdir}/nc_attr__{self.attrname}__{testname}_{basename}.nc"
        )
        return path_str

    @staticmethod
    def _default_vars_and_attrvalues(vars_and_attrvalues):
        # Simple default strategy : turn a simple value into {'var': value}
        if not isinstance(vars_and_attrvalues, dict):
            # Treat single non-dict argument as a value for a single variable
            vars_and_attrvalues = {"var": vars_and_attrvalues}
        return vars_and_attrvalues

    def _create_testcase_files(
        self,
        global_attr_value: Optional[str] = None,
        vars_and_attrvalues: Union[None, str, dict] = None,
        globalval_file2: Optional[str] = None,
        var_values_file2: Union[None, str, dict] = None,
    ):
        """
        Create temporary input netcdf files with specific content.

        A generalised routine for creating a netcdf testfile to test behaviour of a
        specific attribute (name).
        Create a temporary input netcdf file (or two) with specific global and
        variable-local versions of a specific attribute.
        """
        # Make some input file paths.
        filepath1 = self._testfile_path("testfile")
        filepath2 = self._testfile_path("testfile2")

        def make_file(
            filepath: str, global_value=None, var_values=None
        ) -> str:
            ds = netCDF4.Dataset(filepath, "w")
            if global_value is not None:
                ds.setncattr(self.attrname, global_value)
            ds.createDimension("x", 3)
            # Rationalise the per-variable requirements
            # N.B. this *always* makes at least one variable, as otherwise we would
            # load no cubes.
            var_values = self._default_vars_and_attrvalues(var_values)
            for var_name, value in var_values.items():
                v = ds.createVariable(var_name, int, ("x",))
                if value is not None:
                    v.setncattr(self.attrname, value)
            ds.close()
            return filepath

        # Create one input file (always).
        filepaths = [
            make_file(
                filepath1,
                global_value=global_attr_value,
                var_values=vars_and_attrvalues,
            )
        ]
        if globalval_file2 is not None or var_values_file2 is not None:
            # Make a second testfile and add it to files-to-be-loaded.
            filepaths.append(
                make_file(
                    filepath2,
                    global_value=globalval_file2,
                    var_values=var_values_file2,
                ),
            )
        return filepaths

    def _roundtrip_load_and_save(
        self, input_filepaths: Union[str, Iterable[str]], output_filepath: str
    ) -> None:
        """
        Load netcdf input file(s) and re-write all to a given output file.
        """
        # Do a load+save to produce a testable output result in a new file.
        cubes = iris.load(input_filepaths)
        iris.save(cubes, output_filepath)

    def _print_files_debug(self):
        import os

        print("Inputs.....")
        for fp in self.input_filepaths + [self.result_filepath]:
            print(f"FILE>>>>>{fp}")
            os.system("ncdump -h " + fp)

    @pytest.fixture
    def attribute_testcase(self, tmp_path_factory):
        """
        Fixture to setup an individual testcase.
        Returns a callable to be used by the test routine to configure the testcase.

        N.B. "tmp_path_factory" is a standard PyTest fixture, providing a temporary
        dirpath shared by all tests, which is a bit quicker and more debuggable than
        having one-per-testcase.
        This fixture stores that path that on the instance, from where various
        subsidiary routines can get it, since this is always called first.
        """
        # Store the temporary directory path on the test instance
        self.tmpdir = str(tmp_path_factory.getbasetemp())

        # The fixture returns a callable, which is used to configure each testcase.
        def create_testcase_call(
            attr_name,
            global_attr_value=None,
            vars_and_attrvalues=None,
            globalval_file2=None,
            var_values_file2=None,
        ):
            """
            Initialise the testcase from the passed-in controls, configure the input
            files and run a save-load roundtrip to produce the output file.

            The name of the tested attribute and all the temporary filepaths are stored
            on the instance, from where check_expected_results can get them.

            """
            self.attrname = attr_name
            self.input_filepaths = self._create_testcase_files(
                global_attr_value=global_attr_value,
                vars_and_attrvalues=vars_and_attrvalues,
                globalval_file2=globalval_file2,
                var_values_file2=var_values_file2,
            )
            self.result_filepath = self._testfile_path("result")
            self._roundtrip_load_and_save(
                self.input_filepaths, self.result_filepath
            )
            # self._print_files_debug()
            return self.result_filepath

        return create_testcase_call

    def check_expected_results(
        self, global_attr_value=None, vars_and_attrvalues=None
    ):
        # The counterpart to _create_testcase, with similar control arguments.
        # Check existence (or not) and values of expected global and local attributes
        # in the test result file (in self.result_filepath).
        # N.B. only ever one result-file, but it can still have multiple variables
        ds = netCDF4.Dataset(self.result_filepath)
        if global_attr_value is None:
            assert self.attrname not in ds.ncattrs()
        else:
            assert self.attrname in ds.ncattrs()
            assert ds.getncattr(self.attrname) == global_attr_value
        if vars_and_attrvalues:
            vars_and_attrvalues = self._default_vars_and_attrvalues(
                vars_and_attrvalues
            )
            for var_name, value in vars_and_attrvalues.items():
                assert var_name in ds.variables
                v = ds.variables[var_name]
                if value is None:
                    assert self.attrname not in v.ncattrs()
                else:
                    assert self.attrname in v.ncattrs()
                    assert v.getncattr(self.attrname) == value

    def test_usertype_single_global(self, attribute_testcase):
        attribute_testcase(
            attr_name="myname",  # A generic "user" attribute with no special handling
            global_attr_value="single-value",
            vars_and_attrvalues={
                "myvar": None
            },  # the variable has no such attribute
        )
        # Default behaviour for a general global user-attribute.
        # It simply remains global.
        self.check_expected_results(
            global_attr_value="single-value",  # local values eclipse the global ones
            vars_and_attrvalues={
                "myvar": None
            },  # the variable has no such attribute
        )

    def test_usertype_single_local(self, attribute_testcase):
        # Default behaviour for a general local user-attribute.
        # It results in a "promoted" global attribute.
        attribute_testcase(
            attr_name="myname",  # A generic "user" attribute with no special handling
            vars_and_attrvalues={"myvar": "single-value"},
        )
        self.check_expected_results(
            global_attr_value="single-value",  # local values eclipse the global ones
            # N.B. the output var has NO such attribute
        )

    def test_usertype_multiple_different(self, attribute_testcase):
        # Default behaviour for general user-attributes.
        # The global attribute is lost because there are local ones.
        vars1 = {"f1_v1": "f1v1", "f1_v2": "f2v2"}
        vars2 = {"f2_v1": "x1", "f2_v2": "x2"}
        attribute_testcase(
            attr_name="random",  # A generic "user" attribute with no special handling
            global_attr_value="global_file1",
            vars_and_attrvalues=vars1,
            globalval_file2="global_file2",
            var_values_file2=vars2,
        )
        # combine all 4 vars in one dict
        all_vars_and_attrs = vars1.copy()
        all_vars_and_attrs.update(vars2)
        # TODO: replace with "|", when we drop Python 3.8
        # see: https://peps.python.org/pep-0584/
        # just check they are all there and distinct
        assert len(all_vars_and_attrs) == len(vars1) + len(vars2)
        self.check_expected_results(
            global_attr_value=None,  # local values eclipse the global ones
            vars_and_attrvalues=all_vars_and_attrs,
        )

    def test_usertype_matching_promoted(self, attribute_testcase):
        # matching local user-attributes are "promoted" to a global one.
        attribute_testcase(
            attr_name="random",
            global_attr_value="global_file1",
            vars_and_attrvalues={"v1": "same-value", "v2": "same-value"},
        )
        self.check_expected_results(
            global_attr_value="same-value",
            vars_and_attrvalues={"v1": None, "v2": None},
        )

    def test_usertype_matching_crossfile_promoted(self, attribute_testcase):
        # matching user-attributes are promoted, even across input files.
        attribute_testcase(
            attr_name="random",
            global_attr_value="global_file1",
            vars_and_attrvalues={"v1": "same-value", "v2": "same-value"},
            var_values_file2={"f2_v1": "same-value", "f2_v2": "same-value"},
        )
        self.check_expected_results(
            global_attr_value="same-value",
            vars_and_attrvalues={
                x: None for x in ("v1", "v2", "f2_v1", "f2_v2")
            },
        )

    def test_usertype_nonmatching_remainlocal(self, attribute_testcase):
        # Non-matching user attributes remain 'local' to the individual variables.
        attribute_testcase(
            attr_name="random",
            global_attr_value="global_file1",
            vars_and_attrvalues={"v1": "same-value", "v2": "different-value"},
        )
        self.check_expected_results(
            global_attr_value=None,  # NB it still destroys the global one !!
            vars_and_attrvalues={"v1": "same-value", "v2": "different-value"},
        )

    # #####################################
    # # WIP ...
    # # We have a number of different "classes" of recognised attributes which are
    # # handled differently.
    # # We may not test all cases, but only one of each "class".  Or we might be able
    # # to do them all (not yet clear how).
    #
    # #####################################
    # # === "Conventions" ===  - which is a case to itself
    # # Note: the usual 'Conventions' behaviour is already tested elsewhere
    # # - see TestConventionsAttributes above

    def test_conventions_var_local(self, attribute_testcase):
        # What happens if 'Conventions' appears as a variable-local attribute.
        # N.B. this is not good CF, but we'll see what happens anyway.
        attribute_testcase(
            attr_name="Conventions",
            global_attr_value=None,
            vars_and_attrvalues="user_set",
        )
        self.check_expected_results(
            global_attr_value="CF-1.7",  # this is standard output from
            vars_and_attrvalues=None,
        )

    def test_conventions_var_both(self, attribute_testcase):
        # What happens if 'Conventions' appears as both global + local attribute.
        attribute_testcase(
            attr_name="Conventions",
            global_attr_value="global-setting",
            vars_and_attrvalues="local-setting",
        )
        self.check_expected_results(
            global_attr_value="CF-1.7",  # this is standard output from
            vars_and_attrvalues=None,
        )

    #######################################################
    # Tests on "global" style attributes
    #  = those specific ones which 'ought' only to be global (except on collisions)
    #

    def test_globalstyle__global(self, global_attr, attribute_testcase):
        attr_content = f"Global tracked {global_attr}"
        attribute_testcase(
            attr_name=global_attr,
            global_attr_value=attr_content,
        )
        self.check_expected_results(global_attr_value=attr_content)

    def test_globalstyle__local(self, global_attr, attribute_testcase):
        # Strictly, not correct CF, but let's see what it does with it.
        attr_content = f"Local tracked {global_attr}"
        attribute_testcase(
            attr_name=global_attr,
            vars_and_attrvalues=attr_content,
        )
        self.check_expected_results(
            global_attr_value=attr_content
        )  # "promoted"

    def test_globalstyle__both(self, global_attr, attribute_testcase):
        attr_global = f"Global-{global_attr}"
        attr_local = f"Local-{global_attr}"
        attribute_testcase(
            attr_name=global_attr,
            global_attr_value=attr_global,
            vars_and_attrvalues=attr_local,
        )
        self.check_expected_results(
            global_attr_value=attr_local  # promoted local setting "wins"
        )

    def test_globalstyle__multivar_different(
        self, global_attr, attribute_testcase
    ):
        # Multiple *different* local settings are retained, not promoted
        attr_1 = f"Local-{global_attr}-1"
        attr_2 = f"Local-{global_attr}-2"
        with pytest.warns(
            UserWarning, match="should only be a CF global attribute"
        ):
            # A warning should be raised when writing the result.
            attribute_testcase(
                attr_name=global_attr,
                vars_and_attrvalues={"v1": attr_1, "v2": attr_2},
            )
        self.check_expected_results(
            global_attr_value=None,
            vars_and_attrvalues={"v1": attr_1, "v2": attr_2},
        )

    def test_globalstyle__multivar_same(self, global_attr, attribute_testcase):
        # Multiple *same* local settings are promoted to a common global one
        attrval = f"Locally-defined-{global_attr}"
        attribute_testcase(
            attr_name=global_attr,
            vars_and_attrvalues={"v1": attrval, "v2": attrval},
        )
        self.check_expected_results(
            global_attr_value=attrval,
            vars_and_attrvalues={"v1": None, "v2": None},
        )

    def test_globalstyle__multifile_different(
        self, global_attr, attribute_testcase
    ):
        # Different global attributes from multiple files are retained as local ones
        attr_1 = f"Global-{global_attr}-1"
        attr_2 = f"Global-{global_attr}-2"
        with pytest.warns(
            UserWarning, match="should only be a CF global attribute"
        ):
            # A warning should be raised when writing the result.
            attribute_testcase(
                attr_name=global_attr,
                global_attr_value=attr_1,
                vars_and_attrvalues={"v1": None},
                globalval_file2=attr_2,
                var_values_file2={"v2": None},
            )
        self.check_expected_results(
            # Combining them "demotes" the common global attributes to local ones
            vars_and_attrvalues={"v1": attr_1, "v2": attr_2}
        )

    def test_globalstyle__multifile_same(
        self, global_attr, attribute_testcase
    ):
        # Matching global attributes in multiple files are retained as global
        attrval = f"Global-{global_attr}"
        attribute_testcase(
            attr_name=global_attr,
            global_attr_value=attrval,
            vars_and_attrvalues={"v1": None},
            globalval_file2=attrval,
            var_values_file2={"v2": None},
        )
        self.check_expected_results(
            # Combining them "demotes" the common global attributes to local ones
            global_attr_value=attrval,
            vars_and_attrvalues={"v1": None, "v2": None},
        )

    #######################################################
    # Tests on "data" style attributes
    #  = those specific ones which 'ought' only to be data-local
    #

    @pytest.mark.parametrize("origin_style", ["input_global", "input_local"])
    def test_datastyle(self, data_attr, attribute_testcase, origin_style):
        # data-style attributes should *not* get 'promoted' to global ones
        # Set the name extension to avoid tests with different 'style' params having
        # collissions over identical testfile names
        self.testname_extension = origin_style

        attrval = f"Attr-setting-{data_attr}"
        if data_attr == "missing_value":
            # Special-cases : 'missing_value' type must be compatible with the variable
            attrval = 303
        elif data_attr == "ukmo__process_flags":
            # What this does when a GLOBAL attr seems to be weird + unintended.
            # 'this' --> 't h i s'
            attrval = "process"
            # NOTE: it's also supposed to handle vector values - which we are not
            # testing.

        # NOTE: results *should* be the same whether the original attribute is written
        # as global or a variable attribute
        if origin_style == "input_global":
            # Record in source as a global attribute
            attribute_testcase(attr_name=data_attr, global_attr_value=attrval)
        else:
            assert origin_style == "input_local"
            # Record in source as a variable-local attribute
            attribute_testcase(
                attr_name=data_attr, vars_and_attrvalues=attrval
            )

        if data_attr in iris.fileformats.netcdf._CF_DATA_ATTRS:
            # These ones are simply discarded on loading.
            # By experiment, this overlap between _CF_ATTRS and _CF_DATA_ATTRS
            # currently contains only 'missing_value' and 'standard_error_multiplier'.
            expect_global = None
            expect_var = None
        else:
            expect_global = None
            if (
                data_attr == "ukmo__process_flags"
                and origin_style == "input_global"
            ):
                # This is very odd behaviour + surely unintended.
                # It's supposed to handle vector values (which we are not checking).
                # But the weird behaviour only applies to the 'global' test, which is
                # obviously not normal usage anyway.
                attrval = "p r o c e s s"
            expect_var = attrval

        if data_attr == "STASH":
            # A special case, output translates this to a different attribute name.
            self.attrname = "um_stash_source"

        self.check_expected_results(
            global_attr_value=expect_global,
            vars_and_attrvalues=expect_var,
        )


if __name__ == "__main__":
    tests.main()
