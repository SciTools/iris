# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Integration tests for loading and saving netcdf files."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from contextlib import contextmanager
from itertools import repeat
import os.path
from os.path import join as path_join
import shutil
from subprocess import check_call
import tempfile
from unittest import mock
import warnings

import netCDF4 as nc
import numpy as np
import numpy.ma as ma

import iris
from iris.coords import CellMethod
from iris.cube import Cube, CubeList
from iris.fileformats.netcdf import CF_CONVENTIONS_VERSION
from iris.fileformats.netcdf import Saver
from iris.fileformats.netcdf import UnknownCellMethodWarning
import iris.tests.stock as stock


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
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(acube)
        self.assertTrue(acube.has_lazy_data())


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
        self.assertTrue(
            (
                "\n     Cell measures:\n          cell_area"
                "                           -         -    "
                "    x         x"
            )
            in printed
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
        levels[:] = np.linspace(0, 10 ** 5, 3)

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
    def test_load_laea_grid(self):
        cube = iris.load_cube(
            tests.get_data_path(
                ("NetCDF", "lambert_azimuthal_equal_area", "euro_air_temp.nc")
            )
        )
        self.assertCML(cube, ("netcdf", "netcdf_laea.cml"))


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
        scale_factor = (cmax - cmin) / (2 ** n - 2)
    else:
        scale_factor = (cmax - cmin) / (2 ** n - 1)
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
        """Test saving a single CF-netCDF file with packing into unsigned. """
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
        # Create a reference file from the CDL text.
        with open(cls.path_test_cdl, "w") as f_out:
            f_out.write(cls._geostationary_problem_cdl)
        # Call 'ncgen' to make an actual netCDF file from the CDL.
        command = "ncgen -o {} {}".format(cls.path_test_nc, cls.path_test_cdl)
        check_call(command, shell=True)

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


if __name__ == "__main__":
    tests.main()
