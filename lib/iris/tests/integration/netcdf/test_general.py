# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Integration tests for loading and saving netcdf files."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from itertools import repeat
import os.path
import shutil
import tempfile
import warnings

import numpy as np
import numpy.ma as ma
import pytest

import iris
import iris.coord_systems
from iris.coords import CellMethod
from iris.cube import Cube, CubeList
import iris.exceptions
from iris.fileformats.netcdf import Saver, UnknownCellMethodWarning
from iris.tests.stock.netcdf import ncgen_from_cdl


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
                (
                    "integration",
                    "netcdf",
                    "general",
                    "TestPackedData",
                    CDLfilename,
                ),
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
                (
                    "integration",
                    "netcdf",
                    "general",
                    "TestPackedData",
                    CDLfilename,
                ),
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
