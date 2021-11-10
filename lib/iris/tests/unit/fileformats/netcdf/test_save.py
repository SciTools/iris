# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.fileformats.netcdf.save` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from unittest import mock

import netCDF4 as nc
import numpy as np

import iris
from iris.coords import DimCoord
from iris.cube import Cube, CubeList
from iris.fileformats.netcdf import CF_CONVENTIONS_VERSION, save
from iris.tests.stock import lat_lon_cube


class Test_conventions(tests.IrisTest):
    def setUp(self):
        self.cube = Cube([0])
        self.custom_conventions = "convention1 convention2"
        self.cube.attributes["Conventions"] = self.custom_conventions
        self.options = iris.config.netcdf

    def test_custom_conventions__ignored(self):
        # Ensure that we drop existing conventions attributes and replace with
        # CF convention.
        with self.temp_filename(".nc") as nc_path:
            save(self.cube, nc_path, "NETCDF4")
            ds = nc.Dataset(nc_path)
            res = ds.getncattr("Conventions")
            ds.close()
        self.assertEqual(res, CF_CONVENTIONS_VERSION)

    def test_custom_conventions__allowed(self):
        # Ensure that existing conventions attributes are passed through if the
        # relevant Iris option is set.
        with mock.patch.object(self.options, "conventions_override", True):
            with self.temp_filename(".nc") as nc_path:
                save(self.cube, nc_path, "NETCDF4")
                ds = nc.Dataset(nc_path)
                res = ds.getncattr("Conventions")
                ds.close()
        self.assertEqual(res, self.custom_conventions)

    def test_custom_conventions__allowed__missing(self):
        # Ensure the default conventions attribute is set if the relevant Iris
        # option is set but there is no custom conventions attribute.
        del self.cube.attributes["Conventions"]
        with mock.patch.object(self.options, "conventions_override", True):
            with self.temp_filename(".nc") as nc_path:
                save(self.cube, nc_path, "NETCDF4")
                ds = nc.Dataset(nc_path)
                res = ds.getncattr("Conventions")
                ds.close()
        self.assertEqual(res, CF_CONVENTIONS_VERSION)


class Test_attributes(tests.IrisTest):
    def test_attributes_arrays(self):
        # Ensure that attributes containing NumPy arrays can be equality
        # checked and their cubes saved as appropriate.
        c1 = Cube([1], attributes={"bar": np.arange(2)})
        c2 = Cube([2], attributes={"bar": np.arange(2)})

        with self.temp_filename("foo.nc") as nc_out:
            save([c1, c2], nc_out)
            ds = nc.Dataset(nc_out)
            res = ds.getncattr("bar")
            ds.close()
        self.assertArrayEqual(res, np.arange(2))

    def test_no_special_attribute_clash(self):
        # Ensure that saving multiple cubes with netCDF4 protected attributes
        # works as expected.
        # Note that here we are testing variable attribute clashes only - by
        # saving multiple cubes the attributes are saved as variable
        # attributes rather than global attributes.
        c1 = Cube([0], var_name="test", attributes={"name": "bar"})
        c2 = Cube([0], var_name="test_1", attributes={"name": "bar_1"})

        with self.temp_filename("foo.nc") as nc_out:
            save([c1, c2], nc_out)
            ds = nc.Dataset(nc_out)
            res = ds.variables["test"].getncattr("name")
            res_1 = ds.variables["test_1"].getncattr("name")
            ds.close()
        self.assertEqual(res, "bar")
        self.assertEqual(res_1, "bar_1")


class Test_unlimited_dims(tests.IrisTest):
    def test_no_unlimited_dims(self):
        cube = lat_lon_cube()
        with self.temp_filename("foo.nc") as nc_out:
            save(cube, nc_out)
            ds = nc.Dataset(nc_out)
            self.assertFalse(ds.dimensions["latitude"].isunlimited())

    def test_unlimited_dim_latitude(self):
        cube = lat_lon_cube()
        unlim_dim_name = "latitude"
        with self.temp_filename("foo.nc") as nc_out:
            save(cube, nc_out, unlimited_dimensions=[unlim_dim_name])
            ds = nc.Dataset(nc_out)
            self.assertTrue(ds.dimensions[unlim_dim_name].isunlimited())


class Test_fill_value(tests.IrisTest):
    def setUp(self):
        self.standard_names = [
            "air_temperature",
            "air_potential_temperature",
            "air_temperature_anomaly",
        ]

    def _make_cubes(self):
        lat = DimCoord(np.arange(3), "latitude", units="degrees")
        lon = DimCoord(np.arange(4), "longitude", units="degrees")
        data = np.arange(12, dtype="f4").reshape(3, 4)
        return CubeList(
            Cube(
                data,
                standard_name=name,
                units="K",
                dim_coords_and_dims=[(lat, 0), (lon, 1)],
            )
            for name in self.standard_names
        )

    def test_None(self):
        # Test that when no fill_value argument is passed, the fill_value
        # argument to Saver.write is None or not present.
        cubes = self._make_cubes()
        with mock.patch("iris.fileformats.netcdf.Saver") as Saver:
            save(cubes, "dummy.nc")

        # Get the Saver.write mock
        with Saver() as saver:
            write = saver.write

        self.assertEqual(3, write.call_count)
        for call in write.mock_calls:
            _, _, kwargs = call
            if "fill_value" in kwargs:
                self.assertIs(None, kwargs["fill_value"])

    def test_single(self):
        # Test that when a single value is passed as the fill_value argument,
        # that value is passed to each call to Saver.write
        cubes = self._make_cubes()
        fill_value = 12345.0
        with mock.patch("iris.fileformats.netcdf.Saver") as Saver:
            save(cubes, "dummy.nc", fill_value=fill_value)

        # Get the Saver.write mock
        with Saver() as saver:
            write = saver.write

        self.assertEqual(3, write.call_count)
        for call in write.mock_calls:
            _, _, kwargs = call
            self.assertEqual(fill_value, kwargs["fill_value"])

    def test_multiple(self):
        # Test that when a list is passed as the fill_value argument,
        # each element is passed to separate calls to Saver.write
        cubes = self._make_cubes()
        fill_values = [123.0, 456.0, 789.0]
        with mock.patch("iris.fileformats.netcdf.Saver") as Saver:
            save(cubes, "dummy.nc", fill_value=fill_values)

        # Get the Saver.write mock
        with Saver() as saver:
            write = saver.write

        self.assertEqual(3, write.call_count)
        for call, fill_value in zip(write.mock_calls, fill_values):
            _, _, kwargs = call
            self.assertEqual(fill_value, kwargs["fill_value"])

    def test_single_string(self):
        # Test that when a string is passed as the fill_value argument,
        # that value is passed to calls to Saver.write
        cube = Cube(["abc", "def", "hij"])
        fill_value = "xyz"
        with mock.patch("iris.fileformats.netcdf.Saver") as Saver:
            save(cube, "dummy.nc", fill_value=fill_value)

        # Get the Saver.write mock
        with Saver() as saver:
            write = saver.write

        self.assertEqual(1, write.call_count)
        _, _, kwargs = write.mock_calls[0]
        self.assertEqual(fill_value, kwargs["fill_value"])

    def test_multi_wrong_length(self):
        # Test that when a list of a different length to the number of cubes
        # is passed as the fill_value argument, an error is raised
        cubes = self._make_cubes()
        fill_values = [1.0, 2.0, 3.0, 4.0]
        with mock.patch("iris.fileformats.netcdf.Saver"):
            with self.assertRaises(ValueError):
                save(cubes, "dummy.nc", fill_value=fill_values)


class Test_HdfSaveBug(tests.IrisTest):
    """
    Check for a known problem with netcdf4.

    If you create dimension with the same name as an existing variable, there
    is a specific problem, relating to HDF so limited to netcdf-4 formats.
    See : https://github.com/Unidata/netcdf-c/issues/1772

    """

    def test_problem_case(self):
        n_x = 2
        x_dim = DimCoord(
            np.arange(n_x), long_name="dim_x", var_name="same_name"
        )
        cube_x = Cube(np.arange(n_x), long_name="cube_x", var_name="same_name")
        cube_y = Cube(
            np.arange(n_x), long_name="cube_y", var_name="different_name"
        )
        cube_y.add_dim_coord(x_dim, 0)
        # In this case, a straightforward translation to the file will be able
        # to save [cube_y, cube_x], but *not* [cube_x, cube_y], because the
        # latter makes a dim of the same name as the 'cube_x' data variable.
        # Here, we are testing the specific workaround in Iris netcdf save which
        # avoids that problem.
        tempdir = Path(mkdtemp())
        filepath = tempdir / "tmp.nc"
        try:
            cubes = [cube_x, cube_y]
            save(cubes, filepath)  # , netcdf_format='NETCDF3_CLASSIC')
            # from subprocess import check_call
            # check_call(f'ncdump -h {filepath}', shell=True)
            loadback_cube_y = iris.load_cube(str(filepath), "cube_y")
            self.assertEqual(
                loadback_cube_y.coord("dim_x").var_name, "same_name_0"
            )
        finally:
            rmtree(tempdir)


if __name__ == "__main__":
    tests.main()
