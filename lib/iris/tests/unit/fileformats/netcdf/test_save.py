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
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD
from iris.fileformats.netcdf import CF_CONVENTIONS_VERSION, save
from iris.tests.stock import lat_lon_cube
from iris.tests.stock.mesh import sample_mesh_cube


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
        with mock.patch("iris.fileformats.netcdf.saver.Saver") as Saver:
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
        with mock.patch("iris.fileformats.netcdf.saver.Saver") as Saver:
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
        with mock.patch("iris.fileformats.netcdf.saver.Saver") as Saver:
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
        with mock.patch("iris.fileformats.netcdf.saver.Saver") as Saver:
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
        with mock.patch("iris.fileformats.netcdf.saver.Saver"):
            with self.assertRaises(ValueError):
                save(cubes, "dummy.nc", fill_value=fill_values)


class Test_HdfSaveBug(tests.IrisTest):
    """
    Check for a known problem with netcdf4.

    If you create dimension with the same name as an existing variable, there
    is a specific problem, relating to HDF so limited to netcdf-4 formats.
    See : https://github.com/Unidata/netcdf-c/issues/1772

    In all these testcases, a straightforward translation to the file would be
    able to save [cube_2, cube_1], but *not* [cube_1, cube_2],
    because the latter creates a dim of the same name as the 'cube_1' data
    variable.

    Here, we are testing the specific workarounds in Iris netcdf save which
    avoids that problem.
    Unfortunately, owing to the complexity of the iris.fileformats.netcdf.Saver
    code, there are several separate places where this had to be fixed.

    N.B. we also check that the data (mostly) survives a save-load roundtrip.
    To identify the read-back cubes with the originals, we use var-names,
    which works because the save code opts to adjust dimension names _instead_.

    """

    def _check_save_and_reload(self, cubes):
        tempdir = Path(mkdtemp())
        filepath = tempdir / "tmp.nc"
        try:
            # Save the given cubes.
            save(cubes, filepath)

            # Load them back for roundtrip testing.
            with PARSE_UGRID_ON_LOAD.context():
                new_cubes = iris.load(str(filepath))

            # There should definitely still be the same number of cubes.
            self.assertEqual(len(new_cubes), len(cubes))

            # Get results in the input order, matching by var_names.
            result = [new_cubes.extract_cube(cube.var_name) for cube in cubes]

            # Check that input + output match cube-for-cube.
            # NB in this codeblock, before we destroy the temporary file.
            for cube_in, cube_out in zip(cubes, result):
                # Using special tolerant equivalence-check.
                self.assertSameCubes(cube_in, cube_out)

        finally:
            rmtree(tempdir)

        # Return result cubes for any additional checks.
        return result

    def assertSameCubes(self, cube1, cube2):
        """
        A special tolerant cube compare.

        Ignore any 'Conventions' attributes.
        Ignore all var-names.

        """

        def clean_cube(cube):
            cube = cube.copy()  # dont modify the original
            # Remove any 'Conventions' attributes
            cube.attributes.pop("Conventions", None)
            # Remove var-names (as original mesh components wouldn't have them)
            cube.var_name = None
            for coord in cube.coords():
                coord.var_name = None
            mesh = cube.mesh
            if mesh:
                mesh.var_name = None
                for component in mesh.coords() + mesh.connectivities():
                    component.var_name = None

            return cube

        self.assertEqual(clean_cube(cube1), clean_cube(cube2))

    def test_dimcoord_varname_collision(self):
        cube_2 = Cube([0, 1], var_name="cube_2")
        x_dim = DimCoord([0, 1], long_name="dim_x", var_name="dimco_name")
        cube_2.add_dim_coord(x_dim, 0)
        # First cube has a varname which collides with the dimcoord.
        cube_1 = Cube([0, 1], long_name="cube_1", var_name="dimco_name")
        # Test save + loadback
        reload_1, reload_2 = self._check_save_and_reload([cube_1, cube_2])
        # As re-loaded, the coord will have a different varname.
        self.assertEqual(reload_2.coord("dim_x").var_name, "dimco_name_0")

    def test_anonymous_dim_varname_collision(self):
        # Second cube is going to name an anonymous dim.
        cube_2 = Cube([0, 1], var_name="cube_2")
        # First cube has a varname which collides with the dim-name.
        cube_1 = Cube([0, 1], long_name="cube_1", var_name="dim0")
        # Add a dimcoord to prevent the *first* cube having an anonymous dim.
        x_dim = DimCoord([0, 1], long_name="dim_x", var_name="dimco_name")
        cube_1.add_dim_coord(x_dim, 0)
        # Test save + loadback
        self._check_save_and_reload([cube_1, cube_2])

    def test_bounds_dim_varname_collision(self):
        cube_2 = Cube([0, 1], var_name="cube_2")
        x_dim = DimCoord([0, 1], long_name="dim_x", var_name="dimco_name")
        x_dim.guess_bounds()
        cube_2.add_dim_coord(x_dim, 0)
        # First cube has a varname which collides with the bounds dimension.
        cube_1 = Cube([0], long_name="cube_1", var_name="bnds")
        # Test save + loadback
        self._check_save_and_reload([cube_1, cube_2])

    def test_string_dim_varname_collision(self):
        cube_2 = Cube([0, 1], var_name="cube_2")
        # NOTE: it *should* be possible for a cube with string data to cause
        # this collision, but cubes with string data are currently not working.
        # See : https://github.com/SciTools/iris/issues/4412
        x_dim = AuxCoord(
            ["this", "that"], long_name="dim_x", var_name="string_auxco"
        )
        cube_2.add_aux_coord(x_dim, 0)
        cube_1 = Cube([0], long_name="cube_1", var_name="string4")
        # Test save + loadback
        self._check_save_and_reload([cube_1, cube_2])

    def test_mesh_location_dim_varname_collision(self):
        cube_2 = sample_mesh_cube()
        cube_2.var_name = "cube_2"  # Make it identifiable
        cube_1 = Cube([0], long_name="cube_1", var_name="Mesh2d_node")
        # Test save + loadback
        self._check_save_and_reload([cube_1, cube_2])

    def test_connectivity_dim_varname_collision(self):
        cube_2 = sample_mesh_cube()
        cube_2.var_name = "cube_2"  # Make it identifiable
        cube_1 = Cube([0], long_name="cube_1", var_name="Mesh_2d_face_N_nodes")
        # Test save + loadback
        self._check_save_and_reload([cube_1, cube_2])


if __name__ == "__main__":
    tests.main()
