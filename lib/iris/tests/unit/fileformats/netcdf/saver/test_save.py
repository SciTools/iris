# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :func:`iris.fileformats.netcdf.save` function."""

import numpy as np
import pytest

import iris
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList
from iris.fileformats.netcdf import CF_CONVENTIONS_VERSION, Saver, _thread_safe_nc, save
from iris.tests import _shared_utils
from iris.tests.stock import lat_lon_cube
from iris.tests.stock.mesh import sample_mesh_cube


class Test_conventions:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = Cube([0])
        self.custom_conventions = "convention1 convention2"
        self.cube.attributes["Conventions"] = self.custom_conventions
        self.options = iris.config.netcdf

    def test_custom_conventions__ignored(self, tmp_path):
        # Ensure that we drop existing conventions attributes and replace with
        # CF convention.
        nc_path = tmp_path / "dummy.nc"
        save(self.cube, nc_path, "NETCDF4")
        ds = _thread_safe_nc.DatasetWrapper(nc_path)
        res = ds.getncattr("Conventions")
        ds.close()
        assert res == CF_CONVENTIONS_VERSION

    def test_custom_conventions__allowed(self, mocker, tmp_path):
        # Ensure that existing conventions attributes are passed through if the
        # relevant Iris option is set.
        nc_path = tmp_path / "dummy.nc"
        mocker.patch.object(self.options, "conventions_override", True)
        save(self.cube, nc_path, "NETCDF4")
        ds = _thread_safe_nc.DatasetWrapper(nc_path)
        res = ds.getncattr("Conventions")
        ds.close()
        assert res == self.custom_conventions

    def test_custom_conventions__allowed__missing(self, mocker, tmp_path):
        # Ensure the default conventions attribute is set if the relevant Iris
        # option is set but there is no custom conventions attribute.
        del self.cube.attributes["Conventions"]
        mocker.patch.object(self.options, "conventions_override", True)
        nc_path = tmp_path / "dummy.nc"
        save(self.cube, nc_path, "NETCDF4")
        ds = _thread_safe_nc.DatasetWrapper(nc_path)
        res = ds.getncattr("Conventions")
        ds.close()
        assert res == CF_CONVENTIONS_VERSION


class Test_attributes:
    def test_attributes_arrays(self, tmp_path):
        # Ensure that attributes containing NumPy arrays can be equality
        # checked and their cubes saved as appropriate.
        c1 = Cube([1], attributes={"bar": np.arange(2)})
        c2 = Cube([2], attributes={"bar": np.arange(2)})

        nc_out = tmp_path / "foo.nc"
        save([c1, c2], nc_out)
        ds = _thread_safe_nc.DatasetWrapper(nc_out)
        res = ds.getncattr("bar")
        ds.close()
        _shared_utils.assert_array_equal(res, np.arange(2))

    def test_attributes_arrays_incompatible_shapes(self, tmp_path):
        # Ensure successful comparison without raising a broadcast error.
        c1 = Cube([1], attributes={"bar": np.arange(2)})
        c2 = Cube([2], attributes={"bar": np.arange(3)})

        nc_out = tmp_path / "foo.nc"
        save([c1, c2], nc_out)
        ds = _thread_safe_nc.DatasetWrapper(nc_out)
        with pytest.raises(AttributeError):
            _ = ds.getncattr("bar")
        for var in ds.variables.values():
            res = var.getncattr("bar")
            assert isinstance(res, np.ndarray)
        ds.close()

    def test_no_special_attribute_clash(self, tmp_path):
        # Ensure that saving multiple cubes with netCDF4 protected attributes
        # works as expected.
        # Note that here we are testing variable attribute clashes only - by
        # saving multiple cubes the attributes are saved as variable
        # attributes rather than global attributes.
        c1 = Cube([0], var_name="test", attributes={"name": "bar"})
        c2 = Cube([0], var_name="test_1", attributes={"name": "bar_1"})

        nc_out = tmp_path / "foo.nc"
        save([c1, c2], nc_out)
        ds = _thread_safe_nc.DatasetWrapper(nc_out)
        res = ds.variables["test"].getncattr("name")
        res_1 = ds.variables["test_1"].getncattr("name")
        ds.close()
        assert res == "bar"
        assert res_1 == "bar_1"


class Test_unlimited_dims:
    def test_no_unlimited_dims(self, tmp_path):
        cube = lat_lon_cube()
        nc_out = tmp_path / "foo.nc"
        save(cube, nc_out)
        ds = _thread_safe_nc.DatasetWrapper(nc_out)
        assert not ds.dimensions["latitude"].isunlimited()

    def test_unlimited_dim_latitude(self, tmp_path):
        cube = lat_lon_cube()
        unlim_dim_name = "latitude"
        nc_out = tmp_path / "foo.nc"
        save(cube, nc_out, unlimited_dimensions=[unlim_dim_name])
        ds = _thread_safe_nc.DatasetWrapper(nc_out)
        assert ds.dimensions[unlim_dim_name].isunlimited()


class Test_fill_value:
    @pytest.fixture(autouse=True)
    def _setup(self):
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

    def test_none(self, mocker):
        # Test that when no fill_value argument is passed, the fill_value
        # argument to Saver.write is None or not present.
        cubes = self._make_cubes()
        Saver = mocker.patch("iris.fileformats.netcdf.saver.Saver")
        save(cubes, "dummy.nc")

        # Get the Saver.write mock
        with Saver() as saver:
            write = saver.write

        assert 3 == write.call_count
        for call in write.mock_calls:
            _, _, kwargs = call
            if "fill_value" in kwargs:
                assert None is kwargs["fill_value"]

    def test_single(self, mocker):
        # Test that when a single value is passed as the fill_value argument,
        # that value is passed to each call to Saver.write
        cubes = self._make_cubes()
        fill_value = 12345.0
        Saver = mocker.patch("iris.fileformats.netcdf.saver.Saver")
        save(cubes, "dummy.nc", fill_value=fill_value)

        # Get the Saver.write mock
        with Saver() as saver:
            write = saver.write

        assert 3 == write.call_count
        for call in write.mock_calls:
            _, _, kwargs = call
            assert fill_value == kwargs["fill_value"]

    def test_multiple(self, mocker):
        # Test that when a list is passed as the fill_value argument,
        # each element is passed to separate calls to Saver.write
        cubes = self._make_cubes()
        fill_values = [123.0, 456.0, 789.0]
        Saver = mocker.patch("iris.fileformats.netcdf.saver.Saver")
        save(cubes, "dummy.nc", fill_value=fill_values)

        # Get the Saver.write mock
        with Saver() as saver:
            write = saver.write

        assert 3 == write.call_count
        for call, fill_value in zip(write.mock_calls, fill_values):
            _, _, kwargs = call
            assert fill_value == kwargs["fill_value"]

    def test_single_string(self, mocker):
        # Test that when a string is passed as the fill_value argument,
        # that value is passed to calls to Saver.write
        cube = Cube(["abc", "def", "hij"])
        fill_value = "xyz"
        Saver = mocker.patch("iris.fileformats.netcdf.saver.Saver")
        save(cube, "dummy.nc", fill_value=fill_value)

        # Get the Saver.write mock
        with Saver() as saver:
            write = saver.write

        assert 1 == write.call_count
        _, _, kwargs = write.mock_calls[0]
        assert fill_value == kwargs["fill_value"]

    def test_multi_wrong_length(self, mocker):
        # Test that when a list of a different length to the number of cubes
        # is passed as the fill_value argument, an error is raised
        cubes = self._make_cubes()
        fill_values = [1.0, 2.0, 3.0, 4.0]
        msg = "If fill_value is a list, it must have the same number of elements as the cube argument."
        with mocker.patch("iris.fileformats.netcdf.saver.Saver"):
            with pytest.raises(ValueError, match=msg):
                save(cubes, "dummy.nc", fill_value=fill_values)


class Test_HdfSaveBug:
    """Check for a known problem with netcdf4.

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

    @pytest.fixture
    def _check_save_and_reload(self, tmp_path):
        def check_save_and_reload(cubes):
            filepath = tmp_path / "temp.nc"
            # Save the given cubes.
            save(cubes, filepath)

            # Load them back for roundtrip testing.
            new_cubes = iris.load(str(filepath))

            # There should definitely still be the same number of cubes.
            assert len(new_cubes) == len(cubes)

            # Get results in the input order, matching by var_names.
            result = [new_cubes.extract_cube(cube.var_name) for cube in cubes]

            # Check that input + output match cube-for-cube.
            # NB in this codeblock, before we destroy the temporary file.
            for cube_in, cube_out in zip(cubes, result):
                # Using special tolerant equivalence-check.
                self.assert_same_cubes(cube_in, cube_out)

            # Return result cubes for any additional checks.
            return result

        return check_save_and_reload

    def assert_same_cubes(self, cube1, cube2):
        """A special tolerant cube compare.

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

        assert clean_cube(cube1) == clean_cube(cube2)

    def test_dimcoord_varname_collision(self, _check_save_and_reload):
        cube_2 = Cube([0, 1], var_name="cube_2")
        x_dim = DimCoord([0, 1], long_name="dim_x", var_name="dimco_name")
        cube_2.add_dim_coord(x_dim, 0)
        # First cube has a varname which collides with the dimcoord.
        cube_1 = Cube([0, 1], long_name="cube_1", var_name="dimco_name")
        # Test save + loadback
        reload_1, reload_2 = _check_save_and_reload([cube_1, cube_2])
        # As re-loaded, the coord will have a different varname.
        assert reload_2.coord("dim_x").var_name == "dimco_name_0"

    def test_anonymous_dim_varname_collision(self, _check_save_and_reload):
        # Second cube is going to name an anonymous dim.
        cube_2 = Cube([0, 1], var_name="cube_2")
        # First cube has a varname which collides with the dim-name.
        cube_1 = Cube([0, 1], long_name="cube_1", var_name="dim0")
        # Add a dimcoord to prevent the *first* cube having an anonymous dim.
        x_dim = DimCoord([0, 1], long_name="dim_x", var_name="dimco_name")
        cube_1.add_dim_coord(x_dim, 0)
        # Test save + loadback
        _check_save_and_reload([cube_1, cube_2])

    def test_bounds_dim_varname_collision(self, _check_save_and_reload):
        cube_2 = Cube([0, 1], var_name="cube_2")
        x_dim = DimCoord([0, 1], long_name="dim_x", var_name="dimco_name")
        x_dim.guess_bounds()
        cube_2.add_dim_coord(x_dim, 0)
        # First cube has a varname which collides with the bounds dimension.
        cube_1 = Cube([0], long_name="cube_1", var_name="bnds")
        # Test save + loadback
        _check_save_and_reload([cube_1, cube_2])

    def test_string_dim_varname_collision(self, _check_save_and_reload):
        cube_2 = Cube([0, 1], var_name="cube_2")
        # NOTE: it *should* be possible for a cube with string data to cause
        # this collision, but cubes with string data are currently not working.
        # See : https://github.com/SciTools/iris/issues/4412
        x_dim = AuxCoord(["this", "that"], long_name="dim_x", var_name="string_auxco")
        cube_2.add_aux_coord(x_dim, 0)
        cube_1 = Cube([0], long_name="cube_1", var_name="string4")
        # Test save + loadback
        _check_save_and_reload([cube_1, cube_2])

    def test_mesh_location_dim_varname_collision(self, _check_save_and_reload):
        cube_2 = sample_mesh_cube()
        cube_2.var_name = "cube_2"  # Make it identifiable
        cube_1 = Cube([0], long_name="cube_1", var_name="Mesh2d_node")
        # Test save + loadback
        _check_save_and_reload([cube_1, cube_2])

    def test_connectivity_dim_varname_collision(self, _check_save_and_reload):
        cube_2 = sample_mesh_cube()
        cube_2.var_name = "cube_2"  # Make it identifiable
        cube_1 = Cube([0], long_name="cube_1", var_name="Mesh_2d_face_N_nodes")
        # Test save + loadback
        _check_save_and_reload([cube_1, cube_2])


class Test_compute_usage:
    """Test the operation  of the save function 'compute' keyword.

    In actual use, this keyword controls 'delayed saving'.  That is tested elsewhere,
    in testing the 'Saver' class itself.
    """

    # A fixture to mock out Saver object creation in a 'save' call.
    @staticmethod
    @pytest.fixture
    def mock_saver_creation(mocker):
        # A mock for a Saver object.
        mock_saver = mocker.MagicMock(spec=Saver)
        # make an __enter__ call return the object itself (as the real Saver does).
        mock_saver.__enter__ = mocker.Mock(return_value=mock_saver)
        # A mock for the Saver() constructor call.
        mock_new_saver_call = mocker.Mock(return_value=mock_saver)

        # Replace the whole Saver class with a simple function, which thereby emulates
        # the constructor call.  This avoids complications due to the fact that Mock
        # patching does not work in the usual way for __init__ and __new__ methods.
        def mock_saver_class_create(*args, **kwargs):
            return mock_new_saver_call(*args, **kwargs)

        # Patch the Saver() creation to return our mock Saver object.
        mocker.patch("iris.fileformats.netcdf.saver.Saver", mock_saver_class_create)
        # Return mocks for both constructor call, and Saver object.
        return mock_new_saver_call, mock_saver

    # A fixture to provide some mock args for 'Saver' creation.
    @staticmethod
    @pytest.fixture
    def mock_saver_args(mocker):
        from collections import namedtuple

        # A special object for the cube, since cube.attributes must be indexable
        mock_cube = mocker.MagicMock()
        args = namedtuple("saver_args", ["cube", "filename", "format", "compute"])(
            cube=mock_cube,
            filename=mocker.sentinel.filepath,
            format=mocker.sentinel.netcdf4,
            compute=mocker.sentinel.compute,
        )
        return args

    def test_saver_creation(self, mock_saver_creation, mock_saver_args):
        # Check that 'save' creates a Saver, passing the 'compute' keyword.
        mock_saver_new, mock_saver = mock_saver_creation
        args = mock_saver_args
        save(
            cube=args.cube,
            filename=args.filename,
            netcdf_format=args.format,
            compute=args.compute,
        )
        # Check the Saver create call it made, in particular that the compute arg is
        # passed in.
        mock_saver_new.assert_called_once_with(
            args.filename, args.format, compute=args.compute
        )

    def test_compute_true(self, mock_saver_creation, mock_saver_args):
        # Check operation when compute=True.
        mock_saver_new, mock_saver = mock_saver_creation
        args = mock_saver_args
        result = save(
            cube=args.cube,
            filename=args.filename,
            netcdf_format=args.format,
            compute=True,
        )
        # It should NOT have called 'delayed_completion'
        assert mock_saver.delayed_completion.call_count == 0
        # Result should be None
        assert result is None

    def test_compute_false_result_delayed(self, mock_saver_creation, mock_saver_args):
        # Check operation when compute=False.
        mock_saver_new, mock_saver = mock_saver_creation
        args = mock_saver_args
        result = save(
            cube=args.cube,
            filename=args.filename,
            netcdf_format=args.format,
            compute=False,
        )
        # It should have called 'delayed_completion' ..
        assert mock_saver.delayed_completion.call_count == 1
        # .. and should return the result of that.
        assert result is mock_saver.delayed_completion.return_value
