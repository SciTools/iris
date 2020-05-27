# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test iris.load functions support for loader-specific keywords.

"""

import os.path
import shutil
import tempfile

import iris
from iris.cube import Cube, CubeList
from iris.fileformats import FORMAT_AGENT
from iris.io.format_picker import FileExtension, FormatSpecification

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests


# Define an extra 'dummy' FormatHandler for load-kwarg testing.
# We add this, temporarily, to the Iris file-format picker for testing.
def format_x_handler_function(fnames, callback, **loader_kwargs):
    # A format handler function.
    # Yield a single cube, with its attributes set to the call arguments.
    call_args = dict(
        fnames=fnames, callback=callback, loader_kwargs=loader_kwargs
    )
    cube = Cube(1, attributes=call_args)
    yield cube


# A Format spec for the fake file format.
FORMAT_X_EXTENSION = "._xtst_"
FORMAT_X = FormatSpecification(
    "Testing file handler",
    FileExtension(),
    FORMAT_X_EXTENSION,
    format_x_handler_function,
)


class LoadFunctionMixin:
    # Common code to test load/load_cube/load_cubes/load_raw.

    # Inheritor must set this to the name of an iris load function.
    # Note: storing the function itself causes it to be mis-identified as an
    # instance method when called, so doing it by name is clearer.
    load_function_name = "xxx"

    @classmethod
    def setUpClass(cls):
        # Add our dummy format handler to the common Iris io file-picker.
        FORMAT_AGENT.add_spec(FORMAT_X)
        # Create a temporary working directory.
        cls._temp_dir = tempfile.mkdtemp()
        # Store the path of a dummy file whose name matches the picker.
        filename = "testfile" + FORMAT_X_EXTENSION
        cls.test_filepath = os.path.join(cls._temp_dir, filename)
        # Create the dummy file.
        with open(cls.test_filepath, "w") as f_open:
            # Write some data to satisfy the other, signature-based pickers.
            # TODO: this really shouldn't be necessary ??
            f_open.write("\x00" * 100)

    @classmethod
    def tearDownClass(cls):
        # Remove the dummy format handler.
        # N.B. no public api, so uses a private property of FormatAgent.
        FORMAT_AGENT._format_specs.remove(FORMAT_X)
        # Delete the temporary directory.
        shutil.rmtree(cls._temp_dir)

    def _load_a_cube(self, *args, **kwargs):
        load_function = getattr(iris, self.load_function_name)
        result = load_function(*args, loader_kwargs=kwargs)
        if load_function is not iris.load_cube:
            # Handle 'other' load functions, which return CubeLists ...
            self.assertIsInstance(result, CubeList)
            # ... however, we intend that all uses will return only 1 cube.
            self.assertEqual(len(result), 1)
            result = result[0]
        self.assertIsInstance(result, Cube)
        return result

    def test_extra_args(self):
        test_kwargs = {"loader_a": 1, "loader_b": "two"}
        result = self._load_a_cube(self.test_filepath, **test_kwargs)
        self.assertEqual(
            result.attributes,
            dict(
                fnames=[self.test_filepath],
                callback=None,
                loader_kwargs=test_kwargs,
            ),
        )

    def test_no_extra_args(self):
        result = self._load_a_cube(self.test_filepath)
        self.assertEqual(
            result.attributes,
            dict(fnames=[self.test_filepath], callback=None, loader_kwargs={}),
        )

    @tests.skip_data
    def test_wrong_loader_noargs_ok(self):
        filepath = tests.get_data_path(
            ["NetCDF", "global", "xyz_t", "GEMS_CO2_Apr2006.nc"]
        )
        result = self._load_a_cube(filepath, "co2")
        self.assertIsNot(result, None)

    @tests.skip_data
    def test_wrong_loader_withargs__fail(self):
        filepath = tests.get_data_path(
            ["NetCDF", "global", "xyz_t", "GEMS_CO2_Apr2006.nc"]
        )
        test_kwargs = {"junk": "this"}
        msg = "load.* got an unexpected keyword argument 'junk'"
        with self.assertRaisesRegex(TypeError, msg):
            _ = self._load_a_cube(filepath, "co2", **test_kwargs)


class TestLoad(LoadFunctionMixin, tests.IrisTest):
    load_function_name = "load"


class TestLoadCubes(LoadFunctionMixin, tests.IrisTest):
    load_function_name = "load_cubes"


class TestLoadCube(LoadFunctionMixin, tests.IrisTest):
    load_function_name = "load_cube"


class TestLoadRaw(LoadFunctionMixin, tests.IrisTest):
    load_function_name = "load_raw"


if __name__ == "__main__":
    tests.main()
