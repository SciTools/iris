# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the module :mod:`iris.fileformats._nc_load_rules.actions`.

"""
from pathlib import Path
import shutil
import subprocess
import tempfile

import iris.fileformats._nc_load_rules.engine
from iris.fileformats.cf import CFReader
import iris.fileformats.netcdf
from iris.fileformats.netcdf import _load_cube

"""
Notes on testing method.

IN cf : "def _load_cube(engine, cf, cf_var, filename)"
WHERE:
  - engine is a :class:`iris.fileformats._nc_load_rules.engine.Engine`
  - cf is a :class:`iris.fileformats.cf.CFReader`
  - cf_var is a :class:`iris.fileformats.cf.CFDataVariable`

As it's hard to construct a suitable CFReader from scratch, it would seem
simpler (for now) to use an ACTUAL FILE.
Likewise, the easiest approach to that is with CDL and "ncgen".
To do this, we need a test "fixture" that can create suitable test files in a
temporary directory.

"""


class Mixin__nc_load_actions:
    """
    Class to make testcases for rules or actions code, and check results.

    Defines standard setUpClass/tearDownClass methods, to create a temporary
    directory for intermediate files.
    NOTE: owing to peculiarities of unittest, these must be explicitly called
    from a setUpClass/tearDownClass within the 'final' inheritor, i.e. the
    actual Test_XXX class which also inherits unittest.TestCase.

    Testcases are manufactured by the '_make_testcase_cdl' method.
    The 'run_testcase' method takes the '_make_testcase_cdl' kwargs and makes
    a result cube (by: producing cdl, converting to netcdf, and loading the
    'phenom' variable only).
    Likewise, a generalised 'check_result' method will be used to perform result
    checking.
    Both '_make_testcase_cdl' and 'check_result' are not defined here :  They
    are to be variously implemented by the inheritors.

    """

    # "global" test setting : whether to output various debug info
    debug = False

    @classmethod
    def setUpClass(cls):
        # Create a temp directory for temp files.
        cls.temp_dirpath = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        # Destroy a temp directory for temp files.
        shutil.rmtree(cls.temp_dirpath)

    def load_cube_from_cdl(self, cdl_string, cdl_path, nc_path):
        """
        Load the 'phenom' data variable in a CDL testcase, as a cube.

        Using ncgen, CFReader and the _load_cube call.

        """
        # Write the CDL to a file.
        with open(cdl_path, "w") as f_out:
            f_out.write(cdl_string)

        # Create a netCDF file from the CDL file.
        command = "ncgen -o {} {}".format(nc_path, cdl_path)
        subprocess.check_call(command, shell=True)

        # Simulate the inner part of the file reading process.
        cf = CFReader(nc_path)
        # Grab a data variable : FOR NOW always grab the 'phenom' variable.
        cf_var = cf.cf_group.data_variables["phenom"]

        engine = iris.fileformats.netcdf._actions_engine()

        # If debug enabled, switch on the activation summary debug output.
        # Use 'patch' so it is restored after the test.
        self.patch("iris.fileformats.netcdf.DEBUG", self.debug)

        # Call the main translation function to load a single cube.
        # _load_cube establishes per-cube facts, activates rules and
        # produces an actual cube.
        cube = _load_cube(engine, cf, cf_var, nc_path)

        # Also Record, on the cubes, which hybrid coord elements were identified
        # by the rules operation.
        # Unlike the other translations, _load_cube does *not* convert this
        # information into actual cube elements.  That is instead done by
        # `iris.fileformats.netcdf._load_aux_factory`.
        # For rules testing, it is anyway more convenient to deal with the raw
        # data, as each factory type has different validity requirements to
        # build it, and none of that is relevant to the rules operation.
        cube._formula_type_name = engine.requires.get("formula_type")
        cube._formula_terms_byname = engine.requires.get("formula_terms")

        # Always returns a single cube.
        return cube

    def run_testcase(self, warning=None, **testcase_kwargs):
        """
        Run a testcase with chosen options, returning a test cube.

        The kwargs apply to the '_make_testcase_cdl' method.

        """
        cdl_path = str(self.temp_dirpath / "test.cdl")
        nc_path = cdl_path.replace(".cdl", ".nc")

        cdl_string = self._make_testcase_cdl(**testcase_kwargs)
        if self.debug:
            print("CDL file content:")
            print(cdl_string)
            print("------\n")

        if warning is None:
            context = self.assertNoWarningsRegexp()
        else:
            context = self.assertWarnsRegexp(warning)
        with context:
            cube = self.load_cube_from_cdl(cdl_string, cdl_path, nc_path)

        if self.debug:
            print("\nCube:")
            print(cube)
            print("")
        return cube

    def _make_testcase_cdl(self, **kwargs):
        """Make a testcase CDL string."""
        # Override for specific uses...
        raise NotImplementedError()

    def check_result(self, cube, **kwargs):
        """Test a result cube."""
        # Override for specific uses...
        raise NotImplementedError()
