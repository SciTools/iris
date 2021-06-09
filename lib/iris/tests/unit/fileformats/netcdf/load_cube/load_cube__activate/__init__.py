# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the engine.activate() call within the
`iris.fileformats.netcdf._load_cube` function.

For now, these tests are designed to function with **either** the "old"
Pyke-rules implementation in :mod:`iris.fileformats._pyke_rules`, **or** the
"new" :mod:`iris.fileformats._nc_load_rules`.
Both of those supply an "engine" with an "activate" method
 -- at least for now : may be simplified in future.

"""
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from iris.fileformats.cf import CFReader
import iris.fileformats.netcdf
from iris.fileformats.netcdf import _load_cube
import iris.fileformats._nc_load_rules.engine

"""
Notes on testing method.

IN cf : "def _load_cube(engine, cf, cf_var, filename)"
WHERE:
  - engine is a :class:`pyke.knowledge_engine.engine`
          -- **OR** :class:`iris.fileformats._nc_load_rules.engine.Engine`
  - cf is a CFReader
  - cf_var is a CFDAtaVariable

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
    These are based on a 'standard simple latlon grid' example.
    Various kwargs control variations on this.

    The 'run_testcase' method takes the '_make_testcase_cdl' kwargs and makes
    a result cube (by: producing cdl, converting to netcdf, and loading).

    The 'check_result' method performs various checks on the result, with
    kwargs controlling the expected properties to be tested against.
    This usage is *also* based on the 'standard simple latlon grid' example,
    the kwargs specify expected differences from that.

    Can also test with either the Pyke(rules) or non-Pyke (actions)
    implementations (for now).

    """

    #
    # "global" test settings
    #

    # whether to test 'rules' or 'actions' implementations
    # TODO: remove when Pyke is gone
    use_pyke = True

    # whether to output various debug info
    # TODO: ?possibly? remove when development is complete
    debug = False

    # whether to perform action in both ways and compare results.
    compare_pyke_nonpyke = True

    @classmethod
    def setUpClass(cls):
        # # Control which testing method we are applying.
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
        Can use a genuine Pyke engine, or the actions mimic engine,
        selected by `self.use_pyke`.

        """
        # Write the CDL to a file.
        with open(cdl_path, "w") as f_out:
            f_out.write(cdl_string)

        # Create a netCDF file from the CDL file.
        command = "ncgen -o {} {}".format(nc_path, cdl_path)
        subprocess.check_call(command, shell=True)

        # Simulate the inner part of the file reading process.
        cf = CFReader(nc_path)
        # Grab a data variable : FOR NOW, should be only 1
        cf_var = list(cf.cf_group.data_variables.values())[0]
        cf_var = cf.cf_group.data_variables["phenom"]

        do_pyke = self.use_pyke or self.compare_pyke_nonpyke
        do_nonpyke = not self.use_pyke or self.compare_pyke_nonpyke
        if do_pyke:
            pyke_engine = iris.fileformats.netcdf._pyke_kb_engine_real()
        if do_nonpyke:
            nonpyke_engine = iris.fileformats._nc_load_rules.engine.Engine()

        iris.fileformats.netcdf.DEBUG = self.debug

        # Call the main translation function to load a single cube.
        def load_single_cube(engine):
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

            return cube

        if do_pyke:
            pyke_cube = load_single_cube(pyke_engine)
        if do_nonpyke:
            nonpyke_cube = load_single_cube(nonpyke_engine)

        # If requested, directly compare the pyke and non-pyke outputs.
        if self.compare_pyke_nonpyke:
            # Compare the loaded cubes from both engines.
            print("\nPYKE-NONPYKE COMPARE")

            # First zap cube-data, as masked data does not compare well.
            def unmask_cube(cube):
                # preserve the original, we're going to realise..
                cube = cube.copy()
                if isinstance(cube.data, np.ma.MaskedArray):
                    cube.data = cube.data.filled(0)
                return cube

            pyke_cube_copy = unmask_cube(pyke_cube)
            nonpyke_cube_copy = unmask_cube(nonpyke_cube)
            if self.debug:
                if nonpyke_cube_copy != pyke_cube_copy:

                    def show_cube(cube):
                        result = str(cube)
                        result += "\n--coords--"
                        for coord in cube.coords():
                            result += "\n  " + str(coord)
                        result += "\n--attributes--"
                        if not cube.attributes:
                            result += "\n  (none)"
                        else:
                            for key, value in cube.attributes.items():
                                result += f"\n  {key}: {value}"
                        return result

                    print("\nPyke/nonpyke mismatch.")
                    print("Pyke cube:\n----")
                    print(show_cube(pyke_cube))
                    print()
                    print("NONPyke cube:\n----")
                    print(show_cube(nonpyke_cube))
                    print("")
            else:
                self.assertEqual(pyke_cube_copy, nonpyke_cube_copy)

        # Return the right thing, whether we did 'both' or not
        if self.use_pyke:
            result_cube = pyke_cube
        else:
            result_cube = nonpyke_cube

        # Always returns a single cube.
        return result_cube

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
