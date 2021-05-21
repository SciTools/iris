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
Both of those supply an "activate" call (for now : may be simplified in future).

"""
import iris.tests as tests

from pathlib import Path
import shutil
import subprocess
import tempfile

from iris.fileformats.cf import CFReader
import iris.fileformats.netcdf
from iris.fileformats.netcdf import _load_cube
import iris.fileformats._nc_load_rules.engine

"""
Testing method.
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


class Test__grid_mappings(tests.IrisTest):
    @classmethod
    def setUpClass(cls):
        # # Control which testing method we are applying.
        # Create a temp directory for temp files.
        cls.temp_dirpath = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        # Destroy a temp directory for temp files.
        shutil.rmtree(cls.temp_dirpath)

    def _call_with_testfile(self):
        # FOR NOW: absolutely basic example.
        cdl_string = r"""
            netcdf test {
            dimensions:
                lats = 2 ;
                lons = 3 ;
            variables:
                double phenom(lats, lons) ;
                    phenom:standard_name = "air_temperature" ;
                    phenom:units = "K" ;
                    phenom:grid_mapping = "grid" ;
                double lats(lats) ;
                    lats:axis = "Y" ;
                    lats:units = "degrees_north" ;
                    lats:standard_name = "latitude" ;
                double lons(lons) ;
                    lons:axis = "X" ;
                    lons:units = "degrees" ;  // THIS IS A BUG!
                    lons:standard_name = "longitude" ;
                int grid ;
                    grid:grid_mapping_name = "latitude_longitude";
                    grid:earth_radius = 6.e6 ;
                data:
                    lats = 10., 20. ;
                    lons = 100., 110., 120. ;
            }
        """
        cdl_path = str(self.temp_dirpath / "test.cdl")
        nc_path = str(self.temp_dirpath / "test.nc")
        with open(cdl_path, "w") as f_out:
            f_out.write(cdl_string)
        # Create reference netCDF file from reference CDL.
        command = "ncgen -o {} {}".format(nc_path, cdl_path)
        subprocess.check_call(command, shell=True)

        cf = CFReader(nc_path)
        # Grab a data variable : FOR NOW, should be only 1
        # (cf_var,) = cf.cf_group.data_variables.values()
        cf_var = cf.cf_group.data_variables["phenom"]

        use_pyke = True
        if use_pyke:
            engine = iris.fileformats.netcdf._pyke_kb_engine_real()
        else:
            engine = iris.fileformats._nc_load_rules.engine.Engine()

        iris.fileformats.netcdf.DEBUG = True
        # iris.fileformats.netcdf.LOAD_PYKE = False
        return _load_cube(engine, cf, cf_var, nc_path)

    def _check_result(self, cube):
        self.assertEqual(cube.standard_name, "air_temperature")
        self.assertEqual(cube.var_name, "phenom")

    def test_latlon(self):
        options = {}
        result = self._call_with_testfile(**options)
        print(result)
        print("coord-system = ", type(result.coord_system()))
        print("  X cs = ", type(result.coord(axis="x").coord_system))
        print("  Y cs = ", type(result.coord(axis="y").coord_system))
        self._check_result(result, **options)
