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


class Mixin_Test__nc_load_actions:
    """
    Class to make testcases for rules or actions code and check results.

    Defines standard setUp/tearDown-Class to create intermediate files in a
    temporary directory.

    Testcase manufacture in _make_testcase_file', based on a simple latlon grid
    example with various kwargs to control variations.
    Testing in 'test_result', with various kwargs controlling expected results.

    Can also switch between testing Pyke and non-Pyke implementations (for now).

    """

    @classmethod
    def setUpClass(cls):
        # # Control which testing method we are applying.
        # Create a temp directory for temp files.
        cls.temp_dirpath = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        # Destroy a temp directory for temp files.
        shutil.rmtree(cls.temp_dirpath)

    def make_testcase_cdl(
        self,
        cdl_path,
        latitude_units=None,
        gridmapvar_name=None,
        gridmapvar_mappropertyname=None,
        gridmapvar_missingradius=False,
    ):
        """
        Write a testcase example into a CDL file.
        """
        if latitude_units is None:
            latitude_units = "degrees_north"
        grid_mapping_name = "grid"
        g_varname = gridmapvar_name
        g_mapname = gridmapvar_mappropertyname
        if g_varname is None:
            g_varname = grid_mapping_name
        if g_mapname is None:
            g_mapname = "grid_mapping_name"
        if gridmapvar_missingradius:
            g_radius_string = ""
        else:
            g_radius_string = f"{g_varname}:earth_radius = 6.e6 ;"

        cdl_string = f"""
            netcdf test {{
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
                    lats:units = "{latitude_units}" ;
                    lats:standard_name = "latitude" ;
                double lons(lons) ;
                    lons:axis = "X" ;
                    lons:units = "degrees_east" ;
                    lons:standard_name = "longitude" ;
                int {g_varname} ;
                    {g_varname}:{g_mapname} = "latitude_longitude";
                    {g_radius_string}
                data:
                    lats = 10., 20. ;
                    lons = 100., 110., 120. ;
            }}
        """
        # print('File content:')
        # print(cdl_string)
        # print('------\n')
        with open(cdl_path, "w") as f_out:
            f_out.write(cdl_string)
        return cdl_path

    def create_cube_from_cdl(self, cdl_path, nc_path, use_pyke=False):
        """
        Load the 'phenom' data variable in a CDL testcase, as a cube.

        Using ncgen and the selected _load_cube call.

        FOR NOW: can select whether load uses Pyke (rules) or newer actions
        code.
        TODO: remove when Pyke implementation is gone.

        """
        # Create reference netCDF file from reference CDL.
        command = "ncgen -o {} {}".format(nc_path, cdl_path)
        subprocess.check_call(command, shell=True)

        cf = CFReader(nc_path)
        # Grab a data variable : FOR NOW, should be only 1
        cf_var = list(cf.cf_group.data_variables.values())[0]
        cf_var = cf.cf_group.data_variables["phenom"]

        if use_pyke:
            engine = iris.fileformats.netcdf._pyke_kb_engine_real()
        else:
            engine = iris.fileformats._nc_load_rules.engine.Engine()

        iris.fileformats.netcdf.DEBUG = True
        # iris.fileformats.netcdf.LOAD_PYKE = False
        return _load_cube(engine, cf, cf_var, nc_path)

    def _run_testcase(self, **testcase_kwargs):
        """
        Run a testcase with chosen optionsm returning a test cube.

        The kwargs apply to the 'make_testcase_cdl' method.

        """
        cdl_path = str(self.temp_dirpath / "test.cdl")
        nc_path = cdl_path.replace(".cdl", ".nc")
        self.make_testcase_cdl(cdl_path, **testcase_kwargs)
        cube = self.create_cube_from_cdl(cdl_path, nc_path)
        return cube

    def _check_result(self, cube, cube_no_cs=False, latitude_no_cs=False):
        """
        Check key properties of a result cube.

        Various options control the expected things which are tested.
        """
        self.assertEqual(cube.standard_name, "air_temperature")
        self.assertEqual(cube.var_name, "phenom")

        lon_coord = cube.coord("longitude")
        lat_coord = cube.coord("latitude")
        expected_dim_coords = [lon_coord, lat_coord]
        expected_aux_coords = []
        # These are exactly the coords we have.
        self.assertEqual(
            set(expected_dim_coords), set(cube.coords(dim_coords=True))
        )
        # These are exactly the coords we have.
        self.assertEqual(
            set(expected_aux_coords), set(cube.coords(dim_coords=False))
        )

        cube_cs = cube.coord_system()
        lat_cs = lat_coord.coord_system
        lon_cs = lon_coord.coord_system
        if cube_no_cs:
            self.assertIsNone(cube_cs)
            self.assertIsNone(lat_cs)
            self.assertIsNone(lon_cs)
        else:
            self.assertEqual(lon_cs, cube_cs)
            if latitude_no_cs:
                self.assertIsNone(lat_cs)
            else:
                self.assertEqual(lat_cs, cube_cs)


class Test__grid_mapping(Mixin_Test__nc_load_actions, tests.IrisTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def test_basic_latlon(self):
        # A basic reference example with a lat-long grid.
        result = self._run_testcase()
        self._check_result(result)

    def test_missing_latlon_radius(self):
        # Lat-long with a missing earth-radius causes an error.
        # One of very few cases where activation may encounter an error.
        # N.B. doesn't really test rule-activation, but maybe worth doing.
        with self.assertRaisesRegex(ValueError, "No ellipsoid"):
            self._run_testcase(gridmapvar_missingradius=True)

    def test_bad_gridmapping_nameproperty(self):
        # Fix the 'grid' var so it does not register as a grid-mapping.
        result = self._run_testcase(gridmapvar_mappropertyname="mappy")
        self._check_result(result, cube_no_cs=True)

    def test_latlon_bad_gridmapping_varname(self):
        # rename the grid-mapping variable so it is effectively 'missing'.
        with self.assertWarnsRegexp("Missing.*grid mapping variable 'grid'"):
            result = self._run_testcase(gridmapvar_name="grid_2")
        self._check_result(result, cube_no_cs=True)

    def test_latlon_bad_latlon_unit(self):
        # Check with bad latitude units : 'degrees' in place of 'degrees_north'.
        result = self._run_testcase(latitude_units="degrees")
        self._check_result(result, latitude_no_cs=True)


if __name__ == "__main__":
    tests.main()
