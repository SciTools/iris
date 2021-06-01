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

import iris.coord_systems as ics
from iris.fileformats.cf import CFReader
import iris.fileformats.netcdf
from iris.fileformats.netcdf import _load_cube
import iris.fileformats._nc_load_rules.engine
import iris.fileformats._nc_load_rules.helpers as hh

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


class Mixin_Test__nc_load_actions(tests.IrisTest):
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
        mapping_name=None,
        use_bad_mapping_params=False,
    ):
        """
        Write a testcase example into a CDL file.
        """
        # Grid-mapping options are standard-latlon, rotated, or non-latlon.
        # This affects names+units of the X and Y coords.
        if mapping_name is None:
            # Default grid-mapping and coords are standard lat-lon.
            mapping_name = hh.CF_GRID_MAPPING_LAT_LON
            xco_name = hh.CF_VALUE_STD_NAME_LON
            yco_name = hh.CF_VALUE_STD_NAME_LAT
            xco_units = "degrees_east"
            # Special cases override some of the values.
            if latitude_units is None:
                yco_units = "degrees_north"
            else:
                # Override the latitude units (to invalidate).
                yco_units = latitude_units

        elif mapping_name == hh.CF_GRID_MAPPING_ROTATED_LAT_LON:
            xco_name = hh.CF_VALUE_STD_NAME_GRID_LON
            yco_name = hh.CF_VALUE_STD_NAME_GRID_LAT
            xco_units = "degrees"
            yco_units = "degrees"
        else:
            # General non-latlon coordinates
            # Exactly which depends on the grid_mapping name.
            xco_name = hh.CF_VALUE_STD_NAME_PROJ_X
            yco_name = hh.CF_VALUE_STD_NAME_PROJ_Y
            xco_units = "m"
            yco_units = "m"

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
        g_string = f"""
            int {g_varname} ;
                {g_varname}:{g_mapname} = "{mapping_name}";
                {g_radius_string}
        """
        if use_bad_mapping_params:
            if mapping_name == hh.CF_GRID_MAPPING_MERCATOR:
                # Mercator mapping with nonzero false-easting is unsupported.
                g_string += f"""
                    {g_varname}:{hh.CF_ATTR_GRID_FALSE_EASTING} = 1.0 ;
                """
            elif False:
                pass
            else:
                # Key is only valid for specific grid-mappings.
                assert mapping_name in (
                    hh.CF_GRID_MAPPING_MERCATOR,
                    hh.CF_GRID_MAPPING_STEREO,
                )

        cdl_string = f"""
            netcdf test {{
            dimensions:
                yco = 2 ;
                xco = 3 ;
            variables:
                double phenom(yco, xco) ;
                    phenom:standard_name = "air_temperature" ;
                    phenom:units = "K" ;
                    phenom:grid_mapping = "grid" ;
                double yco(yco) ;
                    yco:axis = "Y" ;
                    yco:units = "{yco_units}" ;
                    yco:standard_name = "{yco_name}" ;
                double xco(xco) ;
                    xco:axis = "X" ;
                    xco:units = "{xco_units}" ;
                    xco:standard_name = "{xco_name}" ;
                {g_string}
                data:
                    yco = 10., 20. ;
                    xco = 100., 110., 120. ;
            }}
        """
        print("File content:")
        print(cdl_string)
        print("------\n")
        with open(cdl_path, "w") as f_out:
            f_out.write(cdl_string)
        return cdl_path

    def create_cube_from_cdl(self, cdl_path, nc_path, use_pyke=True):
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

    def run_testcase(self, **testcase_kwargs):
        """
        Run a testcase with chosen optionsm returning a test cube.

        The kwargs apply to the 'make_testcase_cdl' method.

        """
        cdl_path = str(self.temp_dirpath / "test.cdl")
        nc_path = cdl_path.replace(".cdl", ".nc")
        self.make_testcase_cdl(cdl_path, **testcase_kwargs)
        cube = self.create_cube_from_cdl(cdl_path, nc_path)
        print("\nCube:")
        print(cube)
        print("")
        return cube

    def check_result(
        self,
        cube,
        cube_cstype=None,
        cube_no_cs=False,
        cube_no_xycoords=False,
        latitude_no_cs=False,
    ):
        """
        Check key properties of a result cube.

        Various options control the expected things which are tested.
        """
        self.assertEqual(cube.standard_name, "air_temperature")
        self.assertEqual(cube.var_name, "phenom")

        x_coords = cube.coords(axis="x")
        y_coords = cube.coords(axis="y")
        expected_dim_coords = x_coords + y_coords
        self.assertEqual(
            set(expected_dim_coords), set(cube.coords(dim_coords=True))
        )
        # These are exactly the coords we have.
        if cube_no_xycoords:
            self.assertEqual(expected_dim_coords, [])
            x_coord = None
            y_coord = None
        else:
            self.assertEqual(len(x_coords), 1)
            (x_coord,) = x_coords
            self.assertEqual(len(y_coords), 1)
            (y_coord,) = y_coords

        expected_aux_coords = []
        # These are exactly the coords we have.
        self.assertEqual(
            set(expected_aux_coords), set(cube.coords(dim_coords=False))
        )

        cube_cs = cube.coord_system()
        if cube_no_xycoords:
            lat_cs = None
            lon_cs = None
        else:
            lat_cs = y_coord.coord_system
            lon_cs = x_coord.coord_system
        if cube_no_cs:
            self.assertIsNone(cube_cs)
            self.assertIsNone(lat_cs)
            self.assertIsNone(lon_cs)
        else:
            if cube_cstype is not None:
                self.assertIsInstance(cube_cs, cube_cstype)
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
        result = self.run_testcase()
        self.check_result(result)

    def test_missing_latlon_radius(self):
        # Lat-long with a missing earth-radius causes an error.
        # One of very few cases where activation may encounter an error.
        # N.B. doesn't really test rule-activation, but maybe worth doing.
        with self.assertRaisesRegex(ValueError, "No ellipsoid"):
            self.run_testcase(gridmapvar_missingradius=True)

    def test_bad_gridmapping_nameproperty(self):
        # Fix the 'grid' var so it does not register as a grid-mapping.
        result = self.run_testcase(gridmapvar_mappropertyname="mappy")
        self.check_result(result, cube_no_cs=True)

    def test_latlon_bad_gridmapping_varname(self):
        # rename the grid-mapping variable so it is effectively 'missing'.
        with self.assertWarnsRegexp("Missing.*grid mapping variable 'grid'"):
            result = self.run_testcase(gridmapvar_name="grid_2")
        self.check_result(result, cube_no_cs=True)

    def test_latlon_bad_latlon_unit(self):
        # Check with bad latitude units : 'degrees' in place of 'degrees_north'.
        result = self.run_testcase(latitude_units="degrees")
        self.check_result(result, latitude_no_cs=True)

    def test_mapping_rotated(self):
        result = self.run_testcase(
            mapping_name=hh.CF_GRID_MAPPING_ROTATED_LAT_LON
        )
        self.check_result(result, cube_cstype=ics.RotatedGeogCS)

    def test_mapping_albers(self):
        result = self.run_testcase(mapping_name=hh.CF_GRID_MAPPING_ALBERS)
        self.check_result(result, cube_cstype=ics.AlbersEqualArea)

    def test_mapping_mercator(self):
        result = self.run_testcase(mapping_name=hh.CF_GRID_MAPPING_MERCATOR)
        self.check_result(result, cube_cstype=ics.Mercator)

    def test_mapping_mercator__fail_unsupported(self):
        with self.assertWarnsRegexp("not yet supported for Mercator"):
            result = self.run_testcase(
                mapping_name=hh.CF_GRID_MAPPING_MERCATOR,
                use_bad_mapping_params=True,
            )
        self.check_result(result, cube_no_cs=True, cube_no_xycoords=True)


if __name__ == "__main__":
    tests.main()
