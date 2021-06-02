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

    @classmethod
    def setUpClass(cls):
        # # Control which testing method we are applying.
        # Create a temp directory for temp files.
        cls.temp_dirpath = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        # Destroy a temp directory for temp files.
        shutil.rmtree(cls.temp_dirpath)

    def _make_testcase_cdl(
        self,
        latitude_units=None,
        gridmapvar_name=None,
        gridmapvar_mappropertyname=None,
        mapping_missingradius=False,
        mapping_type_name=None,
        mapping_scalefactor=None,
        yco_values=None,
        xco_name=None,
        yco_name=None,
        xco_units=None,
        yco_units=None,
    ):
        """
        Create a CDL string for a testcase.

        This is the "master" routine for creating all our testcases.
        Kwarg options modify a simple default testcase with a latlon grid.
        The routine handles the various testcase options and their possible
        interactions.  This includes knowing what extra changes are required
        to support different grid-mapping types (for example).

        """
        # The grid-mapping options are standard-latlon, rotated, or non-latlon.
        # This affects names+units of the X and Y coords.
        # We don't have an option to *not* include a grid-mapping variable, but
        # we can mimic a missing grid-mapping by changing the varname from that
        # which the data-variable refers to, with "gridmapvar_name=xxx".
        # Likewise, an invalid (unrecognised) grid-mapping can be mimicked by
        # selecting an unkown 'grid_mapping_name' property, with
        # "gridmapvar_mappropertyname=xxx".
        if mapping_type_name is None:
            # Default grid-mapping and coords are standard lat-lon.
            mapping_type_name = hh.CF_GRID_MAPPING_LAT_LON
            xco_name_default = hh.CF_VALUE_STD_NAME_LON
            yco_name_default = hh.CF_VALUE_STD_NAME_LAT
            xco_units_default = "degrees_east"
            # Special kwarg overrides some of the values.
            if latitude_units is None:
                yco_units_default = "degrees_north"
            else:
                # Override the latitude units (to invalidate).
                yco_units_default = latitude_units

        elif mapping_type_name == hh.CF_GRID_MAPPING_ROTATED_LAT_LON:
            # Rotated lat-lon coordinates.
            xco_name_default = hh.CF_VALUE_STD_NAME_GRID_LON
            yco_name_default = hh.CF_VALUE_STD_NAME_GRID_LAT
            xco_units_default = "degrees"
            yco_units_default = "degrees"

        else:
            # General non-latlon coordinates
            # Exactly which depends on the grid_mapping name.
            xco_name_default = hh.CF_VALUE_STD_NAME_PROJ_X
            yco_name_default = hh.CF_VALUE_STD_NAME_PROJ_Y
            xco_units_default = "m"
            yco_units_default = "m"

        # Options can override coord (standard) names and units.
        if xco_name is None:
            xco_name = xco_name_default
        if yco_name is None:
            yco_name = yco_name_default
        if xco_units is None:
            xco_units = xco_units_default
        if yco_units is None:
            yco_units = yco_units_default

        grid_mapping_name = "grid"
        # Options can override the gridvar name and properties.
        g_varname = gridmapvar_name
        g_mapname = gridmapvar_mappropertyname
        if g_varname is None:
            g_varname = grid_mapping_name
        if g_mapname is None:
            # If you change this, it is no longer a valid grid-mapping var.
            g_mapname = "grid_mapping_name"

        # Omit the earth radius, if requested.
        if mapping_missingradius:
            g_radius_string = ""
        else:
            g_radius_string = f"{g_varname}:earth_radius = 6.e6 ;"
        g_string = f"""
            int {g_varname} ;
                {g_varname}:{g_mapname} = "{mapping_type_name}";
                {g_radius_string}
        """

        # Add a specified scale-factor, if requested.
        if mapping_scalefactor is not None:
            # Add a specific scale-factor term to the grid mapping.
            # (Non-unity scale is not supported for Mercator/Stereographic).
            sfapo_name = hh.CF_ATTR_GRID_SCALE_FACTOR_AT_PROJ_ORIGIN
            g_string += f"""
                {g_varname}:{sfapo_name} = {mapping_scalefactor} ;
            """

        #
        # Add various additional (minimal) required properties for different
        # grid mapping types.
        #

        # Those which require 'latitude of projection origin'
        if mapping_type_name in (
            hh.CF_GRID_MAPPING_TRANSVERSE,
            hh.CF_GRID_MAPPING_STEREO,
            hh.CF_GRID_MAPPING_GEOSTATIONARY,
            hh.CF_GRID_MAPPING_VERTICAL,
        ):
            latpo_name = hh.CF_ATTR_GRID_LAT_OF_PROJ_ORIGIN
            g_string += f"""
                {g_varname}:{latpo_name} = 0.0 ;
            """
        # Those which require 'longitude of projection origin'
        if mapping_type_name in (
            hh.CF_GRID_MAPPING_STEREO,
            hh.CF_GRID_MAPPING_GEOSTATIONARY,
            hh.CF_GRID_MAPPING_VERTICAL,
        ):
            lonpo_name = hh.CF_ATTR_GRID_LON_OF_PROJ_ORIGIN
            g_string += f"""
                {g_varname}:{lonpo_name} = 0.0 ;
            """
        # Those which require 'longitude of central meridian'
        if mapping_type_name in (hh.CF_GRID_MAPPING_TRANSVERSE,):
            latcm_name = hh.CF_ATTR_GRID_LON_OF_CENT_MERIDIAN
            g_string += f"""
                {g_varname}:{latcm_name} = 0.0 ;
            """
        # Those which require 'perspective point height'
        if mapping_type_name in (
            hh.CF_GRID_MAPPING_VERTICAL,
            hh.CF_GRID_MAPPING_GEOSTATIONARY,
        ):
            pph_name = hh.CF_ATTR_GRID_PERSPECTIVE_HEIGHT
            g_string += f"""
                {g_varname}:{pph_name} = 600000.0 ;
            """
        # Those which require 'sweep angle axis'
        if mapping_type_name in (hh.CF_GRID_MAPPING_GEOSTATIONARY,):
            saa_name = hh.CF_ATTR_GRID_SWEEP_ANGLE_AXIS
            g_string += f"""
                {g_varname}:{saa_name} = "y" ;
            """

        # y-coord values
        if yco_values is None:
            yco_values = [10.0, 20.0]
        yco_value_strings = [str(val) for val in yco_values]
        yco_values_string = ", ".join(yco_value_strings)

        # Construct the total CDL string
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
                    yco = {yco_values_string} ;
                    xco = 100., 110., 120. ;
            }}
        """
        if self.debug:
            print("File content:")
            print(cdl_string)
            print("------\n")
        return cdl_string

    def _load_cube_from_cdl(self, cdl_string, cdl_path, nc_path):
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

        if self.use_pyke:
            engine = iris.fileformats.netcdf._pyke_kb_engine_real()
        else:
            engine = iris.fileformats._nc_load_rules.engine.Engine()

        iris.fileformats.netcdf.DEBUG = self.debug
        # iris.fileformats.netcdf.LOAD_PYKE = False
        return _load_cube(engine, cf, cf_var, nc_path)

    def run_testcase(self, warning=None, **testcase_kwargs):
        """
        Run a testcase with chosen options, returning a test cube.

        The kwargs apply to the '_make_testcase_cdl' method.

        """
        cdl_path = str(self.temp_dirpath / "test.cdl")
        nc_path = cdl_path.replace(".cdl", ".nc")
        cdl_string = self._make_testcase_cdl(**testcase_kwargs)
        if warning is None:
            context = self.assertNoWarningsRegexp()
        else:
            context = self.assertWarnsRegexp(warning)
        with context:
            cube = self._load_cube_from_cdl(cdl_string, cdl_path, nc_path)
        if self.debug:
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
        xco_no_cs=False,  # N.B. no effect if cube_no_cs is True
        yco_no_cs=False,  # N.B. no effect if cube_no_cs is True
        yco_is_aux=False,
        xco_stdname=True,
        yco_stdname=True,
    ):
        """
        Check key properties of a result cube.

        Various options control the expected things which are tested.
        """
        self.assertEqual(cube.standard_name, "air_temperature")
        self.assertEqual(cube.var_name, "phenom")

        x_coords = cube.coords(dimensions=(1,))
        y_coords = cube.coords(dimensions=(0,))
        if yco_is_aux:
            expected_dim_coords = x_coords
            expected_aux_coords = y_coords
        else:
            expected_dim_coords = x_coords + y_coords
            expected_aux_coords = []

        self.assertEqual(
            set(expected_dim_coords), set(cube.coords(dim_coords=True))
        )
        if cube_no_xycoords:
            self.assertEqual(expected_dim_coords, [])
            x_coord = None
            y_coord = None
        else:
            self.assertEqual(len(x_coords), 1)
            (x_coord,) = x_coords
            self.assertEqual(len(y_coords), 1)
            (y_coord,) = y_coords

        self.assertEqual(
            set(expected_aux_coords), set(cube.coords(dim_coords=False))
        )

        if x_coord:
            if xco_stdname is None:
                # no check
                pass
            elif xco_stdname is True:
                self.assertIsNotNone(x_coord.standard_name)
            elif xco_stdname is False:
                self.assertIsNone(x_coord.standard_name)
            else:
                self.assertEqual(x_coord.standard_name, xco_stdname)

        if y_coord:
            if yco_stdname is None:
                # no check
                pass
            if yco_stdname is True:
                self.assertIsNotNone(y_coord.standard_name)
            elif yco_stdname is False:
                self.assertIsNone(y_coord.standard_name)
            else:
                self.assertEqual(y_coord.standard_name, yco_stdname)

        cube_cs = cube.coord_system()
        if cube_no_xycoords:
            yco_cs = None
            xco_cs = None
        else:
            yco_cs = y_coord.coord_system
            xco_cs = x_coord.coord_system
        if cube_no_cs:
            self.assertIsNone(cube_cs)
            self.assertIsNone(yco_cs)
            self.assertIsNone(xco_cs)
        else:
            if cube_cstype is not None:
                self.assertIsInstance(cube_cs, cube_cstype)
            if xco_no_cs:
                self.assertIsNone(xco_cs)
            else:
                self.assertEqual(xco_cs, cube_cs)
            if yco_no_cs:
                self.assertIsNone(yco_cs)
            else:
                self.assertEqual(yco_cs, cube_cs)
