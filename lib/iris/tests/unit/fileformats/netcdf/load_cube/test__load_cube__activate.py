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


class Mixin_Test__nc_load_actions:
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
        cdl_path,
        latitude_units=None,
        gridmapvar_name=None,
        gridmapvar_mappropertyname=None,
        gridmapvar_missingradius=False,
        mapping_name=None,
        mapping_scalefactor=None,
    ):
        """
        Write a testcase example into a CDL file.

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
        if mapping_name is None:
            # Default grid-mapping and coords are standard lat-lon.
            mapping_name = hh.CF_GRID_MAPPING_LAT_LON
            xco_name = hh.CF_VALUE_STD_NAME_LON
            yco_name = hh.CF_VALUE_STD_NAME_LAT
            xco_units = "degrees_east"
            # Special kwarg overrides some of the values.
            if latitude_units is None:
                yco_units = "degrees_north"
            else:
                # Override the latitude units (to invalidate).
                yco_units = latitude_units

        elif mapping_name == hh.CF_GRID_MAPPING_ROTATED_LAT_LON:
            # Rotated lat-lon coordinates.
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
        # Options can override the gridvar name, and its 'grid+mapping_name'
        # property.
        g_varname = gridmapvar_name
        g_mapname = gridmapvar_mappropertyname
        if g_varname is None:
            g_varname = grid_mapping_name
        if g_mapname is None:
            g_mapname = "grid_mapping_name"

        # Omit the earth radius, if requested.
        if gridmapvar_missingradius:
            g_radius_string = ""
        else:
            g_radius_string = f"{g_varname}:earth_radius = 6.e6 ;"
        g_string = f"""
            int {g_varname} ;
                {g_varname}:{g_mapname} = "{mapping_name}";
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
        if mapping_name in (
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
        if mapping_name in (
            hh.CF_GRID_MAPPING_STEREO,
            hh.CF_GRID_MAPPING_GEOSTATIONARY,
            hh.CF_GRID_MAPPING_VERTICAL,
        ):
            lonpo_name = hh.CF_ATTR_GRID_LON_OF_PROJ_ORIGIN
            g_string += f"""
                {g_varname}:{lonpo_name} = 0.0 ;
            """
        # Those which require 'longitude of central meridian'
        if mapping_name in (hh.CF_GRID_MAPPING_TRANSVERSE,):
            latcm_name = hh.CF_ATTR_GRID_LON_OF_CENT_MERIDIAN
            g_string += f"""
                {g_varname}:{latcm_name} = 0.0 ;
            """
        # Those which require 'perspective point height'
        if mapping_name in (
            hh.CF_GRID_MAPPING_VERTICAL,
            hh.CF_GRID_MAPPING_GEOSTATIONARY,
        ):
            pph_name = hh.CF_ATTR_GRID_PERSPECTIVE_HEIGHT
            g_string += f"""
                {g_varname}:{pph_name} = 600000.0 ;
            """
        # Those which require 'sweep angle axis'
        if mapping_name in (hh.CF_GRID_MAPPING_GEOSTATIONARY,):
            saa_name = hh.CF_ATTR_GRID_SWEEP_ANGLE_AXIS
            g_string += f"""
                {g_varname}:{saa_name} = "y" ;
            """

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
                    yco = 10., 20. ;
                    xco = 100., 110., 120. ;
            }}
        """
        if self.debug:
            print("File content:")
            print(cdl_string)
            print("------\n")
        with open(cdl_path, "w") as f_out:
            f_out.write(cdl_string)
        return cdl_path

    def _load_cube_from_cdl(self, cdl_path, nc_path):
        """
        Load the 'phenom' data variable in a CDL testcase, as a cube.

        Using ncgen and the selected _load_cube call.

        """
        # Create reference netCDF file from reference CDL.
        command = "ncgen -o {} {}".format(nc_path, cdl_path)
        subprocess.check_call(command, shell=True)

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

    def run_testcase(self, **testcase_kwargs):
        """
        Run a testcase with chosen optionsm returning a test cube.

        The kwargs apply to the '_make_testcase_cdl' method.

        """
        cdl_path = str(self.temp_dirpath / "test.cdl")
        nc_path = cdl_path.replace(".cdl", ".nc")
        self._make_testcase_cdl(cdl_path, **testcase_kwargs)
        cube = self._load_cube_from_cdl(cdl_path, nc_path)
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


class Mixin__grid_mapping(Mixin_Test__nc_load_actions):
    # Various testcases for translation of grid-mappings

    def test_basic_latlon(self):
        # A basic reference example with a lat-long grid.
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_grid_mapping_latitude_longitude
        #     003 : fc_provides_coordinate_latitude
        #     004 : fc_provides_coordinate_longitude
        #     005 : fc_build_coordinate_latitude
        #     006 : fc_build_coordinate_longitude
        # Notes:
        #     grid-mapping: regular latlon
        #     dim-coords: lat+lon
        #     coords-build: standard latlon coords (with latlon coord-system)
        result = self.run_testcase()
        self.check_result(result)

    def test_missing_latlon_radius(self):
        # Lat-long with a missing earth-radius causes an error.
        # One of very few cases where activation may encounter an error.
        # N.B. doesn't really test rules-activation, but maybe worth doing.
        # (no rules trigger)
        with self.assertRaisesRegex(ValueError, "No ellipsoid"):
            self.run_testcase(gridmapvar_missingradius=True)

    def test_bad_gridmapping_nameproperty(self):
        # Fix the 'grid' var so it does not register as a grid-mapping.
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_coordinate_latitude
        #     003 : fc_provides_coordinate_longitude
        #     004 : fc_build_coordinate_latitude_nocs
        #     005 : fc_build_coordinate_longitude_nocs
        # Notes:
        #     grid-mapping: NONE
        #     dim-coords: lat+lon
        #     coords-build: latlon coords NO coord-system
        result = self.run_testcase(gridmapvar_mappropertyname="mappy")
        self.check_result(result, cube_no_cs=True)

    def test_latlon_bad_gridmapping_varname(self):
        # rename the grid-mapping variable so it is effectively 'missing'
        # (I.E. the var named in "data-variable:grid_mapping" does not exist).
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_coordinate_latitude
        #     003 : fc_provides_coordinate_longitude
        #     004 : fc_build_coordinate_latitude_nocs
        #     005 : fc_build_coordinate_longitude_nocs
        # Notes:
        #     no coord-system
        #     all the same as test_bad_gridmapping_nameproperty
        with self.assertWarnsRegexp("Missing.*grid mapping variable 'grid'"):
            result = self.run_testcase(gridmapvar_name="grid_2")
        self.check_result(result, cube_no_cs=True)

    def test_latlon_bad_latlon_unit(self):
        # Check with bad latitude units : 'degrees' in place of 'degrees_north'.
        #
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_grid_mapping_latitude_longitude
        #     003 : fc_provides_coordinate_longitude
        #     004 : fc_build_coordinate_longitude
        #     005 : fc_default_coordinate
        # Notes:
        #     grid-mapping: regular latlon
        #     dim-coords:
        #         x is regular longitude dim-coord
        #         y is 'default' coord ==> builds as an 'extra' dim-coord
        #     coords-build:
        #         x(lon) is regular latlon with coord-system
        #         y(lat) is a dim-coord, but NO coord-system
        # = "fc_provides_coordinate_latitude" does not trigger, because it is
        #   not a valid latitude coordinate.
        result = self.run_testcase(latitude_units="degrees")
        self.check_result(result, latitude_no_cs=True)

    def test_mapping_rotated(self):
        # Test with rotated-latlon grid-mapping
        # Distinct from both regular-latlon and non-latlon cases, as the
        # coordinate standard names and units are different.
        # (run_testcase/_make_testcase_cdl know how to handle that).
        #
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_grid_mapping_rotated_latitude_longitude
        #     003 : fc_provides_coordinate_latitude
        #     004 : fc_provides_coordinate_longitude
        #     005 : fc_build_coordinate_latitude_rotated
        #     006 : fc_build_coordinate_longitude_rotated
        # Notes:
        #     grid-mapping: rotated lat-lon
        #     dim-coords: lat+lon
        #     coords-build: lat+lon coords ROTATED, with coord-system
        #         (rotated means different name + units)
        result = self.run_testcase(
            mapping_name=hh.CF_GRID_MAPPING_ROTATED_LAT_LON
        )
        self.check_result(result, cube_cstype=ics.RotatedGeogCS)

    #
    # All non-latlon coordinate systems ...
    # These all have projection-x/y coordinates with units of metres.
    # They all work the same way, except that Mercator/Stereographic have
    # parameter checking routines that can fail.
    # NOTE: various mapping types *require* certain addtional properties
    #   - without which an error will occur during translation.
    #   - run_testcase/_make_testcase_cdl know how to provide these
    #
    # Rules Triggered:
    #     001 : fc_default
    #     002 : fc_provides_grid_mapping_<XXX-mapping-name-XXX>
    #     003 : fc_provides_projection_x_coordinate
    #     004 : fc_provides_projection_y_coordinate
    #     005 : fc_build_coordinate_projection_x_<XXX-mapping-name-XXX>
    #     006 : fc_build_coordinate_projection_y_<XXX-mapping-name-XXX>
    # Notes:
    #     grid-mapping: <XXX>
    #     dim-coords: proj-x and -y
    #     coords-build: proj-x/-y_<XXX>, with coord-system

    def test_mapping_albers(self):
        result = self.run_testcase(mapping_name=hh.CF_GRID_MAPPING_ALBERS)
        self.check_result(result, cube_cstype=ics.AlbersEqualArea)

    def test_mapping_geostationary(self):
        result = self.run_testcase(
            mapping_name=hh.CF_GRID_MAPPING_GEOSTATIONARY
        )
        self.check_result(result, cube_cstype=ics.Geostationary)

    def test_mapping_lambert_azimuthal(self):
        result = self.run_testcase(
            mapping_name=hh.CF_GRID_MAPPING_LAMBERT_AZIMUTHAL
        )
        self.check_result(result, cube_cstype=ics.LambertAzimuthalEqualArea)

    def test_mapping_lambert_conformal(self):
        result = self.run_testcase(
            mapping_name=hh.CF_GRID_MAPPING_LAMBERT_CONFORMAL
        )
        self.check_result(result, cube_cstype=ics.LambertConformal)

    def test_mapping_mercator(self):
        result = self.run_testcase(mapping_name=hh.CF_GRID_MAPPING_MERCATOR)
        self.check_result(result, cube_cstype=ics.Mercator)

    def test_mapping_mercator__fail_unsupported(self):
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_projection_x_coordinate
        #     003 : fc_provides_projection_y_coordinate
        # Notes:
        #     grid-mapping: NONE
        #     dim-coords: proj-x and -y
        #     coords-build: NONE
        # = NO coord-system
        # = NO dim-coords built (cube has no coords)
        with self.assertWarnsRegexp("not yet supported for Mercator"):
            # Set a non-unity scale factor, which mercator cannot handle.
            result = self.run_testcase(
                mapping_name=hh.CF_GRID_MAPPING_MERCATOR,
                mapping_scalefactor=2.0,
            )
        self.check_result(result, cube_no_cs=True, cube_no_xycoords=True)

    def test_mapping_stereographic(self):
        result = self.run_testcase(mapping_name=hh.CF_GRID_MAPPING_STEREO)
        self.check_result(result, cube_cstype=ics.Stereographic)

    def test_mapping_stereographic__fail_unsupported(self):
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_projection_x_coordinate
        #     003 : fc_provides_projection_y_coordinate
        # Notes:
        #     as for 'mercator__fail_unsupported', above
        #     = NO dim-coords built (cube has no coords)
        with self.assertWarnsRegexp("not yet supported for stereographic"):
            # Set a non-unity scale factor, which stereo cannot handle.
            result = self.run_testcase(
                mapping_name=hh.CF_GRID_MAPPING_STEREO,
                mapping_scalefactor=2.0,
            )
        self.check_result(result, cube_no_cs=True, cube_no_xycoords=True)

    def test_mapping_transverse_mercator(self):
        result = self.run_testcase(mapping_name=hh.CF_GRID_MAPPING_TRANSVERSE)
        self.check_result(result, cube_cstype=ics.TransverseMercator)

    def test_mapping_vertical_perspective(self):
        result = self.run_testcase(mapping_name=hh.CF_GRID_MAPPING_VERTICAL)
        self.check_result(result, cube_cstype=ics.VerticalPerspective)


class Test__grid_mapping__pyke_rules(Mixin__grid_mapping, tests.IrisTest):
    # Run grid-mapping tests with Pyke (rules)
    use_pyke = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()


from unittest import skip


@skip
class Test__grid_mapping__nonpyke_actions(Mixin__grid_mapping, tests.IrisTest):
    # Run grid-mapping tests with non-Pyke (actions)
    use_pyke = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()


if __name__ == "__main__":
    tests.main()
