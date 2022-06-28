# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the engine.activate() call within the
`iris.fileformats.netcdf._load_cube` function.

Here, *specifically* testcases relating to grid-mappings and dim-coords.

"""
import iris.tests as tests  # isort: skip

import iris.coord_systems as ics
import iris.fileformats._nc_load_rules.helpers as hh
from iris.tests.unit.fileformats.nc_load_rules.actions import (
    Mixin__nc_load_actions,
)


class Mixin__grid_mapping(Mixin__nc_load_actions):
    # Testcase support routines for testing translation of grid-mappings
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
        xco_is_dim=True,
        yco_is_dim=True,
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

        phenom_auxcoord_names = []
        if xco_is_dim:
            # xdim has same name as xco, making xco a dim-coord
            xdim_name = "xco"
        else:
            # use alternate dim-name, and put xco on the 'coords' list
            # This makes the X coord an aux-coord
            xdim_name = "xdim_altname"
            phenom_auxcoord_names.append("xco")
        if yco_is_dim:
            # ydim has same name as yco, making yco a dim-coord
            ydim_name = "yco"  # This makes the Y coord a dim-coord
        else:
            # use alternate dim-name, and put yco on the 'coords' list
            # This makes the Y coord an aux-coord
            ydim_name = "ydim_altname"
            phenom_auxcoord_names.append("yco")

        # Build a 'phenom:coords' string if needed.
        if phenom_auxcoord_names:
            phenom_coords_string = " ".join(phenom_auxcoord_names)
            phenom_coords_string = f"""
                    phenom:coordinates = "{phenom_coords_string}" ;
"""
        else:
            phenom_coords_string = ""

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
        # Polar stereo needs a special 'latitude of projection origin', a
        # 'straight_vertical_longitude_from_pole' and a `standard_parallel` or
        # `scale_factor_at_projection_origin` so treat it specially
        if mapping_type_name in (hh.CF_GRID_MAPPING_POLAR,):
            latpo_name = hh.CF_ATTR_GRID_LAT_OF_PROJ_ORIGIN
            g_string += f"""
                {g_varname}:{latpo_name} = 90.0 ;
            """
            svl_name = hh.CF_ATTR_GRID_STRAIGHT_VERT_LON
            g_string += f"""
                {g_varname}:{svl_name} = 0.0 ;
            """
            stanpar_name = hh.CF_ATTR_GRID_STANDARD_PARALLEL
            g_string += f"""
                {g_varname}:{stanpar_name} = 1.0 ;
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
                {ydim_name} = 2 ;
                {xdim_name} = 3 ;
            variables:
                double phenom({ydim_name}, {xdim_name}) ;
                    phenom:standard_name = "air_temperature" ;
                    phenom:units = "K" ;
                    phenom:grid_mapping = "grid" ;
{phenom_coords_string}
                double yco({ydim_name}) ;
                    yco:axis = "Y" ;
                    yco:units = "{yco_units}" ;
                    yco:standard_name = "{yco_name}" ;
                double xco({xdim_name}) ;
                    xco:axis = "X" ;
                    xco:units = "{xco_units}" ;
                    xco:standard_name = "{xco_name}" ;
                {g_string}
                data:
                    yco = {yco_values_string} ;
                    xco = 100., 110., 120. ;
            }}
        """
        return cdl_string

    def check_result(
        self,
        cube,
        cube_cstype=None,
        cube_no_cs=False,
        cube_no_xycoords=False,
        xco_no_cs=False,  # N.B. no effect if cube_no_cs is True
        yco_no_cs=False,  # N.B. no effect if cube_no_cs is True
        xco_is_aux=False,
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
        expected_dim_coords = []
        expected_aux_coords = []
        if yco_is_aux:
            expected_aux_coords += y_coords
        else:
            expected_dim_coords += y_coords
        if xco_is_aux:
            expected_aux_coords += x_coords
        else:
            expected_dim_coords += x_coords

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


class Test__grid_mapping(Mixin__grid_mapping, tests.IrisTest):
    # Various testcases for translation of grid-mappings
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def test_basic_latlon(self):
        # A basic reference example with a lat-long grid.
        #
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_grid_mapping_(latitude_longitude)
        #     003 : fc_provides_coordinate_(latitude)
        #     004 : fc_provides_coordinate_(longitude)
        #     005 : fc_build_coordinate_(latitude)
        #     006 : fc_build_coordinate_(longitude)
        # Notes:
        #     * grid-mapping identified : regular latlon
        #     * dim-coords identified : lat+lon
        #     * coords built : standard latlon (with latlon coord-system)
        result = self.run_testcase()
        self.check_result(result)

    def test_missing_latlon_radius(self):
        # Lat-long with a missing earth-radius causes an error.
        # One of very few cases where activation may encounter an error.
        # N.B. doesn't really test rules-activation, but maybe worth doing.
        # (no rules trigger)
        with self.assertRaisesRegex(ValueError, "No ellipsoid"):
            self.run_testcase(mapping_missingradius=True)

    def test_bad_gridmapping_nameproperty(self):
        # Fix the 'grid' var so it does not register as a grid-mapping.
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_grid_mapping --FAILED(no grid-mapping attr)
        #     003 : fc_provides_coordinate_(latitude)
        #     004 : fc_provides_coordinate_(longitude)
        #     005 : fc_build_coordinate_(latitude)(no-cs)
        #     006 : fc_build_coordinate_(longitude)(no-cs)
        # Notes:
        #     * grid-mapping identified :  NONE  (thus, no coord-system)
        #     * dim-coords identified : lat+lon
        #     * coords built : lat+lon coords, with NO coord-system
        result = self.run_testcase(gridmapvar_mappropertyname="mappy")
        self.check_result(result, cube_no_cs=True)

    def test_latlon_bad_gridmapping_varname(self):
        # rename the grid-mapping variable so it is effectively 'missing'
        # (I.E. the var named in "data-variable:grid_mapping" does not exist).
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_coordinate_(latitude)
        #     003 : fc_provides_coordinate_(longitude)
        #     004 : fc_build_coordinate_(latitude)(no-cs)
        #     005 : fc_build_coordinate_(longitude)(no-cs)
        # Notes:
        #     * behaviours all the same as 'test_bad_gridmapping_nameproperty'
        warning = "Missing.*grid mapping variable 'grid'"
        result = self.run_testcase(
            warning_regex=warning, gridmapvar_name="grid_2"
        )
        self.check_result(result, cube_no_cs=True)

    def test_latlon_bad_latlon_unit(self):
        # Check with bad latitude units : 'degrees' in place of 'degrees_north'.
        #
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_grid_mapping_(latitude_longitude)
        #     003 : fc_default_coordinate_(provide-phase)
        #     004 : fc_provides_coordinate_(longitude)
        #     005 : fc_build_coordinate_(miscellaneous)
        #     006 : fc_build_coordinate_(longitude)
        # Notes:
        #     * grid-mapping identified : regular latlon
        #     * dim-coords identified :
        #         x is regular longitude dim-coord
        #         y is 'default' coord ==> builds as an 'extra' dim-coord
        #     * coords built :
        #         x(lon) is regular latlon with coord-system
        #         y(lat) is a dim-coord, but with NO coord-system
        #     * additional :
        #         "fc_provides_coordinate_latitude" did not trigger,
        #         because it is not a valid latitude coordinate.
        result = self.run_testcase(latitude_units="degrees")
        self.check_result(result, yco_no_cs=True)

    def test_mapping_rotated(self):
        # Test with rotated-latlon grid-mapping
        # Distinct from both regular-latlon and non-latlon cases, as the
        # coordinate standard names and units are different.
        # ('_make_testcase_cdl' and 'check_result' know how to handle that).
        #
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_grid_mapping_(rotated_latitude_longitude)
        #     003 : fc_provides_coordinate_(rotated_latitude)
        #     004 : fc_provides_coordinate_(rotated_longitude)
        #     005 : fc_build_coordinate_(rotated_latitude)(rotated)
        #     006 : fc_build_coordinate_(rotated_longitude)(rotated)
        # Notes:
        #     * grid-mapping identified : rotated lat-lon
        #     * dim-coords identified : lat+lon
        #     * coords built: lat+lon coords ROTATED, with coord-system
        #         - "rotated" means that they have a different name + units
        result = self.run_testcase(
            mapping_type_name=hh.CF_GRID_MAPPING_ROTATED_LAT_LON
        )
        self.check_result(result, cube_cstype=ics.RotatedGeogCS)

    #
    # All non-latlon coordinate systems ...
    # These all have projection-x/y coordinates with units of metres.
    # They all work the same way.
    # NOTE: various mapping types *require* certain addtional properties
    #   - without which an error will occur during translation.
    #   - run_testcase/_make_testcase_cdl know how to provide these
    #
    # Rules Triggered:
    #     001 : fc_default
    #     002 : fc_provides_grid_mapping_(<XXX-mapping-name-XXX>)
    #     003 : fc_provides_coordinate_(projection_y)
    #     004 : fc_provides_coordinate_(projection_x)
    #     005 : fc_build_coordinate_(projection_y)
    #     006 : fc_build_coordinate_(projection_x)
    # Notes:
    #     * grid-mapping identified : <XXX>
    #     * dim-coords identified : projection_<x,y>_coordinate
    #     * coords built : projection_<x,y>_coordinate, with coord-system
    def test_mapping_albers(self):
        result = self.run_testcase(mapping_type_name=hh.CF_GRID_MAPPING_ALBERS)
        self.check_result(result, cube_cstype=ics.AlbersEqualArea)

    def test_mapping_geostationary(self):
        result = self.run_testcase(
            mapping_type_name=hh.CF_GRID_MAPPING_GEOSTATIONARY
        )
        self.check_result(result, cube_cstype=ics.Geostationary)

    def test_mapping_lambert_azimuthal(self):
        result = self.run_testcase(
            mapping_type_name=hh.CF_GRID_MAPPING_LAMBERT_AZIMUTHAL
        )
        self.check_result(result, cube_cstype=ics.LambertAzimuthalEqualArea)

    def test_mapping_lambert_conformal(self):
        result = self.run_testcase(
            mapping_type_name=hh.CF_GRID_MAPPING_LAMBERT_CONFORMAL
        )
        self.check_result(result, cube_cstype=ics.LambertConformal)

    def test_mapping_mercator(self):
        result = self.run_testcase(
            mapping_type_name=hh.CF_GRID_MAPPING_MERCATOR
        )
        self.check_result(result, cube_cstype=ics.Mercator)

    def test_mapping_stereographic(self):
        result = self.run_testcase(mapping_type_name=hh.CF_GRID_MAPPING_STEREO)
        self.check_result(result, cube_cstype=ics.Stereographic)

    def test_mapping_polar_stereographic(self):
        result = self.run_testcase(mapping_type_name=hh.CF_GRID_MAPPING_POLAR)
        self.check_result(result, cube_cstype=ics.PolarStereographic)

    def test_mapping_transverse_mercator(self):
        result = self.run_testcase(
            mapping_type_name=hh.CF_GRID_MAPPING_TRANSVERSE
        )
        self.check_result(result, cube_cstype=ics.TransverseMercator)

    def test_mapping_vertical_perspective(self):
        result = self.run_testcase(
            mapping_type_name=hh.CF_GRID_MAPPING_VERTICAL
        )
        self.check_result(result, cube_cstype=ics.VerticalPerspective)

    def test_mapping_unsupported(self):
        # Use azimuthal, which is a real thing but we don't yet support it.
        #
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_grid_mapping --FAILED(unhandled type azimuthal_equidistant)
        #     003 : fc_provides_coordinate_(projection_y)
        #     004 : fc_provides_coordinate_(projection_x)
        #     005 : fc_build_coordinate_(projection_y)(FAILED projected coord with non-projected cs)
        #     006 : fc_build_coordinate_(projection_x)(FAILED projected coord with non-projected cs)
        # Notes:
        #     * NO grid-mapping is identified (or coord-system built)
        #     * There is no warning for this : it fails silently.
        #         TODO: perhaps there _should_ be a warning in such cases ?
        result = self.run_testcase(
            mapping_type_name=hh.CF_GRID_MAPPING_AZIMUTHAL
        )
        self.check_result(result, cube_no_cs=True, cube_no_xycoords=True)

    def test_mapping_undefined(self):
        # Use a random, unknown "mapping type".
        #
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_grid_mapping --FAILED(unhandled type unknown)
        #     003 : fc_provides_coordinate_(projection_y)
        #     004 : fc_provides_coordinate_(projection_x)
        #     005 : fc_build_coordinate_(projection_y)(FAILED projected coord with non-projected cs)
        #     006 : fc_build_coordinate_(projection_x)(FAILED projected coord with non-projected cs)
        # Notes:
        #     * There is no warning for this : it fails silently.
        #         TODO: perhaps there _should_ be a warning in such cases ?
        result = self.run_testcase(mapping_type_name="unknown")
        self.check_result(result, cube_no_cs=True, cube_no_xycoords=True)

    #
    # Cases where names(+units) of coords don't match the grid-mapping type.
    # Effectively, there are 9 possibilities for (latlon/rotated/projected)
    # coords mismatched to (latlon/rotated/projected/missing) coord-systems.
    #
    # N.B. the results are not all the same :
    #
    #  1. when a coord and the grid-mapping have the same 'type',
    #     i.e. plain-latlon, rotated-latlon or non-latlon, then dim-coords are
    #     built with a coord-system (as seen previously).
    #  2. when there is no grid-mapping, we can build coords of any type,
    #     but with no coord-system.
    #  3. when one of (coord + grid-mapping) is plain-latlon or rotated-latlon,
    #     and the other is non-latlon (i.e. any other type),
    #     then we build coords *without* a coord-system
    #  4. when one of (coord + grid-mapping) is plain-latlon, and the other is
    #     rotated-latlon, we don't build coords at all.
    #     TODO: it's not clear why this needs to behave differently from case
    #     (3.) : possibly, these two should be made consistent.
    #
    # TODO: *all* these 'mismatch' cases should probably generate warnings,
    # except for plain-latlon coords with no grid-mapping.
    # At present, we _only_ warn when an expected grid-mapping is absent.
    #

    def test_mapping__mismatch__latlon_coords_rotated_system(self):
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_grid_mapping_(rotated_latitude_longitude)
        #     003 : fc_provides_coordinate_(latitude)
        #     004 : fc_provides_coordinate_(longitude)
        #     005 : fc_build_coordinate_(latitude)(FAILED : latlon coord with rotated cs)
        #     006 : fc_build_coordinate_(longitude)(FAILED : latlon coord with rotated cs)
        # Notes:
        #     * coords built : NONE  (see above)
        result = self.run_testcase(
            mapping_type_name=hh.CF_GRID_MAPPING_ROTATED_LAT_LON,
            xco_name="longitude",
            xco_units="degrees_east",
            yco_name="latitude",
            yco_units="degrees_north",
        )
        self.check_result(result, cube_no_cs=True, cube_no_xycoords=True)

    def test_mapping__mismatch__latlon_coords_nonll_system(self):
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_grid_mapping_(albers_conical_equal_area)
        #     003 : fc_provides_coordinate_(latitude)
        #     004 : fc_provides_coordinate_(longitude)
        #     005 : fc_build_coordinate_(latitude)(no-cs : discarded projected cs)
        #     006 : fc_build_coordinate_(longitude)(no-cs : discarded projected cs)
        # Notes:
        #     * coords built : lat + lon, with no coord-system (see above)
        result = self.run_testcase(
            mapping_type_name=hh.CF_GRID_MAPPING_ALBERS,
            xco_name="longitude",
            xco_units="degrees_east",
            yco_name="latitude",
            yco_units="degrees_north",
        )
        self.check_result(result, cube_no_cs=True)

    def test_mapping__mismatch__latlon_coords_missing_system(self):
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_coordinate_(latitude)
        #     003 : fc_provides_coordinate_(longitude)
        #     004 : fc_build_coordinate_(latitude)(no-cs)
        #     005 : fc_build_coordinate_(longitude)(no-cs)
        # Notes:
        #     * coords built : lat + lon, with no coord-system (see above)
        warning = "Missing.*grid mapping variable 'grid'"
        result = self.run_testcase(
            warning_regex=warning,
            gridmapvar_name="moved",
            xco_name="longitude",
            xco_units="degrees_east",
            yco_name="latitude",
            yco_units="degrees_north",
        )
        self.check_result(result, cube_no_cs=True)

    def test_mapping__mismatch__rotated_coords_latlon_system(self):
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_grid_mapping_(latitude_longitude)
        #     003 : fc_provides_coordinate_(rotated_latitude)
        #     004 : fc_provides_coordinate_(rotated_longitude)
        #     005 : fc_build_coordinate_(rotated_latitude)(FAILED rotated coord with latlon cs)
        #     006 : fc_build_coordinate_(rotated_longitude)(FAILED rotated coord with latlon cs)
        # Notes:
        #     * coords built : NONE  (see above)
        result = self.run_testcase(
            xco_name="grid_longitude",
            xco_units="degrees",
            yco_name="grid_latitude",
            yco_units="degrees",
        )
        self.check_result(result, cube_no_cs=True, cube_no_xycoords=True)

    def test_mapping__mismatch__rotated_coords_nonll_system(self):
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_grid_mapping_(albers_conical_equal_area)
        #     003 : fc_provides_coordinate_(rotated_latitude)
        #     004 : fc_provides_coordinate_(rotated_longitude)
        #     005 : fc_build_coordinate_(rotated_latitude)(rotated no-cs : discarded projected cs)
        #     006 : fc_build_coordinate_(rotated_longitude)(rotated no-cs : discarded projected cs)
        # Notes:
        #     * coords built : rotated-lat + lon, with no coord-system (see above)
        result = self.run_testcase(
            mapping_type_name=hh.CF_GRID_MAPPING_ALBERS,
            xco_name="grid_longitude",
            xco_units="degrees",
            yco_name="grid_latitude",
            yco_units="degrees",
        )
        self.check_result(result, cube_no_cs=True)

    def test_mapping__mismatch__rotated_coords_missing_system(self):
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_coordinate_(rotated_latitude)
        #     003 : fc_provides_coordinate_(rotated_longitude)
        #     004 : fc_build_coordinate_(rotated_latitude)(rotated no-cs)
        #     005 : fc_build_coordinate_(rotated_longitude)(rotated no-cs)
        # Notes:
        #     * coords built : rotated lat + lon, with no coord-system (see above)
        warning = "Missing.*grid mapping variable 'grid'"
        result = self.run_testcase(
            warning_regex=warning,
            gridmapvar_name="moved",
            xco_name="grid_longitude",
            xco_units="degrees",
            yco_name="grid_latitude",
            yco_units="degrees",
        )
        self.check_result(result, cube_no_cs=True)

    def test_mapping__mismatch__nonll_coords_latlon_system(self):
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_grid_mapping_(latitude_longitude)
        #     003 : fc_default_coordinate_(provide-phase)
        #     004 : fc_default_coordinate_(provide-phase)
        #     005 : fc_build_coordinate_(miscellaneous)
        #     006 : fc_build_coordinate_(miscellaneous)
        # Notes:
        #     * coords built : projection x + y, with no coord-system (see above)
        #     * the coords build as "default" type : they have no standard-name
        result = self.run_testcase(
            xco_name="projection_x",
            xco_units="m",
            yco_name="projection_y",
            yco_units="m",
        )
        self.check_result(
            result, cube_no_cs=True, xco_stdname=False, yco_stdname=False
        )

    def test_mapping__mismatch__nonll_coords_rotated_system(self):
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_grid_mapping_(rotated_latitude_longitude)
        #     003 : fc_default_coordinate_(provide-phase)
        #     004 : fc_default_coordinate_(provide-phase)
        #     005 : fc_build_coordinate_(miscellaneous)
        #     006 : fc_build_coordinate_(miscellaneous)
        # Notes:
        #     * as previous case '__mismatch__nonll_'
        result = self.run_testcase(
            mapping_type_name=hh.CF_GRID_MAPPING_ROTATED_LAT_LON,
            xco_name="projection_x",
            xco_units="m",
            yco_name="projection_y",
            yco_units="m",
        )
        self.check_result(
            result, cube_no_cs=True, xco_stdname=False, yco_stdname=False
        )

    def test_mapping__mismatch__nonll_coords_missing_system(self):
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_default_coordinate_(provide-phase)
        #     003 : fc_default_coordinate_(provide-phase)
        #     004 : fc_build_coordinate_(miscellaneous)
        #     005 : fc_build_coordinate_(miscellaneous)
        # Notes:
        #     * effectively, just like previous 2 cases
        warning = "Missing.*grid mapping variable 'grid'"
        result = self.run_testcase(
            warning_regex=warning,
            gridmapvar_name="moved",
            xco_name="projection_x",
            xco_units="m",
            yco_name="projection_y",
            yco_units="m",
        )
        self.check_result(
            result, cube_no_cs=True, xco_stdname=False, yco_stdname=False
        )


class Test__aux_latlons(Mixin__grid_mapping, tests.IrisTest):
    # Testcases for translating auxiliary latitude+longitude variables
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def test_aux_lon(self):
        # Change the name of xdim, and put xco on the coords list.
        #
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_grid_mapping_(latitude_longitude)
        #     003 : fc_provides_coordinate_(latitude)
        #     004 : fc_build_coordinate_(latitude)
        #     005 : fc_build_auxiliary_coordinate_longitude
        result = self.run_testcase(xco_is_dim=False)
        self.check_result(result, xco_is_aux=True, xco_no_cs=True)

    def test_aux_lat(self):
        # As previous, but with the Y coord.
        #
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_grid_mapping_(latitude_longitude)
        #     003 : fc_provides_coordinate_(longitude)
        #     004 : fc_build_coordinate_(longitude)
        #     005 : fc_build_auxiliary_coordinate_latitude
        result = self.run_testcase(yco_is_dim=False)
        self.check_result(result, yco_is_aux=True, yco_no_cs=True)

    def test_aux_lat_and_lon(self):
        # Make *both* X and Y coords into aux-coords.
        #
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_grid_mapping_(latitude_longitude)
        #     003 : fc_build_auxiliary_coordinate_longitude
        #     004 : fc_build_auxiliary_coordinate_latitude
        # Notes:
        #     * a grid-mapping is recognised, but discarded, as in this case
        #       there are no dim-coords to reference it.
        result = self.run_testcase(xco_is_dim=False, yco_is_dim=False)
        self.check_result(
            result, xco_is_aux=True, yco_is_aux=True, cube_no_cs=True
        )

    def test_aux_lon_rotated(self):
        # Rotated-style lat + lon coords, X is an aux-coord.
        #
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_grid_mapping_(rotated_latitude_longitude)
        #     003 : fc_provides_coordinate_(rotated_latitude)
        #     004 : fc_build_coordinate_(rotated_latitude)(rotated)
        #     005 : fc_build_auxiliary_coordinate_longitude_rotated
        # Notes:
        #     * as the plain-latlon case 'test_aux_lon'.
        result = self.run_testcase(
            mapping_type_name=hh.CF_GRID_MAPPING_ROTATED_LAT_LON,
            xco_is_dim=False,
        )
        self.check_result(result, xco_is_aux=True, xco_no_cs=True)

    def test_aux_lat_rotated(self):
        # Rotated-style lat + lon coords, Y is an aux-coord.
        #
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_grid_mapping_(rotated_latitude_longitude)
        #     003 : fc_provides_coordinate_(rotated_longitude)
        #     004 : fc_build_coordinate_(rotated_longitude)(rotated)
        #     005 : fc_build_auxiliary_coordinate_latitude_rotated
        # Notes:
        #     * as the plain-latlon case 'test_aux_lat'.
        result = self.run_testcase(
            mapping_type_name=hh.CF_GRID_MAPPING_ROTATED_LAT_LON,
            yco_is_dim=False,
        )
        self.check_result(result, yco_is_aux=True, yco_no_cs=True)


class Test__nondimcoords(Mixin__grid_mapping, tests.IrisTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def test_nondim_lats(self):
        # Fix a coord's values so it cannot be a dim-coord.
        #
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_grid_mapping_(latitude_longitude)
        #     003 : fc_provides_coordinate_(latitude)
        #     004 : fc_provides_coordinate_(longitude)
        #     005 : fc_build_coordinate_(latitude)
        #     006 : fc_build_coordinate_(longitude)
        # Notes:
        #     * in terms of rule triggering, this is not distinct from the
        #       "normal" case : but latitude is now created as an aux-coord.
        warning = "must be.* monotonic"
        result = self.run_testcase(
            warning_regex=warning, yco_values=[0.0, 0.0]
        )
        self.check_result(result, yco_is_aux=True)


if __name__ == "__main__":
    tests.main()
