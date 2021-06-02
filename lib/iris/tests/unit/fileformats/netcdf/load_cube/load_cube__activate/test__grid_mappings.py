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
import iris.tests as tests

import iris.coord_systems as ics
import iris.fileformats._nc_load_rules.helpers as hh

from iris.tests.unit.fileformats.netcdf.load_cube.load_cube__activate import (
    Mixin__nc_load_actions,
)


class Mixin__grid_mapping(Mixin__nc_load_actions):
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
            self.run_testcase(mapping_missingradius=True)

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
        warning = "Missing.*grid mapping variable 'grid'"
        result = self.run_testcase(warning=warning, gridmapvar_name="grid_2")
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
        self.check_result(result, yco_no_cs=True)

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
            mapping_type_name=hh.CF_GRID_MAPPING_ROTATED_LAT_LON
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
        # Set a non-unity scale factor, which mercator cannot handle.
        warning = "not yet supported for Mercator"
        result = self.run_testcase(
            warning=warning,
            mapping_type_name=hh.CF_GRID_MAPPING_MERCATOR,
            mapping_scalefactor=2.0,
        )
        self.check_result(result, cube_no_cs=True, cube_no_xycoords=True)

    def test_mapping_stereographic(self):
        result = self.run_testcase(mapping_type_name=hh.CF_GRID_MAPPING_STEREO)
        self.check_result(result, cube_cstype=ics.Stereographic)

    def test_mapping_stereographic__fail_unsupported(self):
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_projection_x_coordinate
        #     003 : fc_provides_projection_y_coordinate
        # Notes:
        #     as for 'mercator__fail_unsupported', above
        #     = NO dim-coords built (cube has no coords)
        #
        # Set a non-unity scale factor, which stereo cannot handle.
        warning = "not yet supported for stereographic"
        result = self.run_testcase(
            warning=warning,
            mapping_type_name=hh.CF_GRID_MAPPING_STEREO,
            mapping_scalefactor=2.0,
        )
        self.check_result(result, cube_no_cs=True, cube_no_xycoords=True)

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
        #     002 : fc_provides_projection_x_coordinate
        #     003 : fc_provides_projection_y_coordinate
        # NOTES:
        #   - there is no warning for this.
        # TODO: perhaps there should be ?
        result = self.run_testcase(
            mapping_type_name=hh.CF_GRID_MAPPING_AZIMUTHAL
        )
        self.check_result(result, cube_no_cs=True, cube_no_xycoords=True)

    def test_mapping_undefined(self):
        # Use a random, unknown "mapping type".
        #
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_projection_x_coordinate
        #     003 : fc_provides_projection_y_coordinate
        # NOTES:
        #   - there is no warning for this.
        # TODO: perhaps there should be ?
        result = self.run_testcase(mapping_type_name="unknown")
        self.check_result(result, cube_no_cs=True, cube_no_xycoords=True)

    #
    # Cases where names(+units) of coords don't match the grid-mapping type
    # Effectively, there are 9 possibilities for (latlon/rotated/projected)
    # coords against (latlon/rotated/projected/missing) coord-systems.
    # N.B. the results are not all the same ...
    #

    def test_mapping__mismatch__latlon_coords_rotated_system(self):
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_grid_mapping_rotated_latitude_longitude
        #     003 : fc_provides_coordinate_latitude
        #     004 : fc_provides_coordinate_longitude
        # NOTES:
        #   no build_coord triggers, as it requires the correct mapping type
        #   so no dim-coords at all in this case
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
        #     002 : fc_provides_grid_mapping_albers_equal_area
        #     003 : fc_provides_coordinate_latitude
        #     004 : fc_provides_coordinate_longitude
        #     005 : fc_build_coordinate_latitude_nocs
        #     006 : fc_build_coordinate_longitude_nocs
        # NOTES:
        #   build_coord_XXX_cs triggers, requires NO latlon/rotated mapping
        #   - but a non-ll mapping is 'ok'.
        # TODO: not really clear why this is right ?
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
        #     002 : fc_provides_coordinate_latitude
        #     003 : fc_provides_coordinate_longitude
        #     004 : fc_build_coordinate_latitude_nocs
        #     005 : fc_build_coordinate_longitude_nocs
        # NOTES:
        #  same as nonll, except *NO* grid-mapping is detected,
        #  - which makes no practical difference
        warning = "Missing.*grid mapping variable 'grid'"
        result = self.run_testcase(
            warning=warning,
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
        #     002 : fc_provides_grid_mapping_latitude_longitude
        #     003 : fc_provides_coordinate_latitude
        #     004 : fc_provides_coordinate_longitude
        # NOTES:
        #   no build_coord triggers : requires NO latlon/rotated mapping
        #   hence no coords at all
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
        #     002 : fc_provides_grid_mapping_albers_equal_area
        #     003 : fc_provides_coordinate_latitude
        #     004 : fc_provides_coordinate_longitude
        #     005 : fc_build_coordinate_latitude_nocs
        #     006 : fc_build_coordinate_longitude_nocs
        # NOTES:
        #  this is different from the previous
        #  build_coord.._nocs triggers : requires NO latlon/rotated mapping
        #  - which seems odd + inconsistent (with previous) ?
        # TODO: should this change ??
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
        #     002 : fc_provides_coordinate_latitude
        #     003 : fc_provides_coordinate_longitude
        #     004 : fc_build_coordinate_latitude_nocs
        #     005 : fc_build_coordinate_longitude_nocs
        # NOTES:
        #  as previous, but no grid-mapping (which makes no difference)
        warning = "Missing.*grid mapping variable 'grid'"
        result = self.run_testcase(
            warning=warning,
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
        #     002 : fc_provides_grid_mapping_latitude_longitude
        #     003 : fc_default_coordinate
        #     004 : fc_default_coordinate
        # NOTES:
        #  dim-coords built as "defaults" : dim-coords, but NO standard name
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
        #     002 : fc_provides_grid_mapping_rotated_latitude_longitude
        #     003 : fc_default_coordinate
        #     004 : fc_default_coordinate
        # NOTES:
        #  same as previous __mismatch__nonll_
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
        #     002 : fc_default_coordinate
        #     003 : fc_default_coordinate
        # NOTES:
        #  effectively, just like previous 2 __mismatch__nonll_
        warning = "Missing.*grid mapping variable 'grid'"
        result = self.run_testcase(
            warning=warning,
            gridmapvar_name="moved",
            xco_name="projection_x",
            xco_units="m",
            yco_name="projection_y",
            yco_units="m",
        )
        self.check_result(
            result, cube_no_cs=True, xco_stdname=False, yco_stdname=False
        )


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
