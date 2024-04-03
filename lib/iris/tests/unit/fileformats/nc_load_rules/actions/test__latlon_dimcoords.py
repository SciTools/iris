# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the engine.activate() call within the
`iris.fileformats.netcdf._load_cube` function.

Tests for rules behaviour in identifying latitude/longitude dim-coords, both
rotated and non-rotated.

"""

import iris.tests as tests  # isort: skip

from iris.coord_systems import GeogCS, RotatedGeogCS
from iris.tests.unit.fileformats.nc_load_rules.actions import Mixin__nc_load_actions


class Mixin_latlon_dimcoords(Mixin__nc_load_actions):
    # Tests for the recognition and construction of latitude/longitude coords.

    # Control to test either longitude or latitude coords.
    # Set by inheritor classes, which are actual TestCases.
    lat_1_or_lon_0 = None

    def setUp(self):
        super().setUp()
        # Generate some useful settings : just to generalise operation over
        # both latitude and longitude.
        islat = self.lat_1_or_lon_0
        assert islat in (0, 1)
        self.unrotated_name = "latitude" if islat else "longitude"
        self.rotated_name = "grid_latitude" if islat else "grid_longitude"
        self.unrotated_units = "degrees_north" if islat else "degrees_east"
        # Note: there are many alternative valid forms for the rotated units,
        # but we are not testing that here.
        self.rotated_units = "degrees"  # NB this one is actually constant
        self.axis = "y" if islat else "x"

    def _make_testcase_cdl(
        self,
        standard_name=None,
        long_name=None,
        var_name=None,
        units=None,
        axis=None,
        grid_mapping=None,
    ):
        # Inner routine called by 'run_testcase' (in Mixin__nc_load_actions),
        # to generate CDL which is then translated into a testfile and loaded.
        if var_name is None:
            # Can't have *no* var-name
            # N.B. it is also the name of the dimension.
            var_name = "dim"

        def attribute_str(name, value):
            if value is None or value == "":
                result = ""
            else:
                result = f'{var_name}:{name} = "{value}" ;'

            return result

        standard_name_str = attribute_str("standard_name", standard_name)
        long_name_str = attribute_str("long_name", long_name)
        units_str = attribute_str("units", units)
        axis_str = attribute_str("axis", axis)
        if grid_mapping:
            grid_mapping_str = 'phenom:grid_mapping = "crs" ;'
        else:
            grid_mapping_str = ""

        assert grid_mapping in (None, "latlon", "rotated")
        if grid_mapping is None:
            crs_str = ""
        elif grid_mapping == "latlon":
            crs_str = """
      int crs ;
        crs:grid_mapping_name = "latitude_longitude" ;
        crs:semi_major_axis = 6371000.0 ;
        crs:inverse_flattening = 1000. ;
"""
        elif grid_mapping == "rotated":
            crs_str = """
      int crs ;
        crs:grid_mapping_name = "rotated_latitude_longitude" ;
        crs:grid_north_pole_latitude = 32.5 ;
        crs:grid_north_pole_longitude = 170. ;
"""

        cdl_string = f"""
netcdf test {{
    dimensions:
        {var_name} = 2 ;
    variables:
        double {var_name}({var_name}) ;
            {standard_name_str}
            {units_str}
            {long_name_str}
            {axis_str}
        double phenom({var_name}) ;
            phenom:standard_name = "air_temperature" ;
            phenom:units = "K" ;
            {grid_mapping_str}
        {crs_str}
    data:
        {var_name} = 0., 1. ;
}}
"""
        return cdl_string

    def check_result(
        self,
        cube,
        standard_name,
        long_name,
        units,
        crs=None,
        context_message="",
    ):
        # Check the existence, standard-name, long-name, units and coord-system
        # of the resulting coord.  Also that it is always a dim-coord.
        # NOTE: there is no "axis" arg, as this information does *not* appear
        # as a separate property (or attribute) of the resulting coord.
        # However, whether the file variable has an axis attribute *does*
        # affect the results here, in some cases.
        coords = cube.coords()
        # There should be one and only one coord.
        self.assertEqual(1, len(coords))
        # It should also be a dim-coord
        self.assertEqual(1, len(cube.coords(dim_coords=True)))
        (coord,) = coords
        if self.debug:
            print("")
            print("DEBUG : result coord =", coord)
            print("")

        coord_stdname, coord_longname, coord_units, coord_crs = [
            getattr(coord, name)
            for name in ("standard_name", "long_name", "units", "coord_system")
        ]
        self.assertEqual(standard_name, coord_stdname, context_message)
        self.assertEqual(long_name, coord_longname, context_message)
        self.assertEqual(units, coord_units, context_message)
        assert crs in (None, "latlon", "rotated")
        if crs is None:
            self.assertEqual(None, coord_crs, context_message)
        elif crs == "latlon":
            self.assertIsInstance(coord_crs, GeogCS, context_message)
        elif crs == "rotated":
            self.assertIsInstance(coord_crs, RotatedGeogCS, context_message)

    #
    # Testcase routines
    #
    # NOTE: all these testcases have been verified against the older behaviour
    # in v3.0.4, based on Pyke rules.
    #

    def test_minimal(self):
        # Nothing but a var-name --> unrecognised dim-coord.
        result = self.run_testcase()
        self.check_result(result, None, None, "unknown")

    def test_fullinfo_unrotated(self):
        # Check behaviour with all normal info elements for 'unrotated' case.
        # Includes a grid-mapping, but no axis (should not be needed).
        result = self.run_testcase(
            standard_name=self.unrotated_name,
            units=self.unrotated_units,
            grid_mapping="latlon",
        )
        self.check_result(result, self.unrotated_name, None, "degrees", "latlon")

    def test_fullinfo_rotated(self):
        # Check behaviour with all normal info elements for 'rotated' case.
        # Includes a grid-mapping, but no axis (should not be needed).
        result = self.run_testcase(
            standard_name=self.rotated_name,
            units=self.rotated_units,
            grid_mapping="rotated",
        )
        self.check_result(result, self.rotated_name, None, "degrees", "rotated")

    def test_axis(self):
        # A suitable axis --> unrotated lat/lon coord, but unknown units.
        result = self.run_testcase(axis=self.axis)
        self.check_result(result, self.unrotated_name, None, "unknown")

    def test_units_unrotated(self):
        # With a unit like 'degrees_east', we automatically identify this as a
        # latlon coord, *and* convert units to plain 'degrees' on loading.
        result = self.run_testcase(units=self.unrotated_units)
        self.check_result(result, self.unrotated_name, None, "degrees")

    def test_units_rotated(self):
        # With no info except a "degrees" unit, we **don't** identify a latlon,
        # i.e. we do not set the standard-name
        result = self.run_testcase(units="degrees")
        self.check_result(result, None, None, "degrees")

    def test_units_unrotated_gridmapping(self):
        # With an unrotated unit *AND* a suitable grid-mapping, we identify a
        # rotated latlon coordinate + assign it the coord-system.
        result = self.run_testcase(units=self.unrotated_units, grid_mapping="latlon")
        self.check_result(result, self.unrotated_name, None, "degrees", "latlon")

    def test_units_rotated_gridmapping_noname(self):
        # Rotated units and grid-mapping, but *without* the expected name.
        # Does not translate, no coord-system (i.e. grid-mapping is discarded).
        result = self.run_testcase(
            units="degrees",
            grid_mapping="rotated",
        )
        self.check_result(result, None, None, "degrees", None)

    def test_units_rotated_gridmapping_withname(self):
        # With a "degrees" unit, a rotated grid-mapping *AND* a suitable
        # standard-name, it recognises a rotated dimcoord.
        result = self.run_testcase(
            standard_name=self.rotated_name,
            units="degrees",
            grid_mapping="rotated",
        )
        self.check_result(result, self.rotated_name, None, "degrees", "rotated")

    def test_units_rotated_gridmapping_varname(self):
        # Same but with var-name containing the standard-name : in this case we
        # get NO COORDINATE-SYSTEM (which is a bit weird).
        result = self.run_testcase(
            var_name=self.rotated_name,
            units="degrees",
            grid_mapping="rotated",
        )
        self.check_result(result, self.rotated_name, None, "degrees", None)

    def test_varname_unrotated(self):
        # With a recognised name in the var-name, we set standard-name.
        # But units are left undetermined.
        result = self.run_testcase(var_name=self.unrotated_name)
        self.check_result(result, self.unrotated_name, None, "unknown")

    def test_varname_rotated(self):
        # With a *rotated* name in the var-name, we set standard-name.
        # But units are left undetermined.
        result = self.run_testcase(var_name=self.rotated_name)
        self.check_result(result, self.rotated_name, None, "unknown")

    def test_varname_unrotated_units_rotated(self):
        # With a "degrees" unit and a suitable var-name, we do identify
        # (= set standard-name).
        # N.B. this accepts "degrees" as a generic term, and so does *not*
        # interpret it as a rotated coordinate.
        result = self.run_testcase(var_name=self.unrotated_name, units="degrees")
        self.check_result(result, self.unrotated_name, None, "degrees")

    def test_longname(self):
        # A recognised form in long-name is *not* translated into standard-name.
        result = self.run_testcase(long_name=self.unrotated_name)
        self.check_result(result, None, self.unrotated_name, "unknown")

    def test_stdname_unrotated(self):
        # Only an (unrotated) standard name : units is not specified
        result = self.run_testcase(standard_name=self.unrotated_name)
        self.check_result(result, self.unrotated_name, None, None)

    def test_stdname_rotated(self):
        # Only a (rotated) standard name : units is not specified
        result = self.run_testcase(standard_name=self.rotated_name)
        self.check_result(result, self.rotated_name, None, None)

    def test_stdname_unrotated_gridmapping(self):
        # An unrotated standard-name and grid-mapping, translates into a
        # coordinate system.
        result = self.run_testcase(
            standard_name=self.unrotated_name, grid_mapping="latlon"
        )
        self.check_result(result, self.unrotated_name, None, "unknown", "latlon")

    def test_stdname_rotated_gridmapping(self):
        # An *rotated* standard-name and grid-mapping, translates into a
        # coordinate system.
        result = self.run_testcase(
            standard_name=self.rotated_name, grid_mapping="rotated"
        )
        self.check_result(result, self.rotated_name, None, None, "rotated")


class Test__longitude_coords(Mixin_latlon_dimcoords, tests.IrisTest):
    lat_1_or_lon_0 = 0

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def setUp(self):
        super().setUp()


class Test__latitude_coords(Mixin_latlon_dimcoords, tests.IrisTest):
    lat_1_or_lon_0 = 1

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def setUp(self):
        super().setUp()


if __name__ == "__main__":
    tests.main()
