# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the engine.activate() call within the
`iris.fileformats.netcdf._load_cube` function.

Tests for rules behaviour in identifying latitude/longitude dim-coords, both
rotated and non-rotated.

"""
import iris.tests as tests  # isort: skip

from iris.coord_systems import GeogCS, RotatedGeogCS
from iris.tests.unit.fileformats.nc_load_rules.actions import (
    Mixin__nc_load_actions,
)


class Test__lon_dimcoords(Mixin__nc_load_actions, tests.IrisTest):
    # Tests for handling of the special UM-specific data-var attributes.
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def _make_testcase_cdl(
        self,
        standard_name=None,
        long_name=None,
        var_name=None,
        units=None,
        axis=None,
        grid_mapping=None,
    ):

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
        # Check the standard-name, long-name and units of the resulting coord.
        # NOTE: there is no "axis" arg, as this aspect does *not* appear as
        # a property (or attribute) of the resulting coord.  It *does* affect
        # the "identification process" though (what we are testing here).
        coords = cube.coords()
        self.assertEqual(len(coords), 1)
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
            self.assertEqual(None, coord_crs)
        elif crs == "latlon":
            self.assertIsInstance(coord_crs, GeogCS)
        elif crs == "rotated":
            self.assertIsInstance(coord_crs, RotatedGeogCS)

    #
    # Testcase routines
    #

    def test_minimal(self):
        result = self.run_testcase()  # Nothing but the var-name.
        self.check_result(result, None, None, "unknown")

    def test_axis(self):
        result = self.run_testcase(axis="x")
        self.check_result(result, "longitude", None, "unknown")

    def test_axis_units_unrotated(self):
        result = self.run_testcase(units="degrees_east", axis="x")
        self.check_result(result, "longitude", None, "degrees")

    def test_unrotated_units(self):
        # With a unit of 'degrees_east', we automatically identify as longitude
        # *And* units are converted to plain 'degrees' on loading.
        result = self.run_testcase(units="degrees_east")
        self.check_result(result, "longitude", None, "degrees")

    def test_rotated_units(self):
        # With just a "degrees" unit, we don't identify as latlon
        result = self.run_testcase(units="degrees")
        self.check_result(result, None, None, "degrees")

    def test_varname_longitude(self):
        # With a recognised var-name, we do identify a longitude
        # But the units are not determined
        result = self.run_testcase(var_name="longitude")
        self.check_result(result, "longitude", None, "unknown")

    def test_varname_lons(self):
        # A var-name that matches a regexp, but is not exact.
        result = self.run_testcase(var_name="lons")
        self.check_result(result, None, None, "unknown")

    def test_longname_lons(self):
        # This does not match.
        result = self.run_testcase(long_name="lons")
        self.check_result(result, None, "lons", "unknown")

    def test_longname_longitude(self):
        # A var-name that matches a regexp, but is not exact.
        result = self.run_testcase(long_name="longitude")
        self.check_result(result, None, "longitude", "unknown")

    def test_longname_gridlongitude(self):
        # This is not recognised
        result = self.run_testcase(long_name="grid_longitude")
        self.check_result(result, None, "grid_longitude", "unknown")

    def test_rotated_units_name(self):
        # With a "degrees" unit and a suitable long-name, we do identify
        # N.B. this is *not* interpreted as rotated
        result = self.run_testcase(units="degrees", var_name="longitude")
        self.check_result(result, "longitude", None, "degrees")

    def test_rotated_units_name_gridmapping(self):
        # With a "degrees" unit and a suitable long-name, *AND* a suitable
        # grid-mapping, we do identify as rotated.
        result = self.run_testcase(
            standard_name="grid_longitude",
            units="degrees",
            grid_mapping="rotated",
        )
        self.check_result(result, "grid_longitude", None, "degrees", "rotated")

    def test_rotated_units_name_wrong_gridmapping(self):
        # With a "degrees" unit and a suitable long-name, *AND* a suitable
        # grid-mapping, we do identify as rotated.
        result = self.run_testcase(
            units="degrees", var_name="xxx", grid_mapping="latlon"
        )
        self.check_result(result, None, None, "degrees")

    def test_unrotated_units_name(self):
        # With a "degrees_east" unit and a suitable long-name, we identify
        result = self.run_testcase(units="degrees_east")
        self.check_result(result, "longitude", None, "degrees")

    def test_unrotated_units_name_gridmapping(self):
        # With a "degrees_east" unit and a suitable long-name, we identify
        result = self.run_testcase(units="degrees_east", grid_mapping="latlon")
        self.check_result(result, "longitude", None, "degrees", "latlon")

    def test_units_wrongaxis(self):
        result = self.run_testcase(units="degrees_east", axis="y")
        self.check_result(result, "longitude", None, "degrees")

    def test_axis_units_rotated(self):
        # This *might* have been interpreted as rotated
        # - but it is not (in the absence of a grid-mapping)
        result = self.run_testcase(units="degrees", axis="x")
        self.check_result(result, "longitude", None, "degrees")

    def test_axis_units_rotated_gridmapping(self):
        # Extension to the previous..
        result = self.run_testcase(
            units="degrees", axis="x", grid_mapping="rotated"
        )
        self.check_result(result, "grid_longitude", None, "degrees", "rotated")

    def test_multifactor(self):
        # Check combinations of the key metadata elements
        # -- that is : standard_name; units; axis
        # We omit both 'long_name' and 'var_name' because, from inspection of
        # the rules code, they are not involved in latlon identification.

        # Function which encodes our expectation of the relevant "rules".
        # TODO: incomplete : need to add 'grid_mapping' as a factor here
        # Note : this logic has been validated against the original Pyke-rules
        # implementation, since we want to preserve all that behaviour.
        def expected_results(standard_name, units, axis):
            """
            Encode our understanding of the longitude indentifying logic.
            NOTE: we don't include long_name in this.  The rules can set
            long_name, when provided with an invalid standard_name, but we have
            no need to test that behavour here.

            Returns:
                (expected_coord_standard_name, expected_coord_units)

            """
            expected_stdname = standard_name
            expected_units = units
            valid = False
            rotated = False

            # First simulate action of 'is_longitude' routine
            if units is not None:
                if units == "degrees":
                    # This one (alone) indicates *rotated* longitudes
                    if standard_name is None:
                        # No standard name, is OK only if we have an axis too.
                        valid = axis == "x"
                    else:
                        # With standard name, only the rotated form is OK.
                        valid = standard_name == "grid_longitude"
                else:
                    # Extended units are all "true" longitude,
                    # i.e. non-rotated, e.g. "degrees_east"
                    valid = units.startswith("degree")
            else:
                if standard_name is not None:
                    # No units, but we do have a standard-name.
                    if "latitude" in standard_name:
                        valid = True
                elif axis == "x":
                    valid = True

            # Next simulate the operation of the 'is_rotated' check.
            if valid:
                if standard_name is not None:
                    rotated = standard_name == "grid_longitude"
                else:
                    rotated = units == "degrees"  # i.e. *not* "_north" et al

            # Now implement in the result : in some circumstances, this will
            # reset either the standard-name or units of the result
            if valid:
                if not rotated:
                    expected_stdname = "longitude"
                elif axis == "x":
                    # If standard name is missing, we don't identify a (rotated)
                    # longitude coord, unless confirmed by the 'axis' setting.
                    expected_stdname = "grid_longitude"

                if expected_units is not None and expected_units.startswith(
                    "degree"
                ):
                    expected_units = "degrees"

            return expected_stdname, expected_units

        # list test options for each factor
        std_name_opts = [
            None,
            "air_temperature",
            "longitude",
            "grid_longitude",
        ]
        units_opts = [None, "degrees", "degrees_east", "K"]
        axis_opts = [None, "x", "z"]
        std_name_opts = [None]
        units_opts = ["degrees"]
        axis_opts = ["x"]

        for std in std_name_opts:
            for units in units_opts:
                for axis in axis_opts:
                    expect_stdname, expect_units = expected_results(
                        std, units, axis
                    )
                    result = self.run_testcase(
                        standard_name=std,
                        long_name=None,
                        var_name=None,
                        units=units,
                        axis=axis,
                    )
                    (coord,) = result.coords()
                    context_message = (
                        "\nTesting with : "
                        f"stdname={std!r}, longname=None, varname=None, units={units!r}, axis={axis!r}"
                        "\n  expected coord with : "
                        f'stdname={expect_stdname!r}, longname=None, varname="dim", units={expect_units!r}'
                        "\n  got : "
                        f"\nstdname={coord.standard_name!r}, longname={coord.long_name!r}, "
                        f"varname={coord.var_name!r}, units={coord.units!r}"
                    )
                    self.check_result(
                        result,
                        expect_stdname,
                        None,
                        expect_units,
                        context_message=context_message,
                    )


if __name__ == "__main__":
    tests.main()
