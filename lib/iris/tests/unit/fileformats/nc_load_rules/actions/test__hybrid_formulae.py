# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the engine.activate() call within the
`iris.fileformats.netcdf._load_cube` function.

Test rules activation relating to hybrid vertical coordinates.

"""
import iris.tests as tests  # isort: skip

import iris.fileformats._nc_load_rules.helpers as hh
from iris.tests.unit.fileformats.nc_load_rules.actions import (
    Mixin__nc_load_actions,
)


class Test__formulae_tests(Mixin__nc_load_actions, tests.IrisTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def _make_testcase_cdl(
        self, formula_root_name=None, term_names=None, extra_formula_type=None
    ):
        """Construct a testcase CDL for data with hybrid vertical coords."""
        if formula_root_name is None:
            formula_root_name = "atmosphere_hybrid_height_coordinate"
        if term_names is None:
            term_names = hh.CF_COORD_VERTICAL.get(formula_root_name)
            if term_names is None:
                # unsupported type : just make something up
                term_names = ["term1"]

        # Arrange to create additional term variables for an 'extra' hybrid
        # formula, if requested.
        if extra_formula_type is None:
            term_names_extra = []
            phenom_coord_names = ["vert"]  # always include the root variable
        else:
            phenom_coord_names = ["vert", "vert_2"]  # two formula coords
            term_names_extra = hh.CF_COORD_VERTICAL.get(extra_formula_type)

        # Build strings to define term variables.
        formula_term_strings = []
        extra_formula_term_strings = []
        terms_string = ""
        for term_name in term_names + term_names_extra:
            term_varname = "v_" + term_name
            # Include in the phenom coordinates list.
            phenom_coord_names.append(term_varname)
            term_string = f"{term_name}: {term_varname}"
            if term_name in term_names:
                # Include in the 'main' terms list.
                formula_term_strings.append(term_string)
            else:
                # Include in the 'extra' terms list.
                extra_formula_term_strings.append(term_string)
            terms_string += f"""
    double {term_varname}(h) ;
        {term_varname}:long_name = "{term_name}_long_name" ;
        {term_varname}:units = "m" ;
"""

        # Construct the reference strings.
        phenom_coords_string = " ".join(phenom_coord_names)
        formula_terms_string = " ".join(formula_term_strings)
        extra_formula_terms_string = " ".join(extra_formula_term_strings)

        # Construct the 'extra' hybrid coord if requested.
        if extra_formula_type is None:
            extra_formula_string = ""
        else:
            # Create the lines to add an 'extra' formula.
            # For now, put this on the same dim : makes no difference.
            extra_formula_string = f"""
    double vert_2(h) ;
        vert_2:standard_name = "{extra_formula_type}" ;
        vert_2:units = "m" ;
        vert_2:formula_terms = "{extra_formula_terms_string}" ;
"""

        # Create the main result string.
        cdl_str = f"""
netcdf test {{
dimensions:
    h = 2 ;
variables:
    double phenom(h) ;
        phenom:standard_name = "air_temperature" ;
        phenom:units = "K" ;
        phenom:coordinates = "{phenom_coords_string}" ;
    double vert(h) ;
        vert:standard_name = "{formula_root_name}" ;
        vert:long_name = "hybrid_vertical" ;
        vert:units = "m" ;
        vert:formula_terms = "{formula_terms_string}" ;
{terms_string}
{extra_formula_string}
}}
"""
        return cdl_str

    def check_result(self, cube, factory_type="_auto", formula_terms="_auto"):
        """Check the result of a cube load with a hybrid vertical coord."""
        if factory_type == "_auto":
            # replace with our 'default', which is hybrid-height.
            # N.B. 'None' is different: it means expect *no* factory.
            factory_type = "atmosphere_hybrid_height_coordinate"
        self.assertEqual(cube._formula_type_name, factory_type)

        if formula_terms == "_auto":
            # Set default terms-expected, according to the expected factory
            # type.
            if factory_type is None:
                # If no factory, expect no identified terms.
                formula_terms = []
            else:
                # Expect the correct ones defined for the factory type.
                formula_terms = hh.CF_COORD_VERTICAL[factory_type]

        # Compare the formula_terms list with the 'expected' ones.
        # N.B. first make the 'expected' list lower case, as the lists in
        # hh.CF_COORD_VERTICAL include uppercase, but rules outputs don't.
        formula_terms = [term.lower() for term in formula_terms]

        # N.B. the terms dictionary can be missing, if there were none
        actual_terms = cube._formula_terms_byname or {}
        self.assertEqual(sorted(formula_terms), sorted(actual_terms.keys()))

        # Check that there is an aux-coord of the expected name for each term
        for var_name in actual_terms.values():
            coords = cube.coords(var_name=var_name, dim_coords=False)
            self.assertEqual(len(coords), 1)

    #
    # Actual testcase routines
    #

    def test_basic_hybridheight(self):
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_build_auxiliary_coordinate
        #     003 : fc_build_auxiliary_coordinate
        #     004 : fc_build_auxiliary_coordinate
        #     005 : fc_build_auxiliary_coordinate
        #     008 : fc_formula_type_atmosphere_hybrid_height_coordinate
        #     009 : fc_formula_term(a)
        #     010 : fc_formula_term(b)
        #     011 : fc_formula_term(orog)
        result = self.run_testcase()
        self.check_result(result)

    def test_missing_term(self):
        # Check behaviour when a term is missing.
        # For the test, omit "orography", which is common in practice.
        #
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_build_auxiliary_coordinate
        #     003 : fc_build_auxiliary_coordinate
        #     004 : fc_build_auxiliary_coordinate
        #     007 : fc_formula_type_atmosphere_hybrid_height_coordinate
        #     008 : fc_formula_term(a)
        #     009 : fc_formula_term(b)
        result = self.run_testcase(
            term_names=["a", "b"]  # missing the 'orog' term
        )
        self.check_result(result, formula_terms=["a", "b"])

    def test_no_terms(self):
        # Check behaviour when *all* terms are missing.
        # N.B. for any _actual_ type, this is probably invalid and would fail?
        #
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_build_auxiliary_coordinate
        result = self.run_testcase(
            formula_root_name="atmosphere_hybrid_height_coordinate",
            term_names=[],
        )
        # This does *not* trigger
        # 'fc_formula_type_atmosphere_hybrid_height_coordinate'
        # This is because, within the 'assert_case_specific_facts' routine,
        # formula_roots are only recognised by scanning the identified
        # formula_terms.
        self.check_result(result, factory_type=None)

    def test_unrecognised_verticaltype(self):
        # Set the root variable name to something NOT a recognised hybrid type.
        #
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_build_auxiliary_coordinate
        #     003 : fc_build_auxiliary_coordinate
        #     004 : fc_build_auxiliary_coordinate
        #     007 : fc_formula_type(FAILED - unrecognised formula type = 'unknown')
        #     008 : fc_formula_term(a)
        #     009 : fc_formula_term(b)
        result = self.run_testcase(
            formula_root_name="unknown",
            term_names=["a", "b"],
            warning="Ignored formula of unrecognised type: 'unknown'.",
        )
        # Check that it picks up the terms, but *not* the factory root coord,
        # which is simply discarded.
        self.check_result(result, factory_type=None, formula_terms=["a", "b"])

    def test_two_formulae(self):
        # Construct an example with TWO hybrid coords.
        # This is not errored, but we don't correctly support it.
        #
        # NOTE: the original Pyke implementation does not detect this problem
        # By design, the new mechanism does + will raise a warning.
        warning = (
            "Omitting factories for some hybrid coordinates.*"
            "multiple hybrid coordinates.* not supported"
        )

        extra_type = "ocean_sigma_coordinate"
        result = self.run_testcase(
            extra_formula_type=extra_type, warning=warning
        )
        # NOTE: FOR NOW, check expected behaviour : only one factory will be
        # built, but there are coordinates (terms) for both types.
        # TODO: this is a bug and needs fixing : translation should handle
        # multiple hybrid coordinates in a sensible way.
        self.check_result(
            result,
            factory_type=extra_type,
            formula_terms=["a", "b", "depth", "eta", "orog", "sigma"],
        )


# Add in tests methods to exercise each (supported) vertical coordinate type
# individually.
# NOTE: hh.CF_COORD_VERTICAL lists all the valid types, but we don't yet
# support all of them.
_SUPPORTED_FORMULA_TYPES = (
    # NOTE: omit "atmosphere_hybrid_height_coordinate" : our basic testcase
    "atmosphere_sigma_coordinate",
    "atmosphere_hybrid_sigma_pressure_coordinate",
    "ocean_sigma_z_coordinate",
    "ocean_sigma_coordinate",
    "ocean_s_coordinate",
    "ocean_s_coordinate_g1",
    "ocean_s_coordinate_g2",
)
for hybrid_type in _SUPPORTED_FORMULA_TYPES:

    def construct_inner_func(hybrid_type):
        term_names = hh.CF_COORD_VERTICAL[hybrid_type]

        def inner(self):
            result = self.run_testcase(
                formula_root_name=hybrid_type, term_names=term_names
            )
            self.check_result(
                result, factory_type=hybrid_type, formula_terms=term_names
            )

        return inner

    # Note: use an intermediate function to generate each test method, simply to
    # generate a new local variable for 'hybrid_type' on each iteration.
    # Otherwise all the test methods will refer to the *same* 'hybrid_type'
    # variable, i.e. the loop variable, which does not work !
    method_name = f"test_{hybrid_type}_coord"
    setattr(
        Test__formulae_tests, method_name, construct_inner_func(hybrid_type)
    )


if __name__ == "__main__":
    tests.main()
