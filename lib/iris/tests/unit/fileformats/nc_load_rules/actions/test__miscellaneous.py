# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the engine.activate() call within the
`iris.fileformats.netcdf._load_cube` function.

Tests for rules activation relating to some isolated aspects :
    * UKMO um-specific metadata
    * label coordinates
    * cell measures
    * ancillary variables

"""

import iris.tests as tests  # isort: skip

from iris.coords import AncillaryVariable, AuxCoord, CellMeasure
from iris.fileformats.pp import STASH
from iris.tests.unit.fileformats.nc_load_rules.actions import Mixin__nc_load_actions


class Test__ukmo_attributes(Mixin__nc_load_actions, tests.IrisTest):
    # Tests for handling of the special UM-specific data-var attributes.
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def _make_testcase_cdl(self, **add_attrs):
        phenom_attrs_string = ""
        for key, value in add_attrs.items():
            phenom_attrs_string += f"""
            phenom:{key} = "{value}" ;
"""

        cdl_string = f"""
netcdf test {{
    dimensions:
        xdim = 2 ;
    variables:
        double phenom(xdim) ;
            phenom:standard_name = "air_temperature" ;
            phenom:units = "K" ;
{phenom_attrs_string}
}}
"""
        return cdl_string

    def check_result(self, cube, stashcode=None, processflags=None):
        cube_stashattr = cube.attributes.get("STASH")
        cube_processflags = cube.attributes.get("ukmo__process_flags")

        if stashcode is not None:
            self.assertIsInstance(cube_stashattr, STASH)
            self.assertEqual(str(stashcode), str(cube_stashattr))
        else:
            self.assertIsNone(cube_stashattr)

        if processflags is not None:
            self.assertIsInstance(cube_processflags, tuple)
            self.assertEqual(set(cube_processflags), set(processflags))
        else:
            self.assertIsNone(cube_processflags)

    #
    # Testcase routines
    #
    stashcode = "m01s02i034"  # Just one valid STASH msi string for testing

    def test_stash(self):
        cube = self.run_testcase(um_stash_source=self.stashcode)
        self.check_result(cube, stashcode=self.stashcode)

    def test_stash_altname(self):
        cube = self.run_testcase(ukmo__um_stash_source=self.stashcode)
        self.check_result(cube, stashcode=self.stashcode)

    def test_stash_empty(self):
        value = ""
        cube = self.run_testcase(ukmo__um_stash_source=value)
        self.assertNotIn("STASH", cube.attributes)
        self.assertEqual(cube.attributes["ukmo__um_stash_source"], value)

    def test_stash_invalid(self):
        value = "XXX"
        cube = self.run_testcase(ukmo__um_stash_source="XXX")
        self.assertNotIn("STASH", cube.attributes)
        self.assertEqual(cube.attributes["ukmo__um_stash_source"], value)

    def test_processflags_single(self):
        cube = self.run_testcase(ukmo__process_flags="this")
        self.check_result(cube, processflags=["this"])

    def test_processflags_multi_with_underscores(self):
        flags_testinput = "this that_1 the_other_one x"
        flags_expectresult = ["this", "that 1", "the other one", "x"]
        cube = self.run_testcase(ukmo__process_flags=flags_testinput)
        self.check_result(cube, processflags=flags_expectresult)

    def test_processflags_empty(self):
        cube = self.run_testcase(ukmo__process_flags="")
        expected_result = [""]  # May seem odd, but that's what it does.
        self.check_result(cube, processflags=expected_result)


class Test__labels_cellmeasures_ancils(Mixin__nc_load_actions, tests.IrisTest):
    # Tests for some simple rules that translate facts directly into cube data,
    # with no alternative actions, complications or failure modes to test.
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def _make_testcase_cdl(
        self,
        include_label=False,
        include_cellmeasure=False,
        include_ancil=False,
    ):
        phenom_extra_attrs_string = ""
        extra_vars_string = ""

        if include_label:
            phenom_extra_attrs_string += """
                    phenom:coordinates = "v_label" ;
"""
            extra_vars_string += """
                char v_label(xdim, strdim) ;
                    v_label:long_name = "string data" ;
"""

        if include_cellmeasure:
            # One simple case : a valid link + a variable definition.
            phenom_extra_attrs_string += """
                    phenom:cell_measures = "area: v_cellm" ;
"""
            extra_vars_string += """
                double v_cellm(xdim) ;
                    v_cellm:long_name = "cell areas" ;
"""

        if include_ancil:
            # One simple case : a valid link + a variable definition.
            phenom_extra_attrs_string += """
                    phenom:ancillary_variables = "v_ancil" ;
"""
            extra_vars_string += """
                double v_ancil(xdim) ;
                    v_ancil:long_name = "ancillary values" ;
"""
        cdl_string = f"""
        netcdf test {{
            dimensions:
                xdim = 2 ;
                strdim = 5 ;
            variables:
                double phenom(xdim) ;
                    phenom:standard_name = "air_temperature" ;
                    phenom:units = "K" ;
{phenom_extra_attrs_string}
{extra_vars_string}
        }}
        """
        return cdl_string

    def check_result(
        self,
        cube,
        expect_label=False,
        expect_cellmeasure=False,
        expect_ancil=False,
    ):
        label_coords = cube.coords(var_name="v_label")
        if expect_label:
            self.assertEqual(len(label_coords), 1)
            (coord,) = label_coords
            self.assertIsInstance(coord, AuxCoord)
            self.assertEqual(coord.dtype.kind, "U")
        else:
            self.assertEqual(len(label_coords), 0)

        cell_measures = cube.cell_measures()
        if expect_cellmeasure:
            self.assertEqual(len(cell_measures), 1)
            (cellm,) = cell_measures
            self.assertIsInstance(cellm, CellMeasure)
        else:
            self.assertEqual(len(cell_measures), 0)

        ancils = cube.ancillary_variables()
        if expect_ancil:
            self.assertEqual(len(ancils), 1)
            (ancil,) = ancils
            self.assertIsInstance(ancil, AncillaryVariable)
        else:
            self.assertEqual(len(ancils), 0)

    def test_label(self):
        cube = self.run_testcase(include_label=True)
        self.check_result(cube, expect_label=True)

    def test_ancil(self):
        cube = self.run_testcase(include_ancil=True)
        self.check_result(cube, expect_ancil=True)

    def test_cellmeasure(self):
        cube = self.run_testcase(include_cellmeasure=True)
        self.check_result(cube, expect_cellmeasure=True)


if __name__ == "__main__":
    tests.main()
