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
import iris.tests as tests
from iris.tests.unit.fileformats.netcdf.load_cube.load_cube__activate.test__grid_mappings import (
    Mixin__grid_mapping,
)


class Test__additional(Mixin__grid_mapping, tests.IrisTest):
    # Run grid-mapping tests with non-Pyke (actions)
    use_pyke = True
    debug = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def test_nondim_lats(self):
        # Check what happens when values don't allow a coord to be dim-coord.
        #
        # Rules Triggered:
        #     001 : fc_default
        #     002 : fc_provides_grid_mapping_latitude_longitude
        #     003 : fc_provides_coordinate_latitude
        #     004 : fc_provides_coordinate_longitude
        #     005 : fc_build_coordinate_latitude
        #     006 : fc_build_coordinate_longitude
        # NOTES:
        #  in terms of rule triggers, this is not distinct from a normal case
        #  - but the latitude is now an aux-coord.
        warning = "must be.* monotonic"
        result = self.run_testcase(warning=warning, yco_values=[0.0, 0.0])
        self.check_result(result, yco_is_aux=True)


if __name__ == "__main__":
    tests.main()
