# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests  # isort:skip

import os

import iris


@tests.skip_data
class TestCallbacks(tests.IrisTest):
    def test_pp_callback(self):
        def pp_callback(cube, field, filename):
            cube.attributes["filename"] = os.path.basename(filename)
            cube.attributes["lbyr"] = field.lbyr

        fname = tests.get_data_path(("PP", "aPPglob1", "global.pp"))
        cube = iris.load_cube(fname, callback=pp_callback)
        self.assertCML(cube, ["uri_callback", "pp_global.cml"])


if __name__ == "__main__":
    tests.main()
