# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

import os

import iris
from iris.tests import _shared_utils


@_shared_utils.skip_data
class TestCallbacks:
    def test_pp_callback(self, request):
        def pp_callback(cube, field, filename):
            cube.attributes["filename"] = os.path.basename(filename)
            cube.attributes["lbyr"] = field.lbyr

        fname = _shared_utils.get_data_path(("PP", "aPPglob1", "global.pp"))
        cube = iris.load_cube(fname, callback=pp_callback)
        _shared_utils.assert_CML(request, cube, ["uri_callback", "pp_global.cml"])
