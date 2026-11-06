# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test the fast loading of structured Fieldsfiles."""

import pytest

from iris.cube import CubeList
from iris.fileformats.um import load_cubes as load
from iris.tests import _shared_utils


@_shared_utils.skip_data
class TestStructuredLoadFF:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.fname = _shared_utils.get_data_path(("FF", "structured", "small"))

    def _merge_cubes(self, cubes):
        # Merge the 2D cubes returned by `iris.fileformats.um.load_cubes`.
        return CubeList(cubes).merge_cube()

    def test_simple(self, request):
        list_of_cubes = list(load(self.fname, None))
        cube = self._merge_cubes(list_of_cubes)
        _shared_utils.assert_CML(request, cube)

    def test_simple_callback(self, request):
        def callback(cube, field, filename):
            cube.attributes["processing"] = "fast-ff"

        list_of_cubes = list(load(self.fname, callback=callback))
        cube = self._merge_cubes(list_of_cubes)
        _shared_utils.assert_CML(request, cube)
