# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test the fast loading of structured Fieldsfiles.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip
from iris.cube import CubeList
from iris.fileformats.um import load_cubes as load


@tests.skip_data
class TestStructuredLoadFF(tests.IrisTest):
    def setUp(self):
        self.fname = tests.get_data_path(("FF", "structured", "small"))

    def _merge_cubes(self, cubes):
        # Merge the 2D cubes returned by `iris.fileformats.um.load_cubes`.
        return CubeList(cubes).merge_cube()

    def test_simple(self):
        list_of_cubes = list(load(self.fname, None))
        cube = self._merge_cubes(list_of_cubes)
        self.assertCML(cube)

    def test_simple_callback(self):
        def callback(cube, field, filename):
            cube.attributes["processing"] = "fast-ff"

        list_of_cubes = list(load(self.fname, callback=callback))
        cube = self._merge_cubes(list_of_cubes)
        self.assertCML(cube)


if __name__ == "__main__":
    tests.main()
