# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.fileformats.pp.save_pairs_from_cube` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from iris.fileformats.pp import save_pairs_from_cube
import iris.tests.stock as stock


class TestSaveFields(tests.IrisTest):
    def setUp(self):
        self.cube = stock.realistic_3d()

    def test_cube_only(self):
        slices_and_fields = save_pairs_from_cube(self.cube)
        for aslice, field in slices_and_fields:
            self.assertEqual(aslice.shape, (9, 11))
            self.assertEqual(field.lbcode, 101)

    def test_field_coords(self):
        slices_and_fields = save_pairs_from_cube(
            self.cube, field_coords=["grid_longitude", "grid_latitude"]
        )
        for aslice, field in slices_and_fields:
            self.assertEqual(aslice.shape, (11, 9))
            self.assertEqual(field.lbcode, 101)

    def test_lazy_data(self):
        cube = self.cube.copy()
        # "Rebase" the cube onto a lazy version of its data.
        cube.data = cube.lazy_data()
        # Check that lazy data is preserved in save-pairs generation.
        slices_and_fields = save_pairs_from_cube(cube)
        for aslice, _ in slices_and_fields:
            self.assertTrue(aslice.has_lazy_data())

    def test_default_bmdi(self):
        slices_and_fields = save_pairs_from_cube(self.cube)
        _, field = next(slices_and_fields)
        self.assertEqual(field.bmdi, -1e30)


if __name__ == "__main__":
    tests.main()
