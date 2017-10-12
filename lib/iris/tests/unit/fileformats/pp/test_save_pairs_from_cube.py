# (C) British Crown Copyright 2015 - 2017, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""Unit tests for the `iris.fileformats.pp.save_pairs_from_cube` function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import iris.tests.stock as stock

from iris.fileformats.pp import save_pairs_from_cube


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
            self.cube,
            field_coords=['grid_longitude',
                          'grid_latitude'])
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
