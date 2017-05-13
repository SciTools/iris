# (C) British Crown Copyright 2014, Met Office
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
"""Test function :func:`iris.util.remap_cube_dimensions`."""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import biggus

from iris.util import remap_cube_dimensions
import iris.tests.stock


class Test(tests.IrisTest):

    def setUp(self):
        self.cube = iris.tests.stock.realistic_4d()[:, :, 0:1, :]
        self.orig = self.cube.copy()

    def test_remove_axes(self):
        remap_cube_dimensions(self.cube, axes_to_remove=[2])
        self.assertEqual(self.cube, self.orig[:, :, 0])

    def test_remove_multiple_axes(self):
        cube = self.cube[:, 0:1]
        remap_cube_dimensions(cube, axes_to_remove=[1, 2])
        self.assertEqual(cube, self.orig[:, 0, 0, :])

    def test_remove_and_add_axes(self):
        remap_cube_dimensions(self.cube, axes_to_remove=[2], axes_to_add=[1])
        self.assertCML(self.cube, ('util', 'remap_cube_dim', 'smaller.cml'))

    def test_add_axes(self):
        # Check that adding then subsequently removing an axes results in the
        # same cube.
        remap_cube_dimensions(self.cube, axes_to_add=[2])
        self.assertNotEqual(self.orig, self.cube)
        remap_cube_dimensions(self.cube, axes_to_remove=[2])
        self.assertEqual(self.cube, self.orig)
        self.assertCML(self.cube, ('util', 'remap_cube_dim', 'bigger.cml'))

    def test_add_repeated_axes(self):
        # Check that adding then subsequently removing an axes results in the
        # same cube.
        remap_cube_dimensions(self.cube, axes_to_add=[2, 2, 4])
        remap_cube_dimensions(self.orig, axes_to_add=[2])
        remap_cube_dimensions(self.orig, axes_to_add=[2])
        remap_cube_dimensions(self.orig, axes_to_add=[6])
        self.assertEqual(self.cube.shape, (6, 70, 1, 1, 1, 100, 1))
        self.assertEqual(self.orig, self.cube)

    def test_add_axes_out_of_range(self):
        msg = ('The axis to be added \(5\) is out of range for '
               'a cube of 4 dimensions.')
        with self.assertRaisesRegexp(ValueError, msg):
            remap_cube_dimensions(self.cube, axes_to_add=[2, 5])

        msg = ('The axis to be added \(-1\) is out of range for '
               'a cube of 4 dimensions.')
        with self.assertRaisesRegexp(ValueError, msg):
            remap_cube_dimensions(self.cube, axes_to_add=[2, -1])

    def test_remove_axes_out_of_range(self):
        msg = ('The axis to be removed \(5\) is out of range for a '
               'cube of 4 dimensions.')
        with self.assertRaisesRegexp(ValueError, msg):
            remap_cube_dimensions(self.cube, axes_to_remove=[2, 5])

        msg = ('The axis to be removed \(-1\) is out of range for a '
               'cube of 4 dimensions.')
        with self.assertRaisesRegexp(ValueError, msg):
            remap_cube_dimensions(self.cube, axes_to_remove=[2, -1])

    def test_remove_axes_of_incorrect_length(self):
        with self.assertRaisesRegexp(ValueError,
                                     'Axis 1 does not have a length of 1.'):
            remap_cube_dimensions(self.cube, axes_to_remove=[1])

    def test_add_multiple_axes(self):
        cube2 = self.cube.copy()
        remap_cube_dimensions(cube2, axes_to_add=[0])
        remap_cube_dimensions(cube2, axes_to_add=[2])
        remap_cube_dimensions(cube2, axes_to_add=[6])
        remap_cube_dimensions(self.cube, axes_to_add=[0, 1, 4])
        self.assertEqual(self.cube.shape, (1, 6, 1, 70, 1, 100, 1))
        self.assertEqual(cube2.shape, (1, 6, 1, 70, 1, 100, 1))
        self.assertEqual(self.cube, cube2)

    def test_biggus_array(self):
        # This functionality will fail until None indexing is implemented
        # in biggus.
        self.cube._my_data = biggus.NumpyArrayAdapter(self.cube.data)
        msg = 'None indexing not yet implemented in biggus.'
        with self.assertRaisesRegexp(ValueError, msg):
            remap_cube_dimensions(
                self.cube, axes_to_add=[2], axes_to_remove=[2])


if __name__ == '__main__':
    tests.main()
