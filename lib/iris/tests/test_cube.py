# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import numpy as np

import iris.cube


class Test_CubeList_getitem(tests.IrisTest):
    def setUp(self):
        self.cube0 = iris.cube.Cube(0)
        self.cube1 = iris.cube.Cube(1)
        self.src_list = [self.cube0, self.cube1]
        self.cube_list = iris.cube.CubeList(self.src_list)

    def test_single(self):
        # Check that simple indexing returns the relevant member Cube.
        for i, cube in enumerate(self.src_list):
            self.assertIs(self.cube_list[i], cube)

    def _test_slice(self, keys):
        subset = self.cube_list[keys]
        self.assertIsInstance(subset, iris.cube.CubeList)
        self.assertEqual(subset, self.src_list[keys])

    def test_slice(self):
        # Check that slicing returns a CubeList containing the relevant
        # members.
        self._test_slice(slice(None))
        self._test_slice(slice(1))
        self._test_slice(slice(1, None))
        self._test_slice(slice(0, 1))
        self._test_slice(slice(None, None, -1))


class Test_CubeList_getslice(tests.IrisTest):
    def setUp(self):
        self.cube0 = iris.cube.Cube(0)
        self.cube1 = iris.cube.Cube(1)
        self.src_list = [self.cube0, self.cube1]
        self.cube_list = iris.cube.CubeList(self.src_list)

    def _test_slice(self, cube_list, equivalent):
        self.assertIsInstance(cube_list, iris.cube.CubeList)
        self.assertEqual(cube_list, equivalent)

    def test_slice(self):
        # Check that slicing returns a CubeList containing the relevant
        # members.
        # NB. We have to use explicit [:1] syntax to trigger the call
        # to __getslice__. Using [slice(1)] still calls __getitem__!
        self._test_slice(self.cube_list[:1], self.src_list[:1])
        self._test_slice(self.cube_list[1:], self.src_list[1:])
        self._test_slice(self.cube_list[0:1], self.src_list[0:1:])


class Test_Cube_add_dim_coord(tests.IrisTest):
    def setUp(self):
        self.cube = iris.cube.Cube(np.arange(4).reshape(2, 2))

    def test_no_dim(self):
        self.assertRaises(
            TypeError,
            self.cube.add_dim_coord,
            iris.coords.DimCoord(np.arange(2), "latitude"),
        )

    def test_adding_aux_coord(self):
        coord = iris.coords.AuxCoord(np.arange(2), "latitude")
        with self.assertRaises(ValueError):
            self.cube.add_dim_coord(coord, 0)


class TestEquality(tests.IrisTest):
    def test_not_implmemented(self):
        class Terry:
            pass

        cube = iris.cube.Cube(0)
        self.assertIs(cube.__eq__(Terry()), NotImplemented)
        self.assertIs(cube.__ne__(Terry()), NotImplemented)


if __name__ == "__main__":
    tests.main()
