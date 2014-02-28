# (C) British Crown Copyright 2010 - 2014, Met Office
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

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy as np

import iris.cube


@iris.tests.skip_data
class Test_CubeList_extract_overlapping(tests.IrisTest):
    def setUp(self):
        self.cube = iris.load_cube(iris.sample_data_path('GloSea4',
                                                         'ensemble_001.pp'))

    def test_extract_one_str_dim(self):
        a, b = iris.cube.CubeList([self.cube[2:],
                                   self.cube[:4]]).extract_overlapping('time')

        self.assertEquals(a.coord('time'), self.cube.coord('time')[2:4])
        self.assertEquals(b.coord('time'), self.cube.coord('time')[2:4])

    def test_extract_one_list_dim(self):
        a, b = iris.cube.CubeList([self.cube[2:],
                                   self.cube[:4]])\
            .extract_overlapping(['time'])

        self.assertEquals(a.coord('time'), self.cube.coord('time')[2:4])
        self.assertEquals(b.coord('time'), self.cube.coord('time')[2:4])

    def test_extract_two_dims(self):
        a, b = iris.cube.CubeList([self.cube[2:, 5:],
                                   self.cube[:4, :10]])\
            .extract_overlapping(['time', 'latitude'])

        self.assertEquals(a.coord('time'),
                          self.cube.coord('time')[2:4])
        self.assertEquals(a.coord('latitude'),
                          self.cube.coord('latitude')[5:10])
        self.assertEquals(b.coord('time'),
                          self.cube.coord('time')[2:4])
        self.assertEquals(b.coord('latitude'),
                          self.cube.coord('latitude')[5:10])

    def test_different_orders(self):
        reverse_order = np.arange(self.cube.shape[0], 0, -1) - 1
        a, b = iris.cube.CubeList([self.cube[reverse_order, ...][:4],
                                   self.cube[:4]]).extract_overlapping('time')
        self.assertEquals(a.coord('time'),
                          self.cube[reverse_order, ...].coord('time')[2:4])
        self.assertEquals(b.coord('time'),
                          self.cube.coord('time')[2:4])


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
        self.assertRaises(TypeError,
                          self.cube.add_dim_coord,
                          iris.coords.DimCoord(np.arange(2), "latitude"))

    def test_adding_aux_coord(self):
        coord = iris.coords.AuxCoord(np.arange(2), "latitude")
        with self.assertRaises(ValueError):
            self.cube.add_dim_coord(coord, 0)


class TestEquality(tests.IrisTest):
    def test_not_implmemented(self):
        class Terry(object):
            pass
        cube = iris.cube.Cube(0)
        self.assertIs(cube.__eq__(Terry()), NotImplemented)
        self.assertIs(cube.__ne__(Terry()), NotImplemented)


if __name__ == "__main__":
    tests.main()
