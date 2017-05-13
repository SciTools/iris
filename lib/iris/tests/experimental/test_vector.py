# (C) British Crown Copyright 2013, Met Office
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

import numpy as np

import iris
import iris.experimental.vector as ivector
import iris.tests as tests


@iris.tests.skip_data
class TestVector(tests.IrisTest):
    def setUp(self):
        self.u = iris.\
            load_cube(tests.get_data_path(('NetCDF',
                                           'global',
                                           'xyt',
                                           'SMALL_hires_wind_u_for_ipcc4.nc')))
        self.v = self.u.copy()
        self.v.rename("northward_wind")
        self.w = self.u.copy()
        self.w.rename("vertical_wind")

        self.three_d_vec = ivector.Vector(self.u, self.v, self.w)
        self.two_d_vec = ivector.Vector(self.u, self.v)

    def test_constructors_equal(self):
        self.assertArrayEqual(self.two_d_vec.u.data, self.three_d_vec.u.data)
        self.assertArrayEqual(self.two_d_vec.v.data, self.three_d_vec.v.data)

    def test_add(self):
        sum_vec = self.three_d_vec + self.three_d_vec
        self.assertArrayEqual(sum_vec.u.data, self.three_d_vec.u.data * 2)
        self.assertArrayEqual(sum_vec.v.data, self.three_d_vec.v.data * 2)
        self.assertArrayEqual(sum_vec.w.data, self.three_d_vec.w.data * 2)

    def test_add_2d_to_3d(self):
        with self.assertRaises(AttributeError):
            self.three_d_vec + self.two_d_vec

    def test_subtract_2d_from_3d(self):
        with self.assertRaises(AttributeError):
            self.three_d_vec - self.two_d_vec

    def test_subtract(self):
        sum_vec = self.three_d_vec - self.three_d_vec
        self.assertTrue(np.all(sum_vec.u.data == 0.0))
        self.assertTrue(np.all(sum_vec.v.data == 0.0))
        self.assertTrue(np.all(sum_vec.w.data == 0.0))

    def test_vector_indexing(self):
        vec_slice = self.three_d_vec[2]
        self.assertEqual(vec_slice.u, self.three_d_vec.u[2])
        self.assertEqual(vec_slice.v, self.three_d_vec.v[2])
        self.assertEqual(vec_slice.w, self.three_d_vec.w[2])

    def test_magnitude(self):
        mag = self.three_d_vec.magnitude()
        answer = np.sqrt(self.three_d_vec.u.data**2 +
                         self.three_d_vec.v.data**2 +
                         self.three_d_vec.w.data**2)
        self.assertArrayEqual(mag.data, answer)


if __name__ == "__main__":
    tests.main()
