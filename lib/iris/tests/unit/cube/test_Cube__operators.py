# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.cube.Cube` class operators."""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests  # isort:skip

import operator

import dask.array as da
import numpy as np
import numpy.ma as ma

import iris
from iris._lazy_data import as_lazy_data
from iris.coords import DimCoord


class Test_lazy_maths(tests.IrisTest):
    def build_lazy_cube(self, points, dtype=np.float64, bounds=None, nx=10):
        data = np.arange(len(points) * nx, dtype=dtype) + 1  # Just avoid 0.
        data = data.reshape(len(points), nx)
        data = as_lazy_data(data)
        cube = iris.cube.Cube(data, standard_name="air_temperature", units="K")
        lat = DimCoord(points, "latitude", bounds=bounds)
        lon = DimCoord(np.arange(nx), "longitude")
        cube.add_dim_coord(lat, 0)
        cube.add_dim_coord(lon, 1)
        return cube

    def check_common(self, base_cube, result):
        self.assertTrue(base_cube.has_lazy_data())
        self.assertTrue(result.has_lazy_data())
        self.assertIsInstance(result.lazy_data(), da.core.Array)

    def cube_cube_math_op(self, c1, math_op):
        result = math_op(c1, c1)
        self.check_common(c1, result)
        expected = math_op(c1.data, c1.data)
        self.assertArrayAlmostEqual(result.data, expected)

    def cube_scalar_math_op(self, c1, scalar, math_op, commutative=True):
        result = math_op(c1, scalar)
        if commutative:
            self.assertEqual(math_op(c1, scalar), math_op(scalar, c1))
        self.check_common(c1, result)
        expected = math_op(c1.data, scalar)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_add_cubes__float(self):
        c1 = self.build_lazy_cube([1, 2])
        op = operator.add
        self.cube_cube_math_op(c1, op)

    def test_add_scalar__float(self):
        c1 = self.build_lazy_cube([1, 2])
        scalar = 5
        op = operator.add
        self.cube_scalar_math_op(c1, scalar, op)

    def test_mul_cubes__float(self):
        c1 = self.build_lazy_cube([1, 2])
        op = operator.mul
        self.cube_cube_math_op(c1, op)

    def test_mul_scalar__float(self):
        c1 = self.build_lazy_cube([1, 2])
        scalar = 5
        op = operator.mul
        self.cube_scalar_math_op(c1, scalar, op)

    def test_sub_cubes__float(self):
        c1 = self.build_lazy_cube([1, 2])
        op = operator.sub
        self.cube_cube_math_op(c1, op)

    def test_sub_scalar__float(self):
        c1 = self.build_lazy_cube([1, 2])
        scalar = 5
        op = operator.sub
        self.cube_scalar_math_op(c1, scalar, op, commutative=False)

    def test_div_cubes__float(self):
        c1 = self.build_lazy_cube([1, 2])
        op = operator.truediv
        self.cube_cube_math_op(c1, op)

    def test_div_scalar__float(self):
        c1 = self.build_lazy_cube([1, 2])
        scalar = 5
        op = operator.truediv
        self.cube_scalar_math_op(c1, scalar, op, commutative=False)

    def test_add_cubes__int(self):
        c1 = self.build_lazy_cube([1, 2], dtype=np.int64)
        op = operator.add
        self.cube_cube_math_op(c1, op)

    def test_add_scalar__int(self):
        c1 = self.build_lazy_cube([1, 2], dtype=np.int64)
        scalar = 5
        op = operator.add
        self.cube_scalar_math_op(c1, scalar, op)

    def test_mul_cubes__int(self):
        c1 = self.build_lazy_cube([1, 2], dtype=np.int64)
        op = operator.mul
        self.cube_cube_math_op(c1, op)

    def test_mul_scalar__int(self):
        c1 = self.build_lazy_cube([1, 2], dtype=np.int64)
        scalar = 5
        op = operator.mul
        self.cube_scalar_math_op(c1, scalar, op)

    def test_sub_cubes__int(self):
        c1 = self.build_lazy_cube([1, 2], dtype=np.int64)
        op = operator.sub
        self.cube_cube_math_op(c1, op)

    def test_sub_scalar__int(self):
        c1 = self.build_lazy_cube([1, 2], dtype=np.int64)
        scalar = 5
        op = operator.sub
        self.cube_scalar_math_op(c1, scalar, op, commutative=False)

    def test_div_cubes__int(self):
        c1 = self.build_lazy_cube([1, 2], dtype=np.int64)
        op = operator.truediv
        self.cube_cube_math_op(c1, op)

    def test_div_scalar__int(self):
        c1 = self.build_lazy_cube([1, 2], dtype=np.int64)
        scalar = 5
        op = operator.truediv
        self.cube_scalar_math_op(c1, scalar, op, commutative=False)


class Test_lazy_maths__scalar_cube(tests.IrisTest):
    def build_lazy_cube(self, value, dtype=np.float64):
        data = as_lazy_data(np.array(value, dtype=dtype))
        return iris.cube.Cube(data, standard_name="air_temperature", units="K")

    def setUp(self):
        self.c1 = self.build_lazy_cube(3)
        self.c2 = self.build_lazy_cube(4)
        self.c3 = self.build_lazy_cube(3, dtype=np.int64)
        self.c4 = self.build_lazy_cube(4, dtype=np.int64)

    def check_common(self, c1, c2, math_op):
        cube = math_op(c1, c2)
        data = cube.data
        self.assertTrue(isinstance(data, np.ndarray))
        self.assertEqual(data.shape, ())

    def test_add_scalar__int(self):
        c3, c4, op = self.c3, 5, operator.add
        self.check_common(c3, c4, op)

    def test_add_cubes__int(self):
        c3, c4, op = self.c3, self.c4, operator.add
        self.check_common(c3, c4, op)

    def test_mul_scalar__int(self):
        c3, c4, op = self.c3, 5, operator.mul
        self.check_common(c3, c4, op)

    def test_mul_cubes__int(self):
        c3, c4, op = self.c3, self.c4, operator.mul
        self.check_common(c3, c4, op)

    def test_sub_scalar__int(self):
        c3, c4, op = self.c3, 5, operator.sub
        self.check_common(c3, c4, op)

    def test_sub_cubes__int(self):
        c3, c4, op = self.c3, self.c4, operator.sub
        self.check_common(c3, c4, op)

    def test_div_scalar__int(self):
        c3, c4, op = self.c3, 5, operator.truediv
        self.check_common(c3, c4, op)

    def test_div_cubes__int(self):
        c3, c4, op = self.c3, self.c4, operator.truediv
        self.check_common(c3, c4, op)

    def test_add_scalar__float(self):
        c1, c2, op = self.c1, 5, operator.add
        self.check_common(c1, c2, op)

    def test_add_cubes__float(self):
        c1, c2, op = self.c1, self.c2, operator.add
        self.check_common(c1, c2, op)

    def test_mul_scalar__float(self):
        c1, c2, op = self.c1, 5, operator.mul
        self.check_common(c1, c2, op)

    def test_mul_cubes__float(self):
        c1, c2, op = self.c1, self.c2, operator.mul
        self.check_common(c1, c2, op)

    def test_sub_scalar__float(self):
        c1, c2, op = self.c1, 5, operator.sub
        self.check_common(c1, c2, op)

    def test_sub_cubes__float(self):
        c1, c2, op = self.c1, self.c2, operator.sub
        self.check_common(c1, c2, op)

    def test_div_scalar__float(self):
        c1, c2, op = self.c1, 5, operator.truediv
        self.check_common(c1, c2, op)

    def test_div_cubes__float(self):
        c1, c2, op = self.c1, self.c2, operator.truediv
        self.check_common(c1, c2, op)


class Test_lazy_maths__masked_data(tests.IrisTest):
    def build_lazy_cube(self, dtype=np.float64):
        data = ma.array(
            [[1.0, 1.0], [1.0, 100000.0]], mask=[[0, 0], [0, 1]], dtype=dtype
        )
        data = as_lazy_data(data)
        cube = iris.cube.Cube(data, standard_name="air_temperature", units="K")
        lat = DimCoord([-10, 10], "latitude")
        lon = DimCoord([10, 20], "longitude")
        cube.add_dim_coord(lat, 0)
        cube.add_dim_coord(lon, 1)
        return cube

    def test_subtract__float(self):
        cube_a = self.build_lazy_cube()
        cube_b = self.build_lazy_cube()
        cube_c = cube_a - cube_b
        self.assertTrue(ma.isMaskedArray(cube_c.data))

    def test_subtract__int(self):
        cube_a = self.build_lazy_cube(dtype=np.int64)
        cube_b = self.build_lazy_cube(dtype=np.int64)
        cube_c = cube_a - cube_b
        self.assertTrue(ma.isMaskedArray(cube_c.data))


if __name__ == "__main__":
    tests.main()
