# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the function :func:`iris.analysis.maths._get_dtype`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np
from numpy import ma

from iris.analysis.maths import _get_dtype
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube


class Test(tests.IrisTest):
    def _check_call(self, obj, expected_dtype):
        result = _get_dtype(obj)
        self.assertEqual(expected_dtype, result)

    def test_int8(self):
        n = -128
        self._check_call(n, np.int8)

    def test_int16(self):
        n = -129
        self._check_call(n, np.int16)

    def test_uint8(self):
        n = 255
        self._check_call(n, np.uint8)

    def test_uint16(self):
        n = 256
        self._check_call(n, np.uint16)

    def test_float16(self):
        n = 60000.0
        self._check_call(n, np.float16)

    def test_float32(self):
        n = 65000.0
        self._check_call(n, np.float32)

    def test_float64(self):
        n = 1e40
        self._check_call(n, np.float64)

    def test_scalar_demote(self):
        n = np.int64(10)
        self._check_call(n, np.uint8)

    def test_array(self):
        a = np.array([1, 2, 3], dtype=np.int16)
        self._check_call(a, np.int16)

    def test_scalar_array(self):
        dtype = np.int32
        a = np.array(1, dtype=dtype)
        self._check_call(a, dtype)

    def test_masked_array(self):
        dtype = np.float16
        m = ma.masked_array([1, 2, 3], [1, 0, 1], dtype=dtype)
        self._check_call(m, dtype)

    def test_masked_constant(self):
        m = ma.masked
        self._check_call(m, m.dtype)

    def test_cube(self):
        dtype = np.float32
        data = np.array([1, 2, 3], dtype=dtype)
        cube = Cube(data)
        self._check_call(cube, dtype)

    def test_aux_coord(self):
        dtype = np.int64
        points = np.array([1, 2, 3], dtype=dtype)
        aux_coord = AuxCoord(points)
        self._check_call(aux_coord, dtype)

    def test_dim_coord(self):
        dtype = np.float16
        points = np.array([1, 2, 3], dtype=dtype)
        dim_coord = DimCoord(points)
        self._check_call(dim_coord, dtype)


if __name__ == "__main__":
    tests.main()
