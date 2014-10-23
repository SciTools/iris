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
"""
Test function
:func:`iris.experimental.regrid._regrid_bilinear_array`.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy as np

import iris
from iris.coords import DimCoord
from iris.experimental.regrid import _regrid_bilinear_array


class Test(tests.IrisTest):
    def setUp(self):
        self.x = iris.coords.DimCoord(np.linspace(-2, 57, 60))
        self.y = iris.coords.DimCoord(np.linspace(0, 49, 50))
        self.xs, self.ys = np.meshgrid(self.x.points, self.y.points)

        transformation = lambda x, y: x + y ** 2
        # Construct a function which adds dimensions to the 2D data array
        # so that we can test higher dimensional functionality.
        dim_extender = lambda arr: (arr[np.newaxis, ..., np.newaxis] * [1, 2])

        self.data = dim_extender(transformation(self.xs, self.ys))

        target_x = np.linspace(-3, 60, 4)
        target_y = np.linspace(0.5, 51, 3)
        self.target_x, self.target_y = np.meshgrid(target_x, target_y)

        #: Expected values, which not quite the analytical value, but
        #: representative of the bilinear interpolation scheme.
        self.expected = np.array([[[[np.nan, np.nan],
                                    [18.5, 37.],
                                    [39.5, 79.],
                                    [np.nan, np.nan]],
                                   [[np.nan, np.nan],
                                    [681.25, 1362.5],
                                    [702.25, 1404.5],
                                    [np.nan, np.nan]],
                                   [[np.nan, np.nan],
                                    [np.nan, np.nan],
                                    [np.nan, np.nan],
                                    [np.nan, np.nan]]]])

        self.x_dim = 2
        self.y_dim = 1

    def assert_values(self, values):
        # values is a list of [x, y, [val1, val2]]
        xs, ys, expecteds = zip(*values)
        expecteds = np.array(expecteds)[None, None, ...]
        result = _regrid_bilinear_array(self.data, self.x_dim, self.y_dim,
                                        self.x, self.y,
                                        np.array([xs]), np.array([ys]))
        self.assertArrayAllClose(result, expecteds, rtol=1e-04)

        # Check that transposing the input data results in the same values
        ndim = self.data.ndim
        result2 = _regrid_bilinear_array(self.data.T, ndim - self.x_dim - 1,
                                         ndim - self.y_dim - 1,
                                         self.x, self.y,
                                         np.array([xs]), np.array([ys]))
        self.assertArrayEqual(result.T, result2)

    def test_single_values(self):
        # Check that the values are sensible e.g. (3 + 4**2 == 19)
        self.assert_values([[3, 4, [19, 38]],
                            [-2, 0, [-2, -4]],
                            [-2.01, 0, [np.nan, np.nan]],
                            [2, -0.01, [np.nan, np.nan]],
                            [57, 0, [57, 114]],
                            [57.01, 0, [np.nan, np.nan]],
                            [57, 49, [2458, 4916]],
                            [57, 49.01, [np.nan, np.nan]]])

    def test_simple_result(self):
        result = _regrid_bilinear_array(self.data, self.x_dim, self.y_dim,
                                        self.x, self.y,
                                        self.target_x, self.target_y)
        self.assertArrayEqual(result, self.expected)

    def test_simple_masked(self):
        data = np.ma.MaskedArray(self.data, mask=True)
        data.mask[:, 1:30, 1:30] = False
        result = _regrid_bilinear_array(data, self.x_dim, self.y_dim,
                                        self.x, self.y,
                                        self.target_x, self.target_y)
        expected_mask = np.array([[[[True, True], [True, True],
                                    [True, True], [True, True]],
                                   [[True, True], [False, False],
                                    [True, True], [True, True]],
                                   [[True, True], [True, True],
                                    [True, True], [True, True]]]], dtype=bool)
        expected = np.ma.MaskedArray(self.expected,
                                     mask=expected_mask)
        self.assertMaskedArrayEqual(result, expected)

    def test_simple_masked_no_mask(self):
        data = np.ma.MaskedArray(self.data, mask=False)
        result = _regrid_bilinear_array(data, self.x_dim, self.y_dim,
                                        self.x, self.y,
                                        self.target_x, self.target_y)
        self.assertIsInstance(result, np.ma.MaskedArray)

    def test_result_transpose_shape(self):
        ndim = self.data.ndim
        result = _regrid_bilinear_array(self.data.T, ndim - self.x_dim - 1,
                                        ndim - self.y_dim - 1, self.x, self.y,
                                        self.target_x, self.target_y)
        self.assertArrayEqual(result, self.expected.T)

    def test_reverse_x_coord(self):
        index = [slice(None)] * self.data.ndim
        index[self.x_dim] = slice(None, None, -1)
        result = _regrid_bilinear_array(self.data[index], self.x_dim,
                                        self.y_dim, self.x[::-1], self.y,
                                        self.target_x, self.target_y)
        self.assertArrayEqual(result, self.expected)

    def test_circular_x_coord(self):
        # Check that interpolation of a circular src coordinate doesn't result
        # in an out of bounds value.
        self.x.circular = True
        self.x.units = 'degree'
        result = _regrid_bilinear_array(self.data, self.x_dim, self.y_dim,
                                        self.x, self.y, np.array([[58]]),
                                        np.array([[0]]))
        self.assertArrayAlmostEqual(result,
                                    np.array([56.80398671, 113.60797342],
                                             ndmin=self.data.ndim))


# Check what happens to NaN values, extrapolated values, and
# masked values.
class TestModes(tests.IrisTest):
    values = [[np.nan, np.nan, 2, 3, np.nan],
              [np.nan, np.nan, 6, 7, np.nan],
              [8, 9, 10, 11, np.nan]]

    linear_values = [[np.nan, np.nan, 2, 3, 4],
                     [np.nan, np.nan, 6, 7, 8],
                     [8, 9, 10, 11, 12]]

    def _regrid(self, data, extrapolation_mode=None):
        x = np.arange(4)
        y = np.arange(3)
        x_coord = DimCoord(x)
        y_coord = DimCoord(y)
        x_dim, y_dim = 1, 0
        grid_x, grid_y = np.meshgrid(np.arange(5), y)
        kwargs = {}
        if extrapolation_mode is not None:
            kwargs['extrapolation_mode'] = extrapolation_mode
        result = _regrid_bilinear_array(data, x_dim, y_dim, x_coord, y_coord,
                                        grid_x, grid_y, **kwargs)
        return result

    def test_default_ndarray(self):
        # NaN           -> NaN
        # Extrapolated  -> NaN
        data = np.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        result = self._regrid(data)
        self.assertNotIsInstance(result, np.ma.MaskedArray)
        self.assertArrayEqual(result, self.values)

    def test_default_maskedarray(self):
        # NaN           -> NaN
        # Extrapolated  -> Masked
        # Masked        -> Masked
        data = np.ma.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        data[2, 3] = np.ma.masked
        result = self._regrid(data)
        self.assertIsInstance(result, np.ma.MaskedArray)
        mask = [[0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 1]]
        expected = np.ma.MaskedArray(self.values, mask)
        self.assertMaskedArrayEqual(result, expected)

    def test_default_maskedarray_none_masked(self):
        # NaN           -> NaN
        # Extrapolated  -> Masked
        # Masked        -> N/A
        data = np.ma.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        result = self._regrid(data)
        self.assertIsInstance(result, np.ma.MaskedArray)
        mask = [[0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1]]
        expected = np.ma.MaskedArray(self.values, mask)
        self.assertMaskedArrayEqual(result, expected)

    def test_default_maskedarray_none_masked_expanded(self):
        # NaN           -> NaN
        # Extrapolated  -> Masked
        # Masked        -> N/A
        data = np.ma.arange(12, dtype=np.float).reshape(3, 4)
        # Make sure the mask has been expanded
        data.mask = False
        data[0, 0] = np.nan
        result = self._regrid(data)
        self.assertIsInstance(result, np.ma.MaskedArray)
        mask = [[0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1]]
        expected = np.ma.MaskedArray(self.values, mask)
        self.assertMaskedArrayEqual(result, expected)

    def test_linear_ndarray(self):
        # NaN           -> NaN
        # Extrapolated  -> linear
        data = np.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        result = self._regrid(data, 'extrapolate')
        self.assertNotIsInstance(result, np.ma.MaskedArray)
        self.assertArrayEqual(result, self.linear_values)

    def test_linear_maskedarray(self):
        # NaN           -> NaN
        # Extrapolated  -> linear
        # Masked        -> Masked
        data = np.ma.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        data[2, 3] = np.ma.masked
        result = self._regrid(data, 'extrapolate')
        self.assertIsInstance(result, np.ma.MaskedArray)
        mask = [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1]]
        expected = np.ma.MaskedArray(self.linear_values, mask)
        self.assertMaskedArrayEqual(result, expected)

    def test_nan_ndarray(self):
        # NaN           -> NaN
        # Extrapolated  -> NaN
        data = np.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        result = self._regrid(data, 'nan')
        self.assertNotIsInstance(result, np.ma.MaskedArray)
        self.assertArrayEqual(result, self.values)

    def test_nan_maskedarray(self):
        # NaN           -> NaN
        # Extrapolated  -> NaN
        # Masked        -> Masked
        data = np.ma.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        data[2, 3] = np.ma.masked
        result = self._regrid(data, 'nan')
        self.assertIsInstance(result, np.ma.MaskedArray)
        mask = [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0]]
        expected = np.ma.MaskedArray(self.values, mask)
        self.assertMaskedArrayEqual(result, expected)

    def test_error_ndarray(self):
        # Values irrelevant - the function raises an error.
        data = np.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        with self.assertRaisesRegexp(ValueError, 'out of bounds'):
            self._regrid(data, 'error')

    def test_error_maskedarray(self):
        # Values irrelevant - the function raises an error.
        data = np.ma.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        data[2, 3] = np.ma.masked
        with self.assertRaisesRegexp(ValueError, 'out of bounds'):
            self._regrid(data, 'error')

    def test_mask_ndarray(self):
        # NaN           -> NaN
        # Extrapolated  -> Masked (this is different from all the other
        #                          modes)
        data = np.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        result = self._regrid(data, 'mask')
        self.assertIsInstance(result, np.ma.MaskedArray)
        mask = [[0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1]]
        expected = np.ma.MaskedArray(self.values, mask)
        self.assertMaskedArrayEqual(result, expected)

    def test_mask_maskedarray(self):
        # NaN           -> NaN
        # Extrapolated  -> Masked
        # Masked        -> Masked
        data = np.ma.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        data[2, 3] = np.ma.masked
        result = self._regrid(data, 'mask')
        self.assertIsInstance(result, np.ma.MaskedArray)
        mask = [[0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 1]]
        expected = np.ma.MaskedArray(self.values, mask)
        self.assertMaskedArrayEqual(result, expected)

    def test_nanmask_ndarray(self):
        # NaN           -> NaN
        # Extrapolated  -> NaN
        data = np.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        result = self._regrid(data, 'nanmask')
        self.assertNotIsInstance(result, np.ma.MaskedArray)
        self.assertArrayEqual(result, self.values)

    def test_nanmask_maskedarray(self):
        # NaN           -> NaN
        # Extrapolated  -> Masked
        # Masked        -> Masked
        data = np.ma.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        data[2, 3] = np.ma.masked
        result = self._regrid(data, 'nanmask')
        self.assertIsInstance(result, np.ma.MaskedArray)
        mask = [[0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 1]]
        expected = np.ma.MaskedArray(self.values, mask)
        self.assertMaskedArrayEqual(result, expected)

    def test_invalid(self):
        data = np.arange(12, dtype=np.float).reshape(3, 4)
        with self.assertRaisesRegexp(ValueError, 'Invalid extrapolation mode'):
            self._regrid(data, 'BOGUS')


if __name__ == '__main__':
    tests.main()
