# (C) British Crown Copyright 2014 - 2015, Met Office
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
"""Unit tests for :class:`iris.analysis._regrid.RectilinearRegridder`."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import numpy as np

from iris.analysis._regrid import RectilinearRegridder as Regridder
from iris.aux_factory import HybridHeightFactory
from iris.coord_systems import GeogCS, OSGB
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.tests.stock import global_pp, lat_lon_cube, realistic_4d


RESULT_DIR = ('analysis', 'regrid')

# Convenience to access Regridder static method.
regrid = Regridder._regrid


class Test__regrid__linear(tests.IrisTest):
    def setUp(self):
        self.x = DimCoord(np.linspace(-2, 57, 60))
        self.y = DimCoord(np.linspace(0, 49, 50))
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
        result = regrid(self.data, self.x_dim, self.y_dim,
                        self.x, self.y,
                        np.array([xs]), np.array([ys]))
        self.assertArrayAllClose(result, expecteds, rtol=1e-04)

        # Check that transposing the input data results in the same values
        ndim = self.data.ndim
        result2 = regrid(self.data.T, ndim - self.x_dim - 1,
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
        result = regrid(self.data, self.x_dim, self.y_dim,
                        self.x, self.y,
                        self.target_x, self.target_y)
        self.assertArrayEqual(result, self.expected)

    def test_simple_masked(self):
        data = np.ma.MaskedArray(self.data, mask=True)
        data.mask[:, 1:30, 1:30] = False
        result = regrid(data, self.x_dim, self.y_dim,
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
        result = regrid(data, self.x_dim, self.y_dim,
                        self.x, self.y,
                        self.target_x, self.target_y)
        self.assertIsInstance(result, np.ma.MaskedArray)

    def test_result_transpose_shape(self):
        ndim = self.data.ndim
        result = regrid(self.data.T, ndim - self.x_dim - 1,
                        ndim - self.y_dim - 1, self.x, self.y,
                        self.target_x, self.target_y)
        self.assertArrayEqual(result, self.expected.T)

    def test_reverse_x_coord(self):
        index = [slice(None)] * self.data.ndim
        index[self.x_dim] = slice(None, None, -1)
        result = regrid(self.data[index], self.x_dim,
                        self.y_dim, self.x[::-1], self.y,
                        self.target_x, self.target_y)
        self.assertArrayEqual(result, self.expected)

    def test_circular_x_coord(self):
        # Check that interpolation of a circular src coordinate doesn't result
        # in an out of bounds value.
        self.x.circular = True
        self.x.units = 'degree'
        result = regrid(self.data, self.x_dim, self.y_dim,
                        self.x, self.y, np.array([[58]]),
                        np.array([[0]]))
        self.assertArrayAlmostEqual(result,
                                    np.array([56.80398671, 113.60797342],
                                             ndmin=self.data.ndim))


# Check what happens to NaN values, extrapolated values, and
# masked values.
class Test__regrid__extrapolation_modes(tests.IrisTest):
    values_by_method = {'linear': [[np.nan, np.nan, 2, 3, np.nan],
                                   [np.nan, np.nan, 6, 7, np.nan],
                                   [8, 9, 10, 11, np.nan]],
                        'nearest': [[np.nan, 1, 2, 3, np.nan],
                                    [4, 5, 6, 7, np.nan],
                                    [8, 9, 10, 11, np.nan]]}

    extrapolate_values_by_method = {'linear': [[np.nan, np.nan, 2, 3, 4],
                                               [np.nan, np.nan, 6, 7, 8],
                                               [8, 9, 10, 11, 12]],
                                    'nearest': [[np.nan, 1, 2, 3, 3],
                                                [4, 5, 6, 7, 7],
                                                [8, 9, 10, 11, 11]]}

    def setUp(self):
        self.methods = ('linear', 'nearest')

    def _regrid(self, data, method, extrapolation_mode=None):
        x = np.arange(4)
        y = np.arange(3)
        x_coord = DimCoord(x)
        y_coord = DimCoord(y)
        x_dim, y_dim = 1, 0
        grid_x, grid_y = np.meshgrid(np.arange(5), y)
        kwargs = dict(method=method)
        if extrapolation_mode is not None:
            kwargs['extrapolation_mode'] = extrapolation_mode
        result = regrid(data, x_dim, y_dim, x_coord, y_coord,
                        grid_x, grid_y, **kwargs)
        return result

    def test_default_ndarray(self):
        # NaN           -> NaN
        # Extrapolated  -> NaN
        data = np.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        for method in self.methods:
            result = self._regrid(data, method)
            self.assertNotIsInstance(result, np.ma.MaskedArray)
            expected = self.values_by_method[method]
            self.assertArrayEqual(result, expected)

    def test_default_maskedarray(self):
        # NaN           -> NaN
        # Extrapolated  -> Masked
        # Masked        -> Masked
        data = np.ma.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        data[2, 3] = np.ma.masked
        for method in self.methods:
            result = self._regrid(data, method)
            self.assertIsInstance(result, np.ma.MaskedArray)
            mask = [[0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 1]]
            values = self.values_by_method[method]
            expected = np.ma.MaskedArray(values, mask)
            self.assertMaskedArrayEqual(result, expected)

    def test_default_maskedarray_none_masked(self):
        # NaN           -> NaN
        # Extrapolated  -> Masked
        # Masked        -> N/A
        data = np.ma.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        for method in self.methods:
            result = self._regrid(data, method)
            self.assertIsInstance(result, np.ma.MaskedArray)
            mask = [[0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1]]
            values = self.values_by_method[method]
            expected = np.ma.MaskedArray(values, mask)
            self.assertMaskedArrayEqual(result, expected)

    def test_default_maskedarray_none_masked_expanded(self):
        # NaN           -> NaN
        # Extrapolated  -> Masked
        # Masked        -> N/A
        data = np.ma.arange(12, dtype=np.float).reshape(3, 4)
        # Make sure the mask has been expanded
        data.mask = False
        data[0, 0] = np.nan
        for method in self.methods:
            result = self._regrid(data, method)
            self.assertIsInstance(result, np.ma.MaskedArray)
            mask = [[0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1]]
            values = self.values_by_method[method]
            expected = np.ma.MaskedArray(values, mask)
            self.assertMaskedArrayEqual(result, expected)

    def test_method_ndarray(self):
        # NaN           -> NaN
        # Extrapolated  -> linear
        data = np.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        for method in self.methods:
            result = self._regrid(data, method, 'extrapolate')
            self.assertNotIsInstance(result, np.ma.MaskedArray)
            expected = self.extrapolate_values_by_method[method]
            self.assertArrayEqual(result, expected)

    def test_method_maskedarray(self):
        # NaN           -> NaN
        # Extrapolated  -> linear
        # Masked        -> Masked
        data = np.ma.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        data[2, 3] = np.ma.masked
        for method in self.methods:
            result = self._regrid(data, method, 'extrapolate')
            self.assertIsInstance(result, np.ma.MaskedArray)
            mask = [[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1]]
            values = self.extrapolate_values_by_method[method]
            expected = np.ma.MaskedArray(values, mask)
            self.assertMaskedArrayEqual(result, expected)

    def test_nan_ndarray(self):
        # NaN           -> NaN
        # Extrapolated  -> NaN
        data = np.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        for method in self.methods:
            result = self._regrid(data, method, 'nan')
            self.assertNotIsInstance(result, np.ma.MaskedArray)
            expected = self.values_by_method[method]
            self.assertArrayEqual(result, expected)

    def test_nan_maskedarray(self):
        # NaN           -> NaN
        # Extrapolated  -> NaN
        # Masked        -> Masked
        data = np.ma.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        data[2, 3] = np.ma.masked
        for method in self.methods:
            result = self._regrid(data, method, 'nan')
            self.assertIsInstance(result, np.ma.MaskedArray)
            mask = [[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0]]
            values = self.values_by_method[method]
            expected = np.ma.MaskedArray(values, mask)
            self.assertMaskedArrayEqual(result, expected)

    def test_error_ndarray(self):
        # Values irrelevant - the function raises an error.
        data = np.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        for method in self.methods:
            with self.assertRaisesRegexp(ValueError, 'out of bounds'):
                self._regrid(data, method, 'error')

    def test_error_maskedarray(self):
        # Values irrelevant - the function raises an error.
        data = np.ma.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        data[2, 3] = np.ma.masked
        for method in self.methods:
            with self.assertRaisesRegexp(ValueError, 'out of bounds'):
                self._regrid(data, method, 'error')

    def test_mask_ndarray(self):
        # NaN           -> NaN
        # Extrapolated  -> Masked (this is different from all the other
        #                          modes)
        data = np.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        for method in self.methods:
            result = self._regrid(data, method, 'mask')
            self.assertIsInstance(result, np.ma.MaskedArray)
            mask = [[0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1]]
            values = self.values_by_method[method]
            expected = np.ma.MaskedArray(values, mask)
            self.assertMaskedArrayEqual(result, expected)

    def test_mask_maskedarray(self):
        # NaN           -> NaN
        # Extrapolated  -> Masked
        # Masked        -> Masked
        data = np.ma.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        data[2, 3] = np.ma.masked
        for method in self.methods:
            result = self._regrid(data, method, 'mask')
            self.assertIsInstance(result, np.ma.MaskedArray)
            mask = [[0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 1]]
            values = self.values_by_method[method]
            expected = np.ma.MaskedArray(values, mask)
            self.assertMaskedArrayEqual(result, expected)

    def test_nanmask_ndarray(self):
        # NaN           -> NaN
        # Extrapolated  -> NaN
        data = np.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        for method in self.methods:
            result = self._regrid(data, method, 'nanmask')
            self.assertNotIsInstance(result, np.ma.MaskedArray)
            expected = self.values_by_method[method]
            self.assertArrayEqual(result, expected)

    def test_nanmask_maskedarray(self):
        # NaN           -> NaN
        # Extrapolated  -> Masked
        # Masked        -> Masked
        data = np.ma.arange(12, dtype=np.float).reshape(3, 4)
        data[0, 0] = np.nan
        data[2, 3] = np.ma.masked
        for method in self.methods:
            result = self._regrid(data, method, 'nanmask')
            self.assertIsInstance(result, np.ma.MaskedArray)
            mask = [[0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 1]]
            values = self.values_by_method[method]
            expected = np.ma.MaskedArray(values, mask)
            self.assertMaskedArrayEqual(result, expected)

    def test_invalid(self):
        data = np.arange(12, dtype=np.float).reshape(3, 4)
        emsg = 'Invalid extrapolation mode'
        for method in self.methods:
            with self.assertRaisesRegexp(ValueError, emsg):
                self._regrid(data, method, 'BOGUS')


class Test___call____invalid_types(tests.IrisTest):
    def setUp(self):
        self.cube = lat_lon_cube()
        # Regridder method and extrapolation-mode.
        self.args = ('linear', 'mask')
        self.regridder = Regridder(self.cube, self.cube, *self.args)

    def test_src_as_array(self):
        arr = np.zeros((3, 4))
        with self.assertRaises(TypeError):
            Regridder(arr, self.cube, *self.args)
        with self.assertRaises(TypeError):
            self.regridder(arr)

    def test_grid_as_array(self):
        with self.assertRaises(TypeError):
            Regridder(self.cube, np.zeros((3, 4)), *self.args)

    def test_src_as_int(self):
        with self.assertRaises(TypeError):
            Regridder(42, self.cube, *self.args)
        with self.assertRaises(TypeError):
            self.regridder(42)

    def test_grid_as_int(self):
        with self.assertRaises(TypeError):
            Regridder(self.cube, 42, *self.args)


class Test___call____missing_coords(tests.IrisTest):
    def setUp(self):
        self.args = ('linear', 'mask')

    def ok_bad(self, coord_names):
        # Deletes the named coords from `bad`.
        ok = lat_lon_cube()
        bad = lat_lon_cube()
        for name in coord_names:
            bad.remove_coord(name)
        return ok, bad

    def test_src_missing_lat(self):
        ok, bad = self.ok_bad(['latitude'])
        with self.assertRaises(ValueError):
            Regridder(bad, ok, *self.args)
        regridder = Regridder(ok, ok, *self.args)
        with self.assertRaises(ValueError):
            regridder(bad)

    def test_grid_missing_lat(self):
        ok, bad = self.ok_bad(['latitude'])
        with self.assertRaises(ValueError):
            Regridder(ok, bad, *self.args)

    def test_src_missing_lon(self):
        ok, bad = self.ok_bad(['longitude'])
        with self.assertRaises(ValueError):
            Regridder(bad, ok, *self.args)
        regridder = Regridder(ok, ok, *self.args)
        with self.assertRaises(ValueError):
            regridder(bad)

    def test_grid_missing_lon(self):
        ok, bad = self.ok_bad(['longitude'])
        with self.assertRaises(ValueError):
            Regridder(ok, bad, *self.args)

    def test_src_missing_lat_lon(self):
        ok, bad = self.ok_bad(['latitude', 'longitude'])
        with self.assertRaises(ValueError):
            Regridder(bad, ok, *self.args)
        regridder = Regridder(ok, ok, *self.args)
        with self.assertRaises(ValueError):
            regridder(bad)

    def test_grid_missing_lat_lon(self):
        ok, bad = self.ok_bad(['latitude', 'longitude'])
        with self.assertRaises(ValueError):
            Regridder(ok, bad, *self.args)


class Test___call____not_dim_coord(tests.IrisTest):
    def setUp(self):
        self.args = ('linear', 'mask')

    def ok_bad(self, coord_name):
        # Demotes the named DimCoord on `bad` to an AuxCoord.
        ok = lat_lon_cube()
        bad = lat_lon_cube()
        coord = bad.coord(coord_name)
        dims = bad.coord_dims(coord)
        bad.remove_coord(coord_name)
        aux_coord = AuxCoord.from_coord(coord)
        bad.add_aux_coord(aux_coord, dims)
        return ok, bad

    def test_src_with_aux_lat(self):
        ok, bad = self.ok_bad('latitude')
        with self.assertRaises(ValueError):
            Regridder(bad, ok, *self.args)
        regridder = Regridder(ok, ok, *self.args)
        with self.assertRaises(ValueError):
            regridder(bad)

    def test_grid_with_aux_lat(self):
        ok, bad = self.ok_bad('latitude')
        with self.assertRaises(ValueError):
            Regridder(ok, bad, *self.args)

    def test_src_with_aux_lon(self):
        ok, bad = self.ok_bad('longitude')
        with self.assertRaises(ValueError):
            Regridder(bad, ok, *self.args)
        regridder = Regridder(ok, ok, *self.args)
        with self.assertRaises(ValueError):
            regridder(bad)

    def test_grid_with_aux_lon(self):
        ok, bad = self.ok_bad('longitude')
        with self.assertRaises(ValueError):
            Regridder(ok, bad, *self.args)


class Test___call____not_dim_coord_share(tests.IrisTest):
    def setUp(self):
        self.args = ('linear', 'mask')

    def ok_bad(self):
        # Make lat/lon share a single dimension on `bad`.
        ok = lat_lon_cube()
        bad = lat_lon_cube()
        lat = bad.coord('latitude')
        bad = bad[0, :lat.shape[0]]
        bad.remove_coord('latitude')
        bad.add_aux_coord(lat, 0)
        return ok, bad

    def test_src_shares_dim(self):
        ok, bad = self.ok_bad()
        with self.assertRaises(ValueError):
            Regridder(bad, ok, *self.args)
        regridder = Regridder(ok, ok, *self.args)
        with self.assertRaises(ValueError):
            regridder(bad)

    def test_grid_shares_dim(self):
        ok, bad = self.ok_bad()
        with self.assertRaises(ValueError):
            Regridder(ok, bad, *self.args)


class Test___call____bad_georeference(tests.IrisTest):
    def setUp(self):
        self.args = ('linear', 'mask')

    def ok_bad(self, lat_cs, lon_cs):
        # Updates `bad` to use the given coordinate systems.
        ok = lat_lon_cube()
        bad = lat_lon_cube()
        bad.coord('latitude').coord_system = lat_cs
        bad.coord('longitude').coord_system = lon_cs
        return ok, bad

    def test_src_no_cs(self):
        ok, bad = self.ok_bad(None, None)
        regridder = Regridder(bad, ok, *self.args)
        with self.assertRaises(ValueError):
            regridder(bad)

    def test_grid_no_cs(self):
        ok, bad = self.ok_bad(None, None)
        regridder = Regridder(ok, bad, *self.args)
        with self.assertRaises(ValueError):
            regridder(ok)

    def test_src_one_cs(self):
        ok, bad = self.ok_bad(None, GeogCS(6371000))
        with self.assertRaises(ValueError):
            Regridder(bad, ok, *self.args)

    def test_grid_one_cs(self):
        ok, bad = self.ok_bad(None, GeogCS(6371000))
        with self.assertRaises(ValueError):
            Regridder(ok, bad, *self.args)

    def test_src_inconsistent_cs(self):
        ok, bad = self.ok_bad(GeogCS(6370000), GeogCS(6371000))
        with self.assertRaises(ValueError):
            Regridder(bad, ok, *self.args)

    def test_grid_inconsistent_cs(self):
        ok, bad = self.ok_bad(GeogCS(6370000), GeogCS(6371000))
        with self.assertRaises(ValueError):
            Regridder(ok, bad, *self.args)


class Test___call____bad_angular_units(tests.IrisTest):
    def ok_bad(self):
        # Changes the longitude coord to radians on `bad`.
        ok = lat_lon_cube()
        bad = lat_lon_cube()
        bad.coord('longitude').units = 'radians'
        return ok, bad

    def test_src_radians(self):
        ok, bad = self.ok_bad()
        regridder = Regridder(bad, ok, 'linear', 'mask')
        with self.assertRaises(ValueError):
            regridder(bad)

    def test_grid_radians(self):
        ok, bad = self.ok_bad()
        with self.assertRaises(ValueError):
            Regridder(ok, bad, 'linear', 'mask')


def uk_cube():
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    uk = Cube(data)
    cs = OSGB()
    y_coord = DimCoord(range(3), 'projection_y_coordinate', units='m',
                       coord_system=cs)
    x_coord = DimCoord(range(4), 'projection_x_coordinate', units='m',
                       coord_system=cs)
    uk.add_dim_coord(y_coord, 0)
    uk.add_dim_coord(x_coord, 1)
    surface = AuxCoord(data * 10, 'surface_altitude', units='m')
    uk.add_aux_coord(surface, (0, 1))
    uk.add_aux_factory(HybridHeightFactory(orography=surface))
    return uk


class Test___call____bad_linear_units(tests.IrisTest):
    def ok_bad(self):
        # Defines `bad` with an x coordinate in km.
        ok = lat_lon_cube()
        bad = uk_cube()
        bad.coord(axis='x').units = 'km'
        return ok, bad

    def test_src_km(self):
        ok, bad = self.ok_bad()
        regridder = Regridder(bad, ok, 'linear', 'mask')
        with self.assertRaises(ValueError):
            regridder(bad)

    def test_grid_km(self):
        ok, bad = self.ok_bad()
        with self.assertRaises(ValueError):
            Regridder(ok, bad, 'linear', 'mask')


class Test___call____no_coord_systems(tests.IrisTest):
    # Test behaviour in the absence of any coordinate systems.

    def setUp(self):
        self.mode = 'mask'
        self.methods = ('linear', 'nearest')

    def remove_coord_systems(self, cube):
        for coord in cube.coords():
            coord.coord_system = None

    def test_ok(self):
        # Ensure regridding is supported when the coordinate definitions match.
        # NB. We change the coordinate *values* to ensure that does not
        # prevent the regridding operation.
        src = uk_cube()
        self.remove_coord_systems(src)
        grid = src.copy()
        for coord in grid.dim_coords:
            coord.points = coord.points + 1
        for method in self.methods:
            regridder = Regridder(src, grid, method, self.mode)
            result = regridder(src)
            for coord in result.dim_coords:
                self.assertEqual(coord, grid.coord(coord))
            expected = np.ma.arange(12).reshape((3, 4)) + 5
            expected[:, 3] = np.ma.masked
            expected[2, :] = np.ma.masked
            self.assertMaskedArrayEqual(result.data, expected)

    def test_matching_units(self):
        # Check we are insensitive to the units provided they match.
        # NB. We change the coordinate *values* to ensure that does not
        # prevent the regridding operation.
        src = uk_cube()
        self.remove_coord_systems(src)
        # Move to unusual units (i.e. not metres or degrees).
        for coord in src.dim_coords:
            coord.units = 'feet'
        grid = src.copy()
        for coord in grid.dim_coords:
            coord.points = coord.points + 1
        for method in self.methods:
            regridder = Regridder(src, grid, method, self.mode)
            result = regridder(src)
            for coord in result.dim_coords:
                self.assertEqual(coord, grid.coord(coord))
            expected = np.ma.arange(12).reshape((3, 4)) + 5
            expected[:, 3] = np.ma.masked
            expected[2, :] = np.ma.masked
            self.assertMaskedArrayEqual(result.data, expected)

    def test_different_units(self):
        src = uk_cube()
        self.remove_coord_systems(src)
        # Move to unusual units (i.e. not metres or degrees).
        for coord in src.coords():
            coord.units = 'feet'
        grid = src.copy()
        grid.coord('projection_y_coordinate').units = 'yards'
        # We change the coordinate *values* to ensure that does not
        # prevent the regridding operation.
        for coord in grid.dim_coords:
            coord.points = coord.points + 1
        for method in self.methods:
            regridder = Regridder(src, grid, method, self.mode)
            emsg = 'matching coordinate metadata'
            with self.assertRaisesRegexp(ValueError, emsg):
                regridder(src)

    def test_coord_metadata_mismatch(self):
        # Check for failure when coordinate definitions differ.
        uk = uk_cube()
        self.remove_coord_systems(uk)
        lat_lon = lat_lon_cube()
        self.remove_coord_systems(lat_lon)
        for method in self.methods:
            regridder = Regridder(uk, lat_lon, method, self.mode)
            with self.assertRaises(ValueError):
                regridder(uk)


class Test___call____extrapolation_modes(tests.IrisTest):
    values = [[np.nan, 6, 7, np.nan],
              [9, 10, 11, np.nan],
              [np.nan, np.nan, np.nan, np.nan]]

    extrapolate_values_by_method = {'linear': [[np.nan, 6, 7, 8],
                                               [9, 10, 11, 12],
                                               [13, 14, 15, 16]],
                                    'nearest': [[np.nan, 6, 7, 7],
                                                [9, 10, 11, 11],
                                                [9, 10, 11, 11]]}

    surface_values = [[50, 60, 70, np.nan],
                      [90, 100, 110, np.nan],
                      [np.nan, np.nan, np.nan, np.nan]]

    def setUp(self):
        self.methods = ('linear', 'nearest')

    def _ndarray_cube(self, method):
        assert method in self.methods
        src = uk_cube()
        index = (0, 0) if method == 'linear' else (1, 1)
        src.data[index] = np.nan
        return src

    def _masked_cube(self, method):
        assert method in self.methods
        src = uk_cube()
        src.data = np.ma.asarray(src.data)
        nan_index = (0, 0) if method == 'linear' else (1, 1)
        mask_index = (2, 3)
        src.data[nan_index] = np.nan
        src.data[mask_index] = np.ma.masked
        return src

    def _regrid(self, src, method, extrapolation_mode='mask'):
        grid = src.copy()
        for coord in grid.dim_coords:
            coord.points = coord.points + 1
        regridder = Regridder(src, grid, method, extrapolation_mode)
        result = regridder(src)

        surface = result.coord('surface_altitude').points
        self.assertNotIsInstance(surface, np.ma.MaskedArray)
        self.assertArrayEqual(surface, self.surface_values)

        return result.data

    def test_default_ndarray(self):
        # NaN           -> NaN
        # Extrapolated  -> Masked
        for method in self.methods:
            src = self._ndarray_cube(method)
            result = self._regrid(src, method)
            self.assertIsInstance(result, np.ma.MaskedArray)
            mask = [[0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [1, 1, 1, 1]]
            expected = np.ma.MaskedArray(self.values, mask)
            self.assertMaskedArrayEqual(result, expected)

    def test_default_maskedarray(self):
        # NaN           -> NaN
        # Extrapolated  -> Masked
        # Masked        -> Masked
        for method in self.methods:
            src = self._masked_cube(method)
            result = self._regrid(src, method)
            self.assertIsInstance(result, np.ma.MaskedArray)
            mask = [[0, 0, 0, 1],
                    [0, 0, 1, 1],
                    [1, 1, 1, 1]]
            expected = np.ma.MaskedArray(self.values, mask)
            self.assertMaskedArrayEqual(result, expected)

    def test_default_maskedarray_none_masked(self):
        # NaN           -> NaN
        # Extrapolated  -> Masked
        # Masked        -> N/A
        for method in self.methods:
            src = uk_cube()
            src.data = np.ma.asarray(src.data)
            index = (0, 0) if method == 'linear' else (1, 1)
            src.data[index] = np.nan
            result = self._regrid(src, method)
            self.assertIsInstance(result, np.ma.MaskedArray)
            mask = [[0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [1, 1, 1, 1]]
            expected = np.ma.MaskedArray(self.values, mask)
            self.assertMaskedArrayEqual(result, expected)

    def test_default_maskedarray_none_masked_expanded(self):
        # NaN           -> NaN
        # Extrapolated  -> Masked
        # Masked        -> N/A
        for method in self.methods:
            src = uk_cube()
            src.data = np.ma.asarray(src.data)
            # Make sure the mask has been expanded
            src.data.mask = False
            index = (0, 0) if method == 'linear' else (1, 1)
            src.data[index] = np.nan
            result = self._regrid(src, method)
            self.assertIsInstance(result, np.ma.MaskedArray)
            mask = [[0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [1, 1, 1, 1]]
            expected = np.ma.MaskedArray(self.values, mask)
            self.assertMaskedArrayEqual(result, expected)

    def test_method_ndarray(self):
        # NaN           -> NaN
        # Extrapolated  -> linear
        for method in self.methods:
            src = self._ndarray_cube(method)
            result = self._regrid(src, method, 'extrapolate')
            self.assertNotIsInstance(result, np.ma.MaskedArray)
            expected = self.extrapolate_values_by_method[method]
            self.assertArrayEqual(result, expected)

    def test_nan_ndarray(self):
        # NaN           -> NaN
        # Extrapolated  -> NaN
        for method in self.methods:
            src = self._ndarray_cube(method)
            result = self._regrid(src, method, 'nan')
            self.assertNotIsInstance(result, np.ma.MaskedArray)
            self.assertArrayEqual(result, self.values)

    def test_nan_maskedarray(self):
        # NaN           -> NaN
        # Extrapolated  -> NaN
        # Masked        -> Masked
        for method in self.methods:
            src = self._masked_cube(method)
            result = self._regrid(src, method, 'nan')
            self.assertIsInstance(result, np.ma.MaskedArray)
            mask = [[0, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 0]]
            expected = np.ma.MaskedArray(self.values, mask)
            self.assertMaskedArrayEqual(result, expected)

    def test_error_ndarray(self):
        # Values irrelevant - the function raises an error.
        for method in self.methods:
            src = self._ndarray_cube(method)
            with self.assertRaisesRegexp(ValueError, 'out of bounds'):
                self._regrid(src, method, 'error')

    def test_error_maskedarray(self):
        # Values irrelevant - the function raises an error.
        for method in self.methods:
            src = self._masked_cube(method)
            with self.assertRaisesRegexp(ValueError, 'out of bounds'):
                self._regrid(src, method, 'error')

    def test_mask_ndarray(self):
        # NaN           -> NaN
        # Extrapolated  -> Masked (this is different from all the other
        #                          modes)
        for method in self.methods:
            src = self._ndarray_cube(method)
            result = self._regrid(src, method, 'mask')
            self.assertIsInstance(result, np.ma.MaskedArray)
            mask = [[0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [1, 1, 1, 1]]
            expected = np.ma.MaskedArray(self.values, mask)
            self.assertMaskedArrayEqual(result, expected)

    def test_mask_maskedarray(self):
        # NaN           -> NaN
        # Extrapolated  -> Masked
        # Masked        -> Masked
        for method in self.methods:
            src = self._masked_cube(method)
            result = self._regrid(src, method, 'mask')
            self.assertIsInstance(result, np.ma.MaskedArray)
            mask = [[0, 0, 0, 1],
                    [0, 0, 1, 1],
                    [1, 1, 1, 1]]
            expected = np.ma.MaskedArray(self.values, mask)
            self.assertMaskedArrayEqual(result, expected)

    def test_nanmask_ndarray(self):
        # NaN           -> NaN
        # Extrapolated  -> NaN
        for method in self.methods:
            src = self._ndarray_cube(method)
            result = self._regrid(src, method, 'nanmask')
            self.assertNotIsInstance(result, np.ma.MaskedArray)
            self.assertArrayEqual(result, self.values)

    def test_nanmask_maskedarray(self):
        # NaN           -> NaN
        # Extrapolated  -> Masked
        # Masked        -> Masked
        for method in self.methods:
            src = self._masked_cube(method)
            result = self._regrid(src, method, 'nanmask')
            self.assertIsInstance(result, np.ma.MaskedArray)
            mask = [[0, 0, 0, 1],
                    [0, 0, 1, 1],
                    [1, 1, 1, 1]]
            expected = np.ma.MaskedArray(self.values, mask)
            self.assertMaskedArrayEqual(result, expected)

    def test_invalid(self):
        src = uk_cube()
        emsg = 'Invalid extrapolation mode'
        for method in self.methods:
            with self.assertRaisesRegexp(ValueError, emsg):
                self._regrid(src, method, 'BOGUS')


@tests.skip_data
class Test___call____rotated_to_lat_lon(tests.IrisTest):
    def setUp(self):
        self.src = realistic_4d()[:5, :2, ::40, ::30]
        # Regridder method and extrapolation-mode.
        self.args = ('linear', 'mask')
        self.mode = 'mask'
        self.methods = ('linear', 'nearest')

    def test_single_point(self):
        src = self.src[0, 0]
        grid = global_pp()[:1, :1]
        # These coordinate values have been derived by converting the
        # rotated coordinates of src[1, 1] into lat/lon by using cs2cs.
        grid.coord('longitude').points = -3.144870
        grid.coord('latitude').points = 52.406444
        for method in self.methods:
            regridder = Regridder(src, grid, method, self.mode)
            result = regridder(src)
            self.assertEqual(src.data[1, 1], result.data)

    def test_transposed_src(self):
        # The source dimensions are in a non-standard order.
        src = self.src
        src.transpose([3, 1, 2, 0])
        grid = self._grid_subset()
        for method in self.methods:
            regridder = Regridder(src, grid, method, self.mode)
            result = regridder(src)
            result.transpose([3, 1, 2, 0])
            cml = RESULT_DIR + ('{}_subset.cml'.format(method),)
            self.assertCMLApproxData(result, cml)

    def _grid_subset(self):
        # The destination grid points are entirely contained within the
        # src grid points.
        grid = global_pp()[:4, :5]
        grid.coord('longitude').points = np.linspace(-3.182, -3.06, 5)
        grid.coord('latitude').points = np.linspace(52.372, 52.44, 4)
        return grid

    def test_reversed(self):
        src = self.src
        grid = self._grid_subset()

        for method in self.methods:
            cml = RESULT_DIR + ('{}_subset.cml'.format(method),)
            regridder = Regridder(src, grid[::-1], method, self.mode)
            result = regridder(src)
            self.assertCMLApproxData(result[:, :, ::-1], cml)

            sample = src[:, :, ::-1]
            regridder = Regridder(sample, grid[::-1], method, self.mode)
            result = regridder(sample)
            self.assertCMLApproxData(result[:, :, ::-1], cml)

            sample = src[:, :, :, ::-1]
            regridder = Regridder(sample, grid[::-1], method, self.mode)
            result = regridder(sample)
            self.assertCMLApproxData(result[:, :, ::-1], cml)

            sample = src[:, :, ::-1, ::-1]
            regridder = Regridder(sample, grid[::-1], method, self.mode)
            result = regridder(sample)
            self.assertCMLApproxData(result[:, :, ::-1], cml)

            regridder = Regridder(src, grid[:, ::-1], method, self.mode)
            result = regridder(src)
            self.assertCMLApproxData(result[:, :, :, ::-1], cml)

            sample = src[:, :, ::-1]
            regridder = Regridder(sample, grid[:, ::-1], method, self.mode)
            result = regridder(sample)
            self.assertCMLApproxData(result[:, :, :, ::-1], cml)

            sample = src[:, :, :, ::-1]
            regridder = Regridder(sample, grid[:, ::-1], method, self.mode)
            result = regridder(sample)
            self.assertCMLApproxData(result[:, :, :, ::-1], cml)

            sample = src[:, :, ::-1, ::-1]
            regridder = Regridder(sample, grid[:, ::-1], method, self.mode)
            result = regridder(sample)
            self.assertCMLApproxData(result[:, :, :, ::-1], cml)

            regridder = Regridder(src, grid[::-1, ::-1], method, self.mode)
            result = regridder(src)
            self.assertCMLApproxData(result[:, :, ::-1, ::-1], cml)

            sample = src[:, :, ::-1]
            regridder = Regridder(sample, grid[::-1, ::-1], method, self.mode)
            result = regridder(sample)
            self.assertCMLApproxData(result[:, :, ::-1, ::-1], cml)

            sample = src[:, :, :, ::-1]
            regridder = Regridder(sample, grid[::-1, ::-1], method, self.mode)
            result = regridder(sample)
            self.assertCMLApproxData(result[:, :, ::-1, ::-1], cml)

            sample = src[:, :, ::-1, ::-1]
            regridder = Regridder(sample, grid[::-1, ::-1], method, self.mode)
            result = regridder(sample)
            self.assertCMLApproxData(result[:, :, ::-1, ::-1], cml)

    def test_grid_subset(self):
        # The destination grid points are entirely contained within the
        # src grid points.
        grid = self._grid_subset()
        for method in self.methods:
            regridder = Regridder(self.src, grid, method, self.mode)
            result = regridder(self.src)
            cml = RESULT_DIR + ('{}_subset.cml'.format(method),)
            self.assertCMLApproxData(result, cml)

    def _big_grid(self):
        grid = self._grid_subset()
        big_grid = Cube(np.zeros((5, 10, 3, 4, 5)))
        big_grid.add_dim_coord(grid.coord('latitude'), 3)
        big_grid.add_dim_coord(grid.coord('longitude'), 4)
        return big_grid

    def test_grid_subset_big(self):
        # Add some extra dimensions to the destination Cube and
        # these should be safely ignored.
        big_grid = self._big_grid()
        for method in self.methods:
            regridder = Regridder(self.src, big_grid, method, self.mode)
            result = regridder(self.src)
            cml = RESULT_DIR + ('{}_subset.cml'.format(method),)
            self.assertCMLApproxData(result, cml)

    def test_grid_subset_big_transposed(self):
        # The order of the grid's dimensions (including the X and Y
        # dimensions) must not affect the result.
        big_grid = self._big_grid()
        big_grid.transpose([4, 0, 3, 1, 2])
        for method in self.methods:
            regridder = Regridder(self.src, big_grid, method, self.mode)
            result = regridder(self.src)
            cml = RESULT_DIR + ('{}_subset.cml'.format(method),)
            self.assertCMLApproxData(result, cml)

    def test_grid_subset_anon(self):
        # Must cope OK with anonymous source dimensions.
        src = self.src
        src.remove_coord('time')
        grid = self._grid_subset()
        for method in self.methods:
            regridder = Regridder(src, grid, method, self.mode)
            result = regridder(src)
            cml = RESULT_DIR + ('{}_subset_anon.cml'.format(method),)
            self.assertCMLApproxData(result, cml)

    def test_grid_subset_missing_data_1(self):
        # The destination grid points are entirely contained within the
        # src grid points AND we have missing data.
        src = self.src
        src.data = np.ma.MaskedArray(src.data)
        src.data[:, :, 0, 0] = np.ma.masked
        grid = self._grid_subset()
        for method in self.methods:
            regridder = Regridder(src, grid, method, self.mode)
            result = regridder(src)
            cml = RESULT_DIR + ('{}_subset_masked_1.cml'.format(method),)
            self.assertCMLApproxData(result, cml)

    def test_grid_subset_missing_data_2(self):
        # The destination grid points are entirely contained within the
        # src grid points AND we have missing data.
        src = self.src
        src.data = np.ma.MaskedArray(src.data)
        src.data[:, :, 1, 2] = np.ma.masked
        grid = self._grid_subset()
        for method in self.methods:
            regridder = Regridder(src, grid, method, self.mode)
            result = regridder(src)
            cml = RESULT_DIR + ('{}_subset_masked_2.cml'.format(method),)
            self.assertCMLApproxData(result, cml)

    def test_grid_partial_overlap(self):
        # The destination grid points are partially contained within the
        # src grid points.
        grid = global_pp()[:4, :4]
        grid.coord('longitude').points = np.linspace(-3.3, -3.06, 4)
        grid.coord('latitude').points = np.linspace(52.377, 52.43, 4)
        for method in self.methods:
            regridder = Regridder(self.src, grid, method, self.mode)
            result = regridder(self.src)
            cml = RESULT_DIR + ('{}_partial_overlap.cml'.format(method),)
            self.assertCMLApproxData(result, cml)

    def test_grid_no_overlap(self):
        # The destination grid points are NOT contained within the
        # src grid points.
        grid = global_pp()[:4, :4]
        grid.coord('longitude').points = np.linspace(-3.3, -3.2, 4)
        grid.coord('latitude').points = np.linspace(52.377, 52.43, 4)
        for method in self.methods:
            regridder = Regridder(self.src, grid, method, self.mode)
            result = regridder(self.src)
            self.assertCMLApproxData(result, RESULT_DIR + ('no_overlap.cml',))

    def test_grid_subset_missing_data_aux(self):
        # The destination grid points are entirely contained within the
        # src grid points AND we have missing data on the aux coordinate.
        src = self.src
        src.coord('surface_altitude').points[1, 2] = np.ma.masked
        grid = self._grid_subset()
        for method in self.methods:
            regridder = Regridder(src, grid, method, self.mode)
            result = regridder(src)
            cml = RESULT_DIR + ('{}_masked_altitude.cml'.format(method),)
            self.assertCMLApproxData(result, cml)


class Test___call____NOP(tests.IrisTest):
    def setUp(self):
        # The destination grid points are exactly the same as the
        # src grid points.
        self.src = realistic_4d()[:5, :2, ::40, ::30]
        self.grid = self.src.copy()

    def test_nop__linear(self):
        regridder = Regridder(self.src, self.grid, 'linear', 'mask')
        result = regridder(self.src)
        self.assertEqual(result, self.src)

    def test_nop__nearest(self):
        regridder = Regridder(self.src, self.grid, 'nearest', 'mask')
        result = regridder(self.src)
        self.assertEqual(result, self.src)


@tests.skip_data
class Test___call____circular(tests.IrisTest):
    def setUp(self):
        src = global_pp()[::10, ::10]
        level_height = AuxCoord(0, long_name='level_height', units='m',
                                attributes={'positive': 'up'})
        sigma = AuxCoord(1, long_name='sigma')
        surface_altitude = AuxCoord((src.data - src.data.min()) * 50,
                                    'surface_altitude', units='m')
        src.add_aux_coord(level_height)
        src.add_aux_coord(sigma)
        src.add_aux_coord(surface_altitude, [0, 1])
        hybrid_height = HybridHeightFactory(level_height, sigma,
                                            surface_altitude)
        src.add_aux_factory(hybrid_height)
        self.src = src

        grid = global_pp()[:4, :4]
        grid.coord('longitude').points = grid.coord('longitude').points - 5
        self.grid = grid
        # Regridder method and extrapolation-mode.
        self.args = ('linear', 'mask')

    def test_non_circular(self):
        # Non-circular src -> non-circular grid
        regridder = Regridder(self.src, self.grid, *self.args)
        result = regridder(self.src)
        self.assertFalse(result.coord('longitude').circular)
        self.assertCMLApproxData(result, RESULT_DIR + ('non_circular.cml',))

    def test_circular_src(self):
        # Circular src -> non-circular grid
        src = self.src
        src.coord('longitude').circular = True
        regridder = Regridder(src, self.grid, *self.args)
        result = regridder(src)
        self.assertFalse(result.coord('longitude').circular)
        self.assertCMLApproxData(result, RESULT_DIR + ('circular_src.cml',))

    def test_circular_grid(self):
        # Non-circular src -> circular grid
        grid = self.grid
        grid.coord('longitude').circular = True
        regridder = Regridder(self.src, grid, *self.args)
        result = regridder(self.src)
        self.assertTrue(result.coord('longitude').circular)
        self.assertCMLApproxData(result, RESULT_DIR + ('circular_grid.cml',))

    def test_circular_src_and_grid(self):
        # Circular src -> circular grid
        src = self.src
        src.coord('longitude').circular = True
        grid = self.grid
        grid.coord('longitude').circular = True
        regridder = Regridder(src, grid, *self.args)
        result = regridder(src)
        self.assertTrue(result.coord('longitude').circular)
        self.assertCMLApproxData(result, RESULT_DIR + ('both_circular.cml',))


if __name__ == '__main__':
    tests.main()
