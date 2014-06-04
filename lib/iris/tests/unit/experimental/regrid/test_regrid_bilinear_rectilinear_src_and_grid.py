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
Unit tests for
:func:`iris.experimental.regrid.regrid_bilinear_rectilinear_src_and_grid`.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy as np

from iris.aux_factory import HybridHeightFactory
from iris.coord_systems import GeogCS, OSGB
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.experimental.regrid import \
    regrid_bilinear_rectilinear_src_and_grid as regrid
from iris.tests.stock import lat_lon_cube


class TestInvalidTypes(tests.IrisTest):
    def test_src_as_array(self):
        with self.assertRaises(TypeError):
            regrid(np.zeros((3, 4)), Cube())

    def test_grid_as_array(self):
        with self.assertRaises(TypeError):
            regrid(Cube(), np.zeros((3, 4)))

    def test_src_as_int(self):
        with self.assertRaises(TypeError):
            regrid(42, Cube())

    def test_grid_as_int(self):
        with self.assertRaises(TypeError):
            regrid(Cube(), 42)


class TestMissingCoords(tests.IrisTest):
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
            regrid(bad, ok)

    def test_grid_missing_lat(self):
        ok, bad = self.ok_bad(['latitude'])
        with self.assertRaises(ValueError):
            regrid(ok, bad)

    def test_src_missing_lon(self):
        ok, bad = self.ok_bad(['longitude'])
        with self.assertRaises(ValueError):
            regrid(bad, ok)

    def test_grid_missing_lon(self):
        ok, bad = self.ok_bad(['longitude'])
        with self.assertRaises(ValueError):
            regrid(ok, bad)

    def test_src_missing_lat_lon(self):
        ok, bad = self.ok_bad(['latitude', 'longitude'])
        with self.assertRaises(ValueError):
            regrid(bad, ok)

    def test_grid_missing_lat_lon(self):
        ok, bad = self.ok_bad(['latitude', 'longitude'])
        with self.assertRaises(ValueError):
            regrid(ok, bad)


class TestNotDimCoord(tests.IrisTest):
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
            regrid(bad, ok)

    def test_grid_with_aux_lat(self):
        ok, bad = self.ok_bad('latitude')
        with self.assertRaises(ValueError):
            regrid(ok, bad)

    def test_src_with_aux_lon(self):
        ok, bad = self.ok_bad('longitude')
        with self.assertRaises(ValueError):
            regrid(bad, ok)

    def test_grid_with_aux_lon(self):
        ok, bad = self.ok_bad('longitude')
        with self.assertRaises(ValueError):
            regrid(ok, bad)


class TestNotDimCoord(tests.IrisTest):
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
            regrid(bad, ok)

    def test_grid_shares_dim(self):
        ok, bad = self.ok_bad()
        with self.assertRaises(ValueError):
            regrid(ok, bad)


class TestBadGeoreference(tests.IrisTest):
    def ok_bad(self, lat_cs, lon_cs):
        # Updates `bad` to use the given coordinate systems.
        ok = lat_lon_cube()
        bad = lat_lon_cube()
        bad.coord('latitude').coord_system = lat_cs
        bad.coord('longitude').coord_system = lon_cs
        return ok, bad

    def test_src_no_cs(self):
        ok, bad = self.ok_bad(None, None)
        with self.assertRaises(ValueError):
            regrid(bad, ok)

    def test_grid_no_cs(self):
        ok, bad = self.ok_bad(None, None)
        with self.assertRaises(ValueError):
            regrid(ok, bad)

    def test_src_one_cs(self):
        ok, bad = self.ok_bad(None, GeogCS(6371000))
        with self.assertRaises(ValueError):
            regrid(bad, ok)

    def test_grid_one_cs(self):
        ok, bad = self.ok_bad(None, GeogCS(6371000))
        with self.assertRaises(ValueError):
            regrid(ok, bad)

    def test_src_inconsistent_cs(self):
        ok, bad = self.ok_bad(GeogCS(6370000), GeogCS(6371000))
        with self.assertRaises(ValueError):
            regrid(bad, ok)

    def test_grid_inconsistent_cs(self):
        ok, bad = self.ok_bad(GeogCS(6370000), GeogCS(6371000))
        with self.assertRaises(ValueError):
            regrid(ok, bad)


class TestBadAngularUnits(tests.IrisTest):
    def ok_bad(self):
        # Changes the longitude coord to radians on `bad`.
        ok = lat_lon_cube()
        bad = lat_lon_cube()
        bad.coord('longitude').units = 'radians'
        return ok, bad

    def test_src_radians(self):
        ok, bad = self.ok_bad()
        with self.assertRaises(ValueError):
            regrid(bad, ok)

    def test_grid_radians(self):
        ok, bad = self.ok_bad()
        with self.assertRaises(ValueError):
            regrid(ok, bad)


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


class TestBadLinearUnits(tests.IrisTest):
    def ok_bad(self):
        # Defines `bad` with an x coordinate in km.
        ok = lat_lon_cube()
        bad = uk_cube()
        bad.coord(axis='x').units = 'km'
        return ok, bad

    def test_src_km(self):
        ok, bad = self.ok_bad()
        with self.assertRaises(ValueError):
            regrid(bad, ok)

    def test_grid_km(self):
        ok, bad = self.ok_bad()
        with self.assertRaises(ValueError):
            regrid(ok, bad)


class TestNoCoordSystems(tests.IrisTest):
    # Test behaviour in the absence of any coordinate systems.

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
        result = regrid(src, grid)
        for coord in result.dim_coords:
            self.assertEqual(coord, grid.coord(coord))
        expected = np.ma.arange(12).reshape((3, 4)) + 5
        expected[:, 3] = np.ma.masked
        expected[2, :] = np.ma.masked
        self.assertMaskedArrayEqual(result.data, expected)

    def test_coord_metadata_mismatch(self):
        # Check for failure when coordinate definitions differ.
        uk = uk_cube()
        self.remove_coord_systems(uk)
        lat_lon = lat_lon_cube()
        self.remove_coord_systems(lat_lon)
        with self.assertRaises(ValueError):
            regrid(uk, lat_lon)


# Check what happens to NaN values, extrapolated values, and
# masked values.
class TestModes(tests.IrisTest):
    values = [[np.nan, 6, 7, np.nan],
              [9, 10, 11, np.nan],
              [np.nan, np.nan, np.nan, np.nan]]

    linear_values = [[np.nan, 6, 7, 8],
                     [9, 10, 11, 12],
                     [13, 14, 15, 16]]

    surface_values = [[50, 60, 70, np.nan],
                      [90, 100, 110, np.nan],
                      [np.nan, np.nan, np.nan, np.nan]]

    def _ndarray_cube(self):
        src = uk_cube()
        src.data[0, 0] = np.nan
        return src

    def _masked_cube(self):
        src = uk_cube()
        src.data = np.ma.asarray(src.data)
        src.data[0, 0] = np.nan
        src.data[2, 3] = np.ma.masked
        return src

    def _regrid(self, src, extrapolation_mode=None):
        grid = src.copy()
        for coord in grid.dim_coords:
            coord.points = coord.points + 1
        kwargs = {}
        if extrapolation_mode is not None:
            kwargs['extrapolation_mode'] = extrapolation_mode
        result = regrid(src, grid, **kwargs)

        surface = result.coord('surface_altitude').points
        self.assertNotIsInstance(surface, np.ma.MaskedArray)
        self.assertArrayEqual(surface, self.surface_values)

        return result.data

    def test_default_ndarray(self):
        # NaN           -> NaN
        # Extrapolated  -> Masked
        src = self._ndarray_cube()
        result = self._regrid(src)
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
        src = self._masked_cube()
        result = self._regrid(src)
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
        src = uk_cube()
        src.data = np.ma.asarray(src.data)
        src.data[0, 0] = np.nan
        result = self._regrid(src)
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
        src = uk_cube()
        src.data = np.ma.asarray(src.data)
        # Make sure the mask has been expanded
        src.data.mask = False
        src.data[0, 0] = np.nan
        result = self._regrid(src)
        self.assertIsInstance(result, np.ma.MaskedArray)
        mask = [[0, 0, 0, 1],
                [0, 0, 0, 1],
                [1, 1, 1, 1]]
        expected = np.ma.MaskedArray(self.values, mask)
        self.assertMaskedArrayEqual(result, expected)

    def test_linear_ndarray(self):
        # NaN           -> NaN
        # Extrapolated  -> linear
        src = self._ndarray_cube()
        result = self._regrid(src, 'linear')
        self.assertNotIsInstance(result, np.ma.MaskedArray)
        self.assertArrayEqual(result, self.linear_values)

    def test_linear_maskedarray(self):
        # NaN           -> NaN
        # Extrapolated  -> linear
        # Masked        -> Masked
        src = self._masked_cube()
        result = self._regrid(src, 'linear')
        self.assertIsInstance(result, np.ma.MaskedArray)
        mask = [[0, 0, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1]]
        expected = np.ma.MaskedArray(self.linear_values, mask)
        self.assertMaskedArrayEqual(result, expected)

    def test_nan_ndarray(self):
        # NaN           -> NaN
        # Extrapolated  -> NaN
        src = self._ndarray_cube()
        result = self._regrid(src, 'nan')
        self.assertNotIsInstance(result, np.ma.MaskedArray)
        self.assertArrayEqual(result, self.values)

    def test_nan_maskedarray(self):
        # NaN           -> NaN
        # Extrapolated  -> NaN
        # Masked        -> Masked
        src = self._masked_cube()
        result = self._regrid(src, 'nan')
        self.assertIsInstance(result, np.ma.MaskedArray)
        mask = [[0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0]]
        expected = np.ma.MaskedArray(self.values, mask)
        self.assertMaskedArrayEqual(result, expected)

    def test_error_ndarray(self):
        # Values irrelevant - the function raises an error.
        src = self._ndarray_cube()
        with self.assertRaisesRegexp(ValueError, 'out of bounds'):
            self._regrid(src, 'error')

    def test_error_maskedarray(self):
        # Values irrelevant - the function raises an error.
        src = self._masked_cube()
        with self.assertRaisesRegexp(ValueError, 'out of bounds'):
            self._regrid(src, 'error')

    def test_mask_ndarray(self):
        # NaN           -> NaN
        # Extrapolated  -> Masked (this is different from all the other
        #                          modes)
        src = self._ndarray_cube()
        result = self._regrid(src, 'mask')
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
        src = self._masked_cube()
        result = self._regrid(src, 'mask')
        self.assertIsInstance(result, np.ma.MaskedArray)
        mask = [[0, 0, 0, 1],
                [0, 0, 1, 1],
                [1, 1, 1, 1]]
        expected = np.ma.MaskedArray(self.values, mask)
        self.assertMaskedArrayEqual(result, expected)

    def test_nanmask_ndarray(self):
        # NaN           -> NaN
        # Extrapolated  -> NaN
        src = self._ndarray_cube()
        result = self._regrid(src, 'nanmask')
        self.assertNotIsInstance(result, np.ma.MaskedArray)
        self.assertArrayEqual(result, self.values)

    def test_nanmask_maskedarray(self):
        # NaN           -> NaN
        # Extrapolated  -> Masked
        # Masked        -> Masked
        src = self._masked_cube()
        result = self._regrid(src, 'nanmask')
        self.assertIsInstance(result, np.ma.MaskedArray)
        mask = [[0, 0, 0, 1],
                [0, 0, 1, 1],
                [1, 1, 1, 1]]
        expected = np.ma.MaskedArray(self.values, mask)
        self.assertMaskedArrayEqual(result, expected)

    def test_invalid(self):
        src = uk_cube()
        with self.assertRaisesRegexp(ValueError, 'Invalid extrapolation mode'):
            self._regrid(src, 'BOGUS')


if __name__ == '__main__':
    tests.main()
