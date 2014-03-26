# (C) British Crown Copyright 2013 - 2014, Met Office
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
"""Unit tests for the `iris.cube.Cube` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import biggus
import mock
import numpy as np

import iris.aux_factory
import iris.coords
import iris.exceptions
from iris import FUTURE
from iris.analysis import WeightedAggregator, Aggregator
from iris.analysis import MEAN, MIN
from iris.cube import Cube
from iris.coords import AuxCoord, DimCoord
from iris.exceptions import CoordinateNotFoundError, CoordinateCollapseError
from iris.exceptions import LazyAggregatorError


class Test___init___data(tests.IrisTest):
    def test_ndarray(self):
        # np.ndarray should be allowed through
        data = np.arange(12).reshape(3, 4)
        cube = Cube(data)
        self.assertEqual(type(cube.data), np.ndarray)
        self.assertArrayEqual(cube.data, data)

    def test_masked(self):
        # np.ma.MaskedArray should be allowed through
        data = np.ma.masked_greater(np.arange(12).reshape(3, 4), 1)
        cube = Cube(data)
        self.assertEqual(type(cube.data), np.ma.MaskedArray)
        self.assertMaskedArrayEqual(cube.data, data)

    def test_matrix(self):
        # Subclasses of np.ndarray should be coerced back to np.ndarray.
        # (Except for np.ma.MaskedArray.)
        data = np.matrix([[1, 2, 3], [4, 5, 6]])
        cube = Cube(data)
        self.assertEqual(type(cube.data), np.ndarray)
        self.assertArrayEqual(cube.data, data)


class Test_xml(tests.IrisTest):
    def test_checksum_ignores_masked_values(self):
        # Mask out an single element.
        data = np.ma.arange(12).reshape(3, 4)
        data[1, 2] = np.ma.masked
        cube = Cube(data)
        self.assertCML(cube)

        # If we change the underlying value before masking it, the
        # checksum should be unaffected.
        data = np.ma.arange(12).reshape(3, 4)
        data[1, 2] = 42
        data[1, 2] = np.ma.masked
        cube = Cube(data)
        self.assertCML(cube)


class Test_collapsed__lazy(tests.IrisTest):
    def setUp(self):
        self.data = np.arange(6.0).reshape((2, 3))
        self.lazydata = biggus.NumpyArrayAdapter(self.data)
        cube = Cube(self.lazydata)
        for i_dim, name in enumerate(('y', 'x')):
            npts = cube.shape[i_dim]
            coord = DimCoord(np.arange(npts), long_name=name)
            cube.add_dim_coord(coord, i_dim)
        self.cube = cube

    def test_dim0_lazy(self):
        cube_collapsed = self.cube.collapsed('y', MEAN, lazy=True)
        self.assertTrue(cube_collapsed.has_lazy_data())
        self.assertArrayAlmostEqual(cube_collapsed.data, [1.5, 2.5, 3.5])
        self.assertFalse(cube_collapsed.has_lazy_data())

    def test_fail_dim1(self):
        # Check that MEAN produces a suitable error message for dim != 0.
        # N.B. non-lazy op can do this
        with self.assertRaises(AssertionError) as err:
            cube_collapsed = self.cube.collapsed('x', MEAN, lazy=True)

    def test_fail_multidims(self):
        # Check that MEAN produces a suitable error message for multiple dims.
        # N.B. non-lazy op can do this
        with self.assertRaises(AssertionError) as err:
            cube_collapsed = self.cube.collapsed(('x', 'y'), MEAN, lazy=True)

    def test_fail_no_lazy(self):
        dummy_agg = Aggregator('custom_op', lambda x: 1)
        with self.assertRaises(LazyAggregatorError) as err:
            cube_collapsed = self.cube.collapsed('x', dummy_agg, lazy=True)
        msg = err.exception.message
        self.assertIn('custom_op', msg)
        self.assertIn('lazy', msg)
        self.assertIn('not support', msg)


class Test_collapsed__warning(tests.IrisTest):
    def setUp(self):
        self.cube = Cube([[1, 2], [1, 2]])
        lat = DimCoord([1, 2], standard_name='latitude')
        lon = DimCoord([1, 2], standard_name='longitude')
        grid_lat = AuxCoord([1, 2], standard_name='grid_latitude')
        grid_lon = AuxCoord([1, 2], standard_name='grid_longitude')
        wibble = AuxCoord([1, 2], long_name='wibble')

        self.cube.add_dim_coord(lat, 0)
        self.cube.add_dim_coord(lon, 1)
        self.cube.add_aux_coord(grid_lat, 0)
        self.cube.add_aux_coord(grid_lon, 1)
        self.cube.add_aux_coord(wibble, 1)

    def _aggregator(self, uses_weighting):
        # Returns a mock aggregator with a mocked method (uses_weighting)
        # which returns the given True/False condition.
        aggregator = mock.Mock(spec=WeightedAggregator)
        aggregator.cell_method = None
        aggregator.uses_weighting = mock.Mock(return_value=uses_weighting)

        return aggregator

    def _assert_warn_collapse_without_weight(self, coords, warn):
        # Ensure that warning is raised.
        msg = "Collapsing spatial coordinate {!r} without weighting"
        for coord in coords:
            self.assertIn(mock.call(msg.format(coord)), warn.call_args_list)

    def _assert_nowarn_collapse_without_weight(self, coords, warn):
        # Ensure that warning is not rised.
        msg = "Collapsing spatial coordinate {!r} without weighting"
        for coord in coords:
            self.assertNotIn(mock.call(msg.format(coord)), warn.call_args_list)

    def test_lat_lon_noweighted_aggregator(self):
        # Collapse latitude coordinate with unweighted aggregator.
        aggregator = mock.Mock(spec=Aggregator)
        aggregator.cell_method = None
        coords = ['latitude', 'longitude']

        with mock.patch('warnings.warn') as warn:
            self.cube.collapsed(coords, aggregator, somekeyword='bla')

        self._assert_nowarn_collapse_without_weight(coords, warn)

    def test_lat_lon_weighted_aggregator(self):
        # Collapse latitude coordinate with weighted aggregator without
        # providing weights.
        aggregator = self._aggregator(False)
        coords = ['latitude', 'longitude']

        with mock.patch('warnings.warn') as warn:
            self.cube.collapsed(coords, aggregator)

        coords = filter(lambda coord: 'latitude' in coord, coords)
        self._assert_warn_collapse_without_weight(coords, warn)

    def test_lat_lon_weighted_aggregator_with_weights(self):
        # Collapse latitude coordinate with a weighted aggregators and
        # providing suitable weights.
        weights = np.array([[0.1, 0.5], [0.3, 0.2]])
        aggregator = self._aggregator(True)
        coords = ['latitude', 'longitude']

        with mock.patch('warnings.warn') as warn:
            self.cube.collapsed(coords, aggregator, weights=weights)

        self._assert_nowarn_collapse_without_weight(coords, warn)

    def test_lat_lon_weighted_aggregator_alt(self):
        # Collapse grid_latitude coordinate with weighted aggregator without
        # providing weights.  Tests coordinate matching logic.
        aggregator = self._aggregator(False)
        coords = ['grid_latitude', 'grid_longitude']

        with mock.patch('warnings.warn') as warn:
            self.cube.collapsed(coords, aggregator)

        coords = filter(lambda coord: 'latitude' in coord, coords)
        self._assert_warn_collapse_without_weight(coords, warn)

    def test_no_lat_weighted_aggregator_mixed(self):
        # Collapse grid_latitude and an unmatched coordinate (not lat/lon)
        # with weighted aggregator without providing weights.
        # Tests coordinate matching logic.
        aggregator = self._aggregator(False)
        coords = ['wibble']

        with mock.patch('warnings.warn') as warn:
            self.cube.collapsed(coords, aggregator)

        self._assert_nowarn_collapse_without_weight(coords, warn)


class Test_summary(tests.IrisTest):
    def test_cell_datetime_objects(self):
        # Check the scalar coordinate summary still works even when
        # iris.FUTURE.cell_datetime_objects is True.
        cube = Cube(0)
        cube.add_aux_coord(AuxCoord(42, units='hours since epoch'))
        with FUTURE.context(cell_datetime_objects=True):
            summary = cube.summary()
        self.assertIn('1970-01-02 18:00:00', summary)


class Test_is_compatible(tests.IrisTest):
    def setUp(self):
        self.test_cube = Cube([1.])
        self.other_cube = self.test_cube.copy()

    def test_noncommon_array_attrs_compatible(self):
        # Non-common array attributes should be ok.
        self.test_cube.attributes['array_test'] = np.array([1.0, 2, 3])
        self.assertTrue(self.test_cube.is_compatible(self.other_cube))

    def test_matching_array_attrs_compatible(self):
        # Matching array attributes should be ok.
        self.test_cube.attributes['array_test'] = np.array([1.0, 2, 3])
        self.other_cube.attributes['array_test'] = np.array([1.0, 2, 3])
        self.assertTrue(self.test_cube.is_compatible(self.other_cube))

    def test_different_array_attrs_incompatible(self):
        # Differing array attributes should make the cubes incompatible.
        self.test_cube.attributes['array_test'] = np.array([1.0, 2, 3])
        self.other_cube.attributes['array_test'] = np.array([1.0, 2, 777.7])
        self.assertFalse(self.test_cube.is_compatible(self.other_cube))


class Test_aggregated_by(tests.IrisTest):
    def setUp(self):
        self.cube = Cube(np.arange(11))
        val_coord = AuxCoord([0, 0, 0, 1, 1, 2, 0, 0, 2, 0, 1],
                             long_name="val")
        label_coord = AuxCoord(['alpha', 'alpha', 'beta',
                                'beta', 'alpha', 'gamma',
                                'alpha', 'alpha', 'alpha',
                                'gamma', 'beta'],
                               long_name='label', units='no_unit')
        self.cube.add_aux_coord(val_coord, 0)
        self.cube.add_aux_coord(label_coord, 0)
        self.mock_agg = mock.Mock(spec=Aggregator)
        self.mock_agg.aggregate = mock.Mock(
            return_value=mock.Mock(dtype='object'))

    def test_string_coord_agg_by_label(self):
        # Aggregate a cube on a string coordinate label where label
        # and val entries are not in step; the resulting cube has a val
        # coord of bounded cells and a label coord of single string entries.
        res_cube = self.cube.aggregated_by('label', self.mock_agg)
        val_coord = AuxCoord(np.array([1., 0.5, 1.]),
                             bounds=np.array([[0, 2], [0, 1], [2, 0]]),
                             long_name='val')
        label_coord = AuxCoord(np.array(['alpha', 'beta', 'gamma']),
                               long_name='label', units='no_unit')
        self.assertEqual(res_cube.coord('val'), val_coord)
        self.assertEqual(res_cube.coord('label'), label_coord)

    def test_string_coord_agg_by_val(self):
        # Aggregate a cube on a numeric coordinate val where label
        # and val entries are not in step; the resulting cube has a label
        # coord with serialised labels from the aggregated cells.
        res_cube = self.cube.aggregated_by('val', self.mock_agg)
        val_coord = AuxCoord(np.array([0,  1,  2]), long_name='val')
        exp0 = 'alpha|alpha|beta|alpha|alpha|gamma'
        exp1 = 'beta|alpha|beta'
        exp2 = 'gamma|alpha'
        label_coord = AuxCoord(np.array((exp0, exp1, exp2)),
                               long_name='label', units='no_unit')
        self.assertEqual(res_cube.coord('val'), val_coord)
        self.assertEqual(res_cube.coord('label'), label_coord)


def create_cube(lon_min, lon_max, bounds=False):
    n_lons = max(lon_min, lon_max) - min(lon_max, lon_min)
    data = np.arange(4 * 3 * n_lons, dtype='f4').reshape(4, 3, n_lons)
    data = biggus.NumpyArrayAdapter(data)
    cube = Cube(data, standard_name='x_wind', units='ms-1')
    cube.add_dim_coord(iris.coords.DimCoord([0, 20, 40, 80],
                                            long_name='level_height',
                                            units='m'), 0)
    cube.add_aux_coord(iris.coords.AuxCoord([1.0, 0.9, 0.8, 0.6],
                                            long_name='sigma'), 0)
    cube.add_dim_coord(iris.coords.DimCoord([-45, 0, 45], 'latitude',
                                            units='degrees'), 1)
    step = 1 if lon_max > lon_min else -1
    cube.add_dim_coord(iris.coords.DimCoord(np.arange(lon_min, lon_max, step),
                                            'longitude', units='degrees'), 2)
    if bounds:
        cube.coord('longitude').guess_bounds()
    cube.add_aux_coord(iris.coords.AuxCoord(
        np.arange(3 * n_lons).reshape(3, n_lons) * 10, 'surface_altitude',
        units='m'), [1, 2])
    cube.add_aux_factory(iris.aux_factory.HybridHeightFactory(
        cube.coord('level_height'), cube.coord('sigma'),
        cube.coord('surface_altitude')))
    return cube


# Ensure all the other coordinates and factories are correctly preserved.
class Test_intersection__Metadata(tests.IrisTest):
    def test_metadata(self):
        cube = create_cube(0, 360)
        result = cube.intersection(longitude=(170, 190))
        self.assertCMLApproxData(result)

    def test_metadata_wrapped(self):
        cube = create_cube(-180, 180)
        result = cube.intersection(longitude=(170, 190))
        self.assertCMLApproxData(result)


# Check the various error conditions.
class Test_intersection__Invalid(tests.IrisTest):
    def test_reversed_min_max(self):
        cube = create_cube(0, 360)
        with self.assertRaises(ValueError):
            cube.intersection(longitude=(30, 10))

    def test_dest_too_large(self):
        cube = create_cube(0, 360)
        with self.assertRaises(ValueError):
            cube.intersection(longitude=(30, 500))

    def test_src_too_large(self):
        cube = create_cube(0, 400)
        with self.assertRaises(ValueError):
            cube.intersection(longitude=(10, 30))

    def test_missing_coord(self):
        cube = create_cube(0, 360)
        with self.assertRaises(iris.exceptions.CoordinateNotFoundError):
            cube.intersection(parrots=(10, 30))

    def test_multi_dim_coord(self):
        cube = create_cube(0, 360)
        with self.assertRaises(iris.exceptions.CoordinateMultiDimError):
            cube.intersection(surface_altitude=(10, 30))

    def test_null_region(self):
        # 10 <= v < 10
        cube = create_cube(0, 360)
        with self.assertRaises(IndexError):
            cube.intersection(longitude=(10, 10, False, False))


class Test_intersection__Lazy(tests.IrisTest):
    def test_real_data(self):
        cube = create_cube(0, 360)
        cube.data
        result = cube.intersection(longitude=(170, 190))
        self.assertFalse(result.has_lazy_data())
        self.assertArrayEqual(result.coord('longitude').points,
                              range(170, 191))
        self.assertEqual(result.data[0, 0, 0], 170)
        self.assertEqual(result.data[0, 0, -1], 190)

    def test_real_data_wrapped(self):
        cube = create_cube(-180, 180)
        cube.data
        result = cube.intersection(longitude=(170, 190))
        self.assertFalse(result.has_lazy_data())
        self.assertArrayEqual(result.coord('longitude').points,
                              range(170, 191))
        self.assertEqual(result.data[0, 0, 0], 350)
        self.assertEqual(result.data[0, 0, -1], 10)

    def test_lazy_data(self):
        cube = create_cube(0, 360)
        result = cube.intersection(longitude=(170, 190))
        self.assertTrue(result.has_lazy_data())
        self.assertArrayEqual(result.coord('longitude').points,
                              range(170, 191))
        self.assertEqual(result.data[0, 0, 0], 170)
        self.assertEqual(result.data[0, 0, -1], 190)

    def test_lazy_data_wrapped(self):
        cube = create_cube(-180, 180)
        result = cube.intersection(longitude=(170, 190))
        self.assertTrue(result.has_lazy_data())
        self.assertArrayEqual(result.coord('longitude').points,
                              range(170, 191))
        self.assertEqual(result.data[0, 0, 0], 350)
        self.assertEqual(result.data[0, 0, -1], 10)


# Check what happens with a regional, points-only circular intersection
# coordinate.
class Test_intersection__RegionalSrcModulus(tests.IrisTest):
    def test_request_subset(self):
        cube = create_cube(40, 60)
        result = cube.intersection(longitude=(45, 50))
        self.assertArrayEqual(result.coord('longitude').points, range(45, 51))
        self.assertArrayEqual(result.data[0, 0], range(5, 11))

    def test_request_left(self):
        cube = create_cube(40, 60)
        result = cube.intersection(longitude=(35, 45))
        self.assertArrayEqual(result.coord('longitude').points, range(40, 46))
        self.assertArrayEqual(result.data[0, 0], range(0, 6))

    def test_request_right(self):
        cube = create_cube(40, 60)
        result = cube.intersection(longitude=(55, 65))
        self.assertArrayEqual(result.coord('longitude').points, range(55, 60))
        self.assertArrayEqual(result.data[0, 0], range(15, 20))

    def test_request_superset(self):
        cube = create_cube(40, 60)
        result = cube.intersection(longitude=(35, 65))
        self.assertArrayEqual(result.coord('longitude').points, range(40, 60))
        self.assertArrayEqual(result.data[0, 0], range(0, 20))

    def test_request_subset_modulus(self):
        cube = create_cube(40, 60)
        result = cube.intersection(longitude=(45 + 360, 50 + 360))
        self.assertArrayEqual(result.coord('longitude').points,
                              range(45 + 360, 51 + 360))
        self.assertArrayEqual(result.data[0, 0], range(5, 11))

    def test_request_left_modulus(self):
        cube = create_cube(40, 60)
        result = cube.intersection(longitude=(35 + 360, 45 + 360))
        self.assertArrayEqual(result.coord('longitude').points,
                              range(40 + 360, 46 + 360))
        self.assertArrayEqual(result.data[0, 0], range(0, 6))

    def test_request_right_modulus(self):
        cube = create_cube(40, 60)
        result = cube.intersection(longitude=(55 + 360, 65 + 360))
        self.assertArrayEqual(result.coord('longitude').points,
                              range(55 + 360, 60 + 360))
        self.assertArrayEqual(result.data[0, 0], range(15, 20))

    def test_request_superset_modulus(self):
        cube = create_cube(40, 60)
        result = cube.intersection(longitude=(35 + 360, 65 + 360))
        self.assertArrayEqual(result.coord('longitude').points,
                              range(40 + 360, 60 + 360))
        self.assertArrayEqual(result.data[0, 0], range(0, 20))

    def test_tolerance_f4(self):
        cube = create_cube(0, 5)
        cube.coord('longitude').points = np.array([0., 3.74999905, 7.49999809,
                                                   11.24999714, 14.99999619],
                                                  dtype='f4')
        result = cube.intersection(longitude=(0, 5))

    def test_tolerance_f8(self):
        cube = create_cube(0, 5)
        cube.coord('longitude').points = np.array([0., 3.74999905, 7.49999809,
                                                   11.24999714, 14.99999619],
                                                  dtype='f8')
        result = cube.intersection(longitude=(0, 5))


# Check what happens with a global, points-only circular intersection
# coordinate.
class Test_intersection__GlobalSrcModulus(tests.IrisTest):
    def test_global(self):
        cube = create_cube(0, 360)
        result = cube.intersection(longitude=(0, 360))
        self.assertEqual(result.coord('longitude').points[0], 0)
        self.assertEqual(result.coord('longitude').points[-1], 359)
        self.assertEqual(result.data[0, 0, 0], 0)
        self.assertEqual(result.data[0, 0, -1], 359)

    def test_global_wrapped(self):
        cube = create_cube(0, 360)
        result = cube.intersection(longitude=(-180, 180))
        self.assertEqual(result.coord('longitude').points[0], -180)
        self.assertEqual(result.coord('longitude').points[-1], 179)
        self.assertEqual(result.data[0, 0, 0], 180)
        self.assertEqual(result.data[0, 0, -1], 179)

    def test_aux_coord(self):
        cube = create_cube(0, 360)
        cube.replace_coord(iris.coords.AuxCoord.from_coord(
            cube.coord('longitude')))
        result = cube.intersection(longitude=(0, 360))
        self.assertEqual(result.coord('longitude').points[0], 0)
        self.assertEqual(result.coord('longitude').points[-1], 359)
        self.assertEqual(result.data[0, 0, 0], 0)
        self.assertEqual(result.data[0, 0, -1], 359)

    def test_aux_coord_wrapped(self):
        cube = create_cube(0, 360)
        cube.replace_coord(iris.coords.AuxCoord.from_coord(
            cube.coord('longitude')))
        result = cube.intersection(longitude=(-180, 180))
        self.assertEqual(result.coord('longitude').points[0], 0)
        self.assertEqual(result.coord('longitude').points[-1], -1)
        self.assertEqual(result.data[0, 0, 0], 0)
        self.assertEqual(result.data[0, 0, -1], 359)

    def test_aux_coord_non_contiguous_wrapped(self):
        cube = create_cube(0, 360)
        coord = iris.coords.AuxCoord.from_coord(cube.coord('longitude'))
        coord.points = (coord.points * 1.5) % 360
        cube.replace_coord(coord)
        result = cube.intersection(longitude=(-90, 90))
        self.assertEqual(result.coord('longitude').points[0], 0)
        self.assertEqual(result.coord('longitude').points[-1], 90)
        self.assertEqual(result.data[0, 0, 0], 0)
        self.assertEqual(result.data[0, 0, -1], 300)

    def test_decrementing(self):
        cube = create_cube(360, 0)
        result = cube.intersection(longitude=(40, 60))
        self.assertEqual(result.coord('longitude').points[0], 60)
        self.assertEqual(result.coord('longitude').points[-1], 40)
        self.assertEqual(result.data[0, 0, 0], 300)
        self.assertEqual(result.data[0, 0, -1], 320)

    def test_decrementing_wrapped(self):
        cube = create_cube(360, 0)
        result = cube.intersection(longitude=(-10, 10))
        self.assertEqual(result.coord('longitude').points[0], 10)
        self.assertEqual(result.coord('longitude').points[-1], -10)
        self.assertEqual(result.data[0, 0, 0], 350)
        self.assertEqual(result.data[0, 0, -1], 10)

    def test_no_wrap_after_modulus(self):
        cube = create_cube(0, 360)
        result = cube.intersection(longitude=(170 + 360, 190 + 360))
        self.assertEqual(result.coord('longitude').points[0], 170 + 360)
        self.assertEqual(result.coord('longitude').points[-1], 190 + 360)
        self.assertEqual(result.data[0, 0, 0], 170)
        self.assertEqual(result.data[0, 0, -1], 190)

    def test_wrap_after_modulus(self):
        cube = create_cube(-180, 180)
        result = cube.intersection(longitude=(170 + 360, 190 + 360))
        self.assertEqual(result.coord('longitude').points[0], 170 + 360)
        self.assertEqual(result.coord('longitude').points[-1], 190 + 360)
        self.assertEqual(result.data[0, 0, 0], 350)
        self.assertEqual(result.data[0, 0, -1], 10)

    def test_select_by_coord(self):
        cube = create_cube(0, 360)
        coord = iris.coords.DimCoord(0, 'longitude', units='degrees')
        result = cube.intersection(iris.coords.CoordExtent(coord, 10, 30))
        self.assertEqual(result.coord('longitude').points[0], 10)
        self.assertEqual(result.coord('longitude').points[-1], 30)
        self.assertEqual(result.data[0, 0, 0], 10)
        self.assertEqual(result.data[0, 0, -1], 30)

    def test_inclusive_exclusive(self):
        cube = create_cube(0, 360)
        result = cube.intersection(longitude=(170, 190, True, False))
        self.assertEqual(result.coord('longitude').points[0], 170)
        self.assertEqual(result.coord('longitude').points[-1], 189)
        self.assertEqual(result.data[0, 0, 0], 170)
        self.assertEqual(result.data[0, 0, -1], 189)

    def test_exclusive_inclusive(self):
        cube = create_cube(0, 360)
        result = cube.intersection(longitude=(170, 190, False))
        self.assertEqual(result.coord('longitude').points[0], 171)
        self.assertEqual(result.coord('longitude').points[-1], 190)
        self.assertEqual(result.data[0, 0, 0], 171)
        self.assertEqual(result.data[0, 0, -1], 190)

    def test_exclusive_exclusive(self):
        cube = create_cube(0, 360)
        result = cube.intersection(longitude=(170, 190, False, False))
        self.assertEqual(result.coord('longitude').points[0], 171)
        self.assertEqual(result.coord('longitude').points[-1], 189)
        self.assertEqual(result.data[0, 0, 0], 171)
        self.assertEqual(result.data[0, 0, -1], 189)

    def test_single_point(self):
        # 10 <= v <= 10
        cube = create_cube(0, 360)
        result = cube.intersection(longitude=(10, 10))
        self.assertEqual(result.coord('longitude').points[0], 10)
        self.assertEqual(result.coord('longitude').points[-1], 10)
        self.assertEqual(result.data[0, 0, 0], 10)
        self.assertEqual(result.data[0, 0, -1], 10)

    def test_wrap_radians(self):
        cube = create_cube(0, 360)
        cube.coord('longitude').convert_units('radians')
        result = cube.intersection(longitude=(-1, 0.5))
        self.assertEqual(result.coord('longitude').points[0],
                         -0.99483767363676634)
        self.assertEqual(result.coord('longitude').points[-1],
                         0.48869219055841207)
        self.assertEqual(result.data[0, 0, 0], 303)
        self.assertEqual(result.data[0, 0, -1], 28)


# Check what happens with a global, points-and-bounds circular
# intersection coordinate.
class Test_intersection__ModulusBounds(tests.IrisTest):
    def test_misaligned_points_inside(self):
        cube = create_cube(0, 360, bounds=True)
        result = cube.intersection(longitude=(169.75, 190.25))
        self.assertArrayEqual(result.coord('longitude').bounds[0],
                              [169.5, 170.5])
        self.assertArrayEqual(result.coord('longitude').bounds[-1],
                              [189.5, 190.5])
        self.assertEqual(result.data[0, 0, 0], 170)
        self.assertEqual(result.data[0, 0, -1], 190)

    def test_misaligned_points_outside(self):
        cube = create_cube(0, 360, bounds=True)
        result = cube.intersection(longitude=(170.25, 189.75))
        self.assertArrayEqual(result.coord('longitude').bounds[0],
                              [169.5, 170.5])
        self.assertArrayEqual(result.coord('longitude').bounds[-1],
                              [189.5, 190.5])
        self.assertEqual(result.data[0, 0, 0], 170)
        self.assertEqual(result.data[0, 0, -1], 190)

    def test_aligned_inclusive(self):
        cube = create_cube(0, 360, bounds=True)
        result = cube.intersection(longitude=(170.5, 189.5))
        self.assertArrayEqual(result.coord('longitude').bounds[0],
                              [169.5, 170.5])
        self.assertArrayEqual(result.coord('longitude').bounds[-1],
                              [189.5, 190.5])
        self.assertEqual(result.data[0, 0, 0], 170)
        self.assertEqual(result.data[0, 0, -1], 190)

    def test_aligned_exclusive(self):
        cube = create_cube(0, 360, bounds=True)
        result = cube.intersection(longitude=(170.5, 189.5, False, False))
        self.assertArrayEqual(result.coord('longitude').bounds[0],
                              [170.5, 171.5])
        self.assertArrayEqual(result.coord('longitude').bounds[-1],
                              [188.5, 189.5])
        self.assertEqual(result.data[0, 0, 0], 171)
        self.assertEqual(result.data[0, 0, -1], 189)


def unrolled_cube():
    data = np.arange(5, dtype='f4')
    cube = Cube(data)
    cube.add_aux_coord(iris.coords.AuxCoord([5.0, 10.0, 8.0, 5.0, 3.0],
                                            'longitude', units='degrees'), 0)
    cube.add_aux_coord(iris.coords.AuxCoord([1.0, 3.0, -2.0, -1.0, -4.0],
                                            'latitude'), 0)
    return cube


# Check what happens with a "unrolled" scatter-point data with a circular
# intersection coordinate.
class Test_intersection__ScatterModulus(tests.IrisTest):
    def test_subset(self):
        cube = unrolled_cube()
        result = cube.intersection(longitude=(5, 8))
        self.assertArrayEqual(result.coord('longitude').points, [5, 8, 5])
        self.assertArrayEqual(result.data, [0, 2, 3])

    def test_subset_wrapped(self):
        cube = unrolled_cube()
        result = cube.intersection(longitude=(5 + 360, 8 + 360))
        self.assertArrayEqual(result.coord('longitude').points,
                              [365, 368, 365])
        self.assertArrayEqual(result.data, [0, 2, 3])

    def test_superset(self):
        cube = unrolled_cube()
        result = cube.intersection(longitude=(0, 15))
        self.assertArrayEqual(result.coord('longitude').points,
                              [5, 10, 8, 5, 3])
        self.assertArrayEqual(result.data, range(5))


if __name__ == "__main__":
    tests.main()
