# (C) British Crown Copyright 2013 - 2018, Met Office
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
"""Unit tests for the :class:`iris.coords.Coord` class."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import collections

import numpy as np

import iris
from iris.coords import DimCoord, AuxCoord, Coord
from iris.tests import mock
from iris.exceptions import UnitConversionError
from iris.tests.unit.coords import CoordTestMixin


Pair = collections.namedtuple('Pair', 'points bounds')


class Test_nearest_neighbour_index__ascending(tests.IrisTest):
    def setUp(self):
        points = [0., 90., 180., 270.]
        self.coord = DimCoord(points, circular=False,
                              units='degrees')

    def _test_nearest_neighbour_index(self, target, bounds=None,
                                      circular=False):
        _bounds = [[-20, 10], [10, 100], [100, 260], [260, 340]]
        ext_pnts = [-70, -10, 110, 275, 370]
        if bounds is True:
            self.coord.bounds = _bounds
        else:
            self.coord.bounds = bounds
        self.coord.circular = circular
        results = [self.coord.nearest_neighbour_index(ind) for ind in ext_pnts]
        self.assertEqual(results, target)

    def test_nobounds(self):
        target = [0, 0, 1, 3, 3]
        self._test_nearest_neighbour_index(target)

    def test_nobounds_circular(self):
        target = [3, 0, 1, 3, 0]
        self._test_nearest_neighbour_index(target, circular=True)

    def test_bounded(self):
        target = [0, 0, 2, 3, 3]
        self._test_nearest_neighbour_index(target, bounds=True)

    def test_bounded_circular(self):
        target = [3, 0, 2, 3, 0]
        self._test_nearest_neighbour_index(target, bounds=True, circular=True)

    def test_bounded_overlapping(self):
        _bounds = [[-20, 50], [10, 150], [100, 300], [260, 340]]
        target = [0, 0, 1, 2, 3]
        self._test_nearest_neighbour_index(target, bounds=_bounds)

    def test_bounded_disjointed(self):
        _bounds = [[-20, 10], [80, 170], [180, 190], [240, 340]]
        target = [0, 0, 1, 3, 3]
        self._test_nearest_neighbour_index(target, bounds=_bounds)

    def test_scalar(self):
        self.coord = DimCoord([0], circular=False, units='degrees')
        target = [0, 0, 0, 0, 0]
        self._test_nearest_neighbour_index(target)


class Test_nearest_neighbour_index__descending(tests.IrisTest):
    def setUp(self):
        points = [270., 180., 90., 0.]
        self.coord = DimCoord(points, circular=False,
                              units='degrees')

    def _test_nearest_neighbour_index(self, target, bounds=False,
                                      circular=False):
        _bounds = [[340, 260], [260, 100], [100, 10], [10, -20]]
        ext_pnts = [-70, -10, 110, 275, 370]
        if bounds:
            self.coord.bounds = _bounds
        self.coord.circular = circular
        results = [self.coord.nearest_neighbour_index(ind) for ind in ext_pnts]
        self.assertEqual(results, target)

    def test_nobounds(self):
        target = [3, 3, 2, 0, 0]
        self._test_nearest_neighbour_index(target)

    def test_nobounds_circular(self):
        target = [0, 3, 2, 0, 3]
        self._test_nearest_neighbour_index(target, circular=True)

    def test_bounded(self):
        target = [3, 3, 1, 0, 0]
        self._test_nearest_neighbour_index(target, bounds=True)

    def test_bounded_circular(self):
        target = [0, 3, 1, 0, 3]
        self._test_nearest_neighbour_index(target, bounds=True, circular=True)


class Test_guess_bounds(tests.IrisTest):
    def setUp(self):
        self.coord = DimCoord(np.array([-160, -120, 0, 30, 150, 170]),
                              units='degree', standard_name='longitude',
                              circular=True)

    def test_non_circular(self):
        self.coord.circular = False
        self.coord.guess_bounds()
        target = np.array([[-180., -140.], [-140., -60.], [-60., 15.],
                           [15., 90.], [90., 160.], [160., 180.]])
        self.assertArrayEqual(target, self.coord.bounds)

    def test_circular_increasing(self):
        self.coord.guess_bounds()
        target = np.array([[-175., -140.], [-140., -60.], [-60., 15.],
                           [15., 90.], [90., 160.], [160., 185.]])
        self.assertArrayEqual(target, self.coord.bounds)

    def test_circular_decreasing(self):
        self.coord.points = self.coord.points[::-1]
        self.coord.guess_bounds()
        target = np.array([[185., 160.], [160., 90.], [90., 15.],
                           [15., -60.], [-60., -140.], [-140., -175.]])
        self.assertArrayEqual(target, self.coord.bounds)

    def test_circular_increasing_alt_range(self):
        self.coord.points = np.array([10, 30, 90, 150, 210, 220])
        self.coord.guess_bounds()
        target = np.array([[-65., 20.], [20., 60.], [60., 120.],
                           [120., 180.], [180., 215.], [215., 295.]])
        self.assertArrayEqual(target, self.coord.bounds)

    def test_circular_decreasing_alt_range(self):
        self.coord.points = np.array([10, 30, 90, 150, 210, 220])[::-1]
        self.coord.guess_bounds()
        target = np.array([[295, 215], [215, 180], [180, 120], [120, 60],
                           [60, 20], [20, -65]])
        self.assertArrayEqual(target, self.coord.bounds)


class Test_guess_bounds__default_enabled_latitude_clipping(tests.IrisTest):
    def test_all_inside(self):
        lat = DimCoord([-10, 0, 20], units='degree', standard_name='latitude')
        lat.guess_bounds()
        self.assertArrayEqual(lat.bounds, [[-15, -5], [-5, 10], [10, 30]])

    def test_points_inside_bounds_outside(self):
        lat = DimCoord([-80, 0, 70], units='degree', standard_name='latitude')
        lat.guess_bounds()
        self.assertArrayEqual(lat.bounds, [[-90, -40], [-40, 35], [35, 90]])

    def test_points_inside_bounds_outside_grid_latitude(self):
        lat = DimCoord([-80, 0, 70], units='degree',
                       standard_name='grid_latitude')
        lat.guess_bounds()
        self.assertArrayEqual(lat.bounds, [[-90, -40], [-40, 35], [35, 90]])

    def test_points_to_edges_bounds_outside(self):
        lat = DimCoord([-90, 0, 90], units='degree', standard_name='latitude')
        lat.guess_bounds()
        self.assertArrayEqual(lat.bounds, [[-90, -45], [-45, 45], [45, 90]])

    def test_points_outside(self):
        lat = DimCoord([-100, 0, 120], units='degree',
                       standard_name='latitude')
        lat.guess_bounds()
        self.assertArrayEqual(lat.bounds, [[-150, -50], [-50, 60], [60, 180]])

    def test_points_inside_bounds_outside_wrong_unit(self):
        lat = DimCoord([-80, 0, 70], units='feet', standard_name='latitude')
        lat.guess_bounds()
        self.assertArrayEqual(lat.bounds, [[-120, -40], [-40, 35], [35, 105]])

    def test_points_inside_bounds_outside_wrong_name(self):
        lat = DimCoord([-80, 0, 70], units='degree', standard_name='longitude')
        lat.guess_bounds()
        self.assertArrayEqual(lat.bounds, [[-120, -40], [-40, 35], [35, 105]])

    def test_points_inside_bounds_outside_wrong_name_2(self):
        lat = DimCoord([-80, 0, 70], units='degree',
                       long_name='other_latitude')
        lat.guess_bounds()
        self.assertArrayEqual(lat.bounds, [[-120, -40], [-40, 35], [35, 105]])


class Test_cell(tests.IrisTest):
    def _mock_coord(self):
        coord = mock.Mock(spec=Coord, ndim=1,
                          points=np.array([mock.sentinel.time]),
                          bounds=np.array([[mock.sentinel.lower,
                                            mock.sentinel.upper]]))
        return coord

    def test_time_as_object(self):
        # Ensure Coord.cell() converts the point/bound values to
        # "datetime" objects.
        coord = self._mock_coord()
        coord.units.num2date = mock.Mock(
            side_effect=[mock.sentinel.datetime,
                         (mock.sentinel.datetime_lower,
                          mock.sentinel.datetime_upper)])
        cell = Coord.cell(coord, 0)
        self.assertIs(cell.point, mock.sentinel.datetime)
        self.assertEqual(cell.bound,
                         (mock.sentinel.datetime_lower,
                          mock.sentinel.datetime_upper))
        self.assertEqual(coord.units.num2date.call_args_list,
                         [mock.call((mock.sentinel.time,)),
                          mock.call((mock.sentinel.lower,
                                     mock.sentinel.upper))])


class Test_collapsed(tests.IrisTest, CoordTestMixin):

    def test_serialize(self):
        # Collapse a string AuxCoord, causing it to be serialised.
        string = Pair(np.array(['two', 'four', 'six', 'eight']),
                      np.array([['one', 'three'],
                                ['three', 'five'],
                                ['five', 'seven'],
                                ['seven', 'nine']]))
        string_nobounds = Pair(np.array(['ecks', 'why', 'zed']),
                               None)
        string_multi = Pair(np.array(['three', 'six', 'nine']),
                            np.array([['one', 'two', 'four', 'five'],
                                      ['four', 'five', 'seven', 'eight'],
                                      ['seven', 'eight', 'ten', 'eleven']]))

        def _serialize(data):
            return '|'.join(str(item) for item in data.flatten())

        for units in ['unknown', 'no_unit']:
            for points, bounds in [string, string_nobounds, string_multi]:
                coord = AuxCoord(points=points, bounds=bounds, units=units)
                collapsed_coord = coord.collapsed()
                self.assertArrayEqual(collapsed_coord.points,
                                      _serialize(points))
                if bounds is not None:
                    for index in np.ndindex(bounds.shape[1:]):
                        index_slice = (slice(None),) + tuple(index)
                        self.assertArrayEqual(
                            collapsed_coord.bounds[index_slice],
                            _serialize(bounds[index_slice]))

    def test_dim_1d(self):
        # Numeric coords should not be serialised.
        coord = DimCoord(points=np.array([2, 4, 6, 8]),
                         bounds=np.array([[1, 3], [3, 5], [5, 7], [7, 9]]))
        for units in ['unknown', 'no_unit', 1, 'K']:
            coord.units = units
            collapsed_coord = coord.collapsed()
            self.assertArrayEqual(collapsed_coord.points,
                                  np.mean(coord.points))
            self.assertArrayEqual(collapsed_coord.bounds,
                                  [[coord.bounds.min(), coord.bounds.max()]])

    def test_numeric_nd(self):
        coord = AuxCoord(points=np.array([[1, 2, 4, 5],
                                          [4, 5, 7, 8],
                                          [7, 8, 10, 11]]))

        collapsed_coord = coord.collapsed()
        self.assertArrayEqual(collapsed_coord.points, np.array([6]))
        self.assertArrayEqual(collapsed_coord.bounds, np.array([[1, 11]]))

        # Test partially collapsing one dimension...
        collapsed_coord = coord.collapsed(1)
        self.assertArrayEqual(collapsed_coord.points, np.array([3.,  6.,  9.]))
        self.assertArrayEqual(collapsed_coord.bounds, np.array([[1,  5],
                                                                [4,  8],
                                                                [7, 11]]))

        # ... and the other
        collapsed_coord = coord.collapsed(0)
        self.assertArrayEqual(collapsed_coord.points, np.array([4,  5,  7, 8]))
        self.assertArrayEqual(collapsed_coord.bounds, np.array([[1,  7],
                                                                [2,  8],
                                                                [4, 10],
                                                                [5, 11]]))

    def test_lazy_nd_bounds(self):
        import dask.array as da

        self.setupTestArrays((3, 4))
        coord = AuxCoord(self.pts_real, bounds=self.bds_lazy)

        collapsed_coord = coord.collapsed()

        # Note that the new points get recalculated from the lazy bounds
        #  and so end up as lazy
        self.assertTrue(collapsed_coord.has_lazy_points())
        self.assertTrue(collapsed_coord.has_lazy_bounds())

        self.assertArrayEqual(collapsed_coord.points, np.array([55]))
        self.assertArrayEqual(collapsed_coord.bounds, da.array([[-2, 112]]))

    def test_lazy_nd_points_and_bounds(self):
        import dask.array as da

        self.setupTestArrays((3, 4))
        coord = AuxCoord(self.pts_lazy, bounds=self.bds_lazy)

        collapsed_coord = coord.collapsed()

        self.assertTrue(collapsed_coord.has_lazy_points())
        self.assertTrue(collapsed_coord.has_lazy_bounds())

        self.assertArrayEqual(collapsed_coord.points, da.array([55]))
        self.assertArrayEqual(collapsed_coord.bounds, da.array([[-2, 112]]))


class Test_is_compatible(tests.IrisTest):
    def setUp(self):
        self.test_coord = AuxCoord([1.])
        self.other_coord = self.test_coord.copy()

    def test_noncommon_array_attrs_compatible(self):
        # Non-common array attributes should be ok.
        self.test_coord.attributes['array_test'] = np.array([1.0, 2, 3])
        self.assertTrue(self.test_coord.is_compatible(self.other_coord))

    def test_matching_array_attrs_compatible(self):
        # Matching array attributes should be ok.
        self.test_coord.attributes['array_test'] = np.array([1.0, 2, 3])
        self.other_coord.attributes['array_test'] = np.array([1.0, 2, 3])
        self.assertTrue(self.test_coord.is_compatible(self.other_coord))

    def test_different_array_attrs_incompatible(self):
        # Differing array attributes should make coords incompatible.
        self.test_coord.attributes['array_test'] = np.array([1.0, 2, 3])
        self.other_coord.attributes['array_test'] = np.array([1.0, 2, 777.7])
        self.assertFalse(self.test_coord.is_compatible(self.other_coord))


class Test_convert_units(tests.IrisTest):
    def test_convert_unknown_units(self):
        coord = iris.coords.AuxCoord(1, units='unknown')
        emsg = ('Cannot convert from unknown units. '
                'The "coord.units" attribute may be set directly.')
        with self.assertRaisesRegexp(UnitConversionError, emsg):
            coord.convert_units('degrees')


class Test___str__(tests.IrisTest):

    def test_short_time_interval(self):
        coord = DimCoord([5], standard_name='time',
                         units='days since 1970-01-01')
        expected = ("DimCoord([1970-01-06 00:00:00], standard_name='time', "
                    "calendar='gregorian')")
        result = coord.__str__()
        self.assertEqual(expected, result)

    def test_long_time_interval(self):
        coord = DimCoord([5], standard_name='time',
                         units='years since 1970-01-01')
        expected = "DimCoord([5], standard_name='time', calendar='gregorian')"
        result = coord.__str__()
        self.assertEqual(expected, result)

    def test_non_time_unit(self):
        coord = DimCoord([1.])
        expected = repr(coord)
        result = coord.__str__()
        self.assertEqual(expected, result)


if __name__ == '__main__':
    tests.main()
