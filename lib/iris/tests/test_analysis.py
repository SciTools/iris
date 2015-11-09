# (C) British Crown Copyright 2010 - 2015, Met Office
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


from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import cartopy.crs as ccrs
import cf_units
import numpy as np
import numpy.ma as ma

import iris
import iris.analysis.cartography
import iris.analysis.maths
import iris.coord_systems
import iris.coords
import iris.cube
import iris.tests.stock

# Run tests in no graphics mode if matplotlib is not available.
if tests.MPL_AVAILABLE:
    import matplotlib
    import matplotlib.pyplot as plt


class TestAnalysisCubeCoordComparison(tests.IrisTest):
    def assertComparisonDict(self, comparison_dict, reference_filename):
        string = ''
        for key in sorted(comparison_dict):
            coord_groups = comparison_dict[key]
            string += ('%40s  ' % key)
            names = [[coord.name() if coord is not None else 'None'
                      for coord in coords]
                     for coords in coord_groups]
            string += str(sorted(names))
            string += '\n'
        self.assertString(string, reference_filename)

    def test_coord_comparison(self):
        cube1 = iris.cube.Cube(np.zeros((41, 41)))
        lonlat_cs = iris.coord_systems.GeogCS(6371229)
        lon_points1 = -180 + 4.5 * np.arange(41, dtype=np.float32)
        lat_points = -90 + 4.5 * np.arange(41, dtype=np.float32)
        cube1.add_dim_coord(iris.coords.DimCoord(lon_points1, 'longitude', units='degrees', coord_system=lonlat_cs), 0)
        cube1.add_dim_coord(iris.coords.DimCoord(lat_points, 'latitude', units='degrees', coord_system=lonlat_cs), 1)
        cube1.add_aux_coord(iris.coords.AuxCoord(0, long_name='z'))
        cube1.add_aux_coord(iris.coords.AuxCoord(['foobar'], long_name='f', units='no_unit'))

        cube2 = iris.cube.Cube(np.zeros((41, 41, 5)))
        lonlat_cs = iris.coord_systems.GeogCS(6371229)
        lon_points2 = -160 + 4.5 * np.arange(41, dtype=np.float32)
        cube2.add_dim_coord(iris.coords.DimCoord(lon_points2, 'longitude', units='degrees', coord_system=lonlat_cs), 0)
        cube2.add_dim_coord(iris.coords.DimCoord(lat_points, 'latitude', units='degrees', coord_system=lonlat_cs), 1)
        cube2.add_dim_coord(iris.coords.DimCoord([5, 7, 9, 11, 13], long_name='z'), 2)

        cube3 = cube1.copy()
        lon = cube3.coord("longitude")
        lat = cube3.coord("latitude")
        cube3.remove_coord(lon)
        cube3.remove_coord(lat)
        cube3.add_dim_coord(lon, 1)
        cube3.add_dim_coord(lat, 0)
        cube3.coord('z').points = [20]

        cube4 = cube2.copy()
        lon = cube4.coord("longitude")
        lat = cube4.coord("latitude")
        cube4.remove_coord(lon)
        cube4.remove_coord(lat)
        cube4.add_dim_coord(lon, 1)
        cube4.add_dim_coord(lat, 0)

        coord_comparison = iris.analysis.coord_comparison

        self.assertComparisonDict(coord_comparison(cube1, cube1), ('analysis', 'coord_comparison', 'cube1_cube1.txt'))
        self.assertComparisonDict(coord_comparison(cube1, cube2), ('analysis', 'coord_comparison', 'cube1_cube2.txt'))
        self.assertComparisonDict(coord_comparison(cube1, cube3), ('analysis', 'coord_comparison', 'cube1_cube3.txt'))
        self.assertComparisonDict(coord_comparison(cube1, cube4), ('analysis', 'coord_comparison', 'cube1_cube4.txt'))
        self.assertComparisonDict(coord_comparison(cube2, cube3), ('analysis', 'coord_comparison', 'cube2_cube3.txt'))
        self.assertComparisonDict(coord_comparison(cube2, cube4), ('analysis', 'coord_comparison', 'cube2_cube4.txt'))
        self.assertComparisonDict(coord_comparison(cube3, cube4), ('analysis', 'coord_comparison', 'cube3_cube4.txt'))

        self.assertComparisonDict(coord_comparison(cube1, cube1, cube1), ('analysis', 'coord_comparison', 'cube1_cube1_cube1.txt'))
        self.assertComparisonDict(coord_comparison(cube1, cube2, cube1), ('analysis', 'coord_comparison', 'cube1_cube2_cube1.txt'))

        # get a coord comparison result and check that we are getting back what was expected
        coord_group = coord_comparison(cube1, cube2)['grouped_coords'][0]
        self.assertIsInstance(coord_group, iris.analysis._CoordGroup)
        self.assertIsInstance(list(coord_group)[0], iris.coords.Coord)


class TestAnalysisWeights(tests.IrisTest):
    def test_weighted_mean_little(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        weights = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.float32)

        cube = iris.cube.Cube(data, long_name="test_data", units="1")
        hcs = iris.coord_systems.GeogCS(6371229)
        lat_coord = iris.coords.DimCoord(np.array([1, 2, 3], dtype=np.float32), long_name="lat", units="1", coord_system=hcs)
        lon_coord = iris.coords.DimCoord(np.array([1, 2, 3], dtype=np.float32), long_name="lon", units="1", coord_system=hcs)
        cube.add_dim_coord(lat_coord, 0)
        cube.add_dim_coord(lon_coord, 1)
        cube.add_aux_coord(iris.coords.AuxCoord(np.arange(3, dtype=np.float32), long_name="dummy", units=1), 1)
        self.assertCML(cube, ('analysis', 'weighted_mean_source.cml'))

        a = cube.collapsed('lat', iris.analysis.MEAN, weights=weights)
        self.assertCMLApproxData(a, ('analysis', 'weighted_mean_lat.cml'))

        b = cube.collapsed(lon_coord, iris.analysis.MEAN, weights=weights)
        b.data = np.asarray(b.data)
        self.assertCMLApproxData(b, ('analysis', 'weighted_mean_lon.cml'))
        self.assertEqual(b.coord('dummy').shape, (1, ))

        # test collapsing multiple coordinates (and the fact that one of the coordinates isn't the same coordinate instance as on the cube)
        c = cube.collapsed([lat_coord[:], lon_coord], iris.analysis.MEAN, weights=weights)
        self.assertCMLApproxData(c, ('analysis', 'weighted_mean_latlon.cml'))
        self.assertEqual(c.coord('dummy').shape, (1, ))

        # Check new coord bounds - made from points
        self.assertArrayEqual(c.coord('lat').bounds, [[1, 3]])

        # Check new coord bounds - made from bounds
        cube.coord('lat').bounds = [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]]
        c = cube.collapsed(['lat', 'lon'], iris.analysis.MEAN, weights=weights)
        self.assertArrayEqual(c.coord('lat').bounds, [[0.5, 3.5]])
        cube.coord('lat').bounds = None

        # Check there was no residual change
        self.assertCML(cube, ('analysis', 'weighted_mean_source.cml'))

    @tests.skip_data
    def test_weighted_mean(self):
        ### compare with pp_area_avg - which collapses both lat and lon
        #
        #     pp = ppa('/data/local/dataZoo/PP/simple_pp/global.pp', 0)
        #     print, pp_area(pp, /box)
        #     print, pp_area_avg(pp, /box)  #287.927
        #     ;gives an answer of 287.927
        #
        ###
        e = iris.tests.stock.simple_pp()
        self.assertCML(e, ('analysis', 'weighted_mean_original.cml'))
        e.coord('latitude').guess_bounds()
        e.coord('longitude').guess_bounds()
        area_weights = iris.analysis.cartography.area_weights(e)
        e.coord('latitude').bounds = None
        e.coord('longitude').bounds = None
        f, collapsed_area_weights = e.collapsed('latitude', iris.analysis.MEAN, weights=area_weights, returned=True)
        g = f.collapsed('longitude', iris.analysis.MEAN, weights=collapsed_area_weights)
        # check it's a 0d, scalar cube
        self.assertEqual(g.shape, ())
        # check the value - pp_area_avg's result of 287.927 differs by factor of 1.00002959
        np.testing.assert_approx_equal(g.data, 287.935, significant=5)

        #check we get summed weights even if we don't give any
        h, summed_weights = e.collapsed('latitude', iris.analysis.MEAN, returned=True)
        assert(summed_weights is not None)

        # Check there was no residual change
        e.coord('latitude').bounds = None
        e.coord('longitude').bounds = None
        self.assertCML(e, ('analysis', 'weighted_mean_original.cml'))

        # Test collapsing of missing coord
        self.assertRaises(iris.exceptions.CoordinateNotFoundError, e.collapsed, 'platitude', iris.analysis.MEAN)

        # Test collpasing of non data coord
        self.assertRaises(iris.exceptions.CoordinateCollapseError, e.collapsed, 'pressure', iris.analysis.MEAN)

@tests.skip_data
class TestAnalysisBasic(tests.IrisTest):
    def setUp(self):
        file = tests.get_data_path(('PP', 'aPProt1', 'rotatedMHtimecube.pp'))
        cubes = iris.load(file)
        self.cube = cubes[0]
        self.assertCML(self.cube, ('analysis', 'original.cml'))

    def _common(self, name, aggregate, original_name='original_common.cml', *args, **kwargs):
        self.cube.data = self.cube.data.astype(np.float64)

        self.assertCML(self.cube, ('analysis', original_name))

        a = self.cube.collapsed('grid_latitude', aggregate)
        self.assertCMLApproxData(a, ('analysis', '%s_latitude.cml' % name), *args, **kwargs)

        b = a.collapsed('grid_longitude', aggregate)
        self.assertCMLApproxData(b, ('analysis', '%s_latitude_longitude.cml' % name), *args, **kwargs)

        c = self.cube.collapsed(['grid_latitude', 'grid_longitude'], aggregate)
        self.assertCMLApproxData(c, ('analysis', '%s_latitude_longitude_1call.cml' % name), *args, **kwargs)

        # Check there was no residual change
        self.assertCML(self.cube, ('analysis', original_name))

    def test_mean(self):
        self._common('mean', iris.analysis.MEAN, decimal=1)

    def test_std_dev(self):
        # as the numbers are so high, trim off some trailing digits & compare to 0dp
        self._common('std_dev', iris.analysis.STD_DEV, decimal=1)

    def test_hmean(self):
        # harmonic mean requires data > 0
        self.cube.data *= self.cube.data
        self._common('hmean', iris.analysis.HMEAN, 'original_hmean.cml', decimal=1)

    def test_gmean(self):
        self._common('gmean', iris.analysis.GMEAN, decimal=1)

    def test_variance(self):
        # as the numbers are so high, trim off some trailing digits & compare to 0dp
        self._common('variance', iris.analysis.VARIANCE, decimal=1)

    def test_median(self):
        self._common('median', iris.analysis.MEDIAN)

    def test_sum(self):
        # as the numbers are so high, trim off some trailing digits & compare to 0dp
        self._common('sum', iris.analysis.SUM, decimal=1)

    def test_max(self):
        self._common('max', iris.analysis.MAX)

    def test_min(self):
        self._common('min', iris.analysis.MIN)

    def test_rms(self):
        self._common('rms', iris.analysis.RMS)

    def test_duplicate_coords(self):
        self.assertRaises(ValueError, tests.stock.track_1d, duplicate_x=True)


class TestMissingData(tests.IrisTest):
    def setUp(self):
        self.cube_with_nan = tests.stock.simple_2d()

        data = self.cube_with_nan.data.astype(np.float32)
        self.cube_with_nan.data = data.copy()
        self.cube_with_nan.data[1, 0] = np.nan
        self.cube_with_nan.data[2, 2] = np.nan
        self.cube_with_nan.data[2, 3] = np.nan

        self.cube_with_mask = tests.stock.simple_2d()
        self.cube_with_mask.data = ma.array(self.cube_with_nan.data,
                                                  mask=np.isnan(self.cube_with_nan.data))

    def test_max(self):
        cube = self.cube_with_nan.collapsed('foo', iris.analysis.MAX)
        np.testing.assert_array_equal(cube.data, np.array([3, np.nan, np.nan]))

        cube = self.cube_with_mask.collapsed('foo', iris.analysis.MAX)
        np.testing.assert_array_equal(cube.data, np.array([3, 7, 9]))

    def test_min(self):
        cube = self.cube_with_nan.collapsed('foo', iris.analysis.MIN)
        np.testing.assert_array_equal(cube.data, np.array([0, np.nan, np.nan]))

        cube = self.cube_with_mask.collapsed('foo', iris.analysis.MIN)
        np.testing.assert_array_equal(cube.data, np.array([0, 5, 8]))

    def test_sum(self):
        cube = self.cube_with_nan.collapsed('foo', iris.analysis.SUM)
        np.testing.assert_array_equal(cube.data, np.array([6, np.nan, np.nan]))

        cube = self.cube_with_mask.collapsed('foo', iris.analysis.SUM)
        np.testing.assert_array_equal(cube.data, np.array([6, 18, 17]))


class TestAggregator_mdtol_keyword(tests.IrisTest):
    def setUp(self):
        data = ma.array([[1, 2], [4, 5]], dtype=np.float32,
                        mask=[[False, True], [False, True]])
        cube = iris.cube.Cube(data, long_name="test_data", units="1")
        lat_coord = iris.coords.DimCoord(np.array([1, 2], dtype=np.float32),
                                         long_name="lat", units="1")
        lon_coord = iris.coords.DimCoord(np.array([3, 4], dtype=np.float32),
                                         long_name="lon", units="1")
        cube.add_dim_coord(lat_coord, 0)
        cube.add_dim_coord(lon_coord, 1)
        self.cube = cube

    def test_single_coord_no_mdtol(self):
        collapsed = self.cube.collapsed(
            self.cube.coord('lat'), iris.analysis.MEAN)
        t = ma.array([2.5, 5.], mask=[False, True])
        self.assertMaskedArrayEqual(collapsed.data, t)

    def test_single_coord_mdtol(self):
        self.cube.data.mask = np.array([[False, True], [False, False]])
        collapsed = self.cube.collapsed(
            self.cube.coord('lat'), iris.analysis.MEAN, mdtol=0.5)
        t = ma.array([2.5, 5], mask=[False, False])
        self.assertMaskedArrayEqual(collapsed.data, t)

    def test_single_coord_mdtol_alt(self):
        self.cube.data.mask = np.array([[False, True], [False, False]])
        collapsed = self.cube.collapsed(
            self.cube.coord('lat'), iris.analysis.MEAN, mdtol=0.4)
        t = ma.array([2.5, 5], mask=[False, True])
        self.assertMaskedArrayEqual(collapsed.data, t)

    def test_multi_coord_no_mdtol(self):
        collapsed = self.cube.collapsed(
            [self.cube.coord('lat'), self.cube.coord('lon')],
            iris.analysis.MEAN)
        t = np.array(2.5)
        self.assertArrayEqual(collapsed.data, t)

    def test_multi_coord_mdtol(self):
        collapsed = self.cube.collapsed(
            [self.cube.coord('lat'), self.cube.coord('lon')],
            iris.analysis.MEAN, mdtol=0.4)
        t = ma.array(2.5, mask=True)
        self.assertMaskedArrayEqual(collapsed.data, t)


class TestAggregators(tests.IrisTest):
    def test_percentile_1d(self):
        cube = tests.stock.simple_1d()

        first_quartile = cube.collapsed('foo', iris.analysis.PERCENTILE,
                                        percent=25)
        np.testing.assert_array_almost_equal(first_quartile.data,
                                             np.array([2.5], dtype=np.float32))
        self.assertCML(first_quartile, ('analysis',
                                        'first_quartile_foo_1d.cml'),
                       checksum=False)

        third_quartile = cube.collapsed('foo', iris.analysis.PERCENTILE,
                                        percent=75)
        np.testing.assert_array_almost_equal(third_quartile.data,
                                             np.array([7.5],
                                             dtype=np.float32))
        self.assertCML(third_quartile,
                       ('analysis', 'third_quartile_foo_1d.cml'),
                       checksum=False)

    def test_percentile_2d(self):
        cube = tests.stock.simple_2d()

        first_quartile = cube.collapsed('foo', iris.analysis.PERCENTILE,
                                        percent=25)
        np.testing.assert_array_almost_equal(first_quartile.data,
                                             np.array([0.75, 4.75, 8.75],
                                             dtype=np.float32))
        self.assertCML(first_quartile, ('analysis',
                                        'first_quartile_foo_2d.cml'),
                       checksum=False)

        first_quartile = cube.collapsed(('foo', 'bar'),
                                        iris.analysis.PERCENTILE, percent=25)
        np.testing.assert_array_almost_equal(first_quartile.data,
                                             np.array([2.75],
                                             dtype=np.float32))
        self.assertCML(first_quartile, ('analysis',
                                        'first_quartile_foo_bar_2d.cml'),
                       checksum=False)

    def test_percentile_3d(self):
        array_3d = np.arange(24, dtype=np.int32).reshape((2, 3, 4))

        last_quartile = iris.analysis._percentile(array_3d, 0, 50)
        np.testing.assert_array_almost_equal(last_quartile,
                                             np.array([[6., 7., 8., 9.],
                                                       [10., 11., 12., 13.],
                                                       [14., 15., 16., 17.]],
                                             dtype=np.float32))

    def test_percentile_3d_axis_one(self):
        array_3d = np.arange(24, dtype=np.int32).reshape((2, 3, 4))

        last_quartile = iris.analysis._percentile(array_3d, 1, 50)
        np.testing.assert_array_almost_equal(last_quartile,
                                             np.array([[4., 5., 6., 7.],
                                                       [16., 17., 18., 19.]],
                                             dtype=np.float32))

    def test_percentile_3d_axis_two(self):
        array_3d = np.arange(24, dtype=np.int32).reshape((2, 3, 4))

        last_quartile = iris.analysis._percentile(array_3d, 2, 50)
        np.testing.assert_array_almost_equal(last_quartile,
                                             np.array([[1.5, 5.5, 9.5],
                                                       [13.5, 17.5, 21.5]],
                                             dtype=np.float32))

    def test_percentile_3d_masked(self):
        cube = tests.stock.simple_3d_mask()

        last_quartile = cube.collapsed('wibble',
                                       iris.analysis.PERCENTILE, percent=75)
        np.testing.assert_array_almost_equal(last_quartile.data,
                                             np.array([[12., 13., 14., 15.],
                                                       [16., 17., 18., 19.],
                                                       [20., 18., 19., 20.]],
                                             dtype=np.float32))
        self.assertCML(last_quartile, ('analysis',
                                       'last_quartile_foo_3d_masked.cml'),
                       checksum=False)

    def test_percentile_3d_notmasked(self):
        cube = tests.stock.simple_3d()

        last_quartile = cube.collapsed('wibble',
                                       iris.analysis.PERCENTILE, percent=75)
        np.testing.assert_array_almost_equal(last_quartile.data,
                                             np.array([[9., 10., 11., 12.],
                                                       [13., 14., 15., 16.],
                                                       [17., 18., 19., 20.]],
                                             dtype=np.float32))
        self.assertCML(last_quartile, ('analysis',
                                       'last_quartile_foo_3d_notmasked.cml'),
                       checksum=False)

    def test_proportion(self):
        cube = tests.stock.simple_1d()
        r = cube.data >= 5
        gt5 = cube.collapsed('foo', iris.analysis.PROPORTION, function=lambda val: val >= 5)
        np.testing.assert_array_almost_equal(gt5.data, np.array([6 / 11.]))
        self.assertCML(gt5, ('analysis', 'proportion_foo_1d.cml'), checksum=False)

    def test_proportion_2d(self):
        cube = tests.stock.simple_2d()

        gt6 = cube.collapsed('foo', iris.analysis.PROPORTION, function=lambda val: val >= 6)
        np.testing.assert_array_almost_equal(gt6.data, np.array([0, 0.5, 1], dtype=np.float32))
        self.assertCML(gt6, ('analysis', 'proportion_foo_2d.cml'), checksum=False)

        gt6 = cube.collapsed('bar', iris.analysis.PROPORTION, function=lambda val: val >= 6)
        np.testing.assert_array_almost_equal(gt6.data, np.array([1 / 3, 1 / 3, 2 / 3, 2 / 3], dtype=np.float32))
        self.assertCML(gt6, ('analysis', 'proportion_bar_2d.cml'), checksum=False)

        gt6 = cube.collapsed(('foo', 'bar'), iris.analysis.PROPORTION, function=lambda val: val >= 6)
        np.testing.assert_array_almost_equal(gt6.data, np.array([0.5], dtype=np.float32))
        self.assertCML(gt6, ('analysis', 'proportion_foo_bar_2d.cml'), checksum=False)

        # mask the data
        cube.data = ma.array(cube.data, mask=cube.data % 2)
        cube.data.mask[1, 2] = True
        gt6_masked = cube.collapsed('bar', iris.analysis.PROPORTION, function=lambda val: val >= 6)
        np.testing.assert_array_almost_equal(gt6_masked.data, ma.array([1 / 3, None, 1 / 2, None],
                                                                                mask=[False, True, False, True],
                                                                                dtype=np.float32))
        self.assertCML(gt6_masked, ('analysis', 'proportion_foo_2d_masked.cml'), checksum=False)

    def test_count(self):
        cube = tests.stock.simple_1d()
        gt5 = cube.collapsed('foo', iris.analysis.COUNT, function=lambda val: val >= 5)
        np.testing.assert_array_almost_equal(gt5.data, np.array([6]))
        gt5.data = gt5.data.astype('i8')
        self.assertCML(gt5, ('analysis', 'count_foo_1d.cml'), checksum=False)

    def test_count_2d(self):
        cube = tests.stock.simple_2d()

        gt6 = cube.collapsed('foo', iris.analysis.COUNT, function=lambda val: val >= 6)
        np.testing.assert_array_almost_equal(gt6.data, np.array([0, 2, 4], dtype=np.float32))
        gt6.data = gt6.data.astype('i8')
        self.assertCML(gt6, ('analysis', 'count_foo_2d.cml'), checksum=False)

        gt6 = cube.collapsed('bar', iris.analysis.COUNT, function=lambda val: val >= 6)
        np.testing.assert_array_almost_equal(gt6.data, np.array([1, 1, 2, 2], dtype=np.float32))
        gt6.data = gt6.data.astype('i8')
        self.assertCML(gt6, ('analysis', 'count_bar_2d.cml'), checksum=False)

        gt6 = cube.collapsed(('foo', 'bar'), iris.analysis.COUNT, function=lambda val: val >= 6)
        np.testing.assert_array_almost_equal(gt6.data, np.array([6], dtype=np.float32))
        gt6.data = gt6.data.astype('i8')
        self.assertCML(gt6, ('analysis', 'count_foo_bar_2d.cml'), checksum=False)

    def test_weighted_sum_consistency(self):
        # weighted sum with unit weights should be the same as a sum
        cube = tests.stock.simple_1d()
        normal_sum = cube.collapsed('foo', iris.analysis.SUM)
        weights = np.ones_like(cube.data)
        weighted_sum = cube.collapsed('foo', iris.analysis.SUM, weights=weights)
        self.assertArrayAlmostEqual(normal_sum.data, weighted_sum.data)

    def test_weighted_sum_1d(self):
        # verify 1d weighted sum is correct
        cube = tests.stock.simple_1d()
        weights = np.array([.05, .05, .1, .1, .2, .3, .2, .1, .1, .05, .05])
        result = cube.collapsed('foo', iris.analysis.SUM, weights=weights)
        self.assertAlmostEqual(result.data, 6.5)
        self.assertCML(result, ('analysis', 'sum_weighted_1d.cml'),
                       checksum=False)

    def test_weighted_sum_2d(self):
        # verify 2d weighted sum is correct
        cube = tests.stock.simple_2d()
        weights = np.array([.3, .4, .3])
        weights = iris.util.broadcast_to_shape(weights, cube.shape, [0])
        result = cube.collapsed('bar', iris.analysis.SUM, weights=weights)
        self.assertArrayAlmostEqual(result.data, np.array([4., 5., 6., 7.]))
        self.assertCML(result, ('analysis', 'sum_weighted_2d.cml'),
                       checksum=False)

    def test_weighted_rms(self):
        cube = tests.stock.simple_2d()
        # modify cube data so that the results are nice numbers
        cube.data = np.array([[4, 7, 10, 8],
                              [21, 30, 12, 24],
                              [14, 16, 20, 8]],
                             dtype=np.float64)
        weights = np.array([[1, 4, 3, 2],
                            [6, 4.5, 1.5, 3],
                            [2, 1, 1.5, 0.5]],
                           dtype=np.float64)
        expected_result = np.array([8.0, 24.0, 16.0])
        result = cube.collapsed('foo', iris.analysis.RMS, weights=weights)
        self.assertArrayAlmostEqual(result.data, expected_result)
        self.assertCML(result, ('analysis', 'rms_weighted_2d.cml'),
                       checksum=False)


@tests.skip_data
class TestRotatedPole(tests.GraphicsTest):
    @tests.skip_plot
    def _check_both_conversions(self, cube):
        rlons, rlats = iris.analysis.cartography.get_xy_grids(cube)
        rcs = cube.coord_system('RotatedGeogCS')
        x, y = iris.analysis.cartography.unrotate_pole(
            rlons, rlats, rcs.grid_north_pole_longitude,
            rcs.grid_north_pole_latitude)
        plt.scatter(x, y)
        self.check_graphic()

        plt.scatter(rlons, rlats)
        self.check_graphic()

    def test_all(self):
        path = tests.get_data_path(('PP', 'ukVorog', 'ukv_orog_refonly.pp'))
        master_cube = iris.load_cube(path)

        # Check overall behaviour.
        cube = master_cube[::10, ::10]
        self._check_both_conversions(cube)

        # Check numerical stability.
        cube = master_cube[210:238, 424:450]
        self._check_both_conversions(cube)

    def test_unrotate_nd(self):
        rlons = np.array([[350., 352.], [350., 352.]])
        rlats = np.array([[-5., -0.], [-4., -1.]])

        resx, resy = iris.analysis.cartography.unrotate_pole(rlons, rlats,
                                                             178.0, 38.0)

        # Solutions derived by proj4 direct.
        solx = np.array([[-16.42176094, -14.85892262],
                            [-16.71055023, -14.58434624]])
        soly = np.array([[ 46.00724251,  51.29188893],
                            [ 46.98728486,  50.30706042]])

        self.assertArrayAlmostEqual(resx, solx)
        self.assertArrayAlmostEqual(resy, soly)

    def test_unrotate_1d(self):
        rlons = np.array([350., 352., 354., 356.])
        rlats = np.array([-5., -0., 5., 10.])

        resx, resy = iris.analysis.cartography.unrotate_pole(
            rlons.flatten(), rlats.flatten(), 178.0, 38.0)

        # Solutions derived by proj4 direct.
        solx = np.array([-16.42176094, -14.85892262,
                            -12.88946157, -10.35078336])
        soly = np.array([46.00724251, 51.29188893,
                            56.55031485, 61.77015703])

        self.assertArrayAlmostEqual(resx, solx)
        self.assertArrayAlmostEqual(resy, soly)

    def test_rotate_nd(self):
        rlons = np.array([[350., 351.], [352., 353.]])
        rlats = np.array([[10., 15.], [20., 25.]])

        resx, resy = iris.analysis.cartography.rotate_pole(rlons, rlats,
                                                           20., 80.)

        # Solutions derived by proj4 direct.
        solx = np.array([[148.69672569, 149.24727087],
                            [149.79067025, 150.31754368]])
        soly = np.array([[18.60905789, 23.67749384],
                            [28.74419024, 33.8087963 ]])

        self.assertArrayAlmostEqual(resx, solx)
        self.assertArrayAlmostEqual(resy, soly)

    def test_rotate_1d(self):
        rlons = np.array([350., 351., 352., 353.])
        rlats = np.array([10., 15., 20., 25.])

        resx, resy = iris.analysis.cartography.rotate_pole(rlons.flatten(),
                                     rlats.flatten(), 20., 80.)

        # Solutions derived by proj4 direct.
        solx = np.array([148.69672569, 149.24727087,
                            149.79067025, 150.31754368])
        soly = np.array([18.60905789, 23.67749384,
                            28.74419024, 33.8087963 ])

        self.assertArrayAlmostEqual(resx, solx)
        self.assertArrayAlmostEqual(resy, soly)


@tests.skip_data
class TestAreaWeights(tests.IrisTest):
    def test_area_weights(self):
        small_cube = iris.tests.stock.simple_pp()
        # Get offset, subsampled region: small enough to test against literals
        small_cube = small_cube[10:, 35:]
        small_cube = small_cube[::8, ::8]
        small_cube = small_cube[:5, :4]
        # pre-check non-data properties
        self.assertCML(small_cube, ('analysis', 'areaweights_original.cml'),
                       checksum=False)

        # check area-weights values
        small_cube.coord('latitude').guess_bounds()
        small_cube.coord('longitude').guess_bounds()
        area_weights = iris.analysis.cartography.area_weights(small_cube)
        expected_results = np.array(
            [[3.11955916e+12, 3.11956058e+12, 3.11955916e+12, 3.11956058e+12],
             [5.21950793e+12, 5.21951031e+12, 5.21950793e+12, 5.21951031e+12],
             [6.68991432e+12, 6.68991737e+12, 6.68991432e+12, 6.68991737e+12],
             [7.35341320e+12, 7.35341655e+12, 7.35341320e+12, 7.35341655e+12],
             [7.12998265e+12, 7.12998589e+12, 7.12998265e+12, 7.12998589e+12]],
            dtype=np.float64)
        self.assertArrayAllClose(area_weights, expected_results, rtol=1e-8)

        # Check there was no residual change
        small_cube.coord('latitude').bounds = None
        small_cube.coord('longitude').bounds = None
        self.assertCML(small_cube, ('analysis', 'areaweights_original.cml'),
                       checksum=False)

class TestAreaWeightGeneration(tests.IrisTest):
    def setUp(self):
        self.cube = iris.tests.stock.realistic_4d()

    def test_area_weights_std(self):
        # weights for stock 4d data
        weights = iris.analysis.cartography.area_weights(self.cube)
        self.assertEqual(weights.shape, self.cube.shape)

    def test_area_weights_order(self):
        # weights for data with dimensions in a different order
        order = [3, 2, 1, 0] # (lon, lat, level, time)
        self.cube.transpose(order)
        weights = iris.analysis.cartography.area_weights(self.cube)
        self.assertEqual(weights.shape, self.cube.shape)

    def test_area_weights_non_adjacent(self):
        # weights for cube with non-adjacent latitude/longitude dimensions
        order = [0, 3, 1, 2] # (time, lon, level, lat)
        self.cube.transpose(order)
        weights = iris.analysis.cartography.area_weights(self.cube)
        self.assertEqual(weights.shape, self.cube.shape)

    def test_area_weights_scalar_latitude(self):
        # weights for cube with a scalar latitude dimension
        cube = self.cube[:, :, 0, :]
        weights = iris.analysis.cartography.area_weights(cube)
        self.assertEqual(weights.shape, cube.shape)

    def test_area_weights_scalar_longitude(self):
        # weights for cube with a scalar longitude dimension
        cube = self.cube[:, :, :, 0]
        weights = iris.analysis.cartography.area_weights(cube)
        self.assertEqual(weights.shape, cube.shape)

    def test_area_weights_scalar(self):
        # weights for cube with scalar latitude and longitude dimensions
        cube = self.cube[:, :, 0, 0]
        weights = iris.analysis.cartography.area_weights(cube)
        self.assertEqual(weights.shape, cube.shape)

    def test_area_weights_singleton_latitude(self):
        # singleton (1-point) latitude dimension
        cube = self.cube[:, :, 0:1, :]
        weights = iris.analysis.cartography.area_weights(cube)
        self.assertEqual(weights.shape, cube.shape)

    def test_area_weights_singleton_longitude(self):
        # singleton (1-point) longitude dimension
        cube = self.cube[:, :, :, 0:1]
        weights = iris.analysis.cartography.area_weights(cube)
        self.assertEqual(weights.shape, cube.shape)

    def test_area_weights_singletons(self):
        # singleton (1-point) latitude and longitude dimensions
        cube = self.cube[:, :, 0:1, 0:1]
        weights = iris.analysis.cartography.area_weights(cube)
        self.assertEqual(weights.shape, cube.shape)

    def test_area_weights_normalized(self):
        # normalized area weights must sum to one over lat/lon dimensions.
        weights = iris.analysis.cartography.area_weights(self.cube,
                                                         normalize=True)
        sumweights = weights.sum(axis=3).sum(axis=2)  # sum over lon and lat
        self.assertArrayAlmostEqual(sumweights, 1)

    def test_area_weights_non_contiguous(self):
        # Slice the cube so that we have non-contiguous longitude
        # bounds.
        ind = (0, 1, 2, -3, -2, -1)
        cube = self.cube[..., ind]
        weights = iris.analysis.cartography.area_weights(cube)
        expected = iris.analysis.cartography.area_weights(self.cube)[..., ind]
        self.assertArrayEqual(weights, expected)

    def test_area_weights_no_lon_bounds(self):
        self.cube.coord('grid_longitude').bounds = None
        with self.assertRaises(ValueError):
            iris.analysis.cartography.area_weights(self.cube)

    def test_area_weights_no_lat_bounds(self):
        self.cube.coord('grid_latitude').bounds = None
        with self.assertRaises(ValueError):
            iris.analysis.cartography.area_weights(self.cube)


@tests.skip_data
class TestLatitudeWeightGeneration(tests.IrisTest):

    def setUp(self):
        path = iris.tests.get_data_path(['NetCDF', 'rotated', 'xyt',
                                         'small_rotPole_precipitation.nc'])
        self.cube = iris.load_cube(path)
        self.cube_dim_lat = self.cube.copy()
        self.cube_dim_lat.remove_coord('latitude')
        self.cube_dim_lat.remove_coord('longitude')
        # The 2d cubes are unrealistic, you would not want to weight by
        # anything other than grid latitude in real-world scenarios. However,
        # the technical details are suitable for testing purposes, providing
        # a nice analog for a 2d latitude coordinate from a curvilinear grid.
        self.cube_aux_lat = self.cube.copy()
        self.cube_aux_lat.remove_coord('grid_latitude')
        self.cube_aux_lat.remove_coord('grid_longitude')
        self.lat1d = self.cube.coord('grid_latitude').points
        self.lat2d = self.cube.coord('latitude').points

    def test_cosine_latitude_weights_range(self):
        # check the range of returned values, needs a cube that spans the full
        # latitude range
        lat_coord = iris.coords.DimCoord(np.linspace(-90, 90, 73),
                                         standard_name='latitude',
                                         units=cf_units.Unit('degrees_north'))
        cube = iris.cube.Cube(np.ones([73], dtype=np.float64),
                              long_name='test_cube', units='1')
        cube.add_dim_coord(lat_coord, 0)
        weights = iris.analysis.cartography.cosine_latitude_weights(cube)
        self.assertTrue(weights.max() <= 1)
        self.assertTrue(weights.min() >= 0)

    def test_cosine_latitude_weights_0d(self):
        # 0d latitude dimension (scalar coordinate)
        weights = iris.analysis.cartography.cosine_latitude_weights(
            self.cube_dim_lat[:, 0, :])
        self.assertEqual(weights.shape, self.cube_dim_lat[:, 0, :].shape)
        self.assertAlmostEqual(weights[0, 0],
                               np.cos(np.deg2rad(self.lat1d[0])))

    def test_cosine_latitude_weights_1d_singleton(self):
        # singleton (1-point) 1d latitude coordinate (time, lat, lon)
        cube = self.cube_dim_lat[:, 0:1, :]
        weights = iris.analysis.cartography.cosine_latitude_weights(cube)
        self.assertEqual(weights.shape, cube.shape)
        self.assertAlmostEqual(weights[0, 0, 0],
                               np.cos(np.deg2rad(self.lat1d[0])))

    def test_cosine_latitude_weights_1d(self):
        # 1d latitude coordinate (time, lat, lon)
        weights = iris.analysis.cartography.cosine_latitude_weights(
            self.cube_dim_lat)
        self.assertEqual(weights.shape, self.cube.shape)
        self.assertArrayAlmostEqual(weights[0, :, 0],
                                    np.cos(np.deg2rad(self.lat1d)))

    def test_cosine_latitude_weights_1d_latitude_first(self):
        # 1d latitude coordinate with latitude first (lat, time, lon)
        order = [1, 0, 2] # (lat, time, lon)
        self.cube_dim_lat.transpose(order)
        weights = iris.analysis.cartography.cosine_latitude_weights(
            self.cube_dim_lat)
        self.assertEqual(weights.shape, self.cube_dim_lat.shape)
        self.assertArrayAlmostEqual(weights[:, 0, 0],
                                    np.cos(np.deg2rad(self.lat1d)))

    def test_cosine_latitude_weights_1d_latitude_last(self):
        # 1d latitude coordinate with latitude last (time, lon, lat)
        order = [0, 2, 1] # (time, lon, lat)
        self.cube_dim_lat.transpose(order)
        weights = iris.analysis.cartography.cosine_latitude_weights(
            self.cube_dim_lat)
        self.assertEqual(weights.shape, self.cube_dim_lat.shape)
        self.assertArrayAlmostEqual(weights[0, 0, :],
                                    np.cos(np.deg2rad(self.lat1d)))

    def test_cosine_latitude_weights_2d_singleton1(self):
        # 2d latitude coordinate with first dimension singleton
        cube = self.cube_aux_lat[:, 0:1, :]
        weights = iris.analysis.cartography.cosine_latitude_weights(cube)
        self.assertEqual(weights.shape, cube.shape)
        self.assertArrayAlmostEqual(weights[0, :, :],
                                    np.cos(np.deg2rad(self.lat2d[0:1, :])))

    def test_cosine_latitude_weights_2d_singleton2(self):
        # 2d latitude coordinate with second dimension singleton
        cube = self.cube_aux_lat[:, :, 0:1]
        weights = iris.analysis.cartography.cosine_latitude_weights(cube)
        self.assertEqual(weights.shape, cube.shape)
        self.assertArrayAlmostEqual(weights[0, :, :],
                                    np.cos(np.deg2rad(self.lat2d[:, 0:1])))

    def test_cosine_latitude_weights_2d_singleton3(self):
        # 2d latitude coordinate with both dimensions singleton
        cube = self.cube_aux_lat[:, 0:1, 0:1]
        weights = iris.analysis.cartography.cosine_latitude_weights(cube)
        self.assertEqual(weights.shape, cube.shape)
        self.assertArrayAlmostEqual(weights[0, :, :],
                                    np.cos(np.deg2rad(self.lat2d[0:1, 0:1])))

    def test_cosine_latitude_weights_2d(self):
        # 2d latitude coordinate (time, lat, lon)
        weights = iris.analysis.cartography.cosine_latitude_weights(
            self.cube_aux_lat)
        self.assertEqual(weights.shape, self.cube_aux_lat.shape)
        self.assertArrayAlmostEqual(weights[0, :, :],
                                    np.cos(np.deg2rad(self.lat2d)))

    def test_cosine_latitude_weights_2d_latitude_first(self):
        # 2d latitude coordinate with latitude first (lat, time, lon)
        order = [1, 0, 2] # (lat, time, lon)
        self.cube_aux_lat.transpose(order)
        weights = iris.analysis.cartography.cosine_latitude_weights(
            self.cube_aux_lat)
        self.assertEqual(weights.shape, self.cube_aux_lat.shape)
        self.assertArrayAlmostEqual(weights[:, 0, :],
                                    np.cos(np.deg2rad(self.lat2d)))

    def test_cosine_latitude_weights_2d_latitude_last(self):
        # 2d latitude coordinate with latitude last (time, lon, lat)
        order = [0, 2, 1] # (time, lon, lat)
        self.cube_aux_lat.transpose(order)
        weights = iris.analysis.cartography.cosine_latitude_weights(
            self.cube_aux_lat)
        self.assertEqual(weights.shape, self.cube_aux_lat.shape)
        self.assertArrayAlmostEqual(weights[0, :, :],
                                    np.cos(np.deg2rad(self.lat2d.T)))

    def test_cosine_latitude_weights_no_latitude(self):
        # no coordinate identified as latitude
        self.cube_dim_lat.remove_coord('grid_latitude')
        with self.assertRaises(ValueError):
            weights = iris.analysis.cartography.cosine_latitude_weights(
                self.cube_dim_lat)

    def test_cosine_latitude_weights_multiple_latitude(self):
        # two coordinates identified as latitude
        with self.assertRaises(ValueError):
            weights = iris.analysis.cartography.cosine_latitude_weights(
                self.cube)


class TestRollingWindow(tests.IrisTest):
    def setUp(self):
        # XXX Comes from test_aggregated_by
        cube = iris.cube.Cube(np.array([[6, 10, 12, 18], [8, 12, 14, 20], [18, 12, 10, 6]]), long_name='temperature', units='kelvin')
        cube.add_dim_coord(iris.coords.DimCoord(np.array([0, 5, 10], dtype=np.float64), 'latitude', units='degrees'), 0)
        cube.add_dim_coord(iris.coords.DimCoord(np.array([0, 2, 4, 6], dtype=np.float64), 'longitude', units='degrees'), 1)

        self.cube = cube

    def test_non_mean_operator(self):
        res_cube = self.cube.rolling_window('longitude', iris.analysis.MAX, window=2)
        expected_result = np.array([[10, 12, 18],
                                       [12, 14, 20],
                                       [18, 12, 10]], dtype=np.float64)
        self.assertArrayEqual(expected_result, res_cube.data)

    def test_longitude_simple(self):
        res_cube = self.cube.rolling_window('longitude', iris.analysis.MEAN, window=2)

        expected_result = np.array([[  8., 11., 15.],
                                      [ 10., 13., 17.],
                                      [ 15., 11., 8.]], dtype=np.float64)

        self.assertArrayEqual(expected_result, res_cube.data)

        self.assertCML(res_cube, ('analysis', 'rolling_window', 'simple_longitude.cml'))

        self.assertRaises(ValueError, self.cube.rolling_window, 'longitude', iris.analysis.MEAN, window=0)

    def test_longitude_masked(self):
        self.cube.data = ma.array(self.cube.data,
                                  mask=[[True, True, True, True],
                                        [True, False, True, True],
                                        [False, False, False, False]])
        res_cube = self.cube.rolling_window('longitude',
                                            iris.analysis.MEAN,
                                            window=2)

        expected_result = np.ma.array([[-99., -99., -99.],
                                       [12., 12., -99.],
                                       [15., 11., 8.]],
                                      mask=[[True, True, True],
                                            [False, False, True],
                                            [False, False, False]],
                                      dtype=np.float64)

        self.assertMaskedArrayEqual(expected_result, res_cube.data)

    def test_longitude_circular(self):
        cube = self.cube
        cube.coord('longitude').circular = True
        self.assertRaises(iris.exceptions.NotYetImplementedError, self.cube.rolling_window, 'longitude', iris.analysis.MEAN, window=0)

    def test_different_length_windows(self):
        res_cube = self.cube.rolling_window('longitude', iris.analysis.MEAN, window=4)

        expected_result = np.array([[ 11.5],
                                       [ 13.5],
                                       [ 11.5]], dtype=np.float64)

        self.assertArrayEqual(expected_result, res_cube.data)

        self.assertCML(res_cube, ('analysis', 'rolling_window', 'size_4_longitude.cml'))

        # Window too long:
        self.assertRaises(ValueError, self.cube.rolling_window, 'longitude', iris.analysis.MEAN, window=6)
        # Window too small:
        self.assertRaises(ValueError, self.cube.rolling_window, 'longitude', iris.analysis.MEAN, window=0)

    def test_bad_coordinate(self):
        self.assertRaises(KeyError, self.cube.rolling_window, 'wibble', iris.analysis.MEAN, window=0)

    def test_latitude_simple(self):
        res_cube = self.cube.rolling_window('latitude', iris.analysis.MEAN, window=2)

        expected_result = np.array([[  7., 11., 13., 19.],
                                       [ 13., 12., 12., 13.]], dtype=np.float64)

        self.assertArrayEqual(expected_result, res_cube.data)

        self.assertCML(res_cube, ('analysis', 'rolling_window', 'simple_latitude.cml'))

    def test_mean_with_weights_consistency(self):
        # equal weights should be the same as the mean with no weights
        wts = np.array([0.5, 0.5], dtype=np.float64)
        res_cube = self.cube.rolling_window('longitude',
                                            iris.analysis.MEAN,
                                            window=2,
                                            weights=wts)
        expected_result = self.cube.rolling_window('longitude',
                                                   iris.analysis.MEAN,
                                                   window=2)
        self.assertArrayEqual(expected_result.data, res_cube.data)

    def test_mean_with_weights(self):
        # rolling window mean with weights
        wts = np.array([0.1, 0.6, 0.3], dtype=np.float64)
        res_cube = self.cube.rolling_window('longitude',
                                            iris.analysis.MEAN,
                                            window=3,
                                            weights=wts)
        expected_result = np.array([[10.2, 13.6],
                                       [12.2, 15.6],
                                       [12.0, 9.0]], dtype=np.float64)
        # use almost equal to compare floats
        self.assertArrayAlmostEqual(expected_result, res_cube.data)


class TestProject(tests.GraphicsTest):
    def setUp(self):
        cube = iris.tests.stock.realistic_4d_no_derived()
        # Remove some slices to speed testing.
        self.cube = cube[0:2, 0:3]
        self.target_proj = ccrs.Robinson()

    def test_bad_resolution(self):
        with self.assertRaises(ValueError):
            iris.analysis.cartography.project(self.cube,
                                              self.target_proj,
                                              nx=-200, ny=200)
        with self.assertRaises(ValueError):
            iris.analysis.cartography.project(self.cube,
                                              self.target_proj,
                                              nx=200, ny='abc')

    def test_missing_latlon(self):
        cube = self.cube.copy()
        cube.remove_coord('grid_latitude')
        with self.assertRaises(ValueError):
            iris.analysis.cartography.project(cube, self.target_proj)
        cube = self.cube.copy()
        cube.remove_coord('grid_longitude')
        with self.assertRaises(ValueError):
            iris.analysis.cartography.project(cube, self.target_proj)
        self.cube.remove_coord('grid_longitude')
        self.cube.remove_coord('grid_latitude')
        with self.assertRaises(ValueError):
            iris.analysis.cartography.project(self.cube, self.target_proj)

    def test_default_resolution(self):
        new_cube, extent = iris.analysis.cartography.project(self.cube,
                                                             self.target_proj)
        self.assertEqual(new_cube.shape, self.cube.shape)

    @tests.skip_data
    @tests.skip_plot
    def test_cartopy_projection(self):
        cube = iris.load_cube(tests.get_data_path(('PP', 'aPPglob1',
                                                   'global.pp')))
        projections = {}
        projections['RotatedPole'] = ccrs.RotatedPole(pole_longitude=177.5,
                                                      pole_latitude=37.5)
        projections['Robinson'] = ccrs.Robinson()
        projections['PlateCarree'] = ccrs.PlateCarree()
        projections['NorthPolarStereo'] = ccrs.NorthPolarStereo()
        projections['Orthographic'] = ccrs.Orthographic(central_longitude=-90,
                                                        central_latitude=45)
        projections['InterruptedGoodeHomolosine'] = ccrs.InterruptedGoodeHomolosine()
        projections['LambertCylindrical'] = ccrs.LambertCylindrical()

        # Set up figure
        fig = plt.figure(figsize=(10, 10))
        gs = matplotlib.gridspec.GridSpec(nrows=3, ncols=3, hspace=1.5, wspace=0.5)
        for subplot_spec, name in zip(gs, sorted(projections)):
            target_proj = projections[name]
            # Set up axes and title
            ax = plt.subplot(subplot_spec, frameon=False, projection=target_proj)
            ax.set_title(name)
            # Transform cube to target projection
            new_cube, extent = iris.analysis.cartography.project(cube, target_proj,
                                                                 nx=150, ny=150)
            # Plot
            plt.pcolor(new_cube.coord('projection_x_coordinate').points,
                       new_cube.coord('projection_y_coordinate').points,
                       new_cube.data)
            # Add coastlines
            ax.coastlines()

        # Tighten up layout
        gs.tight_layout(plt.gcf())

        # Verify resulting plot
        self.check_graphic(tol=1.0)

    @tests.skip_data
    def test_no_coord_system(self):
        cube = iris.load_cube(tests.get_data_path(('PP', 'aPPglob1', 'global.pp')))
        cube.coord('longitude').coord_system = None
        cube.coord('latitude').coord_system = None
        new_cube, extent = iris.analysis.cartography.project(cube,
                                                             self.target_proj)
        self.assertCML(new_cube,
                       ('analysis', 'project', 'default_source_cs.cml'))


if __name__ == "__main__":
    tests.main()
