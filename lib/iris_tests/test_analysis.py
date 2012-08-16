# (C) British Crown Copyright 2010 - 2012, Met Office
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


from __future__ import division
# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import os
import unittest
import zlib

import matplotlib.pyplot as plt
import numpy
import shapely.geometry

import iris
import iris.analysis.cartography
import iris.analysis.geometry
import iris.analysis.maths
import iris.coord_systems
import iris.coords
import iris.cube
import iris.tests.stock


class TestAnalysisCubeCoordComparison(tests.IrisTest):
    def assertComparisonDict(self, comarison_dict, reference_filename):
        string = ''
        for key, coord_groups in comarison_dict.iteritems():
            string += ('%40s  ' % key)
            names = [[coord.name() if coord is not None else 'None' for coord in coords] for coords in coord_groups]
            string += str(sorted(names))
            string += '\n'
        self.assertString(string, reference_filename)
        
    def test_coord_comparison(self):
        cube1 = iris.cube.Cube(numpy.zeros((41, 41)))
        lonlat_cs = iris.coord_systems.GeogCS()
        lon_points1 = -180 + 4.5 * numpy.arange(41, dtype=numpy.float32) 
        lat_points = -90 + 4.5 * numpy.arange(41, dtype=numpy.float32) 
        cube1.add_dim_coord(iris.coords.DimCoord(lon_points1, 'longitude', units='degrees', coord_system=lonlat_cs), 0) 
        cube1.add_dim_coord(iris.coords.DimCoord(lat_points, 'latitude', units='degrees', coord_system=lonlat_cs), 1)
        cube1.add_aux_coord(iris.coords.AuxCoord(0, long_name='z')) 
        cube1.add_aux_coord(iris.coords.AuxCoord(['foobar'], long_name='f', units='no_unit')) 
    
        cube2 = iris.cube.Cube(numpy.zeros((41, 41, 5)))
        lonlat_cs = iris.coord_systems.GeogCS()
        lon_points2 = -160 + 4.5 * numpy.arange(41, dtype=numpy.float32)
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
        data = numpy.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]], dtype=numpy.float32)
        weights = numpy.array([[9, 8, 7],[6, 5, 4],[3, 2, 1]], dtype=numpy.float32)
        
        cube = iris.cube.Cube(data, long_name="test_data", units="1") 
        hcs = iris.coord_systems.GeogCS()
        lat_coord = iris.coords.DimCoord(numpy.array([1, 2, 3], dtype=numpy.float32), long_name="lat", units="1", coord_system=hcs) 
        lon_coord = iris.coords.DimCoord(numpy.array([1, 2, 3], dtype=numpy.float32), long_name="lon", units="1", coord_system=hcs)
        cube.add_dim_coord(lat_coord, 0) 
        cube.add_dim_coord(lon_coord, 1) 
        cube.add_aux_coord(iris.coords.AuxCoord(numpy.arange(3, dtype=numpy.float32), long_name="dummy", units=1), 1)
        self.assertCML(cube, ('analysis', 'weighted_mean_source.cml'))

        a = cube.collapsed('lat', iris.analysis.MEAN, weights=weights)
        self.assertCMLApproxData(a, ('analysis', 'weighted_mean_lat.cml'))
        
        b = cube.collapsed(lon_coord, iris.analysis.MEAN, weights=weights)
        b.data = numpy.asarray(b.data)
        self.assertCMLApproxData(b, ('analysis', 'weighted_mean_lon.cml'))
        self.assertEquals(b.coord("dummy").shape, (1,))

        # test collapsing multiple coordinates (and the fact that one of the coordinates isn't the same coordinate instance as on the cube)
        c = cube.collapsed([lat_coord[:], lon_coord], iris.analysis.MEAN, weights=weights)
        self.assertCMLApproxData(c, ('analysis', 'weighted_mean_latlon.cml'))
        self.assertEquals(c.coord("dummy").shape, (1,))
        
        # Check new coord bounds - made from points
        self.assertArrayEqual(c.coord('lat').bounds, [[1, 3]])
        
        # Check new coord bounds - made from bounds
        cube.coord('lat').bounds = [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]]
        c = cube.collapsed(['lat','lon'], iris.analysis.MEAN, weights=weights)
        self.assertArrayEqual(c.coord('lat').bounds, [[0.5, 3.5]])
        cube.coord('lat').bounds = None

        # Check there was no residual change
        self.assertCML(cube, ('analysis', 'weighted_mean_source.cml'))

    @iris.tests.skip_data
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
        # check it's a 1D, scalar cube (actually, it should really be 0D)!
        self.assertEquals(g.data.shape, (1,))
        # check the value - pp_area_avg's result of 287.927 differs by factor of 1.00002959
        numpy.testing.assert_approx_equal(g.data[0], 287.935, significant=5)
        
        #check we get summed weights even if we don't give any
        h,summed_weights = e.collapsed('latitude', iris.analysis.MEAN, returned=True)
        assert(summed_weights is not None)
        
        # Check there was no residual change
        e.coord('latitude').bounds = None
        e.coord('longitude').bounds = None
        self.assertCML(e, ('analysis', 'weighted_mean_original.cml'))
        
        # Test collapsing of missing coord
        self.assertRaises(iris.exceptions.CoordinateNotFoundError, e.collapsed, 'platitude', iris.analysis.MEAN)

        # Test collpasing of non data coord
        self.assertRaises(iris.exceptions.CoordinateCollapseError, e.collapsed, 'pressure', iris.analysis.MEAN)

@iris.tests.skip_data
class TestAnalysisBasic(tests.IrisTest):
    def setUp(self):
        file = tests.get_data_path(('PP', 'aPProt1', 'rotatedMHtimecube.pp'))
        cubes = iris.load(file)        
        self.cube = cubes[0]
        self.assertCML(self.cube, ('analysis', 'original.cml'))
        
    def _common(self, name, aggregate, original_name='original_common.cml', *args, **kwargs):
        self.cube.data = self.cube.data.astype(numpy.float64)
           
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

    def test_duplicate_coords(self):
        self.assertRaises(ValueError, tests.stock.track_1d, duplicate_x=True)

    def test_lat_lon_range(self):
        # Test with non-circular longitude
        result_non_circ = iris.analysis.cartography.lat_lon_range(self.cube)
        numpy.testing.assert_array_almost_equal(result_non_circ, ((15, 77), (-88, 72)), decimal=0)

        # Set longitude to be circular
        self.cube.coord('grid_longitude').circular = True
        result_circ = iris.analysis.cartography.lat_lon_range(self.cube)
        
        # lon range of circular coord grid_longitude will be approx -88 + 360 = 272
        numpy.testing.assert_array_almost_equal(result_circ, ((15, 77), (-88, 272)), decimal=0)


class TestMissingData(tests.IrisTest):
    def setUp(self):
        self.cube_with_nan = tests.stock.simple_2d()
        
        data = self.cube_with_nan.data.astype(numpy.float32)
        self.cube_with_nan.data = data.copy()
        self.cube_with_nan.data[1, 0] = numpy.nan
        self.cube_with_nan.data[2, 2] = numpy.nan
        self.cube_with_nan.data[2, 3] = numpy.nan
        
        self.cube_with_mask = tests.stock.simple_2d()
        self.cube_with_mask.data = numpy.ma.array(self.cube_with_nan.data, 
                                                  mask=numpy.isnan(self.cube_with_nan.data))
        
    def test_max(self):
        cube = self.cube_with_nan.collapsed('foo', iris.analysis.MAX)
        numpy.testing.assert_array_equal(cube.data, numpy.array([3, numpy.nan, numpy.nan]))
        
        cube = self.cube_with_mask.collapsed('foo', iris.analysis.MAX)
        numpy.testing.assert_array_equal(cube.data, numpy.array([3, 7, 9]))

    def test_min(self):
        cube = self.cube_with_nan.collapsed('foo', iris.analysis.MIN)
        numpy.testing.assert_array_equal(cube.data, numpy.array([0, numpy.nan, numpy.nan]))
        
        cube = self.cube_with_mask.collapsed('foo', iris.analysis.MIN)
        numpy.testing.assert_array_equal(cube.data, numpy.array([0, 5, 8]))

    def test_sum(self):
        cube = self.cube_with_nan.collapsed('foo', iris.analysis.SUM)
        numpy.testing.assert_array_equal(cube.data, numpy.array([6, numpy.nan, numpy.nan]))
        
        cube = self.cube_with_mask.collapsed('foo', iris.analysis.SUM)
        numpy.testing.assert_array_equal(cube.data, numpy.array([6, 18, 17]))


class TestAggregators(tests.IrisTest):
    def test_percentile_1d(self):
        cube = tests.stock.simple_1d()

        first_quartile = cube.collapsed('foo', iris.analysis.PERCENTILE, percent=25)
        numpy.testing.assert_array_almost_equal(first_quartile.data, numpy.array([2.5], dtype=numpy.float32))
        self.assertCML(first_quartile, ('analysis', 'first_quartile_foo_1d.cml'), checksum=False)

        third_quartile = cube.collapsed('foo', iris.analysis.PERCENTILE, percent=75)
        numpy.testing.assert_array_almost_equal(third_quartile.data, numpy.array([7.5], dtype=numpy.float32))
        self.assertCML(third_quartile, ('analysis', 'third_quartile_foo_1d.cml'), checksum=False)

    def test_percentile_2d(self):
        cube = tests.stock.simple_2d()

        first_quartile = cube.collapsed('foo', iris.analysis.PERCENTILE, percent=25)
        numpy.testing.assert_array_almost_equal(first_quartile.data, numpy.array([0.75, 4.75, 8.75], dtype=numpy.float32))
        self.assertCML(first_quartile, ('analysis', 'first_quartile_foo_2d.cml'), checksum=False)

        first_quartile = cube.collapsed(('foo', 'bar'), iris.analysis.PERCENTILE, percent=25)
        numpy.testing.assert_array_almost_equal(first_quartile.data, numpy.array([2.75], dtype=numpy.float32))
        self.assertCML(first_quartile, ('analysis', 'first_quartile_foo_bar_2d.cml'), checksum=False)

    def test_proportion(self):
        cube = tests.stock.simple_1d()
        r = cube.data >= 5
        gt5 = cube.collapsed('foo', iris.analysis.PROPORTION, function=lambda val: val >= 5)
        numpy.testing.assert_array_almost_equal(gt5.data, numpy.array([6/11.]))
        self.assertCML(gt5, ('analysis', 'proportion_foo_1d.cml'), checksum=False)
        
    def test_proportion_2d(self):
        cube = tests.stock.simple_2d()

        gt6 = cube.collapsed('foo', iris.analysis.PROPORTION, function=lambda val: val >= 6)
        numpy.testing.assert_array_almost_equal(gt6.data, numpy.array([0, 0.5, 1], dtype=numpy.float32))
        self.assertCML(gt6, ('analysis', 'proportion_foo_2d.cml'), checksum=False)
        
        gt6 = cube.collapsed('bar', iris.analysis.PROPORTION, function=lambda val: val >= 6)
        numpy.testing.assert_array_almost_equal(gt6.data, numpy.array([1/3, 1/3, 2/3, 2/3], dtype=numpy.float32))
        self.assertCML(gt6, ('analysis', 'proportion_bar_2d.cml'), checksum=False)
        
        gt6 = cube.collapsed(('foo', 'bar'), iris.analysis.PROPORTION, function=lambda val: val >= 6)
        numpy.testing.assert_array_almost_equal(gt6.data, numpy.array([0.5], dtype=numpy.float32))
        self.assertCML(gt6, ('analysis', 'proportion_foo_bar_2d.cml'), checksum=False)
        
        # mask the data
        cube.data = numpy.ma.array(cube.data, mask=cube.data % 2)
        cube.data.mask[1, 2] = True
        gt6_masked = cube.collapsed('bar', iris.analysis.PROPORTION, function=lambda val: val >= 6)
        numpy.testing.assert_array_almost_equal(gt6_masked.data, numpy.ma.array([1/3, None, 1/2, None], 
                                                                                mask=[False, True, False, True], 
                                                                                dtype=numpy.float32))
        self.assertCML(gt6_masked, ('analysis', 'proportion_foo_2d_masked.cml'), checksum=False)
                
    def test_count(self):
        cube = tests.stock.simple_1d()
        r = cube.data >= 5
        gt5 = cube.collapsed('foo', iris.analysis.COUNT, function=lambda val: val >= 5)
        numpy.testing.assert_array_almost_equal(gt5.data, numpy.array([6]))
        self.assertCML(gt5, ('analysis', 'count_foo_1d.cml'), checksum=False)
        
    def test_count_2d(self):
        cube = tests.stock.simple_2d()

        gt6 = cube.collapsed('foo', iris.analysis.COUNT, function=lambda val: val >= 6)
        numpy.testing.assert_array_almost_equal(gt6.data, numpy.array([0, 2, 4], dtype=numpy.float32))
        self.assertCML(gt6, ('analysis', 'count_foo_2d.cml'), checksum=False)
        
        gt6 = cube.collapsed('bar', iris.analysis.COUNT, function=lambda val: val >= 6)
        numpy.testing.assert_array_almost_equal(gt6.data, numpy.array([1, 1, 2, 2], dtype=numpy.float32))
        self.assertCML(gt6, ('analysis', 'count_bar_2d.cml'), checksum=False)
        
        gt6 = cube.collapsed(('foo', 'bar'), iris.analysis.COUNT, function=lambda val: val >= 6)
        numpy.testing.assert_array_almost_equal(gt6.data, numpy.array([6], dtype=numpy.float32))
        self.assertCML(gt6, ('analysis', 'count_foo_bar_2d.cml'), checksum=False)


@iris.tests.skip_data
class TestRotatedPole(tests.IrisTest):
    def _check_both_conversions(self, cube):
        lats, lons = iris.analysis.cartography.get_lat_lon_grids(cube)
        plt.scatter(lons, lats)
        self.check_graphic()

        n_pole = cube.coord_system('RotatedGeogCS').grid_north_pole
        rlons, rlats = iris.analysis.cartography.rotate_pole(lons, lats, n_pole.lon, n_pole.lat)
        plt.scatter(rlons, rlats)
        self.check_graphic()

    def test_all(self):
        path = tests.get_data_path(('PP', 'ukVorog', 'ukv_orog_refonly.pp'))
        master_cube = iris.load_strict(path)

        # Check overall behaviour
        cube = master_cube[::10, ::10]
        self._check_both_conversions(cube)

        # Check numerical stability
        cube = master_cube[210:238, 424:450]
        self._check_both_conversions(cube)

@iris.tests.skip_data
class TestAreaWeights(tests.IrisTest):
    def setUp(self):
        self.cube = iris.tests.stock.simple_pp()
        self.assertCML(self.cube, ('analysis', 'areaweights_original.cml'))

    def test_area_weights(self):
        self.cube.coord('latitude').guess_bounds()
        self.cube.coord('longitude').guess_bounds()
        area_weights = iris.analysis.cartography.area_weights(self.cube)
        self.assertEquals(zlib.crc32(area_weights), 253962218)

        # Check there was no residual change
        self.cube.coord('latitude').bounds = None
        self.cube.coord('longitude').bounds = None
        self.assertCML(self.cube, ('analysis', 'areaweights_original.cml'))

    def test_quadrant_area(self):
        
        degrees = iris.unit.Unit("degrees")
        radians = iris.unit.Unit("radians")
        
        def lon2radlon(lons):
            return [degrees.convert(lon, radians) for lon in lons]

        def lat2radcolat(lats):
            return [degrees.convert(lat+90, radians) for lat in lats]
        
        lats = numpy.array([lat2radcolat([-80, -70])])
        lons = numpy.array([lon2radlon([0, 10])])
        area = iris.analysis.cartography._quadrant_area(lats, lons, iris.analysis.cartography.DEFAULT_SPHERICAL_EARTH_RADIUS)
        self.assertAlmostEquals(area, [[319251.84598076]])
    
        lats = numpy.array([lat2radcolat([0, 10])])
        lons = numpy.array([lon2radlon([0, 10])])
        area = iris.analysis.cartography._quadrant_area(lats, lons, iris.analysis.cartography.DEFAULT_SPHERICAL_EARTH_RADIUS)
        self.assertAlmostEquals(area, [[1228800.59385144]])

        lats = numpy.array([lat2radcolat([10, 0])])
        lons = numpy.array([lon2radlon([0, 10])])
        area = iris.analysis.cartography._quadrant_area(lats, lons, iris.analysis.cartography.DEFAULT_SPHERICAL_EARTH_RADIUS)
        self.assertAlmostEquals(area, [[1228800.59385144]])
    
        lats = numpy.array([lat2radcolat([70, 80])])
        lons = numpy.array([lon2radlon([0, 10])])
        area = iris.analysis.cartography._quadrant_area(lats, lons, iris.analysis.cartography.DEFAULT_SPHERICAL_EARTH_RADIUS)
        self.assertAlmostEquals(area, [[319251.84598076]])
    
        lats = numpy.array([lat2radcolat([-80, -70]), lat2radcolat([0, 10]), lat2radcolat([70, 80])])
        lons = numpy.array([lon2radlon([0, 10])])
        area = iris.analysis.cartography._quadrant_area(lats, lons, iris.analysis.cartography.DEFAULT_SPHERICAL_EARTH_RADIUS)
        self.assertAlmostEquals(area[0], [319251.84598076])
        self.assertAlmostEquals(area[1], [1228800.59385144])
        self.assertAlmostEquals(area[2], [319251.84598076])


class TestRollingWindow(tests.IrisTest):
    def setUp(self):
        # XXX Comes from test_aggregated_by
        cube = iris.cube.Cube(numpy.array([[6, 10, 12, 18], [8, 12, 14, 20], [18, 12, 10, 6]]), long_name='temperature', units='kelvin')
        cube.add_dim_coord(iris.coords.DimCoord(numpy.array([0, 5, 10], dtype=numpy.float64), 'latitude', units='degrees'), 0)
        cube.add_dim_coord(iris.coords.DimCoord(numpy.array([0, 2, 4, 6], dtype=numpy.float64), 'longitude', units='degrees'), 1)

        self.cube = cube

    def test_non_mean_operator(self):
        res_cube = self.cube.rolling_window('longitude', iris.analysis.MAX, window=2)
        expected_result = numpy.array([[10, 12, 18],
                                       [12, 14, 20],
                                       [18, 12, 10]], dtype=numpy.float64)
        self.assertArrayEqual(expected_result, res_cube.data)

    def test_longitude_simple(self):
        res_cube = self.cube.rolling_window('longitude', iris.analysis.MEAN, window=2)
        
        expected_result = numpy.array([[  8.,  11.,  15.],
                                      [ 10.,  13.,  17.],
                                      [ 15.,  11.,   8.]], dtype=numpy.float64)
        
        self.assertArrayEqual(expected_result, res_cube.data)
        
        self.assertCML(res_cube, ('analysis', 'rolling_window', 'simple_longitude.cml'))
        
        self.assertRaises(ValueError, self.cube.rolling_window, 'longitude', iris.analysis.MEAN, window=0)
        
    def test_longitude_circular(self):
        cube = self.cube
        cube.coord('longitude').circular = True
        self.assertRaises(iris.exceptions.NotYetImplementedError, self.cube.rolling_window, 'longitude', iris.analysis.MEAN, window=0)
                
    def test_different_length_windows(self):
        res_cube = self.cube.rolling_window('longitude', iris.analysis.MEAN, window=4)
        
        expected_result = numpy.array([[ 11.5],
                                       [ 13.5],
                                       [ 11.5]], dtype=numpy.float64)
        
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
        
        expected_result = numpy.array([[  7.,  11.,  13.,  19.],
                                       [ 13.,  12.,  12.,  13.]], dtype=numpy.float64)
        
        self.assertArrayEqual(expected_result, res_cube.data)
        
        self.assertCML(res_cube, ('analysis', 'rolling_window', 'simple_latitude.cml'))

    def test_returned_weights(self):
        self.assertRaises(ValueError, self.cube.rolling_window, 'longitude', iris.analysis.MEAN, window=2, returned=True) 
        self.assertRaises(ValueError, self.cube.rolling_window, 'longitude', iris.analysis.MEAN, window=2, weights=[1,2,3,4,5]) 


class TestGeometry(tests.IrisTest):

    @iris.tests.skip_data
    def test_distinct_xy(self):
        cube = iris.tests.stock.simple_pp()
        cube = cube[:4, :4]
        lon = cube.coord('longitude')
        lat = cube.coord('latitude')
        lon.guess_bounds()
        lat.guess_bounds()
        from iris.fileformats.rules import regular_step
        quarter = abs(regular_step(lon) * regular_step(lat) * 0.25)
        half = abs(regular_step(lon) * regular_step(lat) * 0.5)
        minx = 3.7499990463256836
        maxx = 7.499998092651367
        miny = 84.99998474121094
        maxy = 89.99998474121094
        geometry = shapely.geometry.box(minx, miny, maxx, maxy)
        weights = iris.analysis.geometry.geometry_area_weights(cube, geometry)
        target = numpy.array([
            [0, quarter, quarter, 0],
            [0, half,    half,    0],
            [0, quarter, quarter, 0],
            [0, 0,       0,       0]])
        self.assertTrue(numpy.allclose(weights, target))

    def test_shared_xy(self):
        cube = tests.stock.track_1d()
        geometry = shapely.geometry.box(1, 4, 3.5, 7)
        weights = iris.analysis.geometry.geometry_area_weights(cube, geometry)
        target = numpy.array([0, 0, 2, 0.5, 0, 0, 0, 0, 0, 0, 0])
        self.assertTrue(numpy.allclose(weights, target))


if __name__ == "__main__":
    unittest.main()
    
