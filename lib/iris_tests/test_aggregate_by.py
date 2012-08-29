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


# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import numpy as np
import unittest

import iris
import iris.analysis
import iris.coord_systems
import iris.coords


class TestAggregateBy(tests.IrisTest):
    
    def setUp(self):
        #
        # common
        #
        cs_latlon = iris.coord_systems.GeogCS()
        points = np.arange(3, dtype=np.float32) * 3
        coord_lat = iris.coords.DimCoord(points, 'latitude', units='degrees', coord_system=cs_latlon)
        coord_lon = iris.coords.DimCoord(points, 'longitude', units='degrees', coord_system=cs_latlon)
        
        #
        # single coordinate aggregate-by
        #
        a = np.arange(9, dtype=np.int32).reshape(3, 3) + 1
        b = np.arange(36, dtype=np.int32).reshape(36, 1, 1)
        data = b * a

        self.cube_single = iris.cube.Cube(data, long_name='temperature', units='kelvin')

        z_points = np.array([1,2,2,3,3,3,4,4,4,4,5,5,5,5,5,6,6,6,6,6,6,
                             7,7,7,7,7,7,7,8,8,8,8,8,8,8,8], dtype=np.int32)
        self.coord_z_single = iris.coords.AuxCoord(z_points, long_name='height', units='m')
        
        self.cube_single.add_aux_coord(self.coord_z_single, 0)
        self.cube_single.add_dim_coord(coord_lon, 1)
        self.cube_single.add_dim_coord(coord_lat, 2)
    
        #
        # multi coordinate aggregate-by
        #
        a = np.arange(9, dtype=np.int32).reshape(3, 3) + 1
        b = np.arange(20, dtype=np.int32).reshape(20, 1, 1)
        data = b * a
        
        self.cube_multi = iris.cube.Cube(data, long_name='temperature', units='kelvin')

        z1_points = np.array([1,1,1, 1,1, 2, 2,2,2, 3, 4,4, 4,4,4, 1,  5, 5,  2,2], dtype=np.int32)
        self.coord_z1_multi = iris.coords.AuxCoord(z1_points, long_name='height', units='m')
        z2_points = np.array([1,1,1, 3,3, 3, 5,5,5, 7, 7,7, 9,9,9, 11, 11,11, 3,3], dtype=np.int32)
        self.coord_z2_multi = iris.coords.AuxCoord(z2_points, long_name='level', units='1')
        
        self.cube_multi.add_aux_coord(self.coord_z1_multi, 0)
        self.cube_multi.add_aux_coord(self.coord_z2_multi, 0)
        self.cube_multi.add_dim_coord(coord_lon.copy(), 1)
        self.cube_multi.add_dim_coord(coord_lat.copy(), 2)

        #
        # expected data results
        #
        self.single_expected = np.array([[[0., 0., 0.],      [0., 0., 0.],        [0., 0., 0.]],
                                         [[1.5, 3., 4.5],    [6., 7.5, 9.],       [10.5, 12., 13.5]],
                                         [[4. , 8., 12.],    [16., 20., 24.],     [28., 32., 36.]],
                                         [[7.5, 15., 22.5],  [30., 37.5, 45.],    [52.5, 60., 67.5]],
                                         [[12., 24., 36.],   [48., 60., 72.],     [84., 96., 108.]],
                                         [[17.5, 35., 52.5], [70., 87.5, 105.],   [122.5, 140., 157.5]],
                                         [[24., 48., 72.],   [96., 120., 144.],   [168., 192., 216.]],
                                         [[31.5, 63., 94.5], [126., 157.5, 189.], [220.5, 252., 283.5]]], dtype=np.float64)
    
        self.multi_expected = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                                        [[3.5, 7., 10.5], [14., 17.5, 21.], [24.5, 28., 31.5]],
                                        [[14., 28., 42.], [56., 70., 84.], [98., 112., 126.]],
                                        [[7., 14., 21.], [28., 35., 42.], [49., 56., 63.]],
                                        [[9., 18., 27.], [36., 45., 54.], [63., 72., 81.]],
                                        [[10.5, 21., 31.5], [42., 52.5, 63.], [73.5, 84., 94.5]],
                                        [[13., 26., 39.], [52., 65., 78.], [91., 104., 117.]],
                                        [[15., 30., 45.], [60., 75., 90.], [105., 120., 135.]],
                                        [[16.5, 33., 49.5], [66., 82.5, 99.], [115.5, 132., 148.5]]], dtype=np.float64)
    
    def test_single(self):
        # group-by with single coordinate name.
        aggregateby_cube = self.cube_single.aggregated_by('height', iris.analysis.MEAN)
        self.assertCML(aggregateby_cube, ('analysis', 'aggregated_by', 'single.cml'))

        # group-by with single coordinate.
        aggregateby_cube = self.cube_single.aggregated_by(self.coord_z_single, iris.analysis.MEAN)
        self.assertCML(aggregateby_cube, ('analysis', 'aggregated_by', 'single.cml'))
        
        np.testing.assert_almost_equal(aggregateby_cube.data, self.single_expected)

    def test_single_shared(self):
        z2_points = np.arange(36, dtype=np.int32)
        coord_z2 = iris.coords.AuxCoord(z2_points, long_name='model_level', units='1')
        self.cube_single.add_aux_coord(coord_z2, 0)

        # group-by with single coordinate name on shared axis.
        aggregateby_cube = self.cube_single.aggregated_by('height', iris.analysis.MEAN)
        self.assertCML(aggregateby_cube, ('analysis', 'aggregated_by', 'single_shared.cml'))

        # group-by with single coordinate on shared axis.
        aggregateby_cube = self.cube_single.aggregated_by(self.coord_z_single, iris.analysis.MEAN)
        self.assertCML(aggregateby_cube, ('analysis', 'aggregated_by', 'single_shared.cml'))

        np.testing.assert_almost_equal(aggregateby_cube.data, self.single_expected)

    def test_multi(self):
        # group-by with multiple coordinate names.
        aggregateby_cube = self.cube_multi.aggregated_by(['height', 'level'], iris.analysis.MEAN)
        self.assertCML(aggregateby_cube, ('analysis', 'aggregated_by', 'multi.cml'))

        # group-by with multiple coordinate names (different order).
        aggregateby_cube = self.cube_multi.aggregated_by(['level', 'height'], iris.analysis.MEAN)
        self.assertCML(aggregateby_cube, ('analysis', 'aggregated_by', 'multi.cml'))

        # group-by with multiple coordinates.
        aggregateby_cube = self.cube_multi.aggregated_by([self.coord_z1_multi, self.coord_z2_multi], iris.analysis.MEAN)
        self.assertCML(aggregateby_cube, ('analysis', 'aggregated_by', 'multi.cml'))

        # group-by with multiple coordinates (different order).
        aggregateby_cube = self.cube_multi.aggregated_by([self.coord_z2_multi, self.coord_z1_multi], iris.analysis.MEAN)
        self.assertCML(aggregateby_cube, ('analysis', 'aggregated_by', 'multi.cml'))

        np.testing.assert_almost_equal(aggregateby_cube.data, self.multi_expected)

    def test_multi_shared(self):
        z3_points = np.arange(20, dtype=np.int32)
        coord_z3 = iris.coords.AuxCoord(z3_points, long_name='sigma', units='1')
        z4_points = np.arange(19, -1, -1, dtype=np.int32)
        coord_z4 = iris.coords.AuxCoord(z4_points, long_name='gamma', units='1')
        
        self.cube_multi.add_aux_coord(coord_z3, 0) 
        self.cube_multi.add_aux_coord(coord_z4, 0)

        # group-by with multiple coordinate names on shared axis.
        aggregateby_cube = self.cube_multi.aggregated_by(['height', 'level'], iris.analysis.MEAN)
        self.assertCML(aggregateby_cube, ('analysis', 'aggregated_by', 'multi_shared.cml'))

        # group-by with multiple coordinate names on shared axis (different order).
        aggregateby_cube = self.cube_multi.aggregated_by(['level', 'height'], iris.analysis.MEAN)
        self.assertCML(aggregateby_cube, ('analysis', 'aggregated_by', 'multi_shared.cml'))

        # group-by with multiple coordinates on shared axis.
        aggregateby_cube = self.cube_multi.aggregated_by([self.coord_z1_multi, self.coord_z2_multi], iris.analysis.MEAN)
        self.assertCML(aggregateby_cube, ('analysis', 'aggregated_by', 'multi_shared.cml'))

        # group-by with multiple coordinates on shared axis (different order).
        aggregateby_cube = self.cube_multi.aggregated_by([self.coord_z2_multi, self.coord_z1_multi], iris.analysis.MEAN)
        self.assertCML(aggregateby_cube, ('analysis', 'aggregated_by', 'multi_shared.cml'))

        np.testing.assert_almost_equal(aggregateby_cube.data, self.multi_expected)

    def test_easy(self):
        data = np.array([[6, 10, 12, 18], [8, 12, 14, 20], [18, 12, 10, 6]], dtype=np.float32)
        cube = iris.cube.Cube(data, long_name='temperature', units='kelvin')

        llcs = iris.coord_systems.GeogCS()
        cube.add_aux_coord(iris.coords.AuxCoord(np.array([0, 0, 10], dtype=np.float32),
                                                'latitude', units='degrees', coord_system=llcs), 0)
        cube.add_aux_coord(iris.coords.AuxCoord(np.array([0, 0, 10, 10], dtype=np.float32), 
                                                'longitude', units='degrees', coord_system=llcs),1)


        #
        # Easy mean aggregate test by each coordinate.
        #
        aggregateby_cube = cube.aggregated_by('longitude', iris.analysis.MEAN)
        np.testing.assert_almost_equal(aggregateby_cube.data, np.array([[8., 15.], [10., 17.], [15., 8.]], dtype=np.float32))
        self.assertCML(aggregateby_cube, ('analysis', 'aggregated_by', 'easy.cml'), checksum=False)

        aggregateby_cube = cube.aggregated_by('latitude', iris.analysis.MEAN)
        np.testing.assert_almost_equal(aggregateby_cube.data, np.array([[7., 11., 13., 19.], [18., 12., 10., 6.]], dtype=np.float32))

        #
        # Easy max aggregate test by each coordinate.
        #
        aggregateby_cube = cube.aggregated_by('longitude', iris.analysis.MAX)
        np.testing.assert_almost_equal(aggregateby_cube.data, np.array([[10., 18.], [12., 20.], [18., 10.]], dtype=np.float32))

        aggregateby_cube = cube.aggregated_by('latitude', iris.analysis.MAX)
        np.testing.assert_almost_equal(aggregateby_cube.data, np.array([[8., 12., 14., 20.], [18., 12., 10., 6.]], dtype=np.float32))

        #
        # Easy sum aggregate test by each coordinate.
        #
        aggregateby_cube = cube.aggregated_by('longitude', iris.analysis.SUM)
        np.testing.assert_almost_equal(aggregateby_cube.data, np.array([[16., 30.], [20., 34.], [30., 16.]], dtype=np.float32))
        
        aggregateby_cube = cube.aggregated_by('latitude', iris.analysis.SUM)
        np.testing.assert_almost_equal(aggregateby_cube.data, np.array([[14., 22., 26., 38.], [18., 12., 10., 6.]], dtype=np.float32))

        #
        # Easy percentile aggregate test by each coordinate.
        #
        aggregateby_cube = cube.aggregated_by('longitude', iris.analysis.PERCENTILE, percent=25)
        np.testing.assert_almost_equal(aggregateby_cube.data, np.array([[7.0, 13.5], [9.0, 15.5], [13.5, 7.0]], dtype=np.float32))
        
        aggregateby_cube = cube.aggregated_by('latitude', iris.analysis.PERCENTILE, percent=25)
        np.testing.assert_almost_equal(aggregateby_cube.data, np.array([[6.5, 10.5, 12.5, 18.5], [18., 12., 10., 6.]], dtype=np.float32))

    def test_returned_weights(self):
        self.assertRaises(ValueError, self.cube_single.aggregated_by, 'height', iris.analysis.MEAN, returned=True) 
        self.assertRaises(ValueError, self.cube_single.aggregated_by, 'height', iris.analysis.MEAN, weights=[1,2,3,4,5]) 


if __name__ == '__main__':
    unittest.main()
