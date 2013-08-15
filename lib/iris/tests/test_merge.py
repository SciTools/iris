# (C) British Crown Copyright 2010 - 2013, Met Office
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
Test the cube merging mechanism.

"""

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import numpy as np

import iris
import iris.cube
import iris.exceptions
from iris.coords import DimCoord, AuxCoord
import iris.coords
import iris.tests.stock


class TestMixin(object):
    """
    Mix-in class for attributes & utilities common to these test cases.
    
    """
    def test_normal_cubes(self):
        cubes = iris.load(self._data_path)
        self.assertEqual(len(cubes), self._num_cubes)
        names = ['forecast_period', 'forecast_reference_time', 'level_height', 'model_level_number', 'sigma', 'source']
        axes = ['forecast_period', 'rt', 'z', 'z', 'z', 'source']
        self.assertCML(cubes, ['merge', self._prefix + '.cml'])
        
    def test_remerge(self):
        # After the merge process the coordinates within each cube can be in a
        # different order. Until that changes we can't compare the cubes
        # directly or with the CML ... so we just make sure the count stays
        # the same.
        cubes = iris.load(self._data_path)
        cubes2 = cubes.merge()
        self.assertEqual(len(cubes), len(cubes2))

    def test_duplication(self):
        cubes = iris.load(self._data_path)
        self.assertRaises(iris.exceptions.DuplicateDataError, (cubes + cubes).merge)
        cubes2 = (cubes + cubes).merge(unique=False)
        self.assertEqual(len(cubes2), 2 * len(cubes))


@iris.tests.skip_data
class TestSingleCube(tests.IrisTest, TestMixin):
    def setUp(self):
        self._data_path = tests.get_data_path(('PP', 'globClim1', 'theta.pp'))
        self._num_cubes = 1
        self._prefix = 'theta'


@iris.tests.skip_data
class TestMultiCube(tests.IrisTest, TestMixin):
    def setUp(self):
        self._data_path = tests.get_data_path(('PP', 'globClim1', 'dec_subset.pp'))
        self._num_cubes = 4
        self._prefix = 'dec'

    def test_coord_attributes(self):
        def custom_coord_callback(cube, field, filename):
            cube.coord('time').attributes['monty'] = 'python'
            cube.coord('time').attributes['brain'] = 'hurts'
        
        # Load slices, decorating a coord with custom attributes
        cubes = iris.load_raw(self._data_path, callback=custom_coord_callback)
        # Merge
        merged = iris.cube.CubeList._extract_and_merge(cubes, constraints=None, strict=False, merge_unique=False)
        # Check the custom attributes are in the merged cube
        for cube in merged:
            assert(cube.coord('time').attributes['monty'] == 'python')
            assert(cube.coord('time').attributes['brain'] == 'hurts')
        

@iris.tests.skip_data
class TestColpex(tests.IrisTest):
    def setUp(self):
        self._data_path = tests.get_data_path(('PP', 'COLPEX', 'small_colpex_theta_p_alt.pp'))

    def test_colpex(self):
        cubes = iris.load(self._data_path)
        self.assertEqual(len(cubes), 3)
        self.assertCML(cubes, ('COLPEX', 'small_colpex_theta_p_alt.cml'))


@iris.tests.skip_data
class TestDataMerge(tests.IrisTest):
    def test_extended_proxy_data(self):
        # Get the empty theta cubes for T+1.5 and T+2
        data_path = tests.get_data_path(
            ('PP', 'COLPEX', 'theta_and_orog_subset.pp'))
        phenom_constraint = iris.Constraint('air_potential_temperature')
        time_value_1 = 347921.33333332836627960205
        time_value_2 = 347921.83333333209156990051
        time_constraint1 = iris.Constraint(time=time_value_1)
        time_constraint2 = iris.Constraint(time=time_value_2)
        time_constraint_1_and_2 = iris.Constraint(
            time=lambda c: c in (time_value_1, time_value_2))
        cube1 = iris.load_cube(data_path, phenom_constraint & time_constraint1)
        cube2 = iris.load_cube(data_path, phenom_constraint & time_constraint2)
        
        # Merge the two halves
        cubes = iris.cube.CubeList([cube1, cube2]).merge(True)
        self.assertCML(cubes, ('merge', 'theta_two_times.cml'))

        # Make sure we get the same result directly from load
        cubes = iris.load_cube(data_path, 
            phenom_constraint & time_constraint_1_and_2)
        self.assertCML(cubes, ('merge', 'theta_two_times.cml'))

    def test_real_data(self):
        data_path = tests.get_data_path(('PP', 'globClim1', 'theta.pp'))
        cubes = iris.load_raw(data_path)
        # Force the source 2-D cubes to load their data before the merge
        for cube in cubes:
            data = cube.data
        cubes = cubes.merge()
        self.assertCML(cubes, ['merge', 'theta.cml'])


class TestDimensionSplitting(tests.IrisTest):
    def _make_cube(self, a, b, c, data):
        cube_data = np.empty((4, 5), dtype=np.float32)
        cube_data[:] = data
        cube = iris.cube.Cube(cube_data)
        cube.add_dim_coord(DimCoord(np.array([0, 1, 2, 3, 4], dtype=np.int32), long_name='x', units='1'), 1)
        cube.add_dim_coord(DimCoord(np.array([0, 1, 2, 3], dtype=np.int32), long_name='y', units='1'), 0)
        cube.add_aux_coord(DimCoord(np.array([a], dtype=np.int32), long_name='a', units='1'))
        cube.add_aux_coord(DimCoord(np.array([b], dtype=np.int32), long_name='b', units='1'))
        cube.add_aux_coord(DimCoord(np.array([c], dtype=np.int32), long_name='c', units='1'))
        return cube
        
    def test_single_split(self):
        # Test what happens when a cube forces a simple, two-way split.
        cubes = []
        cubes.append(self._make_cube(0, 0, 0, 0))
        cubes.append(self._make_cube(0, 1, 1, 1))
        cubes.append(self._make_cube(1, 0, 2, 2))
        cubes.append(self._make_cube(1, 1, 3, 3))
        cubes.append(self._make_cube(2, 0, 4, 4))
        cubes.append(self._make_cube(2, 1, 5, 5))
        cube = iris.cube.CubeList(cubes).merge()
        self.assertCML(cube, ('merge', 'single_split.cml'))

    def test_multi_split(self):
        # Test what happens when a cube forces a three-way split.
        cubes = []
        cubes.append(self._make_cube(0, 0, 0, 0))
        cubes.append(self._make_cube(0, 0, 1, 1))
        cubes.append(self._make_cube(0, 1, 0, 2))
        cubes.append(self._make_cube(0, 1, 1, 3))
        cubes.append(self._make_cube(1, 0, 0, 4))
        cubes.append(self._make_cube(1, 0, 1, 5))
        cubes.append(self._make_cube(1, 1, 0, 6))
        cubes.append(self._make_cube(1, 1, 1, 7))
        cubes.append(self._make_cube(2, 0, 0, 8))
        cubes.append(self._make_cube(2, 0, 1, 9))
        cubes.append(self._make_cube(2, 1, 0, 10))
        cubes.append(self._make_cube(2, 1, 1, 11))
        cube = iris.cube.CubeList(cubes).merge()
        self.assertCML(cube, ('merge', 'multi_split.cml'))


class TestCombination(tests.IrisTest):
    def _make_cube(self, a, b, c, d, data=0):
        cube_data = np.empty((4, 5), dtype=np.float32)
        cube_data[:] = data
        cube = iris.cube.Cube(cube_data)
        cube.add_dim_coord(DimCoord(np.array([0, 1, 2, 3, 4], dtype=np.int32),
                                    long_name='x', units='1'), 1)
        cube.add_dim_coord(DimCoord(np.array([0, 1, 2, 3], dtype=np.int32),
                                    long_name='y', units='1'), 0)
        
        for name, value in zip(['a', 'b', 'c', 'd'], [a, b, c, d]):
            dtype = np.str if isinstance(value, basestring) else np.float32
            cube.add_aux_coord(AuxCoord(np.array([value], dtype=dtype),
                                        long_name=name, units='1'))

        return cube

    def test_separable_combination(self):
        cubes = iris.cube.CubeList()
        cubes.append(self._make_cube('2005', 'ECMWF',
                                     'HOPE-E, Sys 1, Met 1, ENSEMBLES', 0))
        cubes.append(self._make_cube('2005', 'ECMWF',
                                     'HOPE-E, Sys 1, Met 1, ENSEMBLES', 1))
        cubes.append(self._make_cube('2005', 'ECMWF',
                                     'HOPE-E, Sys 1, Met 1, ENSEMBLES', 2))
        cubes.append(self._make_cube('2026', 'UK Met Office',
                                     'HadGEM2, Sys 1, Met 1, ENSEMBLES', 0))
        cubes.append(self._make_cube('2026', 'UK Met Office',
                                     'HadGEM2, Sys 1, Met 1, ENSEMBLES', 1))
        cubes.append(self._make_cube('2026', 'UK Met Office',
                                     'HadGEM2, Sys 1, Met 1, ENSEMBLES', 2))
        cubes.append(self._make_cube('2002', 'CERFACS',
                                     'GELATO, Sys 0, Met 1, ENSEMBLES', 0))
        cubes.append(self._make_cube('2002', 'CERFACS',
                                     'GELATO, Sys 0, Met 1, ENSEMBLES', 1))
        cubes.append(self._make_cube('2002', 'CERFACS',
                                     'GELATO, Sys 0, Met 1, ENSEMBLES', 2))
        cubes.append(self._make_cube('2002', 'IFM-GEOMAR',
                                     'ECHAM5, Sys 1, Met 10, ENSEMBLES', 0))
        cubes.append(self._make_cube('2002', 'IFM-GEOMAR',
                                     'ECHAM5, Sys 1, Met 10, ENSEMBLES', 1))
        cubes.append(self._make_cube('2002', 'IFM-GEOMAR',
                                     'ECHAM5, Sys 1, Met 10, ENSEMBLES', 2))
        cubes.append(self._make_cube('2502', 'UK Met Office',
                                     'HadCM3, Sys 51, Met 10, ENSEMBLES', 0))
        cubes.append(self._make_cube('2502', 'UK Met Office',
                                     'HadCM3, Sys 51, Met 11, ENSEMBLES', 0))
        cubes.append(self._make_cube('2502', 'UK Met Office',
                                     'HadCM3, Sys 51, Met 12, ENSEMBLES', 0))
        cubes.append(self._make_cube('2502', 'UK Met Office',
                                     'HadCM3, Sys 51, Met 13, ENSEMBLES', 0))
        cubes.append(self._make_cube('2502', 'UK Met Office',
                                     'HadCM3, Sys 51, Met 14, ENSEMBLES', 0))
        cubes.append(self._make_cube('2502', 'UK Met Office',
                                     'HadCM3, Sys 51, Met 15, ENSEMBLES', 0))
        cubes.append(self._make_cube('2502', 'UK Met Office',
                                     'HadCM3, Sys 51, Met 16, ENSEMBLES', 0))
        cubes.append(self._make_cube('2502', 'UK Met Office',
                                     'HadCM3, Sys 51, Met 17, ENSEMBLES', 0))
        cubes.append(self._make_cube('2502', 'UK Met Office',
                                     'HadCM3, Sys 51, Met 18, ENSEMBLES', 0))
        cube = cubes.merge()
        self.assertCML(cube, ('merge', 'separable_combination.cml'),
                       checksum=False)


class TestDimSelection(tests.IrisTest):
    def _make_cube(self, a, b, data=0, a_dim=False, b_dim=False):
        cube_data = np.empty((4, 5), dtype=np.float32)
        cube_data[:] = data
        cube = iris.cube.Cube(cube_data)
        cube.add_dim_coord(DimCoord(np.array([0, 1, 2, 3, 4], dtype=np.int32),
                                    long_name='x', units='1'), 1)
        cube.add_dim_coord(DimCoord(np.array([0, 1, 2, 3], dtype=np.int32),
                                    long_name='y', units='1'), 0)

        for name, value, dim in zip(['a', 'b'], [a, b], [a_dim, b_dim]):
            dtype = np.str if isinstance(value, basestring) else np.float32
            ctype = DimCoord if dim else AuxCoord
            coord = ctype(np.array([value], dtype=dtype),
                          long_name=name, units='1')
            cube.add_aux_coord(coord)

        return cube

    def test_string_a_with_aux(self):
        templates = (('a', 0), ('b', 1), ('c', 2), ('d', 3))
        cubes = [self._make_cube(a, b) for a, b in templates]
        cube = iris.cube.CubeList(cubes).merge()[0]
        self.assertCML(cube, ('merge', 'string_a_with_aux.cml'),
                       checksum=False)
        self.assertTrue(isinstance(cube.coord('a'), AuxCoord))
        self.assertTrue(isinstance(cube.coord('b'), DimCoord))
        self.assertTrue(cube.coord('b') in cube.dim_coords)

    def test_string_b_with_aux(self):
        templates = ((0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'))
        cubes = [self._make_cube(a, b) for a, b in templates]
        cube = iris.cube.CubeList(cubes).merge()[0]
        self.assertCML(cube, ('merge', 'string_b_with_aux.cml'),
                       checksum=False)
        self.assertTrue(isinstance(cube.coord('a'), DimCoord))
        self.assertTrue(cube.coord('a') in cube.dim_coords)
        self.assertTrue(isinstance(cube.coord('b'), AuxCoord))

    def test_string_a_with_dim(self):
        templates = (('a', 0), ('b', 1), ('c', 2), ('d', 3))
        cubes = [self._make_cube(a, b, b_dim=True) for a, b in templates]
        cube = iris.cube.CubeList(cubes).merge()[0]
        self.assertCML(cube, ('merge', 'string_a_with_dim.cml'),
                       checksum=False)
        self.assertTrue(isinstance(cube.coord('a'), AuxCoord))
        self.assertTrue(isinstance(cube.coord('b'), DimCoord))
        self.assertTrue(cube.coord('b') in cube.dim_coords)

    def test_string_b_with_aux(self):
        templates = ((0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'))
        cubes = [self._make_cube(a, b, a_dim=True) for a, b in templates]
        cube = iris.cube.CubeList(cubes).merge()[0]
        self.assertCML(cube, ('merge', 'string_b_with_dim.cml'),
                       checksum=False)
        self.assertTrue(isinstance(cube.coord('a'), DimCoord))
        self.assertTrue(cube.coord('a') in cube.dim_coords)
        self.assertTrue(isinstance(cube.coord('b'), AuxCoord))

    def test_string_a_b(self):
        templates = (('a', '0'), ('b', '1'), ('c', '2'), ('d', '3'))
        cubes = [self._make_cube(a, b) for a, b in templates]
        cube = iris.cube.CubeList(cubes).merge()[0]
        self.assertCML(cube, ('merge', 'string_a_b.cml'),
                       checksum=False)
        self.assertTrue(isinstance(cube.coord('a'), AuxCoord))
        self.assertTrue(isinstance(cube.coord('b'), AuxCoord))

    def test_a_aux_b_aux(self):
        templates = ((0, 10), (1, 11), (2, 12), (3, 13))
        cubes = [self._make_cube(a, b) for a, b in templates]
        cube = iris.cube.CubeList(cubes).merge()[0]
        self.assertCML(cube, ('merge', 'a_aux_b_aux.cml'),
                       checksum=False)
        self.assertTrue(isinstance(cube.coord('a'), DimCoord))
        self.assertTrue(cube.coord('a') in cube.dim_coords)
        self.assertTrue(isinstance(cube.coord('b'), DimCoord))
        self.assertTrue(cube.coord('b') in cube.aux_coords)

    def test_a_aux_b_dim(self):
        templates = ((0, 10), (1, 11), (2, 12), (3, 13))
        cubes = [self._make_cube(a, b, b_dim=True) for a, b in templates]
        cube = iris.cube.CubeList(cubes).merge()[0]
        self.assertCML(cube, ('merge', 'a_aux_b_dim.cml'),
                       checksum=False)
        self.assertTrue(isinstance(cube.coord('a'), DimCoord))
        self.assertTrue(cube.coord('a') in cube.aux_coords)
        self.assertTrue(isinstance(cube.coord('b'), DimCoord))
        self.assertTrue(cube.coord('b') in cube.dim_coords)

    def test_a_dim_b_aux(self):
        templates = ((0, 10), (1, 11), (2, 12), (3, 13))
        cubes = [self._make_cube(a, b, a_dim=True) for a, b in templates]
        cube = iris.cube.CubeList(cubes).merge()[0]
        self.assertCML(cube, ('merge', 'a_dim_b_aux.cml'),
                       checksum=False)
        self.assertTrue(isinstance(cube.coord('a'), DimCoord))
        self.assertTrue(cube.coord('a') in cube.dim_coords)
        self.assertTrue(isinstance(cube.coord('b'), DimCoord))
        self.assertTrue(cube.coord('b') in cube.aux_coords)

    def test_a_dim_b_dim(self):
        templates = ((0, 10), (1, 11), (2, 12), (3, 13))
        cubes = [self._make_cube(a, b, a_dim=True, b_dim=True) \
                     for a, b in templates]
        cube = iris.cube.CubeList(cubes).merge()[0]
        self.assertCML(cube, ('merge', 'a_dim_b_dim.cml'),
                       checksum=False)
        self.assertTrue(isinstance(cube.coord('a'), DimCoord))
        self.assertTrue(cube.coord('a') in cube.dim_coords)
        self.assertTrue(isinstance(cube.coord('b'), DimCoord))
        self.assertTrue(cube.coord('b') in cube.aux_coords)


class TestTimeTripleMerging(tests.IrisTest):
    def _make_cube(self, a, b, c, data=0):
        cube_data = np.empty((4, 5), dtype=np.float32)
        cube_data[:] = data
        cube = iris.cube.Cube(cube_data)
        cube.add_dim_coord(DimCoord(np.array([0, 1, 2, 3, 4], dtype=np.int32), long_name='x', units='1'), 1)
        cube.add_dim_coord(DimCoord(np.array([0, 1, 2, 3], dtype=np.int32), long_name='y', units='1'), 0)
        cube.add_aux_coord(DimCoord(np.array([a], dtype=np.int32), standard_name='forecast_period', units='1'))
        cube.add_aux_coord(DimCoord(np.array([b], dtype=np.int32), standard_name='forecast_reference_time', units='1'))
        cube.add_aux_coord(DimCoord(np.array([c], dtype=np.int32), standard_name='time', units='1'))
        return cube
    
    def _test_triples(self, triples, filename):
        cubes = [self._make_cube(fp, rt, t) for fp, rt, t in triples]
        cube = iris.cube.CubeList(cubes).merge()
        self.assertCML(cube, ('merge', 'time_triple_' + filename + '.cml'), checksum=False)
                          
    def test_single_forecast(self):
        # A single forecast series (i.e. from a single reference time)
        # => fp, t: 4; rt: scalar
        triples = (
            (0, 10, 10), (1, 10, 11), (2, 10, 12), (3, 10, 13),
        )
        self._test_triples(triples, 'single_forecast')

    def test_successive_forecasts(self):
        # Three forecast series from successively later reference times
        # => rt, t: 3; fp, t: 4
        triples = (
            (0, 10, 10), (1, 10, 11), (2, 10, 12), (3, 10, 13),
            (0, 11, 11), (1, 11, 12), (2, 11, 13), (3, 11, 14),
            (0, 12, 12), (1, 12, 13), (2, 12, 14), (3, 12, 15),
        )
        self._test_triples(triples, 'successive_forecasts')

    def test_time_vs_ref_time(self):
        # => fp, t: 4; fp, rt: 3
        triples = (
            (2, 10, 12), (3, 10, 13), (4, 10, 14), (5, 10, 15),
            (1, 11, 12), (2, 11, 13), (3, 11, 14), (4, 11, 15),
            (0, 12, 12), (1, 12, 13), (2, 12, 14), (3, 12, 15),
        )
        self._test_triples(triples, 'time_vs_ref_time')

    def test_time_vs_forecast(self):
        # => rt, t: 4, fp, rt: 3
        triples = (
            (0, 10, 10), (0, 11, 11), (0, 12, 12), (0, 13, 13),
            (1,  9, 10), (1, 10, 11), (1, 11, 12), (1, 12, 13),
            (2,  8, 10), (2,  9, 11), (2, 10, 12), (2, 11, 13),
        )
        self._test_triples(triples, 'time_vs_forecast')
        
    def test_time_non_dim_coord(self):
        # => rt: 1 fp, t (bounded): 2
        triples = (
            (5, 0, 2.5), (10, 0, 5),
        )
        cubes = [self._make_cube(fp, rt, t) for fp, rt, t in triples]
        for end_time, cube in zip([5, 10], cubes):
            cube.coord('time').bounds = [0, end_time] 
        cube, = iris.cube.CubeList(cubes).merge()
        self.assertCML(cube, ('merge', 'time_triple_time_non_dim_coord.cml'), checksum=False)
        # make sure that forecast_period is the dimensioned coordinate (as time becomes an AuxCoord)
        self.assertEqual(cube.coord(dimensions=0, dim_coords=True).name(), 'forecast_period')
                          
    def test_independent(self):
        # => fp: 2; rt: 2; t: 2
        triples = (
            (0, 10, 10), (0, 11, 10),
            (0, 10, 11), (0, 11, 11),
            (1, 10, 10), (1, 11, 10),
            (1, 10, 11), (1, 11, 11),
        )
        self._test_triples(triples, 'independent')

    def test_series(self):
        # => fp, rt, t: 5 (with only t being definitive).
        triples = (
            (0, 10, 10),
            (0, 11, 11),
            (0, 12, 12),
            (1, 12, 13),
            (2, 12, 14),
        )
        self._test_triples(triples, 'series')

    def test_non_expanding_dimension(self):
        triples = (
            (0, 10, 0), (0, 20, 1), (0, 20, 0),
        )
        # => fp: scalar; rt, t: 3 (with no time being definitive)
        self._test_triples(triples, 'non_expanding')
    
    def test_duplicate_data(self):
        # test what happens when we have repeated time coordinates (i.e. duplicate data)
        cube1 = self._make_cube(0, 10, 0)
        cube2 = self._make_cube(1, 20, 1)
        cube3 = self._make_cube(1, 20, 1)
        
        # check that we get a duplicate data error when unique is True
        with self.assertRaises(iris.exceptions.DuplicateDataError):
            iris.cube.CubeList([cube1, cube2, cube3]).merge()
        
        cubes = iris.cube.CubeList([cube1, cube2, cube3]).merge(unique=False)
        self.assertCML(cubes, ('merge', 'time_triple_duplicate_data.cml'), checksum=False)
    
    def test_simple1(self):
        cube1 = self._make_cube(0, 10, 0)
        cube2 = self._make_cube(1, 20, 1)
        cube3 = self._make_cube(2, 20, 0)
        cube = iris.cube.CubeList([cube1, cube2, cube3]).merge()
        self.assertCML(cube, ('merge', 'time_triple_merging1.cml'), checksum=False)
        
    def test_simple2(self):
        cubes = iris.cube.CubeList([
                                    self._make_cube(0, 0, 0),
                                    self._make_cube(1, 0, 1),
                                    self._make_cube(2, 0, 2),
                                    self._make_cube(0, 1, 3),
                                    self._make_cube(1, 1, 4),          
                                    self._make_cube(2, 1, 5),
                                   ])
        cube = cubes.merge()[0]
        self.assertCML(cube, ('merge', 'time_triple_merging2.cml'), checksum=False)
        self.assertIsNone(cube.assert_valid())
        
        cube = iris.cube.CubeList(cubes[:-1]).merge()[0]
        self.assertCML(cube, ('merge', 'time_triple_merging3.cml'), checksum=False)
        self.assertIsNone(cube.assert_valid())
        
    def test_simple3(self):
        cubes = iris.cube.CubeList([
                                    self._make_cube(0, 0, 0),
                                    self._make_cube(0, 1, 1),
                                    self._make_cube(0, 2, 2),
                                    self._make_cube(1, 0, 3),
                                    self._make_cube(1, 1, 4),          
                                    self._make_cube(1, 2, 5),
                                   ])
        cube = cubes.merge()[0]
        self.assertCML(cube, ('merge', 'time_triple_merging4.cml'), checksum=False)
        self.assertIsNone(cube.assert_valid())
        
        cube = iris.cube.CubeList(cubes[:-1]).merge()[0]
        self.assertCML(cube, ('merge', 'time_triple_merging5.cml'), checksum=False)
        self.assertIsNone(cube.assert_valid())


class TestCubeMergeTheoretical(tests.IrisTest):
    def test_simple_bounds_merge(self):
        cube1 = iris.tests.stock.simple_2d()
        cube2 = iris.tests.stock.simple_2d()
        
        cube1.add_aux_coord(DimCoord(np.int32(10), long_name='pressure', units='Pa'))
        cube2.add_aux_coord(DimCoord(np.int32(11), long_name='pressure', units='Pa'))
        
        r = iris.cube.CubeList([cube1, cube2]).merge()
        self.assertCML(r, ('cube_merge', 'test_simple_bound_merge.cml'))
        
    def test_simple_multidim_merge(self):
        cube1 = iris.tests.stock.simple_2d_w_multidim_coords()
        cube2 = iris.tests.stock.simple_2d_w_multidim_coords()
        
        cube1.add_aux_coord(DimCoord(np.int32(10), long_name='pressure', units='Pa'))
        cube2.add_aux_coord(DimCoord(np.int32(11), long_name='pressure', units='Pa'))

        r = iris.cube.CubeList([cube1, cube2]).merge()[0]
        self.assertCML(r, ('cube_merge', 'multidim_coord_merge.cml'))
        
        # try transposing the cubes first
        cube1.transpose([1, 0])
        cube2.transpose([1, 0])
        r = iris.cube.CubeList([cube1, cube2]).merge()[0]
        self.assertCML(r, ('cube_merge', 'multidim_coord_merge_transpose.cml'))
        
    def test_simple_points_merge(self):
        cube1 = iris.tests.stock.simple_2d(with_bounds=False)
        cube2 = iris.tests.stock.simple_2d(with_bounds=False)
        
        cube1.add_aux_coord(DimCoord(np.int32(10), long_name='pressure', units='Pa'))
        cube2.add_aux_coord(DimCoord(np.int32(11), long_name='pressure', units='Pa'))
        
        r = iris.cube.CubeList([cube1, cube2]).merge()
        self.assertCML(r, ('cube_merge', 'test_simple_merge.cml'))
        
        # check that the unique merging raises a Duplicate data error 
        self.assertRaises(iris.exceptions.DuplicateDataError, iris.cube.CubeList([cube1, cube1]).merge, unique=True)
        
        # check that non unique merging returns both cubes 
        r = iris.cube.CubeList([cube1, cube1]).merge(unique=False)
        self.assertCML(r[0], ('cube_merge', 'test_orig_point_cube.cml'))
        self.assertCML(r[1], ('cube_merge', 'test_orig_point_cube.cml'))
        
        # test local_attributes merging
        cube1.local_attributes['my_attr1'] = 'foo'
        r = iris.cube.CubeList([cube1, cube2]).merge()
        # result should be 2 cubes
        self.assertCML(r, ('cube_merge', 'test_simple_attributes1.cml'))
        
        cube2.local_attributes['my_attr1'] = 'bar'
        r = iris.cube.CubeList([cube1, cube2]).merge()
        # result should be 2 cubes
        self.assertCML(r, ('cube_merge', 'test_simple_attributes2.cml'))
        
        cube2.local_attributes['my_attr1'] = 'foo'
        r = iris.cube.CubeList([cube1, cube2]).merge()
        # result should be 1 cube
        self.assertCML(r, ('cube_merge', 'test_simple_attributes3.cml'))

        # test global_attributes merging
        cube1.global_attributes['my_global_attr'] = 'foo'
        r = iris.cube.CubeList([cube1, cube2]).merge()
        # result should be 2 cubes
        self.assertCML(r, ('cube_merge', 'test_simple_attributes4.cml'))
        
        cube2.global_attributes['my_global_attr'] = 'bar'
        r = iris.cube.CubeList([cube1, cube2]).merge()
        # result should be 2 cubes
        self.assertCML(r, ('cube_merge', 'test_simple_attributes5.cml'))
        
        cube2.global_attributes['my_global_attr'] = 'foo'
        r = iris.cube.CubeList([cube1, cube2]).merge()
        # result should be 1 cube
        self.assertCML(r, ('cube_merge', 'test_simple_attributes6.cml'))

        # test mixed global/local attributes merging
        cube1.local_attributes['my_attr2'] = 'bar'
        cube2.global_attributes['my_attr2'] = 'bar'
        r = iris.cube.CubeList([cube1, cube2]).merge()
        # result should be 2 cubes
        self.assertCML(r, ('cube_merge', 'test_simple_attributes7.cml'))


if __name__ == "__main__":
    tests.main()
