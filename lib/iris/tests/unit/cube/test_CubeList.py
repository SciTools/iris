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
"""Unit tests for the `iris.cube.CubeList` class."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.cube import Cube, CubeList
from iris.coords import AuxCoord, DimCoord
from cf_units import Unit
import iris.coord_systems
import iris.exceptions


class Test_concatenate_cube(tests.IrisTest):
    def setUp(self):
        self.units = Unit('days since 1970-01-01 00:00:00',
                          calendar='gregorian')
        self.cube1 = Cube([1, 2, 3], 'air_temperature', units='K')
        self.cube1.add_dim_coord(DimCoord([0, 1, 2], 'time', units=self.units),
                                 0)

    def test_pass(self):
        self.cube2 = Cube([1, 2, 3], 'air_temperature', units='K')
        self.cube2.add_dim_coord(DimCoord([3, 4, 5], 'time', units=self.units),
                                 0)
        result = CubeList([self.cube1, self.cube2]).concatenate_cube()
        self.assertIsInstance(result, Cube)

    def test_fail(self):
        units = Unit('days since 1970-01-02 00:00:00',
                     calendar='gregorian')
        cube2 = Cube([1, 2, 3], 'air_temperature', units='K')
        cube2.add_dim_coord(DimCoord([0, 1, 2], 'time', units=units), 0)
        with self.assertRaises(iris.exceptions.ConcatenateError):
            CubeList([self.cube1, cube2]).concatenate_cube()

    def test_empty(self):
        exc_regexp = "can't concatenate an empty CubeList"
        with self.assertRaisesRegexp(ValueError, exc_regexp):
            CubeList([]).concatenate_cube()


class Test_extract_overlapping(tests.IrisTest):
    def setUp(self):
        shape = (6, 14, 19)
        n_time, n_lat, n_lon = shape
        n_data = n_time * n_lat * n_lon
        cube = Cube(np.arange(n_data, dtype=np.int32).reshape(shape))
        coord = iris.coords.DimCoord(points=np.arange(n_time),
                                     standard_name='time',
                                     units='hours since epoch')
        cube.add_dim_coord(coord, 0)
        cs = iris.coord_systems.GeogCS(6371229)
        coord = iris.coords.DimCoord(points=np.linspace(-90, 90, n_lat),
                                     standard_name='latitude',
                                     units='degrees',
                                     coord_system=cs)
        cube.add_dim_coord(coord, 1)
        coord = iris.coords.DimCoord(points=np.linspace(-180, 180, n_lon),
                                     standard_name='longitude',
                                     units='degrees',
                                     coord_system=cs)
        cube.add_dim_coord(coord, 2)
        self.cube = cube

    def test_extract_one_str_dim(self):
        cubes = iris.cube.CubeList([self.cube[2:], self.cube[:4]])
        a, b = cubes.extract_overlapping('time')
        self.assertEqual(a.coord('time'), self.cube.coord('time')[2:4])
        self.assertEqual(b.coord('time'), self.cube.coord('time')[2:4])

    def test_extract_one_list_dim(self):
        cubes = iris.cube.CubeList([self.cube[2:], self.cube[:4]])
        a, b = cubes.extract_overlapping(['time'])
        self.assertEqual(a.coord('time'), self.cube.coord('time')[2:4])
        self.assertEqual(b.coord('time'), self.cube.coord('time')[2:4])

    def test_extract_two_dims(self):
        cubes = iris.cube.CubeList([self.cube[2:, 5:], self.cube[:4, :10]])
        a, b = cubes.extract_overlapping(['time', 'latitude'])
        self.assertEqual(a.coord('time'),
                         self.cube.coord('time')[2:4])
        self.assertEqual(a.coord('latitude'),
                         self.cube.coord('latitude')[5:10])
        self.assertEqual(b.coord('time'),
                         self.cube.coord('time')[2:4])
        self.assertEqual(b.coord('latitude'),
                         self.cube.coord('latitude')[5:10])

    def test_different_orders(self):
        cubes = iris.cube.CubeList([self.cube[::-1][:4], self.cube[:4]])
        a, b = cubes.extract_overlapping('time')
        self.assertEqual(a.coord('time'), self.cube[::-1].coord('time')[2:4])
        self.assertEqual(b.coord('time'), self.cube.coord('time')[2:4])


class Test_merge_cube(tests.IrisTest):
    def setUp(self):
        self.cube1 = Cube([1, 2, 3], "air_temperature", units="K")
        self.cube1.add_aux_coord(AuxCoord([0], "height", units="m"))

    def test_pass(self):
        cube2 = self.cube1.copy()
        cube2.coord("height").points = [1]
        result = CubeList([self.cube1, cube2]).merge_cube()
        self.assertIsInstance(result, Cube)

    def test_fail(self):
        cube2 = self.cube1.copy()
        cube2.rename("not air temperature")
        with self.assertRaises(iris.exceptions.MergeError):
            CubeList([self.cube1, cube2]).merge_cube()

    def test_empty(self):
        with self.assertRaises(ValueError):
            CubeList([]).merge_cube()

    def test_single_cube(self):
        result = CubeList([self.cube1]).merge_cube()
        self.assertEqual(result, self.cube1)
        self.assertIsNot(result, self.cube1)

    def test_repeated_cube(self):
        with self.assertRaises(iris.exceptions.MergeError):
            CubeList([self.cube1, self.cube1]).merge_cube()


class Test_merge__time_triple(tests.IrisTest):
    @staticmethod
    def _make_cube(fp, rt, t, realization=None):
        cube = Cube(np.arange(20).reshape(4, 5))
        cube.add_dim_coord(DimCoord(np.arange(5), long_name='x'), 1)
        cube.add_dim_coord(DimCoord(np.arange(4), long_name='y'), 0)
        cube.add_aux_coord(DimCoord(fp, standard_name='forecast_period'))
        cube.add_aux_coord(DimCoord(rt,
                                    standard_name='forecast_reference_time'))
        cube.add_aux_coord(DimCoord(t, standard_name='time'))
        if realization is not None:
            cube.add_aux_coord(DimCoord(realization,
                                        standard_name='realization'))
        return cube

    def test_orthogonal_with_realization(self):
        # => fp: 2; rt: 2; t: 2; realization: 2
        triples = ((0, 10, 1),
                   (0, 10, 2),
                   (0, 11, 1),
                   (0, 11, 2),
                   (1, 10, 1),
                   (1, 10, 2),
                   (1, 11, 1),
                   (1, 11, 2))
        en1_cubes = [self._make_cube(*triple, realization=1) for
                     triple in triples]
        en2_cubes = [self._make_cube(*triple, realization=2) for
                     triple in triples]
        cubes = CubeList(en1_cubes) + CubeList(en2_cubes)
        cube, = cubes.merge()
        self.assertCML(cube, checksum=False)

    def test_combination_with_realization(self):
        # => fp, rt, t: 8; realization: 2
        triples = ((0, 10, 1),
                   (0, 10, 2),
                   (0, 11, 1),
                   (0, 11, 3),  # This '3' breaks the pattern.
                   (1, 10, 1),
                   (1, 10, 2),
                   (1, 11, 1),
                   (1, 11, 2))
        en1_cubes = [self._make_cube(*triple, realization=1) for
                     triple in triples]
        en2_cubes = [self._make_cube(*triple, realization=2) for
                     triple in triples]
        cubes = CubeList(en1_cubes) + CubeList(en2_cubes)
        cube, = cubes.merge()
        self.assertCML(cube, checksum=False)

    def test_combination_with_extra_realization(self):
        # => fp, rt, t, realization: 17
        triples = ((0, 10, 1),
                   (0, 10, 2),
                   (0, 11, 1),
                   (0, 11, 2),
                   (1, 10, 1),
                   (1, 10, 2),
                   (1, 11, 1),
                   (1, 11, 2))
        en1_cubes = [self._make_cube(*triple, realization=1) for
                     triple in triples]
        en2_cubes = [self._make_cube(*triple, realization=2) for
                     triple in triples]
        # Add extra that is a duplicate of one of the time triples
        # but with a different realisation.
        en3_cubes = [self._make_cube(0, 10, 2, realization=3)]
        cubes = CubeList(en1_cubes) + CubeList(en2_cubes) + CubeList(en3_cubes)
        cube, = cubes.merge()
        self.assertCML(cube, checksum=False)

    def test_combination_with_extra_triple(self):
        # => fp, rt, t, realization: 17
        triples = ((0, 10, 1),
                   (0, 10, 2),
                   (0, 11, 1),
                   (0, 11, 2),
                   (1, 10, 1),
                   (1, 10, 2),
                   (1, 11, 1),
                   (1, 11, 2))
        en1_cubes = [self._make_cube(*triple, realization=1) for
                     triple in triples]
        # Add extra time triple on the end.
        en2_cubes = [self._make_cube(*triple, realization=2) for
                     triple in triples + ((1, 11, 3),)]
        cubes = CubeList(en1_cubes) + CubeList(en2_cubes)
        cube, = cubes.merge()
        self.assertCML(cube, checksum=False)


class Test_xml(tests.IrisTest):
    def setUp(self):
        self.cubes = CubeList([Cube(np.arange(3)),
                               Cube(np.arange(3))])

    def test_byteorder_default(self):
        self.assertIn('byteorder', self.cubes.xml())

    def test_byteorder_false(self):
        self.assertNotIn('byteorder', self.cubes.xml(byteorder=False))

    def test_byteorder_true(self):
        self.assertIn('byteorder', self.cubes.xml(byteorder=True))


class Test_extract(tests.IrisTest):
    def setUp(self):
        self.scalar_cubes = CubeList()
        for i in range(5):
            for letter in 'abcd':
                self.scalar_cubes.append(Cube(i, long_name=letter))

    def test_scalar_cube_name_constraint(self):
        # Test the name based extraction of a CubeList containing scalar cubes.
        res = self.scalar_cubes.extract('a')
        expected = CubeList([Cube(i, long_name='a') for i in range(5)])
        self.assertEqual(res, expected)

    def test_scalar_cube_data_constraint(self):
        # Test the extraction of a CubeList containing scalar cubes
        # when using a cube_func.
        val = 2
        constraint = iris.Constraint(cube_func=lambda c: c.data == val)
        res = self.scalar_cubes.extract(constraint)
        expected = CubeList([Cube(val, long_name=letter) for letter in 'abcd'])
        self.assertEqual(res, expected)


if __name__ == "__main__":
    tests.main()
