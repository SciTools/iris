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
"""Unit tests for the `iris.cube.CubeList` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.cube import Cube, CubeList
from iris.coords import DimCoord


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


if __name__ == "__main__":
    tests.main()
