# (C) British Crown Copyright 2016, Met Office
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
"""Integration tests for merge."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import iris.coords
import iris.cube


class TestMerge(tests.IrisTest):
    def test(self):
        coord_vals = [
                      (0, 30, 100),
                      (0, 40, 100),
                      (0, 40, 110),
                      (1, 30, 110),
                      (1, 30, 100),
                      (1, 40, 110),
                      ]

        test_raw = iris.cube.CubeList([])

        for r, t, frt in coord_vals:
            cube = iris.cube.Cube([0])
            cube.add_aux_coord(iris.coords.DimCoord(r, long_name='a'), [])
            cube.add_aux_coord(iris.coords.DimCoord(t, long_name='b'), [])
            cube.add_aux_coord(iris.coords.DimCoord(frt, long_name='c'), [])
            test_raw.append(cube)

        test_raw.merge()


if __name__ == '__main__':
    tests.main()
