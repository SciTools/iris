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
"""Test saving to PP files."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

import iris
import iris.fileformats.pp as pp


class TestCoordinateForms(tests.IrisTest):
    def test_save_awkward_case_is_regular(self):
        # Check that specific "awkward" values still save in a regular form.
        nx = 3
        ny = 2
        x0 = 355.626
        dx = 0.0135
        data = np.zeros((ny, nx), dtype=np.float32)
        test_cube = iris.cube.Cube(data)
        x_coord = iris.coords.DimCoord.from_regular(
            zeroth=x0,
            step=dx,
            count=nx,
            standard_name='longitude',
            units='degrees_east')
        test_cube.add_dim_coord(x_coord, 1)
        y0 = 20.5
        dy = 3.72
        y_coord = iris.coords.DimCoord.from_regular(
            zeroth=y0,
            step=dy,
            count=ny,
            standard_name='latitude',
            units='degrees_north')
        test_cube.add_dim_coord(y_coord, 0)
        # Write to a temporary PP file and read it back as a PPField
        with self.temp_filename('.pp') as pp_filepath:
            iris.save(test_cube, pp_filepath)
            pp_loader = pp.load(pp_filepath)
            pp_field = pp_loader.next()
        # Check that the result has the regular coordinates as expected.
        self.assertAlmostEqual(pp_field.bzx, x0, places=4)  # N.B. *not* exact.
        self.assertAlmostEqual(pp_field.lbnpt, nx)
        self.assertAlmostEqual(pp_field.bzy, y0)
        self.assertAlmostEqual(pp_field.bdy, dy)
        self.assertAlmostEqual(pp_field.lbrow, ny)

    def test_save_irregular(self):
        # Check that a non-regular coordinate saves as expected.
        nx = 3
        ny = 2
        x_values = [0.0, 1.1, 2.0]
        data = np.zeros((ny, nx), dtype=np.float32)
        test_cube = iris.cube.Cube(data)
        x_coord = iris.coords.DimCoord(x_values,
                                       standard_name='longitude',
                                       units='degrees_east')
        test_cube.add_dim_coord(x_coord, 1)
        y0 = 20.5
        dy = 3.72
        y_coord = iris.coords.DimCoord.from_regular(
            zeroth=y0,
            step=dy,
            count=ny,
            standard_name='latitude',
            units='degrees_north')
        test_cube.add_dim_coord(y_coord, 0)
        # Write to a temporary PP file and read it back as a PPField
        with self.temp_filename('.pp') as pp_filepath:
            iris.save(test_cube, pp_filepath)
            pp_loader = pp.load(pp_filepath)
            pp_field = pp_loader.next()
        # Check that the result has the regular/irregular Y and X as expected.
        self.assertAlmostEqual(pp_field.bdx, 0.0)
        self.assertArrayAllClose(pp_field.x, x_values)
        self.assertAlmostEqual(pp_field.lbnpt, nx)
        self.assertAlmostEqual(pp_field.bzy, y0)
        self.assertAlmostEqual(pp_field.bdy, dy)
        self.assertAlmostEqual(pp_field.lbrow, ny)


if __name__ == '__main__':
    tests.main()
