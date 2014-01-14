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
"""Integration tests for loading and saving PP files."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import numpy as np

import iris.fileformats.pp


class TestVertical(tests.IrisTest):
    def test_soil_level_round_trip(self):
        # Use pp.load_cubes() to convert a fake PPField into a Cube.
        # NB. Use MagicMock so that SplittableInt header items, such as
        # LBCODE, support len().
        soil_level = 1234
        field = mock.MagicMock(lbvc=6, blev=soil_level,
                               stash=iris.fileformats.pp.STASH(1, 0, 9),
                               lbuser=[0] * 7, lbrsvd=[0] * 4)
        load = mock.Mock(return_value=iter([field]))
        with mock.patch('iris.fileformats.pp.load', new=load) as load:
            cube = next(iris.fileformats.pp.load_cubes('DUMMY'))

        self.assertIn('soil', cube.standard_name)
        self.assertEqual(len(cube.coords('model_level_number')), 1)
        self.assertEqual(cube.coord('model_level_number').points, soil_level)

        # Now use the save rules to convert the Cube back into a PPField.
        field = iris.fileformats.pp.PPField3()
        field.lbfc = 0
        field.lbvc = 0
        iris.fileformats.pp._ensure_save_rules_loaded()
        iris.fileformats.pp._save_rules.verify(cube, field)

        # Check the vertical coordinate is as originally specified.
        self.assertEqual(field.lbvc, 6)
        self.assertEqual(field.blev, soil_level)

    def test_potential_temperature_level_round_trip(self):
        """Check save+load for data on 'potential temperature' levels."""
        # Use pp.load_cubes() to convert a fake PPField into a Cube.
        # NB. Use MagicMock so that SplittableInt header items, such as
        # LBCODE, support len().
        potm_value = 22.5
        field = mock.MagicMock(lbvc=19, blev=potm_value,
                               lbuser=[0] * 7, lbrsvd=[0] * 4)
        load = mock.Mock(return_value=iter([field]))
        with mock.patch('iris.fileformats.pp.load', new=load) as load:
            cube = next(iris.fileformats.pp.load_cubes('DUMMY'))

        self.assertEqual(len(cube.coords('air_potential_temperature')), 1)
        self.assertEqual(cube.coord('air_potential_temperature').points,
                         potm_value)

        # Now use the save rules to convert the Cube back into a PPField.
        field = iris.fileformats.pp.PPField3()
        field.lbfc = 0
        field.lbvc = 0
        iris.fileformats.pp._ensure_save_rules_loaded()
        iris.fileformats.pp._save_rules.verify(cube, field)

        # Check the vertical coordinate is as originally specified.
        self.assertEqual(field.lbvc, 19)
        self.assertEqual(field.blev, potm_value)


class TestCoordinateForms(tests.IrisTest):
    def test_save_awkward_case_is_regular(self):
        # Check that specific "awkward" values still save in a regular form.
        nx = 3
        ny = 2
        x0 = np.float32(355.626)
        dx = np.float32(0.0135)
        data = np.zeros((ny, nx), dtype=np.float32)
        test_cube = iris.cube.Cube(data)
        x_coord = iris.coords.DimCoord.from_regular(
            zeroth=x0,
            step=dx,
            count=nx,
            standard_name='longitude',
            units='degrees_east')
        test_cube.add_dim_coord(x_coord, 1)
        y0 = np.float32(20.5)
        dy = np.float32(3.72)
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
            pp_loader = iris.fileformats.pp.load(pp_filepath)
            pp_field = pp_loader.next()
        # Check that the result has the regular coordinates as expected.
        self.assertAlmostEqual(pp_field.bzx, x0)  # N.B. *not* exact.
        self.assertAlmostEqual(pp_field.bdx, dx)
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
            pp_loader = iris.fileformats.pp.load(pp_filepath)
            pp_field = pp_loader.next()
        # Check that the result has the regular/irregular Y and X as expected.
        self.assertAlmostEqual(pp_field.bdx, 0.0)
        self.assertArrayAllClose(pp_field.x, x_values)
        self.assertAlmostEqual(pp_field.lbnpt, nx)
        self.assertAlmostEqual(pp_field.bzy, y0)
        self.assertAlmostEqual(pp_field.bdy, dy)
        self.assertAlmostEqual(pp_field.lbrow, ny)


if __name__ == "__main__":
    tests.main()
