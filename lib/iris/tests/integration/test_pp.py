# (C) British Crown Copyright 2013, Met Office
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


if __name__ == "__main__":
    tests.main()
