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
import iris.fileformats.pp_rules
from iris.fileformats.pp_rules import \
    PP_HYBRID_COORDINATE_REFERENCE_PRESSURE as REFERENCE_PRESSURE


class TestVertical(tests.IrisTest):
    def _test_coord(self, cube, point, bounds=None, **kwargs):
        coords = cube.coords(**kwargs)
        self.assertEqual(len(coords), 1, 'failed to find exactly one coord'
                                         ' using: {}'.format(kwargs))
        self.assertEqual(coords[0].points, point)
        if bounds is not None:
            self.assertArrayEqual(coords[0].bounds, [bounds])


    def test_soil_level_round_trip(self):
        # Use pp.load_cubes() to convert a fake PPField into a Cube.
        # NB. Use MagicMock so that SplittableInt header items, such as
        # LBCODE, support len().
        soil_level = 1234
        field = mock.MagicMock(lbvc=6, lblev=soil_level,
                               stash=iris.fileformats.pp.STASH(1, 0, 9),
                               lbuser=[0] * 7, lbrsvd=[0] * 4)
        load = mock.Mock(return_value=iter([field]))
        with mock.patch('iris.fileformats.pp.load', new=load) as load:
            cube = next(iris.fileformats.pp.load_cubes('DUMMY'))

        self.assertIn('soil', cube.standard_name)
        self._test_coord(cube, soil_level, long_name='soil_model_level_number')

        # Now use the save rules to convert the Cube back into a PPField.
        field = iris.fileformats.pp.PPField3()
        field.lbfc = 0
        field.lbvc = 0
        iris.fileformats.pp._ensure_save_rules_loaded()
        iris.fileformats.pp._save_rules.verify(cube, field)

        # Check the vertical coordinate is as originally specified.
        self.assertEqual(field.lbvc, 6)
        self.assertEqual(field.lblev, soil_level)

    def test_potential_temperature_level_round_trip(self):
        # Check save+load for data on 'potential temperature' levels.

        # Use pp.load_cubes() to convert a fake PPField into a Cube.
        # NB. Use MagicMock so that SplittableInt header items, such as
        # LBCODE, support len().
        potm_value = 22.5
        field = mock.MagicMock(lbvc=19, blev=potm_value,
                               lbuser=[0] * 7, lbrsvd=[0] * 4)
        load = mock.Mock(return_value=iter([field]))
        with mock.patch('iris.fileformats.pp.load', new=load) as load:
            cube = next(iris.fileformats.pp.load_cubes('DUMMY'))

        self._test_coord(cube, potm_value,
                         standard_name='air_potential_temperature')

        # Now use the save rules to convert the Cube back into a PPField.
        field = iris.fileformats.pp.PPField3()
        field.lbfc = 0
        field.lbvc = 0
        iris.fileformats.pp._ensure_save_rules_loaded()
        iris.fileformats.pp._save_rules.verify(cube, field)

        # Check the vertical coordinate is as originally specified.
        self.assertEqual(field.lbvc, 19)
        self.assertEqual(field.blev, potm_value)

    def test_hybrid_pressure_round_trip(self):
        # Use pp.load_cubes() to convert fake PPFields into Cubes.
        # NB. Use MagicMock so that SplittableInt header items, such as
        # LBCODE, support len().
        def field_with_data(scale=1):
            x, y = 40, 30
            field = mock.MagicMock(_data_manager=None,
                                   data=np.arange(1200).reshape(y, x) * scale,
                                   lbcode=[1], lbnpt=x, lbrow=y,
                                   bzx=350, bdx=1.5, bzy=40, bdy=1.5,
                                   lbuser=[0] * 7, lbrsvd=[0] * 4)
            field._x_coord_name = lambda: 'longitude'
            field._y_coord_name = lambda: 'latitude'
            field.coord_system = lambda: None
            return field

        # Make a fake reference surface field.
        pressure_field = field_with_data(10)
        pressure_field.stash = iris.fileformats.pp.STASH(1, 0, 409)
        pressure_field.lbuser[3] = 409

        # Make a fake data field which needs the reference surface.
        model_level = 5678
        sigma_lower, sigma, sigma_upper = 0.85, 0.9, 0.95
        delta_lower, delta, delta_upper = 0.05, 0.1, 0.15
        data_field = field_with_data()
        data_field.configure_mock(lbvc=9, lblev=model_level,
                                  blev=delta, brlev=delta_lower,
                                  bhlev=sigma, bhrlev=sigma_lower,
                                  brsvd=[delta_upper, sigma_upper])

        # Convert both fields to cubes.
        load = mock.Mock(return_value=iter([pressure_field, data_field]))
        with mock.patch('iris.fileformats.pp.load', new=load) as load:
            pressure_cube, data_cube = iris.fileformats.pp.load_cubes('DUMMY')

        # Check the reference surface cube looks OK.
        self.assertEqual(pressure_cube.standard_name, 'surface_air_pressure')
        self.assertEqual(pressure_cube.units, 'Pa')

        # Check the data cube is set up to use hybrid-pressure.
        self._test_coord(data_cube, model_level,
                         standard_name='model_level_number')
        self._test_coord(data_cube, delta, [delta_lower, delta_upper],
                         long_name='delta')
        self._test_coord(data_cube, REFERENCE_PRESSURE,
                         long_name='reference_pressure')
        self._test_coord(data_cube, sigma, [sigma_lower, sigma_upper],
                         long_name='sigma')
        aux_factories = data_cube.aux_factories
        self.assertEqual(len(aux_factories), 1)
        surface_coord = aux_factories[0].dependencies['surface_air_pressure']
        self.assertArrayEqual(surface_coord.points,
                              np.arange(12000, step=10).reshape(30, 40))

        # Now use the save rules to convert the Cubes back into PPFields.
        pressure_field = iris.fileformats.pp.PPField3()
        pressure_field.lbfc = 0
        pressure_field.lbvc = 0
        pressure_field.brsvd = [None, None]
        pressure_field.lbuser = [None] * 7
        iris.fileformats.pp._ensure_save_rules_loaded()
        iris.fileformats.pp._save_rules.verify(pressure_cube, pressure_field)

        data_field = iris.fileformats.pp.PPField3()
        data_field.lbfc = 0
        data_field.lbvc = 0
        data_field.brsvd = [None, None]
        data_field.lbuser = [None] * 7
        data_cube.standard_name = 'air_temperature'  # XXX Workaround until #892
        iris.fileformats.pp._save_rules.verify(data_cube, data_field)

        # The reference surface field should have STASH=409
        self.assertArrayEqual(pressure_field.lbuser,
                              [None, None, None, 409, None, None, 1])

        # Check the data field has the vertical coordinate as originally
        # specified.
        self.assertEqual(data_field.lbvc, 9)
        self.assertEqual(data_field.lblev, model_level)
        self.assertEqual(data_field.blev, delta)
        self.assertEqual(data_field.brlev, delta_lower)
        self.assertEqual(data_field.bhlev, sigma)
        self.assertEqual(data_field.bhrlev, sigma_lower)
        self.assertEqual(data_field.brsvd, [delta_upper, sigma_upper])


class TestCoordinateForms(tests.IrisTest):
    def _common(self, x_coord):
        nx = len(x_coord.points)
        ny = 2
        data = np.zeros((ny, nx), dtype=np.float32)
        test_cube = iris.cube.Cube(data)
        y0 = np.float32(20.5)
        dy = np.float32(3.72)
        y_coord = iris.coords.DimCoord.from_regular(
            zeroth=y0,
            step=dy,
            count=ny,
            standard_name='latitude',
            units='degrees_north')
        test_cube.add_dim_coord(x_coord, 1)
        test_cube.add_dim_coord(y_coord, 0)
        # Write to a temporary PP file and read it back as a PPField
        with self.temp_filename('.pp') as pp_filepath:
            iris.save(test_cube, pp_filepath)
            pp_loader = iris.fileformats.pp.load(pp_filepath)
            pp_field = pp_loader.next()
        return pp_field

    def test_save_awkward_case_is_regular(self):
        # Check that specific "awkward" values still save in a regular form.
        nx = 3
        x0 = np.float32(355.626)
        dx = np.float32(0.0135)
        x_coord = iris.coords.DimCoord.from_regular(
            zeroth=x0,
            step=dx,
            count=nx,
            standard_name='longitude',
            units='degrees_east')
        pp_field = self._common(x_coord)
        # Check that the result has the regular coordinates as expected.
        self.assertEqual(pp_field.bzx, x0)
        self.assertEqual(pp_field.bdx, dx)
        self.assertEqual(pp_field.lbnpt, nx)

    def test_save_irregular(self):
        # Check that a non-regular coordinate saves as expected.
        nx = 3
        x_values = [0.0, 1.1, 2.0]
        x_coord = iris.coords.DimCoord(x_values,
                                       standard_name='longitude',
                                       units='degrees_east')
        pp_field = self._common(x_coord)
        # Check that the result has the regular/irregular Y and X as expected.
        self.assertEqual(pp_field.bdx, 0.0)
        self.assertArrayAllClose(pp_field.x, x_values)
        self.assertEqual(pp_field.lbnpt, nx)


if __name__ == "__main__":
    tests.main()
