# (C) British Crown Copyright 2013 - 2015, Met Office
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

from __future__ import (absolute_import, division, print_function)

import six

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from contextlib import nested

import mock
import numpy as np

from iris.aux_factory import HybridHeightFactory, HybridPressureFactory
from iris.coords import AuxCoord, CellMethod
from iris.cube import Cube
import iris.fileformats.pp
import iris.fileformats.pp_rules


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
            field = mock.MagicMock(_data=np.arange(1200).reshape(y, x) * scale,
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
                                  bhlev=delta, bhrlev=delta_lower,
                                  blev=sigma, brlev=sigma_lower,
                                  brsvd=[sigma_upper, delta_upper])

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
                         long_name='level_pressure')
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
        iris.fileformats.pp._save_rules.verify(data_cube, data_field)

        # The reference surface field should have STASH=409
        self.assertArrayEqual(pressure_field.lbuser,
                              [None, None, None, 409, None, None, 1])

        # Check the data field has the vertical coordinate as originally
        # specified.
        self.assertEqual(data_field.lbvc, 9)
        self.assertEqual(data_field.lblev, model_level)
        self.assertEqual(data_field.bhlev, delta)
        self.assertEqual(data_field.bhrlev, delta_lower)
        self.assertEqual(data_field.blev, sigma)
        self.assertEqual(data_field.brlev, sigma_lower)
        self.assertEqual(data_field.brsvd, [sigma_upper, delta_upper])

    def test_hybrid_pressure_with_duplicate_references(self):
        def field_with_data(scale=1):
            x, y = 40, 30
            field = mock.MagicMock(_data=np.arange(1200).reshape(y, x) * scale,
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
                                  bhlev=delta, bhrlev=delta_lower,
                                  blev=sigma, brlev=sigma_lower,
                                  brsvd=[sigma_upper, delta_upper])

        # Convert both fields to cubes.
        load = mock.Mock(return_value=iter([data_field,
                                            pressure_field,
                                            pressure_field]))
        msg = 'Multiple reference cubes for surface_air_pressure'
        with nested(mock.patch('iris.fileformats.pp.load', new=load),
                    mock.patch('warnings.warn')) as (load, warn):
            _, _, _ = iris.fileformats.pp.load_cubes('DUMMY')
            warn.assert_called_with(msg)

    def test_hybrid_height_with_non_standard_coords(self):
        # Check the save rules are using the AuxFactory to find the
        # hybrid height coordinates and not relying on their names.
        ny, nx = 30, 40
        sigma_lower, sigma, sigma_upper = 0.75, 0.8, 0.75
        delta_lower, delta, delta_upper = 150, 200, 250

        cube = Cube(np.zeros((ny, nx)), 'air_temperature')
        level_coord = AuxCoord(0, 'model_level_number')
        cube.add_aux_coord(level_coord)
        delta_coord = AuxCoord(delta, bounds=[[delta_lower, delta_upper]],
                               long_name='moog', units='m')
        sigma_coord = AuxCoord(sigma, bounds=[[sigma_lower, sigma_upper]],
                               long_name='mavis')
        surface_altitude_coord = AuxCoord(np.zeros((ny, nx)),
                                          'surface_altitude', units='m')
        cube.add_aux_coord(delta_coord)
        cube.add_aux_coord(sigma_coord)
        cube.add_aux_coord(surface_altitude_coord, (0, 1))
        cube.add_aux_factory(HybridHeightFactory(delta_coord, sigma_coord,
                                                 surface_altitude_coord))

        field = iris.fileformats.pp.PPField3()
        field.lbfc = 0
        field.lbvc = 0
        field.brsvd = [None, None]
        field.lbuser = [None] * 7
        iris.fileformats.pp._ensure_save_rules_loaded()
        iris.fileformats.pp._save_rules.verify(cube, field)

        self.assertEqual(field.blev, delta)
        self.assertEqual(field.brlev, delta_lower)
        self.assertEqual(field.bhlev, sigma)
        self.assertEqual(field.bhrlev, sigma_lower)
        self.assertEqual(field.brsvd, [delta_upper, sigma_upper])

    def test_hybrid_pressure_with_non_standard_coords(self):
        # Check the save rules are using the AuxFactory to find the
        # hybrid pressure coordinates and not relying on their names.
        ny, nx = 30, 40
        sigma_lower, sigma, sigma_upper = 0.75, 0.8, 0.75
        delta_lower, delta, delta_upper = 0.15, 0.2, 0.25

        cube = Cube(np.zeros((ny, nx)), 'air_temperature')
        level_coord = AuxCoord(0, 'model_level_number')
        cube.add_aux_coord(level_coord)
        delta_coord = AuxCoord(delta, bounds=[[delta_lower, delta_upper]],
                               long_name='moog', units='Pa')
        sigma_coord = AuxCoord(sigma, bounds=[[sigma_lower, sigma_upper]],
                               long_name='mavis')
        surface_air_pressure_coord = AuxCoord(np.zeros((ny, nx)),
                                              'surface_air_pressure',
                                              units='Pa')
        cube.add_aux_coord(delta_coord)
        cube.add_aux_coord(sigma_coord)
        cube.add_aux_coord(surface_air_pressure_coord, (0, 1))
        cube.add_aux_factory(HybridPressureFactory(
            delta_coord, sigma_coord, surface_air_pressure_coord))

        field = iris.fileformats.pp.PPField3()
        field.lbfc = 0
        field.lbvc = 0
        field.brsvd = [None, None]
        field.lbuser = [None] * 7
        iris.fileformats.pp._ensure_save_rules_loaded()
        iris.fileformats.pp._save_rules.verify(cube, field)

        self.assertEqual(field.bhlev, delta)
        self.assertEqual(field.bhrlev, delta_lower)
        self.assertEqual(field.blev, sigma)
        self.assertEqual(field.brlev, sigma_lower)
        self.assertEqual(field.brsvd, [sigma_upper, delta_upper])

    def test_hybrid_height_round_trip_no_reference(self):
        # Use pp.load_cubes() to convert fake PPFields into Cubes.
        # NB. Use MagicMock so that SplittableInt header items, such as
        # LBCODE, support len().
        def field_with_data(scale=1):
            x, y = 40, 30
            field = mock.MagicMock(_data=np.arange(1200).reshape(y, x) * scale,
                                   lbcode=[1], lbnpt=x, lbrow=y,
                                   bzx=350, bdx=1.5, bzy=40, bdy=1.5,
                                   lbuser=[0] * 7, lbrsvd=[0] * 4)
            field._x_coord_name = lambda: 'longitude'
            field._y_coord_name = lambda: 'latitude'
            field.coord_system = lambda: None
            return field

        # Make a fake data field which needs the reference surface.
        model_level = 5678
        sigma_lower, sigma, sigma_upper = 0.85, 0.9, 0.95
        delta_lower, delta, delta_upper = 0.05, 0.1, 0.15
        data_field = field_with_data()
        data_field.configure_mock(lbvc=65, lblev=model_level,
                                  bhlev=sigma, bhrlev=sigma_lower,
                                  blev=delta, brlev=delta_lower,
                                  brsvd=[delta_upper, sigma_upper])

        # Convert field to a cube.
        load = mock.Mock(return_value=iter([data_field]))
        with mock.patch('iris.fileformats.pp.load', new=load) as load, \
                mock.patch('warnings.warn') as warn:
            data_cube, = iris.fileformats.pp.load_cubes('DUMMY')

        msg = "Unable to create instance of HybridHeightFactory. " \
              "The file(s) ['DUMMY'] don't contain field(s) for 'orography'."
        warn.assert_called_once_with(msg)

        # Check the data cube is set up to use hybrid height.
        self._test_coord(data_cube, model_level,
                         standard_name='model_level_number')
        self._test_coord(data_cube, delta, [delta_lower, delta_upper],
                         long_name='level_height')
        self._test_coord(data_cube, sigma, [sigma_lower, sigma_upper],
                         long_name='sigma')
        # Check that no aux factory is created (due to missing
        # reference surface).
        aux_factories = data_cube.aux_factories
        self.assertEqual(len(aux_factories), 0)

        # Now use the save rules to convert the Cube back into a PPField.
        data_field = iris.fileformats.pp.PPField3()
        data_field.lbfc = 0
        data_field.lbvc = 0
        data_field.brsvd = [None, None]
        data_field.lbuser = [None] * 7
        iris.fileformats.pp._ensure_save_rules_loaded()
        iris.fileformats.pp._save_rules.verify(data_cube, data_field)

        # Check the data field has the vertical coordinate as originally
        # specified.
        self.assertEqual(data_field.lbvc, 65)
        self.assertEqual(data_field.lblev, model_level)
        self.assertEqual(data_field.bhlev, sigma)
        self.assertEqual(data_field.bhrlev, sigma_lower)
        self.assertEqual(data_field.blev, delta)
        self.assertEqual(data_field.brlev, delta_lower)
        self.assertEqual(data_field.brsvd, [delta_upper, sigma_upper])


class TestSaveLBFT(tests.IrisTest):
    def create_cube(self, fp_min, fp_mid, fp_max, ref_offset, season=None):
        cube = Cube(np.zeros((3, 4)))
        cube.add_aux_coord(AuxCoord(standard_name='forecast_period',
                                    units='hours',
                                    points=fp_mid, bounds=[fp_min, fp_max]))
        cube.add_aux_coord(AuxCoord(standard_name='time',
                                    units='hours since epoch',
                                    points=ref_offset + fp_mid,
                                    bounds=[ref_offset + fp_min,
                                            ref_offset + fp_max]))
        if season:
            cube.add_aux_coord(AuxCoord(long_name='clim_season',
                                        points=season))
            cube.add_cell_method(CellMethod('DUMMY', 'clim_season'))
        return cube

    def convert_cube_to_field(self, cube):
        # Use the save rules to convert the Cube back into a PPField.
        field = iris.fileformats.pp.PPField3()
        field.lbfc = 0
        field.lbvc = 0
        field.lbtim = 0
        iris.fileformats.pp._ensure_save_rules_loaded()
        iris.fileformats.pp._save_rules.verify(cube, field)
        return field

    def test_time_mean_from_forecast_period(self):
        cube = self.create_cube(24, 36, 48, 72)
        field = self.convert_cube_to_field(cube)
        self.assertEqual(field.lbft, 48)

    def test_time_mean_from_forecast_reference_time(self):
        cube = Cube(np.zeros((3, 4)))
        cube.add_aux_coord(AuxCoord(standard_name='forecast_reference_time',
                                    units='hours since epoch',
                                    points=72))
        cube.add_aux_coord(AuxCoord(standard_name='time',
                                    units='hours since epoch',
                                    points=72 + 36, bounds=[72 + 24, 72 + 48]))
        field = self.convert_cube_to_field(cube)
        self.assertEqual(field.lbft, 48)

    def test_climatological_mean_single_year(self):
        cube = Cube(np.zeros((3, 4)))
        cube.add_aux_coord(AuxCoord(standard_name='forecast_period',
                                    units='hours',
                                    points=36, bounds=[24, 4 * 24]))
        cube.add_aux_coord(AuxCoord(standard_name='time',
                                    units='hours since epoch',
                                    points=240 + 36, bounds=[240 + 24,
                                                             240 + 4 * 24]))
        cube.add_aux_coord(AuxCoord(long_name='clim_season', points='DUMMY'))
        cube.add_cell_method(CellMethod('DUMMY', 'clim_season'))
        field = self.convert_cube_to_field(cube)
        self.assertEqual(field.lbft, 4 * 24)

    def test_climatological_mean_multi_year_djf(self):
        delta_start = 24
        delta_mid = 36
        delta_end = 369 * 24
        ref_offset = 10 * 24
        cube = self.create_cube(24, 36, 369 * 24, 240, 'djf')
        field = self.convert_cube_to_field(cube)
        self.assertEqual(field.lbft, 369 * 24)

    def test_climatological_mean_multi_year_mam(self):
        cube = self.create_cube(24, 36, 369 * 24, 240, 'mam')
        field = self.convert_cube_to_field(cube)
        self.assertEqual(field.lbft, 369 * 24)

    def test_climatological_mean_multi_year_jja(self):
        cube = self.create_cube(24, 36, 369 * 24, 240, 'jja')
        field = self.convert_cube_to_field(cube)
        self.assertEqual(field.lbft, 369 * 24)

    def test_climatological_mean_multi_year_son(self):
        cube = self.create_cube(24, 36, 369 * 24, 240, 'son')
        field = self.convert_cube_to_field(cube)
        self.assertEqual(field.lbft, 369 * 24)


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
            pp_field = next(pp_loader)
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
