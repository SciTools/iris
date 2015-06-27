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
"""Integration tests for loading and saving GRIB2 files."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy.ma as ma

from iris import FUTURE, load_cube

from subprocess import check_output

import iris
from iris import FUTURE, load_cube, save
from iris.coords import CellMethod
from iris.coord_systems import RotatedGeogCS
from iris.fileformats.pp import EARTH_RADIUS as UM_DEFAULT_EARTH_RADIUS
from iris.util import is_regular


@tests.skip_data
class TestImport(tests.IrisTest):
    def test_gdt1(self):
        with FUTURE.context(strict_grib_load=True):
            path = tests.get_data_path(('GRIB', 'rotated_nae_t',
                                        'sensible_pole.grib2'))
            cube = load_cube(path)
            self.assertCMLApproxData(cube)

    def test_gdt90_with_bitmap(self):
        with FUTURE.context(strict_grib_load=True):
            path = tests.get_data_path(('GRIB', 'umukv', 'ukv_chan9.grib2'))
            cube = load_cube(path)
            # Pay particular attention to the orientation.
            self.assertIsNot(cube.data[0, 0], ma.masked)
            self.assertIs(cube.data[-1, 0], ma.masked)
            self.assertIs(cube.data[0, -1], ma.masked)
            self.assertIs(cube.data[-1, -1], ma.masked)
            x = cube.coord('projection_x_coordinate').points
            y = cube.coord('projection_y_coordinate').points
            self.assertGreater(x[0], x[-1])  # Decreasing X coordinate
            self.assertLess(y[0], y[-1])  # Increasing Y coordinate
            # Check everything else.
            self.assertCMLApproxData(cube)


@tests.skip_data
class TestPDT8(tests.IrisTest):
    def setUp(self):
        # Load from the test file.
        file_path = tests.get_data_path(('GRIB', 'time_processed',
                                         'time_bound.grib2'))
        with FUTURE.context(strict_grib_load=True):
            self.cube = load_cube(file_path)

    def test_coords(self):
        # Check the result has main coordinates as expected.
        for name, shape, is_bounded in [
                ('forecast_reference_time', (1,), False),
                ('time', (1,), True),
                ('forecast_period', (1,), True),
                ('pressure', (1,), False),
                ('latitude', (73,), False),
                ('longitude', (96,), False)]:
            coords = self.cube.coords(name)
            self.assertEqual(len(coords), 1,
                             'expected one {!r} coord, found {}'.format(
                                 name, len(coords)))
            coord, = coords
            self.assertEqual(coord.shape, shape,
                             'coord {!r} shape is {} instead of {!r}.'.format(
                                 name, coord.shape, shape))
            self.assertEqual(coord.has_bounds(), is_bounded,
                             'coord {!r} has_bounds={}, expected {}.'.format(
                                 name, coord.has_bounds(), is_bounded))

    def test_cell_method(self):
        # Check the result has the expected cell method.
        cell_methods = self.cube.cell_methods
        self.assertEqual(len(cell_methods), 1,
                         'result has {} cell methods, expected one.'.format(
                             len(cell_methods)))
        cell_method, = cell_methods
        self.assertEqual(cell_method.coord_names, ('time',))


@tests.skip_data
class TestPDT11(tests.IrisTest):
    def test_perturbation(self):
        path = tests.get_data_path(('NetCDF', 'global', 'xyt',
                                    'SMALL_hires_wind_u_for_ipcc4.nc'))
        cube = load_cube(path)
        # trim to 1 time and regular lats
        cube = cube[0, 12:144, :]
        crs = iris.coord_systems.GeogCS(6371229)
        cube.coord('latitude').coord_system = crs
        cube.coord('longitude').coord_system = crs
        # add a realization coordinate
        cube.add_aux_coord(iris.coords.DimCoord(points=1,
                                                standard_name='realization',
                                                units='1'))
        with self.temp_filename('testPDT11.GRIB2') as temp_file_path:
            iris.save(cube, temp_file_path)
            # Get a grib_dump of the output file.
            dump_text = check_output(('grib_dump -O -wcount=1 ' +
                                      temp_file_path),
                                     shell=True).decode()

            # Check that various aspects of the saved file are as expected.
            expect_strings = (
                'editionNumber = 2',
                'gridDefinitionTemplateNumber = 0',
                'productDefinitionTemplateNumber = 11',
                'perturbationNumber = 1',
                'typeOfStatisticalProcessing = 0',
                'numberOfForecastsInEnsemble = 255')
            for expect in expect_strings:
                self.assertIn(expect, dump_text)


@tests.skip_data
class TestGDT5(tests.IrisTest):
    def test_save_load(self):
        # Load sample UKV data (variable-resolution rotated grid).
        path = tests.get_data_path(('PP', 'ukV1', 'ukVpmslont.pp'))
        cube = load_cube(path)

        # Extract a single 2D field, for simplicity.
        self.assertEqual(cube.ndim, 3)
        self.assertEqual(cube.coord_dims('time'), (0,))
        cube = cube[0]

        # FOR NOW: **also** fix the data so that it is square, i.e. nx=ny.
        # This is needed because of a bug in the gribapi.
        # See : https://software.ecmwf.int/issues/browse/SUP-1096
        ny, nx = cube.shape
        nn = min(nx, ny)
        cube = cube[:nn, :nn]

        # Check that it has a rotated-pole variable-spaced grid, as expected.
        x_coord = cube.coord(axis='x')
        self.assertIsInstance(x_coord.coord_system, RotatedGeogCS)
        self.assertFalse(is_regular(x_coord))

        # Write to temporary file, check grib_dump output, and load back in.
        with self.temp_filename('ukv_sample.grib2') as temp_file_path:
            save(cube, temp_file_path)

            # Get a grib_dump of the output file.
            dump_text = check_output(('grib_dump -O -wcount=1 ' +
                                      temp_file_path),
                                     shell=True).decode()

            # Check that various aspects of the saved file are as expected.
            expect_strings = (
                'editionNumber = 2',
                'gridDefinitionTemplateNumber = 5',
                'Ni = {:d}'.format(cube.shape[-1]),
                'Nj = {:d}'.format(cube.shape[-2]),
                'shapeOfTheEarth = 1',
                'scaledValueOfRadiusOfSphericalEarth = {:d}'.format(
                    int(UM_DEFAULT_EARTH_RADIUS)),
                'resolutionAndComponentFlags = 0',
                'latitudeOfSouthernPole = -37500000',
                'longitudeOfSouthernPole = 357500000',
                'angleOfRotation = 0')
            for expect in expect_strings:
                self.assertIn(expect, dump_text)

            # Load the Grib file back into a new cube.
            with FUTURE.context(strict_grib_load=True):
                cube_loaded_from_saved = load_cube(temp_file_path)
                # Also load data, before the temporary file gets deleted.
                cube_loaded_from_saved.data

        # The re-loaded result will not match the original in every respect:
        #  * cube attributes are discarded
        #  * horizontal coordinates are rounded to an integer representation
        #  * bounds on horizontal coords are lost
        # Thus the following "equivalence tests" are rather piecemeal..

        # Check those re-loaded properties which should match the original.
        for test_cube in (cube, cube_loaded_from_saved):
            self.assertEqual(test_cube.standard_name,
                             'air_pressure_at_sea_level')
            self.assertEqual(test_cube.units, 'Pa')
            self.assertEqual(test_cube.shape, (744, 744))
            self.assertEqual(test_cube.cell_methods, ())

        # Check no cube attributes on the re-loaded cube.
        # Note: this does *not* match the original, but is as expected.
        self.assertEqual(cube_loaded_from_saved.attributes, {})

        # Now remaining to check: coordinates + data...

        # Check they have all the same coordinates.
        co_names = [coord.name() for coord in cube.coords()]
        co_names_reload = [coord.name()
                           for coord in cube_loaded_from_saved.coords()]
        self.assertEqual(sorted(co_names_reload), sorted(co_names))

        # Check all the coordinates.
        for coord_name in co_names:
            try:
                co_orig = cube.coord(coord_name)
                co_load = cube_loaded_from_saved.coord(coord_name)

                # Check shape.
                self.assertEqual(co_load.shape, co_orig.shape,
                                 'Shape of re-loaded "{}" coord is {} '
                                 'instead of {}'.format(coord_name,
                                                        co_load.shape,
                                                        co_orig.shape))

                # Check coordinate points equal, within a tolerance.
                self.assertArrayAllClose(co_load.points, co_orig.points,
                                         rtol=1.0e-6)

                # Check all coords are unbounded.
                # (NOTE: this is not so for the original X and Y coordinates,
                # but Grib does not store those bounds).
                self.assertIsNone(co_load.bounds)

            except AssertionError as err:
                self.assertTrue(False,
                                'Failed on coordinate "{}" : {}'.format(
                                    coord_name, str(err)))

        # Check that main data array also matches.
        self.assertArrayAllClose(cube.data, cube_loaded_from_saved.data)


@tests.skip_data
class TestGDT30(tests.IrisTest):

    def test_lambert(self):
        path = tests.get_data_path(('GRIB', 'lambert', 'lambert.grib2'))
        with FUTURE.context(strict_grib_load=True):
            cube = load_cube(path)
        self.assertCMLApproxData(cube)


@tests.skip_data
class TestGDT40(tests.IrisTest):

    def test_regular(self):
        path = tests.get_data_path(('GRIB', 'gaussian', 'regular_gg.grib2'))
        with FUTURE.context(strict_grib_load=True):
            cube = load_cube(path)
        self.assertCMLApproxData(cube)

    def test_reduced(self):
        path = tests.get_data_path(('GRIB', 'reduced', 'reduced_gg.grib2'))
        with FUTURE.context(strict_grib_load=True):
            cube = load_cube(path)
        self.assertCMLApproxData(cube)


@tests.skip_data
class TestDRT3(tests.IrisTest):

    def test_grid_complex_spatial_differencing(self):
        path = tests.get_data_path(('GRIB', 'missing_values',
                                    'missing_values.grib2'))
        with FUTURE.context(strict_grib_load=True):
            cube = load_cube(path)
        self.assertCMLApproxData(cube)


if __name__ == '__main__':
    tests.main()
