# (C) British Crown Copyright 2014 - 2020, Met Office
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

from cf_units import Unit
import numpy.ma as ma

import iris
from iris import load_cube, save
from iris.coords import CellMethod, DimCoord
from iris.coord_systems import RotatedGeogCS
from iris.fileformats.pp import EARTH_RADIUS as UM_DEFAULT_EARTH_RADIUS
import iris.tests.stock as stock
from iris.util import is_regular

# Grib support is optional.
if tests.GRIB_AVAILABLE:
    from iris_grib import load_pairs_from_fields
    from iris_grib.message import GribMessage
    try:
        from iris_grib.grib_phenom_translation import GRIBCode
    except ImportError:
        GRIBCode = None


@tests.skip_data
@tests.skip_grib
class TestImport(tests.IrisTest):
    def test_gdt1(self):
        path = tests.get_data_path(('GRIB', 'rotated_nae_t',
                                    'sensible_pole.grib2'))
        cube = load_cube(path)
        self.assertCMLApproxData(cube)

    def test_gdt90_with_bitmap(self):
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
@tests.skip_grib
class TestPDT8(tests.IrisTest):
    def setUp(self):
        # Load from the test file.
        file_path = tests.get_data_path(('GRIB', 'time_processed',
                                         'time_bound.grib2'))
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
@tests.skip_grib
class TestPDT11(tests.TestGribMessage):
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

            # Check that various aspects of the saved file are as expected.
            expect_values = (
                (0, 'editionNumber',  2),
                (3, 'gridDefinitionTemplateNumber', 0),
                (4, 'productDefinitionTemplateNumber', 11),
                (4, 'perturbationNumber', 1),
                (4, 'typeOfStatisticalProcessing', 0),
                (4, 'numberOfForecastsInEnsemble', 255))
            self.assertGribMessageContents(temp_file_path, expect_values)


@tests.skip_grib
class TestPDT40(tests.IrisTest):
    def test_save_load(self):
        cube = stock.lat_lon_cube()
        cube.rename('atmosphere_mole_content_of_ozone')
        cube.units = Unit('Dobson')
        tcoord = DimCoord(23, 'time',
                          units=Unit('days since epoch', calendar='standard'))
        fpcoord = DimCoord(24, 'forecast_period', units=Unit('hours'))
        cube.add_aux_coord(tcoord)
        cube.add_aux_coord(fpcoord)
        cube.attributes["WMO_constituent_type"] = 0
        if GRIBCode is not None:
            cube.attributes["GRIB_PARAM"] = GRIBCode("GRIB2:d000c014n000")

        with self.temp_filename('test_grib_pdt40.grib2') as temp_file_path:
            save(cube, temp_file_path)
            loaded = load_cube(temp_file_path)
            self.assertEqual(loaded.attributes, cube.attributes)


@tests.skip_data
@tests.skip_grib
class TestGDT5(tests.TestGribMessage):
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

        # Write to temporary file, check that key contents are in the file,
        # then load back in.
        with self.temp_filename('ukv_sample.grib2') as temp_file_path:
            save(cube, temp_file_path)

            # Check that various aspects of the saved file are as expected.
            expect_values = (
                (0, 'editionNumber', 2),
                (3, 'gridDefinitionTemplateNumber', 5),
                (3, 'Ni', cube.shape[-1]),
                (3, 'Nj', cube.shape[-2]),
                (3, 'shapeOfTheEarth', 1),
                (3, 'scaledValueOfRadiusOfSphericalEarth',
                 int(UM_DEFAULT_EARTH_RADIUS)),
                (3, 'resolutionAndComponentFlags', 0),
                (3, 'latitudeOfSouthernPole', -37500000),
                (3, 'longitudeOfSouthernPole', 357500000),
                (3, 'angleOfRotation', 0))
            self.assertGribMessageContents(temp_file_path, expect_values)

            # Load the Grib file back into a new cube.
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

        if GRIBCode is not None:
            # Python3 only --> iris-grib version >= 0.15
            # Check only the GRIB_PARAM attribute exists on the re-loaded cube.
            # Note: this does *not* match the original, but is as expected.
            self.assertEqual(
                cube_loaded_from_saved.attributes,
                {"GRIB_PARAM": GRIBCode("GRIB2:d000c003n001")},
            )

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
@tests.skip_grib
class TestGDT30(tests.IrisTest):

    def test_lambert(self):
        path = tests.get_data_path(('GRIB', 'lambert', 'lambert.grib2'))
        cube = load_cube(path)
        self.assertCMLApproxData(cube)


@tests.skip_data
@tests.skip_grib
class TestGDT40(tests.IrisTest):

    def test_regular(self):
        path = tests.get_data_path(('GRIB', 'gaussian', 'regular_gg.grib2'))
        cube = load_cube(path)
        self.assertCMLApproxData(cube)

    def test_reduced(self):
        path = tests.get_data_path(('GRIB', 'reduced', 'reduced_gg.grib2'))
        cube = load_cube(path)
        self.assertCMLApproxData(cube)


@tests.skip_data
@tests.skip_grib
class TestDRT3(tests.IrisTest):

    def test_grid_complex_spatial_differencing(self):
        path = tests.get_data_path(('GRIB', 'missing_values',
                                    'missing_values.grib2'))
        cube = load_cube(path)
        self.assertCMLApproxData(cube)


@tests.skip_data
@tests.skip_grib
class TestAsCubes(tests.IrisTest):
    def setUp(self):
        # Load from the test file.
        self.file_path = tests.get_data_path(('GRIB', 'time_processed',
                                              'time_bound.grib2'))

    def test_year_filter(self):
        msgs = GribMessage.messages_from_filename(self.file_path)
        chosen_messages = []
        for gmsg in msgs:
            if gmsg.sections[1]['year'] == 1998:
                chosen_messages.append(gmsg)
        cubes_msgs = list(load_pairs_from_fields(chosen_messages))
        self.assertEqual(len(cubes_msgs), 1)

    def test_year_filter_none(self):
        msgs = GribMessage.messages_from_filename(self.file_path)
        chosen_messages = []
        for gmsg in msgs:
            if gmsg.sections[1]['year'] == 1958:
                chosen_messages.append(gmsg)
        cubes_msgs = list(load_pairs_from_fields(chosen_messages))
        self.assertEqual(len(cubes_msgs), 0)

    def test_as_pairs(self):
        messages = GribMessage.messages_from_filename(self.file_path)
        cubes = []
        cube_msg_pairs = load_pairs_from_fields(messages)
        for cube, gmsg in cube_msg_pairs:
            if gmsg.sections[1]['year'] == 1998:
                cube.attributes['the year is'] = gmsg.sections[1]['year']
                cubes.append(cube)
        self.assertEqual(len(cubes), 1)
        self.assertEqual(cubes[0].attributes['the year is'], 1998)


if __name__ == '__main__':
    tests.main()
