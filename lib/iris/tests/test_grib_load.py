# (C) British Crown Copyright 2010 - 2013, Met Office
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

# Import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import datetime
import os

import gribapi
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mock
import numpy as np

import iris
import iris.exceptions
import iris.fileformats.grib
import iris.plot as iplt
import iris.quickplot as qplt
import iris.tests.stock
import iris.util

# Construct a mock object to mimic the gribapi for GribWrapper testing.
_mock_gribapi = mock.Mock(spec=gribapi)
_mock_gribapi.GribInternalError = Exception


def _mock_gribapi_fetch(message, key):
    """
    Fake the gribapi key-fetch.

    Fetch key-value from the fake message (dictionary).
    If the key is not present, raise the diagnostic exception.

    """
    if key in message.keys():
        return message[key]
    else:
        raise _mock_gribapi.GribInternalError


def _mock_gribapi__grib_is_missing(grib_message, keyname):
    """
    Fake the gribapi key-existence enquiry.

    Return whether the key exists in the fake message (dictionary).

    """
    return (keyname not in grib_message)


def _mock_gribapi__grib_get_native_type(grib_message, keyname):
    """
    Fake the gribapi type-discovery operation.

    Return type of key-value in the fake message (dictionary).
    If the key is not present, raise the diagnostic exception.

    """
    if keyname in grib_message:
        return type(grib_message[keyname])
    raise _mock_gribapi.GribInternalError(keyname)

_mock_gribapi.grib_get_long = mock.Mock(side_effect=_mock_gribapi_fetch)
_mock_gribapi.grib_get_string = mock.Mock(side_effect=_mock_gribapi_fetch)
_mock_gribapi.grib_get_double = mock.Mock(side_effect=_mock_gribapi_fetch)
_mock_gribapi.grib_get_double_array = mock.Mock(
    side_effect=_mock_gribapi_fetch)
_mock_gribapi.grib_is_missing = mock.Mock(
    side_effect=_mock_gribapi__grib_is_missing)
_mock_gribapi.grib_get_native_type = mock.Mock(
    side_effect=_mock_gribapi__grib_get_native_type)

# define seconds in an hour, for general test usage
_hour_secs = 3600.0


class FakeGribMessage(dict):
    """
    A 'fake grib message' object, for testing GribWrapper construction.

    Behaves as a dictionary, containing key-values for message keys.

    """
    def __init__(self, **kwargs):
        """
        Create a fake message object.

        General keys can be set/add as required via **kwargs.
        The keys 'edition' and 'time_code' are specially managed.

        """
        # Start with a bare dictionary
        dict.__init__(self)
        # Extract specially-recognised keys.
        edition = kwargs.pop('edition', 1)
        time_code = kwargs.pop('time_code', None)
        # Set the minimally required keys.
        self._init_minimal_message(edition=edition)
        # Also set a time-code, if given.
        if time_code is not None:
            self.set_timeunit_code(time_code)
        # Finally, add any remaining passed key-values.
        self.update(**kwargs)

    def _init_minimal_message(self, edition=1):
        # Set values for all the required keys.
        # 'edition' controls the edition-specific keys.
        self.update({
            'Ni': 1,
            'Nj': 1,
            'alternativeRowScanning': 0,
            'centre': 'ecmf',
            'year': 2007,
            'month': 3,
            'day': 23,
            'hour': 12,
            'minute': 0,
            'indicatorOfUnitOfTimeRange': 1,
            'shapeOfTheEarth': 6,
            'gridType': 'rotated_ll',
            'angleOfRotation': 0.0,
            'iDirectionIncrementInDegrees': 0.036,
            'jDirectionIncrementInDegrees': 0.036,
            'iScansNegatively': 0,
            'jScansPositively': 1,
            'longitudeOfFirstGridPointInDegrees': -5.70,
            'latitudeOfFirstGridPointInDegrees': -4.452,
            'jPointsAreConsecutive': 0,
            'values': np.array([[1.0]]),
            'indicatorOfParameter': 9999,
            'parameterNumber': 9999,
        })
        # Add edition-dependent settings.
        self['edition'] = edition
        if edition == 1:
            self.update({
                'startStep': 24,
                'timeRangeIndicator': 1,
                'P1': 2, 'P2': 0,
                # time unit - needed AS WELL as 'indicatorOfUnitOfTimeRange'
                'unitOfTime': 1,
                'table2Version': 9999,
            })
        if edition == 2:
            self.update({
                'iDirectionIncrementGiven': 1,
                'jDirectionIncrementGiven': 1,
                'uvRelativeToGrid': 0,
                'forecastTime': 24,
                'productDefinitionTemplateNumber': 0,
                'stepRange': 24,
                'discipline': 9999,
                'parameterCategory': 9999,
                'tablesVersion': 4
            })

    def set_timeunit_code(self, timecode):
        # Do timecode setting (somewhat edition-dependent).
        self['indicatorOfUnitOfTimeRange'] = timecode
        if self['edition'] == 1:
            # for some odd reason, GRIB1 code uses *both* of these
            # NOTE kludge -- the 2 keys are really the same thing
            self['unitOfTime'] = timecode


@iris.tests.skip_data
class TestGribLoad(tests.GraphicsTest):

    def setUp(self):
        iris.fileformats.grib.hindcast_workaround = True

    def tearDown(self):
        iris.fileformats.grib.hindcast_workaround = False

    def test_load(self):

        cubes = iris.load(tests.get_data_path(('GRIB', 'rotated_uk',
                                               "uk_wrongparam.grib1")))
        self.assertCML(cubes, ("grib_load", "rotated.cml"))

        cubes = iris.load(tests.get_data_path(('GRIB', "time_processed",
                                               "time_bound.grib1")))
        self.assertCML(cubes, ("grib_load", "time_bound_grib1.cml"))

        cubes = iris.load(tests.get_data_path(('GRIB', "time_processed",
                                               "time_bound.grib2")))
        self.assertCML(cubes, ("grib_load", "time_bound_grib2.cml"))

        cubes = iris.load(tests.get_data_path(('GRIB', "3_layer_viz",
                                               "3_layer.grib2")))
        cubes = iris.cube.CubeList([cubes[1], cubes[0], cubes[2]])
        self.assertCML(cubes, ("grib_load", "3_layer.cml"))

    def test_load_masked(self):

        gribfile = tests.get_data_path(
            ('GRIB', 'missing_values', 'missing_values.grib2'))
        cubes = iris.load(gribfile)
        self.assertCML(cubes, ('grib_load', 'missing_values_grib2.cml'))
        
    def test_y_fastest(self):
        cubes = iris.load(tests.get_data_path(("GRIB", "y_fastest",
                                               "y_fast.grib2")))
        self.assertCML(cubes, ("grib_load", "y_fastest.cml"))
        iplt.contourf(cubes[0])
        plt.gca().coastlines()
        plt.title("y changes fastest")
        self.check_graphic()

    def test_ij_directions(self):

        def old_compat_load(name):
            cube = iris.load(tests.get_data_path(('GRIB', 'ij_directions',
                                                  name)))[0]
            return [cube]

        cubes = old_compat_load("ipos_jpos.grib2")
        self.assertCML(cubes, ("grib_load", "ipos_jpos.cml"))
        iplt.contourf(cubes[0])
        plt.gca().coastlines()
        plt.title("ipos_jpos cube")
        self.check_graphic()

        cubes = old_compat_load("ipos_jneg.grib2")
        self.assertCML(cubes, ("grib_load", "ipos_jneg.cml"))
        iplt.contourf(cubes[0])
        plt.gca().coastlines()
        plt.title("ipos_jneg cube")
        self.check_graphic()

        cubes = old_compat_load("ineg_jneg.grib2")
        self.assertCML(cubes, ("grib_load", "ineg_jneg.cml"))
        iplt.contourf(cubes[0])
        plt.gca().coastlines()
        plt.title("ineg_jneg cube")
        self.check_graphic()

        cubes = old_compat_load("ineg_jpos.grib2")
        self.assertCML(cubes, ("grib_load", "ineg_jpos.cml"))
        iplt.contourf(cubes[0])
        plt.gca().coastlines()
        plt.title("ineg_jpos cube")
        self.check_graphic()

    def test_shape_of_earth(self):

        def old_compat_load(name):
            cube = iris.load(tests.get_data_path(('GRIB', 'shape_of_earth',
                                                  name)))[0]
            return cube

        #pre-defined sphere
        cube = old_compat_load("0.grib2")
        self.assertCML(cube, ("grib_load", "earth_shape_0.cml"))

        #custom sphere
        cube = old_compat_load("1.grib2")
        self.assertCML(cube, ("grib_load", "earth_shape_1.cml"))

        #IAU65 oblate sphere 
        cube = old_compat_load("2.grib2")
        self.assertCML(cube, ("grib_load", "earth_shape_2.cml"))

        #custom oblate spheroid (km) 
        cube = old_compat_load("3.grib2")
        self.assertCML(cube, ("grib_load", "earth_shape_3.cml"))

        #IAG-GRS80 oblate spheroid 
        cube = old_compat_load("4.grib2")
        self.assertCML(cube, ("grib_load", "earth_shape_4.cml"))

        #WGS84
        cube = old_compat_load("5.grib2")
        self.assertCML(cube, ("grib_load", "earth_shape_5.cml"))

        #pre-defined sphere
        cube = old_compat_load("6.grib2")
        self.assertCML(cube, ("grib_load", "earth_shape_6.cml"))

        #custom oblate spheroid (m)
        cube = old_compat_load("7.grib2")
        self.assertCML(cube, ("grib_load", "earth_shape_7.cml"))

        #grib1 - same as grib2 shape 6, above
        cube = old_compat_load("global.grib1")
        self.assertCML(cube, ("grib_load", "earth_shape_grib1.cml"))

    def test_custom_rules(self):
        # Test custom rule evaluation.
        # Default behaviour
        cube = tests.stock.global_grib2()
        self.assertEqual(cube.name(), 'air_temperature')

        # Custom behaviour
        temp_path = iris.util.create_temp_filename()
        f = open(temp_path, 'w')
        f.write('\n'.join((
            'IF',
            'grib.edition == 2',
            'grib.discipline == 0',
            'grib.parameterCategory == 0',
            'grib.parameterNumber == 0',
            'THEN',
            'CMAttribute("long_name", "customised")',
            'CMAttribute("standard_name", None)')))
        f.close()
        iris.fileformats.grib.add_load_rules(temp_path)
        cube = tests.stock.global_grib2()
        self.assertEqual(cube.name(), 'customised')
        os.remove(temp_path)

        # Back to default
        iris.fileformats.grib.reset_load_rules()
        cube = tests.stock.global_grib2()
        self.assertEqual(cube.name(), 'air_temperature')

    def test_polar_stereo_grib1(self):
        cube = iris.load_cube(tests.get_data_path(
            ("GRIB", "polar_stereo", "ST4.2013052210.01h")))
        self.assertCML(cube, ("grib_load", "polar_stereo_grib1.cml"))
        qplt.contourf(cube, norm=LogNorm())
        plt.gca().coastlines()
        plt.gca().gridlines()
        plt.title("polar stereo grib1")
        self.check_graphic()

    def test_polar_stereo_grib2(self):
        cube = iris.load_cube(tests.get_data_path(
            ("GRIB", "polar_stereo",
             "CMC_glb_TMP_ISBL_1015_ps30km_2013052000_P006.grib2")))
        self.assertCML(cube, ("grib_load", "polar_stereo_grib2.cml"))

        qplt.contourf(cube)
        plt.gca().coastlines()
        plt.gca().gridlines()
        plt.title("polar stereo grib2")
        self.check_graphic()

    def test_lambert_grib1(self):
        cube = iris.load_cube(tests.get_data_path(
            ("GRIB", "lambert", "lambert.grib1")))
        self.assertCML(cube, ("grib_load", "lambert_grib1.cml"))

        qplt.contourf(cube)
        plt.gca().coastlines()
        plt.gca().gridlines()
        plt.title("lambert grib1")
        self.check_graphic()

    def test_lambert_grib2(self):
        cube = iris.load_cube(tests.get_data_path(
            ("GRIB", "lambert", "lambert.grib2")))
        self.assertCML(cube, ("grib_load", "lambert_grib2.cml"))

        qplt.contourf(cube)
        plt.gca().coastlines()
        plt.gca().gridlines()
        plt.title("lambert grib2")
        self.check_graphic()


class TestGribTimecodes(tests.GraphicsTest):
    def _run_timetests(self, test_set):
        # Check the unit-handling for given units-codes and editions.

        # Operates on lists of cases for various time-units and grib-editions.
        # Format: (edition, code, expected-exception,
        #          equivalent-seconds, description-string)
        with mock.patch('iris.fileformats.grib.gribapi', _mock_gribapi):
            for test_controls in test_set:
                (
                    grib_edition, timeunit_codenum,
                    expected_error,
                    timeunit_secs, timeunit_str
                ) = test_controls

                # Construct a suitable fake test message.
                message = FakeGribMessage(
                    edition=grib_edition,
                    time_code=timeunit_codenum
                )

                if expected_error:
                    # Expect GribWrapper construction to fail.
                    with self.assertRaises(type(expected_error)) as ar_context:
                        msg = iris.fileformats.grib.GribWrapper(message)
                    self.assertEqual(
                        ar_context.exception.args,
                        expected_error.args)
                    continue

                # 'ELSE'...
                # Expect the wrapper construction to work.
                # Make a GribWrapper object and test it.
                wrapped_msg = iris.fileformats.grib.GribWrapper(message)

                # Check the units string.
                forecast_timeunit = wrapped_msg._forecastTimeUnit
                self.assertEqual(
                    forecast_timeunit, timeunit_str,
                    'Bad unit string for edition={ed:01d}, '
                    'unitcode={code:01d} : '
                    'expected="{wanted}" GOT="{got}"'.format(
                        ed=grib_edition,
                        code=timeunit_codenum,
                        wanted=timeunit_str,
                        got=forecast_timeunit
                    )
                )

                # Check the data-starttime calculation.
                interval_start_to_end = (
                    wrapped_msg._phenomenonDateTime
                    - wrapped_msg._referenceDateTime
                )
                if grib_edition == 1:
                    interval_from_units = wrapped_msg.P1
                else:
                    interval_from_units = wrapped_msg.forecastTime
                interval_from_units *= datetime.timedelta(0, timeunit_secs)
                self.assertEqual(
                    interval_start_to_end, interval_from_units,
                    'Inconsistent start time offset for edition={ed:01d}, '
                    'unitcode={code:01d} : '
                    'from-unit="{unit_str}" '
                    'from-phenom-minus-ref="{e2e_str}"'.format(
                        ed=grib_edition,
                        code=timeunit_codenum,
                        unit_str=interval_from_units,
                        e2e_str=interval_start_to_end
                    )
                )

    # Test groups of testcases for various time-units and grib-editions.
    # Format: (edition, code, expected-exception,
    #          equivalent-seconds, description-string)
    def test_timeunits_common(self):
        tests = (
            (1, 0, None, 60.0, 'minutes'),
            (1, 1, None, _hour_secs, 'hours'),
            (1, 2, None, 24.0 * _hour_secs, 'days'),
            (1, 10, None, 3.0 * _hour_secs, '3 hours'),
            (1, 11, None, 6.0 * _hour_secs, '6 hours'),
            (1, 12, None, 12.0 * _hour_secs, '12 hours'),
        )
        TestGribTimecodes._run_timetests(self, tests)

    @staticmethod
    def _err_bad_timeunit(code):
        return iris.exceptions.NotYetImplementedError(
            'Unhandled time unit for forecast '
            'indicatorOfUnitOfTimeRange : {code}'.format(code=code)
        )

    def test_timeunits_grib1_specific(self):
        tests = (
            (1, 13, None, 0.25 * _hour_secs, '15 minutes'),
            (1, 14, None, 0.5 * _hour_secs, '30 minutes'),
            (1, 254, None, 1.0, 'seconds'),
            (1, 111, TestGribTimecodes._err_bad_timeunit(111), 1.0, '??'),
        )
        TestGribTimecodes._run_timetests(self, tests)

    def test_timeunits_grib2_specific(self):
        tests = (
            (2, 13, None, 1.0, 'seconds'),
            # check the extra grib1 keys FAIL
            (2, 14, TestGribTimecodes._err_bad_timeunit(14), 0.0, '??'),
            (2, 254, TestGribTimecodes._err_bad_timeunit(254), 0.0, '??'),
        )
        TestGribTimecodes._run_timetests(self, tests)

    def test_timeunits_calendar(self):
        tests = (
            (1, 3, TestGribTimecodes._err_bad_timeunit(3), 0.0, 'months'),
            (1, 4, TestGribTimecodes._err_bad_timeunit(4), 0.0, 'years'),
            (1, 5, TestGribTimecodes._err_bad_timeunit(5), 0.0, 'decades'),
            (1, 6, TestGribTimecodes._err_bad_timeunit(6), 0.0, '30 years'),
            (1, 7, TestGribTimecodes._err_bad_timeunit(7), 0.0, 'centuries'),
        )
        TestGribTimecodes._run_timetests(self, tests)

    def test_timeunits_invalid(self):
        tests = (
            (1, 111, TestGribTimecodes._err_bad_timeunit(111), 1.0, '??'),
            (2, 27, TestGribTimecodes._err_bad_timeunit(27), 1.0, '??'),
        )
        TestGribTimecodes._run_timetests(self, tests)

    def test_load_probability_forecast(self):
        # Test GribWrapper interpretation of PDT 4.9 data.
        # NOTE: 
        #   Currently Iris has only partial support for PDT 4.9.
        #   Though it can load the data, key metadata (thresholds) is lost.
        #   At present, we are not testing for this.

        # Make a testing grib message in memory, with gribapi.
        grib_message = gribapi.grib_new_from_samples('GRIB2')
        gribapi.grib_set_long(grib_message, 'productDefinitionTemplateNumber',
                              9)
        gribapi.grib_set_string(grib_message, 'stepRange', '10-55')
        grib_wrapper = iris.fileformats.grib.GribWrapper(grib_message)
        
        # Check that it captures the statistics time period info.
        # (And for now, nothing else)
        self.assertEqual(
            grib_wrapper._referenceDateTime,
            datetime.datetime(year=2007, month=03, day=23, 
                              hour=12, minute=0, second=0)
        )
        self.assertEqual(
            grib_wrapper._periodStartDateTime,
            datetime.datetime(year=2007, month=03, day=23,
                              hour=22, minute=0, second=0)
        )
        self.assertEqual(
            grib_wrapper._periodEndDateTime,
            datetime.datetime(year=2007, month=03, day=25,
                              hour=12, minute=0, second=0)
        )

    def test_warn_unknown_pdts(self):
        # Test loading of an unrecognised GRIB Product Definition Template.
        
        # Get a temporary file by name (deleted afterward by context).
        with self.temp_filename() as temp_gribfile_path:
            # Write a test grib message to the temporary file.
            with open(temp_gribfile_path, 'wb') as temp_gribfile:
                grib_message = gribapi.grib_new_from_samples('GRIB2')
                # Set the PDT to something unexpected.
                gribapi.grib_set_long(
                    grib_message, 'productDefinitionTemplateNumber', 5)
                gribapi.grib_write(grib_message, temp_gribfile)

            # Load the message from the file as a cube.
            cube_generator = iris.fileformats.grib.load_cubes(
                temp_gribfile_path)
            cube = cube_generator.next()

            # Check the cube has an extra "warning" attribute.
            self.assertEqual(
                cube.attributes['GRIB_LOAD_WARNING'],
                'unsupported GRIB2 ProductDefinitionTemplate: #4.5'
            )


class TestGribSimple(tests.IrisTest):
    # A testing class that does not need the test data.
    def mock_grib(self):
        # A mock grib message, with attributes that can't be Mocks themselves.
        grib = mock.Mock()
        grib.startStep = 0
        grib.phenomenon_points = lambda unit: 3
        grib._forecastTimeUnit = "hours"
        grib.productDefinitionTemplateNumber = 0
        # define a level type (NB these 2 are effectively the same)
        grib.levelType = 1
        grib.typeOfFirstFixedSurface = 1
        return grib

    def cube_from_message(self, grib):
        cube = iris.cube.Cube([[1.0]])
        # Parameter translation now uses the GribWrapper, so we must convert
        # the Mock-based fake message to a FakeGribMessage.
        with mock.patch('iris.fileformats.grib.gribapi', _mock_gribapi):
                grib_message = FakeGribMessage(**grib.__dict__)
                wrapped_msg = iris.fileformats.grib.GribWrapper(grib_message)
                iris.fileformats.grib.load_rules.convert(cube, wrapped_msg)
        return cube


class TestGrib1LoadPhenomenon(TestGribSimple):
    # Test recognition of grib phenomenon types.
    def mock_grib(self):
        grib = super(TestGrib1LoadPhenomenon, self).mock_grib()
        grib.edition = 1
        return grib

    def test_grib1_unknownparam(self):
        grib = self.mock_grib()
        grib.table2Version = 0
        grib.indicatorOfParameter = 9999
        cube = self.cube_from_message(grib)
        self.assertEqual(cube.standard_name, None)
        self.assertEqual(cube.long_name, None)
        self.assertEqual(cube.units, iris.unit.Unit("???"))

    def test_grib1_unknown_local_param(self):
        grib = self.mock_grib()
        grib.table2Version = 128
        grib.indicatorOfParameter = 999
        cube = self.cube_from_message(grib)
        self.assertEqual(cube.standard_name, None)
        self.assertEqual(cube.long_name, 'UNKNOWN LOCAL PARAM 999.128')
        self.assertEqual(cube.units, iris.unit.Unit("???"))

    def test_grib1_unknown_standard_param(self):
        grib = self.mock_grib()
        grib.table2Version = 1
        grib.indicatorOfParameter = 975
        cube = self.cube_from_message(grib)
        self.assertEqual(cube.standard_name, None)
        self.assertEqual(cube.long_name, 'UNKNOWN LOCAL PARAM 975.1')
        self.assertEqual(cube.units, iris.unit.Unit("???"))

    def known_grib1(self, param, standard_str, units_str):
        grib = self.mock_grib()
        grib.table2Version = 1
        grib.indicatorOfParameter = param
        cube = self.cube_from_message(grib)
        self.assertEqual(cube.standard_name, standard_str)
        self.assertEqual(cube.long_name, None)
        self.assertEqual(cube.units, iris.unit.Unit(units_str))

    def test_grib1_known_standard_params(self):
        # at present, there are just a very few of these
        self.known_grib1(11, 'air_temperature', 'kelvin')
        self.known_grib1(33, 'x_wind', 'm s-1')
        self.known_grib1(34, 'y_wind', 'm s-1')


class TestGrib2LoadPhenomenon(TestGribSimple):
    # Test recognition of grib phenomenon types.
    def mock_grib(self):
        grib = super(TestGrib2LoadPhenomenon, self).mock_grib()
        grib.edition = 2
        grib._forecastTimeUnit = 'hours'
        grib._forecastTime = 0.0
        grib.phenomenon_points = lambda unit: [0.0]
        return grib

    def known_grib2(self, discipline, category, param,
                    standard_name, long_name, units_str):
        grib = self.mock_grib()
        grib.discipline = discipline
        grib.parameterCategory = category
        grib.parameterNumber = param
        cube = self.cube_from_message(grib)
        try:
            iris_unit = iris.unit.Unit(units_str)
        except ValueError:
            iris_unit = iris.unit.Unit('???')
        self.assertEqual(cube.standard_name, standard_name)
        self.assertEqual(cube.long_name, long_name)
        self.assertEqual(cube.units, iris_unit)

    def test_grib2_unknownparam(self):
        grib = self.mock_grib()
        grib.discipline = 999
        grib.parameterCategory = 999
        grib.parameterNumber = 9999
        cube = self.cube_from_message(grib)
        self.assertEqual(cube.standard_name, None)
        self.assertEqual(cube.long_name, None)
        self.assertEqual(cube.units, iris.unit.Unit("???"))

    def test_grib2_known_standard_params(self):
        # check we know how to translate at least these params
        # I.E. all the ones the older scheme provided.
        full_set = [
            (0, 0, 0, "air_temperature", None, "kelvin"),
            (0, 0, 2, "air_potential_temperature", None, "K"),
            (0, 1, 0, "specific_humidity", None, "kg kg-1"),
            (0, 1, 1, "relative_humidity", None, "%"),
            (0, 1, 3, None, "precipitable_water", "kg m-2"),
            (0, 1, 22, None, "cloud_mixing_ratio", "kg kg-1"),
            (0, 1, 13, "liquid_water_content_of_surface_snow", None, "kg m-2"),
            (0, 2, 1, "wind_speed", None, "m s-1"),
            (0, 2, 2, "x_wind", None, "m s-1"),
            (0, 2, 3, "y_wind", None, "m s-1"),
            (0, 2, 8, "lagrangian_tendency_of_air_pressure", None, "Pa s-1"),
            (0, 2, 10, "atmosphere_absolute_vorticity", None, "s-1"),
            (0, 3, 0, "air_pressure", None, "Pa"),
            (0, 3, 1, "air_pressure_at_sea_level", None, "Pa"),
            (0, 3, 3, None, "icao_standard_atmosphere_reference_height", "m"),
            (0, 3, 5, "geopotential_height", None, "m"),
            (0, 3, 9, "geopotential_height_anomaly", None, "m"),
            (0, 6, 1, "cloud_area_fraction", None, "%"),
            (0, 6, 6, "atmosphere_mass_content_of_cloud_liquid_water", None,
                "kg m-2"),
            (0, 7, 6,
             "atmosphere_specific_convective_available_potential_energy",
             None, "J kg-1"),
            (0, 7, 7, None, "convective_inhibition", "J kg-1"),
            (0, 7, 8, None, "storm_relative_helicity", "J kg-1"),
            (0, 14, 0, "atmosphere_mole_content_of_ozone", None, "Dobson"),
            (2, 0, 0, "land_area_fraction", None, "1"),
            (10, 2, 0, "sea_ice_area_fraction", None, "1")]

        for (discipline, category, number,
             standard_name, long_name, units) in full_set:
            self.known_grib2(discipline, category, number,
                             standard_name, long_name, units)


if __name__ == "__main__":
    tests.main()
