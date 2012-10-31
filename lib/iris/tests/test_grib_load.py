# (C) British Crown Copyright 2010 - 2012, Met Office
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

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import os
import datetime

import gribapi

import matplotlib.pyplot as plt

import iris
import iris.fileformats.grib
import iris.plot as iplt
import iris.util
import iris.tests.stock


#
# helper code
#
class Fake_GribApi(object):
    """ Object to replace the gribapi interface with a few callable functions.

        Works with a values dictionary replacing a 'real' grib message.
        Implement only a few methods, just enough to allow GribWrapper creation.
    """

    class GribInternalError(Exception):
        """ Fake exception class.

            Detected by grib.py as key-access fail (must be an attribute of the 'gribapi module')
        """
        pass  
      
    @staticmethod
    def _fake_grib_get_value_from_name(grib_message, keyname):
        """ Fake fetching a key from a grib message.
        
            Works with a values dictionary replacing a 'real' grib message.
            Implement key-fetch as dictionary access.
            If absent, raise diagnostic exception.
        """
        if keyname in grib_message:
            return grib_message[keyname]
        raise Fake_GribApi.GribInternalError(keyname)

    # fakeup various get_XXX methods by converting to dictionary access on the (fake) message object
    grib_get_long = _fake_grib_get_value_from_name
    grib_get_string = _fake_grib_get_value_from_name
    grib_get_double = _fake_grib_get_value_from_name
    grib_get_double_array = _fake_grib_get_value_from_name

    @staticmethod
    def grib_is_missing(grib_message, keyname):
        """ Fake enquiring key existence.
        
            Convert to (fake) message object attribute existence.
        """ 
        return (keyname not in grib_message)
    
    @staticmethod
    def grib_get_native_type(grib_message, keyname):
        """ Fake gribapi type discovery operation. 

            If absent, raise diagnostic exception.
        """
        if keyname in grib_message:
            return type(grib_message[keyname])
        raise Fake_GribApi.GribInternalError(keyname)


@contextmanager
def Fakeup_Gribapi_Context(fake_gribapi_class=Fake_GribApi):
    """ Define a context within which iris.fileformats.grib.gribapi is replaced with a fake interface. """
    orig_gribapi_module = iris.fileformats.grib.gribapi
    iris.fileformats.grib.gribapi = fake_gribapi_class()
    try:
        yield None
    finally:
        # NOTE: the 'try/finally' is *necessary*, or it won't tidy up after an exception inside the context
        iris.fileformats.grib.gribapi = orig_gribapi_module


#
# main Testcase class
#        
@iris.tests.skip_data
class TestGribLoad(tests.GraphicsTest):
  
    def test_load(self):
                
        cubes = iris.load(tests.get_data_path(('GRIB', 'rotated_uk', "uk_wrongparam.grib1")))
        self.assertCML(cubes, ("grib_load", "rotated.cml"))
        
        cubes = iris.load(tests.get_data_path(('GRIB', "time_processed", "time_bound.grib1")))
        self.assertCML(cubes, ("grib_load", "time_bound_grib1.cml"))

        cubes = iris.load(tests.get_data_path(('GRIB', "time_processed", "time_bound.grib2")))
        self.assertCML(cubes, ("grib_load", "time_bound_grib2.cml"))
        
        cubes = iris.load(tests.get_data_path(('GRIB', "3_layer_viz", "3_layer.grib2")))
        cubes = iris.cube.CubeList([cubes[1], cubes[0], cubes[2]])
        self.assertCML(cubes, ("grib_load", "3_layer.cml"))
        
    def test_y_fastest(self):
        cubes = iris.load(tests.get_data_path(("GRIB", "y_fastest", "y_fast.grib2")))
        self.assertCML(cubes, ("grib_load", "y_fastest.cml"))
        iplt.contourf(cubes[0])
        plt.gca().coastlines()
        plt.title("y changes fastest")
        self.check_graphic()

    def test_ij_directions(self):
        
        def old_compat_load(name):
            cube = iris.load(tests.get_data_path(('GRIB', 'ij_directions', name)))[0]
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
            cube = iris.load(tests.get_data_path(('GRIB', 'shape_of_earth', name)))[0]
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
#        data_path = tests.get_data_path(('GRIB', 'global_t', 'global.grib2'))
#        cube = iris.load_cube(data_path)
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
        
    def test_fp_units(self):
        # Test different units for forecast period (just the ones we care about)

        # define basic 'fake message' contents, for mockup testing on the underlying methods
        #  - these contain just the minimum keys needed to create a GribWrapper
          
        # edition-1 test message data ...
        fake_message_ed1 = {
            'Ni': 1,
            'Nj': 1,
            'edition': 1,
            'alternativeRowScanning': 0,
            'startStep': 24,
            'centre': 'ecmf',
            'year': 2007,
            'month': 3,
            'day': 23,
            'hour': 12,
            'minute': 0,
            'timeRangeIndicator': 0,
            'P1':2, 'P2': 0,
            'unitOfTime': None,                   # NOTE: kludge, these two are the same thing !
            'indicatorOfUnitOfTimeRange': None,   # NOTE: kludge, these two are the same thing !
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
        }
        # edition-2 test message data ...
        fake_message_ed2 = {
            'Ni': 1,
            'Nj': 1,
            'edition': 2,
            'alternativeRowScanning': 0,
            'iDirectionIncrementGiven': 1,
            'jDirectionIncrementGiven': 1,
            'uvRelativeToGrid': 0,
            'forecastTime': 24,
            'centre': 'ecmf',
            'year': 2007,
            'month': 3,
            'day': 23,
            'hour': 12,
            'minute': 0,
            'productDefinitionTemplateNumber': 0,
            'stepRange':24,
            'shapeOfTheEarth':6,
            'gridType': 'rotated_ll',
            'angleOfRotation': 0.0,
            'unitOfTime': None,                   # NOTE: kludge, these two are the same thing !
            'indicatorOfUnitOfTimeRange': None,   # NOTE: kludge, these two are the same thing !
            'iDirectionIncrementInDegrees': 0.036,
            'jDirectionIncrementInDegrees': 0.036,
            'iScansNegatively': 0,
            'jScansPositively': 1,
            'longitudeOfFirstGridPointInDegrees': -5.70,
            'latitudeOfFirstGridPointInDegrees': -4.452,
            'jPointsAreConsecutive': 0,
            'values': np.array([[1.0]]),
        }

        # setup a list of test control values for each supported unit/edition testcase
        hour_secs = 3600.0
        test_set = (
            # edition, code, unit-equivalent-seconds, description-string
# edition-1
            (1, 0, 60.0, 'minutes'),
            (1, 1, hour_secs, 'hours'),
            (1, 2, 24.0*hour_secs, 'days'),
# ..these ones are possible but not yet supported ...
#            (1, 10, 3.0*hour_secs, '3 hours'),
#            (1, 11, 6.0*hour_secs, '6 hours'),
#            (1, 12, 12.0*hour_secs, '12 hours'),
#            (1, 13, 0.25*hour_secs, '15 minutes'),
#            (1, 14, 0.5*hour_secs, '30 minutes'),
#            (1, 254, 1.0, 'seconds'),

# edition 2
            (2, 0, 60.0, 'minutes'),
            (2, 1, hour_secs, 'hours'),
            (2, 2, 24.0*hour_secs, 'days'),
            (2, 13, 1.0, 'seconds'),
# ..these ones are possible but not yet supported ...
#            (2, 10, 3.0*hour_secs, '3 hours'),
#            (2, 11, 6.0*hour_secs, '6 hours'),
#            (2, 12, 12.0*hour_secs, '12 hours'),
        )

        # check unit-handling for each supported unit-code and grib-edition
        with Fakeup_Gribapi_Context():
            for (grib_edition, timeunit_codenum, timeunit_secs, timeunit_str) in test_set:
#                print 'checking: ed=',grib_edition, ' code=',timeunit_codenum, ' str=', timeunit_str
                # select grib-1 or grib-2 basic test message
                fake_message = [fake_message_ed1, fake_message_ed2][grib_edition-1]
                # set timeunit (NOTE slight kludge -- these 2 keys are aliases in the real gribapi)
                fake_message['indicatorOfUnitOfTimeRange'] = timeunit_codenum
                fake_message['unitOfTime'] = timeunit_codenum

                # make the GribWrapper object to test 
                wrapped_msg = iris.fileformats.grib.GribWrapper(fake_message)
                
                # check the units string
                forecast_timeunit = wrapped_msg._forecastTimeUnit
#                forecast_timeunit += '?'
                self.assertEqual(forecast_timeunit, timeunit_str, 
                    ('Bad unit string for edition=%01d, unitcode=%01d : expected="%s" GOT="%s"'
                      % (grib_edition, timeunit_codenum, timeunit_str, forecast_timeunit)
                    )
                )
                interval_start_to_end = wrapped_msg._phenomenonDateTime - wrapped_msg._referenceDateTime
                
                # check the data-starttime calculation
                if grib_edition == 1:
                    interval_time_units = wrapped_msg.P1*datetime.timedelta(0, timeunit_secs)
                else:
                    interval_time_units = wrapped_msg.forecastTime*datetime.timedelta(0, timeunit_secs)
#                interval_time_units += datetime.timedelta(0, 1)
                self.assertEqual(interval_start_to_end, interval_time_units,
                    ('Inconsistent start time offset for edition=%01d, unitcode=%01d : from-unit="%s" from-phenom-minus-ref="%s"'
                      % (grib_edition, timeunit_codenum, str(interval_time_units), str(interval_start_to_end))
                    )
                )


    def test_load_probability_forecast(self):
        # Test GribWrapper interpretation of PDT 4.9 data.
        # NOTE: 
        #   Currently Iris has only partial support for PDT 4.9.
        #   Though it can load the data, key metadata (thresholds) is lost.
        #   At present, we are not testing for this.

        # Make a testing grib message in memory, with gribapi.
        grib_message = gribapi.grib_new_from_samples('GRIB2')
        gribapi.grib_set_long(grib_message, 'productDefinitionTemplateNumber', 9)
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
                temp_gribfile_path )
            cube = cube_generator.next()

            # Check the cube has an extra "warning" attribute.
            self.assertEqual(
                cube.attributes['GRIB_LOAD_WARNING'],
                'unsupported GRIB2 ProductDefinitionTemplate: #4.5'
            )

        
if __name__ == "__main__":
    tests.main()
    print "finished"
