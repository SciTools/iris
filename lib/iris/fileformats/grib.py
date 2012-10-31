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
"""
Conversion of cubes to/from GRIB.

See also: `ECMWF GRIB API <http://www.ecmwf.int/publications/manuals/grib_api/index.html>`_.

"""

import datetime
import math  #for fmod
import os
import warnings

import numpy

import iris.proxy
iris.proxy.apply_proxy('gribapi', globals())

import iris.coords as coords
import iris.coord_systems as coord_systems
import iris.unit
import grib_save_rules


# rules for converting a grib message to a cm cube
_load_rules = None
_cross_reference_rules = None

CENTRE_TITLES = {'egrr': 'U.K. Met Office - Exeter',
                 'ecmf': 'European Centre for Medium Range Weather Forecasts',
                 'rjtd': 'Tokyo, Japan Meteorological Agency',
                 '55'  : 'San Francisco',
                 'kwbc': 'US National Weather Service, National Centres for Environmental Prediction'}

TIME_RANGE_INDICATORS = {0:'none', 1:'none', 3:'time mean', 4:'time sum',
                         5:'time _difference', 10:'none',
                         # TODO #567 Further exploration of the following mappings
                         51:'time mean', 113:'time mean', 114:'time sum',
                         115:'time mean', 116:'time sum', 117:'time mean',
                         118:'time _covariance', 123:'time mean',
                         124:'time sum', 125:'time standard_deviation'}

PROCESSING_TYPES = {0:'time mean', 1:'time sum', 2:'time maximum', 3:'time minimum',
                    4:'time _difference', 5:'time _root mean square',
                    6:'time standard_deviation', 7:'time _convariance',
                    8:'time _difference', 9:'time _ratio'}

unknown_string = "???"


def _ensure_load_rules_loaded():
    """Makes sure the standard conversion rules are loaded."""

    # Uses this module-level variables
    global _load_rules, _cross_reference_rules

    if _load_rules is None:
        rules_path = os.path.join(iris.config.CONFIG_PATH, 'grib_rules.txt')
        _load_rules = iris.fileformats.rules.RulesContainer(rules_path)

    if _cross_reference_rules is None:
        basepath = iris.config.CONFIG_PATH
        _cross_reference_rules = iris.fileformats.rules.RulesContainer(
            os.path.join(basepath, 'grib_cross_reference_rules.txt'),
            rule_type=iris.fileformats.rules.ObjectReturningRule)


def add_load_rules(filename):
    """
    Registers a rules file for use during the GRIB load process.

    Registered files are processed after the standard rules, and in the order
    they were registered.
    
    """
    _ensure_load_rules_loaded()
    _load_rules.import_rules(filename)


def reset_load_rules():
    """Resets the GRIB load process to use only the standard conversion rules."""
    
    # Uses this module-level variable
    global _load_rules
    
    _load_rules = None


class GribWrapper(object):
    """
    Contains a pygrib object plus some extra keys of our own.
    
    """
    def __init__(self, grib_message):
        """Store the grib message and compute our extra keys."""
        self.grib_message = grib_message
        
        # initialise the key-extension dictionary
        # NOTE: this attribute *must* exist, or the __getattr__ overload can hit an infinite loop
        self.extra_keys = {}
        
        self._confirm_in_scope()
        
        self._compute_extra_keys()

        #this is something pygrib did for us - reshape,
        #but it flipped the data - which we don't want
        ni = self.Ni
        nj = self.Nj
        self.data = self.values
        j_fast = gribapi.grib_get_long(grib_message, "jPointsAreConsecutive")
        if j_fast == 0:
            self.data = self.data.reshape(nj, ni)
        else:
            self.data = self.data.reshape(ni, nj)
            
        
    def _confirm_in_scope(self):
        """Ensure we have a grib flavour that we choose to support."""
        #forbid quasi-regular grids
        if (gribapi.grib_is_missing(self.grib_message, "Ni") or
            gribapi.grib_is_missing(self.grib_message, "Nj")):
            raise iris.exceptions.IrisError("Quasi-regular grids not yet handled.")

        #forbid alternate row scanning
        #(uncommon entry from GRIB2 flag table 3.4, also in GRIB1)
        if self.alternativeRowScanning == 1:
            raise iris.exceptions.IrisError("alternativeRowScanning == 1 not handled.")
        
        #grib2 specifics
        if self.edition == 2:
            #forbid uncommon entries from flag table 3.3
            if self.iDirectionIncrementGiven == 0 or self.jDirectionIncrementGiven == 0:
                raise iris.exceptions.IrisError("Quasi-regular grids not yet handled.")
            if self.uvRelativeToGrid == 1:
                raise iris.exceptions.IrisError("uvRelativeToGrid == 1 "
                                                "(in Flag table 3.3) not handled.")
            
        
    def __getattr__(self, key):
        """Return a grib key, or one of our extra keys."""
        
        # is it in the grib message?
        try:
            # we just get <type 'float'> as the type of the "values" array...special case here...
            if key in ["values", "pv"]:
                res = gribapi.grib_get_double_array(self.grib_message, key)
            elif key in ('typeOfFirstFixedSurface','typeOfSecondFixedSurface'):
                res = numpy.int32(gribapi.grib_get_long(self.grib_message, key))
            else:
                key_type = gribapi.grib_get_native_type(self.grib_message, key)
                if key_type == int:
                    res = numpy.int32(gribapi.grib_get_long(self.grib_message, key))
                elif key_type == float:
                    # Because some computer keys are floats, like
                    # longitudeOfFirstGridPointInDegrees, a float32 is not always enough...
                    res = numpy.float64(gribapi.grib_get_double(self.grib_message, key))
                elif key_type == str:
                    res = gribapi.grib_get_string(self.grib_message, key)
                else:
                    raise ValueError("Unknown type for %s : %s" % (key, str(key_type)))
        except gribapi.GribInternalError:
            res = None
        
        #...or is it in our list of extras?
        if res == None:
            if key in self.extra_keys:
                res = self.extra_keys[key]
            else:
                #must raise an exception for the hasattr() mechanism to work
                raise AttributeError("Cannot find GRIB key %s" % key)

        return res

    def _compute_extra_keys(self):
        """Compute our extra keys."""
        global unknown_string
        
        self.extra_keys = {}

        # work out stuff based on these values from the message
        edition = self.edition

        # time-processed forcast time is from reference time to start of period
        if edition == 2:
            forecastTime = self.forecastTime
            
            # Warn if we detect 'bad' (2's compliment) forecastTime.
            # Don't fix it, we're not doing that at present. It's not grib compliant.
            if abs(forecastTime) > 2**24:
                warnings.warn("Bad forecastTime detected! "
                              "Please contact the Iris team with the file you tried to load.")
        else:
            forecastTime = self.startStep

        #regular or rotated grid?
        try:
            longitudeOfSouthernPoleInDegrees = self.longitudeOfSouthernPoleInDegrees
            latitudeOfSouthernPoleInDegrees = self.latitudeOfSouthernPoleInDegrees
        except AttributeError:
            longitudeOfSouthernPoleInDegrees = 0.0
            latitudeOfSouthernPoleInDegrees = 90.0
        
        centre = gribapi.grib_get_string(self.grib_message, "centre")
        
        
        #default values
        self.extra_keys = {'_referenceDateTime':-1.0, '_phenomenonDateTime':-1.0, 
            '_periodStartDateTime':-1.0, '_periodEndDateTime':-1.0,
            '_levelTypeName':unknown_string,
            '_levelTypeUnits':unknown_string, '_firstLevelTypeName':unknown_string,
            '_firstLevelTypeUnits':unknown_string, '_firstLevel':-1.0,
            '_secondLevelTypeName':unknown_string, '_secondLevel':-1.0,
            '_originatingCentre':unknown_string, '_forecastTimeUnit':unknown_string,
            '_coord_system':None, '_x_circular':False,
            '_x_coord_name':unknown_string, '_y_coord_name':unknown_string,
            # These are here to avoid repetition in the rules files,
            # and reduce the very long line lengths.
            '_x_points':None, '_y_points':None}

        #reference date
        self.extra_keys['_referenceDateTime'] = \
            datetime.datetime(int(self.year), int(self.month), int(self.day),
                              int(self.hour), int(self.minute))
        
        #verification date
        processingDone = self._get_processing_done()
        #time processed?
        if processingDone.startswith("time"):
            if self.edition == 1:
                validityDate = str(self.validityDate)
                validityTime = "{:04}".format(int(self.validityTime))
                endYear   = int(validityDate[:4])
                endMonth  = int(validityDate[4:6])
                endDay    = int(validityDate[6:8])
                endHour   = int(validityTime[:2])
                endMinute = int(validityTime[2:4])
            elif self.edition == 2:
                endYear   = self.yearOfEndOfOverallTimeInterval
                endMonth  = self.monthOfEndOfOverallTimeInterval
                endDay    = self.dayOfEndOfOverallTimeInterval
                endHour   = self.hourOfEndOfOverallTimeInterval
                endMinute = self.minuteOfEndOfOverallTimeInterval

            # fixed forecastTime in hours
            self.extra_keys['_periodStartDateTime'] = \
                (self.extra_keys['_referenceDateTime'] + 
                 datetime.timedelta(0, 0, 0, 0, 0, int(forecastTime)))
            self.extra_keys['_periodEndDateTime'] = \
                datetime.datetime(endYear, endMonth, endDay, endHour, endMinute)
        else:
            self.extra_keys['_phenomenonDateTime'] = self._get_verification_date()

        
        #shape of the earth

        #pre-defined sphere
        if self.shapeOfTheEarth == 0:
            self.extra_keys['_coord_system'] = coord_systems.GeogCS(semi_major_axis=6367470)
            
        #custom sphere
        elif self.shapeOfTheEarth == 1:
            self.extra_keys['_coord_system'] = \
                coord_systems.GeogCS(self.scaledValueOfRadiusOfSphericalEarth * \
                                     self.scaleFactorOfRadiusOfSphericalEarth)
                    
        #IAU65 oblate sphere
        elif self.shapeOfTheEarth == 2:
            self.extra_keys['_coord_system'] = coord_systems.GeogCS(6378160, inverse_flattening=297.0)
                
        #custom oblate spheroid (km)
        elif self.shapeOfTheEarth == 3:
            self.extra_keys['_coord_system'] = coord_systems.GeogCS(
                semi_major_axis=self.scaledValueOfEarthMajorAxis * self.scaleFactorOfEarthMajorAxis * 1000.0,
                semi_minor_axis=self.scaledValueOfEarthMinorAxis * self.scaleFactorOfEarthMinorAxis * 1000.0)
            
        #IAG-GRS80 oblate spheroid
        elif self.shapeOfTheEarth == 4:
            self.extra_keys['_coord_system'] = coord_systems.GeogCS(6378137, None, 298.257222101)

        #WGS84
        elif self.shapeOfTheEarth == 5:
            self.extra_keys['_coord_system'] = \
                coord_systems.GeogCS(6378137, inverse_flattening=298.257223563)
        
        #pre-defined sphere
        elif self.shapeOfTheEarth == 6:
            self.extra_keys['_coord_system'] = coord_systems.GeogCS(6371229)
        
        #custom oblate spheroid (m)
        elif self.shapeOfTheEarth == 7:
            self.extra_keys['_coord_system'] = coord_systems.GeogCS(
                semi_major_axis=self.scaledValueOfEarthMajorAxis * self.scaleFactorOfEarthMajorAxis,
                semi_minor_axis=self.scaledValueOfEarthMinorAxis * self.scaleFactorOfEarthMinorAxis)
        
        elif self.shapeOfTheEarth == 8:
            raise ValueError("unhandled shape of earth : grib earth shape = 8")
        
        else:
            raise ValueError("undefined shape of earth")


        #rotated pole
        gridType = gribapi.grib_get_string(self.grib_message, "gridType")
        if gridType == 'rotated_ll':
            # Replace the llcs with a rotated one
            southPoleLon = longitudeOfSouthernPoleInDegrees
            southPoleLat = latitudeOfSouthernPoleInDegrees
            # TODO: Confirm the translation from angleOfRotation to north_pole_lon (usually 0 for both)
            self.extra_keys['_coord_system'] = \
                iris.coord_systems.RotatedGeogCS(-southPoleLat, math.fmod(southPoleLon + 180.0, 360.0),
                                                 self.angleOfRotation, self.extra_keys['_coord_system']) 

        
        #originating centre
        #TODO #574 Expand to include sub-centre
        self.extra_keys['_originatingCentre'] = CENTRE_TITLES.get(centre, "unknown centre %s" % centre)
            
        #forecast time unit as a cm string
        #TODO #575 Do we want PP or GRIB style forecast delta?
        if self.indicatorOfUnitOfTimeRange == 0:
            self.extra_keys['_forecastTimeUnit'] = 'minutes'
        elif self.indicatorOfUnitOfTimeRange == 1:
            self.extra_keys['_forecastTimeUnit'] = 'hours'
        elif self.indicatorOfUnitOfTimeRange == 2:
            self.extra_keys['_forecastTimeUnit'] = 'days'
#        elif self.indicatorOfUnitOfTimeRange == 3:
#            self.extra_keys['_forecastTimeUnit'] = 'months'
#        elif self.indicatorOfUnitOfTimeRange == 4:
#            self.extra_keys['_forecastTimeUnit'] = 'years'
#        elif self.indicatorOfUnitOfTimeRange == 5:
#            self.extra_keys['_forecastTimeUnit'] = 'decades'
#        elif self.indicatorOfUnitOfTimeRange == 6:
#            self.extra_keys['_forecastTimeUnit'] = '30 years'
#        elif self.indicatorOfUnitOfTimeRange == 7:
#            self.extra_keys['_forecastTimeUnit'] = 'century'
        elif self.indicatorOfUnitOfTimeRange == 10:
            self.extra_keys['_forecastTimeUnit'] = '3 hours'
        elif self.indicatorOfUnitOfTimeRange == 11:
            self.extra_keys['_forecastTimeUnit'] = '6 hours'
        elif self.indicatorOfUnitOfTimeRange == 12:
            self.extra_keys['_forecastTimeUnit'] = '12 hours'
        elif self.indicatorOfUnitOfTimeRange == 13 and edition==2:
            self.extra_keys['_forecastTimeUnit'] = 'seconds'
        else:
            message = "Only hours are currently handled for forecast indicatorOfUnitOfTimeRange"
            raise iris.exceptions.NotYetImplementedError(message)
        
        if self.gridType=="regular_ll":
            self.extra_keys['_x_coord_name'] = "longitude"
            self.extra_keys['_y_coord_name'] = "latitude"
        else:
            self.extra_keys['_x_coord_name'] = "grid_longitude"
            self.extra_keys['_y_coord_name'] = "grid_latitude"

        i_step = self.iDirectionIncrementInDegrees
        j_step = self.jDirectionIncrementInDegrees
        if self.iScansNegatively:
            i_step = -i_step
        if not self.jScansPositively:
            j_step = -j_step
        self._x_points = (numpy.arange(self.Ni, dtype=numpy.float64) * i_step +
                          self.longitudeOfFirstGridPointInDegrees)
        self._y_points = (numpy.arange(self.Nj, dtype=numpy.float64) * j_step +
                          self.latitudeOfFirstGridPointInDegrees)

        # circular x coord?
        if "longitude" in self.extra_keys['_x_coord_name'] and self.Ni > 1:
            # Is the gap from end to start smaller or about equal to the max step?
            points = self._x_points
            gap = 360.0 - abs(points[-1] - points[0]) 
            max_step = abs(numpy.diff(points)).max()
            if gap <= max_step:
                self.extra_keys['_x_circular'] = True
            else:
                try:
                    numpy.testing.assert_almost_equal(gap / max_step, 1.0, decimal=3)
                    self.extra_keys['_x_circular'] = True
                except:
                    pass
        
    def _get_processing_done(self):
        """Determine the type of processing that was done on the data."""
        
        processingDone = 'unknown'
        edition = self.edition
        
        #grib1
        if edition == 1:
            timeRangeIndicator = self.timeRangeIndicator
            processingDone = TIME_RANGE_INDICATORS.get(timeRangeIndicator,
                                'time _grib1_process_unknown_%i' % timeRangeIndicator)
        
        #grib2
        else:
        
            pdt = self.productDefinitionTemplateNumber
            
            #pdt 4.0? (standard forecast)
            if pdt == 0:
                processingDone = 'none'
                
            #pdt 4.8 or 4.9? (time-processed)
            elif pdt in (8, 9):
                typeOfStatisticalProcessing = self.typeOfStatisticalProcessing
                processingDone = PROCESSING_TYPES.get(typeOfStatisticalProcessing,
                                    'time _grib2_process_unknown_%i' % typeOfStatisticalProcessing)
                        
        return processingDone
    
    def _get_verification_date(self):
        reference_date_time = self._referenceDateTime
        
        if self.edition == 1:
    
            time_range_indicator = self.timeRangeIndicator
            P1 = self.P1
            P2 = self.P2
            unit_of_time = self.unitOfTime
    
            if time_range_indicator == 0:    time_diff = P1       #Forecast product valid at reference time + P1 P1>0), or Uninitialized analysis product for reference time (P1=0). Or Image product for reference time (P1=0)
            elif time_range_indicator == 1:    time_diff = P1     #Initialized analysis product for reference time (P1=0).
            elif time_range_indicator == 2:    time_diff = (P1 + P2) * 0.5    #Product with a valid time ranging between reference time + P1 and reference time + P2
            elif time_range_indicator == 3:    time_diff = (P1 + P2) * 0.5    #Average(reference time + P1 to reference time + P2)
            elif time_range_indicator == 4:    time_diff = P2     #Accumulation (reference time + P1 to reference time + P2) product considered valid at reference time + P2
            elif time_range_indicator == 5:    time_diff = P2     #Difference(reference time + P2 minus reference time + P1) product considered valid at reference time + P2
            elif time_range_indicator == 10:   time_diff = P1 * 256 + P2    #P1 occupies octets 19 and 20; product valid at reference time + P1
            elif time_range_indicator == 51:                      #Climatological Mean Value: multiple year averages of quantities which are themselves means over some period of time (P2) less than a year. The reference time (R) indicates the date and time of the start of a period of time, given by R to R + P2, over which a mean is formed; N indicates the number of such period-means that are averaged together to form the climatological value, assuming that the N period-mean fields are separated by one year. The reference time indicates the start of the N-year climatology. N is given in octets 22-23 of the PDS. If P1 = 0 then the data averaged in the basic interval P2 are assumed to be continuous, i.e., all available data are simply averaged together. If P1 = 1 (the units of time - octet 18, code table 4 - are not relevant here) then the data averaged together in the basic interval P2 are valid only at the time (hour, minute) given in the reference time, for all the days included in the P2 period. The units of P2 are given by the contents of octet 18 and Table 4.
                raise iris.exceptions.TranslationError("unhandled grib1 timeRangeIndicator = 51 (avg of avgs)")
            elif time_range_indicator == 113:    time_diff = P1    #Average of N forecasts (or initialized analyses); each product has forecast period of P1 (P1=0 for initialized analyses); products have reference times at intervals of P2, beginning at the given reference time.
            elif time_range_indicator == 114:    time_diff = P1    #Accumulation of N forecasts (or initialized analyses); each product has forecast period of P1 (P1=0 for initialized analyses); products have reference times at intervals of P2, beginning at the given reference time.
            elif time_range_indicator == 115:    time_diff = P1    #Average of N forecasts, all with the same reference time; the first has a forecast period of P1, the remaining forecasts follow at intervals of P2.
            elif time_range_indicator == 116:    time_diff = P1    #Accumulation of N forecasts, all with the same reference time; the first has a forecast period of P1, the remaining follow at intervals of P2.
            elif time_range_indicator == 117:    time_diff = P1    #Average of N forecasts, the first has a period of P1, the subsequent ones have forecast periods reduced from the previous one by an interval of P2; the reference time for the first is given in octets 13-17, the subsequent ones have reference times increased from the previous one by an interval of P2. Thus all the forecasts have the same valid time, given by the initial reference time + P1.
            elif time_range_indicator == 118:    time_diff = P1    #Temporal variance, or covariance, of N initialized analyses; each product has forecast period P1=0; products have reference times at intervals of P2, beginning at the given reference time.
            elif time_range_indicator == 123:    time_diff = P1    #Average of N uninitialized analyses, starting at the reference time, at intervals of P2.
            elif time_range_indicator == 124:    time_diff = P1    #Accumulation of N uninitialized analyses, starting at the reference time, at intervals of P2.
            else:
                raise iris.exceptions.TranslationError("unhandled grib1 timeRangeIndicator = %i" %
                                                       time_range_indicator)
    
            if unit_of_time == 0:  # minutes
                verification_date = (reference_date_time + 
                    datetime.timedelta(0, 0, 0, 0, int(time_diff)))  
            elif unit_of_time == 1:  # hours
                verification_date = (reference_date_time + 
                    datetime.timedelta(0, 0, 0, 0, 0, int(time_diff)))
            elif unit_of_time == 2:  # days
                verification_date = (reference_date_time +
                    datetime.timedelta(int(time_diff)))
            elif unit_of_time == 10:  # 3 hours
                verification_date = (reference_date_time +
                    datetime.timedelta(0, 0, 0, 0, 0, int(time_diff*3)))
            else:
                raise iris.exceptions.TranslationError("Unhandled grib1 unitOfTime = %i" %
                                                       unit_of_time)
        
        if self.edition == 2:
        
            time_diff = int(self.stepRange) # gribapi gives us a string!
            forecast_time_unit = self.indicatorOfUnitOfTimeRange  # P1 and P2 units
            
            if forecast_time_unit == 0:  # minutes
                verification_date = (reference_date_time + 
                    datetime.timedelta(0, 0, 0, 0, int(time_diff)))
            elif forecast_time_unit == 1:  # hours
                verification_date = (reference_date_time +
                    datetime.timedelta(0, 0, 0, 0, 0, int(time_diff)))
            elif forecast_time_unit == 2:  # days
                verification_date = (reference_date_time +
                    datetime.timedelta(int(time_diff)))
            elif forecast_time_unit == 13:  # seconds
                verification_date = (reference_date_time +
                    datetime.timedelta(0, int(time_diff)))
            else:
                raise iris.exceptions.TranslationError("Unhandled grib2 unitOfTime = %i" %
                                                       forecast_time_unit)
    
        return verification_date
    
    def phenomenon_points(self, time_unit):
        """
        Return the phenomenon time point offset from the epoch time reference 
        measured in the appropriate time units.
       
        """

        time_reference = '%s since epoch' % time_unit
        return iris.unit.date2num(self._phenomenonDateTime, time_reference,
                                  iris.unit.CALENDAR_GREGORIAN)
    
    def phenomenon_bounds(self, time_unit):
        """
        Return the phenomenon time bound offsets from the epoch time reference 
        measured in the appropriate time units.
    
        """
        # TODO #576 Investigate when it's valid to get phenomenon_bounds
        time_reference = '%s since epoch' % time_unit
        unit = iris.unit.Unit(time_reference, iris.unit.CALENDAR_GREGORIAN)
        return [unit.date2num(self._periodStartDateTime),
                unit.date2num(self._periodEndDateTime)]


def grib_generator(filename):
    """Returns a generator of GribWrapper fields from the given filename."""
    with open(filename, 'rb') as grib_file:
        while True:
            grib_message = gribapi.grib_new_from_file(grib_file)
            if grib_message is None:
                break

            grib_wrapper = GribWrapper(grib_message)

            yield grib_wrapper

            # finished with the grib message - claimed by the ecmwf c library.
            gribapi.grib_release(grib_message)


def load_cubes(filenames, callback=None):
    """Returns a generator of cubes from the given list of filenames."""
    _ensure_load_rules_loaded()
    rules = iris.fileformats.rules
    grib_loader = rules.Loader(grib_generator, _load_rules,
                               _cross_reference_rules, 'GRIB_LOAD')
    return rules.load_cubes(filenames, callback, grib_loader)


def save_grib2(cube, target, append=False, **kwargs):
    """
    Save a cube to a GRIB2 file.
    
    Args:

        * cube      - A :class:`iris.cube.Cube`, :class:`iris.cube.CubeList` or list of cubes.
        * target    - A filename or open file handle.

    Kwargs:
    
        * append    - Whether to start a new file afresh or add the cube(s) to the end of the file. 
                      Only applicable when target is a filename, not a file handle.
                      Default is False.

    See also :func:`iris.io.save`.
    
    """

    # grib file (this bit is common to the pp and grib savers...)
    if isinstance(target, basestring):
        grib_file = open(target, "ab" if append else "wb")
    elif hasattr(target, "write"):
        if hasattr(target, "mode") and "b" not in target.mode:
            raise ValueError("Target not binary")
        grib_file = target
    else:
        raise ValueError("Can only save grib to filename or writable")

    # discover the lat and lon coords (this bit is common to the pp and grib savers...)
    lat_coords = filter(lambda coord: "latitude" in coord.name(), cube.coords())
    lon_coords = filter(lambda coord: "longitude" in coord.name(), cube.coords())
    if len(lat_coords) != 1 or len(lon_coords) != 1:
        raise iris.exceptions.TranslationError("Did not find one (and only one) "
                                               "latitude or longitude coord")

    # Save each latlon slice2D in the cube 
    for slice2D in cube.slices([lat_coords[0], lon_coords[0]]):

        # Save this slice to the grib file
        grib_message = gribapi.grib_new_from_samples("GRIB2")
        grib_save_rules.run(slice2D, grib_message)
        gribapi.grib_write(grib_message, grib_file)
        gribapi.grib_release(grib_message)

    # (this bit is common to the pp and grib savers...)
    if isinstance(target, basestring):
        grib_file.close()



