# (C) British Crown Copyright 2010 - 2014, Met Office
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

import biggus
import cartopy
import numpy as np
import numpy.ma as ma
import scipy.interpolate

import iris.proxy
iris.proxy.apply_proxy('gribapi', globals())

from iris.analysis.interpolate import Linear1dExtrapolator
import iris.coord_systems as coord_systems
from iris.exceptions import TranslationError
# NOTE: careful here, to avoid circular imports (as iris imports grib)
from iris.fileformats.grib import grib_phenom_translation as gptx
from iris.fileformats.grib import grib_save_rules
import iris.fileformats.grib.load_rules
import iris.unit


__all__ = ['grib_generator', 'load_cubes',
           'reset_load_rules', 'save_grib2', 'GribWrapper',
           'hindcast_workaround']


#: Set this flag to True to enable support of negative forecast periods
#: when loading and saving GRIB files.
hindcast_workaround = False


# rules for converting a grib message to a cm cube
_load_rules = None


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

TIME_CODES_EDITION1 = {
    0: ('minutes', 60),
    1: ('hours', 60*60),
    2: ('days', 24*60*60),
    # NOTE: do *not* support calendar-dependent units at all.
    # So the following possible keys remain unsupported:
    #  3: 'months',
    #  4: 'years',
    #  5: 'decades',
    #  6: '30 years',
    #  7: 'century',
    10: ('3 hours', 3*60*60),
    11: ('6 hours', 6*60*60),
    12: ('12 hours', 12*60*60),
    13: ('15 minutes', 15*60),
    14: ('30 minutes', 30*60),
    254: ('seconds', 1),
}

TIME_CODES_EDITION2 = {
    0: ('minutes', 60),
    1: ('hours', 60*60),
    2: ('days', 24*60*60),
    # NOTE: do *not* support calendar-dependent units at all.
    # So the following possible keys remain unsupported:
    #  3: 'months',
    #  4: 'years',
    #  5: 'decades',
    #  6: '30 years',
    #  7: 'century',
    10: ('3 hours', 3*60*60),
    11: ('6 hours', 6*60*60),
    12: ('12 hours', 12*60*60),
    13: ('seconds', 1),
}

unknown_string = "???"


def reset_load_rules():
    """
    Resets the GRIB load process to use only the standard conversion rules.

    .. deprecated:: 1.7

    """
    # Uses this module-level variable
    global _load_rules

    warnings.warn('reset_load_rules was deprecated in v1.7.')

    _load_rules = None


class GribDataProxy(object):
    """A reference to the data payload of a single Grib message."""

    __slots__ = ('shape', 'dtype', 'fill_value', 'path', 'offset', 'regularise')

    def __init__(self, shape, dtype, fill_value, path, offset, regularise):
        self.shape = shape
        self.dtype = dtype
        self.fill_value = fill_value
        self.path = path
        self.offset = offset
        self.regularise = regularise

    @property
    def ndim(self):
        return len(self.shape)

    def __getitem__(self, keys):
        with open(self.path, 'rb') as grib_fh:
            grib_fh.seek(self.offset)
            grib_message = gribapi.grib_new_from_file(grib_fh)

            if self.regularise and _is_quasi_regular_grib(grib_message):
                _regularise(grib_message)

            data = _message_values(grib_message, self.shape)
            gribapi.grib_release(grib_message)

        return data.__getitem__(keys)

    def __repr__(self):
        msg = '<{self.__class__.__name__} shape={self.shape} ' \
          'dtype={self.dtype!r} fill_value={self.fill_value!r} ' \
          'path={self.path!r} offset={self.offset} ' \
          'regularise={self.regularise}>'
        return msg.format(self=self)

    def __getstate__(self):
        return {attr:getattr(self, attr) for attr in self.__slots__}

    def __setstate__(self, state):
        for key, value in state.iteritems():
            setattr(self, key, value)


class GribWrapper(object):
    """
    Contains a pygrib object plus some extra keys of our own.

    """
    def __init__(self, grib_message, grib_fh=None, auto_regularise=True):
        """Store the grib message and compute our extra keys."""
        self.grib_message = grib_message
        deferred = grib_fh is not None

        # Store the file pointer and message length from the current
        # grib message before it's changed by calls to the grib-api.
        if deferred:
            # Note that, the grib-api has already read this message and 
            # advanced the file pointer to the end of the message.
            offset = grib_fh.tell()
            message_length = gribapi.grib_get_long(grib_message, 'totalLength')

        if auto_regularise and _is_quasi_regular_grib(grib_message):
            warnings.warn('Regularising GRIB message.')
            if deferred:
                self._regularise_shape(grib_message)
            else:
                _regularise(grib_message)

        # Initialise the key-extension dictionary.
        # NOTE: this attribute *must* exist, or the the __getattr__ overload
        # can hit an infinite loop.
        self.extra_keys = {}
        self._confirm_in_scope()
        self._compute_extra_keys()

        # Calculate the data payload shape.
        shape = (gribapi.grib_get_long(grib_message, 'numberOfValues'),)

        if not self.gridType.startswith('reduced'):
            ni, nj = self.Ni, self.Nj
            j_fast = gribapi.grib_get_long(grib_message,
                                           'jPointsAreConsecutive')
            shape = (nj, ni) if j_fast == 0 else (ni, nj)

        if deferred:
            # Wrap the reference to the data payload within the data proxy
            # in order to support deferred data loading.
            # The byte offset requires to be reset back to the first byte
            # of this message. The file pointer offset is always at the end 
            # of the current message due to the grib-api reading the message.
            proxy = GribDataProxy(shape, np.zeros(.0).dtype, np.nan,
                                  grib_fh.name,
                                  offset - message_length,
                                  auto_regularise)
            self._data = biggus.NumpyArrayAdapter(proxy)
        else:
            self.data = _message_values(grib_message, shape)

    @staticmethod
    def _regularise_shape(grib_message):
        """
        Calculate the regularised shape of the reduced message and push
        dummy regularised values into the message to force the gribapi
        to update the message grid type from reduced to regular.

        """
        # Make sure to read any missing values as NaN.
        gribapi.grib_set_double(grib_message, "missingValue", np.nan)

        # Get full longitude values, these describe the longitude value of
        # *every* point in the grid, they are not 1d monotonic coordinates.
        lons = gribapi.grib_get_double_array(grib_message, "longitudes")

        # Compute the new longitude coordinate for the regular grid.
        new_nx = max(gribapi.grib_get_long_array(grib_message, "pl"))
        new_x_step = (max(lons) - min(lons)) / (new_nx - 1)
        if gribapi.grib_get_long(grib_message, "iScansNegatively"):
            new_x_step *= -1

        gribapi.grib_set_long(grib_message, "Nx", int(new_nx))
        gribapi.grib_set_double(grib_message, "iDirectionIncrementInDegrees",
                                float(new_x_step))
        # Spoof gribapi with false regularised values.
        nj = gribapi.grib_get_long(grib_message, 'Nj')
        temp = np.zeros((nj * new_nx,), dtype=np.float)
        gribapi.grib_set_double_array(grib_message, 'values', temp)
        gribapi.grib_set_long(grib_message, "jPointsAreConsecutive", 0)
        gribapi.grib_set_long(grib_message, "PLPresent", 0)

    def _confirm_in_scope(self):
        """Ensure we have a grib flavour that we choose to support."""

        #forbid alternate row scanning
        #(uncommon entry from GRIB2 flag table 3.4, also in GRIB1)
        if self.alternativeRowScanning == 1:
            raise iris.exceptions.IrisError("alternativeRowScanning == 1 not handled.")

    def __getattr__(self, key):
        """Return a grib key, or one of our extra keys."""

        # is it in the grib message?
        try:
            # we just get <type 'float'> as the type of the "values" array...special case here...
            if key in ["values", "pv", "latitudes", "longitudes"]:
                res = gribapi.grib_get_double_array(self.grib_message, key)
            elif key in ('typeOfFirstFixedSurface','typeOfSecondFixedSurface'):
                res = np.int32(gribapi.grib_get_long(self.grib_message, key))
            else:
                key_type = gribapi.grib_get_native_type(self.grib_message, key)
                if key_type == int:
                    res = np.int32(gribapi.grib_get_long(self.grib_message, key))
                elif key_type == float:
                    # Because some computer keys are floats, like
                    # longitudeOfFirstGridPointInDegrees, a float32 is not always enough...
                    res = np.float64(gribapi.grib_get_double(self.grib_message, key))
                elif key_type == str:
                    res = gribapi.grib_get_string(self.grib_message, key)
                else:
                    raise ValueError("Unknown type for %s : %s" % (key, str(key_type)))
        except gribapi.GribInternalError:
            res = None

        #...or is it in our list of extras?
        if res is None:
            if key in self.extra_keys:
                res = self.extra_keys[key]
            else:
                #must raise an exception for the hasattr() mechanism to work
                raise AttributeError("Cannot find GRIB key %s" % key)

        return res

    def _timeunit_detail(self):
        """Return the (string, seconds) describing the message time unit."""
        if self.edition == 1:
            code_to_detail = TIME_CODES_EDITION1
        else:
            code_to_detail = TIME_CODES_EDITION2
        unit_code = self.indicatorOfUnitOfTimeRange
        if unit_code not in code_to_detail:
            message = 'Unhandled time unit for forecast ' \
                      'indicatorOfUnitOfTimeRange : ' + str(unit_code)
            raise iris.exceptions.NotYetImplementedError(message)
        return code_to_detail[unit_code]

    def _timeunit_string(self):
        """Get the udunits string for the message time unit."""
        return self._timeunit_detail()[0]

    def _timeunit_seconds(self):
        """Get the number of seconds in the message time unit."""
        return self._timeunit_detail()[1]

    def _compute_extra_keys(self):
        """Compute our extra keys."""
        global unknown_string

        self.extra_keys = {}

        # work out stuff based on these values from the message
        edition = self.edition

        # time-processed forcast time is from reference time to start of period
        if edition == 2:
            forecastTime = self.forecastTime

            uft = np.uint32(forecastTime)
            BILL = 2**30

            # Workaround grib api's assumption that forecast time is positive.
            # Handles correctly encoded -ve forecast times up to one -1 billion.
            if hindcast_workaround:
                if 2 * BILL < uft < 3 * BILL:
                    msg = "Re-interpreting negative forecastTime from " \
                            + str(forecastTime)
                    forecastTime = -(uft - 2 * BILL)
                    msg += " to " + str(forecastTime)
                    warnings.warn(msg)

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
            '_originatingCentre':unknown_string,
            '_forecastTime':None, '_forecastTimeUnit':unknown_string,
            '_coord_system':None, '_x_circular':False,
            '_x_coord_name':unknown_string, '_y_coord_name':unknown_string,
            # These are here to avoid repetition in the rules files,
            # and reduce the very long line lengths.
            '_x_points':None, '_y_points':None,
            '_cf_data':None}

        # cf phenomenon translation
        if edition == 1:
            # Get centre code (N.B. self.centre has default type = string)
            centre_number = gribapi.grib_get_long(self.grib_message, "centre")
            # Look for a known grib1-to-cf translation (or None).
            cf_data = gptx.grib1_phenom_to_cf_info(
                table2_version=self.table2Version,
                centre_number=centre_number,
                param_number=self.indicatorOfParameter)
            self.extra_keys['_cf_data'] = cf_data
        elif edition == 2:
            # Don't attempt to interpret params if 'master tables version' is
            # 255, as local params may then have same codes as standard ones.
            if self.tablesVersion != 255:
                # Look for a known grib2-to-cf translation (or None).
                cf_data = gptx.grib2_phenom_to_cf_info(
                    param_discipline=self.discipline,
                    param_category=self.parameterCategory,
                    param_number=self.parameterNumber)
                self.extra_keys['_cf_data'] = cf_data

        #reference date
        self.extra_keys['_referenceDateTime'] = \
            datetime.datetime(int(self.year), int(self.month), int(self.day),
                              int(self.hour), int(self.minute))

        # forecast time with workarounds
        self.extra_keys['_forecastTime'] = forecastTime

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
                 datetime.timedelta(hours=int(forecastTime)))
            self.extra_keys['_periodEndDateTime'] = \
                datetime.datetime(endYear, endMonth, endDay, endHour, endMinute)
        else:
            self.extra_keys['_phenomenonDateTime'] = self._get_verification_date()


        #originating centre
        #TODO #574 Expand to include sub-centre
        self.extra_keys['_originatingCentre'] = CENTRE_TITLES.get(
                                        centre, "unknown centre %s" % centre)

        #forecast time unit as a cm string
        #TODO #575 Do we want PP or GRIB style forecast delta?
        self.extra_keys['_forecastTimeUnit'] = self._timeunit_string()


        #shape of the earth

        #pre-defined sphere
        if self.shapeOfTheEarth == 0:
            geoid = coord_systems.GeogCS(semi_major_axis=6367470)

        #custom sphere
        elif self.shapeOfTheEarth == 1:
            geoid = coord_systems.GeogCS(
                self.scaledValueOfRadiusOfSphericalEarth *
                10 ** -self.scaleFactorOfRadiusOfSphericalEarth)

        #IAU65 oblate sphere
        elif self.shapeOfTheEarth == 2:
            geoid = coord_systems.GeogCS(6378160, inverse_flattening=297.0)

        #custom oblate spheroid (km)
        elif self.shapeOfTheEarth == 3:
            geoid = coord_systems.GeogCS(
                semi_major_axis=self.scaledValueOfEarthMajorAxis *
                10 ** -self.scaleFactorOfEarthMajorAxis * 1000.,
                semi_minor_axis=self.scaledValueOfEarthMinorAxis *
                10 ** -self.scaleFactorOfEarthMinorAxis * 1000.)

        #IAG-GRS80 oblate spheroid
        elif self.shapeOfTheEarth == 4:
            geoid = coord_systems.GeogCS(6378137, None, 298.257222101)

        #WGS84
        elif self.shapeOfTheEarth == 5:
            geoid = \
                coord_systems.GeogCS(6378137, inverse_flattening=298.257223563)

        #pre-defined sphere
        elif self.shapeOfTheEarth == 6:
            geoid = coord_systems.GeogCS(6371229)

        #custom oblate spheroid (m)
        elif self.shapeOfTheEarth == 7:
            geoid = coord_systems.GeogCS(
                semi_major_axis=self.scaledValueOfEarthMajorAxis *
                10 ** -self.scaleFactorOfEarthMajorAxis,
                semi_minor_axis=self.scaledValueOfEarthMinorAxis *
                10 ** -self.scaleFactorOfEarthMinorAxis)

        elif self.shapeOfTheEarth == 8:
            raise ValueError("unhandled shape of earth : grib earth shape = 8")

        else:
            raise ValueError("undefined shape of earth")

        gridType = gribapi.grib_get_string(self.grib_message, "gridType")

        if gridType in ["regular_ll", "regular_gg", "reduced_ll", "reduced_gg"]:
            self.extra_keys['_x_coord_name'] = "longitude"
            self.extra_keys['_y_coord_name'] = "latitude"
            self.extra_keys['_coord_system'] = geoid
        elif gridType == 'rotated_ll':
            # TODO: Confirm the translation from angleOfRotation to
            # north_pole_lon (usually 0 for both)
            self.extra_keys['_x_coord_name'] = "grid_longitude"
            self.extra_keys['_y_coord_name'] = "grid_latitude"
            southPoleLon = longitudeOfSouthernPoleInDegrees
            southPoleLat = latitudeOfSouthernPoleInDegrees
            self.extra_keys['_coord_system'] = \
                iris.coord_systems.RotatedGeogCS(
                                        -southPoleLat,
                                        math.fmod(southPoleLon + 180.0, 360.0),
                                        self.angleOfRotation, geoid)
        elif gridType == 'polar_stereographic':
            self.extra_keys['_x_coord_name'] = "projection_x_coordinate"
            self.extra_keys['_y_coord_name'] = "projection_y_coordinate"

            if self.projectionCentreFlag == 0:
                pole_lat = 90
            elif self.projectionCentreFlag == 1:
                pole_lat = -90
            else:
                raise TranslationError("Unhandled projectionCentreFlag")

            # Note: I think the grib api defaults LaDInDegrees to 60 for grib1.
            self.extra_keys['_coord_system'] = \
                iris.coord_systems.Stereographic(
                    pole_lat, self.orientationOfTheGridInDegrees, 0, 0,
                    self.LaDInDegrees, ellipsoid=geoid)

        elif gridType == 'lambert':
            self.extra_keys['_x_coord_name'] = "projection_x_coordinate"
            self.extra_keys['_y_coord_name'] = "projection_y_coordinate"

            if self.edition == 1:
                flag_name = "projectionCenterFlag"
            else:
                flag_name = "projectionCentreFlag"

            if getattr(self, flag_name) == 0:
                pole_lat = 90
            elif getattr(self, flag_name) == 1:
                pole_lat = -90
            else:
                raise TranslationError("Unhandled projectionCentreFlag")

            LambertConformal = iris.coord_systems.LambertConformal
            self.extra_keys['_coord_system'] = LambertConformal(
                self.LaDInDegrees, self.LoVInDegrees, 0, 0,
                secant_latitudes=(self.Latin1InDegrees, self.Latin2InDegrees),
                ellipsoid=geoid)
        else:
            raise TranslationError("unhandled grid type: {}".format(gridType))

        if gridType in ["regular_ll", "rotated_ll"]:
            self._regular_longitude_common()
            j_step = self.jDirectionIncrementInDegrees
            if not self.jScansPositively:
                j_step = -j_step
            self._y_points = (np.arange(self.Nj, dtype=np.float64) * j_step +
                              self.latitudeOfFirstGridPointInDegrees)

        elif gridType in ['regular_gg']:
            # longitude coordinate is straight-forward
            self._regular_longitude_common()
            # get the distinct latitudes, and make sure they are sorted
            # (south-to-north) and then put them in the right direction
            # depending on the scan direction
            latitude_points = gribapi.grib_get_double_array(
                self.grib_message, 'distinctLatitudes').astype(np.float64)
            latitude_points.sort()
            if not self.jScansPositively:
                # we require latitudes north-to-south
                self._y_points = latitude_points[::-1]
            else:
                self._y_points = latitude_points

        elif gridType in ["polar_stereographic", "lambert"]:
            # convert the starting latlon into meters
            cartopy_crs = self.extra_keys['_coord_system'].as_cartopy_crs()
            x1, y1 = cartopy_crs.transform_point(
                                self.longitudeOfFirstGridPointInDegrees,
                                self.latitudeOfFirstGridPointInDegrees,
                                cartopy.crs.Geodetic())

            if not np.all(np.isfinite([x1, y1])):
                raise TranslationError("Could not determine the first latitude"
                                       " and/or longitude grid point.")

            self._x_points = x1 + self.DxInMetres * np.arange(self.Nx,
                                                              dtype=np.float64)
            self._y_points = y1 + self.DyInMetres * np.arange(self.Ny,
                                                              dtype=np.float64)

        elif gridType in ["reduced_ll", "reduced_gg"]:
            self._x_points = self.longitudes
            self._y_points = self.latitudes

        else:
            raise TranslationError("unhandled grid type")

    def _regular_longitude_common(self):
        """Define a regular longitude dimension."""
        i_step = self.iDirectionIncrementInDegrees
        if self.iScansNegatively:
            i_step = -i_step
        self._x_points = (np.arange(self.Ni, dtype=np.float64) * i_step +
                          self.longitudeOfFirstGridPointInDegrees)
        if "longitude" in self.extra_keys['_x_coord_name'] and self.Ni > 1:
            if _longitude_is_cyclic(self._x_points):
                self.extra_keys['_x_circular'] = True

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

        # calculate start time (edition-dependent)
        if self.edition == 1:
            time_range_indicator = self.timeRangeIndicator
            P1 = self.P1
            P2 = self.P2
            if time_range_indicator == 0:    time_diff = P1       #Forecast product valid at reference time + P1 P1>0), or Uninitialized analysis product for reference time (P1=0). Or Image product for reference time (P1=0)
            elif time_range_indicator == 1:    time_diff = P1     #Initialized analysis product for reference time (P1=0).
            elif time_range_indicator == 2:    time_diff = (P1 + P2) * 0.5    #Product with a valid time ranging between reference time + P1 and reference time + P2
            elif time_range_indicator == 3:    time_diff = (P1 + P2) * 0.5    #Average(reference time + P1 to reference time + P2)
            elif time_range_indicator == 4:    time_diff = P2     #Accumulation (reference time + P1 to reference time + P2) product considered valid at reference time + P2
            elif time_range_indicator == 5:    time_diff = P2     #Difference(reference time + P2 minus reference time + P1) product considered valid at reference time + P2
            elif time_range_indicator == 10:   time_diff = P1 * 256 + P2    #P1 occupies octets 19 and 20; product valid at reference time + P1
            elif time_range_indicator == 51:                      #Climatological Mean Value: multiple year averages of quantities which are themselves means over some period of time (P2) less than a year. The reference time (R) indicates the date and time of the start of a period of time, given by R to R + P2, over which a mean is formed; N indicates the number of such period-means that are averaged together to form the climatological value, assuming that the N period-mean fields are separated by one year. The reference time indicates the start of the N-year climatology. N is given in octets 22-23 of the PDS. If P1 = 0 then the data averaged in the basic interval P2 are assumed to be continuous, i.e., all available data are simply averaged together. If P1 = 1 (the units of time - octet 18, code table 4 - are not relevant here) then the data averaged together in the basic interval P2 are valid only at the time (hour, minute) given in the reference time, for all the days included in the P2 period. The units of P2 are given by the contents of octet 18 and Table 4.
                raise TranslationError("unhandled grib1 timeRangeIndicator "
                                       "= 51 (avg of avgs)")
            elif time_range_indicator == 113:    time_diff = P1    #Average of N forecasts (or initialized analyses); each product has forecast period of P1 (P1=0 for initialized analyses); products have reference times at intervals of P2, beginning at the given reference time.
            elif time_range_indicator == 114:    time_diff = P1    #Accumulation of N forecasts (or initialized analyses); each product has forecast period of P1 (P1=0 for initialized analyses); products have reference times at intervals of P2, beginning at the given reference time.
            elif time_range_indicator == 115:    time_diff = P1    #Average of N forecasts, all with the same reference time; the first has a forecast period of P1, the remaining forecasts follow at intervals of P2.
            elif time_range_indicator == 116:    time_diff = P1    #Accumulation of N forecasts, all with the same reference time; the first has a forecast period of P1, the remaining follow at intervals of P2.
            elif time_range_indicator == 117:    time_diff = P1    #Average of N forecasts, the first has a period of P1, the subsequent ones have forecast periods reduced from the previous one by an interval of P2; the reference time for the first is given in octets 13-17, the subsequent ones have reference times increased from the previous one by an interval of P2. Thus all the forecasts have the same valid time, given by the initial reference time + P1.
            elif time_range_indicator == 118:    time_diff = P1    #Temporal variance, or covariance, of N initialized analyses; each product has forecast period P1=0; products have reference times at intervals of P2, beginning at the given reference time.
            elif time_range_indicator == 123:    time_diff = P1    #Average of N uninitialized analyses, starting at the reference time, at intervals of P2.
            elif time_range_indicator == 124:    time_diff = P1    #Accumulation of N uninitialized analyses, starting at the reference time, at intervals of P2.
            else:
                raise TranslationError("unhandled grib1 timeRangeIndicator "
                                       "= %i" % time_range_indicator)
        elif self.edition == 2:
            time_diff = int(self.stepRange)  # gribapi gives us a string!

        else:
            raise TranslationError(
                "unhandled grib edition = {}".format(self.edition)
            )

        # Get the timeunit interval.
        interval_secs = self._timeunit_seconds()
        # Multiply by start-offset and convert to a timedelta.
        #     NOTE: a 'float' conversion is required here, as time_diff may be
        #     a numpy scalar, which timedelta will not accept.
        interval_delta = datetime.timedelta(
            seconds=float(time_diff * interval_secs))
        # Return validity_time = (reference_time + start_offset*time_unit).
        return reference_date_time + interval_delta

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


def _longitude_is_cyclic(points):
    """Work out if a set of longitude points is cyclic."""
    # Is the gap from end to start smaller, or about equal to the max step?
    gap = 360.0 - abs(points[-1] - points[0])
    max_step = abs(np.diff(points)).max()
    cyclic = False
    if gap <= max_step:
        cyclic = True
    else:
        delta = 0.001
        if abs(1.0 - gap / max_step) < delta:
            cyclic = True
    return cyclic


def _message_values(grib_message, shape):
    gribapi.grib_set_double(grib_message, 'missingValue', np.nan)
    data = gribapi.grib_get_double_array(grib_message, 'values')
    data = data.reshape(shape)

    # Handle missing values in a sensible way.
    mask = np.isnan(data)
    if mask.any():
        data = ma.array(data, mask=mask, fill_value=np.nan)
    return data


def _is_quasi_regular_grib(grib_message):
    """Detect GRIB 'thinned' a.k.a 'reduced' a.k.a 'quasi-regular' grid."""
    reduced_grids = ("reduced_ll", "reduced_gg")
    return gribapi.grib_get(grib_message, 'gridType') in reduced_grids


def _regularise(grib_message):
    """
    Transform a reduced grid to a regular grid using interpolation.

    Uses 1d linear interpolation at constant latitude to make the grid
    regular. If the longitude dimension is circular then this is taken
    into account by the interpolation. If the longitude dimension is not
    circular then extrapolation is allowed to make sure all end regular
    grid points get a value. In practice this extrapolation is likely to
    be minimal.

    """
    # Make sure to read any missing values as NaN.
    gribapi.grib_set_double(grib_message, "missingValue", np.nan)

    # Get full longitude values, these describe the longitude value of
    # *every* point in the grid, they are not 1d monotonic coordinates.
    lons = gribapi.grib_get_double_array(grib_message, "longitudes")

    # Compute the new longitude coordinate for the regular grid.
    new_nx = max(gribapi.grib_get_long_array(grib_message, "pl"))
    new_x_step = (max(lons) - min(lons)) / (new_nx - 1)
    if gribapi.grib_get_long(grib_message, "iScansNegatively"):
        new_x_step *= -1

    new_lons = np.arange(new_nx) * new_x_step + lons[0]
    # Get full latitude and data values, these describe the latitude and
    # data values of *every* point in the grid, they are not 1d monotonic
    # coordinates.
    lats = gribapi.grib_get_double_array(grib_message, "latitudes")
    values = gribapi.grib_get_double_array(grib_message, "values")

    # Retrieve the distinct latitudes from the GRIB message. GRIBAPI docs
    # don't specify if these points are guaranteed to be oriented correctly so
    # the safe option is to sort them into ascending (south-to-north) order
    # and then reverse the order if necessary.
    new_lats = gribapi.grib_get_double_array(grib_message, "distinctLatitudes")
    new_lats.sort()
    if not gribapi.grib_get_long(grib_message, "jScansPositively"):
        new_lats = new_lats[::-1]
    ny = new_lats.shape[0]

    # Use 1d linear interpolation along latitude circles to regularise the
    # reduced data.
    cyclic = _longitude_is_cyclic(new_lons)
    new_values = np.empty([ny, new_nx], dtype=values.dtype)
    for ilat, lat in enumerate(new_lats):
        idx = np.where(lats == lat)
        llons = lons[idx]
        vvalues = values[idx]
        if cyclic:
            # For cyclic data we insert dummy points at each end to ensure
            # we can interpolate to all output longitudes using pure
            # interpolation.
            cgap = (360 - llons[-1] - llons[0])
            llons = np.concatenate(
                (llons[0:1] - cgap, llons, llons[-1:] + cgap))
            vvalues = np.concatenate(
                (vvalues[-1:], vvalues, vvalues[0:1]))
            fixed_latitude_interpolator = scipy.interpolate.interp1d(
                llons, vvalues)
        else:
            # Allow extrapolation for non-cyclic data sets to ensure we can
            # interpolate to all output longitudes.
            fixed_latitude_interpolator = Linear1dExtrapolator(
                scipy.interpolate.interp1d(llons, vvalues))
        new_values[ilat] = fixed_latitude_interpolator(new_lons)
    new_values = new_values.flatten()

    # Set flags for the regularised data.
    if np.isnan(new_values).any():
        # Account for any missing data.
        gribapi.grib_set_double(grib_message, "missingValue", np.inf)
        gribapi.grib_set(grib_message, "bitmapPresent", 1)
        new_values = np.where(np.isnan(new_values), np.inf, new_values)

    gribapi.grib_set_long(grib_message, "Nx", int(new_nx))
    gribapi.grib_set_double(grib_message,
                            "iDirectionIncrementInDegrees", float(new_x_step))
    gribapi.grib_set_double_array(grib_message, "values", new_values)
    gribapi.grib_set_long(grib_message, "jPointsAreConsecutive", 0)
    gribapi.grib_set_long(grib_message, "PLPresent", 0)


def grib_generator(filename, auto_regularise=True):
    """
    Returns a generator of :class:`~iris.fileformats.grib.GribWrapper`
    fields from the given filename.

    Args:

    * filename (string):
        Name of the file to generate fields from.

    Kwargs:

    * auto_regularise (*True* | *False*):
        If *True*, any field defined on a reduced grid will be interpolated
        to an equivalent regular grid. If *False*, any field defined on a
        reduced grid will be loaded on the raw reduced grid with no shape
        information. The default behaviour is to interpolate fields on a
        reduced grid to an equivalent regular grid.

    """
    with open(filename, 'rb') as grib_fh:
        while True:
            grib_message = gribapi.grib_new_from_file(grib_fh)
            if grib_message is None:
                break

            grib_wrapper = GribWrapper(grib_message, grib_fh, auto_regularise)

            yield grib_wrapper

            # finished with the grib message - claimed by the ecmwf c library.
            gribapi.grib_release(grib_message)


def load_cubes(filenames, callback=None, auto_regularise=True):
    """
    Returns a generator of cubes from the given list of filenames.

    Args:

    * filenames (string/list):
        One or more GRIB filenames to load from.

    Kwargs:

    * callback (callable function):
        Function which can be passed on to :func:`iris.io.run_callback`.

    * auto_regularise (*True* | *False*):
        If *True*, any cube defined on a reduced grid will be interpolated
        to an equivalent regular grid. If *False*, any cube defined on a
        reduced grid will be loaded on the raw reduced grid with no shape
        information. The default behaviour is to interpolate cubes on a
        reduced grid to an equivalent regular grid.

    .. note::

       To make use of the *auto_regularise* keyword the normal Iris loading
       pipeline cannot be used, the loading must be performed manually::

           cube_generator = iris.fileformats.grib.load_cubes(
               "reduced.grib", auto_regularise=False)
           cubes = iris.cube.CubeList(cube_generator).merge()

    """
    grib_loader = iris.fileformats.rules.Loader(
        grib_generator, {'auto_regularise': auto_regularise},
        iris.fileformats.grib.load_rules.convert,
        _load_rules)
    return iris.fileformats.rules.load_cubes(filenames, callback, grib_loader)


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
        raise TranslationError("Did not find one (and only one) "
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
