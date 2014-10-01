# (C) British Crown Copyright 2014, Met Office
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
Module to support the loading and convertion of a GRIB2 message into
cube metadata.

"""

from collections import namedtuple, Iterable, OrderedDict
from datetime import datetime, timedelta
import math
import threading
import warnings

import numpy as np
import numpy.ma as ma

from iris.aux_factory import HybridPressureFactory
import iris.coord_systems as icoord_systems
from iris.coords import AuxCoord, DimCoord, CellMethod
from iris.exceptions import TranslationError
from iris.fileformats.grib import grib_phenom_translation as itranslation
from iris.fileformats.rules import Factory, Reference
from iris.unit import CALENDAR_GREGORIAN, date2num, Unit
from iris.util import _is_circular


# Restrict the names imported from this namespace.
__all__ = ['convert']

options = threading.local()
options.warn_on_unsupported = False
options.support_hindcast_values = True

ScanningMode = namedtuple('ScanningMode', ['i_negative',
                                           'j_positive',
                                           'j_consecutive',
                                           'i_alternative'])

FixedSurface = namedtuple('FixedSurface', ['standard_name',
                                           'long_name',
                                           'units'])

# Regulations 92.1.6.
_GRID_ACCURACY_IN_DEGREES = 1e-6  # 1/1,000,000 of a degree

# Reference Common Code Table C-1.
_CENTRES = {
    'ecmf': 'European Centre for Medium Range Weather Forecasts'
}

# Reference Code Table 1.0
_CODE_TABLES_MISSING = 255

# UDUNITS-2 units time string. Reference GRIB2 Code Table 4.4.
_TIME_RANGE_UNITS = {
    0: 'minutes',
    1: 'hours',
    2: 'days',
    # 3: 'months',     Unsupported
    # 4: 'years',      Unsupported
    # 5: '10 years',   Unsupported
    # 6: '30 years',   Unsupported
    # 7: '100 years',  Unsupported
    # 8-9              Reserved
    10: '3 hours',
    11: '6 hours',
    12: '12 hours',
    13: 'seconds'
}

# Reference Code Table 4.5.
_FIXED_SURFACE = {
    100: FixedSurface(None, 'pressure', 'Pa'),  # Isobaric surface
    103: FixedSurface(None, 'height', 'm')      # Height level above ground
}
_FIXED_SURFACE_MISSING = 255

# Reference Code Table 6.0
_BITMAP_CODE_NONE = 255

# Reference Code Table 4.10.
_STATISTIC_TYPE_NAMES = {
    0: 'mean',
    2: 'maximum',
    3: 'minimum'
}

# Reference Code Table 4.11.
_STATISTIC_TYPE_OF_TIME_INTERVAL = {
    2: 'same start time of forecast, forecast time is incremented'
}
# NOTE: Our test data contains the value 2, which is all we currently support.
# The exact interpretation of this is still unclear.


# Regulation 92.1.12
def unscale(value, factor):
    """
    Implements Regulation 92.1.12.

    Args:

    * value:
        Scaled value or sequence of scaled values.

    * factor:
        Scale factor or sequence of scale factors.

    Returns:
        For scalar value and factor, the unscaled floating point
        result is returned. If either value and/or factor are
        MDI, then :data:`numpy.ma.masked` is returned.

        For sequence value and factor, the unscaled floating point
        :class:`numpy.ndarray` is returned. If either value and/or
        factor contain MDI, then :class:`numpy.ma.core.MaskedArray`
        is returned.

    """
    _unscale = lambda v, f: v / 10.0 ** f
    if isinstance(value, Iterable) or isinstance(factor, Iterable):
        def _masker(item):
            result = ma.masked_equal(item, _MDI)
            if ma.count_masked(result):
                # Circumvent downstream NumPy "RuntimeWarning"
                # of "overflow encountered in power" in _unscale
                # for data containing _MDI.
                result.data[result.mask] = 0
            return result
        value = _masker(value)
        factor = _masker(factor)
        result = _unscale(value, factor)
        if ma.count_masked(result) == 0:
            result = result.data
    else:
        result = ma.masked
        if value != _MDI and factor != _MDI:
            result = _unscale(value, factor)
    return result


# Regulations 92.1.4 and 92.1.5.
_MDI = 2 ** 32 - 1
# Note:
#   1. Integer "on-disk" values (aka. coded keys) in GRIB messages:
#       - Are 8-, 16-, or 32-bit.
#       - Are either signed or unsigned, with signed values stored as
#         sign-and-magnitude (*not* twos-complement).
#       - Use all bits set to indicate a missing value (MDI).
#   2. Irrespective of the on-disk form, the ECMWF GRIB API *always*:
#       - Returns values as 64-bit signed integers, either as native
#         Python 'int' or numpy 'int64'.
#       - Returns missing values as 2**32 - 1, but not all keys are
#         defined as supporting missing values.
#   NB. For keys which support missing values, the MDI value is reliably
#   distinct from the valid range of either signed or unsigned 8-, 16-,
#   or 32-bit values. For example:
#       unsigned 32-bit:
#           min = 0b000...000 = 0
#           max = 0b111...110 = 2**32 - 2
#           MDI = 0b111...111 = 2**32 - 1
#       signed 32-bit:
#           MDI = 0b111...111 = 2**32 - 1
#           min = 0b111...110 = -(2**31 - 2)
#           max = 0b011...111 = 2**31 - 1


# Non-standardised usage for negative forecast times.
def _hindcast_fix(forecast_time):
    """Return a forecast time interpreted as a possibly negative value."""
    uft = np.uint32(forecast_time)
    HIGHBIT = 2**30

    # Workaround grib api's assumption that forecast time is positive.
    # Handles correctly encoded -ve forecast times up to one -1 billion.
    if 2 * HIGHBIT < uft < 3 * HIGHBIT:
        original_forecast_time = forecast_time
        forecast_time = -(uft - 2 * HIGHBIT)
        if options.warn_on_unsupported:
            msg = ('Re-interpreting large grib forecastTime '
                   'from {} to {}.'.format(original_forecast_time,
                                           forecast_time))
            warnings.warn(msg)

    return forecast_time


###############################################################################
#
# Identification Section 1
#
###############################################################################

def reference_time_coord(section):
    """
    Translate section 1 reference time according to its significance.

    Reference section 1, year octets 13-14, month octet 15, day octet 16,
    hour octet 17, minute octet 18, second octet 19.

    Returns:
        The scalar reference time :class:`iris.coords.DimCoord`.

    """
    # Look-up standard name by significanceOfReferenceTime.
    _lookup = {1: 'forecast_reference_time',
               3: 'time'}

    # Calculate the reference time and units.
    dt = datetime(section['year'], section['month'], section['day'],
                  section['hour'], section['minute'], section['second'])
    # XXX Defaulting to a Gregorian calendar.
    # Current GRIBAPI does not cover GRIB Section 1 - Octets 22-nn (optional)
    # which are part of GRIB spec v12.
    unit = Unit('hours since epoch', calendar=CALENDAR_GREGORIAN)
    point = unit.date2num(dt)

    # Reference Code Table 1.2.
    significanceOfReferenceTime = section['significanceOfReferenceTime']
    standard_name = _lookup.get(significanceOfReferenceTime)

    if standard_name is None:
        msg = 'Identificaton section 1 contains an unsupported significance ' \
            'of reference time [{}]'.format(significanceOfReferenceTime)
        raise TranslationError(msg)

    # Create the associated reference time of data coordinate.
    coord = DimCoord(point, standard_name=standard_name, units=unit)

    return coord


###############################################################################
#
# Grid Definition Section 3
#
###############################################################################

def scanning_mode(scanningMode):
    """
    Translate the scanning mode bitmask.

    Reference GRIB2 Flag Table 3.4.

    Args:

    * scanningMode:
        Message section 3, octet 72.

    Returns:
        A :class:`collections.namedtuple` representation.

    """
    i_negative = bool(scanningMode & 0x80)
    j_positive = bool(scanningMode & 0x40)
    j_consecutive = bool(scanningMode & 0x20)
    i_alternative = bool(scanningMode & 0x10)

    if i_alternative:
        msg = 'Grid definition section 3 contains unsupported ' \
            'alternative row scanning mode'
        raise TranslationError(msg)

    return ScanningMode(i_negative, j_positive,
                        j_consecutive, i_alternative)


def ellipsoid(shapeOfTheEarth, major, minor, radius):
    """
    Translate the shape of the earth to an appropriate coordinate
    reference system.

    For MDI set either major and minor or radius to :data:`numpy.ma.masked`

    Reference GRIB2 Code Table 3.2.

    Args:

    * shapeOfTheEarth:
        Message section 3, octet 15.

    * major:
        Semi-major axis of the oblate spheroid in units determined by
        the shapeOfTheEarth.

    * minor:
        Semi-minor axis of the oblate spheroid in units determined by
        the shapeOfTheEarth.

    * radius:
        Radius of sphere (in m).

    Returns:
        :class:`iris.coord_systems.CoordSystem`

    """
    # Supported shapeOfTheEarth values.
    if shapeOfTheEarth not in (0, 1, 3, 6, 7):
        msg = 'Grid definition section 3 contains an unsupported ' \
            'shape of the earth [{}]'.format(shapeOfTheEarth)
        raise TranslationError(msg)

    if shapeOfTheEarth == 0:
        # Earth assumed spherical with radius of 6 367 470.0m
        result = icoord_systems.GeogCS(6367470)
    elif shapeOfTheEarth == 1:
        # Earth assumed spherical with radius specified (in m) by
        # data producer.
        if radius is ma.masked:
            msg = 'Ellipsoid for shape of the earth {} requires a' \
                'radius to be specified.'.format(shapeOfTheEarth)
            raise ValueError(msg)
        result = icoord_systems.GeogCS(radius)
    elif shapeOfTheEarth in [3, 7]:
        # Earth assumed oblate spheroid with major and minor axes
        # specified (in km)/(in m) by data producer.
        emsg_oblate = 'Ellipsoid for shape of the earth [{}] requires a' \
            'semi-{} axis to be specified.'
        if major is ma.masked:
            raise ValueError(emsg_oblate.format(shapeOfTheEarth, 'major'))
        if minor is ma.masked:
            raise ValueError(emsg_oblate.format(shapeOfTheEarth, 'minor'))
        # Check whether to convert from km to m.
        if shapeOfTheEarth == 3:
            major *= 1000
            minor *= 1000
        result = icoord_systems.GeogCS(major, minor)
    elif shapeOfTheEarth == 6:
        # Earth assumed spherical with radius of 6 371 229.0m
        result = icoord_systems.GeogCS(6371229)

    return result


def ellipsoid_geometry(section):
    """
    Calculated the unscaled ellipsoid major-axis, minor-axis and radius.

    Args:

    * section:
        Dictionary of coded key/value pairs from section 3 of the message.

    Returns:
        Tuple containing the major-axis, minor-axis and radius.

    """
    major = unscale(section['scaledValueOfEarthMajorAxis'],
                    section['scaleFactorOfEarthMajorAxis'])
    minor = unscale(section['scaledValueOfEarthMinorAxis'],
                    section['scaleFactorOfEarthMinorAxis'])
    radius = unscale(section['scaledValueOfRadiusOfSphericalEarth'],
                     section['scaleFactorOfRadiusOfSphericalEarth'])
    return major, minor, radius


def grid_definition_template_0_and_1(section, metadata, y_name, x_name, cs):
    """
    Translate templates representing regularly spaced latitude/longitude
    on either a standard or rotated grid.

    Updates the metadata in-place with the translations.

    Args:

    * section:
        Dictionary of coded key/value pairs from section 3 of the message

    * metadata:
        :class:`collections.OrderedDict` of metadata.

    * y_name:
        Name of the Y coordinate, e.g. latitude or grid_latitude.

    * x_name:
        Name of the X coordinate, e.g. longitude or grid_longitude.

    * cs:
        The :class:`iris.coord_systems.CoordSystem` to use when creating
        the X and Y coordinates.

    """
    scan = scanning_mode(section['scanningMode'])

    # Calculate longitude points.
    x_inc = section['iDirectionIncrement'] * _GRID_ACCURACY_IN_DEGREES
    x_offset = section['longitudeOfFirstGridPoint'] * _GRID_ACCURACY_IN_DEGREES
    x_direction = -1 if scan.i_negative else 1
    Ni = section['Ni']
    x_points = np.arange(Ni, dtype=np.float64) * x_inc * x_direction + x_offset

    # Determine whether the x-points (in degrees) are circular.
    circular = _is_circular(x_points, 360.0)

    # Calculate latitude points.
    y_inc = section['jDirectionIncrement'] * _GRID_ACCURACY_IN_DEGREES
    y_offset = section['latitudeOfFirstGridPoint'] * _GRID_ACCURACY_IN_DEGREES
    y_direction = 1 if scan.j_positive else -1
    Nj = section['Nj']
    y_points = np.arange(Nj, dtype=np.float64) * y_inc * y_direction + y_offset

    # Create the lat/lon coordinates.
    y_coord = DimCoord(y_points, standard_name=y_name, units='degrees',
                       coord_system=cs)
    x_coord = DimCoord(x_points, standard_name=x_name, units='degrees',
                       coord_system=cs, circular=circular)

    # Determine the lat/lon dimensions.
    y_dim, x_dim = 0, 1
    if scan.j_consecutive:
        y_dim, x_dim = 1, 0

    # Add the lat/lon coordinates to the metadata dim coords.
    metadata['dim_coords_and_dims'].append((y_coord, y_dim))
    metadata['dim_coords_and_dims'].append((x_coord, x_dim))


def grid_definition_template_0(section, metadata):
    """
    Translate template representing regular latitude/longitude
    grid (regular_ll).

    Updates the metadata in-place with the translations.

    Args:

    * section:
        Dictionary of coded key/value pairs from section 3 of the message

    * metadata:
        :class:`collections.OrderedDict` of metadata.

    """
    # Determine the coordinate system.
    major, minor, radius = ellipsoid_geometry(section)
    cs = ellipsoid(section['shapeOfTheEarth'], major, minor, radius)
    grid_definition_template_0_and_1(section, metadata,
                                     'latitude', 'longitude', cs)


def grid_definition_template_1(section, metadata):
    """
    Translate template representing rotated latitude/longitude grid.

    Updates the metadata in-place with the translations.

    Args:

    * section:
        Dictionary of coded key/value pairs from section 3 of the message

    * metadata:
        :class:`collections.OrderedDict` of metadata.

    """
    # Determine the coordinate system.
    major, minor, radius = ellipsoid_geometry(section)
    south_pole_lat = (section['latitudeOfSouthernPole'] *
                      _GRID_ACCURACY_IN_DEGREES)
    south_pole_lon = (section['longitudeOfSouthernPole'] *
                      _GRID_ACCURACY_IN_DEGREES)
    cs = icoord_systems.RotatedGeogCS(-south_pole_lat,
                                      math.fmod(south_pole_lon + 180, 360),
                                      section['angleOfRotation'],
                                      ellipsoid(section['shapeOfTheEarth'],
                                                major, minor, radius))
    grid_definition_template_0_and_1(section, metadata,
                                     'grid_latitude', 'grid_longitude', cs)


def grid_definition_section(section, metadata):
    """
    Translate section 3 from the GRIB2 message.

    Update the metadata in-place with the translations.

    Args:

    * section:
        Dictionary of coded key/value pairs from section 3 of the message.

    * metadata:
        :class:`collections.OrderedDict` of metadata.

    """
    # Reference GRIB2 Code Table 3.0.
    value = section['sourceOfGridDefinition']
    if value != 0:
        msg = 'Grid definition section 3 contains unsupported ' \
            'source of grid definition [{}]'.format(value)
        raise TranslationError(msg)

    if section['numberOfOctectsForNumberOfPoints'] != 0 or \
            section['interpretationOfNumberOfPoints'] != 0:
        msg = 'Grid definition section 3 contains unsupported ' \
            'quasi-regular grid'
        raise TranslationError(msg)

    # Reference GRIB2 Code Table 3.1.
    template = section['gridDefinitionTemplateNumber']

    if template == 0:
        # Process regular latitude/longitude grid (regular_ll)
        grid_definition_template_0(section, metadata)
    elif template == 1:
        # Process rotated latitude/longitude grid.
        grid_definition_template_1(section, metadata)
    else:
        msg = 'Grid definition template [{}] is not supported'.format(template)
        raise TranslationError(msg)


###############################################################################
#
# Product Definition Section 4
#
###############################################################################

def translate_phenomenon(metadata, discipline, parameterCategory,
                         parameterNumber):
    """
    Translate GRIB2 phenomenon to CF phenomenon.

    Updates the metadata in-place with the translations.

    Args:

    * metadata:
        :class:`collections.OrderedDict` of metadata.

    * discipline:
        Message section 0, octet 7.

    * parameterCategory:
        Message section 4, octet 10.

    * parameterNumber:
        Message section 4, octet 11.

    """
    cf = itranslation.grib2_phenom_to_cf_info(param_discipline=discipline,
                                              param_category=parameterCategory,
                                              param_number=parameterNumber)
    if cf is not None:
        metadata['standard_name'] = cf.standard_name
        metadata['long_name'] = cf.long_name
        metadata['units'] = cf.units


def time_range_unit(indicatorOfUnitOfTimeRange):
    """
    Translate the time range indicator to an equivalent
    :class:`iris.unit.Unit`.

    Args:

    * indicatorOfUnitOfTimeRange:
        Message section 4, octet 18.

    Returns:
        :class:`iris.unit.Unit`.

    """
    try:
        unit = Unit(_TIME_RANGE_UNITS[indicatorOfUnitOfTimeRange])
    except (KeyError, ValueError):
        msg = 'Product definition section 4 contains unsupported ' \
            'time range unit [{}]'.format(indicatorOfUnitOfTimeRange)
        raise TranslationError(msg)
    return unit


def hybrid_factories(section, metadata):
    """
    Translate the section 4 optional hybrid vertical coordinates.

    Updates the metadata in-place with the translations.

    Reference GRIB2 Code Table 4.5.

    Relevant notes:
    [3] Hybrid pressure level (119) shall be used instead of Hybrid level (105)

    Args:

    * section:
        Dictionary of coded key/value pairs from section 4 of the message.

    * metadata:
        :class:`collections.OrderedDict` of metadata.

    """
    NV = section['NV']
    if NV > 0:
        typeOfFirstFixedSurface = section['typeOfFirstFixedSurface']
        if typeOfFirstFixedSurface == _FIXED_SURFACE_MISSING:
            msg = 'Product definition section 4 contains missing ' \
                'type of first fixed surface'
            raise TranslationError(msg)

        typeOfSecondFixedSurface = section['typeOfSecondFixedSurface']
        if typeOfSecondFixedSurface != _FIXED_SURFACE_MISSING:
            msg = 'Product definition section 4 contains unsupported type ' \
                'of second fixed surface [{}]'.format(typeOfSecondFixedSurface)
            raise TranslationError(msg)

        if typeOfFirstFixedSurface in [105, 119]:
            # Hybrid level (105) and Hybrid pressure level (119).
            scaleFactor = section['scaleFactorOfFirstFixedSurface']
            if scaleFactor != 0:
                msg = 'Product definition section 4 contains invalid scale ' \
                    'factor of first fixed surface [{}]'.format(scaleFactor)
                raise TranslationError(msg)

            # Create the model level number scalar coordinate.
            scaledValue = section['scaledValueOfFirstFixedSurface']
            coord = DimCoord(scaledValue, standard_name='model_level_number',
                             attributes=dict(positive='up'))
            metadata['aux_coords_and_dims'].append((coord, None))
            # Create the level pressure scalar coordinate.
            pv = section['pv']
            offset = scaledValue
            coord = DimCoord(pv[offset], long_name='level_pressure',
                             units='Pa')
            metadata['aux_coords_and_dims'].append((coord, None))
            # Create the sigma scalar coordinate.
            offset += NV / 2
            coord = AuxCoord(pv[offset], long_name='sigma')
            metadata['aux_coords_and_dims'].append((coord, None))
            # Create the associated factory reference.
            args = [{'long_name': 'level_pressure'}, {'long_name': 'sigma'},
                    Reference('surface_air_pressure')]
            factory = Factory(HybridPressureFactory, args)
            metadata['factories'].append(factory)
        else:
            msg = 'Product definition section 4 contains unsupported ' \
                'first fixed surface [{}]'.format(typeOfFirstFixedSurface)
            raise TranslationError(msg)


def vertical_coords(section, metadata):
    """
    Translate the vertical coordinates or hybrid vertical coordinates.

    Updates the metadata in-place with the translations.

    Reference GRIB2 Code Table 4.5.

    Args:

    * section:
        Dictionary of coded key/value pairs from section 4 of the message.

    * metadata:
        :class:`collections.OrderedDict` of metadata.

    """
    if section['NV'] > 0:
        # Generate hybrid vertical coordinates.
        hybrid_factories(section, metadata)
    else:
        # Generate vertical coordinate.
        typeOfFirstFixedSurface = section['typeOfFirstFixedSurface']
        key = 'scaledValueOfFirstFixedSurface'
        scaledValueOfFirstFixedSurface = section[key]
        fixed_surface = _FIXED_SURFACE.get(typeOfFirstFixedSurface)

        if fixed_surface is None:
            if typeOfFirstFixedSurface != _FIXED_SURFACE_MISSING:
                if scaledValueOfFirstFixedSurface == _FIXED_SURFACE_MISSING:
                    if options.warn_on_unsupported:
                        msg = 'Unable to translate type of first fixed ' \
                            'surface with missing scaled value.'
                        warnings.warn(msg)
                else:
                    if options.warn_on_unsupported:
                        msg = 'Unable to translate type of first fixed ' \
                            'surface with scaled value.'
                        warnings.warn(msg)
        else:
            key = 'scaleFactorOfFirstFixedSurface'
            scaleFactorOfFirstFixedSurface = section[key]
            typeOfSecondFixedSurface = section['typeOfSecondFixedSurface']

            if typeOfSecondFixedSurface != _FIXED_SURFACE_MISSING:
                if typeOfFirstFixedSurface != typeOfSecondFixedSurface:
                    msg = 'Product definition section 4 has different ' \
                        'types of first and second fixed surface'
                    raise TranslationError(msg)

                key = 'scaledValueOfSecondFixedSurface'
                scaledValueOfSecondFixedSurface = section[key]

                if scaledValueOfSecondFixedSurface == _MDI:
                    msg = 'Product definition section 4 has missing ' \
                        'scaled value of second fixed surface'
                    raise TranslationError(msg)
                else:
                    key = 'scaleFactorOfSecondFixedSurface'
                    scaleFactorOfSecondFixedSurface = section[key]
                    first = unscale(scaledValueOfFirstFixedSurface,
                                    scaleFactorOfFirstFixedSurface)
                    second = unscale(scaledValueOfSecondFixedSurface,
                                     scaleFactorOfSecondFixedSurface)
                    point = 0.5 * (first + second)
                    bounds = [first, second]
                    coord = DimCoord(point,
                                     standard_name=fixed_surface.standard_name,
                                     long_name=fixed_surface.long_name,
                                     units=fixed_surface.units,
                                     bounds=bounds)
                    # Add the vertical coordinate to metadata aux coords.
                    metadata['aux_coords_and_dims'].append((coord, None))
            else:
                point = unscale(scaledValueOfFirstFixedSurface,
                                scaleFactorOfFirstFixedSurface)
                coord = DimCoord(point,
                                 standard_name=fixed_surface.standard_name,
                                 long_name=fixed_surface.long_name,
                                 units=fixed_surface.units)
                # Add the vertical coordinate to metadata aux coords.
                metadata['aux_coords_and_dims'].append((coord, None))


def forecast_period_coord(indicatorOfUnitOfTimeRange, forecastTime):
    """
    Create the forecast period coordinate.

    Args:

    * indicatorOfUnitOfTimeRange:
        Message section 4, octets 18.

    * forecastTime:
        Message section 4, octets 19-22.

    Returns:
        The scalar forecast period :class:`iris.coords.DimCoord`.

    """
    # Determine the forecast period and associated units.
    unit = time_range_unit(indicatorOfUnitOfTimeRange)
    point = unit.convert(forecastTime, 'hours')
    # Create the forecast period scalar coordinate.
    coord = DimCoord(point, standard_name='forecast_period', units='hours')
    return coord


def statistical_forecast_period_coord(section, frt_coord):
    """
    Create a forecast period coordinate for a time-statistic message.

    This applies only with a product definition template 4.8.

    Args:

    * section:
        Dictionary of coded key/value pairs from section 4 of the message.

    * frt_coord:
        The scalar forecast reference time :class:`iris.coords.DimCoord`.

    Returns:
        The scalar forecast period :class:`iris.coords.DimCoord`, containing a
        single, bounded point (period value).

    """
    # Get the period end time as a datetime.
    end_time = datetime(section['yearOfEndOfOverallTimeInterval'],
                        section['monthOfEndOfOverallTimeInterval'],
                        section['dayOfEndOfOverallTimeInterval'],
                        section['hourOfEndOfOverallTimeInterval'],
                        section['minuteOfEndOfOverallTimeInterval'],
                        section['secondOfEndOfOverallTimeInterval'])

    # Get forecast reference time (frt) as a datetime.
    frt_point = frt_coord.units.num2date(frt_coord.points[0])

    # Get the period start time (as a timedelta relative to the frt).
    forecast_time = section['forecastTime']
    if options.support_hindcast_values:
        # Apply the hindcast fix.
        forecast_time = _hindcast_fix(forecast_time)
    forecast_units = time_range_unit(section['indicatorOfUnitOfTimeRange'])
    forecast_seconds = forecast_units.convert(forecast_time, 'seconds')
    start_time_delta = timedelta(seconds=forecast_seconds)

    # Get the period end time (as a timedelta relative to the frt).
    end_time_delta = end_time - frt_point

    # Get the middle of the period (as a timedelta relative to the frt).
    mid_time_delta = (start_time_delta + end_time_delta) / 2

    # Create and return the forecast period coordinate.
    def timedelta_hours(timedelta):
        return timedelta.total_seconds() / 3600.0

    mid_point_hours = timedelta_hours(mid_time_delta)
    bounds_hours = [timedelta_hours(start_time_delta),
                    timedelta_hours(end_time_delta)]
    fp_coord = DimCoord(mid_point_hours, bounds=bounds_hours,
                        standard_name='forecast_period', units='hours')
    return fp_coord


def validity_time_coord(frt_coord, fp_coord):
    """
    Create the validity or phenomenon time coordinate.

    Args:

    * frt_coord:
        The scalar forecast reference time :class:`iris.coords.DimCoord`.

    * fp_coord:
        The scalar forecast period :class:`iris.coords.DimCoord`.

    Returns:
        The scalar time :class:`iris.coords.DimCoord`.
        It has bounds if the period coord has them, otherwise not.

    """
    if frt_coord.shape != (1,):
        msg = 'Expected scalar forecast reference time coordinate when ' \
            'calculating validity time, got shape {!r}'.format(frt_coord.shape)
        raise ValueError(msg)

    if fp_coord.shape != (1,):
        msg = 'Expected scalar forecast period coordinate when ' \
            'calculating validity time, got shape {!r}'.format(fp_coord.shape)
        raise ValueError(msg)

    def coord_timedelta(coord, value):
        # Helper to convert a time coordinate value into a timedelta.
        seconds = coord.units.convert(value, 'seconds')
        return timedelta(seconds=seconds)

    # Calculate validity (phenomenon) time in forecast-reference-time units.
    frt_point = frt_coord.units.num2date(frt_coord.points[0])
    point_delta = coord_timedelta(fp_coord, fp_coord.points[0])
    point = frt_coord.units.date2num(frt_point + point_delta)

    # Calculate bounds (if any) in the same way.
    if fp_coord.bounds is None:
        bounds = None
    else:
        bounds_deltas = [coord_timedelta(fp_coord, bound_point)
                         for bound_point in fp_coord.bounds[0]]
        bounds = [frt_coord.units.date2num(frt_point + delta)
                  for delta in bounds_deltas]

    # Create the time scalar coordinate.
    coord = DimCoord(point, bounds=bounds,
                     standard_name='time', units=frt_coord.units)
    return coord


def generating_process(section):
    if options.warn_on_unsupported:
        # Reference Code Table 4.3.
        warnings.warn('Unable to translate type of generating process.')
        warnings.warn('Unable to translate background generating '
                      'process identifier.')
        warnings.warn('Unable to translate forecast generating '
                      'process identifier.')


def data_cutoff(hoursAfterDataCutoff, minutesAfterDataCutoff):
    """
    Handle the after reference time data cutoff.

    Args:

    * hoursAfterDataCutoff:
        Message section 4, octets 15-16.

    * minutesAfterDataCutoff:
        Message section 4, octet 17.

    """
    if (hoursAfterDataCutoff != _MDI or
            minutesAfterDataCutoff != _MDI):
        if options.warn_on_unsupported:
            warnings.warn('Unable to translate "hours and/or minutes '
                          'after data cutoff".')


def statistical_cell_method(section):
    """
    Create a cell method representing a time statistic.

    This applies only with a product definition template 4.8.

    Args:

    * section:
        Dictionary of coded key/value pairs from section 4 of the message.

    Returns:
        A cell method over 'time'.

    """
    # Handle the number of time ranges -- we currently only support one.
    n_time_ranges = section['numberOfTimeRange']
    if n_time_ranges != 1:
        if n_time_ranges == 0:
            msg = ('Product definition section 4 specifies aggregation over '
                   '"0 time ranges".')
            raise TranslationError(msg)
        else:
            msg = ('Product definition section 4 specifies aggregation over '
                   'multiple time ranges [{}], which is not yet '
                   'supported.'.format(n_time_ranges))
            raise TranslationError(msg)

    # Decode the type of statistic (aggregation method).
    statistic_code = section['typeOfStatisticalProcessing']
    statistic_name = _STATISTIC_TYPE_NAMES.get(statistic_code)
    if statistic_name is None:
        msg = ('grib statistical process type [{}] '
               'is not supported'.format(statistic_code))
        raise TranslationError(msg)

    # Decode the type of time increment.
    increment_typecode = section['typeOfTimeIncrement']
    if increment_typecode != 2:
        # NOTE: All our current test data seems to contain the value 2, which
        # is all we currently support.
        # The exact interpretation of this is still unclear.
        msg = ('grib statistic time-increment type [{}] '
               'is not supported.'.format(increment_typecode))
        raise TranslationError(msg)

    interval_number = section['timeIncrement']
    if interval_number == 0:
        intervals_string = None
    else:
        units_string = _TIME_RANGE_UNITS[
            section['indicatorOfUnitForTimeIncrement']]
        intervals_string = '{} {}'.format(interval_number, units_string)

    # Create a cell method to represent the time aggregation.
    cell_method = CellMethod(method=statistic_name,
                             coords='time',
                             intervals=intervals_string)
    return cell_method


def product_definition_template_0(section, metadata, frt_coord):
    """
    Translate template representing an analysis or forecast at a horizontal
    level or in a horizontal layer at a point in time.

    Updates the metadata in-place with the translations.

    Args:

    * section:
        Dictionary of coded key/value pairs from section 4 of the message.

    * metadata:
        :class:`collections.OrderedDict` of metadata.

    * frt_coord:
        The scalar forecast reference time :class:`iris.coords.DimCoord`.

    """
    # Handle generating process details.
    generating_process(section)

    # Handle the data cutoff.
    data_cutoff(section['hoursAfterDataCutoff'],
                section['minutesAfterDataCutoff'])

    # Calculate the forecast period coordinate.
    fp_coord = forecast_period_coord(section['indicatorOfUnitOfTimeRange'],
                                     section['forecastTime'])
    # Add the forecast period coordinate to the metadata aux coords.
    metadata['aux_coords_and_dims'].append((fp_coord, None))

    # Calculate the validity (phenomenon) time.
    t_coord = validity_time_coord(frt_coord, fp_coord)
    # Add the time coordinate to the metadata aux coords.
    metadata['aux_coords_and_dims'].append((t_coord, None))

    # Add the forecast reference time coordinate to the metadata aux coords.
    metadata['aux_coords_and_dims'].append((frt_coord, None))

    # Check for vertical coordinates.
    vertical_coords(section, metadata)


def product_definition_template_1(section, metadata, frt_coord):
    """
    Translate template representing individual ensemble forecast, control
    and perturbed, at a horizontal level or in a horizontal layer at a
    point in time.

    Updates the metadata in-place with the translations.

    Args:

    * section:
        Dictionary of coded key/value pairs from section 4 of the message.

    * metadata:
        :class:`collectins.OrderedDict` of metadata.

    * frt_coord:
        The scalar forecast reference time :class:`iris.coords.DimCoord`.

    """
    # Perform identical message processing.
    product_definition_template_0(section, metadata, frt_coord)

    if options.warn_on_unsupported:
        # Reference Code Table 4.6.
        warnings.warn('Unable to translate type of ensemble forecast.')
        warnings.warn('Unable to translate number of forecasts in ensemble.')

    # Create the realization coordinates.
    realization = DimCoord(section['perturbationNumber'],
                           standard_name='realization',
                           units='no_unit')
    # Add the realization coordinate to the metadata aux coords.
    metadata['aux_coords_and_dims'].append((realization, None))


def product_definition_template_8(section, metadata, frt_coord):
    """
    Translate template representing average, accumulation and/or extreme values
    or other statistically processed values at a horizontal level or in a
    horizontal layer in a continuous or non-continuous time interval.

    Updates the metadata in-place with the translations.

    Args:

    * section:
        Dictionary of coded key/value pairs from section 4 of the message.

    * metadata:
        :class:`collections.OrderedDict` of metadata.

    * frt_coord:
        The scalar forecast reference time :class:`iris.coords.DimCoord`.

    """
    # Handle generating process details.
    generating_process(section)

    # Handle the data cutoff.
    data_cutoff(section['hoursAfterDataCutoff'],
                section['minutesAfterDataCutoff'])

    # Create a cell method to represent the time statistic.
    time_statistic_cell_method = statistical_cell_method(section)
    # Add the forecast cell method to the metadata.
    metadata['cell_methods'].append(time_statistic_cell_method)

    # Add the forecast reference time coordinate to the metadata aux coords.
    metadata['aux_coords_and_dims'].append((frt_coord, None))

    # Add a bounded forecast period coordinate.
    fp_coord = statistical_forecast_period_coord(section, frt_coord)
    metadata['aux_coords_and_dims'].append((fp_coord, None))

    # Calculate a bounded validity time coord matching the forecast period.
    t_coord = validity_time_coord(frt_coord, fp_coord)
    # Add the time coordinate to the metadata aux coords.
    metadata['aux_coords_and_dims'].append((t_coord, None))

    # Check for vertical coordinates.
    vertical_coords(section, metadata)


def product_definition_template_31(section, metadata, rt_coord):
    """
    Translate template representing a satellite product.

    Updates the metadata in-place with the translations.

    Args:

    * section:
        Dictionary of coded key/value pairs from section 4 of the message.

    * metadata:
        :class:`collections.OrderedDict` of metadata.

    * rt_coord:
        The scalar observation time :class:`iris.coords.DimCoord'.

    """
    if options.warn_on_unsupported:
        warnings.warn('Unable to translate type of generating process.')
        warnings.warn('Unable to translate observation generating '
                      'process identifier.')

    # Number of contributing spectral bands.
    NB = section['NB']

    if NB > 0:
        # Create the satellite series coordinate.
        satelliteSeries = section['satelliteSeries']
        coord = AuxCoord(satelliteSeries, long_name='satellite_series')
        # Add the satellite series coordinate to the metadata aux coords.
        metadata['aux_coords_and_dims'].append((coord, None))

        # Create the satellite number coordinate.
        satelliteNumber = section['satelliteNumber']
        coord = AuxCoord(satelliteNumber, long_name='satellite_number')
        # Add the satellite number coordinate to the metadata aux coords.
        metadata['aux_coords_and_dims'].append((coord, None))

        # Create the satellite instrument type coordinate.
        instrumentType = section['instrumentType']
        coord = AuxCoord(instrumentType, long_name='instrument_type')
        # Add the instrument type coordinate to the metadata aux coords.
        metadata['aux_coords_and_dims'].append((coord, None))

        # Create the central wave number coordinate.
        scaleFactor = section['scaleFactorOfCentralWaveNumber']
        scaledValue = section['scaledValueOfCentralWaveNumber']
        wave_number = unscale(scaledValue, scaleFactor)
        standard_name = 'sensor_band_central_radiation_wavenumber'
        coord = AuxCoord(wave_number,
                         standard_name=standard_name,
                         units=Unit('m-1'))
        # Add the central wave number coordinate to the metadata aux coords.
        metadata['aux_coords_and_dims'].append((coord, None))

        # Add the observation time coordinate.
        metadata['aux_coords_and_dims'].append((rt_coord, None))


def product_definition_section(section, metadata, discipline, tablesVersion,
                               rt_coord):
    """
    Translate section 4 from the GRIB2 message.

    Updates the metadata in-place with the translations.

    Args:

    * section:
        Dictionary of coded key/value pairs from section 4 of the message.

    * metadata:
        :class:`collections.OrderedDict` of metadata.

    * discipline:
        Message section 0, octet 7.

    * tablesVersion:
        Message section 1, octet 10.

    * rt_coord:
        The scalar reference time :class:`iris.coords.DimCoord`.

    """
    # Reference GRIB2 Code Table 4.0.
    template = section['productDefinitionTemplateNumber']

    if template == 0:
        # Process analysis or forecast at a horizontal level or
        # in a horizontal layer at a point in time.
        product_definition_template_0(section, metadata, rt_coord)
    elif template == 1:
        # Process individual ensemble forecast, control and perturbed, at
        # a horizontal level or in a horizontal layer at a point in time.
        product_definition_template_1(section, metadata, rt_coord)
    elif template == 8:
        # Process statistically processed values at a horizontal level or in a
        # horizontal layer in a continuous or non-continuous time interval.
        product_definition_template_8(section, metadata, rt_coord)
    elif template == 31:
        # Process satellite product.
        product_definition_template_31(section, metadata, rt_coord)
    else:
        msg = 'Product definition template [{}] is not ' \
            'supported'.format(template)
        raise TranslationError(msg)

    # Translate GRIB2 phenomenon to CF phenomenon.
    if tablesVersion != _CODE_TABLES_MISSING:
        translate_phenomenon(metadata, discipline,
                             section['parameterCategory'],
                             section['parameterNumber'])


###############################################################################
#
# Data Representation Section 5
#
###############################################################################

def data_representation_section(section):
    """
    Translate section 5 from the GRIB2 message.

    """
    # Reference GRIB2 Code Table 5.0.
    template = section['dataRepresentationTemplateNumber']

    if template != 0:
        msg = 'Data Representation Section Template [{}] is not ' \
            'supported'.format(template)
        raise TranslationError(msg)


###############################################################################
#
# Bitmap Section 6
#
###############################################################################

def bitmap_section(section):
    """
    Translate section 6 from the GRIB2 message.

    """
    # Reference GRIB2 Code Table 6.0.
    bitMapIndicator = section['bitMapIndicator']

    if bitMapIndicator != _BITMAP_CODE_NONE:
        msg = 'Bitmap Section 6 contains unsupported ' \
            'bitmap indicator [{}]'.format(bitMapIndicator)
        raise TranslationError(msg)


###############################################################################

def grib2_convert(field, metadata):
    """
    Translate the GRIB2 message into the appropriate cube metadata.

    Updates the metadata in-place with the translations.

    Args:

    * field:
        GRIB2 message to be translated.

    * metadata:
        :class:`collections.OrderedDict` of metadata.

    """
    # Section 1 - Identification Section.
    centre = _CENTRES.get(field.sections[1]['centre'])
    if centre is not None:
        metadata['attributes']['centre'] = centre
    rt_coord = reference_time_coord(field.sections[1])

    # Section 3 - Grid Definition Section (Grid Definition Template)
    grid_definition_section(field.sections[3], metadata)

    # Section 4 - Product Definition Section (Product Definition Template)
    product_definition_section(field.sections[4], metadata,
                               field.sections[0]['discipline'],
                               field.sections[1]['tablesVersion'],
                               rt_coord)

    # Section 5 - Data Representation Section (Data Representation Template)
    data_representation_section(field.sections[5])

    # Section 6 - Bitmap Section.
    bitmap_section(field.sections[6])


###############################################################################

def convert(field):
    """
    Translate the GRIB message into the appropriate cube metadata.

    Args:

    * field:
        GRIB message to be translated.

    Returns:
        Translated cube metadata tuple containing factories list, references
        list, standard_name, long_name, units, attributes dictionary, cell
        methods list, list containing dimension coordinate and associated
        dimension tuple pairs, and a list containing auxiliary coordinate and
        associated dimensions tuple pairs.

    """
    editionNumber = field.sections[0]['editionNumber']
    if editionNumber != 2:
        msg = 'GRIB edition {} is not supported'.format(editionNumber)
        raise TranslationError(msg)

    # Initialise the cube metadata.
    metadata = OrderedDict()
    metadata['factories'] = []
    metadata['references'] = []
    metadata['standard_name'] = None
    metadata['long_name'] = None
    metadata['units'] = None
    metadata['attributes'] = {}
    metadata['cell_methods'] = []
    metadata['dim_coords_and_dims'] = []
    metadata['aux_coords_and_dims'] = []

    # Convert GRIB2 message to cube metadata.
    grib2_convert(field, metadata)

    return tuple(metadata.values())
