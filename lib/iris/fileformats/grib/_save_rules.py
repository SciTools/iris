# (C) British Crown Copyright 2010 - 2015, Met Office
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
Grib save implementation.

This module replaces the deprecated
:mod:`iris.fileformats.grib.grib_save_rules`. It is a private module
with no public API. It is invoked from
:meth:`iris.fileformats.grib.save_grib2`.

"""

from __future__ import (absolute_import, division, print_function)

import warnings

import gribapi
import numpy as np
import numpy.ma as ma

import iris
import iris.exceptions
import iris.unit
from iris.coord_systems import GeogCS, RotatedGeogCS, TransverseMercator
from iris.fileformats.grib import grib_phenom_translation as gptx
from iris.fileformats.grib._load_convert import (_STATISTIC_TYPE_NAMES,
                                                 _TIME_RANGE_UNITS)
from iris.util import is_regular, regular_step


# Invert code tables from :mod:`iris.fileformats.grib._load_convert`.
_STATISTIC_TYPE_NAMES = {val: key for key, val in
                         _STATISTIC_TYPE_NAMES.items()}
_TIME_RANGE_UNITS = {val: key for key, val in _TIME_RANGE_UNITS.items()}


def fixup_float32_as_int32(value):
    """
    Workaround for use when the ECMWF GRIB API treats an IEEE 32-bit
    floating-point value as a signed, 4-byte integer.

    Returns the integer value which will result in the on-disk
    representation corresponding to the IEEE 32-bit floating-point
    value.

    """
    value_as_float32 = np.array(value, dtype='f4')
    value_as_uint32 = value_as_float32.view(dtype='u4')
    if value_as_uint32 >= 0x80000000:
        # Convert from two's-complement to sign-and-magnitude.
        # NB. Because of the silly representation of negative
        # integers in GRIB2, there is no value we can pass to
        # grib_set that will result in the bit pattern 0x80000000.
        # But since that bit pattern corresponds to a floating
        # point value of negative-zero, we can safely treat it as
        # positive-zero instead.
        value_as_grib_int = 0x80000000 - int(value_as_uint32)
    else:
        value_as_grib_int = int(value_as_uint32)
    return value_as_grib_int


def fixup_int32_as_uint32(value):
    """
    Workaround for use when the ECMWF GRIB API treats a signed, 4-byte
    integer value as an unsigned, 4-byte integer.

    Returns the unsigned integer value which will result in the on-disk
    representation corresponding to the signed, 4-byte integer value.

    """
    value = int(value)
    if -0x7fffffff <= value <= 0x7fffffff:
        if value < 0:
            # Convert from two's-complement to sign-and-magnitude.
            value = 0x80000000 - value
    else:
        msg = '{} out of range -2147483647 to 2147483647.'.format(value)
        raise ValueError(msg)
    return value


def ensure_set_int32_value(grib, key, value):
    """
    Ensure the workaround function :func:`fixup_int32_as_uint32` is applied as
    necessary to problem keys.

    """
    try:
        gribapi.grib_set(grib, key, value)
    except gribapi.GribInternalError:
        value = fixup_int32_as_uint32(value)
        gribapi.grib_set(grib, key, value)


###############################################################################
#
# Constants
#
###############################################################################

# Reference Flag Table 3.3
_RESOLUTION_AND_COMPONENTS_GRID_WINDS_BIT = 3  # NB "bit5", from MSB=1.

# Reference Regulation 92.1.6
_DEFAULT_DEGREES_UNITS = 1.0e-6


###############################################################################
#
# Identification Section 1
#
###############################################################################


def centre(cube, grib):
    # TODO: read centre from cube
    gribapi.grib_set_long(grib, "centre", 74)  # UKMO
    gribapi.grib_set_long(grib, "subCentre", 0)  # exeter is not in the spec


def reference_time(cube, grib):
    # Set the reference time.
    # (analysis, forecast start, verify time, obs time, etc)
    try:
        fp_coord = cube.coord("forecast_period")
    except iris.exceptions.CoordinateNotFoundError:
        fp_coord = None

    if fp_coord is not None:
        rt, rt_meaning, _, _ = _non_missing_forecast_period(cube)
    else:
        rt, rt_meaning, _, _ = _missing_forecast_period(cube)

    gribapi.grib_set_long(grib, "significanceOfReferenceTime", rt_meaning)
    gribapi.grib_set_long(
        grib, "dataDate", "%04d%02d%02d" % (rt.year, rt.month, rt.day))
    gribapi.grib_set_long(
        grib, "dataTime", "%02d%02d" % (rt.hour, rt.minute))

    # TODO: Set the calendar, when we find out what happened to the proposal!
    # http://tinyurl.com/oefqgv6
    # I was sure it was approved for pre-operational use but it's not there.


def identification(cube, grib):
    centre(cube, grib)
    reference_time(cube, grib)

    # operational product, operational test, research product, etc
    # (missing for now)
    gribapi.grib_set_long(grib, "productionStatusOfProcessedData", 255)

    # Code table 1.4
    # analysis, forecast, processed satellite, processed radar,
    if cube.coords('realization'):
        # assume realization will always have 1 and only 1 point
        # as cubes saving to GRIB2 a 2D horizontal slices
        if cube.coord('realization').points[0] != 0:
            gribapi.grib_set_long(grib, "typeOfProcessedData", 4)
        else:
            gribapi.grib_set_long(grib, "typeOfProcessedData", 3)
    else:
        gribapi.grib_set_long(grib, "typeOfProcessedData", 2)


###############################################################################
#
# Grid Definition Section 3
#
###############################################################################


def shape_of_the_earth(cube, grib):
    # assume latlon
    cs = cube.coord(dimensions=[0]).coord_system

    # Initially set shape_of_earth keys to missing (255 for byte, -1 for long).
    gribapi.grib_set_long(grib, "scaleFactorOfRadiusOfSphericalEarth", 255)
    gribapi.grib_set_long(grib, "scaledValueOfRadiusOfSphericalEarth", -1)
    gribapi.grib_set_long(grib, "scaleFactorOfEarthMajorAxis", 255)
    gribapi.grib_set_long(grib, "scaledValueOfEarthMajorAxis", -1)
    gribapi.grib_set_long(grib, "scaleFactorOfEarthMinorAxis", 255)
    gribapi.grib_set_long(grib, "scaledValueOfEarthMinorAxis", -1)

    if isinstance(cs, GeogCS):
        ellipsoid = cs
    else:
        ellipsoid = cs.ellipsoid
        if ellipsoid is None:
            msg = "Could not determine shape of the earth from coord system "\
                  "of horizontal grid."
            raise iris.exceptions.TranslationError(msg)

    # Spherical earth.
    if ellipsoid.inverse_flattening == 0.0:
        gribapi.grib_set_long(grib, "shapeOfTheEarth", 1)
        gribapi.grib_set_long(grib, "scaleFactorOfRadiusOfSphericalEarth", 0)
        gribapi.grib_set_long(grib, "scaledValueOfRadiusOfSphericalEarth",
                              ellipsoid.semi_major_axis)
    # Oblate spheroid earth.
    else:
        gribapi.grib_set_long(grib, "shapeOfTheEarth", 7)
        gribapi.grib_set_long(grib, "scaleFactorOfEarthMajorAxis", 0)
        gribapi.grib_set_long(grib, "scaledValueOfEarthMajorAxis",
                              ellipsoid.semi_major_axis)
        gribapi.grib_set_long(grib, "scaleFactorOfEarthMinorAxis", 0)
        gribapi.grib_set_long(grib, "scaledValueOfEarthMinorAxis",
                              ellipsoid.semi_minor_axis)


def grid_dims(x_coord, y_coord, grib):
    gribapi.grib_set_long(grib, "Ni", x_coord.shape[0])
    gribapi.grib_set_long(grib, "Nj", y_coord.shape[0])


def latlon_first_last(x_coord, y_coord, grib):
    if x_coord.has_bounds() or y_coord.has_bounds():
        warnings.warn("Ignoring xy bounds")

# XXX Pending #1125
#    gribapi.grib_set_double(grib, "latitudeOfFirstGridPointInDegrees",
#                            float(y_coord.points[0]))
#    gribapi.grib_set_double(grib, "latitudeOfLastGridPointInDegrees",
#                            float(y_coord.points[-1]))
#    gribapi.grib_set_double(grib, "longitudeOfFirstGridPointInDegrees",
#                            float(x_coord.points[0]))
#    gribapi.grib_set_double(grib, "longitudeOfLastGridPointInDegrees",
#                            float(x_coord.points[-1]))
# WORKAROUND
    gribapi.grib_set_long(grib, "latitudeOfFirstGridPoint",
                          int(y_coord.points[0]*1000000))
    gribapi.grib_set_long(grib, "latitudeOfLastGridPoint",
                          int(y_coord.points[-1]*1000000))
    gribapi.grib_set_long(grib, "longitudeOfFirstGridPoint",
                          int((x_coord.points[0] % 360)*1000000))
    gribapi.grib_set_long(grib, "longitudeOfLastGridPoint",
                          int((x_coord.points[-1] % 360)*1000000))


def dx_dy(x_coord, y_coord, grib):
    x_step = regular_step(x_coord)
    y_step = regular_step(y_coord)
    gribapi.grib_set(grib, "DxInDegrees", float(abs(x_step)))
    gribapi.grib_set(grib, "DyInDegrees", float(abs(y_step)))


def scanning_mode_flags(x_coord, y_coord, grib):
    gribapi.grib_set_long(grib, "iScansPositively",
                          int(x_coord.points[1] - x_coord.points[0] > 0))
    gribapi.grib_set_long(grib, "jScansPositively",
                          int(y_coord.points[1] - y_coord.points[0] > 0))


def horizontal_grid_common(cube, grib):
    # Grib encoding of the sequences of X and Y points.
    y_coord = cube.coord(dimensions=[0])
    x_coord = cube.coord(dimensions=[1])
    shape_of_the_earth(cube, grib)
    grid_dims(x_coord, y_coord, grib)
    scanning_mode_flags(x_coord, y_coord, grib)


def latlon_points_regular(cube, grib):
    y_coord = cube.coord(dimensions=[0])
    x_coord = cube.coord(dimensions=[1])
    latlon_first_last(x_coord, y_coord, grib)
    dx_dy(x_coord, y_coord, grib)


def latlon_points_irregular(cube, grib):
    y_coord = cube.coord(dimensions=[0])
    x_coord = cube.coord(dimensions=[1])

    # Distinguish between true-north and grid-oriented vectors.
    is_grid_wind = cube.name() in ('x_wind', 'y_wind', 'grid_eastward_wind',
                                   'grid_northward_wind')
    # Encode in bit "5" of 'resolutionAndComponentFlags' (other bits unused).
    component_flags = 0
    if is_grid_wind:
        component_flags |= 2 ** _RESOLUTION_AND_COMPONENTS_GRID_WINDS_BIT
    gribapi.grib_set(grib, 'resolutionAndComponentFlags', component_flags)

    # Record the  X and Y coordinate values.
    # NOTE: there is currently a bug in the gribapi which means that the size
    # of the longitudes array does not equal 'Nj', as it should.
    # See : https://software.ecmwf.int/issues/browse/SUP-1096
    # So, this only works at present if the x and y dimensions are **equal**.
    lon_values = x_coord.points / _DEFAULT_DEGREES_UNITS
    lat_values = y_coord.points / _DEFAULT_DEGREES_UNITS
    gribapi.grib_set_array(grib, 'longitudes',
                           np.array(np.round(lon_values), dtype=np.int64))
    gribapi.grib_set_array(grib, 'latitudes',
                           np.array(np.round(lat_values), dtype=np.int64))


def rotated_pole(cube, grib):
    # Grib encoding of a rotated pole coordinate system.
    cs = cube.coord(dimensions=[0]).coord_system

    if cs.north_pole_grid_longitude != 0.0:
        raise iris.exceptions.TranslationError(
            'Grib save does not yet support Rotated-pole coordinates with '
            'a rotated prime meridian.')
# XXX Pending #1125
#    gribapi.grib_set_double(grib, "latitudeOfSouthernPoleInDegrees",
#                            float(cs.n_pole.latitude))
#    gribapi.grib_set_double(grib, "longitudeOfSouthernPoleInDegrees",
#                            float(cs.n_pole.longitude))
#    gribapi.grib_set_double(grib, "angleOfRotationInDegrees", 0)
# WORKAROUND
    latitude = cs.grid_north_pole_latitude / _DEFAULT_DEGREES_UNITS
    longitude = (((cs.grid_north_pole_longitude + 180) % 360) /
                 _DEFAULT_DEGREES_UNITS)
    gribapi.grib_set(grib, "latitudeOfSouthernPole", - int(round(latitude)))
    gribapi.grib_set(grib, "longitudeOfSouthernPole", int(round(longitude)))
    gribapi.grib_set(grib, "angleOfRotation", 0)


def grid_definition_template_0(cube, grib):
    """
    Set keys within the provided grib message based on
    Grid Definition Template 3.0.

    Template 3.0 is used to represent "latitude/longitude (or equidistant
    cylindrical, or Plate Carree)".
    The coordinates are regularly spaced, true latitudes and longitudes.

    """
    # Constant resolution, aka 'regular' true lat-lon grid.
    gribapi.grib_set_long(grib, "gridDefinitionTemplateNumber", 0)
    horizontal_grid_common(cube, grib)
    latlon_points_regular(cube, grib)


def grid_definition_template_1(cube, grib):
    """
    Set keys within the provided grib message based on
    Grid Definition Template 3.1.

    Template 3.1 is used to represent "rotated latitude/longitude (or
    equidistant cylindrical, or Plate Carree)".
    The coordinates are regularly spaced, rotated latitudes and longitudes.

    """
    # Constant resolution, aka 'regular' rotated lat-lon grid.
    gribapi.grib_set_long(grib, "gridDefinitionTemplateNumber", 1)

    # Record details of the rotated coordinate system.
    rotated_pole(cube, grib)

    # Encode the lat/lon points.
    horizontal_grid_common(cube, grib)
    latlon_points_regular(cube, grib)


def grid_definition_template_5(cube, grib):
    """
    Set keys within the provided grib message based on
    Grid Definition Template 3.5.

    Template 3.5 is used to represent "variable resolution rotated
    latitude/longitude".
    The coordinates are irregularly spaced, rotated latitudes and longitudes.

    """
    # NOTE: we must set Ni=Nj=1 before establishing the template.
    # Without this, setting "gridDefinitionTemplateNumber" = 5 causes an
    # immediate error.
    # See: https://software.ecmwf.int/issues/browse/SUP-1095
    # This is acceptable, as the subsequent call to 'horizontal_grid_common'
    # will set these to the correct horizontal dimensions
    # (by calling 'grid_dims').
    gribapi.grib_set(grib, "Ni", 1)
    gribapi.grib_set(grib, "Nj", 1)
    gribapi.grib_set(grib, "gridDefinitionTemplateNumber", 5)

    # Record details of the rotated coordinate system.
    rotated_pole(cube, grib)
    # Encode the lat/lon points.
    horizontal_grid_common(cube, grib)
    latlon_points_irregular(cube, grib)


def grid_definition_template_12(cube, grib):
    """
    Set keys within the provided grib message based on
    Grid Definition Template 3.12.

    Template 3.12 is used to represent a Transverse Mercator grid.

    """
    gribapi.grib_set(grib, "gridDefinitionTemplateNumber", 12)

    # Retrieve some information from the cube.
    y_coord = cube.coord(dimensions=[0])
    x_coord = cube.coord(dimensions=[1])
    cs = y_coord.coord_system

    # Normalise the coordinate values to centimetres - the resolution
    # used in the GRIB message.
    def points_in_cm(coord):
        points = coord.units.convert(coord.points, 'cm')
        points = np.around(points).astype(int)
        return points
    y_cm = points_in_cm(y_coord)
    x_cm = points_in_cm(x_coord)

    # Set some keys specific to GDT12.
    # Encode the horizontal points.

    # NB. Since we're already in centimetres, our tolerance for
    # discrepancy in the differences is 1.
    def step(points):
        diffs = points[1:] - points[:-1]
        mean_diff = np.mean(diffs).astype(points.dtype)
        if not np.allclose(diffs, mean_diff, atol=1):
            msg = ('Irregular coordinates not supported for transverse '
                   'Mercator.')
            raise iris.exceptions.TranslationError(msg)
        return int(mean_diff)

    gribapi.grib_set(grib, 'Di', abs(step(x_cm)))
    gribapi.grib_set(grib, 'Dj', abs(step(y_cm)))
    horizontal_grid_common(cube, grib)

    # GRIBAPI expects unsigned ints in X1, X2, Y1, Y2 but it should accept
    # signed ints, so work around this.
    # See https://software.ecmwf.int/issues/browse/SUP-1101
    ensure_set_int32_value(grib, 'Y1', int(y_cm[0]))
    ensure_set_int32_value(grib, 'Y2', int(y_cm[-1]))
    ensure_set_int32_value(grib, 'X1', int(x_cm[0]))
    ensure_set_int32_value(grib, 'X2', int(x_cm[-1]))

    # Lat and lon of reference point are measured in millionths of a degree.
    gribapi.grib_set(grib, "latitudeOfReferencePoint",
                     cs.latitude_of_projection_origin / _DEFAULT_DEGREES_UNITS)
    gribapi.grib_set(grib, "longitudeOfReferencePoint",
                     cs.longitude_of_central_meridian / _DEFAULT_DEGREES_UNITS)

    # Convert a value in metres into the closest integer number of
    # centimetres.
    def m_to_cm(value):
        return int(round(value * 100))

    # False easting and false northing are measured in units of (10^-2)m.
    gribapi.grib_set(grib, 'XR', m_to_cm(cs.false_easting))
    gribapi.grib_set(grib, 'YR', m_to_cm(cs.false_northing))

    # GRIBAPI expects a signed int for scaleFactorAtReferencePoint
    # but it should accept a float, so work around this.
    # See https://software.ecmwf.int/issues/browse/SUP-1100
    value = cs.scale_factor_at_central_meridian
    key_type = gribapi.grib_get_native_type(grib,
                                            "scaleFactorAtReferencePoint")
    if key_type is not float:
        value = fixup_float32_as_int32(value)
    gribapi.grib_set(grib, "scaleFactorAtReferencePoint", value)


def grid_definition_section(cube, grib):
    """
    Set keys within the grid definition section of the provided grib message,
    based on the properties of the cube.

    """
    x_coord = cube.coord(dimensions=[1])
    y_coord = cube.coord(dimensions=[0])
    cs = x_coord.coord_system  # N.B. already checked same cs for x and y.
    regular_x_and_y = is_regular(x_coord) and is_regular(y_coord)

    if isinstance(cs, GeogCS):
        if not regular_x_and_y:
            raise iris.exceptions.TranslationError(
                'Saving an irregular latlon grid to GRIB (PDT3.4) is not '
                'yet supported.')

        grid_definition_template_0(cube, grib)

    elif isinstance(cs, RotatedGeogCS):
        # Rotated coordinate system cases.
        # Choose between GDT 3.1 and 3.5 according to coordinate regularity.
        if regular_x_and_y:
            grid_definition_template_1(cube, grib)
        else:
            grid_definition_template_5(cube, grib)

    elif isinstance(cs, TransverseMercator):
        # Transverse Mercator coordinate system (template 3.12).
        grid_definition_template_12(cube, grib)

    else:
        raise ValueError('Grib saving is not supported for coordinate system: '
                         '{:s}'.format(cs))


###############################################################################
#
# Product Definition Section 4
#
###############################################################################

def set_discipline_and_parameter(cube, grib):
    # NOTE: for now, can match by *either* standard_name or long_name.
    # This allows workarounds for data with no identified standard_name.
    grib2_info = gptx.cf_phenom_to_grib2_info(cube.standard_name,
                                              cube.long_name)
    if grib2_info is not None:
        gribapi.grib_set(grib, "discipline", grib2_info.discipline)
        gribapi.grib_set(grib, "parameterCategory", grib2_info.category)
        gribapi.grib_set(grib, "parameterNumber", grib2_info.number)
    else:
        gribapi.grib_set(grib, "discipline", 255)
        gribapi.grib_set(grib, "parameterCategory", 255)
        gribapi.grib_set(grib, "parameterNumber", 255)
        warnings.warn('Unable to determine Grib2 parameter code for cube.\n'
                      'discipline, parameterCategory and parameterNumber '
                      'have been set to "missing".')


def _non_missing_forecast_period(cube):
    # Calculate "model start time" to use as the reference time.
    fp_coord = cube.coord("forecast_period")

    # Convert fp and t to hours so we can subtract to calculate R.
    cf_fp_hrs = fp_coord.units.convert(fp_coord.points[0], 'hours')
    t_coord = cube.coord("time").copy()
    hours_since = iris.unit.Unit("hours since epoch",
                                 calendar=t_coord.units.calendar)
    t_coord.convert_units(hours_since)

    rt_num = t_coord.points[0] - cf_fp_hrs
    rt = hours_since.num2date(rt_num)
    rt_meaning = 1  # "start of forecast"

    # Forecast period
    if fp_coord.units == iris.unit.Unit("hours"):
        grib_time_code = 1
    elif fp_coord.units == iris.unit.Unit("minutes"):
        grib_time_code = 0
    elif fp_coord.units == iris.unit.Unit("seconds"):
        grib_time_code = 13
    else:
        raise iris.exceptions.TranslationError(
            "Unexpected units for 'forecast_period' : %s" % fp_coord.units)

    if not t_coord.has_bounds():
        fp = fp_coord.points[0]
    else:
        if not fp_coord.has_bounds():
            raise iris.exceptions.TranslationError(
                "bounds on 'time' coordinate requires bounds on"
                " 'forecast_period'.")
        fp = fp_coord.bounds[0][0]

    if fp - int(fp):
        warnings.warn("forecast_period encoding problem: "
                      "scaling required.")
    fp = int(fp)

    # Turn negative forecast times into grib negative numbers?
    from iris.fileformats.grib import hindcast_workaround
    if hindcast_workaround and fp < 0:
        msg = "Encoding negative forecast period from {} to ".format(fp)
        fp = 2**31 + abs(fp)
        msg += "{}".format(np.int32(fp))
        warnings.warn(msg)

    return rt, rt_meaning, fp, grib_time_code


def _missing_forecast_period(cube):
    """
    Returns a reference time and significance code together with a forecast
    period and corresponding units type code.

    """
    t_coord = cube.coord("time")

    if cube.coords('forecast_reference_time'):
        # Make copies and convert them to common "hours since" units.
        hours_since = iris.unit.Unit('hours since epoch',
                                     calendar=t_coord.units.calendar)
        frt_coord = cube.coord('forecast_reference_time').copy()
        frt_coord.convert_units(hours_since)
        t_coord = t_coord.copy()
        t_coord.convert_units(hours_since)
        # Extract values.
        t = t_coord.bounds[0, 0] if t_coord.has_bounds() else t_coord.points[0]
        frt = frt_coord.points[0]
        # Calculate GRIB parameters.
        rt = frt_coord.units.num2date(frt)
        rt_meaning = 1  # Forecast reference time.
        fp = t - frt
        integer_fp = int(fp)
        if integer_fp != fp:
            msg = 'Truncating floating point forecast period {} to ' \
                  'integer value {}'
            warnings.warn(msg.format(fp, integer_fp))
        fp = integer_fp
        fp_meaning = 1  # Hours
    else:
        # With no forecast period or forecast reference time set assume a
        # reference time significance of "Observation time" and set the
        # forecast period to 0h.
        t = t_coord.bounds[0, 0] if t_coord.has_bounds() else t_coord.points[0]
        rt = t_coord.units.num2date(t)
        rt_meaning = 3  # Observation time
        fp = 0
        fp_meaning = 1  # Hours

    return rt, rt_meaning, fp, fp_meaning


def set_forecast_time(cube, grib):
    """
    Set the forecast time keys based on the forecast_period coordinate. In
    the absence of a forecast_period and forecast_reference_time,
    the forecast time is set to zero.

    """
    try:
        fp_coord = cube.coord("forecast_period")
    except iris.exceptions.CoordinateNotFoundError:
        fp_coord = None

    if fp_coord is not None:
        _, _, fp, grib_time_code = _non_missing_forecast_period(cube)
    else:
        _, _, fp, grib_time_code = _missing_forecast_period(cube)

    gribapi.grib_set(grib, "indicatorOfUnitOfTimeRange", grib_time_code)
    gribapi.grib_set(grib, "forecastTime", fp)


def set_fixed_surfaces(cube, grib):

    # Look for something we can export
    v_coord = grib_v_code = output_unit = None

    # pressure
    if cube.coords("air_pressure") or cube.coords("pressure"):
        grib_v_code = 100
        output_unit = iris.unit.Unit("Pa")
        v_coord = (cube.coords("air_pressure") or cube.coords("pressure"))[0]

    # altitude
    elif cube.coords("altitude"):
        grib_v_code = 102
        output_unit = iris.unit.Unit("m")
        v_coord = cube.coord("altitude")

    # height
    elif cube.coords("height"):
        grib_v_code = 103
        output_unit = iris.unit.Unit("m")
        v_coord = cube.coord("height")

    elif cube.coords("air_potential_temperature"):
        grib_v_code = 107
        output_unit = iris.unit.Unit('K')
        v_coord = cube.coord("air_potential_temperature")

    # unknown / absent
    else:
        # check for *ANY* height coords at all...
        v_coords = cube.coords(axis='z')
        if v_coords:
            # There are vertical coordinate(s), but we don't understand them...
            v_coords_str = ' ,'.join(["'{}'".format(c.name())
                                      for c in v_coords])
            raise iris.exceptions.TranslationError(
                'The vertical-axis coordinate(s) ({}) '
                'are not recognised or handled.'.format(v_coords_str))

    # What did we find?
    if v_coord is None:
        # No vertical coordinate: record as 'surface' level (levelType=1).
        # NOTE: may *not* be truly correct, but seems to be common practice.
        # Still under investigation :
        # See https://github.com/SciTools/iris/issues/519
        gribapi.grib_set(grib, "typeOfFirstFixedSurface", 1)
        gribapi.grib_set(grib, "scaleFactorOfFirstFixedSurface", 0)
        gribapi.grib_set(grib, "scaledValueOfFirstFixedSurface", 0)
        # Set secondary surface = 'missing'.
        gribapi.grib_set(grib, "typeOfSecondFixedSurface", -1)
        gribapi.grib_set(grib, "scaleFactorOfSecondFixedSurface", 255)
        gribapi.grib_set(grib, "scaledValueOfSecondFixedSurface", -1)
    elif not v_coord.has_bounds():
        # No second surface
        output_v = v_coord.units.convert(v_coord.points[0], output_unit)
        if output_v - abs(output_v):
            warnings.warn("Vertical level encoding problem: scaling required.")
        output_v = int(output_v)

        gribapi.grib_set(grib, "typeOfFirstFixedSurface", grib_v_code)
        gribapi.grib_set(grib, "scaleFactorOfFirstFixedSurface", 0)
        gribapi.grib_set(grib, "scaledValueOfFirstFixedSurface", output_v)
        gribapi.grib_set(grib, "typeOfSecondFixedSurface", -1)
        gribapi.grib_set(grib, "scaleFactorOfSecondFixedSurface", 255)
        gribapi.grib_set(grib, "scaledValueOfSecondFixedSurface", -1)
    else:
        # bounded : set lower+upper surfaces
        output_v = v_coord.units.convert(v_coord.bounds[0], output_unit)
        if output_v[0] - abs(output_v[0]) or output_v[1] - abs(output_v[1]):
            warnings.warn("Vertical level encoding problem: scaling required.")
        gribapi.grib_set(grib, "typeOfFirstFixedSurface", grib_v_code)
        gribapi.grib_set(grib, "typeOfSecondFixedSurface", grib_v_code)
        gribapi.grib_set(grib, "scaleFactorOfFirstFixedSurface", 0)
        gribapi.grib_set(grib, "scaleFactorOfSecondFixedSurface", 0)
        gribapi.grib_set(grib, "scaledValueOfFirstFixedSurface",
                         int(output_v[0]))
        gribapi.grib_set(grib, "scaledValueOfSecondFixedSurface",
                         int(output_v[1]))


def set_time_range(time_coord, grib):
    """
    Set the time range keys in the specified message
    based on the bounds of the provided time coordinate.

    """
    if len(time_coord.points) != 1:
        msg = 'Expected length one time coordinate, got {} points'
        raise ValueError(msg.format(len(time_coord.points)))

    if time_coord.nbounds != 2:
        msg = 'Expected time coordinate with two bounds, got {} bounds'
        raise ValueError(msg.format(time_coord.nbounds))

    # Set type to hours and convert period to this unit.
    gribapi.grib_set(grib, "indicatorOfUnitForTimeRange",
                     _TIME_RANGE_UNITS['hours'])
    hours_since_units = iris.unit.Unit('hours since epoch',
                                       calendar=time_coord.units.calendar)
    start_hours, end_hours = time_coord.units.convert(time_coord.bounds[0],
                                                      hours_since_units)
    # Cast from np.float to Python int. The lengthOfTimeRange key is a
    # 4 byte integer so we cast to highlight truncation of any floating
    # point value. The grib_api will do the cast from float to int, but it
    # cannot handle numpy floats.
    time_range_in_hours = end_hours - start_hours
    integer_hours = int(time_range_in_hours)
    if integer_hours != time_range_in_hours:
        msg = 'Truncating floating point lengthOfTimeRange {} to ' \
              'integer value {}'
        warnings.warn(msg.format(time_range_in_hours, integer_hours))
    gribapi.grib_set(grib, "lengthOfTimeRange", integer_hours)


def set_time_increment(cell_method, grib):
    """
    Set the time increment keys in the specified message
    based on the provided cell method.

    """
    # Type of time increment, e.g incrementing forecast period, incrementing
    # forecast reference time, etc. Set to missing, but we could use the
    # cell method coord to infer a value (see code table 4.11).
    gribapi.grib_set(grib, "typeOfTimeIncrement", 255)

    # Default values for the time increment value and units type.
    inc = 0
    units_type = 255
    # Attempt to determine time increment from cell method intervals string.
    intervals = cell_method.intervals
    if intervals is not None and len(intervals) == 1:
        interval, = intervals
        try:
            inc, units = interval.split()
            inc = float(inc)
            if units in ('hr', 'hour', 'hours'):
                units_type = _TIME_RANGE_UNITS['hours']
            else:
                raise ValueError('Unable to parse units of interval')
        except ValueError:
            # Problem interpreting the interval string.
            inc = 0
            units_type = 255
        else:
            # Cast to int as timeIncrement key is a 4 byte integer.
            integer_inc = int(inc)
            if integer_inc != inc:
                warnings.warn('Truncating floating point timeIncrement {} to '
                              'integer value {}'.format(inc, integer_inc))
            inc = integer_inc

    gribapi.grib_set(grib, "indicatorOfUnitForTimeIncrement", units_type)
    gribapi.grib_set(grib, "timeIncrement", inc)


def _cube_is_time_statistic(cube):
    """
    Test whether we can identify this cube as a statistic over time.

    At present, accept anything whose latest cell method operates over a single
    coordinate that "looks like" a time factor (i.e. some specific names).
    In particular, we recognise the coordinate names defined in
    :py:mod:`iris.coord_categorisation`.

    """
    # The *only* relevant information is in cell_methods, as coordinates or
    # dimensions of aggregation may no longer exist.  So it's not possible to
    # be definitive, but we handle *some* useful cases.
    # In other cases just say "no", which is safe even when not ideal.

    # Identify a single coordinate from the latest cell_method.
    if not cube.cell_methods:
        return False
    latest_coordnames = cube.cell_methods[-1].coord_names
    if len(latest_coordnames) != 1:
        return False
    coord_name = latest_coordnames[0]

    # Define accepted time names, including those from coord_categorisations.
    recognised_time_names = ['time', 'year', 'month', 'day', 'weekday',
                             'season']

    # Accept it if the name is recognised.
    # Currently does *not* recognise related names like 'month_number' or
    # 'years', as that seems potentially unsafe.
    return coord_name in recognised_time_names


def product_definition_template_common(cube, grib):
    """
    Set keys within the provided grib message that are common across
    all of the supported product definition templates.

    """
    set_discipline_and_parameter(cube, grib)

    # Various missing values.
    gribapi.grib_set(grib, "typeOfGeneratingProcess", 255)
    gribapi.grib_set(grib, "backgroundProcess", 255)
    gribapi.grib_set(grib, "generatingProcessIdentifier", 255)

    # Generic time handling.
    set_forecast_time(cube, grib)

    # Handle vertical coords.
    set_fixed_surfaces(cube, grib)


def product_definition_template_0(cube, grib):
    """
    Set keys within the provided grib message based on Product
    Definition Template 4.0.

    Template 4.0 is used to represent an analysis or forecast at
    a horizontal level at a point in time.

    """
    gribapi.grib_set_long(grib, "productDefinitionTemplateNumber", 0)
    product_definition_template_common(cube, grib)


def product_definition_template_8(cube, grib):
    """
    Set keys within the provided grib message based on Product
    Definition Template 4.8.

    Template 4.8 is used to represent an aggregation over a time
    interval.

    """
    gribapi.grib_set(grib, "productDefinitionTemplateNumber", 8)
    product_definition_template_common(cube, grib)

    # Check for time coordinate.
    time_coord = cube.coord('time')

    if len(time_coord.points) != 1:
        msg = 'Expected length one time coordinate, got {} points'
        raise ValueError(msg.format(time_coord.points))

    if time_coord.nbounds != 2:
        msg = 'Expected time coordinate with two bounds, got {} bounds'
        raise ValueError(msg.format(time_coord.nbounds))

    # Check that there is one and only one cell method related to the
    # time coord.
    time_cell_methods = [cell_method for cell_method in cube.cell_methods if
                         'time' in cell_method.coord_names]
    if not time_cell_methods:
        raise ValueError("Expected a cell method with a coordinate name "
                         "of 'time'")
    if len(time_cell_methods) > 1:
        raise ValueError("Cannot handle multiple 'time' cell methods")
    cell_method, = time_cell_methods

    if len(cell_method.coord_names) > 1:
        raise ValueError("Cannot handle multiple coordinate names in "
                         "the time related cell method. Expected ('time',), "
                         "got {!r}".format(cell_method.coord_names))

    # Extract the datetime-like object corresponding to the end of
    # the overall processing interval.
    end = time_coord.units.num2date(time_coord.bounds[0, -1])

    # Set the associated keys for the end of the interval (octets 35-41
    # in section 4).
    gribapi.grib_set(grib, "yearOfEndOfOverallTimeInterval", end.year)
    gribapi.grib_set(grib, "monthOfEndOfOverallTimeInterval", end.month)
    gribapi.grib_set(grib, "dayOfEndOfOverallTimeInterval", end.day)
    gribapi.grib_set(grib, "hourOfEndOfOverallTimeInterval", end.hour)
    gribapi.grib_set(grib, "minuteOfEndOfOverallTimeInterval", end.minute)
    gribapi.grib_set(grib, "secondOfEndOfOverallTimeInterval", end.second)

    # Only one time range specification. If there were a series of aggregations
    # (e.g. the mean of an accumulation) one might set this to a higher value,
    # but we currently only handle a single time related cell method.
    gribapi.grib_set(grib, "numberOfTimeRange", 1)
    gribapi.grib_set(grib, "numberOfMissingInStatisticalProcess", 0)

    # Type of statistical process (see code table 4.10)
    statistic_type = _STATISTIC_TYPE_NAMES.get(cell_method.method, 255)
    gribapi.grib_set(grib, "typeOfStatisticalProcessing", statistic_type)

    # Period over which statistical processing is performed.
    set_time_range(time_coord, grib)

    # Time increment i.e. interval of cell method (if any)
    set_time_increment(cell_method, grib)


def product_definition_section(cube, grib):
    """
    Set keys within the product definition section of the provided
    grib message based on the properties of the cube.

    """
    if not cube.coord("time").has_bounds():
        # forecast (template 4.0)
        product_definition_template_0(cube, grib)
    elif _cube_is_time_statistic(cube):
        # time processed (template 4.8)
        try:
            product_definition_template_8(cube, grib)
        except ValueError as e:
            raise ValueError('Saving to GRIB2 failed: the cube is not suitable'
                             ' for saving as a time processed statistic GRIB'
                             ' message. {}'.format(e))
    else:
        # Don't know how to handle this kind of data
        msg = 'A suitable product template could not be deduced'
        raise iris.exceptions.TranslationError(msg)


###############################################################################
#
# Data Representation Section 5
#
###############################################################################

def data_section(cube, grib):
    # Masked data?
    if isinstance(cube.data, ma.core.MaskedArray):
        # What missing value shall we use?
        if not np.isnan(cube.data.fill_value):
            # Use the data's fill value.
            fill_value = float(cube.data.fill_value)
        else:
            # We can't use the data's fill value if it's NaN,
            # the GRIB API doesn't like it.
            # Calculate an MDI outside the data range.
            min, max = cube.data.min(), cube.data.max()
            fill_value = min - (max - min) * 0.1
        # Prepare the unmaksed data array, using fill_value as the MDI.
        data = cube.data.filled(fill_value)
    else:
        fill_value = None
        data = cube.data

    # units scaling
    grib2_info = gptx.cf_phenom_to_grib2_info(cube.standard_name,
                                              cube.long_name)
    if grib2_info is None:
        # for now, just allow this
        warnings.warn('Unable to determine Grib2 parameter code for cube.\n'
                      'Message data may not be correctly scaled.')
    else:
        if cube.units != grib2_info.units:
            data = cube.units.convert(data, grib2_info.units)
            if fill_value is not None:
                fill_value = cube.units.convert(fill_value, grib2_info.units)

    if fill_value is None:
        # Disable missing values in the grib message.
        gribapi.grib_set(grib, "bitmapPresent", 0)
    else:
        # Enable missing values in the grib message.
        gribapi.grib_set(grib, "bitmapPresent", 1)
        gribapi.grib_set_double(grib, "missingValue", fill_value)
    gribapi.grib_set_double_array(grib, "values", data.flatten())

    # todo: check packing accuracy?
#    print("packingError", gribapi.getb_get_double(grib, "packingError"))


###############################################################################

def gribbability_check(cube):
    "We always need the following things for grib saving."

    # GeogCS exists?
    cs0 = cube.coord(dimensions=[0]).coord_system
    cs1 = cube.coord(dimensions=[1]).coord_system
    if cs0 is None or cs1 is None:
        raise iris.exceptions.TranslationError("CoordSystem not present")
    if cs0 != cs1:
        raise iris.exceptions.TranslationError("Inconsistent CoordSystems")

    # Time period exists?
    if not cube.coords("time"):
        raise iris.exceptions.TranslationError("time coord not found")


def run(cube, grib):
    """
    Set the keys of the grib message based on the contents of the cube.

    Args:

    * cube:
        An instance of :class:`iris.cube.Cube`.

    * grib_message_id:
        ID of a grib message in memory. This is typically the return value of
        :func:`gribapi.grib_new_from_samples`.

    """
    gribbability_check(cube)

    # Section 1 - Identification Section.
    identification(cube, grib)

    # Section 3 - Grid Definition Section (Grid Definition Template)
    grid_definition_section(cube, grib)

    # Section 4 - Product Definition Section (Product Definition Template)
    product_definition_section(cube, grib)

    # Section 5 - Data Representation Section (Data Representation Template)
    data_section(cube, grib)
