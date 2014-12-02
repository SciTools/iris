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
from iris.fileformats.rules import is_regular, regular_step
from iris.fileformats.grib import grib_phenom_translation as gptx
from iris.fileformats.grib._load_convert import (_STATISTIC_TYPE_NAMES,
                                                 _TIME_RANGE_UNITS)

# Invert code tables from :mod:`iris.fileformats.grib._load_convert`.
_STATISTIC_TYPE_NAMES = {val: key for key, val in
                         _STATISTIC_TYPE_NAMES.items()}
_TIME_RANGE_UNITS = {val: key for key, val in _TIME_RANGE_UNITS.items()}


def gribbability_check(cube):
    "We always need the following things for grib saving."

    # GeogCS exists?
    cs0 = cube.coord(dimensions=[0]).coord_system
    cs1 = cube.coord(dimensions=[1]).coord_system
    if cs0 is None or cs1 is None:
        raise iris.exceptions.TranslationError("CoordSystem not present")
    if cs0 != cs1:
        raise iris.exceptions.TranslationError("Inconsistent CoordSystems")

    # Regular?
    y_coord = cube.coord(dimensions=[0])
    x_coord = cube.coord(dimensions=[1])
    if not is_regular(x_coord) or not is_regular(y_coord):
        raise iris.exceptions.TranslationError(
            "Cannot save irregular grids to grib")

    # Time period exists?
    if not cube.coords("time"):
        raise iris.exceptions.TranslationError("time coord not found")


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
    # analysis, forecast, processed satellite, processed radar,
    # (analysis and forecast products for now)
    gribapi.grib_set_long(grib, "typeOfProcessedData", 2)


###############################################################################
#
# Grid Definition Section 3
#
###############################################################################

def shape_of_the_earth(cube, grib):

    # assume latlon
    cs = cube.coord(dimensions=[0]).coord_system

    # Turn them all missing to start with (255 for byte, -1 for long)
    gribapi.grib_set_long(grib, "scaleFactorOfRadiusOfSphericalEarth", 255)
    gribapi.grib_set_long(grib, "scaledValueOfRadiusOfSphericalEarth", -1)
    gribapi.grib_set_long(grib, "scaleFactorOfEarthMajorAxis", 255)
    gribapi.grib_set_long(grib, "scaledValueOfEarthMajorAxis", -1)
    gribapi.grib_set_long(grib, "scaleFactorOfEarthMinorAxis", 255)
    gribapi.grib_set_long(grib, "scaledValueOfEarthMinorAxis", -1)

    ellipsoid = cs
    if isinstance(cs, iris.coord_systems.RotatedGeogCS):
        ellipsoid = cs.ellipsoid

    if ellipsoid.inverse_flattening == 0.0:
        gribapi.grib_set_long(grib, "shapeOfTheEarth", 1)
        gribapi.grib_set_long(grib, "scaleFactorOfRadiusOfSphericalEarth", 0)
        gribapi.grib_set_long(grib, "scaledValueOfRadiusOfSphericalEarth",
                              ellipsoid.semi_major_axis)
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
    # TODO: THIS USED BE "Dx" and "Dy"!!! DID THE API CHANGE AGAIN???
    gribapi.grib_set_double(grib, "DxInDegrees", float(abs(x_step)))
    gribapi.grib_set_double(grib, "DyInDegrees", float(abs(y_step)))


def scanning_mode_flags(x_coord, y_coord, grib):
    gribapi.grib_set_long(grib, "iScansPositively",
                          int(x_coord.points[1] - x_coord.points[0] > 0))
    gribapi.grib_set_long(grib, "jScansPositively",
                          int(y_coord.points[1] - y_coord.points[0] > 0))


def latlon_common(cube, grib):
    y_coord = cube.coord(dimensions=[0])
    x_coord = cube.coord(dimensions=[1])
    shape_of_the_earth(cube, grib)
    grid_dims(x_coord, y_coord, grib)
    latlon_first_last(x_coord, y_coord, grib)
    dx_dy(x_coord, y_coord, grib)
    scanning_mode_flags(x_coord, y_coord, grib)


def rotated_pole(cube, grib):
    cs = cube.coord(dimensions=[0]).coord_system

# XXX Pending #1125
#    gribapi.grib_set_double(grib, "latitudeOfSouthernPoleInDegrees",
#                            float(cs.n_pole.latitude))
#    gribapi.grib_set_double(grib, "longitudeOfSouthernPoleInDegrees",
#                            float(cs.n_pole.longitude))
#    gribapi.grib_set_double(grib, "angleOfRotationInDegrees", 0)
# WORKAROUND
    latitude = -int(cs.grid_north_pole_latitude*1000000)
    longitude = int(((cs.grid_north_pole_longitude+180) % 360)*1000000)
    gribapi.grib_set_long(grib, "latitudeOfSouthernPole", latitude)
    gribapi.grib_set_long(grib, "longitudeOfSouthernPole", longitude)
    gribapi.grib_set_long(grib, "angleOfRotation", 0)


def grid_template(cube, grib):
    cs = cube.coord(dimensions=[0]).coord_system
    if isinstance(cs, iris.coord_systems.GeogCS):
        # template 3.0
        gribapi.grib_set_long(grib, "gridDefinitionTemplateNumber", 0)
        latlon_common(cube, grib)

    # rotated
    elif isinstance(cs, iris.coord_systems.RotatedGeogCS):
        # template 3.1
        gribapi.grib_set_long(grib, "gridDefinitionTemplateNumber", 1)
        latlon_common(cube, grib)
        rotated_pole(cube, grib)
    else:
        raise ValueError("Currently unhandled CoordSystem: %s" % cs)


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
    # We have no way of knowing the CF forecast reference time.
    # Set GRIB reference time to "verifying time of forecast",
    # and the forecast period to 0h.
    t_coord = cube.coord("time")
    t = t_coord.bounds[0, 0] if t_coord.has_bounds() else t_coord.points[0]
    rt = t_coord.units.num2date(t)
    rt_meaning = 2  # "verification time of forecast"

    fp = 0
    fp_meaning = 1  # hours

    return rt, rt_meaning, fp, fp_meaning


def set_forecast_time(cube, grib):
    """
    Set the forecast time keys based on the forecast_period coordinate. In
    the absence of a forecast_period, the forecast time is set to zero.

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
        warnings.warn('Truncating floating point lengthOfTimeRange {} to integer '
                      'value {}'.format(time_range_in_hours, integer_hours))
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
    grid_template(cube, grib)

    # Section 4 - Product Definition Section (Product Definition Template)
    product_definition_section(cube, grib)

    # Section 5 - Data Representation Section (Data Representation Template)
    data_section(cube, grib)
