# (C) British Crown Copyright 2013 - 2017, Met Office
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

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Historically this was auto-generated from
# SciTools/iris-code-generators:tools/gen_rules.py

import warnings

from cf_units import CALENDAR_GREGORIAN, Unit
import numpy as np

from iris.aux_factory import HybridPressureFactory
from iris.coords import AuxCoord, CellMethod, DimCoord
from iris.exceptions import TranslationError
from iris.fileformats.rules import (ConversionMetadata, Factory, Reference,
                                    ReferenceTarget)


def grib1_convert(grib):
    """
    Converts a GRIB1 message into the corresponding items of Cube metadata.

    Args:

    * grib:
        A :class:`~iris.fileformats.grib.GribWrapper` object.

    Returns:
        A :class:`iris.fileformats.rules.ConversionMetadata` object.

    """
    if grib.edition != 1:
        emsg = 'GRIB edition {} is not supported by {!r}.'
        raise TranslationError(emsg.format(grib.edition,
                                           type(grib).__name__))

    factories = []
    references = []
    standard_name = None
    long_name = None
    units = None
    attributes = {}
    cell_methods = []
    dim_coords_and_dims = []
    aux_coords_and_dims = []

    if \
            (grib.gridType=="reduced_gg"):
        aux_coords_and_dims.append((AuxCoord(grib._y_points, grib._y_coord_name, units='degrees', coord_system=grib._coord_system), 0))
        aux_coords_and_dims.append((AuxCoord(grib._x_points, grib._x_coord_name, units='degrees', coord_system=grib._coord_system), 0))

    if \
            (grib.gridType=="regular_ll") and \
            (grib.jPointsAreConsecutive == 0):
        dim_coords_and_dims.append((DimCoord(grib._y_points, grib._y_coord_name, units='degrees', coord_system=grib._coord_system), 0))
        dim_coords_and_dims.append((DimCoord(grib._x_points, grib._x_coord_name, units='degrees', coord_system=grib._coord_system, circular=grib._x_circular), 1))

    if \
            (grib.gridType=="regular_ll") and \
            (grib.jPointsAreConsecutive == 1):
        dim_coords_and_dims.append((DimCoord(grib._y_points, grib._y_coord_name, units='degrees', coord_system=grib._coord_system), 1))
        dim_coords_and_dims.append((DimCoord(grib._x_points, grib._x_coord_name, units='degrees', coord_system=grib._coord_system, circular=grib._x_circular), 0))

    if \
            (grib.gridType=="regular_gg") and \
            (grib.jPointsAreConsecutive == 0):
        dim_coords_and_dims.append((DimCoord(grib._y_points, grib._y_coord_name, units='degrees', coord_system=grib._coord_system), 0))
        dim_coords_and_dims.append((DimCoord(grib._x_points, grib._x_coord_name, units='degrees', coord_system=grib._coord_system, circular=grib._x_circular), 1))

    if \
            (grib.gridType=="regular_gg") and \
            (grib.jPointsAreConsecutive == 1):
        dim_coords_and_dims.append((DimCoord(grib._y_points, grib._y_coord_name, units='degrees', coord_system=grib._coord_system), 1))
        dim_coords_and_dims.append((DimCoord(grib._x_points, grib._x_coord_name, units='degrees', coord_system=grib._coord_system, circular=grib._x_circular), 0))

    if \
            (grib.gridType=="rotated_ll") and \
            (grib.jPointsAreConsecutive == 0):
        dim_coords_and_dims.append((DimCoord(grib._y_points, grib._y_coord_name, units='degrees', coord_system=grib._coord_system), 0))
        dim_coords_and_dims.append((DimCoord(grib._x_points, grib._x_coord_name, units='degrees', coord_system=grib._coord_system, circular=grib._x_circular), 1))

    if \
            (grib.gridType=="rotated_ll") and \
            (grib.jPointsAreConsecutive == 1):
        dim_coords_and_dims.append((DimCoord(grib._y_points, grib._y_coord_name, units='degrees', coord_system=grib._coord_system), 1))
        dim_coords_and_dims.append((DimCoord(grib._x_points, grib._x_coord_name, units='degrees', coord_system=grib._coord_system, circular=grib._x_circular), 0))

    if grib.gridType in ["polar_stereographic", "lambert"]:
        dim_coords_and_dims.append((DimCoord(grib._y_points, grib._y_coord_name, units="m", coord_system=grib._coord_system), 0))
        dim_coords_and_dims.append((DimCoord(grib._x_points, grib._x_coord_name, units="m", coord_system=grib._coord_system), 1))

    if \
            (grib.table2Version < 128) and \
            (grib.indicatorOfParameter == 11) and \
            (grib._cf_data is None):
        standard_name = "air_temperature"
        units = "kelvin"

    if \
            (grib.table2Version < 128) and \
            (grib.indicatorOfParameter == 33) and \
            (grib._cf_data is None):
        standard_name = "x_wind"
        units = "m s-1"

    if \
            (grib.table2Version < 128) and \
            (grib.indicatorOfParameter == 34) and \
            (grib._cf_data is None):
        standard_name = "y_wind"
        units = "m s-1"

    if \
            (grib._cf_data is not None):
        standard_name = grib._cf_data.standard_name
        long_name = grib._cf_data.standard_name or grib._cf_data.long_name
        units = grib._cf_data.units

    if \
            (grib.table2Version >= 128) and \
            (grib._cf_data is None):
        long_name = "UNKNOWN LOCAL PARAM " + str(grib.indicatorOfParameter) + "." + str(grib.table2Version)
        units = "???"

    if \
            (grib.table2Version == 1) and \
            (grib.indicatorOfParameter >= 128):
        long_name = "UNKNOWN LOCAL PARAM " + str(grib.indicatorOfParameter) + "." + str(grib.table2Version)
        units = "???"

    if \
            (grib._phenomenonDateTime != -1.0):
        aux_coords_and_dims.append((DimCoord(points=grib.startStep, standard_name='forecast_period', units=grib._forecastTimeUnit), None))
        aux_coords_and_dims.append((DimCoord(points=grib.phenomenon_points('hours'), standard_name='time', units=Unit('hours since epoch', CALENDAR_GREGORIAN)), None))

    def add_bounded_time_coords(aux_coords_and_dims, grib):
        t_bounds = grib.phenomenon_bounds('hours')
        period = Unit('hours').convert(t_bounds[1] - t_bounds[0],
                                       grib._forecastTimeUnit)
        aux_coords_and_dims.append((
            DimCoord(standard_name='forecast_period',
                     units=grib._forecastTimeUnit,
                     points=grib._forecastTime + 0.5 * period,
                     bounds=[grib._forecastTime, grib._forecastTime + period]),
            None))
        aux_coords_and_dims.append((
            DimCoord(standard_name='time',
                     units=Unit('hours since epoch', CALENDAR_GREGORIAN),
                     points=0.5 * (t_bounds[0] + t_bounds[1]),
                     bounds=t_bounds),
            None))

    if \
            (grib.timeRangeIndicator == 2):
        add_bounded_time_coords(aux_coords_and_dims, grib)

    if \
            (grib.timeRangeIndicator == 3):
        add_bounded_time_coords(aux_coords_and_dims, grib)
        cell_methods.append(CellMethod("mean", coords="time"))

    if \
            (grib.timeRangeIndicator == 4):
        add_bounded_time_coords(aux_coords_and_dims, grib)
        cell_methods.append(CellMethod("sum", coords="time"))

    if \
            (grib.timeRangeIndicator == 5):
        add_bounded_time_coords(aux_coords_and_dims, grib)
        cell_methods.append(CellMethod("_difference", coords="time"))

    if \
            (grib.timeRangeIndicator == 51):
        add_bounded_time_coords(aux_coords_and_dims, grib)
        cell_methods.append(CellMethod("mean", coords="time"))

    if \
            (grib.timeRangeIndicator == 113):
        add_bounded_time_coords(aux_coords_and_dims, grib)
        cell_methods.append(CellMethod("mean", coords="time"))

    if \
            (grib.timeRangeIndicator == 114):
        add_bounded_time_coords(aux_coords_and_dims, grib)
        cell_methods.append(CellMethod("sum", coords="time"))

    if \
            (grib.timeRangeIndicator == 115):
        add_bounded_time_coords(aux_coords_and_dims, grib)
        cell_methods.append(CellMethod("mean", coords="time"))

    if \
            (grib.timeRangeIndicator == 116):
        add_bounded_time_coords(aux_coords_and_dims, grib)
        cell_methods.append(CellMethod("sum", coords="time"))

    if \
            (grib.timeRangeIndicator == 117):
        add_bounded_time_coords(aux_coords_and_dims, grib)
        cell_methods.append(CellMethod("mean", coords="time"))

    if \
            (grib.timeRangeIndicator == 118):
        add_bounded_time_coords(aux_coords_and_dims, grib)
        cell_methods.append(CellMethod("_covariance", coords="time"))

    if \
            (grib.timeRangeIndicator == 123):
        add_bounded_time_coords(aux_coords_and_dims, grib)
        cell_methods.append(CellMethod("mean", coords="time"))

    if \
            (grib.timeRangeIndicator == 124):
        add_bounded_time_coords(aux_coords_and_dims, grib)
        cell_methods.append(CellMethod("sum", coords="time"))

    if \
            (grib.timeRangeIndicator == 125):
        add_bounded_time_coords(aux_coords_and_dims, grib)
        cell_methods.append(CellMethod("standard_deviation", coords="time"))

    if \
            (grib.levelType == 'pl'):
        aux_coords_and_dims.append((DimCoord(points=grib.level,  long_name="pressure", units="hPa"), None))

    if \
            (grib.levelType == 'sfc'):

            if (grib._cf_data is not None) and \
            (grib._cf_data.set_height is not None):
                aux_coords_and_dims.append((DimCoord(points=grib._cf_data.set_height,  long_name="height", units="m", attributes={'positive':'up'}), None))
            elif grib.typeOfLevel == 'heightAboveGround': # required for NCAR
                aux_coords_and_dims.append((DimCoord(points=grib.level,  long_name="height", units="m", attributes={'positive':'up'}), None))

    if \
            (grib.levelType == 'ml') and \
            (hasattr(grib, 'pv')):
        aux_coords_and_dims.append((AuxCoord(grib.level, standard_name='model_level_number', attributes={'positive': 'up'}), None))
        aux_coords_and_dims.append((DimCoord(grib.pv[grib.level], long_name='level_pressure', units='Pa'), None))
        aux_coords_and_dims.append((AuxCoord(grib.pv[grib.numberOfCoordinatesValues//2 + grib.level], long_name='sigma'), None))
        factories.append(Factory(HybridPressureFactory, [{'long_name': 'level_pressure'}, {'long_name': 'sigma'}, Reference('surface_pressure')]))

    if grib._originatingCentre != 'unknown':
        aux_coords_and_dims.append((AuxCoord(points=grib._originatingCentre, long_name='originating_centre', units='no_unit'), None))

    return ConversionMetadata(factories, references, standard_name, long_name,
                              units, attributes, cell_methods,
                              dim_coords_and_dims, aux_coords_and_dims)
