# (C) British Crown Copyright 2013, Met Office
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

import warnings

import numpy as np

from iris.aux_factory import HybridHeightFactory
from iris.coords import AuxCoord, CellMethod, DimCoord
from iris.fileformats.mosig_cf_map import MOSIG_STASH_TO_CF
from iris.fileformats.rules import Factory, Reference, ReferenceTarget
from iris.fileformats.um_cf_map import LBFC_TO_CF, STASH_TO_CF
from iris.unit import Unit
import iris.fileformats.pp
import iris.unit


def convert(cube, grib):
    cm = cube
    factories = []
    references = []

    if \
            (grib.gridType=="regular_ll") and \
            (grib.jPointsAreConsecutive == 0):
        cube.add_dim_coord(DimCoord(grib._y_points, grib._y_coord_name, units='degrees', coord_system=grib._coord_system), 0)
        cube.add_dim_coord(DimCoord(grib._x_points, grib._x_coord_name, units='degrees', coord_system=grib._coord_system, circular=grib._x_circular), 1)

    if \
            (grib.gridType=="regular_ll") and \
            (grib.jPointsAreConsecutive == 1):
        cube.add_dim_coord(DimCoord(grib._y_points, grib._y_coord_name, units='degrees', coord_system=grib._coord_system), 1)
        cube.add_dim_coord(DimCoord(grib._x_points, grib._x_coord_name, units='degrees', coord_system=grib._coord_system, circular=grib._x_circular), 0)

    if \
            (grib.gridType=="rotated_ll") and \
            (grib.jPointsAreConsecutive == 0):
        cube.add_dim_coord(DimCoord(grib._y_points, grib._y_coord_name, units='degrees', coord_system=grib._coord_system), 0)
        cube.add_dim_coord(DimCoord(grib._x_points, grib._x_coord_name, units='degrees', coord_system=grib._coord_system, circular=grib._x_circular), 1)

    if \
            (grib.gridType=="rotated_ll") and \
            (grib.jPointsAreConsecutive == 1):
        cube.add_dim_coord(DimCoord(grib._y_points, grib._y_coord_name, units='degrees', coord_system=grib._coord_system), 1)
        cube.add_dim_coord(DimCoord(grib._x_points, grib._x_coord_name, units='degrees', coord_system=grib._coord_system, circular=grib._x_circular), 0)

    if grib.gridType=="polar_stereographic":
        cube.add_dim_coord(DimCoord(grib._y_points, grib._y_coord_name, units=None, coord_system=grib._coord_system), 0)
        cube.add_dim_coord(DimCoord(grib._x_points, grib._x_coord_name, units=None, coord_system=grib._coord_system), 1)

    if \
            (grib.edition == 1) and \
            (grib.table2Version < 128) and \
            (grib.indicatorOfParameter == 11) and \
            (grib._cf_data is None):
        cube.rename("air_temperature")
        units = "kelvin"
        try:
            setattr(cube, 'units', units)
        except ValueError:
            msg = 'Ignoring PP invalid units {!r}'.format(units)
            warnings.warn(msg)
            cube.attributes['invalid_units'] = units
            cube.units = iris.unit._UNKNOWN_UNIT_STRING

    if \
            (grib.edition == 1) and \
            (grib.table2Version < 128) and \
            (grib.indicatorOfParameter == 33) and \
            (grib._cf_data is None):
        cube.rename("x_wind")
        units = "m s-1"
        try:
            setattr(cube, 'units', units)
        except ValueError:
            msg = 'Ignoring PP invalid units {!r}'.format(units)
            warnings.warn(msg)
            cube.attributes['invalid_units'] = units
            cube.units = iris.unit._UNKNOWN_UNIT_STRING

    if \
            (grib.edition == 1) and \
            (grib.table2Version < 128) and \
            (grib.indicatorOfParameter == 34) and \
            (grib._cf_data is None):
        cube.rename("y_wind")
        units = "m s-1"
        try:
            setattr(cube, 'units', units)
        except ValueError:
            msg = 'Ignoring PP invalid units {!r}'.format(units)
            warnings.warn(msg)
            cube.attributes['invalid_units'] = units
            cube.units = iris.unit._UNKNOWN_UNIT_STRING

    if \
            (grib.edition == 1) and \
            (grib._cf_data is not None):
        cube.rename(grib._cf_data.standard_name)
        cube.long_name = grib._cf_data.standard_name or grib._cf_data.long_name
        units = grib._cf_data.units
        try:
            setattr(cube, 'units', units)
        except ValueError:
            msg = 'Ignoring PP invalid units {!r}'.format(units)
            warnings.warn(msg)
            cube.attributes['invalid_units'] = units
            cube.units = iris.unit._UNKNOWN_UNIT_STRING

    if \
            (grib.edition == 1) and \
            (grib.table2Version >= 128) and \
            (grib._cf_data is None):
        cube.long_name = "UNKNOWN LOCAL PARAM " + str(grib.indicatorOfParameter) + "." + str(grib.table2Version)
        units = "???"
        try:
            setattr(cube, 'units', units)
        except ValueError:
            msg = 'Ignoring PP invalid units {!r}'.format(units)
            warnings.warn(msg)
            cube.attributes['invalid_units'] = units
            cube.units = iris.unit._UNKNOWN_UNIT_STRING

    if \
            (grib.edition == 1) and \
            (grib.table2Version == 1) and \
            (grib.indicatorOfParameter >= 128):
        cube.long_name = "UNKNOWN LOCAL PARAM " + str(grib.indicatorOfParameter) + "." + str(grib.table2Version)
        units = "???"
        try:
            setattr(cube, 'units', units)
        except ValueError:
            msg = 'Ignoring PP invalid units {!r}'.format(units)
            warnings.warn(msg)
            cube.attributes['invalid_units'] = units
            cube.units = iris.unit._UNKNOWN_UNIT_STRING

    if \
            (grib.edition == 2) and \
            (grib._cf_data is not None):
        cube.rename(grib._cf_data.standard_name)
        cube.long_name = grib._cf_data.long_name
        units = grib._cf_data.units
        try:
            setattr(cube, 'units', units)
        except ValueError:
            msg = 'Ignoring PP invalid units {!r}'.format(units)
            warnings.warn(msg)
            cube.attributes['invalid_units'] = units
            cube.units = iris.unit._UNKNOWN_UNIT_STRING

    if \
            (grib.edition == 1) and \
            (grib._phenomenonDateTime != -1.0):
        cube.add_aux_coord(DimCoord(points=grib.startStep, standard_name='forecast_period', units=grib._forecastTimeUnit))
        cube.add_aux_coord(DimCoord(points=grib.phenomenon_points('hours'), standard_name='time', units=Unit('hours since epoch', iris.unit.CALENDAR_GREGORIAN)))

    if \
            (grib.edition == 1) and \
            (grib.timeRangeIndicator == 3):
        cube.add_aux_coord(DimCoord(points=grib.startStep, standard_name='forecast_period', units=grib._forecastTimeUnit))
        cube.add_aux_coord(DimCoord(points=np.mean(grib.phenomenon_bounds('hours')), bounds=grib.phenomenon_bounds('hours'),  standard_name='time', units=Unit('hours since epoch', iris.unit.CALENDAR_GREGORIAN)))
        cube.add_cell_method(CellMethod("mean", cm.coord("time")))

    if \
            (grib.edition == 1) and \
            (grib.timeRangeIndicator == 4):
        cube.add_aux_coord(DimCoord(points=grib.startStep, standard_name='forecast_period', units=grib._forecastTimeUnit))
        cube.add_aux_coord(DimCoord(points=np.mean(grib.phenomenon_bounds('hours')), bounds=grib.phenomenon_bounds('hours'),  standard_name='time', units=Unit('hours since epoch', iris.unit.CALENDAR_GREGORIAN)))
        cube.add_cell_method(CellMethod("sum", cm.coord("time")))

    if \
            (grib.edition == 1) and \
            (grib.timeRangeIndicator == 5):
        cube.add_aux_coord(DimCoord(points=grib.startStep, standard_name='forecast_period', units=grib._forecastTimeUnit))
        cube.add_aux_coord(DimCoord(points=np.mean(grib.phenomenon_bounds('hours')), bounds=grib.phenomenon_bounds('hours'),  standard_name='time', units=Unit('hours since epoch', iris.unit.CALENDAR_GREGORIAN)))
        cube.add_cell_method(CellMethod("_difference", cm.coord("time")))

    if \
            (grib.edition == 1) and \
            (grib.timeRangeIndicator == 51):
        cube.add_aux_coord(DimCoord(points=grib.startStep, standard_name='forecast_period', units=grib._forecastTimeUnit))
        cube.add_aux_coord(DimCoord(points=np.mean(grib.phenomenon_bounds('hours')), bounds=grib.phenomenon_bounds('hours'),  standard_name='time', units=Unit('hours since epoch', iris.unit.CALENDAR_GREGORIAN)))
        cube.add_cell_method(CellMethod("mean", cm.coord("time")))

    if \
            (grib.edition == 1) and \
            (grib.timeRangeIndicator == 113):
        cube.add_aux_coord(DimCoord(points=grib.startStep, standard_name='forecast_period', units=grib._forecastTimeUnit))
        cube.add_aux_coord(DimCoord(points=np.mean(grib.phenomenon_bounds('hours')), bounds=grib.phenomenon_bounds('hours'),  standard_name='time', units=Unit('hours since epoch', iris.unit.CALENDAR_GREGORIAN)))
        cube.add_cell_method(CellMethod("mean", cm.coord("time")))

    if \
            (grib.edition == 1) and \
            (grib.timeRangeIndicator == 114):
        cube.add_aux_coord(DimCoord(points=grib.startStep, standard_name='forecast_period', units=grib._forecastTimeUnit))
        cube.add_aux_coord(DimCoord(points=np.mean(grib.phenomenon_bounds('hours')), bounds=grib.phenomenon_bounds('hours'),  standard_name='time', units=Unit('hours since epoch', iris.unit.CALENDAR_GREGORIAN)))
        cube.add_cell_method(CellMethod("sum", cm.coord("time")))

    if \
            (grib.edition == 1) and \
            (grib.timeRangeIndicator == 115):
        cube.add_aux_coord(DimCoord(points=grib.startStep, standard_name='forecast_period', units=grib._forecastTimeUnit))
        cube.add_aux_coord(DimCoord(points=np.mean(grib.phenomenon_bounds('hours')), bounds=grib.phenomenon_bounds('hours'),  standard_name='time', units=Unit('hours since epoch', iris.unit.CALENDAR_GREGORIAN)))
        cube.add_cell_method(CellMethod("mean", cm.coord("time")))

    if \
            (grib.edition == 1) and \
            (grib.timeRangeIndicator == 116):
        cube.add_aux_coord(DimCoord(points=grib.startStep, standard_name='forecast_period', units=grib._forecastTimeUnit))
        cube.add_aux_coord(DimCoord(points=np.mean(grib.phenomenon_bounds('hours')), bounds=grib.phenomenon_bounds('hours'),  standard_name='time', units=Unit('hours since epoch', iris.unit.CALENDAR_GREGORIAN)))
        cube.add_cell_method(CellMethod("sum", cm.coord("time")))

    if \
            (grib.edition == 1) and \
            (grib.timeRangeIndicator == 117):
        cube.add_aux_coord(DimCoord(points=grib.startStep, standard_name='forecast_period', units=grib._forecastTimeUnit))
        cube.add_aux_coord(DimCoord(points=np.mean(grib.phenomenon_bounds('hours')), bounds=grib.phenomenon_bounds('hours'),  standard_name='time', units=Unit('hours since epoch', iris.unit.CALENDAR_GREGORIAN)))
        cube.add_cell_method(CellMethod("mean", cm.coord("time")))

    if \
            (grib.edition == 1) and \
            (grib.timeRangeIndicator == 118):
        cube.add_aux_coord(DimCoord(points=grib.startStep, standard_name='forecast_period', units=grib._forecastTimeUnit))
        cube.add_aux_coord(DimCoord(points=np.mean(grib.phenomenon_bounds('hours')), bounds=grib.phenomenon_bounds('hours'),  standard_name='time', units=Unit('hours since epoch', iris.unit.CALENDAR_GREGORIAN)))
        cube.add_cell_method(CellMethod("_covariance", cm.coord("time")))

    if \
            (grib.edition == 1) and \
            (grib.timeRangeIndicator == 123):
        cube.add_aux_coord(DimCoord(points=grib.startStep, standard_name='forecast_period', units=grib._forecastTimeUnit))
        cube.add_aux_coord(DimCoord(points=np.mean(grib.phenomenon_bounds('hours')), bounds=grib.phenomenon_bounds('hours'),  standard_name='time', units=Unit('hours since epoch', iris.unit.CALENDAR_GREGORIAN)))
        cube.add_cell_method(CellMethod("mean", cm.coord("time")))

    if \
            (grib.edition == 1) and \
            (grib.timeRangeIndicator == 124):
        cube.add_aux_coord(DimCoord(points=grib.startStep, standard_name='forecast_period', units=grib._forecastTimeUnit))
        cube.add_aux_coord(DimCoord(points=np.mean(grib.phenomenon_bounds('hours')), bounds=grib.phenomenon_bounds('hours'),  standard_name='time', units=Unit('hours since epoch', iris.unit.CALENDAR_GREGORIAN)))
        cube.add_cell_method(CellMethod("sum", cm.coord("time")))

    if \
            (grib.edition == 1) and \
            (grib.timeRangeIndicator == 125):
        cube.add_aux_coord(DimCoord(points=grib.startStep, standard_name='forecast_period', units=grib._forecastTimeUnit))
        cube.add_aux_coord(DimCoord(points=np.mean(grib.phenomenon_bounds('hours')), bounds=grib.phenomenon_bounds('hours'),  standard_name='time', units=Unit('hours since epoch', iris.unit.CALENDAR_GREGORIAN)))
        cube.add_cell_method(CellMethod("standard_deviation", cm.coord("time")))

    if \
            (grib.edition == 2) and \
            (grib.productDefinitionTemplateNumber == 0):
        cube.add_aux_coord(DimCoord(points=Unit(grib._forecastTimeUnit).convert(np.int32(grib._forecastTime), "hours"), standard_name='forecast_period', units="hours"))
        cube.add_aux_coord(DimCoord(points=grib.phenomenon_points('hours'), standard_name='time', units=Unit('hours since epoch', iris.unit.CALENDAR_GREGORIAN)))

    if \
            (grib.edition == 2) and \
            (grib.productDefinitionTemplateNumber in (8, 9)):
        cube.add_aux_coord(DimCoord(points=Unit(grib._forecastTimeUnit).convert(np.int32(grib._forecastTime), "hours"), standard_name='forecast_period', units="hours"))
        cube.add_aux_coord(DimCoord(points=np.mean(grib.phenomenon_bounds('hours')), standard_name='time', units=Unit('hours since epoch', iris.unit.CALENDAR_GREGORIAN), bounds=grib.phenomenon_bounds('hours')))

    if \
            (grib.edition == 2) and \
            (grib.productDefinitionTemplateNumber == 8) and \
            (grib.typeOfStatisticalProcessing == 0):
        cube.add_cell_method(CellMethod("mean", cm.coord("time")))

    if \
            (grib.edition == 2) and \
            (grib.productDefinitionTemplateNumber == 8) and \
            (grib.typeOfStatisticalProcessing == 1):
        cube.add_cell_method(CellMethod("sum", cm.coord("time")))

    if \
            (grib.edition == 2) and \
            (grib.productDefinitionTemplateNumber == 8) and \
            (grib.typeOfStatisticalProcessing == 2):
        cube.add_cell_method(CellMethod("maximum", cm.coord("time")))

    if \
            (grib.edition == 2) and \
            (grib.productDefinitionTemplateNumber == 8) and \
            (grib.typeOfStatisticalProcessing == 3):
        cube.add_cell_method(CellMethod("minimum", cm.coord("time")))

    if \
            (grib.edition == 2) and \
            (grib.productDefinitionTemplateNumber == 8) and \
            (grib.typeOfStatisticalProcessing == 4):
        cube.add_cell_method(CellMethod("_difference", cm.coord("time")))

    if \
            (grib.edition == 2) and \
            (grib.productDefinitionTemplateNumber == 8) and \
            (grib.typeOfStatisticalProcessing == 5):
        cube.add_cell_method(CellMethod("_root_mean_square", cm.coord("time")))

    if \
            (grib.edition == 2) and \
            (grib.productDefinitionTemplateNumber == 8) and \
            (grib.typeOfStatisticalProcessing == 6):
        cube.add_cell_method(CellMethod("standard_deviation", cm.coord("time")))

    if \
            (grib.edition == 2) and \
            (grib.productDefinitionTemplateNumber == 8) and \
            (grib.typeOfStatisticalProcessing == 7):
        cube.add_cell_method(CellMethod("_convariance", cm.coord("time")))

    if \
            (grib.edition == 2) and \
            (grib.productDefinitionTemplateNumber == 8) and \
            (grib.typeOfStatisticalProcessing == 8):
        cube.add_cell_method(CellMethod("_difference", cm.coord("time")))

    if \
            (grib.edition == 2) and \
            (grib.productDefinitionTemplateNumber == 8) and \
            (grib.typeOfStatisticalProcessing == 9):
        cube.add_cell_method(CellMethod("_ratio", cm.coord("time")))

    if \
            (grib.edition == 1) and \
            (grib.levelType == 'pl'):
        cube.add_aux_coord(DimCoord(points=grib.level,  long_name="pressure", units="hPa"))

    if \
            (grib.edition == 1) and \
            (grib.levelType == 'sfc') and \
            (grib._cf_data is not None) and \
            (grib._cf_data.set_height is not None):
        cube.add_aux_coord(DimCoord(points=grib._cf_data.set_height,  long_name="height", units="m", attributes={'positive':'up'}))

    if \
            (grib.edition == 2) and \
            (grib.typeOfFirstFixedSurface == 100) and \
            (grib.typeOfSecondFixedSurface == 255):
        cube.add_aux_coord(DimCoord(points=grib.scaledValueOfFirstFixedSurface/(10.0**grib.scaleFactorOfFirstFixedSurface), long_name="pressure", units="Pa"))

    if \
            (grib.edition == 2) and \
            (grib.typeOfFirstFixedSurface == 100) and \
            (grib.typeOfSecondFixedSurface != 255):
        cube.add_aux_coord(DimCoord(points=0.5*(grib.scaledValueOfFirstFixedSurface/(10.0**grib.scaleFactorOfFirstFixedSurface) + grib.scaledValueOfSecondFixedSurface/(10.0**grib.scaleFactorOfSecondFixedSurface)), long_name="pressure", units="Pa", bounds=[grib.scaledValueOfFirstFixedSurface/(10.0**grib.scaleFactorOfFirstFixedSurface) , grib.scaledValueOfSecondFixedSurface/(10.0**grib.scaleFactorOfSecondFixedSurface)]))

    if \
            (grib.edition == 2) and \
            (grib.typeOfFirstFixedSurface in [105, 119]) and \
            (grib.numberOfCoordinatesValues > 0):
        cube.add_aux_coord(AuxCoord(grib.scaledValueOfFirstFixedSurface, standard_name='model_level_number', attributes={'positive': 'up'}))
        cube.add_aux_coord(DimCoord(grib.pv[grib.scaledValueOfFirstFixedSurface], long_name='level_pressure', units='Pa'))
        cube.add_aux_coord(AuxCoord(grib.pv[grib.numberOfCoordinatesValues/2 + grib.scaledValueOfFirstFixedSurface], long_name='sigma'))
        factories.append(Factory(HybridPressureFactory, [{'long_name': 'level_pressure'}, {'long_name': 'sigma'}, Reference('surface_pressure')]))

    if grib._originatingCentre != 'unknown':
        cube.add_aux_coord(AuxCoord(points=grib._originatingCentre, long_name='originating_centre', units='no_unit'))

    if \
            (grib.edition == 2) and \
            (grib.productDefinitionTemplateNumber == 1):
        cube.add_aux_coord(DimCoord(points=grib.perturbationNumber, long_name='ensemble_member', units='no_unit'))

    if grib.productDefinitionTemplateNumber not in (0, 8):
        cube.attributes["GRIB_LOAD_WARNING"] = ("unsupported GRIB%d ProductDefinitionTemplate: #4.%d" % (grib.edition, grib.productDefinitionTemplateNumber))

    if \
            (grib.edition == 2) and \
            (grib.centre == 'ecmf') and \
            (grib.discipline == 0) and \
            (grib.parameterCategory == 3) and \
            (grib.parameterNumber == 25) and \
            (grib.typeOfFirstFixedSurface == 105):
        references.append(ReferenceTarget('surface_pressure', lambda cube: {'standard_name': 'surface_air_pressure', 'units': 'Pa', 'data': np.exp(cube.data)}))

    return factories, references
