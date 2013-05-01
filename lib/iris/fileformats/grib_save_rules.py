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


import warnings

import gribapi
import numpy as np
import numpy.ma as ma

import iris
import iris.unit
from iris.fileformats.rules import is_regular, regular_step


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
        raise iris.exceptions.TranslationError("Cannot save irregular grids to grib")

    # Time period exists?
    if not cube.coords("time"):
        raise iris.exceptions.TranslationError("time coord not found")

    # Forecast period exists?
    if not cube.coords("forecast_period"):
        raise iris.exceptions.TranslationError("forecast_period coord not found")

###########################
### grid template stuff ###
###########################


def shape_of_the_earth(cube, grib):
    
    # assume latlon
    cs = cube.coord(dimensions=[0]).coord_system

    # Turn them all missing to start with
    gribapi.grib_set_long(grib, "scaleFactorOfRadiusOfSphericalEarth", 255)  # missing, byte
    gribapi.grib_set_long(grib, "scaledValueOfRadiusOfSphericalEarth", -1)  # missing, long
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
        gribapi.grib_set_long(grib, "scaledValueOfRadiusOfSphericalEarth", ellipsoid.semi_major_axis)
        
    else:
        gribapi.grib_set_long(grib, "shapeOfTheEarth", 7)
        gribapi.grib_set_long(grib, "scaleFactorOfEarthMajorAxis", 0)
        gribapi.grib_set_long(grib, "scaledValueOfEarthMajorAxis", ellipsoid.semi_major_axis)
        gribapi.grib_set_long(grib, "scaleFactorOfEarthMinorAxis", 0)
        gribapi.grib_set_long(grib, "scaledValueOfEarthMinorAxis", ellipsoid.semi_minor_axis)
        

def grid_dims(x_coord, y_coord, grib):

    gribapi.grib_set_long(grib, "Ni", x_coord.shape[0])
    gribapi.grib_set_long(grib, "Nj", y_coord.shape[0])


def latlon_first_last(x_coord, y_coord, grib):
    
    if x_coord.has_bounds() or y_coord.has_bounds():
        warnings.warn("Ignoring xy bounds")

# XXX Pending #1125
#    gribapi.grib_set_double(grib, "latitudeOfFirstGridPointInDegrees", float(y_coord.points[0]))
#    gribapi.grib_set_double(grib, "latitudeOfLastGridPointInDegrees", float(y_coord.points[-1]))
#    gribapi.grib_set_double(grib, "longitudeOfFirstGridPointInDegrees", float(x_coord.points[0]))
#    gribapi.grib_set_double(grib, "longitudeOfLastGridPointInDegrees", float(x_coord.points[-1]))
# WORKAROUND
    gribapi.grib_set_long(grib, "latitudeOfFirstGridPoint", int(y_coord.points[0]*1000000))
    gribapi.grib_set_long(grib, "latitudeOfLastGridPoint", int(y_coord.points[-1]*1000000))
    gribapi.grib_set_long(grib, "longitudeOfFirstGridPoint", int((x_coord.points[0]%360)*1000000))
    gribapi.grib_set_long(grib, "longitudeOfLastGridPoint", int((x_coord.points[-1]%360)*1000000))


def dx_dy(x_coord, y_coord, grib):

    x_step = regular_step(x_coord) 
    y_step = regular_step(y_coord) 

    # TODO: THIS USED BE "Dx" and "Dy"!!! DID THE API CHANGE AGAIN???
    gribapi.grib_set_double(grib, "DxInDegrees", float(abs(x_step)))
    gribapi.grib_set_double(grib, "DyInDegrees", float(abs(y_step)))

    
def scanning_mode_flags(x_coord, y_coord, grib):
    
    gribapi.grib_set_long(grib, "iScansPositively", int(x_coord.points[1] - x_coord.points[0] > 0))
    gribapi.grib_set_long(grib, "jScansPositively", int(y_coord.points[1] - y_coord.points[0] > 0))


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
#    gribapi.grib_set_double(grib, "latitudeOfSouthernPoleInDegrees", float(cs.n_pole.latitude))
#    gribapi.grib_set_double(grib, "longitudeOfSouthernPoleInDegrees", float(cs.n_pole.longitude))
#    gribapi.grib_set_double(grib, "angleOfRotationInDegrees", 0)
# WORKAROUND
    gribapi.grib_set_long(grib, "latitudeOfSouthernPole", -int(cs.grid_north_pole_latitude*1000000))
    gribapi.grib_set_long(grib, "longitudeOfSouthernPole", int(((cs.grid_north_pole_longitude+180)%360)*1000000))
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


##############################
### product template stuff ###
##############################


def param_code(cube, grib):
    gribapi.grib_set_long(grib, "discipline", 0)
    gribapi.grib_set_long(grib, "parameterCategory", 0)
    gribapi.grib_set_long(grib, "parameterNumber", 0)
    warnings.warn("Not yet translating standard name into grib param codes.\n"
                  "discipline, parameterCategory and parameterNumber have been zeroed.")
    
    
def generating_process_type(cube, grib):
    
    # analysis = 0
    # initialisation = 1
    # forecast = 2
    # more...
    
    # missing
    gribapi.grib_set_long(grib, "typeOfGeneratingProcess", 255)
    
    
def background_process_id(cube, grib):
    # locally defined    
    gribapi.grib_set_long(grib, "backgroundProcess", 255)
    

def generating_process_id(cube, grib):
    # locally defined    
    gribapi.grib_set_long(grib, "generatingProcessIdentifier", 255)


def obs_time_after_cutoff(cube, grib):
    # nothing stored in iris for this at present
    gribapi.grib_set_long(grib, "hoursAfterDataCutoff", 0)
    gribapi.grib_set_long(grib, "minutesAfterDataCutoff", 0)


def time_range(cube, grib):
    """Grib encoding of forecast_period.""" 
    
    fp_coord = cube.coord("forecast_period")
    if fp_coord.has_bounds():
        raise iris.exceptions.TranslationError("Bounds not expected for 'forecast_period'")
    
    if fp_coord.units == iris.unit.Unit("hours"):
        grib_time_code = 1
    elif fp_coord.units == iris.unit.Unit("minutes"):
        grib_time_code = 0
    elif fp_coord.units == iris.unit.Unit("seconds"):
        grib_time_code = 13
    else:
        raise iris.exceptions.TranslationError("Unexpected units for 'forecast_period' : %s" % fp_coord.units)
    
    fp = fp_coord.points[0]
    if fp - int(fp):
        warnings.warn("forecast_period encoding problem : Scaling required.")
    fp = int(fp)
    
    # Turn negative forecast times into grib negative numbers?
    from iris.fileformats.grib import hindcast_workaround
    if hindcast_workaround and fp < 0:
        msg = "Encoding negative forecast period from {} to ".format(fp)
        fp = 2**31 + abs(fp)
        msg += "{}".format(np.int32(fp))
        warnings.warn(msg)
        
    gribapi.grib_set_long(grib, "indicatorOfUnitOfTimeRange", grib_time_code)
    gribapi.grib_set_long(grib, "forecastTime", fp)


def hybrid_surfaces(cube, grib):
    
    is_hybrid = False
    
# XXX Addressed in #1118 pending #1039 for hybrid levels
#
#    # hybrid height? (assume points)
#    if cube.coords("model_level") and cube.coords("level_height") and cube.coords("sigma") \
#    and isinstance(cube.coord("sigma").coord_system, iris.coord_systems.HybridHeightCS):
#        is_hybrid = True
#        gribapi.grib_set_long(grib, "typeOfFirstFixedSurface", 118)
#        gribapi.grib_set_long(grib, "scaledValueOfFirstFixedSurface", long(cube.coord("model_level").points[0]))
#        gribapi.grib_set_long(grib, "PVPresent", 1)
#        gribapi.grib_set_long(grib, "numberOfVerticalCoordinateValues", 2)
#        gribapi.grib_set_double_array(grib, "pv", [cube.coord("level_height").points[0], cube.coord("sigma").points[0]])
#
#    # hybrid pressure?
#    if XXX:
#        pass

    return is_hybrid


def non_hybrid_surfaces(cube, grib):

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

    else:
        raise iris.exceptions.TranslationError("Vertical coordinate not found / handled")

    # is it a surface layer (no thickness)?
    if not v_coord.has_bounds():
        output_v = v_coord.units.convert(v_coord.points[0], output_unit)
        if output_v - abs(output_v):
            warnings.warn("Vertical level encoding problem : Scaling required.")
        output_v = int(output_v)
        
        gribapi.grib_set_long(grib, "typeOfFirstFixedSurface", grib_v_code)
        gribapi.grib_set_long(grib, "scaleFactorOfFirstFixedSurface", 0)
        gribapi.grib_set_long(grib, "scaledValueOfFirstFixedSurface", output_v)
        
        gribapi.grib_set_long(grib, "typeOfSecondFixedSurface", -1)
        gribapi.grib_set_long(grib, "scaleFactorOfSecondFixedSurface", 255)  # missing, byte
        gribapi.grib_set_long(grib, "scaledValueOfSecondFixedSurface", -1)   # missing, long        
    
    # it's a bounded layer
    else:
        output_v = v_coord.units.convert(v_coord.bounds[0,0], output_unit)
        if (output_v[0] - abs(output_v[0])) or (output_v[1] - abs(output_v[1])):
            warnings.warn("Vertical level encoding problem : Scaling required.")
        
        gribapi.grib_set_long(grib, "typeOfFirstFixedSurface", grib_v_code)
        gribapi.grib_set_long(grib, "scaleFactorOfFirstFixedSurface", 0)
        gribapi.grib_set_long(grib, "scaledValueOfFirstFixedSurface", output_v[0])
    
        gribapi.grib_set_long(grib, "typeOfSecondFixedSurface", grib_v_code)
        gribapi.grib_set_long(grib, "scaleFactorOfSecondFixedSurface", 0)
        gribapi.grib_set_long(grib, "scaledValueOfSecondFixedSurface", output_v[1])


def surfaces(cube, grib):
    
    if not hybrid_surfaces(cube, grib):
        non_hybrid_surfaces(cube, grib)
    
    
    
def product_common(cube, grib):

    param_code(cube, grib)
    generating_process_type(cube, grib)
    background_process_id(cube, grib)
    generating_process_id(cube, grib)
    obs_time_after_cutoff(cube, grib)
    time_range(cube, grib)
    surfaces(cube, grib)


def type_of_statistical_processing(cube, grib, coord):
    """Search for processing over the given coord."""
    
    stat_code = 255  # (grib code table 4.10)
    
    # if the last cell method applies only to the given coord...
    cell_method = cube.cell_methods[-1]
    if len(cell_method.coord_names) == 1 and cell_method.coord_names[0] == coord.name():
        if cell_method.method == 'mean':
            stat_code = 0
        elif cell_method.method == 'accumulation':
            stat_code = 1
        elif cell_method.method == 'minimum':
            stat_code = 2
        elif cell_method.method == 'maximum':
            stat_code = 3
        elif cell_method.method == 'standard_deviation':
            stat_code = 6

    if stat_code == 255:    
        warnings.warn("Unable to determine type of statistical processing")
        
    gribapi.grib_set_long(grib, "typeOfStatisticalProcessing", stat_code)  


def time_processing_period(cube, grib):
    """For template 4.8 (time mean, time max, etc)"""
    # We could probably split this function up a bit
    
    # Can safely assume bounded pt.
    pt_coord = cube.coord("time")
    end = iris.unit.num2date(pt_coord.bounds[0,1], pt_coord.units.name, pt_coord.units.calendar)
    
    gribapi.grib_set_long(grib, "yearOfEndOfOverallTimeInterval", end.year)
    gribapi.grib_set_long(grib, "monthOfEndOfOverallTimeInterval", end.month)
    gribapi.grib_set_long(grib, "dayOfEndOfOverallTimeInterval", end.day)
    gribapi.grib_set_long(grib, "hourOfEndOfOverallTimeInterval", end.hour)
    gribapi.grib_set_long(grib, "minuteOfEndOfOverallTimeInterval", end.minute)
    gribapi.grib_set_long(grib, "secondOfEndOfOverallTimeInterval", end.second)

    gribapi.grib_set_long(grib, "numberOfTimeRange", 1)
    gribapi.grib_set_long(grib, "numberOfMissingInStatisticalProcess", 0)
    

    type_of_statistical_processing(cube, grib, pt_coord)
    

    #type of time increment e.g incrementing fp, incrementing ref time, etc (code table 4.11)
    gribapi.grib_set_long(grib, "typeOfTimeIncrement", 255)
    
    gribapi.grib_set_long(grib, "indicatorOfUnitForTimeRange", 1) # time unit for period over which statistical processing is done (hours)
    gribapi.grib_set_long(grib, "lengthOfTimeRange", float(pt_coord.bounds[0,1] - pt_coord.bounds[0,0]))  # period over which statistical processing is done
    
    gribapi.grib_set_long(grib, "indicatorOfUnitForTimeIncrement", 255)  # time unit between successive source fields (not setting this at present)
    gribapi.grib_set_long(grib, "timeIncrement", 0)  # between successive source fields (just set to 0 for now)
    

def product_template(cube, grib):
    # This will become more complex if we cover more templates, such as 4.15

    # forecast (template 4.0)
    if not cube.coord("time").has_bounds():
        gribapi.grib_set_long(grib, "productDefinitionTemplateNumber", 0)
        product_common(cube, grib)
    
    # time processed (template 4.8)
    elif cube.cell_methods and cube.cell_methods[-1].coord_names[0] == "time":
        gribapi.grib_set_long(grib, "productDefinitionTemplateNumber", 8)
        product_common(cube, grib)
        time_processing_period(cube, grib)
        
    else:
        raise iris.exceptions.TranslationError("A suitable product template could not be deduced")


def centre(cube, grib):
    # TODO: read centre from cube
    gribapi.grib_set_long(grib, "centre", 74)  # UKMO
    gribapi.grib_set_long(grib, "subCentre", 0)  # exeter is not in the spec


def reference_time(cube, grib):

    # analysis, forecast start, verify time, obs time, (start of forecast for now)
    gribapi.grib_set_long(grib, "significanceOfReferenceTime", 1)
      
    # calculate reference time
    pt_coord = cube.coord("time")
    pt = pt_coord.bounds[0,0] if pt_coord.has_bounds() else pt_coord.points[0]  # always in hours
    ft = cube.coord("forecast_period").points[0]   # always in hours
    rt = pt - ft
    rt = iris.unit.num2date(rt, pt_coord.units.name, pt_coord.units.calendar)
    
    gribapi.grib_set_long(grib, "dataDate", "%04d%02d%02d" % (rt.year, rt.month, rt.day))
    gribapi.grib_set_long(grib, "dataTime", "%02d%02d" % (rt.hour, rt.minute))


def identification(cube, grib):

    centre(cube, grib)
    reference_time(cube, grib)
    
    # operational product, operational test, research product, etc (missing for now)
    gribapi.grib_set_long(grib, "productionStatusOfProcessedData", 255)
    
    # analysis, forecast, processed satellite, processed radar, (analysis and forecast products for now)
    gribapi.grib_set_long(grib, "typeOfProcessedData", 2)


def data(cube, grib):

    # mdi
    if isinstance(cube.data, ma.core.MaskedArray):
        gribapi.grib_set_double(grib, "missingValue", float(cube.data.fill_value))
        data = cube.data.filled()
    else:
        gribapi.grib_set_double(grib, "missingValue", float(-1e9))
        data = cube.data
    
    # values
    gribapi.grib_set_double_array(grib, "values", data.flatten())
    
    # todo: check packing accuracy?
    #print "packingError", gribapi.getb_get_double(grib, "packingError")


def run(cube, grib):

    gribbability_check(cube)
    
    identification(cube, grib)
    grid_template(cube, grib)
    product_template(cube, grib)
    data(cube, grib)
    
