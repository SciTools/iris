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
"""An attempt to break down the NIMROD loading rules into atomic functions."""


import netcdftime
import numpy

import warnings

import iris
from iris.coords import DimCoord
from iris.exceptions import TranslationError


def name(cube, field):
    cube.rename(field.title.strip())


def units(cube, field):
    units = field.units.strip()
    try:
        cube.units = units
    except ValueError:
        # Just add it as an attribute.
        warnings.warn("Unhandled units '{0}' recorded in cube attributes.".
                      format(units))
        cube.attributes["invalid_units"] = units


def time(cube, field):
    valid_date = netcdftime.datetime(field.vt_year, field.vt_month,
                                     field.vt_day, field.vt_hour,
                                     field.vt_minute, field.vt_second)
    time_coord = DimCoord(iris.unit.date2num(valid_date,
                                             'hours since 1970-01-01 00:00:00', 
                                             iris.unit.CALENDAR_STANDARD), 
                          standard_name='time', units='hours')
    cube.add_aux_coord(time_coord)


def reference_time(cube, field):
    if field.dt_year != field.int_mdi:        
        data_date = netcdftime.datetime(field.dt_year, field.dt_month,
                                        field.dt_day, field.dt_hour,
                                        field.dt_minute)
        ref_time_coord = DimCoord(iris.unit.date2num(
                                      data_date, 
                                      'hours since 1970-01-01 00:00:00', 
                                      iris.unit.CALENDAR_STANDARD), 
                                  standard_name='forecast_reference_time',
                                  units='hours')
        cube.add_aux_coord(ref_time_coord)


def experiment(cube, field):
    if field.experiment_num != field.int_mdi:
        cube.add_aux_coord(DimCoord(field.experiment_num, 
                                    long_name="experiment_number"))


def period_minutes(cube, field):
    if field.period_minutes != field.int_mdi:
        if field.period_minutes != 0:
            raise TranslationError("Period_minutes not yet handled")


def proj_biaxial_ellipsoid(cube, field):    
    if field.proj_biaxial_ellipsoid not in [field.int_mdi, 0]:
        raise TranslationError("Biaxial ellipsoid %d not yet handled" %
                               field.proj_biaxial_ellipsoid)


def tm_meridian_scaling(cube, field):
    if field.tm_meridian_scaling != field.int_mdi:
        if int(field.tm_meridian_scaling*1e6) == 999601:
            pass  # This is the expected value for British National Grid
        else:
            raise TranslationError("tm_meridian_scaling not yet handled: {}".
                                   format(field.tm_meridian_scaling))


def NG_X(cube, field):
    # British National Grid x coord
    cube.add_dim_coord(
        DimCoord(numpy.arange(field.num_cols) * field.column_step + field.x_origin,
                 long_name="x", units="m", coord_system=iris.coord_systems.OSGB()), 1)


def NG_Y(cube, field):
    # British National Grid y coord
    if field.origin_corner == 0:  # top left
        cube.add_dim_coord(
            DimCoord(numpy.arange(field.num_rows)[::-1] * -field.row_step + field.y_origin,
                     long_name="y", units="m", coord_system=iris.coord_systems.OSGB()), 0)
    else:
        raise TranslationError("Corner {0} not yet implemented".
                               format(field.origin_corner))


def horizontal_grid(cube, field):
    # "NG" (British National Grid)
    if field.horizontal_grid_type == 0:
        NG_X(cube, field)
        NG_Y(cube, field)
    else:
        raise TranslationError("Grid type %d not yet implemented" %
                               field.horizontal_grid_type)


def orography_vertical_coord(cube, field):
    # We can find values in the vertical coord, such as 9999,
    # for orography fields. Don't make a vertical coord from these.
    pass


def height_vertical_coord(cube, field):
    if (field.reference_vertical_coord_type == field.int_mdi or
        field.reference_vertical_coord == field.float32_mdi):
        cube.add_aux_coord(DimCoord(field.vertical_coord,
                            standard_name="height", units="m"))
    else:
        raise TranslationError("Bounded vertical not yet implemented")


def altitude_vertical_coord(cube, field):
    if (field.reference_vertical_coord_type == field.int_mdi or
        field.reference_vertical_coord == field.float32_mdi):
        cube.add_aux_coord(DimCoord(field.vertical_coord,
                            standard_name="altitude", units="m"))
    else:
        raise TranslationError("Bounded vertical not yet implemented")


def vertical_coord(cube, field):
    if field.vertical_coord_type != field.int_mdi:
        if field.field_code == 73:
            orography_vertical_coord(cube, field)
        elif field.vertical_coord_type == 0:
            height_vertical_coord(cube, field)
        elif field.vertical_coord_type == 1:
            altitude_vertical_coord(cube, field)
        else:
            raise TranslationError("Vertical coord type %d not yet handled" %
                                   field.vertical_coord_type)


def ensemble_member(cube, field):
    ensemble_member = getattr(field, "ensemble_member")
    if ensemble_member != field.int_mdi:
        cube.add_aux_coord(DimCoord(ensemble_member, "realization"))


def origin_corner(cube, field):
    if field.origin_corner == 0:  # top left
        cube.data = cube.data[::-1, :].copy()
    else:
        raise TranslationError("Corner {0} not yet implemented".
                               format(field.origin_corner))
    return cube


def attributes(cube, field):
        
    def add_attr(name):
        if hasattr(field, name):
            value = getattr(field, name)
            if value not in [field.int_mdi, field.float32_mdi]:
                cube.attributes[name] = value

    add_attr("nimrod_version")
    add_attr("field_code")
    add_attr("num_model_levels")
    add_attr("sat_calib")
    add_attr("sat_space_count")
    add_attr("ducting_index")
    add_attr("elevation_angle")
    add_attr("radar_num")
    add_attr("radars_bitmask")
    add_attr("more_radars_bitmask")
    add_attr("clutter_map_num")
    add_attr("calibration_type")
    add_attr("bright_band_height")
    add_attr("bright_band_intensity")
    add_attr("bright_band_test1")
    add_attr("bright_band_test2")
    add_attr("infill_flag")
    add_attr("stop_elevation")
    add_attr("sensor_id")
    add_attr("meteosat_id")
    add_attr("alphas_available")

    cube.attributes["source"] = field.source.strip()
    

def run(field):
    """
    Convert a NIMROD field to an Iris cube. 
    
    Args:
    
        * field - a :class:`~iris.fileformats.nimrod.NimrodField`

    Returns:
    
        * A new :class:`~iris.cube.Cube`, created from the NimrodField.
    
    """

    cube = iris.cube.Cube(field.data)

    name(cube, field)
    units(cube, field)

    # time
    time(cube, field)
    reference_time(cube, field)                
    period_minutes(cube, field)
    
    experiment(cube, field)
        
    # horizontal grid
    proj_biaxial_ellipsoid(cube, field)    
    tm_meridian_scaling(cube, field)
    horizontal_grid(cube, field)
    
    # vertical
    vertical_coord(cube, field)

    # add other stuff, if present
    ensemble_member(cube, field)
    attributes(cube, field)

    origin_corner(cube, field)
    
    return cube
