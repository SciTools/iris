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
"""Rules for converting NIMROD fields into cubes."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa


import warnings

import netcdftime
import numpy as np

import iris
from iris.coords import DimCoord
from iris.exceptions import TranslationError


__all__ = ['run']


# Meridian scaling for British National grid.
MERIDIAN_SCALING_BNG = 0.9996012717

NIMROD_DEFAULT = -32767.0

TIME_UNIT = iris.unit.Unit('hours since 1970-01-01 00:00:00',
                           calendar=iris.unit.CALENDAR_STANDARD)


FIELD_CODES = {73: "orography"}
VERTICAL_CODES = {0: "height", 1: "altitude", 12: "levels_below_ground"}


class TranslationWarning(Warning):
    pass


def name(cube, field):
    """Set the cube's name from the field."""

    cube.rename(field.title.strip())


def units(cube, field):
    """
    Set the cube's units from the field.

    Unhandled units are stored in an "invalid_units" attribute instead.

    """
    units = field.units.strip()
    try:
        cube.units = units
    except ValueError:
        # Just add it as an attribute.
        warnings.warn("Unhandled units '{0}' recorded in cube attributes.".
                      format(units))
        cube.attributes["invalid_units"] = units


def time(cube, field):
    """Add a time coord to the cube."""
    valid_date = netcdftime.datetime(field.vt_year, field.vt_month,
                                     field.vt_day, field.vt_hour,
                                     field.vt_minute, field.vt_second)
    point = TIME_UNIT.date2num(valid_date)

    bounds = None
    if field.period_minutes != field.int_mdi and field.period_minutes != 0:
        # Create a bound array to handle the Period of Interest if set.
        bounds = (point - (field.period_minutes / 60.0), point)

    time_coord = DimCoord(points=point,
                          bounds=bounds,
                          standard_name='time',
                          units=TIME_UNIT)

    cube.add_aux_coord(time_coord)


def reference_time(cube, field):
    """Add a 'reference time' to the cube, if present in the field."""
    if field.dt_year != field.int_mdi:
        data_date = netcdftime.datetime(field.dt_year, field.dt_month,
                                        field.dt_day, field.dt_hour,
                                        field.dt_minute)

        ref_time_coord = DimCoord(TIME_UNIT.date2num(data_date),
                                  standard_name='forecast_reference_time',
                                  units=TIME_UNIT)

        cube.add_aux_coord(ref_time_coord)


def experiment(cube, field):
    """Add an 'experiment number' to the cube, if present in the field."""
    if field.experiment_num != field.int_mdi:
        cube.add_aux_coord(DimCoord(field.experiment_num,
                                    long_name="experiment_number"))


def proj_biaxial_ellipsoid(cube, field):
    """
    Ellipsoid definition is currently ignored.

    """
    pass


def tm_meridian_scaling(cube, field):
    """
    Deal with the scale factor on the central meridian for transverse mercator
    projections if present in the field.

    Currently only caters for British National Grid.

    """
    if field.tm_meridian_scaling not in [field.float32_mdi, NIMROD_DEFAULT]:
        if abs(field.tm_meridian_scaling - MERIDIAN_SCALING_BNG) < 1e-6:
            pass  # This is the expected value for British National Grid
        else:
            warnings.warn("tm_meridian_scaling not yet handled: {}"
                          "".format(field.tm_meridian_scaling),
                          TranslationWarning)


def british_national_grid_x(cube, field):
    """Add a British National Grid X coord to the cube."""
    x_coord = DimCoord(np.arange(field.num_cols) * field.column_step +
                       field.x_origin, standard_name="projection_x_coordinate",
                       units="m", coord_system=iris.coord_systems.OSGB())
    cube.add_dim_coord(x_coord, 1)


def british_national_grid_y(cube, field):
    """
    Add a British National Grid Y coord to the cube.

    Currently only handles origin in the top left corner.

    """
    if field.origin_corner == 0:  # top left
        y_coord = DimCoord(np.arange(field.num_rows)[::-1] *
                           -field.row_step + field.y_origin,
                           standard_name="projection_y_coordinate", units="m",
                           coord_system=iris.coord_systems.OSGB())
        cube.add_dim_coord(y_coord, 0)
    else:
        raise TranslationError("Corner {0} not yet implemented".
                               format(field.origin_corner))


def horizontal_grid(cube, field):
    """Add X and Y coords to the cube.

    Currently only handles British National Grid.

    """
    # "NG" (British National Grid)
    if field.horizontal_grid_type == 0:
        british_national_grid_x(cube, field)
        british_national_grid_y(cube, field)
    else:
        raise TranslationError("Grid type %d not yet implemented" %
                               field.horizontal_grid_type)


def orography_vertical_coord(cube, field):
    """Special handling of vertical coords for orography fields: Do nothing."""
    # We can find values in the vertical coord, such as 9999,
    # for orography fields. Don't make a vertical coord from these.
    pass


def height_vertical_coord(cube, field):
    """Add a height coord to the cube, if present in the field."""
    if (field.reference_vertical_coord_type == field.int_mdi or
            field.reference_vertical_coord == field.float32_mdi):
        height_coord = DimCoord(field.vertical_coord,
                                standard_name="height", units="m",
                                attributes={"positive": "up"})
        cube.add_aux_coord(height_coord)
    else:
        raise TranslationError("Bounded vertical not yet implemented")


def altitude_vertical_coord(cube, field):
    """Add an altitude coord to the cube, if present in the field."""
    if (field.reference_vertical_coord_type == field.int_mdi or
            field.reference_vertical_coord == field.float32_mdi):
                alti_coord = DimCoord(field.vertical_coord,
                                      standard_name="altitude",
                                      units="m",
                                      attributes={"positive": "up"})
                cube.add_aux_coord(alti_coord)
    else:
        raise TranslationError("Bounded vertical not yet implemented")


def levels_below_ground_vertical_coord(cube, field):
    """Add a levels_below_ground coord to the cube, if present in the field."""
    if (field.reference_vertical_coord_type == field.int_mdi or
            field.reference_vertical_coord == field.float32_mdi):
                lev_coord = DimCoord(field.vertical_coord,
                                     long_name="levels_below_ground",
                                     units="1",
                                     attributes={"positive": "down"})
                cube.add_aux_coord(lev_coord)
    else:
        raise TranslationError("Bounded vertical not yet implemented")


def vertical_coord(cube, field):
    """Add a vertical coord to the cube."""
    v_type = field.vertical_coord_type

    if v_type not in [field.int_mdi, NIMROD_DEFAULT]:
        if FIELD_CODES.get(field.field_code, None) == "orography":
            orography_vertical_coord(cube, field)
        else:
            vertical_code_name = VERTICAL_CODES.get(v_type, None)
            if vertical_code_name == "height":
                height_vertical_coord(cube, field)
            elif vertical_code_name == "altitude":
                altitude_vertical_coord(cube, field)
            elif vertical_code_name == "levels_below_ground":
                levels_below_ground_vertical_coord(cube, field)
            else:
                warnings.warn("Vertical coord {!r} not yet handled"
                              "".format(v_type), TranslationWarning)


def ensemble_member(cube, field):
    """Add an 'ensemble member' coord to the cube, if present in the field."""
    ensemble_member = getattr(field, "ensemble_member")
    if ensemble_member != field.int_mdi:
        cube.add_aux_coord(DimCoord(ensemble_member, "realization"))


def origin_corner(cube, field):
    """Ensure the data matches the order of the coords we've made."""
    if field.origin_corner == 0:  # top left
        cube.data = cube.data[::-1, :].copy()
    else:
        raise TranslationError("Corner {0} not yet implemented".
                               format(field.origin_corner))
    return cube


def attributes(cube, field):
    """Add attributes to the cube."""
    def add_attr(name):
        """Add an attribute to the cube."""
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
