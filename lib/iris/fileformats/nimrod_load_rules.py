# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Rules for converting NIMROD fields into cubes."""

import warnings
import re

import cf_units
import cftime
import numpy as np

import iris
import iris.coord_systems
from iris.coords import DimCoord
from iris.exceptions import TranslationError


__all__ = ["run"]


# Meridian scaling for British National grid.
MERIDIAN_SCALING_BNG = 0.9996012717

NIMROD_DEFAULT = -32767.0

TIME_UNIT = cf_units.Unit(
    "seconds since 1970-01-01 00:00:00", calendar=cf_units.CALENDAR_GREGORIAN
)


DEFAULT_UNITS = {817: 'm s^-1',
                 422: 'min^-1',
                 218: 'cm',
                 155: 'm',
                 101: 'm',
                 63: 'mm hr^-1',
                 61: 'mm',
                 58: 'Celsius',
                 12: 'mb'}
FIELD_CODES = {73: "orography"}
# VERTICAL_CODES contains conversions from the Nimrod Documentation for the
# header entry 20 for the vertical coordinate type
VERTICAL_CODES = {0: {'standard_name': 'height', 'units': 'm',
                      'attributes': {"positive": "up"}},
                  1: {'standard_name': 'altitude', 'units': 'm',
                      'attributes': {"positive": "up"}},
                  2: {'standard_name': 'pressure', 'units': 'hPa',
                      'attributes': {"positive": "down"}},
                  6: {'standard_name': 'temperature', 'units': 'K'},
                  12: {'long_name': 'levels_below_ground', 'units': 'unknown',
                       'attributes': {"positive": "down"}}}
# Unhandled VERTICAL_CODES values (no use case identified):
#  3: ['sigma', 'model level'],
#  4: ['eta', 'model level'],
#  5: ['radar beam number', 'unknown'],
#  7: ['potential temperature', 'unknown'],
#  8: ['equivalent potential temperature', 'unknown'],
#  9: ['wet bulb potential temperature', 'unknown'],
#  10: ['potential vorticity', 'unknown'],
#  11: ['cloud boundary', 'unknown'],

SOIL_TYPE_CODES = {1: "broadleaf_tree",
                   2: "needleleaf_tree",
                   3: "c3_grass",
                   4: "c4_grass",
                   5: "crop",
                   6: "shrub",
                   7: "urban",
                   8: "water",
                   9: "soil",
                   10: "ice",
                   601: "urban_canyon",
                   602: "urban_roof",
                   }
TIME_AVERAGING_CODES = {
    8192: "maximum in period",
    4096: 'minimum in period',
    2048: 'unknown(2048)',
    1024: 'unknown(1024)',
    512: 'time lagged',
    256: 'extrapolation',
    128: 'accumulation or average',
    64: 'from UM 150m',
    32: 'scaled to UM resolution',
    16: 'averaged over multiple surface types',
    8: 'only observations used',
    4: 'smoothed',
    2: 'cold bias applied',
    1: 'warm bias applied',
}


class TranslationWarning(Warning):
    pass


def name(cube, field):
    """Set the cube's name from the field.
    Modifies the Nimrod object title based on other meta-data in the
    Nimrod field and known use cases.
    """
    if field.field_code == 161 and field.threshold_value >= 0.:
        field.title = "minimum_cloud_base_above_threshold"
    if field.field_code == 12:
        field.title = "pressure"
    if field.field_code == 28:
        field.title = "snow probability"
    if field.field_code == 29 and field.threshold_value >= 0.:
        field.title = "fog fraction"
    if field.field_code == 58:
        field.title = "temperature"
    if field.field_code == 61:
        field.title = "precipitation"
    if field.field_code == 63:
        field.title = "precipitation"
    if field.field_code == 817:
        field.title = "wind_speed_of_gust"
    if field.field_code == 155:
        field.title = "Visibility"
    if field.field_code == 218:
        field.title = "snowfall"
    if field.field_code == 101:
        field.title = "snowmelt_above_sea_level"
    if field.field_code == 172:
        field.title = "cloud_area_fraction_in_atmosphere"
    if field.field_code == 421:
        field.title = "precipitation type"
    if field.field_code == 804 and field.vertical_coord >= 0.:
        field.title = "wind speed"
    if field.field_code == 806 and field.vertical_coord >= 0.:
        field.title = "wind direction"
    if field.field_code == 821:
        field.title = "Probabilistic Gust Risk Analysis from Observations"
        field.source = "Nimrod pwind routine"
    if field.source.strip() == "pwind":
        field.source = "Nimrod pwind routine"

    if getattr(field, "ensemble_member") == -98 and 'mean' not in field.title:
        field.title = 'mean_of_' + field.title
    if getattr(field, "ensemble_member") == -99 and 'spread' not in field.title:
        field.title = 'standard_deviation_of_' + field.title

    cube.rename(field.title.strip())


def units(cube, field):
    """
    Set the cube's units from the field.

    Takes into account nimrod unit strings of the form unit*?? where the data
    needs to converted by dividing by ??. Also converts units we know Iris
    can't handle into appropriate units Iris can handle. This is mostly when
    there is an inappropriate capital letter in the unit in the Nimrod file.
    Some units still can't be handled by Iris so in these cases empty strings
    are added as the cube's unit. The most notable unit Iris can't handle is
    oktas for cloud cover.

    Unhandled units are stored in an "invalid_units" attribute instead.

    """
    unit_exception_dictionary = {'Knts': 'knots',
                                 'knts': 'knots',
                                 'J/Kg': 'J/kg',
                                 'logical': '',
                                 'Code': '',
                                 'mask': '',
                                 'oktas': '',
                                 'm/2-25k': '',
                                 'g/Kg': '',
                                 'unitless': '',
                                 'Fraction': '1',
                                 'index': '',
                                 'Beaufort': '',
                                 'mmh2o': 'kg/m2',
                                 'n/a': ''}

    field_units = field.units.strip()
    if '*' in field_units:
        # Split into unit string and integer
        unit_list = field_units.split('*')
        if '^' in unit_list[1]:
            # Split out magnitude
            unit_sublist = unit_list[1].split('^')
            cube.data = cube.data / float(unit_sublist[0]) ** float(
                unit_sublist[1])
        else:
            cube.data = cube.data / float(unit_list[1])
        field_units = unit_list[0]
    if 'ug/m3E1' in field_units:
        # Split into unit string and integer
        unit_list = field_units.split('E')
        cube.data = cube.data / 10.**float(unit_list[1])
        field_units = unit_list[0]
    if '%' in field_units:
        # Convert any percentages into fraction
        unit_list = field_units.split('%')
        if len(''.join(unit_list)) == 0:
            field_units = '1'
            cube.data = cube.data / 100.
    if field_units in unit_exception_dictionary.keys():
        field_units = unit_exception_dictionary[field_units]
    if len(field_units) > 0 and field_units[0] == '/':
        # Deal with the case where the units are of the form '/unit' eg
        # '/second' in the Nimrod file. This converts to the form unit^-1
        field_units = field_units[1:] + '^-1'
    try:
        cube.units = field_units
    except ValueError:
        # Just add it as an attribute.
        warnings.warn(
            "Unhandled units '{0}' recorded in cube attributes.".format(
                field_units)
        )
        cube.attributes["invalid_units"] = field_units


def time(cube, field):
    """Add a time coord to the cube."""
    if field.vt_year <= 0:
        # Some ancillary files, eg land sea mask do not
        # have a validity time. So make one up for the
        # start of the year.
        # This will screw up the forecast_period for these fields,
        # although if the valid time is missing too, it will be
        # made to be the same, so the forecast_period will always
        # be zero for these files.
        valid_date = cftime.datetime(
            2016, 1, 1, 0, 0, 0)
    else:
        valid_date = cftime.datetime(
            field.vt_year,
            field.vt_month,
            field.vt_day,
            field.vt_hour,
            field.vt_minute,
            field.vt_second,
        )
    point = np.array(TIME_UNIT.date2num(valid_date), dtype=np.int64)

    lb_delta = None
    if field.period_minutes == 32767:
        lb_delta = int(field.period_seconds)
    elif field.period_minutes != field.int_mdi and field.period_minutes != 0:
        lb_delta = int(field.period_minutes) * 60
    if lb_delta:
        bounds = np.array([point - lb_delta, point], dtype=np.int64)
    else:
        bounds = None

    time_coord = DimCoord(
        points=point, bounds=bounds, standard_name="time", units=TIME_UNIT
    )

    cube.add_aux_coord(time_coord)


def reference_time(cube, field):
    """Add a 'reference time' to the cube, if present in the field."""
    if field.dt_year != field.int_mdi and field.dt_year > 0:
        data_date = cftime.datetime(
            field.dt_year,
            field.dt_month,
            field.dt_day,
            field.dt_hour,
            field.dt_minute,
        )

        ref_time_coord = DimCoord(
            np.array(TIME_UNIT.date2num(data_date), dtype=np.int64),
            standard_name="forecast_reference_time",
            units=TIME_UNIT,
        )

        cube.add_aux_coord(ref_time_coord)


def mask_cube(cube, field):
    """
    Updates cube.data to be a masked array if appropriate.

    """

    if field.datum_type == 1:
        # field.data are integers
        if np.any(field.data == field.int_mdi):
            cube.data = np.ma.masked_equal(field.data,
                                           field.int_mdi)
    elif field.datum_type == 0:
        # field.data are floats
        if np.any(np.isclose(field.data, field.float32_mdi)):
            cube.data = np.ma.masked_inside(
                field.data,
                field.float32_mdi - 0.5,
                field.float32_mdi + 0.5)


def experiment(cube, field):
    """Add an 'experiment number' to the cube, if present in the field."""
    if field.experiment_num != field.int_mdi:
        cube.add_aux_coord(
            DimCoord(field.experiment_num, long_name="experiment_number")
        )


def proj_biaxial_ellipsoid(field):
    """
    Returns the correct dictionary of arguments needed to define an
    iris.coord_systems.GeogCS.

    Based firstly on the value given by ellipsoid, then by grid if ellipsoid is
    missing, select the right pre-defined ellipsoid dictionary (Airy_1830 or
    International_1924).

    """
    # Reference for Airy_1830 and international_1924 ellipsoids:
    # http://fcm9/projects/PostProc/wiki/PostProcDocDomains#ProjectionConstants
    # Reference for GRS80:
    Airy_1830 = {'semi_major_axis': 6377563.396, 'semi_minor_axis': 6356256.910}
    International_1924 = {'semi_major_axis': 6378388.000,
                          'semi_minor_axis': 6356911.946}
    if field.proj_biaxial_ellipsoid == 0:
        ellipsoid = Airy_1830
    elif field.proj_biaxial_ellipsoid == 1:
        ellipsoid = International_1924
    elif field.proj_biaxial_ellipsoid == field.int_mdi or \
            field.proj_biaxial_ellipsoid == -32767:
        if field.horizontal_grid_type == 0:
            ellipsoid = Airy_1830
        elif field.horizontal_grid_type == 1 or field.horizontal_grid_type == 4:
            ellipsoid = International_1924
        else:
            raise TranslationError('''Unsupported grid type, only NG, EuroPP
                                     and lat/long are possible''')
    else:
        raise TranslationError(
            'Ellipsoid not supported, proj_biaxial_ellipsoid:{}, '
            'horizontal_grid_type:{}'.format(field.proj_biaxial_ellipsoid,
                                             field.horizontal_grid_type))
    return ellipsoid


def set_british_national_grid_defaults(field):
    """Check for missing coord-system meta-data and set default values for
    the Ordnance Survey GB Transverse Mercator projection. Some Radarnet
    files are missing these."""
    invalid_values = (-32767., float(field.int_mdi))

    if np.any([field.true_origin_latitude == v for v in invalid_values]):
        field.true_origin_latitude = 49.
    if np.any([field.true_origin_longitude == v for v in invalid_values]):
        field.true_origin_longitude = -2.
    if np.any([field.true_origin_easting == v for v in invalid_values]):
        field.true_origin_easting = 400000.
    if np.any([field.true_origin_northing == v for v in invalid_values]):
        field.true_origin_northing = -100000.
    if np.any([field.tm_meridian_scaling == v for v in invalid_values]):
        field.tm_meridian_scaling = 0.9996012717

    ng_central_meridian_sf_dp = 0.9996012717
    if abs(field.tm_meridian_scaling - ng_central_meridian_sf_dp) < 1.0e-04:
        # Update the National Grid scaling factor to double
        # precision accuracy to improve the accuracy of
        # reprojection calculations that use it.
        field.tm_meridian_scaling = ng_central_meridian_sf_dp


def coord_system(field):
    """Define the coordinate system for the field."""
    ellipsoid = proj_biaxial_ellipsoid(field)

    if field.horizontal_grid_type == 0:
        # Check for missing grid meta-data and insert OSGB definitions.
        # Some Radarnet files are missing these.
        set_british_national_grid_defaults(field)
    if field.horizontal_grid_type == 0 or field.horizontal_grid_type == 4:
        coord_sys = iris.coord_systems.TransverseMercator(
            field.true_origin_latitude, field.true_origin_longitude, field.true_origin_easting,
            field.true_origin_northing,
            field.tm_meridian_scaling, iris.coord_systems.GeogCS(**ellipsoid))
    elif field.horizontal_grid_type == 1:
        coord_sys = iris.coord_systems.GeogCS(**ellipsoid)
    else:
        raise TranslationError(
            "Coordinate system for field type {} not implemented".format(
                field.horizontal_grid_type
            )
        )
    return coord_sys


def horizontal_grid(cube, field):
    """Add X and Y coordinates to the cube.

    """
    if field.origin_corner != 0:
        raise TranslationError(
            "Corner {0} not yet implemented".format(field.origin_corner)
        )
    crs = coord_system(field)
    if field.horizontal_grid_type == 0 or field.horizontal_grid_type == 4:
        units_name = 'm'
        x_coord_name = 'projection_x_coordinate'
        y_coord_name = 'projection_y_coordinate'
    elif field.horizontal_grid_type == 1:
        units_name = 'degrees'
        x_coord_name = 'longitude'
        y_coord_name = 'latitude'
    else:
        raise TranslationError("Horizontal grid type {} not "
                               "implemented".format(field.horizontal_grid_type))
    points = (np.arange(field.num_cols) * field.column_step +
              field.x_origin).astype(np.float32)
    x_coord = DimCoord(
        points,
        standard_name=x_coord_name,
        units=units_name,
        coord_system=crs,
    )
    cube.add_dim_coord(x_coord, 1)
    points = (np.arange(field.num_rows)[::-1] * -field.row_step +
              field.y_origin).astype(np.float32)
    y_coord = DimCoord(
        points,
        standard_name=y_coord_name,
        units=units_name,
        coord_system=crs,
    )
    cube.add_dim_coord(y_coord, 0)


def vertical_coord(cube, field):
    """Add a vertical coord to the cube, if appropriate."""
    coord_point = getattr(field, "vertical_coord")
    if (coord_point == 9999. and
            (field.reference_vertical_coord in [9999., field.int_mdi,
                                                NIMROD_DEFAULT])):
        return
    if coord_point in [9999., 8888.]:
        # A bounded vertical coord starting from the surface
        # Relies on correct field.vertical_coord_type to distinguish between
        # these meanings.
        coord_point = 0.
    coord_point = np.array(coord_point, dtype=np.float32)
    if (field.reference_vertical_coord >= 0. and
            field.reference_vertical_coord != coord_point):
        bounds = np.array([coord_point, field.reference_vertical_coord],
                          dtype=np.float32)
    else:
        bounds = None

    coord_args = VERTICAL_CODES.get(field.vertical_coord_type, None)
    if coord_args:
        new_coord = iris.coords.AuxCoord(coord_point, bounds=bounds,
                                         **coord_args)
        # Add coordinate to cube
        cube.add_aux_coord(new_coord)
        return

    warnings.warn(
        "Vertical coord {!r} not yet handled" "".format(
            field.vertical_coord_type),
        TranslationWarning,
    )


def ensemble_member(cube, field):
    """Add an 'ensemble member' coord to the cube, if present in the field."""
    ensemble_member_value = getattr(field, "ensemble_member")
    if ensemble_member_value != field.int_mdi:
        cube.add_aux_coord(DimCoord(np.array(ensemble_member_value,
                                             dtype=np.int32),
                                    "realization"))


def origin_corner(cube, field):
    """Ensure the data matches the order of the coordinates we've made."""
    if field.origin_corner == 0:  # top left
        cube.data = cube.data[::-1, :].copy()
    else:
        raise TranslationError(
            "Corner {0} not yet implemented".format(field.origin_corner)
        )
    return cube


def attributes(cube, field):
    """Add attributes to the cube."""

    def add_attr(item):
        """Add an attribute to the cube."""
        if hasattr(field, item):
            value = getattr(field, item)
            if value not in [field.int_mdi, field.float32_mdi]:
                cube.attributes[item] = value

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
    for key in ["neighbourhood_radius", "recursive_filter_iterations",
                "recursive_filter_alpha", "threshold_vicinity_radius",
                "probability_period_of_event"]:
        add_attr(key)

    source = field.source.strip()
    rematcher = re.compile('^ek\d\d$')
    if (rematcher.match(source) is not None
            or source.find('umek') == 0):
        source = 'MOGREPS-UK'
    cube.attributes['source'] = source


def threshold_coord(cube, field):
    """
    Adds a scalar threshold coord to the cube for known use cases.
    """
    coord_keys = None
    if field.field_code == 161 and field.threshold_value >= 0.:
        coord_keys = {"standard_name": "cloud_area_fraction",
                      "units": ""}
    if field.field_code == 29 and field.threshold_value >= 0.:
        coord_keys = {"standard_name": "visibility_in_air",
                      "units": "metres"}
    if field.field_code == 821:
        coord_keys = {"standard_name": "wind_speed_of_gust",
                      "units": "m/s"}
    if coord_keys:
        cube.add_aux_coord(iris.coords.AuxCoord(
            np.array(field.threshold_value, dtype=np.float32),
            var_name='threshold',
            **coord_keys))


def probability_coord(cube, field):
    """
    Adds a coord relating to probability meta-data from the header to the
    cube if appropriate.
    Returns True if this is a blended multi-member field
    """
    probtype_lookup = {1: {'var_name': 'threshold',
                           'attributes': {'relative_to_threshold': 'above'}},
                       2: {'var_name': 'threshold',
                           'attributes': {'relative_to_threshold': 'below'}},
                       3: {'standard_name': 'percentile', 'units': "%"},
                       4: {'var_name': 'threshold',
                           'attributes': {'relative_to_threshold': 'equal'}}}
    probmethod_lookup = {1: 'AOT (Any One Time)',
                         2: 'ST (Some Time)',
                         4: 'AT (All Time)',
                         8: 'AOL (Any One Location)',
                         16: 'SW (Some Where)'}
    is_multi_member_field = False
    coord_keys = probtype_lookup.get(field.threshold_type, None)
    if not coord_keys:
        return is_multi_member_field
    if not coord_keys.get('units', None):
        coord_keys['units'] = DEFAULT_UNITS.get(field.field_code, None)
    coord_val = None
    if field.threshold_value_alt > -32766.:
        coord_val = field.threshold_value_alt
    elif field.threshold_value > -32766.:
        coord_val = field.threshold_value
    if field.chead.find('pc') > 0:
        try:
            coord_val = [int(x.strip('pc')) for x in field.chead.split(' ')
                         if x.find('pc') > 0][0]
        except IndexError:
            pass
    if coord_val is not None:
        if field.threshold_fuzziness > -32766.:
            bounds = [coord_val * field.threshold_fuzziness,
                      coord_val * (2. - field.threshold_fuzziness)]
        else:
            bounds = None
        new_coord = iris.coords.AuxCoord(
            np.array(coord_val, dtype=np.float32),
            bounds=np.array(bounds, dtype=np.float32),
            **coord_keys)
        cube.add_aux_coord(new_coord)
        cube.units = '1'
        is_multi_member_field = True

    if field.probability_method > 0:
        probability_attributes = []
        num = field.probability_method
        for key in sorted(probmethod_lookup.keys(), reverse=True):
            if num >= key:
                probability_attributes.append(probmethod_lookup[key])
                num = num - key
        cube.attributes['Probability methods'] = probability_attributes
    if field.member_count == 1:
        is_multi_member_field = False
    return is_multi_member_field


def soil_type_coord(cube, field):
    """Add soil type as a coord if appropriate"""
    if field.threshold_type != 0:
        soil_name = SOIL_TYPE_CODES.get(field.soil_type, None)
        if soil_name:
            cube.add_aux_coord(iris.coords.AuxCoord(
                soil_name,
                standard_name='soil_type', units=None))


def time_averaging(cube, field):
    """Decode the averagingtype code - similar to the PP LBPROC code."""
    num = field.averagingtype
    averaging_attributes = []
    for key in sorted(TIME_AVERAGING_CODES.keys(), reverse=True):
        if num >= key:
            averaging_attributes.append(TIME_AVERAGING_CODES[key])
            num = num - key
    if averaging_attributes:
        cube.attributes['processing'] = averaging_attributes


def run(field):
    """
    Convert a NIMROD field to an Iris cube.

    Args:

        * field - a :class:`~iris.fileformats.nimrod.NimrodField`

    Returns:

        * A new :class:`~iris.cube.Cube`, created from the NimrodField.

    """
    cube = iris.cube.Cube(field.data.astype(np.float32))

    name(cube, field)
    units(cube, field)

    mask_cube(cube, field)

    # time
    time(cube, field)
    reference_time(cube, field)

    experiment(cube, field)

    # horizontal grid
    horizontal_grid(cube, field)

    # vertical
    vertical_coord(cube, field)

    # add other stuff, if present
    soil_type_coord(cube, field)
    threshold_coord(cube, field)
    if not probability_coord(cube, field):
        ensemble_member(cube, field)
    time_averaging(cube, field)
    attributes(cube, field)

    origin_corner(cube, field)

    cube.data = cube.data.astype(np.float32)

    return cube
