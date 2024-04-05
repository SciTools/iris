# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Rules for converting NIMROD fields into cubes."""

import re
import string
import warnings

import cf_units
import cftime
import numpy as np

import iris
import iris.coord_systems
from iris.coords import DimCoord
from iris.exceptions import (
    CoordinateNotFoundError,
    TranslationError,
)
from iris.warnings import IrisNimrodTranslationWarning

__all__ = ["run"]


NIMROD_DEFAULT = -32767.0

TIME_UNIT = cf_units.Unit(
    "seconds since 1970-01-01 00:00:00", calendar=cf_units.CALENDAR_STANDARD
)


class TranslationWarning(IrisNimrodTranslationWarning):
    """Backwards compatible form of :class:`iris.warnings.IrisNimrodTranslationWarning`."""

    # TODO: remove at the next major release.
    pass


def is_missing(field, value):
    """Return True if value matches an "is-missing" number."""
    return any(np.isclose(value, [field.int_mdi, field.float32_mdi, NIMROD_DEFAULT]))


def name(cube, field, handle_metadata_errors):
    """Set the cube's name from the field.

    Modifies the Nimrod object title based on other meta-data in the
    Nimrod field and known use cases.

    Adds "mean_of" or "standard_deviation_of_" to the cube name if appropriate.

    """
    title_from_field_code = {
        12: "air_pressure",
        27: "snow fraction",
        28: "snow probability",
        58: "temperature",
        61: "amount_of_precipitation",
        63: "rate_of_precipitation",
        29: "fog fraction",
        101: "snow_melting_level_above_sea_level",
        102: "rain_melted_level_above_sea_level",
        155: "Visibility",
        156: "Worst visibility in grid point",
        161: "minimum_cloud_base",
        172: "cloud_area_fraction_in_atmosphere",
        218: "snowfall",
        421: "precipitation type",
        501: "vector_wind_shear",
        508: "low_level_jet_u_component",
        509: "low_level_jet_curvature",
        514: "low_level_jet_v_component",
        804: "wind speed",
        806: "wind direction",
        817: "wind_speed_of_gust",
        821: "Probabilistic Gust Risk Analysis from Observations",
    }
    if handle_metadata_errors:
        cube_title = title_from_field_code.get(field.field_code, field.title)
    else:
        cube_title = field.title
    if field.ensemble_member == -98:
        if not re.match("(?i)^.*(mean).*", cube_title):
            cube_title = "mean_of_" + cube_title
    if field.ensemble_member == -99:
        if not re.match("(?i)^.*(spread).*", cube_title):
            cube_title = "standard_deviation_of_" + cube_title

    cube.rename(remove_unprintable_chars(cube_title))


def remove_unprintable_chars(input_str):
    """Remove unprintable characters from a string and return the result."""
    return "".join(c if c in string.printable else " " for c in input_str).strip()


def units(cube, field):
    """Set the cube's units from the field.

    Takes into account nimrod unit strings of the form unit*?? where the data
    needs to converted by dividing by ??. Also converts units we know Iris
    can't handle into appropriate units Iris can handle. This is mostly when
    there is an inappropriate capital letter in the unit in the Nimrod file.
    Some units still can't be handled by Iris so in these cases empty strings
    are added as the cube's unit. The most notable unit Iris can't handle is
    oktas for cloud cover.

    Unhandled units are stored in an "invalid_units" attribute instead.

    """
    unit_exception_dictionary = {
        "Knts": "knots",
        "knts": "knots",
        "J/Kg": "J/kg",
        "logical": "1",
        "Code": "1",
        "mask": "1",
        "mb": "hPa",
        "unitless": "1",
        "Fraction": "1",
        "index": "1",
        "Beaufort": "",
        "mmh2o": "kg/m2",
        "n/a": "1",
    }

    field_units = remove_unprintable_chars(field.units)
    if field_units == "m/2-25k":
        # Handle strange visibility units
        cube.data = (cube.data.astype(np.float32) + 25000.0) * 2
        field_units = "m"
    if "*" in field_units:
        # Split into unit string and integer
        unit_list = field_units.split("*")
        if "^" in unit_list[1]:
            # Split out magnitude
            unit_sublist = unit_list[1].split("^")
            cube.data = cube.data.astype(np.float32) / float(unit_sublist[0]) ** float(
                unit_sublist[1]
            )
        else:
            cube.data = cube.data.astype(np.float32) / float(unit_list[1])
        field_units = unit_list[0]
    if "ug/m3E1" in field_units:
        # Split into unit string and integer
        unit_list = field_units.split("E")
        cube.data = cube.data.astype(np.float32) / 10.0
        field_units = unit_list[0]
    if field_units == "%":
        # Convert any percentages into fraction
        field_units = "1"
        cube.data = cube.data.astype(np.float32) / 100.0
    if field_units == "oktas":
        field_units = "1"
        cube.data = cube.data.astype(np.float32) / 8.0
    if field_units == "dBZ":
        # cf_units doesn't recognise decibels (dBZ), but does know BZ
        field_units = "BZ"
        cube.data = cube.data.astype(np.float32) / 10.0
    if field_units == "g/Kg":
        field_units = "kg/kg"
        cube.data = cube.data.astype(np.float32) / 1000.0
    if not field_units:
        if field.field_code == 8:
            # Relative Humidity data are unitless, but not "unknown"
            field_units = "1"
        if field.field_code in [505, 515]:
            # CAPE units are not always set correctly. Assume J/kg
            field_units = "J/kg"
    if field_units in unit_exception_dictionary.keys():
        field_units = unit_exception_dictionary[field_units]
    if len(field_units) > 0 and field_units[0] == "/":
        # Deal with the case where the units are of the form '/unit' eg
        # '/second' in the Nimrod file. This converts to the form unit^-1
        field_units = field_units[1:] + "^-1"
    try:
        cube.units = field_units
    except ValueError:
        # Just add it as an attribute.
        warnings.warn(
            "Unhandled units '{0}' recorded in cube attributes.".format(field_units),
            category=IrisNimrodTranslationWarning,
        )
        cube.attributes["invalid_units"] = field_units


def time(cube, field):
    """Add a time coord to the cube based on validity time and time-window."""
    if field.vt_year <= 0:
        # Some ancillary files, eg land sea mask do not
        # have a validity time.
        return
    else:
        valid_date = cftime.datetime(
            field.vt_year,
            field.vt_month,
            field.vt_day,
            field.vt_hour,
            field.vt_minute,
            field.vt_second,
        )
    point = np.around(TIME_UNIT.date2num(valid_date)).astype(np.int64)

    period_seconds = None
    if field.period_minutes == 32767:
        period_seconds = field.period_seconds
    elif not is_missing(field, field.period_minutes) and field.period_minutes != 0:
        period_seconds = field.period_minutes * 60
    if period_seconds:
        bounds = np.array([point - period_seconds, point], dtype=np.int64)
    else:
        bounds = None

    time_coord = DimCoord(
        points=point, bounds=bounds, standard_name="time", units=TIME_UNIT
    )

    cube.add_aux_coord(time_coord)


def reference_time(cube, field):
    """Add a 'reference time' to the cube, if present in the field."""
    if not is_missing(field, field.dt_year) and field.dt_year > 0:
        data_date = cftime.datetime(
            field.dt_year,
            field.dt_month,
            field.dt_day,
            field.dt_hour,
            field.dt_minute,
        )
        ref_time_coord = DimCoord(
            np.array(np.round(TIME_UNIT.date2num(data_date)), dtype=np.int64),
            standard_name="forecast_reference_time",
            units=TIME_UNIT,
        )

        cube.add_aux_coord(ref_time_coord)


def forecast_period(cube):
    """Add forecast_period coord based on existing time and forecast_reference_time coords.

    Must be run after time() and reference_time()

    """
    try:
        time_coord = cube.coord("time")
        frt_coord = cube.coord("forecast_reference_time")
    except CoordinateNotFoundError:
        return
    time_delta = time_coord.cell(0).point - frt_coord.cell(0).point

    points = np.array(time_delta.total_seconds(), dtype=np.int32)
    forecast_period_unit = cf_units.Unit("second")
    if cube.coord("time").has_bounds():
        time_window = time_coord.cell(0).bound
        time_window = time_window[1] - time_window[0]
        bounds = np.array(
            [points - time_window.total_seconds(), points], dtype=np.int32
        )
    else:
        bounds = None
    cube.add_aux_coord(
        iris.coords.AuxCoord(
            points,
            standard_name="forecast_period",
            bounds=bounds,
            units=forecast_period_unit,
        )
    )


def mask_cube(cube, field):
    """Update cube.data to be a masked array if appropriate."""
    dtype = cube.dtype
    masked_points = None
    if field.datum_type == 1:
        # field.data are integers
        masked_points = field.data == field.int_mdi
    elif field.datum_type == 0:
        # field.data are floats
        masked_points = np.isclose(field.data, field.float32_mdi)
    if np.any(masked_points):
        cube.data = np.ma.masked_array(cube.data, mask=masked_points, dtype=dtype)


def experiment(cube, field):
    """Add an 'experiment number' to the cube, if present in the field."""
    if not is_missing(field, field.experiment_num):
        cube.add_aux_coord(
            DimCoord(field.experiment_num, long_name="experiment_number", units="1")
        )


def proj_biaxial_ellipsoid(field, handle_metadata_errors):
    """Return correct dict of arguments needed to define an iris.coord_systems.GeogCS.

    Based firstly on the value given by ellipsoid, then by grid if ellipsoid is
    missing, select the right pre-defined ellipsoid dictionary (Airy_1830 or
    international_1924).

    References
    ----------
    Airy 1830: https://georepository.com/ellipsoid_7001/Airy-1830.html
    International 1924: https://georepository.com/ellipsoid_7022/International-1924.html

    """
    airy_1830 = {
        "semi_major_axis": 6377563.396,
        "semi_minor_axis": 6356256.910,
    }
    international_1924 = {
        "semi_major_axis": 6378388.000,
        "semi_minor_axis": 6356911.946,
    }
    if field.proj_biaxial_ellipsoid == 0:
        ellipsoid = airy_1830
    elif field.proj_biaxial_ellipsoid == 1:
        ellipsoid = international_1924
    elif is_missing(field, field.proj_biaxial_ellipsoid) and handle_metadata_errors:
        if field.horizontal_grid_type == 0:
            ellipsoid = airy_1830
        elif field.horizontal_grid_type == 1 or field.horizontal_grid_type == 4:
            ellipsoid = international_1924
        else:
            raise TranslationError(
                """Unsupported grid type, only NG, EuroPP
                                     and lat/long are possible"""
            )
    else:
        raise TranslationError(
            "Ellipsoid not supported, proj_biaxial_ellipsoid:{}, "
            "horizontal_grid_type:{}".format(
                field.proj_biaxial_ellipsoid, field.horizontal_grid_type
            )
        )
    return ellipsoid


def set_british_national_grid_defaults(field, handle_metadata_errors):
    """Check for missing coord-system meta-data and set default values.

    Check for missing coord-system meta-data and set default values for
    the Ordnance Survey GB Transverse Mercator projection. Some Radarnet
    files are missing these.

    """
    if handle_metadata_errors:
        if is_missing(field, field.true_origin_latitude):
            field.true_origin_latitude = 49.0
        if is_missing(field, field.true_origin_longitude):
            field.true_origin_longitude = -2.0
        if is_missing(field, field.true_origin_easting) or np.isclose(
            # Some old files misquote the value in km instead of m
            field.true_origin_easting,
            400.0,
        ):
            field.true_origin_easting = 400000.0
        if is_missing(field, field.true_origin_northing) or np.isclose(
            # Some old files misquote the value in km instead of m
            field.true_origin_northing,
            -100.0,
        ):
            field.true_origin_northing = -100000.0
        if is_missing(field, field.tm_meridian_scaling):
            field.tm_meridian_scaling = 0.9996012717

    ng_central_meridian_sf_dp = 0.9996012717
    if abs(field.tm_meridian_scaling - ng_central_meridian_sf_dp) < 1.0e-04:
        # Update the National Grid scaling factor to double
        # precision accuracy to improve the accuracy of
        # reprojection calculations that use it.
        field.tm_meridian_scaling = ng_central_meridian_sf_dp


def coord_system(field, handle_metadata_errors):
    """Define the coordinate system for the field.

    Handles Transverse Mercator, Universal Transverse Mercator and Plate Carree.

    Transverse Mercator projections will default to the British National Grid if any
    parameters are missing.
    """
    ellipsoid = proj_biaxial_ellipsoid(field, handle_metadata_errors)

    if field.horizontal_grid_type == 0:
        # Check for missing grid meta-data and insert OSGB definitions.
        # Some Radarnet files are missing these.
        set_british_national_grid_defaults(field, handle_metadata_errors)
    if field.horizontal_grid_type == 0 or field.horizontal_grid_type == 4:
        crs_args = (
            field.true_origin_latitude,
            field.true_origin_longitude,
            field.true_origin_easting,
            field.true_origin_northing,
            field.tm_meridian_scaling,
        )
        if any([is_missing(field, v) for v in crs_args]):
            warnings.warn(
                "Coordinate Reference System is not completely defined. "
                "Plotting and reprojection may be impaired.",
                category=IrisNimrodTranslationWarning,
            )
        coord_sys = iris.coord_systems.TransverseMercator(
            *crs_args, iris.coord_systems.GeogCS(**ellipsoid)
        )
    elif field.horizontal_grid_type == 1:
        coord_sys = iris.coord_systems.GeogCS(**ellipsoid)
    else:
        coord_sys = None
    return coord_sys


def horizontal_grid(cube, field, handle_metadata_errors):
    """Add X and Y coordinates to the cube.

    Handles Transverse Mercator, Universal Transverse Mercator and Plate Carree.

    coordinate reference system is supplied by coord_system(field)

    Must be run AFTER origin_corner()
    """
    crs = coord_system(field, handle_metadata_errors)
    if field.horizontal_grid_type == 0 or field.horizontal_grid_type == 4:
        units_name = "m"
        x_coord_name = "projection_x_coordinate"
        y_coord_name = "projection_y_coordinate"
    elif field.horizontal_grid_type == 1:
        units_name = "degrees"
        x_coord_name = "longitude"
        y_coord_name = "latitude"
    else:
        raise TranslationError(
            "Horizontal grid type {} not implemented".format(field.horizontal_grid_type)
        )
    points = np.linspace(
        field.x_origin,
        field.x_origin + field.num_cols * field.column_step,
        field.num_cols,
        endpoint=False,
        dtype=np.float32,
    )
    x_coord = DimCoord(
        points, standard_name=x_coord_name, units=units_name, coord_system=crs
    )
    cube.add_dim_coord(x_coord, 1)
    points = np.linspace(
        field.y_origin - (field.num_rows - 1) * field.row_step,
        field.y_origin,
        field.num_rows,
        endpoint=True,
        dtype=np.float32,
    )
    y_coord = DimCoord(
        points, standard_name=y_coord_name, units=units_name, coord_system=crs
    )
    cube.add_dim_coord(y_coord, 0)


def vertical_coord(cube, field):
    """Add a vertical coord to the cube, with bounds.

    Add a vertical coord to the cube, with bounds, if appropriate.
    Handles special numbers for "at-sea-level" (8888) and "at-ground-level"
    (9999).

    """
    # vertical_codes contains conversions from the Nimrod Documentation for the
    # header entry 20 for the vertical coordinate type
    # Unhandled vertical_codes values (no use case identified):
    #  3: ['sigma', 'model level'],
    #  4: ['eta', 'model level'],
    #  5: ['radar beam number', 'unknown'],
    #  7: ['potential temperature', 'unknown'],
    #  8: ['equivalent potential temperature', 'unknown'],
    #  9: ['wet bulb potential temperature', 'unknown'],
    #  10: ['potential vorticity', 'unknown'],
    #  11: ['cloud boundary', 'unknown'],
    vertical_codes = {
        0: {
            "standard_name": "height",
            "units": "m",
            "attributes": {"positive": "up"},
        },
        1: {
            "standard_name": "altitude",
            "units": "m",
            "attributes": {"positive": "up"},
        },
        2: {
            "standard_name": "air_pressure",
            "units": "hPa",
            "attributes": {"positive": "down"},
        },
        6: {"standard_name": "air_temperature", "units": "K"},
        12: {
            "long_name": "depth_below_ground",
            "units": "m",
            "attributes": {"positive": "down"},
        },
    }
    if all(
        [
            is_missing(field, x)
            for x in [
                field.vertical_coord,
                field.vertical_coord_type,
                field.reference_vertical_coord,
                field.reference_vertical_coord_type,
            ]
        ]
    ):
        return

    if (
        not is_missing(field, field.reference_vertical_coord_type)
        and field.reference_vertical_coord_type != field.vertical_coord_type
        and not is_missing(field, field.reference_vertical_coord)
    ):
        msg = (
            "Unmatched vertical coord types "
            f"{field.vertical_coord_type} != {field.reference_vertical_coord_type}. "
            f"Assuming {field.vertical_coord_type}"
        )
        warnings.warn(msg, category=IrisNimrodTranslationWarning)

    coord_point = field.vertical_coord
    if coord_point == 8888.0:
        if "sea_level" not in cube.name():
            cube.rename(f"{cube.name()}_at_mean_sea_level")
        coord_point = 0.0
        if np.isclose(field.reference_vertical_coord, 8888.0) or is_missing(
            field, field.reference_vertical_coord
        ):
            # This describes a surface field. No changes needed.
            return

    coord_args = vertical_codes.get(field.vertical_coord_type, None)
    if np.isclose(coord_point, 9999.0):
        if np.isclose(field.reference_vertical_coord, 9999.0) or is_missing(
            field, field.reference_vertical_coord
        ):
            # This describes a surface field. No changes needed.
            return
        # A bounded vertical coord starting from the surface
        coord_point = 0.0
        coord_args = vertical_codes.get(field.reference_vertical_coord_type, None)
    coord_point = np.array(coord_point, dtype=np.float32)
    if (
        field.reference_vertical_coord >= 0.0
        and field.reference_vertical_coord != coord_point
    ):
        bounds = np.array(
            [coord_point, field.reference_vertical_coord], dtype=np.float32
        )
    else:
        bounds = None

    if coord_args:
        new_coord = iris.coords.AuxCoord(coord_point, bounds=bounds, **coord_args)
        # Add coordinate to cube
        cube.add_aux_coord(new_coord)
        return

    warnings.warn(
        "Vertical coord {!r} not yet handled".format(field.vertical_coord_type),
        category=TranslationWarning,
    )


def ensemble_member(cube, field):
    """Add an 'ensemble member' coord to the cube, if present in the field."""
    ensemble_member_value = field.ensemble_member

    if ensemble_member_value in [-98, -99]:
        # ignore these special values handled in name()
        return
    if not is_missing(field, ensemble_member_value):
        cube.add_aux_coord(
            DimCoord(
                np.array(ensemble_member_value, dtype=np.int32),
                "realization",
                units="1",
            )
        )


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
            if is_missing(field, value):
                return
            if "radius" in item:
                value = f"{value} km"
            cube.attributes[item] = value

    add_attr("nimrod_version")
    add_attr("field_code")
    add_attr("num_model_levels")
    add_attr("sat_calib")
    add_attr("sat_space_count")
    add_attr("ducting_index")
    add_attr("elevation_angle")
    cube_source = field.source.strip()

    # Handle a few known meta-data errors:
    if field.field_code == 501:
        cube_source = ""
    if field.field_code == 821:
        cube_source = "Nimrod pwind routine"
    if field.source.strip() == "pwind":
        cube_source = "Nimrod pwind routine"
    for key in [
        "neighbourhood_radius",
        "recursive_filter_iterations",
        "recursive_filter_alpha",
        "threshold_vicinity_radius",
        "probability_period_of_event",
    ]:
        add_attr(key)

    # Remove member number from cube_source. This commonly takes the form ek04 where ek
    # indicates the model and 04 is the realization number. As the number is represented
    # by a realization coord, stripping it from here allows cubes to be merged.
    match = re.match(r"^(?P<model_code>\w\w)(?P<realization>\d\d)$", cube_source)
    try:
        r_coord = cube.coord("realization")
    except CoordinateNotFoundError:
        r_coord = None
    if match is not None:
        if r_coord:
            if int(match["realization"]) == r_coord.points[0]:
                cube_source = match["model_code"]
    cube.attributes["source"] = cube_source
    cube.attributes["title"] = "Unknown"
    cube.attributes["institution"] = "Met Office"


def known_threshold_coord(field):
    """Supply known threshold coord meta-data for known use cases.

    threshold_value_alt exists because some meta-data are mis-assigned in the
    Nimrod data.

    """
    coord_keys = {}
    if (
        field.field_code == 161
        and field.threshold_value >= 0.0
        and "pc" not in field.title
    ):
        coord_keys = {"var_name": "threshold"}
        if field.threshold_value_alt > 8.0:
            coord_keys["standard_name"] = "height"
            coord_keys["units"] = "metres"
        else:
            coord_keys["standard_name"] = "cloud_area_fraction"
            coord_keys["units"] = "oktas"
    elif field.field_code == 29 and field.threshold_value >= 0.0:
        if is_missing(field, field.threshold_type):
            coord_keys = {
                "standard_name": "visibility_in_air",
                "var_name": "threshold",
                "units": "metres",
            }
        else:
            coord_keys = {
                "long_name": "fog_fraction",
                "var_name": "threshold",
                "units": "1",
            }
    elif (
        field.field_code == 422
        and field.threshold_value >= 0.0
        and is_missing(field, field.threshold_type)
    ):
        coord_keys = {"long_name": "radius_of_max", "units": "km"}
    elif field.field_code == 821:
        coord_keys = {
            "standard_name": "wind_speed_of_gust",
            "var_name": "threshold",
            "units": "m/s",
        }
    return coord_keys


def probability_coord(cube, field, handle_metadata_errors):
    """Add a coord relating to probability meta-data from the header to the cube.

    Add a coord relating to probability meta-data from the header to the
    cube if appropriate.

    Must be run after the name method.

    """
    probtype_lookup = {
        1: {
            "var_name": "threshold",
            "attributes": {"relative_to_threshold": "above"},
        },
        2: {
            "var_name": "threshold",
            "attributes": {"relative_to_threshold": "below"},
        },
        3: {"long_name": "percentile", "units": "1"},
        4: {
            "var_name": "threshold",
            "attributes": {"relative_to_threshold": "equal"},
        },
    }
    probmethod_lookup = {
        1: "AOT (Any One Time)",
        2: "ST (Some Time)",
        4: "AT (All Time)",
        8: "AOL (Any One Location)",
        16: "SW (Some Where)",
    }
    # The units for the threshold coord are not defined in the Nimrod meta-data.
    # These represent the known use-cases.
    units_from_field_code = {
        817: "m s^-1",
        804: "knots",
        422: "min^-1",
        421: "1",
        218: "cm",
        172: "oktas",
        155: "m",
        101: "m",
        63: "mm hr^-1",
        61: "mm",
        58: "Celsius",
        12: "mb",
        8: "1",
    }
    is_probability_field = False
    coord_keys = probtype_lookup.get(field.threshold_type, {})
    if coord_keys.get("var_name") == "threshold":
        is_probability_field = True
    if handle_metadata_errors:
        coord_keys.update(known_threshold_coord(field))
    if not coord_keys.get("units"):
        coord_keys["units"] = units_from_field_code.get(field.field_code, "unknown")
    coord_val = None
    # coord_val could come from the threshold_value or threshold_value_alt:
    if field.threshold_value_alt > -32766.0:
        coord_val = field.threshold_value_alt
    elif field.threshold_value > -32766.0:
        coord_val = field.threshold_value

    # coord_val could also be encoded in the cube name if we have a percentile
    # (this overrides the threshold_value which may be unrelated in the case of
    # the 50th %ile of 3okta cloud cover)
    if (
        coord_keys.get("long_name") == "percentile"
        and cube.name().find("pc") > 0
        and handle_metadata_errors
    ):
        try:
            coord_val = [
                int(x.strip("pc")) for x in cube.name().split(" ") if x.find("pc") > 0
            ][0]
        except IndexError:
            pass

    # If we found a coord_val, build the coord (with bounds) and add to cube)
    if coord_val is not None:
        if not is_missing(field, field.threshold_fuzziness):
            bounds = [
                coord_val * field.threshold_fuzziness,
                coord_val * (2.0 - field.threshold_fuzziness),
            ]
            bounds = np.array(bounds, dtype=np.float32)
            # TODO: Enable filtering of zero-length bounds once Iris doesn't strip bounds
            #  in merge_cube
            # if np.isclose(bounds[0], bounds[1]):
            #    bounds = None
        else:
            bounds = None
        if coord_keys.get("units", None) == "oktas":
            coord_keys["units"] = "1"
            coord_val /= 8.0
            if bounds is not None:
                bounds /= 8.0
        if coord_keys["units"] == "unknown":
            coord_name = coord_keys.get(
                "standard_name",
                coord_keys.get("long_name", coord_keys.get("var_name", None)),
            )
            warnings.warn(
                f"No default units for {coord_name} coord of {cube.name()}. "
                "Meta-data may be incomplete.",
                category=IrisNimrodTranslationWarning,
            )
        new_coord = iris.coords.AuxCoord(
            np.array(coord_val, dtype=np.float32), bounds=bounds, **coord_keys
        )
        cube.add_aux_coord(new_coord)
        if field.threshold_type == 3:
            pass
        else:
            if is_probability_field:
                # Some probability fields have inappropriate units (those of the threshold)
                if "%" in cube.name():
                    # If the cube name has % in it, convert to fraction
                    cube.rename(cube.name().replace("%", "fraction"))
                    cube.data = cube.data.astype(np.float32) / 100.0
                cube.units = "1"
                cube.rename(f"probability_of_{cube.name()}")

    if field.probability_method > 0:
        probability_attributes = []
        num = field.probability_method
        for key in sorted(probmethod_lookup.keys(), reverse=True):
            if num >= key:
                probability_attributes.append(probmethod_lookup[key])
                num = num - key
        cube.attributes["Probability methods"] = probability_attributes
    return


def soil_type_coord(cube, field):
    """Add soil type as a coord if appropriate."""
    soil_type_codes = {
        1: "broadleaf_tree",
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
    soil_name = soil_type_codes.get(field.soil_type, None)
    if soil_name:
        cube.add_aux_coord(
            iris.coords.AuxCoord(soil_name, standard_name="soil_type", units=None)
        )


def time_averaging(cube, field):
    """Decode the averagingtype code - similar to the PP LBPROC code."""
    time_averaging_codes = {
        8192: "maximum in period",
        4096: "minimum in period",
        2048: "unknown(2048)",
        1024: "unknown(1024)",
        512: "time lagged",
        256: "extrapolation",
        128: "accumulation or average",
        64: "from UM 150m",
        32: "scaled to UM resolution",
        16: "averaged over multiple surface types",
        8: "only observations used",
        4: "smoothed",
        2: "cold bias applied",
        1: "warm bias applied",
    }

    num = field.averagingtype
    averaging_attributes = []
    for key in sorted(time_averaging_codes.keys(), reverse=True):
        if num >= key:
            averaging_attributes.append(time_averaging_codes[key])
            num = num - key
    if averaging_attributes:
        cube.attributes["processing"] = averaging_attributes


def run(field, handle_metadata_errors=True):
    """Convert a NIMROD field to an Iris cube.

    Parameters
    ----------
    field : :class:`~iris.fileformats.nimrod.NimrodField`
    handle_metadata_errors : bool, default=True
        Set to False to omit handling of known meta-data deficiencies
        in Nimrod-format data.

    Returns
    -------
    :class:`~iris.cube.Cube`
        A new :class:`~iris.cube.Cube`, created from the NimrodField.

    """
    cube = iris.cube.Cube(field.data)

    name(cube, field, handle_metadata_errors)
    mask_cube(cube, field)
    units(cube, field)

    # time
    time(cube, field)
    reference_time(cube, field)
    forecast_period(cube)

    experiment(cube, field)

    # horizontal grid
    origin_corner(cube, field)
    horizontal_grid(cube, field, handle_metadata_errors)

    # vertical
    vertical_coord(cube, field)

    # add other stuff, if present
    soil_type_coord(cube, field)
    probability_coord(cube, field, handle_metadata_errors)
    ensemble_member(cube, field)
    time_averaging(cube, field)
    attributes(cube, field)

    return cube
