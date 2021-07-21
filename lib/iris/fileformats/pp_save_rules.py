# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

import warnings

import cftime

import iris
from iris.aux_factory import HybridHeightFactory, HybridPressureFactory
from iris.fileformats._ff_cross_references import STASH_TRANS
from iris.fileformats._pp_lbproc_pairs import LBPROC_MAP
from iris.fileformats.rules import (
    aux_factory,
    has_aux_factory,
    scalar_cell_method,
    scalar_coord,
    vector_coord,
)
from iris.fileformats.um_cf_map import CF_TO_LBFC
from iris.util import is_regular, regular_step


def _basic_coord_system_rules(cube, pp):
    """
    Rules for setting the coord system of the PP field.

    Args:
        cube: the cube being saved as a series of PP fields.
        pp: the current PP field having save rules applied.

    Returns:
        The PP field with updated metadata.

    """
    if (
        cube.coord_system("GeogCS") is not None
        or cube.coord_system(None) is None
    ):
        pp.bplat = 90
        pp.bplon = 0
    elif cube.coord_system("RotatedGeogCS") is not None:
        pp.bplat = cube.coord_system("RotatedGeogCS").grid_north_pole_latitude
        pp.bplon = cube.coord_system("RotatedGeogCS").grid_north_pole_longitude
    return pp


def _um_version_rules(cube, pp):
    from_um_str = "Data from Met Office Unified Model"
    source_attr = cube.attributes.get("source")
    if source_attr is not None:
        um_version = source_attr.rsplit(from_um_str, 1)

    if (
        "um_version" not in cube.attributes
        and "source" in cube.attributes
        and len(um_version) > 1
        and len(um_version[1]) == 0
    ):
        # UM - no version number.
        pp.lbsrce = 1111
    elif (
        "um_version" not in cube.attributes
        and "source" in cube.attributes
        and len(um_version) > 1
        and len(um_version[1]) > 0
    ):
        # UM - with version number.
        pp.lbsrce = int(float(um_version[1]) * 1000000) + 1111
    elif "um_version" in cube.attributes:
        # UM - from 'um_version' attribute.
        um_ver_minor = int(cube.attributes["um_version"].split(".")[1])
        um_ver_major = int(cube.attributes["um_version"].split(".")[0])
        pp.lbsrce = 1111 + 10000 * um_ver_minor + 1000000 * um_ver_major
    return pp


def _stash_rules(cube, pp):
    """
    Attributes rules for setting the STASH attribute of the PP field.

    Args:
        cube: the cube being saved as a series of PP fields.
        pp: the current PP field having save rules applied.

    Returns:
        The PP field with updated metadata.

    """
    if "STASH" in cube.attributes:
        stash = cube.attributes["STASH"]
        if isinstance(stash, iris.fileformats.pp.STASH):
            pp.lbuser[3] = 1000 * (stash.section or 0) + (stash.item or 0)
            pp.lbuser[6] = stash.model or 0
    return pp


def _general_time_rules(cube, pp):
    """
    Rules for setting time metadata of the PP field.

    Args:
        cube: the cube being saved as a series of PP fields.
        pp: the current PP field having save rules applied.

    Returns:
        The PP field with updated metadata.

    """
    time_coord = scalar_coord(cube, "time")
    fp_coord = scalar_coord(cube, "forecast_period")
    frt_coord = scalar_coord(cube, "forecast_reference_time")
    clim_season_coord = scalar_coord(cube, "clim_season")

    cm_time_mean = scalar_cell_method(cube, "mean", "time")
    cm_time_min = scalar_cell_method(cube, "minimum", "time")
    cm_time_max = scalar_cell_method(cube, "maximum", "time")

    # No forecast.
    if time_coord is not None and fp_coord is None and frt_coord is None:
        pp.lbtim.ia = 0
        pp.lbtim.ib = 0
        pp.t1 = time_coord.units.num2date(time_coord.points[0])
        pp.t2 = cftime.datetime(0, 0, 0)

    # Forecast.
    if (
        time_coord is not None
        and not time_coord.has_bounds()
        and fp_coord is not None
    ):
        pp.lbtim.ia = 0
        pp.lbtim.ib = 1
        pp.t1 = time_coord.units.num2date(time_coord.points[0])
        pp.t2 = time_coord.units.num2date(
            time_coord.points[0] - fp_coord.points[0]
        )
        pp.lbft = fp_coord.points[0]

    # Time mean (non-climatological).
    # XXX This only works when we have a single timestep.
    if (
        time_coord is not None
        and time_coord.has_bounds()
        and clim_season_coord is None
        and fp_coord is not None
        and fp_coord.has_bounds()
    ):
        # XXX How do we know *which* time to use if there are more than
        # one? *Can* there be more than one?
        pp.lbtim.ib = 2
        pp.t1 = time_coord.units.num2date(time_coord.bounds[0, 0])
        pp.t2 = time_coord.units.num2date(time_coord.bounds[0, 1])
        pp.lbft = fp_coord.units.convert(fp_coord.bounds[0, 1], "hours")

    if (
        time_coord is not None
        and time_coord.has_bounds()
        and clim_season_coord is None
        and fp_coord is None
        and frt_coord is not None
    ):
        # Handle missing forecast period using time and forecast ref time.
        pp.lbtim.ib = 2
        pp.t1 = time_coord.units.num2date(time_coord.bounds[0, 0])
        pp.t2 = time_coord.units.num2date(time_coord.bounds[0, 1])
        stop = time_coord.units.convert(
            time_coord.bounds[0, 1], "hours since epoch"
        )
        start = frt_coord.units.convert(
            frt_coord.points[0], "hours since epoch"
        )
        pp.lbft = stop - start

    if (
        time_coord is not None
        and time_coord.has_bounds()
        and clim_season_coord is None
        and cm_time_mean is not None
    ):
        pp.lbtim.ib = 2
        pp.t1 = time_coord.units.num2date(time_coord.bounds[0, 0])
        pp.t2 = time_coord.units.num2date(time_coord.bounds[0, 1])

    if (
        time_coord is not None
        and time_coord.has_bounds()
        and clim_season_coord is None
        and cm_time_mean is not None
        and cm_time_mean.intervals != ()
        and cm_time_mean.intervals[0].endswith("hour")
    ):
        pp.lbtim.ia = int(cm_time_mean.intervals[0][:-5])

    if (
        time_coord is not None
        and time_coord.has_bounds()
        and clim_season_coord is None
        and (fp_coord is not None or frt_coord is not None)
        and (
            cm_time_mean is None
            or cm_time_mean.intervals == ()
            or not cm_time_mean.intervals[0].endswith("hour")
        )
    ):
        pp.lbtim.ia = 0

    # If the cell methods contain a minimum then overwrite lbtim.ia with this
    # interval.
    if (
        time_coord is not None
        and time_coord.has_bounds()
        and clim_season_coord is None
        and (fp_coord is not None or frt_coord is not None)
        and cm_time_min is not None
        and cm_time_min.intervals != ()
        and cm_time_min.intervals[0].endswith("hour")
    ):
        # Set lbtim.ia with the integer part of the cell method's interval
        # e.g. if interval is '24 hour' then lbtim.ia becomes 24.
        pp.lbtim.ia = int(cm_time_min.intervals[0][:-5])

    # If the cell methods contain a maximum then overwrite lbtim.ia with this
    # interval.
    if (
        time_coord is not None
        and time_coord.has_bounds()
        and clim_season_coord is None
        and (fp_coord is not None or frt_coord is not None)
        and cm_time_max is not None
        and cm_time_max.intervals != ()
        and cm_time_max.intervals[0].endswith("hour")
    ):
        # Set lbtim.ia with the integer part of the cell method's interval
        # e.g. if interval is '1 hour' then lbtim.ia becomes 1.
        pp.lbtim.ia = int(cm_time_max.intervals[0][:-5])

    if time_coord is not None and time_coord.has_bounds():
        lower_bound_yr = time_coord.units.num2date(
            time_coord.bounds[0, 0]
        ).year
        upper_bound_yr = time_coord.units.num2date(
            time_coord.bounds[0, 1]
        ).year
    else:
        lower_bound_yr = None
        upper_bound_yr = None

    # Climatological time means.
    if (
        time_coord is not None
        and time_coord.has_bounds()
        and lower_bound_yr == upper_bound_yr
        and fp_coord is not None
        and fp_coord.has_bounds()
        and clim_season_coord is not None
        and "clim_season" in cube.cell_methods[-1].coord_names
    ):
        # Climatological time mean - single year.
        pp.lbtim.ia = 0
        pp.lbtim.ib = 2
        pp.t1 = time_coord.units.num2date(time_coord.bounds[0, 0])
        pp.t2 = time_coord.units.num2date(time_coord.bounds[0, 1])
        pp.lbft = fp_coord.units.convert(fp_coord.bounds[0, 1], "hours")

    elif (
        time_coord is not None
        and time_coord.has_bounds()
        and lower_bound_yr != upper_bound_yr
        and fp_coord is not None
        and fp_coord.has_bounds()
        and clim_season_coord is not None
        and "clim_season" in cube.cell_methods[-1].coord_names
        and clim_season_coord.points[0] == "djf"
    ):
        # Climatological time mean - spanning years - djf.
        pp.lbtim.ia = 0
        pp.lbtim.ib = 3
        pp.t1 = time_coord.units.num2date(time_coord.bounds[0, 0])
        pp.t2 = time_coord.units.num2date(time_coord.bounds[0, 1])
        if pp.t1.month == 12:
            pp.t1 = cftime.datetime(pp.t1.year)
        else:
            pp.t1 = cftime.datetime(pp.t1.year - 1, 12, 1, 0, 0, 0)
        pp.t2 = cftime.datetime(pp.t2.year, 3, 1, 0, 0, 0)
        _conditional_warning(
            time_coord.bounds[0, 0] != time_coord.units.date2num(pp.t1),
            "modified t1 for climatological seasonal mean",
        )
        _conditional_warning(
            time_coord.bounds[0, 1] != time_coord.units.date2num(pp.t2),
            "modified t2 for climatological seasonal mean",
        )
        pp.lbft = fp_coord.units.convert(fp_coord.bounds[0, 1], "hours")

    elif (
        time_coord is not None
        and time_coord.has_bounds()
        and lower_bound_yr != upper_bound_yr
        and fp_coord is not None
        and fp_coord.has_bounds()
        and clim_season_coord is not None
        and "clim_season" in cube.cell_methods[-1].coord_names
        and clim_season_coord.points[0] == "mam"
    ):
        # Climatological time mean - spanning years - mam.
        pp.lbtim.ia = 0
        pp.lbtim.ib = 3
        # TODO: wut?
        pp.t1 = time_coord.units.num2date(time_coord.bounds[0, 0])
        pp.t2 = time_coord.units.num2date(time_coord.bounds[0, 1])
        pp.t1 = cftime.datetime(pp.t1.year, 3, 1, 0, 0, 0)
        pp.t2 = cftime.datetime(pp.t2.year, 6, 1, 0, 0, 0)
        _conditional_warning(
            time_coord.bounds[0, 0] != time_coord.units.date2num(pp.t1),
            "modified t1 for climatological seasonal mean",
        )
        _conditional_warning(
            time_coord.bounds[0, 1] != time_coord.units.date2num(pp.t2),
            "modified t2 for climatological seasonal mean",
        )
        pp.lbft = fp_coord.units.convert(fp_coord.bounds[0, 1], "hours")

    elif (
        time_coord is not None
        and time_coord.has_bounds()
        and lower_bound_yr != upper_bound_yr
        and fp_coord is not None
        and fp_coord.has_bounds()
        and clim_season_coord is not None
        and "clim_season" in cube.cell_methods[-1].coord_names
        and clim_season_coord.points[0] == "jja"
    ):
        # Climatological time mean - spanning years - jja.
        pp.lbtim.ia = 0
        pp.lbtim.ib = 3
        # TODO: wut?
        pp.t1 = time_coord.units.num2date(time_coord.bounds[0, 0])
        pp.t2 = time_coord.units.num2date(time_coord.bounds[0, 1])
        pp.t1 = cftime.datetime(pp.t1.year, 6, 1, 0, 0, 0)
        pp.t2 = cftime.datetime(pp.t2.year, 9, 1, 0, 0, 0)
        _conditional_warning(
            time_coord.bounds[0, 0] != time_coord.units.date2num(pp.t1),
            "modified t1 for climatological seasonal mean",
        )
        _conditional_warning(
            time_coord.bounds[0, 1] != time_coord.units.date2num(pp.t2),
            "modified t2 for climatological seasonal mean",
        )
        pp.lbft = fp_coord.units.convert(fp_coord.bounds[0, 1], "hours")

    elif (
        time_coord is not None
        and time_coord.has_bounds()
        and lower_bound_yr != upper_bound_yr
        and fp_coord is not None
        and fp_coord.has_bounds()
        and clim_season_coord is not None
        and "clim_season" in cube.cell_methods[-1].coord_names
        and clim_season_coord.points[0] == "son"
    ):
        # Climatological time mean - spanning years - son.
        pp.lbtim.ia = 0
        pp.lbtim.ib = 3
        # TODO: wut?
        pp.t1 = time_coord.units.num2date(time_coord.bounds[0, 0])
        pp.t2 = time_coord.units.num2date(time_coord.bounds[0, 1])
        pp.t1 = cftime.datetime(pp.t1.year, 9, 1, 0, 0, 0)
        pp.t2 = cftime.datetime(pp.t2.year, 12, 1, 0, 0, 0)
        _conditional_warning(
            time_coord.bounds[0, 0] != time_coord.units.date2num(pp.t1),
            "modified t1 for climatological seasonal mean",
        )
        _conditional_warning(
            time_coord.bounds[0, 1] != time_coord.units.date2num(pp.t2),
            "modified t2 for climatological seasonal mean",
        )
        pp.lbft = fp_coord.units.convert(fp_coord.bounds[0, 1], "hours")

    return pp


def _calendar_rules(cube, pp):
    """
    Rules for setting the calendar of the PP field.

    Args:
        cube: the cube being saved as a series of PP fields.
        pp: the current PP field having save rules applied.

    Returns:
        The PP field with updated metadata.

    """
    time_coord = scalar_coord(cube, "time")
    if time_coord is not None:
        if time_coord.units.calendar == "360_day":
            pp.lbtim.ic = 2
        elif time_coord.units.calendar == "gregorian":
            pp.lbtim.ic = 1
        elif time_coord.units.calendar == "365_day":
            pp.lbtim.ic = 4
    return pp


def _grid_and_pole_rules(cube, pp):
    """
    Rules for setting the horizontal grid and pole location of the PP field.

    Args:
        cube: the cube being saved as a series of PP fields.
        pp: the current PP field having save rules applied.

    Returns:
        The PP field with updated metadata.

    """
    lon_coord = vector_coord(cube, "longitude")
    grid_lon_coord = vector_coord(cube, "grid_longitude")
    lat_coord = vector_coord(cube, "latitude")
    grid_lat_coord = vector_coord(cube, "grid_latitude")

    if lon_coord and not is_regular(lon_coord):
        pp.bzx = 0
        pp.bdx = 0
        pp.lbnpt = lon_coord.shape[0]
        pp.x = lon_coord.points
    elif grid_lon_coord and not is_regular(grid_lon_coord):
        pp.bzx = 0
        pp.bdx = 0
        pp.lbnpt = grid_lon_coord.shape[0]
        pp.x = grid_lon_coord.points
    elif lon_coord and is_regular(lon_coord):
        pp.bzx = lon_coord.points[0] - regular_step(lon_coord)
        pp.bdx = regular_step(lon_coord)
        pp.lbnpt = len(lon_coord.points)
    elif grid_lon_coord and is_regular(grid_lon_coord):
        pp.bzx = grid_lon_coord.points[0] - regular_step(grid_lon_coord)
        pp.bdx = regular_step(grid_lon_coord)
        pp.lbnpt = len(grid_lon_coord.points)

    if lat_coord and not is_regular(lat_coord):
        pp.bzy = 0
        pp.bdy = 0
        pp.lbrow = lat_coord.shape[0]
        pp.y = lat_coord.points
    elif grid_lat_coord and not is_regular(grid_lat_coord):
        pp.bzy = 0
        pp.bdy = 0
        pp.lbrow = grid_lat_coord.shape[0]
        pp.y = grid_lat_coord.points
    elif lat_coord and is_regular(lat_coord):
        pp.bzy = lat_coord.points[0] - regular_step(lat_coord)
        pp.bdy = regular_step(lat_coord)
        pp.lbrow = len(lat_coord.points)
    elif grid_lat_coord and is_regular(grid_lat_coord):
        pp.bzy = grid_lat_coord.points[0] - regular_step(grid_lat_coord)
        pp.bdy = regular_step(grid_lat_coord)
        pp.lbrow = len(grid_lat_coord.points)

    # Check if we have a rotated coord system.
    if cube.coord_system("RotatedGeogCS") is not None:
        pp.lbcode = int(pp.lbcode) + 100

    # Check if we have a circular x-coord.
    for lon_coord in (lon_coord, grid_lon_coord):
        if lon_coord is not None:
            if lon_coord.circular:
                pp.lbhem = 0
            else:
                pp.lbhem = 3

    return pp


def _non_std_cross_section_rules(cube, pp):
    """
    Rules for applying non-standard cross-sections to the PP field.

    Args:
        cube: the cube being saved as a series of PP fields.
        pp: the current PP field having save rules applied.

    Returns:
        The PP field with updated metadata.

    """
    # Define commonly-used coords.
    air_pres_coord = vector_coord(cube, "air_pressure")
    depth_coord = vector_coord(cube, "depth")
    eta_coord = vector_coord(cube, "eta")
    lat_coord = vector_coord(cube, "latitude")
    time_coord = vector_coord(cube, "time")

    # Non-standard cross-section with bounds - x=latitude, y=air_pressure.
    if (
        air_pres_coord is not None
        and not air_pres_coord.circular
        and air_pres_coord.has_bounds()
        and lat_coord is not None
        and not lat_coord.circular
        and lat_coord.has_bounds()
    ):
        pp.lbcode = 10000 + int(100 * 10) + 1
        pp.bgor = 0
        pp.y = air_pres_coord.points
        pp.y_lower_bound = air_pres_coord.bounds[:, 0]
        pp.y_upper_bound = air_pres_coord.bounds[:, 1]
        pp.x = lat_coord.points
        pp.x_lower_bound = lat_coord.bounds[:, 0]
        pp.x_upper_bound = lat_coord.bounds[:, 1]
        pp.lbrow = air_pres_coord.shape[0]
        pp.lbnpt = lat_coord.shape[0]
        pp.bzx = pp.bzy = pp.bdx = pp.bdy = 0

    # Non-standard cross-section with bounds - x=latitude, y=depth.
    if (
        depth_coord is not None
        and not depth_coord.circular
        and depth_coord.has_bounds()
        and lat_coord is not None
        and not lat_coord.circular
        and lat_coord.has_bounds()
    ):
        pp.lbcode = 10000 + int(100 * 10) + 4
        pp.bgor = 0
        pp.y = depth_coord.points
        pp.y_lower_bound = depth_coord.bounds[:, 0]
        pp.y_upper_bound = depth_coord.bounds[:, 1]
        pp.x = lat_coord.points
        pp.x_lower_bound = lat_coord.bounds[:, 0]
        pp.x_upper_bound = lat_coord.bounds[:, 1]
        pp.lbrow = depth_coord.shape[0]
        pp.lbnpt = lat_coord.shape[0]
        pp.bzx = pp.bzy = pp.bdx = pp.bdy = 0

    # Non-standard cross-section with bounds - x=latitude, y=eta.
    if (
        eta_coord is not None
        and not eta_coord.circular
        and eta_coord.has_bounds()
        and lat_coord is not None
        and not lat_coord.circular
        and lat_coord.has_bounds()
    ):
        pp.lbcode = 10000 + int(100 * 10) + 3
        pp.bgor = 0
        pp.y = eta_coord.points
        pp.y_lower_bound = eta_coord.bounds[:, 0]
        pp.y_upper_bound = eta_coord.bounds[:, 1]
        pp.x = lat_coord.points
        pp.x_lower_bound = lat_coord.bounds[:, 0]
        pp.x_upper_bound = lat_coord.bounds[:, 1]
        pp.lbrow = eta_coord.shape[0]
        pp.lbnpt = lat_coord.shape[0]
        pp.bzx = pp.bzy = pp.bdx = pp.bdy = 0

    # Non-standard cross-section with bounds - x=days (360 calendar), y=depth.
    if (
        depth_coord is not None
        and not depth_coord.circular
        and depth_coord.has_bounds()
        and time_coord is not None
        and not time_coord.circular
        and time_coord.has_bounds()
    ):
        pp.lbcode = 10000 + int(100 * 23) + 4
        pp.bgor = 0
        pp.y = depth_coord.points
        pp.y_lower_bound = depth_coord.bounds[:, 0]
        pp.y_upper_bound = depth_coord.bounds[:, 1]
        pp.x = time_coord.points
        pp.x_lower_bound = time_coord.bounds[:, 0]
        pp.x_upper_bound = time_coord.bounds[:, 1]
        pp.lbrow = depth_coord.shape[0]
        pp.lbnpt = time_coord.shape[0]
        pp.bzx = pp.bzy = pp.bdx = pp.bdy = 0

    # Non-standard cross-section with bounds -
    # x=days (360 calendar), y=air_pressure.
    if (
        air_pres_coord is not None
        and not air_pres_coord.circular
        and air_pres_coord.has_bounds()
        and time_coord is not None
        and not time_coord.circular
        and time_coord.has_bounds()
    ):
        pp.lbcode = 10000 + int(100 * 23) + 1
        pp.bgor = 0
        pp.y = air_pres_coord.points
        pp.y_lower_bound = air_pres_coord.bounds[:, 0]
        pp.y_upper_bound = air_pres_coord.bounds[:, 1]
        pp.x = time_coord.points
        pp.x_lower_bound = time_coord.bounds[:, 0]
        pp.x_upper_bound = time_coord.bounds[:, 1]
        pp.lbrow = air_pres_coord.shape[0]
        pp.lbnpt = time_coord.shape[0]
        pp.bzx = pp.bzy = pp.bdx = pp.bdy = 0

    return pp


def _lbproc_rules(cube, pp):
    """
    Rules for setting the horizontal grid and pole location of the PP field.

    Note: `pp.lbproc` must be set to 0 before these rules are run.

    Args:
        cube: the cube being saved as a series of PP fields.
        pp: the current PP field having save rules applied.

    Returns:
        The PP field with updated metadata.

    """
    # Basic setting (this may be overridden by subsequent rules).
    pp.lbproc = 0

    if cube.attributes.get("ukmo__process_flags", None):
        pp.lbproc += sum(
            [
                LBPROC_MAP[name]
                for name in cube.attributes["ukmo__process_flags"]
            ]
        )

    # Zonal-mean: look for a CellMethod which is a "mean" over "longitude" or
    # "grid_longitude".
    if (
        scalar_cell_method(cube, "mean", "longitude") is not None
        or scalar_cell_method(cube, "mean", "grid_longitude") is not None
    ):
        pp.lbproc += 64

    # Time-mean: look for a CellMethod which is a "mean" over "time".
    if scalar_cell_method(cube, "mean", "time") is not None:
        pp.lbproc += 128

    # Time-minimum: look for a CellMethod which is a "minimum" over "time".
    if scalar_cell_method(cube, "minimum", "time") is not None:
        pp.lbproc += 4096

    # Time-maximum: look for a CellMethod which is a "maximum" over "time".
    if scalar_cell_method(cube, "maximum", "time") is not None:
        pp.lbproc += 8192

    return pp


def _vertical_rules(cube, pp):
    """
    Rules for setting vertical levels for the PP field.

    Args:
        cube: the cube being saved as a series of PP fields.
        pp: the current PP field having save rules applied.

    Returns:
        The PP field with updated metadata.

    """
    # Define commonly-used coords.
    air_pres_coord = scalar_coord(cube, "air_pressure")
    apt_coord = scalar_coord(cube, "air_potential_temperature")
    depth_coord = scalar_coord(cube, "depth")
    height_coord = scalar_coord(cube, "height")
    level_height_coord = scalar_coord(cube, "level_height")
    mln_coord = scalar_coord(cube, "model_level_number")
    pressure_coord = scalar_coord(cube, "pressure")
    pseudo_level_coord = scalar_coord(cube, "pseudo_level")
    sigma_coord = scalar_coord(cube, "sigma")
    soil_mln_coord = scalar_coord(cube, "soil_model_level_number")

    # Define commonly-used aux factories.
    try:
        height_factory = aux_factory(cube, HybridHeightFactory)
    except ValueError:
        height_factory = None
    try:
        pressure_factory = aux_factory(cube, HybridPressureFactory)
    except ValueError:
        pressure_factory = None

    # Set `lbuser[5]`.
    if pseudo_level_coord is not None and not pseudo_level_coord.bounds:
        pp.lbuser[4] = pseudo_level_coord.points[0]

    # Single height level.
    if (
        height_coord is not None
        and not height_coord.bounds
        and height_coord.points[0] == 1.5
        and cube.name() == "air_temperature"
    ):
        pp.lbvc = 129
        pp.blev = -1

    if pp.lbvc == 0 and height_coord is not None and not height_coord.bounds:
        pp.lbvc = 1
        pp.blev = cube.coord("height").points[0]

    # Single air_pressure level.
    if air_pres_coord is not None and not air_pres_coord.bounds:
        pp.lbvc = 8
        pp.blev = air_pres_coord.points[0]

    # Single pressure level.
    if pressure_coord is not None and not pressure_coord.bounds:
        pp.lbvc = 8
        pp.blev = pressure_coord.points[0]

    # Single depth level (non cross-section).
    if (
        mln_coord is not None
        and not mln_coord.bounds
        and depth_coord is not None
        and not depth_coord.bounds
    ):
        pp.lbvc = 2
        pp.lblev = mln_coord.points[0]
        pp.blev = depth_coord.points[0]

    # Single depth level (Non-dimensional soil model level).
    if (
        soil_mln_coord is not None
        and not soil_mln_coord.has_bounds()
        and air_pres_coord is None
        and depth_coord is None
        and height_coord is None
        and pressure_coord is None
        and cube.standard_name is not None
        and "soil" in cube.standard_name
    ):
        pp.lbvc = 6
        pp.lblev = soil_mln_coord.points[0]
        pp.blev = pp.lblev
        pp.brsvd[0] = 0
        pp.brlev = 0

    # Single depth level (soil depth).
    if (
        depth_coord is not None
        and depth_coord.has_bounds()
        and air_pres_coord is None
        and soil_mln_coord is None
        and mln_coord is None
        and height_coord is None
        and pressure_coord is None
        and cube.standard_name is not None
        and "soil" in cube.standard_name
    ):
        pp.lbvc = 6
        pp.blev = depth_coord.points[0]
        pp.brsvd[0] = depth_coord.bounds[0, 0]
        pp.brlev = depth_coord.bounds[0, 1]

    # Single potential-temperature level.
    if (
        apt_coord is not None
        and not apt_coord.bounds
        and air_pres_coord is None
        and depth_coord is None
        and height_coord is None
        and pressure_coord is None
        and mln_coord is None
    ):
        pp.lbvc = 19
        pp.lblev = apt_coord.points[0]
        pp.blev = apt_coord.points[0]

    # Single hybrid_height level
    # (without aux factory e.g. due to missing orography).
    if (
        not has_aux_factory(cube, HybridHeightFactory)
        and mln_coord is not None
        and mln_coord.bounds is None
        and level_height_coord is not None
        and level_height_coord.bounds is not None
        and sigma_coord is not None
        and sigma_coord.bounds is not None
    ):
        pp.lbvc = 65
        pp.lblev = mln_coord.points[0]
        pp.blev = level_height_coord.points[0]
        pp.brlev = level_height_coord.bounds[0, 0]
        pp.brsvd[0] = level_height_coord.bounds[0, 1]
        pp.bhlev = sigma_coord.points[0]
        pp.bhrlev = sigma_coord.bounds[0, 0]
        pp.brsvd[1] = sigma_coord.bounds[0, 1]

    # Single hybrid_height level (with aux factory).
    if (
        has_aux_factory(cube, HybridHeightFactory)
        and mln_coord is not None
        and mln_coord.bounds is None
        and height_factory.dependencies["delta"] is not None
        and height_factory.dependencies["delta"].bounds is not None
        and height_factory.dependencies["sigma"] is not None
        and height_factory.dependencies["sigma"].bounds is not None
    ):
        pp.lbvc = 65
        pp.lblev = mln_coord.points[0]
        pp.blev = height_factory.dependencies["delta"].points[0]
        pp.brlev = height_factory.dependencies["delta"].bounds[0, 0]
        pp.brsvd[0] = height_factory.dependencies["delta"].bounds[0, 1]
        pp.bhlev = height_factory.dependencies["sigma"].points[0]
        pp.bhrlev = height_factory.dependencies["sigma"].bounds[0, 0]
        pp.brsvd[1] = height_factory.dependencies["sigma"].bounds[0, 1]

    # Single hybrid pressure level.
    if (
        has_aux_factory(cube, HybridPressureFactory)
        and mln_coord is not None
        and mln_coord.bounds is None
        and pressure_factory.dependencies["delta"] is not None
        and pressure_factory.dependencies["delta"].bounds is not None
        and pressure_factory.dependencies["sigma"] is not None
        and pressure_factory.dependencies["sigma"].bounds is not None
    ):
        pp.lbvc = 9
        pp.lblev = mln_coord.points[0]
        pp.blev = pressure_factory.dependencies["sigma"].points[0]
        pp.brlev = pressure_factory.dependencies["sigma"].bounds[0, 0]
        pp.brsvd[0] = pressure_factory.dependencies["sigma"].bounds[0, 1]
        pp.bhlev = pressure_factory.dependencies["delta"].points[0]
        pp.bhrlev = pressure_factory.dependencies["delta"].bounds[0, 0]
        pp.brsvd[1] = pressure_factory.dependencies["delta"].bounds[0, 1]

    return pp


def _all_other_rules(cube, pp):
    """
    Rules for setting the horizontal grid and pole location of the PP field.

    Args:
        cube: the cube being saved as a series of PP fields.
        pp: the current PP field having save rules applied.

    Returns:
        The PP field with updated metadata.

    """
    # "CFNAME mega-rule."
    check_items = (cube.standard_name, cube.long_name, str(cube.units))
    if check_items in CF_TO_LBFC:
        pp.lbfc = CF_TO_LBFC[check_items]

    # Set STASH code.
    if (
        "STASH" in cube.attributes
        and str(cube.attributes["STASH"]) in STASH_TRANS
    ):
        pp.lbfc = STASH_TRANS[str(cube.attributes["STASH"])].field_code

    return pp


def verify(cube, field):
    # Rules functions.
    field = _basic_coord_system_rules(cube, field)
    field = _um_version_rules(cube, field)
    field = _stash_rules(cube, field)
    field = _general_time_rules(cube, field)
    field = _calendar_rules(cube, field)
    field = _grid_and_pole_rules(cube, field)
    field = _non_std_cross_section_rules(cube, field)
    field = _lbproc_rules(cube, field)
    field = _vertical_rules(cube, field)
    field = _all_other_rules(cube, field)

    return field


# Helper functions used when running the rules.


def _conditional_warning(condition, warning):
    if condition:
        warnings.warn(warning)
