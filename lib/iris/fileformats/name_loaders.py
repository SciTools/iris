# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""NAME file format loading functions."""

import collections
import datetime
from operator import itemgetter
import re
import warnings

import cf_units
import numpy as np

import iris.coord_systems
from iris.coords import AuxCoord, CellMethod, DimCoord
import iris.cube
from iris.exceptions import TranslationError
import iris.util

EARTH_RADIUS = 6371229.0
NAMEIII_DATETIME_FORMAT = "%d/%m/%Y  %H:%M %Z"
NAMETRAJ_DATETIME_FORMAT = "%d/%m/%Y  %H:%M:%S %Z"
NAMEII_FIELD_DATETIME_FORMAT = "%H%M%Z %d/%m/%Y"
NAMEII_TIMESERIES_DATETIME_FORMAT = "%d/%m/%Y  %H:%M:%S"


NAMECoord = collections.namedtuple(
    "NAMECoord", ["name", "dimension", "values"]
)


def _split_name_and_units(name):
    units = None
    if "(" in name and ")" in name:
        split = name.rsplit("(", 1)
        try_units = split[1].replace(")", "").strip()
        try:
            try_units = cf_units.Unit(try_units)
        except ValueError:
            pass
        else:
            name = split[0].strip()
            units = try_units
    return name, units


def read_header(file_handle):
    """
    Return a dictionary containing the header information extracted
    from the the provided NAME file object.

    Args:

    * file_handle (file-like object):
        A file-like object from which to read the header information.

    Returns:
        A dictionary containing the extracted header information.

    """
    header = {}
    header["NAME Version"] = next(file_handle).strip()
    for line in file_handle:
        words = line.split(":", 1)
        if len(words) != 2:
            if "Forward" in line or "Backward" in line:
                header["Trajectory direction"] = words[0].strip()
            break
        key, value = [word.strip() for word in words]
        header[key] = value

    # Cast some values into floats or integers if they match a
    # given name. Set any empty string values to None.
    for key, value in header.items():
        if value:
            if key in [
                "X grid origin",
                "Y grid origin",
                "X grid resolution",
                "Y grid resolution",
            ]:
                header[key] = float(value)
            elif key in [
                "X grid size",
                "Y grid size",
                "Number of preliminary cols",
                "Number of field cols",
                "Number of fields",
                "Number of series",
            ]:
                header[key] = int(value)
        else:
            header[key] = None

    return header


def _read_data_arrays(file_handle, n_arrays, shape):
    """
    Return a list of NumPy arrays containing the data extracted from
    the provided file object. The number and shape of the arrays
    must be specified.

    """
    data_arrays = [np.zeros(shape, dtype=np.float32) for i in range(n_arrays)]

    # Iterate over the remaining lines which represent the data in
    # a column form.
    for line in file_handle:
        # Split the line by comma, removing the last empty column
        # caused by the trailing comma
        vals = line.split(",")[:-1]

        # Cast the x and y grid positions to integers and convert
        # them to zero based indices
        x = int(float(vals[0])) - 1
        y = int(float(vals[1])) - 1

        # Populate the data arrays (i.e. all columns but the leading 4).
        for i, data_array in enumerate(data_arrays):
            data_array[y, x] = float(vals[i + 4])

    return data_arrays


def _build_lat_lon_for_NAME_field(
    header, dimindex, x_or_y, coord_names=["longitude", "latitude"]
):
    """
    Return regular latitude and longitude coordinates extracted from
    the provided header dictionary.
    """

    if x_or_y == "X":
        start = header["X grid origin"]
        step = header["X grid resolution"]
        count = header["X grid size"]
        pts = start + np.arange(count, dtype=np.float64) * step
        lat_lon = NAMECoord(
            name=coord_names[0], dimension=dimindex, values=pts
        )
    else:
        start = header["Y grid origin"]
        step = header["Y grid resolution"]
        count = header["Y grid size"]
        pts = start + np.arange(count, dtype=np.float64) * step
        lat_lon = NAMECoord(
            name=coord_names[1], dimension=dimindex, values=pts
        )

    return lat_lon


def _build_lat_lon_for_NAME_timeseries(column_headings):
    """
    Return regular latitude and longitude coordinates extracted from
    the provided column_headings dictionary.

    """
    # Pattern to match a number
    pattern = re.compile(
        r"""
        [-+]?          # Optional sign
        (?:
            \d+\.\d*   # Float: integral part required
        |
            \d*\.\d+   # Float: fractional part required
        |
            \d+        # Integer
        )
        (?![0-9.])     # Not followed by a numeric character
        """,
        re.VERBOSE,
    )

    # Extract numbers from the X and Y column headings, which are currently
    # strings of the form "X = -1.9 Lat-Long"
    for key in ("X", "Y"):
        new_headings = []
        for heading in column_headings[key]:
            match = pattern.search(heading)
            if match and "Lat-Long" in heading:
                new_headings.append(float(match.group(0)))
            else:
                new_headings.append(heading)
        column_headings[key] = new_headings

    lon = NAMECoord(
        name="longitude", dimension=None, values=column_headings["X"]
    )
    lat = NAMECoord(
        name="latitude", dimension=None, values=column_headings["Y"]
    )

    return lat, lon


def _calc_integration_period(time_avgs):
    """
    Return a list of datetime.timedelta objects determined from the provided
    list of averaging/integration period column headings.

    """
    integration_periods = []
    pattern = re.compile(
        r"\s*(\d{1,2}day)?\s*(\d{1,2}hr)?\s*(\d{1,2}min)?\s*(\w*)\s*"
    )
    for time_str in time_avgs:
        days = 0
        hours = 0
        minutes = 0
        matches = pattern.search(time_str)
        if matches:
            _days = matches.group(1)
            if _days is not None and len(_days) > 0:
                days = float(_days.rstrip("day"))
            _hours = matches.group(2)
            if _hours is not None and len(_hours) > 0:
                hours = float(_hours.rstrip("hr"))
            _minutes = matches.group(3)
            if _minutes is not None and len(_minutes) > 0:
                minutes = float(_minutes.rstrip("min"))
        total_hours = days * 24.0 + hours + minutes / 60.0
        integration_periods.append(datetime.timedelta(hours=total_hours))
    return integration_periods


def _parse_units(units):
    """
    Return a known :class:`cf_units.Unit` given a NAME unit

    .. note::

        * Some NAME units are not currently handled.
        * Units which are in the wrong case (case is ignored in NAME)
        * Units where the space between SI units is missing
        * Units where the characters used are non-standard (i.e. 'mc' for
          micro instead of 'u')

    Args:

    * units (string):
        NAME units.

    Returns:
        An instance of :class:`cf_units.Unit`.

    """

    unit_mapper = {
        "Risks/m3": "1",  # Used for Bluetongue
        "TCID50s/m3": "1",  # Used for Foot and Mouth
        "TCID50/m3": "1",  # Used for Foot and Mouth
        "N/A": "1",  # Used for CHEMET area at risk
        "lb": "pounds",  # pounds
        "oz": "1",  # ounces
        "deg": "degree",  # angular degree
        "oktas": "1",  # oktas
        "deg C": "deg_C",  # degrees Celsius
        "FL": "unknown",  # flight level
    }

    units = unit_mapper.get(units, units)

    units = units.replace("Kg", "kg")
    units = units.replace("gs", "g s")
    units = units.replace("Bqs", "Bq s")
    units = units.replace("mcBq", "uBq")
    units = units.replace("mcg", "ug")
    try:
        units = cf_units.Unit(units)
    except ValueError:
        warnings.warn("Unknown units: {!r}".format(units))
        units = cf_units.Unit(None)

    return units


def _cf_height_from_name(z_coord, lower_bound=None, upper_bound=None):
    """
    Parser for the z component of field headings.

    This parse is specifically for handling the z component of NAME field
    headings, which include height above ground level, height above sea level
    and flight level etc.  This function returns an iris coordinate
    representing this field heading.

    Args:

    * z_coord (list):
        A field heading, specifically the z component.

    Returns:
        An instance of :class:`iris.coords.AuxCoord` representing the
        interpretation of the supplied field heading.

    """

    # NAMEII - integer/float support.
    # Match against height agl, asl and Pa.
    pattern = re.compile(
        r"^From\s*"
        r"(?P<lower_bound>[0-9]+(\.[0-9]+)?)"
        r"\s*-\s*"
        r"(?P<upper_bound>[0-9]+(\.[0-9]+)?)"
        r"\s*(?P<type>m\s*asl|m\s*agl|Pa)"
        r"(?P<extra>.*)"
    )

    # Match against flight level.
    pattern_fl = re.compile(
        r"^From\s*"
        r"(?P<type>FL)"
        r"(?P<lower_bound>[0-9]+(\.[0-9]+)?)"
        r"\s*-\s*FL"
        r"(?P<upper_bound>[0-9]+(\.[0-9]+)?)"
        r"(?P<extra>.*)"
    )

    # NAMEIII - integer/float support.
    # Match scalar against height agl, asl, Pa, FL
    pattern_scalar = re.compile(
        r"Z\s*=\s*"
        r"(?P<point>[0-9]+(\.[0-9]+)?([eE][+-]?\d+)?)"
        r"\s*(?P<type>m\s*agl|m\s*asl|FL|Pa)"
        r"(?P<extra>.*)"
    )

    type_name = {
        "magl": "height",
        "masl": "altitude",
        "FL": "flight_level",
        "Pa": "air_pressure",
    }
    patterns = [pattern, pattern_fl, pattern_scalar]

    units = "no-unit"
    points = z_coord
    bounds = None
    standard_name = None
    long_name = "z"

    if upper_bound is not None and lower_bound is not None:
        match_ub = pattern_scalar.match(upper_bound)
        match_lb = pattern_scalar.match(lower_bound)

    for pattern in patterns:
        match = pattern.match(z_coord)
        if match:
            match = match.groupdict()
            # Do not interpret if there is additional information to the match
            if match["extra"]:
                break
            units = match["type"].replace(" ", "")
            name = type_name[units]

            # Interpret points if present.
            if "point" in match:
                points = float(match["point"])
                if upper_bound is not None and lower_bound is not None:
                    bounds = np.array(
                        [
                            float(match_lb.groupdict()["point"]),
                            float(match_ub.groupdict()["point"]),
                        ]
                    )
            # Interpret points from bounds.
            else:
                bounds = np.array(
                    [float(match["lower_bound"]), float(match["upper_bound"])]
                )
                points = bounds.sum() / 2.0

            long_name = None
            if name == "altitude":
                units = units[0]
                standard_name = name
                long_name = "altitude above sea level"
            elif name == "height":
                units = units[0]
                standard_name = name
                long_name = "height above ground level"
            elif name == "air_pressure":
                standard_name = name
            elif name == "flight_level":
                long_name = name
            units = _parse_units(units)

            break

    coord = AuxCoord(
        points,
        units=units,
        standard_name=standard_name,
        long_name=long_name,
        bounds=bounds,
        attributes={"positive": "up"},
    )

    return coord


def _generate_cubes(
    header, column_headings, coords, data_arrays, cell_methods=None
):
    """
    Yield :class:`iris.cube.Cube` instances given
    the headers, column headings, coords and data_arrays extracted
    from a NAME file.

    """
    for i, data_array in enumerate(data_arrays):
        # Turn the dictionary of column headings with a list of header
        # information for each field into a dictionary of headings for
        # just this field.
        field_headings = {k: v[i] for k, v in column_headings.items()}

        # Make a cube.
        cube = iris.cube.Cube(data_array)

        # Determine the name and units.
        name = "{} {}".format(
            field_headings["Species"], field_headings["Quantity"]
        )
        name = name.upper().replace(" ", "_")
        cube.rename(name)

        # Some units are not in SI units, are missing spaces or typed
        # in the wrong case. _parse_units returns units that are
        # recognised by Iris.
        cube.units = _parse_units(field_headings["Units"])

        # Define and add the singular coordinates of the field (flight
        # level, time etc.)
        if "Z" in field_headings:
            (upper_bound,) = [
                field_headings["... to [Z]"]
                if "... to [Z]" in field_headings
                else None
            ]
            (lower_bound,) = [
                field_headings["... from [Z]"]
                if "... from [Z]" in field_headings
                else None
            ]
            z_coord = _cf_height_from_name(
                field_headings["Z"],
                upper_bound=upper_bound,
                lower_bound=lower_bound,
            )
            cube.add_aux_coord(z_coord)

        # Define the time unit and use it to serialise the datetime for
        # the time coordinate.
        time_unit = cf_units.Unit(
            "hours since epoch", calendar=cf_units.CALENDAR_STANDARD
        )

        # Build time, height, latitude and longitude coordinates.
        for coord in coords:
            pts = coord.values
            coord_sys = None
            if coord.name == "latitude" or coord.name == "longitude":
                coord_units = "degrees"
                coord_sys = iris.coord_systems.GeogCS(EARTH_RADIUS)
            if (
                coord.name == "projection_x_coordinate"
                or coord.name == "projection_y_coordinate"
            ):
                coord_units = "m"
                coord_sys = iris.coord_systems.OSGB()
            if coord.name == "height":
                coord_units = "m"
                long_name = "height above ground level"
                pts = coord.values
            if coord.name == "altitude":
                coord_units = "m"
                long_name = "altitude above sea level"
                pts = coord.values
            if coord.name == "air_pressure":
                coord_units = "Pa"
                pts = coord.values
            if coord.name == "flight_level":
                pts = coord.values
                long_name = "flight_level"
                coord_units = _parse_units("FL")
            if coord.name == "time":
                coord_units = time_unit
                pts = time_unit.date2num(coord.values).astype(float)

            if coord.dimension is not None:
                if coord.name == "longitude":
                    circular = iris.util._is_circular(pts, 360.0)
                else:
                    circular = False
                if coord.name == "flight_level":
                    icoord = DimCoord(
                        points=pts, units=coord_units, long_name=long_name
                    )
                else:
                    icoord = DimCoord(
                        points=pts,
                        standard_name=coord.name,
                        units=coord_units,
                        coord_system=coord_sys,
                        circular=circular,
                    )
                if coord.name == "height" or coord.name == "altitude":
                    icoord.long_name = long_name
                if (
                    coord.name == "time"
                    and "Av or Int period" in field_headings
                ):
                    dt = coord.values - field_headings["Av or Int period"]
                    bnds = time_unit.date2num(np.vstack((dt, coord.values)).T)
                    icoord.bounds = bnds.astype(float)
                else:
                    icoord.guess_bounds()
                cube.add_dim_coord(icoord, coord.dimension)
            else:
                icoord = AuxCoord(
                    points=pts[i],
                    standard_name=coord.name,
                    coord_system=coord_sys,
                    units=coord_units,
                )
                if (
                    coord.name == "time"
                    and "Av or Int period" in field_headings
                ):
                    dt = coord.values - field_headings["Av or Int period"]
                    bnds = time_unit.date2num(np.vstack((dt, coord.values)).T)
                    icoord.bounds = bnds[i, :].astype(float)
                cube.add_aux_coord(icoord)

        # Headings/column headings which are encoded elsewhere.
        headings = [
            "X",
            "Y",
            "Z",
            "Time",
            "T",
            "Units",
            "Av or Int period",
            "... from [Z]",
            "... to [Z]",
            "X grid origin",
            "Y grid origin",
            "X grid size",
            "Y grid size",
            "X grid resolution",
            "Y grid resolution",
            "Number of field cols",
            "Number of preliminary cols",
            "Number of fields",
            "Number of series",
            "Output format",
        ]

        # Add the Main Headings as attributes.
        for key, value in header.items():
            if value is not None and value != "" and key not in headings:
                cube.attributes[key] = value

        # Add the Column Headings as attributes
        for key, value in field_headings.items():
            if value is not None and value != "" and key not in headings:
                cube.attributes[key] = value

        if cell_methods is not None:
            cell_method = cell_methods[i]
            if cell_method is not None:
                cube.add_cell_method(cell_method)

        yield cube


def _build_cell_methods(av_or_ints, coord):
    """
    Return a list of :class:`iris.coords.CellMethod` instances
    based on the provided list of column heading entries and the
    associated coordinate. If a given entry does not correspond to a cell
    method (e.g. "No time averaging"), a value of None is inserted.

    Args:

    * av_or_ints (iterable of strings):
        An iterable of strings containing the column heading entries
        to be parsed.
    * coord (string or :class:`iris.coords.Coord`):
        The coordinate name (or :class:`iris.coords.Coord` instance)
        to which the column heading entries refer.

    Returns:
        A list that is the same length as `av_or_ints` containing
        :class:`iris.coords.CellMethod` instances or values of None.

    """
    cell_methods = []
    no_avg_pattern = re.compile(r"^(no( (.* )?averaging)?)?$", re.IGNORECASE)
    for av_or_int in av_or_ints:
        if no_avg_pattern.search(av_or_int) is not None:
            cell_method = None
        elif "average" in av_or_int or "averaged" in av_or_int:
            cell_method = CellMethod("mean", coord)
        elif "integral" in av_or_int or "integrated" in av_or_int:
            cell_method = CellMethod("sum", coord)
        else:
            cell_method = None
            msg = "Unknown {} statistic: {!r}. Unable to create cell method."
            warnings.warn(msg.format(coord, av_or_int))
        cell_methods.append(cell_method)  # NOTE: this can be a None
    return cell_methods


def load_NAMEIII_field(filename):
    """
    Load a NAME III grid output file returning a
    generator of :class:`iris.cube.Cube` instances.

    Args:

    * filename (string):
        Name of file to load.

    Returns:
        A generator :class:`iris.cube.Cube` instances.

    """
    # Loading a file gives a generator of lines which can be progressed using
    # the next() function. This will come in handy as we wish to progress
    # through the file line by line.
    with open(filename, "r") as file_handle:
        # Create a dictionary which can hold the header metadata about this
        # file.
        header = read_header(file_handle)

        # Skip the next line (contains the word Fields:) in the file.
        next(file_handle)

        # Read the lines of column definitions.
        # In this version a fixed order of column headings is assumed (and
        # first 4 columns are ignored).
        column_headings = {}
        for column_header_name in [
            "Species Category",
            "Name",
            "Quantity",
            "Species",
            "Units",
            "Sources",
            "Ensemble Av",
            "Time Av or Int",
            "Horizontal Av or Int",
            "Vertical Av or Int",
            "Prob Perc",
            "Prob Perc Ens",
            "Prob Perc Time",
            "Time",
            "Z",
            "D",
        ]:
            cols = [col.strip() for col in next(file_handle).split(",")]
            column_headings[column_header_name] = cols[4:-1]

        # Read in the column titles to determine the coordinate system
        col_titles = next(file_handle).split(",")
        if "National Grid" in col_titles[2]:
            coord_names = [
                "projection_x_coordinate",
                "projection_y_coordinate",
            ]
        else:
            coord_names = ["longitude", "latitude"]

        # Convert the time to python datetimes.
        new_time_column_header = []
        for i, t in enumerate(column_headings["Time"]):
            dt = datetime.datetime.strptime(t, NAMEIII_DATETIME_FORMAT)
            new_time_column_header.append(dt)
        column_headings["Time"] = new_time_column_header

        # Convert averaging/integrating period to timedeltas.
        column_headings["Av or Int period"] = _calc_integration_period(
            column_headings["Time Av or Int"]
        )

        # Build a time coordinate.
        tdim = NAMECoord(
            name="time",
            dimension=None,
            values=np.array(column_headings["Time"]),
        )

        cell_methods = _build_cell_methods(
            column_headings["Time Av or Int"], tdim.name
        )

        # Build regular latitude and longitude coordinates.
        lon = _build_lat_lon_for_NAME_field(
            header, 1, "X", coord_names=coord_names
        )
        lat = _build_lat_lon_for_NAME_field(
            header, 0, "Y", coord_names=coord_names
        )

        coords = [lon, lat, tdim]

        # Create data arrays to hold the data for each column.
        n_arrays = header["Number of field cols"]
        shape = (header["Y grid size"], header["X grid size"])
        data_arrays = _read_data_arrays(file_handle, n_arrays, shape)

    return _generate_cubes(
        header, column_headings, coords, data_arrays, cell_methods
    )


def load_NAMEII_field(filename):
    """
    Load a NAME II grid output file returning a
    generator of :class:`iris.cube.Cube` instances.

    Args:

    * filename (string):
        Name of file to load.

    Returns:
        A generator :class:`iris.cube.Cube` instances.

    """
    with open(filename, "r") as file_handle:
        # Create a dictionary which can hold the header metadata about this
        # file.
        header = read_header(file_handle)

        # Origin in namever=2 format is bottom-left hand corner so alter this
        # to centre of a grid box
        header["X grid origin"] = (
            header["X grid origin"] + header["X grid resolution"] / 2
        )
        header["Y grid origin"] = (
            header["Y grid origin"] + header["Y grid resolution"] / 2
        )

        # Read the lines of column definitions.
        # In this version a fixed order of column headings is assumed (and
        # first 4 columns are ignored).
        column_headings = {}
        for column_header_name in [
            "Species Category",
            "Species",
            "Time Av or Int",
            "Quantity",
            "Units",
            "Z",
            "Time",
        ]:
            cols = [col.strip() for col in next(file_handle).split(",")]
            column_headings[column_header_name] = cols[4:-1]

        # Convert the time to python datetimes
        new_time_column_header = []
        for i, t in enumerate(column_headings["Time"]):
            dt = datetime.datetime.strptime(t, NAMEII_FIELD_DATETIME_FORMAT)
            new_time_column_header.append(dt)
        column_headings["Time"] = new_time_column_header

        # Convert averaging/integrating period to timedeltas.
        pattern = re.compile(r"\s*(\d{3})\s*(hr)?\s*(time)\s*(\w*)")
        column_headings["Av or Int period"] = []
        for i, t in enumerate(column_headings["Time Av or Int"]):
            matches = pattern.search(t)
            hours = 0
            if matches:
                if len(matches.group(1)) > 0:
                    hours = float(matches.group(1))
            column_headings["Av or Int period"].append(
                datetime.timedelta(hours=hours)
            )

        # Build a time coordinate.
        tdim = NAMECoord(
            name="time",
            dimension=None,
            values=np.array(column_headings["Time"]),
        )

        cell_methods = _build_cell_methods(
            column_headings["Time Av or Int"], tdim.name
        )

        # Build regular latitude and longitude coordinates.
        lon = _build_lat_lon_for_NAME_field(header, 1, "X")
        lat = _build_lat_lon_for_NAME_field(header, 0, "Y")

        coords = [lon, lat, tdim]

        # Skip the blank line after the column headings.
        next(file_handle)

        # Create data arrays to hold the data for each column.
        n_arrays = header["Number of fields"]
        shape = (header["Y grid size"], header["X grid size"])
        data_arrays = _read_data_arrays(file_handle, n_arrays, shape)

    return _generate_cubes(
        header, column_headings, coords, data_arrays, cell_methods
    )


def load_NAMEIII_timeseries(filename):
    """
    Load a NAME III time series file returning a
    generator of :class:`iris.cube.Cube` instances.

    Args:

    * filename (string):
        Name of file to load.

    Returns:
        A generator :class:`iris.cube.Cube` instances.

    """
    with open(filename, "r") as file_handle:
        # Create a dictionary which can hold the header metadata about this
        # file.
        header = read_header(file_handle)

        # skip the next line (contains the word Fields:) in the file.
        next(file_handle)

        # Read the lines of column definitions - currently hardwired
        column_headings = {}
        for column_header_name in [
            "Species Category",
            "Name",
            "Quantity",
            "Species",
            "Units",
            "Sources",
            "Ens Av",
            "Time Av or Int",
            "Horizontal Av or Int",
            "Vertical Av or Int",
            "Prob Perc",
            "Prob Perc Ens",
            "Prob Perc Time",
            "Location",
            "X",
            "Y",
            "Z",
            "D",
        ]:
            cols = [col.strip() for col in next(file_handle).split(",")]
            column_headings[column_header_name] = cols[1:-1]

        # Determine the coordinates of the data and store in namedtuples.
        # Extract latitude and longitude information from X, Y location
        # headings.
        lat, lon = _build_lat_lon_for_NAME_timeseries(column_headings)

        # Convert averaging/integrating period to timedeltas.
        column_headings["Av or Int period"] = _calc_integration_period(
            column_headings["Time Av or Int"]
        )

        # Skip the line after the column headings.
        next(file_handle)

        # Make a list of data lists to hold the data for each column.
        data_lists = [[] for i in range(header["Number of field cols"])]
        time_list = []

        # Iterate over the remaining lines which represent the data in a
        # column form.
        for line in file_handle:
            # Split the line by comma, removing the last empty column caused
            # by the trailing comma.
            vals = line.split(",")[:-1]

            # Time is stored in the first column.
            t = vals[0].strip()
            dt = datetime.datetime.strptime(t, NAMEIII_DATETIME_FORMAT)
            time_list.append(dt)

            # Populate the data arrays.
            for i, data_list in enumerate(data_lists):
                data_list.append(float(vals[i + 1]))

        data_arrays = [np.array(dl) for dl in data_lists]
        time_array = np.array(time_list)
        tdim = NAMECoord(name="time", dimension=0, values=time_array)

        coords = [lon, lat, tdim]

    return _generate_cubes(header, column_headings, coords, data_arrays)


def load_NAMEII_timeseries(filename):
    """
    Load a NAME II Time Series file returning a
    generator of :class:`iris.cube.Cube` instances.

    Args:

    * filename (string):
        Name of file to load.

    Returns:
        A generator :class:`iris.cube.Cube` instances.

    """
    with open(filename, "r") as file_handle:
        # Create a dictionary which can hold the header metadata about this
        # file.
        header = read_header(file_handle)

        # Read the lines of column definitions.
        column_headings = {}
        for column_header_name in [
            "Y",
            "X",
            "Location",
            "Species Category",
            "Species",
            "Quantity",
            "Z",
            "Units",
        ]:
            cols = [col.strip() for col in next(file_handle).split(",")]
            column_headings[column_header_name] = cols[1:-1]

        # Determine the coordinates of the data and store in namedtuples.
        # Extract latitude and longitude information from X, Y location
        # headings.
        lat, lon = _build_lat_lon_for_NAME_timeseries(column_headings)

        # Skip the blank line after the column headings.
        next(file_handle)

        # Make a list of data arrays to hold the data for each column.
        data_lists = [[] for i in range(header["Number of series"])]
        time_list = []

        # Iterate over the remaining lines which represent the data in a
        # column form.
        for line in file_handle:
            # Split the line by comma, removing the last empty column caused
            # by the trailing comma.
            vals = line.split(",")[:-1]

            # Time is stored in the first two columns.
            t = vals[0].strip() + " " + vals[1].strip()
            dt = datetime.datetime.strptime(
                t, NAMEII_TIMESERIES_DATETIME_FORMAT
            )
            time_list.append(dt)

            # Populate the data arrays.
            for i, data_list in enumerate(data_lists):
                data_list.append(float(vals[i + 2]))

        data_arrays = [np.array(dl) for dl in data_lists]
        time_array = np.array(time_list)
        tdim = NAMECoord(name="time", dimension=0, values=time_array)

        coords = [lon, lat, tdim]

    return _generate_cubes(header, column_headings, coords, data_arrays)


def load_NAMEIII_version2(filename):
    """
    Load a NAME III version 2 file returning a
    generator of :class:`iris.cube.Cube` instances.

    Args:

    * filename (string):
        Name of file to load.

    Returns:
        A generator :class:`iris.cube.Cube` instances.

    """

    # loading a file gives a generator of lines which can be progressed
    # using the next() method. This will come in handy as we wish to
    # progress through the file line by line.
    with open(filename, "r") as file_handle:
        # define a dictionary to hold the header metadata about this file
        header = read_header(file_handle)

        # Skip next line which contains (Fields:)
        next(file_handle)

        # Now carry on and read column headers
        column_headings = {}
        datacol1 = header["Number of preliminary cols"]
        for line in file_handle:
            data = [col.strip() for col in line.split(",")][:-1]

            # If first column is not zero we have reached the end
            #  of the headers
            if data[0] != "":
                break

            column_key = data[datacol1 - 1].strip(":")

            # This will filter out any zero columns
            if filter(None, data[datacol1:]):
                column_headings[column_key] = data[datacol1:]

        # Some tidying up
        if "T" in column_headings:
            new_time_column_header = []
            for i, t in enumerate(column_headings["T"]):
                dt = datetime.datetime.strptime(t, NAMEIII_DATETIME_FORMAT)
                new_time_column_header.append(dt)
            column_headings["T"] = new_time_column_header

        # Convert averaging/integrating period to timedeltas.
        column_headings["Av or Int period"] = _calc_integration_period(
            column_headings["Time av/int info"]
        )

        # Next we need to figure out what we have in the preliminary columns
        # For X and Y we want the index
        # And the values can be extracted from the header information
        xindex = None
        yindex = None
        dim_coords = []

        # First determine whether we are using National Grid or lat/lon
        if "X (UK National Grid (m))" in data:
            coord_names = [
                "projection_x_coordinate",
                "projection_y_coordinate",
            ]
        else:
            coord_names = ["longitude", "latitude"]

        if "Y Index" in data:
            yindex = data.index("Y Index")
            dim_coords.append("Y")
            lat = _build_lat_lon_for_NAME_field(
                header, dim_coords.index("Y"), "Y", coord_names=coord_names
            )
        if "X Index" in data:
            xindex = data.index("X Index")
            dim_coords.append("X")
            lon = _build_lat_lon_for_NAME_field(
                header, dim_coords.index("X"), "X", coord_names=coord_names
            )

        # For all other variables we need the values (note that for Z the units
        # will also be given in the column header)
        tindex = None
        zindex = None
        if "T" in data:
            tindex = data.index("T")
            dim_coords.append("T")

        if "Z" in line:
            zgrid = [item for item in data if item[0:3] == "Z ("]
            zunits = zgrid[0].split("(")[1].strip(")")
            if zunits == "m asl":
                z_name = "altitude"
            elif zunits == "m agl":
                z_name = "height"
            elif zunits == "FL":
                z_name = "flight_level"
            elif zunits == "Pa":
                z_name = "air_pressure"
            else:
                ValueError("Vertical coordinate unknown")
            zindex = data.index(zgrid[0])
            dim_coords.append("Z")

        # Make a list of data lists to hold the data
        # for each column.(aimed at T-Z data)
        data_lists = [[] for i in range(header["Number of field cols"])]
        coord_lists = [
            [] for i in range(header["Number of preliminary cols"] - 1)
        ]

        # Iterate over the remaining lines which represent the data in a
        # column form.
        for line in file_handle:
            # Split the line by comma, removing the last empty column caused
            # by the trailing comma.
            vals = line.split(",")[:-1]

            # Time is stored in the column labelled T index
            if tindex is not None:
                t = vals[tindex].strip()
                dt = datetime.datetime.strptime(t, NAMEIII_DATETIME_FORMAT)
                coord_lists[dim_coords.index("T")].append(dt)

            # Z is stored in the column labelled ZIndex
            if zindex is not None:
                z = vals[zindex].strip()
                coord_lists[dim_coords.index("Z")].append(float(z))

            # For X and Y we are extracting indices not values
            if yindex is not None:
                yind = vals[yindex].strip()
                coord_lists[dim_coords.index("Y")].append(int(yind) - 1)
            if xindex is not None:
                xind = vals[xindex].strip()
                coord_lists[dim_coords.index("X")].append(int(xind) - 1)

            # Populate the data arrays.
            for i, data_list in enumerate(data_lists):
                data_list.append(float(vals[i + datacol1]))

        data_arrays = [np.array(dl) for dl in data_lists]

        # Convert Z and T arrays into arrays of indices
        zind = []
        if zindex is not None:
            z_array = np.array(coord_lists[dim_coords.index("Z")])
            z_unique = sorted(list(set(coord_lists[dim_coords.index("Z")])))
            z_coord = NAMECoord(
                name=z_name, dimension=dim_coords.index("Z"), values=z_unique
            )
            for z in z_array:
                zind.append(z_unique.index(z))
            coord_lists[dim_coords.index("Z")] = zind

        tind = []
        if tindex is not None:
            time_array = np.array(coord_lists[dim_coords.index("T")])
            t_unique = sorted(list(set(coord_lists[dim_coords.index("T")])))
            time = NAMECoord(
                name="time",
                dimension=dim_coords.index("T"),
                values=np.array(t_unique),
            )
            for t in time_array:
                tind.append(t_unique.index(t))
            coord_lists[dim_coords.index("T")] = tind

        # Now determine the shape of the multidimensional array to store
        # the data in based on the length of the coordinates
        array_shape_list = []
        coords = []
        for cname in dim_coords:
            if cname == "X":
                coords.append(lon)
                array_shape_list.append(len(lon.values))
            elif cname == "Y":
                coords.append(lat)
                array_shape_list.append(len(lat.values))
            elif cname == "Z":
                coords.append(z_coord)
                array_shape_list.append(len(z_coord.values))
            elif cname == "T":
                coords.append(time)
                array_shape_list.append(len(time.values))
        array_shape = np.array(array_shape_list)

        # Reshape the data to the new multidimensional shape
        new_data_arrays = []
        for data_array in data_arrays:
            new_data_array = np.zeros(array_shape, dtype=np.float32)
            for ind1, item in enumerate(data_array):
                index_list = []
                for column, dcoord in enumerate(dim_coords):
                    index_list.append(coord_lists[column][ind1])
                index_array = np.array(index_list)
                mindex = np.ravel_multi_index(index_array, array_shape)
                new_data_array[np.unravel_index(mindex, array_shape)] = item
            new_data_arrays.append(new_data_array)

    # If X and Y are in the column headings build coordinates
    if "X" in column_headings and "Y" in column_headings:
        lat, lon = _build_lat_lon_for_NAME_timeseries(column_headings)
        coords.append(lat)
        coords.append(lon)

    # If time is in the column heading build a coordinate
    if "T" in column_headings:
        tdim = NAMECoord(
            name="time", dimension=None, values=np.array(column_headings["T"])
        )
        coords.append(tdim)

    return _generate_cubes(
        header, column_headings, coords, new_data_arrays, cell_methods=None
    )


def load_NAMEIII_trajectory(filename):
    """
    Load a NAME III trajectory file returning a
    generator of :class:`iris.cube.Cube` instances.

    Args:

    * filename (string):
        Name of file to load.

    Returns:
        A generator :class:`iris.cube.Cube` instances.

    """
    time_unit = cf_units.Unit(
        "hours since epoch", calendar=cf_units.CALENDAR_STANDARD
    )

    with open(filename, "r") as infile:
        header = read_header(infile)

        # read the column headings
        for line in infile:
            if line.startswith("    "):
                break
        headings = [heading.strip() for heading in line.split(",")]

        # read the columns
        columns = [[] for i in range(len(headings))]
        for line in infile:
            values = [v.strip() for v in line.split(",")]
            for c, v in enumerate(values):
                if "UTC" in v:
                    v = datetime.datetime.strptime(v, NAMETRAJ_DATETIME_FORMAT)
                else:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
                columns[c].append(v)

    # Sort columns according to PP Index
    columns_t = list(map(list, zip(*columns)))
    columns_t.sort(key=itemgetter(1))
    columns = list(map(list, zip(*columns_t)))

    # Where's the Z column?
    z_column = None
    for i, heading in enumerate(headings):
        if heading.startswith("Z "):
            z_column = i
            break
    if z_column is None:
        raise TranslationError("Expected a Z column")

    # Every column up to Z becomes a coordinate.
    coords = []
    for name, values in zip(headings[: z_column + 1], columns[: z_column + 1]):
        values = np.array(values)
        if np.all(np.array(values) == values[0]):
            values = [values[0]]

        long_name = units = None
        if isinstance(values[0], datetime.datetime):
            values = time_unit.date2num(values).astype(float)
            units = time_unit
            if name == "Time":
                name = "time"
        elif " (Lat-Long)" in name:
            if name.startswith("X"):
                name = "longitude"
            elif name.startswith("Y"):
                name = "latitude"
            units = "degrees"
        elif name == "Z (m asl)":
            name = "altitude"
            units = "m"
            long_name = "altitude above sea level"
        elif name == "Z (m agl)":
            name = "height"
            units = "m"
            long_name = "height above ground level"
        elif name == "Z (FL)":
            name = "flight_level"
            long_name = name

        try:
            coord = DimCoord(values, units=units)
        except ValueError:
            coord = AuxCoord(values, units=units)
        coord.rename(name)
        if coord.long_name is None and long_name is not None:
            coord.long_name = long_name
        coords.append(coord)

    # Every numerical column after the Z becomes a cube.
    for name, values in zip(headings[z_column + 1 :], columns[z_column + 1 :]):
        try:
            float(values[0])
        except ValueError:
            continue
        # units embedded in column heading?
        name, units = _split_name_and_units(name)
        cube = iris.cube.Cube(values, units=units)
        cube.rename(name)
        # Add the Main Headings as attributes.
        for key, value in header.items():
            if value is not None and value != "" and key not in headings:
                cube.attributes[key] = value
        # Add coordinates
        for coord in coords:
            dim = 0 if len(coord.points) > 1 else None
            if dim == 0 and coord.name() == "time":
                cube.add_dim_coord(coord.copy(), dim)
            elif dim == 0 and coord.name() == "PP Index":
                cube.add_dim_coord(coord.copy(), dim)
            else:
                cube.add_aux_coord(coord.copy(), dim)
        yield cube
