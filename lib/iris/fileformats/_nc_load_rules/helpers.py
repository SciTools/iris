# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Helper functions for NetCDF loading rules.

All the pure-Python 'helper' functions which were previously included in the
Pyke rules database 'fc_rules_cf.krb'.

The 'action' routines now call these, as the rules used to do.
They have not changed, **except** that the 'build_coordinate_system' routine
acquired an extra initial 'engine' argument, purely for consistency with other
build routines, and which it does not use.

"""

from __future__ import annotations

import contextlib
from functools import partial
import re
from typing import TYPE_CHECKING, Any, List, Optional
import warnings

import cf_units
import numpy as np
import numpy.ma as ma
import pyproj

import iris
from iris._deprecation import warn_deprecated
import iris.aux_factory
from iris.common.mixin import LimitedAttributeDict, _get_valid_standard_name
import iris.coord_systems
import iris.coords
from iris.cube import Cube
import iris.exceptions
import iris.fileformats.cf as cf
import iris.fileformats.netcdf
from iris.fileformats.netcdf.loader import _get_cf_var_data
from iris.loading import LOAD_PROBLEMS, LoadProblems
import iris.std_names
import iris.util
import iris.warnings

if TYPE_CHECKING:
    from numpy.ma import MaskedArray

    from iris.fileformats.cf import CFBoundaryVariable

    from .engine import Engine

# TODO: should un-addable coords / cell measures / etcetera be skipped? iris#5068.

#
# UD Units Constants (based on Unidata udunits.dat definition file)
#
UD_UNITS_LAT = [
    "degrees_north",
    "degree_north",
    "degree_n",
    "degrees_n",
    "degreen",
    "degreesn",
    "degrees",
    "degrees north",
    "degree north",
    "degree n",
    "degrees n",
]
UD_UNITS_LON = [
    "degrees_east",
    "degree_east",
    "degree_e",
    "degrees_e",
    "degreee",
    "degreese",
    "degrees",
    "degrees east",
    "degree east",
    "degree e",
    "degrees e",
]
UNKNOWN_UNIT_STRING = "?"
NO_UNIT_STRING = "-"

#
# CF Dimensionless Vertical Coordinates
#
CF_COORD_VERTICAL = {
    "atmosphere_ln_pressure_coordinate": ["p0", "lev"],
    "atmosphere_sigma_coordinate": ["sigma", "ps", "ptop"],
    "atmosphere_hybrid_sigma_pressure_coordinate": ["a", "b", "ps", "p0"],
    "atmosphere_hybrid_height_coordinate": ["a", "b", "orog"],
    "atmosphere_sleve_coordinate": [
        "a",
        "b1",
        "b2",
        "ztop",
        "zsurf1",
        "zsurf2",
    ],
    "ocean_sigma_coordinate": ["sigma", "eta", "depth"],
    "ocean_s_coordinate": ["s", "eta", "depth", "a", "b", "depth_c"],
    "ocean_sigma_z_coordinate": [
        "sigma",
        "eta",
        "depth",
        "depth_c",
        "nsigma",
        "zlev",
    ],
    "ocean_double_sigma_coordinate": [
        "sigma",
        "depth",
        "z1",
        "z2",
        "a",
        "href",
        "k_c",
    ],
    "ocean_s_coordinate_g1": ["s", "eta", "depth", "depth_c", "C"],
    "ocean_s_coordinate_g2": ["s", "eta", "depth", "depth_c", "C"],
}

#
# CF Grid Mappings
#
CF_GRID_MAPPING_ALBERS = "albers_conical_equal_area"
CF_GRID_MAPPING_AZIMUTHAL = "azimuthal_equidistant"
CF_GRID_MAPPING_LAMBERT_AZIMUTHAL = "lambert_azimuthal_equal_area"
CF_GRID_MAPPING_LAMBERT_CONFORMAL = "lambert_conformal_conic"
CF_GRID_MAPPING_LAMBERT_CYLINDRICAL = "lambert_cylindrical_equal_area"
CF_GRID_MAPPING_LAT_LON = "latitude_longitude"
CF_GRID_MAPPING_MERCATOR = "mercator"
CF_GRID_MAPPING_ORTHO = "orthographic"
CF_GRID_MAPPING_POLAR = "polar_stereographic"
CF_GRID_MAPPING_ROTATED_LAT_LON = "rotated_latitude_longitude"
CF_GRID_MAPPING_STEREO = "stereographic"
CF_GRID_MAPPING_TRANSVERSE = "transverse_mercator"
CF_GRID_MAPPING_VERTICAL = "vertical_perspective"
CF_GRID_MAPPING_GEOSTATIONARY = "geostationary"
CF_GRID_MAPPING_OBLIQUE = "oblique_mercator"
CF_GRID_MAPPING_ROTATED_MERCATOR = "rotated_mercator"

#
# Regex for parsing grid_mapping (extended format)
# Link to online regex101 playground: https://regex101.com/r/NcKzkQ/1
#
#   (\w+):                # Matches '<word>:' and stores in CAPTURE GROUP 1
#   (                     # CAPTURE GROUP 2 for capturing multiple coords
#       (?:               #  Non-capturing group for composing match
#           \s+           #   Matches one or more blank characters
#           (?!\w+:)      #   Negative look-ahead: don't match <word> followed by colon
#           \w+           #   Matches a <word>
#       )+                #  Repeats non-capturing group at least once.
#   )                     # End of CAPTURE GROUP 2
_GRID_MAPPING_PARSE_EXTENDED = re.compile(
    r"""
        (\w+):
        (
            (?:
                \s+
                (?!\w+:)
                \w+
            )+
        )+
    """,
    re.VERBOSE,
)
_GRID_MAPPING_PARSE_SIMPLE = re.compile(r"^\w+$")
_GRID_MAPPING_VALIDATORS = (  # fmt: skip
    (
        re.compile(r"\w+: +\w+:"),
        "`<coord_system>:` identifier followed immediately by another `<coord_system>:` identifier",
    ),
    (
        re.compile(r"\w+: *$"),
        "`<coord_system>:` is empty - missing coordinate list",
    ),
    (
        re.compile(r"^\w+ +\w+"),
        "Multiple coordinates found without `<coord_system>:` identifier",
    ),
)
#
# CF Attribute Names.
#
CF_ATTR_AXIS = "axis"
CF_ATTR_BOUNDS = "bounds"
CF_ATTR_CALENDAR = "calendar"
CF_ATTR_CLIMATOLOGY = "climatology"
CF_ATTR_GRID_CRS_WKT = "crs_wkt"
CF_ATTR_GRID_DATUM = "horizontal_datum_name"
CF_ATTR_GRID_INVERSE_FLATTENING = "inverse_flattening"
CF_ATTR_GRID_EARTH_RADIUS = "earth_radius"
CF_ATTR_GRID_MAPPING_NAME = "grid_mapping_name"
CF_ATTR_GRID_NORTH_POLE_LAT = "grid_north_pole_latitude"
CF_ATTR_GRID_NORTH_POLE_LON = "grid_north_pole_longitude"
CF_ATTR_GRID_NORTH_POLE_GRID_LON = "north_pole_grid_longitude"
CF_ATTR_GRID_SEMI_MAJOR_AXIS = "semi_major_axis"
CF_ATTR_GRID_SEMI_MINOR_AXIS = "semi_minor_axis"
CF_ATTR_GRID_LAT_OF_PROJ_ORIGIN = "latitude_of_projection_origin"
CF_ATTR_GRID_LON_OF_PROJ_ORIGIN = "longitude_of_projection_origin"
CF_ATTR_GRID_STRAIGHT_VERT_LON = "straight_vertical_longitude_from_pole"
CF_ATTR_GRID_STANDARD_PARALLEL = "standard_parallel"
CF_ATTR_GRID_FALSE_EASTING = "false_easting"
CF_ATTR_GRID_FALSE_NORTHING = "false_northing"
CF_ATTR_GRID_SCALE_FACTOR_AT_PROJ_ORIGIN = "scale_factor_at_projection_origin"
CF_ATTR_GRID_SCALE_FACTOR_AT_CENT_MERIDIAN = "scale_factor_at_central_meridian"
CF_ATTR_GRID_LON_OF_CENT_MERIDIAN = "longitude_of_central_meridian"
CF_ATTR_GRID_PERSPECTIVE_HEIGHT = "perspective_point_height"
CF_ATTR_GRID_SWEEP_ANGLE_AXIS = "sweep_angle_axis"
CF_ATTR_GRID_AZIMUTH_CENT_LINE = "azimuth_of_central_line"
CF_ATTR_POSITIVE = "positive"
CF_ATTR_STD_NAME = "standard_name"
CF_ATTR_LONG_NAME = "long_name"
CF_ATTR_UNITS = "units"
CF_ATTR_CELL_METHODS = "cell_methods"

#
# CF Attribute Value Constants.
#
# Attribute - axis.
CF_VALUE_AXIS_X = "x"
CF_VALUE_AXIS_Y = "y"
CF_VALUE_AXIS_T = "t"
CF_VALUE_AXIS_Z = "z"


# Attribute - positive.
CF_VALUE_POSITIVE = ["down", "up"]

# Attribute - standard_name.
CF_VALUE_STD_NAME_LAT = "latitude"
CF_VALUE_STD_NAME_LON = "longitude"
CF_VALUE_STD_NAME_GRID_LAT = "grid_latitude"
CF_VALUE_STD_NAME_GRID_LON = "grid_longitude"
CF_VALUE_STD_NAME_PROJ_X = "projection_x_coordinate"
CF_VALUE_STD_NAME_PROJ_Y = "projection_y_coordinate"


################################################################################
# Handling of cell-methods.

_CM_COMMENT = "comment"
_CM_EXTRA = "extra"
_CM_INTERVAL = "interval"
_CM_METHOD = "method"
_CM_NAME = "name"
_CM_PARSE_NAME = re.compile(r"([\w_]+\s*?:\s*)+")
_CM_PARSE = re.compile(
    r"""
                           (?P<name>([\w_]+\s*?:\s*)+)
                           (?P<method>[^\s][\w_\s]+(?![\w_]*\s*?:))\s*
                           (?:
                               \(\s*
                               (?P<extra>.+)
                               \)\s*
                           )?
                       """,
    re.VERBOSE,
)

# Cell methods.
_CM_KNOWN_METHODS = [
    "point",
    "sum",
    "mean",
    "maximum",
    "minimum",
    "mid_range",
    "standard_deviation",
    "variance",
    "mode",
    "median",
]


class _WarnComboIgnoringLoad(
    iris.warnings.IrisIgnoringWarning,
    iris.warnings.IrisLoadWarning,
):
    """One-off combination of warning classes - enhances user filtering."""

    pass


class _WarnComboDefaultingLoad(
    iris.warnings.IrisDefaultingWarning,
    iris.warnings.IrisLoadWarning,
):
    """One-off combination of warning classes - enhances user filtering."""

    pass


class _WarnComboDefaultingCfLoad(
    iris.warnings.IrisCfLoadWarning,
    iris.warnings.IrisDefaultingWarning,
):
    """One-off combination of warning classes - enhances user filtering."""

    pass


class _WarnComboIgnoringCfLoad(
    iris.warnings.IrisIgnoringWarning,
    iris.warnings.IrisCfLoadWarning,
):
    """One-off combination of warning classes - enhances user filtering."""

    pass


def _split_cell_methods(nc_cell_methods: str) -> List[re.Match]:
    """Split a CF cell_methods.

    Split a CF cell_methods attribute string into a list of zero or more cell
    methods, each of which is then parsed with a regex to return a list of match
    objects.

    Parameters
    ----------
    nc_cell_methods : str
        The value of the cell methods attribute to be split.

    Returns
    -------
    nc_cell_methods_matches: list of re.Match objects
        A list of re.Match objects associated with each parsed cell method.

    Notes
    -----
    Splitting is done based on words followed by colons outside of any brackets.
    Validation of anything other than being laid out in the expected format is
    left to the calling function.

    """
    # Find name candidates
    name_start_inds = []
    for m in _CM_PARSE_NAME.finditer(nc_cell_methods):
        name_start_inds.append(m.start())

    # No matches? Must be malformed cell_method string; warn and return
    if not name_start_inds:
        msg = f"Failed to parse cell method string: {nc_cell_methods}"
        warnings.warn(msg, category=iris.warnings.IrisCfLoadWarning, stacklevel=2)
        return []

    # Remove those that fall inside brackets
    bracket_depth = 0
    for ind, cha in enumerate(nc_cell_methods):
        if cha == "(":
            bracket_depth += 1
        elif cha == ")":
            bracket_depth -= 1
            if bracket_depth < 0:
                msg = (
                    "Cell methods may be incorrectly parsed due to mismatched brackets"
                )
                warnings.warn(
                    msg,
                    category=iris.warnings.IrisCfLoadWarning,
                    stacklevel=2,
                )
        if bracket_depth > 0 and ind in name_start_inds:
            name_start_inds.remove(ind)

    # List tuples of indices of starts and ends of the cell methods in the string
    method_indices = []
    for ii in range(len(name_start_inds) - 1):
        method_indices.append((name_start_inds[ii], name_start_inds[ii + 1]))
    method_indices.append((name_start_inds[-1], len(nc_cell_methods)))

    # Index the string and match against each substring
    nc_cell_methods_matches = []
    for start_ind, end_ind in method_indices:
        nc_cell_method_str = nc_cell_methods[start_ind:end_ind]
        nc_cell_method_match = _CM_PARSE.match(nc_cell_method_str.strip())
        if not nc_cell_method_match:
            msg = f"Failed to fully parse cell method string: {nc_cell_methods}"
            warnings.warn(msg, category=iris.warnings.IrisCfLoadWarning, stacklevel=2)
            continue
        nc_cell_methods_matches.append(nc_cell_method_match)

    return nc_cell_methods_matches


class UnknownCellMethodWarning(iris.warnings.IrisUnknownCellMethodWarning):
    """Backwards compatible form of :class:`iris.warnings.IrisUnknownCellMethodWarning`."""

    # TODO: remove at the next major release.
    pass


def parse_cell_methods(nc_cell_methods, cf_name=None):
    """Parse a CF cell_methods attribute string into a tuple of zero or more CellMethod instances.

    Parameters
    ----------
    nc_cell_methods : str
        The value of the cell methods attribute to be parsed.
    cf_name : optional

    Returns
    -------
    iterable of :class:`iris.coords.CellMethod`.

    Notes
    -----
    Multiple coordinates, intervals and comments are supported.
    If a method has a non-standard name a warning will be issued, but the
    results are not affected.

    """
    msg = None
    cell_methods = []
    if nc_cell_methods is not None:
        for m in _split_cell_methods(nc_cell_methods):
            d = m.groupdict()
            method = d[_CM_METHOD]
            method = method.strip()
            # Check validity of method, allowing for multi-part methods
            # e.g. mean over years.
            method_words = method.split()
            if method_words[0].lower() not in _CM_KNOWN_METHODS:
                msg = "NetCDF variable contains unknown cell method {!r}"
                msg = msg.format(method_words[0])
                if cf_name:
                    name = "{}".format(cf_name)
                    msg = msg.replace("variable", "variable {!r}".format(name))
                else:
                    warnings.warn(
                        msg,
                        category=UnknownCellMethodWarning,
                    )
                    msg = None
            d[_CM_METHOD] = method
            name = d[_CM_NAME]
            name = name.replace(" ", "")
            name = name.rstrip(":")
            d[_CM_NAME] = tuple([n for n in name.split(":")])
            interval = []
            comment = []
            if d[_CM_EXTRA] is not None:
                #
                # tokenise the key words and field colon marker
                #
                d[_CM_EXTRA] = d[_CM_EXTRA].replace("comment:", "<<comment>><<:>>")
                d[_CM_EXTRA] = d[_CM_EXTRA].replace("interval:", "<<interval>><<:>>")
                d[_CM_EXTRA] = d[_CM_EXTRA].split("<<:>>")
                if len(d[_CM_EXTRA]) == 1:
                    comment.extend(d[_CM_EXTRA])
                else:
                    next_field_type = comment
                    for field in d[_CM_EXTRA]:
                        field_type = next_field_type
                        index = field.rfind("<<interval>>")
                        if index == 0:
                            next_field_type = interval
                            continue
                        elif index > 0:
                            next_field_type = interval
                        else:
                            index = field.rfind("<<comment>>")
                            if index == 0:
                                next_field_type = comment
                                continue
                            elif index > 0:
                                next_field_type = comment
                        if index != -1:
                            field = field[:index]
                        field_type.append(field.strip())
            #
            # cater for a shared interval over multiple axes
            #
            if len(interval):
                if len(d[_CM_NAME]) != len(interval) and len(interval) == 1:
                    interval = interval * len(d[_CM_NAME])
            #
            # cater for a shared comment over multiple axes
            #
            if len(comment):
                if len(d[_CM_NAME]) != len(comment) and len(comment) == 1:
                    comment = comment * len(d[_CM_NAME])
            d[_CM_INTERVAL] = tuple(interval)
            d[_CM_COMMENT] = tuple(comment)
            cell_method = iris.coords.CellMethod(
                d[_CM_METHOD],
                coords=d[_CM_NAME],
                intervals=d[_CM_INTERVAL],
                comments=d[_CM_COMMENT],
            )
            cell_methods.append(cell_method)
        # only prints one warning, rather than each loop
        if msg:
            warnings.warn(msg, category=UnknownCellMethodWarning)
    return tuple(cell_methods)


def _add_or_capture(
    build_func: partial,
    add_method: partial,
    cf_var: iris.fileformats.cf.CFVariable,
    destination: LoadProblems.Problem.Destination,
    attr_key: Optional[str] = None,
) -> Optional[LoadProblems.Problem]:
    """Build & add objects to the Cube, capturing problem objects - common code.

    Problems are captured in :const:`iris.loading.LOAD_PROBLEMS`.

    Parameters
    ----------
    build_func : ``functools.partial``
        A function that builds the object-to-be-added. Passed as a
        :class:`~functools.partial` instance so
        that argument complexities can be handled by the caller, while execution
        is deferred until the appropriate time within :func:`_add_or_capture`.
        The passed :class:`~functools.partial` instance must have ALL arguments
        already bound, and when called it must return the object that will be
        added to the Cube.
    add_method : ``functools.partial``
        A function that takes the object returned by `build_func` and adds it to
        the Cube. Passed as a :class:`~functools.partial` instance to allow
        further arguments to be bound by the caller.
    cf_var : iris.fileformats.cf.CFVariable
        The CFVariable object that provides the info for building the
        object-to-be-added. Used in case of an error, to build the most basic
        :class:`~iris.cube.Cube` possible - for adding to
        :const:`iris.loading.LOAD_PROBLEMS`.
    destination : LoadProblems.Problem.Destination
        Info about where the object will be added, e.g. a ``standard_name``
        might be added to :class:`~iris.cube.Cube`,
        :class:`~iris.coords.DimCoord`, etcetera. Used to provide the maximum
        information if a problem gets captured.
    attr_key : str, optional
        The attribute-of-interest on `cf_var`, if applicable. For example: in
        some cases we are building a coordinate using the entire of `cf_var` -
        no `attr_key` needed - but in other cases we are 'building' a
        standard_name by getting this key from `cf_var`.

    Returns
    -------
    iris.loading.LoadProblems.Problem or None
        The captured problem, if any; the same object that is added to
        :const:`iris.loading.LOAD_PROBLEMS`.

    See Also
    --------
    iris.loading.LoadProblems.Problem: The type of the returned object.
    iris.loading.LOAD_PROBLEMS: The destination for captured problems.
    """
    captured: Cube | dict[str, Any] | None = None
    load_problems_entry: LoadProblems.Problem | None = None

    try:
        built = build_func()

    except Exception as exc_build:
        # Problems CREATING the desired object.
        # Fully suppress further problems since we're just trying to do our
        #  best to capture objects IF possible.
        if attr_key is not None:
            captured_attr = None
            with contextlib.suppress(AttributeError):
                captured_attr = getattr(cf_var, attr_key)
            captured = {attr_key: captured_attr}
        else:
            with contextlib.suppress(Exception):
                captured = build_raw_cube(cf_var)

        load_problems_entry = LOAD_PROBLEMS.record(
            filename=cf_var.filename,
            loaded=captured,
            exception=exc_build,
            destination=destination,
            handled=False,
        )

    else:
        try:
            add_method(built)
        except Exception as exc_add:
            # Problems ADDING the built object to the Cube.
            if attr_key is not None:
                captured = {attr_key: built}
            else:
                captured = built

            load_problems_entry = LOAD_PROBLEMS.record(
                filename=cf_var.filename,
                loaded=captured,
                exception=exc_add,
                destination=destination,
                handled=False,
            )

    return load_problems_entry


################################################################################
def build_raw_cube(cf_var: cf.CFVariable) -> Cube:
    """Build the most basic Cube possible - used as a 'last resort' fallback."""
    # TODO: dataless Cubes might be an opportunity for _get_cf_var_data() to return None?
    data = _get_cf_var_data(cf_var)
    raw_attributes = {key: value for key, value in cf_var.cf_attrs()}
    # Not a real attribute, but this is 'Iris language'.
    raw_attributes["var_name"] = cf_var.cf_name
    attributes = {LimitedAttributeDict.IRIS_RAW: raw_attributes}
    return Cube(data=data, attributes=attributes)


################################################################################
def _build_name_standard(cf_var: cf.CFVariable) -> str | None:
    value = getattr(cf_var, CF_ATTR_STD_NAME, None)
    if value is not None:
        standard_name = _get_valid_standard_name(value)
    else:
        standard_name = value
    return standard_name


def _build_name_long(cf_var: cf.CFVariable) -> str | None:
    return getattr(cf_var, CF_ATTR_LONG_NAME, None)


def _build_name_var(cf_var: cf.CFVariable) -> str | None:
    return cf_var.cf_name


def build_and_add_names(engine: Engine) -> None:
    """Add standard_, long_, var_name to the Cube."""
    assert engine.cf_var is not None
    assert engine.cube is not None

    destination = LoadProblems.Problem.Destination(
        iris_class=Cube,
        identifier=engine.cf_var.cf_name,
    )

    def setter(attr_name):
        return partial(setattr, engine.cube, attr_name)

    problem = _add_or_capture(
        build_func=partial(_build_name_standard, engine.cf_var),
        add_method=setter("standard_name"),
        cf_var=engine.cf_var,
        attr_key=CF_ATTR_STD_NAME,
        destination=destination,
    )
    if problem is not None and hasattr(problem.loaded, "get"):
        assert isinstance(problem.loaded, dict)
        invalid_std_name = problem.loaded.get(CF_ATTR_STD_NAME)
        problem.handled = True
    else:
        invalid_std_name = None

    long_name_kwargs = dict(
        add_method=setter("long_name"),
        cf_var=engine.cf_var,
        attr_key=CF_ATTR_LONG_NAME,
        destination=destination,
    )
    _ = _add_or_capture(
        build_func=partial(_build_name_long, engine.cf_var),
        **long_name_kwargs,
    )

    # Store as long_name is there is space, or as attribute if not.
    if invalid_std_name is not None:
        if engine.cube.long_name is None:
            _ = _add_or_capture(
                build_func=partial(lambda: invalid_std_name),
                **long_name_kwargs,
            )
        else:
            engine.cube.attributes["invalid_standard_name"] = invalid_std_name

    _ = _add_or_capture(
        build_func=partial(_build_name_var, engine.cf_var),
        add_method=setter("var_name"),
        cf_var=engine.cf_var,
        attr_key="cf_name",
        destination=destination,
    )


################################################################################
def _add_global_attribute(cube: Cube, attr_name: Any, attr_value: Any):
    cube.attributes.globals[str(attr_name)] = attr_value


def build_and_add_global_attributes(engine: Engine):
    """Create global attributes for the Cube then add them to the Cube."""
    assert engine.cf_var is not None
    assert engine.cube is not None

    for attr_name, attr_value in engine.cf_var.cf_group.global_attributes.items():
        problem = _add_or_capture(
            build_func=partial(lambda: attr_value),
            add_method=partial(_add_global_attribute, engine.cube, attr_name),
            cf_var=engine.cf_var,
            attr_key=attr_name,
            destination=LoadProblems.Problem.Destination(
                iris_class=Cube,
                identifier=engine.cf_var.cf_name,
            ),
        )
        if problem is not None:
            stack_notes = problem.stack_trace.__notes__
            if stack_notes is None:
                stack_notes = []
            stack_notes.append(
                f"Skipping disallowed global attribute '{attr_name}' (see above error)"
            )
            problem.stack_trace.__notes__ = stack_notes


################################################################################
def build_and_add_units(engine: Engine):
    """Create a Units instance and add it to the Cube."""
    assert engine.cf_var is not None
    assert engine.cube is not None

    _ = _add_or_capture(
        build_func=partial(
            get_attr_units,
            engine.cf_var,
            getattr(engine.cube, "attributes", None),
            capture_invalid=True,
        ),
        add_method=partial(setattr, engine.cube, "units"),
        cf_var=engine.cf_var,
        attr_key=CF_ATTR_UNITS,
        destination=LoadProblems.Problem.Destination(
            iris_class=Cube,
            identifier=engine.cf_var.cf_name,
        ),
    )


################################################################################
def _build_cell_methods(cf_var: cf.CFDataVariable) -> List[iris.coords.CellMethod]:
    nc_att_cell_methods = getattr(cf_var, CF_ATTR_CELL_METHODS, None)
    return parse_cell_methods(nc_att_cell_methods, cf_var.cf_name)


def build_and_add_cell_methods(engine: Engine):
    """Create CellMethod instances and add them to the Cube."""
    assert engine.cf_var is not None
    assert engine.cube is not None

    _ = _add_or_capture(
        build_func=partial(_build_cell_methods, engine.cf_var),
        add_method=partial(setattr, engine.cube, "cell_methods"),
        cf_var=engine.cf_var,
        attr_key=CF_ATTR_CELL_METHODS,
        destination=LoadProblems.Problem.Destination(
            iris_class=Cube,
            identifier=engine.cf_var.cf_name,
        ),
    )


################################################################################
def _get_ellipsoid(cf_grid_var):
    """Build a :class:`iris.coord_systems.GeogCS`.

    Return a :class:`iris.coord_systems.GeogCS` using the relevant properties of
    `cf_grid_var`. Returns None if no relevant properties are specified.

    """
    major = getattr(cf_grid_var, CF_ATTR_GRID_SEMI_MAJOR_AXIS, None)
    minor = getattr(cf_grid_var, CF_ATTR_GRID_SEMI_MINOR_AXIS, None)
    inverse_flattening = getattr(cf_grid_var, CF_ATTR_GRID_INVERSE_FLATTENING, None)

    # Avoid over-specification exception.
    if major is not None and minor is not None:
        inverse_flattening = None

    # Check for a default spherical earth.
    if major is None and minor is None and inverse_flattening is None:
        major = getattr(cf_grid_var, CF_ATTR_GRID_EARTH_RADIUS, None)

    datum = getattr(cf_grid_var, CF_ATTR_GRID_DATUM, None)
    # Check crs_wkt if no datum
    if datum is None:
        crs_wkt = getattr(cf_grid_var, CF_ATTR_GRID_CRS_WKT, None)
        if crs_wkt is not None:
            proj_crs = pyproj.crs.CRS.from_wkt(crs_wkt)
            if proj_crs.datum is not None:
                datum = proj_crs.datum.name

    # An unknown crs datum will be treated as None
    if datum == "unknown":
        datum = None

    if datum is not None and not iris.FUTURE.datum_support:
        wmsg = (
            "Ignoring a datum in netCDF load for consistency with existing "
            "behaviour. In a future version of Iris, this datum will be "
            "applied. To apply the datum when loading, use the "
            "iris.FUTURE.datum_support flag."
        )
        warnings.warn(wmsg, category=FutureWarning, stacklevel=14)
        datum = None

    if datum is not None:
        return iris.coord_systems.GeogCS.from_datum(datum)
    elif major is None and minor is None and inverse_flattening is None:
        return None
    else:
        return iris.coord_systems.GeogCS(major, minor, inverse_flattening)


################################################################################
def build_coordinate_system(engine, cf_grid_var):
    """Create a coordinate system from the CF-netCDF grid mapping variable."""
    coord_system = _get_ellipsoid(cf_grid_var)
    if coord_system is None:
        raise ValueError("No ellipsoid specified")
    else:
        return coord_system


################################################################################
def build_rotated_coordinate_system(engine, cf_grid_var):
    """Create a rotated coordinate system from the CF-netCDF grid mapping variable."""
    ellipsoid = _get_ellipsoid(cf_grid_var)

    north_pole_latitude = getattr(cf_grid_var, CF_ATTR_GRID_NORTH_POLE_LAT, 90.0)
    north_pole_longitude = getattr(cf_grid_var, CF_ATTR_GRID_NORTH_POLE_LON, 0.0)
    if north_pole_latitude is None or north_pole_longitude is None:
        warnings.warn(
            "Rotated pole position is not fully specified",
            category=iris.warnings.IrisCfLoadWarning,
        )

    north_pole_grid_lon = getattr(cf_grid_var, CF_ATTR_GRID_NORTH_POLE_GRID_LON, 0.0)

    rcs = iris.coord_systems.RotatedGeogCS(
        north_pole_latitude,
        north_pole_longitude,
        north_pole_grid_lon,
        ellipsoid,
    )

    return rcs


################################################################################
def build_transverse_mercator_coordinate_system(engine, cf_grid_var):
    """Create a transverse Mercator coordinate system from the CF-netCDF grid mapping variable."""
    ellipsoid = _get_ellipsoid(cf_grid_var)

    latitude_of_projection_origin = getattr(
        cf_grid_var, CF_ATTR_GRID_LAT_OF_PROJ_ORIGIN, None
    )
    longitude_of_central_meridian = getattr(
        cf_grid_var, CF_ATTR_GRID_LON_OF_CENT_MERIDIAN, None
    )
    false_easting = getattr(cf_grid_var, CF_ATTR_GRID_FALSE_EASTING, None)
    false_northing = getattr(cf_grid_var, CF_ATTR_GRID_FALSE_NORTHING, None)
    scale_factor_at_central_meridian = getattr(
        cf_grid_var, CF_ATTR_GRID_SCALE_FACTOR_AT_CENT_MERIDIAN, None
    )

    # The following accounts for the inconsistency in the transverse
    # mercator description within the CF spec.
    if longitude_of_central_meridian is None:
        longitude_of_central_meridian = getattr(
            cf_grid_var, CF_ATTR_GRID_LON_OF_PROJ_ORIGIN, None
        )
    if scale_factor_at_central_meridian is None:
        scale_factor_at_central_meridian = getattr(
            cf_grid_var, CF_ATTR_GRID_SCALE_FACTOR_AT_PROJ_ORIGIN, None
        )

    cs = iris.coord_systems.TransverseMercator(
        latitude_of_projection_origin,
        longitude_of_central_meridian,
        false_easting,
        false_northing,
        scale_factor_at_central_meridian,
        ellipsoid,
    )

    return cs


################################################################################
def build_lambert_conformal_coordinate_system(engine, cf_grid_var):
    """Create a Lambert conformal conic coordinate system from the CF-netCDF grid mapping variable."""
    ellipsoid = _get_ellipsoid(cf_grid_var)

    latitude_of_projection_origin = getattr(
        cf_grid_var, CF_ATTR_GRID_LAT_OF_PROJ_ORIGIN, None
    )
    longitude_of_central_meridian = getattr(
        cf_grid_var, CF_ATTR_GRID_LON_OF_CENT_MERIDIAN, None
    )
    false_easting = getattr(cf_grid_var, CF_ATTR_GRID_FALSE_EASTING, None)
    false_northing = getattr(cf_grid_var, CF_ATTR_GRID_FALSE_NORTHING, None)
    standard_parallel = getattr(cf_grid_var, CF_ATTR_GRID_STANDARD_PARALLEL, None)

    cs = iris.coord_systems.LambertConformal(
        latitude_of_projection_origin,
        longitude_of_central_meridian,
        false_easting,
        false_northing,
        standard_parallel,
        ellipsoid,
    )

    return cs


################################################################################
def build_stereographic_coordinate_system(engine, cf_grid_var):
    """Create a stereographic coordinate system from the CF-netCDF grid mapping variable."""
    ellipsoid = _get_ellipsoid(cf_grid_var)

    latitude_of_projection_origin = getattr(
        cf_grid_var, CF_ATTR_GRID_LAT_OF_PROJ_ORIGIN, None
    )
    longitude_of_projection_origin = getattr(
        cf_grid_var, CF_ATTR_GRID_LON_OF_PROJ_ORIGIN, None
    )
    scale_factor_at_projection_origin = getattr(
        cf_grid_var, CF_ATTR_GRID_SCALE_FACTOR_AT_PROJ_ORIGIN, None
    )

    false_easting = getattr(cf_grid_var, CF_ATTR_GRID_FALSE_EASTING, None)
    false_northing = getattr(cf_grid_var, CF_ATTR_GRID_FALSE_NORTHING, None)

    cs = iris.coord_systems.Stereographic(
        latitude_of_projection_origin,
        longitude_of_projection_origin,
        false_easting,
        false_northing,
        true_scale_lat=None,
        scale_factor_at_projection_origin=scale_factor_at_projection_origin,
        ellipsoid=ellipsoid,
    )

    return cs


################################################################################
def build_polar_stereographic_coordinate_system(engine, cf_grid_var):
    """Create a polar stereographic coordinate system from the CF-netCDF grid mapping variable."""
    ellipsoid = _get_ellipsoid(cf_grid_var)

    latitude_of_projection_origin = getattr(
        cf_grid_var, CF_ATTR_GRID_LAT_OF_PROJ_ORIGIN, None
    )
    longitude_of_projection_origin = getattr(
        cf_grid_var, CF_ATTR_GRID_STRAIGHT_VERT_LON, None
    )
    true_scale_lat = getattr(cf_grid_var, CF_ATTR_GRID_STANDARD_PARALLEL, None)
    scale_factor_at_projection_origin = getattr(
        cf_grid_var, CF_ATTR_GRID_SCALE_FACTOR_AT_PROJ_ORIGIN, None
    )

    false_easting = getattr(cf_grid_var, CF_ATTR_GRID_FALSE_EASTING, None)
    false_northing = getattr(cf_grid_var, CF_ATTR_GRID_FALSE_NORTHING, None)

    cs = iris.coord_systems.PolarStereographic(
        latitude_of_projection_origin,
        longitude_of_projection_origin,
        false_easting,
        false_northing,
        true_scale_lat,
        scale_factor_at_projection_origin,
        ellipsoid=ellipsoid,
    )

    return cs


################################################################################
def build_mercator_coordinate_system(engine, cf_grid_var):
    """Create a Mercator coordinate system from the CF-netCDF grid mapping variable."""
    ellipsoid = _get_ellipsoid(cf_grid_var)

    longitude_of_projection_origin = getattr(
        cf_grid_var, CF_ATTR_GRID_LON_OF_PROJ_ORIGIN, None
    )
    standard_parallel = getattr(cf_grid_var, CF_ATTR_GRID_STANDARD_PARALLEL, None)
    false_easting = getattr(cf_grid_var, CF_ATTR_GRID_FALSE_EASTING, None)
    false_northing = getattr(cf_grid_var, CF_ATTR_GRID_FALSE_NORTHING, None)
    scale_factor_at_projection_origin = getattr(
        cf_grid_var, CF_ATTR_GRID_SCALE_FACTOR_AT_PROJ_ORIGIN, None
    )

    cs = iris.coord_systems.Mercator(
        longitude_of_projection_origin,
        ellipsoid=ellipsoid,
        standard_parallel=standard_parallel,
        scale_factor_at_projection_origin=scale_factor_at_projection_origin,
        false_easting=false_easting,
        false_northing=false_northing,
    )

    return cs


################################################################################
def build_lambert_azimuthal_equal_area_coordinate_system(engine, cf_grid_var):
    """Create a lambert azimuthal equal area coordinate system from the CF-netCDF grid mapping variable."""
    ellipsoid = _get_ellipsoid(cf_grid_var)

    latitude_of_projection_origin = getattr(
        cf_grid_var, CF_ATTR_GRID_LAT_OF_PROJ_ORIGIN, None
    )
    longitude_of_projection_origin = getattr(
        cf_grid_var, CF_ATTR_GRID_LON_OF_PROJ_ORIGIN, None
    )
    false_easting = getattr(cf_grid_var, CF_ATTR_GRID_FALSE_EASTING, None)
    false_northing = getattr(cf_grid_var, CF_ATTR_GRID_FALSE_NORTHING, None)

    cs = iris.coord_systems.LambertAzimuthalEqualArea(
        latitude_of_projection_origin,
        longitude_of_projection_origin,
        false_easting,
        false_northing,
        ellipsoid,
    )

    return cs


################################################################################
def build_albers_equal_area_coordinate_system(engine, cf_grid_var):
    """Create a albers conical equal area coordinate system from the CF-netCDF grid mapping variable."""
    ellipsoid = _get_ellipsoid(cf_grid_var)

    latitude_of_projection_origin = getattr(
        cf_grid_var, CF_ATTR_GRID_LAT_OF_PROJ_ORIGIN, None
    )
    longitude_of_central_meridian = getattr(
        cf_grid_var, CF_ATTR_GRID_LON_OF_CENT_MERIDIAN, None
    )
    false_easting = getattr(cf_grid_var, CF_ATTR_GRID_FALSE_EASTING, None)
    false_northing = getattr(cf_grid_var, CF_ATTR_GRID_FALSE_NORTHING, None)
    standard_parallels = getattr(cf_grid_var, CF_ATTR_GRID_STANDARD_PARALLEL, None)

    cs = iris.coord_systems.AlbersEqualArea(
        latitude_of_projection_origin,
        longitude_of_central_meridian,
        false_easting,
        false_northing,
        standard_parallels,
        ellipsoid,
    )

    return cs


################################################################################
def build_vertical_perspective_coordinate_system(engine, cf_grid_var):
    """Create a vertical perspective coordinate system from the CF-netCDF grid mapping variables."""
    ellipsoid = _get_ellipsoid(cf_grid_var)

    latitude_of_projection_origin = getattr(
        cf_grid_var, CF_ATTR_GRID_LAT_OF_PROJ_ORIGIN, None
    )
    longitude_of_projection_origin = getattr(
        cf_grid_var, CF_ATTR_GRID_LON_OF_PROJ_ORIGIN, None
    )
    perspective_point_height = getattr(
        cf_grid_var, CF_ATTR_GRID_PERSPECTIVE_HEIGHT, None
    )
    false_easting = getattr(cf_grid_var, CF_ATTR_GRID_FALSE_EASTING, None)
    false_northing = getattr(cf_grid_var, CF_ATTR_GRID_FALSE_NORTHING, None)

    cs = iris.coord_systems.VerticalPerspective(
        latitude_of_projection_origin,
        longitude_of_projection_origin,
        perspective_point_height,
        false_easting,
        false_northing,
        ellipsoid,
    )

    return cs


################################################################################
def build_geostationary_coordinate_system(engine, cf_grid_var):
    """Create a geostationary coordinate system from the CF-netCDF grid mapping variable."""
    ellipsoid = _get_ellipsoid(cf_grid_var)

    latitude_of_projection_origin = getattr(
        cf_grid_var, CF_ATTR_GRID_LAT_OF_PROJ_ORIGIN, None
    )
    longitude_of_projection_origin = getattr(
        cf_grid_var, CF_ATTR_GRID_LON_OF_PROJ_ORIGIN, None
    )
    perspective_point_height = getattr(
        cf_grid_var, CF_ATTR_GRID_PERSPECTIVE_HEIGHT, None
    )
    false_easting = getattr(cf_grid_var, CF_ATTR_GRID_FALSE_EASTING, None)
    false_northing = getattr(cf_grid_var, CF_ATTR_GRID_FALSE_NORTHING, None)
    sweep_angle_axis = getattr(cf_grid_var, CF_ATTR_GRID_SWEEP_ANGLE_AXIS, None)

    cs = iris.coord_systems.Geostationary(
        latitude_of_projection_origin,
        longitude_of_projection_origin,
        perspective_point_height,
        sweep_angle_axis,
        false_easting,
        false_northing,
        ellipsoid,
    )

    return cs


################################################################################
def build_oblique_mercator_coordinate_system(engine, cf_grid_var):
    """Create an oblique mercator coordinate system from the CF-netCDF grid mapping variable."""
    ellipsoid = _get_ellipsoid(cf_grid_var)

    azimuth_of_central_line = getattr(cf_grid_var, CF_ATTR_GRID_AZIMUTH_CENT_LINE, None)
    latitude_of_projection_origin = getattr(
        cf_grid_var, CF_ATTR_GRID_LAT_OF_PROJ_ORIGIN, None
    )
    longitude_of_projection_origin = getattr(
        cf_grid_var, CF_ATTR_GRID_LON_OF_PROJ_ORIGIN, None
    )
    scale_factor_at_projection_origin = getattr(
        cf_grid_var, CF_ATTR_GRID_SCALE_FACTOR_AT_PROJ_ORIGIN, None
    )
    false_easting = getattr(cf_grid_var, CF_ATTR_GRID_FALSE_EASTING, None)
    false_northing = getattr(cf_grid_var, CF_ATTR_GRID_FALSE_NORTHING, None)
    kwargs = dict(
        azimuth_of_central_line=azimuth_of_central_line,
        latitude_of_projection_origin=latitude_of_projection_origin,
        longitude_of_projection_origin=longitude_of_projection_origin,
        scale_factor_at_projection_origin=scale_factor_at_projection_origin,
        false_easting=false_easting,
        false_northing=false_northing,
        ellipsoid=ellipsoid,
    )

    # Handle the alternative form noted in CF: rotated mercator.
    grid_mapping_name = getattr(cf_grid_var, CF_ATTR_GRID_MAPPING_NAME)
    candidate_systems = dict(
        oblique_mercator=iris.coord_systems.ObliqueMercator,
        rotated_mercator=iris.coord_systems.RotatedMercator,
    )
    if grid_mapping_name == "rotated_mercator":
        message = (
            "Iris will stop loading the rotated_mercator grid mapping name in "
            "a future release, in accordance with CF version 1.11 . Instead "
            "please use oblique_mercator with azimuth_of_central_line = 90 ."
        )
        warn_deprecated(message)
        del kwargs[CF_ATTR_GRID_AZIMUTH_CENT_LINE]

    cs = candidate_systems[grid_mapping_name](**kwargs)
    return cs


################################################################################
def get_attr_units(cf_var, attributes, capture_invalid=False):
    attr_units = getattr(cf_var, CF_ATTR_UNITS, UNKNOWN_UNIT_STRING)
    if not attr_units:
        attr_units = UNKNOWN_UNIT_STRING

    # Sanitise lat/lon units.
    if attr_units in UD_UNITS_LAT or attr_units in UD_UNITS_LON:
        attr_units = "degrees"

    # Graceful loading of invalid units.
    invalid_units_message = (
        f"{{prefix}} units '{attr_units}' on netCDF variable '{cf_var.cf_name}'."
    )
    try:
        cf_units.as_unit(attr_units)
    except ValueError as invalid_units_error:
        if capture_invalid:
            # This block is only expected when getting Cube units.
            assert isinstance(cf_var, cf.CFDataVariable)
            try:
                raise invalid_units_error.__class__(
                    invalid_units_message.format(prefix="Invalid")
                ) from invalid_units_error
            except invalid_units_error.__class__ as error:
                _ = LOAD_PROBLEMS.record(
                    filename=cf_var.filename,
                    loaded={CF_ATTR_UNITS: attr_units},
                    exception=error,
                    destination=LoadProblems.Problem.Destination(
                        iris_class=Cube,
                        identifier=cf_var.cf_name,
                    ),
                    handled=True,
                )

        else:
            warnings.warn(
                invalid_units_message.format(prefix="Ignoring invalid"),
                category=_WarnComboIgnoringCfLoad,
            )

        attributes["invalid_units"] = attr_units
        attr_units = UNKNOWN_UNIT_STRING

    if np.issubdtype(cf_var.dtype, np.str_):
        attr_units = NO_UNIT_STRING

    if any(
        hasattr(cf_var.cf_data, name)
        for name in ("flag_values", "flag_masks", "flag_meanings")
    ):
        attr_units = cf_units._NO_UNIT_STRING

    # Get any associated calendar for a time reference coordinate.
    if cf_units.as_unit(attr_units).is_time_reference():
        attr_calendar = getattr(cf_var, CF_ATTR_CALENDAR, None)

        if attr_calendar:
            attr_units = cf_units.Unit(attr_units, calendar=attr_calendar)

    return attr_units


################################################################################
def get_names(cf_coord_var, coord_name, attributes):
    """Determine the standard_name, long_name and var_name attributes."""
    standard_name = getattr(cf_coord_var, CF_ATTR_STD_NAME, None)
    long_name = getattr(cf_coord_var, CF_ATTR_LONG_NAME, None)
    cf_name = str(cf_coord_var.cf_name)

    if standard_name is not None:
        try:
            standard_name = _get_valid_standard_name(standard_name)
        except ValueError:
            if long_name is not None:
                attributes["invalid_standard_name"] = standard_name
                if coord_name is not None:
                    standard_name = coord_name
                else:
                    standard_name = None
            else:
                if coord_name is not None:
                    attributes["invalid_standard_name"] = standard_name
                    standard_name = coord_name
                else:
                    standard_name = None

    else:
        if coord_name is not None:
            standard_name = coord_name

    # Last attempt to set the standard name to something meaningful.
    if standard_name is None:
        if cf_name in iris.std_names.STD_NAMES:
            standard_name = cf_name

    return (standard_name, long_name, cf_name)


################################################################################
def get_cf_bounds_var(cf_coord_var):
    """Return the CF variable representing the bounds of a coordinate variable."""
    attr_bounds = getattr(cf_coord_var, CF_ATTR_BOUNDS, None)
    attr_climatology = getattr(cf_coord_var, CF_ATTR_CLIMATOLOGY, None)

    # Determine bounds, preferring standard bounds over climatology.
    # NB. No need to raise a warning if the bounds/climatology
    # variable is missing, as that will already have been done by
    # iris.fileformats.cf.
    cf_bounds_var = None
    climatological = False
    if attr_bounds is not None:
        bounds_vars = cf_coord_var.cf_group.bounds
        if attr_bounds in bounds_vars:
            cf_bounds_var = bounds_vars[attr_bounds]
    elif attr_climatology is not None:
        climatology_vars = cf_coord_var.cf_group.climatology
        if attr_climatology in climatology_vars:
            cf_bounds_var = climatology_vars[attr_climatology]
            climatological = True

    if attr_bounds is not None and attr_climatology is not None:
        warnings.warn(
            "Ignoring climatology in favour of bounds attribute "
            "on NetCDF variable {!r}.".format(cf_coord_var.cf_name),
            category=_WarnComboIgnoringCfLoad,
        )

    return cf_bounds_var, climatological


################################################################################
def reorder_bounds_data(bounds_data, cf_bounds_var, cf_coord_var):
    """Return a bounds_data array with the vertex dimension as the most rapidly varying.

    .. note::

        This function assumes the dimension names of the coordinate
        variable match those of the bounds variable in order to determine
        which is the vertex dimension.


    """
    vertex_dim_names = set(cf_bounds_var.dimensions).difference(cf_coord_var.dimensions)
    if len(vertex_dim_names) != 1:
        msg = (
            "Too many dimension names differ between coordinate "
            "variable {!r} and the bounds variable {!r}. "
            "Expected 1, got {}."
        )
        raise ValueError(
            msg.format(
                str(cf_coord_var.cf_name),
                str(cf_bounds_var.cf_name),
                len(vertex_dim_names),
            )
        )
    vertex_dim = cf_bounds_var.dimensions.index(*vertex_dim_names)
    bounds_data = np.rollaxis(bounds_data.view(), vertex_dim, len(bounds_data.shape))
    return bounds_data


################################################################################
def _normalise_bounds_units(
    points_units: str | None,
    cf_bounds_var: CFBoundaryVariable,
    bounds_data: MaskedArray,
) -> Optional[MaskedArray]:
    """Ensure bounds have units compatible with points.

    If required, the `bounds_data` will be converted to the `points_units`.
    If the bounds units are not convertible, a warning will be issued and
    the `bounds_data` will be ignored.

    Bounds with invalid units will be gracefully left unconverted and passed through.

    Parameters
    ----------
    points_units : str
        The units of the coordinate points.
    cf_bounds_var : CFBoundaryVariable
        The serialized NetCDF bounds variable.
    bounds_data : MaskedArray
        The pre-processed data of the bounds variable.

    Returns
    -------
    MaskedArray or None
        The bounds data with the same units as the points, or ``None``
        if the bounds units are not convertible to the points units.

    """
    bounds_units = get_attr_units(cf_bounds_var, {})
    result: MaskedArray | None = bounds_data

    if bounds_units != UNKNOWN_UNIT_STRING:
        p_units = cf_units.Unit(points_units)
        b_units = cf_units.Unit(bounds_units)

        if b_units != p_units:
            if b_units.is_convertible(p_units):
                result = b_units.convert(bounds_data, p_units)
            else:
                wmsg = (
                    f"Ignoring bounds on NetCDF variable {cf_bounds_var.cf_name!r}. "
                    f"Expected units compatible with {p_units.origin!r}, got "
                    f"{b_units.origin!r}."
                )
                warnings.warn(
                    wmsg, category=iris.warnings.IrisCfLoadWarning, stacklevel=2
                )
                result = None

    return result


################################################################################
def _build_dimension_coordinate(
    cf_coord_var: cf.CFCoordinateVariable,
    coord_name: Optional[str] = None,
    coord_system: Optional[iris.coord_systems.CoordSystem] = None,
) -> iris.coords.Coord:
    attributes: dict[str, Any] = {}

    attr_units = get_attr_units(cf_coord_var, attributes)
    points_data = cf_coord_var[:]
    # Gracefully fill points masked array.
    if ma.is_masked(points_data):
        points_data = ma.filled(points_data)
        msg = "Gracefully filling {!r} dimension coordinate masked points"
        warnings.warn(
            msg.format(str(cf_coord_var.cf_name)),
            category=_WarnComboDefaultingLoad,
        )

    # Get any coordinate bounds.
    cf_bounds_var, climatological = get_cf_bounds_var(cf_coord_var)
    if cf_bounds_var is not None:
        bounds_data = cf_bounds_var[:]
        # Gracefully fill bounds masked array.
        if ma.is_masked(bounds_data):
            bounds_data = ma.filled(bounds_data)
            msg = "Gracefully filling {!r} dimension coordinate masked bounds"
            warnings.warn(
                msg.format(str(cf_coord_var.cf_name)),
                category=_WarnComboDefaultingLoad,
            )
        # Handle transposed bounds where the vertex dimension is not
        # the last one. Test based on shape to support different
        # dimension names.
        if cf_bounds_var.shape[:-1] != cf_coord_var.shape:
            bounds_data = reorder_bounds_data(bounds_data, cf_bounds_var, cf_coord_var)

        bounds_data = _normalise_bounds_units(attr_units, cf_bounds_var, bounds_data)
    else:
        bounds_data = None

    # Determine whether the coordinate is circular.
    circular = False
    if (
        points_data.ndim == 1
        and coord_name in [CF_VALUE_STD_NAME_LON, CF_VALUE_STD_NAME_GRID_LON]
        and cf_units.Unit(attr_units)
        in [cf_units.Unit("radians"), cf_units.Unit("degrees")]
    ):
        modulus_value = cf_units.Unit(attr_units).modulus
        circular = iris.util._is_circular(
            points_data, modulus_value, bounds=bounds_data
        )

    # Determine the standard_name, long_name and var_name
    standard_name, long_name, var_name = get_names(cf_coord_var, coord_name, attributes)

    coord = iris.coords.DimCoord(
        points_data,
        standard_name=standard_name,
        long_name=long_name,
        var_name=var_name,
        units=attr_units,
        bounds=bounds_data,
        attributes=attributes,
        circular=circular,
        climatological=climatological,
    )

    assert coord.var_name is not None
    # Part 2 - only adding - following on from building in
    #  actions.action_provides_grid_mapping()
    _ = _add_or_capture(
        build_func=partial(lambda: coord_system),
        add_method=partial(setattr, coord, "coord_system"),
        # cf_var is usually the variable for the thing we are building.
        #  In this case coord_system was built earlier; cf_coord_var is here
        #  only to provide the filename.
        cf_var=cf_coord_var,
        destination=LoadProblems.Problem.Destination(
            iris_class=iris.coords.DimCoord,
            identifier=coord.var_name,
        ),
    )

    return coord


def _add_dimension_coordinate(
    engine: Engine,
    cf_coord_var: cf.CFCoordinateVariable,
    coord: iris.coords.DimCoord | iris.coords.AuxCoord,
) -> None:
    assert engine.cf_var is not None
    assert engine.cube is not None
    assert engine.cube_parts is not None

    # Determine the name of the dimension/s shared between the CF-netCDF
    #  data variable and the coordinate being built.
    common_dims = [
        dim for dim in cf_coord_var.dimensions if dim in engine.cf_var.dimensions
    ]
    data_dims = None
    if common_dims:
        # Calculate the offset of each common dimension.
        data_dims = [int(engine.cf_var.dimensions.index(dim)) for dim in common_dims]

    if hasattr(coord, "circular") and data_dims is not None:
        # Appease MyPy. The check itself uses duck typing to avoid any
        #  silent errors when Mocking.
        assert isinstance(coord, iris.coords.DimCoord)
        try:
            (data_dim,) = data_dims
        except ValueError:
            message = (
                "Expected single dimension for dimension coordinate "
                f"{coord.var_name}, got: {data_dims}."
            )
            raise ValueError(message)
        engine.cube.add_dim_coord(coord, data_dim)
    else:
        # Should work fine for scalar coords - data_dims passed as None.
        engine.cube.add_aux_coord(coord, data_dims)

    # Update the coordinate to CF-netCDF variable mapping.
    engine.cube_parts["coordinates"].append((coord, cf_coord_var.cf_name))


def build_and_add_dimension_coordinate(
    engine: Engine,
    cf_coord_var: cf.CFCoordinateVariable,
    coord_name: Optional[str] = None,
    coord_system: Optional[iris.coord_systems.CoordSystem] = None,
):
    """Create a DimCoord instance and add it to the Cube."""
    assert engine.cf_var is not None

    destination = LoadProblems.Problem.Destination(
        iris_class=Cube,
        identifier=engine.cf_var.cf_name,
    )

    problem = _add_or_capture(
        build_func=partial(
            _build_dimension_coordinate,
            cf_coord_var,
            coord_name,
            coord_system,
        ),
        add_method=partial(_add_dimension_coordinate, engine, cf_coord_var),
        cf_var=cf_coord_var,
        destination=destination,
    )
    if problem is not None:
        coord_var_name = str(cf_coord_var.cf_name)
        stack_notes = problem.stack_trace.__notes__
        if stack_notes is None:
            stack_notes = []
        stack_notes.append(
            f"Failed to create {coord_var_name} dimension coordinate:\n"
            f"Gracefully creating {coord_var_name!r} auxiliary coordinate instead."
        )
        problem.stack_trace.__notes__ = stack_notes
        problem.handled = True

        _ = _add_or_capture(
            build_func=partial(
                _build_auxiliary_coordinate,
                engine,
                cf_coord_var,
                coord_name,
                coord_system,
            ),
            add_method=partial(_add_auxiliary_coordinate, engine, cf_coord_var),
            cf_var=cf_coord_var,
            destination=destination,
        )


################################################################################
def _build_auxiliary_coordinate(
    engine: Engine,
    cf_coord_var: cf.CFCoordinateVariable | cf.CFAuxiliaryCoordinateVariable,
    coord_name: Optional[str] = None,
    coord_system: Optional[iris.coord_systems.CoordSystem] = None,
) -> iris.coords.AuxCoord:
    assert engine.cf_var is not None

    attributes: dict[str, Any] = {}

    # Get units
    attr_units = get_attr_units(cf_coord_var, attributes)

    # Get any coordinate point data.
    if isinstance(cf_coord_var, cf.CFLabelVariable):
        points_data = cf_coord_var.cf_label_data(engine.cf_var)
    else:
        points_data = _get_cf_var_data(cf_coord_var)

    # Get any coordinate bounds.
    cf_bounds_var, climatological = get_cf_bounds_var(cf_coord_var)
    if cf_bounds_var is not None:
        bounds_data = _get_cf_var_data(cf_bounds_var)

        # Handle transposed bounds where the vertex dimension is not
        # the last one. Test based on shape to support different
        # dimension names.
        if cf_bounds_var.shape[:-1] != cf_coord_var.shape:
            # Resolving the data to a numpy array (i.e. *not* masked) for
            # compatibility with array creators (i.e. dask)
            bounds_data = np.asarray(bounds_data)
            bounds_data = reorder_bounds_data(bounds_data, cf_bounds_var, cf_coord_var)

        bounds_data = _normalise_bounds_units(attr_units, cf_bounds_var, bounds_data)
    else:
        bounds_data = None

    # Determine the standard_name, long_name and var_name
    standard_name, long_name, var_name = get_names(cf_coord_var, coord_name, attributes)

    # Create the coordinate
    coord = iris.coords.AuxCoord(
        points_data,
        standard_name=standard_name,
        long_name=long_name,
        var_name=var_name,
        units=attr_units,
        bounds=bounds_data,
        attributes=attributes,
        climatological=climatological,
    )

    assert coord.var_name is not None
    # Part 2 - only adding - following on from building in
    #  actions.action_provides_grid_mapping()
    _ = _add_or_capture(
        build_func=partial(lambda: coord_system),
        add_method=partial(setattr, coord, "coord_system"),
        # cf_var is usually the variable for the thing we are building.
        #  In this case coord_system was built earlier; cf_coord_var is here
        #  only to provide the filename.
        cf_var=cf_coord_var,
        destination=LoadProblems.Problem.Destination(
            iris_class=iris.coords.AuxCoord,
            identifier=coord.var_name,
        ),
    )

    return coord


def _add_auxiliary_coordinate(
    engine: Engine,
    cf_coord_var: cf.CFCoordinateVariable | cf.CFAuxiliaryCoordinateVariable,
    coord: iris.coords.DimCoord | iris.coords.AuxCoord,
) -> None:
    assert engine.cf_var is not None
    assert engine.cube is not None
    assert engine.cube_parts is not None

    # Determine the name of the dimension/s shared between the CF-netCDF data variable
    # and the coordinate being built.
    common_dims = [
        dim for dim in cf_coord_var.dimensions if dim in engine.cf_var.dimensions
    ]
    data_dims = None
    if common_dims:
        # Calculate the offset of each common dimension.
        data_dims = [engine.cf_var.dimensions.index(dim) for dim in common_dims]

    engine.cube.add_aux_coord(coord, data_dims)

    # Make a list with names, stored on the engine, so we can find them all later.
    engine.cube_parts["coordinates"].append((coord, cf_coord_var.cf_name))


def build_and_add_auxiliary_coordinate(
    engine: Engine,
    cf_coord_var: cf.CFAuxiliaryCoordinateVariable,
    coord_name: Optional[str] = None,
    coord_system: Optional[iris.coord_systems.CoordSystem] = None,
):
    """Create a AuxCoord instance and add it to the Cube."""
    assert engine.cf_var is not None

    _ = _add_or_capture(
        build_func=partial(
            _build_auxiliary_coordinate,
            engine,
            cf_coord_var,
            coord_name,
            coord_system,
        ),
        add_method=partial(_add_auxiliary_coordinate, engine, cf_coord_var),
        cf_var=cf_coord_var,
        destination=LoadProblems.Problem.Destination(
            iris_class=Cube,
            identifier=engine.cf_var.cf_name,
        ),
    )


################################################################################
def _build_cell_measure(cf_cm_var: cf.CFMeasureVariable) -> iris.coords.CellMeasure:
    attributes: dict[str, Any] = {}

    # Get units
    attr_units = get_attr_units(cf_cm_var, attributes)

    # Get (lazy) content array
    data = _get_cf_var_data(cf_cm_var)

    # Determine the standard_name, long_name and var_name
    standard_name, long_name, var_name = get_names(cf_cm_var, None, attributes)

    # Obtain the cf_measure.
    measure = cf_cm_var.cf_measure

    # Create the CellMeasure
    cell_measure = iris.coords.CellMeasure(
        data,
        standard_name=standard_name,
        long_name=long_name,
        var_name=var_name,
        units=attr_units,
        attributes=attributes,
        measure=measure,
    )

    return cell_measure


def _add_cell_measure(
    engine: Engine,
    cf_cm_var: cf.CFMeasureVariable,
    cell_measure: iris.coords.CellMeasure,
) -> None:
    assert engine.cf_var is not None
    assert engine.cube is not None
    assert engine.cube_parts is not None

    cf_var = engine.cf_var
    cube = engine.cube

    # Determine the name of the dimension/s shared between the CF-netCDF data
    #  variable and the coordinate being built.
    common_dims = [dim for dim in cf_cm_var.dimensions if dim in cf_var.dimensions]
    data_dims = None
    if common_dims:
        # Calculate the offset of each common dimension.
        data_dims = [cf_var.dimensions.index(dim) for dim in common_dims]

    # Add it to the cube
    cube.add_cell_measure(cell_measure, data_dims)
    # Make a list with names, stored on the engine, so we can find them all later.
    engine.cube_parts["cell_measures"].append((cell_measure, cf_cm_var.cf_name))


def build_and_add_cell_measure(
    engine: Engine,
    cf_cm_var: cf.CFMeasureVariable,
) -> None:
    """Create a CellMeasure instance and add it to the Cube."""
    assert engine.cf_var is not None

    _ = _add_or_capture(
        build_func=partial(_build_cell_measure, cf_cm_var),
        add_method=partial(_add_cell_measure, engine, cf_cm_var),
        cf_var=cf_cm_var,
        destination=LoadProblems.Problem.Destination(
            iris_class=Cube,
            identifier=engine.cf_var.cf_name,
        ),
    )


################################################################################
def _build_ancil_var(
    cf_av_var: cf.CFAncillaryDataVariable,
) -> iris.coords.AncillaryVariable:
    attributes: dict[str, Any] = {}

    # Get units
    attr_units = get_attr_units(cf_av_var, attributes)

    # Get (lazy) content array
    data = _get_cf_var_data(cf_av_var)

    # Determine the standard_name, long_name and var_name
    standard_name, long_name, var_name = get_names(cf_av_var, None, attributes)

    # Create the AncillaryVariable
    av = iris.coords.AncillaryVariable(
        data,
        standard_name=standard_name,
        long_name=long_name,
        var_name=var_name,
        units=attr_units,
        attributes=attributes,
    )

    return av


def _add_ancil_var(
    engine: Engine,
    cf_av_var: cf.CFAncillaryDataVariable,
    av: iris.coords.AncillaryVariable,
) -> None:
    assert engine.cf_var is not None
    assert engine.cube is not None
    assert engine.cube_parts is not None

    cf_var = engine.cf_var
    cube = engine.cube

    # Determine the name of the dimension/s shared between the CF-netCDF data variable
    #  and the AV being built.
    common_dims = [dim for dim in cf_av_var.dimensions if dim in cf_var.dimensions]
    data_dims = None
    if common_dims:
        # Calculate the offset of each common dimension.
        data_dims = [cf_var.dimensions.index(dim) for dim in common_dims]

    # Add it to the cube
    cube.add_ancillary_variable(av, data_dims)
    # Make a list with names, stored on the engine, so we can find them all later.
    engine.cube_parts["ancillary_variables"].append((av, cf_av_var.cf_name))


def build_and_add_ancil_var(
    engine: Engine,
    cf_av_var: cf.CFAncillaryDataVariable,
) -> None:
    """Create an AncillaryVariable instance and add it to the Cube."""
    assert engine.cf_var is not None

    _ = _add_or_capture(
        build_func=partial(_build_ancil_var, cf_av_var),
        add_method=partial(_add_ancil_var, engine, cf_av_var),
        cf_var=cf_av_var,
        destination=LoadProblems.Problem.Destination(
            iris_class=Cube,
            identifier=engine.cf_var.cf_name,
        ),
    )


################################################################################
def _is_lat_lon(cf_var, ud_units, std_name, std_name_grid, axis_name, prefixes):
    """Determine whether the CF coordinate variable is a latitude/longitude variable.

    Ref:

    * [CF] Section 4.1 Latitude Coordinate.
    * [CF] Section 4.2 Longitude Coordinate.

    """
    is_valid = False
    attr_units = getattr(cf_var, CF_ATTR_UNITS, None)

    if isinstance(attr_units, str):
        attr_units = attr_units.lower()
        is_valid = attr_units in ud_units

        # Special case - Check for rotated pole.
        if attr_units == "degrees":
            attr_std_name = getattr(cf_var, CF_ATTR_STD_NAME, None)
            if attr_std_name is not None:
                is_valid = attr_std_name.lower() == std_name_grid
            else:
                is_valid = False
                # TODO: check that this interpretation of axis is correct.
                attr_axis = getattr(cf_var, CF_ATTR_AXIS, None)
                if attr_axis is not None:
                    is_valid = attr_axis.lower() == axis_name
    else:
        # Alternative is to check standard_name or axis.
        attr_std_name = getattr(cf_var, CF_ATTR_STD_NAME, None)

        if attr_std_name is not None:
            attr_std_name = attr_std_name.lower()
            is_valid = attr_std_name in [std_name, std_name_grid]
            if not is_valid:
                is_valid = any(
                    [attr_std_name.startswith(prefix) for prefix in prefixes]
                )
        else:
            attr_axis = getattr(cf_var, CF_ATTR_AXIS, None)

            if attr_axis is not None:
                is_valid = attr_axis.lower() == axis_name

    return is_valid


################################################################################
def is_latitude(engine, cf_name):
    """Determine whether the CF coordinate variable is a latitude variable."""
    cf_var = engine.cf_var.cf_group[cf_name]
    return _is_lat_lon(
        cf_var,
        UD_UNITS_LAT,
        CF_VALUE_STD_NAME_LAT,
        CF_VALUE_STD_NAME_GRID_LAT,
        CF_VALUE_AXIS_Y,
        ["lat", "rlat"],
    )


################################################################################
def is_longitude(engine, cf_name):
    """Determine whether the CF coordinate variable is a longitude variable."""
    cf_var = engine.cf_var.cf_group[cf_name]
    return _is_lat_lon(
        cf_var,
        UD_UNITS_LON,
        CF_VALUE_STD_NAME_LON,
        CF_VALUE_STD_NAME_GRID_LON,
        CF_VALUE_AXIS_X,
        ["lon", "rlon"],
    )


################################################################################
def is_projection_x_coordinate(engine, cf_name):
    """Determine whether the CF coordinate variable is a projection_x_coordinate variable."""
    cf_var = engine.cf_var.cf_group[cf_name]
    attr_name = getattr(cf_var, CF_ATTR_STD_NAME, None) or getattr(
        cf_var, CF_ATTR_LONG_NAME, None
    )
    return attr_name == CF_VALUE_STD_NAME_PROJ_X


################################################################################
def is_projection_y_coordinate(engine, cf_name):
    """Determine whether the CF coordinate variable is a projection_y_coordinate variable."""
    cf_var = engine.cf_var.cf_group[cf_name]
    attr_name = getattr(cf_var, CF_ATTR_STD_NAME, None) or getattr(
        cf_var, CF_ATTR_LONG_NAME, None
    )
    return attr_name == CF_VALUE_STD_NAME_PROJ_Y


################################################################################
def is_time(engine, cf_name):
    """Determine whether the CF coordinate variable is a time variable.

    Ref: [CF] Section 4.4 Time Coordinate.

    """
    cf_var = engine.cf_var.cf_group[cf_name]
    attr_units = getattr(cf_var, CF_ATTR_UNITS, None)

    attr_std_name = getattr(cf_var, CF_ATTR_STD_NAME, None)
    attr_axis = getattr(cf_var, CF_ATTR_AXIS, "")
    try:
        is_time_reference = cf_units.Unit(attr_units or 1).is_time_reference()
    except ValueError:
        is_time_reference = False

    return is_time_reference and (
        attr_std_name == "time" or attr_axis.lower() == CF_VALUE_AXIS_T
    )


################################################################################
def is_time_period(engine, cf_name):
    """Determine whether the CF coordinate variable represents a time period."""
    is_valid = False
    cf_var = engine.cf_var.cf_group[cf_name]
    attr_units = getattr(cf_var, CF_ATTR_UNITS, None)

    if attr_units is not None:
        try:
            is_valid = cf_units.is_time(attr_units)
        except ValueError:
            is_valid = False

    return is_valid


################################################################################
def is_grid_mapping(engine, cf_name, grid_mapping):
    """Determine whether the CF grid mapping variable is of the appropriate type."""
    is_valid = False
    cf_var = engine.cf_var.cf_group[cf_name]
    attr_mapping_name = getattr(cf_var, CF_ATTR_GRID_MAPPING_NAME, None)

    if attr_mapping_name is not None:
        is_valid = attr_mapping_name.lower() == grid_mapping

    return is_valid


################################################################################
def _parse_extended_grid_mapping(grid_mapping: str) -> dict[None | str, str]:
    """Parse `grid_mapping` attribute and return list of coordinate system variables and associated coords."""
    # Handles extended grid_mapping too. Possibilities:
    #  grid_mapping = "crs"  : simple mapping; a single variable name with no coords
    #  grid_mapping = "crs: lat lon"  : extended mapping; a variable name and list of coords
    #  grid_mapping = "crs: lat lon other: var1 var2"  : multiple extended mappings

    mappings: dict[None | str, str]

    # try simple mapping first
    if _GRID_MAPPING_PARSE_SIMPLE.match(grid_mapping):
        mappings = {None: grid_mapping}  # simple single grid mapping variable
    else:
        # Try extended mapping:
        # 1. Run validators to check for invalid expressions:
        for v_re, v_msg in _GRID_MAPPING_VALIDATORS:
            if len(match := v_re.findall(grid_mapping)):
                msg = f"Invalid syntax in extended grid_mapping: {grid_mapping!r}\n{v_msg} : {match}"
                raise iris.exceptions.CFParseError(msg)

        # 2. Parse grid_mapping into list of [cs, (coords, ...)]:
        result = _GRID_MAPPING_PARSE_EXTENDED.findall(grid_mapping)
        if len(result) == 0:
            msg = f"Failed to parse grid_mapping: {grid_mapping!r}"
            raise iris.exceptions.CFParseError(msg)

        # split second match group into list of coordinates:
        mappings = {}
        for cs, coords in result:
            mappings.update({coord: cs for coord in coords.split()})

    return mappings


################################################################################
def _is_rotated(engine, cf_name, cf_attr_value):
    """Determine whether the CF coordinate variable is rotated."""
    is_valid = False
    cf_var = engine.cf_var.cf_group[cf_name]
    attr_std_name = getattr(cf_var, CF_ATTR_STD_NAME, None)

    if attr_std_name is not None:
        is_valid = attr_std_name.lower() == cf_attr_value
    else:
        attr_units = getattr(cf_var, CF_ATTR_UNITS, None)
        if attr_units is not None:
            is_valid = attr_units.lower() == "degrees"

    return is_valid


################################################################################
def is_rotated_latitude(engine, cf_name):
    """Determine whether the CF coordinate variable is rotated latitude."""
    return _is_rotated(engine, cf_name, CF_VALUE_STD_NAME_GRID_LAT)


###############################################################################
def is_rotated_longitude(engine, cf_name):
    """Determine whether the CF coordinate variable is rotated longitude."""
    return _is_rotated(engine, cf_name, CF_VALUE_STD_NAME_GRID_LON)


################################################################################
def has_supported_mercator_parameters(engine, cf_name):
    """Determine whether the CF grid mapping variable has the supported values.

    Determine whether the CF grid mapping variable has the supported
    values for the parameters of the Mercator projection.
    """
    is_valid = True
    cf_grid_var = engine.cf_var.cf_group[cf_name]

    standard_parallel = getattr(cf_grid_var, CF_ATTR_GRID_STANDARD_PARALLEL, None)
    scale_factor_at_projection_origin = getattr(
        cf_grid_var, CF_ATTR_GRID_SCALE_FACTOR_AT_PROJ_ORIGIN, None
    )

    if scale_factor_at_projection_origin is not None and standard_parallel is not None:
        warnings.warn(
            "It does not make sense to provide both "
            '"scale_factor_at_projection_origin" and "standard_parallel".',
            category=iris.warnings.IrisCfInvalidCoordParamWarning,
        )
        is_valid = False

    return is_valid


################################################################################
def has_supported_polar_stereographic_parameters(engine, cf_name):
    """Determine whether CF grid mapping variable supports Polar Stereographic.

    Determine whether the CF grid mapping variable has the supported
    values for the parameters of the Polar Stereographic projection.

    """
    is_valid = True
    cf_grid_var = engine.cf_var.cf_group[cf_name]

    latitude_of_projection_origin = getattr(
        cf_grid_var, CF_ATTR_GRID_LAT_OF_PROJ_ORIGIN, None
    )

    standard_parallel = getattr(cf_grid_var, CF_ATTR_GRID_STANDARD_PARALLEL, None)
    scale_factor_at_projection_origin = getattr(
        cf_grid_var, CF_ATTR_GRID_SCALE_FACTOR_AT_PROJ_ORIGIN, None
    )

    if latitude_of_projection_origin != 90 and latitude_of_projection_origin != -90:
        warnings.warn(
            '"latitude_of_projection_origin" must be +90 or -90.',
            category=iris.warnings.IrisCfInvalidCoordParamWarning,
        )
        is_valid = False

    if scale_factor_at_projection_origin is not None and standard_parallel is not None:
        warnings.warn(
            "It does not make sense to provide both "
            '"scale_factor_at_projection_origin" and "standard_parallel".',
            category=iris.warnings.IrisCfInvalidCoordParamWarning,
        )
        is_valid = False

    if scale_factor_at_projection_origin is None and standard_parallel is None:
        warnings.warn(
            'One of "scale_factor_at_projection_origin" and '
            '"standard_parallel" is required.',
            category=iris.warnings.IrisCfInvalidCoordParamWarning,
        )
        is_valid = False

    return is_valid
