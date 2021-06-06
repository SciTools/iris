# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Replacement code for the Pyke rules.

For now, we are still emulating various aspects of how our original Pyke-based
code used the Pyke 'engine' to hold translation data, both Pyke-specific and
not :
1) basic details from the iris.fileformats.cf analysis of the file are
   recorded before translating each output cube, using
   "engine.assert_case_specific_fact(name, args)".

2) this is also used to store intermediate info passed between rules, which
   used to be done with a "facts_cf.provides" statement in rule actions.

3) Iris-specific info is (still) stored in additional properties created on
   the engine object :
       engine.cf_var, .cube, .cube_parts, .requires, .rule_triggered, .filename

Our "rules" are just action routines.
The top-level 'run_actions' routine decides which actions to call, based on the
info recorded when processing each cube output.  It does this in a simple
explicit way, which doesn't use any clever chaining, "trigger conditions" or
other rule-type logic.

Each 'action' function can replace several similar 'rules'.
E.G. 'action_provides_grid_mapping' replaces all 'fc_provides_grid+mapping_<X>'.
To aid debug, each returns a 'rule_name' string, indicating which original rule
this particular action call is emulating :  In some cases, this may include a
textual note that this rule 'failed', aka "did not trigger", which would not be
recorded in the original implementation.

The top-level 'run_actions' ensures that the individual rules actions are
called, with various arguments, as appropriate to ensure the whole cube is
built as it was by the original rules implementation.

TODO: remove the use of intermediate "facts" to carry information between
actions.  This mimics older behaviour, so is still useful while we are still
comparing behaviour with the old Pyke rules (debugging).  But once that is no
longer useful, this can be considerably simplified.

"""

from . import helpers as hh
from functools import wraps


def convert_actionname_to_rulename(func_name):
    # Given the name of an action-func, return the name of the rule.
    funcname_prefix = "action_"
    rulename_prefix = "fc_"  # To match existing behaviours
    rule_name = func_name
    if rule_name.startswith(funcname_prefix):
        rule_name = rule_name[len(funcname_prefix) :]
    if not rule_name.startswith(rulename_prefix):
        rule_name = rulename_prefix + rule_name
    return rule_name


def _default_rulenamesfunc(func_name):
    # A simple default function to deduce the rules-name from an action-name.
    rule_name = convert_actionname_to_rulename(func_name)
    return rule_name


def action_function(func):
    # Wrap an action function with some standard behaviour.
    # Notably : engages with the rules logging process.
    @wraps(func)
    def inner(engine, *args, **kwargs):
        # Call the original rules-func
        rule_name = func(engine, *args, **kwargs)
        if rule_name is None:
            # Work out the corresponding rule name, and log it.
            # Note: an action returns a name string, which identifies it,
            # but also may vary depending on whether it successfully
            # triggered, and if so what it matched.
            rule_name = _default_rulenamesfunc(func.__name__)
        engine.rule_triggered.add(rule_name)

    func._rulenames_func = _default_rulenamesfunc
    return inner


@action_function
def action_default(engine):
    hh.build_cube_metadata(engine)


# Lookup table used by 'action_provides_grid_mapping'.
# Maps each supported CF grid-mapping-name to a pair of handling ("helper")
# routines:
#  (@0) a validity-checker (or None)
#  (@1) a coord-system builder function.
_grid_types_to_checker_builder = {
    hh.CF_GRID_MAPPING_LAT_LON: (None, hh.build_coordinate_system),
    hh.CF_GRID_MAPPING_ROTATED_LAT_LON: (
        None,
        hh.build_rotated_coordinate_system,
    ),
    hh.CF_GRID_MAPPING_MERCATOR: (
        hh.has_supported_mercator_parameters,
        hh.build_mercator_coordinate_system,
    ),
    hh.CF_GRID_MAPPING_TRANSVERSE: (
        None,
        hh.build_transverse_mercator_coordinate_system,
    ),
    hh.CF_GRID_MAPPING_STEREO: (
        hh.has_supported_stereographic_parameters,
        hh.build_stereographic_coordinate_system,
    ),
    hh.CF_GRID_MAPPING_LAMBERT_CONFORMAL: (
        None,
        hh.build_lambert_conformal_coordinate_system,
    ),
    hh.CF_GRID_MAPPING_LAMBERT_AZIMUTHAL: (
        None,
        hh.build_lambert_azimuthal_equal_area_coordinate_system,
    ),
    hh.CF_GRID_MAPPING_ALBERS: (
        None,
        hh.build_albers_equal_area_coordinate_system,
    ),
    hh.CF_GRID_MAPPING_VERTICAL: (
        None,
        hh.build_vertical_perspective_coordinate_system,
    ),
    hh.CF_GRID_MAPPING_GEOSTATIONARY: (
        None,
        hh.build_geostationary_coordinate_system,
    ),
}


@action_function
def action_provides_grid_mapping(engine, gridmapping_fact):
    (var_name,) = gridmapping_fact
    rule_name = "fc_provides_grid_mapping"
    cf_var = engine.cf_var.cf_group[var_name]
    grid_mapping_type = getattr(cf_var, hh.CF_ATTR_GRID_MAPPING_NAME, None)
    succeed = True
    if grid_mapping_type is None:
        succeed = False
        rule_name += " --FAILED(no grid-mapping attr)"
    else:
        grid_mapping_type = grid_mapping_type.lower()
    if succeed:
        if grid_mapping_type in _grid_types_to_checker_builder:
            checker, builder = _grid_types_to_checker_builder[
                grid_mapping_type
            ]
            rule_name += f"_({grid_mapping_type})"
        else:
            succeed = False
            rule_name += f" --FAILED(unhandled type {grid_mapping_type})"
    # We DON'T call this, as we already identified the type in the call.
    # if succeed and not is_grid_mapping(engine, var_name, grid_mapping_type):
    #     succeed = False
    #     rule_name += f' --(FAILED is_grid_mapping)'
    if succeed:
        if checker is not None and not checker(engine, var_name):
            succeed = False
            rule_name += f" --(FAILED check {checker.__name__})"

    if succeed:
        coordinate_system = builder(engine, cf_var)
        # Check there is not an existing one.
        old_gridtype_fact = engine.fact_list("grid-type")
        if old_gridtype_fact:
            (old_gridtype,) = old_gridtype_fact
            succeed = False
            rule_name += (
                f" --(FAILED overwrite coord-sytem "
                f"{old_gridtype} with {grid_mapping_type})"
            )
    if succeed:
        engine.cube_parts["coordinate_system"] = coordinate_system
        engine.add_fact("grid-type", (grid_mapping_type,))

    return rule_name


@action_function
def action_provides_coordinate(engine, dimcoord_fact):
    (var_name,) = dimcoord_fact

    # Identify the coord type
    # N.B. *only* to "name" the rule, for debug : no functional need.
    coord_type = None
    if hh.is_latitude(engine, var_name):
        coord_type = "latitude"
    elif hh.is_longitude(engine, var_name):
        coord_type = "longitude"
    elif hh.is_rotated_latitude(engine, var_name):
        coord_type = "rotated_latitude"
    elif hh.is_rotated_longitude(engine, var_name):
        coord_type = "rotated_longitude"
    elif hh.is_time(engine, var_name):
        coord_type = "time"
    elif hh.is_time_period(engine, var_name):
        coord_type = "time_period"
    elif hh.is_projection_x_coordinate(engine, var_name):
        coord_type = "projection_x"
    elif hh.is_projection_y_coordinate(engine, var_name):
        coord_type = "projection_y"

    if coord_type is None:
        # Not identified as a specific known coord_type.
        # N.B. in the original rules, this does *not* trigger separate
        # 'provides' and 'build' phases : there is just a single
        # 'fc_default_coordinate' rule.
        # Rationalise this for now by making it like the others.
        # FOR NOW: ~matching old code, but they could *all* be simplified.
        # TODO: combine 2 operation into 1 for ALL of these.
        coord_type = "miscellaneous"
        rule_name = "fc_default_coordinate_(provide-phase)"
    else:
        rule_name = f"fc_provides_coordinate_({coord_type})"

    engine.add_fact("provides-coordinate-(oftype)", (coord_type, var_name))
    return rule_name


# Lookup table used by 'action_build_dimension_coordinate'.
# Maps each supported coordinate-type name (a rules-internal concept) to a pair
# of information values :
#  (@0) the CF grid_mapping_name (or None)
#       If set, the cube should have a coord-system, which is set on the
#       resulting coordinate.  If None, the coord has no coord_system.
#  (@1) an (optional) fixed standard-name for the coordinate, or None
#       If None, the coordinate name is copied from the source variable
_coordtype_to_gridtype_coordname = {
    "latitude": ("latitude_longitude", hh.CF_VALUE_STD_NAME_LAT),
    "longitude": ("latitude_longitude", hh.CF_VALUE_STD_NAME_LON),
    "rotated_latitude": (
        "rotated_latitude_longitude",
        hh.CF_VALUE_STD_NAME_GRID_LAT,
    ),
    "rotated_longitude": (
        "rotated_latitude_longitude",
        hh.CF_VALUE_STD_NAME_GRID_LON,
    ),
    "projection_x": ("projected", hh.CF_VALUE_STD_NAME_PROJ_X),
    "projection_y": ("projected", hh.CF_VALUE_STD_NAME_PROJ_Y),
    "time": (None, None),
    "time_period": (None, None),
    "miscellaneous": (None, None),
}


@action_function
def action_build_dimension_coordinate(engine, providescoord_fact):
    coord_type, var_name = providescoord_fact
    cf_var = engine.cf_var.cf_group[var_name]
    rule_name = f"fc_build_coordinate_({coord_type})"
    grid_type, coord_name = _coordtype_to_gridtype_coordname[coord_type]
    coord_system = None
    if grid_type is not None:
        # If a type is identified with a grid, use the coordinate system
        # N.B. this requires each grid-type identification to validate the
        # coord var (e.g. "is_longitude").
        # Non-conforming lon/lat/projection coords will be classed as
        # dim-coords by cf.py, but 'action_provides_coordinate' will give them
        # a coord-type of 'miscellaneous' : hence, they have no coord-system.
        coord_system = engine.cube_parts.get("coordinate_system")
    hh.build_dimension_coordinate(
        engine, cf_var, coord_name=coord_name, coord_system=coord_system
    )

    return rule_name


@action_function
def action_build_auxiliary_coordinate(engine, auxcoord_fact):
    (var_name,) = auxcoord_fact
    rule_name = "fc_build_auxiliary_coordinate"

    # FOR NOW: attempt to identify type
    # TODO: eventually remove much of this, which only affects rule_name.
    # (but could possibly retain for future debugging purposes)
    coord_type = ""  # unidentified : can be OK
    coord_name = None
    if hh.is_time(engine, var_name):
        coord_type = "time"
    elif hh.is_time_period(engine, var_name):
        coord_type = "time_period"
    elif hh.is_longitude(engine, var_name):
        coord_type = "longitude"
        if hh.is_rotated_longitude(engine, var_name):
            coord_type += "_rotated"
            coord_name = hh.CF_VALUE_STD_NAME_GRID_LON
        else:
            coord_name = hh.CF_VALUE_STD_NAME_LON
    elif hh.is_latitude(engine, var_name):
        coord_type = "latitude"
        if hh.is_rotated_latitude(engine, var_name):
            coord_type += "_rotated"
            coord_name = hh.CF_VALUE_STD_NAME_GRID_LAT
        else:
            coord_name = hh.CF_VALUE_STD_NAME_LAT

    if coord_type:
        rule_name += f"_{coord_type}"

    cf_var = engine.cf_var.cf_group.auxiliary_coordinates[var_name]
    hh.build_auxiliary_coordinate(engine, cf_var, coord_name=coord_name)

    return rule_name


def run_actions(engine):
    # default (all cubes) action, always runs
    action_default(engine)  # This should run the default rules.

    # deal with grid-mappings
    grid_mapping_facts = engine.fact_list("grid_mapping")
    # For now, there should be at most *one* of these.
    assert len(grid_mapping_facts) in (0, 1)
    for grid_mapping_fact in grid_mapping_facts:
        action_provides_grid_mapping(engine, grid_mapping_fact)

    # identify + record aka "PROVIDE" specific named coordinates
    # N.B. cf.py has identified that these are dim-coords, NOT aux-coords
    # (which are recorded separately).
    # TODO: can probably remove this step ??
    dimcoord_facts = engine.fact_list("coordinate")
    for dimcoord_fact in dimcoord_facts:
        action_provides_coordinate(engine, dimcoord_fact)

    # build coordinates
    providescoord_facts = engine.fact_list("provides-coordinate-(oftype)")
    for providescoord_fact in providescoord_facts:
        action_build_dimension_coordinate(engine, providescoord_fact)

    # build aux-coords
    auxcoord_facts = engine.fact_list("auxiliary_coordinate")
    for auxcoord_fact in auxcoord_facts:
        action_build_auxiliary_coordinate(engine, auxcoord_fact)
