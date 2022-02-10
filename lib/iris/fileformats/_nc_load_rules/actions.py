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
E.G. 'action_provides_grid_mapping' replaces all 'fc_provides_grid_mapping_<X>'.
To aid debug, each returns a 'rule_name' string, indicating which original rule
this particular action call is emulating :  In some cases, this may include a
textual note that this rule 'failed', aka "did not trigger", which would not be
recorded in the original implementation.

TODO: remove the use of intermediate "facts" to carry information between
actions.  This mimics older behaviour, so is still useful while we are still
comparing behaviour with the old Pyke rules (debugging).  But once that is no
longer useful, this can be considerably simplified.

"""

from functools import wraps
import warnings

from iris.config import get_logger
import iris.fileformats.cf
import iris.fileformats.pp as pp

from . import helpers as hh

# Configure the logger.
logger = get_logger(__name__, fmt="[%(funcName)s]")


def _default_rulenamesfunc(func_name):
    # A simple default function to deduce the rules-name from an action-name.
    funcname_prefix = "action_"
    rulename_prefix = "fc_"  # To match existing behaviours
    rule_name = func_name
    if rule_name.startswith(funcname_prefix):
        rule_name = rule_name[len(funcname_prefix) :]
    if not rule_name.startswith(rulename_prefix):
        rule_name = rulename_prefix + rule_name
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
    """Standard operations for every cube."""
    hh.build_cube_metadata(engine)


# Lookup table used by 'action_provides_grid_mapping'.
# Maps each supported CF grid-mapping-name to a pair of handling ("helper")
# routines:
#  (@0) a validity-checker (or None)
#  (@1) a coord-system builder function.
_GRIDTYPE_CHECKER_AND_BUILDER = {
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
    """Convert a CFGridMappingVariable into a cube coord-system."""
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
        if grid_mapping_type in _GRIDTYPE_CHECKER_AND_BUILDER:
            checker, builder = _GRIDTYPE_CHECKER_AND_BUILDER[grid_mapping_type]
            rule_name += f"_({grid_mapping_type})"
        else:
            succeed = False
            rule_name += f" --FAILED(unhandled type {grid_mapping_type})"

    if succeed:
        if checker is not None and not checker(engine, var_name):
            succeed = False
            rule_name += f" --(FAILED check {checker.__name__})"

    if succeed:
        coordinate_system = builder(engine, cf_var)
        engine.cube_parts["coordinate_system"] = coordinate_system

        # Check there is not an existing one.
        # ATM this is guaranteed by the caller, "run_actions".
        assert engine.fact_list("grid-type") == []

        engine.add_fact("grid-type", (grid_mapping_type,))

    return rule_name


@action_function
def action_provides_coordinate(engine, dimcoord_fact):
    """Identify the coordinate 'type' of a CFCoordinateVariable."""
    (var_name,) = dimcoord_fact

    # Identify the "type" of a coordinate variable
    coord_type = None
    # NOTE: must test for rotated cases *first*, as 'is_longitude' and
    # 'is_latitude' functions also accept rotated cases.
    if hh.is_rotated_latitude(engine, var_name):
        coord_type = "rotated_latitude"
    elif hh.is_rotated_longitude(engine, var_name):
        coord_type = "rotated_longitude"
    elif hh.is_latitude(engine, var_name):
        coord_type = "latitude"
    elif hh.is_longitude(engine, var_name):
        coord_type = "longitude"
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
        # Rationalise this for now by making it more like the others.
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
#  (@0) A grid "type", one of latlon/rotated/projected (or None)
#       If set, the cube should have a coord-system, which is set on the
#       resulting coordinate.  If None, the coord has no coord_system.
#  (@1) an (optional) fixed standard-name for the coordinate, or None
#       If None, the coordinate name is copied from the source variable
_COORDTYPE_GRIDTYPES_AND_COORDNAMES = {
    "latitude": ("latlon", hh.CF_VALUE_STD_NAME_LAT),
    "longitude": ("latlon", hh.CF_VALUE_STD_NAME_LON),
    "rotated_latitude": (
        "rotated",
        hh.CF_VALUE_STD_NAME_GRID_LAT,
    ),
    "rotated_longitude": (
        "rotated",
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
    """Convert a CFCoordinateVariable into a cube dim-coord."""
    coord_type, var_name = providescoord_fact
    cf_var = engine.cf_var.cf_group[var_name]
    rule_name = f"fc_build_coordinate_({coord_type})"
    coord_grid_class, coord_name = _COORDTYPE_GRIDTYPES_AND_COORDNAMES[
        coord_type
    ]
    if coord_grid_class is None:
        # Coordinates not identified with a specific grid-type class (latlon,
        # rotated or projected) are always built, but can have no coord-system.
        coord_system = None  # no coord-system can be used
        succeed = True
    else:
        grid_classes = ("latlon", "rotated", "projected")
        assert coord_grid_class in grid_classes
        # If a coord is of a type identified with a grid, we may have a
        # coordinate system (i.e. a valid grid-mapping).
        # N.B. this requires each grid-type identification to validate the
        # coord var (e.g. "is_longitude").
        # Non-conforming lon/lat/projection coords will be classed as
        # dim-coords by cf.py, but 'action_provides_coordinate' will give them
        # a coord-type of 'miscellaneous' : hence, they have no coord-system.
        coord_system = engine.cube_parts.get("coordinate_system")
        # Translate the specific grid-mapping type to a grid-class
        if coord_system is None:
            succeed = True
            cs_gridclass = None
        else:
            # Get a grid-class from the grid-type
            # i.e. one of latlon/rotated/projected, as for coord_grid_class.
            gridtypes_factlist = engine.fact_list("grid-type")
            (gridtypes_fact,) = gridtypes_factlist  # only 1 fact
            (cs_gridtype,) = gridtypes_fact  # fact contains 1 term
            if cs_gridtype == "latitude_longitude":
                cs_gridclass = "latlon"
            elif cs_gridtype == "rotated_latitude_longitude":
                cs_gridclass = "rotated"
            else:
                # Other specific projections
                assert cs_gridtype is not None
                cs_gridclass = "projected"

        assert cs_gridclass in grid_classes + (None,)

        if coord_grid_class == "latlon":
            if cs_gridclass == "latlon":
                succeed = True
            elif cs_gridclass is None:
                succeed = True
                rule_name += "(no-cs)"
            elif cs_gridclass == "rotated":
                # We disallow this case
                succeed = False
                rule_name += "(FAILED : latlon coord with rotated cs)"
            else:
                assert cs_gridclass == "projected"
                # succeed, no error, but discards the coord-system
                # TODO: could issue a warning in this case ?
                succeed = True
                coord_system = None
                rule_name += "(no-cs : discarded projected cs)"
        elif coord_grid_class == "rotated":
            if cs_gridclass == "rotated":
                succeed = True
                rule_name += "(rotated)"
            elif cs_gridclass is None:
                succeed = True
                rule_name += "(rotated no-cs)"
            elif cs_gridclass == "latlon":
                # We disallow this case
                succeed = False
                rule_name += "(FAILED rotated coord with latlon cs)"
            else:
                assert cs_gridclass == "projected"
                succeed = True
                coord_system = None
                rule_name += "(rotated no-cs : discarded projected cs)"
        elif coord_grid_class == "projected":
            # In this case, can *only* build a coord at all if there is a
            # coord-system of the correct class (i.e. 'projected').
            succeed = cs_gridclass == "projected"
            if not succeed:
                rule_name += "(FAILED projected coord with non-projected cs)"
        else:
            # Just FYI : literally not possible, as we already asserted this.
            assert coord_grid_class in grid_classes

    if succeed:
        hh.build_dimension_coordinate(
            engine, cf_var, coord_name=coord_name, coord_system=coord_system
        )
    return rule_name


@action_function
def action_build_auxiliary_coordinate(engine, auxcoord_fact):
    """Convert a CFAuxiliaryCoordinateVariable into a cube aux-coord."""
    (var_name,) = auxcoord_fact
    rule_name = "fc_build_auxiliary_coordinate"

    # Identify any known coord "type" : latitude/longitude/time/time_period
    # If latitude/longitude, this sets the standard_name of the built AuxCoord
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


@action_function
def action_ukmo_stash(engine):
    """Convert 'ukmo stash' cf property into a cube attribute."""
    rule_name = "fc_attribute_ukmo__um_stash_source"
    var = engine.cf_var
    attr_name = "ukmo__um_stash_source"
    attr_value = getattr(var, attr_name, None)
    if attr_value is None:
        attr_name = "um_stash_source"  # legacy form
        attr_value = getattr(var, attr_name, None)
    if attr_value is None:
        rule_name += "(NOT-TRIGGERED)"
    else:
        # No helper routine : just do it
        try:
            stash_code = pp.STASH.from_msi(attr_value)
        except (TypeError, ValueError):
            engine.cube.attributes[attr_name] = attr_value
            msg = (
                "Unable to set attribute STASH as not a valid MSI "
                f'string "mXXsXXiXXX", got "{attr_value}"'
            )
            logger.debug(msg)
        else:
            engine.cube.attributes["STASH"] = stash_code

    return rule_name


@action_function
def action_ukmo_processflags(engine):
    """Convert 'ukmo process flags' cf property into a cube attribute."""
    rule_name = "fc_attribute_ukmo__process_flags"
    var = engine.cf_var
    attr_name = "ukmo__process_flags"
    attr_value = getattr(var, attr_name, None)
    if attr_value is None:
        rule_name += "(NOT-TRIGGERED)"
    else:
        # No helper routine : just do it
        flags = [x.replace("_", " ") for x in attr_value.split(" ")]
        engine.cube.attributes["ukmo__process_flags"] = tuple(flags)

    return rule_name


@action_function
def action_build_cell_measure(engine, cellm_fact):
    """Convert a CFCellMeasureVariable into a cube cell-measure."""
    (var_name,) = cellm_fact
    var = engine.cf_var.cf_group.cell_measures[var_name]
    hh.build_cell_measures(engine, var)


@action_function
def action_build_ancil_var(engine, ancil_fact):
    """Convert a CFAncillaryVariable into a cube ancil-var."""
    (var_name,) = ancil_fact
    var = engine.cf_var.cf_group.ancillary_variables[var_name]
    hh.build_ancil_var(engine, var)


@action_function
def action_build_label_coordinate(engine, label_fact):
    """Convert a CFLabelVariable into a cube string-type aux-coord."""
    (var_name,) = label_fact
    var = engine.cf_var.cf_group.labels[var_name]
    hh.build_auxiliary_coordinate(engine, var)


@action_function
def action_formula_type(engine, formula_root_fact):
    """Register a CFVariable as a formula root."""
    rule_name = "fc_formula_type"
    (var_name,) = formula_root_fact
    cf_var = engine.cf_var.cf_group[var_name]
    # cf_var.standard_name is a formula type (or we should never get here).
    formula_type = getattr(cf_var, "standard_name", None)
    succeed = True
    if formula_type not in iris.fileformats.cf.reference_terms:
        succeed = False
        rule_name += f"(FAILED - unrecognised formula type = {formula_type!r})"
        msg = f"Ignored formula of unrecognised type: {formula_type!r}."
        warnings.warn(msg)
    if succeed:
        # Check we don't already have one.
        existing_type = engine.requires.get("formula_type")
        if existing_type:
            # NOTE: in this case, for now, we will accept the last appearing,
            # which matches the older behaviour.
            # TODO: this needs resolving, somehow.
            succeed = False
            msg = (
                "Omitting factories for some hybrid coordinates, as multiple "
                "hybrid coordinates on a single variable are not supported: "
                f"Formula of type ={formula_type!r} "
                f"overrides another of type ={existing_type!r}.)"
            )
            warnings.warn(msg)
        rule_name += f"_{formula_type}"
        # Set 'requires' info for iris.fileformats.netcdf._load_aux_factory.
        engine.requires["formula_type"] = formula_type

    return rule_name


@action_function
def action_formula_term(engine, formula_term_fact):
    """Register a CFVariable as a formula term."""
    # Must run AFTER formula root identification.
    (termvar_name, rootvar_name, term_name) = formula_term_fact
    # The rootname is implicit :  have only one per cube
    # TODO: change when we adopt cf-1.7 advanced grid-mapping syntax
    engine.requires.setdefault("formula_terms", {})[term_name] = termvar_name
    rule_name = f"fc_formula_term({term_name})"
    return rule_name


def run_actions(engine):
    """
    Run all actions for a cube.

    This is the top-level "activation" function which runs all the appropriate
    rules actions to translate facts and build all the cube elements.

    The specific cube being translated is "engine.cube".

    """

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

    # build (dimension) coordinates
    # The 'provides' step and the grid-mapping must have already been done.
    providescoord_facts = engine.fact_list("provides-coordinate-(oftype)")
    for providescoord_fact in providescoord_facts:
        action_build_dimension_coordinate(engine, providescoord_fact)

    # build aux-coords
    auxcoord_facts = engine.fact_list("auxiliary_coordinate")
    for auxcoord_fact in auxcoord_facts:
        action_build_auxiliary_coordinate(engine, auxcoord_fact)

    # Detect + process and special 'ukmo' attributes
    # Run on every cube : they choose themselves whether to trigger.
    action_ukmo_stash(engine)
    action_ukmo_processflags(engine)

    # cell measures
    cellm_facts = engine.fact_list("cell_measure")
    for cellm_fact in cellm_facts:
        action_build_cell_measure(engine, cellm_fact)

    # ancillary variables
    ancil_facts = engine.fact_list("ancillary_variable")
    for ancil_fact in ancil_facts:
        action_build_ancil_var(engine, ancil_fact)

    # label coords
    label_facts = engine.fact_list("label")
    for label_fact in label_facts:
        action_build_label_coordinate(engine, label_fact)

    # formula root variables
    formula_root_facts = engine.fact_list("formula_root")
    for root_fact in formula_root_facts:
        action_formula_type(engine, root_fact)

    # formula terms
    # The 'formula_root's must have already been done.
    formula_term_facts = engine.fact_list("formula_term")
    for term_fact in formula_term_facts:
        action_formula_term(engine, term_fact)
