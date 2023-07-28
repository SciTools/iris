# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Handling of standard names and standard name aliases.
"""

import warnings

import iris.std_names


def get_convention():
    """Return the 'Conventions' string of the CF standard name table."""
    try:
        convention = iris.std_names.CONVENTIONS_STRING
    except AttributeError:
        convention = None
    return convention


def set_alias_processing(mode):
    """
    Set how standard name aliases are handled.

    Arg:

    * mode `string` specifying handling:
            'accept' - aliases are handled as any other standard name,
            'warn' - as above, but a warning is issued,
            'replace' - aliased standard names are replaced with the current one.
    """
    if not hasattr(iris.std_names, "ALIASES"):
        raise ValueError("The standard name table has no aliases defined.")
    if mode == "default":
        iris.std_names._MODE = iris.std_names._DEFAULT
    elif mode in iris.std_names._ALTERNATIVE_MODES:
        iris.std_names._MODE = mode
    else:
        raise ValueError(
            "{!r} is not a valid alternative for processing "
            "of standard name aliases.".format(mode)
        )


def get_description(name):
    """
    Return the standard name description as a `string`.

    Arg:

    * name `string` containing the standard name.
    """
    if not hasattr(iris.std_names, "DESCRIPTIONS"):
        return None

    error = False
    if name in iris.std_names.STD_NAMES:
        descr = iris.std_names.DESCRIPTIONS[name]
    elif hasattr(iris.std_names, "ALIASES"):
        if name in iris.std_names.ALIASES:
            descr = iris.std_names.DESCRIPTIONS[iris.std_names.ALIASES[name]]
            if iris.std_names._MODE == iris.std_names._REPLACE:
                msg = (
                    "\nStandard name {!r} is aliased and is \nreplaced by {!r}.\n"
                    "The description for the latter will be used."
                )
                warnings.warn(msg.format(name, iris.std_names.ALIASES[name]))
        else:
            error = True
    else:
        error = True

    if error:
        raise ValueError("{!r} is not a valid standard name.".format(name))
    return descr


def check_valid_std_name(name):
    """
    Check and return if argument is a valid standard name or alias.

    Arg:

    * name `string` containing the prospective standard name.

    Depending on the setting of the alias proceessing the following will
    happen if 'name' is an aliased standard name:
    "accept" - the aliased standard name is accepted as valid and returned,
    "warn" - a warning is issued, otherwise the same as "accept",
    "replace" - the valid standard name is returned without warning.

    When 'name' is neither a standard name nor an alias an error results.
    """
    error = False
    if name in iris.std_names.STD_NAMES:
        std_name = name
    elif hasattr(iris.std_names, "ALIASES"):
        if name in iris.std_names.ALIASES:
            if iris.std_names._MODE == iris.std_names._REPLACE:
                std_name = iris.std_names.ALIASES[name]
            else:
                std_name = name
                if iris.std_names._MODE == iris.std_names._WARN:
                    msg = "\nThe standard name {!r} is aliased should be \nreplaced by {!r}."
                    warnings.warn(
                        msg.format(name, iris.std_names.ALIASES[name])
                    )
        else:
            error = True
    else:
        error = True

    if error:
        raise ValueError("{!r} is not a valid standard_name.".format(name))
    return std_name
