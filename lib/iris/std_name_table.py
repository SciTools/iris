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

    Requesting the description of a aliased standard name results in a error
    if the alias proceessing is set to "replace", because the aliased standard
    name should already have been replaced.
    """
    if not hasattr(iris.std_names, "DESCRIPTIONS"):
        return None

    if name in iris.std_names.STD_NAMES:
        descr = iris.std_names.DESCRIPTIONS[name]
        action = iris.std_names._ALTERNATIVE_MODES[0]
    elif hasattr(iris.std_names, "ALIASES"):
        if name in iris.std_names.ALIASES:
            descr = iris.std_names.DESCRIPTIONS[iris.std_names.ALIASES[name]]
            action = iris.std_names._MODE
        else:
            action = iris.std_names._ALTERNATIVE_MODES[2]
    else:
        action = iris.std_names._ALTERNATIVE_MODES[2]

    if action == iris.std_names._ALTERNATIVE_MODES[1]:
        msg = (
            "\nStandard name {!r} is aliased and is \nreplaced by {!r}.\n"
            "The description for the latter will be used."
        )
        warnings.warn(msg.format(name, iris.std_names.ALIASES[name]))
    elif action == iris.std_names._ALTERNATIVE_MODES[2]:
        raise ValueError(
            "{!r} is not a valid standard name (or it may have been aliased).".format(
                name
            )
        )
    return descr


def check_valid_std_name(name):
    """
    Returning standard name as a `string`.

    Arg:

    * name `string` containing the prospective standard name.

    Depending on the setting of the alias proceessing the following will
    happen if 'name' is an aliased standard name:
    "accept" - the aliased standard name is accepted as valid and returned,
    "warn" - a warning is issued and the valid standard name is returned,
    "replace" - the valid standard name is returned without warning.

    When 'name' is neither a standard name nor an alias an error results.
    """
    if name in iris.std_names.STD_NAMES:
        std_name = name
        action = iris.std_names._ALTERNATIVE_MODES[0]
    elif hasattr(iris.std_names, "ALIASES"):
        if name in iris.std_names.ALIASES:
            if iris.std_names._MODE == iris.std_names._ALTERNATIVE_MODES[0]:
                std_name = name
                action = iris.std_names._ALTERNATIVE_MODES[0]
            else:
                std_name = iris.std_names.ALIASES[name]
                if (
                    iris.std_names._MODE
                    == iris.std_names._ALTERNATIVE_MODES[1]
                ):
                    action = iris.std_names._MODE
                else:
                    action = iris.std_names._ALTERNATIVE_MODES[0]
        else:
            action = iris.std_names._ALTERNATIVE_MODES[2]
    else:
        action = iris.std_names._ALTERNATIVE_MODES[2]

    if action == iris.std_names._ALTERNATIVE_MODES[2]:
        raise ValueError("{repr(name)} is not a valid standard_name.")
    elif action == iris.std_names._ALTERNATIVE_MODES[1]:
        msg = "\nThe standard name {!r} is aliased and is \nreplaced by {!r}."
        warnings.warn(msg.format(name, iris.std_names.ALIASES[name]))
    return std_name
