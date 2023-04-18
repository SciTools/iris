# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Provides NAME file format loading capabilities."""


def _get_NAME_loader(filename):
    """
    Return the appropriate load function for a NAME file based
    on the contents of its header.

    """
    # Lazy import to avoid importing name_loaders until
    # attempting to load a NAME file.
    import iris.fileformats.name_loaders as name_loaders

    load = None
    with open(filename, "r") as file_handle:
        header = name_loaders.read_header(file_handle)

    # Infer file type based on contents of header.
    if "Run name" in header and "Output format" not in header:
        if "X grid origin" not in header:
            load = name_loaders.load_NAMEIII_trajectory
        elif header.get("X grid origin") is not None:
            load = name_loaders.load_NAMEIII_field
        else:
            load = name_loaders.load_NAMEIII_timeseries

    elif "Output format" in header:
        load = name_loaders.load_NAMEIII_version2

    elif "Title" in header:
        if "Number of series" in header:
            load = name_loaders.load_NAMEII_timeseries
        else:
            load = name_loaders.load_NAMEII_field

    if load is None:
        raise ValueError(
            "Unable to determine NAME file type " "of {!r}.".format(filename)
        )

    return load


def load_cubes(filenames, callback):
    """
    Return a generator of cubes given one or more filenames and an
    optional callback.

    Args:

    * filenames (string/list):
        One or more NAME filenames to load.

    Kwargs:

    * callback (callable function):
        A function which can be passed on to :func:`iris.io.run_callback`.

    Returns:
         A generator of :class:`iris.cubes.Cube` instances.

    """
    from iris.io import run_callback

    if isinstance(filenames, str):
        filenames = [filenames]

    for filename in filenames:
        load = _get_NAME_loader(filename)
        for cube in load(filename):
            if callback is not None:
                cube = run_callback(callback, cube, None, filename)
            if cube is not None:
                yield cube
