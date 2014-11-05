# (C) British Crown Copyright 2013 - 2014, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""Provides NAME file format loading capabilities."""

from __future__ import (absolute_import, division, print_function)

import iris.io


def _get_NAME_loader(filename):
    """
    Return the approriate load function for a NAME file based
    on the contents of its header.

    """
    # Lazy import to avoid importing name_loaders until
    # attempting to load a NAME file.
    import iris.fileformats.name_loaders as name_loaders

    load = None
    with open(filename, 'r') as file_handle:
        header = name_loaders.read_header(file_handle)

    # Infer file type based on contents of header.
    if 'Run name' in header:
        if 'X grid origin' not in header:
            load = name_loaders.load_NAMEIII_trajectory
        elif header.get('X grid origin') is not None:
            load = name_loaders.load_NAMEIII_field
        else:
            load = name_loaders.load_NAMEIII_timeseries
    elif 'Title' in header:
        if 'Number of series' in header:
            load = name_loaders.load_NAMEII_timeseries
        else:
            load = name_loaders.load_NAMEII_field

    if load is None:
        raise ValueError('Unable to determine NAME file type '
                         'of {!r}.'.format(filename))

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
    if isinstance(filenames, basestring):
        filenames = [filenames]

    for filename in filenames:
        load = _get_NAME_loader(filename)
        for cube in load(filename):
            if callback is not None:
                cube = iris.io.run_callback(callback, cube,
                                            None, filename)
            if cube is not None:
                yield cube
