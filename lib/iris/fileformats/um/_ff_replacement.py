# (C) British Crown Copyright 2014 - 2017, Met Office
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
"""
Support for UM "fieldsfile-like" files.

At present, the only UM file types supported are true FieldsFiles and LBCs.
Other types of UM file may fail to load correctly (or at all).

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

from iris.fileformats._ff import FF2PP
from iris.fileformats.pp import _load_cubes_variable_loader


def um_to_pp(filename, read_data=False, word_depth=None):
    """
    Extract individual PPFields from within a UM Fieldsfile-like file.

    Returns an iterator over the fields contained within the FieldsFile,
    returned as :class:`iris.fileformats.pp.PPField` instances.

    Args:

    * filename (string):
        Specify the name of the FieldsFile.

    Kwargs:

    * read_data (boolean):
        Specify whether to read the associated PPField data within
        the FieldsFile.  Default value is False.

    Returns:
        Iteration of :class:`iris.fileformats.pp.PPField`\s.

    For example::

        >>> for field in um.um_to_pp(filename):
        ...     print(field)

    """
    if word_depth is None:
        ff2pp = FF2PP(filename, read_data=read_data)
    else:
        ff2pp = FF2PP(filename, read_data=read_data,
                      word_depth=word_depth)

    # Note: unlike the original wrapped case, we will return an actual
    # iterator, rather than an object that can provide an iterator.
    return iter(ff2pp)


def load_cubes(filenames, callback, constraints=None,
               _loader_kwargs=None):
    """
    Loads cubes from filenames of UM fieldsfile-like files.

    Args:

    * filenames - list of filenames to load

    Kwargs:

    * callback - a function which can be passed on to
        :func:`iris.io.run_callback`

    .. note::

        The resultant cubes may not be in the order that they are in the
        file (order is not preserved when there is a field with
        orography references).

    """
    return _load_cubes_variable_loader(
        filenames, callback, FF2PP, constraints=constraints,
        loading_function_kwargs=_loader_kwargs)


def load_cubes_32bit_ieee(filenames, callback, constraints=None):
    """
    Loads cubes from filenames of 32bit ieee converted UM fieldsfile-like
    files.

    .. seealso::

        :func:`load_cubes` for keyword details

    """
    return load_cubes(filenames, callback, constraints=constraints,
                      _loader_kwargs={'word_depth': 4})
