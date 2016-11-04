# (C) British Crown Copyright 2016, Met Office
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
Support for fast matrix loading of structured UM files.

This works with either PP or Fieldsfiles.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

import iris
from iris.cube import CubeList


def _basic_load_function(filename, pp_filter=None, **kwargs):
    # The low-level 'fields from filename' loader.
    # Referred to in generic rules processing as the 'generator' function.
    #
    # Called by generic rules code.
    # replaces pp.load (and the like)
    #
    # It yields a sequence of "fields".
    #
    # In this case our 'fields' are :
    # iris.fileformats.um._fast_load_structured_fields.FieldCollation
    #
    # Also in our case, we need to apply the basic single-field filtering
    # operation that speeds up phenomenon selection.
    # Therefore, the actual loader will pass us this as a keyword, if it is
    # needed.
    # The remaining keywords are 'passed on' to the lower-level function.
    #
    # NOTE: so, this is what is passed as the 'field' to user callbacks.
    from iris.experimental.fieldsfile import _structured_loader
    from iris.fileformats.um._fast_load_structured_fields import \
        group_structured_fields
    loader = _structured_loader(filename)
    fields = iter(field
                  for field in loader(filename, **kwargs)
                  if pp_filter is None or pp_filter(field))
    return group_structured_fields(fields)


def _convert(collation):
    # The call recorded in the 'loader' structure of the the generic rules
    # code (iris.fileformats.rules), that converts a 'field' into a 'raw cube'.
    from iris.experimental.fieldsfile import _convert_collation
    return _convert_collation(collation)


def _combine_structured_cubes(cubes):
    # Combine structured cubes from different sourcefiles, in the style of
    # merge/concatenate.
    #
    # Because standard Cube.merge employed in loading can't do this.
    return iter(CubeList(cubes).concatenate())


def _fast_load_common(iris_call, uris, constraints, callback,
                      do_raw_load=False):
    import iris.fileformats.pp as pp
    try:
        old_structured_flag = pp._DO_STRUCTURED_LOAD
        old_raw_flag = pp._STRUCTURED_LOAD_IS_RAW
        pp._DO_STRUCTURED_LOAD = True
        pp._STRUCTURED_LOAD_IS_RAW = do_raw_load
        result = iris_call(uris, constraints, callback)
    finally:
        pp._PP_LOAD_VIA_STRUCTURED = old_structured_flag
        pp._STRUCTURED_LOAD_IS_RAW = old_raw_flag
    return result


def fast_load(uris, constraints=None, callback=None):
    """
    Load cubes from structured UM Fieldsfile and PP files.

    This function is an alternative to :meth:`iris.load` that can be used on
    'structured' UM files, providing much faster load times.

    Within a given input file, each phenomenon must occur as a set of fields
    repeating regularly over a set of dimensions, in which order within the
    file is signficant.  Additionally, all combinations of...
    In addition, all dimension combinations must be present...
    The function will normally return a single cube for each phenomenon
    contained :  Within each source file, the fields of each different phenomon
    must be arranged in a regular repeating structure.

    Otherwise ...

    Args:

    * filename:
        One or more filenames, with optional wildcard characters.
        These can be UM (Fieldsfile-like) or PP files, the content in either
        case being UM 'fields'.

    Kwargs:

    * constraints:
        One or more constraints.
    * callback:
        A modifier/filter function.

    Returns:
        An :class:`iris.cube.CubeList`.

    """
    return _fast_load_common(iris.load, uris, constraints, callback)


def fast_load_cube(uris, constraints=None, callback=None):
    """
    Fast-load data, yielding a single result cube.

    If not all data can be combined into a single cube, an exception is raised.

    """
    return _fast_load_common(iris.load_cube, uris, constraints, callback)


def fast_load_cubes(uris, constraints=None, callback=None):
    """
    Fast-load data, yielding a single result cube for each constraint.

    If any constraint does not yield a single cube, an exception is raised.

    """
    return _fast_load_common(iris.load_cubes, uris, constraints, callback)


def fast_load_raw(uris, constraints=None, callback=None):
    """
    Fast-load data, without merging cubes from different files.

    The result of loading each file is returned as one or more cubes.

    """
    return _fast_load_common(iris.load_raw, uris, constraints, callback,
                             do_raw_load=True)
