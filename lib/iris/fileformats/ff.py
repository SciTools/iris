# (C) British Crown Copyright 2010 - 2016, Met Office
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
Provides UK Met Office Fields File (FF) format specific capabilities.

.. deprecated:: 1.10

    This module has now been *deprecated*.
    Please use :mod:`iris.fileformats.um` instead :
    That contains equivalents for the key features of this module.

    The following replacements may be used.

    * for :class:`FF2PP`, use :meth:`iris.fileformats.um.um_to_pp`
    * for :meth:`load_cubes`, use :meth:`iris.fileformats.um.load_cubes`
    * for :meth:`load_cubes_32bit_ieee`, use
      :meth:`iris.fileformats.um.load_cubes_32bit_ieee`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

from functools import wraps
import warnings

from iris.fileformats import _ff
from iris.fileformats import pp
from iris._deprecation_helpers import ClassWrapperSameDocstring


_FF_DEPRECATION_WARNING = "The module 'iris.fileformats.ff' is deprecated."


# Define a standard mechanism for deprecation messages.
def _warn_deprecated(message=None):
    if message is None:
        message = _FF_DEPRECATION_WARNING
    warnings.warn(message)

# Issue a deprecation message when the module is loaded.
_warn_deprecated()

# Directly import various simple data items from the 'old' ff module.
from iris.fileformats._ff import (
    IMDI,
    FF_HEADER_DEPTH,
    DEFAULT_FF_WORD_DEPTH,
    _FF_LOOKUP_TABLE_TERMINATE,
    UM_FIXED_LENGTH_HEADER,
    UM_TO_FF_HEADER_OFFSET,
    FF_HEADER,
    _FF_HEADER_POINTERS,
    _LBUSER_DTYPE_LOOKUP,
    X_COORD_U_GRID,
    Y_COORD_V_GRID,
    HANDLED_GRIDS,
    REAL_EW_SPACING,
    REAL_NS_SPACING,
    REAL_FIRST_LAT,
    REAL_FIRST_LON,
    REAL_POLE_LAT,
    REAL_POLE_LON
)


# Define wrappers to all public classes, that emit deprecation warnings.
# Note: it seems we are obliged to provide an __init__ with a suitable matching
# signature for each one, as Sphinx will take its constructor signature from
# any overriding definition of __init__ or __new__.
# So without this, the docs don't look right.

class Grid(six.with_metaclass(ClassWrapperSameDocstring, _ff.Grid)):
    @wraps(_ff.Grid.__init__)
    def __init__(self, column_dependent_constants, row_dependent_constants,
                 real_constants, horiz_grid_type):
        _warn_deprecated()
        super(Grid, self).__init__(
            column_dependent_constants, row_dependent_constants,
            real_constants, horiz_grid_type)


class ArakawaC(six.with_metaclass(ClassWrapperSameDocstring, _ff.ArakawaC)):
    @wraps(_ff.ArakawaC.__init__)
    def __init__(self, column_dependent_constants, row_dependent_constants,
                 real_constants, horiz_grid_type):
        _warn_deprecated()
        super(ArakawaC, self).__init__(
            column_dependent_constants, row_dependent_constants,
            real_constants, horiz_grid_type)


class NewDynamics(six.with_metaclass(ClassWrapperSameDocstring,
                                     _ff.NewDynamics)):
    @wraps(_ff.NewDynamics.__init__)
    def __init__(self, column_dependent_constants, row_dependent_constants,
                 real_constants, horiz_grid_type):
        _warn_deprecated()
        super(NewDynamics, self).__init__(
            column_dependent_constants, row_dependent_constants,
            real_constants, horiz_grid_type)


class ENDGame(six.with_metaclass(ClassWrapperSameDocstring, _ff.ENDGame)):
    @wraps(_ff.ENDGame.__init__)
    def __init__(self, column_dependent_constants, row_dependent_constants,
                 real_constants, horiz_grid_type):
        _warn_deprecated()
        super(ENDGame, self).__init__(
            column_dependent_constants, row_dependent_constants,
            real_constants, horiz_grid_type)


class FFHeader(six.with_metaclass(ClassWrapperSameDocstring, _ff.FFHeader)):
    @wraps(_ff.FFHeader.__init__)
    def __init__(self, filename, word_depth=DEFAULT_FF_WORD_DEPTH):
        _warn_deprecated()
        super(FFHeader, self).__init__(filename, word_depth=word_depth)


class FF2PP(six.with_metaclass(ClassWrapperSameDocstring, _ff.FF2PP)):
    @wraps(_ff.FF2PP.__init__)
    def __init__(self, filename, read_data=False,
                 word_depth=DEFAULT_FF_WORD_DEPTH):
        # Provide an enhanced deprecation message for this one.
        msg = (_FF_DEPRECATION_WARNING + '\n' +
               "Please use 'iris.fileformats.um.um_to_pp' in place of "
               "'iris.fileformats.ff.FF2PP.")
        _warn_deprecated(msg)
        super(FF2PP, self).__init__(filename, read_data=read_data,
                                    word_depth=word_depth)


# Provide alternative loader functions which issue a deprecation warning,
# using the original dosctrings with an appended deprecation warning.

def load_cubes(filenames, callback, constraints=None):
    _warn_deprecated("The module 'iris.fileformats.ff' is deprecated. "
                     "\nPlease use 'iris.fileformat.um.load_cubes' "
                     "in place of 'iris.fileformats.ff.load_cubes'.")
    return pp._load_cubes_variable_loader(filenames, callback, FF2PP,
                                          constraints=constraints)

load_cubes.__doc__ = _ff.load_cubes.__doc__ + """
    .. deprecated:: 1.10
        Please use :meth:`iris.fileformats.um.load_cubes` as a replacement.

"""


def load_cubes_32bit_ieee(filenames, callback, constraints=None):
    _warn_deprecated("The module 'iris.fileformats.ff' is deprecated. "
                     "\nPlease use 'iris.fileformat.um.load_cubes_32bit_ieee' "
                     "in place of 'iris.fileformats.ff.load_cubes_32bit_ieee'"
                     ".")
    return pp._load_cubes_variable_loader(filenames, callback, FF2PP,
                                          {'word_depth': 4},
                                          constraints=constraints)

load_cubes_32bit_ieee.__doc__ = _ff.load_cubes_32bit_ieee.__doc__ + """
    .. deprecated:: 1.10
        Please use :meth:`iris.fileformats.um.load_cubes_32bit_ieee` as a
        replacement.

"""
