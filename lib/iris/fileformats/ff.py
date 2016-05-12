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

import warnings

from iris._deprecation import warn_deprecated

# Issue a deprecation message when the module is loaded.
warn_deprecated("The module 'iris.fileformats.ff' is deprecated. "
                "Please use iris.fileformats.um as a replacement, which "
                "contains equivalents for all important features.")

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
    REAL_POLE_LON,
    Grid,
    ArakawaC,
    NewDynamics,
    ENDGame,
    FFHeader,
    FF2PP,
    load_cubes,
    load_cubes_32bit_ieee
)

# Ensure we reproduce documentation as it appeared in v1.9,
# but with a somewhat improved order of appearance.
__all__ = (
    'load_cubes',
    'load_cubes_32bit_ieee',
    'FF2PP',
    'Grid',
    'ArakawaC',
    'NewDynamics',
    'ENDGame',
    'FFHeader',
)
