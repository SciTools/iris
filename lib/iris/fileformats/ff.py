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

from iris.fileformats import _ff
from iris.fileformats import pp

_DEPRECATION_WARNSTRING = "The module 'iris.fileformats.ff' is deprecated."


# Define a standard mechanism for deprecation messages.
def _warn_deprecated(message=None):
    if message is None:
        message = _DEPRECATION_WARNSTRING
    warnings.warn(message)

# Issue a deprecation message when the module is loaded, if enabled.
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


# Make new wrappers for classes and functions, with the original public names,
# but which emit deprecation warnings when used.
class _DeprecationWrapperMetaclass(type):
    def __new__(metacls, classname, bases, class_dict):
        # Patch the subclass to duplicate docstrings from the parent class, and
        # provide an __init__ that issues a deprecation warning and then calls
        # the parent constructor.
        parent_class = bases[0]
        # Copy the original class docstring.
        class_dict['__doc__'] = parent_class.__doc__

        # Copy the init docstring too.
        init_docstring = parent_class.__init__.__doc__

        # Define a warning message.
        depr_warnstring = _DEPRECATION_WARNSTRING
        # Use a special class variable for an augmented warning message.
        alternative_note = class_dict.get('_DEPRECATION_ALTERNATIVE_NOTE')
        if alternative_note:
            depr_warnstring = '{}\n{}\n'.format(
                depr_warnstring, alternative_note)

        # Save the parent class *on* the wrapper class, so we can chain to its
        #  __init__ call.
        class_dict['_target_parent_class'] = parent_class

        # Create a wrapper init function which issues the deprecation.
        def initfn(self, *args, **kwargs):
            _warn_deprecated(depr_warnstring)
            self._target_parent_class.__init__(self, *args, **kwargs)

        # Set this as the init for the wrapper class.
        initfn.func_name = '__init__'
        initfn.__doc__ = init_docstring
        class_dict['__init__'] = initfn

        # Return the result.
        return super(_DeprecationWrapperMetaclass, metacls).__new__(
            metacls, classname, bases, class_dict)


class Grid(six.with_metaclass(_DeprecationWrapperMetaclass, _ff.Grid)):
    pass


class ArakawaC(six.with_metaclass(_DeprecationWrapperMetaclass, _ff.ArakawaC)):
    pass


class NewDynamics(six.with_metaclass(_DeprecationWrapperMetaclass,
                                     _ff.NewDynamics)):
    pass


class ENDGame(six.with_metaclass(_DeprecationWrapperMetaclass, _ff.ENDGame)):
    pass


class FFHeader(six.with_metaclass(_DeprecationWrapperMetaclass, _ff.FFHeader)):
    pass


class FF2PP(six.with_metaclass(_DeprecationWrapperMetaclass, _ff.FF2PP)):
    _DEPRECATION_ALTERNATIVE_NOTE = (
        "Please use 'iris.fileformats.um.um_to_pp' in place of "
        "'iris.fileformats.ff.FF2PP.")


# Make wrappers to the loader functions which also issue a warning,
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
