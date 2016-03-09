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
This extension module provides access to the underlying libmo_unpack library
functionality.

.. deprecated:: 1.10
    :mod:`iris.fileformats.pp_packing` is deprecated.
    Please install mo_pack (https://github.com/SciTools/mo_pack) instead.
    This provides additional pack/unpacking functionality.

"""
from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

import warnings

from iris.fileformats import _old_pp_packing as old_pp_packing


_DEPRECATION_DOCSTRING_SUFFIX = """
.. deprecated:: 1.10
    :mod:`iris.fileformats.pp_packing` is deprecated.
    Please install mo_pack (https://github.com/SciTools/mo_pack) instead.
    This provides additional pack/unpacking functionality.

"""

_DEPRECATION_WARNING = (
    'Module "iris.fileformats.pp_packing" is deprecated.  '
    'Please install mo_pack (https://github.com/SciTools/mo_pack) instead.  '
    'This provides additional pack/unpacking functionality.')


# Emit a deprecation warning when anyone tries to import this.
# For quiet, can still use _old_pp_packing instead, as fileformats.pp does.
warnings.warn(_DEPRECATION_WARNING)


# Define simple wrappers for functions in pp_packing.
# N.B. signatures must match the originals !
def wgdos_unpack(data, lbrow, lbnpt, bmdi):
    warnings.warn(_DEPRECATION_WARNING)
    return old_pp_packing.wgdos_unpack(data, lbrow, lbnpt, bmdi)


def rle_decode(data, lbrow, lbnpt, bmdi):
    warnings.warn(_DEPRECATION_WARNING)
    return old_pp_packing.rle_decode(data, lbrow, lbnpt, bmdi)


def _add_fixed_up_docstring(new_fn, original_fn):
    # Add docstring to a wrapper function, based on the original function.
    # This would be simpler if Sphinx were less fussy about formatting.
    docstring = original_fn.__doc__
    lines = [line for line in docstring.split('\n')]
    # Strip off last blank lines, and add deprecation notice.
    while len(lines[-1].strip()) == 0:
        lines = lines[:-1]
    docstring = '\n'.join(lines)
    docstring += _DEPRECATION_DOCSTRING_SUFFIX
    new_fn.__doc__ = docstring


_add_fixed_up_docstring(wgdos_unpack, old_pp_packing.wgdos_unpack)
_add_fixed_up_docstring(rle_decode, old_pp_packing.rle_decode)
