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
Interpolation and re-gridding routines.

See also: :mod:`NumPy <numpy>`, and :ref:`SciPy <scipy:modindex>`.

.. deprecated:: 1.10

    The module :mod:`iris.analysis.interpolate` is deprecated.
    Please use :meth:`iris.cube.regrid` or :meth:`iris.cube.interpolate` with
    the appropriate regridding and interpolation schemes from
    :mod:`iris.analysis` instead.

The actual content of this module is all taken from
'iris.analysis._interpolate_backdoor'.
The only difference is that this module also emits a deprecation warning when
it is imported.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

from iris.analysis._interpolate_backdoor import *
from iris.analysis._interpolate_backdoor import _warn_deprecated

# Issue a deprecation message when the module is loaded.
_warn_deprecated()
