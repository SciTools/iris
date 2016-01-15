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
Exceptions specific to the Iris package.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

import iris.coords


class IrisError(Exception):
    """Base class for errors in the Iris package."""
    pass


class CoordinateCollapseError(IrisError):
    """Raised when a requested coordinate cannot be collapsed."""
    pass


class CoordinateNotFoundError(KeyError):
    """Raised when a search yields no coordinates."""
    pass


class CoordinateMultiDimError(ValueError):
    """Raised when a routine doesn't support multi-dimensional coordinates."""
    def __init__(self, msg):
        if isinstance(msg, iris.coords.Coord):
            fmt = "Multi-dimensional coordinate not supported: '%s'"
            msg = fmt % msg.name()
        ValueError.__init__(self, msg)


class CoordinateNotRegularError(ValueError):
    """Raised when a coordinate is unexpectedly irregular."""
    pass


class InvalidCubeError(IrisError):
    """Raised when a Cube validation check fails."""
    pass


class ConstraintMismatchError(IrisError):
    """
    Raised when a constraint operation has failed to find the correct number
    of results.

    """
    pass


class NotYetImplementedError(IrisError):
    """
    Raised by missing functionality.

    Different meaning to NotImplementedError, which is for abstract methods.

    """
    pass


class TranslationError(IrisError):
    """Raised when Iris is unable to translate format-specific codes."""
    pass


class IgnoreCubeException(IrisError):
    """
    Raised from a callback function when a cube should be ignored on load.

    """
    pass


class ConcatenateError(IrisError):
    """
    Raised when concatenate is expected to produce a single cube, but fails to
    do so.

    """
    def __init__(self, differences):
        """
        Creates a ConcatenateError with a list of textual descriptions of
        the differences which prevented a concatenate.

        Args:

        * differences:
            The list of strings which describe the differences.

        """
        self.differences = differences

    def __str__(self):
        return '\n  '.join(['failed to concatenate into a single cube.'] +
                           list(self.differences))


class MergeError(IrisError):
    """
    Raised when merge is expected to produce a single cube, but fails to
    do so.

    """
    def __init__(self, differences):
        """
        Creates a MergeError with a list of textual descriptions of
        the differences which prevented a merge.

        Args:

        * differences:
            The list of strings which describe the differences.

        """
        self.differences = differences

    def __str__(self):
        return '\n  '.join(['failed to merge into a single cube.'] +
                           list(self.differences))


class DuplicateDataError(MergeError):
    """Raised when merging two or more cubes that have identical metadata."""
    def __init__(self, msg):
        self.differences = [msg]


class LazyAggregatorError(Exception):
    pass


class FieldLoadFault(Warning):
    """
    Warning raised when a fault-tolerant loader encounters a fault with
    a field it is attempting to load.

    """
    pass
