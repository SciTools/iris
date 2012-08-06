# (C) British Crown Copyright 2010 - 2012, Met Office
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
            msg = "Multi-dimensional coordinate not supported: '%s'" % msg.name()
        ValueError.__init__(self, msg)


class CoordinateNotRegularError(ValueError):
    """Raised when a coordinate is unexpectedly irregular."""
    pass


class DuplicateDataError(IrisError):
    """Raised when merging two or more cubes that have identical metadata."""
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
    """Raised from a callback function when a cube should be ignored on load."""
    pass
    
