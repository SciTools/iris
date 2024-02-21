# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Exceptions specific to the Iris package."""


class IrisError(Exception):
    """Base class for errors in the Iris package."""

    pass


class CoordinateCollapseError(IrisError):
    """Raised when a requested coordinate cannot be collapsed."""

    pass


class CoordinateNotFoundError(KeyError):
    """Raised when a search yields no coordinates."""

    pass


class CellMeasureNotFoundError(KeyError):
    """Raised when a search yields no cell measures."""

    pass


class AncillaryVariableNotFoundError(KeyError):
    """Raised when a search yields no ancillary variables."""

    pass


class ConnectivityNotFoundError(KeyError):
    """Raised when a search yields no connectivities."""

    pass


class CoordinateMultiDimError(ValueError):
    """Raised when a routine doesn't support multi-dimensional coordinates."""

    def __init__(self, msg):
        # N.B. deferred import to avoid a circular import dependency.
        import iris.coords

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
    """Raised when a constraint operation has failed to find the correct number of results."""

    pass


class NotYetImplementedError(IrisError):
    """Raised by missing functionality.

    Different meaning to NotImplementedError, which is for abstract methods.

    """

    pass


class TranslationError(IrisError):
    """Raised when Iris is unable to translate format-specific codes."""

    pass


class IgnoreCubeException(IrisError):
    """Raised from a callback function when a cube should be ignored on load."""

    pass


class ConcatenateError(IrisError):
    """Raised when concatenate is expected to produce a single cube, but fails to do so."""

    def __init__(self, differences):
        """Create a ConcatenateError with a list of textual descriptions of differences.

        Create a ConcatenateError with a list of textual descriptions of
        the differences which prevented a concatenate.

        Parameters
        ----------
        differences : list of str
            The list of strings which describe the differences.

        """
        self.differences = differences

    def __str__(self):
        return "\n  ".join(
            ["failed to concatenate into a single cube."] + list(self.differences)
        )


class MergeError(IrisError):
    """Raised when merge is expected to produce a single cube, but fails to do so."""

    def __init__(self, differences):
        """Create a MergeError with a list of textual descriptions of the differences.

        Creates a MergeError with a list of textual descriptions of
        the differences which prevented a merge.

        Parameters
        ----------
        differences : list of str
            The list of strings which describe the differences.

        """
        self.differences = differences

    def __str__(self):
        return "\n  ".join(
            ["failed to merge into a single cube."] + list(self.differences)
        )


class DuplicateDataError(MergeError):
    """Raised when merging two or more cubes that have identical metadata."""

    def __init__(self, msg):
        self.differences = [msg]


class LazyAggregatorError(Exception):
    pass


class UnitConversionError(IrisError):
    """Raised when Iris is unable to convert a unit."""

    pass


class CannotAddError(ValueError):
    """Raised when an object (e.g. coord) cannot be added to a :class:`~iris.cube.Cube`."""

    pass
