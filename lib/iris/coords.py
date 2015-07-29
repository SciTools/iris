# (C) British Crown Copyright 2010 - 2015, Met Office
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
Definitions of coordinates.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

from abc import ABCMeta, abstractproperty
import collections
import copy
from itertools import chain
from six.moves import zip_longest
import operator
import warnings
import zlib

import biggus
import netcdftime
import numpy as np

import iris.aux_factory
import iris.exceptions
import iris.time
import iris.unit
import iris.util

from iris._cube_coord_common import CFVariableMixin
from iris.util import is_regular


class CoordDefn(collections.namedtuple('CoordDefn',
                                       ['standard_name', 'long_name',
                                        'var_name', 'units',
                                        'attributes', 'coord_system'])):
    """
    Criterion for identifying a specific type of :class:`DimCoord` or
    :class:`AuxCoord` based on its metadata.

    """
    def name(self, default='unknown'):
        """
        Returns a human-readable name.

        First it tries self.standard_name, then it tries the 'long_name'
        attribute, then the 'var_name' attribute, before falling back to
        the value of `default` (which itself defaults to 'unknown').

        """
        return self.standard_name or self.long_name or self.var_name or default

    def __lt__(self, other):
        if not isinstance(other, CoordDefn):
            return NotImplemented

        def _sort_key(defn):
            # Emulate Python 2 behaviour with None
            return (defn.standard_name is not None, defn.standard_name,
                    defn.long_name is not None, defn.long_name,
                    defn.var_name is not None, defn.var_name,
                    defn.units is not None, defn.units,
                    defn.coord_system is not None, defn.coord_system)

        return _sort_key(self) < _sort_key(other)


class CoordExtent(collections.namedtuple('_CoordExtent', ['name_or_coord',
                                                          'minimum',
                                                          'maximum',
                                                          'min_inclusive',
                                                          'max_inclusive'])):
    """Defines a range of values for a coordinate."""

    def __new__(cls, name_or_coord, minimum, maximum,
                min_inclusive=True, max_inclusive=True):
        """
        Create a CoordExtent for the specified coordinate and range of
        values.

        Args:

        * name_or_coord
            Either a coordinate name or a coordinate, as defined in
            :meth:`iris.cube.Cube.coords()`.

        * minimum
            The minimum value of the range to select.

        * maximum
            The maximum value of the range to select.

        Kwargs:

        * min_inclusive
            If True, coordinate values equal to `minimum` will be included
            in the selection. Default is True.

        * max_inclusive
            If True, coordinate values equal to `maximum` will be included
            in the selection. Default is True.

        """
        return super(CoordExtent, cls).__new__(cls, name_or_coord, minimum,
                                               maximum, min_inclusive,
                                               max_inclusive)


# Coordinate cell styles. Used in plot and cartography.
POINT_MODE = 0
BOUND_MODE = 1

BOUND_POSITION_START = 0
BOUND_POSITION_MIDDLE = 0.5
BOUND_POSITION_END = 1


# Private named tuple class for coordinate groups.
_GroupbyItem = collections.namedtuple('GroupbyItem',
                                      'groupby_point, groupby_slice')


class Cell(collections.namedtuple('Cell', ['point', 'bound'])):
    """
    An immutable representation of a single cell of a coordinate, including the
    sample point and/or boundary position.

    Notes on cell comparison:

    Cells are compared in two ways, depending on whether they are
    compared to another Cell, or to a number/string.

    Cell-Cell comparison is defined to produce a strict ordering. If
    two cells are not exactly equal (i.e. including whether they both
    define bounds or not) then they will have a consistent relative
    order.

    Cell-number and Cell-string comparison is defined to support
    Constraint matching. The number/string will equal the Cell if, and
    only if, it is within the Cell (including on the boundary). The
    relative comparisons (lt, le, ..) are defined to be consistent with
    this interpretation. So for a given value `n` and Cell `cell`, only
    one of the following can be true:

    |    n < cell
    |    n == cell
    |    n > cell

    Similarly, `n <= cell` implies either `n < cell` or `n == cell`.
    And `n >= cell` implies either `n > cell` or `n == cell`.

    """

    # This subclass adds no attributes.
    __slots__ = ()

    def __new__(cls, point=None, bound=None):
        """
        Construct a Cell from point or point-and-bound information.

        """
        if point is None:
            raise ValueError('Point must be defined.')

        if bound is not None:
            bound = tuple(bound)

        if isinstance(point, np.ndarray):
            point = tuple(point.flatten())

        if isinstance(point, (tuple, list)):
            if len(point) != 1:
                raise ValueError('Point may only be a list or tuple if it has '
                                 'length 1.')
            point = point[0]

        return super(Cell, cls).__new__(cls, point, bound)

    def __mod__(self, mod):
        point = self.point
        bound = self.bound
        if point is not None:
            point = point % mod
        if bound is not None:
            bound = tuple([val % mod for val in bound])
        return Cell(point, bound)

    def __add__(self, mod):
        point = self.point
        bound = self.bound
        if point is not None:
            point = point + mod
        if bound is not None:
            bound = tuple([val + mod for val in bound])
        return Cell(point, bound)

    def __eq__(self, other):
        """
        Compares Cell equality depending on the type of the object to be
        compared.

        """
        if isinstance(other, (int, float, np.number)) or \
                hasattr(other, 'timetuple'):
            if self.bound is not None:
                return self.contains_point(other)
            else:
                return self.point == other
        elif isinstance(other, Cell):
            return (self.point == other.point) and (self.bound == other.bound)
        elif (isinstance(other, six.string_types) and self.bound is None and
              isinstance(self.point, six.string_types)):
            return self.point == other
        else:
            return NotImplemented

    # Must supply __ne__, Python does not defer to __eq__ for negative equality
    def __ne__(self, other):
        result = self.__eq__(other)
        if result is not NotImplemented:
            result = not result
        return result

    def __common_cmp__(self, other, operator_method):
        """
        Common method called by the rich comparison operators. The method of
        checking equality depends on the type of the object to be compared.

        Cell vs Cell comparison is used to define a strict order.
        Non-Cell vs Cell comparison is used to define Constraint matching.

        """
        if not (isinstance(other, (int, float, np.number, Cell)) or
                hasattr(other, 'timetuple')):
            raise TypeError("Unexpected type of other "
                            "{}.".format(type(other)))
        if operator_method not in (operator.gt, operator.lt,
                                   operator.ge, operator.le):
            raise ValueError("Unexpected operator_method")

        # Prevent silent errors resulting from missing netcdftime
        # behaviour.
        if (isinstance(other, netcdftime.datetime) or
                (isinstance(self.point, netcdftime.datetime) and
                 not isinstance(other, iris.time.PartialDateTime))):
            raise TypeError('Cannot determine the order of '
                            'netcdftime.datetime objects')

        if isinstance(other, Cell):
            # Cell vs Cell comparison for providing a strict sort order
            if self.bound is None:
                if other.bound is None:
                    # Point vs point
                    # - Simple ordering
                    result = operator_method(self.point, other.point)
                else:
                    # Point vs point-and-bound
                    # - Simple ordering of point values, but if the two
                    #   points are equal, we make the arbitrary choice
                    #   that the point-only Cell is defined as less than
                    #   the point-and-bound Cell.
                    if self.point == other.point:
                        result = operator_method in (operator.lt, operator.le)
                    else:
                        result = operator_method(self.point, other.point)
            else:
                if other.bound is None:
                    # Point-and-bound vs point
                    # - Simple ordering of point values, but if the two
                    #   points are equal, we make the arbitrary choice
                    #   that the point-only Cell is defined as less than
                    #   the point-and-bound Cell.
                    if self.point == other.point:
                        result = operator_method in (operator.gt, operator.ge)
                    else:
                        result = operator_method(self.point, other.point)
                else:
                    # Point-and-bound vs point-and-bound
                    # - Primarily ordered on minimum-bound. If the
                    #   minimum-bounds are equal, then ordered on
                    #   maximum-bound. If the maximum-bounds are also
                    #   equal, then ordered on point values.
                    if self.bound[0] == other.bound[0]:
                        if self.bound[1] == other.bound[1]:
                            result = operator_method(self.point, other.point)
                        else:
                            result = operator_method(self.bound[1],
                                                     other.bound[1])
                    else:
                        result = operator_method(self.bound[0], other.bound[0])
        else:
            # Cell vs number (or string, or datetime-like) for providing
            # Constraint behaviour.
            if self.bound is None:
                # Point vs number
                # - Simple matching
                me = self.point
            else:
                if hasattr(other, 'timetuple'):
                    raise TypeError('Cannot determine whether a point lies '
                                    'within a bounded region for '
                                    'datetime-like objects.')
                # Point-and-bound vs number
                # - Match if "within" the Cell
                if operator_method in [operator.gt, operator.le]:
                    me = min(self.bound)
                else:
                    me = max(self.bound)
            result = operator_method(me, other)

        return result

    def __ge__(self, other):
        return self.__common_cmp__(other, operator.ge)

    def __le__(self, other):
        return self.__common_cmp__(other, operator.le)

    def __gt__(self, other):
        return self.__common_cmp__(other, operator.gt)

    def __lt__(self, other):
        return self.__common_cmp__(other, operator.lt)

    def __str__(self):
        if self.bound is not None:
            return repr(self)
        else:
            return str(self.point)

    def contains_point(self, point):
        """
        For a bounded cell, returns whether the given point lies within the
        bounds.

        .. note:: The test carried out is equivalent to min(bound)
                  <= point <= max(bound).

        """
        if self.bound is None:
            raise ValueError('Point cannot exist inside an unbounded cell.')
        if hasattr(point, 'timetuple') or np.any([hasattr(val, 'timetuple') for
                                                  val in self.bound]):
            raise TypeError('Cannot determine whether a point lies within '
                            'a bounded region for datetime-like objects.')

        return np.min(self.bound) <= point <= np.max(self.bound)


class Coord(six.with_metaclass(ABCMeta, CFVariableMixin)):
    """
    Abstract superclass for coordinates.

    """

    _MODE_ADD = 1
    _MODE_SUB = 2
    _MODE_MUL = 3
    _MODE_DIV = 4
    _MODE_RDIV = 5
    _MODE_SYMBOL = {_MODE_ADD: '+', _MODE_SUB: '-',
                    _MODE_MUL: '*', _MODE_DIV: '/',
                    _MODE_RDIV: '/'}

    def __init__(self, points, standard_name=None, long_name=None,
                 var_name=None, units='1', bounds=None, attributes=None,
                 coord_system=None):

        """
        Constructs a single coordinate.

        Args:

        * points:
            The values (or value in the case of a scalar coordinate) of the
            coordinate for each cell.

        Kwargs:

        * standard_name:
            CF standard name of coordinate
        * long_name:
            Descriptive name of coordinate
        * var_name:
            CF variable name of coordinate
        * units
            The :class:`~iris.unit.Unit` of the coordinate's values.
            Can be a string, which will be converted to a Unit object.
        * bounds
            An array of values describing the bounds of each cell. Given n
            bounds for each cell, the shape of the bounds array should be
            points.shape + (n,). For example, a 1d coordinate with 100 points
            and two bounds per cell would have a bounds array of shape
            (100, 2)

        * attributes
            A dictionary containing other cf and user-defined attributes.
        * coord_system
            A :class:`~iris.coord_systems.CoordSystem`,
            e.g. a :class:`~iris.coord_systems.GeogCS` for a longitude Coord.

        """
        #: CF standard name of the quantity that the coordinate represents.
        self.standard_name = standard_name

        #: Descriptive name of the coordinate.
        self.long_name = long_name

        #: The CF variable name for the coordinate.
        self.var_name = var_name

        #: Unit of the quantity that the coordinate represents.
        self.units = units

        #: Other attributes, including user specified attributes that
        #: have no meaning to Iris.
        self.attributes = attributes

        #: Relevant CoordSystem (if any).
        self.coord_system = coord_system

        self.points = points
        self.bounds = bounds

    def __getitem__(self, key):
        """
        Returns a new Coord whose values are obtained by conventional array
        indexing.

        .. note::

            Indexing of a circular coordinate results in a non-circular
            coordinate if the overall shape of the coordinate changes after
            indexing.

        """
        # Turn the key(s) into a full slice spec - i.e. one entry for
        # each dimension of the coord.
        full_slice = iris.util._build_full_slice_given_keys(key, self.ndim)

        # If it's a "null" indexing operation (e.g. coord[:, :]) then
        # we can preserve deferred loading by avoiding promoting _points
        # and _bounds to full ndarray instances.
        def is_full_slice(s):
            return isinstance(s, slice) and s == slice(None, None)
        if all(is_full_slice(s) for s in full_slice):
            points = self._points
            bounds = self._bounds
        else:
            points = self._points
            if isinstance(points, iris.aux_factory._LazyArray):
                # This triggers the LazyArray to compute its values
                # (if it hasn't already), which will also trigger any
                # deferred loading of its dependencies.
                points = points.view()
            bounds = self._bounds
            if isinstance(bounds, iris.aux_factory._LazyArray):
                bounds = bounds.view()

            # Make indexing on the cube column based by using the
            # column_slices_generator (potentially requires slicing the
            # data multiple times).
            _, slice_gen = iris.util.column_slices_generator(full_slice,
                                                             self.ndim)
            for keys in slice_gen:
                if points is not None:
                    points = points[keys]
                    if points.shape and min(points.shape) == 0:
                        raise IndexError('Cannot index with zero length '
                                         'slice.')
                if bounds is not None:
                    bounds = bounds[keys + (Ellipsis, )]

        new_coord = self.copy(points=points, bounds=bounds)
        return new_coord

    def copy(self, points=None, bounds=None):
        """
        Returns a copy of this coordinate.

        Kwargs:

        * points: A points array for the new coordinate.
                  This may be a different shape to the points of the coordinate
                  being copied.

        * bounds: A bounds array for the new coordinate.
                  Given n bounds for each cell, the shape of the bounds array
                  should be points.shape + (n,). For example, a 1d coordinate
                  with 100 points and two bounds per cell would have a bounds
                  array of shape (100, 2).

        .. note:: If the points argument is specified and bounds are not, the
                  resulting coordinate will have no bounds.

        """

        if points is None and bounds is not None:
            raise ValueError('If bounds are specified, points must also be '
                             'specified')

        new_coord = copy.deepcopy(self)
        if points is not None:
            # Explicitly not using the points property as we don't want the
            # shape the new points to be constrained by the shape of
            # self.points
            new_coord._points = None
            new_coord.points = points
            # Regardless of whether bounds are provided as an argument, new
            # points will result in new bounds, discarding those copied from
            # self.
            new_coord.bounds = bounds

        return new_coord

    @abstractproperty
    def points(self):
        """Property containing the points values as a numpy array"""

    @abstractproperty
    def bounds(self):
        """Property containing the bound values as a numpy array"""

    def _repr_other_metadata(self):
        fmt = ''
        if self.long_name:
            fmt = ', long_name={self.long_name!r}'
        if self.var_name:
            fmt += ', var_name={self.var_name!r}'
        if len(self.attributes) > 0:
            fmt += ', attributes={self.attributes}'
        if self.coord_system:
            fmt += ', coord_system={self.coord_system}'
        result = fmt.format(self=self)
        return result

    def _str_dates(self, dates_as_numbers):
        date_obj_array = self.units.num2date(dates_as_numbers)
        kwargs = {'separator': ', ', 'prefix': '      '}
        try:
            # With NumPy 1.7 we need to ask for 'str' formatting.
            result = np.core.arrayprint.array2string(
                date_obj_array, formatter={'numpystr': str}, **kwargs)
        except TypeError:
            # But in 1.6 we don't need to ask, and the option doesn't
            # even exist!
            result = np.core.arrayprint.array2string(date_obj_array, **kwargs)
        return result

    def __str__(self):
        if self.units.is_time_reference():
            fmt = '{cls}({points}{bounds}' \
                  ', standard_name={self.standard_name!r}' \
                  ', calendar={self.units.calendar!r}{other_metadata})'
            points = self._str_dates(self.points)
            bounds = ''
            if self.bounds is not None:
                bounds = ', bounds=' + self._str_dates(self.bounds)
            result = fmt.format(self=self, cls=type(self).__name__,
                                points=points, bounds=bounds,
                                other_metadata=self._repr_other_metadata())
        else:
            result = repr(self)
        return result

    def __repr__(self):
        fmt = '{cls}({self.points!r}{bounds}' \
              ', standard_name={self.standard_name!r}, units={self.units!r}' \
              '{other_metadata})'
        bounds = ''
        if self.bounds is not None:
            bounds = ', bounds=' + repr(self.bounds)
        result = fmt.format(self=self, cls=type(self).__name__,
                            bounds=bounds,
                            other_metadata=self._repr_other_metadata())
        return result

    def __eq__(self, other):
        eq = NotImplemented
        # If the other object has a means of getting its definition, and
        # whether or not it has_points and has_bounds, then do the
        # comparison, otherwise return a NotImplemented to let Python try to
        # resolve the operator elsewhere.
        if hasattr(other, '_as_defn'):
            # metadata comparison
            eq = self._as_defn() == other._as_defn()
            # points comparison
            if eq:
                eq = iris.util.array_equal(self.points, other.points)
            # bounds comparison
            if eq:
                if self.bounds is not None and other.bounds is not None:
                    eq = iris.util.array_equal(self.bounds, other.bounds)
                else:
                    eq = self.bounds is None and other.bounds is None

        return eq

    # Must supply __ne__, Python does not defer to __eq__ for negative equality
    def __ne__(self, other):
        result = self.__eq__(other)
        if result is not NotImplemented:
            result = not result
        return result

    def _as_defn(self):
        defn = CoordDefn(self.standard_name, self.long_name, self.var_name,
                         self.units, self.attributes, self.coord_system)
        return defn

    def __binary_operator__(self, other, mode_constant):
        """
        Common code which is called by add, sub, mult and div

        Mode constant is one of ADD, SUB, MUL, DIV, RDIV

        .. note::

            The unit is *not* changed when doing scalar operations on a
            coordinate. This means that a coordinate which represents
            "10 meters" when multiplied by a scalar i.e. "1000" would result
            in a coordinate of "10000 meters". An alternative approach could
            be taken to multiply the *unit* by 1000 and the resultant
            coordinate would represent "10 kilometers".

        """
        if isinstance(other, Coord):
            raise iris.exceptions.NotYetImplementedError(
                'coord %s coord' % Coord._MODE_SYMBOL[mode_constant])

        elif isinstance(other, (int, float, np.number)):

            if mode_constant == Coord._MODE_ADD:
                points = self.points + other
            elif mode_constant == Coord._MODE_SUB:
                points = self.points - other
            elif mode_constant == Coord._MODE_MUL:
                points = self.points * other
            elif mode_constant == Coord._MODE_DIV:
                points = self.points / other
            elif mode_constant == Coord._MODE_RDIV:
                points = other / self.points

            if self.bounds is not None:
                if mode_constant == Coord._MODE_ADD:
                    bounds = self.bounds + other
                elif mode_constant == Coord._MODE_SUB:
                    bounds = self.bounds - other
                elif mode_constant == Coord._MODE_MUL:
                    bounds = self.bounds * other
                elif mode_constant == Coord._MODE_DIV:
                    bounds = self.bounds / other
                elif mode_constant == Coord._MODE_RDIV:
                    bounds = other / self.bounds
            else:
                bounds = None
            new_coord = self.copy(points, bounds)
            return new_coord
        else:
            return NotImplemented

    def __add__(self, other):
        return self.__binary_operator__(other, Coord._MODE_ADD)

    def __sub__(self, other):
        return self.__binary_operator__(other, Coord._MODE_SUB)

    def __mul__(self, other):
        return self.__binary_operator__(other, Coord._MODE_MUL)

    def __div__(self, other):
        return self.__binary_operator__(other, Coord._MODE_DIV)

    def __truediv__(self, other):
        return self.__binary_operator__(other, Coord._MODE_DIV)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return (-self) + other

    def __rdiv__(self, other):
        return self.__binary_operator__(other, Coord._MODE_RDIV)

    def __rtruediv__(self, other):
        return self.__binary_operator__(other, Coord._MODE_RDIV)

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self.copy(-self.points, -self.bounds if self.bounds is not
                         None else None)

    def convert_units(self, unit):
        """
        Change the coordinate's units, converting the values in its points
        and bounds arrays.

        For example, if a coordinate's :attr:`~iris.coords.Coord.units`
        attribute is set to radians then::

            coord.convert_units('degrees')

        will change the coordinate's
        :attr:`~iris.coords.Coord.units` attribute to degrees and
        multiply each value in :attr:`~iris.coords.Coord.points` and
        :attr:`~iris.coords.Coord.bounds` by 180.0/:math:`\pi`.

        """
        # If the coord has units convert the values in points (and bounds if
        # present).
        if not self.units.is_unknown():
            self.points = self.units.convert(self.points, unit)
            if self.bounds is not None:
                self.bounds = self.units.convert(self.bounds, unit)
        self.units = unit

    def cells(self):
        """
        Returns an iterable of Cell instances for this Coord.

        For example::

           for cell in self.cells():
              ...

        """
        return _CellIterator(self)

    def _sanity_check_contiguous(self):
        if self.ndim != 1:
            raise iris.exceptions.CoordinateMultiDimError(
                'Invalid operation for {!r}. Contiguous bounds are not defined'
                ' for multi-dimensional coordinates.'.format(self.name()))
        if self.nbounds != 2:
            raise ValueError(
                'Invalid operation for {!r}, with {} bounds. Contiguous bounds'
                ' are only defined for coordinates with 2 bounds.'.format(
                    self.name(), self.nbounds))

    def is_contiguous(self, rtol=1e-05, atol=1e-08):
        """
        Return True if, and only if, this Coord is bounded with contiguous
        bounds to within the specified relative and absolute tolerances.

        Args:

        * rtol:
            The relative tolerance parameter (default is 1e-05).
        * atol:
            The absolute tolerance parameter (default is 1e-08).

        Returns:
            Boolean.

        """
        if self.bounds is not None:
            self._sanity_check_contiguous()
            return np.allclose(self.bounds[1:, 0], self.bounds[:-1, 1],
                               rtol=rtol, atol=atol)
        else:
            return False

    def contiguous_bounds(self):
        """
        Returns the N+1 bound values for a contiguous bounded coordinate
        of length N.

        .. note::

            If the coordinate is does not have bounds, this method will
            return bounds positioned halfway between the coordinate's points.

        """
        if self.bounds is None:
            warnings.warn('Coordinate {!r} is not bounded, guessing '
                          'contiguous bounds.'.format(self.name()))
            bounds = self._guess_bounds()
        else:
            self._sanity_check_contiguous()
            bounds = self.bounds

        c_bounds = np.resize(bounds[:, 0], bounds.shape[0] + 1)
        c_bounds[-1] = bounds[-1, 1]
        return c_bounds

    def is_monotonic(self):
        """Return True if, and only if, this Coord is monotonic."""

        if self.ndim != 1:
            raise iris.exceptions.CoordinateMultiDimError(self)

        if self.shape == (1,):
            return True

        if self.points is not None:
            if not iris.util.monotonic(self.points, strict=True):
                return False

        if self.bounds is not None:
            for b_index in range(self.nbounds):
                if not iris.util.monotonic(self.bounds[..., b_index],
                                           strict=True):
                    return False

        return True

    def is_compatible(self, other, ignore=None):
        """
        Return whether the coordinate is compatible with another.

        Compatibility is determined by comparing
        :meth:`iris.coords.Coord.name()`, :attr:`iris.coords.Coord.units`,
        :attr:`iris.coords.Coord.coord_system` and
        :attr:`iris.coords.Coord.attributes` that are present in both objects.

        Args:

        * other:
            An instance of :class:`iris.coords.Coord` or
            :class:`iris.coords.CoordDefn`.
        * ignore:
           A single attribute key or iterable of attribute keys to ignore when
           comparing the coordinates. Default is None. To ignore all
           attributes, set this to other.attributes.

        Returns:
           Boolean.

        """
        compatible = (self.name() == other.name() and
                      self.units == other.units and
                      self.coord_system == other.coord_system)

        if compatible:
            common_keys = set(self.attributes).intersection(other.attributes)
            if ignore is not None:
                if isinstance(ignore, six.string_types):
                    ignore = (ignore,)
                common_keys = common_keys.difference(ignore)
            for key in common_keys:
                if np.any(self.attributes[key] != other.attributes[key]):
                    compatible = False
                    break

        return compatible

    @property
    def dtype(self):
        """
        Abstract property which returns the Numpy data type of the Coordinate.

        """
        return self.points.dtype

    @property
    def ndim(self):
        """
        Return the number of dimensions of the coordinate (not including the
        bounded dimension).

        """
        return len(self.shape)

    @property
    def nbounds(self):
        """
        Return the number of bounds that this coordinate has (0 for no bounds).

        """
        nbounds = 0
        if self.bounds is not None:
            nbounds = self.bounds.shape[-1]
        return nbounds

    def has_bounds(self):
        return self.bounds is not None

    @property
    def shape(self):
        """The fundamental shape of the Coord, expressed as a tuple."""
        # Access the underlying _points attribute to avoid triggering
        # a deferred load unnecessarily.
        return self._points.shape

    def cell(self, index):
        """
        Return the single :class:`Cell` instance which results from slicing the
        points/bounds with the given index.

        .. note::

            If `iris.FUTURE.cell_datetime_objects` is True, then this
            method will return Cell objects whose `points` and `bounds`
            attributes contain either datetime.datetime instances or
            netcdftime.datetime instances (depending on the calendar).

        """
        index = iris.util._build_full_slice_given_keys(index, self.ndim)

        point = tuple(np.array(self.points[index], ndmin=1).flatten())
        if len(point) != 1:
            raise IndexError('The index %s did not uniquely identify a single '
                             'point to create a cell with.' % (index, ))

        bound = None
        if self.bounds is not None:
            bound = tuple(np.array(self.bounds[index], ndmin=1).flatten())

        if iris.FUTURE.cell_datetime_objects:
            if self.units.is_time_reference():
                point = self.units.num2date(point)
                if bound is not None:
                    bound = self.units.num2date(bound)

        return Cell(point, bound)

    def collapsed(self, dims_to_collapse=None):
        """
        Returns a copy of this coordinate which has been collapsed along
        the specified dimensions.

        Replaces the points & bounds with a simple bounded region.

        """
        if isinstance(dims_to_collapse, (int, np.integer)):
            dims_to_collapse = [dims_to_collapse]

        if dims_to_collapse is not None and \
                set(range(self.ndim)) != set(dims_to_collapse):
            raise ValueError('Cannot partially collapse a coordinate (%s).'
                             % self.name())

        if np.issubdtype(self.dtype, np.str):
            # Collapse the coordinate by serializing the points and
            # bounds as strings.
            serialize = lambda x: '|'.join([str(i) for i in x.flatten()])
            bounds = None
            if self.bounds is not None:
                shape = self.bounds.shape[1:]
                bounds = []
                for index in np.ndindex(shape):
                    index_slice = (slice(None),) + tuple(index)
                    bounds.append(serialize(self.bounds[index_slice]))
                dtype = np.dtype('S{}'.format(max(map(len, bounds))))
                bounds = np.array(bounds, dtype=dtype).reshape((1,) + shape)
            points = serialize(self.points)
            dtype = np.dtype('S{}'.format(len(points)))
            # Create the new collapsed coordinate.
            coord = self.copy(points=np.array(points, dtype=dtype),
                              bounds=bounds)
        else:
            # Collapse the coordinate by calculating the bounded extremes.
            if self.ndim > 1:
                msg = 'Collapsing a multi-dimensional coordinate. ' \
                    'Metadata may not be fully descriptive for {!r}.'
                warnings.warn(msg.format(self.name()))
            elif not self.is_contiguous():
                msg = 'Collapsing a non-contiguous coordinate. ' \
                    'Metadata may not be fully descriptive for {!r}.'
                warnings.warn(msg.format(self.name()))

            # Create bounds for the new collapsed coordinate.
            item = self.bounds if self.bounds is not None else self.points
            lower, upper = np.min(item), np.max(item)
            bounds_dtype = item.dtype
            bounds = [lower, upper]
            # Create points for the new collapsed coordinate.
            points_dtype = self.points.dtype
            points = [(lower + upper) * 0.5]

            # Create the new collapsed coordinate.
            coord = self.copy(points=np.array(points, dtype=points_dtype),
                              bounds=np.array(bounds, dtype=bounds_dtype))
        return coord

    def _guess_bounds(self, bound_position=0.5):
        """
        Return bounds for this coordinate based on its points.

        Kwargs:

        * bound_position - The desired position of the bounds relative to the
                           position of the points.

        Returns:
            A numpy array of shape (len(self.points), 2).

        .. note::

            This method only works for coordinates with ``coord.ndim == 1``.

        """
        # XXX Consider moving into DimCoord
        # ensure we have monotonic points
        if not self.is_monotonic():
            raise ValueError("Need monotonic points to generate bounds for %s"
                             % self.name())

        if self.ndim != 1:
            raise iris.exceptions.CoordinateMultiDimError(self)

        if self.shape[0] < 2:
            raise ValueError('Cannot guess bounds for a coordinate of length '
                             '1.')

        if self.bounds is not None:
            raise ValueError('Coord already has bounds. Remove the bounds '
                             'before guessing new ones.')

        if getattr(self, 'circular', False):
            points = np.empty(self.points.shape[0] + 2)
            points[1:-1] = self.points
            direction = 1 if self.points[-1] > self.points[0] else -1
            points[0] = self.points[-1] - (self.units.modulus * direction)
            points[-1] = self.points[0] + (self.units.modulus * direction)
            diffs = np.diff(points)
        else:
            diffs = np.diff(self.points)
            diffs = np.insert(diffs, 0, diffs[0])
            diffs = np.append(diffs, diffs[-1])

        min_bounds = self.points - diffs[:-1] * bound_position
        max_bounds = self.points + diffs[1:] * (1 - bound_position)

        bounds = np.array([min_bounds, max_bounds]).transpose()

        return bounds

    def guess_bounds(self, bound_position=0.5):
        """
        Add contiguous bounds to a coordinate, calculated from its points.

        Puts a cell boundary at the specified fraction between each point and
        the next, plus extrapolated lowermost and uppermost bound points, so
        that each point lies within a cell.

        With regularly spaced points, the resulting bounds will also be
        regular, and all points lie at the same position within their cell.
        With irregular points, the first and last cells are given the same
        widths as the ones next to them.

        Kwargs:

        * bound_position - The desired position of the bounds relative to the
                           position of the points.

        .. note::

            An error is raised if the coordinate already has bounds, is not
            one-dimensional, or is not monotonic.

        .. note::

            Unevenly spaced values, such from a wrapped longitude range, can
            produce unexpected results :  In such cases you should assign
            suitable values directly to the bounds property, instead.

        """
        self.bounds = self._guess_bounds(bound_position)

    def intersect(self, other, return_indices=False):
        """
        Returns a new coordinate from the intersection of two coordinates.

        Both coordinates must be compatible as defined by
        :meth:`~iris.coords.Coord.is_compatible`.

        Kwargs:

        * return_indices:
            If True, changes the return behaviour to return the intersection
            indices for the "self" coordinate.

        """
        if not self.is_compatible(other):
            msg = 'The coordinates cannot be intersected. They are not ' \
                  'compatible because of differing metadata.'
            raise ValueError(msg)

        # Cache self.cells for speed. We can also use the index operation on a
        # list conveniently.
        self_cells = [cell for cell in self.cells()]

        # Maintain a list of indices on self for which cells exist in both self
        # and other.
        self_intersect_indices = []
        for cell in other.cells():
            try:
                self_intersect_indices.append(self_cells.index(cell))
            except ValueError:
                pass

        if return_indices is False and self_intersect_indices == []:
            raise ValueError('No intersection between %s coords possible.' %
                             self.name())

        self_intersect_indices = np.array(self_intersect_indices)

        # Return either the indices, or a Coordinate instance of the
        # intersection.
        if return_indices:
            return self_intersect_indices
        else:
            return self[self_intersect_indices]

    def nearest_neighbour_index(self, point):
        """
        Returns the index of the cell nearest to the given point.

        Only works for one-dimensional coordinates.

        .. note:: If the coordinate contains bounds, these will be used to
            determine the nearest neighbour instead of the point values.

        .. note:: For circular coordinates, the 'nearest' point can wrap around
            to the other end of the values.

        """
        points = self.points
        bounds = self.bounds if self.has_bounds() else np.array([])
        if self.ndim != 1:
            raise ValueError('Nearest-neighbour is currently limited'
                             ' to one-dimensional coordinates.')
        do_circular = getattr(self, 'circular', False)
        if do_circular:
            wrap_modulus = self.units.modulus
            # wrap 'point' to a range based on lowest points or bounds value.
            wrap_origin = np.min(np.hstack((points, bounds.flatten())))
            point = wrap_origin + (point - wrap_origin) % wrap_modulus

        # Calculate the nearest neighbour.
        # The algorithm:  given a single value (V),
        #   if coord has bounds,
        #     make bounds cells complete and non-overlapping
        #     return first cell containing V
        #   else (no bounds),
        #     find the point which is closest to V
        #     or if two are equally close, return the lowest index
        if self.has_bounds():
            # make bounds ranges complete+separate, so point is in at least one
            increasing = self.bounds[0, 1] > self.bounds[0, 0]
            bounds = bounds.copy()
            # sort the bounds cells by their centre values
            sort_inds = np.argsort(np.mean(bounds, axis=1))
            bounds = bounds[sort_inds]
            # replace all adjacent bounds with their averages
            if increasing:
                mid_bounds = 0.5 * (bounds[:-1, 1] + bounds[1:, 0])
                bounds[:-1, 1] = mid_bounds
                bounds[1:, 0] = mid_bounds
            else:
                mid_bounds = 0.5 * (bounds[:-1, 0] + bounds[1:, 1])
                bounds[:-1, 0] = mid_bounds
                bounds[1:, 1] = mid_bounds

            # if point lies beyond either end, fix the end cell to include it
            bounds[0, 0] = min(point, bounds[0, 0])
            bounds[-1, 1] = max(point, bounds[-1, 1])
            # get index of first-occurring cell that contains the point
            inside_cells = np.logical_and(point >= np.min(bounds, axis=1),
                                          point <= np.max(bounds, axis=1))
            result_index = np.where(inside_cells)[0][0]
            # return the original index of the cell (before the bounds sort)
            result_index = sort_inds[result_index]

        # Or, if no bounds, we always have points ...
        else:
            if do_circular:
                # add an extra, wrapped max point (simpler than bounds case)
                # NOTE: circular implies a DimCoord, so *must* be monotonic
                if points[-1] >= points[0]:
                    # ascending value order : add wrapped lowest value to end
                    index_offset = 0
                    points = np.hstack((points, points[0] + wrap_modulus))
                else:
                    # descending order : add wrapped lowest value at start
                    index_offset = 1
                    points = np.hstack((points[-1] + wrap_modulus, points))
            # return index of first-occurring nearest point
            distances = np.abs(points - point)
            result_index = np.where(distances == np.min(distances))[0][0]
            if do_circular:
                # convert index back from circular-adjusted points
                result_index = (result_index - index_offset) % self.shape[0]

        return result_index

    def xml_element(self, doc):
        """Return a DOM element describing this Coord."""
        # Create the XML element as the camelCaseEquivalent of the
        # class name.
        element_name = type(self).__name__
        element_name = element_name[0].lower() + element_name[1:]
        element = doc.createElement(element_name)

        element.setAttribute('id', self._xml_id())

        if self.standard_name:
            element.setAttribute('standard_name', str(self.standard_name))
        if self.long_name:
            element.setAttribute('long_name', str(self.long_name))
        if self.var_name:
            element.setAttribute('var_name', str(self.var_name))
        element.setAttribute('units', repr(self.units))

        if self.attributes:
            attributes_element = doc.createElement('attributes')
            for name in sorted(six.iterkeys(self.attributes)):
                attribute_element = doc.createElement('attribute')
                attribute_element.setAttribute('name', name)
                attribute_element.setAttribute('value',
                                               str(self.attributes[name]))
                attributes_element.appendChild(attribute_element)
            element.appendChild(attributes_element)

        # Add a coord system sub-element?
        if self.coord_system:
            element.appendChild(self.coord_system.xml_element(doc))

        # Add the values
        element.setAttribute('value_type', str(self._value_type_name()))
        element.setAttribute('shape', str(self.shape))
        if hasattr(self.points, 'to_xml_attr'):
            element.setAttribute('points', self.points.to_xml_attr())
        else:
            element.setAttribute('points', iris.util.format_array(self.points))

        if self.bounds is not None:
            if hasattr(self.bounds, 'to_xml_attr'):
                element.setAttribute('bounds', self.bounds.to_xml_attr())
            else:
                element.setAttribute('bounds',
                                     iris.util.format_array(self.bounds))

        return element

    def _xml_id(self):
        # Returns a consistent, unique string identifier for this coordinate.
        unique_value = b''
        if self.standard_name:
            unique_value += self.standard_name.encode('utf-8')
        unique_value += b'\0'
        if self.long_name:
            unique_value += self.long_name.encode('utf-8')
        unique_value += b'\0'
        unique_value += str(self.units).encode('utf-8') + b'\0'
        for k, v in sorted(self.attributes.items()):
            unique_value += (str(k) + ':' + str(v)).encode('utf-8') + b'\0'
        unique_value += str(self.coord_system).encode('utf-8') + b'\0'
        # Mask to ensure consistency across Python versions & platforms.
        crc = zlib.crc32(unique_value) & 0xffffffff
        return '%08x' % (crc, )

    def _value_type_name(self):
        """
        A simple, readable name for the data type of the Coord point/bound
        values.

        """
        values = self.points
        value_type_name = values.dtype.name
        if self.points.dtype.kind == 'S':
            value_type_name = 'string'
        elif self.points.dtype.kind == 'U':
            value_type_name = 'unicode'
        return value_type_name


class DimCoord(Coord):
    """
    A coordinate that is 1D, numeric, and strictly monotonic.

    """
    @staticmethod
    def from_coord(coord):
        """Create a new DimCoord from the given coordinate."""
        return DimCoord(coord.points, standard_name=coord.standard_name,
                        long_name=coord.long_name, var_name=coord.var_name,
                        units=coord.units, bounds=coord.bounds,
                        attributes=coord.attributes,
                        coord_system=copy.deepcopy(coord.coord_system),
                        circular=getattr(coord, 'circular', False))

    @classmethod
    def from_regular(cls, zeroth, step, count, standard_name=None,
                     long_name=None, var_name=None, units='1', attributes=None,
                     coord_system=None, circular=False, with_bounds=False):
        """
        Create a :class:`DimCoord` with regularly spaced points, and
        optionally bounds.

        The majority of the arguments are defined as for
        :meth:`Coord.__init__`, but those which differ are defined below.

        Args:

        * zeroth:
            The value *prior* to the first point value.
        * step:
            The numeric difference between successive point values.
        * count:
            The number of point values.

        Kwargs:

        * with_bounds:
            If True, the resulting DimCoord will possess bound values
            which are equally spaced around the points. Otherwise no
            bounds values will be defined. Defaults to False.

        """
        coord = DimCoord.__new__(cls)

        coord.standard_name = standard_name
        coord.long_name = long_name
        coord.var_name = var_name
        coord.units = units
        coord.attributes = attributes
        coord.coord_system = coord_system
        coord.circular = circular

        points = (zeroth+step) + step*np.arange(count, dtype=np.float32)
        points.flags.writeable = False
        coord._points = points
        if not is_regular(coord) and count > 1:
            points = (zeroth+step) + step*np.arange(count, dtype=np.float64)
            points.flags.writeable = False
            coord._points = points

        if with_bounds:
            delta = 0.5 * step
            bounds = np.concatenate([[points - delta], [points + delta]]).T
            bounds.flags.writeable = False
            coord._bounds = bounds
        else:
            coord._bounds = None

        return coord

    def __init__(self, points, standard_name=None, long_name=None,
                 var_name=None, units='1', bounds=None, attributes=None,
                 coord_system=None, circular=False):
        """
        Create a 1D, numeric, and strictly monotonic :class:`Coord` with
        read-only points and bounds.

        """
        Coord.__init__(self, points, standard_name=standard_name,
                       long_name=long_name, var_name=var_name,
                       units=units, bounds=bounds, attributes=attributes,
                       coord_system=coord_system)

        #: Whether the coordinate wraps by ``coord.units.modulus``.
        self.circular = bool(circular)

    def copy(self, points=None, bounds=None):
        new_coord = super(DimCoord, self).copy(points=points, bounds=bounds)
        # Make the array read-only.
        new_coord._points.flags.writeable = False
        if new_coord._bounds is not None:
            new_coord._bounds.flags.writeable = False
        return new_coord

    def __eq__(self, other):
        # TODO investigate equality of AuxCoord and DimCoord if circular is
        # False.
        result = NotImplemented
        if isinstance(other, DimCoord):
            result = (Coord.__eq__(self, other) and self.circular ==
                      other.circular)
        return result

    # The __ne__ operator from Coord implements the not __eq__ method.

    def __getitem__(self, key):
        coord = super(DimCoord, self).__getitem__(key)
        coord.circular = self.circular and coord.shape == self.shape
        return coord

    def collapsed(self, dims_to_collapse=None):
        coord = Coord.collapsed(self, dims_to_collapse=dims_to_collapse)
        if self.circular and self.units.modulus is not None:
            bnds = coord.bounds.copy()
            bnds[0, 1] = coord.bounds[0, 0] + self.units.modulus
            coord.bounds = bnds
            coord.points = np.array(np.sum(coord.bounds) * 0.5,
                                    dtype=self.points.dtype)
        # XXX This isn't actually correct, but is ported from the old world.
        coord.circular = False
        return coord

    def _repr_other_metadata(self):
        result = Coord._repr_other_metadata(self)
        if self.circular:
            result += ', circular=%r' % self.circular
        return result

    @property
    def points(self):
        """The local points values as a read-only NumPy array."""
        points = self._points.view()
        return points

    @points.setter
    def points(self, points):
        points = np.array(points, ndmin=1)
        # If points are already defined for this coordinate,
        if hasattr(self, '_points') and self._points is not None:
            # Check that setting these points wouldn't change self.shape
            if points.shape != self.shape:
                raise ValueError("New points shape must match existing points "
                                 "shape.")

        # Checks for 1d, numeric, monotonic
        if points.ndim != 1:
            raise ValueError('The points array must be 1-dimensional.')
        if not np.issubdtype(points.dtype, np.number):
            raise ValueError('The points array must be numeric.')
        if len(points) > 1 and not iris.util.monotonic(points, strict=True):
            raise ValueError('The points array must be strictly monotonic.')
        # Make the array read-only.
        points.flags.writeable = False

        self._points = points

    @property
    def bounds(self):
        """
        The bounds values as a read-only NumPy array, or None if no
        bounds have been set.

        """
        bounds = None
        if self._bounds is not None:
            bounds = self._bounds.view()
        return bounds

    @bounds.setter
    def bounds(self, bounds):
        if bounds is not None:
            # Ensure the bounds are a compatible shape.
            bounds = np.array(bounds, ndmin=2)
            if self.shape != bounds.shape[:-1]:
                raise ValueError(
                    "The shape of the bounds array should be "
                    "points.shape + (n_bounds,)")
            # Checks for numeric and monotonic
            if not np.issubdtype(bounds.dtype, np.number):
                raise ValueError('The bounds array must be numeric.')

            n_bounds = bounds.shape[-1]
            n_points = bounds.shape[0]
            if n_points > 1:

                directions = set()
                for b_index in range(n_bounds):
                    monotonic, direction = iris.util.monotonic(
                        bounds[:, b_index], strict=True, return_direction=True)
                    if not monotonic:
                        raise ValueError('The bounds array must be strictly '
                                         'monotonic.')
                    directions.add(direction)

                if len(directions) != 1:
                    raise ValueError('The direction of monotonicity must be '
                                     'consistent across all bounds')

            # Ensure the array is read-only.
            bounds.flags.writeable = False

        self._bounds = bounds

    def is_monotonic(self):
        return True

    def xml_element(self, doc):
        """Return DOM element describing this :class:`iris.coords.DimCoord`."""
        element = super(DimCoord, self).xml_element(doc)
        if self.circular:
            element.setAttribute('circular', str(self.circular))
        return element


class AuxCoord(Coord):
    """A CF auxiliary coordinate."""
    @staticmethod
    def from_coord(coord):
        """Create a new AuxCoord from the given coordinate."""
        new_coord = AuxCoord(coord.points, standard_name=coord.standard_name,
                             long_name=coord.long_name,
                             var_name=coord.var_name,
                             units=coord.units, bounds=coord.bounds,
                             attributes=coord.attributes,
                             coord_system=copy.deepcopy(coord.coord_system))

        return new_coord

    def _sanitise_array(self, src, ndmin):
        # Ensure the array is writeable.
        # NB. Returns the *same object* if src is already writeable.
        result = np.require(src, requirements='W')
        # Ensure the array has enough dimensions.
        # NB. Returns the *same object* if result.ndim >= ndmin
        result = np.array(result, ndmin=ndmin, copy=False)
        # We don't need to copy the data, but we do need to have our
        # own view so we can control the shape, etc.
        result = result.view()
        return result

    @property
    def points(self):
        """Property containing the points values as a numpy array"""
        points = self._points
        if isinstance(points, biggus.Array):
            points = points.ndarray()
            self._points = points
        return points.view()

    @points.setter
    def points(self, points):
        # Set the points to a new array - as long as it's the same shape.

        # With the exception of LazyArrays, ensure points has an ndmin
        # of 1 and is either a numpy or biggus array.
        # This will avoid Scalar coords with points of shape () rather
        # than the desired (1,)
        if isinstance(points, biggus.Array):
            if points.shape == ():
                points = biggus.ConstantArray((1,), points.ndarray(),
                                              points.dtype)
        elif not isinstance(points, iris.aux_factory._LazyArray):
            points = self._sanitise_array(points, 1)
        # If points are already defined for this coordinate,
        if hasattr(self, '_points') and self._points is not None:
            # Check that setting these points wouldn't change self.shape
            if points.shape != self.shape:
                raise ValueError("New points shape must match existing points "
                                 "shape.")

        self._points = points

    @property
    def bounds(self):
        """
        Property containing the bound values, as a numpy array,
        or None if no bound values are defined.

        .. note:: The shape of the bound array should be: ``points.shape +
            (n_bounds, )``.

        """
        if self._bounds is not None:
            bounds = self._bounds
            if isinstance(bounds, biggus.Array):
                bounds = bounds.ndarray()
                self._bounds = bounds
            bounds = bounds.view()
        else:
            bounds = None

        return bounds

    @bounds.setter
    def bounds(self, bounds):
        # Ensure the bounds are a compatible shape.
        if bounds is not None:
            if not isinstance(bounds, (iris.aux_factory._LazyArray,
                                       biggus.Array)):
                bounds = self._sanitise_array(bounds, 2)
            # NB. Use _points to avoid triggering any lazy array.
            if self._points.shape != bounds.shape[:-1]:
                raise ValueError("Bounds shape must be compatible with points "
                                 "shape.")
        self._bounds = bounds


class CellMethod(iris.util._OrderedHashable):
    """
    Represents a sub-cell pre-processing operation.

    """

    # Declare the attribute names relevant to the _OrderedHashable behaviour.
    _names = ('method', 'coord_names', 'intervals', 'comments')

    #: The name of the operation that was applied. e.g. "mean", "max", etc.
    method = None

    #: The tuple of coordinate names over which the operation was applied.
    coord_names = None

    #: A description of the original intervals over which the operation
    #: was applied.
    intervals = None

    #: Additional comments.
    comments = None

    def __init__(self, method, coords=None, intervals=None, comments=None):
        """
        Args:

        * method:
            The name of the operation.

        Kwargs:

        * coords:
            A single instance or sequence of :class:`.Coord` instances or
            coordinate names.

        * intervals:
            A single string, or a sequence strings, describing the intervals
            within the cell method.

        * comments:
            A single string, or a sequence strings, containing any additional
            comments.

        """
        if not isinstance(method, six.string_types):
            raise TypeError("'method' must be a string - got a '%s'" %
                            type(method))

        _coords = []
        if coords is None:
            pass
        elif isinstance(coords, Coord):
            _coords.append(coords.name())
        elif isinstance(coords, six.string_types):
            _coords.append(coords)
        else:
            normalise = (lambda coord: coord.name() if
                         isinstance(coord, Coord) else coord)
            _coords.extend([normalise(coord) for coord in coords])

        _intervals = []
        if intervals is None:
            pass
        elif isinstance(intervals, six.string_types):
            _intervals = [intervals]
        else:
            _intervals.extend(intervals)

        _comments = []
        if comments is None:
            pass
        elif isinstance(comments, six.string_types):
            _comments = [comments]
        else:
            _comments.extend(comments)

        self._init(method, tuple(_coords), tuple(_intervals), tuple(_comments))

    def __str__(self):
        """Return a custom string representation of CellMethod"""
        # Group related coord names intervals and comments together
        cell_components = zip_longest(self.coord_names, self.intervals,
                                      self.comments, fillvalue="")

        collection_summaries = []
        cm_summary = "%s: " % self.method

        for coord_name, interval, comment in cell_components:
            other_info = ", ".join(filter(None, chain((interval, comment))))
            if other_info:
                coord_summary = "%s (%s)" % (coord_name, other_info)
            else:
                coord_summary = "%s" % coord_name

            collection_summaries.append(coord_summary)

        return cm_summary + ", ".join(collection_summaries)

    def __add__(self, other):
        # Disable the default tuple behaviour of tuple concatenation
        raise NotImplementedError()

    def xml_element(self, doc):
        """
        Return a dom element describing itself

        """
        cellMethod_xml_element = doc.createElement('cellMethod')
        cellMethod_xml_element.setAttribute('method', self.method)

        for coord_name, interval, comment in zip_longest(self.coord_names,
                                                         self.intervals,
                                                         self.comments):
            coord_xml_element = doc.createElement('coord')
            if coord_name is not None:
                coord_xml_element.setAttribute('name', coord_name)
                if interval is not None:
                    coord_xml_element.setAttribute('interval', interval)
                if comment is not None:
                    coord_xml_element.setAttribute('comment', comment)
                cellMethod_xml_element.appendChild(coord_xml_element)

        return cellMethod_xml_element


# See Coord.cells() for the description/context.
class _CellIterator(collections.Iterator):
    def __init__(self, coord):
        self._coord = coord
        if coord.ndim != 1:
            raise iris.exceptions.CoordinateMultiDimError(coord)
        self._indices = iter(range(coord.shape[0]))

    def next(self):
        # NB. When self._indices runs out it will raise StopIteration for us.
        i = next(self._indices)
        return self._coord.cell(i)


# See ExplicitCoord._group() for the description/context.
class _GroupIterator(collections.Iterator):
    def __init__(self, points):
        self._points = points
        self._start = 0

    def next(self):
        num_points = len(self._points)
        if self._start >= num_points:
            raise StopIteration

        stop = self._start + 1
        m = self._points[self._start]
        while stop < num_points and self._points[stop] == m:
            stop += 1

        group = _GroupbyItem(m, slice(self._start, stop))
        self._start = stop
        return group
