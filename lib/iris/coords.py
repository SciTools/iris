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
Definitions of coordinates.

"""
from __future__ import division

from abc import ABCMeta, abstractmethod, abstractproperty
from copy import deepcopy
import collections
from itertools import izip, chain, izip_longest
import operator
import re
import warnings

import numpy

import iris.aux_factory
import iris.exceptions
import iris.unit
import iris.util

from iris._cube_coord_common import CFVariableMixin, LimitedAttributeDict


class CoordDefn(collections.namedtuple('CoordDefn', 
                                       ['standard_name', 'long_name', 'units',
                                        'attributes', 'coord_system'])):
    """
    Criterion for identifying a specific type of :class:`DimCoord` or :class:`AuxCoord`
    based on its metadata.

    """
    def name(self, default='unknown'):
        """
        Returns a human-readable name.
        
        First it tries self.standard_name, then it tries the 'long_name'
        attributes, before falling back to the value of `default` (which
        itself defaults to 'unknown').

        """
        return self.standard_name or self.long_name or default


# Coordinate cell styles. Used in plot and cartography.
POINT_MODE = 0
BOUND_MODE = 1

BOUND_POSITION_START = 0
BOUND_POSITION_MIDDLE = 0.5
BOUND_POSITION_END = 1

# Private named tuple class for coordinate groups.
_GroupbyItem = collections.namedtuple('GroupbyItem', 'groupby_point, groupby_slice')

class Cell(iris.util._OrderedHashable):
    """
    An immutable representation of a single cell of a coordinate, including the sample point
    and/or boundary position.
    """

    # Declare the attribute names relevant to the _OrderedHashable behaviour.
    _names = ('point', 'bound')

    def __init__(self, point=None, bound=None):
        """
        Construct a Cell from point or point-and-bound information. 
        
        Notes on cell comparison:
    
        Cells can be compared in a number of ways depending on how they are configured and the data
        type they are being compared to.
        
        For equality comparison (i.e. checking that a cell is equal to something) the following rules apply:
        
        1. When comparing to another cell, equality is determined by comparing tuples of attribute values from both objects.
            
        2. When comparing to a string value and cell point is also a string then comparison is done by string comparison.
        
        3. When comparing to an integer or float and there are bounds set on the cell then the comparison is True if the numeric value
           'other' is within the cells bounds. If no bounds are set on the cell then the comparison is performed on the cells point value.
        
        For rich comparisions i.e. (less than, greater than, less than or equals and greater than or equals) the following rules apply:
        
        1. When comparing to an integer, float or numpy.number and the cell is bounded, the comparison is performed 
           on the max or min of the cells bounds. If the cell is unbounded then the numeric value is compared with
           the cells point value.
        
        2. If both comparison objects are cells the following table describes how the objects are compared:
           
           # TODO: This seem wrong, due to the inconsistency between the first two scenarios. See #1278.
           
           ========================================= ==================================================================
           Primary and comparison cell styles        Comparison method
           ========================================= ==================================================================
           Bounded vs unbounded                      Comparison point is compared to primary bounds.                           
           Unbounded vs bounded                      Point values are compared.                           
           Bounded vs bounded                        Points values and max or min of both bounds are compared                   
           ========================================= ==================================================================

        """ 
        if point is None:
            raise ValueError('Point must be defined.')
        
        if bound is not None:
            bound = tuple(bound)
            
        if isinstance(point, numpy.ndarray):
            point = tuple(point.flatten())
            
        if isinstance(point, (tuple, list)):
            if len(point) != 1:
                raise ValueError('Point may only be a list or tuple if it has length 1.')
            point = point[0]

        self._init(point, bound) 

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
        Compares Cell equality depending on the type of the object to be compared.    
        """
        if isinstance(other, (int, float)):
            if self.bound is not None:
                return self.contains_point(other)
            else:
                return self.point == other
        elif isinstance(other, Cell):
            return self._as_tuple() == other._as_tuple()
        elif isinstance(other, basestring) and self.bound is None and isinstance(self.point, basestring):
            return self.point == other
        else:
            return NotImplemented

    def __common_cmp__(self, other, operator_method):
        """
        Common method called by the rich comparison operators. The method of checking equality
        depends on the type of the object to be compared.
        """
        if not isinstance(other, (int, float, numpy.number, Cell)):
            raise ValueError("Unexpected type of other")
        if operator_method not in [operator.gt, operator.lt, operator.ge, operator.le]:
            raise ValueError("Unexpected operator_method")
        
        # Do we have bounds?
        if self.bound is not None:
            if operator_method in [operator.gt, operator.le]:
                me = min(self.bound)
            else:
                me = max(self.bound)
        else:
            me = self.point
    
        # Does other have bounds?
        if isinstance(other, (int, float, numpy.number)):
            it = other
        elif other.bound is not None:
            if operator_method in [operator.lt, operator.ge]:
                it = min(other.bound)
            else:
                it = max(other.bound)
        else:
            it = other.point
        
        # Compare the designated components
        res = operator_method(me, it)
        
        # If we both have bounds, we must also check our points
        if self.bound is not None and isinstance(other, Cell) and other.bound is not None:
            res = res and operator_method(self.point, other.point) 
        
        return res
    
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
        For a bounded cell, returns whether the given point lies within the bounds.

        .. note:: The test carried out is equivalent to min(bound) <= point <= max(bound).

        """
        if self.bound is None:
            raise ValueError('Point cannot exist inside an unbounded cell.')

        return numpy.min(self.bound) <= point <= numpy.max(self.bound)


class Coord(CFVariableMixin):
    """
    Abstract superclass for coordinates.

    """
    __metaclass__ = ABCMeta

    _MODE_ADD = 1
    _MODE_SUB = 2
    _MODE_MUL = 3
    _MODE_DIV = 4
    _MODE_RDIV = 5
    _MODE_SYMBOL = { _MODE_ADD: '+', _MODE_SUB: '-',
                     _MODE_MUL: '*', _MODE_DIV: '/',
                     _MODE_RDIV:'/',                 }

    def __init__(self, points, standard_name=None, long_name=None, 
                 units='1', bounds=None, attributes=None,
                 coord_system=None):
    
        """
        Constructs a single coordinate.

        Args:

        * points:
            The values (or value in the case of a scalar coordinate) of the coordinate for each cell.

        Kwargs:

        * standard_name:
            CF standard name of coordinate
        * long_name:
            Descriptive name of coordinate
        * units
            Unit for coordinate's values
        * bounds
            An array of values describing the bounds of each cell. The shape 
            of the array must be compatible with the points array.
        * attributes
            A dictionary containing other cf and user-defined attributes.
        * coord_system
            A :class:`~iris.coord_systems.CoordSystem`, e.g a :class:`~iris.coord_systems.GeogCS` for a longitude Coord.

        """
        self.standard_name = standard_name
        """CF standard name of the quantity that the coordinate represents."""

        self.long_name = long_name
        """Descriptive name of the coordinate."""

        self.units = iris.unit.as_unit(units)
        """Unit of the quantity that the coordinate represents."""

        self.attributes = attributes
        """Other attributes, including user specified attributes that have no meaning to Iris."""

        self.coord_system = coord_system
        """Relevant CoordSystem (if any)."""    
        
        self.points = points
        self.bounds = bounds

    def __getitem__(self, key):
        """
        Returns a new Coord whose values are obtained by conventional array indexing.

        .. note:: 
            Indexing of a circular coordinate results in a non-circular coordinate if the overall shape of
            the coordinate changes after indexing.

        """
        points = self.points
        bounds = self.bounds
        
        # turn the key(s) into a full slice spec (all dims) 
        full_slice = iris.util._build_full_slice_given_keys(key, self.ndim)
        
        # make indexing on the cube column based by using the column_slices_generator 
        # (potentially requires slicing the data multiple times)
        _, slice_gen = iris.util.column_slices_generator(full_slice, self.ndim)
        
        for keys in slice_gen:
            if points is not None:
                points = points[keys]
                if points.shape and min(points.shape) == 0:
                    raise IndexError('Cannot index with zero length slice.')
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
                  This must be the compatible with the points array of the 
                  coordinate being created.
        
        .. note:: If the points argument is specified and bounds are not, the
                  resulting coordinate will have no bounds.
        
        """

        if points is None and bounds is not None:
            raise ValueError('If bounds are specified, points must also be specified')

        new_coord = deepcopy(self)
        if points is not None:
            # Explicitly not using the points property as we don't want the shape
            # of the new points to be constrained by the shape of self.points
            new_coord._points = None
            new_coord.points = points
            # Regardless of whether bounds are provided as an argument, new points 
            # will result in new bounds, discarding those copied from self.
            new_coord.bounds = bounds  
            
        return new_coord

    @abstractproperty
    def points(self):
        """Property containing the points values as a numpy array"""
        
    @abstractproperty
    def bounds(self):
        """Property containing the bound values as a numpy array"""

    def _format_metadata(self):
        result = "standard_name=%r, units=%r" % (self.standard_name, self.units)
        if self.long_name:
            result += ', long_name=%r' % self.long_name
        if len(self.attributes) > 0:
            result += ", attributes=" + str(self.attributes)
        if self.coord_system:
            result += ", coord_system=" + str(self.coord_system)
        return result
    
    def __repr__(self):
        result = '%s(%r' % (self.__class__.__name__, self.points)
        if self.bounds is not None:
            result += ', bounds=%r' % self.bounds
        result += ', %s)' % self._format_metadata()
        return result
    
    def __eq__(self, other):
        eq = NotImplemented
        # if the other object has a means of getting its definition, and whether or not it has_points and has_bounds,
        # then do the comparison, otherwise return a NotImplemented to let Python try to resolve the operator elsewhere.
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
        result = self == other
        if result != NotImplemented:
            result = not result
        return result

    def _as_defn(self):
        defn = CoordDefn(self.standard_name, self.long_name, self.units,
                         self.attributes, self.coord_system)
        return defn

    def __binary_operator__(self, other, mode_constant):
        """
        Common code which is called by add, sub, mult and div

        Mode constant is one of ADD, SUB, MUL, DIV, RDIV

        .. note:: The unit is *not* changed when doing scalar operations on a coordinate. This means that
                  a coordinate which represents "10 meters" when multiplied by a scalar i.e. "1000" would result
                  in a coordinate of "10000 meters". An alternative approach could be taken to multiply the *unit*
                  by 1000 and the resultant coordinate would represent "10 kilometers".
        """
        if isinstance(other, Coord):
            raise iris.exceptions.NotYetImplementedError('coord %s coord' % Coord._MODE_SYMBOL[mode_constant])

        elif isinstance(other, (int, float, numpy.number)):
        
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
        return self.copy(-self.points, -self.bounds if self.bounds is not None else None)
        
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
                    self.name()))

    def is_contiguous(self):
        """
        Return True if, and only if, this Coord is bounded with contiguous bounds.

        """
        if self.bounds is not None:
            self._sanity_check_contiguous()
            return numpy.all(self.bounds[1:, 0] == self.bounds[:-1, 1])
        else:
            return False

    def contiguous_bounds(self):
        """Returns the N+1 bound values for a contiguous bounded coordinate of length N."""
        self._sanity_check_contiguous()
        bounds = self.bounds
        if bounds is None:
            raise ValueError('Coordinate is not bounded, cannot get contiguous bounds.')
        c_bounds = numpy.resize(bounds[:, 0], bounds.shape[0] + 1)
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
            for b_index in xrange(self.nbounds):
                if not iris.util.monotonic(self.bounds[..., b_index], strict=True):
                    return False
                
        return True 

    @property
    def dtype(self):
        """
        Abstract property which returns the Numpy data type of the Coordinate.

        """
        return self.points.dtype

    @property
    def ndim(self):
        """Return the number of dimensions of the coordinate (not including the bounded dimension)."""
        return len(self.shape)
    
    @property
    def nbounds(self):
        """Return the number of bounds that this coordinate has (0 for no bounds)."""
        nbounds = 0
        if self.bounds is not None:
            nbounds = self.bounds.shape[-1]
        return nbounds
    
    def has_bounds(self):
        return self.bounds is not None

    @property
    def shape(self):
        """The fundamental shape of the Coord, expressed as a tuple."""
        return self.points.shape
        
    def index(self, cell):
        """
        Return the index of a given Cell in this Coord.

        """
        raise IrisError('Coord.index() is no longer available. Use Coord.nearest_neighbour_index() instead.')

    def cell(self, index):
        """
        Return the single :class:`Cell` instance which results from slicing the points/bounds with the given index.

        """
        index = iris.util._build_full_slice_given_keys(index, self.ndim)
        
        point = tuple(numpy.array(self.points[index], ndmin=1).flatten())
        if len(point) != 1:
            raise IndexError('The index %s did not uniquely identify a single point to create a cell with.' % (index, ))
        
        bound = None
        if self.bounds is not None:
            bound = tuple(numpy.array(self.bounds[index], ndmin=1).flatten())
            if len(bound) != self.nbounds:
                raise IndexError('The index %s did not uniquely identify a single bound to create a cell with.' % (index, ))
        
        return Cell(point, bound)
        
    def collapsed(self, dims_to_collapse=None):
        """
        Returns a copy of this coordinate which has been collapsed along
        the specified dimensions.

        Replaces the points & bounds with a simple bounded region.
        
        """
        if isinstance(dims_to_collapse, (int, numpy.integer)):
            dims_to_collapse = set([dims_to_collapse])

        if dims_to_collapse is None:
            dims_to_collapse = set(range(self.ndim))
        else:
            if set(range(self.ndim)) != set(dims_to_collapse):
                raise ValueError('Cannot partially collapse a coordinate (%s).' % self.name())
        
        # Warn about non-contiguity.
        if self.ndim > 1:
            warnings.warn('Collapsing a multi-dimensional coordinate. Metadata may not be fully descriptive for "%s".' % self.name())
        elif not self.is_contiguous():
            warnings.warn('Collapsing a non-contiguous coordinate. Metadata may not be fully descriptive for "%s".' % self.name())

        # Create bounds for the new collapsed coordinate.
        if self.bounds is not None:
            lower_bound, upper_bound = numpy.min(self.bounds), numpy.max(self.bounds)
            bounds_dtype = self.bounds.dtype
        else:
            lower_bound, upper_bound = numpy.min(self.points), numpy.max(self.points)
            bounds_dtype = self.points.dtype
            
        points_dtype = self.points.dtype
        
        # Create the new collapsed coordinate.
        coord_collapsed = self.copy(points=numpy.array([(lower_bound + upper_bound) * 0.5], dtype=points_dtype), 
                                    bounds=numpy.array([lower_bound, upper_bound], dtype=bounds_dtype))
        return coord_collapsed

    def guess_bounds(self, bound_position=0.5):
        """
        Try to add bounds to this coordinate using the coordinate's points.
        
        Kwargs:
        
        * bound_position - The desired position of the bounds relative to the position of the points.

        This method only works for coordinates with ``coord.ndim == 1``.
        
        """
        # XXX Consider moving into DimCoord
        # ensure we have monotonic points
        if not self.is_monotonic():
            raise ValueError("Need monotonic points to generate bounds for %s" % self.name())

        if self.ndim != 1:
            raise iris.exceptions.CoordinateMultiDimError(self)
        
        if self.shape[0] < 2:
            raise ValueError('Cannot guess bounds for a coordinate of length 1.')

        if self.bounds is not None:
            raise ValueError('Coord already has bounds. Remove the bounds before guessing new ones.')

        diffs = numpy.diff(self.points)

        diffs = numpy.insert(diffs, 0, diffs[0])
        diffs = numpy.append(diffs, diffs[-1])

        min_bounds = self.points - diffs[:-1] * bound_position
        max_bounds = self.points + diffs[1:] * (1 - bound_position)

        self.bounds = numpy.array([min_bounds, max_bounds]).transpose()

    def intersect(self, other, return_indices=False):
        """
        Returns a new coordinate from the intersection of two coordinates.

        Both coordinates must have the same metadata, i.e. standard_name, long_name,
        units, attributes and coord_system.

        Kwargs:

        * return_indices:
            If True, changes the return behaviour to return the intersection indices
            for the "self" coordinate.

        """
        if self._as_defn() != other._as_defn():
            raise ValueError('Coords cannot be intersected, differing metadata.')

        # Cache self.cells for speed. We can also use the index operation on a list conveniently.
        self_cells = [cell for cell in self.cells()]

        # maintain a list of indices on self for which cells exist in both self and other
        self_intersect_indices = []
        for cell in other.cells():
            try:
                self_intersect_indices.append(self_cells.index(cell))
            except ValueError:
                pass

        if return_indices == False and self_intersect_indices == []:
            raise ValueError('No intersection between %s coords possible.' % self.name())

        self_intersect_indices = numpy.array(self_intersect_indices)

        # return either the indices, or a Coordinate instance of the intersection
        if return_indices:
            return self_intersect_indices
        else:
            return self[self_intersect_indices]

    def nearest_neighbour_index(self, point):
        """
        Returns the index of the cell nearest to the given point.

        .. note:: If the coordinate contain bounds, these will be used to determine
            the nearest neighbour instead of the point values.
            
        .. note:: Does not take into account the circular attribute of a coordinate.

        """
        # Calculate the nearest neighbour. The algorithm:  given a single value (V),
        # if the coord has bounds then find the bound (upper or lower) which is closest to V
        #     if "closest" results in two matches then return the index for which a cell contains V,
        #         if no such cell exists then pick the lowest index
        # if the coord has points then find the point which is closest to V
        #     if "closest" results in two matches then return the lowest index
        if self.has_bounds():
            diff = numpy.abs(self.bounds - point)
            # where will look like [[first dimension matches], [second dimension matches]]
            # we will just take the first match (in this case, it does not matter what the second dimension was)
            minimized_diff_indices = numpy.where(diff == numpy.min(diff))[0]

            min_index = None
            # If we have more than one result, try picking the result which actually contains the requested point
            if len(minimized_diff_indices) > 1:
                for index in minimized_diff_indices:
                    if self.cell(index).contains_point(point):
                        min_index = index
                        break
            # Pick the first index that WHERE returned if len(minimized_diff_indices) == 1 or we could not
            # find a cell which contained the point
            if min_index is None:
                min_index = minimized_diff_indices[0]

        # Then we have points
        else:
            diff = numpy.abs(self.points - point)
            min_index = numpy.where(diff == numpy.min(diff))[0][0]

        return min_index

    def sin(self):
        """
        Return a coordinate which represents sin(this coordinate).

        .. deprecated::
            This method has been deprecated.

        """
        warnings.warn('Coord.sin() has been deprecated.') 
        import iris.analysis.calculus
        return iris.analysis.calculus._coord_sin(self)

    def cos(self):
        """
        Return a coordinate which represents cos(this coordinate).

        .. deprecated::
            This method has been deprecated.

        """
        warnings.warn('Coord.cos() has been deprecated.') 
        import iris.analysis.calculus
        return iris.analysis.calculus._coord_cos(self)

    def unit_converted(self, new_unit):
        """Return a coordinate converted to a given unit."""
        points = self.units.convert(self.points, new_unit)
        if self.bounds is not None:
            bounds = self.units.convert(self.bounds, new_unit)
        else:
            bounds = None
        new_coord = self.copy(points=points, bounds=bounds)
        new_coord.units = iris.unit.as_unit(new_unit)
        return new_coord
    
    def xml_element(self, doc):
        """Return a DOM element describing this Coord."""
        # Create the xml element
        element_name = type(self).__name__
        element_name.replace(element_name[0], element_name[0].lower(), 1)  # lower case the first char
        element = doc.createElement(element_name)
        
        element.setAttribute('id', self._xml_id())

        if self.standard_name:
            element.setAttribute('standard_name', str(self.standard_name))
        if self.long_name:
            element.setAttribute('long_name', str(self.long_name))
        element.setAttribute('units', repr(self.units))

        if self.attributes:
            attributes_element = doc.createElement('attributes')
            for name in sorted(self.attributes.iterkeys()):
                attribute_element = doc.createElement('attribute')
                attribute_element.setAttribute('name', name)
                attribute_element.setAttribute('value', str(self.attributes[name]))
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
                element.setAttribute('bounds', iris.util.format_array(self.bounds))
            
        return element

    def _xml_id(self):
        # Returns a consistent, unique string identifier for this coordinate.
        # NB. `None` does not have a reliable hash value.
        unique_value = (self.standard_name or 0, str(self.units),
                        tuple(self.attributes.iteritems()))
        return str(hex(hash(unique_value))).lstrip('-0x')

    def _value_type_name(self):
        """A simple, readable name for the data type of the Coord point/bound values."""
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
                        long_name=coord.long_name, units=coord.units, 
                        bounds=coord.bounds, attributes=coord.attributes, 
                        coord_system=coord.coord_system,
                        circular=getattr(coord, 'circular', False))

    def __init__(self, points, standard_name=None, long_name=None, units='1',
                 bounds=None, attributes=None, coord_system=None, circular=False):
        """
        Create a 1D, numeric, and strictly monotonic :class:`Coord` with read-only points and bounds.

        """
        Coord.__init__(self, points, standard_name=standard_name, long_name=long_name, 
                       units=units, bounds=bounds, attributes=attributes, 
                       coord_system=coord_system)

        self.circular = bool(circular)
        """Whether the coordinate wraps by ``coord.units.modulus``."""
    
    def __eq__(self, other):
        # TODO investigate equality of AuxCoord and DimCoord if circular is False
        result = NotImplemented
        if isinstance(other, DimCoord):
            result = Coord.__eq__(self, other) and self.circular == other.circular
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
            coord.points = numpy.array(numpy.sum(coord.bounds) * 0.5, dtype=self.points.dtype)
        # XXX This isn't actually correct, but is ported from the old world.
        coord.circular = False
        return coord
    
    def _format_metadata(self):
        result = Coord._format_metadata(self)
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
        points = numpy.array(points, ndmin=1)
        # If points are already defined for this coordinate,
        if hasattr(self, '_points') and self._points is not None:
            # Check that setting these points wouldn't change self.shape
            if points.shape != self.shape:
                raise ValueError("New points shape must match existing points shape.")

        # Checks for 1d, numeric, monotonic
        if points.ndim != 1:
            raise ValueError('The points array must be 1-dimensional.')
        if not numpy.issubdtype(points.dtype, numpy.number):
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
            bounds = numpy.array(bounds, ndmin=2)
            if self.shape != bounds.shape[:-1]:
                raise ValueError("Bounds shape must be compatible with points shape.")
            # Checks for numeric and monotonic
            if not numpy.issubdtype(bounds.dtype, numpy.number):
                raise ValueError('The bounds array must be numeric.')
            
            n_bounds = bounds.shape[-1]
            n_points = bounds.shape[0]
            if n_points > 1:
                
                directions = set()
                for b_index in xrange(n_bounds):
                    monotonic, direction = iris.util.monotonic(bounds[:, b_index], 
                                                               strict=True, return_direction=True)
                    if not monotonic:
                        raise ValueError('The bounds array must be strictly monotonic.')
                    directions.add(direction)
                    
                if len(directions) != 1:
                    raise ValueError('The direction of monotonicity must be consistent across all bounds')
                
            # Ensure the array is read-only.
            bounds.flags.writeable = False

        self._bounds = bounds
    
    def is_monotonic(self):
        return True 


class AuxCoord(Coord):
    """A CF auxiliary coordinate."""
    @staticmethod
    def from_coord(coord):
        """Create a new AuxCoord from the given coordinate."""
        new_coord = AuxCoord(coord.points, standard_name=coord.standard_name, 
                             long_name=coord.long_name, units=coord.units, 
                             bounds=coord.bounds, attributes=coord.attributes, 
                             coord_system=coord.coord_system)

        return new_coord

    def _sanitise_array(self, src, ndmin):
        # Ensure the array is writeable.
        # NB. Returns the *same object* if src is already writeable.
        result = numpy.require(src, requirements='W')
        # Ensure the array has enough dimensions.
        # NB. Returns the *same object* if result.ndim >= ndmin
        result = numpy.array(result, ndmin=ndmin, copy=False)
        # We don't need to copy the data, but we do need to have our
        # own view so we can control the shape, etc.
        result = result.view()
        return result

    @property
    def points(self):
        """Property containing the points values as a numpy array"""
        return self._points.view()

    @points.setter
    def points(self, points):
        # Set the points to a new array - as long as it's the same shape.

        # With the exception of LazyArrays ensure points is a numpy array with ndmin of 1.
        # This will avoid Scalar coords with points of shape () rather than the desired (1,)
        #   ... could change to: points = lazy.array(points, ndmin=1)
        if not isinstance(points, iris.aux_factory.LazyArray):
            points = self._sanitise_array(points, 1)
        # If points are already defined for this coordinate,
        if hasattr(self, '_points') and self._points is not None:
            # Check that setting these points wouldn't change self.shape
            if points.shape != self.shape:
                raise ValueError("New points shape must match existing points shape.")
        
        self._points = points

    @property
    def bounds(self):
        """
        Property containing the bound values, as a numpy array, 
        or None if no bound values are defined.
        
        .. note:: The shape of the bound array should be: ``points.shape + (n_bounds, )``. 

        """
        if self._bounds is not None:
            bounds = self._bounds.view()
        else:
            bounds = None
            
        return bounds
        
    @bounds.setter
    def bounds(self, bounds):
        # Ensure the bounds are a compatible shape.
        if bounds is not None:
            if not isinstance(bounds, iris.aux_factory.LazyArray):
                bounds = self._sanitise_array(bounds, 2)
            # NB. Use _points to avoid triggering any lazy array.
            if self._points.shape != bounds.shape[:-1]:
                raise ValueError("Bounds shape must be compatible with points shape.")
        self._bounds = bounds


class CellMethod(iris.util._OrderedHashable):
    """
    Represents a sub-cell pre-processing operation.

    """

    # Declare the attribute names relevant to the _OrderedHashable behaviour.
    _names = ('method', 'coord_names', 'intervals', 'comments')

    method = None
    """The name of the operation that was applied. e.g. "mean", "max", etc."""

    coord_names = None
    """The tuple of coordinate names over which the operation was applied."""

    intervals = None
    """A description of the original intervals over which the operation was applied."""

    comments = None
    """Additional comments."""

    def __init__(self, method, coords=None, intervals=None, comments=None):
        """
        Args:

        * method:
            The name of the operation.

        Kwargs:

        * coords:
            A single instance or sequence of :class:`.Coord` instances or coordinate names.

        * intervals:
            A single string, or a sequence strings, describing the intervals within the cell method.

        * comments:
            A single string, or a sequence strings, containing any additional comments.

        """
        if not isinstance(method, basestring):
            raise TypeError("'method' must be a string - got a '%s'" % type(method))

        _coords = []
        if coords is None:
            pass
        elif isinstance(coords, Coord):
            _coords.append(coords.name())
        elif isinstance(coords, basestring):
            _coords.append(coords)
        else:
            normalise = lambda coord: coord.name() if isinstance(coord, Coord) else coord
            _coords.extend([normalise(coord) for coord in coords])

        _intervals = []
        if intervals is None:
            pass
        elif isinstance(intervals, basestring):
            _intervals = [intervals]
        else:
            _intervals.extend(intervals)

        _comments = []
        if comments is None:
            pass
        elif isinstance(comments, basestring):
            _comments = [comments]
        else:
            _comments.extend(comments)

        self._init(method, tuple(_coords), tuple(_intervals), tuple(_comments))

    def __str__(self):
        """Return a custom string representation of CellMethod"""
        # Group related coord names intervals and comments together 
        cell_components = izip_longest(self.coord_names, self.intervals, self.comments, fillvalue="")
        
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

        for coord_name, interval, comment in map(None, self.coord_names, self.intervals, self.comments):
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
        self._indices = iter(xrange(coord.shape[0]))

    def next(self):
        # NB. When self._indices runs out it will raise StopIteration for us.
        i = self._indices.next()
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
