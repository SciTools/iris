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
Provides objects for building up expressions useful for pattern matching.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

import collections
import operator

import numpy as np

import iris.coords
import iris.exceptions


class Constraint(object):
    """
    Constraints are the mechanism by which cubes can be pattern matched and
    filtered according to specific criteria.

    Once a constraint has been defined, it can be applied to cubes using the
    :meth:`Constraint.extract` method.

    """
    def __init__(self, name=None, cube_func=None, coord_values=None, **kwargs):
        """
        Creates a new instance of a Constraint which can be used for filtering
        cube loading or cube list extraction.

        Args:

        * name:   string or None
            If a string, it is used as the name to match against Cube.name().
        * cube_func:   callable or None
            If a callable, it must accept a Cube as its first and only argument
            and return either True or False.
        * coord_values:   dict or None
            If a dict, it must map coordinate name to the condition on the
            associated coordinate.
        * `**kwargs`:
            The remaining keyword arguments are converted to coordinate
            constraints. The name of the argument gives the name of a
            coordinate, and the value of the argument is the condition to meet
            on that coordinate::

                Constraint(model_level_number=10)

            Coordinate level constraints can be of several types:

            * **string, int or float** - the value of the coordinate to match.
              e.g. ``model_level_number=10``

            * **list of values** - the possible values that the coordinate may
              have to match. e.g. ``model_level_number=[10, 12]``

            * **callable** - a function which accepts a
              :class:`iris.coords.Cell` instance as its first and only argument
              returning True or False if the value of the Cell is desired.
              e.g. ``model_level_number=lambda cell: 5 < cell < 10``

        The :ref:`user guide <loading_iris_cubes>` covers cube much of
        constraining in detail, however an example which uses all of the
        features of this class is given here for completeness::

            Constraint(name='air_potential_temperature',
                       cube_func=lambda cube: cube.units == 'kelvin',
                       coord_values={'latitude':lambda cell: 0 < cell < 90},
                       model_level_number=[10, 12])
                       & Constraint(ensemble_member=2)

        Constraint filtering is performed at the cell level.
        For further details on how cell comparisons are performed see
        :class:`iris.coords.Cell`.

        """
        if not (name is None or isinstance(name, basestring)):
            raise TypeError('name must be None or string, got %r' % name)
        if not (cube_func is None or callable(cube_func)):
            raise TypeError('cube_func must be None or callable, got %r'
                            % cube_func)
        if not (coord_values is None or isinstance(coord_values,
                                                   collections.Mapping)):
            raise TypeError('coord_values must be None or a '
                            'collections.Mapping, got %r' % coord_values)

        coord_values = coord_values or {}
        duplicate_keys = coord_values.viewkeys() & kwargs.viewkeys()
        if duplicate_keys:
            raise ValueError('Duplicate coordinate conditions specified for: '
                             '%s' % list(duplicate_keys))

        self._name = name
        self._cube_func = cube_func

        self._coord_values = coord_values.copy()
        self._coord_values.update(kwargs)

        self._coord_constraints = []
        for coord_name, coord_thing in self._coord_values.items():
            self._coord_constraints.append(_CoordConstraint(coord_name,
                                                            coord_thing))

    def __repr__(self):
        args = []
        if self._name:
            args.append(('name', self._name))
        if self._cube_func:
            args.append(('cube_func', self._cube_func))
        if self._coord_values:
            args.append(('coord_values', self._coord_values))
        return 'Constraint(%s)' % ', '.join('%s=%r' % (k, v) for k, v in args)

    def _coordless_match(self, cube):
        """
        Return whether this constraint matches the given cube when not
        taking coordinates into account.

        """
        match = True
        if self._name:
            match = self._name == cube.name()
        if match and self._cube_func:
            match = self._cube_func(cube)
        return match

    def extract(self, cube):
        """
        Return the subset of the given cube which matches this constraint,
        else return None.

        """
        resultant_CIM = self._CIM_extract(cube)
        slice_tuple = resultant_CIM.as_slice()
        result = None
        if slice_tuple is not None:
            # Slicing the cube is an expensive operation.
            if all([item == slice(None) for item in slice_tuple]):
                # Don't perform a full slice, just return the cube.
                result = cube
            else:
                # Performing the partial slice.
                result = cube[slice_tuple]
        return result

    def _CIM_extract(self, cube):
        # Returns _ColumnIndexManager

        # Cater for scalar cubes by setting the dimensionality to 1
        # when cube.ndim is 0.
        resultant_CIM = _ColumnIndexManager(cube.ndim or 1)

        if not self._coordless_match(cube):
            resultant_CIM.all_false()
        else:
            for coord_constraint in self._coord_constraints:
                resultant_CIM = resultant_CIM & coord_constraint.extract(cube)

        return resultant_CIM

    def __and__(self, other):
        return ConstraintCombination(self, other, operator.__and__)

    def __rand__(self, other):
        return ConstraintCombination(other, self, operator.__and__)


class ConstraintCombination(Constraint):
    """Represents the binary combination of two Constraint instances."""
    def __init__(self, lhs, rhs, operator):
        """
        A ConstraintCombination instance is created by providing two
        Constraint instances and the appropriate :mod:`operator`.

        """
        try:
            lhs_constraint = as_constraint(lhs)
            rhs_constraint = as_constraint(rhs)
        except TypeError:
            raise TypeError('Can only combine Constraint instances, '
                            'got: %s and %s' % (type(lhs), type(rhs)))
        self.lhs = lhs_constraint
        self.rhs = rhs_constraint
        self.operator = operator

    def _coordless_match(self, cube):
        return self.operator(self.lhs._coordless_match(cube),
                             self.rhs._coordless_match(cube))

    def __repr__(self):
        return 'ConstraintCombination(%r, %r, %r)' % (self.lhs, self.rhs,
                                                      self.operator)

    def _CIM_extract(self, cube):
        return self.operator(self.lhs._CIM_extract(cube),
                             self.rhs._CIM_extract(cube))


class _CoordConstraint(object):
    """Represents the atomic elements which might build up a Constraint."""
    def __init__(self, coord_name, coord_thing):
        """
        Create a coordinate constraint given the coordinate name and a
        thing to compare it with.

        Arguments:

        * coord_name  -  string
            The name of the coordinate to constrain
        * coord_thing
            The object to compare

        """
        self.coord_name = coord_name
        self._coord_thing = coord_thing

    def __repr__(self):
        return '_CoordConstraint(%r, %r)' % (self.coord_name,
                                             self._coord_thing)

    def extract(self, cube):
        """
        Returns the the column based indices of the given cube which
        match the constraint.

        """
        # Cater for scalar cubes by setting the dimensionality to 1
        # when cube.ndim is 0.
        cube_cim = _ColumnIndexManager(cube.ndim or 1)
        try:
            coord = cube.coord(self.coord_name)
        except iris.exceptions.CoordinateNotFoundError:
            cube_cim.all_false()
            return cube_cim
        dims = cube.coord_dims(coord)
        if len(dims) > 1:
            msg = 'Cannot apply constraints to multidimensional coordinates'
            raise iris.exceptions.CoordinateMultiDimError(msg)

        try_quick = False
        if callable(self._coord_thing):
            call_func = self._coord_thing
        elif (isinstance(self._coord_thing, collections.Iterable) and
                not isinstance(self._coord_thing,
                               (basestring, iris.coords.Cell))):
            call_func = lambda cell: cell in list(self._coord_thing)
        else:
            call_func = lambda c: c == self._coord_thing
            try_quick = (isinstance(coord, iris.coords.DimCoord) and
                         not isinstance(self._coord_thing, iris.coords.Cell))

        # Simple, yet dramatic, optimisation for the monotonic case.
        if try_quick:
            try:
                i = coord.nearest_neighbour_index(self._coord_thing)
            except TypeError:
                try_quick = False
        if try_quick:
            r = np.zeros(coord.shape, dtype=np.bool)
            if coord.cell(i) == self._coord_thing:
                r[i] = True
        else:
            r = np.array([call_func(cell) for cell in coord.cells()])
        if dims:
            cube_cim[dims[0]] = r
        elif not all(r):
            cube_cim.all_false()
        return cube_cim


class _ColumnIndexManager(object):
    """
    A class to represent column aligned slices which can be operated on
    using ``&``, ``|`` or ``^``.

    ::

        # 4 Dimensional slices
        import numpy as np
        cim = _ColumnIndexManager(4)
        cim[1] = np.array([3, 4, 5]) > 3
        print(cim.as_slice())

    """
    def __init__(self, ndims):
        """
        A _ColumnIndexManager is always created to span the given
        number of dimensions.

        """
        self._column_arrays = [True] * ndims
        self.ndims = ndims

    def __and__(self, other):
        return self._bitwise_operator(other, operator.__and__)

    def __or__(self, other):
        return self._bitwise_operator(other, operator.__or__)

    def __xor__(self, other):
        return self._bitwise_operator(other, operator.__xor__)

    def _bitwise_operator(self, other, operator):
        if not isinstance(other, _ColumnIndexManager):
            return NotImplemented

        if self.ndims != other.ndims:
            raise ValueError('Cannot do %s for %r and %r as they have a '
                             'different number of dimensions.' % operator)
        r = _ColumnIndexManager(self.ndims)
        # iterate over each dimension an combine appropriately
        for i, (lhs, rhs) in enumerate(zip(self, other)):
            r[i] = operator(lhs, rhs)
        return r

    def all_false(self):
        """Turn all slices into False."""
        for i in range(self.ndims):
            self[i] = False

    def __getitem__(self, key):
        return self._column_arrays[key]

    def __setitem__(self, key, value):
        is_vector = isinstance(value, np.ndarray) and value.ndim == 1
        if is_vector or isinstance(value, bool):
            self._column_arrays[key] = value
        else:
            raise TypeError('Expecting value to be a 1 dimensional numpy array'
                            ', or a boolean. Got %s' % (type(value)))

    def as_slice(self):
        """
        Turns a _ColumnIndexManager into a tuple which can be used in an
        indexing operation.

        If no index is possible, None will be returned.
        """
        result = [None] * self.ndims

        for dim, dimension_array in enumerate(self):
            # If dimension_array has not been set, span the entire dimension
            if isinstance(dimension_array, np.ndarray):
                where_true = np.where(dimension_array)[0]
                # If the array had no True values in it, then the dimension
                # is equivalent to False
                if len(where_true) == 0:
                    result = None
                    break

                # If there was exactly one match, the key should be an integer
                if where_true.shape == (1,):
                    result[dim] = where_true[0]
                else:
                    # Finally, we can either provide a slice if possible,
                    # or a tuple of indices which match. In order to determine
                    # if we can provide a slice, calculate the deltas between
                    # the indices and check if they are the same.
                    delta = np.diff(where_true, axis=0)
                    # if the diff is consistent we can create a slice object
                    if all(delta[0] == delta):
                        result[dim] = slice(where_true[0], where_true[-1] + 1,
                                            delta[0])
                    else:
                        # otherwise, key is a tuple
                        result[dim] = tuple(where_true)

            # Handle the case where dimension_array is a boolean
            elif dimension_array:
                result[dim] = slice(None, None)
            else:
                result = None
                break

        if result is None:
            return result
        else:
            return tuple(result)


def list_of_constraints(constraints):
    """
    Turns the given constraints into a list of valid constraints
    using :func:`as_constraint`.

    """
    if not isinstance(constraints, (list, tuple)):
        constraints = [constraints]

    return [as_constraint(constraint) for constraint in constraints]


def as_constraint(thing):
    """
    Casts an object into a cube constraint where possible, otherwise
    a TypeError will be raised.

    If the given object is already a valid constraint then the given object
    will be returned, else a TypeError will be raised.

    """
    if isinstance(thing, Constraint):
        return thing
    elif thing is None:
        return Constraint()
    elif isinstance(thing, basestring):
        return Constraint(thing)
    else:
        raise TypeError('%r cannot be cast to a constraint.' % thing)


class AttributeConstraint(Constraint):
    """Provides a simple Cube-attribute based :class:`Constraint`."""
    def __init__(self, **attributes):
        """
        Example usage::

            iris.AttributeConstraint(STASH='m01s16i004')

            iris.AttributeConstraint(
                STASH=lambda stash: stash.endswith('i005'))

        .. note:: Attribute constraint names are case sensitive.

        """
        self._attributes = attributes
        Constraint.__init__(self, cube_func=self._cube_func)

    def _cube_func(self, cube):
        match = True
        for name, value in self._attributes.iteritems():
            if name in cube.attributes:
                cube_attr = cube.attributes.get(name)
                # if we have a callable, then call it with the value,
                # otherwise, assert equality
                if callable(value):
                    if not value(cube_attr):
                        match = False
                        break
                else:
                    if cube_attr != value:
                        match = False
                        break
            else:
                match = False
                break
        return match

    def __repr__(self):
        return 'AttributeConstraint(%r)' % self._attributes
