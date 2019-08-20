# (C) British Crown Copyright 2010 - 2019, Met Office
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
Basic mathematical and statistical operations.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

import inspect
import math
import operator
import six
import warnings

import cf_units
import numpy as np
from numpy import ma

import iris.analysis
import iris.coords
import iris.cube
import iris.exceptions
import iris.util

import dask.array as da
from dask.array.core import broadcast_shapes


_output_dtype_cache = {}


def _output_dtype(op, first_dtype, second_dtype=None, in_place=False):
    """
    Get the numpy dtype corresponding to the result of applying a unary or
    binary operation to arguments of specified dtype.

    Args:

    * op:
        A unary or binary operator which can be applied to array-like objects.
    * first_dtype:
        The dtype of the first or only argument to the operator.

    Kwargs:

    * second_dtype:
        The dtype of the second argument to the operator.

    * in_place:
        Whether the operation is to be performed in place.

    Returns:
        An instance of :class:`numpy.dtype`

    .. note::

        The function always returns the dtype which would result if the
        operation were successful, even if the operation could fail due to
        casting restrictions for in place operations.

    """
    if in_place:
        # Always return the first dtype, even if the operation would fail due
        # to failure to cast the result.
        result = first_dtype
    else:
        operand_dtypes = (first_dtype, second_dtype) \
            if second_dtype is not None \
            else (first_dtype,)
        key = (op, operand_dtypes)
        result = _output_dtype_cache.get(key, None)
        if result is None:
            arrays = [np.array([1], dtype=dtype) for dtype in operand_dtypes]
            result = op(*arrays).dtype
            _output_dtype_cache[key] = result
    return result


def _get_dtype(operand):
    """
    Get the numpy dtype corresponding to the numeric data in the object
    provided.

    Args:

    * operand:
        An instance of :class:`iris.cube.Cube` or :class:`iris.coords.Coord`,
        or a number or :class:`numpy.ndarray`.

    Returns:
        An instance of :class:`numpy.dtype`

    """
    return np.min_scalar_type(operand) if np.isscalar(operand) \
        else operand.dtype


def abs(cube, in_place=False):
    """
    Calculate the absolute values of the data in the Cube provided.

    Args:

    * cube:
        An instance of :class:`iris.cube.Cube`.

    Kwargs:

    * in_place:
        Whether to create a new Cube, or alter the given "cube".

    Returns:
        An instance of :class:`iris.cube.Cube`.

    """
    _assert_is_cube(cube)
    new_dtype = _output_dtype(np.abs, cube.dtype, in_place=in_place)
    op = da.absolute if cube.has_lazy_data() else np.abs
    return _math_op_common(cube, op, cube.units, new_dtype, in_place=in_place)


def intersection_of_cubes(cube, other_cube):
    """
    Return the two Cubes of intersection given two Cubes.

    .. note:: The intersection of cubes function will ignore all single valued
        coordinates in checking the intersection.

    Args:

    * cube:
        An instance of :class:`iris.cube.Cube`.
    * other_cube:
        An instance of :class:`iris.cube.Cube`.

    Returns:
        A pair of :class:`iris.cube.Cube` instances in a tuple corresponding
        to the original cubes restricted to their intersection.

    """
    # Take references of the original cubes (which will be copied when
    # slicing later).
    new_cube_self = cube
    new_cube_other = other_cube

    # This routine has not been written to cope with multi-dimensional
    # coordinates.
    for coord in cube.coords() + other_cube.coords():
        if coord.ndim != 1:
            raise iris.exceptions.CoordinateMultiDimError(coord)

    coord_comp = iris.analysis.coord_comparison(cube, other_cube)

    if coord_comp['ungroupable_and_dimensioned']:
        raise ValueError('Cubes do not share all coordinates in common, '
                         'cannot intersect.')

    # cubes must have matching coordinates
    for coord in cube.coords():
        other_coord = other_cube.coord(coord)

        # Only intersect coordinates which are different, single values
        # coordinates may differ.
        if coord.shape[0] > 1 and coord != other_coord:
            intersected_coord = coord.intersect(other_coord)
            new_cube_self = new_cube_self.subset(intersected_coord)
            new_cube_other = new_cube_other.subset(intersected_coord)

    return new_cube_self, new_cube_other


def _assert_is_cube(cube):
    if not isinstance(cube, iris.cube.Cube):
        raise TypeError('The "cube" argument must be an instance of '
                        'iris.cube.Cube.')


def _assert_compatible(cube, other):
    """
    Checks to see if cube.data and another array can be broadcast to
    the same shape.

    """
    try:
        new_shape = broadcast_shapes(cube.shape, other.shape)
    except ValueError as err:
        # re-raise
        raise ValueError("The array was not broadcastable to the cube's data "
                         "shape. The error message when "
                         "broadcasting:\n{}\nThe cube's shape was {} and the "
                         "array's shape was {}".format(err, cube.shape,
                                                       other.shape))

    if cube.shape != new_shape:
        raise ValueError("The array operation would increase the size or "
                         "dimensionality of the cube. The new cube's data "
                         "would have had to become: {}".format(
                             new_shape))


def _assert_matching_units(cube, other, operation_name):
    """
    Check that the units of the cube and the other item are the same, or if
    the other does not have a unit, skip this test
    """
    if cube.units != getattr(other, 'units', cube.units):
        msg = 'Cannot use {!r} with differing units ({} & {})'.format(
            operation_name, cube.units, other.units)
        raise iris.exceptions.NotYetImplementedError(msg)


def add(cube, other, dim=None, in_place=False):
    """
    Calculate the sum of two cubes, or the sum of a cube and a
    coordinate or scalar value.

    When summing two cubes, they must both have the same coordinate
    systems & data resolution.

    When adding a coordinate to a cube, they must both share the same
    number of elements along a shared axis.

    Args:

    * cube:
        An instance of :class:`iris.cube.Cube`.
    * other:
        An instance of :class:`iris.cube.Cube` or :class:`iris.coords.Coord`,
        or a number or :class:`numpy.ndarray`.

    Kwargs:

    * dim:
        If supplying a coord with no match on the cube, you must supply
        the dimension to process.
    * in_place:
        Whether to create a new Cube, or alter the given "cube".

    Returns:
        An instance of :class:`iris.cube.Cube`.

    """
    _assert_is_cube(cube)
    new_dtype = _output_dtype(operator.add, cube.dtype, _get_dtype(other),
                              in_place=in_place)
    if in_place:
        _inplace_common_checks(cube, other, 'addition')
        op = operator.iadd
    else:
        op = operator.add
    return _add_subtract_common(op, 'add', cube, other, new_dtype, dim=dim,
                                in_place=in_place)


def subtract(cube, other, dim=None, in_place=False):
    """
    Calculate the difference between two cubes, or the difference between
    a cube and a coordinate or scalar value.

    When subtracting two cubes, they must both have the same coordinate
    systems & data resolution.

    When subtracting a coordinate to a cube, they must both share the
    same number of elements along a shared axis.

    Args:

    * cube:
        An instance of :class:`iris.cube.Cube`.
    * other:
        An instance of :class:`iris.cube.Cube` or :class:`iris.coords.Coord`,
        or a number or :class:`numpy.ndarray`.

    Kwargs:

    * dim:
        If supplying a coord with no match on the cube, you must supply
        the dimension to process.
    * in_place:
        Whether to create a new Cube, or alter the given "cube".

    Returns:
        An instance of :class:`iris.cube.Cube`.

    """
    _assert_is_cube(cube)
    new_dtype = _output_dtype(operator.sub, cube.dtype, _get_dtype(other),
                              in_place=in_place)
    if in_place:
        _inplace_common_checks(cube, other, 'subtraction')
        op = operator.isub
    else:
        op = operator.sub
    return _add_subtract_common(op, 'subtract', cube, other, new_dtype,
                                dim=dim, in_place=in_place)


def _add_subtract_common(operation_function, operation_name, cube, other,
                         new_dtype, dim=None, in_place=False):
    """
    Function which shares common code between addition and subtraction
    of cubes.

    operation_function   - function which does the operation
                           (e.g. numpy.subtract)
    operation_name       - the public name of the operation (e.g. 'divide')
    cube                 - the cube whose data is used as the first argument
                           to `operation_function`
    other                - the cube, coord, ndarray or number whose data is
                           used as the second argument
    new_dtype            - the expected dtype of the output. Used in the
                           case of scalar masked arrays
    dim                  - dimension along which to apply `other` if it's a
                           coordinate that is not found in `cube`
    in_place             - whether or not to apply the operation in place to
                           `cube` and `cube.data`

    """
    _assert_is_cube(cube)
    _assert_matching_units(cube, other, operation_name)

    if isinstance(other, iris.cube.Cube):
        # get a coordinate comparison of this cube and the cube to do the
        # operation with
        coord_comp = iris.analysis.coord_comparison(cube, other)

        bad_coord_grps = (coord_comp['ungroupable_and_dimensioned'] +
                          coord_comp['resamplable'])
        if bad_coord_grps:
            raise ValueError('This operation cannot be performed as there are '
                             'differing coordinates (%s) remaining '
                             'which cannot be ignored.'
                             % ', '.join({coord_grp.name() for coord_grp
                                          in bad_coord_grps}))
    else:
        coord_comp = None

    new_cube = _binary_op_common(operation_function, operation_name, cube,
                                 other, cube.units, new_dtype=new_dtype,
                                 dim=dim, in_place=in_place)

    if coord_comp:
        # If a coordinate is to be ignored - remove it
        ignore = filter(None, [coord_grp[0] for coord_grp
                               in coord_comp['ignorable']])
        for coord in ignore:
            new_cube.remove_coord(coord)

    return new_cube


def multiply(cube, other, dim=None, in_place=False):
    """
    Calculate the product of a cube and another cube or coordinate.

    Args:

    * cube:
        An instance of :class:`iris.cube.Cube`.
    * other:
        An instance of :class:`iris.cube.Cube` or :class:`iris.coords.Coord`,
        or a number or :class:`numpy.ndarray`.

    Kwargs:

    * dim:
        If supplying a coord with no match on the cube, you must supply
        the dimension to process.

    Returns:
        An instance of :class:`iris.cube.Cube`.

    """
    _assert_is_cube(cube)
    new_dtype = _output_dtype(operator.mul, cube.dtype, _get_dtype(other),
                              in_place=in_place)
    other_unit = getattr(other, 'units', '1')
    new_unit = cube.units * other_unit
    if in_place:
        _inplace_common_checks(cube, other, 'multiplication')
        op = operator.imul
    else:
        op = operator.mul

    if isinstance(other, iris.cube.Cube):
        # get a coordinate comparison of this cube and the cube to do the
        # operation with
        coord_comp = iris.analysis.coord_comparison(cube, other)
        bad_coord_grps = (coord_comp['ungroupable_and_dimensioned'] +
                          coord_comp['resamplable'])
        if bad_coord_grps:
            raise ValueError('This operation cannot be performed as there are '
                             'differing coordinates (%s) remaining '
                             'which cannot be ignored.'
                             % ', '.join({coord_grp.name() for coord_grp
                                          in bad_coord_grps}))
    else:
        coord_comp = None

    new_cube = _binary_op_common(op, 'multiply', cube, other, new_unit,
                                 new_dtype=new_dtype, dim=dim,
                                 in_place=in_place)

    if coord_comp:
        # If a coordinate is to be ignored - remove it
        ignore = filter(None, [coord_grp[0] for coord_grp
                        in coord_comp['ignorable']])
        for coord in ignore:
            new_cube.remove_coord(coord)

    return new_cube


def _inplace_common_checks(cube, other, math_op):
    """
    Check whether an inplace math operation can take place between `cube` and
    `other`. It cannot if `cube` has integer data and `other` has float data
    as the operation will always produce float data that cannot be 'safely'
    cast back to the integer data of `cube`.

    """
    other_dtype = _get_dtype(other)
    if not np.can_cast(other_dtype, cube.dtype, 'same_kind'):
            aemsg = ('Cannot perform inplace {} between {!r} '
                     'with {} data and {!r} with {} data.')
            raise ArithmeticError(
                aemsg.format(math_op, cube, cube.dtype, other, other_dtype))


def divide(cube, other, dim=None, in_place=False):
    """
    Calculate the division of a cube by a cube or coordinate.

    Args:

    * cube:
        An instance of :class:`iris.cube.Cube`.
    * other:
        An instance of :class:`iris.cube.Cube` or :class:`iris.coords.Coord`,
        or a number or :class:`numpy.ndarray`.

    Kwargs:

    * dim:
        If supplying a coord with no match on the cube, you must supply
        the dimension to process.

    Returns:
        An instance of :class:`iris.cube.Cube`.

    """
    _assert_is_cube(cube)
    new_dtype = _output_dtype(operator.truediv,
                              cube.dtype, _get_dtype(other), in_place=in_place)
    other_unit = getattr(other, 'units', '1')
    new_unit = cube.units / other_unit
    if in_place:
        if cube.dtype.kind in 'iu':
            # Cannot coerce float result from inplace division back to int.
            aemsg = ('Cannot perform inplace division of cube {!r} '
                     'with integer data.')
            raise ArithmeticError(aemsg)
        op = operator.itruediv
    else:
        op = operator.truediv

    if isinstance(other, iris.cube.Cube):
        # get a coordinate comparison of this cube and the cube to do the
        # operation with
        coord_comp = iris.analysis.coord_comparison(cube, other)
        bad_coord_grps = (coord_comp['ungroupable_and_dimensioned'] +
                          coord_comp['resamplable'])
        if bad_coord_grps:
            raise ValueError('This operation cannot be performed as there are '
                             'differing coordinates (%s) remaining '
                             'which cannot be ignored.'
                             % ', '.join({coord_grp.name() for coord_grp
                                          in bad_coord_grps}))
    else:
        coord_comp = None

    new_cube = _binary_op_common(op, 'divide', cube, other, new_unit,
                                 new_dtype=new_dtype, dim=dim,
                                 in_place=in_place)

    if coord_comp:
        # If a coordinate is to be ignored - remove it
        ignore = filter(None, [coord_grp[0] for coord_grp
                        in coord_comp['ignorable']])
        for coord in ignore:
            new_cube.remove_coord(coord)

    return new_cube


def exponentiate(cube, exponent, in_place=False):
    """
    Returns the result of the given cube to the power of a scalar.

    Args:

    * cube:
        An instance of :class:`iris.cube.Cube`.
    * exponent:
        The integer or floating point exponent.

        .. note:: When applied to the cube's unit, the exponent must
            result in a unit that can be described using only integer
            powers of the basic units.

            e.g. Unit('meter^-2 kilogram second^-1')

    Kwargs:

    * in_place:
        Whether to create a new Cube, or alter the given "cube".

    Returns:
        An instance of :class:`iris.cube.Cube`.

    """
    _assert_is_cube(cube)
    new_dtype = _output_dtype(operator.pow, cube.dtype, _get_dtype(exponent),
                              in_place=in_place)
    if cube.has_lazy_data():
        def power(data):
            return operator.pow(data, exponent)
    else:
        def power(data, out=None):
            return np.power(data, exponent, out)

    return _math_op_common(cube, power, cube.units ** exponent, new_dtype,
                           in_place=in_place)


def exp(cube, in_place=False):
    """
    Calculate the exponential (exp(x)) of the cube.

    Args:

    * cube:
        An instance of :class:`iris.cube.Cube`.

    .. note::

        Taking an exponential will return a cube with dimensionless units.

    Kwargs:

    * in_place:
        Whether to create a new Cube, or alter the given "cube".

    Returns:
        An instance of :class:`iris.cube.Cube`.

    """
    _assert_is_cube(cube)
    new_dtype = _output_dtype(np.exp, cube.dtype, in_place=in_place)
    op = da.exp if cube.has_lazy_data() else np.exp
    return _math_op_common(cube, op, cf_units.Unit('1'), new_dtype,
                           in_place=in_place)


def log(cube, in_place=False):
    """
    Calculate the natural logarithm (base-e logarithm) of the cube.

    Args:

    * cube:
        An instance of :class:`iris.cube.Cube`.

    Kwargs:

    * in_place:
        Whether to create a new Cube, or alter the given "cube".

    Returns:
        An instance of :class:`iris.cube.Cube`.

    """
    _assert_is_cube(cube)
    new_dtype = _output_dtype(np.log, cube.dtype, in_place=in_place)
    op = da.log if cube.has_lazy_data() else np.log
    return _math_op_common(cube, op, cube.units.log(math.e), new_dtype,
                           in_place=in_place)


def log2(cube, in_place=False):
    """
    Calculate the base-2 logarithm of the cube.

    Args:

    * cube:
        An instance of :class:`iris.cube.Cube`.

    Kwargs:lib/iris/tests/unit/analysis/maths/test_subtract.py

    * in_place:
        Whether to create a new Cube, or alter the given "cube".

    Returns:
        An instance of :class:`iris.cube.Cube`.

    """
    _assert_is_cube(cube)
    new_dtype = _output_dtype(np.log2, cube.dtype, in_place=in_place)
    op = da.log2 if cube.has_lazy_data() else np.log2
    return _math_op_common(cube, op, cube.units.log(2), new_dtype,
                           in_place=in_place)


def log10(cube, in_place=False):
    """
    Calculate the base-10 logarithm of the cube.

    Args:

    * cube:
        An instance of :class:`iris.cube.Cube`.

    Kwargs:

    * in_place:
        Whether to create a new Cube, or alter the given "cube".

    Returns:
        An instance of :class:`iris.cube.Cube`.

    """
    _assert_is_cube(cube)
    new_dtype = _output_dtype(np.log10, cube.dtype, in_place=in_place)
    op = da.log10 if cube.has_lazy_data() else np.log10
    return _math_op_common(cube, op, cube.units.log(10), new_dtype,
                           in_place=in_place)


def apply_ufunc(ufunc, cube, other_cube=None, new_unit=None, new_name=None,
                in_place=False):
    """
    Apply a `numpy universal function
    <http://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_ to a cube
    or pair of cubes.

    .. note:: Many of the numpy.ufunc have been implemented explicitly in Iris
        e.g. :func:`numpy.abs`, :func:`numpy.add` are implemented in
        :func:`iris.analysis.maths.abs`, :func:`iris.analysis.maths.add`.
        It is usually preferable to use these functions rather than
        :func:`iris.analysis.maths.apply_ufunc` where possible.

    Args:

    * ufunc:
        An instance of :func:`numpy.ufunc` e.g. :func:`numpy.sin`,
        :func:`numpy.mod`.

    * cube:
        An instance of :class:`iris.cube.Cube`.

    Kwargs:

    * other_cube:
        An instance of :class:`iris.cube.Cube` to be given as the second
        argument to :func:`numpy.ufunc`.

    * new_unit:
        Unit for the resulting Cube.

    * new_name:
        Name for the resulting Cube.

    * in_place:
        Whether to create a new Cube, or alter the given "cube".

    Returns:
        An instance of :class:`iris.cube.Cube`.

    Example::

        cube = apply_ufunc(numpy.sin, cube, in_place=True)

    """

    if not isinstance(ufunc, np.ufunc):
        name = getattr(ufunc, '__name__', 'function passed to apply_ufunc')

        raise TypeError('{} is not recognised (it is not an instance of '
                        'numpy.ufunc)'.format(name))

    if ufunc.nout != 1:
        raise ValueError('{} returns {} objects, apply_ufunc currently '
                         'only supports ufunc functions returning a single '
                         'object.'.format(ufunc.__name__, ufunc.nout))

    if ufunc.nin == 2:
        if other_cube is None:
            raise ValueError('{} requires two arguments, so other_cube '
                             'must also be passed to apply_ufunc'.format(
                                 ufunc.__name__))

        _assert_is_cube(other_cube)
        new_dtype = _output_dtype(ufunc, cube.dtype, other_cube.dtype,
                                  in_place=in_place)

        new_cube = _binary_op_common(ufunc, ufunc.__name__, cube, other_cube,
                                     new_unit, new_dtype=new_dtype,
                                     in_place=in_place)

    elif ufunc.nin == 1:
        new_dtype = _output_dtype(ufunc, cube.dtype, in_place=in_place)

        new_cube = _math_op_common(cube, ufunc, new_unit, new_dtype,
                                   in_place=in_place)

    else:
        raise ValueError(ufunc.__name__ + ".nin should be 1 or 2.")

    new_cube.rename(new_name)

    return new_cube


def _binary_op_common(operation_function, operation_name, cube, other,
                      new_unit, new_dtype=None, dim=None, in_place=False):
    """
    Function which shares common code between binary operations.

    operation_function   - function which does the operation
                           (e.g. numpy.divide)
    operation_name       - the public name of the operation (e.g. 'divide')
    cube                 - the cube whose data is used as the first argument
                           to `operation_function`
    other                - the cube, coord, ndarray or number whose data is
                           used as the second argument
    new_dtype            - the expected dtype of the output. Used in the
                           case of scalar masked arrays
    new_unit             - unit for the resulting quantity
    dim                  - dimension along which to apply `other` if it's a
                           coordinate that is not found in `cube`
    in_place             - whether or not to apply the operation in place to
                           `cube` and `cube.data`
    """
    _assert_is_cube(cube)
    if isinstance(other, iris.coords.Coord):
        other = _broadcast_cube_coord_data(cube, other, operation_name, dim)
    elif isinstance(other, iris.cube.Cube):
        try:
            broadcast_shapes(cube.shape, other.shape)
        except ValueError:
            other = iris.util.as_compatible_shape(other, cube)
        other = other.core_data()
    else:
        other = np.asanyarray(other)

    # don't worry about checking for other data types (such as scalars or
    # np.ndarrays) because _assert_compatible validates that they are broadcast
    # compatible with cube.data
    _assert_compatible(cube, other)

    def unary_func(x):
        ret = operation_function(x, other)
        if ret is NotImplemented:
            # explicitly raise the TypeError, so it gets raised even if, for
            # example, `iris.analysis.maths.multiply(cube, other)` is called
            # directly instead of `cube * other`
            raise TypeError('cannot %s %r and %r objects' %
                            (operation_function.__name__, type(x).__name__,
                             type(other).__name__))
        return ret
    return _math_op_common(cube, unary_func, new_unit, new_dtype, in_place)


def _broadcast_cube_coord_data(cube, other, operation_name, dim=None):
    # What dimension are we processing?
    data_dimension = None
    if dim is not None:
        # Ensure the given dim matches the coord
        if other in cube.coords() and cube.coord_dims(other) != [dim]:
            raise ValueError("dim provided does not match dim found for coord")
        data_dimension = dim
    else:
        # Try and get a coord dim
        if other.shape != (1,):
            try:
                coord_dims = cube.coord_dims(other)
                data_dimension = coord_dims[0] if coord_dims else None
            except iris.exceptions.CoordinateNotFoundError:
                raise ValueError("Could not determine dimension for %s. "
                                 "Use %s(cube, coord, dim=dim)"
                                 % (operation_name, operation_name))

    if other.ndim != 1:
        raise iris.exceptions.CoordinateMultiDimError(other)

    if other.has_bounds():
        warnings.warn('Using {!r} with a bounded coordinate is not well '
                      'defined; ignoring bounds.'.format(operation_name))

    points = other.points

    # If the `data_dimension` is defined then shape the provided points for
    # proper array broadcasting
    if data_dimension is not None:
        points_shape = [1] * cube.ndim
        points_shape[data_dimension] = -1
        points = points.reshape(points_shape)

    return points


def _math_op_common(cube, operation_function, new_unit, new_dtype=None,
                    in_place=False):
    _assert_is_cube(cube)

    if in_place:
        new_cube = cube
        if cube.has_lazy_data():
            new_cube.data = operation_function(cube.lazy_data())
        else:
            try:
                operation_function(cube.data, out=cube.data)
            except TypeError:
                # Non ufunc function
                operation_function(cube.data)
    else:
        new_cube = cube.copy(data=operation_function(cube.core_data()))

    # If the result of the operation is scalar and masked, we need to fix up
    # the dtype
    if new_dtype is not None \
            and not new_cube.has_lazy_data() \
            and new_cube.data.shape == () \
            and ma.is_masked(new_cube.data):
        new_cube.data = ma.masked_array(0, 1, dtype=new_dtype)

    iris.analysis.clear_phenomenon_identity(new_cube)
    new_cube.units = new_unit
    return new_cube


class IFunc(object):
    """
    :class:`IFunc` class for functions that can be applied to an iris cube.
    """
    def __init__(self, data_func, units_func):
        """
        Create an ifunc from a data function and units function.

        Args:

        * data_func:

            Function to be applied to one or two data arrays, which
            are given as positional arguments. Should return another
            data array, with the same shape as the first array.

            Can also have keyword arguments.

        * units_func:

            Function to calculate the unit of the resulting cube.
            Should take the cube(s) as input and return
            an instance of :class:`cf_units.Unit`.

        Returns:
            An ifunc.

        **Example usage 1** Using an existing numpy ufunc, such as numpy.sin
        for the data function and a simple lambda function for the units
        function::

            sine_ifunc = iris.analysis.maths.IFunc(
                numpy.sin, lambda cube: cf_units.Unit('1'))
            sine_cube = sine_ifunc(cube)

        **Example usage 2** Define a function for the data arrays of two cubes
        and define a units function that checks the units of the cubes
        for consistency, before giving the resulting cube the same units
        as the first cube::

            def ws_data_func(u_data, v_data):
                return numpy.sqrt( u_data**2 + v_data**2 )

            def ws_units_func(u_cube, v_cube):
                if u_cube.units != getattr(v_cube, 'units', u_cube.units):
                    raise ValueError("units do not match")
                return u_cube.units

            ws_ifunc = iris.analysis.maths.IFunc(ws_data_func, ws_units_func)
            ws_cube = ws_ifunc(u_cube, v_cube, new_name='wind speed')

        **Example usage 3** Using a data function that allows a keyword
        argument::

            cs_ifunc = iris.analysis.maths.IFunc(numpy.cumsum,
                lambda a: a.units)
            cs_cube = cs_ifunc(cube, axis=1)
        """

        if hasattr(data_func, 'nin'):
            self.nin = data_func.nin
        else:
            if six.PY2:
                (args, _, _, defaults) = inspect.getargspec(data_func)
                self.nin = len(args) - (
                    len(defaults) if defaults is not None else 0)
            else:
                sig = inspect.signature(data_func)
                args = [param for param in sig.parameters.values()
                        if (param.kind != param.KEYWORD_ONLY and
                            param.default is param.empty)]
                self.nin = len(args)

        if self.nin not in [1, 2]:
            msg = ('{} requires {} input data arrays, the IFunc class '
                   'currently only supports functions requiring 1 or two '
                   'data arrays as input.')
            raise ValueError(msg.format(data_func.__name__, self.nin))

        if hasattr(data_func, 'nout'):
            if data_func.nout != 1:
                msg = ('{} returns {} objects, the IFunc class currently '
                       'only supports functions returning a single object.')
                raise ValueError(msg.format(data_func.__name__,
                                            data_func.nout))

        self.data_func = data_func

        self.units_func = units_func

    def __repr__(self):
        return 'iris.analysis.maths.IFunc({}, {})'.format(
            self.data_func.__name__, self.units_func.__name__)

    def __str__(self):
        return ('IFunc constructed from the data function {} '
                'and the units function {}'.format(
                    self.data_func.__name__, self.units_func.__name__))

    def __call__(self, cube, other=None, dim=None, in_place=False,
                 new_name=None, **kwargs_data_func):
        """
        Applies the ifunc to the cube(s).

        Args:

        * cube
            An instance of :class:`iris.cube.Cube`, whose data is used
            as the first argument to the data function.

        Kwargs:

        * other
            A cube, coord, ndarray or number whose data is used as the
            second argument to the data function.

        * new_name:
            Name for the resulting Cube.

        * in_place:
            Whether to create a new Cube, or alter the given "cube".

        * dim:
            Dimension along which to apply `other` if it's a coordinate that is
            not found in `cube`

        * **kwargs_data_func:
            Keyword arguments that get passed on to the data_func.

        Returns:
            An instance of :class:`iris.cube.Cube`.

        """
        _assert_is_cube(cube)

        def wrap_data_func(*args, **kwargs):
            kwargs_combined = dict(kwargs_data_func, **kwargs)

            return self.data_func(*args, **kwargs_combined)

        if self.nin == 2:
            if other is None:
                raise ValueError(self.data_func.__name__ +
                                 ' requires two arguments')

            new_unit = self.units_func(cube, other)

            new_cube = _binary_op_common(wrap_data_func,
                                         self.data_func.__name__, cube, other,
                                         new_unit, dim=dim,
                                         in_place=in_place)

        elif self.nin == 1:
            if other is not None:
                raise ValueError(self.data_func.__name__ +
                                 ' requires one argument')

            new_unit = self.units_func(cube)

            new_cube = _math_op_common(cube, wrap_data_func, new_unit,
                                       in_place=in_place)

        else:
            raise ValueError('self.nin should be 1 or 2.')

        if new_name is not None:
            new_cube.rename(new_name)

        return new_cube
