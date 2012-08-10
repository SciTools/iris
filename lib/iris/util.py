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
Miscellaneous utility functions.

"""

import abc
import collections
import inspect
import os
import tempfile
import time

import numpy


def delta(ndarray, dimension, circular=False):
    """
    Calculates the difference between values along a given dimension.
    
    Args:

    * ndarray:
        The array over which to do the difference.
        
    * dimension:
        The dimension over which to do the difference on ndarray.
    
    * circular:
        If not False then return n results in the requested dimension
        with the delta between the last and first element included in
        the result otherwise the result will be of length n-1 (where n
        is the length of ndarray in the given dimension's direction)

        If circular is numeric then the value of circular will be added
        to the last element of the given dimension if the last element
        is negative, otherwise the value of circular will be subtracted
        from the last element.
                 
        The example below illustrates the process::

            original array              -180, -90,  0,    90
            delta (with circular=360):    90,  90, 90, -270+360

    .. note::

        The difference algorithm implemented is forward difference:
        
            >>> import numpy
            >>> import iris.util
            >>> original = numpy.array([-180, -90, 0, 90])
            >>> iris.util.delta(original, 0)
            array([90, 90, 90])
            >>> iris.util.delta(original, 0, circular=360)
            array([90, 90, 90, 90])
        
    """
    if circular is not False:
        _delta = numpy.roll(ndarray, -1, axis=dimension)
        last_element = [slice(None, None)] * ndarray.ndim
        last_element[dimension] = slice(-1, None)
        
        if not isinstance(circular, bool): 
            result = numpy.where(ndarray[last_element] >= _delta[last_element])[0]
            _delta[last_element] -= circular
            _delta[last_element][result] += 2*circular
                    
        numpy.subtract(_delta, ndarray, _delta)         
    else:
        _delta = numpy.diff(ndarray, axis=dimension)
    
    return _delta


def guess_coord_axis(coord):
    """
    Returns a "best guess" axis name of the coordinate.
    
    Heuristic categoration of the coordinate into either label
    'T', 'Z', 'Y', 'X' or None.
    
    Args:

    * coord:
        The :class:`iris.coords.Coord`.

    Returns:
        'T', 'Z', 'Y', 'X', or None.
    
    """
    axis = None
    name = coord.name().lower()

    if coord.standard_name in ('longitude', 'grid_longitude', 'projection_x_coordinate'):
        axis = 'X'
    elif coord.standard_name in ('latitude', 'grid_latitude', 'projection_y_coordinate'):
        axis = 'Y'
    elif 'height' in name or 'depth' in name or \
            coord.units.convertible('hPa') or \
            coord.attributes.get('positive') in ('up', 'down'):
        axis = 'Z'
    elif coord.units.time_reference:    
        axis = 'T'

    if coord.name().upper() == "X":
        axis = "X"
    if coord.name().upper() == "Y":
        axis = "Y"
    if coord.name().upper() == "Z":
        axis = "Z"

    return axis


def rolling_window(a, window=1, step=1, axis=-1):
    """
    Make an ndarray with a rolling window of the last dimension

    Args:
    
    * a : array_like
        Array to add rolling window to
        
    Kwargs:
    
    * window : int
        Size of rolling window
    * step : int
        Size of step between rolling windows
    * axis : int
        Axis to take the rolling window over

    Returns:
        
        Array that is a view of the original array with an added dimension
        of the size of the given window at axis + 1.

    Examples::
        
        >>> x=np.arange(10).reshape((2,5))
        >>> rolling_window(x, 3)
        array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
               [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])
    
    Calculate rolling mean of last dimension::
     
        >>> np.mean(rolling_window(x, 3), -1)
        array([[ 1.,  2.,  3.],
               [ 6.,  7.,  8.]])

    """
    # NOTE: The implementation of this function originates from 
    # https://github.com/numpy/numpy/pull/31#issuecomment-1304851 04/08/2011
    if window < 1:  
        raise ValueError("`window` must be at least 1.")
    if window > a.shape[axis]:
        raise ValueError("`window` is too long.")
    if step < 1:
        raise ValueError("`step` must be at least 1.")
    axis = axis % a.ndim
    num_windows = (a.shape[axis] - window + step) / step
    shape = a.shape[:axis] + (num_windows, window) + a.shape[axis + 1:]
    strides = a.strides[:axis] + (step * a.strides[axis], a.strides[axis]) + a.strides[axis + 1:]
    return numpy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def array_equal(array1, array2):
    """
    Returns whether two arrays have the same shape and elements.
    
    This provides the same functionality as :func:`numpy.array_equal` but with
    additional support for arrays of strings.
    
    """
    # Numpy provides an array_equal method but that does not support arrays of strings.
    if array1.ndim == 0 or array2.ndim == 0:
        eq = array1.ndim == 0 and array2.ndim == 0 and array1 == array2
    elif len(array1) == len(array2):
        eq = array1 == array2
        if not isinstance(eq, bool):
            eq = eq.all()
    else:
        eq = False
    return eq


def approx_equal(a, b, max_absolute_error=1e-10, max_relative_error=1e-10):
    """Returns whether two numbers are almost equal, allowing for the finite precision of floating point numbers."""
    # Deal with numbers close to zero
    if abs(a - b) < max_absolute_error:
        return True
    # Ensure we get consistent results if "a" and "b" are supplied in the opposite order.
    max_ab = max([a, b], key=abs)
    relative_error = abs(a - b) / max_ab
    return relative_error < max_relative_error


def between(lh, rh, lh_inclusive=True, rh_inclusive=True):
    """
    Provides a convenient way of defining a 3 element inequality such as ``a < number < b``.
    
    Arguments:
    
    * lh
        The left hand element of the inequality
    * rh
        The right hand element of the inequality
        
    Keywords:
    
    * lh_inclusive - boolean
        Affects the left hand comparison operator to use in the inequality.
        True for ``<=`` false for ``<``. Defaults to True.
    * rh_inclusive - boolean
        Same as lh_inclusive but for right hand operator.
        
        
    For example::
    
        between_3_and_6 = between(3, 6)
        for i in range(10):
           print i, between_3_and_6(i)


        between_3_and_6 = between(3, 6, rh_inclusive=False)
        for i in range(10):
           print i, between_3_and_6(i)
        
    """
    if lh_inclusive and rh_inclusive:
        return lambda c: lh <= c <= rh    
    elif lh_inclusive and not rh_inclusive:
        return lambda c: lh <= c < rh    
    elif not lh_inclusive and rh_inclusive:
        return lambda c: lh < c <= rh    
    else:
        return lambda c: lh < c < rh


def reverse(array, axes):
    """
    Reverse the array along the given axes.
    
    Args:

    * array
        The array to reverse
    * axes
        A single value or array of values of axes to reverse
       
    ::
    
        >>> import numpy
        >>> a = numpy.arange(24).reshape(2, 3, 4)
        >>> print a
        [[[ 0  1  2  3]
          [ 4  5  6  7]
          [ 8  9 10 11]]
        <BLANKLINE>
         [[12 13 14 15]
          [16 17 18 19]
          [20 21 22 23]]]
        >>> print reverse(a, 1)
        [[[ 8  9 10 11]
          [ 4  5  6  7]
          [ 0  1  2  3]]
        <BLANKLINE>
         [[20 21 22 23]
          [16 17 18 19]
          [12 13 14 15]]]
        >>> print reverse(a, [1, 2])
        [[[11 10  9  8]
          [ 7  6  5  4]
          [ 3  2  1  0]]
        <BLANKLINE>
         [[23 22 21 20]
          [19 18 17 16]
          [15 14 13 12]]]
       
    """
    index = [slice(None, None)] * array.ndim
    axes = numpy.array(axes, ndmin=1)
    if axes.ndim != 1:
        raise ValueError('Reverse was expecting a single axis or a 1d array of axes, got %r' % axes)
    if  numpy.min(axes) < 0 or numpy.max(axes) > array.ndim-1:
        raise ValueError('An axis value out of range for the number of dimensions from the '
                         'given array (%s) was received. Got: %r' % (array.ndim, axes))
    
    for axis in axes:
        index[axis] = slice(None, None, -1)
    
    return array[tuple(index)]


def monotonic(array, strict=False, return_direction=False):
    """
    Return whether the given 1d array is monotonic.
    
    Kwargs:
    
    * strict (boolean)
        Flag to enable strict monotonic checking
    * return_direction (boolean)
        Flag to change return behaviour to return (monotonic_status, direction) 
        Direction will be 1 for positive or -1 for negative. The direction is meaningless
        if the array is not monotonic.
        
    Returns:
    
    * monotonic_status (boolean)
        Whether the array was monotonic.
        
        If the return_direction flag was given then the returned value will be:
            ``(monotonic_status, direction)``
    
    """
    if array.ndim != 1 or len(array) <= 1:
        raise ValueError('The array to check must be 1 dimensional and have more than 1 element.')
    
    d = delta(array, 0)
        
    direction = numpy.sign(max(d, key=numpy.abs))
    
    # ALL step of 0
    if direction == 0 and not strict:
        direction = 1
    
    if direction == 0:
        monotonic = False
    elif (direction > 0 and not strict):
        monotonic = all(d >= 0)
    elif (direction > 0 and strict):
        monotonic = all(d > 0)
    elif (direction < 0 and not strict):
        monotonic = all(d <= 0)
    elif (direction < 0 and strict):
        monotonic = all(d < 0)
    
    if return_direction:    
        return monotonic, direction
    else:    
        return monotonic


class Linear1dExtrapolator(object):
    """
    Extension class to :class:`scipy.interpolate.interp1d` to provide linear extrapolation.
    
    See also: :mod:`scipy.interpolate`.
    
    """    
    def __init__(self, interpolator):
        """
        Given an already created :class:`scipy.interpolate.interp1d` instance, return a callable object
        which supports linear extrapolation.
        
        """
        self._interpolator = interpolator
        self.x = interpolator.x
        # Store the y values given to the interpolator. 
        self.y = interpolator.y
        """
        The y values given to the interpolator object.
        
        .. note:: These are stored with the interpolator.axis last.
        
        """
        
    def all_points_in_range(self, requested_x):
        """Given the x points, do all of the points sit inside the interpolation range."""
        test = (requested_x >= self.x[0]) & (requested_x <= self.x[-1])
        if isinstance(test, numpy.ndarray):
            test = test.all()
        return test        
            
    def __call__(self, requested_x):        
        if not self.all_points_in_range(requested_x):            
            # cast requested_x to a numpy array if it is not already.
            if not isinstance(requested_x, numpy.ndarray):
                requested_x = numpy.array(requested_x)
                        
            # we need to catch the special case of providing a single value...
            remember_that_i_was_0d = requested_x.ndim == 0
                
            requested_x = requested_x.flatten()

            gt = numpy.where(requested_x > self.x[-1])[0]
            lt = numpy.where(requested_x < self.x[0])[0]
            ok = numpy.where( (requested_x >= self.x[0]) & (requested_x <= self.x[-1]) )[0]
            
            data_shape = list(self._interpolator.y.shape)
            data_shape[-1] = len(requested_x)
            result = numpy.empty(data_shape, dtype=self._interpolator(self.x[0]).dtype)
            
            # Make a variable to represent the slice into the resultant data. (This will be updated in each of gt, lt & ok)
            interpolator_result_index = [slice(None, None)] * self._interpolator.y.ndim
            
            if len(ok) != 0:
                interpolator_result_index[-1] = ok
                
                r = self._interpolator(requested_x[ok])
                # Reshape the properly formed array to put the interpolator.axis last i.e. dims 0, 1, 2 -> 0, 2, 1 if axis = 1
                axes = range(r.ndim)
                del axes[self._interpolator.axis]
                axes.append(self._interpolator.axis)
                                
                result[interpolator_result_index] = r.transpose(axes)
                 
            if len(lt) != 0:
                interpolator_result_index[-1] = lt
                
                grad = (self.y[..., 1:2] - self.y[..., 0:1]) / (self.x[1] - self.x[0])
                result[interpolator_result_index] = self.y[..., 0:1] + (requested_x[lt] - self.x[0]) * grad
                
            if len(gt) != 0:
                interpolator_result_index[-1] = gt
                
                grad = (self.y[..., -1:] - self.y[..., -2:-1]) / (self.x[-1] - self.x[-2])
                result[interpolator_result_index] = self.y[..., -1:] + (requested_x[gt] - self.x[-1]) * grad

            axes = range(len(interpolator_result_index))
            axes.insert(self._interpolator.axis, axes.pop(axes[-1]))
            result = result.transpose(axes)

            if remember_that_i_was_0d:
                new_shape = list(result.shape)
                del new_shape[self._interpolator.axis]
                result = result.reshape(new_shape)
            
            return result
        else:
            return self._interpolator(requested_x)

def column_slices_generator(full_slice, ndims):
    """
    Given a full slice full of tuples, return a dictionary mapping old data dimensions to new and a generator which gives 
    the successive slices needed to index correctly (across columns).
    
    This routine deals with the special functionality for tuple based indexing e.g. [0, (3, 5), :, (1, 6, 8)]
    by first providing a slice which takes the non tuple slices out first i.e. [0, :, :, :]
    then subsequently iterates through each of the tuples taking out the appropriate slices
    i.e. [(3, 5), :, :] followed by [:, :, (1, 6, 8)]
    
    This method was developed as numpy does not support the direct approach of [(3, 5), : , (1, 6, 8)] for column based indexing.

    """    
    list_of_slices = []

    # Map current dimensions to new dimensions, or None
    dimension_mapping = {None: None}
    _count_current_dim = 0
    for i, i_key in enumerate(full_slice):
        if isinstance(i_key, int):
            dimension_mapping[i] = None
        else:
            dimension_mapping[i] = _count_current_dim
            _count_current_dim += 1
                
    # Get all of the dimensions for which a tuple of indices were provided (numpy.ndarrays are treated in the same way tuples in this case)
    is_tuple_style_index = lambda key: isinstance(key, tuple) or (isinstance(key, numpy.ndarray) and key.ndim == 1)
    tuple_indices = [i for i, key in enumerate(full_slice) if is_tuple_style_index(key)]

    # stg1: Take a copy of the full_slice specification, turning all tuples into a full slice
    if tuple_indices != range(len(full_slice)):
        first_slice = list(full_slice)
        for tuple_index in tuple_indices:
            first_slice[tuple_index] = slice(None, None)
        # turn first_slice back into a tuple ready for indexing
        first_slice = tuple(first_slice)
        
        list_of_slices.append(first_slice)
    
    data_ndims = max(dimension_mapping.values())
    if data_ndims is not None:
        data_ndims += 1
    
    # stg2 iterate over each of the tuples
    for tuple_index in tuple_indices:
        # Create a list with the indices to span the whole data array that we currently have
        spanning_slice_with_tuple = [slice(None, None)] * data_ndims
        # Replace the slice(None, None) with our current tuple
        spanning_slice_with_tuple[dimension_mapping[tuple_index]] = full_slice[tuple_index]
        
        # if we just have [(0, 1)] turn it into [(0, 1), ...] as this is Numpy's syntax.
        if len(spanning_slice_with_tuple) == 1:
            spanning_slice_with_tuple.append(Ellipsis)
        
        spanning_slice_with_tuple = tuple(spanning_slice_with_tuple)

        list_of_slices.append(spanning_slice_with_tuple)        
    
    # return the dimension mapping and a generator of slices
    return dimension_mapping, iter(list_of_slices)


def _build_full_slice_given_keys(keys, ndim):
    """Given the keys passed to a __getitem__ call, build an equivalent tuple of keys which span ndims."""
    # Ensure that we always have a tuple of keys    
    if not isinstance(keys, tuple):
        keys = tuple([keys])
        
    # catch the case where an extra Ellipsis has been provided which can be discarded iff len(keys)-1 == ndim
    if len(keys)-1 == ndim and Ellipsis in filter(lambda obj: not isinstance(obj, numpy.ndarray), keys):
        keys = list(keys)
        is_ellipsis = [key is Ellipsis for key in keys]
        keys.pop(is_ellipsis.index(True))
        keys = tuple(keys)
    
    # for ndim >= 1 appending a ":" to the slice specification is allowable, remove this now
    if len(keys) > ndim and ndim != 0 and keys[-1] == slice(None, None):
        keys = keys[:-1]
                
    if len(keys) > ndim:
        raise IndexError('More slices requested than dimensions. Requested %r, but there '
                             'were only %s dimensions.' % (keys, ndim))
            
    # For each dimension get the slice which has been requested.
    # If no slice provided, then default to the whole dimension        
    full_slice = [slice(None, None)] * ndim
    
    for i, key in enumerate(keys):
        if key is Ellipsis:
            
            # replace any subsequent Ellipsis objects in keys with slice(None, None) as per Numpy
            keys = keys[:i] + tuple( [slice(None, None) if key is Ellipsis else key for key in keys[i:]] )
            
            # iterate over the remaining keys in reverse to fill in 
            # the gaps from the right hand side
            for j, key in enumerate(keys[:i:-1]):
                full_slice[-j-1] = key
                
            # we've finished with i now so stop the iteration
            break
        else:
            full_slice[i] = key
    
    # remove any tuples on dimensions, turning them into numpy array's for consistent behaviour
    full_slice = tuple([numpy.array(key, ndmin=1) if isinstance(key, tuple) else key for key in full_slice])
    return full_slice 


def _wrap_function_for_method(function, docstring=None):
    """
    Returns a wrapper function modified to be suitable for use as a method.

    The wrapper function renames the first argument as "self" and allows an alternative docstring, thus
    allowing the built-in help(...) routine to display appropriate output.
    
    """
    # Generate the Python source for the wrapper function.
    # NB. The first argument is replaced with "self".
    args, varargs, varkw, defaults = inspect.getargspec(function)
    if defaults is None:
        basic_args = ['self'] + args[1:]
        default_args = []
        simple_default_args = []
    else:
        cutoff = -len(defaults)
        basic_args = ['self'] + args[1:cutoff]
        default_args = ['%s=%r' % pair for pair in zip(args[cutoff:], defaults)]
        simple_default_args = args[cutoff:]
    var_arg = [] if varargs is None else ['*' + varargs]
    var_kw = [] if varkw is None else ['**' + varkw]
    arg_source = ', '.join(basic_args + default_args + var_arg + var_kw)
    simple_arg_source = ', '.join(basic_args + simple_default_args + var_arg + var_kw)
    source = 'def %s(%s):\n    return function(%s)' % (function.func_name, arg_source, simple_arg_source)

    # Compile the wrapper function
    # NB. There's an outstanding bug with "exec" where the locals and globals dictionaries must be the same
    # if we're to get closure behaviour.
    my_locals = {'function': function}
    exec source in my_locals, my_locals

    # Update the docstring if required, and return the modified function
    wrapper = my_locals[function.func_name]
    if docstring is None:
        wrapper.__doc__ = function.__doc__
    else:
        wrapper.__doc__ = docstring
    return wrapper


class _MetaOrderedHashable(abc.ABCMeta):
    """
    A metaclass that ensures that non-abstract subclasses of _OrderedHashable
    without an explicit __init__ method are given a default __init__ method
    with the appropriate method signature.

    Also, an _init method is provided to allow subclasses with their own
    __init__ constructors to initialise their values via an explicit method
    signature.

    NB. This metaclass is used to construct the _OrderedHashable class as well
    as all its subclasses.

    """

    def __new__(cls, name, bases, namespace):
        # We only want to modify concrete classes that have defined the
        # "_names" property.
        if '_names' in namespace and not isinstance(namespace['_names'], abc.abstractproperty):
            args = ', '.join(namespace['_names'])

            # Ensure the class has a constructor with explicit arguments.
            if '__init__' not in namespace:
                # Create a default __init__ method for the class
                method_source = 'def __init__(self, %s):\n self._init_from_tuple((%s,))' % (args, args)
                exec method_source in namespace

            # Ensure the class has a "helper constructor" with explicit arguments.
            if '_init' not in namespace:
                # Create a default _init method for the class
                method_source = 'def _init(self, %s):\n self._init_from_tuple((%s,))' % (args, args)
                exec method_source in namespace

        return super(_MetaOrderedHashable, cls).__new__(cls, name, bases, namespace)


class _OrderedHashable(collections.Hashable):
    """
    Convenience class for creating "immutable", hashable, and ordered classes.
    
    Instance identity is defined by the specific list of attribute names
    declared in the abstract attribute "_names". Subclasses must declare the
    attribute "_names" as an iterable containing the names of all the
    attributes relevant to equality/hash-value/ordering.

    Initial values should be set by using ::
        self._init(self, value1, value2, ..)
    
    .. note::
        It's the responsibility of the subclass to ensure that the values of
        its attributes are themselves hashable.
    
    """

    # The metaclass adds default __init__ methods when appropriate.
    __metaclass__ = _MetaOrderedHashable

    @abc.abstractproperty
    def _names(self):
        """
        Override this attribute to declare the names of all the attributes
        relevant to the hash/comparison semantics.

        """
        pass

    def _init_from_tuple(self, values):
        for name, value in zip(self._names, values):
            object.__setattr__(self, name, value)

    def __repr__(self):
        class_name = type(self).__name__
        attributes = ', '.join('%s=%r' % (name, value) for (name, value) in zip(self._names, self._as_tuple()))
        return '%s(%s)' % (class_name, attributes)

    def _as_tuple(self):
        return tuple(getattr(self, name) for name in self._names)

    # Prevent attribute updates

    def __setattr__(self, name, value):
        raise AttributeError('Instances of %s are immutable' % type(self).__name__)

    def __delattr__(self, name):
        raise AttributeError('Instances of %s are immutable' % type(self).__name__)

    # Provide hash semantics

    def _identity(self):
        return self._as_tuple()

    def __hash__(self):
        return hash(self._identity())

    def __eq__(self, other):
        return isinstance(other, type(self)) and self._identity() == other._identity()

    def __ne__(self, other):
        # Since we've defined __eq__ we should also define __ne__.
        return not self == other

    # Provide default ordering semantics

    def __cmp__(self, other):
        if isinstance(other, _OrderedHashable):
            result = cmp(self._identity(), other._identity())
        else:
            result = NotImplemented
        return result


def create_temp_filename(suffix=''):
    """Return a temporary file name.

    Args:
    
        * suffix  -  Optional filename extension.
    
    """
    temp_file = tempfile.mkstemp(suffix)
    os.close(temp_file[0])
    return temp_file[1]


def clip_string(the_str, clip_length=70, rider = "..."):
    """
    Returns a clipped version of the string based on the specified clip length and whether
    or not any graceful clip points can be found.
    
    If the string to be clipped is shorter than the specified clip length, the original string is returned.
    
    If the string is longer than the clip length, a graceful point (a space character) after the clip length
    is searched for. If a graceful point is found the string is clipped at this point and the rider is added.
    If no graceful point can be found, then the string is clipped exactly where the user requested and the
    rider is added.
    
    Args:
    
    * the_str
        The string to be clipped
    * clip_length
        The length in characters that the input string should be clipped to.
        Defaults to a preconfigured value if not specified.
    * rider
        A series of characters appended at the end of the returned string to show it has been clipped.
        Defaults to a preconfigured value if not specified.
        
    Returns:
        The string clipped to the required length with a rider appended. If the clip length
        was greater than the orignal string, the original string is returned unaltered.
        
    """

    if clip_length >= len(the_str) or clip_length <=0:
        return the_str
    else:
        if the_str[clip_length].isspace():
            return the_str[:clip_length] + rider
        else:
            first_part = the_str[:clip_length]   
            remainder = the_str[clip_length:]
            
            # Try to find a graceful point at which to trim i.e. a space
            # If no graceful point can be found, then just trim where the user specified
            # by adding an empty slice of the remainder ( [:0] )
            termination_point = remainder.find(" ") if remainder.find(" ") != -1 else 0
            
            return first_part + remainder[:termination_point] + rider


def ensure_array(a):
    if not isinstance(a, (numpy.ndarray, numpy.ma.core.MaskedArray)):
        a = numpy.array([a])
    return a



class _Timers(object):
    # See help for timers, below.
    
    def __init__(self):
        self.timers = {}
    
    def start(self, name, step_name):
        self.stop(name)
        timer = self.timers.setdefault(name, {})
        timer[step_name] = time.time()
        timer["active_timer_step"] = step_name

    def restart(self, name, step_name):
        self.stop(name)
        timer = self.timers.setdefault(name, {})
        timer[step_name] = time.time() - timer.get(step_name, 0)
        timer["active_timer_step"] = step_name
    
    def stop(self, name):
        if name in self.timers and "active_timer_step" in self.timers[name]:
            timer = self.timers[name]
            active = timer["active_timer_step"]
            start = timer[active]
            timer[active] = time.time() - start
        return self.get(name)
   
    def get(self, name):
        result = (name, [])
        if name in self.timers:
            result = (name, ", ".join(["'%s':%8.5f"%(k,v) for k, v in self.timers[name].items() if k != "active_timer_step"]))
        return result

    def reset(self, name):
        self.timers[name] = {}
    

timers = _Timers()
"""
Provides multiple named timers, each composed of multiple named steps.

Only one step is active at a time, so calling start(timer_name, step_name)
will stop the current step and start the new one.

Example Usage:

    from iris.util import timers

    def little_func(param):

        timers.restart("little func", "init")
        init()

        timers.restart("little func", "main")
        main(param)

        timers.restart("little func", "cleanup")
        cleanup()

        timers.stop("little func")
        
    def my_big_func():
        
        timers.start("big func", "input")
        input()
        
        timers.start("big func", "processing")
        little_func(123)
        little_func(456)

        timers.start("big func", "output")
        output()
        
        print timers.stop("big func")
        
        print timers.get("little func")
        
"""


def format_array(arr):
    """
    Returns the given array as a string, using the python builtin str function on a piecewise basis.
    
    Useful for xml representation of arrays. 
    
    For customisations, use the :mod:`numpy.core.arrayprint` directly.
    
    """    
    if arr.size > 85:
        summary_insert = "..., "
    else:
        summary_insert = ""
    ffunc = str
    return numpy.core.arrayprint._formatArray(arr, ffunc, len(arr.shape), max_line_len=50,
                                              next_line_prefix='\t\t', separator=', ',
                                              edge_items=3, summary_insert=summary_insert)[:-1]
