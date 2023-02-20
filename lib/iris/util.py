# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Miscellaneous utility functions.

"""

from abc import ABCMeta, abstractmethod
from collections.abc import Hashable, Iterable
import functools
import inspect
import os
import os.path
import sys
import tempfile

import cf_units
from dask import array as da
import numpy as np
import numpy.ma as ma

from iris._deprecation import warn_deprecated
from iris._lazy_data import as_concrete_data, is_lazy_data
from iris.common import SERVICES
from iris.common.lenient import _lenient_client
import iris.exceptions


def broadcast_to_shape(array, shape, dim_map):
    """
    Broadcast an array to a given shape.

    Each dimension of the array must correspond to a dimension in the
    given shape. Striding is used to repeat the array until it matches
    the desired shape, returning repeated views on the original array.
    If you need to write to the resulting array, make a copy first.

    Args:

    * array (:class:`numpy.ndarray`-like)
        An array to broadcast.

    * shape (:class:`list`, :class:`tuple` etc.):
        The shape the array should be broadcast to.

    * dim_map (:class:`list`, :class:`tuple` etc.):
        A mapping of the dimensions of *array* to their corresponding
        element in *shape*. *dim_map* must be the same length as the
        number of dimensions in *array*. Each element of *dim_map*
        corresponds to a dimension of *array* and its value provides
        the index in *shape* which the dimension of *array* corresponds
        to, so the first element of *dim_map* gives the index of *shape*
        that corresponds to the first dimension of *array* etc.

    Examples:

    Broadcasting an array of shape (2, 3) to the shape (5, 2, 6, 3)
    where the first dimension of the array corresponds to the second
    element of the desired shape and the second dimension of the array
    corresponds to the fourth element of the desired shape::

        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = broadcast_to_shape(a, (5, 2, 6, 3), (1, 3))

    Broadcasting an array of shape (48, 96) to the shape (96, 48, 12)::

        # a is an array of shape (48, 96)
        result = broadcast_to_shape(a, (96, 48, 12), (1, 0))

    """
    if len(dim_map) != array.ndim:
        # We must check for this condition here because we cannot rely on
        # getting an error from numpy if the dim_map argument is not the
        # correct length, we might just get a segfault.
        raise ValueError(
            "dim_map must have an entry for every "
            "dimension of the input array"
        )

    def _broadcast_helper(a):
        strides = [0] * len(shape)
        for idim, dim in enumerate(dim_map):
            if shape[dim] != a.shape[idim]:
                # We'll get garbage values if the dimensions of array are not
                # those indicated by shape.
                raise ValueError("shape and array are not compatible")
            strides[dim] = a.strides[idim]
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    array_view = _broadcast_helper(array)
    if ma.isMaskedArray(array):
        if array.mask is ma.nomask:
            # Degenerate masks can be applied as-is.
            mask_view = array.mask
        else:
            # Mask arrays need to be handled in the same way as the data array.
            mask_view = _broadcast_helper(array.mask)
        array_view = ma.array(array_view, mask=mask_view)
    return array_view


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

            >>> import numpy as np
            >>> import iris.util
            >>> original = np.array([-180, -90, 0, 90])
            >>> iris.util.delta(original, 0)
            array([90, 90, 90])
            >>> iris.util.delta(original, 0, circular=360)
            array([90, 90, 90, 90])

    """
    if circular is not False:
        _delta = np.roll(ndarray, -1, axis=dimension)
        last_element = [slice(None, None)] * ndarray.ndim
        last_element[dimension] = slice(-1, None)
        last_element = tuple(last_element)

        if not isinstance(circular, bool):
            result = np.where(ndarray[last_element] >= _delta[last_element])[0]
            _delta[last_element] -= circular
            _delta[last_element][result] += 2 * circular

        np.subtract(_delta, ndarray, _delta)
    else:
        _delta = np.diff(ndarray, axis=dimension)

    return _delta


def describe_diff(cube_a, cube_b, output_file=None):
    """
    Prints the differences that prevent compatibility between two cubes, as
    defined by :meth:`iris.cube.Cube.is_compatible()`.

    Args:

    * cube_a:
        An instance of :class:`iris.cube.Cube` or
        :class:`iris.cube.CubeMetadata`.

    * cube_b:
        An instance of :class:`iris.cube.Cube` or
        :class:`iris.cube.CubeMetadata`.

    * output_file:
        A :class:`file` or file-like object to receive output. Defaults to
        sys.stdout.

    .. seealso::

        :meth:`iris.cube.Cube.is_compatible()`

    .. note::

        Compatibility does not guarantee that two cubes can be merged.
        Instead, this function is designed to provide a verbose description
        of the differences in metadata between two cubes. Determining whether
        two cubes will merge requires additional logic that is beyond the
        scope of this function.

    """

    if output_file is None:
        output_file = sys.stdout

    if cube_a.is_compatible(cube_b):
        output_file.write("Cubes are compatible\n")
    else:
        common_keys = set(cube_a.attributes).intersection(cube_b.attributes)
        for key in common_keys:
            if np.any(cube_a.attributes[key] != cube_b.attributes[key]):
                output_file.write(
                    '"%s" cube_a attribute value "%s" is not '
                    "compatible with cube_b "
                    'attribute value "%s"\n'
                    % (key, cube_a.attributes[key], cube_b.attributes[key])
                )

        if cube_a.name() != cube_b.name():
            output_file.write(
                'cube_a name "%s" is not compatible '
                'with cube_b name "%s"\n' % (cube_a.name(), cube_b.name())
            )

        if cube_a.units != cube_b.units:
            output_file.write(
                'cube_a units "%s" are not compatible with cube_b units "%s"\n'
                % (cube_a.units, cube_b.units)
            )

        if cube_a.cell_methods != cube_b.cell_methods:
            output_file.write(
                "Cell methods\n%s\nand\n%s\nare not compatible\n"
                % (cube_a.cell_methods, cube_b.cell_methods)
            )


def guess_coord_axis(coord):
    """
    Returns a "best guess" axis name of the coordinate.

    Heuristic categorisation of the coordinate into either label
    'T', 'Z', 'Y', 'X' or None.

    Args:

    * coord:
        The :class:`iris.coords.Coord`.

    Returns:
        'T', 'Z', 'Y', 'X', or None.

    """
    axis = None

    if coord.standard_name in (
        "longitude",
        "grid_longitude",
        "projection_x_coordinate",
    ):
        axis = "X"
    elif coord.standard_name in (
        "latitude",
        "grid_latitude",
        "projection_y_coordinate",
    ):
        axis = "Y"
    elif coord.units.is_convertible("hPa") or coord.attributes.get(
        "positive"
    ) in ("up", "down"):
        axis = "Z"
    elif coord.units.is_time_reference():
        axis = "T"

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

        >>> x = np.arange(10).reshape((2, 5))
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
    num_windows = (a.shape[axis] - window + step) // step
    shape = a.shape[:axis] + (num_windows, window) + a.shape[axis + 1 :]
    strides = (
        a.strides[:axis]
        + (step * a.strides[axis], a.strides[axis])
        + a.strides[axis + 1 :]
    )
    rw = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    if ma.isMaskedArray(a):
        mask = ma.getmaskarray(a)
        strides = (
            mask.strides[:axis]
            + (step * mask.strides[axis], mask.strides[axis])
            + mask.strides[axis + 1 :]
        )
        rw = ma.array(
            rw,
            mask=np.lib.stride_tricks.as_strided(
                mask, shape=shape, strides=strides
            ),
        )
    return rw


def array_equal(array1, array2, withnans=False):
    """
    Returns whether two arrays have the same shape and elements.

    Args:

    * array1, array2 (arraylike):
        args to be compared, normalised if necessary with :func:`np.asarray`.

    Kwargs:

    * withnans (bool):
        When unset (default), the result is False if either input contains NaN
        points.  This is the normal floating-point arithmetic result.
        When set, return True if inputs contain the same value in all elements,
        _including_ any NaN values.

    This provides much the same functionality as :func:`numpy.array_equal`, but
    with additional support for arrays of strings and NaN-tolerant operation.

    """

    def normalise_array(array):
        if not is_lazy_data(array):
            array = np.asarray(array)
        return array

    array1, array2 = normalise_array(array1), normalise_array(array2)

    eq = array1.shape == array2.shape
    if eq:
        eqs = array1 == array2

        if withnans and (array1.dtype.kind == "f" or array2.dtype.kind == "f"):
            nans1, nans2 = np.isnan(array1), np.isnan(array2)
            eq = as_concrete_data(np.all(nans1 == nans2))

            if eq:
                eqs = as_concrete_data(eqs)
                if not is_lazy_data(nans1):
                    idxs = nans1
                elif not is_lazy_data(nans2):
                    idxs = nans2
                else:
                    idxs = as_concrete_data(nans1)

                if np.any(idxs):
                    eqs[idxs] = True

        if eq:
            eq = as_concrete_data(np.all(eqs))  # check equal at all points

    return eq


def approx_equal(a, b, max_absolute_error=1e-10, max_relative_error=1e-10):
    """
    Returns whether two numbers are almost equal, allowing for the finite
    precision of floating point numbers.

    .. deprecated:: 3.2.0

       Instead use :func:`math.isclose`. For example, rather than calling
       ``approx_equal(a, b, max_abs, max_rel)`` replace with ``math.isclose(a,
       b, max_rel, max_abs)``. Note that :func:`~math.isclose` will return True
       if the actual error equals the maximum, whereas :func:`util.approx_equal`
       will return False.

    """
    wmsg = (
        "iris.util.approx_equal has been deprecated and will be removed, "
        "please use math.isclose instead."
    )
    warn_deprecated(wmsg)

    # Deal with numbers close to zero
    if abs(a - b) < max_absolute_error:
        return True
    # Ensure we get consistent results if "a" and "b" are supplied in the
    # opposite order.
    max_ab = max([a, b], key=abs)
    relative_error = abs(a - b) / max_ab
    return relative_error < max_relative_error


def between(lh, rh, lh_inclusive=True, rh_inclusive=True):
    """
    Provides a convenient way of defining a 3 element inequality such as
    ``a < number < b``.

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
           print(i, between_3_and_6(i))


        between_3_and_6 = between(3, 6, rh_inclusive=False)
        for i in range(10):
           print(i, between_3_and_6(i))

    """
    if lh_inclusive and rh_inclusive:
        return lambda c: lh <= c <= rh
    elif lh_inclusive and not rh_inclusive:
        return lambda c: lh <= c < rh
    elif not lh_inclusive and rh_inclusive:
        return lambda c: lh < c <= rh
    else:
        return lambda c: lh < c < rh


def reverse(cube_or_array, coords_or_dims):
    """
    Reverse the cube or array along the given dimensions.

    Args:

    * cube_or_array: :class:`iris.cube.Cube` or :class:`numpy.ndarray`
        The cube or array to reverse.
    * coords_or_dims: int, str, :class:`iris.coords.Coord` or sequence of these
        Identify one or more dimensions to reverse.  If cube_or_array is a
        numpy array, use int or a sequence of ints, as in the examples below.
        If cube_or_array is a Cube, a Coord or coordinate name (or sequence of
        these) may be specified instead.

    ::

        >>> import numpy as np
        >>> a = np.arange(24).reshape(2, 3, 4)
        >>> print(a)
        [[[ 0  1  2  3]
          [ 4  5  6  7]
          [ 8  9 10 11]]
        <BLANKLINE>
         [[12 13 14 15]
          [16 17 18 19]
          [20 21 22 23]]]
        >>> print(reverse(a, 1))
        [[[ 8  9 10 11]
          [ 4  5  6  7]
          [ 0  1  2  3]]
        <BLANKLINE>
         [[20 21 22 23]
          [16 17 18 19]
          [12 13 14 15]]]
        >>> print(reverse(a, [1, 2]))
        [[[11 10  9  8]
          [ 7  6  5  4]
          [ 3  2  1  0]]
        <BLANKLINE>
         [[23 22 21 20]
          [19 18 17 16]
          [15 14 13 12]]]

    """
    from iris.cube import Cube

    index = [slice(None, None)] * cube_or_array.ndim

    if isinstance(coords_or_dims, Cube):
        raise TypeError(
            "coords_or_dims must be int, str, coordinate or "
            "sequence of these.  Got cube."
        )

    if isinstance(coords_or_dims, str) or not isinstance(
        coords_or_dims, Iterable
    ):
        coords_or_dims = [coords_or_dims]

    axes = set()
    for coord_or_dim in coords_or_dims:
        if isinstance(coord_or_dim, int):
            axes.add(coord_or_dim)
        elif isinstance(cube_or_array, np.ndarray):
            raise TypeError(
                "To reverse an array, provide an int or sequence of ints."
            )
        else:
            try:
                axes.update(cube_or_array.coord_dims(coord_or_dim))
            except AttributeError:
                raise TypeError(
                    "coords_or_dims must be int, str, coordinate "
                    "or sequence of these."
                )

    axes = np.array(list(axes), ndmin=1)
    if axes.ndim != 1 or axes.size == 0:
        raise ValueError(
            "Reverse was expecting a single axis or a 1d array "
            "of axes, got %r" % axes
        )
    if np.min(axes) < 0 or np.max(axes) > cube_or_array.ndim - 1:
        raise ValueError(
            "An axis value out of range for the number of "
            "dimensions from the given array (%s) was received. "
            "Got: %r" % (cube_or_array.ndim, axes)
        )

    for axis in axes:
        index[axis] = slice(None, None, -1)

    return cube_or_array[tuple(index)]


def monotonic(array, strict=False, return_direction=False):
    """
    Return whether the given 1d array is monotonic.

    Note that, the array must not contain missing data.

    Kwargs:

    * strict (boolean)
        Flag to enable strict monotonic checking
    * return_direction (boolean)
        Flag to change return behaviour to return
        (monotonic_status, direction). Direction will be 1 for positive
        or -1 for negative. The direction is meaningless if the array is
        not monotonic.

    Returns:

    * monotonic_status (boolean)
        Whether the array was monotonic.

        If the return_direction flag was given then the returned value
        will be:

            ``(monotonic_status, direction)``

    """
    if array.ndim != 1 or len(array) <= 1:
        raise ValueError(
            "The array to check must be 1 dimensional and have "
            "more than 1 element."
        )

    if ma.isMaskedArray(array) and ma.count_masked(array) != 0:
        raise ValueError("The array to check contains missing data.")

    # Identify the directions of the largest/most-positive and
    # smallest/most-negative steps.
    d = np.diff(array)

    sign_max_d = np.sign(np.max(d))
    sign_min_d = np.sign(np.min(d))

    if strict:
        monotonic = sign_max_d == sign_min_d and sign_max_d != 0
    else:
        monotonic = (
            (sign_min_d < 0 and sign_max_d <= 0)
            or (sign_max_d > 0 and sign_min_d >= 0)
            or (sign_min_d == sign_max_d == 0)
        )

    if return_direction:
        if sign_max_d == 0:
            direction = sign_min_d
        else:
            direction = sign_max_d

        return monotonic, direction

    return monotonic


def column_slices_generator(full_slice, ndims):
    """
    Given a full slice full of tuples, return a dictionary mapping old
    data dimensions to new and a generator which gives the successive
    slices needed to index correctly (across columns).

    This routine deals with the special functionality for tuple based
    indexing e.g. [0, (3, 5), :, (1, 6, 8)] by first providing a slice
    which takes the non tuple slices out first i.e. [0, :, :, :] then
    subsequently iterates through each of the tuples taking out the
    appropriate slices i.e. [(3, 5), :, :] followed by [:, :, (1, 6, 8)]

    This method was developed as numpy does not support the direct
    approach of [(3, 5), : , (1, 6, 8)] for column based indexing.

    """
    list_of_slices = []

    # Map current dimensions to new dimensions, or None
    dimension_mapping = {None: None}
    _count_current_dim = 0
    for i, i_key in enumerate(full_slice):
        if isinstance(i_key, (int, np.integer)):
            dimension_mapping[i] = None
        else:
            dimension_mapping[i] = _count_current_dim
            _count_current_dim += 1

    # Get all of the dimensions for which a tuple of indices were provided
    # (numpy.ndarrays are treated in the same way tuples in this case)
    def is_tuple_style_index(key):
        return isinstance(key, tuple) or (
            isinstance(key, np.ndarray) and key.ndim == 1
        )

    tuple_indices = [
        i for i, key in enumerate(full_slice) if is_tuple_style_index(key)
    ]

    # stg1: Take a copy of the full_slice specification, turning all tuples
    # into a full slice
    if tuple_indices != list(range(len(full_slice))):
        first_slice = list(full_slice)
        for tuple_index in tuple_indices:
            first_slice[tuple_index] = slice(None, None)
        # turn first_slice back into a tuple ready for indexing
        first_slice = tuple(first_slice)

        list_of_slices.append(first_slice)

    # stg2 iterate over each of the tuples
    for tuple_index in tuple_indices:
        # Create a list with the indices to span the whole data array that we
        # currently have
        spanning_slice_with_tuple = [slice(None, None)] * _count_current_dim
        # Replace the slice(None, None) with our current tuple
        spanning_slice_with_tuple[dimension_mapping[tuple_index]] = full_slice[
            tuple_index
        ]

        # if we just have [(0, 1)] turn it into [(0, 1), ...] as this is
        # Numpy's syntax.
        if len(spanning_slice_with_tuple) == 1:
            spanning_slice_with_tuple.append(Ellipsis)

        spanning_slice_with_tuple = tuple(spanning_slice_with_tuple)

        list_of_slices.append(spanning_slice_with_tuple)

    # return the dimension mapping and a generator of slices
    return dimension_mapping, iter(list_of_slices)


def _build_full_slice_given_keys(keys, ndim):
    """
    Given the keys passed to a __getitem__ call, build an equivalent
    tuple of keys which span ndims.

    """
    # Ensure that we always have a tuple of keys
    if not isinstance(keys, tuple):
        keys = tuple([keys])

    # catch the case where an extra Ellipsis has been provided which can be
    # discarded iff len(keys)-1 == ndim
    if len(keys) - 1 == ndim and Ellipsis in filter(
        lambda obj: not isinstance(obj, np.ndarray), keys
    ):
        keys = list(keys)
        is_ellipsis = [key is Ellipsis for key in keys]
        keys.pop(is_ellipsis.index(True))
        keys = tuple(keys)

    # for ndim >= 1 appending a ":" to the slice specification is allowable,
    # remove this now
    if len(keys) > ndim and ndim != 0 and keys[-1] == slice(None, None):
        keys = keys[:-1]

    if len(keys) > ndim:
        raise IndexError(
            "More slices requested than dimensions. Requested "
            "%r, but there were only %s dimensions." % (keys, ndim)
        )

    # For each dimension get the slice which has been requested.
    # If no slice provided, then default to the whole dimension
    full_slice = [slice(None, None)] * ndim

    for i, key in enumerate(keys):
        if key is Ellipsis:
            # replace any subsequent Ellipsis objects in keys with
            # slice(None, None) as per Numpy
            keys = keys[:i] + tuple(
                [
                    slice(None, None) if key is Ellipsis else key
                    for key in keys[i:]
                ]
            )

            # iterate over the remaining keys in reverse to fill in
            # the gaps from the right hand side
            for j, key in enumerate(keys[:i:-1]):
                full_slice[-j - 1] = key

            # we've finished with i now so stop the iteration
            break
        else:
            full_slice[i] = key

    # remove any tuples on dimensions, turning them into numpy array's for
    # consistent behaviour
    full_slice = tuple(
        [
            np.array(key, ndmin=1) if isinstance(key, tuple) else key
            for key in full_slice
        ]
    )
    return full_slice


def _slice_data_with_keys(data, keys):
    """
    Index an array-like object as "data[keys]", with orthogonal indexing.

    Args:

    * data (array-like):
        array to index.

    * keys (list):
        list of indexes, as received from a __getitem__ call.

    This enforces an orthogonal interpretation of indexing, which means that
    both 'real' (numpy) arrays and other array-likes index in the same way,
    instead of numpy arrays doing 'fancy indexing'.

    Returns (dim_map, data_region), where :

    * dim_map (dict) :
        A dimension map, as returned by :func:`column_slices_generator`.
        i.e. "dim_map[old_dim_index]" --> "new_dim_index" or None.

    * data_region (array-like) :
        The sub-array.

    .. Note::

        Avoids copying the data, where possible.

    """
    # Combines the use of _build_full_slice_given_keys and
    # column_slices_generator.
    # By slicing on only one index at a time, this also mostly avoids copying
    # the data, except some cases when a key contains a list of indices.
    n_dims = len(data.shape)
    full_slice = _build_full_slice_given_keys(keys, n_dims)
    dims_mapping, slices_iter = column_slices_generator(full_slice, n_dims)
    for this_slice in slices_iter:
        data = data[this_slice]
        if data.ndim > 0 and min(data.shape) < 1:
            # Disallow slicings where a dimension has no points, like "[5:5]".
            raise IndexError("Cannot index with zero length slice.")

    return dims_mapping, data


def _wrap_function_for_method(function, docstring=None):
    """
    Returns a wrapper function modified to be suitable for use as a
    method.

    The wrapper function renames the first argument as "self" and allows
    an alternative docstring, thus allowing the built-in help(...)
    routine to display appropriate output.

    """
    # Generate the Python source for the wrapper function.
    # NB. The first argument is replaced with "self".
    args, varargs, varkw, defaults = inspect.getargspec(function)
    if defaults is None:
        basic_args = ["self"] + args[1:]
        default_args = []
        simple_default_args = []
    else:
        cutoff = -len(defaults)
        basic_args = ["self"] + args[1:cutoff]
        default_args = [
            "%s=%r" % pair for pair in zip(args[cutoff:], defaults)
        ]
        simple_default_args = args[cutoff:]
    var_arg = [] if varargs is None else ["*" + varargs]
    var_kw = [] if varkw is None else ["**" + varkw]
    arg_source = ", ".join(basic_args + default_args + var_arg + var_kw)
    simple_arg_source = ", ".join(
        basic_args + simple_default_args + var_arg + var_kw
    )
    source = "def %s(%s):\n    return function(%s)" % (
        function.__name__,
        arg_source,
        simple_arg_source,
    )

    # Compile the wrapper function
    # NB. There's an outstanding bug with "exec" where the locals and globals
    # dictionaries must be the same if we're to get closure behaviour.
    my_locals = {"function": function}
    exec(source, my_locals, my_locals)

    # Update the docstring if required, and return the modified function
    wrapper = my_locals[function.__name__]
    if docstring is None:
        wrapper.__doc__ = function.__doc__
    else:
        wrapper.__doc__ = docstring
    return wrapper


class _MetaOrderedHashable(ABCMeta):
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
        if "_names" in namespace and not getattr(
            namespace["_names"], "__isabstractmethod__", False
        ):
            args = ", ".join(namespace["_names"])

            # Ensure the class has a constructor with explicit arguments.
            if "__init__" not in namespace:
                # Create a default __init__ method for the class
                method_source = (
                    "def __init__(self, %s):\n "
                    "self._init_from_tuple((%s,))" % (args, args)
                )
                exec(method_source, namespace)

            # Ensure the class has a "helper constructor" with explicit
            # arguments.
            if "_init" not in namespace:
                # Create a default _init method for the class
                method_source = (
                    "def _init(self, %s):\n "
                    "self._init_from_tuple((%s,))" % (args, args)
                )
                exec(method_source, namespace)

        return super().__new__(cls, name, bases, namespace)


@functools.total_ordering
class _OrderedHashable(Hashable, metaclass=_MetaOrderedHashable):
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

    @property
    @abstractmethod
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
        attributes = ", ".join(
            "%s=%r" % (name, value)
            for (name, value) in zip(self._names, self._as_tuple())
        )
        return "%s(%s)" % (class_name, attributes)

    def _as_tuple(self):
        return tuple(getattr(self, name) for name in self._names)

    # Prevent attribute updates

    def __setattr__(self, name, value):
        raise AttributeError(
            "Instances of %s are immutable" % type(self).__name__
        )

    def __delattr__(self, name):
        raise AttributeError(
            "Instances of %s are immutable" % type(self).__name__
        )

    # Provide hash semantics

    def _identity(self):
        return self._as_tuple()

    def __hash__(self):
        return hash(self._identity())

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self._identity() == other._identity()
        )

    def __ne__(self, other):
        # Since we've defined __eq__ we should also define __ne__.
        return not self == other

    # Provide default ordering semantics

    def __lt__(self, other):
        if isinstance(other, _OrderedHashable):
            return self._identity() < other._identity()
        else:
            return NotImplemented


def create_temp_filename(suffix=""):
    """Return a temporary file name.

    Args:

        * suffix  -  Optional filename extension.

    """
    temp_file = tempfile.mkstemp(suffix)
    os.close(temp_file[0])
    return temp_file[1]


def clip_string(the_str, clip_length=70, rider="..."):
    """
    Returns a clipped version of the string based on the specified clip
    length and whether or not any graceful clip points can be found.

    If the string to be clipped is shorter than the specified clip
    length, the original string is returned.

    If the string is longer than the clip length, a graceful point (a
    space character) after the clip length is searched for. If a
    graceful point is found the string is clipped at this point and the
    rider is added. If no graceful point can be found, then the string
    is clipped exactly where the user requested and the rider is added.

    Args:

    * the_str
        The string to be clipped
    * clip_length
        The length in characters that the input string should be clipped
        to. Defaults to a preconfigured value if not specified.
    * rider
        A series of characters appended at the end of the returned
        string to show it has been clipped. Defaults to a preconfigured
        value if not specified.

    Returns:
        The string clipped to the required length with a rider appended.
        If the clip length was greater than the original string, the
        original string is returned unaltered.

    """

    if clip_length >= len(the_str) or clip_length <= 0:
        return the_str
    else:
        if the_str[clip_length].isspace():
            return the_str[:clip_length] + rider
        else:
            first_part = the_str[:clip_length]
            remainder = the_str[clip_length:]

            # Try to find a graceful point at which to trim i.e. a space
            # If no graceful point can be found, then just trim where the user
            # specified by adding an empty slice of the remainder ( [:0] )
            termination_point = remainder.find(" ")
            if termination_point == -1:
                termination_point = 0

            return first_part + remainder[:termination_point] + rider


def format_array(arr):
    """
    Returns the given array as a string, using the python builtin str
    function on a piecewise basis.

    Useful for xml representation of arrays.

    For customisations, use the :mod:`numpy.core.arrayprint` directly.

    """

    summary_threshold = 85
    summary_insert = "..." if arr.size > summary_threshold else ""
    edge_items = 3
    ffunc = str
    max_line_len = 50

    # Format the array with version 1.13 legacy behaviour
    with np.printoptions(legacy="1.13"):
        # Use this (private) routine for more control.
        formatArray = np.core.arrayprint._formatArray
        # N.B. the 'legacy' arg had different forms in different numpy versions
        # -- fetch the required form from the internal options dict
        format_options_legacy = np.core.arrayprint._format_options["legacy"]

        result = formatArray(
            arr,
            ffunc,
            max_line_len,
            next_line_prefix="\t\t",
            separator=", ",
            edge_items=edge_items,
            summary_insert=summary_insert,
            legacy=format_options_legacy,
        )

    return result


def new_axis(src_cube, scalar_coord=None, expand_extras=()):
    """
    Create a new axis as the leading dimension of the cube, promoting a scalar
    coordinate if specified.

    Args:

    * src_cube (:class:`iris.cube.Cube`)
        Source cube on which to generate a new axis.

    Kwargs:

    * scalar_coord (:class:`iris.coord.Coord` or 'string')
        Scalar coordinate to promote to a dimension coordinate.

    * expand_extras (iterable)
        Auxiliary coordinates, ancillary variables and cell measures which will
        be expanded so that they map to the new dimension as well as the
        existing dimensions.

    Returns:
        A new :class:`iris.cube.Cube` instance with one extra leading dimension
        (length 1). Chosen auxiliary coordinates, cell measures and ancillary
        variables will also be given an additional dimension, associated with
        the leading dimension of the cube.

    For example::

        >>> cube.shape
        (360, 360)
        >>> ncube = iris.util.new_axis(cube, 'time')
        >>> ncube.shape
        (1, 360, 360)
    """

    def _reshape_data_array(data_manager):
        # Indexing numpy arrays requires loading deferred data here returning a
        # copy of the data with a new leading dimension.
        # If the data of the source cube (or values of the dimensional metadata
        # object) is a Masked Constant, it is changed here to a Masked Array to
        # allow the mask to gain an extra dimension with the data.
        if data_manager.has_lazy_data():
            new_data = data_manager.lazy_data()[None]
        else:
            if isinstance(data_manager.data, ma.core.MaskedConstant):
                new_data = ma.array([np.nan], mask=[True])
            else:
                new_data = data_manager.data[None]
        return new_data

    def _handle_dimensional_metadata(
        cube, dm_item, cube_add_method, expand_extras
    ):
        cube_dims = dm_item.cube_dims(cube)
        if dm_item in expand_extras:
            if cube_dims == ():
                new_dm_item, new_dims = dm_item.copy(), 0
            else:
                new_dims = np.concatenate([(0,), np.array(cube_dims) + 1])
                new_values = _reshape_data_array(dm_item._values_dm)
                kwargs = dm_item.metadata._asdict()
                new_dm_item = dm_item.__class__(new_values, **kwargs)
                try:
                    if dm_item.has_bounds():
                        new_dm_item.bounds = _reshape_data_array(
                            dm_item._bounds_dm
                        )
                except AttributeError:
                    pass
        else:
            new_dims = np.array(cube_dims) + 1
            new_dm_item = dm_item.copy()

        cube_add_method(new_dm_item, new_dims)

    if scalar_coord is not None:
        scalar_coord = src_cube.coord(scalar_coord)
        if not scalar_coord.shape == (1,):
            emsg = scalar_coord.name() + "is not a scalar coordinate."
            raise ValueError(emsg)

    expand_extras = [
        src_cube._dimensional_metadata(item) for item in expand_extras
    ]

    new_cube = iris.cube.Cube(_reshape_data_array(src_cube._data_manager))
    new_cube.metadata = src_cube.metadata

    for coord in src_cube.dim_coords:
        coord_dims = np.array(src_cube.coord_dims(coord)) + 1
        new_cube.add_dim_coord(coord.copy(), coord_dims)

    for coord in src_cube.aux_coords:
        if scalar_coord and scalar_coord == coord:
            dim_coord = iris.coords.DimCoord.from_coord(coord)
            new_cube.add_dim_coord(dim_coord, 0)
        else:
            _handle_dimensional_metadata(
                src_cube, coord, new_cube.add_aux_coord, expand_extras
            )

    for cm in src_cube.cell_measures():
        _handle_dimensional_metadata(
            src_cube, cm, new_cube.add_cell_measure, expand_extras
        )

    for av in src_cube.ancillary_variables():
        _handle_dimensional_metadata(
            src_cube, av, new_cube.add_ancillary_variable, expand_extras
        )

    nonderived_coords = src_cube.dim_coords + src_cube.aux_coords
    coord_mapping = {
        id(old_co): new_cube.coord(old_co) for old_co in nonderived_coords
    }
    for factory in src_cube.aux_factories:
        new_factory = factory.updated(coord_mapping)
        new_cube.add_aux_factory(new_factory)

    return new_cube


def squeeze(cube):
    """
    Removes any dimension of length one. If it has an associated DimCoord or
    AuxCoord, this becomes a scalar coord.

    Args:

    * cube (:class:`iris.cube.Cube`)
        Source cube to remove length 1 dimension(s) from.

    Returns:
        A new :class:`iris.cube.Cube` instance without any dimensions of
        length 1.

    For example::

        >>> cube.shape
        (1, 360, 360)
        >>> ncube = iris.util.squeeze(cube)
        >>> ncube.shape
        (360, 360)

    """

    slices = [
        0 if cube.shape[dim] == 1 else slice(None) for dim in range(cube.ndim)
    ]

    squeezed = cube[tuple(slices)]

    return squeezed


def file_is_newer_than(result_path, source_paths):
    """
    Return whether the 'result' file has a later modification time than all of
    the 'source' files.

    If a stored result depends entirely on known 'sources', it need only be
    re-built when one of them changes.  This function can be used to test that
    by comparing file timestamps.

    Args:

    * result_path (string):
        The filepath of a file containing some derived result data.
    * source_paths (string or iterable of strings):
        The path(s) to the original datafiles used to make the result.  May
        include wildcards and '~' expansions (like Iris load paths), but not
        URIs.

    Returns:
        True if all the sources are older than the result, else False.

        If any of the file paths describes no existing files, an exception will
        be raised.

    .. note::
        There are obvious caveats to using file timestamps for this, as correct
        usage depends on how the sources might change.  For example, a file
        could be replaced by one of the same name, but an older timestamp.

        If wildcards and '~' expansions are used, this introduces even more
        uncertainty, as then you cannot even be sure that the resulting list of
        file names is the same as the originals.  For example, some files may
        have been deleted or others added.

    .. note::
        The result file may often be a :mod:`pickle` file.  In that case, it
        also depends on the relevant module sources, so extra caution is
        required.  Ideally, an additional check on iris.__version__ is advised.

    """
    # Accept a string as a single source path
    if isinstance(source_paths, str):
        source_paths = [source_paths]
    # Fix our chosen timestamp function
    file_date = os.path.getmtime
    # Get the 'result file' time
    result_timestamp = file_date(result_path)
    # Get all source filepaths, with normal Iris.io load helper function
    source_file_paths = iris.io.expand_filespecs(source_paths)
    # Compare each filetime, for each spec, with the 'result time'
    for path in source_file_paths:
        source_timestamp = file_date(path)
        if source_timestamp >= result_timestamp:
            return False
    return True


def is_regular(coord):
    """Determine if the given coord is regular."""
    try:
        regular_step(coord)
    except iris.exceptions.CoordinateNotRegularError:
        return False
    except (TypeError, ValueError):
        return False
    return True


def regular_step(coord):
    """Return the regular step from a coord or fail."""
    if coord.ndim != 1:
        raise iris.exceptions.CoordinateMultiDimError("Expected 1D coord")
    if coord.shape[0] < 2:
        raise ValueError("Expected a non-scalar coord")

    avdiff, regular = points_step(coord.points)
    if not regular:
        msg = "Coord %s is not regular" % coord.name()
        raise iris.exceptions.CoordinateNotRegularError(msg)
    return avdiff.astype(coord.points.dtype)


def regular_points(zeroth, step, count):
    """Make an array of regular points.

    Create an array of `count` points from `zeroth` + `step`, adding `step` each
    time. In float32 if this gives a sufficiently regular array (tested with
    points_step) and float64 if not.

    Parameters
    ----------
    zeroth : number
        The value *prior* to the first point value.

    step : number
        The numeric difference between successive point values.

    count : number
        The number of point values.

    """
    points = (zeroth + step) + step * np.arange(count, dtype=np.float32)
    _, regular = iris.util.points_step(points)
    if not regular:
        points = (zeroth + step) + step * np.arange(count, dtype=np.float64)
    return points


def points_step(points):
    """Determine whether `points` has a regular step.

    Parameters
    ----------
    points : numeric, array-like
        The sequence of values to check for a regular difference.

    Returns
    -------
    numeric, bool
        A tuple containing the average difference between values, and whether the difference is regular.
    """
    # Calculations only make sense with multiple points
    points = np.asanyarray(points)
    if points.size >= 2:
        diffs = np.diff(points)
        avdiff = np.mean(diffs)
        # TODO: This value for `rtol` is set for test_analysis to pass...
        regular = np.allclose(diffs, avdiff, rtol=0.001)
    else:
        avdiff = np.nan
        regular = True
    return avdiff, regular


def unify_time_units(cubes):
    """
    Performs an in-place conversion of the time units of all time coords in the
    cubes in a given iterable. One common epoch is defined for each calendar
    found in the cubes to prevent units being defined with inconsistencies
    between epoch and calendar.

    Each epoch is defined from the first suitable time coordinate found in the
    input cubes.

    Arg:

    * cubes:
        An iterable containing :class:`iris.cube.Cube` instances.

    """
    epochs = {}

    for cube in cubes:
        for time_coord in cube.coords():
            if time_coord.units.is_time_reference():
                epoch = epochs.setdefault(
                    time_coord.units.calendar, time_coord.units.origin
                )
                new_unit = cf_units.Unit(epoch, time_coord.units.calendar)
                time_coord.convert_units(new_unit)


def _is_circular(points, modulus, bounds=None):
    """
    Determine whether the provided points or bounds are circular in nature
    relative to the modulus value.

    If the bounds are provided then these are checked for circularity rather
    than the points.

    Args:

    * points:
        :class:`numpy.ndarray` of point values.

    * modulus:
        Circularity modulus value.

    Kwargs:

    * bounds:
        :class:`numpy.ndarray` of bound values.

    Returns:
        Boolean.

    """
    circular = False
    if bounds is not None:
        # Set circular to True if the bounds ends are equivalent.
        first_bound = last_bound = None
        if bounds.ndim == 1 and bounds.shape[-1] == 2:
            first_bound = bounds[0] % modulus
            last_bound = bounds[1] % modulus
        elif bounds.ndim == 2 and bounds.shape[-1] == 2:
            first_bound = bounds[0, 0] % modulus
            last_bound = bounds[-1, 1] % modulus

        if first_bound is not None and last_bound is not None:
            circular = np.allclose(first_bound, last_bound, rtol=1.0e-5)
    else:
        # set circular if points are regular and last+1 ~= first
        if len(points) > 1:
            diffs = list(set(np.diff(points)))
            diff = np.mean(diffs)
            abs_tol = np.abs(diff * 1.0e-4)
            diff_approx_equal = np.max(np.abs(diffs - diff)) < abs_tol
            if diff_approx_equal:
                circular_value = (points[-1] + diff) % modulus
                try:
                    np.testing.assert_approx_equal(
                        points[0], circular_value, significant=4
                    )
                    circular = True
                except AssertionError:
                    if points[0] == 0:
                        try:
                            np.testing.assert_approx_equal(
                                modulus, circular_value, significant=4
                            )
                            circular = True
                        except AssertionError:
                            pass
        else:
            # XXX - Inherited behaviour from NetCDF PyKE rules.
            # We need to decide whether this is valid!
            circular = points[0] >= modulus
    return circular


def promote_aux_coord_to_dim_coord(cube, name_or_coord):
    """
    Promotes an AuxCoord on the cube to a DimCoord. This AuxCoord must be
    associated with a single cube dimension. If the AuxCoord is associated
    with a dimension that already has a DimCoord, that DimCoord gets
    demoted to an AuxCoord.

    Args:

    * cube
        An instance of :class:`iris.cube.Cube`

    * name_or_coord:
        Either

        (a) An instance of :class:`iris.coords.AuxCoord`

        or

        (b) the :attr:`standard_name`, :attr:`long_name`, or
        :attr:`var_name` of an instance of an instance of
        :class:`iris.coords.AuxCoord`.

    For example,

    .. testsetup:: promote

        import iris
        from iris.coord_categorisation import add_year
        from iris.util import demote_dim_coord_to_aux_coord, promote_aux_coord_to_dim_coord
        cube = iris.load_cube(iris.sample_data_path("E1_north_america.nc"))
        cube.remove_coord("forecast_reference_time")
        cube.remove_coord("height")
        cube.attributes = {}
        cube.cell_methods = ()
        add_year(cube, "time")

    .. doctest:: promote

        >>> print(cube)
        air_temperature / (K)               (time: 240; latitude: 37; longitude: 49)
            Dimension coordinates:
                time                             x              -              -
                latitude                         -              x              -
                longitude                        -              -              x
            Auxiliary coordinates:
                forecast_period                  x              -              -
                year                             x              -              -
        >>> promote_aux_coord_to_dim_coord(cube, "year")
        >>> print(cube)
        air_temperature / (K)               (year: 240; latitude: 37; longitude: 49)
            Dimension coordinates:
                year                             x              -              -
                latitude                         -              x              -
                longitude                        -              -              x
            Auxiliary coordinates:
                forecast_period                  x              -              -
                time                             x              -              -

    """
    from iris.coords import Coord, DimCoord

    if isinstance(name_or_coord, str):
        aux_coord = cube.coord(name_or_coord)
    elif isinstance(name_or_coord, Coord):
        aux_coord = name_or_coord
    else:
        # Don't know how to handle this type
        msg = (
            "Don't know how to handle coordinate of type {}. "
            "Ensure all coordinates are of type str or "
            "iris.coords.Coord."
        )
        msg = msg.format(type(name_or_coord))
        raise TypeError(msg)

    if aux_coord in cube.dim_coords:
        # nothing to do
        return

    if aux_coord not in cube.aux_coords:
        msg = (
            "Attempting to promote an AuxCoord ({}) "
            "which does not exist in the cube."
        )
        msg = msg.format(aux_coord.name())
        raise ValueError(msg)

    coord_dim = cube.coord_dims(aux_coord)

    if len(coord_dim) != 1:
        msg = (
            "Attempting to promote an AuxCoord ({}) "
            "which is associated with {} dimensions."
        )
        msg = msg.format(aux_coord.name(), len(coord_dim))
        raise ValueError(msg)

    try:
        dim_coord = DimCoord.from_coord(aux_coord)
    except ValueError as valerr:
        msg = (
            "Attempt to promote an AuxCoord ({}) fails "
            "when attempting to create a DimCoord from the "
            "AuxCoord because: {}"
        )
        msg = msg.format(aux_coord.name(), str(valerr))
        raise ValueError(msg)

    old_dim_coord = cube.coords(
        dim_coords=True, contains_dimension=coord_dim[0]
    )

    if len(old_dim_coord) == 1:
        demote_dim_coord_to_aux_coord(cube, old_dim_coord[0])

    # order matters here: don't want to remove
    # the aux_coord before have tried to make
    # dim_coord in case that fails
    cube.remove_coord(aux_coord)

    cube.add_dim_coord(dim_coord, coord_dim)


def demote_dim_coord_to_aux_coord(cube, name_or_coord):
    """
    Demotes a dimension coordinate  on the cube to an auxiliary coordinate.

    The DimCoord is demoted to an auxiliary coordinate on the cube.
    The dimension of the cube that was associated with the DimCoord becomes
    anonymous.  The class of the coordinate is left as DimCoord, it is not
    recast as an AuxCoord instance.

    Args:

    * cube
        An instance of :class:`iris.cube.Cube`

    * name_or_coord:
        Either

        (a) An instance of :class:`iris.coords.DimCoord`

        or

        (b) the :attr:`standard_name`, :attr:`long_name`, or
        :attr:`var_name` of an instance of an instance of
        :class:`iris.coords.DimCoord`.

    For example,

    .. testsetup:: demote

        import iris
        from iris.coord_categorisation import add_year
        from iris.util import demote_dim_coord_to_aux_coord, promote_aux_coord_to_dim_coord
        cube = iris.load_cube(iris.sample_data_path("E1_north_america.nc"))
        cube.remove_coord("forecast_reference_time")
        cube.remove_coord("height")
        cube.attributes = {}
        cube.cell_methods = ()
        add_year(cube, "time")

    .. doctest:: demote

        >>> print(cube)
        air_temperature / (K)               (time: 240; latitude: 37; longitude: 49)
            Dimension coordinates:
                time                             x              -              -
                latitude                         -              x              -
                longitude                        -              -              x
            Auxiliary coordinates:
                forecast_period                  x              -              -
                year                             x              -              -
        >>> demote_dim_coord_to_aux_coord(cube, "time")
        >>> print(cube)
        air_temperature / (K)               (-- : 240; latitude: 37; longitude: 49)
            Dimension coordinates:
                latitude                        -              x              -
                longitude                       -              -              x
            Auxiliary coordinates:
                forecast_period                 x              -              -
                time                            x              -              -
                year                            x              -              -

    """
    from iris.coords import Coord

    if isinstance(name_or_coord, str):
        dim_coord = cube.coord(name_or_coord)
    elif isinstance(name_or_coord, Coord):
        dim_coord = name_or_coord
    else:
        # Don't know how to handle this type
        msg = (
            "Don't know how to handle coordinate of type {}. "
            "Ensure all coordinates are of type str or "
            "iris.coords.Coord."
        )
        msg = msg.format(type(name_or_coord))
        raise TypeError(msg)

    if dim_coord not in cube.dim_coords:
        # nothing to do
        return

    coord_dim = cube.coord_dims(dim_coord)

    cube.remove_coord(dim_coord)

    cube.add_aux_coord(dim_coord, coord_dim)


@functools.wraps(np.meshgrid)
def _meshgrid(*xi, **kwargs):
    """
    @numpy v1.13, the dtype of each output nD coordinate is the same as its
    associated input 1D coordinate. This is not the case prior to numpy v1.13,
    where the output dtype is cast up to its highest resolution, regardlessly.

    This convenience function ensures consistent meshgrid behaviour across
    numpy versions.

    Reference: https://github.com/numpy/numpy/pull/5302

    """
    mxi = np.meshgrid(*xi, **kwargs)
    for i, (mxii, xii) in enumerate(zip(mxi, xi)):
        if mxii.dtype != xii.dtype:
            mxi[i] = mxii.astype(xii.dtype)
    return mxi


def find_discontiguities(cube, rel_tol=1e-5, abs_tol=1e-8):
    """
    Searches coord for discontiguities in the bounds array, returned as a
    boolean array (True where discontiguities are present).

    Args:

    * cube (`iris.cube.Cube`):
        The cube to be checked for discontinuities in its 'x' and 'y'
        coordinates.

    Kwargs:

    * rel_tol (float):
        The relative equality tolerance to apply in coordinate bounds
        checking.

    * abs_tol (float):
        The absolute value tolerance to apply in coordinate bounds
        checking.

    Returns:

    * result (`numpy.ndarray` of bool) :
        true/false map of which cells in the cube XY grid have
        discontiguities in the coordinate points array.

        This can be used as the input array for
        :func:`iris.util.mask_cube`.

    Examples::

        # Find any unknown discontiguities in your cube's x and y arrays:
        discontiguities = iris.util.find_discontiguities(cube)

        # Pass the resultant boolean array to `iris.util.mask_cube`
        # with a cube slice; this will use the boolean array to mask
        # any discontiguous data points before plotting:
        masked_cube_slice = iris.util.mask_cube(cube[0], discontiguities)

        # Plot the masked cube slice:
        iplt.pcolormesh(masked_cube_slice)

    """
    lats_and_lons = [
        "latitude",
        "grid_latitude",
        "longitude",
        "grid_longitude",
    ]
    spatial_coords = [
        coord for coord in cube.aux_coords if coord.name() in lats_and_lons
    ]
    dim_err_msg = (
        "Discontiguity searches are currently only supported for "
        "2-dimensional coordinates."
    )
    if len(spatial_coords) != 2:
        raise NotImplementedError(dim_err_msg)

    # Check which dimensions are spanned by each coordinate.
    for coord in spatial_coords:
        if coord.ndim != 2:
            raise NotImplementedError(dim_err_msg)
        else:
            span = set(cube.coord_dims(coord))
        if not span:
            msg = "The coordinate {!r} doesn't span a data dimension."
            raise ValueError(msg.format(coord.name()))

    # Check that the 2d coordinate arrays are the same shape as each other
    if len(spatial_coords) == 2:
        assert spatial_coords[0].points.shape == spatial_coords[1].points.shape

    # Set up unmasked boolean array the same size as the coord points array:
    bad_points_boolean = np.zeros(spatial_coords[0].points.shape, dtype=bool)

    for coord in spatial_coords:
        _, (diffs_x, diffs_y) = coord._discontiguity_in_bounds(
            rtol=rel_tol, atol=abs_tol
        )

        bad_points_boolean[:, :-1] = np.logical_or(
            bad_points_boolean[:, :-1], diffs_x
        )
        # apply mask for y-direction discontiguities:
        bad_points_boolean[:-1, :] = np.logical_or(
            bad_points_boolean[:-1, :], diffs_y
        )
    return bad_points_boolean


def _mask_array(array, points_to_mask, in_place=False):
    """
    Apply masking to array where points_to_mask is True/non-zero.  Designed to
    work with iris.analysis.maths._binary_op_common so array and points_to_mask
    will be broadcastable to each other.  array and points_to_mask may be numpy
    or dask types (or one of each).

    If array is lazy then in_place is ignored: _math_op_common will use the
    returned value regardless of in_place, so we do not need to implement it
    here.  If in_place is True then array must be a
    :class:`numpy.ma.MaskedArray` or :class:`dask.array.Array`
    (must be a dask array if points_to_mask is lazy).

    """
    # Decide which array library to use.
    if is_lazy_data(points_to_mask) or is_lazy_data(array):
        al = da
        if not is_lazy_data(array) and in_place:
            # Non-lazy array and lazy mask should not come up for in_place
            # case, due to _binary_op_common handling added at #3790.
            raise TypeError(
                "Cannot apply lazy mask in-place to a non-lazy array."
            )
        in_place = False

    elif in_place and not isinstance(array, ma.MaskedArray):
        raise TypeError("Cannot apply a mask in-place to a plain numpy array.")
    else:
        al = np

    points_to_mask = points_to_mask.astype(bool)

    # Treat any masked points on our mask as False.
    points_to_mask = al.ma.filled(points_to_mask, False)

    # Get broadcasted views of the arrays.  Note that broadcast_arrays does not
    # preserve masks, so we need to explicitly handle any exising mask on array.
    array_mask = al.ma.getmaskarray(array)

    array_data, array_mask, points_to_mask = al.broadcast_arrays(
        array, array_mask, points_to_mask
    )

    new_mask = al.logical_or(array_mask, points_to_mask)

    if in_place:
        array.mask = new_mask
        result = array  # Resolve uses returned value even if working in place.
    else:
        # Return a new, independent array.
        result = al.ma.masked_array(array_data.copy(), mask=new_mask)

    return result


@_lenient_client(services=SERVICES)
def mask_cube(cube, points_to_mask, in_place=False, dim=None):
    """
    Masks any cells in the cube's data array which correspond to cells marked
    ``True`` (or non zero) in ``points_to_mask``.  ``points_to_mask`` may be
    specified as a :class:`numpy.ndarray`, :class:`iris.coords.Coord` or
    :class:`iris.cube.Cube`, following the same broadcasting approach as cube
    arithmetic (see :ref:`cube maths`).

    Parameters
    ----------

    cube : iris.cube.Cube
        Cube containing data that requires masking.

    points_to_mask : numpy.ndarray, iris.coords.Coord or iris.cube.Cube
        Specifies booleans (or ones and zeros) indicating which points will be masked.

    in_place : bool, default=False
        If `True`, masking is applied to the input cube.  Otherwise a copy is masked
        and returned.

    dim : int, optional
        If `points_to_mask` is a coord which does not exist on the cube, specify the
        dimension to which it should be mapped.

    Returns
    -------

    iris.cube.Cube
        A cube whose data array is masked at points specified by ``points_to_mask``.

    Notes
    -----

    If either ``cube`` or ``points_to_mask`` is lazy, the result will be lazy.

    """
    if in_place and not cube.has_lazy_data():
        # Ensure cube data is masked type so we can work on it in-place.
        cube.data = ma.asanyarray(cube.data)
        mask_function = functools.partial(_mask_array, in_place=True)
    else:
        mask_function = _mask_array

    input_metadata = cube.metadata
    result = iris.analysis.maths._binary_op_common(
        mask_function,
        "mask",
        cube,
        points_to_mask,
        cube.units,
        in_place=in_place,
        dim=dim,
        sanitise_metadata=False,
    )

    # Resolve combines the metadata from the two operands, but we want to
    # preserve the metadata from the (first) input cube.
    result.metadata = input_metadata

    if not in_place:
        return result


def equalise_attributes(cubes):
    """
    Delete cube attributes that are not identical over all cubes in a group.

    This function deletes any attributes which are not the same for all the
    given cubes.  The cubes will then have identical attributes, and the
    removed attributes are returned.  The given cubes are modified in-place.

    Args:

    * cubes (iterable of :class:`iris.cube.Cube`):
        A collection of cubes to compare and adjust.

    Returns:

    * removed (list):
        A list of dicts holding the removed attributes.

    """
    removed = []
    # Work out which attributes are identical across all the cubes.
    common_keys = list(cubes[0].attributes.keys())
    keys_to_remove = set(common_keys)
    for cube in cubes[1:]:
        cube_keys = list(cube.attributes.keys())
        keys_to_remove.update(cube_keys)
        common_keys = [
            key
            for key in common_keys
            if (
                key in cube_keys
                and np.all(cube.attributes[key] == cubes[0].attributes[key])
            )
        ]
    keys_to_remove.difference_update(common_keys)

    # Remove all the other attributes.
    for cube in cubes:
        deleted_attributes = {
            key: cube.attributes.pop(key)
            for key in keys_to_remove
            if key in cube.attributes
        }
        removed.append(deleted_attributes)
    return removed


def is_masked(array):
    """
    Equivalent to :func:`numpy.ma.is_masked`, but works for both lazy AND realised arrays.

    Parameters
    ----------
    array : :class:`numpy.Array` or :class:`dask.array.Array`
            The array to be checked for masks.

    Returns
    -------
    bool
        Whether or not the array has any masks.

    """
    if is_lazy_data(array):
        result = da.ma.getmaskarray(array).any().compute()
    else:
        result = ma.is_masked(array)
    return result


def _strip_metadata_from_dims(cube, dims):
    """
    Remove ancillary variables and cell measures that map to specific dimensions.

    Returns a cube copy with (possibly) some cell-measures and ancillary variables removed.

    To be used by operations that modify or remove dimensions.
    Note: does nothing to (aux)-coordinates.  Those would be handled explicitly by the calling operation.

    """
    reduced_cube = cube.copy()

    # Remove any ancillary variables that span the dimension(s) being collapsed
    for ancil in reduced_cube.ancillary_variables():
        ancil_dims = reduced_cube.ancillary_variable_dims(ancil)
        if set(dims).intersection(ancil_dims):
            reduced_cube.remove_ancillary_variable(ancil)

    # Remove any cell measures that span the dimension(s) being collapsed
    for cm in reduced_cube.cell_measures():
        cm_dims = reduced_cube.cell_measure_dims(cm)
        if set(dims).intersection(cm_dims):
            reduced_cube.remove_cell_measure(cm)

    return reduced_cube
