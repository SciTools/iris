# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
r"""Identification of multi-dimensional structure in a flat sequence of homogeneous objects.

The purpose of this module is to provide utilities for the identification
of multi-dimensional structure in a flat sequence of homogeneous objects.

One application of this is to efficiently identify a higher dimensional
structure from a sorted sequence of PPField instances; for an example, given
a list of 12 PPFields, identification that there are 3 unique "time" values
and 4 unique "height" values where time and height are linearly independent
means that we could construct a resulting cube with a shape of
``(3, 4) + <shape of a single field>``.

An example using numpy arrays:

    >>> import numpy as np
    >>> orig_x, orig_y = np.arange(2), np.arange(3)
    >>> x, y = np.meshgrid(orig_x, orig_y)

    >>> # Remove the dimensional structure from the arrays.
    >>> x, y = x.flatten(), y.flatten()

    >>> print(x)
    [0 1 0 1 0 1]
    >>> print(y)
    [0 0 1 1 2 2]

    >>> arrays = {'x': x, 'y': y}
    >>> group = GroupStructure.from_component_arrays(arrays)
    >>> print(group)
    Group structure:
      Length: 6
      Element names: x, y
      Possible structures ("c" order):
        (y: 3; x: 2)

    >>> built_arrays = group.build_arrays((3, 2), arrays)
    >>> y_array, y_axes = built_arrays['y']
    >>> print(y_array, y_axes)
    [0 1 2] (0,)

"""

from collections import namedtuple

import numpy as np


class _UnstructuredArrayException(Exception):
    """Raised when an array has been incorrectly assumed to be structured in a specific way."""


class ArrayStructure(namedtuple("ArrayStructure", ["stride", "unique_ordered_values"])):
    """Represent the identified structure of an array.

    Represents the identified structure of an array, where stride is the
    step between each unique value being seen in order in the flattened
    version of the array.

    Note: Stride is **not** in bytes, but is instead the number of objects in
    the original list of arrays, thus, stride is dtype independent.

    For column major (aka "F" order) arrays, stride will be one for those
    arrays which vary over the first dimension, and conversely will be one for
    C order arrays when varying over the last dimension.

    Constructing an ArrayStructure is most frequently done through the
    :meth:`ArrayStructure.from_array` class method, which takes a flattened
    array as its input.

    Stride examples:

    >>> ArrayStructure.from_array(np.array([1, 2])).stride
    1
    >>> ArrayStructure.from_array(np.array([1, 1, 2, 2])).stride
    2
    >>> ArrayStructure.from_array(np.array([1, 1, 1, 2, 2, 2])).stride
    3
    >>> ArrayStructure.from_array(np.array([1, 2, 1, 2])).stride
    1
    >>> ArrayStructure.from_array(np.array([1, 1, 2, 2, 1, 1, 2, 2])).stride
    2

    """

    def __new__(cls, stride, unique_ordered_values):
        self = super().__new__(cls, stride, unique_ordered_values)
        return self

    __slots__ = ()

    @property
    def size(self):
        """Number of unique values in the original array.

        The ``size`` attribute is the number of the unique values in the
        original array. It is **not** the length of the original array.

        """
        return len(self.unique_ordered_values)

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        stride = getattr(other, "stride", None)
        arr = getattr(other, "unique_ordered_values", None)

        result = NotImplemented
        if stride is not None or arr is not None:
            result = stride == self.stride and np.all(self.unique_ordered_values == arr)
        return result

    def __ne__(self, other):
        return not (self == other)

    def construct_array(self, size):
        """Build 1D array.

        The inverse operation of :func:`ArrayStructure.from_array`, returning
        a 1D array of the given length with the appropriate repetition
        pattern.

        """
        return np.tile(
            np.repeat(self.unique_ordered_values, self.stride),
            size // (self.size * self.stride),
        )

    def nd_array_and_dims(self, original_array, target_shape, order="c"):
        """Given a 1D array and a target shape, construct an ndarray and associated dimensions.

        Raises an _UnstructuredArrayException if no optimised shape array can
        be returned, in which case, simply reshaping the original_array would
        be just as effective.

        For example:

        >>> orig = np.array([1, 2, 3, 1, 2, 3])
        >>> structure = ArrayStructure.from_array(orig)
        >>> array, dims = structure.nd_array_and_dims(orig, (2, 1, 3))
        >>> array
        array([1, 2, 3])
        >>> dims
        (2,)
        >>> # Filling the array with dimensions of length one should impact
        >>> # dims but not the array which is returned.
        >>> _, dims = structure.nd_array_and_dims(orig, (1, 2, 1, 3, 1))
        >>> dims
        (3,)

        """
        if original_array.shape[0] != np.prod(target_shape):
            raise ValueError("Original array and target shape do not match up.")
        stride_product = 1

        result = None

        if self.size == 1:
            # There is no need to even consider the dimensionality - this
            # array structure only has one unique value, so it is a scalar
            # and has no associated dimension.
            result = (np.array(original_array[0]), ())

        for dim, length in sorted(
            enumerate(target_shape), reverse=order.lower() == "c"
        ):
            if result is not None:
                break

            # Does this array structure represent a flattened array of the
            # given shape? If so, reshape it back to the target shape,
            # then index out any dimensions which are constant.
            if self.stride == stride_product and length == self.size:
                vector = original_array.reshape(target_shape + (-1,), order=order)
                # Reduce the dimensionality to a 1d array by indexing
                # everything but this dimension.
                vector = vector[
                    tuple(
                        0 if dim != i else slice(None) for i in range(len(target_shape))
                    )
                ]
                # Remove any trailing dimension if it is trivial.
                if len(vector.shape) != 1 and vector.shape[-1] == 1:
                    vector = vector[..., 0]

                result = [vector, (dim,)]
                break

            stride_product *= length

        if result is not None:
            return result
        else:
            msg = (
                "Unable to construct an efficient nd_array for the target "
                "shape. Consider reshaping the array to the full shape "
                "instead."
            )
            raise _UnstructuredArrayException(msg)

    @classmethod
    def from_array(cls, arr):
        """Return the computed ArrayStructure for the given flat array.

        Return the computed ArrayStructure for the given flat array
        (if a structure exists, otherwise return None).

        """
        # Note: This algorithm will only find distinct value columns/rows/axes
        # any dimension with repeat values will not have its structure
        # identified and will be considered irregular.

        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        if arr.ndim != 1:
            raise ValueError("The given array must be 1D.")

        if arr.size == 0:
            return cls(1, arr)

        # unique is a *sorted* array of unique values.
        # unique_inds takes us from the sorted unique values back to inds in
        # the input array inds_back_to_orig gives us the indices of each value
        # in the array vs the index in the *sorted* unique array.
        _, unique_inds = np.unique(arr, return_index=True)

        # Return the unique values back into an ordered array.
        unique = arr[np.sort(unique_inds)]

        # what we actually want is inds_back_to_orig in the sort order of the
        # original array.
        new_inds = np.empty(arr.shape, dtype=unique_inds.dtype)

        for ind, unique_val in enumerate(unique):
            new_inds[arr == unique_val] = ind

        inds_back_to_orig = new_inds

        u_len = len(unique)
        n_fields = arr.size

        structure = None

        # If the length of the unique values is not a divisor of the
        # length of the original array, it is going to be an irregular
        # array, so we can avoid some processing.
        if (n_fields % u_len) != 0:
            # No structure.
            pass
        # Shortcut the simple case of all values being distinct.
        elif u_len == 1:
            structure = ArrayStructure(1, unique)
        else:
            # Working in index space, compute indices where values change.
            ind_diffs = np.diff(inds_back_to_orig)

            # Find the indices where a change takes place.
            ind_diffs_which_changed = np.nonzero(ind_diffs)[0]

            # Any index which changed by a different consecutive amount is a
            # stride. For example, an input array of [1,1,2,2,1,1,2,2]
            # results in ind_diffs looking like [0,1,0,-1,0,1,0] and
            # ind_diffs_which_changed being [1,3,5]. So now identifying the
            # "stride" (being the length of any sequence which has
            # consecutively equal values) is a matter of identifying the
            # difference between any two consecutive values from
            # ind_diffs_which_changed. If we don't have enough
            # ind_diffs_which_changed values to compute a difference, then
            # there is either one or two distinct values in the original
            # array, and the stride is therefore the total length / number
            # of unique values.
            try:
                stride = np.diff(ind_diffs_which_changed[:2])[0]
            except IndexError:
                stride = n_fields // u_len

            structure = cls(stride, unique)

            # Do one last sanity check - does the array we've just described
            # actually compute the correct array?
            constructed_array = structure.construct_array(arr.size)
            if not np.all(constructed_array == arr):
                structure = None

        return structure


class GroupStructure:
    """Represent a collection of array structures.

    The GroupStructure class represents a collection of array structures along
    with additional information such as the length of the arrays and the array
    order in which they are found (row-major or column-major).

    """

    def __init__(self, length, component_structure, array_order="c"):
        """Group_component_to_array - a dictionary. See also TODO."""
        self.length = length
        """The size common to all of the original arrays.
           Used to determine possible shape configurations."""

        self._cmpt_structure = component_structure
        """A dictionary mapping component name to ArrayStructure instance
          (or None if no such structure exists for that component)."""

        array_order = array_order.lower()
        if array_order not in ["c", "f"]:
            raise ValueError("Invalid array order {!r}".format(array_order))
        self._array_order = array_order

    @classmethod
    def from_component_arrays(cls, component_arrays, array_order="c"):
        """From component arrays.

        Given a dictionary of component name to flattened numpy array,
        return an :class:`GroupStructure` instance which is representative
        of the underlying array structures.

        Parameters
        ----------
        component_arrays :
            A dictionary mapping component name to the full sized 1d (flattened)
            numpy array.

        """
        cmpt_structure = {
            name: ArrayStructure.from_array(array)
            for name, array in component_arrays.items()
        }

        sizes = np.array([array.size for array in component_arrays.values()])
        if not np.all(sizes == sizes[0]):
            raise ValueError("All array elements must have the same size.")

        return cls(sizes[0], cmpt_structure, array_order=array_order)

    def _potentially_flattened_components(self):
        """Return a generator of the components which could form non-trivial.

        (i.e. ``length > 1``) array dimensions.

        """
        for name, structure in self._cmpt_structure.items():
            if structure is not None and structure.size > 1:
                yield (name, structure)

    @property
    def is_row_major(self):
        return self._array_order == "c"

    def possible_structures(self):
        """Return a tuple containing the possible structures that this group could have.

        A structure in this case is an iterable of
        ``(name, ArrayStructure)`` pairs, one per dimension, of a possible
        array. The shape of the resulting array would then be
        ``tuple(array_struct.size for (name, array_struct) in pair)`` for any
        of the returned structures.

        The algorithm does not deal with incomplete structures, such that
        all components critical for the identification of a shape are
        necessary.

        """
        vector_structures = sorted(self._potentially_flattened_components())

        def filter_strides_of_length(length):
            return [
                (name, struct)
                for (name, struct) in vector_structures
                if struct.stride == length
            ]

        # Keep track of our structures so far. This will be a list of lists
        # containing the actual structure pairs.
        possible = []

        # Get hold of all array structures with a stride of 1. These are the
        # only possible array structures for the first dimension (i.e. on the
        # left-most dimension for column-major ordering). Start a list of
        # structures, one for each possible left hand side dimension
        # component.
        for structure in filter_strides_of_length(1):
            possible.append([structure])

        # Keep track of all structures which are valid, these are ultimately
        # what will be returned from this function.
        allowed_structures = []

        # We will make use of the mutability of the possible list, removing
        # each list representing a potential structure. With the potential
        # just removed, we will find any array structures which could be the
        # next dimension in the potential, adding one new possible structure
        # per array structure found. If at any point, a possible's stride
        # product is the same as the length of this group, we've got an
        # allowed structure.
        while possible:
            for potential in possible[:]:
                possible.remove(potential)
                # If we are to build another dimension on top of this possible
                # structure, we need to compute the stride that would be
                # needed for that dimension.
                next_stride = np.prod([struct.size for (_, struct) in potential])

                # If we've found a structure whose product is the length of
                # the fields of this Group, we've got a valid potential.
                if next_stride == self.length:
                    allowed_structures.append(potential)

                # So let's get all of the potential nd-arrays which would be
                # viable dimensions on this potential structure.
                next_dim_structs = filter_strides_of_length(next_stride)

                # Any we find get added to this potential and put back in the
                # possibles.
                for struct in next_dim_structs:
                    if struct in potential:
                        continue
                    new_potential = potential[:]
                    # Add the structure to the potential on the right hand
                    # side of the structure (column-major).
                    new_potential.append(struct)
                    possible.append(new_potential)

        # We've been working in column-major order, so let's reverse the
        # dimensionality if we are in row-major.
        if self.is_row_major:
            for potential in allowed_structures:
                potential.reverse()

        return tuple(allowed_structures)

    def __str__(self):
        result = [
            "Group structure:",
            "  Length: {}".format(self.length),
            "  Element names: {}".format(
                ", ".join(sorted(self._cmpt_structure.keys()))
            ),
            '  Possible structures ("{}" order):'.format(self._array_order),
        ]

        for structure in self.possible_structures():
            sizes = (
                "{}: {}".format(name, arr_struct.size) for name, arr_struct in structure
            )
            result.append("    ({})".format("; ".join(sizes)))

        return "\n".join(result)

    def build_arrays(self, shape, elements_arrays):
        """Build Arrays.

        Given the target shape, and a dictionary mapping name to 1D array of
        :attr:`.length`, return a dictionary mapping element name to
        ``(ndarray, dims)``.

        Note: Actually the arrays may be more than one dimension, and the
        trailing dimensions will be preserved. This is useful for items such
        as datetimes, where an efficiency exists to avoid the construction
        of datetime objects until the last moment.

        """
        elem_to_nd_and_dims = {}

        if sorted(elements_arrays.keys()) != sorted(self._cmpt_structure):
            raise ValueError(
                "The GroupStructure elements were not the same "
                "as those provided in the element_arrays."
            )

        for name, array in elements_arrays.items():
            struct = self._cmpt_structure[name]
            nd_array_and_dims = None
            if struct is not None:
                try:
                    nd_array_and_dims = struct.nd_array_and_dims(
                        array, shape, order=self._array_order
                    )
                except _UnstructuredArrayException:
                    pass

            if nd_array_and_dims is None:
                reshape_shape = shape
                if array.ndim > 1:
                    reshape_shape = reshape_shape + (-1,)
                nd_array_and_dims = [
                    array.reshape(reshape_shape, order=self._array_order),
                    tuple(range(len(shape))),
                ]
            elem_to_nd_and_dims[name] = nd_array_and_dims
        return elem_to_nd_and_dims
