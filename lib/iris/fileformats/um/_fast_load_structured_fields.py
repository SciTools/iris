# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Code for fast loading of structured UM data.

This module defines which pp-field elements take part in structured loading,
and provides creation of :class:`BasicFieldCollation` objects from lists of
:class:`iris.fileformats.pp.PPField`.

"""

import itertools

import cftime
import numpy as np

from iris._lazy_data import as_lazy_data, multidim_lazy_stack
from iris.fileformats.um._optimal_array_structuring import (
    optimal_array_structure,
)


class BasicFieldCollation:
    """
    An object representing a group of UM fields with array structure that can
    be vectorized into a single cube.

    For example:

    Suppose we have a set of 28 fields repeating over 7 vertical levels for
    each of 4 different data times.  If a BasicFieldCollation is created to
    contain these, it can identify that this is a 4*7 regular array structure.

    This BasicFieldCollation will then have the following properties:

    * within 'element_arrays_and_dims' :
        Element 'blev' have the array shape (7,) and dims of (1,).
        Elements 't1' and 't2' have shape (4,) and dims (0,).
        The other elements (lbft, lbrsvd4 and lbuser5) all have scalar array
        values and dims=None.

    .. note::

        If no array structure is found, the element values are all
        either scalar or full-length 1-D vectors.

    """

    def __init__(self, fields):
        """
        Args:

        * fields (iterable of :class:`iris.fileformats.pp.PPField`):
            The fields in the collation.

        """
        self._fields = tuple(fields)
        self._data_cache = None
        assert len(self.fields) > 0
        self._structure_calculated = False
        self._vector_dims_shape = None
        self._primary_dimension_elements = None
        self._element_arrays_and_dims = None

    @property
    def fields(self):
        return self._fields

    @property
    def data(self):
        if not self._structure_calculated:
            self._calculate_structure()
        if self._data_cache is None:
            stack = np.empty(self.vector_dims_shape, "object")
            for nd_index, field in zip(
                np.ndindex(self.vector_dims_shape), self.fields
            ):
                stack[nd_index] = as_lazy_data(field._data)
            self._data_cache = multidim_lazy_stack(stack)
        return self._data_cache

    def core_data(self):
        return self.data

    @property
    def realised_dtype(self):
        return np.result_type(
            *[field.realised_dtype for field in self._fields]
        )

    @property
    def data_proxy(self):
        return self.data

    @property
    def bmdi(self):
        bmdis = set([f.bmdi for f in self.fields])
        if len(bmdis) != 1:
            raise ValueError("Multiple bmdi values defined in FieldCollection")
        return bmdis.pop()

    @property
    def vector_dims_shape(self):
        """The shape of the array structure."""
        if not self._structure_calculated:
            self._calculate_structure()
        return self._vector_dims_shape

    @property
    def _UNUSED_primary_dimension_elements(self):
        """A set of names of the elements which are array dimensions."""
        if not self._structure_calculated:
            self._calculate_structure()
        return self._primary_dimension_elements

    @property
    def element_arrays_and_dims(self):
        """
        Value arrays for vector metadata elements.

        A dictionary mapping element_name: (value_array, dims).

        The arrays are reduced to their minimum dimensions.  A scalar array
        has an associated 'dims' of None (instead of an empty tuple).

        """
        if not self._structure_calculated:
            self._calculate_structure()
        return self._element_arrays_and_dims

    def _field_vector_element_arrays(self):
        """Define the field components used in the structure analysis."""
        # Define functions to make t1 and t2 values as date-time tuples.
        # These depend on header version (PPField2 has no seconds values).
        def t1_fn(fld):
            return (
                fld.lbyr,
                fld.lbmon,
                fld.lbdat,
                fld.lbhr,
                fld.lbmin,
                getattr(fld, "lbsec", 0),
            )

        def t2_fn(fld):
            return (
                fld.lbyrd,
                fld.lbmond,
                fld.lbdatd,
                fld.lbhrd,
                fld.lbmind,
                getattr(fld, "lbsecd", 0),
            )

        # Return a list of (name, array) for the vectorizable elements.
        component_arrays = [
            ("t1", np.array([t1_fn(fld) for fld in self.fields])),
            ("t2", np.array([t2_fn(fld) for fld in self.fields])),
            ("lbft", np.array([fld.lbft for fld in self.fields])),
            ("blev", np.array([fld.blev for fld in self.fields])),
            ("lblev", np.array([fld.lblev for fld in self.fields])),
            ("bhlev", np.array([fld.bhlev for fld in self.fields])),
            ("bhrlev", np.array([fld.bhrlev for fld in self.fields])),
            ("brsvd1", np.array([fld.brsvd[0] for fld in self.fields])),
            ("brsvd2", np.array([fld.brsvd[1] for fld in self.fields])),
            ("brlev", np.array([fld.brlev for fld in self.fields])),
        ]
        return component_arrays

    # Static factors for the _time_comparable_int routine (seconds per period).
    _TIME_ELEMENT_MULTIPLIERS = np.cumprod([1, 60, 60, 24, 31, 12])[::-1]

    def _time_comparable_int(self, yr, mon, dat, hr, min, sec):
        """
        Return a single unique number representing a date-time tuple.

        This calculation takes no account of the time field's real calendar,
        instead giving every month 31 days, which preserves the required
        time ordering.

        """
        elements = np.array((yr, mon, dat, hr, min, sec))
        return np.sum(elements * self._TIME_ELEMENT_MULTIPLIERS)

    def _calculate_structure(self):
        # Make value arrays for the vectorisable field elements.
        element_definitions = self._field_vector_element_arrays()

        # Identify the vertical elements and payload.
        blev_array = dict(element_definitions).get("blev")
        vertical_elements = (
            "lblev",
            "bhlev",
            "bhrlev",
            "brsvd1",
            "brsvd2",
            "brlev",
        )

        # Make an ordering copy.
        ordering_definitions = element_definitions[:]
        # Replace time value tuples with integers and bind the vertical
        # elements to the (expected) primary vertical element "blev".
        for index, (name, array) in enumerate(ordering_definitions):
            if name in ("t1", "t2"):
                array = np.array(
                    [self._time_comparable_int(*tuple(val)) for val in array]
                )
                ordering_definitions[index] = (name, array)
            if name in vertical_elements and blev_array is not None:
                ordering_definitions[index] = (name, blev_array)

        # Perform the main analysis: get vector dimensions, elements, arrays.
        (
            dims_shape,
            primary_elements,
            vector_element_arrays_and_dims,
        ) = optimal_array_structure(ordering_definitions, element_definitions)

        # Replace time tuples in the result with real datetime-like values.
        # N.B. so we *don't* do this on the whole (expanded) input arrays.
        for name in ("t1", "t2"):
            if name in vector_element_arrays_and_dims:
                arr, dims = vector_element_arrays_and_dims[name]
                arr_shape = arr.shape[:-1]
                extra_length = arr.shape[-1]
                # Flatten out the array apart from the last dimension,
                # convert to cftime objects, then reshape back.
                arr = np.array(
                    [
                        cftime.datetime(*args)
                        for args in arr.reshape(-1, extra_length)
                    ]
                ).reshape(arr_shape)
                vector_element_arrays_and_dims[name] = (arr, dims)

        # Write the private cache values, exposed as public properties.
        self._vector_dims_shape = dims_shape
        self._primary_dimension_elements = primary_elements
        self._element_arrays_and_dims = vector_element_arrays_and_dims
        # Do all this only once.
        self._structure_calculated = True


def _um_collation_key_function(field):
    """
    Standard collation key definition for fast structured field loading.

    The elements used here are the minimum sufficient to define the
    'phenomenon', as described for :meth:`group_structured_fields`.

    """
    return (
        field.lbuser[3],  # stashcode first
        field.lbproc,  # then stats processing
        field.lbuser[6],  # then model
        field.lbuser[4],  # then pseudo-level : this one is a KLUDGE.
    )
    # NOTE: including pseudo-level here makes it treat different pseudo-levels
    # as different phenomena.  These will later be merged in the "ordinary"
    # post-load merge.
    # The current structured-load code fails to handle multiple pseudo-levels
    # correctly: because pseudo-level is not on in its list of "things that may
    # vary within a phenomenon", it will create a scalar pseudo-level
    # coordinate when it should have been a vector of values.
    # This kludge fixes that error, but it is inefficient because it bypasses
    # the structured load, producing n-levels times more 'raw' cubes.
    # It will also fail if any phenomenon occurs with multiple 'normal' levels
    # (i.e. lblev) *and* pseudo-levels (lbuser[4]).
    # TODO: it should be fairly easy to do this properly -- i.e. create a
    # vector pseudo-level coordinate directly in the structured load analysis.


def group_structured_fields(
    field_iterator, collation_class=BasicFieldCollation, **collation_kwargs
):
    """
    Collect structured fields into identified groups whose fields can be
    combined to form a single cube.

    Args:

    * field_iterator (iterator of :class:`iris.fileformats.pp.PPField`):
        A source of PP or FF fields.  N.B. order is significant.

    Kwargs:

    * collation_class (class):
        Type of collation wrapper to create from each group of fields.
    * collation_kwargs (dict):
        Additional constructor keywords for collation creation.

    The function sorts and collates on phenomenon-relevant metadata only,
    defined as the field components: 'lbuser[3]' (stash), 'lbproc' (statistic),
    'lbuser[6]' (model).
    Each distinct combination of these defines a specific phenomenon (or
    statistical aggregation of one), and those fields appear as a single
    iteration result.

    Implicitly, within each result group, *all* other metadata components
    should be either:

    *  the same for all fields,
    *  completely irrelevant, or
    *  used by a vectorised rule function (such as
       :func:`iris.fileformats.pp_load_rules._convert_time_coords`).

    Returns:
        A generator of 'collation_class' objects, each of which contains a
        single collated group from the input fields.

    .. note::

         At present, fields with different values of 'lbuser[4]' (pseudo-level)
         are *also* treated as different phenomena.  This is a temporary fix,
         standing in place for a more correct handling of pseudo-levels.

    """
    _fields = sorted(field_iterator, key=_um_collation_key_function)
    for _, fields in itertools.groupby(_fields, _um_collation_key_function):
        yield collation_class(tuple(fields), **collation_kwargs)
