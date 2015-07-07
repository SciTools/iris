# (C) British Crown Copyright 2014 - 2015, Met Office
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
"""Experimental code for fast loading of structured UM data."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

import itertools

from netCDF4 import netcdftime
import numpy as np

from iris.fileformats.um._optimal_array_structuring import \
    optimal_array_structure

from biggus import ArrayStack
from iris.fileformats.pp import PPField3


class FieldCollation(object):
    """
    An object representing a group of UM fields with array structure that can
    be vectorized into a single cube.

    For example:

    Suppose we have a set of 28 fields repeating over 7 vertical levels for
    each of 4 different data times.  If a FieldCollation is created to contain
    these, it can identify that this is a 4*7 regular array structure.

    This FieldCollation will then have the following properties:

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
            data_arrays = [f._data for f in self.fields]
            self._data_cache = \
                ArrayStack.multidim_array_stack(data_arrays,
                                                self.vector_dims_shape)
        return self._data_cache

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
        t1_fn = lambda fld: (fld.lbyr, fld.lbmon, fld.lbdat,
                             fld.lbhr, fld.lbmin, getattr(fld, 'lbsec', 0))
        t2_fn = lambda fld: (fld.lbyrd, fld.lbmond, fld.lbdatd,
                             fld.lbhrd, fld.lbmind, getattr(fld, 'lbsecd', 0))

        # Return a list of (name, array) for the vectorizable elements.
        component_arrays = [
            ('t1', np.array([t1_fn(fld) for fld in self.fields])),
            ('t2', np.array([t2_fn(fld) for fld in self.fields])),
            ('lbft', np.array([fld.lbft for fld in self.fields])),
            ('blev', np.array([fld.blev for fld in self.fields])),
            ('lblev', np.array([fld.lblev for fld in self.fields])),
            ('bhlev', np.array([fld.bhlev for fld in self.fields])),
            ('bhrlev', np.array([fld.bhrlev for fld in self.fields])),
            ('brsvd1', np.array([fld.brsvd[0] for fld in self.fields])),
            ('brsvd2', np.array([fld.brsvd[1] for fld in self.fields])),
            ('brlev', np.array([fld.brlev for fld in self.fields]))
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
        blev_array = dict(element_definitions).get('blev')
        vertical_elements = ('lblev', 'bhlev', 'bhrlev',
                             'brsvd1', 'brsvd2', 'brlev')

        # Make an ordering copy.
        ordering_definitions = element_definitions[:]
        # Replace time value tuples with integers and bind the vertical
        # elements to the (expected) primary vertical element "blev".
        for index, (name, array) in enumerate(ordering_definitions):
            if name in ('t1', 't2'):
                array = np.array(
                    [self._time_comparable_int(*tuple(val)) for val in array])
                ordering_definitions[index] = (name, array)
            if name in vertical_elements and blev_array is not None:
                ordering_definitions[index] = (name, blev_array)

        # Perform the main analysis: get vector dimensions, elements, arrays.
        dims_shape, primary_elements, vector_element_arrays_and_dims = \
            optimal_array_structure(ordering_definitions,
                                    element_definitions)

        # Replace time tuples in the result with real datetime-like values.
        # N.B. so we *don't* do this on the whole (expanded) input arrays.
        for name in ('t1', 't2'):
            if name in vector_element_arrays_and_dims:
                arr, dims = vector_element_arrays_and_dims[name]
                arr_shape = arr.shape[:-1]
                extra_length = arr.shape[-1]
                # Flatten out the array apart from the last dimension,
                # convert to netcdftime objects, then reshape back.
                arr = np.array([netcdftime.datetime(*args)
                                for args in arr.reshape(-1, extra_length)]
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
    return (field.lbuser[3], field.lbproc, field.lbuser[6])


def group_structured_fields(field_iterator):
    """
    Collect structured fields into identified groups whose fields can be
    combined to form a single cube.

    Args:

    * field_iterator (iterator of :class:`iris.fileformats.pp.PPField`):
        A source of PP or FF fields.  N.B. order is significant.

    The function sorts and collates on phenomenon-relevant metadata only,
    defined as the field components: 'lbuser[3]', 'lbuser[6]' and 'lbproc'.
    Each distinct combination of these defines a specific phenomenon (or
    statistical aggregation of one), and those fields appear as a single
    iteration result.

    Implicitly, within each result group, *all* other metadata components
    should be either:

    *  the same for all fields,
    *  completely irrelevant, or
    *  used by a vectorised rule function (such as
       :func:`iris.fileformats.pp_rules._convert_vector_time_coords`).

    Returns:
        An generator of
        :class:`~iris.experimental.fileformats.um.FieldCollation` objects,
        each of which contains a single collated group from the input fields.

    """
    _fields = sorted(field_iterator, key=_um_collation_key_function)
    for _, fields in itertools.groupby(_fields, _um_collation_key_function):
        yield FieldCollation(tuple(fields))
