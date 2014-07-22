# (C) British Crown Copyright 2014, Met Office
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

import itertools

from netCDF4 import netcdftime
import numpy as np

from iris.fileformats.um._optimal_array_structuring import \
    optimal_array_structure

from iris.fileformats.pp import PPField3


# Static values for the _time_comparable_int routine.
_TIME_ELEMENT_MULTIPLIERS = np.cumprod([1, 60, 60, 24, 31, 12])[::-1]


def _time_comparable_int(yr, mon, dat, hr, min, sec):
    """
    Return a simple number representing a date-time tuple.

    This calculation takes no account of the real calendar, but instead just
    gives every month 31 days, which preserves the required time ordering.

    In fact... for the purposes used here we didn't really neded ordering,
    it is merely important that all datetimes yield a *unique* conversion.

    """
    elements = np.array((yr, mon, dat, hr, min, sec))
    return np.sum(elements * _TIME_ELEMENT_MULTIPLIERS)


class FieldCollation(object):
    """
    An object representing a group of UM fields with array structure that can
    be vectorized into a single cube.

    """
    def __init__(self, fields):
        """
        Kwargs
        ------
        fields - a iterable of PPField instances. All instances must be of the
                 same type (i.e. must all be PPField3/PPField2).
                 Fields are immutable.
        """
        self._fields = tuple(fields)
        assert len(self.fields) > 0
        self._structure_calculated = False
        self._vector_dims_shape = None
        self._primary_dimension_elements = None
        self._element_arrays_and_dims = None

    @property
    def fields(self):
        return self._fields

    @property
    def vector_dims_shape(self):
        if not self._structure_calculated:
            self._calculate_structure()
        return self._vector_dims_shape

    @property
    def primary_dimension_elements(self):
        if not self._structure_calculated:
            self._calculate_structure()
        return self._primary_dimension_elements

    @property
    def element_arrays_and_dims(self):
        if not self._structure_calculated:
            self._calculate_structure()
        return self._element_arrays_and_dims

    @staticmethod
    def _field_vector_element_arrays(fields):
        """"Define the field components used in the structure analysis."""
        # First define functions to access t1 and t2 as date-time tuples.
        # -- this depends on the PPField header version.
        if isinstance(fields[0], PPField3):
            # PPField3 store times to one second.
            t1_fn = lambda fld: (fld.lbyr, fld.lbmon, fld.lbdat,
                                 fld.lbhr, fld.lbmin, fld.lbsec)
            t2_fn = lambda fld: (fld.lbyrd, fld.lbmond, fld.lbdatd,
                                 fld.lbhrd, fld.lbmind, fld.lbsecd)
        else:
            # PPField2 has no seconds elements, fill these with zero.
            t1_fn = lambda fld: (fld.lbyr, fld.lbmon, fld.lbdat,
                                 fld.lbhr, fld.lbmin, 0)
            t2_fn = lambda fld: (fld.lbyrd, fld.lbmond, fld.lbdatd,
                                 fld.lbhrd, fld.lbmind, 0)

        # Return a list of (name, array) for the vectorizable elements.
        component_arrays = [
            ('t1', np.array([t1_fn(fld) for fld in fields])),
            ('t2', np.array([t2_fn(fld) for fld in fields])),
            ('lbft', np.array([fld.lbft for fld in fields])),
            ('blev', np.array([fld.blev for fld in fields])),
            ('lbrsvd4', np.array([fld.lbrsvd[3] for fld in fields])),
            ('lbuser5', np.array([fld.lbuser[4] for fld in fields]))]

        return component_arrays

    def _calculate_structure(self):
        # Make value arrays for the vectorisable field elements.
        element_definitions = self._field_vector_element_arrays(self.fields)

        # Make a copy with time value tuples replaced by integers.
        ordering_definitions = element_definitions[:]
        for index, (name, array) in enumerate(ordering_definitions):
            if name in ('t1', 't2'):
                array = np.array(
                    [_time_comparable_int(*tuple(val)) for val in array])
                ordering_definitions[index] = (name, array)

        # Get a list of the original value arrays for building results.
        actual_value_arrays = [array for name, array in element_definitions]

        # Perform the main analysis --> vector dimensions, elements, arrays.
        dims_shape, primary_elements, vector_element_arrays_and_dims = \
            optimal_array_structure(ordering_definitions,
                                    actual_value_arrays)

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
    """ Standard collation key definition for fast structured field loading."""
    return (field.lbuser[3], field.lbproc, field.lbuser[6])


def group_structured_fields(field_iterator):
    """
    Collect structured fields into identified groups whose fields can be
    combined to form a single cube.

    Args:

    * field_iterator (iterator of :class:`iris.fileformats.pp.PPField`):
        A source of PP or FF fields.  N.B. order is significant.

    The function sorts + collates on phenomenon-relevant metadata only, defined
    as the field components : 'lbuser[3]', 'lbuser[6]' and 'lbproc'.
    Each distinct combination of these defines a specific phenomenon (or
    statistical aggregation of one), and those fields appear as a single
    iteration result.

    Implicitly, within each result group, _all_ other metadata components
    should be either -
    (1) the same for all fields, or
    (2) completely irrelevant, or
    (3) used by a vectorised rule function, such as
        :func:`iris.fileformats.pp_rules._convert_vector_time_coords`.

    Returns:
        An iterator yielding
        :class:`~iris.experimental.fileformats.um.FieldCollation` objects, each
        of which contains a single collated group from the input fields.

    """
    fields = sorted(field_iterator, key=_um_collation_key_function)
    for _, fields in itertools.groupby(fields, _um_collation_key_function):
        yield FieldCollation(tuple(fields))
