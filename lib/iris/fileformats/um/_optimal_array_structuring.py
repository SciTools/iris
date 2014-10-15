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
"""A module to provide an optimal array structure calculation."""

import itertools

import numpy as np

from iris.fileformats._structured_array_identification import \
    GroupStructure


def _optimal_dimensioning_structure(structure, element_priorities):
    """
    Return the optimal array replication structure for a given
    :class:`~iris.fileformats._structured_array_identification.GroupStructure`.

    The optimal, in this case, is defined as that which produces the most
    number of non-trivial dimensions.

    May return an empty list if no structure can be identified.

    """
    permitted_structures = structure.possible_structures()
    if not permitted_structures:
        result = []
    else:
        #
        # TODO: this looks wrong to me (PP) ?
        #  Surely the product of the sizes is just the overall length ???
        #
        result = max(permitted_structures,
                     key=lambda potential: (
                         np.prod([struct.size
                                  for (name, struct) in potential]),
                         len(potential),
                         max(element_priorities[name]
                             for (name, struct) in potential)))
    return result


def optimal_array_structure(ordering_elements, actual_values_elements=None):
    """
    Calculate an optimal array replication structure for a set of vectors.

    Args:

    * ordering_elements (iterable of (name, 1-d array)):
        Input element names and value-vectors.  Must all be the same length
        (but not necessarily type).  Must have at least one.

    Kwargs:

    * actual_values_elements (iterable of (name, 1-d array)):
        The 'real' values used to construct the result arrays, if different
        from 'ordering_elements'.  Must contain  all the same names (but not
        necessarily in the same order).

    The 'ordering_elements' arg contains the pattern used to deduce a
    structure.  The order of this is significant, in that earlier elements get
    priority when associating dimensions with specific elements.

    Returns:

        dims_shape, primary_elements, element_arrays_and_dims

        * 'dims' is the shape of the vector dimensions chosen.

        * 'primary_elements' is a set of dimension names.
            Those input element names which are identified as dimensions:
            At most one for each dimension.

        * 'element_arrays_and_dims' is a dictionary "name --> (array, dims)"
            For all elements which are not dimensionless.  The 'array's are
            reduced to the shape of their mapped dimensions.

    For example::

        >>> import iris.fileformats.um._optimal_array_structuring as optdims
        >>> elements_structure = [('a', np.array([1, 1, 1, 2, 2, 2])),
        ...                       ('b', np.array([0, 1, 2, 0, 1, 2])),
        ...                       ('c', np.array([11, 12, 13, 14, 15, 16]))]
        >>> elements_values = [('a', np.array([10, 10, 10, 12, 12, 12])),
        ...                    ('b', np.array([15, 16, 17, 15, 16, 17])),
        ...                    ('c', np.array([9, 3, 5, 2, 7, 1]))]
        >>> dims_shape, dim_names, arrays_and_dims = \
        ...      optdims.optimal_array_structure(elements_structure,
        ...                                      elements_values)
        >>> print dims_shape
        (2, 3)
        >>> print dim_names
        set(['a', 'b'])
        >>> print arrays_and_dims
        {'a': (array([10, 12]), (0,)), 'c': (array([[9, 3, 5],
               [2, 7, 1]]), (0, 1)), 'b': (array([15, 16, 17]), (1,))}

    """
    # Convert the inputs to dicts.
    element_ordering_arrays = dict(ordering_elements)
    if actual_values_elements is None:
        actual_values_elements = element_ordering_arrays
    actual_value_arrays = dict(actual_values_elements)
    if set(actual_value_arrays.keys()) != set(element_ordering_arrays.keys()):
        raise ValueError("Names of 'actual_values_elements' do not match "
                         "those of the 'ordering_elements_arrays'.")

    # Define element priorities from ordering, to choose between equally good
    # structures, as structure code does not recognise any element ordering.
    n_elements = len(ordering_elements)
    element_priorities = {
        name: n_elements - index
        for index, (name, array) in enumerate(ordering_elements)}

    # Calculate the basic fields-group array structure.
    base_structure = GroupStructure.from_component_arrays(
        element_ordering_arrays)

    # Work out the target cube structure.
    target_structure = _optimal_dimensioning_structure(base_structure,
                                                       element_priorities)

    # Work out result cube dimensions.
    if not target_structure:
        # Get the length of an input array (they are all the same).
        elements_length = len(ordering_elements[0][1])
        vector_dims_shape = (elements_length,)
    else:
        vector_dims_shape = tuple(
            struct.size for (name, struct) in target_structure)

    # Build arrays of element values mapped onto the vectorised dimensions.
    elements_and_dimensions = base_structure.build_arrays(
        vector_dims_shape, actual_value_arrays)

    # Filter out the trivial (scalar) ones.
    elements_and_dimensions = {
        name: (array, dims)
        for name, (array, dims) in elements_and_dimensions.iteritems()
        if len(dims)}

    # Make a list of the 'primary' elements = those in the target structure.
    primary_dimension_elements = set(
        name for name, structure in target_structure)

    # Return all the information.
    return (vector_dims_shape,
            primary_dimension_elements,
            elements_and_dimensions)
