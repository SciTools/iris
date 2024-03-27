# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""A module to provide an optimal array structure calculation."""

from iris.fileformats._structured_array_identification import GroupStructure


def _optimal_dimensioning_structure(structure, element_priorities):
    """Determine the optimal array structure for the :class:`FieldCollation`.

    Uses the structure options provided by the
    :class:`~iris.fileformats._structured_array_identification.GroupStructure`
    to determine the optimal array structure for the :class:`FieldCollation`.

    The optimal structure is that which generates the greatest number of
    non-trivial dimensions. If the number of non-trivial dimensions is equal
    in more than one structure options then dimension priorities as specified
    by `element_priorities` are used to determine optimal structure.

    Parameters
    ----------
    structure :
        A set of structure options, as provided by :class:\
        `~iris.fileformats._structured_array_identification.GroupStructure`.
    element_priorities :
        A dictionary mapping structure element names to their priority as
        defined by their input order to :func:`~optimal_array_structure`.

    Returns
    -------
    array structure or an empty list
        The determined optimal array structure or an empty list if no structure
        options were determined.

    """
    permitted_structures = structure.possible_structures()
    if not permitted_structures:
        result = []
    else:
        result = max(
            permitted_structures,
            key=lambda candidate: (
                len(candidate),
                [element_priorities[name] for (name, struct) in candidate],
            ),
        )
    return result


def optimal_array_structure(ordering_elements, actual_values_elements=None):
    """Calculate an optimal array replication structure for a set of vectors.

    Parameters
    ----------
    ordering_elements : iterable of (name, 1-d array)
        Input element names and value-vectors.  Must all be the same length
        (but not necessarily type).  Must have at least one.

        .. note::

            The 'ordering_elements' arg contains the pattern used to deduce a
            structure.  The order of this is significant, in that earlier
            elements get priority when associating dimensions with specific
            elements.
    actual_values_elements : iterable of (name, 1-d array), optional
        The 'real' values used to construct the result arrays, if different
        from 'ordering_elements'.  Must contain all the same names (but not
        necessarily in the same order).

    Returns
    -------
    dims_shape
        Shape of the vector dimensions chosen.
    primary_elements
        Set of dimension names; the names of input
        elements that are identified as dimensions. At most one for each
        dimension.
    element_arrays_and_dims
        A dictionary [name: (array, dims)],
        for all elements that are not dimensionless. Each array is reduced
        to the shape of its mapped dimension.

    Examples
    --------
    ::

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
        >>> print(dims_shape)
        (2, 3)
        >>> print(dim_names)
        set(['a', 'b'])
        >>> print(arrays_and_dims)
        {'a': (array([10, 12]), (0,)), 'c': (array([[9, 3, 5],
               [2, 7, 1]]), (0, 1)), 'b': (array([15, 16, 17]), (1,))}

    """
    # Convert the inputs to dicts.
    element_ordering_arrays = dict(ordering_elements)
    if actual_values_elements is None:
        actual_values_elements = element_ordering_arrays
    actual_value_arrays = dict(actual_values_elements)
    if set(actual_value_arrays.keys()) != set(element_ordering_arrays.keys()):
        msg = "Names in values arrays do not match those in ordering arrays."
        raise ValueError(msg)

    # Define element priorities from ordering, to choose between equally good
    # structures, as structure code does not recognise any element ordering.
    n_elements = len(ordering_elements)
    element_priorities = {
        name: n_elements - index
        for index, (name, array) in enumerate(ordering_elements)
    }

    # Calculate the basic fields-group array structure.
    base_structure = GroupStructure.from_component_arrays(element_ordering_arrays)

    # Work out the target cube structure.
    target_structure = _optimal_dimensioning_structure(
        base_structure, element_priorities
    )

    # Work out result cube dimensions.
    if not target_structure:
        # Get the length of an input array (they are all the same).
        # Note that no elements map to multiple dimensions.
        elements_length = len(ordering_elements[0][1])
        vector_dims_shape = (elements_length,)
    else:
        vector_dims_shape = tuple(struct.size for (_, struct) in target_structure)

    # Build arrays of element values mapped onto the vectorised dimensions.
    elements_and_dimensions = base_structure.build_arrays(
        vector_dims_shape, actual_value_arrays
    )

    # Filter out the trivial (scalar) ones.
    elements_and_dimensions = {
        name: (array, dims)
        for name, (array, dims) in elements_and_dimensions.items()
        if len(dims)
    }

    # Make a list of 'primary' elements; i.e. those in the target structure.
    primary_dimension_elements = set(name for (name, _) in target_structure)

    if vector_dims_shape == (1,):
        shape = ()
    else:
        shape = vector_dims_shape

    # Return all the information.
    return (shape, primary_dimension_elements, elements_and_dimensions)
