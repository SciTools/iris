# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Automatic collation of cubes into higher-dimensional cubes.

Typically the cube merge process is handled by
:method:`iris.cube.CubeList.merge`.

"""

from collections import OrderedDict, namedtuple
from copy import deepcopy

import numpy as np

from iris._lazy_data import (
    as_concrete_data,
    as_lazy_data,
    is_lazy_data,
    multidim_lazy_stack,
)
from iris.common import CoordMetadata, CubeMetadata
import iris.coords
import iris.cube
import iris.exceptions
import iris.util


#
# Private namedtuple wrapper classes.
#
class _Template(
    namedtuple("Template", ["dims", "points", "bounds", "kwargs"])
):
    """
    Common framework from which to build a dimension or auxiliary coordinate.

    Args:

    * dims:
        Tuple of the associated :class:`iris.cube.Cube` data dimension/s
        spanned by this coordinate template.

    * points:
        A :mod:`numpy` array representing the coordinate point data. No
        points data is represented by None.

    * bounds:
        A :mod:`numpy` array representing the coordinate bounds data. No
        bounds data is represented by None.

    * kwargs:
        A dictionary of key/value pairs required to create a coordinate.

    """

    __slots__ = ()


class _CoordMetaData(
    namedtuple("CoordMetaData", ["points_dtype", "bounds_dtype", "kwargs"])
):
    """
    Bespoke metadata required to build a dimension or auxiliary coordinate.

    Args:

    * points_dtype:
        The points data :class:`numpy.dtype` of an associated coordinate.
        None otherwise.

    * bounds_dtype:
        The bounds data :class:`numpy.dtype` of an associated coordinate.
        None otherwise.

    * kwargs:
        A dictionary of key/value pairs required to create a coordinate.

    """

    __slots__ = ()


class _CoordAndDims(namedtuple("CoordAndDims", ["coord", "dims"])):
    """
    Container for a coordinate and the associated data dimension/s
    spanned over a :class:`iris.cube.Cube`.

    Args:

    * coord:
        A :class:`iris.coords.DimCoord` or :class:`iris.coords.AuxCoord`
        coordinate instance.

    * dims:
        A tuple of the data dimension/s spanned by the coordinate.

    """

    __slots__ = ()


class _ScalarCoordPayload(
    namedtuple("ScalarCoordPayload", ["defns", "values", "metadata"])
):
    """
    Container for all scalar coordinate data and metadata represented
    within a :class:`iris.cube.Cube`.

    All scalar coordinate related data is sorted into ascending order
    of the associated coordinate definition.

    Args:

    * defns:
        A list of scalar coordinate metadata :class:`iris.common.CoordMetadata`
        belonging to a :class:`iris.cube.Cube`.

    * values:
        A list of scalar coordinate values belonging to a
        :class:`iris.cube.Cube`.  Each scalar coordinate value is
        typically an :class:`iris.coords.Cell`.

    * metadata:
        A list of :class:`_CoordMetaData` instances belonging to a
        :class:`iris.cube.Cube`.

    """

    __slots__ = ()


class _VectorCoordPayload(
    namedtuple(
        "VectorCoordPayload", ["dim_coords_and_dims", "aux_coords_and_dims"]
    )
):
    """
    Container for all vector coordinate data and metadata represented
    within a :class:`iris.cube.Cube`.

    Args:

    * dim_coords_and_dims:
        A list of :class:`_CoordAndDim` instances containing non-scalar
        (i.e. multi-valued) :class:`iris.coords.DimCoord` instances and
        the associated data dimension spanned by them for a
        :class:`iris.cube.Cube`.

    * aux_coords_and_dims:
        A list of :class:`_CoordAndDim` instances containing non-scalar
        (i.e. multi-valued) :class:`iris.coords.DimCoord` and/or
        :class:`iris.coords.AuxCoord` instances and the associated data
        dimension/s spanned by them for a :class:`iris.cube.Cube`.

    """

    __slots__ = ()


class _CoordPayload(
    namedtuple("CoordPayload", ["scalar", "vector", "factory_defns"])
):
    """
    Container for all the scalar and vector coordinate data and
    metadata, and auxiliary coordinate factories represented within a
    :class:`iris.cube.Cube`.

    All scalar coordinate and factory related data is sorted into
    ascending order of the associated coordinate definition.

    Args:

    * scalar:
        A :class:`_ScalarCoordPayload` instance.

    * vector:
        A :class:`_VectorCoordPayload` instance.

    * factory_defns:
        A list of :class:`_FactoryDefn` instances.

    """

    __slots__ = ()

    def as_signature(self):
        """Construct and return a :class:`_CoordSignature` from the payload."""

        return _CoordSignature(
            self.scalar.defns,
            self.vector.dim_coords_and_dims,
            self.vector.aux_coords_and_dims,
            self.factory_defns,
        )

    @staticmethod
    def _coords_msgs(msgs, coord_group, defns_a, defns_b):
        if defns_a != defns_b:
            # Get a new list so we can modify it
            defns_b = list(defns_b)
            diff_defns = []
            for defn_a in defns_a:
                try:
                    defns_b.remove(defn_a)
                except ValueError:
                    diff_defns.append(defn_a)
            diff_defns.extend(defns_b)
            if diff_defns:
                names = sorted(set(defn.name() for defn in diff_defns))
                msgs.append(
                    "Coordinates in {} differ: {}.".format(
                        coord_group, ", ".join(names)
                    )
                )
            else:
                msgs.append(
                    "Coordinates in {} differ by dtype or class"
                    " (i.e. DimCoord vs AuxCoord).".format(coord_group)
                )

    def match_signature(self, signature, error_on_mismatch):
        """
        Return whether this _CoordPayload matches the corresponding
        aspects of a _CoordSignature.

        Args:

        * signature (_CoordSignature):
            The _CoordSignature to compare against.

        * error_on_mismatch (bool):
            If True, raise an Exception with detailed explanation.

        Returns:
           Boolean. True if and only if this _CoordPayload matches
           the corresponding aspects `other`.

        """

        def unzip(coords_and_dims):
            if coords_and_dims:
                coords, dims = zip(*coords_and_dims)
            else:
                coords, dims = [], []
            return coords, dims

        def dims_msgs(msgs, coord_group, dimlists_a, dimlists_b):
            if dimlists_a != dimlists_b:
                msgs.append(
                    "Coordinate-to-dimension mapping differs for {}.".format(
                        coord_group
                    )
                )

        msgs = []
        self._coords_msgs(
            msgs,
            "cube.aux_coords (scalar)",
            self.scalar.defns,
            signature.scalar_defns,
        )

        coord_group = "cube.dim_coords"
        self_coords, self_dims = unzip(self.vector.dim_coords_and_dims)
        other_coords, other_dims = unzip(signature.vector_dim_coords_and_dims)
        self._coords_msgs(msgs, coord_group, self_coords, other_coords)
        dims_msgs(msgs, coord_group, self_dims, other_dims)

        coord_group = "cube.aux_coords (non-scalar)"
        self_coords, self_dims = unzip(self.vector.aux_coords_and_dims)
        other_coords, other_dims = unzip(signature.vector_aux_coords_and_dims)
        self._coords_msgs(msgs, coord_group, self_coords, other_coords)
        dims_msgs(msgs, coord_group, self_dims, other_dims)

        if self.factory_defns != signature.factory_defns:
            msgs.append("cube.aux_factories() differ")

        match = not bool(msgs)
        if error_on_mismatch and not match:
            raise iris.exceptions.MergeError(msgs)
        return match


class _CoordSignature(
    namedtuple(
        "CoordSignature",
        [
            "scalar_defns",
            "vector_dim_coords_and_dims",
            "vector_aux_coords_and_dims",
            "factory_defns",
        ],
    )
):
    """
    Criterion for identifying a specific type of :class:`iris.cube.Cube`
    based on its scalar and vector coorinate data and metadata, and
    auxiliary coordinate factories.

    Args:

    * scalar_defns:
        A list of scalar coordinate definitions sorted into ascending order.

    * vector_dim_coords_and_dims:
        A list of :class:`_CoordAndDim` instances containing non-scalar
        (i.e. multi-valued) :class:`iris.coords.DimCoord` instances and
        the associated data dimension spanned by them for a
        :class:`iris.cube.Cube`.

    * vector_aux_coords_and_dims:
        A list of :class:`_CoordAndDim` instances containing non-scalar
        (i.e. multi-valued) :class:`iris.coords.DimCoord` and/or
        :class:`iris.coords.AuxCoord` instances and the associated data
        dimension/s spanned by them for a :class:`iris.cube.Cube`.

    * factory_defns:
        A list of :class:`_FactoryDefn` instances.

    """

    __slots__ = ()


class _CubeSignature(
    namedtuple(
        "CubeSignature",
        [
            "defn",
            "data_shape",
            "data_type",
            "cell_measures_and_dims",
            "ancillary_variables_and_dims",
        ],
    )
):
    """
    Criterion for identifying a specific type of :class:`iris.cube.Cube`
    based on its metadata.

    Args:

    * defn:
        A cube definition tuple.

    * data_shape:
        The data payload shape of a :class:`iris.cube.Cube`.

    * data_type:
        The data payload :class:`numpy.dtype` of a :class:`iris.cube.Cube`.

    * cell_measures_and_dims:
        A list of cell_measures and dims for the cube.

    * ancillary_variables_and_dims:
        A list of ancillary variables and dims for the cube.

    """

    __slots__ = ()

    def _defn_msgs(self, other_defn):
        msgs = []
        self_defn = self.defn
        if self_defn.standard_name != other_defn.standard_name:
            msgs.append(
                "cube.standard_name differs: {!r} != {!r}".format(
                    self_defn.standard_name, other_defn.standard_name
                )
            )
        if self_defn.long_name != other_defn.long_name:
            msgs.append(
                "cube.long_name differs: {!r} != {!r}".format(
                    self_defn.long_name, other_defn.long_name
                )
            )
        if self_defn.var_name != other_defn.var_name:
            msgs.append(
                "cube.var_name differs: {!r} != {!r}".format(
                    self_defn.var_name, other_defn.var_name
                )
            )
        if self_defn.units != other_defn.units:
            msgs.append(
                "cube.units differs: {!r} != {!r}".format(
                    self_defn.units, other_defn.units
                )
            )
        if self_defn.attributes != other_defn.attributes:
            diff_keys = set(self_defn.attributes.keys()) ^ set(
                other_defn.attributes.keys()
            )
            if diff_keys:
                msgs.append(
                    "cube.attributes keys differ: "
                    + ", ".join(repr(key) for key in diff_keys)
                )
            else:
                diff_attrs = [
                    repr(key)
                    for key in self_defn.attributes
                    if np.all(
                        self_defn.attributes[key] != other_defn.attributes[key]
                    )
                ]
                diff_attrs = ", ".join(diff_attrs)
                msgs.append(
                    "cube.attributes values differ for keys: {}".format(
                        diff_attrs
                    )
                )
        if self_defn.cell_methods != other_defn.cell_methods:
            msgs.append("cube.cell_methods differ")
        return msgs

    def match(self, other, error_on_mismatch):
        """
        Return whether this _CubeSignature equals another.

        This is the first step to determine if two "cubes" (either a
        real Cube or a ProtoCube) can be merged, by considering:
            - standard_name, long_name, var_name
            - units
            - attributes
            - cell_methods
            - shape, dtype

        Args:

        * other (_CubeSignature):
            The _CubeSignature to compare against.

        * error_on_mismatch (bool):
            If True, raise a :class:`~iris.exceptions.MergeException`
            with a detailed explanation if the two do not match.

        Returns:
           Boolean. True if and only if this _CubeSignature matches `other`.

        """
        msgs = self._defn_msgs(other.defn)
        if self.data_shape != other.data_shape:
            msg = "cube.shape differs: {} != {}"
            msgs.append(msg.format(self.data_shape, other.data_shape))
        if self.data_type != other.data_type:
            msg = "cube data dtype differs: {} != {}"
            msgs.append(msg.format(self.data_type, other.data_type))
        # Both cell_measures_and_dims and ancillary_variables_and_dims are
        # ordered by the same method, it is therefore not possible for a
        # mismatch to be caused by a difference in order.
        if self.cell_measures_and_dims != other.cell_measures_and_dims:
            msgs.append("cube.cell_measures differ")
        if (
            self.ancillary_variables_and_dims
            != other.ancillary_variables_and_dims
        ):
            msgs.append("cube.ancillary_variables differ")

        match = not bool(msgs)
        if error_on_mismatch and not match:
            raise iris.exceptions.MergeError(msgs)
        return match


class _Skeleton(namedtuple("Skeleton", ["scalar_values", "data"])):
    """
    Basis of a source-cube, containing the associated scalar coordinate values
    and data payload of a :class:`iris.cube.Cube`.

    Args:

    * scalar_values:
        A list of scalar coordinate values belonging to a
        :class:`iris.cube.Cube` sorted into ascending order of the
        associated coordinate definition. Each scalar coordinate value
        is typically an :class:`iris.coords.Cell`.

    * data:
        The data payload of a :class:`iris.cube.Cube`.

    """

    __slots__ = ()


class _FactoryDefn(namedtuple("_FactoryDefn", ["class_", "dependency_defns"])):
    """
    The information required to identify and rebuild a single AuxCoordFactory.

    Args:

    * class_:
        The class of the AuxCoordFactory.

    * dependency_defns:
        A list of pairs, where each pair contains a dependency key and its
        corresponding coordinate definition. Sorted on dependency key.

    """

    __slots__ = ()


class _Relation(namedtuple("Relation", ["separable", "inseparable"])):
    """
    Categorisation of the candidate dimensions belonging to a
    :class:`ProtoCube` into separable 'independent' dimensions, and
    inseparable dependent dimensions.

    Args:

    * separable:
        A set of independent candidate dimension names.

    * inseperable:
        A set of dependent candidate dimension names.

    """

    __slots__ = ()


_COMBINATION_JOIN = "-"


def _is_combination(name):
    """
    Determine whether the candidate dimension is an 'invented' combination
    of candidate dimensions.

    Args:

    * name:
        The candidate dimension.

    Returns:
        Boolean.

    """
    return _COMBINATION_JOIN in str(name)


def build_indexes(positions):
    """
    Construct a mapping for each candidate dimension that maps for each
    of its scalar values the set of values for each of the other candidate
    dimensions.

    For example:

        >>> from iris._merge import build_indexes
        >>> positions = [{'a': 0, 'b': 10, 'c': 100},
        ...              {'a': 1, 'b': 10, 'c': 200},
        ...              {'a': 2, 'b': 20, 'c': 300}]
        ...
        >>> indexes = build_indexes(positions)
        >>> for k in sorted(indexes):
        ...     print('%r:' % k)
        ...     for kk in sorted(indexes[k]):
        ...         print('\t%r:' % kk, end=' ')
        ...         for kkk in sorted(indexes[k][kk]):
        ...             print('%r: %r' % (kkk, indexes[k][kk][kkk]), end=' ')
        ...         print()
        ...
        'a':
             0: 'b': set([10]) 'c': set([100])
             1: 'b': set([10]) 'c': set([200])
             2: 'b': set([20]) 'c': set([300])
        'b':
             10: 'a': set([0, 1]) 'c': set([200, 100])
             20: 'a': set([2]) 'c': set([300])
        'c':
             100: 'a': set([0]) 'b': set([10])
             200: 'a': set([1]) 'b': set([10])
             300: 'a': set([2]) 'b': set([20])

    Args:

    * positions:
        A list containing a dictionary of candidate dimension key to
        scalar value pairs for each source-cube.

    Returns:
        The cross-reference dictionary for each candidate dimension.

    """
    names = positions[0].keys()
    scalar_index_by_name = {name: {} for name in names}

    for position in positions:
        for name, value in position.items():
            name_index_by_scalar = scalar_index_by_name[name]

            if value in name_index_by_scalar:
                value_index_by_name = name_index_by_scalar[value]
                for other_name in names:
                    if other_name != name:
                        value_index_by_name[other_name].add(
                            position[other_name]
                        )
            else:
                name_index_by_scalar[value] = {
                    other_name: set((position[other_name],))
                    for other_name in names
                    if other_name != name
                }

    return scalar_index_by_name


def _separable_pair(name, index):
    """
    Determine whether the candidate dimension is separable.

    A candidate dimension X and Y are separable if each scalar
    value of X maps to the same set of scalar values of Y.

    Args:

    * name1:
        The first candidate dimension to be compared.

    * name2:
        The second candidate dimension to be compared.

    * index:
        The cross-reference dictionary for the first candidate
        dimension.

    Returns:
        Boolean.

    """
    items = iter(index.values())
    reference = next(items)[name]

    return all([item[name] == reference for item in items])


def _separable(name, indexes):
    """
    Determine the candidate dimensions that are separable and
    inseparable relative to the provided candidate dimension.

    A candidate dimension X and Y are separable if each scalar
    value of X maps to the same set of scalar values of Y.

    Args:

    * name:
        The candidate dimension that requires its separable and
        inseparable relationship to be determined.

    * indexes:
        The cross-reference dictionary for each candidate dimension.

    Returns:
        A tuple containing the set of separable and inseparable
        candidate dimensions.

    """
    separable = set()
    inseparable = set()

    for target_name in indexes:
        if name != target_name:
            if _separable_pair(target_name, indexes[name]):
                separable.add(target_name)
            else:
                inseparable.add(target_name)

    return _Relation(separable, inseparable)


def derive_relation_matrix(indexes):
    """
    Construct a mapping for each candidate dimension that specifies
    which of the other candidate dimensions are separable or inseparable.

    A candidate dimension X and Y are separable if each scalar value of
    X maps to the same set of scalar values of Y.

    Also see :func:`iris._merge.build_indexes`.

    For example:

        >>> from iris._merge import build_indexes, derive_relation_matrix
        >>> positions = [{'a': 0, 'b': 10, 'c': 100},
        ...              {'a': 1, 'b': 10, 'c': 200},
        ...              {'a': 2, 'b': 20, 'c': 300}]
        ...
        >>> indexes = build_indexes(positions)
        >>> matrix = derive_relation_matrix(indexes)
        >>> for k, v in matrix.iteritems():
        ...     print('%r: %r' % (k, v))
        ...
        'a': Relation(separable=set([]), inseparable=set(['c', 'b']))
        'c': Relation(separable=set([]), inseparable=set(['a', 'b']))
        'b': Relation(separable=set([]), inseparable=set(['a', 'c']))

    Args:

    * indexes:
        The cross-reference dictionary for each candidate dimension.

    Returns:
        The relation dictionary for each candidate dimension.

    """
    # TODO: This takes twice as long as it could do because it doesn't
    # account for the symmetric nature of the relationship.
    relation_matrix = {name: _separable(name, indexes) for name in indexes}

    return relation_matrix


def derive_groups(relation_matrix):
    """
    Determine all related (chained) groups of inseparable candidate dimensions.

    If candidate dimension A is inseparable for B and C, and B is inseparable
    from D, and E is inseparable from F. Then the groups are ABCD and EF.

    Args:

    * relation_matrix:
        The relation dictionary for each candidate dimension.

    Returns:
        A list of all related (chained) inseparable candidate dimensions.

    """
    names = set(relation_matrix)
    groups = []

    while names:
        name = names.pop()
        group = set([name])
        to_follow = set([name])

        while to_follow:
            name = to_follow.pop()
            new = relation_matrix[name].inseparable - group
            group.update(new)
            to_follow.update(new)
            names -= new

        groups.append(group)

    return groups


def _derive_separable_group(relation_matrix, group):
    """
    Determine which candidate dimensions in the group are separable.

    Args:

    * relation_matrix:
        The relation dictionary for each candidate dimension.

    * group:
        A set of related (chained) inseparable candidate dimensions.

    Returns:
        The set of candidate dimensions within the group that are
        separable.

    """
    result = set()

    for name in group:
        if relation_matrix[name].separable & group:
            result.add(name)

    return result


def _is_dependent(dependent, independent, positions, function_mapping=None):
    """
    Determine whether there exists a one-to-one functional relationship
    between the independent candidate dimension/s and the dependent
    candidate dimension.

    Args:

    * dependent:
        A candidate dimension that requires to be functionally
        dependent on all the independent candidate dimensions.

    * independent:
        A list of candidate dimension/s that require to act as the independent
        variables in a functional relationship.

    * positions:
        A list containing a dictionary of candidate dimension key to
        scalar value pairs for each source-cube.

    Kwargs:

    * function_mapping:
        A dictionary that enumerates a valid functional relationship
        between the dependent candidate dimension and the independent
        candidate dimension/s.

    Returns:
        Boolean.

    """
    valid = True
    relation = {}

    if isinstance(function_mapping, dict):
        relation = function_mapping

    for position in positions:
        item = tuple([position[name] for name in independent])

        if item in relation:
            if position[dependent] != relation[item]:
                valid = False
                break
        else:
            relation[item] = position[dependent]

    return valid


def _derive_consistent_groups(relation_matrix, separable_group):
    """
    Determine the largest combinations of candidate dimensions within the
    separable group that are self consistently separable from one another.

    If the candidate dimension A is separable from the candidate dimensions
    B and C. Then the candidate dimension group ABC is a separable consistent
    group if B is separable from A and C, and C is separable from A and B.

    Args:

    * relation_matrix:
        The relation dictionary for each candidate dimension.

    * separable_group:
        The set of candidate dimensions that are separable.

    Returns:
        A list of candidate dimension groups that are consistently separable.

    """
    result = []

    for name in separable_group:
        name_separable_group = (
            relation_matrix[name].separable & separable_group
        )
        candidate = list(name_separable_group) + [name]
        valid = True

        for _ in range(len(name_separable_group)):
            candidate_separable_group = set(candidate[1:])

            if (
                candidate_separable_group
                & (relation_matrix[candidate[0]].separable & separable_group)
                != candidate_separable_group
            ):
                valid = False
                break

            candidate.append(candidate.pop(0))

        if valid:
            result.append(candidate)

    return result


def _build_separable_group(
    space, group, separable_consistent_groups, positions, function_matrix
):
    """
    Update the space with the first separable consistent group that
    satisfies a valid functional relationship with all other candidate
    dimensions in the group.

    For example, the group ABCD and separable consistent group CD,
    if A = f(C, D) and B = f(C, D) then update the space with
    "A: (C, D), B: (C, D), C: None, D: None". Where "A: (C, D)" means
    that candidate dimension A is dependent on candidate dimensions C
    and D, and "C: None" means that this candidate dimension is
    independent.

    Args:

    * space:
        A dictionary defining for each candidate dimension its
        dependency on any other candidate dimensions within the space.

    * group:
        A set of related (chained) inseparable candidate dimensions.

    * separable_consistent_groups:
        A list of candidate dimension groups that are consistently separable.

    * positions:
        A list containing a dictionary of candidate dimension key to
        scalar value pairs for each source-cube.

    * function_matrix:
        The function mapping dictionary for each candidate dimension that
        participates in a functional relationship.

    Returns:
        Boolean.

    """
    valid = False

    for independent in sorted(separable_consistent_groups):
        dependent = list(group - set(independent))
        dependent_function_matrix = {}

        for name in dependent:
            function_mapping = {}
            valid = _is_dependent(
                name, independent, positions, function_mapping
            )

            if not valid:
                break

            dependent_function_matrix[name] = function_mapping

        if valid:
            break

    if function_matrix is not None:
        function_matrix.update(dependent_function_matrix)

    if valid:
        space.update({name: None for name in independent})
        space.update({name: tuple(independent) for name in dependent})

    return valid


def _build_inseparable_group(space, group, positions, function_matrix):
    """
    Update the space with the first valid scalar functional relationship
    between a candidate dimension within the group and all other
    candidate dimensions.

    For example, with the inseparable group ABCD, a valid scalar
    relationship B = f(A), C = f(A) and D = f(A) results in a space with
    "A: None, B: (A,), C: (A,), D: (A,)".
    Where "A: None" means that this candidate dimension is independent,
    and "B: (A,)" means that candidate dimension B is dependent on
    candidate dimension A.

    The scalar relationship must exist between one candidate dimension
    and all others in the group, as the group is considered inseparable
    in this context.

    Args:

    * space:
        A dictionary defining for each candidate dimension its dependency on
        any other candidate dimensions within the space.

    * group:
        A set of related (chained) inseparable candidate dimensions.

    * positions:
        A list containing a dictionary of candidate dimension key to
        scalar value pairs for each source-cube.

    * function_matrix:
        The function mapping dictionary for each candidate dimension that
        participates in a functional relationship.

    Returns:
        Boolean.

    """
    scalar = False

    for name in sorted(group):
        independent = set([name])
        dependent = set(group) - independent
        valid = False
        dependent_function_matrix = {}

        for name in dependent:
            function_mapping = {}
            valid = _is_dependent(
                name, independent, positions, function_mapping
            )

            if not valid:
                break

            dependent_function_matrix[name] = function_mapping

        if valid:
            scalar = True

            if function_matrix is not None:
                function_matrix.update(dependent_function_matrix)

            space.update({name: None for name in independent})
            space.update({name: tuple(independent) for name in dependent})
            break

    return scalar


def _build_combination_group(space, group, positions, function_matrix):
    """
    Update the space with the new combined or invented dimension
    that each member of this inseparable group depends on.

    As no functional relationship between members of the group can be
    determined, the new combination dimension will not have a dimension
    coordinate associated with it. Rather, it is simply an enumeration
    of the group members for each of the positions (source-cubes).

    Args:

    * space:
        A dictionary defining for each candidate dimension its dependency on
        any other candidate dimensions within the space.

    * group:
        A set of related (chained) inseparable candidate dimensions.

    * positions:
        A list containing a dictionary of candidate dimension key to
        scalar value pairs for each source-cube.

    * function_matrix:
        The function mapping dictionary for each candidate dimension that
        participates in a functional relationship.

    Returns:
        None.

    """
    combination = _COMBINATION_JOIN.join(sorted(map(str, group)))
    space.update({name: None for name in (combination,)})
    space.update({name: (combination,) for name in group})
    members = combination.split(_COMBINATION_JOIN)

    # Populate the function matrix for each member of the group.
    for name in group:
        function_matrix[name] = {}

    for position in positions:
        # Note, the cell double-tuple! This ensures that the cell value for
        # each member of the group is kept bound together as one key.
        cell = (
            tuple(
                [
                    position[int(member) if member.isdigit() else member]
                    for member in members
                ]
            ),
        )
        for name in group:
            function_matrix[name][cell] = position[name]


def derive_space(groups, relation_matrix, positions, function_matrix=None):
    """
    Determine the relationship between all the candidate dimensions.

    Args:
      * groups:
          A list of all related (chained) inseparable candidate dimensions.

      * relation_matrix:
          The relation dictionary for each candidate dimension.

      * positions:
          A list containing a dictionary of candidate dimension key to
          scalar value pairs for each source-cube.

    Kwargs:
      * function_matrix:
          The function mapping dictionary for each candidate dimension that
          participates in a functional relationship.

    Returns:
        A space dictionary describing the relationship between each
        candidate dimension.

    """
    space = {}

    for group in groups:
        separable_group = _derive_separable_group(relation_matrix, group)

        if len(group) == 1 and not separable_group:
            # This single candidate dimension is separable from all other
            # candidate dimensions in the group, therefore it is a genuine
            # dimension of the space.
            space.update({name: None for name in group})
        elif separable_group:
            # Determine the largest combination of the candidate dimensions
            # in the separable group that are consistently separable.
            consistent_groups = _derive_consistent_groups(
                relation_matrix, separable_group
            )
            if not _build_separable_group(
                space, group, consistent_groups, positions, function_matrix
            ):
                # There is no relationship between any of the candidate
                # dimensions in the separable group, so merge them together
                # into a new combined dimension of the space.
                _build_combination_group(
                    space, group, positions, function_matrix
                )
        else:
            # Determine whether there is a scalar relationship between one of
            # the candidate dimensions and each of the other candidate
            # dimensions in this inseparable group.
            if not _build_inseparable_group(
                space, group, positions, function_matrix
            ):
                # There is no relationship between any of the candidate
                # dimensions in this inseparable group, so merge them together
                # into a new combined dimension of the space.
                _build_combination_group(
                    space, group, positions, function_matrix
                )

    return space


class ProtoCube:
    """
    Framework for merging source-cubes into one or more higher
    dimensional cubes.

    """

    def __init__(self, cube):
        """
        Create a new ProtoCube from the given cube and record the cube
        as a source-cube.

        """

        # Default hint ordering for candidate dimension coordinates.
        self._hints = [
            "time",
            "forecast_reference_time",
            "forecast_period",
            "model_level_number",
        ]

        # The proto-cube source.
        self._source = cube

        # The cube signature is metadata that defines this ProtoCube.
        self._cube_signature = self._build_signature(cube)

        # Extract the scalar and vector coordinate data and metadata
        # from the cube.
        coord_payload = self._extract_coord_payload(cube)

        # The coordinate signature defines the scalar and vector
        # coordinates of this ProtoCube.
        self._coord_signature = coord_payload.as_signature()
        self._coord_metadata = coord_payload.scalar.metadata

        # The list of stripped-down source-cubes relevant to this ProtoCube.
        self._skeletons = []
        self._add_cube(cube, coord_payload)

        # Proto-coordinates constructed from merged scalars.
        self._dim_templates = []
        self._aux_templates = []

        # During the merge this will contain the complete, merged shape
        # of a result cube.
        # E.g. Merging three (72, 96) cubes would give:
        #      self._shape = (3, 72, 96).
        self._shape = []
        # During the merge this will contain the shape of the "stack"
        # of cubes used to create a single result cube.
        # E.g. Merging three (72, 96) cubes would give:
        #      self._stack_shape = (3,)
        self._stack_shape = []

        self._nd_names = []
        self._cache_by_name = {}
        self._dim_coords_and_dims = []
        self._aux_coords_and_dims = []

        # Dims offset by merged space higher dimensionality.
        self._vector_dim_coords_dims = []
        self._vector_aux_coords_dims = []

        # cell measures and ancillary variables are not merge candidates
        # they are checked and preserved through merge
        self._cell_measures_and_dims = cube._cell_measures_and_dims
        self._ancillary_variables_and_dims = cube._ancillary_variables_and_dims

    def _report_duplicate(self, nd_indexes, group_by_nd_index):
        # Find the first offending source-cube with duplicate metadata.
        index = [
            group_by_nd_index[nd_index][1]
            for nd_index in nd_indexes
            if len(group_by_nd_index[nd_index]) > 1
        ][0]
        name = self._cube_signature.defn.name()
        scalars = []
        for defn, value in zip(
            self._coord_signature.scalar_defns,
            self._skeletons[index].scalar_values,
        ):
            scalars.append("%s=%r" % (defn.name(), value))
        msg = "Duplicate %r cube, with scalar coordinates %s"
        msg = msg % (name, ", ".join(scalars))
        raise iris.exceptions.DuplicateDataError(msg)

    def merge(self, unique=True):
        """
        Returns the list of cubes resulting from merging the registered
        source-cubes.

        Kwargs:

        * unique:
            If True, raises `iris.exceptions.DuplicateDataError` if
            duplicate cubes are detected.

        Returns:
            A :class:`iris.cube.CubeList` of merged cubes.

        """
        positions = [
            {i: v for i, v in enumerate(skeleton.scalar_values)}
            for skeleton in self._skeletons
        ]
        indexes = build_indexes(positions)
        relation_matrix = derive_relation_matrix(indexes)
        groups = derive_groups(relation_matrix)

        function_matrix = {}
        space = derive_space(
            groups, relation_matrix, positions, function_matrix=function_matrix
        )
        self._define_space(space, positions, indexes, function_matrix)
        self._build_coordinates()

        # All the final, merged cubes will end up here.
        merged_cubes = iris.cube.CubeList()

        # Collate source-cubes by the nd-index.
        group_by_nd_index = {}
        for index, position in enumerate(positions):
            group = group_by_nd_index.setdefault(self._nd_index(position), [])
            group.append(index)

        # Determine the largest group of source-cubes that want to occupy
        # the same nd-index in the final merged cube.
        group_depth = max([len(group) for group in group_by_nd_index.values()])
        nd_indexes = sorted(group_by_nd_index.keys())

        # Check for unique data.
        if unique and group_depth > 1:
            self._report_duplicate(nd_indexes, group_by_nd_index)

        # Generate group-depth merged cubes from the source-cubes.
        for level in range(group_depth):
            # Track the largest dtype of the data to be merged.
            # Unfortunately, da.stack() is not symmetric with regards
            # to dtypes. So stacking float + int yields a float, but
            # stacking an int + float yields an int! We need to ensure
            # that the largest dtype prevails i.e. float, in order to
            # support the masked case for dask.
            # Reference https://github.com/dask/dask/issues/2273.
            dtype = None
            # Stack up all the data from all of the relevant source
            # cubes in a single dask "stacked" array.
            # If it turns out that all the source cubes already had
            # their data loaded then at the end we convert the stack back
            # into a plain numpy array.
            stack = np.empty(self._stack_shape, "object")
            all_have_data = True
            for nd_index in nd_indexes:
                # Get the data of the current existing or last known
                # good source-cube
                group = group_by_nd_index[nd_index]
                offset = min(level, len(group) - 1)
                data = self._skeletons[group[offset]].data
                # Ensure the data is represented as a dask array and
                # slot that array into the stack.
                if is_lazy_data(data):
                    all_have_data = False
                else:
                    data = as_lazy_data(data)
                stack[nd_index] = data
                # Determine the largest dtype.
                if dtype is None:
                    dtype = data.dtype
                else:
                    dtype = np.promote_types(data.dtype, dtype)

            # Coerce to the largest dtype.
            for nd_index in nd_indexes:
                stack[nd_index] = stack[nd_index].astype(dtype)

            merged_data = multidim_lazy_stack(stack)
            if all_have_data:
                # All inputs were concrete, so turn the result back into a
                # normal array.
                dtype = self._cube_signature.data_type
                merged_data = as_concrete_data(merged_data)
            merged_cube = self._get_cube(merged_data)
            merged_cubes.append(merged_cube)

        return merged_cubes

    def register(self, cube, error_on_mismatch=False):
        """
        Add a compatible :class:`iris.cube.Cube` as a source-cube for
        merging under this :class:`ProtoCube`.

        A cube will be deemed compatible based on the signature of the
        cube and the signature of its scalar coordinates and vector
        coordinates being identical to that of the ProtoCube.

        Args:

        * cube:
            Candidate :class:`iris.cube.Cube` to be associated with
            this :class:`ProtoCube`.

        Kwargs:

        * error_on_mismatch:
            If True, raise an informative
            :class:`~iris.exceptions.MergeError` if registration fails.

        Returns:
            True iff the :class:`iris.cube.Cube` is compatible with
            this :class:`ProtoCube`.

        """
        cube_signature = self._cube_signature
        other = self._build_signature(cube)
        match = cube_signature.match(other, error_on_mismatch)
        if match:
            coord_payload = self._extract_coord_payload(cube)
            match = coord_payload.match_signature(
                self._coord_signature, error_on_mismatch
            )
        if match:
            # Register the cube as a source-cube for this ProtoCube.
            self._add_cube(cube, coord_payload)
        return match

    def _guess_axis(self, name):
        """
        Returns a "best guess" axis name of the candidate dimension.

        Heuristic categoration of the candidate dimension
        (i.e. scalar_defn index) into either label 'T', 'Z', 'Y', 'X'
        or None.

        Based on the associated scalar coordinate definition rather than the
        scalar coordinate itself.

        Args:

        * name:
            The candidate dimension.

        Returns:
            'T', 'Z', 'Y', 'X', or None.

        """
        axis = None

        if not _is_combination(name):
            defn = self._coord_signature.scalar_defns[name]
            axis = iris.util.guess_coord_axis(defn)

        return axis

    def _define_space(self, space, positions, indexes, function_matrix):
        """
        Given the derived :class:`ProtoCube` space, define this space in
        terms of its dimensionality, shape, coordinates and associated
        coordinate to space dimension mappings.

        Args:

        * space:
            A dictionary defining for each candidate dimension its
            dependency on any other candidate dimensions within the space.

        * positions:
            A list containing a dictionary of candidate dimension key to
            scalar value pairs for each source-cube.

        * indexes:
            A cross-reference dictionary for each candidate dimension.

        * function_matrix:
            The function mapping dictionary for each candidate dimension that
            participates in a functional relationship.

        """
        # Heuristic reordering of coordinate defintion indexes into
        # preferred dimension order.
        def axis_and_name(name):
            axis_dict = {"T": 1, "Z": 2, "Y": 3, "X": 4}
            axis_index = axis_dict.get(self._guess_axis(name), 0)
            # The middle element ensures sorting is the same as Python 2.
            return (axis_index, not isinstance(name, int), name)

        names = sorted(space, key=axis_and_name)
        dim_by_name = {}

        metadata = self._coord_metadata
        coord_signature = self._coord_signature
        defns = coord_signature.scalar_defns
        vector_dim_coords_and_dims = coord_signature.vector_dim_coords_and_dims
        vector_aux_coords_and_dims = coord_signature.vector_aux_coords_and_dims
        signature = self._cube_signature

        # First pass - Build the dimension coordinate templates for the space.
        for name in names:
            if space[name] is None:
                if _is_combination(name):
                    members = name.split(_COMBINATION_JOIN)
                    # Create list of unique tuples from all combinations of
                    # scalars for each source cube. The keys of an OrderedDict
                    # are used to retain the ordering of source cubes but to
                    # remove any duplicate tuples.
                    cells = OrderedDict(
                        (
                            tuple(
                                position[
                                    int(member) if member.isdigit() else member
                                ]
                                for member in members
                            ),
                            None,
                        )
                        for position in positions
                    ).keys()
                    dim_by_name[name] = len(self._shape)
                    self._nd_names.append(name)
                    self._shape.append(len(cells))
                    self._stack_shape.append(len(cells))
                    self._cache_by_name[name] = {
                        cell: index for index, cell in enumerate(cells)
                    }
                else:
                    # TODO: Consider appropriate sort order (ascending,
                    # decending) i.e. use CF positive attribute.
                    cells = sorted(indexes[name])
                    points = np.array(
                        [cell.point for cell in cells],
                        dtype=metadata[name].points_dtype,
                    )
                    if cells[0].bound is not None:
                        bounds = np.array(
                            [cell.bound for cell in cells],
                            dtype=metadata[name].bounds_dtype,
                        )
                    else:
                        bounds = None
                    kwargs = dict(zip(CoordMetadata._fields, defns[name]))
                    kwargs.update(metadata[name].kwargs)

                    def name_in_independents():
                        return any(
                            name in independents
                            for independents in space.values()
                            if independents is not None
                        )

                    if len(cells) == 1 and not name_in_independents():
                        # A scalar coordinate not participating in a
                        # function dependency.
                        self._aux_templates.append(
                            _Template((), points, bounds, kwargs)
                        )
                    else:
                        # Dimension coordinate (or aux if the data is
                        # string like).
                        dim_by_name[name] = dim = len(self._shape)
                        self._nd_names.append(name)
                        if metadata[name].points_dtype.kind in "SU":
                            self._aux_templates.append(
                                _Template(dim, points, bounds, kwargs)
                            )
                        else:
                            self._dim_templates.append(
                                _Template(dim, points, bounds, kwargs)
                            )
                        self._shape.append(len(cells))
                        self._stack_shape.append(len(cells))
                        self._cache_by_name[name] = {
                            cell: index for index, cell in enumerate(cells)
                        }

        # Second pass - Build the auxiliary coordinate templates for the space.
        for name in names:
            name_independents = space[name]

            # Determine if there is a function dependency.
            if name_independents is not None:
                # Calculate the auxiliary coordinate shape.
                dims = tuple(
                    [
                        dim_by_name[independent]
                        for independent in name_independents
                    ]
                )
                aux_shape = [self._shape[dim] for dim in dims]
                # Create empty points and bounds in preparation to be filled.
                points = np.empty(aux_shape, dtype=metadata[name].points_dtype)
                if positions[0][name].bound is not None:
                    shape = aux_shape + [len(positions[0][name].bound)]
                    bounds = np.empty(shape, dtype=metadata[name].bounds_dtype)
                else:
                    bounds = None

                # Populate the points and bounds based on the appropriate
                # function mapping.
                temp = function_matrix[name].items()
                for function_independents, name_value in temp:
                    # Build the index (and cache it) for the auxiliary
                    # coordinate based on the associated independent
                    # dimension coordinate/s.
                    index = []

                    name_function_pairs = zip(
                        name_independents, function_independents
                    )
                    for independent, independent_value in name_function_pairs:
                        cache = self._cache_by_name[independent]
                        index.append(cache[independent_value])

                    index = tuple(index)
                    if points is not None:
                        points[index] = name_value.point

                    if bounds is not None:
                        bounds[index] = name_value.bound

                kwargs = dict(zip(CoordMetadata._fields, defns[name]))
                self._aux_templates.append(
                    _Template(dims, points, bounds, kwargs)
                )

        # Calculate the dimension mapping for each vector within the space.
        offset = len(self._shape)
        self._vector_dim_coords_dims = [
            tuple([dim + offset for dim in item.dims])
            for item in vector_dim_coords_and_dims
        ]
        self._vector_aux_coords_dims = [
            tuple([dim + offset for dim in item.dims])
            for item in vector_aux_coords_and_dims
        ]

        # Now factor in the vector payload shape. Note that, for
        # deferred loading, this does NOT change the shape.
        self._shape.extend(signature.data_shape)

    def _get_cube(self, data):
        """
        Return a fully constructed cube for the given data, containing
        all its coordinates and metadata.

        """
        signature = self._cube_signature
        dim_coords_and_dims = [
            (deepcopy(coord), dim) for coord, dim in self._dim_coords_and_dims
        ]
        aux_coords_and_dims = [
            (deepcopy(coord), dims)
            for coord, dims in self._aux_coords_and_dims
        ]
        kwargs = dict(zip(CubeMetadata._fields, signature.defn))

        cms_and_dims = [
            (deepcopy(cm), dims) for cm, dims in self._cell_measures_and_dims
        ]
        avs_and_dims = [
            (deepcopy(av), dims)
            for av, dims in self._ancillary_variables_and_dims
        ]
        cube = iris.cube.Cube(
            data,
            dim_coords_and_dims=dim_coords_and_dims,
            aux_coords_and_dims=aux_coords_and_dims,
            cell_measures_and_dims=cms_and_dims,
            ancillary_variables_and_dims=avs_and_dims,
            **kwargs,
        )

        # Add on any aux coord factories.
        for factory_defn in self._coord_signature.factory_defns:
            args = {}
            for key, defn in factory_defn.dependency_defns:
                coord = cube.coord(defn)
                args[key] = coord
            factory = factory_defn.class_(**args)
            cube.add_aux_factory(factory)

        return cube

    def _nd_index(self, position):
        """
        Returns the n-dimensional index of this source-cube (position),
        within the merged cube.

        """

        index = []

        # Determine the index of the source-cube cell for each dimension.
        for name in self._nd_names:
            if _is_combination(name):
                members = name.split(_COMBINATION_JOIN)
                cell = tuple(
                    position[int(member) if member.isdigit() else member]
                    for member in members
                )
                index.append(self._cache_by_name[name][cell])
            else:
                index.append(self._cache_by_name[name][position[name]])

        return tuple(index)

    def _build_coordinates(self):
        """
        Build the dimension and auxiliary coordinates for the final
        merged cube given that the final dimensionality of the target
        merged cube is known and the associated dimension/s that each
        coordinate maps onto in that merged cube.

        The associated vector coordinate/s of the ProtoCube are also
        created with the correct space dimensionality and dimension/s
        mappings.

        """
        # Containers for the newly created coordinates and associated
        # dimension mappings.
        dim_coords_and_dims = self._dim_coords_and_dims
        aux_coords_and_dims = self._aux_coords_and_dims

        # Build the dimension coordinates.
        for template in self._dim_templates:
            # Sometimes it's not possible to build a dim coordinate e.g.
            # the bounds are not monotonic, so try building the coordinate,
            # and if it fails make the coordinate into an auxiliary coordinate.
            # This will ultimately make an anonymous dimension.
            try:
                coord = iris.coords.DimCoord(
                    template.points, bounds=template.bounds, **template.kwargs
                )
                dim_coords_and_dims.append(_CoordAndDims(coord, template.dims))
            except ValueError:
                self._aux_templates.append(template)

        # There is the potential that there are still anonymous dimensions.
        # Get a list of the dimensions which are not anonymous at this stage.
        covered_dims = [
            dim_coord_and_dim.dims for dim_coord_and_dim in dim_coords_and_dims
        ]

        # Build the auxiliary coordinates.
        for template in self._aux_templates:
            # Attempt to build a DimCoord and add it to the cube. If this
            # fails e.g it's non-monontic or multi-dimensional or non-numeric,
            # then build an AuxCoord.
            try:
                coord = iris.coords.DimCoord(
                    template.points, bounds=template.bounds, **template.kwargs
                )
                if (
                    len(template.dims) == 1
                    and template.dims[0] not in covered_dims
                ):
                    dim_coords_and_dims.append(
                        _CoordAndDims(coord, template.dims)
                    )
                    covered_dims.append(template.dims[0])
                else:
                    aux_coords_and_dims.append(
                        _CoordAndDims(coord, template.dims)
                    )
            except ValueError:
                # kwarg not applicable to AuxCoord.
                template.kwargs.pop("circular", None)
                coord = iris.coords.AuxCoord(
                    template.points, bounds=template.bounds, **template.kwargs
                )
                aux_coords_and_dims.append(_CoordAndDims(coord, template.dims))

        # Mix in the vector coordinates.
        for item, dims in zip(
            self._coord_signature.vector_dim_coords_and_dims,
            self._vector_dim_coords_dims,
        ):
            dim_coords_and_dims.append(_CoordAndDims(item.coord, dims))

        for item, dims in zip(
            self._coord_signature.vector_aux_coords_and_dims,
            self._vector_aux_coords_dims,
        ):
            aux_coords_and_dims.append(_CoordAndDims(item.coord, dims))

    def _build_signature(self, cube):
        """
        Generate the signature that defines this cube.

        Args:

        * cube:
            The source cube to create the cube signature from.

        Returns:
            The cube signature.

        """

        return _CubeSignature(
            cube.metadata,
            cube.shape,
            cube.dtype,
            cube._cell_measures_and_dims,
            cube._ancillary_variables_and_dims,
        )

    def _add_cube(self, cube, coord_payload):
        """Create and add the source-cube skeleton to the ProtoCube."""
        skeleton = _Skeleton(coord_payload.scalar.values, cube.core_data())
        # Attempt to do something sensible with mixed scalar dtypes.
        for i, metadata in enumerate(coord_payload.scalar.metadata):
            if metadata.points_dtype > self._coord_metadata[i].points_dtype:
                self._coord_metadata[i] = metadata
        self._skeletons.append(skeleton)

    def _extract_coord_payload(self, cube):
        """
        Extract all relevant coordinate data and metadata from the cube.

        In particular, for each scalar coordinate determine its definition,
        its cell (point and bound) value and all other scalar coordinate
        metadata that allows us to fully reconstruct that scalar
        coordinate. Note that all scalar data is sorted in order of the
        scalar coordinate definition.

        The coordinate payload of the cube also includes any associated vector
        coordinates that describe that cube, and descriptions of any auxiliary
        coordinate factories.

        """
        scalar_defns = []
        scalar_values = []
        scalar_metadata = []
        vector_dim_coords_and_dims = []
        vector_aux_coords_and_dims = []

        cube_aux_coords = cube.aux_coords
        coords = cube.dim_coords + cube_aux_coords
        cube_aux_coord_ids = {id(coord) for coord in cube_aux_coords}

        # Coordinate hint ordering dictionary - from most preferred to least.
        # Copes with duplicate hint entries, where the most preferred is king.
        hint_dict = {
            name: i
            for i, name in zip(
                range(len(self._hints), 0, -1), self._hints[::-1]
            )
        }
        # Coordinate axis ordering dictionary.
        axis_dict = {"T": 0, "Z": 1, "Y": 2, "X": 3}

        # Coordinate sort function.
        # NB. This makes use of two properties which don't end up in
        # the metadata used by scalar_defns: `coord.points.dtype` and
        # `type(coord)`.
        def key_func(coord):
            points_dtype = coord.dtype
            return (
                not np.issubdtype(points_dtype, np.number),
                not isinstance(coord, iris.coords.DimCoord),
                hint_dict.get(coord.name(), len(hint_dict) + 1),
                axis_dict.get(
                    iris.util.guess_coord_axis(coord), len(axis_dict) + 1
                ),
                coord.metadata,
            )

        # Order the coordinates by hints, axis, and definition.
        for coord in sorted(coords, key=key_func):
            if not cube.coord_dims(coord) and coord.shape == (1,):
                # Extract the scalar coordinate data and metadata.
                scalar_defns.append(coord.metadata)
                # Because we know there's a single Cell in the
                # coordinate, it's quicker to roll our own than use
                # Coord.cell().
                points = coord.points
                bounds = coord.bounds
                points_dtype = points.dtype
                if bounds is not None:
                    bounds_dtype = bounds.dtype
                    bounds = bounds[0]
                else:
                    bounds_dtype = None
                scalar_values.append(iris.coords.Cell(points[0], bounds))
                kwargs = {}
                if isinstance(coord, iris.coords.DimCoord):
                    kwargs["circular"] = coord.circular
                scalar_metadata.append(
                    _CoordMetaData(points_dtype, bounds_dtype, kwargs)
                )
            else:
                # Extract the vector coordinate and metadata.
                if id(coord) in cube_aux_coord_ids:
                    vector_aux_coords_and_dims.append(
                        _CoordAndDims(coord, tuple(cube.coord_dims(coord)))
                    )
                else:
                    vector_dim_coords_and_dims.append(
                        _CoordAndDims(coord, tuple(cube.coord_dims(coord)))
                    )

        factory_defns = []
        for factory in sorted(
            cube.aux_factories, key=lambda factory: factory.metadata
        ):
            dependency_defns = []
            dependencies = factory.dependencies
            for key in sorted(dependencies):
                coord = dependencies[key]
                if coord is not None:
                    dependency_defns.append((key, coord.metadata))
            factory_defn = _FactoryDefn(type(factory), dependency_defns)
            factory_defns.append(factory_defn)

        scalar = _ScalarCoordPayload(
            scalar_defns, scalar_values, scalar_metadata
        )
        vector = _VectorCoordPayload(
            vector_dim_coords_and_dims, vector_aux_coords_and_dims
        )

        return _CoordPayload(scalar, vector, factory_defns)
