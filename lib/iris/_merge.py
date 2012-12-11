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
Automatic collation of cubes into higher-dimensional cubes.

Typically the cube merge process is handled by :method:`iris.cube.CubeList.merge`.

"""
from collections import namedtuple, Iterable
from copy import deepcopy

import numpy

import iris.cube
import iris.coords
import iris.exceptions
import iris.unit
import iris.util


#
# Private namedtuple wrapper classes.
#
class _Template(namedtuple('Template',
                           ['dims', 'points', 'bounds', 'kwargs'])):
    """
    Common framework from which to build a dimension or auxiliary coordinate.
    
    Args:

    * dims:
        Tuple of the associated :class:`iris.cube.Cube` data dimension/s spanned
        by this coordinate template.

    * points:
        A :mod:`numpy` array representing the coordinate point data. No points data
        is represented by None.

    * bounds:
        A :mod:`numpy` array representing the coordinate bounds data. No bounds data
        is represented by None.

    * kwargs:
        A dictionary of key/value pairs required to create a coordinate.

    """
        

class _CoordMetaData(namedtuple('CoordMetaData',
                                ['points_dtype', 'bounds_dtype', 'kwargs'])):
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


class _CoordAndDims(namedtuple('CoordAndDims',
                               ['coord', 'dims'])):
    """
    Container for a coordinate and the associated data dimension/s 
    spanned over a :class:`iris.cube.Cube`.

    Args:

    * coord:
        A :class:`iris.coords.DimCoord` or :class:`iris.coords.AuxCoord`
        coordinate instance.

    * dims:
        A tuple of the data dimesion/s spanned by the coordinate.

    """


class _ScalarCoordPayload(namedtuple('ScalarCoordPayload',
                                     ['defns', 'values', 'metadata'])):
    """
    Container for all scalar coordinate data and metadata represented
    within a :class:`iris.cube.Cube`.

    All scalar coordinate related data is sorted into ascending order
    of the associated coordinate definition.

    Args:

    * defns:
        A list of scalar coordinate definitions :class:`iris.coords.CoordDefn`
        belonging to a :class:`iris.cube.Cube`.

    * values:
        A list of scalar coordinate values belonging to a :class:`iris.cube.Cube`.
        Each scalar coordinate value is typically an :class:`iris.coords.Cell`.

    * metadata:
        A list of :class:`_CoordMetaData` instances belonging to a :class:`iris.cube.Cube`.

    """


class _VectorCoordPayload(namedtuple('VectorCoordPayload',
                                     ['dim_coords_and_dims', 'aux_coords_and_dims'])):
    """
    Container for all vector coordinate data and metadata represented
    within a :class:`iris.cube.Cube`.

    Args:

    * dim_coords_and_dims:
        A list of :class:`_CoordAndDim` instances containing non-scalar (i.e. multi-valued)
        :class:`iris.coords.DimCoord` instances and the associated data dimension spanned
        by them for a :class:`iris.cube.Cube`.

    * aux_coords_and_dims:
        A list of :class:`_CoordAndDim` instances containing non-scalar (i.e. multi-valued)
        :class:`iris.coords.DimCoord` and/or :class:`iris.coords.AuxCoord` instances and the 
        associated data dimension/s spanned by them for a :class:`iris.cube.Cube`.

    """


class _CoordPayload(namedtuple('CoordPayload',
                               ['scalar', 'vector', 'factory_defns'])):
    """
    Container for all the scalar and vector coordinate data and metadata, and
    auxiliary coordinate factories represented within a :class:`iris.cube.Cube`.

    All scalar coordinate and factory related data is sorted into ascending order 
    of the associated coordinate definition.

    Args:

    * scalar:
        A :class:`_ScalarCoordPayload` instance.
    
    * vector:
        A :class:`_VectorCoordPayload` instance.

    * factory_defns:
        A list of :class:`_FactoryDefn` instances.

    """
    def as_signature(self):
        """Construct and return a :class:`_CoordSignature` from the payload."""

        return _CoordSignature(self.scalar.defns,
                               self.vector.dim_coords_and_dims, self.vector.aux_coords_and_dims,
                               self.factory_defns)


class _CoordSignature(namedtuple('CoordSignature',
                                 ['scalar_defns',
                                  'vector_dim_coords_and_dims', 'vector_aux_coords_and_dims',
                                  'factory_defns'])):
    """
    Criterion for identifying a specific type of :class:`iris.cube.Cube` based on its
    scalar and vector coorinate data and metadata, and auxiliary coordinate factories.

    Args:

    * scalar_defns:
        A list of scalar coordinate definitions sorted into ascending order.

    * vector_dim_coords_and_dims:
        A list of :class:`_CoordAndDim` instances containing non-scalar (i.e. multi-valued)
        :class:`iris.coords.DimCoord` instances and the associated data dimension spanned
        by them for a :class:`iris.cube.Cube`.

    * vector_aux_coords_and_dims:
        A list of :class:`_CoordAndDim` instances containing non-scalar (i.e. multi-valued)
        :class:`iris.coords.DimCoord` and/or :class:`iris.coords.AuxCoord` instances and the
        associated data dimension/s spanned by them for a :class:`iris.cube.Cube`.

    * factory_defns:
        A list of :class:`_FactoryDefn` instances.

    """


class _CubeSignature(namedtuple('CubeSignature',
                                ['defn', 'data_shape', 'data_manager', 'data_type', 'mdi'])):
    """
    Criterion for identifying a specific type of :class:`iris.cube.Cube` based on
    its metadata.

    Args:

    * defn:
        A cube definition tuple.

    * data_shape:
        The data payload shape of a :class:`iris.cube.Cube`.

    * data_manager:
        The :class:`iris.fileformats.manager.DataManager` instance.

    * data_type:
        The data payload :class:`numpy.dtype` of a :class:`iris.cube.Cube`.

    * mdi:
        The missing data value associated with the data payload of a :class:`iris.cube.Cube`.

    """


class _Skeleton(namedtuple('Skeleton',
                           ['scalar_values', 'data'])):
    """
    Basis of a source-cube, containing the associated scalar coordinate values
    and data payload of a :class:`iris.cube.Cube`.
 
    Args:

    * scalar_values:
        A list of scalar coordinate values belonging to a :class:`iris.cube.Cube`
        sorted into ascending order of the associated coordinate definition. Each 
        scalar coordinate value is typically an :class:`iris.coords.Cell`.

    * data:
        The data payload of a :class:`iris.cube.Cube`.

    """


class _FactoryDefn(namedtuple('_FactoryDefn',
                              ['class_', 'dependency_defns'])):
    """
    The information required to identify and rebuild a single AuxCoordFactory.

    Args:

    * class_:
        The class of the AuxCoordFactory.

    * dependency_defns:
        A list of pairs, where each pair contains a dependency key and its
        corresponding coordinate definition. Sorted on dependency key.

    """


class _Relation(namedtuple('Relation',
                           ['separable', 'inseparable'])):
    """
    Categorisation of the candidate dimensions belonging to a :class:`ProtoCube`
    into separable 'independent' dimensions, and inseparable dependent dimensions.

    Args:

    * separable:
        A set of independent candidate dimension names.

    * inseperable:
        A set of dependent candidate dimension names.

    """


_COMBINATION_JOIN = '-'


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
        ...     print '%r:' % k
        ...     for kk in sorted(indexes[k]):
        ...         print '\t%r:' % kk,
        ...         for kkk in sorted(indexes[k][kk]):
        ...             print '%r: %r' % (kkk, indexes[k][kk][kkk]),
        ...         print
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
        for name, value in position.iteritems():
            name_index_by_scalar = scalar_index_by_name[name]

            if value in name_index_by_scalar:
                value_index_by_name = name_index_by_scalar[value]
                for other_name in names:
                    if other_name != name:
                        value_index_by_name[other_name].add(position[other_name])
            else:
                name_index_by_scalar[value] = {other_name: set((position[other_name],)) for other_name in names if other_name != name}

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
    items = index.itervalues()
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
        ...     print '%r: %r' % (k, v)
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
        A list containing a dictionary of candidate dimension key to scalar value
        pairs for each source-cube.

    Kwargs:

    * function_mapping:
        A dictionary that enumerates a valid functional relationship between
        the dependent candidate dimension and the independent candidate dimension/s.


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


def _derive_separable_consistent_groups(relation_matrix, separable_group):
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
        name_separable_group = relation_matrix[name].separable & separable_group
        candidate = list(name_separable_group) + [name]
        valid = True

        for _ in range(len(name_separable_group)):
            candidate_separable_group = set(candidate[1:])

            if candidate_separable_group & (relation_matrix[candidate[0]].separable & separable_group) != candidate_separable_group:
                valid = False
                break

            candidate.append(candidate.pop(0))

        if valid:
            result.append(candidate)

    return result


def _build_separable_group(space, group, separable_consistent_groups, positions, function_matrix):
    """
    Update the space with the first separable consistent group that satisfies a valid 
    functional relationship with all other candidate dimensions in the group.

    For example, the group ABCD and separable consistent group CD, if A = f(C, D) and B = f(C, D) 
    then update the space with "A: (C, D), B: (C, D), C: None, D: None". Where "A: (C, D)" means 
    that candidate dimension A is dependent on candidate dimensions C and D, and "C: None" means 
    that this candidate dimension is independent. 

    Args:

    * space:
        A dictionary defining for each candidate dimension its dependency on any other
        candidate dimensions within the space.

    * group:
        A set of related (chained) inseparable candidate dimensions.

    * separable_consistent_groups:
        A list of candidate dimension groups that are consistently separable.

    * positions:
        A list containing a dictionary of candidate dimension key to scalar value 
        pairs for each source-cube.

    * function_matrix:
        The function mapping dictionary for each candidate dimension that
        participates in a functional relationship.

    Returns:
        None.

    """
    valid = False

    for independent in sorted(separable_consistent_groups):
        dependent = list(group - set(independent))
        dependent_function_matrix = {}

        for name in dependent:
            function_mapping = {}
            valid = _is_dependent(name, independent, positions, function_mapping)

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
    else:
        raise iris.exceptions.NotYetImplementedError('No functional relationship between separable and inseparable candidate dimensions.')


def _build_inseparable_group(space, group, positions, function_matrix):
    """
    Update the space with the first valid scalar functional relationship between
    a candidate dimension within the group and all other candidate dimensions.

    For example, with the inseparable group ABCD, a valid scalar relationship B = f(A),
    C = f(A) and D = f(A) results in a space with "A: None, B: (A,), C: (A,), D: (A,)".
    Where "A: None" means that this candidate dimension is independent, and
    "B: (A,)" means that candidate dimension B is dependent on candidate dimension A.

    The scalar relationship must exist between one candidate dimension and all 
    others in the group, as the group is considered inseparable in this context.

    Args:
    
    * space:
        A dictionary defining for each candidate dimension its dependency on 
        any other candidate dimensions within the space.

    * group:
        A set of related (chained) inseparable candidate dimensions.

    * positions:
        A list containing a dictionary of candidate dimension key to scalar value
        pairs for each source-cube.

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
            valid = _is_dependent(name, independent, positions, function_mapping)
                    
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

    As no functional relationship between members of the group can be determined,
    the new combination dimension will not have a dimension coordinate associated
    with it. Rather, it is simply an enumeration of the group members for each
    of the positions (source-cubes).

    Args:

    * space:
        A dictionary defining for each candidate dimension its dependency on
        any other candidate dimensions within the space.

    * group:
        A set of related (chained) inseparable candidate dimensions.

    * positions:
        A list containing a dictionary of candidate dimension key to scalar value
        pairs for each source-cube.

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
        cell = (tuple([position[int(member) if member.isdigit() else member] for member in members]),)
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
        A space dictionary describing the relationship between each candidate dimension. 

    """
    space = {}

    for group in groups:
        separable_group = _derive_separable_group(relation_matrix, group)

        if len(group) == 1 and not separable_group:
            # This single candidate dimension is separable from all other candidate dimensions
            # in the group, therefore it is a genuine dimension of the space.
            space.update({name: None for name in group})
        elif separable_group:
            # Determine the largest combination of the candidate dimensions 
            # in the separable group that are consistently separable.
            separable_consistent_groups = _derive_separable_consistent_groups(relation_matrix, separable_group)
            _build_separable_group(space, group, separable_consistent_groups, positions, function_matrix)
        else:
            # Determine whether there is a scalar relationship between one of
            # the candidate dimensions and each of the other candidate dimensions 
            # in this inseparable group.
            if not _build_inseparable_group(space, group, positions, function_matrix):
                # There is no relationship between any of the candidate dimensions in this 
                # inseparable group, so merge them together into a new combined dimension of the space.
                _build_combination_group(space, group, positions, function_matrix)

    return space


class ProtoCube(object):
    """Framework for merging source-cubes into one or more higher dimensional cubes."""

    def __init__(self, cube):
        """Create a new ProtoCube from the given cube and record the cube as a source-cube."""

        # Default hint ordering for candidate dimension coordinates.
        self._hints = ['time', 'forecast_reference_time', 'forecast_period', 'model_level_number']

        # The cube signature is metadata that defines this ProtoCube.
        self._cube_signature = self._build_signature(cube)

        # Extract the scalar and vector coordinate data and metadata from the cube. 
        coord_payload = self._extract_coord_payload(cube)

        # The coordinate signature defines the scalar and vector coordinates of this ProtoCube.
        self._coord_signature = coord_payload.as_signature()
        self._coord_metadata = coord_payload.scalar.metadata

        # The list of stripped-down source-cubes relevant to this ProtoCube.
        self._skeletons = []
        self._add_cube(cube, coord_payload)

        self._dim_templates = []  # Proto-coordinates constructed from merged scalars.
        self._aux_templates = []  # Proto-coordinates constructed from merged scalars.
        self._shape = []
        self._nd_names = []
        self._cache_by_name = {}
        self._dim_coords_and_dims = []
        self._aux_coords_and_dims = []
        self._vector_dim_coords_dims = []  # Dims offset by merged space higher dimensionality.
        self._vector_aux_coords_dims = []  # Dims offset by merged space higher dimensionality.

    def merge(self, unique=True):
        """
        Returns the list of cubes resulting from merging the registered source-cubes.

        Kwargs:

        * unique:
            If True, raises `iris.exceptions.DuplicateDataError` if
            duplicate cubes are detected.

        Returns:
            A :class:`iris.cube.CubeList` of merged cubes.

        """
        positions = [{i: v for i, v in enumerate(skeleton.scalar_values)} for skeleton in self._skeletons]
        indexes = build_indexes(positions)
        relation_matrix = derive_relation_matrix(indexes)
        groups = derive_groups(relation_matrix)

        function_matrix = {}
        space = derive_space(groups, relation_matrix, positions, function_matrix=function_matrix)
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
        nd_indexes = group_by_nd_index.keys()
        nd_indexes.sort()
        
        # Check for unique data.
        if unique and group_depth > 1:
            # Find the first offending source-cube with duplicate metadata.
            index = [group_by_nd_index[nd_index][1] for nd_index in nd_indexes if len(group_by_nd_index[nd_index]) > 1][0]
            name = self._cube_signature.defn.name()
            scalars = []
            for defn, value in zip(self._coord_signature.scalar_defns, self._skeletons[index].scalar_values):
                scalars.append('%s=%r' % (defn.name(), value))
            raise iris.exceptions.DuplicateDataError('Duplicate %r cube, with scalar coordinates %s' % (name, ', '.join(scalars)))

        # Generate group-depth merged cubes from the source-cubes.
        for level in xrange(group_depth):
            # The merged cube's data will be an array of data proxies for deferred loading.
            merged_cube = self._get_cube()

            for nd_index in nd_indexes:
                # Get the data of the current existing or last known good source-cube
                offset = min(level, len(group_by_nd_index[nd_index]) - 1)
                data = self._skeletons[group_by_nd_index[nd_index][offset]].data

                # Slot the data into merged cube. The nd-index will have less dimensionality than
                # that of the merged cube's data. The "missing" dimensions correspond to the 
                # dimensionality of the source-cubes data.
                if nd_index:
                    # The use of "flatten" allows us to cope with a 0-dimensional array.
                    # Otherwise, the assignment copies the 0-d *array* into the merged cube,
                    # and not the contents of the array!
                    if data.ndim == 0:
                        merged_cube._data[nd_index] = data.flatten()[0]
                    else:
                        merged_cube._data[nd_index] = data
                else:
                    merged_cube._data = data
            
            # Unmask the array only if it is filled.
            if isinstance(merged_cube._data, numpy.ma.core.MaskedArray):
                if numpy.ma.count_masked(merged_cube._data) == 0:
                    merged_cube._data = merged_cube._data.filled()

            merged_cubes.append(merged_cube)

        return merged_cubes

    def register(self, cube):
        """
        Add a compatible :class:`iris.cube.Cube` as a source-cube for merging under this 
        :class:`ProtoCube`.

        A cube will be deemed compatible based on the signature of the cube and the signature of its
        scalar coordinates and vector coordinates being identical to that of the ProtoCube.

        Args:

        * cube:
            Candidate :class:`iris.cube.Cube` to be associated with this :class:`ProtoCube`.
        
        Returns:
            True iff the :class:`iris.cube.Cube` is compatible with this :class:`ProtoCube`.

        """
        match = self._cube_signature == self._build_signature(cube)

        if match:
            coord_payload = self._extract_coord_payload(cube)
            signature = coord_payload.as_signature()
            match = self._coord_signature == signature

        if match:
            # Register the cube as a source-cube for this ProtoCube.
            self._add_cube(cube, coord_payload)

        return match

    def _guess_axis(self, name):
        """
        Returns a "best guess" axis name of the candidate dimension.

        Heuristic categoration of the candidate dimension (i.e. scalar_defn index)
        into either label 'T', 'Z', 'Y', 'X' or None.

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
        Given the derived :class:`ProtoCube` space, define this space in terms of its
        dimensionality, shape, coordinates and associated coordinate to space dimension mappings.

        Args:

        * space:
            A dictionary defining for each candidate dimension its dependency on any other
            candidate dimensions within the space.

        * positions:
            A list containing a dictionary of candidate dimension key to
            scalar value pairs for each source-cube.

        * indexes:
            A cross-reference dictionary for each candidate dimension.

        * function_matrix:
            The function mapping dictionary for each candidate dimension that
            participates in a functional relationship.

        """
        # Heuristic reordering of coordinate defintion indexes into preferred dimension order.
        names = sorted(space, 
                       key=lambda name: ({'T':1, 'Z':2, 'Y':3, 'X':4}.get(self._guess_axis(name), 0), name))
        dim_by_name = {}

        metadata = self._coord_metadata
        defns = self._coord_signature.scalar_defns
        vector_dim_coords_and_dims = self._coord_signature.vector_dim_coords_and_dims
        vector_aux_coords_and_dims = self._coord_signature.vector_aux_coords_and_dims
        signature = self._cube_signature

        # First pass - Build the dimension coordinate templates for the space.
        for name in names:
            if space[name] is None:
                if _is_combination(name):
                    members = name.split(_COMBINATION_JOIN)
                    cells = [tuple([position[int(member) if member.isdigit() else member] for member in members]) for position in positions]
                    dim_by_name[name] = len(self._shape)
                    self._nd_names.append(name)
                    self._shape.append(len(cells))
                    self._cache_by_name[name] = {cell:index for index, cell in enumerate(cells)}
                else:
                    # TODO: Consider appropriate sort order (ascending, decending) i.e. use CF positive attribute.
                    cells = sorted(indexes[name])
                    points = numpy.array([cell.point for cell in cells], dtype=metadata[name].points_dtype)
                    bounds = numpy.array([cell.bound for cell in cells], dtype=metadata[name].bounds_dtype) if cells[0].bound is not None else None
                    kwargs = dict(zip(iris.coords.CoordDefn._fields, defns[name]))
                    kwargs.update(metadata[name].kwargs)

                    if len(cells) == 1 and not any([name in independents for independents in 
                                                    space.itervalues() if independents is not None]):
                        # A scalar coordinate not participating in a function dependency.
                        self._aux_templates.append(_Template((), points, bounds, kwargs))
                    else:
                        # Dimension coordinate (or aux if the data is string like).
                        dim_by_name[name] = dim = len(self._shape)
                        self._nd_names.append(name)
                        if metadata[name].points_dtype.kind == 'S':
                            self._aux_templates.append(_Template(dim, points, bounds, kwargs))
                        else:
                            self._dim_templates.append(_Template(dim, points, bounds, kwargs))
                        self._shape.append(len(cells))
                        self._cache_by_name[name] = {cell:index for index, cell in enumerate(cells)}

        # Second pass - Build the auxiliary coordinate templates for the space.
        for name in names:
            name_independents = space[name]

            # Determine if there is a function dependency.
            if name_independents is not None:
                # Calculate the auxiliary coordinate shape.
                dims = tuple([dim_by_name[independent] for independent in name_independents])
                aux_shape = [self._shape[dim] for dim in dims]
                # Create empty points and bounds in preparation to be filled.
                points = numpy.empty(aux_shape, dtype=metadata[name].points_dtype)
                bounds = numpy.empty(aux_shape + [len(positions[0][name].bound)], dtype=metadata[name].bounds_dtype) if positions[0][name].bound is not None else None

                # Populate the points and bounds based on the appropriate function mapping.
                for function_independents, name_value in function_matrix[name].iteritems():
                    # Build the index (and cache it) for the auxiliary coordinate based on the 
                    # associated independent dimension coordinate/s.
                    index = []

                    for independent, independent_value in zip(name_independents, function_independents):
                        index.append(self._cache_by_name[independent][independent_value])

                    index = tuple(index)
                    if points is not None:
                        points[index] = name_value.point

                    if bounds is not None:
                        bounds[index] = name_value.bound
                        
                kwargs = dict(zip(iris.coords.CoordDefn._fields, defns[name]))
                self._aux_templates.append(_Template(dims, points, bounds, kwargs))
                
        # Calculate the dimension mapping for each vector within the space. 
        offset = len(self._shape)
        self._vector_dim_coords_dims = [tuple([dim + offset for dim in item.dims]) for item in vector_dim_coords_and_dims]
        self._vector_aux_coords_dims = [tuple([dim + offset for dim in item.dims]) for item in vector_aux_coords_and_dims]

        # Now factor in the vector payload shape. Note that, for
        # deferred loading, this does NOT change the shape.
        self._shape.extend(signature.data_shape)

    def _get_cube(self):
        """
        Returns a cube containing all its coordinates and appropriately shaped
        data that corresponds to this ProtoCube.

        All the values in the cube's data array are masked.

        """
        signature = self._cube_signature
        dim_coords_and_dims = [(deepcopy(coord), dim) for coord, dim in self._dim_coords_and_dims]
        aux_coords_and_dims = [(deepcopy(coord), dims) for coord, dims in self._aux_coords_and_dims]
        kwargs = dict(zip(iris.cube.CubeMetadata._fields, signature.defn))

        # Create fully masked data, i.e. all missing.
        # (The CubeML checksum doesn't respect the mask, so we zero the
        # underlying data to ensure repeatable checksums.)
        if signature.data_manager is None:
            data = numpy.ma.MaskedArray(numpy.zeros(self._shape,
                                                    signature.data_type),
                                        mask=numpy.ones(self._shape, 'bool'),
                                        fill_value=signature.mdi)
        else:
            data = numpy.ma.MaskedArray(numpy.zeros(self._shape, 'object'),
                                        mask=numpy.ones(self._shape, 'bool'))

        cube = iris.cube.Cube(data,
                              dim_coords_and_dims=dim_coords_and_dims,
                              aux_coords_and_dims=aux_coords_and_dims,
                              data_manager=signature.data_manager, **kwargs)

        # Add on any aux coord factories.
        for factory_defn in self._coord_signature.factory_defns:
            args = {}
            for key, defn in factory_defn.dependency_defns:
                coord = cube.coord(coord=defn)
                args[key] = coord
            factory = factory_defn.class_(**args)
            cube.add_aux_factory(factory)

        return cube

    def _nd_index(self, position):
        """Returns the n-dimensional index of this source-cube (position), within the merged cube."""

        index = []

        # Determine the index of the source-cube cell for each dimension.
        for name in self._nd_names:
            if _is_combination(name):
                members = name.split(_COMBINATION_JOIN)
                cell = tuple([position[int(member) if member.isdigit() else member] for member in members])
                index.append(self._cache_by_name[name][cell])
            else:
                index.append(self._cache_by_name[name][position[name]])

        return tuple(index)

    def _build_coordinates(self):
        """
        Build the dimension and auxiliary coordinates for the final merged cube given that the final 
        dimensionality of the target merged cube is known and the associated dimension/s that each 
        coordinate maps onto in that merged cube.

        The associated vector coordinate/s of the ProtoCube are also created with the correct space
        dimensionality and dimension/s mappings.

        """
        # Containers for the newly created coordinates and associated dimension mappings.
        dim_coords_and_dims = self._dim_coords_and_dims
        aux_coords_and_dims = self._aux_coords_and_dims

        # Build the dimension coordinates.
        for template in self._dim_templates:
            dim_coords_and_dims.append(_CoordAndDims(iris.coords.DimCoord(template.points,
                                                                              bounds=template.bounds,
                                                                              **template.kwargs), template.dims))

        # Build the auxiliary coordinates.
        for template in self._aux_templates:
            # Attempt to build a DimCoord and add it to the cube. If this fails e.g it's
            # non-monontic or multi-dimensional or non-numeric, then build an AuxCoord.
            try:
                aux_coords_and_dims.append(_CoordAndDims(iris.coords.DimCoord(template.points, 
                                                                              bounds=template.bounds, 
                                                                              **template.kwargs), template.dims))
            except ValueError:
                template.kwargs.pop('circular', None) # kwarg not applicable to AuxCoord.
                aux_coords_and_dims.append(_CoordAndDims(iris.coords.AuxCoord(template.points, 
                                                                              bounds=template.bounds, 
                                                                              **template.kwargs), template.dims))

        # Mix in the vector coordinates.
        for item, dims in zip(self._coord_signature.vector_dim_coords_and_dims, self._vector_dim_coords_dims):
            dim_coords_and_dims.append(_CoordAndDims(item.coord, dims))

        for item, dims in zip(self._coord_signature.vector_aux_coords_and_dims, self._vector_aux_coords_dims):
            aux_coords_and_dims.append(_CoordAndDims(item.coord, dims))

    def _build_signature(self, cube):
        """Generate the signature that defines this cube."""

        defn = cube.metadata
        data_shape = cube._data.shape
        data_manager = cube._data_manager
        mdi = None

        if data_manager is None:
            data_type = cube._data.dtype.name
            if isinstance(cube.data, numpy.ma.core.MaskedArray):
                mdi = cube.data.fill_value
        else:
            data_type = data_manager.data_type.name
            mdi = data_manager.mdi

        return _CubeSignature(defn, data_shape, data_manager, data_type, mdi)

    def _add_cube(self, cube, coord_payload):
        """Create and add the source-cube skeleton to the ProtoCube."""

        skeleton = _Skeleton(coord_payload.scalar.values, cube._data)
        self._skeletons.append(skeleton)

    def _extract_coord_payload(self, cube):
        """
        Extract all relevant coordinate data and metadata from the cube.

        In particular, for each scalar coordinate determine its definition,
        its cell (point and bound) value and all other scalar coordinate metadata
        that allows us to fully reconstruct that scalar coordinate. Note that all
        scalar data is sorted in order of the scalar coordinate definition.

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
        
        # Coordinate hint ordering dictionary - from most preferred to least.
        # Copes with duplicate hint entries, where the most preferred is king.
        hint_dict = {name: i for i, name in zip(range(len(self._hints), 0, -1), self._hints[::-1])}
        # Coordinate axis ordering dictionary.
        axis_dict = {'T': 0, 'Z': 1, 'Y': 2, 'X': 3}
        # Coordinate sort function - by coordinate hint, then by guessed coordinate axis, then
        # by coordinate definition, in ascending order.
        key_func = lambda coord: (hint_dict.get(coord.name(), len(hint_dict) + 1),
                                  axis_dict.get(iris.util.guess_coord_axis(coord), len(axis_dict) + 1),
                                  coord._as_defn())

        # Order the coordinates by hints, axis, and definition.
        for coord in sorted(coords, key=key_func):
            if not cube.coord_dims(coord) and coord.shape == (1,):
                # Extract the scalar coordinate data and metadata.
                scalar_defns.append(coord._as_defn())
                scalar_values.append(coord.cell(0))
                points_dtype = coord.points.dtype
                bounds_dtype = coord.bounds.dtype if coord.bounds is not None else None
                kwargs = {'circular': coord.circular} if isinstance(coord, iris.coords.DimCoord) else {}
                scalar_metadata.append(_CoordMetaData(points_dtype, bounds_dtype, kwargs))
            else:
                # Extract the vector coordinate and metadata.
                if coord in cube_aux_coords:
                    vector_aux_coords_and_dims.append(_CoordAndDims(coord, tuple(cube.coord_dims(coord))))
                else:
                    vector_dim_coords_and_dims.append(_CoordAndDims(coord, tuple(cube.coord_dims(coord))))
 
        factory_defns = []
        for factory in sorted(cube.aux_factories, key=lambda factory: factory._as_defn()):
            dependency_defns = []
            dependencies = factory.dependencies
            for key in sorted(dependencies):
                coord = dependencies[key]
                dependency_defns.append((key, coord._as_defn()))
            factory_defn = _FactoryDefn(type(factory), dependency_defns)
            factory_defns.append(factory_defn)

        scalar = _ScalarCoordPayload(scalar_defns, scalar_values, scalar_metadata)
        vector = _VectorCoordPayload(vector_dim_coords_and_dims, vector_aux_coords_and_dims)

        return _CoordPayload(scalar, vector, factory_defns)
