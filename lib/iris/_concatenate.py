# (C) British Crown Copyright 2013 - 2014, Met Office
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
Automatic concatenation of multiple cubes over one or more existing dimensions.

.. warning::

    Currently, the :func:`concatenate` routine will load the data payload
    of all cubes passed to it.

    This restriction will be relaxed in a future release.

"""

from collections import defaultdict, namedtuple

import numpy as np
import numpy.ma as ma

import iris.coords
import iris.cube
from iris.util import guess_coord_axis, array_equal, unify_time_units


#
# TODO:
#
#   * Deal with scalar coordinate promotion to a new dimension
#     e.g. promote scalar z coordinate in 2D cube (y:m, x:n) to
#     give the similar 3D cube (z:1, y:m, x:n). These two types
#     of cubes are one and the same, and as such should concatenate
#     together.
#
#   * Cope with auxiliary coordinate factories.
#
#   * Don't load the cube data payload.
#
#   * Deal with anonymous dimensions.
#
#   * Allow concatentation over a user specified dimension.
#


# Restrict the names imported from this namespace.
__all__ = ['concatenate']

# Direction of dimension coordinate value order.
_CONSTANT = 0
_DECREASING = -1
_INCREASING = 1


class _CoordAndDims(namedtuple('CoordAndDims',
                               ['coord', 'dims'])):
    """
    Container for a coordinate and the associated data dimension(s)
    spanned over a :class:`iris.cube.Cube`.

    Args:

    * coord:
        A :class:`iris.coords.DimCoord` or :class:`iris.coords.AuxCoord`
        coordinate instance.

    * dims:
        A tuple of the data dimension(s) spanned by the coordinate.

    """


class _CoordMetaData(namedtuple('CoordMetaData',
                                ['defn', 'dims', 'points_dtype',
                                 'bounds_dtype', 'kwargs'])):
    """
    Container for the metadata that defines a dimension or auxiliary
    coordinate.

    Args:

    * defn:
        The :class:`iris.coords.CoordDefn` metadata that represents a
        coordinate.

    * dims:
        The dimension(s) associated with the coordinate.

    * points_dtype:
        The points data :class:`np.dtype` of an associated coordinate.

    * bounds_dtype:
        The bounds data :class:`np.dtype` of an associated coordinate.

    * kwargs:
        A dictionary of key/value pairs required to define a coordinate.

    """
    def __new__(cls, coord, dims):
        """
        Create a new :class:`_CoordMetaData` instance.

        Args:

        * coord:
            The :class:`iris.coord.DimCoord` or :class:`iris.coord.AuxCoord`.

        * dims:
            The dimension(s) associated with the coordinate.

        Returns:
            The new class instance.

        """
        defn = coord._as_defn()
        points_dtype = coord.points.dtype
        bounds_dtype = coord.bounds.dtype if coord.bounds is not None \
            else None
        kwargs = {}
        # Add circular flag metadata for dimensional coordinates.
        if hasattr(coord, 'circular'):
            kwargs['circular'] = coord.circular
        if isinstance(coord, iris.coords.DimCoord):
            # Mix the monotonic ordering into the metadata.
            if coord.points[0] == coord.points[-1]:
                order = _CONSTANT
            elif coord.points[-1] > coord.points[0]:
                order = _INCREASING
            else:
                order = _DECREASING
            kwargs['order'] = order
        metadata = super(_CoordMetaData, cls).__new__(cls, defn, dims,
                                                      points_dtype,
                                                      bounds_dtype,
                                                      kwargs)
        return metadata


class _SkeletonCube(namedtuple('SkeletonCube',
                               ['signature', 'data'])):
    """
    Basis of a source-cube, containing the associated coordinate metadata,
    coordinates and cube data payload.

    Args:

    * signature:
        The :class:`_CoordSignature` of an associated source-cube.

    * data:
        The data payload of an associated :class:`iris.cube.Cube` source-cube.

    """


class _Extent(namedtuple('Extent',
                         ['min', 'max'])):
    """
    Container representing the limits of a one-dimensional extent/range.

    Args:

    * min:
        The minimum value of the extent.

    * max:
        The maximum value of the extent.

    """


class _CoordExtent(namedtuple('CoordExtent',
                              ['points', 'bounds'])):
    """
    Container representing the points and bounds extent of a one dimensional
    coordinate.

    Args:

    * points:
        The :class:`_Extent` of the coordinate point values.

    * bounds:
        A list containing the :class:`_Extent` of the coordinate lower
        bound and the upper bound. Defaults to None if no associated
        bounds exist for the coordinate.

    """


def _name(coord, default='unknown'):
    """
    Returns a human-readable name.

    First it tries self.standard_name, then it tries the 'long_name'
    attribute, then the 'var_name' attribute, before falling back to
    the value of `default` (which itself defaults to 'unknown').

    Note this function is an exact duplicate of :meth:`cube.metadata.name`.

    """
    return coord.standard_name or coord.long_name or coord.var_name or default


def concatenate(cubes, error_on_mismatch=False):
    """
    Concatenate the provided cubes over common existing dimensions.

    Args:

    * cubes:
        An iterable containing one or more :class:`iris.cube.Cube` instances
        to be concatenated together.

    Kwargs:

    * error_on_mismatch:
        If True, raise an informative
        :class:`~iris.exceptions.ContatenateError` if registration fails.

    Returns:
        A :class:`iris.cube.CubeList` of concatenated :class:`iris.cube.Cube`
        instances.

    .. warning::

        This routine will load your data payload!

    """
    proto_cubes_by_name = defaultdict(list)
    # Initialise the nominated axis (dimension) of concatenation
    # which requires to be negotiated.
    axis = None

    # Register each cube with its appropriate proto-cube.
    for cube in cubes:
        # TODO: Remove this when new deferred data mechanism is available.
        # Avoid deferred data/data manager issues, and load the cube data!
        cube.data

        name = cube.standard_name or cube.long_name
        proto_cubes = proto_cubes_by_name[name]
        registered = False

        # Register cube with an existing proto-cube.
        for proto_cube in proto_cubes:
            registered = proto_cube.register(cube, axis, error_on_mismatch)
            if registered:
                axis = proto_cube.axis
                break

        # Create a new proto-cube for an unregistered cube.
        if not registered:
            proto_cubes.append(_ProtoCube(cube))

    # Construct a concatenated cube from each of the proto-cubes.
    concatenated_cubes = iris.cube.CubeList()

    for name in sorted(proto_cubes_by_name):
        for proto_cube in proto_cubes_by_name[name]:
            # Construct the concatenated cube.
            concatenated_cubes.append(proto_cube.concatenate())

    # Perform concatenation until we've reached an equilibrium.
    count = len(concatenated_cubes)
    if count != 1 and count != len(cubes):
        concatenated_cubes = concatenate(concatenated_cubes)

    return concatenated_cubes


class _CubeSignature(object):
    """
    Template for identifying a specific type of :class:`iris.cube.Cube` based
    on its metadata and coordinates.

    """
    def __init__(self, cube):
        """
        Represents the cube metadata and associated coordinate metadata that
        allows suitable cubes for concatenation to be identified.

        Args:

        * cube:
            The :class:`iris.cube.Cube` source-cube.

        """
        self.aux_coords_and_dims = []
        self.aux_metadata = []
        self.dim_coords = cube.dim_coords
        self.dim_metadata = []
        self.ndim = cube.ndim
        self.scalar_coords = []

        # Determine whether there are any anonymous cube dimensions.
        covered = set(cube.coord_dims(coord)[0] for coord in self.dim_coords)
        self.anonymous = covered != set(range(self.ndim))

        self.defn = cube.metadata
        self.data_type = cube.data.dtype

        #
        # Collate the dimension coordinate metadata.
        #
        for coord in self.dim_coords:
            metadata = _CoordMetaData(coord, cube.coord_dims(coord))
            self.dim_metadata.append(metadata)

        #
        # Collate the auxiliary coordinate metadata and scalar coordinates.
        #
        axes = dict(T=0, Z=1, Y=2, X=3)
        # Coordinate sort function - by guessed coordinate axis, then
        # by coordinate definition, then by dimensions, in ascending order.
        key_func = lambda coord: (axes.get(guess_coord_axis(coord),
                                           len(axes) + 1),
                                  coord._as_defn(),
                                  cube.coord_dims(coord))

        for coord in sorted(cube.aux_coords, key=key_func):
            dims = cube.coord_dims(coord)
            if dims:
                metadata = _CoordMetaData(coord, dims)
                self.aux_metadata.append(metadata)
                coord_and_dims = _CoordAndDims(coord, tuple(dims))
                self.aux_coords_and_dims.append(coord_and_dims)
            else:
                self.scalar_coords.append(coord)

    def _coordinate_differences(self, other, attr):
        """
        Determine the names of the coordinates that differ between `self` and
        `other` for a coordinate attribute on a _CubeSignature.

        Args:

        * other (_CubeSignature):
            The _CubeSignature to compare against.

        * attr (string):
            The _CubeSignature attribute within which differences exist
            between `self` and `other`.

        Returns:
            Tuple of a descriptive error message and the names of coordinates
            that differ between `self` and `other`.

        """
        # Set up {name: coord_metadata} dictionaries.
        try:
            self_dict = {x.defn.name(): x for x in getattr(self, attr)}
            other_dict = {x.defn.name(): x for x in getattr(other, attr)}
        except AttributeError:
            self_dict = {_name(x): x for x in getattr(self, attr)}
            other_dict = {_name(x): x for x in getattr(other, attr)}
        if len(self_dict.keys()) == 0:
            self_dict = {'< None >': None}
        if len(other_dict.keys()) == 0:
            other_dict = {'< None >': None}
        self_names = self_dict.keys()
        other_names = other_dict.keys()

        # Compare coord metadata.
        if len(self_names) != len(other_names) or self_names != other_names:
            result = ('', ', '.join(self_names), ', '.join(other_names))
        else:
            diff_names = []
            for self_key, self_value in self_dict.iteritems():
                other_value = other_dict[self_key]
                if self_value != other_value:
                    diff_names.append(self_key)
            result = (' metadata',
                      ', '.join(diff_names),
                      ', '.join(diff_names))
        return result

    def match(self, other, error_on_mismatch):
        """
        Return whether this _CubeSignature equals another.

        This is the first step to determine if two "cubes" (either a
        real Cube or a ProtoCube) can be concatenated, by considering:
            - data dimensions
            - dimensions metadata
            - aux coords metadata
            - scalar coords
            - attributes
            - dtype

        Args:

        * other (_CubeSignature):
            The _CubeSignature to compare against.

        * error_on_mismatch (bool):
            If True, raise a :class:`~iris.exceptions.MergeException`
            with a detailed explanation if the two do not match.

        Returns:
           Boolean. True if and only if this _CubeSignature matches the other.

        """
        msg_template = '{}{} differ: {} != {}'
        msgs = []

        # Check if either cube is anonymous.
        if self.anonymous or other.anonymous:
            msg = ('Dimensions differ: one or both cubes have anonymous '
                   'dimensions')
            msgs.append(msg)
        # Check cube definitions.
        if self.defn != other.defn:
            # Note that the case of different phenomenon names is dealt with
            # in :meth:`iris.cube.CubeList.concatenate_cube()`.
            msg = 'Cube metadata differs for phenomenon: {}'
            msgs.append(msg.format(_name(self.defn)))
        # Check dim coordinates.
        if self.dim_metadata != other.dim_metadata:
            differences = self._coordinate_differences(other, 'dim_metadata')
            msgs.append(msg_template.format('Dimension coordinates',
                                            *differences))
        # Check aux coordinates.
        if self.aux_metadata != other.aux_metadata:
            differences = self._coordinate_differences(other, 'aux_metadata')
            msgs.append(msg_template.format('Auxiliary coordinates',
                                            *differences))
        # Check scalar coordinates.
        if self.scalar_coords != other.scalar_coords:
            differences = self._coordinate_differences(other, 'scalar_coords')
            msgs.append(msg_template.format('Scalar coordinates',
                                            *differences))
        # Check ndim.
        if self.ndim != other.ndim:
            msgs.append(msg_template.format('Data dimensions', '',
                                            self.ndim, other.ndim))
        # Check datatype.
        if self.data_type != other.data_type:
            msgs.append(msg_template.format('Datatypes', '',
                                            self.data_type, other.data_type))

        match = not bool(msgs)
        if error_on_mismatch and not match:
            raise iris.exceptions.ConcatenateError(msgs)
        return match


class _CoordSignature(object):
    """
    Template for identifying a specific type of :class:`iris.cube.Cube` based
    on its coordinates.

    """
    def __init__(self, cube_signature):
        """
        Represents the coordinate metadata required to identify suitable
        non-overlapping :class:`iris.cube.Cube` source-cubes for
        concatenation over a common single dimension.

        Args:

        * cube_signature:
            The :class:`_CubeSignature` that defines the source-cube.

        """
        self.aux_coords_and_dims = cube_signature.aux_coords_and_dims
        self.dim_coords = cube_signature.dim_coords
        self.dim_extents = []
        self.dim_order = [metadata.kwargs['order']
                          for metadata in cube_signature.dim_metadata]

        # Calculate the extents for each dimensional coordinate.
        self._calculate_extents()

    @staticmethod
    def _cmp(coord, other):
        """
        Compare the coordinates for concatenation compatibility.

        Returns:
            A boolean tuple pair of whether the coordinates are compatible,
            and whether they represent a candidate axis of concatenation.

        """
        # A candidate axis must have non-identical coordinate points.
        candidate_axis = not array_equal(coord.points, other.points)

        if candidate_axis:
            # Ensure both have equal availability of bounds.
            result = (coord.bounds is None) == (other.bounds is None)
        else:
            if coord.bounds is not None and other.bounds is not None:
                # Ensure equality of bounds.
                result = array_equal(coord.bounds, other.bounds)
            else:
                # Ensure both have equal availability of bounds.
                result = coord.bounds is None and other.bounds is None

        return result, candidate_axis

    def candidate_axis(self, other):
        """
        Determine the candidate axis of concatenation with the
        given coordinate signature.

        If a candidate axis is found, then the coordinate
        signatures are compatible.

        Args:

        * other:
            The :class:`_CoordSignature`

        Returns:
            None if no single candidate axis exists, otherwise
            the candidate axis of concatenation.

        """
        result = False
        candidate_axes = []

        # Compare dimension coordinates.
        for dim, coord in enumerate(self.dim_coords):
            result, candidate_axis = self._cmp(coord,
                                               other.dim_coords[dim])
            if not result:
                break
            if candidate_axis:
                candidate_axes.append(dim)

        # Only permit one degree of dimensional freedom when
        # determining the candidate axis of concatenation.
        if result and len(candidate_axes) == 1:
            result = candidate_axes[0]
        else:
            result = None

        return result

    def _calculate_extents(self):
        """
        Calculate the extent over each dimension coordinates points and bounds.

        """
        self.dim_extents = []
        for coord, order in zip(self.dim_coords, self.dim_order):
            if order == _CONSTANT or order == _INCREASING:
                points = _Extent(coord.points[0], coord.points[-1])
                if coord.bounds is not None:
                    bounds = (_Extent(coord.bounds[0, 0], coord.bounds[-1, 0]),
                              _Extent(coord.bounds[0, 1], coord.bounds[-1, 1]))
                else:
                    bounds = None
            else:
                # The order must be decreasing ...
                points = _Extent(coord.points[-1], coord.points[0])
                if coord.bounds is not None:
                    bounds = (_Extent(coord.bounds[-1, 0], coord.bounds[0, 0]),
                              _Extent(coord.bounds[-1, 1], coord.bounds[0, 1]))
                else:
                    bounds = None

            self.dim_extents.append(_CoordExtent(points, bounds))


class _ProtoCube(object):
    """
    Framework for concatenating multiple source-cubes over one
    common dimension.

    """
    def __init__(self, cube):
        """
        Create a new _ProtoCube from the given cube and record the cube
        as a source-cube.

        Args:

        * cube:
            Source :class:`iris.cube.Cube` of the :class:`_ProtoCube`.

        """
        # Cache the source-cube of this proto-cube.
        self._cube = cube

        # The cube signature is a combination of cube and coordinate
        # metadata that defines this proto-cube.
        self._cube_signature = _CubeSignature(cube)
        self._data_is_masked = ma.isMaskedArray(cube.data)

        # The coordinate signature allows suitable non-overlapping
        # source-cubes to be identified.
        self._coord_signature = _CoordSignature(self._cube_signature)

        # The list of source-cubes relevant to this proto-cube.
        self._skeletons = []
        self._add_skeleton(self._coord_signature, cube.data)

        # The nominated axis of concatenation.
        self._axis = None

    @property
    def axis(self):
        """Return the nominated dimension of concatenation."""

        return self._axis

    def concatenate(self):
        """
        Concatenates all the source-cubes registered with the
        :class:`_ProtoCube` over the nominated common dimension.

        Returns:
            The concatenated :class:`iris.cube.Cube`.

        """
        if len(self._skeletons) > 1:
            skeletons = self._skeletons
            order = self._coord_signature.dim_order[self.axis]
            cube_signature = self._cube_signature

            # Sequence the skeleton segments into the correct order
            # pending concatenation.
            key_func = lambda skeleton: skeleton.signature.dim_extents
            skeletons.sort(key=key_func,
                           reverse=(order == _DECREASING))

            # Concatenate the new dimension coordinate.
            dim_coords_and_dims = self._build_dim_coordinates()

            # Concatenate the new auxiliary coordinates.
            aux_coords_and_dims = self._build_aux_coordinates()

            # Concatenate the new data payload.
            data = self._build_data()

            # Build the new cube.
            kwargs = cube_signature.defn._asdict()
            cube = iris.cube.Cube(data,
                                  dim_coords_and_dims=dim_coords_and_dims,
                                  aux_coords_and_dims=aux_coords_and_dims,
                                  **kwargs)
        else:
            # There are no other source-cubes to concatenate
            # with this proto-cube.
            cube = self._cube

        return cube

    def register(self, cube, axis=None, error_on_mismatch=False):
        """
        Determine whether the given source-cube is suitable for concatenation
        with this :class:`_ProtoCube`.

        Args:

        * cube:
            The :class:`iris.cube.Cube` source-cube candidate for
            concatenation.

        Kwargs:

        * axis:
            Seed the dimension of concatenation for the :class:`_ProtoCube`
            rather than rely on negotiation with source-cubes.

        * error_on_mismatch:
            If True, raise an informative error if registration fails.

        Returns:
            Boolean.

        """
        # Verify and assert the nominated axis.
        if axis is not None and self.axis is not None and self.axis != axis:
            msg = 'Nominated axis [{}] is not equal ' \
                'to negotiated axis [{}]'.format(axis, self.axis)
            raise ValueError(msg)

        # Check for compatible cube signatures.
        cube_signature = _CubeSignature(cube)
        match = self._cube_signature.match(cube_signature, error_on_mismatch)

        # Check for compatible coordinate signatures.
        if match:
            coord_signature = _CoordSignature(cube_signature)
            candidate_axis = self._coord_signature.candidate_axis(
                coord_signature)
            match = candidate_axis is not None and \
                (candidate_axis == axis or axis is None)

        # Check for compatible coordinate extents.
        if match:
            match = self._sequence(coord_signature.dim_extents[candidate_axis],
                                   candidate_axis)

        if match:
            # Register the cube as a source-cube for this proto-cube.
            self._add_skeleton(coord_signature, cube.data)
            self._data_is_masked |= ma.isMaskedArray(cube.data)
            # Declare the nominated axis of concatenation.
            self._axis = candidate_axis

        return match

    def _add_skeleton(self, coord_signature, data):
        """
        Create and add the source-cube skeleton to the
        :class:`_ProtoCube`.

        Args:

        * coord_signature:
            The :class:`_CoordSignature` of the associated
            given source-cube.

        * data:
            The data payload of an associated :class:`iris.cube.Cube`
            source-cube.

        """
        skeleton = _SkeletonCube(coord_signature, data)
        self._skeletons.append(skeleton)

    def _build_aux_coordinates(self):
        """
        Generate the auxiliary coordinates with associated dimension(s)
        mapping for the new concatenated cube.

        Returns:
            A list of auxiliary coordinates and dimension(s) tuple pairs.

        """
        # Setup convenience hooks.
        skeletons = self._skeletons
        cube_signature = self._cube_signature

        aux_coords_and_dims = []

        # Generate all the auxiliary coordinates for the new concatenated cube.
        for i, (coord, dims) in enumerate(cube_signature.aux_coords_and_dims):
            # Check whether the coordinate spans the nominated
            # dimension of concatenation.
            if self.axis in dims:
                # Concatenate the points together.
                dim = dims.index(self.axis)
                points = [skton.signature.aux_coords_and_dims[i].coord.points
                          for skton in skeletons]
                points = np.concatenate(tuple(points), axis=dim)

                # Concatenate the bounds together.
                bnds = None
                if coord.has_bounds():
                    bnds = [skton.signature.aux_coords_and_dims[i].coord.bounds
                            for skton in skeletons]
                    bnds = np.concatenate(tuple(bnds), axis=dim)

                # Generate the associated coordinate metadata.
                kwargs = cube_signature.aux_metadata[i].defn._asdict()

                # Build the concatenated coordinate.
                if isinstance(coord, iris.coords.AuxCoord):
                    coord = iris.coords.AuxCoord(points, bounds=bnds, **kwargs)
                else:
                    # Attempt to create a DimCoord, otherwise default to
                    # an AuxCoord on failure.
                    try:
                        coord = iris.coords.DimCoord(points, bounds=bnds,
                                                     **kwargs)
                    except ValueError:
                        coord = iris.coords.AuxCoord(points, bounds=bnds,
                                                     **kwargs)

            aux_coords_and_dims.append((coord.copy(), dims))

        # Generate all the scalar coordinates for the new concatenated cube.
        for coord in cube_signature.scalar_coords:
            aux_coords_and_dims.append((coord.copy(), ()))

        return aux_coords_and_dims

    def _build_data(self):
        """
        Generate the data payload for the new concatenated cube.

        Returns:
            The concatenated :class:`iris.cube.Cube` data payload.

        """
        skeletons = self._skeletons
        data = [skeleton.data for skeleton in skeletons]

        if self._data_is_masked:
            data = ma.concatenate(tuple(data), axis=self.axis)
        else:
            data = np.concatenate(tuple(data), axis=self.axis)

        return data

    def _build_dim_coordinates(self):
        """
        Generate the dimension coordinates with associated dimension
        mapping for the new concatenated cube.

        Return:
            A list of dimension coordinate and dimension tuple pairs.

        """
        # Setup convenience hooks.
        skeletons = self._skeletons
        axis = self.axis
        defn = self._cube_signature.dim_metadata[axis].defn
        circular = self._cube_signature.dim_metadata[axis].kwargs['circular']

        # Concatenate the points together for the nominated dimension.
        points = [skeleton.signature.dim_coords[axis].points
                  for skeleton in skeletons]
        points = np.concatenate(tuple(points))

        # Concatenate the bounds together for the nominated dimension.
        bounds = None
        if self._cube_signature.dim_coords[axis].has_bounds():
            bounds = [skeleton.signature.dim_coords[axis].bounds
                      for skeleton in skeletons]
            bounds = np.concatenate(tuple(bounds))

        # Populate the new dimension coordinate with the concatenated
        # points, bounds and associated metadata.
        kwargs = defn._asdict()
        kwargs['circular'] = circular
        dim_coord = iris.coords.DimCoord(points, bounds=bounds, **kwargs)

        # Generate all the dimension coordinates for the new concatenated cube.
        dim_coords_and_dims = []
        for dim, coord in enumerate(self._cube_signature.dim_coords):
            if dim == axis:
                dim_coords_and_dims.append((dim_coord, dim))
            else:
                dim_coords_and_dims.append((coord.copy(), dim))

        return dim_coords_and_dims

    def _sequence(self, extent, axis):
        """
        Determine whether the given extent can be sequenced along with
        all the extents of the source-cubes already registered with
        this :class:`_ProtoCube` into non-overlapping segments for the
        given axis.

        Args:

        * extent:
            The :class:`_CoordExtent` of the candidate source-cube.

        * axis:
            The candidate axis of concatenation.

        Returns:
            Boolean.

        """
        result = True

        # Add the new extent to the current extents collection.
        dim_extents = [skeleton.signature.dim_extents[axis]
                       for skeleton in self._skeletons]
        dim_extents.append(extent)

        # Sort into the appropriate dimension order.
        order = self._coord_signature.dim_order[axis]
        dim_extents.sort(reverse=(order == _DECREASING))

        # Ensure that the extents don't overlap.
        if len(dim_extents) > 1:
            for i, extent in enumerate(dim_extents[1:]):
                # Check the points - must be strictly monotonic.
                if order == _DECREASING:
                    left = dim_extents[i].points.min
                    right = extent.points.max
                else:
                    left = dim_extents[i].points.max
                    right = extent.points.min

                if left >= right:
                    result = False
                    break

                # Check the bounds - must be strictly monotonic.
                if extent.bounds is not None:
                    if order == _DECREASING:
                        left_0 = dim_extents[i].bounds[0].min
                        left_1 = dim_extents[1].bounds[1].min
                        right_0 = extent.bounds[0].max
                        right_1 = extent.bounds[1].max
                    else:
                        left_0 = dim_extents[i].bounds[0].max
                        left_1 = dim_extents[i].bounds[1].max
                        right_0 = extent.bounds[0].min
                        right_1 = extent.bounds[1].min

                    lower_bound_fail = left_0 >= right_0
                    upper_bound_fail = left_1 >= right_1

                    if lower_bound_fail or upper_bound_fail:
                        result = False
                        break

        return result
