# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

"""
Infra-structure for unstructured mesh support, based on
CF UGRID Conventions (v1.0), https://ugrid-conventions.github.io/ugrid-conventions/

"""

from abc import ABC, abstractmethod
from collections import Iterable, namedtuple
from functools import wraps
import re

import dask.array as da
import numpy as np

from .. import _lazy_data as _lazy
from ..common.metadata import (
    BaseMetadata,
    metadata_filter,
    metadata_manager_factory,
    SERVICES,
    SERVICES_COMBINE,
    SERVICES_EQUAL,
    SERVICES_DIFFERENCE,
)
from ..common.lenient import _lenient_service as lenient_service
from ..common.mixin import CFVariableMixin
from ..config import get_logger
from ..coords import _DimensionalMetadata, AuxCoord
from ..exceptions import ConnectivityNotFoundError, CoordinateNotFoundError
from ..util import guess_coord_axis


__all__ = [
    "Connectivity",
    "ConnectivityMetadata",
    "Mesh",
    "Mesh1DConnectivities",
    "Mesh1DCoords",
    "Mesh1DNames",
    "Mesh2DConnectivities",
    "Mesh2DCoords",
    "Mesh2DNames",
    "MeshEdgeCoords",
    "MeshFaceCoords",
    "MeshNodeCoords",
    "MeshMetadata",
    "MeshCoordMetadata",
]


#: Numpy "threshold" printoptions default argument.
NP_PRINTOPTIONS_THRESHOLD = 10
#: Numpy "edgeitems" printoptions default argument.
NP_PRINTOPTIONS_EDGEITEMS = 2


# Configure the logger.
logger = get_logger(__name__, fmt="[%(cls)s.%(funcName)s]")

#
# Mesh dimension names namedtuples.
#

#: Namedtuple for 1D mesh topology NetCDF variable dimension names.
Mesh1DNames = namedtuple("Mesh1DNames", ["node_dimension", "edge_dimension"])
#: Namedtuple for 2D mesh topology NetCDF variable dimension names.
Mesh2DNames = namedtuple(
    "Mesh2DNames", ["node_dimension", "edge_dimension", "face_dimension"]
)

#
# Mesh coordinate manager namedtuples.
#

#: Namedtuple for 1D mesh :class:`~iris.coords.AuxCoord` coordinates.
Mesh1DCoords = namedtuple(
    "Mesh1DCoords", ["node_x", "node_y", "edge_x", "edge_y"]
)
#: Namedtuple for 2D mesh :class:`~iris.coords.AuxCoord` coordinates.
Mesh2DCoords = namedtuple(
    "Mesh2DCoords",
    ["node_x", "node_y", "edge_x", "edge_y", "face_x", "face_y"],
)
#: Namedtuple for ``node`` :class:`~iris.coords.AuxCoord` coordinates.
MeshNodeCoords = namedtuple("MeshNodeCoords", ["node_x", "node_y"])
#: Namedtuple for ``edge`` :class:`~iris.coords.AuxCoord` coordinates.
MeshEdgeCoords = namedtuple("MeshEdgeCoords", ["edge_x", "edge_y"])
#: Namedtuple for ``face`` :class:`~iris.coords.AuxCoord` coordinates.
MeshFaceCoords = namedtuple("MeshFaceCoords", ["face_x", "face_y"])

#
# Mesh connectivity manager namedtuples.
#

#: Namedtuple for 1D mesh :class:`Connectivity` coordinates.
Mesh1DConnectivities = namedtuple("Mesh1DConnectivities", ["edge_node"])
#: Namedtuple for 2D mesh :class:`Connectivity` coordinates.
Mesh2DConnectivities = namedtuple(
    "Mesh2DConnectivities",
    [
        "face_node",
        "edge_node",
        "face_edge",
        "face_face",
        "edge_face",
        "boundary_node",
    ],
)


class Connectivity(_DimensionalMetadata):
    """
    A CF-UGRID topology connectivity, describing the topological relationship
    between two lists of dimensional locations. One or more connectivities
    make up a CF-UGRID topology - a constituent of a CF-UGRID mesh.

    See: https://ugrid-conventions.github.io/ugrid-conventions

    """

    UGRID_CF_ROLES = [
        "edge_node_connectivity",
        "face_node_connectivity",
        "face_edge_connectivity",
        "face_face_connectivity",
        "edge_face_connectivity",
        "boundary_node_connectivity",
        "volume_node_connectivity",
        "volume_edge_connectivity",
        "volume_face_connectivity",
        "volume_volume_connectivity",
    ]

    def __init__(
        self,
        indices,
        cf_role,
        standard_name=None,
        long_name=None,
        var_name=None,
        units=None,
        attributes=None,
        start_index=0,
        src_dim=0,
    ):
        """
        Constructs a single connectivity.

        Args:

        * indices (numpy.ndarray or numpy.ma.core.MaskedArray or dask.array.Array):
            The index values describing a topological relationship. Constructed
            of 2 dimensions - the list of locations, and within each location:
            the indices of the 'target locations' it relates to.
            Use a :class:`numpy.ma.core.MaskedArray` if :attr:`src_location`
            lengths vary - mask unused index 'slots' within each
            :attr:`src_location`. Use a :class:`dask.array.Array` to keep
            indices 'lazy'.
        * cf_role (str):
            Denotes the topological relationship that this connectivity
            describes. Made up of this array's locations, and the indexed
            'target location' within each location.
            See :attr:`UGRID_CF_ROLES` for valid arguments.

        Kwargs:

        * standard_name (str):
            CF standard name of the connectivity.
            (NOTE: this is not expected by the UGRID conventions, but will be
            handled in Iris' standard way if provided).
        * long_name (str):
            Descriptive name of the connectivity.
        * var_name (str):
            The NetCDF variable name for the connectivity.
        * units (cf_units.Unit):
            The :class:`~cf_units.Unit` of the connectivity's values.
            Can be a string, which will be converted to a Unit object.
            (NOTE: this is not expected by the UGRID conventions, but will be
            handled in Iris' standard way if provided).
        * attributes (dict):
            A dictionary containing other cf and user-defined attributes.
        * start_index (int):
            Either ``0`` or ``1``. Default is ``0``. Denotes whether
            :attr:`indices` uses 0-based or 1-based indexing (allows support
            for Fortran and legacy NetCDF files).
        * src_dim (int):
            Either ``0`` or ``1``. Default is ``0``. Denotes which dimension
            of :attr:`indices` varies over the :attr:`src_location`'s (the
            alternate dimension therefore varying within individual
            :attr:`src_location`'s). (This parameter allows support for fastest varying index being
            either first or last).
            E.g. for ``face_node_connectivity``, for 10 faces:
            ``indices.shape[src_dim] = 10``.

        """

        def validate_arg_vs_list(arg_name, arg, valid_list):
            if arg not in valid_list:
                error_msg = (
                    f"Invalid {arg_name} . Got: {arg} . Must be one of: "
                    f"{valid_list} ."
                )
                raise ValueError(error_msg)

        # Configure the metadata manager.
        self._metadata_manager = metadata_manager_factory(ConnectivityMetadata)

        validate_arg_vs_list("start_index", start_index, [0, 1])
        # indices array will be 2-dimensional, so must be either 0 or 1.
        validate_arg_vs_list("src_dim", src_dim, [0, 1])
        validate_arg_vs_list("cf_role", cf_role, Connectivity.UGRID_CF_ROLES)

        self._metadata_manager.start_index = start_index
        self._metadata_manager.src_dim = src_dim
        self._metadata_manager.cf_role = cf_role

        self._tgt_dim = 1 - src_dim
        self._src_location, self._tgt_location = cf_role.split("_")[:2]

        super().__init__(
            values=indices,
            standard_name=standard_name,
            long_name=long_name,
            var_name=var_name,
            units=units,
            attributes=attributes,
        )

    def __repr__(self):
        def kwargs_filter(k, v):
            result = False
            if k != "cf_role":
                if v is not None:
                    result = True
                    if (
                        not isinstance(v, str)
                        and isinstance(v, Iterable)
                        and not v
                    ):
                        result = False
                    elif k == "units" and v == "unknown":
                        result = False
            return result

        def array2repr(array):
            if self.has_lazy_indices():
                result = repr(array)
            else:
                with np.printoptions(
                    threshold=NP_PRINTOPTIONS_THRESHOLD,
                    edgeitems=NP_PRINTOPTIONS_EDGEITEMS,
                ):
                    result = re.sub("\n  *", " ", repr(array))
            return result

        # positional arguments
        args = ", ".join(
            [
                f"{array2repr(self.core_indices())}",
                f"cf_role={self.cf_role!r}",
            ]
        )

        # optional arguments (metadata)
        kwargs = ", ".join(
            [
                f"{k}={v!r}"
                for k, v in self.metadata._asdict().items()
                if kwargs_filter(k, v)
            ]
        )

        return f"{self.__class__.__name__}({', '.join([args, kwargs])})"

    def __str__(self):
        args = ", ".join(
            [f"cf_role={self.cf_role!r}", f"start_index={self.start_index!r}"]
        )
        return f"{self.__class__.__name__}({args})"

    @property
    def _values(self):
        # Overridden just to allow .setter override.
        return super()._values

    @_values.setter
    def _values(self, values):
        self._validate_indices(values, shapes_only=True)
        # The recommended way of using the setter in super().
        super(Connectivity, self.__class__)._values.fset(self, values)

    @property
    def cf_role(self):
        """
        The category of topological relationship that this connectivity
        describes.
        **Read-only** - validity of :attr:`indices` is dependent on
        :attr:`cf_role`. A new :class:`Connectivity` must therefore be defined
        if a different :attr:`cf_role` is needed.

        """
        return self._metadata_manager.cf_role

    @property
    def src_location(self):
        """
        Derived from the connectivity's :attr:`cf_role` - the first part, e.g.
        ``face`` in ``face_node_connectivity``. Refers to the locations
        listed by the :attr:`src_dim` of the connectivity's :attr:`indices`
        array.

        """
        return self._src_location

    @property
    def tgt_location(self):
        """
        Derived from the connectivity's :attr:`cf_role` - the second part, e.g.
        ``node`` in ``face_node_connectivity``. Refers to the locations indexed
        by the values in the connectivity's :attr:`indices` array.

        """
        return self._tgt_location

    @property
    def start_index(self):
        """
        The base value of the connectivity's :attr:`indices` array; either
        ``0`` or ``1``.
        **Read-only** - validity of :attr:`indices` is dependent on
        :attr:`start_index`. A new :class:`Connectivity` must therefore be
        defined if a different :attr:`start_index` is needed.

        """
        return self._metadata_manager.start_index

    @property
    def src_dim(self):
        """
        The dimension of the connectivity's :attr:`indices` array that varies
        over the connectivity's :attr:`src_location`'s. Either ``0`` or ``1``.
        **Read-only** - validity of :attr:`indices` is dependent on
        :attr:`src_dim`. Use :meth:`transpose` to create a new, transposed
        :class:`Connectivity` if a different :attr:`src_dim` is needed.

        """
        return self._metadata_manager.src_dim

    @property
    def tgt_dim(self):
        """
        Derived as the alternate value of :attr:`src_dim` - each must equal
        either ``0`` or ``1``.
        The dimension of the connectivity's :attr:`indices` array that varies
        within the connectivity's individual :attr:`src_location`'s.

        """
        return self._tgt_dim

    @property
    def indices(self):
        """
        The index values describing the topological relationship of the
        connectivity, as a NumPy array. Masked points indicate a
        :attr:`src_location` shorter than the longest :attr:`src_location`
        described in this array - unused index 'slots' are masked.
        **Read-only** - index values are only meaningful when combined with
        an appropriate :attr:`cf_role`, :attr:`start_index` and
        :attr:`src_dim`. A new :class:`Connectivity` must therefore be
        defined if different indices are needed.

        """
        return self._values

    def indices_by_src(self, indices=None):
        """
        Return a view of the indices array with :attr:`src_dim` **always** as
        the first index - transposed if necessary. Can optionally pass in an
        identically shaped array on which to perform this operation (e.g. the
        output from :meth:`core_indices` or :meth:`lazy_indices`).

        Kwargs:

        * indices (array):
            The array on which to operate. If ``None``, will operate on
            :attr:`indices`. Default is ``None``.

        Returns:
            A view of the indices array, transposed - if necessary - to put
            :attr:`src_dim` first.

        """
        if indices is None:
            indices = self.indices

        if indices.shape != self.shape:
            raise ValueError(
                f"Invalid indices provided. Must be shape={self.shape} , "
                f"got shape={indices.shape} ."
            )

        if self.src_dim == 0:
            result = indices
        elif self.src_dim == 1:
            result = indices.transpose()
        else:
            raise ValueError("Invalid src_dim.")

        return result

    def _validate_indices(self, indices, shapes_only=False):
        # Use shapes_only=True for a lower resource, less thorough validation
        # of indices by just inspecting the array shape instead of inspecting
        # individual masks. So will not catch individual src_locations being
        # unacceptably small.

        def indices_error(message):
            raise ValueError("Invalid indices provided. " + message)

        indices = self._sanitise_array(indices, 0)

        indices_dtype = indices.dtype
        if not np.issubdtype(indices_dtype, np.integer):
            indices_error(
                f"dtype must be numpy integer subtype, got: {indices_dtype} ."
            )

        indices_min = indices.min()
        if _lazy.is_lazy_data(indices_min):
            indices_min = indices_min.compute()
        if indices_min < self.start_index:
            indices_error(
                f"Lowest index: {indices_min} < start_index: {self.start_index} ."
            )

        indices_shape = indices.shape
        if len(indices_shape) != 2:
            indices_error(
                f"Expected 2-dimensional shape, got: shape={indices_shape} ."
            )

        len_req_fail = False
        if shapes_only:
            src_shape = indices_shape[self.tgt_dim]
            # Wrap as lazy to allow use of the same operations below
            # regardless of shapes_only.
            src_lengths = _lazy.as_lazy_data(np.asarray(src_shape))
        else:
            # Wouldn't be safe to use during __init__ validation, since
            # lazy_src_lengths requires self.indices to exist. Safe here since
            # shapes_only==False is only called manually, i.e. after
            # initialisation.
            src_lengths = self.lazy_src_lengths()
        if self.src_location in ("edge", "boundary"):
            if (src_lengths != 2).any().compute():
                len_req_fail = "len=2"
        else:
            if self.src_location == "face":
                min_size = 3
            elif self.src_location == "volume":
                if self.tgt_location == "edge":
                    min_size = 6
                else:
                    min_size = 4
            else:
                raise NotImplementedError
            if (src_lengths < min_size).any().compute():
                len_req_fail = f"len>={min_size}"
        if len_req_fail:
            indices_error(
                f"Not all src_locations meet requirement: {len_req_fail} - "
                f"needed to describe '{self.cf_role}' ."
            )

    def validate_indices(self):
        """
        Perform a thorough validity check of this connectivity's
        :attr:`indices`. Includes checking the sizes of individual
        :attr:`src_location`'s (specified using masks on the
        :attr:`indices` array) against the :attr:`cf_role`.

        Raises a ``ValueError`` if any problems are encountered, otherwise
        passes silently.

        .. note::

            While this uses lazy computation, it will still be a high
            resource demand for a large :attr:`indices` array.

        """
        self._validate_indices(self.indices, shapes_only=False)

    def __eq__(self, other):
        eq = NotImplemented
        if isinstance(other, Connectivity):
            # Account for the fact that other could be the transposed equivalent
            # of self, which we consider 'safe' since the recommended
            # interaction with the indices array is via indices_by_src, which
            # corrects for this difference. (To enable this, src_dim does
            # not participate in ConnectivityMetadata to ConnectivityMetadata
            # equivalence).
            if hasattr(other, "metadata"):
                # metadata comparison
                eq = self.metadata == other.metadata
                if eq:
                    eq = (
                        self.indices_by_src() == other.indices_by_src()
                    ).all()
        return eq

    def transpose(self):
        """
        Create a new :class:`Connectivity`, identical to this one but with the
        :attr:`indices` array transposed and the :attr:`src_dim` value flipped.

        Returns:
            A new :class:`Connectivity` that is the transposed equivalent of
            the original.

        """
        new_connectivity = Connectivity(
            indices=self.indices.transpose().copy(),
            cf_role=self.cf_role,
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
            units=self.units,
            attributes=self.attributes,
            start_index=self.start_index,
            src_dim=self.tgt_dim,
        )
        return new_connectivity

    def lazy_indices(self):
        """
        Return a lazy array representing the connectivity's indices.

        Accessing this method will never cause the :attr:`indices` values to be
        loaded. Similarly, calling methods on, or indexing, the returned Array
        will not cause the connectivity to have loaded :attr:`indices`.

        If the :attr:`indices` have already been loaded for the connectivity,
        the returned Array will be a new lazy array wrapper.

        Returns:
            A lazy array, representing the connectivity indices array.

        """
        return super()._lazy_values()

    def core_indices(self):
        """
        The indices array at the core of this connectivity, which may be a
        NumPy array or a Dask array.

        Returns:
            numpy.ndarray or numpy.ma.core.MaskedArray or dask.array.Array

        """
        return super()._core_values()

    def has_lazy_indices(self):
        """
        Return a boolean indicating whether the connectivity's :attr:`indices`
        array is a lazy Dask array or not.

        Returns:
            boolean

        """
        return super()._has_lazy_values()

    def lazy_src_lengths(self):
        """
        Return a lazy array representing the lengths of each
        :attr:`src_location` in the :attr:`src_dim` of the connectivity's
        :attr:`indices` array, accounting for masks if present.

        Accessing this method will never cause the :attr:`indices` values to be
        loaded. Similarly, calling methods on, or indexing, the returned Array
        will not cause the connectivity to have loaded :attr:`indices`.

        The returned Array will be lazy regardless of whether the
        :attr:`indices` have already been loaded.

        Returns:
            A lazy array, representing the lengths of each :attr:`src_location`.

        """
        src_mask_counts = da.sum(
            da.ma.getmaskarray(self.indices), axis=self.tgt_dim
        )
        max_src_size = self.indices.shape[self.tgt_dim]
        return max_src_size - src_mask_counts

    def src_lengths(self):
        """
        Return a NumPy array representing the lengths of each
        :attr:`src_location` in the :attr:`src_dim` of the connectivity's
        :attr:`indices` array, accounting for masks if present.

        Returns:
            A NumPy array, representing the lengths of each :attr:`src_location`.

        """
        return self.lazy_src_lengths().compute()

    def cube_dims(self, cube):
        """Not available on :class:`Connectivity`."""
        raise NotImplementedError

    def xml_element(self, doc):
        # Create the XML element as the camelCaseEquivalent of the
        # class name
        element = super().xml_element(doc)

        element.setAttribute("cf_role", self.cf_role)
        element.setAttribute("start_index", self.start_index)
        element.setAttribute("src_dim", self.src_dim)

        return element


class ConnectivityMetadata(BaseMetadata):
    """
    Metadata container for a :class:`~iris.experimental.ugrid.Connectivity`.

    """

    # The "src_dim" member is stateful only, and does not participate in
    # lenient/strict equivalence.
    _members = ("cf_role", "start_index", "src_dim")

    __slots__ = ()

    @wraps(BaseMetadata.__eq__, assigned=("__doc__",), updated=())
    @lenient_service
    def __eq__(self, other):
        return super().__eq__(other)

    def _combine_lenient(self, other):
        """
        Perform lenient combination of metadata members for connectivities.

        Args:

        * other (ConnectivityMetadata):
            The other connectivity metadata participating in the lenient
            combination.

        Returns:
            A list of combined metadata member values.

        """
        # Perform "strict" combination for "cf_role", "start_index", "src_dim".
        def func(field):
            left = getattr(self, field)
            right = getattr(other, field)
            return left if left == right else None

        # Note that, we use "_members" not "_fields".
        values = [func(field) for field in ConnectivityMetadata._members]
        # Perform lenient combination of the other parent members.
        result = super()._combine_lenient(other)
        result.extend(values)

        return result

    def _compare_lenient(self, other):
        """
        Perform lenient equality of metadata members for connectivities.

        Args:

        * other (ConnectivityMetadata):
            The other connectivity metadata participating in the lenient
            comparison.

        Returns:
            Boolean.

        """
        # Perform "strict" comparison for "cf_role", "start_index".
        # The "src_dim" member is not part of lenient equivalence.
        members = filter(
            lambda member: member != "src_dim", ConnectivityMetadata._members
        )
        result = all(
            [
                getattr(self, field) == getattr(other, field)
                for field in members
            ]
        )
        if result:
            # Perform lenient comparison of the other parent members.
            result = super()._compare_lenient(other)

        return result

    def _difference_lenient(self, other):
        """
        Perform lenient difference of metadata members for connectivities.

        Args:

        * other (ConnectivityMetadata):
            The other connectivity metadata participating in the lenient
            difference.

        Returns:
            A list of difference metadata member values.

        """
        # Perform "strict" difference for "cf_role", "start_index", "src_dim".
        def func(field):
            left = getattr(self, field)
            right = getattr(other, field)
            return None if left == right else (left, right)

        # Note that, we use "_members" not "_fields".
        values = [func(field) for field in ConnectivityMetadata._members]
        # Perform lenient difference of the other parent members.
        result = super()._difference_lenient(other)
        result.extend(values)

        return result

    @wraps(BaseMetadata.combine, assigned=("__doc__",), updated=())
    @lenient_service
    def combine(self, other, lenient=None):
        return super().combine(other, lenient=lenient)

    @wraps(BaseMetadata.difference, assigned=("__doc__",), updated=())
    @lenient_service
    def difference(self, other, lenient=None):
        return super().difference(other, lenient=lenient)

    @wraps(BaseMetadata.equal, assigned=("__doc__",), updated=())
    @lenient_service
    def equal(self, other, lenient=None):
        return super().equal(other, lenient=lenient)


class MeshMetadata(BaseMetadata):
    """
    Metadata container for a :class:`~iris.experimental.ugrid.Mesh`.

    """

    # The node_dimension", "edge_dimension" and "face_dimension" members are
    # stateful only; they not participate in lenient/strict equivalence.
    _members = (
        "topology_dimension",
        "node_dimension",
        "edge_dimension",
        "face_dimension",
    )

    __slots__ = ()

    @wraps(BaseMetadata.__eq__, assigned=("__doc__",), updated=())
    @lenient_service
    def __eq__(self, other):
        return super().__eq__(other)

    def _combine_lenient(self, other):
        """
        Perform lenient combination of metadata members for meshes.

        Args:

        * other (MeshMetadata):
            The other mesh metadata participating in the lenient
            combination.

        Returns:
            A list of combined metadata member values.

        """

        # Perform "strict" combination for "topology_dimension",
        # "node_dimension", "edge_dimension" and "face_dimension".
        def func(field):
            left = getattr(self, field)
            right = getattr(other, field)
            return left if left == right else None

        # Note that, we use "_members" not "_fields".
        values = [func(field) for field in MeshMetadata._members]
        # Perform lenient combination of the other parent members.
        result = super()._combine_lenient(other)
        result.extend(values)

        return result

    def _compare_lenient(self, other):
        """
        Perform lenient equality of metadata members for meshes.

        Args:

        * other (MeshMetadata):
            The other mesh metadata participating in the lenient
            comparison.

        Returns:
            Boolean.

        """
        # Perform "strict" comparison for "topology_dimension".
        # "node_dimension", "edge_dimension" and "face_dimension" are not part
        # of lenient equivalence at all.
        result = self.topology_dimension == other.topology_dimension
        if result:
            # Perform lenient comparison of the other parent members.
            result = super()._compare_lenient(other)

        return result

    def _difference_lenient(self, other):
        """
        Perform lenient difference of metadata members for meshes.

        Args:

        * other (MeshMetadata):
            The other mesh metadata participating in the lenient
            difference.

        Returns:
            A list of difference metadata member values.

        """
        # Perform "strict" difference for "topology_dimension",
        # "node_dimension", "edge_dimension" and "face_dimension".
        def func(field):
            left = getattr(self, field)
            right = getattr(other, field)
            return None if left == right else (left, right)

        # Note that, we use "_members" not "_fields".
        values = [func(field) for field in MeshMetadata._members]
        # Perform lenient difference of the other parent members.
        result = super()._difference_lenient(other)
        result.extend(values)

        return result

    @wraps(BaseMetadata.combine, assigned=("__doc__",), updated=())
    @lenient_service
    def combine(self, other, lenient=None):
        return super().combine(other, lenient=lenient)

    @wraps(BaseMetadata.difference, assigned=("__doc__",), updated=())
    @lenient_service
    def difference(self, other, lenient=None):
        return super().difference(other, lenient=lenient)

    @wraps(BaseMetadata.equal, assigned=("__doc__",), updated=())
    @lenient_service
    def equal(self, other, lenient=None):
        return super().equal(other, lenient=lenient)


class Mesh(CFVariableMixin):
    """
    A container representing the UGRID ``cf_role`` ``mesh_topology``, supporting
    1D network, 2D triangular, and 2D flexible mesh topologies.

    .. note::

        The 3D layered and fully 3D unstructured mesh topologies are not supported
        at this time.

    .. seealso::

        The UGRID Conventions, https://ugrid-conventions.github.io/ugrid-conventions/

    """

    # TBD: for volume and/or z-axis support include axis "z" and/or dimension "3"
    #: The supported mesh axes.
    AXES = ("x", "y")
    #: Valid range of values for ``topology_dimension``.
    TOPOLOGY_DIMENSIONS = (1, 2)

    def __init__(
        self,
        topology_dimension,
        node_coords_and_axes,
        connectivities,
        edge_coords_and_axes=None,
        face_coords_and_axes=None,
        standard_name=None,
        long_name=None,
        var_name=None,
        units=None,
        attributes=None,
        node_dimension=None,
        edge_dimension=None,
        face_dimension=None,
    ):
        """
        .. note::

            The purpose of the :attr:`node_dimension`, :attr:`edge_dimension` and
            :attr:`face_dimension` properties are to preserve the original NetCDF
            variable dimension names. Note that, only :attr:`edge_dimension` and
            :attr:`face_dimension` are UGRID attributes, and are only present for
            :attr:`topology_dimension` ``>=2``.

        """
        # TODO: support volumes.
        # TODO: support (coord, "z")

        self._metadata_manager = metadata_manager_factory(MeshMetadata)

        # topology_dimension is read-only, so assign directly to the metadata manager
        if topology_dimension not in self.TOPOLOGY_DIMENSIONS:
            emsg = f"Expected 'topology_dimension' in range {self.TOPOLOGY_DIMENSIONS!r}, got {topology_dimension!r}."
            raise ValueError(emsg)
        self._metadata_manager.topology_dimension = topology_dimension

        self.node_dimension = node_dimension
        self.edge_dimension = edge_dimension
        self.face_dimension = face_dimension

        # assign the metadata to the metadata manager
        self.standard_name = standard_name
        self.long_name = long_name
        self.var_name = var_name
        self.units = units
        self.attributes = attributes

        # based on the topology_dimension, create the appropriate coordinate manager
        def normalise(location, axis):
            result = str(axis).lower()
            if result not in self.AXES:
                emsg = f"Invalid axis specified for {location} coordinate {coord.name()!r}, got {axis!r}."
                raise ValueError(emsg)
            return f"{location}_{axis}"

        if not isinstance(node_coords_and_axes, Iterable):
            node_coords_and_axes = [node_coords_and_axes]

        if not isinstance(connectivities, Iterable):
            connectivities = [connectivities]

        kwargs = {}
        for coord, axis in node_coords_and_axes:
            kwargs[normalise("node", axis)] = coord
        if edge_coords_and_axes is not None:
            for coord, axis in edge_coords_and_axes:
                kwargs[normalise("edge", axis)] = coord
        if face_coords_and_axes is not None:
            for coord, axis in face_coords_and_axes:
                kwargs[normalise("face", axis)] = coord

        # check the UGRID minimum requirement for coordinates
        if "node_x" not in kwargs:
            emsg = (
                "Require a node coordinate that is x-axis like to be provided."
            )
            raise ValueError(emsg)
        if "node_y" not in kwargs:
            emsg = (
                "Require a node coordinate that is y-axis like to be provided."
            )
            raise ValueError(emsg)

        if self.topology_dimension == 1:
            self._coord_manager = _Mesh1DCoordinateManager(**kwargs)
            self._connectivity_manager = _Mesh1DConnectivityManager(
                *connectivities
            )
        elif self.topology_dimension == 2:
            self._coord_manager = _Mesh2DCoordinateManager(**kwargs)
            self._connectivity_manager = _Mesh2DConnectivityManager(
                *connectivities
            )
        else:
            emsg = f"Unsupported 'topology_dimension', got {topology_dimension!r}."
            raise NotImplementedError(emsg)

    def __eq__(self, other):
        # TBD: this is a minimalist implementation and requires to be revisited
        return id(self) == id(other)

    def __getstate__(self):
        return (
            self._metadata_manager,
            self._coord_manager,
            self._connectivity_manager,
        )

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is not NotImplemented:
            result = not result
        return result

    def __repr__(self):
        def to_coord_and_axis(members):
            def axis(member):
                return member.split("_")[1]

            result = [
                f"({coord!s}, {axis(member)!r})"
                for member, coord in members._asdict().items()
                if coord is not None
            ]
            result = f"[{', '.join(result)}]" if result else None
            return result

        node_coords_and_axes = to_coord_and_axis(self.node_coords)
        connectivities = [
            str(connectivity)
            for connectivity in self.all_connectivities
            if connectivity is not None
        ]

        if len(connectivities) == 1:
            connectivities = connectivities[0]
        else:
            connectivities = f"[{', '.join(connectivities)}]"

        # positional arguments
        args = [
            f"topology_dimension={self.topology_dimension!r}",
            f"node_coords_and_axes={node_coords_and_axes}",
            f"connectivities={connectivities}",
        ]

        # optional argument
        edge_coords_and_axes = to_coord_and_axis(self.edge_coords)
        if edge_coords_and_axes:
            args.append(f"edge_coords_and_axes={edge_coords_and_axes}")

        # optional argument
        if self.topology_dimension > 1:
            face_coords_and_axes = to_coord_and_axis(self.face_coords)
            if face_coords_and_axes:
                args.append(f"face_coords_and_axes={face_coords_and_axes}")

        def kwargs_filter(k, v):
            result = False
            if k != "topology_dimension":
                if not (
                    self.topology_dimension == 1 and k == "face_dimension"
                ):
                    if v is not None:
                        result = True
                        if (
                            not isinstance(v, str)
                            and isinstance(v, Iterable)
                            and not v
                        ):
                            result = False
                        elif k == "units" and v == "unknown":
                            result = False
            return result

        # optional arguments (metadata)
        args.extend(
            [
                f"{k}={v!r}"
                for k, v in self.metadata._asdict().items()
                if kwargs_filter(k, v)
            ]
        )

        return f"{self.__class__.__name__}({', '.join(args)})"

    def __setstate__(self, state):
        metadata_manager, coord_manager, connectivity_manager = state
        self._metadata_manager = metadata_manager
        self._coord_manager = coord_manager
        self._connectivity_manager = connectivity_manager

    def _set_dimension_names(self, node, edge, face, reset=False):
        args = (node, edge, face)
        currents = (
            self.node_dimension,
            self.edge_dimension,
            self.face_dimension,
        )
        zipped = zip(args, currents)
        if reset:
            node, edge, face = [
                None if arg else current for arg, current in zipped
            ]
        else:
            node, edge, face = [arg or current for arg, current in zipped]

        self.node_dimension = node
        self.edge_dimension = edge
        self.face_dimension = face

        if self.topology_dimension == 1:
            result = Mesh1DNames(self.node_dimension, self.edge_dimension)
        elif self.topology_dimension == 2:
            result = Mesh2DNames(
                self.node_dimension, self.edge_dimension, self.face_dimension
            )
        else:
            message = (
                f"Unsupported topology_dimension: {self.topology_dimension} ."
            )
            raise NotImplementedError(message)

        return result

    @property
    def all_connectivities(self):
        """
        All the :class:`Connectivity` coordinates of the :class:`Mesh`.

        """
        return self._connectivity_manager.all_members

    @property
    def all_coords(self):
        """
        All the :class:`~iris.coords.AuxCoord` coordinates of the :class:`Mesh`.

        """
        return self._coord_manager.all_members

    @property
    def boundary_node_connectivity(self):
        """
        The *optional* UGRID ``boundary_node_connectivity`` :class:`Connectivity`
        coordinate of the :class:`Mesh`.

        """
        return self._connectivity_manager.boundary_node

    @property
    def edge_coords(self):
        """
        The *optional* UGRID ``edge`` :class:`~iris.coords.AuxCoord` coordinates
        of the :class:`Mesh`.

        """
        return self._coord_manager.edge_coords

    @property
    def edge_dimension(self):
        """
        The *optionally required* UGRID NetCDF variable name for the ``edge``
        dimension.

        """
        return self._metadata_manager.edge_dimension

    @edge_dimension.setter
    def edge_dimension(self, name):
        if not name or not isinstance(name, str):
            edge_dimension = f"Mesh{self.topology_dimension}d_edge"
        else:
            edge_dimension = name
        self._metadata_manager.edge_dimension = edge_dimension

    @property
    def edge_face_connectivity(self):
        """
        The *optional* UGRID ``edge_face_connectivity`` :class:`Connectivity`
        coordinate of the :class:`Mesh`.

        """
        return self._connectivity_manager.edge_face

    @property
    def edge_node_connectivity(self):
        """
        The UGRID ``edge_node_connectivity`` :class:`Connectivity` coordinate
        of the :class:`Mesh`, which is **required** for
        :attr:`Mesh.topology_dimension` of ``1``, and *optionally required* for
        :attr:`Mesh.topology_dimension` ``>=2``.

        """
        return self._connectivity_manager.edge_node

    @property
    def face_coords(self):
        """
        The *optional* UGRID ``face`` :class:`~iris.coords.AuxCoord` coordinates
        of the :class:`Mesh`.

        """
        return self._coord_manager.face_coords

    @property
    def face_dimension(self):
        """
        The *optionally required* UGRID NetCDF variable name for the ``face``
        dimension.

        """
        return self._metadata_manager.face_dimension

    @face_dimension.setter
    def face_dimension(self, name):
        if self.topology_dimension < 2:
            face_dimension = None
            if name:
                # Tell the user it is not being set if they expected otherwise.
                message = (
                    "Not setting face_dimension (inappropriate for "
                    f"topology_dimension={self.topology_dimension} ."
                )
                logger.debug(message)
        elif not name or not isinstance(name, str):
            face_dimension = f"Mesh{self.topology_dimension}d_face"
        else:
            face_dimension = name
        self._metadata_manager.face_dimension = face_dimension

    @property
    def face_edge_connectivity(self):
        """
        The *optional* UGRID ``face_edge_connectivity`` :class:`Connectivity`
        coordinate of the :class:`Mesh`.

        """
        # optional
        return self._connectivity_manager.face_edge

    @property
    def face_face_connectivity(self):
        """
        The *optional* UGRID ``face_face_connectivity`` :class:`Connectivity`
        coordinate of the :class:`Mesh`.

        """
        return self._connectivity_manager.face_face

    @property
    def face_node_connectivity(self):
        """
        The UGRID ``face_node_connectivity`` :class:`Connectivity` coordinate
        of the :class:`Mesh`, which is **required** for
        :attr:`Mesh.topology_dimension` of ``2``, and *optionally required* for
        :attr:`Mesh.topology_dimension` of ``3``.

        """
        return self._connectivity_manager.face_node

    @property
    def node_coords(self):
        """
        The **required** UGRID ``node`` :class:`~iris.coords.AuxCoord` coordinates
        of the :class:`Mesh`.

        """
        return self._coord_manager.node_coords

    @property
    def node_dimension(self):
        """The NetCDF variable name for the ``node`` dimension."""
        return self._metadata_manager.node_dimension

    @node_dimension.setter
    def node_dimension(self, name):
        if not name or not isinstance(name, str):
            node_dimension = f"Mesh{self.topology_dimension}d_node"
        else:
            node_dimension = name
        self._metadata_manager.node_dimension = node_dimension

    def add_connectivities(self, *connectivities):
        """
        Add one or more :class:`Connectivity` coordinates to the :class:`Mesh`.

        Args:

        * connectivities (iterable of object):
            A collection of one or more :class:`Connectivity` coordinates to
            add to the :class:`Mesh`.

        """
        self._connectivity_manager.add(*connectivities)

    def add_coords(
        self,
        node_x=None,
        node_y=None,
        edge_x=None,
        edge_y=None,
        face_x=None,
        face_y=None,
    ):
        """
        Add one or more :class:`~iris.coords.AuxCoord` coordinates to the :class:`Mesh`.

        Kwargs:

        * node_x (object):
            The ``x-axis`` like ``node`` :class:`~iris.coords.AuxCoord`.

        * node_y (object):
            The ``y-axis`` like ``node`` :class:`~iris.coords.AuxCoord`.

        * edge_x (object):
            The ``x-axis`` like ``edge`` :class:`~iris.coords.AuxCoord`.

        * edge_y (object):
            The ``y-axis`` like ``edge`` :class:`~iris.coords.AuxCoord`.

        * face_x (object):
            The ``x-axis`` like ``face`` :class:`~iris.coords.AuxCoord`.

        * face_y (object):
            The ``y-axis`` like ``face`` :class:`~iris.coords.AuxCoord`.

        """
        # Filter out absent arguments - only expecting face coords sometimes,
        # same will be true of volumes in future.
        kwargs = {
            "node_x": node_x,
            "node_y": node_y,
            "edge_x": edge_x,
            "edge_y": edge_y,
            "face_x": face_x,
            "face_y": face_y,
        }
        kwargs = {k: v for k, v in kwargs.items() if v}

        self._coord_manager.add(**kwargs)

    def connectivities(
        self,
        item=None,
        standard_name=None,
        long_name=None,
        var_name=None,
        attributes=None,
        cf_role=None,
        contains_node=None,
        contains_edge=None,
        contains_face=None,
    ):
        """
        Return all :class:`Connectivity` instances from the :class:`Mesh` that
        match the provided criteria.

        Criteria can be either specific properties or other objects with
        metadata to be matched.

        .. seealso::

            :meth:`Mesh.connectivity` for matching exactly one connectivity.

        Kwargs:

        * item (str or object):
            Either,

            * a :attr:`~iris.common.mixin.CFVariableMixin.standard_name`,
              :attr:`~iris.common.mixin.CFVariableMixin.long_name`, or
              :attr:`~iris.common.mixin.CFVariableMixin.var_name` which is
              compared against the :meth:`~iris.common.mixin.CFVariableMixin.name`.

            * a connectivity or metadata instance equal to that of
              the desired objectss e.g., :class:`Connectivity` or
              :class:`ConnectivityMetadata`.

        * standard_name (str):
            The CF standard name of the desired :class:`Connectibity`. If
            ``None``, does not check for ``standard_name``.

        * long_name (str):
            An unconstrained description of the :class:`Connectivity`. If
            ``None``, does not check for ``long_name``.

        * var_name (str):
            The NetCDF variable name of the desired :class:`Connectivity`. If
            ``None``, does not check for ``var_name``.

        * attributes (dict):
            A dictionary of attributes desired on the :class:`Connectivity`. If
            ``None``, does not check for ``attributes``.

        * cf_role (str):
            The UGRID ``cf_role`` of the desired :class:`Connectivity`.

        * contains_node (bool):
            Contains the ``node`` location as part of the
            :attr:`ConnectivityMetadata.cf_role` in the list of objects to be matched.

        * contains_edge (bool):
            Contains the ``edge`` location as part of the
            :attr:`ConnectivityMetadata.cf_role` in the list of objects to be matched.

        * contains_face (bool):
            Contains the ``face`` location as part of the
            :attr:`ConnectivityMetadata.cf_role` in the list of objects to be matched.

        Returns:
            A list of :class:`Connectivity` instances from the :class:`Mesh`
            that matched the given criteria.

        """
        return self._connectivity_manager.filters(
            item=item,
            standard_name=standard_name,
            long_name=long_name,
            var_name=var_name,
            attributes=attributes,
            cf_role=cf_role,
            contains_node=contains_node,
            contains_edge=contains_edge,
            contains_face=contains_face,
        )

    def connectivity(
        self,
        item=None,
        standard_name=None,
        long_name=None,
        var_name=None,
        attributes=None,
        cf_role=None,
        contains_node=None,
        contains_edge=None,
        contains_face=None,
    ):
        """
        Return a single :class:`Connectivity` from the :class:`Mesh` that
        matches the provided criteria.

        Criteria can be either specific properties or other objects with
        metadata to be matched.

        .. note::

            If the given criteria do not return **precisely one**
            :class:`Connectivity`, then a
            :class:`~iris.exceptions.ConnectivityNotFoundError` is raised.

        .. seealso::

            :meth:`Cube.connectivities` for matching zero or more connectivities.

        Kwargs:

        * item (str or object):
            Either,

            * a :attr:`~iris.common.mixin.CFVariableMixin.standard_name`,
              :attr:`~iris.common.mixin.CFVariableMixin.long_name`, or
              :attr:`~iris.common.mixin.CFVariableMixin.var_name` which is
              compared against the :meth:`~iris.common.mixin.CFVariableMixin.name`.

            * a connectivity or metadata instance equal to that of
              the desired object e.g., :class:`Connectivity` or
              :class:`ConnectivityMetadata`.

        * standard_name (str):
            The CF standard name of the desired :class:`Connectivity`. If
            ``None``, does not check for ``standard_name``.

        * long_name (str):
            An unconstrained description of the :class:`Connectivity`. If
            ``None``, does not check for ``long_name``.

        * var_name (str):
            The NetCDF variable name of the desired :class:`Connectivity`. If
            ``None``, does not check for ``var_name``.

        * attributes (dict):
            A dictionary of attributes desired on the :class:`Connectivity`. If
            ``None``, does not check for ``attributes``.

        * cf_role (str):
            The UGRID ``cf_role`` of the desired :class:`Connectivity`.

        * contains_node (bool):
            Contains the ``node`` location as part of the
            :attr:`ConnectivityMetadata.cf_role` in the list of objects to be matched.

        * contains_edge (bool):
            Contains the ``edge`` location as part of the
            :attr:`ConnectivityMetadata.cf_role` in the list of objects to be matched.

        * contains_face (bool):
            Contains the ``face`` location as part of the
            :attr:`ConnectivityMetadata.cf_role` in the list of objects to be matched.

        Returns:
            The :class:`Connectivity` from the :class:`Mesh` that matched the
            given criteria.

        """

        return self._connectivity_manager.filter(
            item=item,
            standard_name=standard_name,
            long_name=long_name,
            var_name=var_name,
            attributes=attributes,
            cf_role=cf_role,
            contains_node=contains_node,
            contains_edge=contains_edge,
            contains_face=contains_face,
        )

    def coord(
        self,
        item=None,
        standard_name=None,
        long_name=None,
        var_name=None,
        attributes=None,
        axis=None,
        include_nodes=None,
        include_edges=None,
        include_faces=None,
    ):
        """
        Return a single :class:`~iris.coords.AuxCoord` coordinate from the
        :class:`Mesh` that matches the provided criteria.

        Criteria can be either specific properties or other objects with
        metadata to be matched.

        .. note::

            If the given criteria do not return **precisely one** coordinate,
            then a :class:`~iris.exceptions.CoordinateNotFoundError` is raised.

        .. seealso::

            :meth:`Cube.coords` for matching zero or more coordinates.

        Kwargs:

        * item (str or object):
            Either,

            * a :attr:`~iris.common.mixin.CFVariableMixin.standard_name`,
              :attr:`~iris.common.mixin.CFVariableMixin.long_name`, or
              :attr:`~iris.common.mixin.CFVariableMixin.var_name` which is
              compared against the :meth:`~iris.common.mixin.CFVariableMixin.name`.

            * a coordinate or metadata instance equal to that of
              the desired coordinate e.g., :class:`~iris.coords.AuxCoord` or
              :class:`~iris.common.metadata.CoordMetadata`.

        * standard_name (str):
            The CF standard name of the desired coordinate. If ``None``, does not
            check for ``standard_name``.

        * long_name (str):
            An unconstrained description of the coordinate. If ``None``, does not
            check for ``long_name``.

        * var_name (str):
            The NetCDF variable name of the desired coordinate. If ``None``, does
            not check for ``var_name``.

        * attributes (dict):
            A dictionary of attributes desired on the coordinates. If ``None``,
            does not check for ``attributes``.

        * axis (str):
            The desired coordinate axis, see :func:`~iris.util.guess_coord_axis`.
            If ``None``, does not check for ``axis``. Accepts the values ``X``,
            ``Y``, ``Z`` and ``T`` (case-insensitive).

        * include_node (bool):
            Include all ``node`` coordinates in the list of objects to be matched.

        * include_edge (bool):
            Include all ``edge`` coordinates in the list of objects to be matched.

        * include_face (bool):
            Include all ``face`` coordinates in the list of objects to be matched.

        Returns:
            The :class:`~iris.coords.AuxCoord` coordinate from the :class:`Mesh`
            that matched the given criteria.

        """
        return self._coord_manager.filter(
            item=item,
            standard_name=standard_name,
            long_name=long_name,
            var_name=var_name,
            attributes=attributes,
            axis=axis,
            include_nodes=include_nodes,
            include_edges=include_edges,
            include_faces=include_faces,
        )

    def coords(
        self,
        item=None,
        standard_name=None,
        long_name=None,
        var_name=None,
        attributes=None,
        axis=None,
        include_nodes=None,
        include_edges=None,
        include_faces=None,
    ):
        """
        Return all :class:`~iris.coords.AuxCoord` coordinates from the :class:`Mesh` that
        match the provided criteria.

        Criteria can be either specific properties or other objects with
        metadata to be matched.

        .. seealso::

            :meth:`Mesh.coord` for matching exactly one coordinate.

        Kwargs:

        * item (str or object):
            Either,

            * a :attr:`~iris.common.mixin.CFVariableMixin.standard_name`,
              :attr:`~iris.common.mixin.CFVariableMixin.long_name`, or
              :attr:`~iris.common.mixin.CFVariableMixin.var_name` which is
              compared against the :meth:`~iris.common.mixin.CFVariableMixin.name`.

            * a coordinate or metadata instance equal to that of
              the desired coordinates e.g., :class:`~iris.coords.AuxCoord` or
              :class:`~iris.common.metadata.CoordMetadata`.

        * standard_name (str):
            The CF standard name of the desired coordinate. If ``None``, does not
            check for ``standard_name``.

        * long_name (str):
            An unconstrained description of the coordinate. If ``None``, does not
            check for ``long_name``.

        * var_name (str):
            The NetCDF variable name of the desired coordinate. If ``None``, does
            not check for ``var_name``.

        * attributes (dict):
            A dictionary of attributes desired on the coordinates. If ``None``,
            does not check for ``attributes``.

        * axis (str):
            The desired coordinate axis, see :func:`~iris.util.guess_coord_axis`.
            If ``None``, does not check for ``axis``. Accepts the values ``X``,
            ``Y``, ``Z`` and ``T`` (case-insensitive).

        * include_node (bool):
            Include all ``node`` coordinates in the list of objects to be matched.

        * include_edge (bool):
            Include all ``edge`` coordinates in the list of objects to be matched.

        * include_face (bool):
            Include all ``face`` coordinates in the list of objects to be matched.

        Returns:
            A list of :class:`~iris.coords.AuxCoord` coordinates from the
            :class:`Mesh` that matched the given criteria.

        """
        return self._coord_manager.filters(
            item=item,
            standard_name=standard_name,
            long_name=long_name,
            var_name=var_name,
            attributes=attributes,
            axis=axis,
            include_nodes=include_nodes,
            include_edges=include_edges,
            include_faces=include_faces,
        )

    def remove_connectivities(
        self,
        item=None,
        standard_name=None,
        long_name=None,
        var_name=None,
        attributes=None,
        cf_role=None,
        contains_node=None,
        contains_edge=None,
        contains_face=None,
    ):
        """
        Remove one or more :class:`Connectivity` from the :class:`Mesh` that
        match the provided criteria.

        Criteria can be either specific properties or other objects with
        metadata to be matched.

        Kwargs:

        * item (str or object):
            Either,

            * a :attr:`~iris.common.mixin.CFVariableMixin.standard_name`,
              :attr:`~iris.common.mixin.CFVariableMixin.long_name`, or
              :attr:`~iris.common.mixin.CFVariableMixin.var_name` which is
              compared against the :meth:`~iris.common.mixin.CFVariableMixin.name`.

            * a coordinate or metadata instance equal to that of
              the desired coordinates e.g., :class:`Connectivity` or
              :class:`ConnectivityMetadata`.

        * standard_name (str):
            The CF standard name of the desired coordinate. If ``None``, does not
            check for ``standard_name``.

        * long_name (str):
            An unconstrained description of the coordinate. If ``None``, does not
            check for ``long_name``.

        * var_name (str):
            The NetCDF variable name of the desired coordinate. If ``None``, does
            not check for ``var_name``.

        * attributes (dict):
            A dictionary of attributes desired on the coordinates. If ``None``,
            does not check for ``attributes``.

        * cf_role (str):
            The UGRID ``cf_role`` of the desired :class:`Connectivity`.

        * contains_node (bool):
            Contains the ``node`` location as part of the
            :attr:`ConnectivityMetadata.cf_role` in the list of objects to be matched
            for potential removal.

        * contains_edge (bool):
            Contains the ``edge`` location as part of the
            :attr:`ConnectivityMetadata.cf_role` in the list of objects to be matched
            for potential removal.

        * contains_face (bool):
            Contains the ``face`` location as part of the
            :attr:`ConnectivityMetadata.cf_role` in the list of objects to be matched
            for potential removal.

        Returns:
            A list of :class:`Connectivity` coordinates removed from the :class:`Mesh`
            that matched the given criteria.

        """
        return self._connectivity_manager.remove(
            item=item,
            standard_name=standard_name,
            long_name=long_name,
            var_name=var_name,
            attributes=attributes,
            cf_role=cf_role,
            contains_node=contains_node,
            contains_edge=contains_edge,
            contains_face=contains_face,
        )

    def remove_coords(
        self,
        item=None,
        standard_name=None,
        long_name=None,
        var_name=None,
        attributes=None,
        axis=None,
        include_nodes=None,
        include_edges=None,
        include_faces=None,
    ):
        """
        Remove one or more :class:`~iris.coords.AuxCoord` from the :class:`Mesh`
        that match the provided criteria.

        Criteria can be either specific properties or other objects with
        metadata to be matched.

        Kwargs:

        * item (str or object):
            Either,

            * a :attr:`~iris.common.mixin.CFVariableMixin.standard_name`,
              :attr:`~iris.common.mixin.CFVariableMixin.long_name`, or
              :attr:`~iris.common.mixin.CFVariableMixin.var_name` which is
              compared against the :meth:`~iris.common.mixin.CFVariableMixin.name`.

            * a coordinate or metadata instance equal to that of
              the desired coordinates e.g., :class:`~iris.coords.AuxCoord` or
              :class:`~iris.common.metadata.CoordMetadata`.

        * standard_name (str):
            The CF standard name of the desired coordinate. If ``None``, does not
            check for ``standard_name``.

        * long_name (str):
            An unconstrained description of the coordinate. If ``None``, does not
            check for ``long_name``.

        * var_name (str):
            The NetCDF variable name of the desired coordinate. If ``None``, does
            not check for ``var_name``.

        * attributes (dict):
            A dictionary of attributes desired on the coordinates. If ``None``,
            does not check for ``attributes``.

        * axis (str):
            The desired coordinate axis, see :func:`~iris.util.guess_coord_axis`.
            If ``None``, does not check for ``axis``. Accepts the values ``X``,
            ``Y``, ``Z`` and ``T`` (case-insensitive).

        * include_node (bool):
            Include all ``node`` coordinates in the list of objects to be matched
            for potential removal.

        * include_edge (bool):
            Include all ``edge`` coordinates in the list of objects to be matched
            for potential removal.

        * include_face (bool):
            Include all ``face`` coordinates in the list of objects to be matched
            for potential removal.

        Returns:
            A list of :class:`~iris.coords.AuxCoord` coordinates removed from
            the :class:`Mesh` that matched the given criteria.

        """
        # Filter out absent arguments - only expecting face coords sometimes,
        # same will be true of volumes in future.
        kwargs = {
            "item": item,
            "standard_name": standard_name,
            "long_name": long_name,
            "var_name": var_name,
            "attributes": attributes,
            "axis": axis,
            "include_nodes": include_nodes,
            "include_edges": include_edges,
            "include_faces": include_faces,
        }
        kwargs = {k: v for k, v in kwargs.items() if v}

        return self._coord_manager.remove(**kwargs)

    def xml_element(self, doc):
        """
        Create the :class:`xml.dom.minidom.Element` that describes this
        :class:`Mesh`.

        Args:

        * doc (object):
            The parent :class:`xml.dom.minidom.Document`.

        Returns:
            The :class:`xml.dom.minidom.Element` that will describe this
            :class:`Mesh`, and the dictionary of attributes that require
            to be added to this element.

        """
        pass

    # the MeshCoord will always have bounds, perhaps points. However the MeshCoord.guess_points() may
    # be a very useful part of its behaviour.
    # after using MeshCoord.guess_points(), the user may wish to add the associated MeshCoord.points into
    # the Mesh as face_coordinates.

    # def to_AuxCoord(self, location, axis):
    #     # factory method
    #     # return the lazy AuxCoord(...) for the given location and axis
    #
    # def to_AuxCoords(self, location):
    #     # factory method
    #     # return the lazy AuxCoord(...), AuxCoord(...)
    #
    # def to_MeshCoord(self, location, axis):
    #     # factory method
    #     # return MeshCoord(..., location=location, axis=axis)
    #     # use Connectivity.indices_by_src() for fetching indices, passing in the lazy_indices() result as an argument.
    #
    # def to_MeshCoords(self, location):
    #     # factory method
    #     # return MeshCoord(..., location=location, axis="x"), MeshCoord(..., location=location, axis="y")
    #     # use Connectivity.indices_by_src for fetching indices, passing in the lazy_indices() result as an argument.

    def dimension_names_reset(self, node=False, edge=False, face=False):
        """
        Reset the name used for the NetCDF variable representing the ``node``,
        ``edge`` and/or ``face`` dimension to ``None``.

        Kwargs:

        * node (bool):
            Reset the name of the ``node`` dimension if ``True``. Default
            is ``False``.

        * edge (bool):
            Reset the name of the ``edge`` dimension if ``True``. Default
            is ``False``.

        * face (bool):
            Reset the name of the ``face`` dimension if ``True``. Default
            is ``False``.

        """
        return self._set_dimension_names(node, edge, face, reset=True)

    def dimension_names(self, node=None, edge=None, face=None):
        """
        Assign the name to be used for the NetCDF variable representing
        the ``node``, ``edge`` and ``face`` dimension.

        The default value of ``None`` will not be assigned to clear the
        associated ``node``, ``edge`` or ``face``. Instead use
        :meth:`Mesh.dimension_names_reset`.

        Kwargs:

        * node (str):
            The name to be used for the NetCDF variable representing the
            ``node`` dimension.

        * edge (str):
            The name to be used for the NetCDF variable representing the
            ``edge`` dimension.

        * face (str):
            The name to be used for the NetCDF variable representing the
            ``face`` dimension.

        """
        return self._set_dimension_names(node, edge, face, reset=False)

    @property
    def cf_role(self):
        """The UGRID ``cf_role`` attribute of the :class:`Mesh`."""
        return "mesh_topology"

    @property
    def topology_dimension(self):
        """
        The UGRID ``topology_dimension`` attribute represents the highest
        dimensionality of all the geometric elements (node, edge, face) represented
        within the :class:`Mesh`.

        """
        return self._metadata_manager.topology_dimension


class _Mesh1DCoordinateManager:
    """

    TBD: require clarity on coord_systems validation
    TBD: require clarity on __eq__ support
    TBD: rationalise self.coords() logic with other manager and Cube

    """

    REQUIRED = (
        "node_x",
        "node_y",
    )
    OPTIONAL = (
        "edge_x",
        "edge_y",
    )

    def __init__(self, node_x, node_y, edge_x=None, edge_y=None):
        # initialise all the coordinates
        self.ALL = self.REQUIRED + self.OPTIONAL
        self._members = {member: None for member in self.ALL}

        # required coordinates
        self.node_x = node_x
        self.node_y = node_y
        # optional coordinates
        self.edge_x = edge_x
        self.edge_y = edge_y

    def __eq__(self, other):
        # TBD: this is a minimalist implementation and requires to be revisited
        return id(self) == id(other)

    def __getstate__(self):
        return self._members

    def __iter__(self):
        for item in self._members.items():
            yield item

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is not NotImplemented:
            result = not result
        return result

    def __repr__(self):
        args = [
            f"{member}={coord!r}"
            for member, coord in self
            if coord is not None
        ]
        return f"{self.__class__.__name__}({', '.join(args)})"

    def __setstate__(self, state):
        self._members = state

    def __str__(self):
        args = [f"{member}" for member, coord in self if coord is not None]
        return f"{self.__class__.__name__}({', '.join(args)})"

    def _remove(self, **kwargs):
        result = {}
        members = self.filters(**kwargs)

        for member in members.keys():
            if member in self.REQUIRED:
                dmsg = f"Ignoring request to remove required coordinate {member!r}"
                logger.debug(dmsg, extra=dict(cls=self.__class__.__name__))
            else:
                result[member] = members[member]
                setattr(self, member, None)

        return result

    def _setter(self, location, axis, coord, shape):
        axis = axis.lower()
        member = f"{location}_{axis}"

        # enforce the UGRID minimum coordinate requirement
        if location == "node" and coord is None:
            emsg = (
                f"{member!r} is a required coordinate, cannot set to 'None'."
            )
            raise ValueError(emsg)

        if coord is not None:
            if not isinstance(coord, AuxCoord):
                emsg = f"{member!r} requires to be an 'AuxCoord', got {type(coord)}."
                raise TypeError(emsg)

            guess_axis = guess_coord_axis(coord)

            if guess_axis and guess_axis.lower() != axis:
                emsg = f"{member!r} requires a {axis}-axis like 'AuxCoord', got a {guess_axis.lower()}-axis like."
                raise TypeError(emsg)

            if coord.climatological:
                emsg = f"{member!r} cannot be a climatological 'AuxCoord'."
                raise TypeError(emsg)

            if shape is not None and coord.shape != shape:
                emsg = f"{member!r} requires to have shape {shape!r}, got {coord.shape!r}."
                raise ValueError(emsg)

        self._members[member] = coord

    def _shape(self, location):
        coord = getattr(self, f"{location}_x")
        shape = coord.shape if coord is not None else None
        if shape is None:
            coord = getattr(self, f"{location}_y")
            if coord is not None:
                shape = coord.shape
        return shape

    @property
    def _edge_shape(self):
        return self._shape(location="edge")

    @property
    def _node_shape(self):
        return self._shape(location="node")

    @property
    def all_members(self):
        return Mesh1DCoords(**self._members)

    @property
    def edge_coords(self):
        return MeshEdgeCoords(edge_x=self.edge_x, edge_y=self.edge_y)

    @property
    def edge_x(self):
        return self._members["edge_x"]

    @edge_x.setter
    def edge_x(self, coord):
        self._setter(
            location="edge", axis="x", coord=coord, shape=self._edge_shape
        )

    @property
    def edge_y(self):
        return self._members["edge_y"]

    @edge_y.setter
    def edge_y(self, coord):
        self._setter(
            location="edge", axis="y", coord=coord, shape=self._edge_shape
        )

    @property
    def node_coords(self):
        return MeshNodeCoords(node_x=self.node_x, node_y=self.node_y)

    @property
    def node_x(self):
        return self._members["node_x"]

    @node_x.setter
    def node_x(self, coord):
        self._setter(
            location="node", axis="x", coord=coord, shape=self._node_shape
        )

    @property
    def node_y(self):
        return self._members["node_y"]

    @node_y.setter
    def node_y(self, coord):
        self._setter(
            location="node", axis="y", coord=coord, shape=self._node_shape
        )

    def _add(self, coords):
        member_x, member_y = coords._fields

        # deal with the special case where both members are changing
        if coords[0] is not None and coords[1] is not None:
            cache_x = self._members[member_x]
            cache_y = self._members[member_y]
            self._members[member_x] = None
            self._members[member_y] = None

            try:
                setattr(self, member_x, coords[0])
                setattr(self, member_y, coords[1])
            except (TypeError, ValueError):
                # restore previous valid state
                self._members[member_x] = cache_x
                self._members[member_y] = cache_y
                # now, re-raise the exception
                raise
        else:
            # deal with the case where one or no member is changing
            if coords[0] is not None:
                setattr(self, member_x, coords[0])
            if coords[1] is not None:
                setattr(self, member_y, coords[1])

    def add(self, node_x=None, node_y=None, edge_x=None, edge_y=None):
        """
        use self.remove(edge_x=True) to remove a coordinate e.g., using the
        pattern self.add(edge_x=None) will not remove the edge_x coordinate

        """
        self._add(MeshNodeCoords(node_x, node_y))
        self._add(MeshEdgeCoords(edge_x, edge_y))

    def filter(self, **kwargs):
        # TODO: rationalise commonality with MeshConnectivityManager.filter and Cube.coord.
        result = self.filters(**kwargs)

        if len(result) > 1:
            names = ", ".join(
                f"{member}={coord!r}" for member, coord in result.items()
            )
            emsg = (
                f"Expected to find exactly 1 coordinate, but found {len(result)}. "
                f"They were: {names}."
            )
            raise CoordinateNotFoundError(emsg)

        if len(result) == 0:
            item = kwargs["item"]
            if item is not None:
                if not isinstance(item, str):
                    item = item.name()
            name = (
                item
                or kwargs["standard_name"]
                or kwargs["long_name"]
                or kwargs["var_name"]
                or None
            )
            name = "" if name is None else f"{name!r} "
            emsg = (
                f"Expected to find exactly 1 {name}coordinate, but found none."
            )
            raise CoordinateNotFoundError(emsg)

        return result

    def filters(
        self,
        item=None,
        standard_name=None,
        long_name=None,
        var_name=None,
        attributes=None,
        axis=None,
        include_nodes=None,
        include_edges=None,
        include_faces=None,
    ):
        # TBD: support coord_systems?

        # Preserve original argument before modifying.
        face_requested = include_faces

        # Rationalise the tri-state behaviour.
        args = [include_nodes, include_edges, include_faces]
        state = not any(set(filter(lambda arg: arg is not None, args)))
        include_nodes, include_edges, include_faces = map(
            lambda arg: arg if arg is not None else state, args
        )

        def populated_coords(coords_tuple):
            return list(filter(None, list(coords_tuple)))

        members = []
        if include_nodes:
            members += populated_coords(self.node_coords)
        if include_edges:
            members += populated_coords(self.edge_coords)
        if hasattr(self, "face_coords"):
            if include_faces:
                members += populated_coords(self.face_coords)
        elif face_requested:
            dmsg = "Ignoring request to filter non-existent 'face_coords'"
            logger.debug(dmsg, extra=dict(cls=self.__class__.__name__))

        result = metadata_filter(
            members,
            item=item,
            standard_name=standard_name,
            long_name=long_name,
            var_name=var_name,
            attributes=attributes,
            axis=axis,
        )

        # Use the results to filter the _members dict for returning.
        result_ids = [id(r) for r in result]
        result_dict = {
            k: v for k, v in self._members.items() if id(v) in result_ids
        }
        return result_dict

    def remove(
        self,
        item=None,
        standard_name=None,
        long_name=None,
        var_name=None,
        attributes=None,
        axis=None,
        include_nodes=None,
        include_edges=None,
    ):
        return self._remove(
            item=item,
            standard_name=standard_name,
            long_name=long_name,
            var_name=var_name,
            attributes=attributes,
            axis=axis,
            include_nodes=include_nodes,
            include_edges=include_edges,
        )


class _Mesh2DCoordinateManager(_Mesh1DCoordinateManager):
    OPTIONAL = (
        "edge_x",
        "edge_y",
        "face_x",
        "face_y",
    )

    def __init__(
        self,
        node_x,
        node_y,
        edge_x=None,
        edge_y=None,
        face_x=None,
        face_y=None,
    ):
        super().__init__(node_x, node_y, edge_x=edge_x, edge_y=edge_y)

        # optional coordinates
        self.face_x = face_x
        self.face_y = face_y

    @property
    def _face_shape(self):
        return self._shape(location="face")

    @property
    def all_members(self):
        return Mesh2DCoords(**self._members)

    @property
    def face_coords(self):
        return MeshFaceCoords(face_x=self.face_x, face_y=self.face_y)

    @property
    def face_x(self):
        return self._members["face_x"]

    @face_x.setter
    def face_x(self, coord):
        self._setter(
            location="face", axis="x", coord=coord, shape=self._face_shape
        )

    @property
    def face_y(self):
        return self._members["face_y"]

    @face_y.setter
    def face_y(self, coord):
        self._setter(
            location="face", axis="y", coord=coord, shape=self._face_shape
        )

    def add(
        self,
        node_x=None,
        node_y=None,
        edge_x=None,
        edge_y=None,
        face_x=None,
        face_y=None,
    ):
        super().add(node_x=node_x, node_y=node_y, edge_x=edge_x, edge_y=edge_y)
        self._add(MeshFaceCoords(face_x, face_y))

    def remove(
        self,
        item=None,
        standard_name=None,
        long_name=None,
        var_name=None,
        attributes=None,
        axis=None,
        include_nodes=None,
        include_edges=None,
        include_faces=None,
    ):
        return self._remove(
            item=item,
            standard_name=standard_name,
            long_name=long_name,
            var_name=var_name,
            attributes=attributes,
            axis=axis,
            include_nodes=include_nodes,
            include_edges=include_edges,
            include_faces=include_faces,
        )


class _MeshConnectivityManagerBase(ABC):
    # Override these in subclasses.
    REQUIRED: tuple = NotImplemented
    OPTIONAL: tuple = NotImplemented

    def __init__(self, *connectivities):
        cf_roles = [c.cf_role for c in connectivities]
        for requisite in self.REQUIRED:
            if requisite not in cf_roles:
                message = f"{type(self).__name__} requires a {requisite} Connectivity."
                raise ValueError(message)

        self.ALL = self.REQUIRED + self.OPTIONAL
        self._members = {member: None for member in self.ALL}
        self.add(*connectivities)

    def __eq__(self, other):
        # TBD: this is a minimalist implementation and requires to be revisited
        return id(self) == id(other)

    def __getstate__(self):
        return self._members

    def __iter__(self):
        for item in self._members.items():
            yield item

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is not NotImplemented:
            result = not result
        return result

    def __repr__(self):
        args = [
            f"{member}={connectivity!r}"
            for member, connectivity in self
            if connectivity is not None
        ]
        return f"{self.__class__.__name__}({', '.join(args)})"

    def __setstate__(self, state):
        self._members = state

    def __str__(self):
        args = [
            f"{member}"
            for member, connectivity in self
            if connectivity is not None
        ]
        return f"{self.__class__.__name__}({', '.join(args)})"

    @property
    @abstractmethod
    def all_members(self):
        return NotImplemented

    def add(self, *connectivities):
        # Since Connectivity classes include their cf_role, no setters will be
        # provided, just a means to add one or more connectivities to the
        # manager.
        # No warning is raised for duplicate cf_roles - user is trusted to
        # validate their outputs.
        add_dict = {}
        for connectivity in connectivities:
            if not isinstance(connectivity, Connectivity):
                message = f"Expected Connectivity, got: {type(connectivity)} ."
                raise TypeError(message)
            cf_role = connectivity.cf_role
            if cf_role not in self.ALL:
                message = (
                    f"Not adding connectivity ({cf_role}: "
                    f"{connectivity!r}) - cf_role must be one of: {self.ALL} ."
                )
                logger.debug(message, extra=dict(cls=self.__class__.__name__))
            else:
                add_dict[cf_role] = connectivity

        # Validate shapes.
        proposed_members = {**self._members, **add_dict}
        locations = set(
            [
                c.src_location
                for c in proposed_members.values()
                if c is not None
            ]
        )
        for location in locations:
            counts = [
                len(c.indices_by_src(c.lazy_indices()))
                for c in proposed_members.values()
                if c is not None and c.src_location == location
            ]
            # Check is list values are identical.
            if not counts.count(counts[0]) == len(counts):
                message = (
                    f"Invalid Connectivities provided - inconsistent "
                    f"{location} counts."
                )
                raise ValueError(message)

        self._members = proposed_members

    def filter(self, **kwargs):
        # TODO: rationalise commonality with MeshCoordManager.filter and Cube.coord.
        result = self.filters(**kwargs)
        if len(result) > 1:
            names = ", ".join(
                f"{member}={connectivity!r}"
                for member, connectivity in result.items()
            )
            message = (
                f"Expected to find exactly 1 connectivity, but found "
                f"{len(result)}. They were: {names}."
            )
            raise ConnectivityNotFoundError(message)
        elif len(result) == 0:
            item = kwargs["item"]
            _name = item
            if item is not None:
                if not isinstance(item, str):
                    _name = item.name()
            bad_name = (
                _name or kwargs["standard_name"] or kwargs["long_name"] or ""
            )
            message = (
                f"Expected to find exactly 1 {bad_name} connectivity, "
                f"but found none."
            )
            raise ConnectivityNotFoundError(message)

        return result

    def filters(
        self,
        item=None,
        standard_name=None,
        long_name=None,
        var_name=None,
        attributes=None,
        cf_role=None,
        contains_node=None,
        contains_edge=None,
        contains_face=None,
    ):
        members = [c for c in self._members.values() if c is not None]

        if cf_role is not None:
            members = [
                instance for instance in members if instance.cf_role == cf_role
            ]

        def location_filter(instances, loc_arg, loc_name):
            if loc_arg is False:
                filtered = [
                    instance
                    for instance in instances
                    if loc_name
                    not in (instance.src_location, instance.tgt_location)
                ]
            elif loc_arg is None:
                filtered = instances
            else:
                # Interpret any other value as =True.
                filtered = [
                    instance
                    for instance in instances
                    if loc_name
                    in (instance.src_location, instance.tgt_location)
                ]

            return filtered

        for arg, loc in (
            (contains_node, "node"),
            (contains_edge, "edge"),
            (contains_face, "face"),
        ):
            members = location_filter(members, arg, loc)

        # No need to actually modify filtering behaviour - already won't return
        # any face cf-roles if none are present.
        supports_faces = any(["face" in role for role in self.ALL])
        if contains_face and not supports_faces:
            message = (
                "Ignoring request to filter for non-existent 'face' cf-roles."
            )
            logger.debug(message, extra=dict(cls=self.__class__.__name__))

        result = metadata_filter(
            members,
            item=item,
            standard_name=standard_name,
            long_name=long_name,
            var_name=var_name,
            attributes=attributes,
        )

        # Use the results to filter the _members dict for returning.
        result_ids = [id(r) for r in result]
        result_dict = {
            k: v for k, v in self._members.items() if id(v) in result_ids
        }
        return result_dict

    def remove(
        self,
        item=None,
        standard_name=None,
        long_name=None,
        var_name=None,
        attributes=None,
        cf_role=None,
        contains_node=None,
        contains_edge=None,
        contains_face=None,
    ):
        removal_dict = self.filters(
            item=item,
            standard_name=standard_name,
            long_name=long_name,
            var_name=var_name,
            attributes=attributes,
            cf_role=cf_role,
            contains_node=contains_node,
            contains_edge=contains_edge,
            contains_face=contains_face,
        )
        for cf_role in self.REQUIRED:
            excluded = removal_dict.pop(cf_role, None)
            if excluded:
                message = (
                    f"Ignoring request to remove required connectivity "
                    f"({cf_role}: {excluded!r})"
                )
                logger.debug(message, extra=dict(cls=self.__class__.__name__))

        for cf_role in removal_dict.keys():
            self._members[cf_role] = None

        return removal_dict


class _Mesh1DConnectivityManager(_MeshConnectivityManagerBase):
    REQUIRED = ("edge_node_connectivity",)
    OPTIONAL = ()

    @property
    def all_members(self):
        return Mesh1DConnectivities(edge_node=self.edge_node)

    @property
    def edge_node(self):
        return self._members["edge_node_connectivity"]


class _Mesh2DConnectivityManager(_MeshConnectivityManagerBase):
    REQUIRED = ("face_node_connectivity",)
    OPTIONAL = (
        "edge_node_connectivity",
        "face_edge_connectivity",
        "face_face_connectivity",
        "edge_face_connectivity",
        "boundary_node_connectivity",
    )

    @property
    def all_members(self):
        return Mesh2DConnectivities(
            face_node=self.face_node,
            edge_node=self.edge_node,
            face_edge=self.face_edge,
            face_face=self.face_face,
            edge_face=self.edge_face,
            boundary_node=self.boundary_node,
        )

    @property
    def boundary_node(self):
        return self._members["boundary_node_connectivity"]

    @property
    def edge_face(self):
        return self._members["edge_face_connectivity"]

    @property
    def edge_node(self):
        return self._members["edge_node_connectivity"]

    @property
    def face_edge(self):
        return self._members["face_edge_connectivity"]

    @property
    def face_face(self):
        return self._members["face_face_connectivity"]

    @property
    def face_node(self):
        return self._members["face_node_connectivity"]


class MeshCoordMetadata(BaseMetadata):
    """
    Metadata container for a :class:`~iris.coords.MeshCoord`.
    """

    _members = ("location", "axis")
    # NOTE: in future, we may add 'mesh' as part of this metadata,
    # as the Mesh seems part of the 'identity' of a MeshCoord.
    # For now we omit it, particularly as we don't yet implement Mesh.__eq__.
    #
    # Thus, for now, the MeshCoord class will need to handle 'mesh' explicitly
    # in identity / comparison, but in future that may be simplified.

    __slots__ = ()

    @wraps(BaseMetadata.__eq__, assigned=("__doc__",), updated=())
    @lenient_service
    def __eq__(self, other):
        return super().__eq__(other)

    def _combine_lenient(self, other):
        """
        Perform lenient combination of metadata members for MeshCoord.

        Args:

        * other (MeshCoordMetadata):
            The other metadata participating in the lenient combination.

        Returns:
            A list of combined metadata member values.

        """
        # It is actually "strict" : return None except where members are equal.
        def func(field):
            left = getattr(self, field)
            right = getattr(other, field)
            return left if left == right else None

        # Note that, we use "_members" not "_fields".
        values = [func(field) for field in self._members]
        # Perform lenient combination of the other parent members.
        result = super()._combine_lenient(other)
        result.extend(values)

        return result

    def _compare_lenient(self, other):
        """
        Perform lenient equality of metadata members for MeshCoord.

        Args:

        * other (MeshCoordMetadata):
            The other metadata participating in the lenient comparison.

        Returns:
            Boolean.

        """
        # Perform "strict" comparison for the MeshCoord specific members
        # 'location', 'axis' : for equality, they must all match.
        result = all(
            [
                getattr(self, field) == getattr(other, field)
                for field in self._members
            ]
        )
        if result:
            # Perform lenient comparison of the other parent members.
            result = super()._compare_lenient(other)

        return result

    def _difference_lenient(self, other):
        """
        Perform lenient difference of metadata members for MeshCoord.

        Args:

        * other (MeshCoordMetadata):
            The other MeshCoord metadata participating in the lenient
            difference.

        Returns:
            A list of different metadata member values.

        """
        # Perform "strict" difference for location / axis.
        def func(field):
            left = getattr(self, field)
            right = getattr(other, field)
            return None if left == right else (left, right)

        # Note that, we use "_members" not "_fields".
        values = [func(field) for field in self._members]
        # Perform lenient difference of the other parent members.
        result = super()._difference_lenient(other)
        result.extend(values)

        return result

    @wraps(BaseMetadata.combine, assigned=("__doc__",), updated=())
    @lenient_service
    def combine(self, other, lenient=None):
        return super().combine(other, lenient=lenient)

    @wraps(BaseMetadata.difference, assigned=("__doc__",), updated=())
    @lenient_service
    def difference(self, other, lenient=None):
        return super().difference(other, lenient=lenient)

    @wraps(BaseMetadata.equal, assigned=("__doc__",), updated=())
    @lenient_service
    def equal(self, other, lenient=None):
        return super().equal(other, lenient=lenient)


# Add our new optional metadata operations into the 'convenience collections'
# of lenient metadata services.
# TODO: when included in 'iris.common.metadata', install each one directly ?
_op_names_and_service_collections = [
    ("combine", SERVICES_COMBINE),
    ("difference", SERVICES_DIFFERENCE),
    ("__eq__", SERVICES_EQUAL),
    ("equal", SERVICES_EQUAL),
]
_metadata_classes = [ConnectivityMetadata, MeshMetadata, MeshCoordMetadata]
for _cls in _metadata_classes:
    for _name, _service_collection in _op_names_and_service_collections:
        _method = getattr(_cls, _name)
        _service_collection.append(_method)
        SERVICES.append(_method)

del (
    _op_names_and_service_collections,
    _metadata_classes,
    _cls,
    _name,
    _service_collection,
    _method,
)
