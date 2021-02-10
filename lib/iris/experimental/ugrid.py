# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

"""
Infra-structure for unstructured mesh support, based on
CF UGRID Convetions (v1.0), https://ugrid-conventions.github.io/ugrid-conventions/

"""

from functools import wraps

import dask.array as da
import numpy as np

from .. import _lazy_data as _lazy
from ..common.metadata import (
    BaseMetadata,
    metadata_manager_factory,
    SERVICES,
    SERVICES_COMBINE,
    SERVICES_EQUAL,
    SERVICES_DIFFERENCE,
)
from ..common.lenient import _lenient_service as lenient_service
from ..config import get_logger
from ..coords import _DimensionalMetadata


__all__ = [
    "Connectivity",
    "ConnectivityMetadata",
]


# Configure the logger.
logger = get_logger(__name__, fmt="[%(cls)s.%(funcName)s]")


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
            The netCDF variable name for the connectivity.
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
    Metadata container for a :class:`~iris.coords.Connectivity`.

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


#: Convenience collection of lenient metadata combine services.
SERVICES_COMBINE.append(ConnectivityMetadata.combine)
SERVICES.append(ConnectivityMetadata.combine)


#: Convenience collection of lenient metadata difference services.
SERVICES_DIFFERENCE.append(ConnectivityMetadata.difference)
SERVICES.append(ConnectivityMetadata.difference)

#: Convenience collection of lenient metadata equality services.
SERVICES_EQUAL.extend(
    [ConnectivityMetadata.__eq__, ConnectivityMetadata.equal]
)
SERVICES.extend([ConnectivityMetadata.__eq__, ConnectivityMetadata.equal])
