# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""The common metadata API classes for :mod:`iris.experimental.ugrid.mesh`.

Eventual destination: :mod:`iris.common.metadata`.

"""
from functools import wraps

from ...common import BaseMetadata
from ...common.lenient import _lenient_service as lenient_service
from ...common.metadata import (
    SERVICES,
    SERVICES_COMBINE,
    SERVICES_DIFFERENCE,
    SERVICES_EQUAL,
)


class ConnectivityMetadata(BaseMetadata):
    """Metadata container for a :class:`~iris.experimental.ugrid.mesh.Connectivity`."""

    # The "location_axis" member is stateful only, and does not participate in
    # lenient/strict equivalence.
    _members = ("cf_role", "start_index", "location_axis")

    __slots__ = ()

    @wraps(BaseMetadata.__eq__, assigned=("__doc__",), updated=())
    @lenient_service
    def __eq__(self, other):
        return super().__eq__(other)

    def _combine_lenient(self, other):
        """Perform lenient combination of metadata members for connectivities.

        Parameters
        ----------
        other : ConnectivityMetadata
            The other connectivity metadata participating in the lenient
            combination.

        Returns
        -------
        A list of combined metadata member values.

        """

        # Perform "strict" combination for "cf_role", "start_index", "location_axis".
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
        """Perform lenient equality of metadata members for connectivities.

        Parameters
        ----------
        other : ConnectivityMetadata
            The other connectivity metadata participating in the lenient
            comparison.

        Returns
        -------
        bool

        """
        # Perform "strict" comparison for "cf_role", "start_index".
        # The "location_axis" member is not part of lenient equivalence.
        members = filter(
            lambda member: member != "location_axis",
            ConnectivityMetadata._members,
        )
        result = all(
            [getattr(self, field) == getattr(other, field) for field in members]
        )
        if result:
            # Perform lenient comparison of the other parent members.
            result = super()._compare_lenient(other)

        return result

    def _difference_lenient(self, other):
        """Perform lenient difference of metadata members for connectivities.

        Parameters
        ----------
        other : ConnectivityMetadata
            The other connectivity metadata participating in the lenient
            difference.

        Returns
        -------
        A list of difference metadata member values.

        """

        # Perform "strict" difference for "cf_role", "start_index", "location_axis".
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
    """Metadata container for a :class:`~iris.experimental.ugrid.mesh.Mesh`."""

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
        """Perform lenient combination of metadata members for meshes.

        Parameters
        ----------
        other : MeshMetadata
            The other mesh metadata participating in the lenient
            combination.

        Returns
        -------
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
        """Perform lenient equality of metadata members for meshes.

        Parameters
        ----------
        other : MeshMetadata
            The other mesh metadata participating in the lenient
            comparison.

        Returns
        -------
        bool

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
        """Perform lenient difference of metadata members for meshes.

        Parameters
        ----------
        other : MeshMetadata
            The other mesh metadata participating in the lenient
            difference.

        Returns
        -------
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


class MeshCoordMetadata(BaseMetadata):
    """Metadata container for a :class:`~iris.coords.MeshCoord`."""

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
        """Perform lenient combination of metadata members for MeshCoord.

        Parameters
        ----------
        other : MeshCoordMetadata
            The other metadata participating in the lenient combination.

        Returns
        -------
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
        """Perform lenient equality of metadata members for MeshCoord.

        Parameters
        ----------
        other : MeshCoordMetadata
            The other metadata participating in the lenient comparison.

        Returns
        -------
        bool

        """
        # Perform "strict" comparison for the MeshCoord specific members
        # 'location', 'axis' : for equality, they must all match.
        result = all(
            [getattr(self, field) == getattr(other, field) for field in self._members]
        )
        if result:
            # Perform lenient comparison of the other parent members.
            result = super()._compare_lenient(other)

        return result

    def _difference_lenient(self, other):
        """Perform lenient difference of metadata members for MeshCoord.

        Parameters
        ----------
        other : MeshCoordMetadata
            The other MeshCoord metadata participating in the lenient
            difference.

        Returns
        -------
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
