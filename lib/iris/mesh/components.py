# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""Iris data model representation of CF UGrid's Mesh and its constituent parts.

Eventual destination: dedicated module in :mod:`iris` root.

"""

from abc import ABC, abstractmethod
from collections import namedtuple
from collections.abc import Container
from contextlib import contextmanager
from typing import Iterable, Literal
import warnings

from cf_units import Unit
from dask import array as da
import numpy as np

from iris.common.metadata import ConnectivityMetadata, MeshCoordMetadata, MeshMetadata

from .. import _lazy_data as _lazy
from ..common import CFVariableMixin, metadata_filter, metadata_manager_factory
from ..common.metadata import BaseMetadata
from ..config import get_logger
from ..coords import AuxCoord, _DimensionalMetadata
from ..exceptions import ConnectivityNotFoundError, CoordinateNotFoundError
from ..util import array_equal, clip_string, guess_coord_axis
from ..warnings import IrisVagueMetadataWarning

# Configure the logger.
logger = get_logger(__name__, propagate=True, handler=False)

#: Numpy "threshold" printoptions default argument.
NP_PRINTOPTIONS_THRESHOLD = 10
#: Numpy "edgeitems" printoptions default argument.
NP_PRINTOPTIONS_EDGEITEMS = 2

#
# MeshXY dimension names namedtuples.
#

#: Namedtuple for 1D mesh topology NetCDF variable dimension names.
Mesh1DNames = namedtuple("Mesh1DNames", ["node_dimension", "edge_dimension"])
#: Namedtuple for 2D mesh topology NetCDF variable dimension names.
Mesh2DNames = namedtuple(
    "Mesh2DNames", ["node_dimension", "edge_dimension", "face_dimension"]
)

#
# MeshXY coordinate manager namedtuples.
#

#: Namedtuple for 1D mesh :class:`~iris.coords.AuxCoord` coordinates.
Mesh1DCoords = namedtuple("Mesh1DCoords", ["node_x", "node_y", "edge_x", "edge_y"])
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
# MeshXY connectivity manager namedtuples.
#

#: Namedtuple for 1D mesh :class:`~iris.mesh.Connectivity` instances.
Mesh1DConnectivities = namedtuple("Mesh1DConnectivities", ["edge_node"])
#: Namedtuple for 2D mesh :class:`~iris.mesh.Connectivity` instances.
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
    """CF-UGRID topology.

    A CF-UGRID topology connectivity, describing the topological relationship
    between two types of mesh element. One or more connectivities make up a
    CF-UGRID topology - a constituent of a CF-UGRID mesh.

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
        location_axis=0,
    ):
        """Construct a single connectivity.

        Parameters
        ----------
        indices : :class:`numpy.ndarray` or :class:`numpy.ma.core.MaskedArray` or :class:`dask.array.Array`
            2D array giving the topological connection relationship between
            :attr:`location` elements and :attr:`connected` elements.
            The :attr:`location_axis` dimension indexes over the
            :attr:`location` dimension of the mesh - i.e. its length matches
            the total number of :attr:`location` elements in the mesh. The
            :attr:`connected_axis` dimension can be any length, corresponding
            to the highest number of :attr:`connected` elements connected to a
            :attr:`location` element. The array values are indices into the
            :attr:`connected` dimension of the mesh. If the number of
            :attr:`connected` elements varies between :attr:`location`
            elements: use a :class:`numpy.ma.core.MaskedArray` and mask the
            :attr:`location` elements' unused index 'slots'. Use a
            :class:`dask.array.Array` to keep indices 'lazy'.
        cf_role : str
            Denotes the topological relationship that this connectivity
            describes. Made up of this array's :attr:`location`, and the
            :attr:`connected` element type that is indexed by the array.
            See :attr:`UGRID_CF_ROLES` for valid arguments.
        standard_name : str, optional
            CF standard name of the connectivity.
            (NOTE: this is not expected by the UGRID conventions, but will be
            handled in Iris' standard way if provided).
        long_name : str, optional
            Descriptive name of the connectivity.
        var_name : str, optional
            The NetCDF variable name for the connectivity.
        units : cf_units.Unit, optional
            The :class:`~cf_units.Unit` of the connectivity's values.
            Can be a string, which will be converted to a Unit object.
            (NOTE: this is not expected by the UGRID conventions, but will be
            handled in Iris' standard way if provided).
        attributes : dict, optional
            A dictionary containing other cf and user-defined attributes.
        start_index : int, optional
            Either ``0`` or ``1``. Default is ``0``. Denotes whether
            :attr:`indices` uses 0-based or 1-based indexing (allows support
            for Fortran and legacy NetCDF files).
        location_axis : int, optional
            Either ``0`` or ``1``. Default is ``0``. Denotes which axis
            of :attr:`indices` varies over the :attr:`location` elements (the
            alternate axis therefore varying over :attr:`connected` elements).
            (This parameter allows support for fastest varying index being
            either first or last).
            E.g. for ``face_node_connectivity``, for 10 faces:
            ``indices.shape[location_axis] == 10``.

        """

        def validate_arg_vs_list(arg_name, arg, valid_list):
            if arg not in valid_list:
                error_msg = (
                    f"Invalid {arg_name} . Got: {arg} . Must be one of: {valid_list} ."
                )
                raise ValueError(error_msg)

        # Configure the metadata manager.
        self._metadata_manager = metadata_manager_factory(ConnectivityMetadata)

        validate_arg_vs_list("start_index", start_index, [0, 1])
        # indices array will be 2-dimensional, so must be either 0 or 1.
        validate_arg_vs_list("location_axis", location_axis, [0, 1])
        validate_arg_vs_list("cf_role", cf_role, Connectivity.UGRID_CF_ROLES)

        self._metadata_manager.start_index = start_index
        self._metadata_manager.location_axis = location_axis
        self._metadata_manager.cf_role = cf_role

        self._connected_axis = 1 - location_axis
        self._location, self._connected = cf_role.split("_")[:2]

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
        """The category of topological relationship that this connectivity describes.

        **Read-only** - validity of :attr:`indices` is dependent on
        :attr:`cf_role`. A new :class:`Connectivity` must therefore be defined
        if a different :attr:`cf_role` is needed.

        """
        return self._metadata_manager.cf_role

    @property
    def location(self):
        """Derived from the connectivity's :attr:`cf_role`.

        Derived from the connectivity's :attr:`cf_role` - the first part, e.g.
        ``face`` in ``face_node_connectivity``. Refers to the elements that
        vary along the :attr:`location_axis` of the connectivity's
        :attr:`indices` array.

        """
        return self._location

    @property
    def connected(self):
        """Derived from the connectivity's :attr:`cf_role`.

        Derived from the connectivity's :attr:`cf_role` - the second part, e.g.
        ``node`` in ``face_node_connectivity``. Refers to the elements indexed
        by the values in the connectivity's :attr:`indices` array.

        """
        return self._connected

    @property
    def start_index(self):
        """The base value of the connectivity's :attr:`indices` array; either ``0`` or ``1``.

        **Read-only** - validity of :attr:`indices` is dependent on
        :attr:`start_index`. A new :class:`Connectivity` must therefore be
        defined if a different :attr:`start_index` is needed.

        """
        return self._metadata_manager.start_index

    @property
    def location_axis(self):
        """The axis of the connectivity's :attr:`indices` array.

        The axis of the connectivity's :attr:`indices` array that varies
        over the connectivity's :attr:`location` elements. Either ``0`` or ``1``.
        **Read-only** - validity of :attr:`indices` is dependent on
        :attr:`location_axis`. Use :meth:`transpose` to create a new, transposed
        :class:`Connectivity` if a different :attr:`location_axis` is needed.

        """
        return self._metadata_manager.location_axis

    @property
    def connected_axis(self):
        """Derived as the alternate value of :attr:`location_axis`.

        Derived as the alternate value of :attr:`location_axis` - each must
        equal either ``0`` or ``1``. The axis of the connectivity's
        :attr:`indices` array that varies over the :attr:`connected` elements
        associated with each :attr:`location` element.

        """
        return self._connected_axis

    @property
    def indices(self):
        """The index values describing the topological relationship of the connectivity.

        The index values describing the topological relationship of the
        connectivity, as a NumPy array. Masked points indicate a
        :attr:`location` element  with fewer :attr:`connected` elements than
        other :attr:`location` elements described in this array - unused index
        'slots' are masked.

        **Read-only** - index values are only meaningful when combined with
        an appropriate :attr:`cf_role`, :attr:`start_index` and
        :attr:`location_axis`. A new :class:`Connectivity` must therefore be
        defined if different indices are needed.

        """
        return self._values

    def indices_by_location(self, indices=None):
        """Return a view of the indices array.

        Return a view of the indices array with :attr:`location_axis` **always** as
        the first axis - transposed if necessary. Can optionally pass in an
        identically shaped array on which to perform this operation (e.g. the
        output from :meth:`core_indices` or :meth:`lazy_indices`).

        Parameters
        ----------
        indices : array, optional
            The array on which to operate. If ``None``, will operate on
            :attr:`indices`. Default is ``None``.

        Returns
        -------
        result :
            A view of the indices array Transposed - if necessary - to put
            :attr:`location_axis` first.

        """
        if indices is None:
            indices = self.indices

        if indices.shape != self.shape:
            raise ValueError(
                f"Invalid indices provided. Must be shape={self.shape} , "
                f"got shape={indices.shape} ."
            )

        if self.location_axis == 0:
            result = indices
        elif self.location_axis == 1:
            result = indices.transpose()
        else:
            raise ValueError("Invalid location_axis.")

        return result

    def _validate_indices(self, indices, shapes_only=False):
        # Use shapes_only=True for a lower resource, less thorough validation
        # of indices by just inspecting the array shape instead of inspecting
        # individual masks. So will not catch individual location elements
        # having unacceptably low numbers of associated connected elements.

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
            indices_error(f"Expected 2-dimensional shape, got: shape={indices_shape} .")

        len_req_fail = False
        if shapes_only:
            location_shape = indices_shape[self.connected_axis]
            # Wrap as lazy to allow use of the same operations below
            # regardless of shapes_only.
            location_lengths = _lazy.as_lazy_data(np.asarray(location_shape))
        else:
            # Wouldn't be safe to use during __init__ validation, since
            # lazy_location_lengths requires self.indices to exist. Safe here since
            # shapes_only==False is only called manually, i.e. after
            # initialisation.
            location_lengths = self.lazy_location_lengths()
        if self.location in ("edge", "boundary"):
            if (location_lengths != 2).any().compute():
                len_req_fail = "len=2"
        else:
            if self.location == "face":
                min_size = 3
            elif self.location == "volume":
                if self.connected == "edge":
                    min_size = 6
                else:
                    min_size = 4
            else:
                raise NotImplementedError
            if (location_lengths < min_size).any().compute():
                len_req_fail = f"len>={min_size}"
        if len_req_fail:
            indices_error(
                f"Not all {self.location}s meet requirement: {len_req_fail} - "
                f"needed to describe '{self.cf_role}' ."
            )

    def validate_indices(self):
        """Perform a thorough validity check of this connectivity's :attr:`indices`.

        Perform a thorough validity check of this connectivity's
        :attr:`indices`. Includes checking the number of :attr:`connected`
        elements associated with each :attr:`location` element (specified using
        masks on the :attr:`indices` array) against the :attr:`cf_role`.

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
            # interaction with the indices array is via indices_by_location, which
            # corrects for this difference. (To enable this, location_axis does
            # not participate in ConnectivityMetadata to ConnectivityMetadata
            # equivalence).
            if hasattr(other, "metadata"):
                # metadata comparison
                eq = self.metadata == other.metadata
                if eq:
                    eq = (
                        self.shape == other.shape
                        and self.location_axis == other.location_axis
                    ) or (
                        self.shape == other.shape[::-1]
                        and self.location_axis == other.connected_axis
                    )
                if eq:
                    eq = array_equal(
                        self.indices_by_location(self.core_indices()),
                        other.indices_by_location(other.core_indices()),
                    )
        return eq

    def transpose(self):
        """Transpose :class:`Connectivity`.

        Create a new :class:`Connectivity`, identical to this one but with the
        :attr:`indices` array transposed and the :attr:`location_axis` value flipped.

        Returns
        -------
        :class:`Connectivity`
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
            location_axis=self.connected_axis,
        )
        return new_connectivity

    def lazy_indices(self):
        """Return a lazy array representing the connectivity's indices.

        Accessing this method will never cause the :attr:`indices` values to be
        loaded. Similarly, calling methods on, or indexing, the returned Array
        will not cause the connectivity to have loaded :attr:`indices`.

        If the :attr:`indices` have already been loaded for the connectivity,
        the returned Array will be a new lazy array wrapper.

        Returns
        -------
        A lazy array, representing the connectivity indices array.

        """
        return super()._lazy_values()

    def core_indices(self):
        """Return the indices array at the core of this connectivity.

        The indices array at the core of this connectivity, which may be a
        NumPy array or a Dask array.

        Returns
        -------
        :class:`numpy.ndarray` or :class:`numpy.ma.core.MaskedArray` or :class:`dask.array.Array`

        """
        return super()._core_values()

    def has_lazy_indices(self):
        """Check if the connectivity's :attr:`indices` array is a lazy Dask array or not.

        Returns
        -------
        bool

        """
        return super()._has_lazy_values()

    def lazy_location_lengths(self):
        """Return a lazy array representing the number of :attr:`connected` elements.

        Return a lazy array representing the number of :attr:`connected`
        elements associated with each of the connectivity's :attr:`location`
        elements, accounting for masks if present.

        Accessing this method will never cause the :attr:`indices` values to be
        loaded. Similarly, calling methods on, or indexing, the returned Array
        will not cause the connectivity to have loaded :attr:`indices`.

        The returned Array will be lazy regardless of whether the
        :attr:`indices` have already been loaded.

        Returns
        -------
        lazy array
            A lazy array, representing the number of :attr:`connected`
            elements associated with each :attr:`location` element.

        """
        location_mask_counts = da.sum(
            da.ma.getmaskarray(self.indices), axis=self.connected_axis
        )
        max_location_size = self.indices.shape[self.connected_axis]
        return max_location_size - location_mask_counts

    def location_lengths(self):
        """Return a NumPy array representing the number of :attr:`connected` elements.

        Return a NumPy array representing the number of :attr:`connected`
        elements associated with each of the connectivity's :attr:`location`
        elements, accounting for masks if present.

        Returns
        -------
        NumPy array
            A NumPy array, representing the number of :attr:`connected`
            elements associated with each :attr:`location` element.

        """
        return self.lazy_location_lengths().compute()

    def cube_dims(self, cube):
        """Not available on :class:`Connectivity`."""
        raise NotImplementedError

    def xml_element(self, doc):
        # Create the XML element as the camelCaseEquivalent of the
        # class name
        element = super().xml_element(doc)

        element.setAttribute("cf_role", self.cf_role)
        element.setAttribute("start_index", self.start_index)
        element.setAttribute("location_axis", self.location_axis)

        return element


class Mesh(CFVariableMixin, ABC):
    """A container representing the UGRID ``cf_role`` ``mesh_topology``.

    Warnings
    --------
    This class is not yet implemented. It is a placeholder for a future
    implementation of the UGRID mesh with the minimum possible assumptions. For
    instance: it is in theory possible for mesh node coordinates to be ANY
    combination of ANY coordinate type, e.g. spherical coordinates or an S2
    coordinate; the current :class:`MeshXY` subclass is based on the assumption
    of an X and a Y coordinate (e.g. longitude and latitude).

    .. todo::
        If SciTools/iris#5994 is agreed upon: implement this class.
          - Move whatever is appropriate from :class:`MeshXY` into this class,
            leaving behind only those elements specific to the assumption of
            X and Y node coordinates.
          - Remove the docstring warning, the NotImplementedError, and the uses
            of ABC/abstractmethod.
          - Add a cross-reference in the docstring for :class:`MeshXY`.
          - Search the Iris codebase for uses of :class:`MeshXY` and work out
            if/how they can be refactored to work with the more flexible
            :class:`Mesh`.

    """

    @abstractmethod
    def __init__(self):
        message = (
            f"The {self.__class__.__name__} class is not yet implemented. "
            "Use the MeshXY class instead."
        )
        raise NotImplementedError(message)


class MeshXY(Mesh):
    """A container representing the UGRID ``cf_role`` ``mesh_topology``.

    A container representing the UGRID [1]_ ``cf_role`` ``mesh_topology``, supporting
    1D network, 2D triangular, and 2D flexible mesh topologies.

    Based on the assumption of 2 :attr:`node_coords` - one associated with the
    X-axis (e.g. longitude) and 1 with the Y-axis (e.g. latitude). UGRID
    describing alternative node coordinates (e.g. spherical) cannot be
    represented.

    Notes
    -----
    The 3D layered and fully 3D unstructured mesh topologies are not supported
    at this time.

    References
    ----------
    .. [1] The UGRID Conventions, https://ugrid-conventions.github.io/ugrid-conventions/

    """

    # TBD: for volume and/or z-axis support include axis "z" and/or dimension "3"
    #: The supported mesh axes.
    AXES = ("x", "y")
    #: Valid range of values for ``topology_dimension``.
    TOPOLOGY_DIMENSIONS = (1, 2)
    #: Valid mesh elements.
    ELEMENTS = ("edge", "node", "face")

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
        """MeshXY initialise.

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
        def normalise(element, axis):
            result = str(axis).lower()
            if result not in self.AXES:
                emsg = f"Invalid axis specified for {element} coordinate {coord.name()!r}, got {axis!r}."
                raise ValueError(emsg)
            return f"{element}_{result}"

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
            emsg = "Require a node coordinate that is x-axis like to be provided."
            raise ValueError(emsg)
        if "node_y" not in kwargs:
            emsg = "Require a node coordinate that is y-axis like to be provided."
            raise ValueError(emsg)

        if self.topology_dimension == 1:
            self._coord_manager = _Mesh1DCoordinateManager(**kwargs)
            self._connectivity_manager = _Mesh1DConnectivityManager(*connectivities)
        elif self.topology_dimension == 2:
            self._coord_manager = _Mesh2DCoordinateManager(**kwargs)
            self._connectivity_manager = _Mesh2DConnectivityManager(*connectivities)
        else:
            emsg = f"Unsupported 'topology_dimension', got {topology_dimension!r}."
            raise NotImplementedError(emsg)

    @classmethod
    def from_coords(cls, *coords):
        r"""Construct a :class:`MeshXY` by derivation from 1/more :class:`~iris.coords.Coord`.

        The :attr:`~MeshXY.topology_dimension`, :class:`~iris.coords.Coord`
        membership and :class:`Connectivity` membership are all determined
        based on the shape of the first :attr:`~iris.coords.Coord.bounds`:

        * ``None`` or ``(n, <2)``:
            Not supported
        * ``(n, 2)``:
            :attr:`~MeshXY.topology_dimension` = ``1``.
            :attr:`~MeshXY.node_coords` and :attr:`~MeshXY.edge_node_connectivity`
            constructed from :attr:`~iris.coords.Coord.bounds`.
            :attr:`~MeshXY.edge_coords` constructed from
            :attr:`~iris.coords.Coord.points`.
        * ``(n, >=3)``:
            :attr:`~MeshXY.topology_dimension` = ``2``.
            :attr:`~MeshXY.node_coords` and :attr:`~MeshXY.face_node_connectivity`
            constructed from :attr:`~iris.coords.Coord.bounds`.
            :attr:`~MeshXY.face_coords` constructed from
            :attr:`~iris.coords.Coord.points`.

        Parameters
        ----------
        *coords : Iterable of :class:`~iris.coords.Coord`
            Coordinates to pass into the :class:`MeshXY`.
            All :attr:`~iris.coords.Coord.points` must have the same shapes;
            all :attr:`~iris.coords.Coord.bounds` must have the same shapes,
            and must not be ``None``.

        Returns
        -------
        :class:`MeshXY`

        Notes
        -----
        .. note::
            Any resulting duplicate nodes are not currently removed, due to the
            computational intensity.

        .. note::
            :class:`MeshXY` currently requires ``X`` and ``Y``
            :class:`~iris.coords.Coord` specifically.
            :meth:`iris.util.guess_coord_axis` is therefore attempted, else the
            first two :class:`~iris.coords.Coord` are taken.

        .. testsetup::

            from iris import load_cube, sample_data_path
            from iris.mesh import (
                MeshXY,
                MeshCoord,
            )

            file_path = sample_data_path("mesh_C4_synthetic_float.nc")
            cube_w_mesh = load_cube(file_path)

        Examples
        --------
        ::

            # Reconstruct a cube-with-mesh after subsetting it.

            >>> print(cube_w_mesh.mesh.name())
            Topology data of 2D unstructured mesh
            >>> mesh_coord_names = [
            ...     coord.name() for coord in cube_w_mesh.coords(mesh_coords=True)
            ... ]
            >>> print(f"MeshCoords: {mesh_coord_names}")
            MeshCoords: ['latitude', 'longitude']

            # Subsetting converts MeshCoords to AuxCoords.
            >>> slices = [slice(None)] * cube_w_mesh.ndim
            >>> slices[cube_w_mesh.mesh_dim()] = slice(-1)
            >>> cube_sub = cube_w_mesh[tuple(slices)]
            >>> print(cube_sub.mesh)
            None
            >>> orig_coords = [cube_sub.coord(c_name) for c_name in mesh_coord_names]
            >>> for coord in orig_coords:
            ...     print(f"{coord.name()}: {type(coord).__name__}")
            latitude: AuxCoord
            longitude: AuxCoord

            >>> new_mesh = MeshXY.from_coords(*orig_coords)
            >>> new_coords = new_mesh.to_MeshCoords(location=cube_w_mesh.location)

            # Replace the AuxCoords with MeshCoords.
            >>> for ix in range(2):
            ...     cube_sub.remove_coord(orig_coords[ix])
            ...     cube_sub.add_aux_coord(new_coords[ix], cube_w_mesh.mesh_dim())

            >>> print(cube_sub.mesh.name())
            Topology data of 2D unstructured mesh
            >>> for coord_name in mesh_coord_names:
            ...     coord = cube_sub.coord(coord_name)
            ...     print(f"{coord_name}: {type(coord).__name__}")
            latitude: MeshCoord
            longitude: MeshCoord

        """

        # Validate points and bounds shape match.
        def check_shape(array_name):
            attr_name = f"core_{array_name}"
            arrays = [getattr(coord, attr_name)() for coord in coords]
            if any(a is None for a in arrays):
                message = f"{array_name} missing from coords[{arrays.index(None)}] ."
                raise ValueError(message)
            shapes = [array.shape for array in arrays]
            if shapes.count(shapes[0]) != len(shapes):
                message = f"{array_name} shapes are not identical for all coords."
                raise ValueError(message)

        for array in ("points", "bounds"):
            check_shape(array)

        # Determine dimensionality, using first coord.
        first_coord = coords[0]

        ndim = first_coord.ndim
        if ndim != 1:
            message = f"Expected coordinate ndim == 1, got: f{ndim} ."
            raise ValueError(message)

        bounds_shape = first_coord.core_bounds().shape
        bounds_dim1 = bounds_shape[1]
        if bounds_dim1 < 2:
            message = (
                f"Expected coordinate bounds.shape (n, >=2), got: {bounds_shape} ."
            )
            raise ValueError(message)
        elif bounds_dim1 == 2:
            topology_dimension = 1
            coord_centring = "edge"
            conn_cf_role = "edge_node_connectivity"
        else:
            topology_dimension = 2
            coord_centring = "face"
            conn_cf_role = "face_node_connectivity"

        # Create connectivity.
        if first_coord.has_lazy_bounds():
            array_lib = da
        else:
            array_lib = np
        indices = array_lib.arange(np.prod(bounds_shape)).reshape(bounds_shape)
        masking = array_lib.ma.getmaskarray(first_coord.core_bounds())
        indices = array_lib.ma.masked_array(indices, masking)
        connectivity = Connectivity(indices, conn_cf_role)

        # Create coords.
        node_coords = []
        centre_coords = []
        for coord in coords:
            coord_kwargs = dict(
                standard_name=coord.standard_name,
                long_name=coord.long_name,
                units=coord.units,
                attributes=coord.attributes,
            )
            node_points = array_lib.ma.filled(coord.core_bounds(), 0.0).flatten()
            node_coords.append(AuxCoord(points=node_points, **coord_kwargs))

            centre_points = coord.core_points()
            centre_coords.append(AuxCoord(points=centre_points, **coord_kwargs))

        #####
        # TODO: remove axis assignment once Mesh supports arbitrary coords.
        # TODO: consider filtering coords as the first action in this method.
        axes_present = [guess_coord_axis(coord) for coord in coords]
        axes_required = ("X", "Y")
        if all([req in axes_present for req in axes_required]):
            axis_indices = [axes_present.index(req) for req in axes_required]
        else:
            message = (
                "Unable to find 'X' and 'Y' using guess_coord_axis. Assuming "
                "X=coords[0], Y=coords[1] ."
            )
            # TODO: reconsider logging level when we have consistent practice.
            logger.info(message, extra=dict(cls=None))
            axis_indices = range(len(axes_required))

        def axes_assign(coord_list):
            coords_sorted = [coord_list[ix] for ix in axis_indices]
            return zip(coords_sorted, axes_required)

        node_coords_and_axes = axes_assign(node_coords)
        centre_coords_and_axes = axes_assign(centre_coords)
        #####

        # Construct the Mesh.
        mesh_kwargs = dict(
            topology_dimension=topology_dimension,
            node_coords_and_axes=node_coords_and_axes,
            connectivities=[connectivity],
        )
        mesh_kwargs[f"{coord_centring}_coords_and_axes"] = centre_coords_and_axes
        return cls(**mesh_kwargs)

    def __eq__(self, other):
        result = NotImplemented

        if isinstance(other, MeshXY):
            result = self.metadata == other.metadata
            if result:
                result = self.all_coords == other.all_coords
            if result:
                result = self.all_connectivities == other.all_connectivities

        return result

    def __hash__(self):
        # Allow use in sets and as dictionary keys, as is done for :class:`iris.cube.Cube`.
        # See https://github.com/SciTools/iris/pull/1772
        return hash(id(self))

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

    def summary(self, shorten=False):
        """Return a string representation of the MeshXY.

        Parameters
        ----------
        shorten : bool, default=False
            If True, produce a oneline string form of the form <MeshXY: ...>.
            If False, produce a multi-line detailed print output.

        Returns
        -------
        str

        """
        if shorten:
            result = self._summary_oneline()
        else:
            result = self._summary_multiline()
        return result

    def __repr__(self):
        return self.summary(shorten=True)

    def __str__(self):
        return self.summary(shorten=False)

    def _summary_oneline(self):
        # We use the repr output to produce short one-line identity summary,
        # similar to the object.__str__ output "<object at xxx>".
        # This form also used in other str() constructions, like MeshCoord.
        # By contrast, __str__ (below) produces a readable multi-line printout.
        mesh_name = self.name()
        if mesh_name in (None, "", "unknown"):
            mesh_name = None
        if mesh_name:
            # Use a more human-readable form
            mesh_string = f"<MeshXY: '{mesh_name}'>"
        else:
            # Mimic the generic object.__str__ style.
            mesh_id = id(self)
            mesh_string = f"<MeshXY object at {hex(mesh_id)}>"

        return mesh_string

    def _summary_multiline(self):
        # Produce a readable multi-line summary of the Mesh content.
        lines = []
        n_indent = 4
        indent_str = " " * n_indent

        def line(text, i_indent=0):
            indent = indent_str * i_indent
            lines.append(f"{indent}{text}")

        line(f"MeshXY : '{self.name()}'")
        line(f"topology_dimension: {self.topology_dimension}", 1)
        for element in ("node", "edge", "face"):
            if element == "node":
                element_exists = True
            else:
                main_conn_name = f"{element}_node_connectivity"
                main_conn = getattr(self, main_conn_name, None)
                element_exists = main_conn is not None
            if element_exists:
                # Include a section for this element
                line(element, 1)
                # Print element dimension
                dim_name = f"{element}_dimension"
                dim = getattr(self, dim_name)
                line(f"{dim_name}: '{dim}'", 2)
                # Print defining connectivity (except node)
                if element != "node":
                    main_conn_string = main_conn.summary(shorten=True, linewidth=0)
                    line(f"{main_conn_name}: {main_conn_string}", 2)
                # Print coords
                coords = self.coords(location=element)
                if coords:
                    line(f"{element} coordinates", 2)
                    for coord in coords:
                        coord_string = coord.summary(shorten=True, linewidth=0)
                        line(coord_string, 3)

        # Having dealt with essential info, now add any optional connectivities
        # N.B. includes boundaries: as optional connectivity, not an "element"
        optional_conn_names = (
            "boundary_connectivity",
            "face_face_connectivity",
            "face_edge_connectivity",
            "edge_face_connectivity",
        )
        optional_conns = [getattr(self, name, None) for name in optional_conn_names]
        optional_conns = {
            name: conn
            for conn, name in zip(optional_conns, optional_conn_names)
            if conn is not None
        }
        if optional_conns:
            line("optional connectivities", 1)
            for name, conn in optional_conns.items():
                conn_string = conn.summary(shorten=True, linewidth=0)
                line(f"{name}: {conn_string}", 2)

        # Output the detail properties, basically those from CFVariableMixin
        for name in BaseMetadata._members:
            val = getattr(self, name, None)
            if val is not None:
                if name == "units":
                    show = val.origin != Unit(None)
                elif isinstance(val, Container):
                    show = bool(val)
                else:
                    show = val is not None
                if show:
                    if name == "attributes":
                        # Use a multi-line form for this.
                        line("attributes:", 1)
                        max_attname_len = max(len(attr) for attr in val.keys())
                        for attrname, attrval in val.items():
                            attrname = attrname.ljust(max_attname_len)
                            if isinstance(attrval, str):
                                # quote strings
                                attrval = repr(attrval)
                                # and abbreviate really long ones
                                attrval = clip_string(attrval)
                            attr_string = f"{attrname}  {attrval}"
                            line(attr_string, 2)
                    else:
                        line(f"{name}: {val!r}", 1)

        result = "\n".join(lines)
        return result

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
            node, edge, face = [None if arg else current for arg, current in zipped]
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
            message = f"Unsupported topology_dimension: {self.topology_dimension} ."
            raise NotImplementedError(message)

        return result

    @property
    def all_connectivities(self):
        """All the :class:`~iris.mesh.Connectivity` instances of the :class:`MeshXY`."""
        return self._connectivity_manager.all_members

    @property
    def all_coords(self):
        """All the :class:`~iris.coords.AuxCoord` coordinates of the :class:`MeshXY`."""
        return self._coord_manager.all_members

    @property
    def boundary_node_connectivity(self):
        """The *optional* UGRID ``boundary_node_connectivity`` :class:`~iris.mesh.Connectivity`.

        The *optional* UGRID ``boundary_node_connectivity``
        :class:`~iris.mesh.Connectivity` of the
        :class:`MeshXY`.

        """
        return self._connectivity_manager.boundary_node

    @property
    def edge_coords(self):
        """The *optional* UGRID ``edge`` :class:`~iris.coords.AuxCoord` coordinates of the :class:`MeshXY`."""
        return self._coord_manager.edge_coords

    @property
    def edge_dimension(self):
        """The *optionally required* UGRID NetCDF variable name for the ``edge`` dimension."""
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
        """The *optional* UGRID ``edge_face_connectivity`` :class:`~iris.mesh.Connectivity`.

        The *optional* UGRID ``edge_face_connectivity``
        :class:`~iris.mesh.Connectivity` of the
        :class:`MeshXY`.

        """
        return self._connectivity_manager.edge_face

    @property
    def edge_node_connectivity(self):
        """The UGRID ``edge_node_connectivity`` :class:`~iris.mesh.Connectivity`.

        The UGRID ``edge_node_connectivity``
        :class:`~iris.mesh.Connectivity` of the
        :class:`MeshXY`, which is **required** for :attr:`MeshXY.topology_dimension`
        of ``1``, and *optionally required* for
        :attr:`MeshXY.topology_dimension` ``>=2``.

        """
        return self._connectivity_manager.edge_node

    @property
    def face_coords(self):
        """The *optional* UGRID ``face`` :class:`~iris.coords.AuxCoord` coordinates of the :class:`MeshXY`."""
        return self._coord_manager.face_coords

    @property
    def face_dimension(self):
        """The *optional* UGRID NetCDF variable name for the ``face`` dimension."""
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
                logger.debug(message, extra=dict(cls=self.__class__.__name__))
        elif not name or not isinstance(name, str):
            face_dimension = f"Mesh{self.topology_dimension}d_face"
        else:
            face_dimension = name
        self._metadata_manager.face_dimension = face_dimension

    @property
    def face_edge_connectivity(self):
        """The *optional* UGRID ``face_edge_connectivity``:class:`~iris.mesh.Connectivity`.

        The *optional* UGRID ``face_edge_connectivity``
        :class:`~iris.mesh.Connectivity` of the
        :class:`MeshXY`.

        """
        # optional
        return self._connectivity_manager.face_edge

    @property
    def face_face_connectivity(self):
        """The *optional* UGRID ``face_face_connectivity`` :class:`~iris.mesh.Connectivity`.

        The *optional* UGRID ``face_face_connectivity``
        :class:`~iris.mesh.Connectivity` of the
        :class:`MeshXY`.

        """
        return self._connectivity_manager.face_face

    @property
    def face_node_connectivity(self):
        """Return ``face_node_connectivity``:class:`~iris.mesh.Connectivity`.

        The UGRID ``face_node_connectivity``
        :class:`~iris.mesh.Connectivity` of the
        :class:`MeshXY`, which is **required** for :attr:`MeshXY.topology_dimension`
        of ``2``, and *optionally required* for :attr:`MeshXY.topology_dimension`
        of ``3``.

        """
        return self._connectivity_manager.face_node

    @property
    def node_coords(self):
        """The **required** UGRID ``node`` :class:`~iris.coords.AuxCoord` coordinates of the :class:`MeshXY`."""
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
        """Add one or more :class:`~iris.mesh.Connectivity` instances to the :class:`MeshXY`.

        Parameters
        ----------
        *connectivities : iterable of object
            A collection of one or more
            :class:`~iris.mesh.Connectivity` instances to
            add to the :class:`MeshXY`.

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
        """Add one or more :class:`~iris.coords.AuxCoord` coordinates to the :class:`MeshXY`.

        Parameters
        ----------
        node_x : optional
            The ``x-axis`` like ``node`` :class:`~iris.coords.AuxCoord`.
        node_y : optional
            The ``y-axis`` like ``node`` :class:`~iris.coords.AuxCoord`.
        edge_x : optional
            The ``x-axis`` like ``edge`` :class:`~iris.coords.AuxCoord`.
        edge_y : optional
            The ``y-axis`` like ``edge`` :class:`~iris.coords.AuxCoord`.
        face_x : optional
            The ``x-axis`` like ``face`` :class:`~iris.coords.AuxCoord`.
        face_y : optional
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
        r"""Return all :class:`~iris.mesh.Connectivity`\s.

        Return all :class:`~iris.mesh.Connectivity`
        instances from the :class:`~iris.mesh.MeshXY` which match the provided criteria.

        Criteria can be either specific properties or other objects with
        metadata to be matched.

        .. seealso::

            :meth:`MeshXY.connectivity` for matching exactly one connectivity.

        Parameters
        ----------
        item : str or object
            Either,

            * a :attr:`~iris.common.mixin.CFVariableMixin.standard_name`,
              :attr:`~iris.common.mixin.CFVariableMixin.long_name`, or
              :attr:`~iris.common.mixin.CFVariableMixin.var_name` which is
              compared against the :meth:`~iris.common.mixin.CFVariableMixin.name`.

            * a connectivity or metadata instance equal to that of
              the desired objects e.g.,
              :class:`~iris.mesh.Connectivity` or
              :class:`~iris.common.metadata.ConnectivityMetadata`.
        standard_name : str, optional
            The CF standard name of the desired
            :class:`~iris.mesh.Connectivity`. If ``None``,
            does not check for ``standard_name``.
        long_name : str, optional
            An unconstrained description of the
            :class:`~iris.mesh.Connectivity`. If ``None``,
            does not check for ``long_name``.
        var_name : str, optional
            The NetCDF variable name of the desired
            :class:`~iris.mesh.Connectivity`. If ``None``,
            does not check for ``var_name``.
        attributes : dict, optional
            A dictionary of attributes desired on the
            :class:`~iris.mesh.Connectivity`. If ``None``,
            does not check for ``attributes``.
        cf_role : str, optional
            The UGRID ``cf_role`` of the desired
            :class:`~iris.mesh.Connectivity`.
        contains_node : bool, optional
            Contains the ``node`` element as part of the
            :attr:`~iris.common.metadata.ConnectivityMetadata.cf_role`
            in the list of objects to be matched.
        contains_edge : bool, optional
            Contains the ``edge`` element as part of the
            :attr:`~iris.common.metadata.ConnectivityMetadata.cf_role`
            in the list of objects to be matched.
        contains_face : bool, optional
            Contains the ``face`` element as part of the
            :attr:`~iris.common.metadata.ConnectivityMetadata.cf_role`
            in the list of objects to be matched.

        Returns
        -------
        list of :class:`~iris.mesh.Connectivity`
            A list of :class:`~iris.mesh.Connectivity`
            instances from the :class:`MeshXY` that matched the given criteria.

        """
        result = self._connectivity_manager.filters(
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
        return list(result.values())

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
        """Return a single :class:`~iris.mesh.Connectivity`.

        Return a single :class:`~iris.mesh.Connectivity`
        from the :class:`MeshXY` that matches the provided criteria.

        Criteria can be either specific properties or other objects with
        metadata to be matched.

        .. note::

            If the given criteria do not return **precisely one**
            :class:`~iris.mesh.Connectivity`, then a
            :class:`~iris.exceptions.ConnectivityNotFoundError` is raised.

        .. seealso::

            :meth:`MeshXY.connectivities` for matching zero or more connectivities.

        Parameters
        ----------
        item : str or object
            Either,

            * a :attr:`~iris.common.mixin.CFVariableMixin.standard_name`,
              :attr:`~iris.common.mixin.CFVariableMixin.long_name`, or
              :attr:`~iris.common.mixin.CFVariableMixin.var_name` which is
              compared against the :meth:`~iris.common.mixin.CFVariableMixin.name`.

            * a connectivity or metadata instance equal to that of
              the desired object e.g.,
              :class:`~iris.mesh.Connectivity` or
              :class:`~iris.common.metadata.ConnectivityMetadata`.
        standard_name : str, optional
            The CF standard name of the desired
            :class:`~iris.mesh.Connectivity`. If ``None``,
            does not check for ``standard_name``.
        long_name : str, optional
            An unconstrained description of the
            :class:`~iris.mesh.Connectivity`. If ``None``,
            does not check for ``long_name``.
        var_name : str, optional
            The NetCDF variable name of the desired
            :class:`~iris.mesh.Connectivity`. If ``None``,
            does not check for ``var_name``.
        attributes : dict, optional
            A dictionary of attributes desired on the
            :class:`~iris.mesh.Connectivity`. If ``None``,
            does not check for ``attributes``.
        cf_role : str, optional
            The UGRID ``cf_role`` of the desired
            :class:`~iris.mesh.Connectivity`.
        contains_node : bool, optional
            Contains the ``node`` element as part of the
            :attr:`~iris.common.metadata.ConnectivityMetadata.cf_role`
            in the list of objects to be matched.
        contains_edge : bool, optional
            Contains the ``edge`` element as part of the
            :attr:`~iris.common.metadata.ConnectivityMetadata.cf_role`
            in the list of objects to be matched.
        contains_face : bool, optional
            Contains the ``face`` element as part of the
            :attr:`~iris.common.metadata.ConnectivityMetadata.cf_role`
            in the list of objects to be matched.

        Returns
        -------
        :class:`~iris.mesh.Connectivity`
            The :class:`~iris.mesh.Connectivity` from the
            :class:`MeshXY` that matched the given criteria.

        """
        result = self._connectivity_manager.filter(
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
        return list(result.values())[0]

    def coord(
        self,
        item=None,
        standard_name=None,
        long_name=None,
        var_name=None,
        attributes=None,
        axis=None,
        location=None,
    ):
        """Return a single :class:`~iris.coords.AuxCoord` coordinate.

        Return a single :class:`~iris.coords.AuxCoord` coordinate from the
        :class:`MeshXY` that matches the provided criteria.

        Criteria can be either specific properties or other objects with
        metadata to be matched.

        .. note::

            If the given criteria do not return **precisely one** coordinate,
            then a :class:`~iris.exceptions.CoordinateNotFoundError` is raised.

        .. seealso::

            :meth:`MeshXY.coords` for matching zero or more coordinates.

        Parameters
        ----------
        item : str or object, optional
            Either,

            * a :attr:`~iris.common.mixin.CFVariableMixin.standard_name`,
              :attr:`~iris.common.mixin.CFVariableMixin.long_name`, or
              :attr:`~iris.common.mixin.CFVariableMixin.var_name` which is
              compared against the :meth:`~iris.common.mixin.CFVariableMixin.name`.

            * a coordinate or metadata instance equal to that of
              the desired coordinate e.g., :class:`~iris.coords.AuxCoord` or
              :class:`~iris.common.metadata.CoordMetadata`.
        standard_name : str, optional
            The CF standard name of the desired coordinate. If ``None``, does not
            check for ``standard_name``.
        long_name : str, optional
            An unconstrained description of the coordinate. If ``None``, does not
            check for ``long_name``.
        var_name : str, optional
            The NetCDF variable name of the desired coordinate. If ``None``, does
            not check for ``var_name``.
        attributes : dict, optional
            A dictionary of attributes desired on the coordinates. If ``None``,
            does not check for ``attributes``.
        axis : str, optional
            The desired coordinate axis, see :func:`~iris.util.guess_coord_axis`.
            If ``None``, does not check for ``axis``. Accepts the values ``X``,
            ``Y``, ``Z`` and ``T`` (case-insensitive).
        location : str, optional
            The desired location. Accepts the values ``node``, ``edge`` or ``face``.

        Returns
        -------
        :class:`~iris.coords.AuxCoord`
            The :class:`~iris.coords.AuxCoord` coordinate from the :class:`MeshXY`
            that matched the given criteria.

        """
        result = self._coord_manager.filter(
            item=item,
            standard_name=standard_name,
            long_name=long_name,
            var_name=var_name,
            attributes=attributes,
            axis=axis,
            location=location,
        )
        return list(result.values())[0]

    def coords(
        self,
        item=None,
        standard_name=None,
        long_name=None,
        var_name=None,
        attributes=None,
        axis=None,
        location=None,
    ):
        """Return all :class:`~iris.coords.AuxCoord` coordinates from the :class:`MeshXY`.

        Return all :class:`~iris.coords.AuxCoord` coordinates from the :class:`MeshXY`
        which match the provided criteria.

        Criteria can be either specific properties or other objects with
        metadata to be matched.

        .. seealso::

            :meth:`MeshXY.coord` for matching exactly one coordinate.

        Parameters
        ----------
        item : str or object, optional
            Either,

            * a :attr:`~iris.common.mixin.CFVariableMixin.standard_name`,
              :attr:`~iris.common.mixin.CFVariableMixin.long_name`, or
              :attr:`~iris.common.mixin.CFVariableMixin.var_name` which is
              compared against the :meth:`~iris.common.mixin.CFVariableMixin.name`.

            * a coordinate or metadata instance equal to that of
              the desired coordinates e.g., :class:`~iris.coords.AuxCoord` or
              :class:`~iris.common.metadata.CoordMetadata`.
        standard_name : str, optional
            The CF standard name of the desired coordinate. If ``None``, does not
            check for ``standard_name``.
        long_name : str, optional
            An unconstrained description of the coordinate. If ``None``, does not
            check for ``long_name``.
        var_name : str, optional
            The NetCDF variable name of the desired coordinate. If ``None``, does
            not check for ``var_name``.
        attributes : dict, optional
            A dictionary of attributes desired on the coordinates. If ``None``,
            does not check for ``attributes``.
        axis : str, optional
            The desired coordinate axis, see :func:`~iris.util.guess_coord_axis`.
            If ``None``, does not check for ``axis``. Accepts the values ``X``,
            ``Y``, ``Z`` and ``T`` (case-insensitive).
        location : str, optional
            The desired location. Accepts the values ``node``, ``edge`` or ``face``.

        Returns
        -------
        list of :class:`~iris.coords.AuxCoord`
            A list of :class:`~iris.coords.AuxCoord` coordinates from the
            :class:`MeshXY` that matched the given criteria.

        """
        result = self._coord_manager.filters(
            item=item,
            standard_name=standard_name,
            long_name=long_name,
            var_name=var_name,
            attributes=attributes,
            axis=axis,
            location=location,
        )
        return list(result.values())

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
        """Remove one or more :class:`~iris.mesh.Connectivity`.

        Remove one or more :class:`~iris.mesh.Connectivity`
        from the :class:`MeshXY` which match the provided criteria.

        Criteria can be either specific properties or other objects with
        metadata to be matched.

        Parameters
        ----------
        item : str or object, optional
            Either,

            * a :attr:`~iris.common.mixin.CFVariableMixin.standard_name`,
              :attr:`~iris.common.mixin.CFVariableMixin.long_name`, or
              :attr:`~iris.common.mixin.CFVariableMixin.var_name` which is
              compared against the :meth:`~iris.common.mixin.CFVariableMixin.name`.

            * a connectivity or metadata instance equal to that of
              the desired objects e.g.,
              :class:`~iris.mesh.Connectivity` or
              :class:`~iris.common.metadata.ConnectivityMetadata`.
        standard_name : str, optional
            The CF standard name of the desired
            :class:`~iris.mesh.Connectivity`. If ``None``,
            does not check for ``standard_name``.
        long_name : str, optional
            An unconstrained description of the
            :class:`~iris.mesh.Connectivity`. If ``None``,
            does not check for ``long_name``.
        var_name : str, optional
            The NetCDF variable name of the desired
            :class:`~iris.mesh.Connectivity`. If ``None``,
            does not check for ``var_name``.
        attributes : dict, optional
            A dictionary of attributes desired on the
            :class:`~iris.mesh.Connectivity`. If ``None``,
            does not check for ``attributes``.
        cf_role : str, optional
            The UGRID ``cf_role`` of the desired
            :class:`~iris.mesh.Connectivity`.
        contains_node : bool, optional
            Contains the ``node`` element as part of the
            :attr:`~iris.common.metadata.ConnectivityMetadata.cf_role`
            in the list of objects to be matched for potential removal.
        contains_edge : bool, optional
            Contains the ``edge`` element as part of the
            :attr:`~iris.common.metadata.ConnectivityMetadata.cf_role`
            in the list of objects to be matched for potential removal.
        contains_face : bool, optional
            Contains the ``face`` element as part of the
            :attr:`~iris.common.metadata.ConnectivityMetadata.cf_role`
            in the list of objects to be matched for potential removal.

        Returns
        -------
        list of :class:`~iris.mesh.Connectivity`
            A list of :class:`~iris.mesh.Connectivity`
            instances removed from the :class:`MeshXY` that matched the given
            criteria.

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
        location=None,
    ):
        """Remove one or more :class:`~iris.coords.AuxCoord` from the :class:`MeshXY`.

        Remove one or more :class:`~iris.coords.AuxCoord` from the :class:`MeshXY`
        which match the provided criteria.

        Criteria can be either specific properties or other objects with
        metadata to be matched.

        Parameters
        ----------
        item : str or object, optional
            Either,

            * a :attr:`~iris.common.mixin.CFVariableMixin.standard_name`,
              :attr:`~iris.common.mixin.CFVariableMixin.long_name`, or
              :attr:`~iris.common.mixin.CFVariableMixin.var_name` which is
              compared against the :meth:`~iris.common.mixin.CFVariableMixin.name`.

            * a coordinate or metadata instance equal to that of
              the desired coordinates e.g., :class:`~iris.coords.AuxCoord` or
              :class:`~iris.common.metadata.CoordMetadata`.
        standard_name : str, optional
            The CF standard name of the desired coordinate. If ``None``, does not
            check for ``standard_name``.
        long_name : str, optional
            An unconstrained description of the coordinate. If ``None``, does not
            check for ``long_name``.
        var_name : str, optional
            The NetCDF variable name of the desired coordinate. If ``None``, does
            not check for ``var_name``.
        attributes : dict, optional
            A dictionary of attributes desired on the coordinates. If ``None``,
            does not check for ``attributes``.
        axis : str, optional
            The desired coordinate axis, see :func:`~iris.util.guess_coord_axis`.
            If ``None``, does not check for ``axis``. Accepts the values ``X``,
            ``Y``, ``Z`` and ``T`` (case-insensitive).
        location : str, optional
            The desired location. Accepts the values ``node``, ``edge`` or ``face``.

        Returns
        -------
        list of :class:`~iris.coords.AuxCoord`
            A list of :class:`~iris.coords.AuxCoord` coordinates removed from
            the :class:`MeshXY` that matched the given criteria.

        """
        result = self._coord_manager.remove(
            item=item,
            standard_name=standard_name,
            long_name=long_name,
            var_name=var_name,
            attributes=attributes,
            axis=axis,
            location=location,
        )

        return result

    def xml_element(self, doc):
        """Create the :class:`xml.dom.minidom.Element` that describes this :class:`MeshXY`.

        Parameters
        ----------
        doc : object
            The parent :class:`xml.dom.minidom.Document`.

        Returns
        -------
        :class:`xml.dom.minidom.Element`
            The :class:`xml.dom.minidom.Element` that will describe this
            :class:`MeshXY`, and the dictionary of attributes that require
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

    def to_MeshCoord(self, location, axis):
        """Generate a :class:`~iris.mesh.MeshCoord`.

        Generate a :class:`~iris.mesh.MeshCoord` that
        references the current :class:`MeshXY`, and passing through the
        ``location`` and ``axis`` arguments.

        .. seealso::

            :meth:`to_MeshCoords` for generating a series of mesh coords.

        Parameters
        ----------
        location : str
            The ``location`` argument for
            :class:`~iris.mesh.MeshCoord` instantiation.
        axis : str
            The ``axis`` argument for
            :class:`~iris.mesh.MeshCoord` instantiation.

        Returns
        -------
        :class:`~iris.mesh.mesh.MeshCoord`
            A :class:`~iris.mesh.mesh.MeshCoord` referencing the
            current :class:`MeshXY`.

        """
        return MeshCoord(mesh=self, location=location, axis=axis)

    def to_MeshCoords(self, location):
        r"""Generate a tuple of :class:`~iris.mesh.mesh.MeshCoord`.

        Generate a tuple of
        :class:`~iris.mesh.mesh.MeshCoord`, each referencing
        the current :class:`MeshXY`, one for each :attr:`AXES` value, passing
        through the ``location`` argument.

        .. seealso::

            :meth:`to_MeshCoord` for generating a single mesh coord.

        Parameters
        ----------
        location : str
            The ``location`` argument for :class:`MeshCoord` instantiation.

        Returns
        -------
        tuple of :class:`~iris.mesh.mesh.MeshCoord`
            Tuple of :class:`~iris.mesh.mesh.MeshCoord`
            referencing the current :class:`MeshXY`. One for each value in
            :attr:`AXES`, using the value for the ``axis`` argument.

        """
        # factory method
        result = [self.to_MeshCoord(location=location, axis=ax) for ax in self.AXES]
        return tuple(result)

    def dimension_names_reset(self, node=False, edge=False, face=False):
        """Reset the name used for the NetCDF variable.

        Reset the name used for the NetCDF variable representing the ``node``,
        ``edge`` and/or ``face`` dimension to ``None``.

        Parameters
        ----------
        node : bool, optional, default=False
            Reset the name of the ``node`` dimension if ``True``. Default
            is ``False``.
        edge : bool, default=False
            Reset the name of the ``edge`` dimension if ``True``. Default
            is ``False``.
        face : bool, default=False
            Reset the name of the ``face`` dimension if ``True``. Default
            is ``False``.

        """
        return self._set_dimension_names(node, edge, face, reset=True)

    def dimension_names(self, node=None, edge=None, face=None):
        """Assign the name to be used for the NetCDF variable.

        Assign the name to be used for the NetCDF variable representing
        the ``node``, ``edge`` and ``face`` dimension.

        The default value of ``None`` will not be assigned to clear the
        associated ``node``, ``edge`` or ``face``. Instead use
        :meth:`MeshXY.dimension_names_reset`.

        Parameters
        ----------
        node : str, optional
            The name to be used for the NetCDF variable representing the
            ``node`` dimension.
        edge : str, optional
            The name to be used for the NetCDF variable representing the
            ``edge`` dimension.
        face : str, optional
            The name to be used for the NetCDF variable representing the
            ``face`` dimension.

        """
        return self._set_dimension_names(node, edge, face, reset=False)

    @property
    def cf_role(self):
        """The UGRID ``cf_role`` attribute of the :class:`MeshXY`."""
        return "mesh_topology"

    @property
    def topology_dimension(self):
        """UGRID ``topology_dimension`` attribute.

        The UGRID ``topology_dimension`` attribute represents the highest
        dimensionality of all the geometric elements (node, edge, face) represented
        within the :class:`MeshXY`.

        """
        return self._metadata_manager.topology_dimension


class _Mesh1DCoordinateManager:
    """TBD: require clarity on coord_systems validation.

    TBD: require clarity on __eq__ support
    TBD: rationalise self.coords() logic with other manager and Cube.

    """

    REQUIRED = (
        "node_x",
        "node_y",
    )
    OPTIONAL: tuple[str, ...] = (
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
        args = [f"{member}={coord!r}" for member, coord in self if coord is not None]
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

    def _setter(self, element, axis, coord, shape):
        axis = axis.lower()
        member = f"{element}_{axis}"

        # enforce the UGRID minimum coordinate requirement
        if element == "node" and coord is None:
            emsg = f"{member!r} is a required coordinate, cannot set to 'None'."
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
                emsg = (
                    f"{member!r} requires to have shape {shape!r}, got {coord.shape!r}."
                )
                raise ValueError(emsg)

        self._members[member] = coord

    def _shape(self, element):
        coord = getattr(self, f"{element}_x")
        shape = coord.shape if coord is not None else None
        if shape is None:
            coord = getattr(self, f"{element}_y")
            if coord is not None:
                shape = coord.shape
        return shape

    @property
    def _edge_shape(self):
        return self._shape(element="edge")

    @property
    def _node_shape(self):
        return self._shape(element="node")

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
        self._setter(element="edge", axis="x", coord=coord, shape=self._edge_shape)

    @property
    def edge_y(self):
        return self._members["edge_y"]

    @edge_y.setter
    def edge_y(self, coord):
        self._setter(element="edge", axis="y", coord=coord, shape=self._edge_shape)

    @property
    def node_coords(self):
        return MeshNodeCoords(node_x=self.node_x, node_y=self.node_y)

    @property
    def node_x(self):
        return self._members["node_x"]

    @node_x.setter
    def node_x(self, coord):
        self._setter(element="node", axis="x", coord=coord, shape=self._node_shape)

    @property
    def node_y(self):
        return self._members["node_y"]

    @node_y.setter
    def node_y(self, coord):
        self._setter(element="node", axis="y", coord=coord, shape=self._node_shape)

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
        """Use self.remove(edge_x=True) to remove a coordinate.

        Use self.remove(edge_x=True) to remove a coordinate e.g., using the
        pattern self.add(edge_x=None) will not remove the edge_x coordinate.

        """
        self._add(MeshNodeCoords(node_x, node_y))
        self._add(MeshEdgeCoords(edge_x, edge_y))

    def filter(self, **kwargs):
        # TODO: rationalise commonality with MeshConnectivityManager.filter and Cube.coord.
        result = self.filters(**kwargs)

        if len(result) > 1:
            names = ", ".join(f"{member}={coord!r}" for member, coord in result.items())
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
            emsg = f"Expected to find exactly 1 {name}coordinate, but found none."
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
        location=None,
    ):
        # TBD: support coord_systems?

        # Determine locations to include.
        if location is not None:
            if location not in ["node", "edge", "face"]:
                raise ValueError(
                    f"Expected location to be one of `node`, `edge` or `face`, got `{location}`"
                )
            include_nodes = location == "node"
            include_edges = location == "edge"
            include_faces = location == "face"
        else:
            include_nodes = include_edges = include_faces = True

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
        elif location == "face":
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
        result_dict = {k: v for k, v in self._members.items() if id(v) in result_ids}
        return result_dict

    def remove(
        self,
        item=None,
        standard_name=None,
        long_name=None,
        var_name=None,
        attributes=None,
        axis=None,
        location=None,
    ):
        return self._remove(
            item=item,
            standard_name=standard_name,
            long_name=long_name,
            var_name=var_name,
            attributes=attributes,
            axis=axis,
            location=location,
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
        return self._shape(element="face")

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
        self._setter(element="face", axis="x", coord=coord, shape=self._face_shape)

    @property
    def face_y(self):
        return self._members["face_y"]

    @face_y.setter
    def face_y(self, coord):
        self._setter(element="face", axis="y", coord=coord, shape=self._face_shape)

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
        location=None,
    ):
        return self._remove(
            item=item,
            standard_name=standard_name,
            long_name=long_name,
            var_name=var_name,
            attributes=attributes,
            axis=axis,
            location=location,
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
            f"{member}" for member, connectivity in self if connectivity is not None
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
        elements = set([c.location for c in proposed_members.values() if c is not None])
        for element in elements:
            counts = [
                len(c.indices_by_location(c.lazy_indices()))
                for c in proposed_members.values()
                if c is not None and c.location == element
            ]
            # Check is list values are identical.
            if not counts.count(counts[0]) == len(counts):
                message = (
                    f"Invalid Connectivities provided - inconsistent {element} counts."
                )
                raise ValueError(message)

        self._members = proposed_members

    def filter(self, **kwargs):
        # TODO: rationalise commonality with MeshCoordManager.filter and Cube.coord.
        result = self.filters(**kwargs)
        if len(result) > 1:
            names = ", ".join(
                f"{member}={connectivity!r}" for member, connectivity in result.items()
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
            bad_name = _name or kwargs["standard_name"] or kwargs["long_name"] or ""
            message = (
                f"Expected to find exactly 1 {bad_name} connectivity, but found none."
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
            members = [instance for instance in members if instance.cf_role == cf_role]

        def element_filter(instances, loc_arg, loc_name):
            if loc_arg is False:
                filtered = [
                    instance
                    for instance in instances
                    if loc_name
                    not in (
                        instance.location,
                        instance.connected,
                    )
                ]
            elif loc_arg is None:
                filtered = instances
            else:
                # Interpret any other value as =True.
                filtered = [
                    instance
                    for instance in instances
                    if loc_name in (instance.location, instance.connected)
                ]

            return filtered

        for arg, loc in (
            (contains_node, "node"),
            (contains_edge, "edge"),
            (contains_face, "face"),
        ):
            members = element_filter(members, arg, loc)

        # No need to actually modify filtering behaviour - already won't return
        # any face cf-roles if none are present.
        supports_faces = any(["face" in role for role in self.ALL])
        if contains_face and not supports_faces:
            message = "Ignoring request to filter for non-existent 'face' cf-roles."
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
        result_dict = {k: v for k, v in self._members.items() if id(v) in result_ids}
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


Location = Literal["edge", "node", "face"]


class MeshCoord(AuxCoord):
    """Geographic coordinate values of data on an unstructured mesh.

    A MeshCoord references a `~iris.mesh.mesh.MeshXY`.
    When contained in a `~iris.cube.Cube` it connects the cube to the Mesh.
    It records (a) which 1-D cube dimension represents the unstructured mesh,
    and (b) which  mesh 'location' the cube data is mapped to -- i.e. is it
    data on 'face's, 'edge's or 'node's.

    A MeshCoord also specifies its 'axis' : 'x' or 'y'.  Its values are then,
    accordingly, longitudes or latitudes.  The values are taken from the
    appropriate coordinates and connectivities in the MeshXY, determined by its
    'location' and 'axis'.

    Any cube with data on a mesh will have a MeshCoord for each axis,
    i.e. an 'X' and a 'Y'.

    The points and bounds contain coordinate values for the mesh elements,
    which depends on location.
    For 'node', the ``.points`` contains node locations.
    For 'edge', the ``.bounds`` contains edge endpoints, and the ``.points`` contain
    edge locations (typically centres), if the MeshXY contains them (optional).
    For 'face', the ``.bounds`` contain the face corners, and the ``.points`` contain the
    face locations (typically centres), if the MeshXY contains them (optional).

    .. note::
        As described above, it is possible for a MeshCoord to have bounds but
        no points.  This is not possible for a regular
        :class:`~iris.coords.AuxCoord` or :class:`~iris.coords.DimCoord`.

    .. note::
        A MeshCoord can not yet actually be created with bounds but no points.
        This is intended in future, but for now it raises an error.

    """

    def __init__(
        self,
        mesh: MeshXY,
        location: Location,
        axis: Literal["x", "y"],
    ):
        # Setup the metadata.
        self._metadata_manager = metadata_manager_factory(MeshCoordMetadata)

        # Validate and record the class-specific constructor args.
        if not isinstance(mesh, MeshXY):
            msg = (  # type: ignore[unreachable]
                f"'mesh' must be an {MeshXY.__module__}.{MeshXY.__name__}, got {mesh}."
            )
            raise TypeError(msg)
        # Handled as a readonly ".mesh" property.
        # NOTE: currently *not* included in metadata. In future it might be.
        self._mesh = mesh

        if location not in MeshXY.ELEMENTS:
            msg = (
                f"'location' of {location} is not a valid MeshXY location', "
                f"must be one of {MeshXY.ELEMENTS}."
            )
            raise ValueError(msg)
        # Held in metadata, readable as self.location, but cannot set it.
        self._metadata_manager.location = location

        if axis not in MeshXY.AXES:
            # The valid axes are defined by the MeshXY class.
            msg = (
                f"'axis' of {axis} is not a valid MeshXY axis', "
                f"must be one of {MeshXY.AXES}."
            )
            raise ValueError(msg)
        # Held in metadata, readable as self.axis, but cannot set it.
        self._metadata_manager.axis = axis

        points, bounds = self._construct_access_arrays()
        if points is None:
            # TODO: we intend to support this in future, but it will require
            #  extra work to refactor the parent classes.
            msg = "Cannot yet create a MeshCoord without points."
            raise ValueError(msg)

        # Get the 'coord identity' metadata from the relevant node-coordinate.
        node_coord = self.mesh.coord(location="node", axis=self.axis)
        node_metadict = node_coord.metadata._asdict()
        # Use node metadata, unless location is face/edge.
        use_metadict = node_metadict.copy()
        if location != "node":
            # Location is either "edge" or "face" - get the relevant coord.
            location_coord = self.mesh.coord(location=location, axis=axis)

            # Take the MeshCoord metadata from the 'location' coord.
            use_metadict = location_coord.metadata._asdict()
            unit_unknown = Unit(None)

            # N.B. at present, coords in a MeshXY are stored+accessed by 'axis', which
            # means they must have a standard_name.  So ...
            # (a) the 'location' (face/edge) coord *always* has a usable phenomenon
            #     identity.
            # (b) we still want to check that location+node coords have the same
            #     phenomenon (i.e. physical meaning identity + units), **but** ...
            # (c) we will accept/ignore some differences : not just "var_name", but
            #     also "long_name" *and* "attributes".  So it is *only* "standard_name"
            #     and "units" that cause an error if they differ.
            for key in ("standard_name", "units"):
                bounds_value = use_metadict[key]
                nodes_value = node_metadict[key]
                if key == "units" and (
                    bounds_value == unit_unknown or nodes_value == unit_unknown
                ):
                    # Allow "any" unit to match no-units (for now)
                    continue
                if bounds_value != nodes_value:

                    def fix_repr(val):
                        # Tidy values appearance by converting Unit to string, and
                        # wrapping strings in '', but leaving other types as a
                        # plain str() representation.
                        if isinstance(val, Unit):
                            val = str(val)
                        if isinstance(val, str):
                            val = repr(val)
                        return val

                    nodes_value, bounds_value = [
                        fix_repr(val) for val in (nodes_value, bounds_value)
                    ]
                    msg = (
                        f"Node coordinate {node_coord!r} disagrees with the "
                        f"{location} coordinate {location_coord!r}, "
                        f'in having a "{key}" value of {nodes_value} '
                        f"instead of {bounds_value}."
                    )
                    raise ValueError(msg)

        # Don't use 'coord_system' as a constructor arg, since for
        # MeshCoords it is deduced from the mesh.
        # (Otherwise a non-None coord_system breaks the 'copy' operation)
        use_metadict.pop("coord_system")

        # Call parent constructor to handle the common constructor args.
        super().__init__(points, bounds=bounds, **use_metadict)

    # Define accessors for MeshCoord-specific properties mesh/location/axis.
    # These are all read-only.

    @property
    def mesh(self):
        return self._mesh

    @property
    def location(self) -> Location:
        return self._metadata_manager.location

    @property
    def axis(self):
        return self._metadata_manager.axis

    # Provide overrides to mimic the Coord-specific properties that are not
    # supported by MeshCoord, i.e. "coord_system" and "climatological".
    # These mimic the Coord properties, but always return fixed 'null' values.
    # They can be set, to the 'null' value only, for the inherited init code.

    @property
    def coord_system(self):
        """The coordinate-system of a MeshCoord.

        It comes from the `related` location coordinate in the mesh.
        """
        # This matches where the coord metadata is drawn from.
        # See : https://github.com/SciTools/iris/issues/4860
        try:
            # NOTE: at present, a MeshCoord *always* references the relevant location
            # coordinate in the mesh, from which its points are taken.
            # However this might change in future ..
            # see : https://github.com/SciTools/iris/discussions/4438#bounds-no-points
            location_coord = self.mesh.coord(location=self.location, axis=self.axis)
            coord_system = location_coord.coord_system
        except CoordinateNotFoundError:
            # No such coord : possible in UGRID, but probably not Iris (at present).
            coord_system = None

        return coord_system

    @coord_system.setter
    def coord_system(self, value):
        if value is not None:
            msg = "Cannot set the coordinate-system of a MeshCoord."
            raise ValueError(msg)

    @property
    def climatological(self):
        """The 'climatological' of a MeshCoord is always 'False'."""
        return False

    @climatological.setter
    def climatological(self, value):
        if value:
            msg = "Cannot set 'climatological' on a MeshCoord."
            raise ValueError(msg)

    def __getitem__(self, keys):
        # Disallow any sub-indexing, permitting *only* "self[:,]".
        # We *don't* intend here to support indexing as such : the exception is
        # just sufficient to enable cube slicing, when it does not affect the
        # mesh dimension.  This works because Cube.__getitem__ passes us keys
        # "normalised" with iris.util._build_full_slice_given_keys.
        if keys != (slice(None),):
            msg = "Cannot index a MeshCoord."
            raise ValueError(msg)

        # Translate "self[:,]" as "self.copy()".
        return self.copy()

    def collapsed(self, dims_to_collapse=None):
        """Return a copy of this coordinate, which has been collapsed along the specified dimensions.

        Replaces the points & bounds with a simple bounded region.

        The coordinate that is collapsed is a :class:`~iris.coords.AuxCoord`
        copy of this :class:`MeshCoord`, since a :class:`MeshCoord`
        does not have its own points/bounds - they are derived from the
        associated :class:`MeshXY`. See :meth:`iris.coords.AuxCoord.collapsed`.
        """

        @contextmanager
        def temp_suppress_warning():
            """Add IrisVagueMetadataWarning filter then removes it after yielding.

            A workaround to mimic catch_warnings(), given python/cpython#73858.
            """
            warnings.filterwarnings("ignore", category=IrisVagueMetadataWarning)
            added_warning = warnings.filters[0]

            yield

            # (warnings.filters is usually mutable but this is not guaranteed).
            new_filters = list(warnings.filters)
            new_filters.remove(added_warning)
            warnings.filters = new_filters

        aux_coord = AuxCoord.from_coord(self)

        # Reuse existing AuxCoord collapse logic, but with a custom
        #  mesh-specific warning.
        message = (
            "Collapsing a mesh coordinate - cannot check for contiguity."
            f"Metadata may not be fully descriptive for '{self.name()}'."
        )
        warnings.warn(message, category=IrisVagueMetadataWarning)
        with temp_suppress_warning():
            collapsed_coord = aux_coord.collapsed(dims_to_collapse)

        return collapsed_coord

    def copy(self, points=None, bounds=None):
        """Make a copy of the MeshCoord.

        Parameters
        ----------
        points, bounds : array, optional
            Provided solely for signature compatibility with other types of
            :class:`~iris.coords.Coord`.
            In this case, if either is not 'None', an error is raised.

        """
        # Override Coord.copy, so that we can ensure it does not duplicate the
        # Mesh object (via deepcopy).
        # This avoids copying Meshes.

        # FOR NOW: also disallow changing points/bounds at all.
        if points is not None or bounds is not None:
            msg = "Cannot change the content of a MeshCoord."
            raise ValueError(msg)

        # Make a new MeshCoord with the same args :  The Mesh is the *same*
        # as the original (not a copy).
        new_coord = MeshCoord(mesh=self.mesh, location=self.location, axis=self.axis)
        return new_coord

    def __deepcopy__(self, memo):
        """Make this equivalent to "shallow" copy.

        Returns a new MeshCoord based on the same MeshXY.

        Required to prevent cube copying from copying the MeshXY, which would
        prevent "cube.copy() == cube" :  see notes for :meth:`copy`.

        """
        return self.copy()

    # Override _DimensionalMetadata.__eq__, to add 'mesh' comparison into the
    # default implementation (which compares metadata, points and bounds).
    # This is needed because 'mesh' is not included in our metadata.
    def __eq__(self, other):
        eq = NotImplemented
        if isinstance(other, MeshCoord):
            # *Don't* use the parent (_DimensionalMetadata) __eq__, as that
            # will try to compare points and bounds arrays.
            # Just compare the mesh, and the (other) metadata.
            eq = self.mesh == other.mesh  # N.B. 'mesh' not in metadata.
            if eq is not NotImplemented and eq:
                # Compare rest of metadata, but not points/bounds.
                eq = self.metadata == other.metadata

        return eq

    # Exactly as for Coord.__hash__ :  See there for why.
    def __hash__(self):
        return hash(id(self))

    def summary(self, *args, **kwargs):
        # We need to specialise _DimensionalMetadata.summary, so that we always
        # print the mesh+location of a MeshCoord.
        if len(args) > 0:
            shorten = args[0]
        else:
            shorten = kwargs.get("shorten", False)

        # Get the default-form result.
        if shorten:
            # NOTE: we simply aren't interested in the values for the repr,
            # so fix linewidth to suppress them
            kwargs["linewidth"] = 1

        # Plug private key, to get back the section structure info
        section_indices = {}
        kwargs["_section_indices"] = section_indices
        result = super().summary(*args, **kwargs)

        # Modify the generic 'default-form' result to produce what we want.
        if shorten:
            # Single-line form : insert mesh+location before the array part
            # Construct a text detailing the mesh + location
            mesh_string = self.mesh.name()
            if mesh_string == "unknown":
                # If no name, replace with the one-line summary
                mesh_string = self.mesh.summary(shorten=True)
            extra_str = f"mesh({mesh_string}) location({self.location})  "
            # find where in the line the data-array text begins
            i_line, i_array = section_indices["data"]
            assert i_line == 0
            # insert the extra text there
            result = result[:i_array] + extra_str + result[i_array:]
            # NOTE: this invalidates the original width calculation and may
            # easily extend the result beyond the intended maximum linewidth.
            # We do treat that as an advisory control over array printing, not
            # an absolute contract, so just ignore the problem for now.
        else:
            # Multiline form
            # find where the "location: ... " section is
            i_location, i_namestart = section_indices["location"]
            lines = result.split("\n")
            location_line = lines[i_location]
            # copy the indent spacing
            indent = location_line[:i_namestart]
            # use that to construct a suitable 'mesh' line
            mesh_string = self.mesh.summary(shorten=True)
            mesh_line = f"{indent}mesh: {mesh_string}"
            # Move the 'location' line, putting it and the 'mesh' line right at
            # the top, immediately after the header line.
            del lines[i_location]
            lines[1:1] = [mesh_line, location_line]
            # Re-join lines to give the result
            result = "\n".join(lines)
        return result

    def _construct_access_arrays(self):
        """Build lazy points and bounds arrays.

        Build lazy points and bounds arrays, providing dynamic access via the
        MeshXY, according to the location and axis.

        Returns
        -------
        array or None
            Tuple of (points, bounds).
            Lazy arrays which calculate the correct points and bounds from the
            MeshXY data, based on the location and axis.
            The MeshXY coordinates accessed are not identified on construction,
            but discovered from the MeshXY at the time of calculation, so that
            the result is always based on current content in the MeshXY.

        """
        mesh, location, axis = self.mesh, self.location, self.axis
        node_coord = mesh.coord(location="node", axis=axis)

        if location == "node":
            points_coord = node_coord
            bounds_connectivity = None
        else:
            points_coord = mesh.coord(location=location, axis=axis)
            bounds_connectivity = getattr(mesh, f"{location}_node_connectivity")

        # The points output is the points of the relevant element-type coord.
        points = points_coord.core_points()
        if bounds_connectivity is None:
            bounds = None
        else:
            # Bounds are calculated from a connectivity and the node points.
            # Data can be real or lazy, so operations must work in Dask, too.
            indices = bounds_connectivity.core_indices()
            # Normalise indices dimension order to [faces/edges, bounds]
            indices = bounds_connectivity.indices_by_location(indices)
            # Normalise the start index
            indices = indices - bounds_connectivity.start_index

            node_points = node_coord.core_points()
            n_nodes = node_points.shape[0]
            # Choose real/lazy array library, to suit array types.
            lazy = _lazy.is_lazy_data(indices) or _lazy.is_lazy_data(node_points)
            al = da if lazy else np
            # NOTE: Dask cannot index with a multidimensional array, so we
            # must flatten it and restore the shape later.
            flat_inds = indices.flatten()
            # NOTE: the connectivity array can have masked points, but we can't
            # effectively index with those.  So use a non-masked index array
            # with "safe" index values, and post-mask the results.
            flat_inds_nomask = al.ma.filled(flat_inds, -1)
            # Note: *also* mask any places where the index is out of range.
            missing_inds = (flat_inds_nomask < 0) | (flat_inds_nomask >= n_nodes)
            flat_inds_safe = al.where(missing_inds, 0, flat_inds_nomask)
            # Here's the core indexing operation.
            # The comma applies all inds-array values to the *first* dimension.
            bounds = node_points[flat_inds_safe,]
            # Fix 'missing' locations, and restore the proper shape.
            bounds = al.ma.masked_array(bounds, missing_inds)
            bounds = bounds.reshape(indices.shape)

        return points, bounds
