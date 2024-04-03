# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

r"""Allow the construction of :class:`~iris.experimental.ugrid.mesh.Mesh`.

Extensions to Iris' NetCDF loading to allow the construction of
:class:`~iris.experimental.ugrid.mesh.Mesh` from UGRID data in the file.

Eventual destination: :mod:`iris.fileformats.netcdf`.

"""

from contextlib import contextmanager
from itertools import groupby
from pathlib import Path
import threading
import warnings

from ...config import get_logger
from ...coords import AuxCoord
from ...fileformats._nc_load_rules.helpers import get_attr_units, get_names
from ...fileformats.netcdf import loader as nc_loader
from ...io import decode_uri, expand_filespecs
from ...util import guess_coord_axis
from ...warnings import IrisCfWarning, IrisDefaultingWarning, IrisIgnoringWarning
from .cf import (
    CFUGridAuxiliaryCoordinateVariable,
    CFUGridConnectivityVariable,
    CFUGridMeshVariable,
    CFUGridReader,
)
from .mesh import Connectivity, Mesh

# Configure the logger.
logger = get_logger(__name__, propagate=True, handler=False)


class _WarnComboCfDefaulting(IrisCfWarning, IrisDefaultingWarning):
    """One-off combination of warning classes - enhances user filtering."""

    pass


class _WarnComboCfDefaultingIgnoring(_WarnComboCfDefaulting, IrisIgnoringWarning):
    """One-off combination of warning classes - enhances user filtering."""

    pass


class ParseUGridOnLoad(threading.local):
    def __init__(self):
        """Thead-safe state to enable UGRID-aware NetCDF loading.

        A flag for dictating whether to use the experimental UGRID-aware
        version of Iris NetCDF loading. Object is thread-safe.

        Use via the run-time switch
        :const:`~iris.experimental.ugrid.load.PARSE_UGRID_ON_LOAD`.
        Use :meth:`context` to temporarily activate.

        .. seealso::

            The UGRID Conventions,
            https://ugrid-conventions.github.io/ugrid-conventions/

        """
        self._state = False

    def __bool__(self):
        return self._state

    @contextmanager
    def context(self):
        """Temporarily activate experimental UGRID-aware NetCDF loading.

        Use the standard Iris loading API while within the context manager. If
        the loaded file(s) include any UGRID content, this will be parsed and
        attached to the resultant cube(s) accordingly.

        Use via the run-time switch
        :const:`~iris.experimental.ugrid.load.PARSE_UGRID_ON_LOAD`.

        For example::

            with PARSE_UGRID_ON_LOAD.context():
                my_cube_list = iris.load([my_file_path, my_file_path2],
                                         constraint=my_constraint,
                                         callback=my_callback)

        """
        try:
            self._state = True
            yield
        finally:
            self._state = False


#: Run-time switch for experimental UGRID-aware NetCDF loading. See :class:`~iris.experimental.ugrid.load.ParseUGridOnLoad`.
PARSE_UGRID_ON_LOAD = ParseUGridOnLoad()


def _meshes_from_cf(cf_reader):
    """Mesh from cf, common behaviour for extracting meshes from a CFReader.

    Simple now, but expected to increase in complexity as Mesh sharing develops.

    """
    # Mesh instances are shared between file phenomena.
    # TODO: more sophisticated Mesh sharing between files.
    # TODO: access external Mesh cache?
    mesh_vars = cf_reader.cf_group.meshes
    meshes = {
        name: _build_mesh(cf_reader, var, cf_reader.filename)
        for name, var in mesh_vars.items()
    }
    return meshes


def load_mesh(uris, var_name=None):
    """Load a single :class:`~iris.experimental.ugrid.mesh.Mesh` object from one or more NetCDF files.

    Raises an error if more/less than one
    :class:`~iris.experimental.ugrid.mesh.Mesh` is found.

    Parameters
    ----------
    uris : str or iterable of str
        One or more filenames/URI's. Filenames can include wildcards. Any URI's
        must support OpenDAP.
    var_name : str, optional
        Only return a :class:`~iris.experimental.ugrid.mesh.Mesh` if its
        var_name matches this value.

    Returns
    -------
    :class:`iris.experimental.ugrid.mesh.Mesh`

    """
    meshes_result = load_meshes(uris, var_name)
    result = set([mesh for file in meshes_result.values() for mesh in file])
    mesh_count = len(result)
    if mesh_count != 1:
        message = f"Expecting 1 mesh, but input file(s) produced: {mesh_count} ."
        raise ValueError(message)
    return result.pop()  # Return the single element


def load_meshes(uris, var_name=None):
    r"""Load :class:`~iris.experimental.ugrid.mesh.Mesh` objects from one or more NetCDF files.

    Parameters
    ----------
    uris : str or iterable of str
        One or more filenames/URI's. Filenames can include wildcards. Any URI's
        must support OpenDAP.
    var_name : str, optional
        Only return :class:`~iris.experimental.ugrid.mesh.Mesh` that have
        var_names matching this value.

    Returns
    -------
    dict
        A dictionary mapping each mesh-containing file path/URL in the input
        ``uris`` to a list of the
        :class:`~iris.experimental.ugrid.mesh.Mesh` returned from each.

    """
    # TODO: rationalise UGRID/mesh handling once experimental.ugrid is folded
    #  into standard behaviour.
    # No constraints or callbacks supported - these assume they are operating
    #  on a Cube.

    from ...fileformats import FORMAT_AGENT

    if not PARSE_UGRID_ON_LOAD:
        # Explicit behaviour, consistent with netcdf.load_cubes(), rather than
        #  an invisible assumption.
        message = (
            f"PARSE_UGRID_ON_LOAD is {bool(PARSE_UGRID_ON_LOAD)}. Must be "
            f"True to enable mesh loading."
        )
        raise ValueError(message)

    if isinstance(uris, str):
        uris = [uris]

    # Group collections of uris by their iris handler
    # Create list of tuples relating schemes to part names.
    uri_tuples = sorted(decode_uri(uri) for uri in uris)

    valid_sources = []
    for scheme, groups in groupby(uri_tuples, key=lambda x: x[0]):
        # Call each scheme handler with the appropriate URIs
        if scheme == "file":
            filenames = [x[1] for x in groups]
            sources = expand_filespecs(filenames)
        elif scheme in ["http", "https"]:
            sources = [":".join(x) for x in groups]
        else:
            message = f"Iris cannot handle the URI scheme: {scheme}"
            raise ValueError(message)

        for source in sources:
            if scheme == "file":
                with open(source, "rb") as fh:
                    handling_format_spec = FORMAT_AGENT.get_spec(Path(source).name, fh)
            else:
                handling_format_spec = FORMAT_AGENT.get_spec(source, None)

            if handling_format_spec.handler == nc_loader.load_cubes:
                valid_sources.append(source)
            else:
                message = f"Ignoring non-NetCDF file: {source}"
                logger.info(msg=message, extra=dict(cls=None))

    result = {}
    for source in valid_sources:
        with CFUGridReader(source) as cf_reader:
            meshes_dict = _meshes_from_cf(cf_reader)
        meshes = list(meshes_dict.values())
        if var_name is not None:
            meshes = list(filter(lambda m: m.var_name == var_name, meshes))
        if meshes:
            result[source] = meshes

    return result


############
# Object construction.
# Helper functions, supporting netcdf.load_cubes ONLY, expected to
# altered/moved when pyke is removed.


def _build_aux_coord(coord_var, file_path):
    """Construct a :class:`~iris.coords.AuxCoord`.

    Construct a :class:`~iris.coords.AuxCoord` from a given
    :class:`~iris.experimental.ugrid.cf.CFUGridAuxiliaryCoordinateVariable`,
    and guess its mesh axis.

    todo: integrate with standard loading API post-pyke.

    """
    # TODO: integrate with standard saving API when no longer 'experimental'.
    assert isinstance(coord_var, CFUGridAuxiliaryCoordinateVariable)
    attributes = {}
    attr_units = get_attr_units(coord_var, attributes)
    points_data = nc_loader._get_cf_var_data(coord_var, file_path)

    # Bounds will not be loaded:
    # Bounds may be present, but the UGRID conventions state this would
    # always be duplication of the same info provided by the mandatory
    # connectivities.

    # Fetch climatological - not allowed for a Mesh, but loading it will
    # mean an informative error gets raised.
    climatological = False
    # TODO: use CF_ATTR_CLIMATOLOGY on re-integration, when no longer
    #  'experimental'.
    attr_climatology = getattr(coord_var, "climatology", None)
    if attr_climatology is not None:
        climatology_vars = coord_var.cf_group.climatology
        climatological = attr_climatology in climatology_vars

    standard_name, long_name, var_name = get_names(coord_var, None, attributes)
    coord = AuxCoord(
        points_data,
        standard_name=standard_name,
        long_name=long_name,
        var_name=var_name,
        units=attr_units,
        attributes=attributes,
        # TODO: coord_system
        climatological=climatological,
    )

    axis = guess_coord_axis(coord)
    if axis is None:
        if var_name[-2] == "_":
            # Fall back on UGRID var_name convention.
            axis = var_name[-1]
        else:
            message = f"Cannot guess axis for UGRID coord: {var_name} ."
            raise ValueError(message)

    return coord, axis


def _build_connectivity(connectivity_var, file_path, element_dims):
    """Construct a :class:`~iris.experimental.ugrid.mesh.Connectivity`.

    Construct a :class:`~iris.experimental.ugrid.mesh.Connectivity` from a
    given :class:`~iris.experimental.ugrid.cf.CFUGridConnectivityVariable`,
    and identify the name of its first dimension.

    todo: integrate with standard loading API post-pyke.

    """
    # TODO: integrate with standard saving API when no longer 'experimental'.
    assert isinstance(connectivity_var, CFUGridConnectivityVariable)
    attributes = {}
    attr_units = get_attr_units(connectivity_var, attributes)
    indices_data = nc_loader._get_cf_var_data(connectivity_var, file_path)

    cf_role = connectivity_var.cf_role
    start_index = connectivity_var.start_index

    dim_names = connectivity_var.dimensions
    # Connectivity arrays must have two dimensions.
    assert len(dim_names) == 2
    if dim_names[1] in element_dims:
        location_axis = 1
    else:
        location_axis = 0

    standard_name, long_name, var_name = get_names(connectivity_var, None, attributes)

    connectivity = Connectivity(
        indices=indices_data,
        cf_role=cf_role,
        standard_name=standard_name,
        long_name=long_name,
        var_name=var_name,
        units=attr_units,
        attributes=attributes,
        start_index=start_index,
        location_axis=location_axis,
    )

    return connectivity, dim_names[0]


def _build_mesh(cf, mesh_var, file_path):
    """Construct a :class:`~iris.experimental.ugrid.mesh.Mesh`.

    Construct a :class:`~iris.experimental.ugrid.mesh.Mesh` from a given
    :class:`~iris.experimental.ugrid.cf.CFUGridMeshVariable`.

    TODO: integrate with standard loading API post-pyke.

    """
    # TODO: integrate with standard saving API when no longer 'experimental'.
    assert isinstance(mesh_var, CFUGridMeshVariable)
    attributes = {}
    attr_units = get_attr_units(mesh_var, attributes)

    cf_role_message = None
    if not hasattr(mesh_var, "cf_role"):
        cf_role_message = f"{mesh_var.cf_name} has no cf_role attribute."
        cf_role = "mesh_topology"
    else:
        cf_role = getattr(mesh_var, "cf_role")
    if cf_role != "mesh_topology":
        cf_role_message = f"{mesh_var.cf_name} has an inappropriate cf_role: {cf_role}."
    if cf_role_message:
        cf_role_message += " Correcting to 'mesh_topology'."
        warnings.warn(
            cf_role_message,
            category=_WarnComboCfDefaulting,
        )

    if hasattr(mesh_var, "volume_node_connectivity"):
        topology_dimension = 3
    elif hasattr(mesh_var, "face_node_connectivity"):
        topology_dimension = 2
    elif hasattr(mesh_var, "edge_node_connectivity"):
        topology_dimension = 1
    else:
        # Nodes only.  We aren't sure yet whether this is a valid option.
        topology_dimension = 0

    if not hasattr(mesh_var, "topology_dimension"):
        msg = (
            f"Mesh variable {mesh_var.cf_name} has no 'topology_dimension'"
            f" : *Assuming* topology_dimension={topology_dimension}"
            ", consistent with the attached connectivities."
        )
        warnings.warn(msg, category=_WarnComboCfDefaulting)
    else:
        quoted_topology_dimension = mesh_var.topology_dimension
        if quoted_topology_dimension != topology_dimension:
            msg = (
                f"*Assuming* 'topology_dimension'={topology_dimension}"
                f", from the attached connectivities of the mesh variable "
                f"{mesh_var.cf_name}.  However, "
                f"{mesh_var.cf_name}:topology_dimension = "
                f"{quoted_topology_dimension}"
                " -- ignoring this as it is inconsistent."
            )
            warnings.warn(
                msg,
                category=_WarnComboCfDefaultingIgnoring,
            )

    node_dimension = None
    edge_dimension = getattr(mesh_var, "edge_dimension", None)
    face_dimension = getattr(mesh_var, "face_dimension", None)

    node_coord_args = []
    edge_coord_args = []
    face_coord_args = []
    for coord_var in mesh_var.cf_group.ugrid_coords.values():
        coord_and_axis = _build_aux_coord(coord_var, file_path)
        coord = coord_and_axis[0]

        if coord.var_name in mesh_var.node_coordinates.split():
            node_coord_args.append(coord_and_axis)
            node_dimension = coord_var.dimensions[0]
        elif coord.var_name in getattr(mesh_var, "edge_coordinates", "").split():
            edge_coord_args.append(coord_and_axis)
        elif coord.var_name in getattr(mesh_var, "face_coordinates", "").split():
            face_coord_args.append(coord_and_axis)
        # TODO: support volume_coordinates.
        else:
            message = (
                f"Invalid UGRID coord: {coord.var_name} . Must be either a"
                f"node_, edge_ or face_coordinate."
            )
            raise ValueError(message)

    if node_dimension is None:
        message = "'node_dimension' could not be identified from mesh node coordinates."
        raise ValueError(message)

    # Used for detecting transposed connectivities.
    element_dims = (edge_dimension, face_dimension)
    connectivity_args = []
    for connectivity_var in mesh_var.cf_group.connectivities.values():
        connectivity, first_dim_name = _build_connectivity(
            connectivity_var, file_path, element_dims
        )
        assert connectivity.var_name == getattr(mesh_var, connectivity.cf_role)
        connectivity_args.append(connectivity)

        # If the mesh_var has not supplied the dimension name, it is safe to
        # fall back on the connectivity's first dimension's name.
        if edge_dimension is None and connectivity.location == "edge":
            edge_dimension = first_dim_name
        if face_dimension is None and connectivity.location == "face":
            face_dimension = first_dim_name

    standard_name, long_name, var_name = get_names(mesh_var, None, attributes)

    mesh = Mesh(
        topology_dimension=topology_dimension,
        node_coords_and_axes=node_coord_args,
        connectivities=connectivity_args,
        edge_coords_and_axes=edge_coord_args,
        face_coords_and_axes=face_coord_args,
        standard_name=standard_name,
        long_name=long_name,
        var_name=var_name,
        units=attr_units,
        attributes=attributes,
        node_dimension=node_dimension,
        edge_dimension=edge_dimension,
        face_dimension=face_dimension,
    )

    mesh_elements = list(mesh.all_coords) + list(mesh.all_connectivities) + [mesh]
    mesh_elements = filter(None, mesh_elements)
    for iris_object in mesh_elements:
        nc_loader._add_unused_attributes(iris_object, cf.cf_group[iris_object.var_name])

    return mesh


def _build_mesh_coords(mesh, cf_var):
    """Construct a tuple of :class:`~iris.experimental.ugrid.mesh.MeshCoord`.

    Construct a tuple of :class:`~iris.experimental.ugrid.mesh.MeshCoord` using
    from a given :class:`~iris.experimental.ugrid.mesh.Mesh`
    and :class:`~iris.fileformats.cf.CFVariable`.

    TODO: integrate with standard loading API post-pyke.

    """
    # TODO: integrate with standard saving API when no longer 'experimental'.
    # Identify the cube's mesh dimension, for attaching MeshCoords.
    element_dimensions = {
        "node": mesh.node_dimension,
        "edge": mesh.edge_dimension,
        "face": mesh.face_dimension,
    }
    mesh_dim_name = element_dimensions[cf_var.location]
    # (Only expecting 1 mesh dimension per cf_var).
    mesh_dim = cf_var.dimensions.index(mesh_dim_name)

    mesh_coords = mesh.to_MeshCoords(location=cf_var.location)
    return mesh_coords, mesh_dim
