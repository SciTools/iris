# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

r"""Allow the construction of :class:`~iris.mesh.MeshXY`.

Extension functions for Iris NetCDF loading, to construct
:class:`~iris.mesh.MeshXY` from UGRID data in files.

.. seealso::

    The UGRID Conventions,
    https://ugrid-conventions.github.io/ugrid-conventions/

"""

from itertools import groupby
from pathlib import Path
import warnings

from iris.config import get_logger
from iris.coords import AuxCoord
from iris.io import decode_uri, expand_filespecs
from iris.mesh.components import Connectivity, MeshXY
from iris.util import guess_coord_axis
from iris.warnings import IrisCfWarning, IrisDefaultingWarning, IrisIgnoringWarning

# NOTE: all imports from iris.fileformats.netcdf must be deferred, to avoid circular
# imports.
# This is needed so that load_mesh/load_meshes can be included in the iris.mesh API.


# Configure the logger.
logger = get_logger(__name__, propagate=True, handler=False)


class _WarnComboCfDefaulting(IrisCfWarning, IrisDefaultingWarning):
    """One-off combination of warning classes - enhances user filtering."""

    pass


class _WarnComboCfDefaultingIgnoring(_WarnComboCfDefaulting, IrisIgnoringWarning):
    """One-off combination of warning classes - enhances user filtering."""

    pass


def _meshes_from_cf(cf_reader):
    """Mesh from cf, common behaviour for extracting meshes from a CFReader.

    Simple now, but expected to increase in complexity as Mesh sharing develops.

    """
    # Mesh instances are shared between file phenomena.
    # TODO: more sophisticated Mesh sharing between files.
    # TODO: access external Mesh cache?
    meshes = {}
    if cf_reader._with_ugrid:
        mesh_vars = cf_reader.cf_group.meshes
        meshes = {
            name: _build_mesh(cf_reader, var, cf_reader.filename)
            for name, var in mesh_vars.items()
        }
    return meshes


def load_mesh(uris, var_name=None):
    """Load a single :class:`~iris.mesh.MeshXY` object from one or more NetCDF files.

    Raises an error if more/less than one
    :class:`~iris.mesh.MeshXY` is found.

    Parameters
    ----------
    uris : str or iterable of str
        One or more filenames/URI's. Filenames can include wildcards. Any URI's
        must support OpenDAP.
    var_name : str, optional
        Only return a :class:`~iris.mesh.MeshXY` if its
        var_name matches this value.

    Returns
    -------
    :class:`iris.mesh.MeshXY`

    """
    meshes_result = load_meshes(uris, var_name)
    result = set([mesh for file in meshes_result.values() for mesh in file])
    mesh_count = len(result)
    if mesh_count != 1:
        message = f"Expecting 1 mesh, but input file(s) produced: {mesh_count} ."
        raise ValueError(message)
    return result.pop()  # Return the single element


def load_meshes(uris, var_name=None):
    r"""Load :class:`~iris.mesh.MeshXY` objects from one or more NetCDF files.

    Parameters
    ----------
    uris : str or iterable of str
        One or more filenames/URI's. Filenames can include wildcards. Any URI's
        must support OpenDAP.
    var_name : str, optional
        Only return :class:`~iris.mesh.MeshXY` that have
        var_names matching this value.

    Returns
    -------
    dict
        A dictionary mapping each mesh-containing file path/URL in the input
        ``uris`` to a list of the
        :class:`~iris.mesh.MeshXY` returned from each.

    """
    # NOTE: no constraints or callbacks supported - these assume they are operating
    #  on a Cube.
    # NOTE: dynamic imports avoid circularity : see note with module imports
    from iris.fileformats import FORMAT_AGENT
    from iris.fileformats.cf import CFReader
    import iris.fileformats.netcdf.loader as nc_loader

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
        with CFReader(source) as cf_reader:
            meshes_dict = _meshes_from_cf(cf_reader)
        meshes = list(meshes_dict.values())
        if var_name is not None:
            meshes = list(filter(lambda m: m.var_name == var_name, meshes))
        if meshes:
            result[source] = meshes

    return result


def _build_aux_coord(coord_var, file_path):
    """Construct a :class:`~iris.coords.AuxCoord`.

    Construct a :class:`~iris.coords.AuxCoord` from a given
    :class:`~iris.fileformats.cf.CFUGridAuxiliaryCoordinateVariable`,
    and guess its mesh axis.

    """
    # NOTE: dynamic imports avoid circularity : see note with module imports
    from iris.fileformats._nc_load_rules.helpers import get_attr_units, get_names
    from iris.fileformats.cf import CFUGridAuxiliaryCoordinateVariable
    from iris.fileformats.netcdf import loader as nc_loader

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
    """Construct a :class:`~iris.mesh.Connectivity`.

    Construct a :class:`~iris.mesh.Connectivity` from a
    given :class:`~iris.fileformats.cf.CFUGridConnectivityVariable`,
    and identify the name of its first dimension.

    """
    # NOTE: dynamic imports avoid circularity : see note with module imports
    from iris.fileformats._nc_load_rules.helpers import get_attr_units, get_names
    from iris.fileformats.cf import CFUGridConnectivityVariable
    from iris.fileformats.netcdf import loader as nc_loader

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
    """Construct a :class:`~iris.mesh.MeshXY`.

    Construct a :class:`~iris.mesh.MeshXY` from a given
    :class:`~iris.fileformats.cf.CFUGridMeshVariable`.

    """
    # NOTE: dynamic imports avoid circularity : see note with module imports
    from iris.fileformats._nc_load_rules.helpers import get_attr_units, get_names
    from iris.fileformats.cf import CFUGridMeshVariable
    from iris.fileformats.netcdf import loader as nc_loader

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
            f"MeshXY variable {mesh_var.cf_name} has no 'topology_dimension'"
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

    mesh = MeshXY(
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
    """Construct a tuple of :class:`~iris.mesh.MeshCoord`.

    Construct a tuple of :class:`~iris.mesh.MeshCoord` using
    from a given :class:`~iris.mesh.MeshXY`
    and :class:`~iris.fileformats.cf.CFVariable`.

    """
    # Identify the cube's mesh dimension, for attaching MeshCoords.
    element_dimensions = {
        "node": mesh.node_dimension,
        "edge": mesh.edge_dimension,
        "face": mesh.face_dimension,
    }
    location = getattr(cf_var, "location", "<empty>")
    if location is None or location not in element_dimensions:
        # We should probably issue warnings and recover, but that is too much
        # work.  Raising a more intelligible error is easy to do though.
        msg = (
            f"mesh data variable {cf_var.name!r} has an invalid "
            f"location={location!r}."
        )
        raise ValueError(msg)
    mesh_dim_name = element_dimensions.get(location)
    if mesh_dim_name is None:
        msg = f"mesh {mesh.name!r} has no {location} dimension."
        raise ValueError(msg)
    if mesh_dim_name in cf_var.dimensions:
        mesh_dim = cf_var.dimensions.index(mesh_dim_name)
    else:
        msg = (
            f"mesh data variable {cf_var.name!r} does not have the "
            f"{location} mesh dimension {mesh_dim_name!r}, in its dimensions."
        )
        raise ValueError(msg)

    mesh_coords = mesh.to_MeshCoords(location=cf_var.location)
    return mesh_coords, mesh_dim
