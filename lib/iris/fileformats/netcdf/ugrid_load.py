# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

r"""Allow the construction of :class:`~iris.mesh.MeshXY`.

.. z_reference:: iris.fileformats.netcdf.ugrid_load
   :tags: topic_load_save;topic_mesh

   API reference

Extension functions for Iris NetCDF loading, to construct
:class:`~iris.mesh.MeshXY` from UGRID data in files.

.. seealso::

    The UGRID Conventions,
    https://ugrid-conventions.github.io/ugrid-conventions/

"""

from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from itertools import groupby
from pathlib import Path
from typing import Iterable
import warnings

from iris.common.mixin import CFVariableMixin
from iris.config import get_logger
from iris.coords import AuxCoord
from iris.io import (
    _REMOTE_STORE_SCHEMES,
    _categorise_uri,
    expand_filespecs,
)
from iris.loading import LoadProblems
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
    from iris.fileformats._nc_load_rules.helpers import _add_or_capture

    meshes = {}

    if cf_reader._with_ugrid:
        mesh_vars = cf_reader.cf_group.meshes
        for name, var in mesh_vars.items():
            _ = _add_or_capture(
                build_func=partial(_build_mesh, cf_reader, var),
                add_method=partial(meshes.__setitem__, name),
                cf_var=var,
                # Meshes are 'linked' to zero or more Cubes; they do not have
                #  a destination in the same way as other Cube components.
                destination=LoadProblems.Problem.Destination(
                    iris_class=CFVariableMixin,
                    identifier="NOT_APPLICABLE",
                ),
            )

    return meshes


@dataclass
class _PendingMeshCoords:
    """MeshCoords built ahead of their cube, waiting to be attached to it."""

    coords: list
    dim: int | None
    capture_kwargs: dict


def _prepare_mesh_coords(cf_reader, meshes, cf_var):
    """Build the MeshCoords a data variable needs, before its cube exists.

    Unconventional 'split' usage of ``_add_or_capture``: this half builds and
    captures, :func:`_attach_mesh_coords` adds. The split is forced by
    attribute handling - built after the cube,
    ``helpers._add_unused_attributes`` has already taken the variable's
    ``mesh`` and ``location`` attributes for cube attributes.

    Parameters
    ----------
    cf_reader : iris.fileformats.cf.CFReader
        The reader the mesh variables were found in.
    meshes : dict
        Meshes by name, as returned by :func:`_meshes_from_cf`.
    cf_var : iris.fileformats.cf.CFDataVariable
        The data variable about to be built into a cube.

    Returns
    -------
    _PendingMeshCoords or None
        None when `cf_var` names no mesh, or names one this dataset does not
        hold.

    """
    # NOTE: dynamic imports avoid circularity : see note with module imports
    from iris.cube import Cube
    from iris.fileformats._nc_load_rules.helpers import _add_or_capture

    # Not this module's `logger`, which sits at INFO and would drop a debug
    #  message at source. `TestsMesh.test_missing_mesh` asserts the package
    #  logger, where this message has always been raised.
    from iris.fileformats.netcdf import logger as netcdf_logger

    mesh_name = cf_var.attributes.get("mesh")
    if mesh_name is None:
        return None

    mesh = meshes.get(mesh_name)
    if mesh is None:
        message = (
            f"Mesh '{mesh_name}' - "
            f"referenced by variable: '{cf_var.cf_name}' - "
            "could not be found in file."
        )
        netcdf_logger.debug(message)
        return None

    pending = _PendingMeshCoords(
        coords=[],
        dim=None,
        capture_kwargs=dict(
            cf_var=cf_reader.cf_group.meshes[mesh_name],
            # MeshCoords are an Iris concept; the best fallback we have is to
            #  capture the CF Mesh.
            destination=LoadProblems.Problem.Destination(
                iris_class=Cube,
                identifier=cf_var.cf_name,
            ),
        ),
    )

    def _stash(coords_and_dim):
        pending.coords, pending.dim = coords_and_dim

    _ = _add_or_capture(
        build_func=partial(_build_mesh_coords, mesh, cf_var),
        add_method=partial(_stash),
        **pending.capture_kwargs,
    )
    return pending


def _attach_mesh_coords(pending, cube):
    """Attach MeshCoords prepared by :func:`_prepare_mesh_coords` to their cube.

    Parameters
    ----------
    pending : _PendingMeshCoords
        The return of :func:`_prepare_mesh_coords`. Its `coords` are empty if
        the build failed, in which case the failure is already recorded and
        this call adds nothing.
    cube : iris.cube.Cube
        The cube built from the data variable the MeshCoords were prepared
        for.

    """
    # NOTE: dynamic imports avoid circularity : see note with module imports
    from iris.fileformats._nc_load_rules.helpers import _add_or_capture

    def _add(coords_and_dim):
        coords, dim = coords_and_dim
        for coord in coords:
            cube.add_aux_coord(coord, dim)

    _ = _add_or_capture(
        build_func=partial(lambda: (pending.coords, pending.dim)),
        add_method=partial(_add),
        **pending.capture_kwargs,
    )


def load_mesh(uris, var_name=None):
    """Load a single :class:`~iris.mesh.MeshXY` from one or more files or stores.

    Raises an error if more/less than one
    :class:`~iris.mesh.MeshXY` is found.

    Parameters
    ----------
    uris : str or iterable of str
        One or more filenames/URI's. Filenames can include wildcards. NetCDF
        files and Zarr stores are read; anything else is ignored. Any URI's
        must support OpenDAP, NcZarr, or a Zarr remote store scheme.
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
    r"""Load :class:`~iris.mesh.MeshXY` objects from files or Zarr stores.

    Parameters
    ----------
    uris : str or iterable of str
        One or more filenames/URI's. Filenames can include wildcards. NetCDF
        files and Zarr stores are read; anything else is ignored. Any URI's
        must support OpenDAP, NcZarr, or a Zarr remote store scheme.
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
    from iris.fileformats import FORMAT_AGENT, _load_zarr
    from iris.fileformats.cf import CFReader
    import iris.fileformats.netcdf.loader as nc_loader

    if (
        isinstance(uris, str)
        or hasattr(uris, "fromcdl")
        or not isinstance(uris, Iterable)
    ):
        # Make a string, Dataset, or other single item, into an iterable.
        uris = [uris]

    # Group collections of uris by their iris handler
    # Create list of tuples relating schemes to part names.
    uri_tuples = sorted(_categorise_uri(uri) for uri in uris)
    valid_sources = []
    for category, groups in groupby(uri_tuples, key=lambda x: x[0]):
        # Call each scheme handler with the appropriate URIs
        if category == "file":
            filenames = [part for _cat, _scheme, part in groups]
            sources = expand_filespecs(filenames)
        elif category in {"http", "https", "nczarr"} | _REMOTE_STORE_SCHEMES:
            sources = [f"{scheme}:{part}" for _cat, scheme, part in groups]
        else:
            message = f"Iris cannot handle the URI scheme: {category}"
            raise ValueError(message)

        for source in sources:
            # iris.loading._generate_cubes makes this same file-or-store
            # distinction independently, for general cube loading - keep
            # both in step.
            if category == "file" and not Path(source).is_dir():
                with open(source, "rb") as fh:
                    handling_format_spec = FORMAT_AGENT.get_spec(Path(source).name, fh)
            else:
                handling_format_spec = FORMAT_AGENT.get_spec(source, None)

            # TODO: drop this format dispatch once the netCDF and Zarr
            #  cube-building loops are unified. This copy, not the mesh code
            #  below, is the only reason a netCDF-named module names Zarr at
            #  all. See the merge-back section of
            #  `docs/superpowers/specs/2026-09-21-zarr-io-design.md`.
            #
            # `_load_zarr` is compared, never `zarr.load_cubes`: a store's
            #  presence is what should reach zarr-python, not this comparison.
            if handling_format_spec.handler == nc_loader.load_cubes:
                valid_sources.append((source, False))
            elif handling_format_spec.handler == _load_zarr:
                valid_sources.append((source, True))
            else:
                message = f"Ignoring file of a format that holds no mesh: {source}"
                logger.info(msg=message, extra=dict(cls=None))

    result = {}
    for source, is_zarr in valid_sources:
        with ExitStack() as stack:
            if is_zarr:
                # Deferred: a caller that named no Zarr store must not reach
                #  zarr-python at all.
                from iris.fileformats.zarr._dataset import ZarrDataset

                # CFReader takes an open CFDataset or a filename, so the mesh
                #  machinery below never learns which it was given.
                target = stack.enter_context(ZarrDataset(source))
            else:
                target = source
            cf_reader = stack.enter_context(CFReader(target))
            meshes_dict = _meshes_from_cf(cf_reader)
        meshes = list(meshes_dict.values())
        if var_name is not None:
            meshes = list(filter(lambda m: m.var_name == var_name, meshes))
        if meshes:
            result[source] = meshes

    return result


def _build_aux_coord(coord_var):
    """Construct a :class:`~iris.coords.AuxCoord`.

    Construct a :class:`~iris.coords.AuxCoord` from a given
    :class:`~iris.fileformats.cf.CFUGridAuxiliaryCoordinateVariable`,
    and guess its mesh axis.

    """
    # NOTE: dynamic imports avoid circularity : see note with module imports
    from iris.fileformats._nc_load_rules.helpers import get_attr_units, get_names
    from iris.fileformats.cf import CFUGridAuxiliaryCoordinateVariable
    from iris.fileformats.cf import loader as cf_loader

    assert isinstance(coord_var, CFUGridAuxiliaryCoordinateVariable)
    attributes = {}
    attr_units = get_attr_units(coord_var, attributes)
    points_data = cf_loader._get_cf_var_data(coord_var)

    # Bounds will not be loaded:
    # Bounds may be present, but the UGRID conventions state this would
    # always be duplication of the same info provided by the mandatory
    # connectivities.

    # Fetch climatological - not allowed for a Mesh, but loading it will
    # mean an informative error gets raised.
    climatological = False
    # TODO: use CF_ATTR_CLIMATOLOGY on re-integration, when no longer
    #  'experimental'.
    attr_climatology = coord_var.attributes.get("climatology")
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


def _build_connectivity(connectivity_var, element_dims):
    """Construct a :class:`~iris.mesh.Connectivity`.

    Construct a :class:`~iris.mesh.Connectivity` from a
    given :class:`~iris.fileformats.cf.CFUGridConnectivityVariable`,
    and identify the name of its first dimension.

    """
    # NOTE: dynamic imports avoid circularity : see note with module imports
    from iris.fileformats._nc_load_rules.helpers import get_attr_units, get_names
    from iris.fileformats.cf import CFUGridConnectivityVariable
    from iris.fileformats.cf import loader as cf_loader

    assert isinstance(connectivity_var, CFUGridConnectivityVariable)
    attributes = {}
    attr_units = get_attr_units(connectivity_var, attributes)
    indices_data = cf_loader._get_cf_var_data(connectivity_var)

    # UGRID requires both of these on a connectivity variable and Iris
    # validates neither. CFUGridConnectivityVariable.identify finds this
    # variable through the mesh variable's role attribute, never through
    # anything on the variable itself, so neither name is guaranteed to be
    # here. A file omitting either raises, and the mesh build is captured into
    # LOAD_PROBLEMS - the cube then loads without a mesh. SciTools/iris#7332.
    cf_role = connectivity_var.attributes["cf_role"]
    start_index = connectivity_var.attributes["start_index"]

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


def _build_mesh(cf, mesh_var):
    """Construct a :class:`~iris.mesh.MeshXY`.

    Construct a :class:`~iris.mesh.MeshXY` from a given
    :class:`~iris.fileformats.cf.CFUGridMeshVariable`.

    """
    # NOTE: dynamic imports avoid circularity : see note with module imports
    from iris.fileformats._nc_load_rules.helpers import get_attr_units, get_names
    from iris.fileformats.cf import CFUGridMeshVariable
    from iris.fileformats.cf import loader as cf_loader

    assert isinstance(mesh_var, CFUGridMeshVariable)
    attributes = {}
    attr_units = get_attr_units(mesh_var, attributes)

    cf_role_message = None
    if "cf_role" not in mesh_var.attributes:
        cf_role_message = f"{mesh_var.cf_name} has no cf_role attribute."
        cf_role = "mesh_topology"
    else:
        cf_role = mesh_var.attributes["cf_role"]
    if cf_role != "mesh_topology":
        cf_role_message = f"{mesh_var.cf_name} has an inappropriate cf_role: {cf_role}."
    if cf_role_message:
        cf_role_message += " Correcting to 'mesh_topology'."
        warnings.warn(
            cf_role_message,
            category=_WarnComboCfDefaulting,
        )

    if "volume_node_connectivity" in mesh_var.attributes:
        topology_dimension = 3
    elif "face_node_connectivity" in mesh_var.attributes:
        topology_dimension = 2
    elif "edge_node_connectivity" in mesh_var.attributes:
        topology_dimension = 1
    else:
        # Nodes only.  We aren't sure yet whether this is a valid option.
        topology_dimension = 0

    if "topology_dimension" not in mesh_var.attributes:
        msg = (
            f"MeshXY variable {mesh_var.cf_name} has no 'topology_dimension'"
            f" : *Assuming* topology_dimension={topology_dimension}"
            ", consistent with the attached connectivities."
        )
        warnings.warn(msg, category=_WarnComboCfDefaulting)
    else:
        quoted_topology_dimension = mesh_var.attributes["topology_dimension"]
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
    edge_dimension = mesh_var.attributes.get("edge_dimension")
    face_dimension = mesh_var.attributes.get("face_dimension")

    node_coord_args = []
    edge_coord_args = []
    face_coord_args = []
    for coord_var in mesh_var.cf_group.ugrid_coords.values():
        coord_and_axis = _build_aux_coord(coord_var)
        coord = coord_and_axis[0]

        if coord.var_name in mesh_var.attributes["node_coordinates"].split():
            node_coord_args.append(coord_and_axis)
            node_dimension = coord_var.dimensions[0]
        elif coord.var_name in mesh_var.attributes.get("edge_coordinates", "").split():
            edge_coord_args.append(coord_and_axis)
        elif coord.var_name in mesh_var.attributes.get("face_coordinates", "").split():
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
            connectivity_var, element_dims
        )
        assert connectivity.var_name == mesh_var.attributes[connectivity.cf_role]
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
        cf_loader._add_unused_attributes(iris_object, cf.cf_group[iris_object.var_name])

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
    location = cf_var.attributes.get("location", "<empty>")
    if location is None or location not in element_dimensions:
        # We should probably issue warnings and recover, but that is too much
        # work.  Raising a more intelligible error is easy to do though.
        msg = (
            f"mesh data variable {cf_var.cf_name!r} has an invalid "
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
            f"mesh data variable {cf_var.cf_name!r} does not have the "
            f"{location} mesh dimension {mesh_dim_name!r}, in its dimensions."
        )
        raise ValueError(msg)

    mesh_coords = mesh.to_MeshCoords(location=location)
    return mesh_coords, mesh_dim
