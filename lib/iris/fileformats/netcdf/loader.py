# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Support loading Iris cubes from NetCDF files using the CF conventions for metadata interpretation.

See : `NetCDF User's Guide <https://docs.unidata.ucar.edu/nug/current/>`_
and `netCDF4 python module <https://github.com/Unidata/netcdf4-python>`_.

Also : `CF Conventions <https://cfconventions.org/>`_.

"""
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from copy import deepcopy
from enum import Enum, auto
import threading
from typing import Union
import warnings

import numpy as np

from iris._lazy_data import as_lazy_data
from iris.aux_factory import (
    AtmosphereSigmaFactory,
    HybridHeightFactory,
    HybridPressureFactory,
    OceanSFactory,
    OceanSg1Factory,
    OceanSg2Factory,
    OceanSigmaFactory,
    OceanSigmaZFactory,
)
import iris.config
import iris.coord_systems
import iris.coords
import iris.exceptions
import iris.fileformats.cf
from iris.fileformats.netcdf import _thread_safe_nc
from iris.fileformats.netcdf.saver import _CF_ATTRS
import iris.io
import iris.util

# Show actions activation statistics.
DEBUG = False

# Get the logger : shared logger for all in 'iris.fileformats.netcdf'.
from . import logger

# An expected part of the public loader API, but includes thread safety
#  concerns so is housed in _thread_safe_nc.
NetCDFDataProxy = _thread_safe_nc.NetCDFDataProxy


class _WarnComboIgnoringBoundsLoad(
    iris.exceptions.IrisIgnoringBoundsWarning,
    iris.exceptions.IrisLoadWarning,
):
    """One-off combination of warning classes - enhances user filtering."""

    pass


def _actions_engine():
    # Return an 'actions engine', which provides a pyke-rules-like interface to
    # the core cf translation code.
    # Deferred import to avoid circularity.
    import iris.fileformats._nc_load_rules.engine as nc_actions_engine

    engine = nc_actions_engine.Engine()
    return engine


def _assert_case_specific_facts(engine, cf, cf_group):
    # Initialise a data store for built cube elements.
    # This is used to patch element attributes *not* setup by the actions
    # process, after the actions code has run.
    engine.cube_parts["coordinates"] = []
    engine.cube_parts["cell_measures"] = []
    engine.cube_parts["ancillary_variables"] = []

    # Assert facts for CF coordinates.
    for cf_name in cf_group.coordinates.keys():
        engine.add_case_specific_fact("coordinate", (cf_name,))

    # Assert facts for CF auxiliary coordinates.
    for cf_name in cf_group.auxiliary_coordinates.keys():
        engine.add_case_specific_fact("auxiliary_coordinate", (cf_name,))

    # Assert facts for CF cell measures.
    for cf_name in cf_group.cell_measures.keys():
        engine.add_case_specific_fact("cell_measure", (cf_name,))

    # Assert facts for CF ancillary variables.
    for cf_name in cf_group.ancillary_variables.keys():
        engine.add_case_specific_fact("ancillary_variable", (cf_name,))

    # Assert facts for CF grid_mappings.
    for cf_name in cf_group.grid_mappings.keys():
        engine.add_case_specific_fact("grid_mapping", (cf_name,))

    # Assert facts for CF labels.
    for cf_name in cf_group.labels.keys():
        engine.add_case_specific_fact("label", (cf_name,))

    # Assert facts for CF formula terms associated with the cf_group
    # of the CF data variable.

    # Collect varnames of formula-root variables as we go.
    # NOTE: use dictionary keys as an 'OrderedSet'
    #   - see: https://stackoverflow.com/a/53657523/2615050
    # This is to ensure that we can handle the resulting facts in a definite
    # order, as using a 'set' led to indeterminate results.
    formula_root = {}
    for cf_var in cf.cf_group.formula_terms.values():
        for cf_root, cf_term in cf_var.cf_terms_by_root.items():
            # Only assert this fact if the formula root variable is
            # defined in the CF group of the CF data variable.
            if cf_root in cf_group:
                formula_root[cf_root] = True
                engine.add_case_specific_fact(
                    "formula_term",
                    (cf_var.cf_name, cf_root, cf_term),
                )

    for cf_root in formula_root.keys():
        engine.add_case_specific_fact("formula_root", (cf_root,))


def _actions_activation_stats(engine, cf_name):
    print("-" * 80)
    print("CF Data Variable: %r" % cf_name)

    engine.print_stats()

    print("Rules Triggered:")

    for rule in sorted(list(engine.rules_triggered)):
        print("\t%s" % rule)

    print("Case Specific Facts:")
    kb_facts = engine.get_kb()

    for key in kb_facts.entity_lists.keys():
        for arg in kb_facts.entity_lists[key].case_specific_facts:
            print("\t%s%s" % (key, arg))


def _set_attributes(attributes, key, value):
    """Set attributes dictionary, converting unicode strings appropriately."""
    if isinstance(value, str):
        try:
            attributes[str(key)] = str(value)
        except UnicodeEncodeError:
            attributes[str(key)] = value
    else:
        attributes[str(key)] = value


def _add_unused_attributes(iris_object, cf_var):
    """Populate the attributes of a cf element with the "unused" attributes.

    Populate the attributes of a cf element with the "unused" attributes
    from the associated CF-netCDF variable. That is, all those that aren't CF
    reserved terms.

    """

    def attribute_predicate(item):
        return item[0] not in _CF_ATTRS

    tmpvar = filter(attribute_predicate, cf_var.cf_attrs_unused())
    attrs_dict = iris_object.attributes
    if hasattr(attrs_dict, "locals"):
        # Treat cube attributes (i.e. a CubeAttrsDict) as a special case.
        # These attrs are "local" (i.e. on the variable), so record them as such.
        attrs_dict = attrs_dict.locals
    for attr_name, attr_value in tmpvar:
        _set_attributes(attrs_dict, attr_name, attr_value)


def _get_actual_dtype(cf_var):
    # Figure out what the eventual data type will be after any scale/offset
    # transforms.
    dummy_data = np.zeros(1, dtype=cf_var.dtype)
    if hasattr(cf_var, "scale_factor"):
        dummy_data = cf_var.scale_factor * dummy_data
    if hasattr(cf_var, "add_offset"):
        dummy_data = cf_var.add_offset + dummy_data
    return dummy_data.dtype


# An arbitrary variable array size, below which we will fetch real data from a variable
# rather than making a lazy array for deferred access.
# Set by experiment at roughly the point where it begins to save us memory, but actually
# mostly done for speed improvement.  See https://github.com/SciTools/iris/pull/5069
_LAZYVAR_MIN_BYTES = 5000


def _get_cf_var_data(cf_var, filename):
    """Get an array representing the data of a CF variable.

    This is typically a lazy array based around a NetCDFDataProxy, but if the variable
    is "sufficiently small", we instead fetch the data as a real (numpy) array.
    The latter is especially valuable for scalar coordinates, which are otherwise
    unnecessarily slow + wasteful of memory.

    """
    global CHUNK_CONTROL
    if hasattr(cf_var, "_data_array"):
        # The variable is not an actual netCDF4 file variable, but an emulating
        # object with an attached data array (either numpy or dask), which can be
        # returned immediately as-is.  This is used as a hook to translate data to/from
        # netcdf data container objects in other packages, such as xarray.
        # See https://github.com/SciTools/iris/issues/4994 "Xarray bridge".
        result = cf_var._data_array
    else:
        total_bytes = cf_var.size * cf_var.dtype.itemsize
        if total_bytes < _LAZYVAR_MIN_BYTES:
            # Don't make a lazy array, as it will cost more memory AND more time to access.
            # Instead fetch the data immediately, as a real array, and return that.
            result = cf_var[:]

        else:
            # Get lazy chunked data out of a cf variable.
            # Creates Dask wrappers around data arrays for any cube components which
            # can have lazy values, e.g. Cube, Coord, CellMeasure, AuxiliaryVariable.
            dtype = _get_actual_dtype(cf_var)

            # Make a data-proxy that mimics array access and can fetch from the file.
            fill_value = getattr(
                cf_var.cf_data,
                "_FillValue",
                _thread_safe_nc.default_fillvals[cf_var.dtype.str[1:]],
            )
            proxy = NetCDFDataProxy(
                cf_var.shape, dtype, filename, cf_var.cf_name, fill_value
            )
            # Get the chunking specified for the variable : this is either a shape, or
            # maybe the string "contiguous".
            if CHUNK_CONTROL.mode is ChunkControl.Modes.AS_DASK:
                result = as_lazy_data(proxy, chunks=None, dask_chunking=True)
            else:
                chunks = cf_var.cf_data.chunking()
                # In the "contiguous" case, pass chunks=None to 'as_lazy_data'.
                if chunks == "contiguous":
                    if (
                        CHUNK_CONTROL.mode is ChunkControl.Modes.FROM_FILE
                        and isinstance(cf_var, iris.fileformats.cf.CFDataVariable)
                    ):
                        raise KeyError(
                            f"{cf_var.cf_name} does not contain pre-existing chunk specifications."
                            f" Instead, you might wish to use CHUNK_CONTROL.set(), or just use default"
                            f" behaviour outside of a context manager. "
                        )
                    # Equivalent to chunks=None, but value required by chunking control
                    chunks = list(cf_var.shape)

                # Modify the chunking in the context of an active chunking control.
                # N.B. settings specific to this named var override global ('*') ones.
                dim_chunks = CHUNK_CONTROL.var_dim_chunksizes.get(
                    cf_var.cf_name
                ) or CHUNK_CONTROL.var_dim_chunksizes.get("*")
                dims = cf_var.cf_data.dimensions
                if CHUNK_CONTROL.mode is ChunkControl.Modes.FROM_FILE:
                    dims_fixed = np.ones(len(dims), dtype=bool)
                elif not dim_chunks:
                    dims_fixed = None
                else:
                    # Modify the chunks argument, and pass in a list of 'fixed' dims, for
                    # any of our dims which are controlled.
                    dims_fixed = np.zeros(len(dims), dtype=bool)
                    for i_dim, dim_name in enumerate(dims):
                        dim_chunksize = dim_chunks.get(dim_name)
                        if dim_chunksize:
                            if dim_chunksize == -1:
                                chunks[i_dim] = cf_var.shape[i_dim]
                            else:
                                chunks[i_dim] = dim_chunksize
                            dims_fixed[i_dim] = True
                if dims_fixed is None:
                    dims_fixed = [dims_fixed]
                result = as_lazy_data(
                    proxy, chunks=chunks, dims_fixed=tuple(dims_fixed)
                )
    return result


class _OrderedAddableList(list):
    """A custom container object for actions recording.

    Used purely in actions debugging, to accumulate a record of which actions
    were activated.

    It replaces a set, so as to preserve the ordering of operations, with
    possible repeats, and it also numbers the entries.

    The actions routines invoke an 'add' method, so this effectively replaces
    a set.add with a list.append.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_add = 0

    def add(self, msg):
        self._n_add += 1
        n_add = self._n_add
        self.append(f"#{n_add:03d} : {msg}")


def _load_cube(engine, cf, cf_var, filename):
    global CHUNK_CONTROL

    # Translate dimension chunk-settings specific to this cube (i.e. named by
    # it's data-var) into global ones, for the duration of this load.
    # Thus, by default, we will create any AuxCoords, CellMeasures et al with
    # any  per-dimension chunksizes specified for the cube.
    these_settings = CHUNK_CONTROL.var_dim_chunksizes.get(cf_var.cf_name, {})
    with CHUNK_CONTROL.set(**these_settings):
        return _load_cube_inner(engine, cf, cf_var, filename)


def _load_cube_inner(engine, cf, cf_var, filename):
    from iris.cube import Cube

    """Create the cube associated with the CF-netCDF data variable."""
    data = _get_cf_var_data(cf_var, filename)
    cube = Cube(data)

    # Reset the actions engine.
    engine.reset()

    # Initialise engine rule processing hooks.
    engine.cf_var = cf_var
    engine.cube = cube
    engine.cube_parts = {}
    engine.requires = {}
    engine.rules_triggered = _OrderedAddableList()
    engine.filename = filename

    # Assert all the case-specific facts.
    # This extracts 'facts' specific to this data-variable (aka cube), from
    # the info supplied in the CFGroup object.
    _assert_case_specific_facts(engine, cf, cf_var.cf_group)

    # Run the actions engine.
    # This creates various cube elements and attaches them to the cube.
    # It also records various other info on the engine, to be processed later.
    engine.activate()

    # Having run the rules, now add the "unused" attributes to each cf element.
    def fix_attributes_all_elements(role_name):
        elements_and_names = engine.cube_parts.get(role_name, [])

        for iris_object, cf_var_name in elements_and_names:
            _add_unused_attributes(iris_object, cf.cf_group[cf_var_name])

    # Populate the attributes of all coordinates, cell-measures and ancillary-vars.
    fix_attributes_all_elements("coordinates")
    fix_attributes_all_elements("ancillary_variables")
    fix_attributes_all_elements("cell_measures")

    # Also populate attributes of the top-level cube itself.
    _add_unused_attributes(cube, cf_var)

    # Work out reference names for all the coords.
    names = {
        coord.var_name: coord.standard_name or coord.var_name or "unknown"
        for coord in cube.coords()
    }

    # Add all the cube cell methods.
    cube.cell_methods = [
        iris.coords.CellMethod(
            method=method.method,
            intervals=method.intervals,
            comments=method.comments,
            coords=[
                names[coord_name] if coord_name in names else coord_name
                for coord_name in method.coord_names
            ],
        )
        for method in cube.cell_methods
    ]

    if DEBUG:
        # Show activation statistics for this data-var (i.e. cube).
        _actions_activation_stats(engine, cf_var.cf_name)

    return cube


def _load_aux_factory(engine, cube):
    """Convert any CF-netCDF dimensionless coordinate to an AuxCoordFactory."""
    formula_type = engine.requires.get("formula_type")
    if formula_type in [
        "atmosphere_sigma_coordinate",
        "atmosphere_hybrid_height_coordinate",
        "atmosphere_hybrid_sigma_pressure_coordinate",
        "ocean_sigma_z_coordinate",
        "ocean_sigma_coordinate",
        "ocean_s_coordinate",
        "ocean_s_coordinate_g1",
        "ocean_s_coordinate_g2",
    ]:

        def coord_from_term(term):
            # Convert term names to coordinates (via netCDF variable names).
            name = engine.requires["formula_terms"].get(term, None)
            if name is not None:
                for coord, cf_var_name in engine.cube_parts["coordinates"]:
                    if cf_var_name == name:
                        return coord
                warnings.warn(
                    "Unable to find coordinate for variable {!r}".format(name),
                    category=iris.exceptions.IrisFactoryCoordNotFoundWarning,
                )

        if formula_type == "atmosphere_sigma_coordinate":
            pressure_at_top = coord_from_term("ptop")
            sigma = coord_from_term("sigma")
            surface_air_pressure = coord_from_term("ps")
            factory = AtmosphereSigmaFactory(
                pressure_at_top, sigma, surface_air_pressure
            )
        elif formula_type == "atmosphere_hybrid_height_coordinate":
            delta = coord_from_term("a")
            sigma = coord_from_term("b")
            orography = coord_from_term("orog")
            factory = HybridHeightFactory(delta, sigma, orography)
        elif formula_type == "atmosphere_hybrid_sigma_pressure_coordinate":
            # Hybrid pressure has two valid versions of its formula terms:
            # "p0: var1 a: var2 b: var3 ps: var4" or
            # "ap: var1 b: var2 ps: var3" where "ap = p0 * a"
            # Attempt to get the "ap" term.
            delta = coord_from_term("ap")
            if delta is None:
                # The "ap" term is unavailable, so try getting terms "p0"
                # and "a" terms in order to derive an "ap" equivalent term.
                coord_p0 = coord_from_term("p0")
                if coord_p0 is not None:
                    if coord_p0.shape != (1,):
                        msg = (
                            "Expecting {!r} to be a scalar reference "
                            "pressure coordinate, got shape {!r}".format(
                                coord_p0.var_name, coord_p0.shape
                            )
                        )
                        raise ValueError(msg)
                    if coord_p0.has_bounds():
                        msg = (
                            "Ignoring atmosphere hybrid sigma pressure "
                            "scalar coordinate {!r} bounds.".format(coord_p0.name())
                        )
                        warnings.warn(
                            msg,
                            category=_WarnComboIgnoringBoundsLoad,
                        )
                    coord_a = coord_from_term("a")
                    if coord_a is not None:
                        if coord_a.units.is_unknown():
                            # Be graceful, and promote unknown to dimensionless units.
                            coord_a.units = "1"
                        delta = coord_a * coord_p0.points[0]
                        delta.units = coord_a.units * coord_p0.units
                        delta.rename("vertical pressure")
                        delta.var_name = "ap"
                        cube.add_aux_coord(delta, cube.coord_dims(coord_a))

            sigma = coord_from_term("b")
            surface_air_pressure = coord_from_term("ps")
            factory = HybridPressureFactory(delta, sigma, surface_air_pressure)
        elif formula_type == "ocean_sigma_z_coordinate":
            sigma = coord_from_term("sigma")
            eta = coord_from_term("eta")
            depth = coord_from_term("depth")
            depth_c = coord_from_term("depth_c")
            nsigma = coord_from_term("nsigma")
            zlev = coord_from_term("zlev")
            factory = OceanSigmaZFactory(sigma, eta, depth, depth_c, nsigma, zlev)
        elif formula_type == "ocean_sigma_coordinate":
            sigma = coord_from_term("sigma")
            eta = coord_from_term("eta")
            depth = coord_from_term("depth")
            factory = OceanSigmaFactory(sigma, eta, depth)
        elif formula_type == "ocean_s_coordinate":
            s = coord_from_term("s")
            eta = coord_from_term("eta")
            depth = coord_from_term("depth")
            a = coord_from_term("a")
            depth_c = coord_from_term("depth_c")
            b = coord_from_term("b")
            factory = OceanSFactory(s, eta, depth, a, b, depth_c)
        elif formula_type == "ocean_s_coordinate_g1":
            s = coord_from_term("s")
            c = coord_from_term("c")
            eta = coord_from_term("eta")
            depth = coord_from_term("depth")
            depth_c = coord_from_term("depth_c")
            factory = OceanSg1Factory(s, c, eta, depth, depth_c)
        elif formula_type == "ocean_s_coordinate_g2":
            s = coord_from_term("s")
            c = coord_from_term("c")
            eta = coord_from_term("eta")
            depth = coord_from_term("depth")
            depth_c = coord_from_term("depth_c")
            factory = OceanSg2Factory(s, c, eta, depth, depth_c)
        cube.add_aux_factory(factory)


def _translate_constraints_to_var_callback(constraints):
    """Translate load constraints into a simple data-var filter function, if possible.

    Returns
    -------
    function : (cf_var:CFDataVariable)
        bool, or None.

    For now, ONLY handles a single NameConstraint with no 'STASH' component.

    """
    import iris._constraints

    constraints = iris._constraints.list_of_constraints(constraints)
    result = None
    if len(constraints) == 1:
        (constraint,) = constraints
        if (
            isinstance(constraint, iris._constraints.NameConstraint)
            and constraint.STASH == "none"
        ):
            # As long as it doesn't use a STASH match, then we can treat it as
            # a testing against name properties of cf_var.
            # That's just like testing against name properties of a cube, except that they may not all exist.
            def inner(cf_datavar):
                match = True
                for name in constraint._names:
                    expected = getattr(constraint, name)
                    if name != "STASH" and expected != "none":
                        attr_name = "cf_name" if name == "var_name" else name
                        # Fetch property : N.B. CFVariable caches the property values
                        # The use of a default here is the only difference from the code in NameConstraint.
                        if not hasattr(cf_datavar, attr_name):
                            continue
                        actual = getattr(cf_datavar, attr_name, "")
                        if actual != expected:
                            match = False
                            break
                return match

            result = inner
    return result


def load_cubes(file_sources, callback=None, constraints=None):
    """Load cubes from a list of NetCDF filenames/OPeNDAP URLs.

    Parameters
    ----------
    file_sources : str or list
        One or more NetCDF filenames/OPeNDAP URLs to load from.
        OR open datasets.

    callback : function, optional
        Function which can be passed on to :func:`iris.io.run_callback`.

    constraints : optional

    Returns
    -------
    Generator of loaded NetCDF :class:`iris.cube.Cube`.

    """
    # TODO: rationalise UGRID/mesh handling once experimental.ugrid is folded
    # into standard behaviour.
    # Deferred import to avoid circular imports.
    from iris.experimental.ugrid.cf import CFUGridReader
    from iris.experimental.ugrid.load import (
        PARSE_UGRID_ON_LOAD,
        _build_mesh_coords,
        _meshes_from_cf,
    )
    from iris.io import run_callback

    # Create a low-level data-var filter from the original load constraints, if they are suitable.
    var_callback = _translate_constraints_to_var_callback(constraints)

    # Create an actions engine.
    engine = _actions_engine()

    if isinstance(file_sources, str) or not isinstance(file_sources, Iterable):
        file_sources = [file_sources]

    for file_source in file_sources:
        # Ingest the file.  At present may be a filepath or an open netCDF4.Dataset.
        meshes = {}
        if PARSE_UGRID_ON_LOAD:
            cf_reader_class = CFUGridReader
        else:
            cf_reader_class = iris.fileformats.cf.CFReader

        with cf_reader_class(file_source) as cf:
            if PARSE_UGRID_ON_LOAD:
                meshes = _meshes_from_cf(cf)

            # Process each CF data variable.
            data_variables = list(cf.cf_group.data_variables.values()) + list(
                cf.cf_group.promoted.values()
            )
            for cf_var in data_variables:
                if var_callback and not var_callback(cf_var):
                    # Deliver only selected results.
                    continue

                # cf_var-specific mesh handling, if a mesh is present.
                # Build the mesh_coords *before* loading the cube - avoids
                # mesh-related attributes being picked up by
                # _add_unused_attributes().
                mesh_name = None
                mesh = None
                mesh_coords, mesh_dim = [], None
                if PARSE_UGRID_ON_LOAD:
                    mesh_name = getattr(cf_var, "mesh", None)
                if mesh_name is not None:
                    try:
                        mesh = meshes[mesh_name]
                    except KeyError:
                        message = (
                            f"File does not contain mesh: '{mesh_name}' - "
                            f"referenced by variable: '{cf_var.cf_name}' ."
                        )
                        logger.debug(message)
                if mesh is not None:
                    mesh_coords, mesh_dim = _build_mesh_coords(mesh, cf_var)

                cube = _load_cube(engine, cf, cf_var, cf.filename)

                # Attach the mesh (if present) to the cube.
                for mesh_coord in mesh_coords:
                    cube.add_aux_coord(mesh_coord, mesh_dim)

                # Process any associated formula terms and attach
                # the corresponding AuxCoordFactory.
                try:
                    _load_aux_factory(engine, cube)
                except ValueError as e:
                    warnings.warn(
                        "{}".format(e),
                        category=iris.exceptions.IrisLoadWarning,
                    )

                # Perform any user registered callback function.
                cube = run_callback(callback, cube, cf_var, file_source)

                # Callback mechanism may return None, which must not be yielded
                if cube is None:
                    continue

                yield cube


class ChunkControl(threading.local):
    """Provide user control of Chunk Control."""

    class Modes(Enum):
        """Modes Enums."""

        DEFAULT = auto()
        FROM_FILE = auto()
        AS_DASK = auto()

    def __init__(self, var_dim_chunksizes=None):
        """Provide user control of Dask chunking.

        The NetCDF loader is controlled by the single instance of this: the
        :data:`~iris.fileformats.netcdf.loader.CHUNK_CONTROL` object.

        A chunk size can be set for a specific (named) file dimension, when
        loading specific (named) variables, or for all variables.

        When a selected variable is a CF data-variable, which loads as a
        :class:`~iris.cube.Cube`, then the given dimension chunk size is *also*
        fixed for all variables which are components of that :class:`~iris.cube.Cube`,
        i.e. any :class:`~iris.coords.Coord`, :class:`~iris.coords.CellMeasure`,
        :class:`~iris.coords.AncillaryVariable` etc.
        This can be overridden, if required, by variable-specific settings.

        For this purpose, :class:`~iris.experimental.ugrid.mesh.MeshCoord` and
        :class:`~iris.experimental.ugrid.mesh.Connectivity` are not
        :class:`~iris.cube.Cube` components, and chunk control on a
        :class:`~iris.cube.Cube` data-variable will not affect them.

        """
        self.var_dim_chunksizes = var_dim_chunksizes or {}
        self.mode = self.Modes.DEFAULT

    @contextmanager
    def set(
        self,
        var_names: Union[str, Iterable[str]] = None,
        **dimension_chunksizes: Mapping[str, int],
    ) -> None:
        r"""Control the Dask chunk sizes applied to NetCDF variables during loading.

        Parameters
        ----------
        var_names : str or list of str, default=None
            apply the `dimension_chunksizes` controls only to these variables,
            or when building :class:`~iris.cube.Cube`\\ s from these data variables.
            If ``None``, settings apply to all loaded variables.
        dimension_chunksizes : dict of {str: int}
            Kwargs specifying chunksizes for dimensions of file variables.
            Each key-value pair defines a chunk size for a named file
            dimension, e.g. ``{'time': 10, 'model_levels':1}``.
            Values of ``-1`` will lock the chunk size to the full size of that
            dimension.

        Notes
        -----
        This function acts as a context manager, for use in a ``with`` block.

        >>> import iris
        >>> from iris.fileformats.netcdf.loader import CHUNK_CONTROL
        >>> with CHUNK_CONTROL.set("air_temperature", time=180, latitude=-1):
        ...     cube = iris.load(iris.sample_data_path("E1_north_america.nc"))[0]

        When `var_names` is present, the chunk size adjustments are applied
        only to the selected variables.  However, for a CF data variable, this
        extends to all components of the (raw) :class:`~iris.cube.Cube` created
        from it.

        **Un**-adjusted dimensions have chunk sizes set in the 'usual' way.
        That is, according to the normal behaviour of
        :func:`iris._lazy_data.as_lazy_data`, which is: chunk size is based on
        the file variable chunking, or full variable shape; this is scaled up
        or down by integer factors to best match the Dask default chunk size,
        i.e. the setting configured by
        ``dask.config.set({'array.chunk-size': '250MiB'})``.

        """
        old_mode = self.mode
        old_var_dim_chunksizes = deepcopy(self.var_dim_chunksizes)
        if var_names is None:
            var_names = ["*"]
        elif isinstance(var_names, str):
            var_names = [var_names]
        try:
            for var_name in var_names:
                # Note: here we simply treat '*' as another name.
                # A specific name match should override a '*' setting, but
                # that is implemented elsewhere.
                if not isinstance(var_name, str):
                    msg = (
                        "'var_names' should be an iterable of strings, "
                        f"not {var_names!r}."
                    )
                    raise ValueError(msg)
                dim_chunks = self.var_dim_chunksizes.setdefault(var_name, {})
                for dim_name, chunksize in dimension_chunksizes.items():
                    if not (isinstance(dim_name, str) and isinstance(chunksize, int)):
                        msg = (
                            "'dimension_chunksizes' kwargs should be a dict "
                            f"of `str: int` pairs, not {dimension_chunksizes!r}."
                        )
                        raise ValueError(msg)
                    dim_chunks[dim_name] = chunksize
            yield
        finally:
            self.var_dim_chunksizes = old_var_dim_chunksizes
            self.mode = old_mode

    @contextmanager
    def from_file(self) -> None:
        r"""Ensure the chunk sizes are loaded in from NetCDF file variables.

        Raises
        ------
        KeyError
            If any NetCDF data variables - those that become
            :class:`~iris.cube.Cube`\\ s - do not specify chunk sizes.

        Notes
        -----
        This function acts as a context manager, for use in a ``with`` block.
        """
        old_mode = self.mode
        old_var_dim_chunksizes = deepcopy(self.var_dim_chunksizes)
        try:
            self.mode = self.Modes.FROM_FILE
            yield
        finally:
            self.mode = old_mode
            self.var_dim_chunksizes = old_var_dim_chunksizes

    @contextmanager
    def as_dask(self) -> None:
        """Relies on Dask :external+dask:doc:`array` to control chunk sizes.

        Notes
        -----
        This function acts as a context manager, for use in a ``with`` block.
        """
        old_mode = self.mode
        old_var_dim_chunksizes = deepcopy(self.var_dim_chunksizes)
        try:
            self.mode = self.Modes.AS_DASK
            yield
        finally:
            self.mode = old_mode
            self.var_dim_chunksizes = old_var_dim_chunksizes


# Note: the CHUNK_CONTROL object controls chunk sizing in the
# :meth:`_get_cf_var_data` method.
# N.B. :meth:`_load_cube` also modifies this when loading each cube,
# introducing an additional context in which any cube-specific settings are
# 'promoted' into being global ones.

#: The global :class:`ChunkControl` object providing user-control of Dask chunking
#: when Iris loads NetCDF files.
CHUNK_CONTROL: ChunkControl = ChunkControl()
