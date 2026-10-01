# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Build Iris cubes from CF-conforming variables, whatever stores them.

.. z_reference:: iris.fileformats.cf.loader
   :tags: topic_load_save

   API reference

This module is the CF half of what used to be
``iris.fileformats.netcdf.loader``. It takes the
:class:`~iris.fileformats.cf.CFVariable` objects a
:class:`~iris.fileformats.cf.CFReader` produced, runs the loading rules over
them and returns :class:`~iris.cube.Cube` objects. None of that is specific to
netCDF, and nothing here imports netCDF4 - though three deferred imports still
reach back into :mod:`iris.fileformats.netcdf` for now: ``saver._CF_ATTRS``
and ``saver.Saver``, because the saver has not moved yet (PR 5), and
``loader.DEBUG``, which never moves (see below). Each is commented at its own
call site with why and, where relevant, when it goes.

Where a storage format *does* show through - reading a variable's array - the
work is delegated to :meth:`~iris.fileformats.cf.dataset.CFDatasetVariable.read_data`,
and this module supplies only the chunking policy to apply if the result comes
back lazy.

Opening files remains the format's own business: see
:func:`iris.fileformats.netcdf.load_cubes`.

Also : `CF Conventions <https://cfconventions.org/>`_.

"""

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from enum import Enum, auto
from functools import partial
import threading
import warnings

import numpy as np

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
import iris.coords
import iris.fileformats.cf
import iris.warnings


class _WarnComboIgnoringBoundsLoad(
    iris.warnings.IrisIgnoringBoundsWarning,
    iris.warnings.IrisLoadWarning,
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
    engine.cube_parts["coordinate_systems"] = {}

    # Add the parsed coordinate reference system mappings
    engine.cube_parts["coordinate_system_mappings"] = cf._coord_system_mappings.get(
        engine.cf_var.cf_name, None
    )

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
    from iris.fileformats._nc_load_rules.helpers import _add_or_capture

    # Deferred import: iris.fileformats.netcdf.saver imports the CF layer, so
    # importing it here at module scope would close the loop. PR 5 moves the
    # saver and this becomes a plain import.
    from iris.fileformats.netcdf.saver import _CF_ATTRS
    from iris.loading import LoadProblems

    def attribute_predicate(item):
        return item[0] not in _CF_ATTRS

    tmpvar = filter(attribute_predicate, cf_var.cf_attrs_unused())
    attrs_dict = iris_object.attributes
    if hasattr(attrs_dict, "locals"):
        # Treat cube attributes (i.e. a CubeAttrsDict) as a special case.
        # These attrs are "local" (i.e. on the variable), so record them as such.
        attrs_dict = attrs_dict.locals

    for attr_name, attr_value in tmpvar:
        _ = _add_or_capture(
            build_func=partial(lambda: attr_value),
            add_method=partial(_set_attributes, attrs_dict, attr_name),
            cf_var=cf_var,
            attr_key=attr_name,
            destination=LoadProblems.Problem.Destination(
                iris_class=iris_object.__class__,
                identifier=cf_var.cf_name,
            ),
        )


def _get_actual_dtype(cf_var):
    # Figure out what the eventual data type will be after any scale/offset
    # transforms.
    dummy_data = np.zeros(1, dtype=cf_var.dtype)
    if "scale_factor" in cf_var.attributes:
        dummy_data = cf_var.attributes["scale_factor"] * dummy_data
    if "add_offset" in cf_var.attributes:
        dummy_data = cf_var.attributes["add_offset"] + dummy_data
    return dummy_data.dtype


# An arbitrary variable array size, below which we will fetch real data from a variable
# rather than making a lazy array for deferred access.
# Set by experiment at roughly the point where it begins to save us memory, but actually
# mostly done for speed improvement.  See https://github.com/SciTools/iris/pull/5069
_LAZYVAR_MIN_BYTES = 5000


def _chunks_from_chunk_control(cf_var):
    """Return the ``(chunks, dims_fixed)`` pair for a lazy array of this variable.

    This is the one piece of a lazy array that is CF's business rather than the
    storage's: how big the dask chunks should be, given the store's own chunking
    and whatever :data:`CHUNK_CONTROL` context is active.

    Parameters
    ----------
    cf_var : :class:`iris.fileformats.cf.CFVariable`
        The variable the array is being built for.

    Returns
    -------
    tuple
        ``("auto", None)`` in :attr:`ChunkControl.Modes.AS_DASK`, otherwise a
        list of chunk sizes and a tuple of per-dimension "fixed" flags, both
        shaped for :func:`iris._lazy_data.as_lazy_data`.

    Raises
    ------
    KeyError
        In :attr:`ChunkControl.Modes.FROM_FILE`, when ``cf_var`` is a cube's
        data variable and the store offers no chunking to adopt.

    """
    if CHUNK_CONTROL.mode is ChunkControl.Modes.AS_DASK:
        return "auto", None

    # Get the chunking specified for the variable : this is either a shape, or
    # None if the variable is unchunked.
    chunks = cf_var.cf_data.chunking
    if chunks is None:
        # Unchunked : either a non-version-4 file, or a contiguous
        # version-4 variable. Neither offers a chunking to adopt.
        if CHUNK_CONTROL.mode is ChunkControl.Modes.FROM_FILE and isinstance(
            cf_var, iris.fileformats.cf.CFDataVariable
        ):
            raise KeyError(
                f"{cf_var.cf_name} does not contain pre-existing chunk specifications."
                f" Instead, you might wish to use CHUNK_CONTROL.set(), or just use default"
                f" behaviour outside of a context manager. "
            )
        # Equivalent to chunks=None, but value required by chunking control
        chunks = list(cf_var.shape)
    else:
        # The chunk-control block below assigns into this.
        chunks = list(chunks)

    # Modify the chunking in the context of an active chunking control.
    # N.B. settings specific to this named var override global ('*') ones.
    dim_chunks = CHUNK_CONTROL.var_dim_chunksizes.get(
        cf_var.cf_name
    ) or CHUNK_CONTROL.var_dim_chunksizes.get("*")
    dims = cf_var.dimensions
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
    return chunks, tuple(dims_fixed)


def _get_cf_var_data(cf_var):
    """Get an array representing the data of a CF variable.

    This is typically a lazy array, but if the variable is "sufficiently small"
    the storage fetches the data as a real (numpy) array instead. The latter is
    especially valuable for scalar coordinates, which are otherwise
    unnecessarily slow + wasteful of memory.

    The store builds the array; this supplies the one decision that is CF's and
    not the store's, namely what chunking a lazy result should have. See section
    4.4 of ``docs/superpowers/specs/2026-09-21-zarr-io-design.md``.

    """
    return cf_var.cf_data.read_data(partial(_chunks_from_chunk_control, cf_var))


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
    from iris.fileformats.netcdf.saver import Saver

    if Saver._DATALESS_ATTRNAME in cf_var.attributes:
        # This data-variable represents a dataless cube.
        # The variable array content was never written (to take up no space).
        data = None
        shape = cf_var.shape
    else:
        data = _get_cf_var_data(cf_var)
        shape = None
    cube = Cube(data=data, shape=shape)

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

    # Set extended_grid_mapping property ONLY if extended grid_mapping was used.
    # This avoids having an unnecessary `iris_extended_grid_mapping` attribute entry.
    if cs_mappings := engine.cube_parts.get("coordinate_system_mappings", None):
        # `None` as a mapping key implies simple mapping syntax (single coord system)
        if None not in cs_mappings:
            cube.extended_grid_mapping = True

    # DEBUG stays in iris.fileformats.netcdf (spec 4.8 lists it among the
    # names that do not move). Read it through the module, not as a
    # from-import, so that assigning to iris.fileformats.netcdf.loader.DEBUG -
    # which is how it is used - still takes effect here.
    from iris.fileformats.netcdf import loader as netcdf_loader

    if netcdf_loader.DEBUG:
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
                    category=iris.warnings.IrisFactoryCoordNotFoundWarning,
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
    bool or None

    Notes
    -----
    For now, ONLY handles NameConstraints with no 'STASH' component.

    """
    import iris._constraints

    constraints = iris._constraints.list_of_constraints(constraints)
    if len(constraints) == 0 or not all(
        isinstance(constraint, iris._constraints.NameConstraint)
        and constraint.STASH == "none"
        for constraint in constraints
    ):
        # We can define a var-filtering function to speedup the load, *ONLY* when we
        #  have some constraints, and all are simple NameConstraints with no STASH.
        result = None
    else:

        def inner(cf_datavar):
            match_any_constraint = False
            for constraint in constraints:
                match_this_constraint = True
                for name in constraint._names:
                    expected = getattr(constraint, name)
                    if name != "STASH" and expected != "none":
                        if name == "var_name":
                            actual = cf_datavar.cf_name
                        elif name in cf_datavar.attributes:
                            actual = cf_datavar.attributes[name]
                        else:
                            # Unlike NameConstraint, a variable that does not
                            # carry the attribute is not a mismatch here: the
                            # cube may still acquire the name later in the load.
                            continue
                        if actual != expected:
                            match_this_constraint = False
                            break
                if match_this_constraint:
                    match_any_constraint = True
                    break
            return match_any_constraint

        result = inner
    return result


class ChunkControl(threading.local):
    """Provide user control of Chunk Control."""

    class Modes(Enum):
        """Modes Enums."""

        DEFAULT = auto()
        FROM_FILE = auto()
        AS_DASK = auto()

    def __init__(self, var_dim_chunksizes=None):
        """Provide user control of Dask chunking.

        The CF loader is controlled by the single instance of this: the
        :data:`~iris.fileformats.cf.loader.CHUNK_CONTROL` object.

        A chunk size can be set for a specific (named) file dimension, when
        loading specific (named) variables, or for all variables.

        When a selected variable is a CF data-variable, which loads as a
        :class:`~iris.cube.Cube`, then the given dimension chunk size is *also*
        fixed for all variables which are components of that :class:`~iris.cube.Cube`,
        i.e. any :class:`~iris.coords.Coord`, :class:`~iris.coords.CellMeasure`,
        :class:`~iris.coords.AncillaryVariable` etc.
        This can be overridden, if required, by variable-specific settings.

        For this purpose, :class:`~iris.mesh.MeshCoord` and
        :class:`~iris.mesh.Connectivity` are not
        :class:`~iris.cube.Cube` components, and chunk control on a
        :class:`~iris.cube.Cube` data-variable will not affect them.

        """
        self.var_dim_chunksizes = var_dim_chunksizes or {}
        self.mode = self.Modes.DEFAULT

    @contextmanager
    def set(
        self,
        var_names: str | Iterable[str] | None = None,
        **dimension_chunksizes: Mapping[str, int],
    ) -> Iterator[None]:
        r"""Control the Dask chunk sizes applied to NetCDF variables during loading.

        This function can also be used to provide a size hint for the unknown
        array lengths when loading "variable-length" NetCDF data types.
        See https://unidata.github.io/netcdf4-python/#netCDF4.Dataset.vltypes

        Parameters
        ----------
        var_names : str or list of str, default=None
            Apply the `dimension_chunksizes` controls only to these variables,
            or when building :class:`~iris.cube.Cube` from these data variables.
            If ``None``, settings apply to all loaded variables.
        **dimension_chunksizes : dict of {str: int}
            Kwargs specifying chunksizes for dimensions of file variables.
            Each key-value pair defines a chunk size for a named file
            dimension, e.g. ``{'time': 10, 'model_levels':1}``.
            Values of ``-1`` will lock the chunk size to the full size of that
            dimension. To specify a size hint for "variable-length"  data types
            use the special name `_vl_hint`.

        Notes
        -----
        This function acts as a context manager, for use in a ``with`` block.

        >>> import iris
        >>> from iris.fileformats.cf.loader import CHUNK_CONTROL
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

        For variable-length data types the size of the variable (or "ragged")
        dimension of the individual array elements cannot be known without
        reading the data. This can make it difficult for Iris to determine
        whether to load the data lazily or not. If the user has some apriori
        knowledge of the mean variable array length this can be passed as
        as a size hint via the special `_vl_hint` name. For example a hint
        that variable-length string array that contains 4 character experiment
        identifiers:
        ``CHUNK_CONTROL.set("expver", _vl_hint=4)``

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
                    msg = (  # type: ignore[unreachable]
                        "'var_names' should be an iterable of strings, "
                        f"not {var_names!r}."
                    )
                    raise ValueError(msg)
                dim_chunks = self.var_dim_chunksizes.setdefault(var_name, {})
                for dim_name, chunksize in dimension_chunksizes.items():
                    if not (isinstance(dim_name, str) and isinstance(chunksize, int)):  # type: ignore[redundant-expr]
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
    def from_file(self) -> Iterator[None]:
        r"""Ensure the chunk sizes are loaded in from NetCDF file variables.

        Raises
        ------
        KeyError
            If any NetCDF data variables - those that become
            :class:`~iris.cube.Cube` - do not specify chunk sizes.

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
    def as_dask(self) -> Iterator[None]:
        """Rely on Dask :external+dask:doc:`array` to control chunk sizes.

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
