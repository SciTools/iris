# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Module to support the loading of a NetCDF file into an Iris cube.

See also: `netCDF4 python <http://code.google.com/p/netcdf4-python/>`_.

Also refer to document 'NetCDF Climate and Forecast (CF) Metadata Conventions'.

"""

import re
import warnings

import netCDF4
import numpy as np

from iris._lazy_data import as_lazy_data
from iris.aux_factory import (
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
import iris.io
import iris.util

# Show actions activation statistics.
DEBUG = False

# Configure the logger.
logger = iris.config.get_logger(__name__)

# Pass through CF attributes:
#  - comment
#  - Conventions
#  - flag_masks
#  - flag_meanings
#  - flag_values
#  - history
#  - institution
#  - reference
#  - source
#  - title
#  - positive
#
_CF_ATTRS = [
    "add_offset",
    "ancillary_variables",
    "axis",
    "bounds",
    "calendar",
    "cell_measures",
    "cell_methods",
    "climatology",
    "compress",
    "coordinates",
    "_FillValue",
    "formula_terms",
    "grid_mapping",
    "leap_month",
    "leap_year",
    "long_name",
    "missing_value",
    "month_lengths",
    "scale_factor",
    "standard_error_multiplier",
    "standard_name",
    "units",
]

# Cell methods.
_CM_KNOWN_METHODS = [
    "point",
    "sum",
    "mean",
    "maximum",
    "minimum",
    "mid_range",
    "standard_deviation",
    "variance",
    "mode",
    "median",
]

_CM_COMMENT = "comment"
_CM_EXTRA = "extra"
_CM_INTERVAL = "interval"
_CM_METHOD = "method"
_CM_NAME = "name"
_CM_PARSE = re.compile(
    r"""
                           (?P<name>([\w_]+\s*?:\s+)+)
                           (?P<method>[\w_\s]+(?![\w_]*\s*?:))\s*
                           (?:
                               \(\s*
                               (?P<extra>[^\)]+)
                               \)\s*
                           )?
                       """,
    re.VERBOSE,
)


class UnknownCellMethodWarning(Warning):
    pass


def parse_cell_methods(nc_cell_methods):
    """
    Parse a CF cell_methods attribute string into a tuple of zero or
    more CellMethod instances.

    Args:

    * nc_cell_methods (str):
        The value of the cell methods attribute to be parsed.

    Returns:

    * cell_methods
        An iterable of :class:`iris.coords.CellMethod`.

    Multiple coordinates, intervals and comments are supported.
    If a method has a non-standard name a warning will be issued, but the
    results are not affected.

    """

    cell_methods = []
    if nc_cell_methods is not None:
        for m in _CM_PARSE.finditer(nc_cell_methods):
            d = m.groupdict()
            method = d[_CM_METHOD]
            method = method.strip()
            # Check validity of method, allowing for multi-part methods
            # e.g. mean over years.
            method_words = method.split()
            if method_words[0].lower() not in _CM_KNOWN_METHODS:
                msg = "NetCDF variable contains unknown cell method {!r}"
                warnings.warn(
                    msg.format("{}".format(method_words[0])),
                    UnknownCellMethodWarning,
                )
            d[_CM_METHOD] = method
            name = d[_CM_NAME]
            name = name.replace(" ", "")
            name = name.rstrip(":")
            d[_CM_NAME] = tuple([n for n in name.split(":")])
            interval = []
            comment = []
            if d[_CM_EXTRA] is not None:
                #
                # tokenise the key words and field colon marker
                #
                d[_CM_EXTRA] = d[_CM_EXTRA].replace(
                    "comment:", "<<comment>><<:>>"
                )
                d[_CM_EXTRA] = d[_CM_EXTRA].replace(
                    "interval:", "<<interval>><<:>>"
                )
                d[_CM_EXTRA] = d[_CM_EXTRA].split("<<:>>")
                if len(d[_CM_EXTRA]) == 1:
                    comment.extend(d[_CM_EXTRA])
                else:
                    next_field_type = comment
                    for field in d[_CM_EXTRA]:
                        field_type = next_field_type
                        index = field.rfind("<<interval>>")
                        if index == 0:
                            next_field_type = interval
                            continue
                        elif index > 0:
                            next_field_type = interval
                        else:
                            index = field.rfind("<<comment>>")
                            if index == 0:
                                next_field_type = comment
                                continue
                            elif index > 0:
                                next_field_type = comment
                        if index != -1:
                            field = field[:index]
                        field_type.append(field.strip())
            #
            # cater for a shared interval over multiple axes
            #
            if len(interval):
                if len(d[_CM_NAME]) != len(interval) and len(interval) == 1:
                    interval = interval * len(d[_CM_NAME])
            #
            # cater for a shared comment over multiple axes
            #
            if len(comment):
                if len(d[_CM_NAME]) != len(comment) and len(comment) == 1:
                    comment = comment * len(d[_CM_NAME])
            d[_CM_INTERVAL] = tuple(interval)
            d[_CM_COMMENT] = tuple(comment)
            cell_method = iris.coords.CellMethod(
                d[_CM_METHOD],
                coords=d[_CM_NAME],
                intervals=d[_CM_INTERVAL],
                comments=d[_CM_COMMENT],
            )
            cell_methods.append(cell_method)
    return tuple(cell_methods)


def _actions_engine():
    # Return an 'actions engine', which provides a pyke-rules-like interface to
    # the core cf translation code.
    # Deferred import to avoid circularity.
    import iris.fileformats._nc_load_rules.engine as nc_actions_engine

    engine = nc_actions_engine.Engine()
    return engine


class NetCDFDataProxy:
    """A reference to the data payload of a single NetCDF file variable."""

    __slots__ = ("shape", "dtype", "path", "variable_name", "fill_value")

    def __init__(self, shape, dtype, path, variable_name, fill_value):
        self.shape = shape
        self.dtype = dtype
        self.path = path
        self.variable_name = variable_name
        self.fill_value = fill_value

    @property
    def ndim(self):
        return len(self.shape)

    def __getitem__(self, keys):
        dataset = netCDF4.Dataset(self.path)
        try:
            variable = dataset.variables[self.variable_name]
            # Get the NetCDF variable data and slice.
            var = variable[keys]
        finally:
            dataset.close()
        return np.asanyarray(var)

    def __repr__(self):
        fmt = (
            "<{self.__class__.__name__} shape={self.shape}"
            " dtype={self.dtype!r} path={self.path!r}"
            " variable_name={self.variable_name!r}>"
        )
        return fmt.format(self=self)

    def __getstate__(self):
        return {attr: getattr(self, attr) for attr in self.__slots__}

    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)


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

    for rule in sorted(list(engine.rule_triggered)):
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
    """
    Populate the attributes of a cf element with the "unused" attributes
    from the associated CF-netCDF variable. That is, all those that aren't CF
    reserved terms.

    """

    def attribute_predicate(item):
        return item[0] not in _CF_ATTRS

    tmpvar = filter(attribute_predicate, cf_var.cf_attrs_unused())
    for attr_name, attr_value in tmpvar:
        _set_attributes(iris_object.attributes, attr_name, attr_value)


def _get_actual_dtype(cf_var):
    # Figure out what the eventual data type will be after any scale/offset
    # transforms.
    dummy_data = np.zeros(1, dtype=cf_var.dtype)
    if hasattr(cf_var, "scale_factor"):
        dummy_data = cf_var.scale_factor * dummy_data
    if hasattr(cf_var, "add_offset"):
        dummy_data = cf_var.add_offset + dummy_data
    return dummy_data.dtype


def _get_cf_var_data(cf_var, filename):
    # Get lazy chunked data out of a cf variable.
    dtype = _get_actual_dtype(cf_var)

    # Create cube with deferred data, but no metadata
    fill_value = getattr(
        cf_var.cf_data,
        "_FillValue",
        netCDF4.default_fillvals[cf_var.dtype.str[1:]],
    )
    proxy = NetCDFDataProxy(
        cf_var.shape, dtype, filename, cf_var.cf_name, fill_value
    )
    # Get the chunking specified for the variable : this is either a shape, or
    # maybe the string "contiguous".
    chunks = cf_var.cf_data.chunking()
    # In the "contiguous" case, pass chunks=None to 'as_lazy_data'.
    if chunks == "contiguous":
        chunks = None
    return as_lazy_data(proxy, chunks=chunks)


class OrderedAddableList(list):
    # Used purely in actions debugging, to accumulate a record of which actions
    # were activated.
    # It replaces a set, so as to record the ordering of operations, with
    # possible repeats, and it also numbers the entries.
    # Actions routines invoke the 'add' method, which thus effectively converts
    # a set.add into a list.append.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_add = 0

    def add(self, msg):
        self._n_add += 1
        n_add = self._n_add
        self.append(f"#{n_add:03d} : {msg}")


def _load_cube(engine, cf, cf_var, filename):
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
    engine.rule_triggered = OrderedAddableList()
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
    """
    Convert any CF-netCDF dimensionless coordinate to an AuxCoordFactory.

    """
    formula_type = engine.requires.get("formula_type")
    if formula_type in [
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
                    "Unable to find coordinate for variable "
                    "{!r}".format(name)
                )

        if formula_type == "atmosphere_hybrid_height_coordinate":
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
                            "scalar coordinate {!r} bounds.".format(
                                coord_p0.name()
                            )
                        )
                        warnings.warn(msg)
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
            factory = OceanSigmaZFactory(
                sigma, eta, depth, depth_c, nsigma, zlev
            )
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


def load_cubes(filenames, callback=None):
    """
    Loads cubes from a list of NetCDF filenames/URLs.

    Args:

    * filenames (string/list):
        One or more NetCDF filenames/DAP URLs to load from.

    Kwargs:

    * callback (callable function):
        Function which can be passed on to :func:`iris.io.run_callback`.

    Returns:
        Generator of loaded NetCDF :class:`iris.cube.Cube`.

    """
    # TODO: rationalise UGRID/mesh handling once experimental.ugrid is folded
    #  into standard behaviour.
    # Deferred import to avoid circular imports.
    from iris.experimental.ugrid import (
        PARSE_UGRID_ON_LOAD,
        CFUGridReader,
        _build_mesh,
        _build_mesh_coords,
    )
    from iris.io import run_callback

    # Create an actions engine.
    engine = _actions_engine()

    if isinstance(filenames, str):
        filenames = [filenames]

    for filename in filenames:
        # Ingest the netCDF file.
        meshes = {}
        if PARSE_UGRID_ON_LOAD:
            cf = CFUGridReader(filename)

            # Mesh instances are shared between file phenomena.
            # TODO: more sophisticated Mesh sharing between files.
            # TODO: access external Mesh cache?
            mesh_vars = cf.cf_group.meshes
            meshes = {
                name: _build_mesh(cf, var, filename)
                for name, var in mesh_vars.items()
            }
        else:
            cf = iris.fileformats.cf.CFReader(filename)

        # Process each CF data variable.
        data_variables = list(cf.cf_group.data_variables.values()) + list(
            cf.cf_group.promoted.values()
        )
        for cf_var in data_variables:
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

            cube = _load_cube(engine, cf, cf_var, filename)

            # Attach the mesh (if present) to the cube.
            for mesh_coord in mesh_coords:
                cube.add_aux_coord(mesh_coord, mesh_dim)

            # Process any associated formula terms and attach
            # the corresponding AuxCoordFactory.
            try:
                _load_aux_factory(engine, cube)
            except ValueError as e:
                warnings.warn("{}".format(e))

            # Perform any user registered callback function.
            cube = run_callback(callback, cube, cf_var, filename)

            # Callback mechanism may return None, which must not be yielded
            if cube is None:
                continue

            yield cube
