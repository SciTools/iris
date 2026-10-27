# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Save Iris cubes to a CF-conforming store.

.. z_reference:: iris.fileformats.cf.saver
   :tags: topic_load_save

   API reference

Everything about writing a cube that does not depend on what it is written
into. The CF data model is here - dimension naming, coordinate and bounds
variables, cell methods, grid mappings, meshes, attribute compliance. Every
storage call goes through :class:`~iris.fileformats.cf.dataset.CFDataset`,
except where a store's own ``_grid_mapping_writer`` writes somewhere that is
not part of it; netCDF's override does that, and every other store's does not.

A format supplies a subclass of ``_CFSaver`` holding its own vocabulary of
storage keywords, and whatever of the five hooks it needs:
``_create_string_variable``, ``_backend_write``, ``_flush_before_close``,
``_patch_dataset`` and ``_grid_mapping_writer``.
:class:`iris.fileformats.netcdf.saver.Saver` is one such subclass, and is where
a user of the netCDF format meets all of this.

``_CFSaver`` is private. What a format must supply is the five hooks, which
are private too, and one subclass is not enough evidence to promise their
shape; everything public on the class is the caller's API, inherited
unchanged by each format's own public saver. Contrast
:class:`~iris.fileformats.cf.dataset.CFDataset`, which is public because every
member of it is something a format must implement, and because a user can
reach one through :attr:`iris.fileformats.cf.CFVariable.cf_data`.

Nothing here is netCDF-specific any longer: the last thing that was is now a
hook like the rest.

``_add_grid_mapping_to_dataset`` writes each CF grid-mapping parameter through
the function ``_grid_mapping_writer`` hands back, so that where a parameter
lands is visible at the point it is written. netCDF's function writes onto its
own variable, because what that does to the type of each attribute in the file
is long-standing behaviour and not this module's to change -- see Q8 in §12.3
of the native Zarr I/O design,
``docs/superpowers/specs/2026-09-21-zarr-io-design.md``. Every other store
writes into the ``attributes`` mapping.

``ATTRIBUTE_HANDLERS`` lives beside this module, in
:mod:`iris.fileformats.cf._attribute_handlers`. The handlers serialise the
Iris-only attribute types that every CF store needs - STASH, GRIB_PARAM,
ukmo__process_flags - and the netCDF load rules read them from there too.

See : `CF Conventions <https://cfconventions.org/>`_.

"""

import collections
import re
import string
from typing import TYPE_CHECKING
import warnings

import cf_units
import dask
import dask.array as da
from dask.delayed import Delayed
import numpy as np

from iris import FUTURE
from iris._deprecation import warn_deprecated
from iris._lazy_data import _co_realise_lazy_arrays, is_lazy_data
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
import iris.coord_systems
import iris.coords
from iris.coords import (
    AncillaryVariable,
    AuxCoord,
    CellMeasure,
    DimCoord,
    _DimensionalMetadata,
)
import iris.exceptions
from iris.fileformats.cf.dataset import CFDatasetVariable
import iris.util
import iris.warnings

from ._attribute_handlers import ATTRIBUTE_HANDLERS

if TYPE_CHECKING:
    from iris.cube import Cube
    from iris.mesh.components import (
        MeshEdgeCoords,
        MeshFaceCoords,
        MeshNodeCoords,
        MeshXY,
    )

# Standard CML spatio-temporal axis names.
SPATIO_TEMPORAL_AXES = ["t", "z", "y", "x"]

# The CF-meaningful attributes which may appear on a data variable.
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

# CF attributes that should not be global.
_CF_DATA_ATTRS = [
    "flag_masks",
    "flag_meanings",
    "flag_values",
    "instance_dimension",
    "missing_value",
    "sample_dimension",
    "standard_error_multiplier",
]

# CF attributes that should only be global.
_CF_GLOBAL_ATTRS = ["conventions", "featureType", "history", "title"]

# UKMO specific attributes that should not be global.
_UKMO_DATA_ATTRS = [
    "STASH",
    "um_stash_source",
    "ukmo__process_flags",
    "iris_extended_grid_mapping",
]

# TODO: whenever we advance to CF-1.11 we should then discuss a completion date
#  for the deprecation of Rotated Mercator in coord_systems.py and
#  _nc_load_rules/helpers.py .
CF_CONVENTIONS_VERSION = "CF-1.7"

_FactoryDefn = collections.namedtuple(
    "_FactoryDefn", ("primary", "std_name", "formula_terms_format")
)
_FACTORY_DEFNS = {
    AtmosphereSigmaFactory: _FactoryDefn(
        primary="sigma",
        std_name="atmosphere_sigma_coordinate",
        formula_terms_format="ptop: {pressure_at_top} sigma: {sigma} "
        "ps: {surface_air_pressure}",
    ),
    HybridHeightFactory: _FactoryDefn(
        primary="delta",
        std_name="atmosphere_hybrid_height_coordinate",
        formula_terms_format="a: {delta} b: {sigma} orog: {orography}",
    ),
    HybridPressureFactory: _FactoryDefn(
        primary="delta",
        std_name="atmosphere_hybrid_sigma_pressure_coordinate",
        formula_terms_format="ap: {delta} b: {sigma} ps: {surface_air_pressure}",
    ),
    OceanSigmaZFactory: _FactoryDefn(
        primary="zlev",
        std_name="ocean_sigma_z_coordinate",
        formula_terms_format="sigma: {sigma} eta: {eta} depth: {depth} "
        "depth_c: {depth_c} nsigma: {nsigma} zlev: {zlev}",
    ),
    OceanSigmaFactory: _FactoryDefn(
        primary="sigma",
        std_name="ocean_sigma_coordinate",
        formula_terms_format="sigma: {sigma} eta: {eta} depth: {depth}",
    ),
    OceanSFactory: _FactoryDefn(
        primary="s",
        std_name="ocean_s_coordinate",
        formula_terms_format="s: {s} eta: {eta} depth: {depth} a: {a} b: {b} "
        "depth_c: {depth_c}",
    ),
    OceanSg1Factory: _FactoryDefn(
        primary="s",
        std_name="ocean_s_coordinate_g1",
        formula_terms_format="s: {s} c: {c} eta: {eta} depth: {depth} "
        "depth_c: {depth_c}",
    ),
    OceanSg2Factory: _FactoryDefn(
        primary="s",
        std_name="ocean_s_coordinate_g2",
        formula_terms_format="s: {s} c: {c} eta: {eta} depth: {depth} "
        "depth_c: {depth_c}",
    ),
}


class CFNameCoordMap:
    """Provide a simple CF name to CF coordinate mapping."""

    _Map = collections.namedtuple("_Map", ["name", "coord"])

    def __init__(self):
        self._map = []

    def append(self, name, coord):
        """Append the given name and coordinate pair to the mapping.

        Parameters
        ----------
        name :
            CF name of the associated coordinate.
        coord :
            The coordinate of the associated CF name.

        Returns
        -------
        None.

        """
        self._map.append(CFNameCoordMap._Map(name, coord))

    @property
    def names(self):
        """Return all the CF names."""
        return [pair.name for pair in self._map]

    @property
    def coords(self):
        """Return all the coordinates."""
        return [pair.coord for pair in self._map]

    def name(self, coord):
        """Return the CF name, given a coordinate, or None if not recognised.

        Parameters
        ----------
        coord :
            The coordinate of the associated CF name.

        Returns
        -------
        Coordinate or None.

        """
        result = None
        for pair in self._map:
            if coord == pair.coord:
                result = pair.name
                break
        return result

    def coord(self, name):
        """Return the coordinate, given a CF name, or None if not recognised.

        Parameters
        ----------
        name :
            CF name of the associated coordinate, or None if not recognised.

        Returns
        -------
        CF name or None.

        """
        result = None
        for pair in self._map:
            if name == pair.name:
                result = pair.coord
                break
        return result


# NOTE : this matches :class:`iris.mesh.MeshXY.ELEMENTS`,
# but in the preferred order for coord/connectivity variables in the file.
MESH_ELEMENTS = ("node", "edge", "face")


class SaverFillValueWarning(iris.warnings.IrisSaverFillValueWarning):
    """Backwards compatible form of :class:`iris.warnings.IrisSaverFillValueWarning`."""

    # TODO: remove at the next major release.
    pass


def marshal_attributes(cubes, local_keys, format_name):
    """Decide which cube attributes a store holds globally, and which locally.

    A store holds one value of a global attribute, and a list of cubes may
    carry several. Settling that is cube logic, not storage logic, so every
    ``_CFSaver`` subclass's own ``save`` function calls this and nothing
    does it twice.

    Which half of this does the work depends on
    :attr:`iris.FUTURE.save_split_attrs`.

    * **Enabled.** ``_CFSaver.write`` writes no global attributes at all,
      and the caller writes the returned `global_attributes` once, after the
      last cube. An attribute the cubes disagree about cannot be global, so
      it is demoted to a local attribute on each cube - on a *copy* of each
      cube, so that the caller's own cubes are untouched.
    * **Legacy.** ``_CFSaver.write`` writes each cube's global attributes
      as it goes, so the returned `global_attributes` is empty and the work
      is to extend `local_keys` with every attribute the cubes do not all
      agree on. Without that, one store-wide value would silently become
      whichever cube was written last.

    Parameters
    ----------
    cubes : :class:`iris.cube.CubeList`
        The cubes about to be saved.
    local_keys : iterable of str or None
        Attribute names the caller asked to be local to a data variable.
        Ignored when split-attribute saving is enabled, which is what makes
        the keyword deprecated.
    format_name : str
        How to name the target format in the legacy mode's deprecation
        warning, e.g. ``"netcdf"``.

    Returns
    -------
    cubes : :class:`iris.cube.CubeList`
        The cubes to save, which are the ones passed in unless an attribute
        had to be demoted.
    local_keys : set of str
        To pass to each ``_CFSaver.write`` call.
    global_attributes : dict
        To write once, after the last cube. Empty in the legacy mode.

    Warns
    -----
    IrisSaveWarning
        Naming the global attributes demoted to local ones, and any that a
        local attribute of the same name blocked entirely.
    IrisDeprecation
        In the legacy mode, which is deprecated since Iris 3.8.

    """
    # TODO: when iris.FUTURE.save_split_attrs is removed, the whole legacy
    #  half of this function goes with it, along with `local_keys` and
    #  `format_name`.
    global_attributes: dict = {}

    if FUTURE.save_split_attrs:
        # 'local_keys' is not used in this case.
        local_keys = set()

        # Find any collisions in the cube global attributes and "demote" all those
        # to local attributes (where possible, else warn they are lost).
        # N.B. "collision" includes when not all cubes *have* that attribute.
        global_names = set()
        for cube in cubes:
            global_names |= set(cube.attributes.globals.keys())

        # Find any global attributes which are not the same on *all* cubes.
        def attr_values_equal(val1, val2):
            # An equality test which also works when some values are numpy arrays (!)
            return iris.util._attribute_equal(val1, val2)

        cube0 = cubes[0]
        invalid_globals = set(
            [
                attrname
                for attrname in global_names
                if not all(
                    attr_values_equal(
                        cube.attributes.globals.get(attrname),
                        cube0.attributes.globals.get(attrname),
                    )
                    for cube in cubes[1:]
                )
            ]
        )

        # Establish all the global attributes which we will write to the file (at end).
        global_attributes = {
            attr: cube0.attributes.globals.get(attr)
            for attr in global_names - invalid_globals
        }
        if invalid_globals:
            # Some cubes have different global attributes: modify cubes as required.
            warnings.warn(
                f"Saving the cube global attributes {sorted(invalid_globals)} as local "
                "(i.e. data-variable) attributes, where possible, since they are not "
                "the same on all input cubes.",
                category=iris.warnings.IrisSaveWarning,
            )
            cubes = cubes.copy()  # avoiding modifying the actual input arg.
            for i_cube in range(len(cubes)):
                # We iterate over cube *index*, so we can replace the list entries with
                # with cube *copies* -- just to avoid changing our call args.
                cube = cubes[i_cube]
                demote_attrs = set(cube.attributes.globals) & invalid_globals
                if any(demote_attrs):
                    # Catch any demoted attrs where there is already a local version
                    blocked_attrs = demote_attrs & set(cube.attributes.locals)
                    if blocked_attrs:
                        warnings.warn(
                            f"Global cube attributes {sorted(blocked_attrs)} "
                            f'of cube "{cube.name()}" were not saved, overlaid '
                            "by existing local attributes with the same names.",
                            category=iris.warnings.IrisSaveWarning,
                        )
                    demote_attrs -= blocked_attrs
                    if demote_attrs:
                        # This cube contains some 'demoted' global attributes.
                        # Replace input cube with a copy, so we can modify attributes.
                        cube = cube.copy()
                        cubes[i_cube] = cube
                        for attr in demote_attrs:
                            # move global to local
                            value = cube.attributes.globals.pop(attr)
                            cube.attributes.locals[attr] = value

    else:
        # Legacy mode: calculate "local_keys" to control which attributes are local
        # and which global.
        message = (
            f"Saving to {format_name} with legacy-style attribute handling for "
            "backwards compatibility.\n"
            "This mode is deprecated since Iris 3.8, and will eventually be removed.\n"
            "Please consider enabling the new split-attributes handling mode, by "
            "setting 'iris.FUTURE.save_split_attrs = True'."
        )
        warn_deprecated(message)

        if local_keys is None:
            local_keys = set()
        else:
            local_keys = set(local_keys)

        # Determine the attribute keys that are common across all cubes and
        # thereby extend the collection of local_keys for attributes
        # that should be attributes on data variables.
        attributes = cubes[0].attributes
        common_keys = set(attributes)
        for cube in cubes[1:]:
            keys = set(cube.attributes)
            local_keys.update(keys.symmetric_difference(common_keys))
            common_keys.intersection_update(keys)
            different_value_keys = []
            for key in common_keys:
                if not iris.util._attribute_equal(
                    attributes[key], cube.attributes[key]
                ):
                    different_value_keys.append(key)
            common_keys.difference_update(different_value_keys)
            local_keys.update(different_value_keys)

    return cubes, local_keys, global_attributes


class _CFSaver:
    """A manager for saving cubes to a CF-conforming store.

    Private: a format's own saver subclasses this and is the public class.
    The module docstring says why.
    """

    def __init__(self, dataset, compute=True, *, owns_dataset=True):
        """Manage saving cubes into an open store.

        Parameters
        ----------
        dataset : :class:`~iris.fileformats.cf.dataset.CFDataset`
            An open, writable store. A subclass opens it and passes it here.
        compute : bool, default=True
            If ``True``, delayed variable saves will be completed on exit from
            the saver context, after the store is closed, equivalent to
            :meth:`complete`.

            If ``False``, the store is created and closed without writing the
            data of variables whose source data was lazy. Those writes can be
            completed later, see :meth:`delayed_completion`.
        owns_dataset : bool, default=True
            Whether leaving the context should close the store. ``False`` when
            the caller supplied an already-open store and will close it itself.

        """
        # All persistent variables
        #: CF name mapping with iris coordinates
        self._name_coord_map = CFNameCoordMap()
        #: Map of dimensions to characteristic coordinates with which they are identified
        self._dim_names_and_coords = CFNameCoordMap()
        #: List of grid mappings added to the file
        self._coord_systems = []
        #: A dictionary, listing dimension names and corresponding length
        self._existing_dim = {}
        #: A map from meshes to their actual file dimensions (names).
        # NB: might not match those of the mesh, if they were 'incremented'.
        self._mesh_dims = {}
        #: A dictionary, mapping formula terms to owner cf variable name
        self._formula_terms_cache = {}
        #: Whether to complete delayed saves on exit.
        self.compute = compute
        # A list of delayed writes for lazy saving
        # a list of couples (source, target).
        self._delayed_writes = []

        self._dataset = dataset
        self._owns_dataset = owns_dataset
        #: Target filepath
        self.filepath = dataset.location

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        """Flush any buffered data to the store before closing."""
        self._flush_before_close()
        self._dataset.sync()
        if self._owns_dataset:
            # Only close if the saver opened it.
            self._dataset.close()
            # Complete after closing, if required
            if self.compute:
                self.complete()

    def _flush_before_close(self) -> None:
        """Write anything that cannot be written once the store is closed.

        Nothing, here. A store that cannot be reopened for a deferred write
        overrides this and writes its lazy data while it is still open.

        """
        pass

    def write(
        self,
        cube,
        local_keys=None,
        unlimited_dimensions=None,
        packing=None,
        fill_value=None,
        compression_kwargs=None,
        **encoding,
    ):
        """Write a cube into the store.

        Parameters
        ----------
        cube : :class:`iris.cube.Cube`
            The cube to save.
        local_keys : iterable of str, optional
            Cube attribute keys that become attributes of the data variable
            rather than global attributes.

            .. note::

                Has no effect if :attr:`iris.FUTURE.save_split_attrs` is ``True``.
        unlimited_dimensions : iterable of str and/or :class:`iris.coords.Coord`, optional
            Coordinate names, or coordinates, whose dimensions are saved with
            unlimited length. By default, none are.
        packing : type or str or dict or list, optional
            How to pack the data, if at all. See
            :meth:`iris.fileformats.netcdf.saver.Saver.write`.
        fill_value : optional
            The value for the data variable's ``_FillValue`` attribute.
        compression_kwargs : dict, optional
            Storage keywords applied to *every* variable this cube creates, not
            only its data variable. The store's own vocabulary.
        **encoding
            Storage keywords applied to the data variable alone. The store's
            own vocabulary.

        Returns
        -------
        None

        """
        # TODO: when iris.FUTURE.save_split_attrs defaults to True, we can deprecate the
        #  "local_keys" arg, and finally remove it when we finally remove the
        #  save_split_attrs switch.
        if unlimited_dimensions is None:
            unlimited_dimensions = []

        cf_profile_available = iris.site_configuration.get("cf_profile") not in [
            None,
            False,
        ]
        if cf_profile_available:
            # Perform a CF profile of the cube. This may result in an exception
            # being raised if mandatory requirements are not satisfied.
            profile = iris.site_configuration["cf_profile"](cube)

        # Ensure that attributes are CF compliant and if possible to make them
        # compliant.
        self.check_attribute_compliance(cube, cube.dtype)
        for coord in cube.coords():
            self.check_attribute_compliance(coord, coord.dtype)

        # Get suitable dimension names.
        mesh_dimensions, cube_dimensions = self._get_dim_names(cube)

        # Create all the CF-netCDF data dimensions.
        # Put mesh dims first, then non-mesh dims in cube-occurring order.
        nonmesh_dimensions = [
            dim for dim in cube_dimensions if dim not in mesh_dimensions
        ]
        all_dimensions = mesh_dimensions + nonmesh_dimensions
        self._create_cf_dimensions(cube, all_dimensions, unlimited_dimensions)

        # Group the storage keywords that apply to every variable, as they are
        # applied to other cube metadata as well as the cube data payload.
        compression_kwargs = dict(compression_kwargs or {})

        # Create the mesh components, if there is a mesh.
        # We do this before creating the data-var, so that mesh vars precede
        # data-vars in the file.
        cf_mesh_name = self._add_mesh(cube, compression_kwargs=compression_kwargs)

        # Create the associated cube CF-netCDF data variable.
        data_encoding = dict(encoding)
        data_encoding.update(compression_kwargs)
        cf_var_cube = self._create_cf_data_variable(
            cube,
            cube_dimensions,
            local_keys=local_keys,
            packing=packing,
            fill_value=fill_value,
            **data_encoding,
        )

        # Associate any mesh with the data-variable.
        # N.B. _add_mesh cannot do this, as we want to put mesh variables
        # before data-variables in the file.
        if cf_mesh_name is not None:
            cf_var_cube.attributes["mesh"] = cf_mesh_name
            cf_var_cube.attributes["location"] = cube.location

        # Add coordinate variables.
        self._add_dim_coords(cube, cube_dimensions)

        # Add the auxiliary coordinate variables and associate the data
        # variable to them
        self._add_aux_coords(
            cube, cf_var_cube, cube_dimensions, compression_kwargs=compression_kwargs
        )

        # Add the cell_measures variables and associate the data
        # variable to them
        self._add_cell_measures(cube, cf_var_cube, cube_dimensions)

        # Add the ancillary_variables variables and associate the data variable
        # to them
        self._add_ancillary_variables(
            cube, cf_var_cube, cube_dimensions, compression_kwargs=compression_kwargs
        )

        # Add the formula terms to the appropriate cf variables for each
        # aux factory in the cube.
        self._add_aux_factories(cube, cf_var_cube, cube_dimensions)

        if not iris.FUTURE.save_split_attrs:
            # In the "old" way, we update global attributes as we go.
            # Add data variable-only attribute names to local_keys.
            if local_keys is None:
                local_keys = set()
            else:
                local_keys = set(local_keys)
            local_keys.update(_CF_DATA_ATTRS, _UKMO_DATA_ATTRS, ["GRIB_PARAM"])

            # Add global attributes taking into account local_keys.
            # NOTE: we never need to "convert" managed global attributes, as they are all
            #  data-local type, i.e. included in "local_keys" above.
            cube_attributes = cube.attributes
            global_attributes = {
                k: v
                for k, v in cube_attributes.items()
                if (k not in local_keys and k.lower() != "conventions")
            }
            self.update_global_attributes(global_attributes)

        if cf_profile_available:
            cf_patch = iris.site_configuration.get("cf_patch")
            if cf_patch is not None:
                self._patch_dataset(cf_patch, profile, cf_var_cube)
            else:
                msg = "cf_profile is available but no {} defined.".format("cf_patch")
                warnings.warn(msg, category=iris.warnings.IrisCfSaveWarning)

    def _patch_dataset(self, cf_patch, profile, cf_var_cube) -> None:
        """Hand the site-configured patch hook the objects it expects.

        Parameters
        ----------
        cf_patch : callable
            The hook, from ``iris.site_configuration``.
        profile : object
            Whatever the configured ``cf_profile`` returned for this cube.
        cf_var_cube : :class:`~iris.fileformats.cf.dataset.CFDatasetVariable`
            The cube's data variable.

        Raises
        ------
        NotImplementedError
            Always, here. The hook has been documented as receiving netCDF4
            objects since Iris 1.3.

        """
        msg = (
            "The 'cf_patch' site configuration hook is documented as receiving "
            f"netCDF4 objects, and {self._dataset.format} has none to give it."
        )
        raise NotImplementedError(msg)

    def check_attribute_compliance(self, container, data_dtype):
        """Check attributte complliance."""

        def _coerce_value(val_attr, val_attr_value, data_dtype):
            val_attr_tmp = np.array(val_attr_value, dtype=data_dtype)
            if (val_attr_tmp != val_attr_value).any():
                msg = '"{}" is not of a suitable value ({})'
                raise ValueError(msg.format(val_attr, val_attr_value))
            return val_attr_tmp

        # Ensure that conflicting attributes are not provided.
        if (
            container.attributes.get("valid_min") is not None
            or container.attributes.get("valid_max") is not None
        ) and container.attributes.get("valid_range") is not None:
            msg = (
                'Both "valid_range" and "valid_min" or "valid_max" attributes present.'
            )
            raise ValueError(msg)

        # Ensure correct datatype
        for val_attr in ["valid_range", "valid_min", "valid_max"]:
            val_attr_value = container.attributes.get(val_attr)
            if val_attr_value is not None:
                val_attr_value = np.asarray(val_attr_value)
                self._ensure_valid_dtype(val_attr_value, val_attr, val_attr_value)
                if data_dtype.itemsize == 1:
                    # Allow signed integral type
                    if val_attr_value.dtype.kind == "i":
                        continue
                new_val = _coerce_value(val_attr, val_attr_value, data_dtype)
                container.attributes[val_attr] = new_val

    def update_global_attributes(self, attributes=None, **kwargs):
        """Update the CF global attributes.

        Update the CF global attributes based on the provided
        iterable/dictionary and/or keyword arguments.

        Parameters
        ----------
        attributes : dict or iterable of key, value pairs, optional
            CF global attributes to be updated.
        """
        # TODO: when when iris.FUTURE.save_split_attrs is removed, this routine will
        # only be called once: it can reasonably be renamed "_set_global_attributes",
        # and the 'kwargs' argument can be removed.
        if attributes is not None:
            # Handle sequence e.g. [('fruit', 'apple'), ...].
            if not hasattr(attributes, "keys"):
                attributes = dict(attributes)

            for attr_name in sorted(attributes):
                self._dataset.attributes[attr_name] = attributes[attr_name]

        for attr_name in sorted(kwargs):
            self._dataset.attributes[attr_name] = kwargs[attr_name]

    def _create_cf_dimensions(self, cube, dimension_names, unlimited_dimensions=None):
        """Create the CF-netCDF data dimensions.

        Parameters
        ----------
        cube : :class:`iris.cube.Cube`
            A :class:`iris.cube.Cube` in which to lookup coordinates.
        dimension_names :
        unlimited_dimensions : iterable of strings and/or :class:`iris.coords.Coord` objects):
            List of coordinates to make unlimited (None by default).

        Returns
        -------
        None.

        """
        unlimited_dim_names = []
        if unlimited_dimensions is not None:
            for coord in unlimited_dimensions:
                try:
                    coord = cube.coord(name_or_coord=coord, dim_coords=True)
                except iris.exceptions.CoordinateNotFoundError:
                    # coordinate isn't used for this cube, but it might be
                    # used for a different one
                    pass
                else:
                    dim_name = self._get_element_variable_name(cube, coord)
                    unlimited_dim_names.append(dim_name)

        for dim_name in dimension_names:
            # NOTE: these dim-names have been chosen by _get_dim_names, and
            # were already checked+fixed to avoid any name collisions.
            if dim_name not in self._dataset.dimensions:
                if dim_name in unlimited_dim_names:
                    size = None
                else:
                    size = self._existing_dim[dim_name]
                self._dataset.create_dimension(dim_name, size)

    def _add_mesh(self, cube_or_mesh, /, *, compression_kwargs=None):
        """Add the cube's mesh, and all related variables to the dataset.

        Add the cube's mesh, and all related variables to the dataset.
        Includes all the mesh-element coordinate and connectivity variables.

        .. note::

            Here, we do *not* add the relevant referencing attributes to the
            data-variable, because we want to create the data-variable later.

        Parameters
        ----------
        cube_or_mesh : :class:`iris.cube.Cube` or :class:`iris.mesh.MeshXY`
            The Cube or Mesh being saved to the netCDF file.
        compression_kwargs : dict, optional
            NetCDF data compression keyword arguments.

        Returns
        -------
        str or None
            The name of the mesh variable created, or None if the cube does not
            have a mesh.

        """
        cf_mesh_name = None

        # Do cube- or -mesh-based save
        from iris.cube import Cube
        from iris.mesh.components import _MeshIndexSet

        if isinstance(cube_or_mesh, Cube):
            mesh = cube_or_mesh.mesh
        else:
            mesh = cube_or_mesh

        if isinstance(mesh, _MeshIndexSet):
            msg = "_MeshIndexSet saving is not yet supported. See https://github.com/SciTools/iris/issues/6123"
            raise ValueError(msg)

        if mesh:
            cf_mesh_name = self._name_coord_map.name(mesh)
            if cf_mesh_name is None:
                # Not already present : create it
                cf_mesh_name = self._create_mesh(mesh)
                self._name_coord_map.append(cf_mesh_name, mesh)

                cf_mesh_var = self._dataset.variables[cf_mesh_name]

                # Get the mesh-element dim names.
                mesh_dims = self._mesh_dims[mesh]

                # Add all the element coordinate variables.
                for location in MESH_ELEMENTS:
                    coords_meshobj_attr = f"{location}_coords"
                    coords_file_attr = f"{location}_coordinates"
                    mesh_coords = getattr(mesh, coords_meshobj_attr, None)
                    if mesh_coords:
                        coord_names = []
                        for coord in mesh_coords:
                            if coord is None:
                                continue  # an awkward thing that mesh.coords does
                            coord_name = self._create_generic_cf_array_var(
                                mesh,
                                [],
                                coord,
                                element_dims=(mesh_dims[location],),
                                compression_kwargs=compression_kwargs,
                            )
                            # Only created once per file, but need to fetch the
                            #  name later in _add_inner_related_vars().
                            self._name_coord_map.append(coord_name, coord)
                            coord_names.append(coord_name)
                        # Record the coordinates (if any) on the mesh variable.
                        if coord_names:
                            coord_names = " ".join(coord_names)
                            cf_mesh_var.attributes[coords_file_attr] = coord_names

                # Add all the connectivity variables.
                # pre-fetch the set + ignore "None"s, which are empty slots.
                conns = [conn for conn in mesh.all_connectivities if conn is not None]
                for conn in conns:
                    # Get the connectivity role, = "{loc1}_{loc2}_connectivity".
                    cf_conn_attr_name = conn.cf_role
                    loc_from, loc_to, _ = cf_conn_attr_name.split("_")
                    # Construct a trailing dimension name.
                    last_dim = f"{cf_mesh_name}_{loc_from}_N_{loc_to}s"
                    # Create if it does not already exist.
                    if last_dim not in self._dataset.dimensions:
                        while last_dim in self._dataset.variables:
                            # Also avoid collision with variable names.
                            # See '_get_dim_names' for reason.
                            last_dim = self._increment_name(last_dim)
                        length = conn.shape[1 - conn.location_axis]
                        self._dataset.create_dimension(last_dim, length)

                    # Create variable.
                    # NOTE: for connectivities *with missing points*, this will use a
                    # fixed standard fill-value of -1.  In that case, we create the
                    # variable with a '_FillValue' property, which can only be done
                    # when it is first created.
                    loc_dim_name = mesh_dims[loc_from]
                    conn_dims = (loc_dim_name, last_dim)
                    if conn.location_axis == 1:
                        # Has the 'other' dimension order, =reversed
                        conn_dims = conn_dims[::-1]
                    if iris.util.is_masked(conn.core_indices()):
                        # Flexible mesh.
                        fill_value = -1
                    else:
                        fill_value = None
                    cf_conn_name = self._create_generic_cf_array_var(
                        mesh,
                        [],
                        conn,
                        element_dims=conn_dims,
                        fill_value=fill_value,
                        compression_kwargs=compression_kwargs,
                    )
                    # Add essential attributes to the Connectivity variable.
                    cf_conn_var = self._dataset.variables[cf_conn_name]
                    cf_conn_var.attributes["cf_role"] = cf_conn_attr_name
                    cf_conn_var.attributes["start_index"] = conn.start_index

                    # Record the connectivity on the parent mesh var.
                    cf_mesh_var.attributes[cf_conn_attr_name] = cf_conn_name
                    # If the connectivity had the 'alternate' dimension order, add the
                    # relevant dimension property
                    if conn.location_axis == 1:
                        loc_dim_attr = f"{loc_from}_dimension"
                        # Should only get here once.
                        assert loc_dim_attr not in cf_mesh_var.attributes
                        cf_mesh_var.attributes[loc_dim_attr] = loc_dim_name

        return cf_mesh_name

    def _add_inner_related_vars(
        self,
        cube,
        cf_var_cube,
        dimension_names,
        coordlike_elements,
        /,
        *,
        compression_kwargs=None,
    ) -> None:
        """Create a set of variables for aux-coords, ancillaries or cell-measures.

        Create a set of variables for aux-coords, ancillaries or cell-measures,
        and attach them to the parent data variable.

        """
        if coordlike_elements:
            # Choose the appropriate parent attribute
            elem_type = type(coordlike_elements[0])
            if elem_type in (AuxCoord, DimCoord):
                role_attribute_name = "coordinates"
            elif elem_type == AncillaryVariable:
                role_attribute_name = "ancillary_variables"
            else:
                # We *only* handle aux-coords, cell-measures and ancillaries
                assert elem_type == CellMeasure
                role_attribute_name = "cell_measures"

            # Add CF-netCDF variables for the given cube components.
            element_names = []
            for element in sorted(
                coordlike_elements, key=lambda element: element.name()
            ):
                # Reuse, or create, the associated CF-netCDF variable.
                cf_name = self._name_coord_map.name(element)
                if cf_name is None:
                    # Not already present : create it
                    cf_name = self._create_generic_cf_array_var(
                        cube,
                        dimension_names,
                        element,
                        compression_kwargs=compression_kwargs,
                    )
                    self._name_coord_map.append(cf_name, element)

                if role_attribute_name == "cell_measures":
                    # In the case of cell-measures, the attribute entries are not just
                    # a var_name, but each have the form "<measure>: <varname>".
                    cf_name = "{}: {}".format(element.measure, cf_name)

                element_names.append(cf_name)

            # Add CF-netCDF references to the primary data variable.
            if element_names:
                variable_names = " ".join(sorted(element_names))
                cf_var_cube.attributes[role_attribute_name] = variable_names

    def _add_aux_coords(
        self, cube, cf_var_cube, dimension_names, /, *, compression_kwargs=None
    ) -> None:
        """Add aux. coordinate to the dataset and associate with the data variable.

        Parameters
        ----------
        cube : :class:`iris.cube.Cube`
            A :class:`iris.cube.Cube` to be saved to a netCDF file.
        cf_var_cube : :class:`~iris.fileformats.netcdf._dataset.NetCDFDatasetVariable`
            A cf variable cube representation.
        dimension_names : list
            Names associated with the dimensions of the cube.
        compression_kwargs : dict, optional
            NetCDF data compression keyword arguments.

        """
        # Exclude any mesh coords, which are bundled in with the aux-coords.
        coords_to_add = [
            coord for coord in cube.aux_coords if not hasattr(coord, "mesh")
        ]

        # Include any relevant mesh location coordinates.
        mesh: MeshXY | None = getattr(cube, "mesh")  # type: ignore[annotation-unchecked]
        mesh_location: str | None = getattr(cube, "location")  # type: ignore[annotation-unchecked]
        if mesh and mesh_location:
            location_coords: MeshNodeCoords | MeshEdgeCoords | MeshFaceCoords = getattr(
                mesh, f"{mesh_location}_coords"
            )  # type: ignore[annotation-unchecked]
            coords_to_add.extend(list(location_coords))

        return self._add_inner_related_vars(
            cube,
            cf_var_cube,
            dimension_names,
            coords_to_add,
            compression_kwargs=compression_kwargs,
        )

    def _add_cell_measures(self, cube, cf_var_cube, dimension_names):
        """Add cell measures to the dataset and associate with the data variable.

        Parameters
        ----------
        cube : :class:`iris.cube.Cube`
            A :class:`iris.cube.Cube` to be saved to a netCDF file.
        cf_var_cube : :class:`~iris.fileformats.netcdf._dataset.NetCDFDatasetVariable`
            A cf variable cube representation.
        dimension_names : list
            Names associated with the dimensions of the cube.
        """
        return self._add_inner_related_vars(
            cube,
            cf_var_cube,
            dimension_names,
            cube.cell_measures(),
        )

    def _add_ancillary_variables(
        self, cube, cf_var_cube, dimension_names, /, *, compression_kwargs=None
    ):
        """Add ancillary variables measures to the dataset and associate with the data variable.

        Parameters
        ----------
        cube : :class:`iris.cube.Cube`
            A :class:`iris.cube.Cube` to be saved to a netCDF file.
        cf_var_cube : :class:`~iris.fileformats.netcdf._dataset.NetCDFDatasetVariable`
            A cf variable cube representation.
        dimension_names : list
            Names associated with the dimensions of the cube.
        compression_kwargs : dict, optional
            NetCDF data compression keyword arguments.

        """
        return self._add_inner_related_vars(
            cube,
            cf_var_cube,
            dimension_names,
            cube.ancillary_variables(),
            compression_kwargs=compression_kwargs,
        )

    def _add_dim_coords(self, cube, dimension_names):
        """Add coordinate variables to NetCDF dataset.

        Parameters
        ----------
        cube : :class:`iris.cube.Cube`
            A :class:`iris.cube.Cube` to be saved to a netCDF file.
        dimension_names : list
            Names associated with the dimensions of the cube.
        """
        # Ensure we create the netCDF coordinate variables first.
        for coord in cube.dim_coords:
            # Create the associated coordinate CF-netCDF variable.
            if coord not in self._name_coord_map.coords:
                # Not already present : create it
                cf_name = self._create_generic_cf_array_var(
                    cube, dimension_names, coord
                )
                self._name_coord_map.append(cf_name, coord)

    def _add_aux_factories(self, cube, cf_var_cube, dimension_names):
        """Represent the presence of dimensionless vertical coordinates.

        Modify the variables of the NetCDF dataset to represent
        the presence of dimensionless vertical coordinates based on
        the aux factories of the cube (if any).

        Parameters
        ----------
        cube : :class:`iris.cube.Cube`
            A :class:`iris.cube.Cube` to be saved to a netCDF file.
        cf_var_cube : :class:`~iris.fileformats.netcdf._dataset.NetCDFDatasetVariable`
            CF variable cube representation.
        dimension_names : list
            Names associated with the dimensions of the cube.
        """
        primaries = []
        for factory in cube.aux_factories:
            factory_defn = _FACTORY_DEFNS.get(type(factory), None)
            if factory_defn is None:
                msg = "Unable to determine formula terms for AuxFactory: {!r}".format(
                    factory
                )
                warnings.warn(msg, category=iris.warnings.IrisSaveWarning)
            else:
                # Override `standard_name`, `long_name`, and `axis` of the
                # primary coord that signals the presence of a dimensionless
                # vertical coord, then set the `formula_terms` attribute.
                primary_coord = factory.dependencies[factory_defn.primary]
                if primary_coord in primaries:
                    msg = (
                        "Cube {!r} has multiple aux factories that share "
                        "a common primary coordinate {!r}. Unable to save "
                        "to netCDF as having multiple formula terms on a "
                        "single coordinate is not supported."
                    )
                    raise ValueError(msg.format(cube, primary_coord.name()))
                primaries.append(primary_coord)

                cf_name = self._name_coord_map.name(primary_coord)
                cf_var = self._dataset.variables[cf_name]

                term_varnames = {
                    term: self._name_coord_map.name(coord)
                    for term, coord in factory.dependencies.items()
                }
                formula_terms = factory_defn.formula_terms_format.format(
                    **term_varnames
                )
                std_name = factory_defn.std_name

                if "formula_terms" in cf_var.attributes:
                    if (
                        cf_var.attributes["formula_terms"] != formula_terms
                        or cf_var.attributes["standard_name"] != std_name
                    ):
                        # TODO: We need to resolve this corner-case where
                        #  the dimensionless vertical coordinate containing
                        #  the formula_terms is a dimension coordinate of
                        #  the associated cube and a new alternatively named
                        #  dimensionless vertical coordinate is required
                        #  with new formula_terms and a renamed dimension.
                        if cf_name in dimension_names:
                            msg = "Unable to create dimensonless vertical coordinate."
                            raise ValueError(msg)
                        key = (cf_name, std_name, formula_terms)
                        name = self._formula_terms_cache.get(key)
                        if name is None:
                            # Create a new variable
                            name = self._create_generic_cf_array_var(
                                cube, dimension_names, primary_coord
                            )
                            cf_var = self._dataset.variables[name]
                            cf_var.attributes["standard_name"] = std_name
                            cf_var.attributes["axis"] = "Z"
                            # Update the formula terms.
                            ft = formula_terms.split()
                            ft = [name if t == cf_name else t for t in ft]
                            cf_var.attributes["formula_terms"] = " ".join(ft)
                            # Update the cache.
                            self._formula_terms_cache[key] = name
                        # Update the associated cube variable.
                        coords = cf_var_cube.attributes["coordinates"].split()
                        coords = [name if c == cf_name else c for c in coords]
                        cf_var_cube.attributes["coordinates"] = " ".join(coords)
                else:
                    cf_var.attributes["standard_name"] = std_name
                    cf_var.attributes["axis"] = "Z"
                    cf_var.attributes["formula_terms"] = formula_terms

                if FUTURE.derived_bounds:
                    # ensure that the primary variable *bounds*, if any, obey the CF
                    #  encoding rule : the bounds variable of a parametric coordinate
                    #  must itself have a "formula_terms" attribute.
                    # See : https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#boundaries-and-formula-terms
                    bounds_varname = cf_var.attributes.get("bounds")
                    cf_bounds_var = self._dataset.variables.get(bounds_varname, None)
                    if (
                        cf_bounds_var is not None
                        and cf_bounds_var.attributes.get("formula_terms") is None
                    ):
                        # We need a bounds formula, and there is none already attached.
                        # Construct and add one, mirroring the main formula.
                        def boundsterm_varname(term_varname):
                            """Identify the boundsvar for a given term var."""
                            # First establish a fallback default, meaning 'no bounds'.
                            result = term_varname
                            # Follow links (if they exist) to find the bounds var.
                            termvar = self._dataset.variables.get(term_varname)
                            boundsname = (
                                None
                                if termvar is None
                                else termvar.attributes.get("bounds")
                            )
                            if boundsname in self._dataset.variables:
                                result = boundsname
                            return result

                        boundsterm_varnames = {
                            term: boundsterm_varname(term_varname)
                            for term, term_varname in term_varnames.items()
                        }
                        bounds_formula_terms = factory_defn.formula_terms_format.format(
                            **boundsterm_varnames
                        )
                        cf_bounds_var.attributes["formula_terms"] = bounds_formula_terms

    def _get_dim_names(self, cube_or_mesh):
        """Determine suitable CF-netCDF data dimension names.

        Parameters
        ----------
        cube_or_mesh : :class:`iris.cube.Cube` or :class:`iris.mesh.MeshXY`
            The Cube or Mesh being saved to the netCDF file.

        Returns
        -------
        mesh_dimensions : list of str
            A list of the mesh dimensions of the attached mesh, if any.
        cube_dimensions : list of str
            A lists of dimension names for each dimension of the cube.

        Notes
        -----
        The returned lists are in the preferred file creation order.
        One of the mesh dimensions will typically also appear in the cube
        dimensions.

        """

        def record_dimension(names_list, dim_name, length, matching_coords=None):
            """Record a file dimension, its length and associated "coordinates".

            Record a file dimension, its length and associated "coordinates"
            (which may in fact also be connectivities).

            If the dimension has been seen already, check that it's length
            matches the earlier finding.

            """
            if matching_coords is None:
                matching_coords = []
            if dim_name not in self._existing_dim:
                self._existing_dim[dim_name] = length
            else:
                # Make sure we never re-write one, though it's really not
                # clear how/if this could ever actually happen.
                existing_length = self._existing_dim[dim_name]
                if length != existing_length:
                    msg = (
                        "Netcdf saving error : existing dimension "
                        f'"{dim_name}" has length {existing_length}, '
                        f"but occurrence in cube {cube} has length {length}"
                    )
                    raise ValueError(msg)

            # Record given "coords" (sort-of, maybe connectivities) to be
            # identified with this dimension: add to the already-seen list.
            existing_coords = self._dim_names_and_coords.coords
            for coord in matching_coords:
                if coord not in existing_coords:
                    self._dim_names_and_coords.append(dim_name, coord)

            # Add the latest name to the list passed in.
            names_list.append(dim_name)

        # Choose cube or mesh saving.
        from iris.cube import Cube

        if isinstance(cube_or_mesh, Cube):
            # there is a mesh, only if the cube has one
            cube = cube_or_mesh
            mesh = cube.mesh
        else:
            # there is no cube, only a mesh
            cube = None
            mesh = cube_or_mesh

        # Get info on mesh, first.
        mesh_dimensions = []
        if mesh is None:
            cube_mesh_dim = None
        else:
            # Identify all the mesh dimensions.
            mesh_location_dimnames = {}
            # NOTE: one of these will be a cube dimension, but that one does not
            # get any special handling.  We *do* want to list/create them in a
            # definite order (node,edge,face), and before non-mesh dimensions.
            for location in MESH_ELEMENTS:
                # Find if this location exists in the mesh, and a characteristic
                # coordinate to identify it with.
                # To use only _required_ UGRID components, we use a location
                # coord for nodes, but a connectivity for faces/edges
                if location == "node":
                    # For nodes, identify the dim with a coordinate variable.
                    dim_coords = mesh.coords(location="node")
                else:
                    # For face/edge, use the relevant "optionally required"
                    # connectivity variable.
                    cf_role = f"{location}_node_connectivity"
                    dim_coords = mesh.connectivities(cf_role=cf_role)
                if len(dim_coords) > 0:
                    # As the mesh contains this location, we want to include this
                    # dim in our returned mesh dims.
                    # We should have 1 identifying variable (of either type).
                    dim_element = dim_coords[0]
                    dim_name = self._dim_names_and_coords.name(dim_element)
                    if dim_name is None:
                        # No existing dim matches this, so assign a new name
                        if location == "node":
                            # always 1-d
                            (dim_length,) = dim_element.shape
                        else:
                            # extract source dim, respecting dim-ordering
                            dim_length = dim_element.shape[dim_element.location_axis]
                        # Name it for the relevant mesh dimension
                        location_dim_attr = f"{location}_dimension"
                        dim_name = getattr(mesh, location_dim_attr)
                        # NOTE: This cannot currently be empty, as a MeshXY
                        # "invents" dimension names which were not given.
                        assert dim_name is not None
                        # Ensure it is a valid variable name.
                        dim_name = self.cf_valid_var_name(dim_name)
                        # Disambiguate if it has the same name as an existing
                        # dimension.
                        # NOTE: *OR* if it matches the name of an existing file
                        # variable.  Because there is a bug ...
                        # See https://github.com/Unidata/netcdf-c/issues/1772
                        # N.B. the workarounds here *ONLY* function because the
                        # caller (write) will not create any more variables
                        # in between choosing dim names (here), and creating
                        # the new dims (via '_create_cf_dimensions').
                        while (
                            dim_name in self._existing_dim
                            or dim_name in self._dataset.variables
                        ):
                            dim_name = self._increment_name(dim_name)

                        # Record the new dimension.
                        record_dimension(
                            mesh_dimensions, dim_name, dim_length, dim_coords
                        )

                    # Store the mesh dims indexed by location
                    mesh_location_dimnames[location] = dim_name

            if cube is None:
                cube_mesh_dim = None
            else:
                # Finally, identify the cube dimension which maps to the mesh: this
                # is used below to recognise the mesh among the cube dimensions.
                any_mesh_coord = cube.coords(mesh_coords=True)[0]
                (cube_mesh_dim,) = any_mesh_coord.cube_dims(cube)

            # Record actual file dimension names for each mesh saved.
            self._mesh_dims[mesh] = mesh_location_dimnames

        # Get the cube dimensions, in order.
        cube_dimensions = []
        if cube is not None:
            for dim in range(cube.ndim):
                if dim == cube_mesh_dim:
                    # Handle a mesh dimension: we already named this.
                    dim_coords = []
                    dim_name = self._mesh_dims[mesh][cube.location]
                else:
                    # Get a name from the dim-coord (if any).
                    dim_coords = cube.coords(dimensions=dim, dim_coords=True)
                    if dim_coords:
                        # Derive a dim name from a coord.
                        coord = dim_coords[0]  # always have at least one

                        # Add only dimensions that have not already been added.
                        dim_name = self._dim_names_and_coords.name(coord)
                        if dim_name is None:
                            # Not already present : create  a unique dimension name
                            # from the coord.
                            dim_name = self._get_element_variable_name(cube, coord)
                            # Disambiguate if it has the same name as an
                            # existing dimension.
                            # OR if it matches an existing file variable name.
                            # NOTE: check against variable names is needed
                            # because of a netcdf bug ... see note in the
                            # mesh dimensions block above.
                            while (
                                dim_name in self._existing_dim
                                or dim_name in self._dataset.variables
                            ):
                                dim_name = self._increment_name(dim_name)

                    else:
                        # No CF-netCDF coordinates describe this data dimension.
                        # Make up a new, distinct dimension name
                        dim_name = f"dim{dim}"
                        # Increment name if conflicted with one already existing
                        # (or planned)
                        # NOTE: check against variable names is needed because
                        # of a netcdf bug ... see note in the mesh dimensions
                        # block above.
                        while (
                            dim_name in self._existing_dim
                            and (self._existing_dim[dim_name] != cube.shape[dim])
                        ) or dim_name in self._dataset.variables:
                            dim_name = self._increment_name(dim_name)

                # Record the dimension.
                record_dimension(cube_dimensions, dim_name, cube.shape[dim], dim_coords)

        return mesh_dimensions, cube_dimensions

    @staticmethod
    def cf_valid_var_name(var_name):
        """Return a valid CF var_name given a potentially invalid name.

        Parameters
        ----------
        var_name : str
            The var_name to normalise.

        Returns
        -------
        str
            The var_name suitable for passing through for variable creation.

        """
        # Replace invalid characters with an underscore ("_").
        var_name = re.sub(r"[^a-zA-Z0-9]", "_", var_name)
        # Ensure the variable name starts with a letter.
        if re.match(r"^[^a-zA-Z]", var_name):
            var_name = "var_{}".format(var_name)
        return var_name

    @staticmethod
    def _cf_coord_standardised_units(coord):
        """Determine a suitable units from a given coordinate.

        Parameters
        ----------
        coord : :class:`iris.coords.Coord`
            A coordinate of a cube.

        Returns
        -------
        units
            The (standard_name, long_name, unit) of the given
            :class:`iris.coords.Coord` instance.
        """
        units = str(coord.units)
        # Set the 'units' of 'latitude' and 'longitude' coordinates specified
        # in 'degrees' to 'degrees_north' and 'degrees_east' respectively,
        # as defined in the CF conventions for netCDF files: sections 4.1 and
        # 4.2.
        if (
            isinstance(coord.coord_system, iris.coord_systems.GeogCS)
            or coord.coord_system is None
        ) and coord.units == "degrees":
            if coord.standard_name == "latitude":
                units = "degrees_north"
            elif coord.standard_name == "longitude":
                units = "degrees_east"

        return units

    def _ensure_valid_dtype(self, values, src_name, src_object):
        # Some formats have no 64-bit or unsigned integer types, so we check
        # whether we can store them as int32 instead. Which dtypes are worth
        # narrowing is the saver's question; whether this store holds them is
        # the store's.
        is_narrowable_integer = np.issubdtype(values.dtype, np.int64) or np.issubdtype(
            values.dtype, np.unsignedinteger
        )
        if is_narrowable_integer and not self._dataset.supports_dtype(values.dtype):
            val_min, val_max = (values.min(), values.max())
            if is_lazy_data(values):
                val_min, val_max = _co_realise_lazy_arrays([val_min, val_max])
            # NumPy will inherit values.dtype even if the scalar numbers work
            #  with a smaller type.
            min_dtype = np.promote_types(
                *[np.min_scalar_type(m) for m in (val_min, val_max)]
            )
            # Cast to an integer type the format supports.
            if not np.can_cast(min_dtype, np.int32):
                msg = (
                    "The data type of {} {!r} is not supported by {} and"
                    " its values cannot be safely cast to a supported"
                    " integer type."
                )
                msg = msg.format(src_name, src_object, self._dataset.format)
                raise ValueError(msg)
            values = values.astype(np.int32)
        return values

    def _create_cf_bounds(self, coord, cf_var, cf_name, /, *, compression_kwargs=None):
        """Create the associated CF-netCDF bounds variable.

        Parameters
        ----------
        coord : :class:`iris.coords.Coord`
            A coordinate of a cube.
        cf_var :
            CF-netCDF variable.
        cf_name : str
            Name of the CF-NetCDF variable.
        compression_kwargs : dict, optional
            NetCDF data compression keyword arguments.

        Returns
        -------
        None

        """
        if hasattr(coord, "has_bounds") and coord.has_bounds():
            if compression_kwargs is None:
                compression_kwargs = {}

            # Get the values in a form which is valid for the file format.
            bounds = self._ensure_valid_dtype(
                coord.core_bounds(), "the bounds of coordinate", coord
            )
            n_bounds = bounds.shape[-1]

            if n_bounds == 2:
                bounds_dimension_name = "bnds"
            else:
                bounds_dimension_name = "bnds_%s" % n_bounds

            if coord.climatological:
                property_name = "climatology"
                varname_extra = "climatology"
            else:
                property_name = "bounds"
                varname_extra = "bnds"

            if bounds_dimension_name not in self._dataset.dimensions:
                # Create the bounds dimension with the appropriate extent.
                while bounds_dimension_name in self._dataset.variables:
                    # Also avoid collision with variable names.
                    # See '_get_dim_names' for reason.
                    bounds_dimension_name = self._increment_name(bounds_dimension_name)
                self._dataset.create_dimension(bounds_dimension_name, n_bounds)

            boundsvar_name = "{}_{}".format(cf_name, varname_extra)
            cf_var.attributes[property_name] = boundsvar_name
            cf_var_bounds = self._dataset.create_variable(
                boundsvar_name,
                bounds.dtype.newbyteorder("="),
                cf_var.dimensions + (bounds_dimension_name,),
                **compression_kwargs,
            )
            self._lazy_stream_data(data=bounds, cf_var=cf_var_bounds)

    def _create_string_variable(
        self, cube_or_mesh, cf_name, element, element_dims, data
    ):
        """Create the variable that will hold this unicode string data.

        Parameters
        ----------
        cube_or_mesh : :class:`iris.cube.Cube` or :class:`iris.mesh.MeshXY`
            The Cube or Mesh being saved.
        cf_name : str
            The name the variable is created with.
        element : :class:`iris.coords._DimensionalMetadata`
            The element being saved.
        element_dims : list of str
            The variable's dimension names. A subclass may append to it.
        data : :class:`numpy.ndarray`
            The string values, used for their dtype.

        Returns
        -------
        :class:`~iris.fileformats.cf.dataset.CFDatasetVariable`

        Raises
        ------
        NotImplementedError
            Always, here. How a store holds a string differs so much between
            formats that there is no generic answer to give.

        """
        msg = (
            f"Cannot save the string variable {cf_name!r}: "
            f"{type(self).__name__} does not say how {self._dataset.format} "
            "stores strings."
        )
        raise NotImplementedError(msg)

    def _get_element_variable_name(
        self, cube_or_mesh: "Cube|MeshXY", element: "_DimensionalMetadata|Cube"
    ) -> str:
        # keep it as _DimensionalMetadata
        # Don't want it as CFVariableMixin - implies a cube can be passed in
        # Could be Coord AncillaryVariable Cellmeasure
        """Return a CF-netCDF variable name for a given coordinate-like element, or cube.

        Parameters
        ----------
        cube_or_mesh : :class:`iris.cube.Cube` or :class:`iris.mesh.MeshXY`
            The Cube or Mesh being saved to the netCDF file.
        element : :class:`iris.coords._DimensionalMetadata` | :class:``iris.cube.Cube``
            An instance of a coordinate (or similar), for which a CF-netCDF
            variable name is required.

        Returns
        -------
        str
            A CF-netCDF variable name as a string.

        """
        # Support cube or mesh save.
        from iris.cube import Cube

        if isinstance(cube_or_mesh, Cube):
            cube = cube_or_mesh
            mesh = cube.mesh
        else:
            cube = None
            mesh = cube_or_mesh

        if element.var_name is not None:
            cf_name = element.var_name
        elif isinstance(element, Cube):
            # Make name for a Cube without a var_name.
            cf_name = "_".join(element.name().lower().split())
        else:
            # Make name for a Coord-like element without a var_name
            name = element.standard_name or element.long_name
            if not name or set(name).intersection(string.whitespace):
                # We need to invent a name, based on its associated dimensions.
                if cube is not None and cube.coords(element):
                    # It is a regular cube coordinate.
                    # Auto-generate a name based on the dims.
                    name = ""
                    for dim in cube.coord_dims(element):
                        name += f"dim{dim}"
                    # Handle scalar coordinate (dims == ()).
                    if not name:
                        name = "unknown_scalar"
                else:
                    # Not a cube coord, so must be a connectivity or
                    # element-coordinate of the mesh.
                    # Name it for it's first dim, i.e. mesh-dim of its location.

                    from iris.mesh import Connectivity

                    if isinstance(element, Connectivity):
                        location = element.cf_role.split("_")[0]
                        location_dim_attr = f"{location}_dimension"
                        name = getattr(mesh, location_dim_attr)
                    else:
                        # 'coord' is a nameless coordinate of a mesh,
                        # so we set it as unknown for saving
                        name = "unknown"

            # Convert to lower case and replace whitespace by underscores.
            cf_name = "_".join(name.lower().split())

        cf_name = self.cf_valid_var_name(cf_name)
        return cf_name

    def _get_mesh_variable_name(self, mesh):
        """Return a CF-netCDF variable name for the mesh.

        Parameters
        ----------
        mesh : :class:`iris.mesh.MeshXY`
            An instance of a Mesh for which a CF-netCDF variable name is
            required.

        Returns
        -------
        str
            A CF-netCDF variable name as a string.

        """
        cf_name = mesh.var_name or mesh.long_name
        # Prefer a var-name, but accept a long_name as an alternative.
        # N.B. we believe it can't (shouldn't) have a standard name.
        if not cf_name:
            # Auto-generate a name based on mesh properties.
            cf_name = f"Mesh_{mesh.topology_dimension}d"

        # Ensure valid form for var-name.
        cf_name = self.cf_valid_var_name(cf_name)
        return cf_name

    def _create_mesh(self, mesh):
        """Create a mesh variable in the netCDF dataset.

        Parameters
        ----------
        mesh : :class:`iris.mesh.MeshXY`
            The Mesh to be saved to CF-netCDF file.

        Returns
        -------
        str
            The string name of the associated CF-netCDF variable saved.

        """
        # First choose a var-name for the mesh variable itself.
        cf_mesh_name = self._get_mesh_variable_name(mesh)
        # Disambiguate any possible clashes.
        while cf_mesh_name in self._dataset.variables:
            cf_mesh_name = self._increment_name(cf_mesh_name)

        # Create the main variable
        cf_mesh_var = self._dataset.create_variable(
            cf_mesh_name,
            np.dtype(np.int32),
        )

        # Add the basic essential attributes
        cf_mesh_var.attributes["cf_role"] = "mesh_topology"
        cf_mesh_var.attributes["topology_dimension"] = np.int32(mesh.topology_dimension)
        # Add the usual names + units attributes
        self._set_cf_var_attributes(cf_mesh_var, mesh)

        return cf_mesh_name

    def _set_cf_var_attributes(self, cf_var, element):
        from iris.cube import Cube

        # Deal with CF-netCDF units, and add the name+units properties.
        if isinstance(element, iris.coords.Coord):
            # Fix "degree" units if needed.
            units_str = self._cf_coord_standardised_units(element)
        else:
            units_str = str(element.units)

        # NB this bit is a nasty hack to preserve existing behaviour through a refactor:
        #  The attributes for Coords are created in the order units, standard_name,
        #   whereas for data-variables (aka Cubes) it is the other way around.
        # Needed now that this routine is also called from _create_cf_data_variable.
        # TODO: when we can break things, rationalise these to be the same.
        def add_units_attr():
            if cf_units.as_unit(units_str).is_udunits():
                cf_var.attributes["units"] = units_str

        def add_names_attrs():
            standard_name = element.standard_name
            if standard_name is not None:
                cf_var.attributes["standard_name"] = standard_name

            long_name = element.long_name
            if long_name is not None:
                cf_var.attributes["long_name"] = long_name

        if isinstance(element, Cube):
            add_names_attrs()
            add_units_attr()
        else:
            add_units_attr()
            add_names_attrs()

        # Add the CF-netCDF calendar attribute.
        if element.units.calendar:
            cf_var.attributes["calendar"] = str(element.units.calendar)

        # Take a copy so we can remove things
        element_attrs = element.attributes.copy()

        # Note: when writing UGRID, "element" can be a Mesh which has no "dtype",
        # and for dataless cubes it will have a 'None' dtype.
        if getattr(element, "dtype", None) is not None:
            # Most attributes are dealt with later.  But _Encoding needs to be defined
            #  *before* we can write to a character variable.
            if element.dtype.kind in "SU" and "_Encoding" in element_attrs:
                encoding = element_attrs.pop("_Encoding")
                cf_var.attributes["_Encoding"] = encoding

        if not isinstance(element, Cube):
            # Add any other custom coordinate attributes.
            # N.B. not Cube, which has specific handling in  _create_cf_data_variable
            for name in sorted(element_attrs):
                value = element_attrs[name]

                if name == "STASH":
                    # Adopting provisional Metadata Conventions for representing MO
                    # Scientific Data encoded in NetCDF Format.
                    name = "um_stash_source"
                    value = str(value)

                # Don't clobber existing attributes.
                if name not in cf_var.attributes:
                    cf_var.attributes[name] = value

    def _create_generic_cf_array_var(
        self,
        cube_or_mesh,
        cube_dim_names,
        element,
        /,
        *,
        element_dims=None,
        fill_value=None,
        compression_kwargs=None,
        packing_controls: dict | None = None,
        is_dataless=False,
    ):
        """Create theCF-netCDF variable given dimensional_metadata.

        Create the associated CF-netCDF variable in the netCDF dataset for the
        given dimensional_metadata.

        .. note::
            If the metadata element is a coord, it may also contain bounds.
            In which case, an additional var is created and linked to it.

        Parameters
        ----------
        cube_or_mesh : :class:`iris.cube.Cube` or :class:`iris.mesh.MeshXY`
            The Cube or Mesh being saved to the netCDF file.
        cube_dim_names : list of str
            The name of each dimension of the cube.
        element : :class:`iris.coords._DimensionalMetadata`
            An Iris :class:`iris.coords._DimensionalMetadata`, belonging to the
            cube.  Provides data, units and standard/long/var names.
            Not used if 'element_dims' is not None.
        element_dims : list of str, optional
            If set, contains the variable dimension (names),
            otherwise these are taken from `element.cube_dims[cube]`.
            For Mesh components (element coordinates and connectivities), this
            *must* be passed in, as "element.cube_dims" does not function.
        fill_value : number, optional
            If set, create the variable with this fill-value, and fill any
            masked data points with this value.
            If not set, standard netcdf4-python behaviour : the variable has no
            '_FillValue' property, and uses the "standard" fill-value for its
            type.
        compression_kwargs : dict, optional
            NetCDF data compression keyword arguments.

        Returns
        -------
        str
            The name of the CF-netCDF variable created.
        """
        # Support cube or mesh save.
        from iris.cube import Cube

        if isinstance(cube_or_mesh, Cube):
            cube = cube_or_mesh
        else:
            cube = None

        if compression_kwargs is None:
            compression_kwargs = {}

        # Work out the var-name to use.
        # N.B. the only part of this routine that may use a mesh _or_ a cube.
        cf_name = self._get_element_variable_name(cube_or_mesh, element)
        while cf_name in self._dataset.variables:
            cf_name = self._increment_name(cf_name)

        if cube and element_dims is None:
            # Get the list of file-dimensions (names), to create the variable.
            element_dims = [
                cube_dim_names[dim] for dim in element.cube_dims(cube)
            ]  # NB using 'cube_dims' as this works for any type of element

        # Get the data values, in a way which works for any element type, as
        # all are subclasses of _DimensionalMetadata.
        # (e.g. =points if a coord, =data if an ancillary, etc)
        if isinstance(element, Cube):
            data = element.core_data()
        else:
            data = element._core_values()

        # This compression contract is *not* applicable to a mesh.
        if cube is not None and data is not None and cube.shape != data.shape:
            compression_kwargs = {}

        if not is_dataless and data.dtype.kind == "U":
            # Unicode string data: typically a CF label variable, sometimes an
            # ancillary variable. How a store holds strings is the store's own
            # business, so the backend creates the variable.
            cf_var = self._create_string_variable(
                cube_or_mesh, cf_name, element, element_dims, data
            )
        else:
            # A non-string variable.
            # ensure a valid datatype for the file format.
            if is_dataless:
                dtype = self._DATALESS_DTYPE
                fill_value = self._DATALESS_FILLVALUE
            else:
                # Normal non-string data.
                # NOTE: this includes byte-arrays (S1 only) : however these must
                # use an actual cube dimension for the 'string dimension', which
                # seriously limits the utility of DECODE_TO_STRINGS_ON_READ.
                # TODO: also support netCDF variable-length strings ("string" type).
                #  Currently hit a **write error here**, being numpy object dtype ("O").
                if data.dtype.kind not in "iufSU" or (
                    data.dtype.kind == "S" and data.dtype.itemsize != 1
                ):
                    # This is a type of data we don't "understand".
                    # NB this includes "Sxx" types other than "S1" :  It seems that
                    # netCDF4 saves Sxx as variable-length strings.
                    # But we don't support that type in Iris.
                    msg = (
                        f"Variable {cf_name!r} has unexpected dtype, {data.dtype!r}."
                        f"Data content arrays must be numeric, or contain "
                        "single-bytes (dtype 'S1'), or unicode strings (dtype 'U<n>')."
                    )
                    raise ValueError(msg)

                element_type = type(element).__name__
                data = self._ensure_valid_dtype(data, element_type, element)
                if not packing_controls:
                    dtype = data.dtype.newbyteorder("=")
                else:
                    dtype = packing_controls["dtype"]

            # Check if this is a dim-coord.
            is_dimcoord = cube is not None and element in cube.dim_coords

            if is_dimcoord:
                # By definition of a CF-netCDF coordinate variable this
                # coordinate must be 1-D and the name of the CF-netCDF variable
                # must be the same as its dimension name.
                cf_name = element_dims[0]

            # Create the CF-netCDF variable.
            cf_var = self._dataset.create_variable(
                cf_name,
                dtype,
                element_dims,
                fill_value=fill_value,
                **compression_kwargs,
            )

            # Add the axis attribute for spatio-temporal CF-netCDF coordinates.
            if is_dimcoord:
                axis = iris.util.guess_coord_axis(element)
                if axis is not None and axis.lower() in SPATIO_TEMPORAL_AXES:
                    cf_var.attributes["axis"] = axis.upper()

            # Create the associated CF-netCDF bounds variable, if any.
            self._create_cf_bounds(
                element, cf_var, cf_name, compression_kwargs=compression_kwargs
            )

        # Add names + units
        # NOTE: *must* now do first, as we may need '_Encoding' set to write it !
        self._set_cf_var_attributes(cf_var, element)

        # Add the data to the CF-netCDF variable.
        if not is_dataless:
            if packing_controls:
                # We must set packing attributes (if any), before assigning values.
                for key, value in packing_controls["attributes"]:
                    cf_var.attributes[key] = value
            self._lazy_stream_data(data=data, cf_var=cf_var)

        return cf_name

    def _create_cf_cell_methods(self, cube, dimension_names):
        """Create CF-netCDF string representation of a cube cell methods.

        Parameters
        ----------
        cube : :class:`iris.cube.Cube` or :class:`iris.cube.CubeList`
            A :class:`iris.cube.Cube`, :class:`iris.cube.CubeList` or list of
            cubes to be saved to a netCDF file.
        dimension_names : list
            Names associated with the dimensions of the cube.

        Returns
        -------
        str
            CF-netCDF string representation of a cube cell methods.

        """
        cell_methods = []

        # Identify the collection of coordinates that represent CF-netCDF
        # coordinate variables.
        cf_coordinates = cube.dim_coords

        for cm in cube.cell_methods:
            names = ""

            for name in cm.coord_names:
                coord = cube.coords(name)

                if coord:
                    coord = coord[0]
                    if coord in cf_coordinates:
                        name = dimension_names[cube.coord_dims(coord)[0]]

                names += "%s: " % name

            interval = " ".join(
                ["interval: %s" % interval for interval in cm.intervals or []]
            )
            comment = " ".join(
                ["comment: %s" % comment for comment in cm.comments or []]
            )
            extra = " ".join([interval, comment]).strip()

            if extra:
                extra = " (%s)" % extra

            cell_methods.append(names + cm.method + extra)

        return " ".join(cell_methods)

    def _add_grid_mapping_to_dataset(self, cs, extended_grid_mapping=False):
        """Create a CF-netCDF grid mapping variable and add to the dataset.

        Parameters
        ----------
        cs : :class:`iris.coord_system.CoordSystem`
            The :class:`iris.coord_system.CoordSystem` used to generate the
            new netCDF CF grid mapping variable.

        Returns
        -------
        None
        """
        cf_var_grid = self._dataset.create_variable(cs.grid_mapping_name, np.int32)
        cf_var_grid.attributes["grid_mapping_name"] = cs.grid_mapping_name

        # Where each parameter below lands decides how it is typed in the
        # store, so the store supplies the function that puts it there. On
        # netCDF that is the netCDF4 variable itself, exactly as it always
        # has been; see Q8 in the design document.
        gm_write_param = self._grid_mapping_writer(cf_var_grid)

        def add_ellipsoid(ellipsoid):
            gm_write_param(
                "longitude_of_prime_meridian", ellipsoid.longitude_of_prime_meridian
            )
            semi_major = ellipsoid.semi_major_axis
            semi_minor = ellipsoid.semi_minor_axis
            if semi_minor == semi_major:
                gm_write_param("earth_radius", semi_major)
            else:
                gm_write_param("semi_major_axis", semi_major)
                gm_write_param("semi_minor_axis", semi_minor)
            if ellipsoid.datum is not None:
                gm_write_param("horizontal_datum_name", ellipsoid.datum)

        # latlon
        if isinstance(cs, iris.coord_systems.GeogCS):
            add_ellipsoid(cs)

        # rotated latlon
        elif isinstance(cs, iris.coord_systems.RotatedGeogCS):
            if cs.ellipsoid:
                add_ellipsoid(cs.ellipsoid)
            gm_write_param("grid_north_pole_latitude", cs.grid_north_pole_latitude)
            gm_write_param("grid_north_pole_longitude", cs.grid_north_pole_longitude)
            gm_write_param("north_pole_grid_longitude", cs.north_pole_grid_longitude)

        # tmerc
        elif isinstance(cs, iris.coord_systems.TransverseMercator):
            if cs.ellipsoid:
                add_ellipsoid(cs.ellipsoid)
            gm_write_param(
                "longitude_of_central_meridian", cs.longitude_of_central_meridian
            )
            gm_write_param(
                "latitude_of_projection_origin", cs.latitude_of_projection_origin
            )
            gm_write_param("false_easting", cs.false_easting)
            gm_write_param("false_northing", cs.false_northing)
            gm_write_param(
                "scale_factor_at_central_meridian", cs.scale_factor_at_central_meridian
            )

        # merc
        elif isinstance(cs, iris.coord_systems.Mercator):
            if cs.ellipsoid:
                add_ellipsoid(cs.ellipsoid)
            gm_write_param(
                "longitude_of_projection_origin", cs.longitude_of_projection_origin
            )
            gm_write_param("false_easting", cs.false_easting)
            gm_write_param("false_northing", cs.false_northing)
            # Only one of these should be set
            if cs.standard_parallel is not None:
                gm_write_param("standard_parallel", cs.standard_parallel)
            elif cs.scale_factor_at_projection_origin is not None:
                gm_write_param(
                    "scale_factor_at_projection_origin",
                    cs.scale_factor_at_projection_origin,
                )

        # lcc
        elif isinstance(cs, iris.coord_systems.LambertConformal):
            if cs.ellipsoid:
                add_ellipsoid(cs.ellipsoid)
            gm_write_param("standard_parallel", cs.secant_latitudes)
            gm_write_param("latitude_of_projection_origin", cs.central_lat)
            gm_write_param("longitude_of_central_meridian", cs.central_lon)
            gm_write_param("false_easting", cs.false_easting)
            gm_write_param("false_northing", cs.false_northing)

        # polar stereo (have to do this before Stereographic because it subclasses it)
        elif isinstance(cs, iris.coord_systems.PolarStereographic):
            if cs.ellipsoid:
                add_ellipsoid(cs.ellipsoid)
            gm_write_param("latitude_of_projection_origin", cs.central_lat)
            gm_write_param("straight_vertical_longitude_from_pole", cs.central_lon)
            gm_write_param("false_easting", cs.false_easting)
            gm_write_param("false_northing", cs.false_northing)
            # Only one of these should be set
            if cs.true_scale_lat is not None:
                gm_write_param("true_scale_lat", cs.true_scale_lat)
            elif cs.scale_factor_at_projection_origin is not None:
                gm_write_param(
                    "scale_factor_at_projection_origin",
                    cs.scale_factor_at_projection_origin,
                )
            else:
                gm_write_param("scale_factor_at_projection_origin", 1.0)

        # stereo
        elif isinstance(cs, iris.coord_systems.Stereographic):
            if cs.ellipsoid:
                add_ellipsoid(cs.ellipsoid)
            gm_write_param("longitude_of_projection_origin", cs.central_lon)
            gm_write_param("latitude_of_projection_origin", cs.central_lat)
            gm_write_param("false_easting", cs.false_easting)
            gm_write_param("false_northing", cs.false_northing)
            # Only one of these should be set
            if cs.true_scale_lat is not None:
                msg = (
                    "It is not valid CF to save a true_scale_lat for "
                    "a Stereographic grid mapping."
                )
                raise ValueError(msg)
            elif cs.scale_factor_at_projection_origin is not None:
                gm_write_param(
                    "scale_factor_at_projection_origin",
                    cs.scale_factor_at_projection_origin,
                )
            else:
                gm_write_param("scale_factor_at_projection_origin", 1.0)

        # osgb (a specific tmerc)
        elif isinstance(cs, iris.coord_systems.OSGB):
            warnings.warn(
                "OSGB coordinate system not yet handled",
                category=iris.warnings.IrisSaveWarning,
            )

        # lambert azimuthal equal area
        elif isinstance(cs, iris.coord_systems.LambertAzimuthalEqualArea):
            if cs.ellipsoid:
                add_ellipsoid(cs.ellipsoid)
            gm_write_param(
                "longitude_of_projection_origin", cs.longitude_of_projection_origin
            )
            gm_write_param(
                "latitude_of_projection_origin", cs.latitude_of_projection_origin
            )
            gm_write_param("false_easting", cs.false_easting)
            gm_write_param("false_northing", cs.false_northing)

        # albers conical equal area
        elif isinstance(cs, iris.coord_systems.AlbersEqualArea):
            if cs.ellipsoid:
                add_ellipsoid(cs.ellipsoid)
            gm_write_param(
                "longitude_of_central_meridian", cs.longitude_of_central_meridian
            )
            gm_write_param(
                "latitude_of_projection_origin", cs.latitude_of_projection_origin
            )
            gm_write_param("false_easting", cs.false_easting)
            gm_write_param("false_northing", cs.false_northing)
            gm_write_param("standard_parallel", cs.standard_parallels)

        # vertical perspective
        elif isinstance(cs, iris.coord_systems.VerticalPerspective):
            if cs.ellipsoid:
                add_ellipsoid(cs.ellipsoid)
            gm_write_param(
                "longitude_of_projection_origin", cs.longitude_of_projection_origin
            )
            gm_write_param(
                "latitude_of_projection_origin", cs.latitude_of_projection_origin
            )
            gm_write_param("false_easting", cs.false_easting)
            gm_write_param("false_northing", cs.false_northing)
            gm_write_param("perspective_point_height", cs.perspective_point_height)

        # geostationary
        elif isinstance(cs, iris.coord_systems.Geostationary):
            if cs.ellipsoid:
                add_ellipsoid(cs.ellipsoid)
            gm_write_param(
                "longitude_of_projection_origin", cs.longitude_of_projection_origin
            )
            gm_write_param(
                "latitude_of_projection_origin", cs.latitude_of_projection_origin
            )
            gm_write_param("false_easting", cs.false_easting)
            gm_write_param("false_northing", cs.false_northing)
            gm_write_param("perspective_point_height", cs.perspective_point_height)
            gm_write_param("sweep_angle_axis", cs.sweep_angle_axis)

        # oblique mercator (and rotated variant)
        # Use duck-typing over isinstance() - subclasses (i.e.
        #  RotatedMercator) upset mock tests.
        elif getattr(cs, "grid_mapping_name", None) == "oblique_mercator":
            # RotatedMercator subclasses ObliqueMercator, and RM
            #  instances are implicitly saved as OM due to inherited
            #  properties. This is correct because CF 1.11 is removing
            #  all mention of RM.
            if cs.ellipsoid:
                add_ellipsoid(cs.ellipsoid)
            gm_write_param("azimuth_of_central_line", cs.azimuth_of_central_line)
            gm_write_param(
                "latitude_of_projection_origin", cs.latitude_of_projection_origin
            )
            gm_write_param(
                "longitude_of_projection_origin", cs.longitude_of_projection_origin
            )
            gm_write_param("false_easting", cs.false_easting)
            gm_write_param("false_northing", cs.false_northing)
            gm_write_param(
                "scale_factor_at_projection_origin",
                cs.scale_factor_at_projection_origin,
            )

        # other
        else:
            warnings.warn(
                "Unable to represent the horizontal "
                "coordinate system. The coordinate system "
                "type %r is not yet implemented." % type(cs),
                category=iris.warnings.IrisSaveWarning,
            )

        # add WKT string
        if extended_grid_mapping:
            gm_write_param("crs_wkt", cs.as_cartopy_crs().to_wkt())

    def _create_cf_grid_mapping(self, cube, cf_var_cube):
        """Create CF-netCDF grid mapping and associated CF-netCDF variable.

        Create CF-netCDF grid mapping variable and associated CF-netCDF
        data variable grid mapping attribute.

        Parameters
        ----------
        cube : :class:`iris.cube.Cube` or :class:`iris.cube.CubeList`
            A :class:`iris.cube.Cube`, :class:`iris.cube.CubeList` or list of
            cubes to be saved to a netCDF file.
        cf_var_cube : :class:`~iris.fileformats.netcdf._dataset.NetCDFDatasetVariable`
            A cf variable cube representation.

        Returns
        -------
        None

        """
        coord_systems = []
        if cube.extended_grid_mapping:
            # get unique list of all coord_systems on cube coords:
            for coord in cube.coords():
                if coord.coord_system and coord.coord_system not in coord_systems:
                    coord_systems.append(coord.coord_system)
        else:
            # will only return a single coord system (if one exists):
            if cs := cube.coord_system("CoordSystem"):
                coord_systems.append(cs)

        grid_mappings = {}
        matched_all_coords = True

        for cs in coord_systems:
            # Grid var not yet created?
            if cs not in self._coord_systems:
                # handle potential duplicate netCDF variable names:
                while cs.grid_mapping_name in self._dataset.variables:
                    aname = self._increment_name(cs.grid_mapping_name)
                    cs.grid_mapping_name = aname

                # create grid mapping variable on dataset for this coordinate system:
                self._add_grid_mapping_to_dataset(
                    cs, extended_grid_mapping=cube.extended_grid_mapping
                )
                self._coord_systems.append(cs)

            # create the `grid_mapping` attribute for the data variable:
            if cube.extended_grid_mapping:
                # Order the coordinates as per the order in the CRS/WKT string.
                # (We should only ever have a coordinate system for horizontal
                # spatial coords, so check for east/north directions)
                ordered_coords = []
                for ax_info in cs.as_cartopy_crs().axis_info:
                    try:
                        match ax_info.direction.lower():
                            case "east":
                                ordered_coords.append(
                                    cube.coord(axis="X", coord_system=cs)
                                )
                            case "north":
                                ordered_coords.append(
                                    cube.coord(axis="Y", coord_system=cs)
                                )
                            case _:
                                msg = (
                                    f"Can't handle axis direction {ax_info.direction!r}"
                                )
                                raise NotImplementedError(msg)
                    except iris.exceptions.CoordinateNotFoundError as e:
                        msg = (
                            f"Failed to assign coordinate for {ax_info.name} axis of "
                            f"coordinate system {cs.grid_mapping_name}: {str(e)}"
                        )
                        warnings.warn(msg, category=iris.warnings.IrisSaveWarning)

                        # fall back to simple grid mapping (single crs entry for DimCoord only)
                        matched_all_coords = False

                # Get variable names, and store them if necessary:
                cfvar_names = []
                for coord in ordered_coords:
                    cfvar = self._name_coord_map.name(coord)
                    if not cfvar:
                        # not found - create and store it:
                        cfvar = self._get_element_variable_name(cube, coord)
                        self._name_coord_map.append(
                            cfvar, self._get_element_variable_name(cube, coord)
                        )
                    cfvar_names.append(cfvar)

                coord_string = " ".join(cfvar_names)
                grid_mappings[cs.grid_mapping_name] = coord_string

        # Refer to grid var
        if len(coord_systems):
            grid_mapping = None
            if cube.extended_grid_mapping:
                if matched_all_coords:
                    grid_mapping = " ".join(
                        f"{cs_name}: {cs_coords}"
                        for cs_name, cs_coords in grid_mappings.items()
                    )
                else:
                    # We didn't match all coords required for grid mapping, default to
                    # single coordinate system for DimCoords, if set.
                    for ax in ["x", "y"]:
                        try:
                            if dim_cs := cube.coord(
                                axis=ax, dim_coords=True
                            ).coord_system:
                                grid_mapping = dim_cs.grid_mapping_name
                                break
                        except iris.exceptions.CoordinateNotFoundError:
                            pass
            else:
                # Not using extended_grid_mapping: only ever be on cs in this case:
                grid_mapping = coord_systems[0].grid_mapping_name

            if grid_mapping:
                cf_var_cube.attributes["grid_mapping"] = grid_mapping

    _DATALESS_ATTRNAME = "iris_dataless_cube"
    _DATALESS_DTYPE = np.dtype("u1")
    _DATALESS_FILLVALUE = 127

    def _create_cf_data_variable(
        self,
        cube,
        dimension_names,
        local_keys=None,
        packing=None,
        fill_value=None,
        **kwargs,
    ):
        """Create CF-netCDF data variable for the cube and any associated grid mapping.

        # TODO: when iris.FUTURE.save_split_attrs is removed, the 'local_keys' arg can
        # be removed.

        Parameters
        ----------
        cube : :class:`iris.cube.Cube`
            The associated cube being saved to CF-netCDF file.
        dimension_names : list
            String names for each dimension of the cube.
        local_keys : iterable of str, optional
            See :func:`iris.fileformats.netcdf.Saver.write`.
        packing : type or str or dict or list, optional
            See :func:`iris.fileformats.netcdf.Saver.write`.
        fill_value : optional
            See :func:`iris.fileformats.netcdf.Saver.write`.

        Returns
        -------
        The newly created CF-netCDF data variable.

        Notes
        -----
        All other keywords are passed through to the dataset's `createVariable`
        method.

        """
        # TODO: when iris.FUTURE.save_split_attrs is removed, the 'local_keys' arg can
        # be removed.
        # Get the values in a form which is valid for the file format.
        is_dataless = cube.is_dataless()

        packing_controls = None
        if packing and not is_dataless:
            data = self._ensure_valid_dtype(cube.core_data(), "cube", cube)
            if isinstance(packing, dict):
                if "dtype" not in packing:
                    msg = "The dtype attribute is required for packing."
                    raise ValueError(msg)
                dtype = np.dtype(packing["dtype"])
                scale_factor = packing.get("scale_factor", None)
                add_offset = packing.get("add_offset", None)
                valid_keys = {"dtype", "scale_factor", "add_offset"}
                invalid_keys = set(packing.keys()) - valid_keys
                if invalid_keys:
                    msg = (
                        "Invalid packing key(s) found: '{}'. The valid "
                        "keys are '{}'.".format(
                            "', '".join(invalid_keys), "', '".join(valid_keys)
                        )
                    )
                    raise ValueError(msg)
            else:
                # We compute the scale_factor and add_offset based on the
                # min/max of the data.
                masked = iris.util.is_masked(data)
                dtype = np.dtype(packing)
                cmin, cmax = (data.min(), data.max())
                if is_lazy_data(data):
                    cmin, cmax = _co_realise_lazy_arrays([cmin, cmax])
                n = dtype.itemsize * 8
                if masked:
                    scale_factor = (cmax - cmin) / (2**n - 2)
                else:
                    scale_factor = (cmax - cmin) / (2**n - 1)
                if dtype.kind == "u":
                    add_offset = cmin
                elif dtype.kind == "i":
                    if masked:
                        add_offset = (cmax + cmin) / 2
                    else:
                        add_offset = cmin + 2 ** (n - 1) * scale_factor

            packing_controls = {
                "dtype": dtype,
                "attributes": [
                    ("scale_factor", scale_factor),
                    ("add_offset", add_offset),
                ],
            }

        # Create the cube CF-netCDF data variable with data payload.
        cf_name = self._create_generic_cf_array_var(
            cube,
            dimension_names,
            cube,
            element_dims=dimension_names,
            fill_value=fill_value,
            compression_kwargs=kwargs,
            packing_controls=packing_controls,
            is_dataless=is_dataless,
        )
        cf_var = self._dataset.variables[cf_name]

        # Set general attrs: NB this part is cube-specific (not the same for components)
        # - so 'set_cf_var_attributes' *doesn't* set these, if element is a Cube
        if iris.FUTURE.save_split_attrs:
            attr_names = cube.attributes.locals.keys()
        else:
            # Add data variable-only attribute names to local_keys.
            if local_keys is None:
                local_keys = set()
            else:
                local_keys = set(local_keys)
            local_keys.update(_CF_DATA_ATTRS, _UKMO_DATA_ATTRS, ["GRIB_PARAM"])

            # Add any cube attributes whose keys are in local_keys as
            # CF-netCDF data variable attributes.
            attr_names = set(cube.attributes).intersection(local_keys)

        # Write all the variable attributes
        for attr_name in sorted(attr_names):
            # Do not output 'conventions' or extended grid mapping attribute.
            if attr_name.lower() in ["conventions", "iris_extended_grid_mapping"]:
                continue

            value = cube.attributes[attr_name]

            # Convert any "managed" attributes. These convert between an internal
            #  convenience representation and what is actually stored in the file.
            handler = ATTRIBUTE_HANDLERS.get(attr_name)
            if handler is not None:
                try:
                    attr_name, value = handler.encode_object(value)
                except (TypeError, ValueError):
                    string_value = str(value)
                    msg = (
                        f"Invalid value in managed '{attr_name}' attribute: {value!r}. "
                        f"File attribute set to raw (string) value {string_value!r}."
                    )
                    warnings.warn(msg, category=iris.warnings.IrisSaveWarning)
                    # Resolve by setting the **Iris** attr-name to the "raw" string repr
                    # NB we expect the default written name to be the last listed
                    attr_name = handler.iris_name
                    value = string_value

            if attr_name in _CF_GLOBAL_ATTRS:
                msg = (
                    "{attr_name!r} is being added as CF data variable "
                    "attribute, but {attr_name!r} should only be a CF "
                    "global attribute.".format(attr_name=attr_name)
                )
                warnings.warn(msg, category=iris.warnings.IrisCfSaveWarning)

            cf_var.attributes[attr_name] = value

        # Add the 'dataless' marker if needed
        if is_dataless:
            cf_var.attributes[self._DATALESS_ATTRNAME] = "true"

        # Create the CF-netCDF data variable cell method attribute.
        cell_methods = self._create_cf_cell_methods(cube, dimension_names)

        if cell_methods:
            cf_var.attributes["cell_methods"] = cell_methods

        # Create the CF-netCDF grid mapping.
        self._create_cf_grid_mapping(cube, cf_var)

        return cf_var

    def _increment_name(self, varname):
        """Increment string name or begin increment.

        Avoidance of conflicts between variable names, where the name is
        incremented to distinguish it from others.

        Parameters
        ----------
        varname : str
            Variable name to increment.

        Returns
        -------
        Incremented varname.

        """
        num = 0
        try:
            name, endnum = varname.rsplit("_", 1)
            if endnum.isdigit():
                num = int(endnum) + 1
                varname = name
        except ValueError:
            pass

        return "{}_{}".format(varname, num)

    def _backend_write(self, data, cf_var) -> bool:
        """Write the data here and now, if this backend handles it itself.

        Parameters
        ----------
        data : array-like
            The values to write.
        cf_var : :class:`~iris.fileformats.cf.dataset.CFDatasetVariable`
            The variable to write them into.

        Returns
        -------
        bool
            Whether the data was written. ``False``, here: the direct and
            delayed writes below cover every store with no special case.

        """
        return False

    def _grid_mapping_writer(self, cf_var):
        """Return the function that writes one CF grid-mapping parameter.

        Parameters
        ----------
        cf_var : :class:`~iris.fileformats.cf.dataset.CFDatasetVariable`
            The grid-mapping variable, newly created and otherwise empty.

        Returns
        -------
        callable
            Called with a parameter name and a value. Here it puts them in
            ``cf_var.attributes``, which is where a CF store keeps them. A
            store that types those attributes some other way overrides this.

        """

        def write_parameter(name, value):
            cf_var.attributes[name] = value

        return write_parameter

    def _lazy_stream_data(
        self,
        data: np.typing.ArrayLike,
        cf_var: CFDatasetVariable,
    ) -> None:
        if hasattr(data, "shape") and data.shape == (1,) + cf_var.shape:
            # (Don't do this check for string data).
            # Reduce dimensionality where the data array has an extra dimension
            #  versus the cf_var - to avoid a broadcasting ambiguity.
            # Happens when bounds data is for a scalar point - array is 2D but
            #  contains just 1 row, so the cf_var is 1D.
            data = data.squeeze(axis=0)

        if self._backend_write(data, cf_var):
            return

        doing_delayed_save = is_lazy_data(data)
        if doing_delayed_save:
            # save lazy data with a delayed operation.  For now, we just record the
            # necessary information -- a single, complete delayed action is
            # constructed later by a call to delayed_completion().
            def store(
                data: np.typing.ArrayLike,
                cf_var: CFDatasetVariable,
            ) -> None:
                # TODO: recheck once Zarr behaviour is settled.
                # Ask the variable for something a worker can stream into
                # after this file is closed.  What that is is the backend's
                # business: netCDF reopens the file, Zarr will hand back the
                # array itself.
                self._delayed_writes.append((data, cf_var.write_handle()))

        else:
            # Real data is always written directly, i.e. not via lazy save.
            def store(
                data: np.typing.ArrayLike,
                cf_var: CFDatasetVariable,
            ) -> None:
                cf_var[:] = data  # type: ignore[index]

        # Store the data.
        store(data, cf_var)

    def delayed_completion(self) -> Delayed:
        """Perform file completion for delayed saves.

        Create and return a :class:`dask.delayed.Delayed` to perform file
        completion for delayed saves.

        Returns
        -------
        :class:`dask.delayed.Delayed`

        Notes
        -----
        The dataset *must* be closed (saver has exited its context) before the
        result can be computed, otherwise computation will hang (never return).
        """
        if self._delayed_writes:
            # Create a single delayed da.store operation to complete the file.
            sources, targets = zip(*self._delayed_writes)
            result = da.store(sources, targets, compute=False, lock=False)

        else:
            # Return a do-nothing delayed, for usage consistency.
            @dask.delayed
            def no_op():
                return None

            result = no_op()

        return result

    def complete(self) -> None:
        """Complete file by computing any delayed variable saves.

        This requires that the Saver has closed the dataset (exited its context).

        """
        if not self._dataset.closed:
            msg = (
                "Cannot call Saver.complete() until its dataset is closed, "
                "i.e. the saver's context has exited."
            )
            raise ValueError(msg)

        # Complete the saves now
        dask.compute(self.delayed_completion())
