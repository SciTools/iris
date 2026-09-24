# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Read a netCDF dataset and interpret it as CF-netCDF variables and groups.

:class:`CFReader` is the entry point to the CF layer: give it a file path or
an already-open dataset and it returns an object whose ``cf_group`` is a fully
populated :class:`~iris.fileformats.cf.CFGroup`. All of the work happens in
:meth:`CFReader.__init__`, in two passes over the dataset's variables.

The first pass, ``_translate``, is classification. It asks each class in
``_variable_types`` to ``identify`` itself among the netCDF variables, then
handles the three special cases that tuple deliberately excludes:
:class:`~iris.fileformats.cf.CFCoordinateVariable` (identified by the netCDF
dimension/variable name coincidence rather than by an attribute),
``_CFFormulaTermsVariable`` (whose members are also something else) and
:class:`~iris.fileformats.cf.CFDataVariable` (what is left over once every
other classification has had its say). It also caches the parse of every
``grid_mapping`` attribute in ``_coord_system_mappings``, mapping a variable
name to the coordinate-system-to-coordinates relationships it declares, so
that the loading rules do not re-parse the same attribute once per coordinate.

The second pass, ``_build_cf_groups``, is association: for each variable it
builds the sub-:class:`~iris.fileformats.cf.CFGroup` of the variables that
variable references, and checks that a referenced variable's dimensions span
the referrer's. One that does not is not an error -- it is dropped from the
referrer's group with a warning and, if nothing else claims it, *promoted* to
a :class:`~iris.fileformats.cf.CFDataVariable` of its own. A final
``_reset`` clears the attribute touch history that classification dirtied, so
that the loading rules start from a clean record of which netCDF attributes
have been consumed.

``CFReader`` may or may not own the file it reads, and ``_own_file`` records
which. Constructed from a path, it opens the dataset and owns it; constructed
from an open dataset, it borrows it. ``_close`` -- reached from ``__exit__``
and from ``__del__`` -- only closes what it owns, so a borrowed dataset
survives garbage collection of the reader. ``_own_file`` is set ``False``
before anything that can raise, so the destructor is safe even if
``__init__`` fails part way.

This is the one module in :mod:`iris.fileformats.cf` still coupled to
:mod:`iris.fileformats.netcdf`: opening a file needs ``_thread_safe_nc`` and
``_bytecoding_datasets``. Everything else in the package already works against
any object that presents netCDF-like ``variables``, ``dimensions``,
``ncattrs`` and ``getncattr``. Removing that last coupling -- replacing the
direct dataset construction with a format-agnostic dataset abstraction, so
that Zarr can be read by the same machinery -- is the subject of §4.2 of the
native Zarr I/O design,
``docs/superpowers/specs/2026-09-21-zarr-io-design.md``.

"""

from collections.abc import Iterable
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import warnings

import iris.exceptions
import iris.fileformats._nc_load_rules.helpers as hh
from iris.fileformats.netcdf import _bytecoding_datasets, _thread_safe_nc
import iris.warnings

from ._group import CFGroup
from ._variables import (
    CFAncillaryDataVariable,
    CFAuxiliaryCoordinateVariable,
    CFBoundaryVariable,
    CFClimatologyVariable,
    CFCoordinateVariable,
    CFDataVariable,
    CFGridMappingVariable,
    CFLabelVariable,
    CFMeasureVariable,
    CFUGridAuxiliaryCoordinateVariable,
    CFUGridConnectivityVariable,
    CFUGridMeshVariable,
    _CFFormulaTermsVariable,
)

#: Supported dimensionless vertical coordinate reference surface/phemomenon
#: formula terms. Ref: [CF] Appendix D.
reference_terms = dict(
    atmosphere_sigma_coordinate=["ps"],
    atmosphere_hybrid_sigma_pressure_coordinate=["ps"],
    atmosphere_hybrid_height_coordinate=["orog"],
    atmosphere_sleve_coordinate=["zsurf1", "zsurf2"],
    ocean_sigma_coordinate=["eta", "depth"],
    ocean_s_coordinate=["eta", "depth"],
    ocean_sigma_z_coordinate=["eta", "depth"],
    ocean_s_coordinate_g1=["eta", "depth"],
    ocean_s_coordinate_g2=["eta", "depth"],
)


################################################################################
class CFReader:
    """Allows the contents of a netCDF file to be interpreted.

    This class allows the contents of a netCDF file to be interpreted according
    to the 'NetCDF Climate and Forecast (CF) Metadata Conventions'.

    """

    # All CF variable types EXCEPT for the "special cases" of
    # CFDataVariable, CFCoordinateVariable and _CFFormulaTermsVariable.
    _variable_types = (
        CFAncillaryDataVariable,
        CFAuxiliaryCoordinateVariable,
        CFBoundaryVariable,
        CFClimatologyVariable,
        CFGridMappingVariable,
        CFLabelVariable,
        CFMeasureVariable,
        CFUGridConnectivityVariable,
        CFUGridAuxiliaryCoordinateVariable,
        CFUGridMeshVariable,
    )

    #: Alias of :class:`~iris.fileformats.cf.CFGroup`, for callers that reach
    #: it through the reader.
    #:
    #: Hidden from the documentation because autodoc would otherwise describe
    #: ``CFGroup`` twice - once as a member of the package and once as this
    #: attribute - and the two descriptions collide, which fails the build
    #: outright under Read the Docs' ``fail_on_warning``. The alias itself is
    #: unaffected; only its duplicate documentation entry goes.
    #:
    #: :meta private:
    CFGroup = CFGroup

    def __init__(self, file_source, warn=False, monotonic=False):
        # Ensure safe operation for destructor, should init fail.
        self._own_file = False
        if isinstance(file_source, str):
            # Create from filepath : open it + own it (=close when we die).
            if not urlparse(file_source).scheme:
                self._filename = Path(file_source).expanduser()
            else:
                self._filename = file_source

            if _bytecoding_datasets.DECODE_TO_STRINGS_ON_READ:
                ds_type = _bytecoding_datasets.EncodedDataset
            else:
                ds_type = _thread_safe_nc.DatasetWrapper

            self._dataset = ds_type(self._filename, mode="r")
            self._own_file = True
        else:
            # We have been passed an open dataset.
            # We use it but don't own it (don't close it).
            self._dataset = file_source
            self._filename = self._dataset.filepath()

        #: Collection of CF-netCDF variables associated with this netCDF file
        self.cf_group = self.CFGroup()

        # Result of parsing "grid_mapping" attribute; mapping of coordinate_system => coordinates
        self._coord_system_mappings = {}

        # Issue load optimisation warning.
        if warn and self._dataset.file_format in [
            "NETCDF3_CLASSIC",
            "NETCDF3_64BIT",
        ]:
            warnings.warn(
                "Optimise CF-netCDF loading by converting data from NetCDF3 "
                'to NetCDF4 file format using the "nccopy" command.',
                category=iris.warnings.IrisLoadWarning,
            )

        self._check_monotonic = monotonic

        self._with_ugrid = True
        if not self._has_meshes():
            self._trim_ugrid_variable_types()
            self._with_ugrid = False

        # Read the variables in the dataset only once to reduce runtime.
        ds = self._dataset
        # Turn off *any* automatic decoding in the underlying netCDF4 dataset.
        ds.set_auto_chartostring(False)
        variables = self._dataset.variables
        self._translate(variables)
        self._build_cf_groups(variables)
        self._reset(variables)

    def __enter__(self):
        # Enable use as a context manager
        # N.B. this **guarantees* closure of the file, when the context is exited.
        # Note: ideally, the class would not do so much work in the __init__ call, and
        # would do all that here, after acquiring necessary permissions/locks.
        # But for legacy reasons, we can't do that.  So **effectively**, the context
        # (in terms of access control) already started, when we created the object.
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # When used as a context-manager, **always** close the file on exit.
        self._close()

    def _has_meshes(self):
        result = False
        for variable in self._dataset.variables.values():
            if hasattr(variable, "mesh") or hasattr(variable, "node_coordinates"):
                result = True
                break
        return result

    def _trim_ugrid_variable_types(self):
        self._variable_types = (
            CFAncillaryDataVariable,
            CFAuxiliaryCoordinateVariable,
            CFBoundaryVariable,
            CFClimatologyVariable,
            CFGridMappingVariable,
            CFLabelVariable,
            CFMeasureVariable,
        )

    @property
    def filename(self):
        """The file that the CFReader is reading."""
        return self._filename

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self._filename)

    def _translate(self, variables):
        """Classify the netCDF variables into CF-netCDF variables."""
        netcdf_variable_names = list(variables.keys())

        # Parse all instances of "grid_mapping" attributes and store in CFReader
        # This avoids re-parsing the grid_mappings each time they are needed.
        for nc_var in variables.values():
            if grid_mapping_attr := getattr(nc_var, "grid_mapping", None):
                try:
                    cs_mappings = hh._parse_extended_grid_mapping(grid_mapping_attr)
                    self._coord_system_mappings[nc_var.name] = cs_mappings
                except iris.exceptions.CFParseError as e:
                    msg = f"Error parsing `grid_mapping` attribute for {nc_var.name}: {str(e)}"
                    warnings.warn(msg, category=iris.warnings.IrisCfWarning)
                    continue

        # Identify all CF coordinate variables first. This must be done
        # first as, by CF convention, the definition of a CF auxiliary
        # coordinate variable may include a scalar CF coordinate variable,
        # whereas we want these two types of variables to be mutually exclusive.
        coords = CFCoordinateVariable.identify(
            variables, monotonic=self._check_monotonic
        )
        self.cf_group.update(coords)
        coordinate_names = list(self.cf_group.coordinates.keys())

        # Identify all CF variables EXCEPT for the "special cases".
        for variable_type in self._variable_types:
            # Prevent grid mapping variables being mis-identified as CF coordinate variables.
            ignore = (
                None
                if issubclass(variable_type, CFGridMappingVariable)
                else coordinate_names
            )
            kwargs = (
                {"coord_system_mappings": self._coord_system_mappings}
                if issubclass(variable_type, CFGridMappingVariable)
                else {}
            )

            self.cf_group.update(
                variable_type.identify(variables, ignore=ignore, **kwargs)
            )

        # Identify global netCDF attributes.
        attr_dict = {
            attr_name: _getncattr(self._dataset, attr_name, "")
            for attr_name in self._dataset.ncattrs()
        }
        self.cf_group.global_attributes.update(attr_dict)

        # Identify and register all CF formula terms.
        formula_terms = _CFFormulaTermsVariable.identify(variables)

        if iris.FUTURE.derived_bounds:
            # Keep track of all the root vars so we can unpick invalid bounds vars
            all_roots = set()

        # cf_var = CFFormulaTermsVariable (loops through everything that appears in formula terms)
        for cf_var in formula_terms.values():
            # Example of a formula term:
            # Suppose in the file eta:formula_terms contains "a: var_A"
            # cf_var = var_A, cf_root = eta and cf_term = 'a'. cf_var.cf_terms_by_root = {eta: 'a'}
            for cf_root, cf_term in cf_var.cf_terms_by_root.items():
                if iris.FUTURE.derived_bounds:
                    # For the "newstyle" derived-bounds implementation, find vars which appear in derived bounds terms
                    #  and turn them into bounds vars (though they don't appear in a "bounds" attribute)

                    # Adds each root only once
                    all_roots.add(cf_root)

                    # cf_root_coord = CFCoordinateVariable or CFAuxiliaryCoordinateVariable of the coordinate relating to the root
                    cf_root_coord = self.cf_group.coordinates.get(cf_root)
                    if cf_root_coord is None:
                        cf_root_coord = self.cf_group.auxiliary_coordinates.get(cf_root)

                    # N.B. cf_root_coord may here be None, if the root var was not a
                    #  coord - that is ok, it will not have a 'bounds', we will skip it.
                    root_bounds_name = None
                    if cf_root_coord is not None:
                        root_bounds_name = cf_root_coord.attributes.get("bounds")
                    if root_bounds_name in self.cf_group:
                        root_bounds_var = self.cf_group.get(root_bounds_name)
                        if "formula_terms" not in root_bounds_var.attributes:
                            # this is an invalid root bounds, according to CF, and therefore should be promoted into a cube
                            root_bounds_var._to_be_promoted = True
                        else:
                            # Found a valid *root* bounds variable : search for a corresponding *term* bounds variable,
                            term_bounds_vars = [
                                # loop through all formula terms and add them if they have a cf_term_by_root
                                # where (bounds of cf_root): cf_term (same as before)
                                f
                                for f in formula_terms.values()
                                if f.cf_terms_by_root.get(root_bounds_name) == cf_term
                            ]
                            if len(term_bounds_vars) == 1:
                                (term_bounds_var,) = term_bounds_vars
                                # N.B. bounds==main-var is valid CF for *no* bounds
                                if term_bounds_var != cf_var:
                                    cf_var.attributes["bounds"] = (
                                        term_bounds_var.cf_name
                                    )
                                    new_var = CFBoundaryVariable(
                                        term_bounds_var.cf_name, term_bounds_var.cf_data
                                    )
                                    new_var.add_formula_term(root_bounds_name, cf_term)
                                    # "Reclassify" this var as a bounds variable
                                    self.cf_group[term_bounds_var.cf_name] = new_var

                if cf_root not in self.cf_group.bounds:
                    # This records all formula terms in the main cf_group that were previously only stored in the formula_terms dictionary.
                    cf_name = cf_var.cf_name
                    if cf_name not in self.cf_group:
                        # If the formula term variable is not already in the group, add it as a coordinate.
                        new_var = CFAuxiliaryCoordinateVariable(cf_name, cf_var.cf_data)
                        if iris.FUTURE.derived_bounds and "bounds" in cf_var.attributes:
                            # Copy "old-style" derived bounds link
                            new_var.attributes["bounds"] = cf_var.attributes["bounds"]
                        self.cf_group[cf_name] = new_var

                    self.cf_group[cf_name].add_formula_term(cf_root, cf_term)

        if iris.FUTURE.derived_bounds:
            for cf_root in all_roots:
                # Invalidate "broken" bounds connections
                root_var = self.cf_group[cf_root]
                if root_var.attributes.get("formula_terms") and root_var.attributes.get(
                    "bounds"
                ):
                    root_bounds_var = self.cf_group.get(root_var.attributes["bounds"])
                    if root_bounds_var is None or not root_bounds_var.attributes.get(
                        "formula_terms"
                    ):
                        # This means it is *not* a valid bounds var, according to CF, and so therefore we are
                        # invalidating the bounds.
                        root_var.attributes["bounds"] = None

        # Determine the CF data variables.
        data_variable_names = (
            set(netcdf_variable_names) - self.cf_group.non_data_variable_names
        )

        for name in data_variable_names:
            self.cf_group[name] = CFDataVariable(name, variables[name])

    def _build_cf_groups(self, variables):
        """Build the first order relationships between CF-netCDF variables."""

        def _build(cf_variable):
            is_mesh_var = isinstance(cf_variable, CFUGridMeshVariable)
            ugrid_coord_names = []
            ugrid_coords = getattr(self.cf_group, "ugrid_coords", None)
            if ugrid_coords is not None:
                ugrid_coord_names = list(ugrid_coords.keys())

            coordinate_names = list(self.cf_group.coordinates.keys())
            cf_group = self.CFGroup()

            def _span_check(
                var_name: str, via_formula_terms: Optional[str] = None
            ) -> None:
                """Sanity check dimensionality."""
                var = self.cf_group[var_name]
                # No span check is necessary if variable is attached to a mesh.
                if (is_mesh_var or var.spans(cf_variable)) and not var._to_be_promoted:
                    cf_group[var_name] = var
                else:
                    # Register the ignored variable.
                    # N.B. 'ignored' variable from enclosing scope.
                    ignored.add(var_name)

                    text_formula = text_via = ""
                    if via_formula_terms:
                        text_formula = " formula terms"
                        text_via = f" via variable {via_formula_terms}"

                    message = (
                        f"Ignoring{text_formula} variable {var_name} "
                        f"referenced by variable {cf_variable.cf_name}"
                        f"{text_via}: Dimensions {var.dimensions} do not span "
                        f"{cf_variable.dimensions}"
                    )
                    warnings.warn(
                        message,
                        category=iris.warnings.IrisCfNonSpanningVarWarning,
                    )

            # Build CF variable relationships.
            for variable_type in self._variable_types:
                ignore = []
                kwargs = {}
                # Avoid UGridAuxiliaryCoordinateVariables also being
                # processed as CFAuxiliaryCoordinateVariables.
                if not is_mesh_var:
                    ignore += ugrid_coord_names
                # Prevent grid mapping variables being mis-identified as CF coordinate variables.
                if issubclass(variable_type, CFGridMappingVariable):
                    # pass parsed grid_mappings to CFGridMappingVariable types
                    kwargs.update(
                        {"coord_system_mappings": self._coord_system_mappings}
                    )
                else:
                    ignore += coordinate_names

                match = variable_type.identify(
                    variables,
                    ignore=ignore,
                    target=cf_variable.cf_name,
                    warn=False,
                    **kwargs,
                )
                # Sanity check dimensionality coverage.
                for cf_name in match:
                    _span_check(cf_name)

            if iris.FUTURE.derived_bounds:
                # Include bounds of every variable, within cf_group attached to the variable.
                if "bounds" in cf_variable.attributes:
                    bounds_name = cf_variable.attributes["bounds"]
                    if bounds_name not in cf_group:
                        bounds_var = self.cf_group.get(bounds_name)
                        if bounds_var:
                            # TODO: warning if span fails
                            if bounds_var.spans(cf_variable):
                                cf_group[bounds_name] = bounds_var

            # Build CF data variable relationships.
            if isinstance(cf_variable, CFDataVariable):
                # Add global netCDF attributes.
                cf_group.global_attributes.update(self.cf_group.global_attributes)
                # Add appropriate "dimensioned" CF coordinate variables.
                cf_group.update(
                    {
                        cf_name: self.cf_group[cf_name]
                        for cf_name in cf_variable.dimensions
                        if cf_name in self.cf_group.coordinates
                    }
                )
                # Add appropriate "dimensionless" CF coordinate variables.
                coordinates_attr = cf_variable.attributes.get("coordinates", "")
                cf_group.update(
                    {
                        cf_name: self.cf_group[cf_name]
                        for cf_name in coordinates_attr.split()
                        if cf_name in self.cf_group.coordinates
                    }
                )
                # Add appropriate formula terms.
                for cf_var in self.cf_group.formula_terms.values():
                    for cf_root in cf_var.cf_terms_by_root:
                        if cf_root in cf_group and cf_var.cf_name not in cf_group:
                            _span_check(cf_var.cf_name, cf_root)

            # Add the CF group to the variable.
            cf_variable.cf_group = cf_group

        # Ignored variables are those that cannot be attached to a
        # data variable as the dimensionality of that variable is not
        # a subset of the dimensionality of the data variable.
        ignored = set()

        for cf_variable in self.cf_group.values():
            _build(cf_variable)

        # Determine whether there are any formula terms that
        # may be promoted to a CFDataVariable and restrict promotion to only
        # those formula terms that are reference surface/phenomenon.
        for cf_var in self.cf_group.formula_terms.values():
            if iris.FUTURE.derived_bounds:
                # Always False: cf_group[...] yields a CFVariable instance, not
                # the class. See https://github.com/SciTools/iris/issues/7296
                if self.cf_group[cf_var.cf_name] is CFBoundaryVariable:
                    continue
            for cf_root, cf_term in cf_var.cf_terms_by_root.items():
                cf_root_var = self.cf_group[cf_root]
                if iris.FUTURE.derived_bounds:
                    if "standard_name" not in cf_root_var.attributes:
                        continue
                name = (
                    cf_root_var.attributes["standard_name"]
                    or cf_root_var.attributes["long_name"]
                )
                terms = reference_terms.get(name, [])
                if isinstance(terms, str) or not isinstance(terms, Iterable):
                    terms = [terms]
                cf_var_name = cf_var.cf_name
                if cf_term in terms and cf_var_name not in self.cf_group.promoted:
                    data_var = CFDataVariable(cf_var_name, cf_var.cf_data)
                    self.cf_group.promoted[cf_var_name] = data_var
                    _build(data_var)
                    break

        # Promote any ignored variables.
        promoted = set()
        not_promoted = ignored.difference(promoted)
        while not_promoted:
            cf_name = not_promoted.pop()
            if (
                cf_name not in self.cf_group.data_variables
                and cf_name not in self.cf_group.promoted
            ):
                data_var = CFDataVariable(cf_name, self.cf_group[cf_name].cf_data)
                self.cf_group.promoted[cf_name] = data_var
                _build(data_var)
            # Determine whether there are still any ignored variables
            # yet to be promoted.
            promoted.add(cf_name)
            not_promoted = ignored.difference(promoted)

    def _reset(self, variables):
        """Reset the attribute touch history of each variable."""
        for nc_var_name in variables.keys():
            self.cf_group[nc_var_name].cf_attrs_reset()

    def _close(self):
        # Explicitly close dataset to prevent file remaining open.
        if self._own_file and self._dataset is not None:
            self._dataset.close()
            self._dataset = None

    def __del__(self):
        # Be sure to close dataset when CFReader is destroyed / garbage-collected.
        self._close()


def _getncattr(dataset, attr, default=None):
    """Wrap `netCDF4.Dataset.getncattr` to make it behave more like `getattr`."""
    try:
        value = dataset.getncattr(attr)
    except AttributeError:
        value = default
    return value
