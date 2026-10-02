# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Collect classified CF-netCDF variables into a single addressable group.

:class:`CFGroup` is the output of classification and the input to rule-based
loading. It is a :class:`~collections.abc.MutableMapping` from CF-netCDF
variable name to the :class:`~iris.fileformats.cf.CFVariable` instance that
:mod:`iris.fileformats.cf._variables` classified it as, so a group holds every
variable in a file exactly once, whatever its type.

On top of that mapping sit one read-only property per variable type --
:attr:`CFGroup.bounds`, :attr:`CFGroup.grid_mappings` and so on -- each of
which filters the mapping by ``isinstance``. They are computed on every access
rather than cached, so a group stays correct when
:class:`~iris.fileformats.cf.CFReader` mutates it during its second pass. A
variable therefore appears in every property whose type it satisfies, and
:attr:`CFGroup.formula_terms` is the one exception to the ``isinstance``
rule: it filters on ``has_formula_terms()`` instead, because participating in
a formula term is a property of the variable rather than of its class.

Two attributes sit alongside the variables and are populated by the reader,
not by this module. ``global_attributes`` holds the file's netCDF global
attributes. ``promoted`` records the variables that the reader re-classified
as :class:`~iris.fileformats.cf.CFDataVariable`, which it does for two
distinct reasons: a formula term that names a reference surface is a data
pay-load in its own right, per [CF] Appendix D; and a variable that was
referenced but whose dimensions are not a subset of the referrer's is dropped
from that referrer with a warning, then promoted so that it is not lost
altogether. Promotion is how Iris avoids silently discarding either.

"""

from collections.abc import MutableMapping

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
    CFVariable,
)


################################################################################
class CFGroup(MutableMapping):
    """Collection of 'NetCDF CF Metadata Conventions variables and netCDF global attributes.

    Represents a collection of 'NetCDF Climate and Forecast (CF) Metadata
    Conventions' variables and netCDF global attributes.

    """

    def __init__(self):
        #: Collection of CF-netCDF variables
        self._cf_variables = {}
        #: Collection of netCDF global attributes
        self.global_attributes = {}
        #: Collection of CF-netCDF variables promoted to a CFDataVariable.
        self.promoted = {}

    def _cf_getter(self, cls):
        # Generate dictionary with dictionary comprehension.
        return {
            cf_name: cf_var
            for cf_name, cf_var in self._cf_variables.items()
            if isinstance(cf_var, cls)
        }

    @property
    def ancillary_variables(self):
        """Collection of CF-netCDF ancillary variables."""
        return self._cf_getter(CFAncillaryDataVariable)

    @property
    def auxiliary_coordinates(self):
        """Collection of CF-netCDF auxiliary coordinate variables."""
        return self._cf_getter(CFAuxiliaryCoordinateVariable)

    @property
    def bounds(self):
        """Collection of CF-netCDF boundary variables."""
        return self._cf_getter(CFBoundaryVariable)

    @property
    def climatology(self):
        """Collection of CF-netCDF climatology variables."""
        return self._cf_getter(CFClimatologyVariable)

    @property
    def coordinates(self):
        """Collection of CF-netCDF coordinate variables."""
        return self._cf_getter(CFCoordinateVariable)

    @property
    def data_variables(self):
        """Collection of CF-netCDF data pay-load variables."""
        return self._cf_getter(CFDataVariable)

    @property
    def formula_terms(self):
        """Collection of CF-netCDF variables that participate in a CF-netCDF formula term."""
        return {
            cf_name: cf_var
            for cf_name, cf_var in self._cf_variables.items()
            if cf_var.has_formula_terms()
        }

    @property
    def grid_mappings(self):
        """Collection of CF-netCDF grid mapping variables."""
        return self._cf_getter(CFGridMappingVariable)

    @property
    def labels(self):
        """Collection of CF-netCDF label variables."""
        return self._cf_getter(CFLabelVariable)

    @property
    def cell_measures(self):
        """Collection of CF-netCDF measure variables."""
        return self._cf_getter(CFMeasureVariable)

    @property
    def non_data_variable_names(self):
        """:class:`set` names of the CF-netCDF variables that are not the data pay-load."""
        non_data_variables = (
            self.ancillary_variables,
            self.auxiliary_coordinates,
            self.bounds,
            self.climatology,
            self.coordinates,
            self.grid_mappings,
            self.labels,
            self.cell_measures,
            self.connectivities,
            self.ugrid_coords,
            self.meshes,
        )
        result = set()
        for variable in non_data_variables:
            result |= set(variable)
        return result

    @property
    def connectivities(self):
        """Collection of CF-UGRID connectivity variables."""
        return self._cf_getter(CFUGridConnectivityVariable)

    @property
    def ugrid_coords(self):
        """Collection of CF-UGRID-relevant auxiliary coordinate variables."""
        return self._cf_getter(CFUGridAuxiliaryCoordinateVariable)

    @property
    def meshes(self):
        """Collection of CF-UGRID mesh variables."""
        return self._cf_getter(CFUGridMeshVariable)

    def keys(self):
        """Return the names of all the CF-netCDF variables in the group."""
        return self._cf_variables.keys()

    def __len__(self):
        return len(self._cf_variables)

    def __iter__(self):
        for item in self._cf_variables:
            yield item

    def __setitem__(self, name, variable):
        if not isinstance(variable, CFVariable):
            raise TypeError(
                "Attempted to add an invalid CF-netCDF variable to the %s"
                % self.__class__.__name__
            )

        if name != variable.cf_name:
            raise ValueError(
                "Mismatch between key name %r and CF-netCDF variable name %r"
                % (str(name), variable.cf_name)
            )

        self._cf_variables[name] = variable

    def __getitem__(self, name):
        if name not in self._cf_variables:
            raise KeyError("Cannot get unknown CF-netCDF variable name %r" % str(name))

        return self._cf_variables[name]

    def __delitem__(self, name):
        if name not in self._cf_variables:
            raise KeyError(
                "Cannot delete unknown CF-netcdf variable name %r" % str(name)
            )

        del self._cf_variables[name]

    def __repr__(self):
        result = []
        result.append("variables:%d" % len(self._cf_variables))
        result.append("global_attributes:%d" % len(self.global_attributes))
        result.append("promoted:%d" % len(self.promoted))

        return "<%s of %s>" % (self.__class__.__name__, ", ".join(result))
