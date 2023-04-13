# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Provides the capability to load netCDF files and interpret them
according to the 'NetCDF Climate and Forecast (CF) Metadata Conventions'.

References:

[CF]  NetCDF Climate and Forecast (CF) Metadata conventions.
[NUG] NetCDF User's Guide, https://www.unidata.ucar.edu/software/netcdf/documentation/NUG/

"""

from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, MutableMapping
import os
import re
import warnings

import numpy as np
import numpy.ma as ma

from iris.fileformats.netcdf import _thread_safe_nc
import iris.util

#
# CF parse pattern common to both formula terms and measure CF variables.
#
_CF_PARSE = re.compile(
    r"""
                           \s*
                           (?P<lhs>[\w_]+)
                           \s*:\s*
                           (?P<rhs>[\w_]+)
                           \s*
                        """,
    re.VERBOSE,
)

# NetCDF variable attributes handled by the netCDF4 module and
# therefore automatically classed as "used" attributes.
_CF_ATTRS_IGNORE = set(
    ["_FillValue", "add_offset", "missing_value", "scale_factor"]
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


# NetCDF returns a different type for strings depending on Python version.
def _is_str_dtype(var):
    return np.issubdtype(var.dtype, np.bytes_)


################################################################################
class CFVariable(metaclass=ABCMeta):
    """Abstract base class wrapper for a CF-netCDF variable."""

    #: Name of the netCDF variable attribute that identifies this
    #: CF-netCDF variable.
    cf_identity = None

    def __init__(self, name, data):
        # Accessing the list of netCDF attributes is surprisingly slow.
        # Since it's used repeatedly, caching the list makes things
        # quite a bit faster.
        self._nc_attrs = data.ncattrs()

        #: NetCDF variable name.
        self.cf_name = name

        #: NetCDF4 Variable data instance.
        self.cf_data = data

        #: Collection of CF-netCDF variables associated with this variable.
        self.cf_group = None

        #: CF-netCDF formula terms that his variable participates in.
        self.cf_terms_by_root = {}

        self.cf_attrs_reset()

    @staticmethod
    def _identify_common(variables, ignore, target):
        if ignore is None:
            ignore = []

        if target is None:
            target = variables
        elif isinstance(target, str):
            if target not in variables:
                raise ValueError(
                    "Cannot identify unknown target CF-netCDF variable %r"
                    % target
                )
            target = {target: variables[target]}
        else:
            raise TypeError("Expect a target CF-netCDF variable name")

        return (ignore, target)

    @abstractmethod
    def identify(self, variables, ignore=None, target=None, warn=True):
        """
        Identify all variables that match the criterion for this CF-netCDF variable class.

        Args:

        * variables:
            Dictionary of netCDF4.Variable instance by variable name.

        Kwargs:

        * ignore:
            List of variable names to ignore.
        * target:
            Name of a single variable to check.
        * warn:
            Issue a warning if a missing variable is referenced.

        Returns:
            Dictionary of CFVariable instance by variable name.

        """
        pass

    def spans(self, cf_variable):
        """
        Determine whether the dimensionality of this variable
        is a subset of the specified target variable.

        Note that, by default scalar variables always span the
        dimensionality of the target variable.

        Args:

        * cf_variable:
            Compare dimensionality with the :class:`CFVariable`.

        Returns:
            Boolean.

        """
        result = set(self.dimensions).issubset(cf_variable.dimensions)
        return result

    def __eq__(self, other):
        # CF variable names are unique.
        return self.cf_name == other.cf_name

    def __ne__(self, other):
        # CF variable names are unique.
        return self.cf_name != other.cf_name

    def __hash__(self):
        # CF variable names are unique.
        return hash(self.cf_name)

    def __getattr__(self, name):
        # Accessing netCDF attributes is surprisingly slow. Since
        # they're often read repeatedly, caching the values makes things
        # quite a bit faster.
        if name in self._nc_attrs:
            self._cf_attrs.add(name)
        value = getattr(self.cf_data, name)
        setattr(self, name, value)
        return value

    def __getitem__(self, key):
        return self.cf_data.__getitem__(key)

    def __len__(self):
        return self.cf_data.__len__()

    def __repr__(self):
        return "%s(%r, %r)" % (
            self.__class__.__name__,
            self.cf_name,
            self.cf_data,
        )

    def cf_attrs(self):
        """Return a list of all attribute name and value pairs of the CF-netCDF variable."""
        return tuple(
            (attr, self.getncattr(attr)) for attr in sorted(self._nc_attrs)
        )

    def cf_attrs_ignored(self):
        """Return a list of all ignored attribute name and value pairs of the CF-netCDF variable."""
        return tuple(
            (attr, self.getncattr(attr))
            for attr in sorted(set(self._nc_attrs) & _CF_ATTRS_IGNORE)
        )

    def cf_attrs_used(self):
        """Return a list of all accessed attribute name and value pairs of the CF-netCDF variable."""
        return tuple(
            (attr, self.getncattr(attr)) for attr in sorted(self._cf_attrs)
        )

    def cf_attrs_unused(self):
        """Return a list of all non-accessed attribute name and value pairs of the CF-netCDF variable."""
        return tuple(
            (attr, self.getncattr(attr))
            for attr in sorted(set(self._nc_attrs) - self._cf_attrs)
        )

    def cf_attrs_reset(self):
        """Reset the history of accessed attribute names of the CF-netCDF variable."""
        self._cf_attrs = set([item[0] for item in self.cf_attrs_ignored()])

    def add_formula_term(self, root, term):
        """
        Register the participation of this CF-netCDF variable in a CF-netCDF formula term.

        Args:

        * root (string):
            The name of CF-netCDF variable that defines the CF-netCDF formula_terms attribute.
        * term (string):
            The associated term name of this variable in the formula_terms definition.

        Returns:
            None.

        """
        self.cf_terms_by_root[root] = term

    def has_formula_terms(self):
        """
        Determine whether this CF-netCDF variable participates in a CF-netcdf formula term.

        Returns:
            Boolean.

        """
        return bool(self.cf_terms_by_root)


class CFAncillaryDataVariable(CFVariable):
    """
    A CF-netCDF ancillary data variable is a variable that provides metadata
    about the individual values of another data variable.

    Identified by the CF-netCDF variable attribute 'ancillary_variables'.

    Ref: [CF] Section 3.4. Ancillary Data.

    """

    cf_identity = "ancillary_variables"

    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)

        # Identify all CF ancillary data variables.
        for nc_var_name, nc_var in target.items():
            # Check for ancillary data variable references.
            nc_var_att = getattr(nc_var, cls.cf_identity, None)

            if nc_var_att is not None:
                for name in nc_var_att.split():
                    if name not in ignore:
                        if name not in variables:
                            if warn:
                                message = "Missing CF-netCDF ancillary data variable %r, referenced by netCDF variable %r"
                                warnings.warn(message % (name, nc_var_name))
                        else:
                            result[name] = CFAncillaryDataVariable(
                                name, variables[name]
                            )

        return result


class CFAuxiliaryCoordinateVariable(CFVariable):
    """
    A CF-netCDF auxiliary coordinate variable is any netCDF variable that contains
    coordinate data, but is not a CF-netCDF coordinate variable by definition.

    There is no relationship between the name of a CF-netCDF auxiliary coordinate
    variable and the name(s) of its dimension(s).

    Identified by the CF-netCDF variable attribute 'coordinates'.
    Also see :class:`iris.fileformats.cf.CFLabelVariable`.

    Ref: [CF] Chapter 5. Coordinate Systems.
         [CF] Section 6.2. Alternative Coordinates.

    """

    cf_identity = "coordinates"

    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)

        # Identify all CF auxiliary coordinate variables.
        for nc_var_name, nc_var in target.items():
            # Check for auxiliary coordinate variable references.
            nc_var_att = getattr(nc_var, cls.cf_identity, None)

            if nc_var_att is not None:
                for name in nc_var_att.split():
                    if name not in ignore:
                        if name not in variables:
                            if warn:
                                message = "Missing CF-netCDF auxiliary coordinate variable %r, referenced by netCDF variable %r"
                                warnings.warn(message % (name, nc_var_name))
                        else:
                            # Restrict to non-string type i.e. not a CFLabelVariable.
                            if not _is_str_dtype(variables[name]):
                                result[name] = CFAuxiliaryCoordinateVariable(
                                    name, variables[name]
                                )

        return result


class CFBoundaryVariable(CFVariable):
    """
    A CF-netCDF boundary variable is associated with a CF-netCDF variable that contains
    coordinate data. When a data value provides information about conditions in a cell
    occupying a region of space/time or some other dimension, the boundary variable
    provides a description of cell extent.

    A CF-netCDF boundary variable will have one more dimension than its associated
    CF-netCDF coordinate variable or CF-netCDF auxiliary coordinate variable.

    Identified by the CF-netCDF variable attribute 'bounds'.

    Ref: [CF] Section 7.1. Cell Boundaries.

    """

    cf_identity = "bounds"

    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)

        # Identify all CF boundary variables.
        for nc_var_name, nc_var in target.items():
            # Check for a boundary variable reference.
            nc_var_att = getattr(nc_var, cls.cf_identity, None)

            if nc_var_att is not None:
                name = nc_var_att.strip()

                if name not in ignore:
                    if name not in variables:
                        if warn:
                            message = "Missing CF-netCDF boundary variable %r, referenced by netCDF variable %r"
                            warnings.warn(message % (name, nc_var_name))
                    else:
                        result[name] = CFBoundaryVariable(
                            name, variables[name]
                        )

        return result

    def spans(self, cf_variable):
        """
        Determine whether the dimensionality of this variable
        is a subset of the specified target variable.

        Note that, by default scalar variables always span the
        dimensionality of the target variable.

        Args:

        * cf_variable:
            Compare dimensionality with the :class:`CFVariable`.

        Returns:
            Boolean.

        """
        # Scalar variables always span the target variable.
        result = True
        if self.dimensions:
            source = self.dimensions
            target = cf_variable.dimensions
            # Ignore the bounds extent dimension.
            result = set(source[:-1]).issubset(target) or set(
                source[1:]
            ).issubset(target)
        return result


class CFClimatologyVariable(CFVariable):
    """
    A CF-netCDF climatology variable is associated with a CF-netCDF variable that contains
    coordinate data. When a data value provides information about conditions in a cell
    occupying a region of space/time or some other dimension, the climatology variable
    provides a climatological description of cell extent.

    A CF-netCDF climatology variable will have one more dimension than its associated
    CF-netCDF coordinate variable.

    Identified by the CF-netCDF variable attribute 'climatology'.

    Ref: [CF] Section 7.4. Climatological Statistics

    """

    cf_identity = "climatology"

    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)

        # Identify all CF climatology variables.
        for nc_var_name, nc_var in target.items():
            # Check for a climatology variable reference.
            nc_var_att = getattr(nc_var, cls.cf_identity, None)

            if nc_var_att is not None:
                name = nc_var_att.strip()

                if name not in ignore:
                    if name not in variables:
                        if warn:
                            message = "Missing CF-netCDF climatology variable %r, referenced by netCDF variable %r"
                            warnings.warn(message % (name, nc_var_name))
                    else:
                        result[name] = CFClimatologyVariable(
                            name, variables[name]
                        )

        return result

    def spans(self, cf_variable):
        """
        Determine whether the dimensionality of this variable
        is a subset of the specified target variable.

        Note that, by default scalar variables always span the
        dimensionality of the target variable.

        Args:

        * cf_variable:
            Compare dimensionality with the :class:`CFVariable`.

        Returns:
            Boolean.

        """
        # Scalar variables always span the target variable.
        result = True
        if self.dimensions:
            source = self.dimensions
            target = cf_variable.dimensions
            # Ignore the climatology extent dimension.
            result = set(source[:-1]).issubset(target) or set(
                source[1:]
            ).issubset(target)
        return result


class CFCoordinateVariable(CFVariable):
    """
    A CF-netCDF coordinate variable is a one-dimensional variable with the same name
    as its dimension, and it is defined as a numeric data type with values that are
    ordered monotonically. Missing values are not allowed in CF-netCDF coordinate
    variables. Also see [NUG] Section 2.3.1.

    Identified by the above criterion, there is no associated CF-netCDF variable
    attribute.

    Ref: [CF] 1.2. Terminology.

    """

    @classmethod
    def identify(
        cls, variables, ignore=None, target=None, warn=True, monotonic=False
    ):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)

        # Identify all CF coordinate variables.
        for nc_var_name, nc_var in target.items():
            if nc_var_name in ignore:
                continue
            # String variables can't be coordinates
            if _is_str_dtype(nc_var):
                continue
            # Restrict to one-dimensional with name as dimension
            if not (nc_var.ndim == 1 and nc_var_name in nc_var.dimensions):
                continue
            # Restrict to monotonic?
            if monotonic:
                data = nc_var[:]
                # Gracefully fill a masked coordinate.
                if ma.isMaskedArray(data):
                    data = ma.filled(data)
                if (
                    nc_var.shape == ()
                    or nc_var.shape == (1,)
                    or iris.util.monotonic(data)
                ):
                    result[nc_var_name] = CFCoordinateVariable(
                        nc_var_name, nc_var
                    )
            else:
                result[nc_var_name] = CFCoordinateVariable(nc_var_name, nc_var)

        return result


class CFDataVariable(CFVariable):
    """
    A CF-netCDF variable containing data pay-load that maps to an Iris :class:`iris.cube.Cube`.

    """

    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True):
        raise NotImplementedError


class _CFFormulaTermsVariable(CFVariable):
    """
    A CF-netCDF formula terms variable corresponds to a term in a formula that
    allows dimensional vertical coordinate values to be computed from dimensionless
    vertical coordinate values and associated variables at specific grid points.

    Identified by the CF-netCDF variable attribute 'formula_terms'.

    Ref: [CF] Section 4.3.2. Dimensional Vertical Coordinate.
         [CF] Appendix D. Dimensionless Vertical Coordinates.

    """

    cf_identity = "formula_terms"

    def __init__(self, name, data, formula_root, formula_term):
        CFVariable.__init__(self, name, data)
        # Register the formula root and term relationship.
        self.add_formula_term(formula_root, formula_term)

    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)

        # Identify all CF formula terms variables.
        for nc_var_name, nc_var in target.items():
            # Check for formula terms variable references.
            nc_var_att = getattr(nc_var, cls.cf_identity, None)

            if nc_var_att is not None:
                for match_item in _CF_PARSE.finditer(nc_var_att):
                    match_group = match_item.groupdict()
                    # Ensure that term name is lower case, as expected.
                    term_name = match_group["lhs"].lower()
                    variable_name = match_group["rhs"]

                    if variable_name not in ignore:
                        if variable_name not in variables:
                            if warn:
                                message = "Missing CF-netCDF formula term variable %r, referenced by netCDF variable %r"
                                warnings.warn(
                                    message % (variable_name, nc_var_name)
                                )
                        else:
                            if variable_name not in result:
                                result[
                                    variable_name
                                ] = _CFFormulaTermsVariable(
                                    variable_name,
                                    variables[variable_name],
                                    nc_var_name,
                                    term_name,
                                )
                            else:
                                result[variable_name].add_formula_term(
                                    nc_var_name, term_name
                                )

        return result

    def __repr__(self):
        return "%s(%r, %r, %r)" % (
            self.__class__.__name__,
            self.cf_name,
            self.cf_data,
            self.cf_terms_by_root,
        )


class CFGridMappingVariable(CFVariable):
    """
    A CF-netCDF grid mapping variable contains a list of specific attributes that
    define a particular grid mapping. A CF-netCDF grid mapping variable must contain
    the attribute 'grid_mapping_name'.

    Based on the value of the 'grid_mapping_name' attribute, there are associated
    standard names of CF-netCDF coordinate variables that contain the mapping's
    independent variables.

    Identified by the CF-netCDF variable attribute 'grid_mapping'.

    Ref: [CF] Section 5.6. Horizontal Coordinate Reference Systems, Grid Mappings, and Projections.
         [CF] Appendix F. Grid Mappings.

    """

    cf_identity = "grid_mapping"

    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)

        # Identify all grid mapping variables.
        for nc_var_name, nc_var in target.items():
            # Check for a grid mapping variable reference.
            nc_var_att = getattr(nc_var, cls.cf_identity, None)

            if nc_var_att is not None:
                name = nc_var_att.strip()

                if name not in ignore:
                    if name not in variables:
                        if warn:
                            message = "Missing CF-netCDF grid mapping variable %r, referenced by netCDF variable %r"
                            warnings.warn(message % (name, nc_var_name))
                    else:
                        result[name] = CFGridMappingVariable(
                            name, variables[name]
                        )

        return result


class CFLabelVariable(CFVariable):
    """
    A CF-netCDF CF label variable is any netCDF variable that contain string
    textual information, or labels.

    Identified by the CF-netCDF variable attribute 'coordinates'.
    Also see :class:`iris.fileformats.cf.CFAuxiliaryCoordinateVariable`.

    Ref: [CF] Section 6.1. Labels.

    """

    cf_identity = "coordinates"

    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)

        # Identify all CF label variables.
        for nc_var_name, nc_var in target.items():
            # Check for label variable references.
            nc_var_att = getattr(nc_var, cls.cf_identity, None)

            if nc_var_att is not None:
                for name in nc_var_att.split():
                    if name not in ignore:
                        if name not in variables:
                            if warn:
                                message = "Missing CF-netCDF label variable %r, referenced by netCDF variable %r"
                                warnings.warn(message % (name, nc_var_name))
                        else:
                            # Register variable, but only allow string type.
                            var = variables[name]
                            if _is_str_dtype(var):
                                result[name] = CFLabelVariable(name, var)

        return result

    def cf_label_data(self, cf_data_var):
        """
        Return the associated CF-netCDF label variable strings.

        Args:

        * cf_data_var (:class:`iris.fileformats.cf.CFDataVariable`):
            The CF-netCDF data variable which the CF-netCDF label variable describes.

        Returns:
            String labels.

        """

        if not isinstance(cf_data_var, CFDataVariable):
            raise TypeError(
                "cf_data_var argument should be of type CFDataVariable. Got %r."
                % type(cf_data_var)
            )

        # Determine the name of the label string (or length) dimension by
        # finding the dimension name that doesn't exist within the data dimensions.
        str_dim_name = list(set(self.dimensions) - set(cf_data_var.dimensions))

        if len(str_dim_name) != 1:
            raise ValueError(
                "Invalid string dimensions for CF-netCDF label variable %r"
                % self.cf_name
            )

        str_dim_name = str_dim_name[0]
        label_data = self[:]

        if ma.isMaskedArray(label_data):
            label_data = label_data.filled()

        # Determine whether we have a string-valued scalar label
        # i.e. a character variable that only has one dimension (the length of the string).
        if self.ndim == 1:
            label_string = b"".join(label_data).strip()
            label_string = label_string.decode("utf8")
            data = np.array([label_string])
        else:
            # Determine the index of the string dimension.
            str_dim = self.dimensions.index(str_dim_name)

            # Calculate new label data shape (without string dimension) and create payload array.
            new_shape = tuple(
                dim_len for i, dim_len in enumerate(self.shape) if i != str_dim
            )
            string_basetype = "|U%d"
            string_dtype = string_basetype % self.shape[str_dim]
            data = np.empty(new_shape, dtype=string_dtype)

            for index in np.ndindex(new_shape):
                # Create the slice for the label data.
                if str_dim == 0:
                    label_index = (slice(None, None),) + index
                else:
                    label_index = index + (slice(None, None),)

                label_string = b"".join(label_data[label_index]).strip()
                label_string = label_string.decode("utf8")
                data[index] = label_string

        return data

    def cf_label_dimensions(self, cf_data_var):
        """
        Return the name of the associated CF-netCDF label variable data dimensions.

        Args:

        * cf_data_var (:class:`iris.fileformats.cf.CFDataVariable`):
            The CF-netCDF data variable which the CF-netCDF label variable describes.

        Returns:
            Tuple of label data dimension names.

        """

        if not isinstance(cf_data_var, CFDataVariable):
            raise TypeError(
                "cf_data_var argument should be of type CFDataVariable. Got %r."
                % type(cf_data_var)
            )

        return tuple(
            [
                dim_name
                for dim_name in self.dimensions
                if dim_name in cf_data_var.dimensions
            ]
        )

    def spans(self, cf_variable):
        """
        Determine whether the dimensionality of this variable
        is a subset of the specified target variable.

        Note that, by default scalar variables always span the
        dimensionality of the target variable.

        Args:

        * cf_variable:
            Compare dimensionality with the :class:`CFVariable`.

        Returns:
            Boolean.

        """
        # Scalar variables always span the target variable.
        result = True
        if self.dimensions:
            source = self.dimensions
            target = cf_variable.dimensions
            # Ignore label string length dimension.
            result = set(source[:-1]).issubset(target) or set(
                source[1:]
            ).issubset(target)
        return result


class CFMeasureVariable(CFVariable):
    """
    A CF-netCDF measure variable is a variable that contains cell areas or volumes.

    Identified by the CF-netCDF variable attribute 'cell_measures'.

    Ref: [CF] Section 7.2. Cell Measures.

    """

    cf_identity = "cell_measures"

    def __init__(self, name, data, measure):
        CFVariable.__init__(self, name, data)
        #: Associated cell measure of the cell variable
        self.cf_measure = measure

    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)

        # Identify all CF measure variables.
        for nc_var_name, nc_var in target.items():
            # Check for measure variable references.
            nc_var_att = getattr(nc_var, cls.cf_identity, None)

            if nc_var_att is not None:
                for match_item in _CF_PARSE.finditer(nc_var_att):
                    match_group = match_item.groupdict()
                    measure = match_group["lhs"]
                    variable_name = match_group["rhs"]

                    var_matches_nc = variable_name != nc_var_name
                    if variable_name not in ignore and var_matches_nc:
                        if variable_name not in variables:
                            if warn:
                                message = "Missing CF-netCDF measure variable %r, referenced by netCDF variable %r"
                                warnings.warn(
                                    message % (variable_name, nc_var_name)
                                )
                        else:
                            result[variable_name] = CFMeasureVariable(
                                variable_name,
                                variables[variable_name],
                                measure,
                            )

        return result


################################################################################
class CFGroup(MutableMapping):
    """
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
        """
        :class:`set` of the names of the CF-netCDF variables that are not
        the data pay-load.

        """
        non_data_variables = (
            self.ancillary_variables,
            self.auxiliary_coordinates,
            self.bounds,
            self.climatology,
            self.coordinates,
            self.grid_mappings,
            self.labels,
            self.cell_measures,
        )
        result = set()
        for variable in non_data_variables:
            result |= set(variable)
        return result

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
            raise KeyError(
                "Cannot get unknown CF-netCDF variable name %r" % str(name)
            )

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


################################################################################
class CFReader:
    """
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
    )

    # TODO: remove once iris.experimental.ugrid.CFUGridReader is folded in.
    CFGroup = CFGroup

    def __init__(self, filename, warn=False, monotonic=False):
        self._dataset = None
        self._filename = os.path.expanduser(filename)

        #: Collection of CF-netCDF variables associated with this netCDF file
        self.cf_group = self.CFGroup()

        self._dataset = _thread_safe_nc.DatasetWrapper(
            self._filename, mode="r"
        )

        # Issue load optimisation warning.
        if warn and self._dataset.file_format in [
            "NETCDF3_CLASSIC",
            "NETCDF3_64BIT",
        ]:
            warnings.warn(
                "Optimise CF-netCDF loading by converting data from NetCDF3 "
                'to NetCDF4 file format using the "nccopy" command.'
            )

        self._check_monotonic = monotonic

        self._translate()
        self._build_cf_groups()
        self._reset()

    def __enter__(self):
        # Enable use as a context manager
        # N.B. this **guarantees* closure of the file, when the context is exited.
        # Note: ideally, the class would not do so much work in the __init__ call, and
        # would do all that here, after acquiring necessary permissions/locks.
        # But for legacy reasons, we can't do that.  So **effectively**, the context
        # (in terms of access control) alreday started, when we created the object.
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # When used as a context-manager, **always** close the file on exit.
        self._close()

    @property
    def filename(self):
        """The file that the CFReader is reading."""
        return self._filename

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self._filename)

    def _translate(self):
        """Classify the netCDF variables into CF-netCDF variables."""

        netcdf_variable_names = list(self._dataset.variables.keys())

        # Identify all CF coordinate variables first. This must be done
        # first as, by CF convention, the definition of a CF auxiliary
        # coordinate variable may include a scalar CF coordinate variable,
        # whereas we want these two types of variables to be mutually exclusive.
        coords = CFCoordinateVariable.identify(
            self._dataset.variables, monotonic=self._check_monotonic
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
            self.cf_group.update(
                variable_type.identify(self._dataset.variables, ignore=ignore)
            )

        # Identify global netCDF attributes.
        attr_dict = {
            attr_name: _getncattr(self._dataset, attr_name, "")
            for attr_name in self._dataset.ncattrs()
        }
        self.cf_group.global_attributes.update(attr_dict)

        # Identify and register all CF formula terms.
        formula_terms = _CFFormulaTermsVariable.identify(
            self._dataset.variables
        )

        for cf_var in formula_terms.values():
            for cf_root, cf_term in cf_var.cf_terms_by_root.items():
                # Ignore formula terms owned by a bounds variable.
                if cf_root not in self.cf_group.bounds:
                    cf_name = cf_var.cf_name
                    if cf_var.cf_name not in self.cf_group:
                        self.cf_group[cf_name] = CFAuxiliaryCoordinateVariable(
                            cf_name, cf_var.cf_data
                        )
                    self.cf_group[cf_name].add_formula_term(cf_root, cf_term)

        # Determine the CF data variables.
        data_variable_names = (
            set(netcdf_variable_names) - self.cf_group.non_data_variable_names
        )

        for name in data_variable_names:
            self.cf_group[name] = CFDataVariable(
                name, self._dataset.variables[name]
            )

    def _build_cf_groups(self):
        """Build the first order relationships between CF-netCDF variables."""

        def _build(cf_variable):
            # TODO: isinstance(cf_variable, UGridMeshVariable)
            #  UGridMeshVariable currently in experimental.ugrid - circular import.
            is_mesh_var = cf_variable.cf_identity == "mesh"
            ugrid_coord_names = []
            ugrid_coords = getattr(self.cf_group, "ugrid_coords", None)
            if ugrid_coords is not None:
                ugrid_coord_names = list(ugrid_coords.keys())

            coordinate_names = list(self.cf_group.coordinates.keys())
            cf_group = self.CFGroup()

            # Build CF variable relationships.
            for variable_type in self._variable_types:
                ignore = []
                # Avoid UGridAuxiliaryCoordinateVariables also being
                # processed as CFAuxiliaryCoordinateVariables.
                if not is_mesh_var:
                    ignore += ugrid_coord_names
                # Prevent grid mapping variables being mis-identified as CF coordinate variables.
                if not issubclass(variable_type, CFGridMappingVariable):
                    ignore += coordinate_names

                match = variable_type.identify(
                    self._dataset.variables,
                    ignore=ignore,
                    target=cf_variable.cf_name,
                    warn=False,
                )
                # Sanity check dimensionality coverage.
                for cf_name, cf_var in match.items():
                    # No span check is necessary if variable is attached to a mesh.
                    if is_mesh_var or cf_var.spans(cf_variable):
                        cf_group[cf_name] = self.cf_group[cf_name]
                    else:
                        # Register the ignored variable.
                        # N.B. 'ignored' variable from enclosing scope.
                        ignored.add(cf_name)
                        msg = (
                            "Ignoring variable {!r} referenced "
                            "by variable {!r}: Dimensions {!r} do not "
                            "span {!r}".format(
                                cf_name,
                                cf_variable.cf_name,
                                cf_var.dimensions,
                                cf_variable.dimensions,
                            )
                        )
                        warnings.warn(msg)

            # Build CF data variable relationships.
            if isinstance(cf_variable, CFDataVariable):
                # Add global netCDF attributes.
                cf_group.global_attributes.update(
                    self.cf_group.global_attributes
                )
                # Add appropriate "dimensioned" CF coordinate variables.
                cf_group.update(
                    {
                        cf_name: self.cf_group[cf_name]
                        for cf_name in cf_variable.dimensions
                        if cf_name in self.cf_group.coordinates
                    }
                )
                # Add appropriate "dimensionless" CF coordinate variables.
                coordinates_attr = getattr(cf_variable, "coordinates", "")
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
                        if (
                            cf_root in cf_group
                            and cf_var.cf_name not in cf_group
                        ):
                            # Sanity check dimensionality.
                            if cf_var.spans(cf_variable):
                                cf_group[cf_var.cf_name] = cf_var
                            else:
                                # Register the ignored variable.
                                # N.B. 'ignored' variable from enclosing scope.
                                ignored.add(cf_var.cf_name)
                                msg = (
                                    "Ignoring formula terms variable {!r} "
                                    "referenced by data variable {!r} via "
                                    "variable {!r}: Dimensions {!r} do not "
                                    "span {!r}".format(
                                        cf_var.cf_name,
                                        cf_variable.cf_name,
                                        cf_root,
                                        cf_var.dimensions,
                                        cf_variable.dimensions,
                                    )
                                )
                                warnings.warn(msg)

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
            for cf_root, cf_term in cf_var.cf_terms_by_root.items():
                cf_root_var = self.cf_group[cf_root]
                name = cf_root_var.standard_name or cf_root_var.long_name
                terms = reference_terms.get(name, [])
                if isinstance(terms, str) or not isinstance(terms, Iterable):
                    terms = [terms]
                cf_var_name = cf_var.cf_name
                if (
                    cf_term in terms
                    and cf_var_name not in self.cf_group.promoted
                ):
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
                data_var = CFDataVariable(
                    cf_name, self.cf_group[cf_name].cf_data
                )
                self.cf_group.promoted[cf_name] = data_var
                _build(data_var)
            # Determine whether there are still any ignored variables
            # yet to be promoted.
            promoted.add(cf_name)
            not_promoted = ignored.difference(promoted)

    def _reset(self):
        """Reset the attribute touch history of each variable."""
        for nc_var_name in self._dataset.variables.keys():
            self.cf_group[nc_var_name].cf_attrs_reset()

    def _close(self):
        # Explicitly close dataset to prevent file remaining open.
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None

    def __del__(self):
        # Be sure to close dataset when CFReader is destroyed / garbage-collected.
        self._close()


def _getncattr(dataset, attr, default=None):
    """
    Simple wrapper round `netCDF4.Dataset.getncattr` to make it behave
    more like `getattr`.

    """
    try:
        value = dataset.getncattr(attr)
    except AttributeError:
        value = default
    return value
