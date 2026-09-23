# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Classify netCDF variables by the role the CF conventions give them.

A CF-netCDF file is a flat bag of variables; the conventions turn it into a
structure by having variables name each other in their attributes. A
``bounds`` attribute names a bounds variable, a ``grid_mapping`` attribute
names a coordinate system variable, and so on. This module is where those
attributes are read and each variable is given a class that says what it is.

:class:`CFVariable` is the abstract base and defines the contract. A subclass
declares which netCDF attribute names it -- ``cf_identity``, a single
attribute name, or ``cf_identities``, a list of them for the classes that
answer to several -- and implements ``identify``. ``identify`` is a
classmethod, not an instance method: it is called on the class, is handed the
whole ``{name: netCDF4.Variable}`` mapping of the file, and returns the subset
of it that belongs to that class, as a ``{name: CFVariable instance}``
mapping. ``ignore`` and ``target`` narrow what it looks at; ``warn`` gates
whether it complains.

Classification is deliberately forgiving. A variable that names a variable
which is not in the file, or names one whose dimensions make no sense for the
role, produces a warning -- :class:`~iris.warnings.IrisCfMissingVarWarning`
and friends -- and is skipped. It does not raise. Iris is expected to load
what it can from a file that is only mostly conformant, and a hard failure
here would make a single bad attribute cost the user the whole file.

The CF-UGRID classes for unstructured meshes live here rather than in
:mod:`iris.mesh` because they are classifiers, not data: they answer the same
``identify`` contract as every other class in this module and are driven by
the same pass in :class:`~iris.fileformats.cf.CFReader`. What they identify is
handed to :mod:`iris.mesh` to build the mesh proper.

This module opens no files. It is given netCDF variables and returns
classifications; :mod:`iris.fileformats.cf._group` collects them and
:mod:`iris.fileformats.cf._reader` drives the whole thing. It does read array
data in exactly one place: ``CFCoordinateVariable.identify`` fetches a
candidate's values when called with ``monotonic=True``, because whether a
coordinate is monotonic cannot be told from its metadata. That is the only
path here that touches data, and it is reached only from
``CFReader(..., monotonic=True)``.

"""

from abc import ABCMeta, abstractmethod
import re
from typing import ClassVar
import warnings

import numpy as np
import numpy.ma as ma

from iris.mesh.components import Connectivity
import iris.util
import iris.warnings
from iris.warnings import IrisCfLabelVarWarning, IrisCfMissingVarWarning

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

# NCZarr stores scalar variables as size-1 arrays on this pseudo-dimension.
_NCZARR_SCALAR_DIMENSION = "_scalar_"

# NetCDF variable attributes handled by the netCDF4 module and
# therefore automatically classed as "used" attributes.
_CF_ATTRS_IGNORE = set(["_FillValue", "add_offset", "missing_value", "scale_factor"])


# NetCDF returns a different type for strings depending on Python version.
def _is_str_dtype(var):
    # N.B. use 'datatype' not 'dtype', to "look inside" variable wrappers which
    #  represent 'S1' type data as 'U<xx>'.
    return np.dtype(var.dtype).kind in "SU"


################################################################################
class CFVariable(metaclass=ABCMeta):
    """Abstract base class wrapper for a CF-netCDF variable."""

    #: Name of the netCDF variable attribute that identifies this
    #: CF-netCDF variable.
    cf_identity: ClassVar[str | None] = None

    def __init__(self, name, data):
        # Accessing the list of netCDF attributes is surprisingly slow.
        # Since it's used repeatedly, caching the list makes things
        # quite a bit faster.
        self._nc_attrs = data.ncattrs()

        self.cf_name = name
        """NetCDF variable name."""

        self.cf_data = data
        """NetCDF4 Variable data instance."""

        """File source of the NetCDF content."""
        try:
            self.filename = data.group().filepath()
        except AttributeError:
            self.filename = "<unknown_filename>"

        self.cf_group = None
        """Collection of CF-netCDF variables associated with this variable."""

        self.cf_terms_by_root = {}
        """CF-netCDF formula terms that his variable participates in."""

        self._to_be_promoted = False

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
                    "Cannot identify unknown target CF-netCDF variable %r" % target
                )
            target = {target: variables[target]}
        else:
            raise TypeError("Expect a target CF-netCDF variable name")

        return (ignore, target)

    @abstractmethod
    def identify(self, variables, ignore=None, target=None, warn=True):
        """Identify all variables that match the criterion for this CF-netCDF variable class.

        Parameters
        ----------
        variables :
            Dictionary of netCDF4.Variable instance by variable name.
        ignore : optional
            List of variable names to ignore.
        target : optional
            Name of a single variable to check.
        warn : bool, default=True
            Issue a warning if a missing variable is referenced.

        Returns
        -------
        Dictionary of CFVariable instance by variable name.

        """
        pass

    def spans(self, cf_variable):
        """Determine dimensionality coverage.

        Determine whether the dimensionality of this variable
        is a subset of the specified target variable.

        Note that, by default scalar variables always span the
        dimensionality of the target variable.

        Parameters
        ----------
        cf_variable :
            Compare dimensionality with the :class:`CFVariable`.

        Returns
        -------
        bool

        """
        dimensions = tuple(self.dimensions)
        if dimensions == (_NCZARR_SCALAR_DIMENSION,):
            return True

        result = set(dimensions).issubset(cf_variable.dimensions)
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
        return tuple((attr, self.getncattr(attr)) for attr in sorted(self._nc_attrs))

    def cf_attrs_ignored(self):
        """Return a list of all ignored attribute name and value pairs of the CF-netCDF variable."""
        return tuple(
            (attr, self.getncattr(attr))
            for attr in sorted(set(self._nc_attrs) & _CF_ATTRS_IGNORE)
        )

    def cf_attrs_used(self):
        """Return a list of all accessed attribute name and value pairs of the CF-netCDF variable."""
        return tuple((attr, self.getncattr(attr)) for attr in sorted(self._cf_attrs))

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
        """Register the participation of this CF-netCDF variable in a CF-netCDF formula term.

        Parameters
        ----------
        root : str
            The name of CF-netCDF variable that defines the CF-netCDF
            formula_terms attribute.
        term : str
            The associated term name of this variable in the formula_terms
            definition.

        Returns
        -------
        None

        """
        self.cf_terms_by_root[root] = term

    def has_formula_terms(self):
        """Determine whether this CF-netCDF variable participates in a CF-netcdf formula term.

        Returns
        -------
        bool

        """
        return bool(self.cf_terms_by_root)


class CFAncillaryDataVariable(CFVariable):
    """CF-netCDF ancillary data variable.

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
                                warnings.warn(
                                    message % (name, nc_var_name),
                                    category=iris.warnings.IrisCfMissingVarWarning,
                                )
                        else:
                            result[name] = CFAncillaryDataVariable(
                                name, variables[name]
                            )

        return result


class CFAuxiliaryCoordinateVariable(CFVariable):
    """CF-netCDF auxiliary coordinate variable.

    A CF-netCDF auxiliary coordinate variable is any netCDF variable that contains
    coordinate data, but is not a CF-netCDF coordinate variable by definition.

    There is no relationship between the name of a CF-netCDF auxiliary coordinate
    variable and the name(s) of its dimension(s).

    Identified by the CF-netCDF variable attribute 'coordinates'.
    Also see :class:`iris.fileformats.cf.CFLabelVariable`.

    Ref:

    * [CF] Chapter 5. Coordinate Systems.
    * [CF] Section 6.2. Alternative Coordinates.

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
                                warnings.warn(
                                    message % (name, nc_var_name),
                                    category=iris.warnings.IrisCfMissingVarWarning,
                                )
                        else:
                            # Restrict to non-string type i.e. not a CFLabelVariable.
                            if not _is_str_dtype(variables[name]):
                                result[name] = CFAuxiliaryCoordinateVariable(
                                    name, variables[name]
                                )

        return result


class CFBoundaryVariable(CFVariable):
    """CF-netCDF boundary variable.

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
                            warnings.warn(
                                message % (name, nc_var_name),
                                category=iris.warnings.IrisCfMissingVarWarning,
                            )
                    else:
                        result[name] = CFBoundaryVariable(name, variables[name])

        return result

    def spans(self, cf_variable):
        """Determine dimensionality coverage.

        Determine whether the dimensionality of this variable
        is a subset of the specified target variable.

        Note that, by default scalar variables always span the
        dimensionality of the target variable.

        Parameters
        ----------
        cf_variable :
            Compare dimensionality with the :class:`CFVariable`.

        Returns
        -------
        bool

        """
        # Scalar variables always span the target variable.
        result = True
        if self.dimensions:
            source = self.dimensions
            target = cf_variable.dimensions
            # Ignore the bounds extent dimension.
            result = set(source[:-1]).issubset(target) or set(source[1:]).issubset(
                target
            )
        return result


class CFClimatologyVariable(CFVariable):
    """CF-netCDF climatology variable.

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
                            warnings.warn(
                                message % (name, nc_var_name),
                                category=iris.warnings.IrisCfMissingVarWarning,
                            )
                    else:
                        result[name] = CFClimatologyVariable(name, variables[name])

        return result

    def spans(self, cf_variable):
        """Determine dimensionality coverage.

        Determine whether the dimensionality of this variable
        is a subset of the specified target variable.

        Note that, by default scalar variables always span the
        dimensionality of the target variable.

        Parameters
        ----------
        cf_variable : :class:`CFVariable`
            Compare dimensionality with the :class:`CFVariable`.

        Returns
        -------
        bool

        """
        # Scalar variables always span the target variable.
        result = True
        if self.dimensions:
            source = self.dimensions
            target = cf_variable.dimensions
            # Ignore the climatology extent dimension.
            result = set(source[:-1]).issubset(target) or set(source[1:]).issubset(
                target
            )
        return result


class CFCoordinateVariable(CFVariable):
    """A CF-netCDF coordinate variable.

    A CF-netCDF coordinate variable is a one-dimensional variable with the same name
    as its dimension, and it is defined as a numeric data type with values that are
    ordered monotonically. Missing values are not allowed in CF-netCDF coordinate
    variables. Also see [NUG] Section 2.3.1.

    Identified by the above criterion, there is no associated CF-netCDF variable
    attribute.

    Ref: [CF] 1.2. Terminology.

    """

    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True, monotonic=False):
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
                    result[nc_var_name] = CFCoordinateVariable(nc_var_name, nc_var)
            else:
                result[nc_var_name] = CFCoordinateVariable(nc_var_name, nc_var)

        return result


class CFDataVariable(CFVariable):
    """A CF-netCDF variable containing data pay-load that maps to an Iris :class:`iris.cube.Cube`."""

    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True):
        raise NotImplementedError


class _CFFormulaTermsVariable(CFVariable):
    """CF-netCDF formula terms variable.

    A CF-netCDF formula terms variable corresponds to a term in a formula that
    allows dimensional vertical coordinate values to be computed from dimensionless
    vertical coordinate values and associated variables at specific grid points.

    Identified by the CF-netCDF variable attribute 'formula_terms'.

    Ref:

    * [CF] Section 4.3.2. Dimensional Vertical Coordinate.
    * [CF] Appendix D. Dimensionless Vertical Coordinates.

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
                                    message % (variable_name, nc_var_name),
                                    category=iris.warnings.IrisCfMissingVarWarning,
                                )
                        else:
                            if variable_name not in result:
                                result[variable_name] = _CFFormulaTermsVariable(
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
    """CF-netCDF grid mapping variable.

    A CF-netCDF grid mapping variable contains a list of specific attributes that
    define a particular grid mapping. A CF-netCDF grid mapping variable must contain
    the attribute 'grid_mapping_name'.

    Based on the value of the 'grid_mapping_name' attribute, there are associated
    standard names of CF-netCDF coordinate variables that contain the mapping's
    independent variables.

    Identified by the CF-netCDF variable attribute 'grid_mapping'.

    Ref:

    * [CF] Section 5.6. Horizontal Coordinate Reference Systems, Grid Mappings, and Projections.
    * [CF] Appendix F. Grid Mappings.

    """

    cf_identity = "grid_mapping"

    @classmethod
    def identify(
        cls, variables, ignore=None, target=None, warn=True, coord_system_mappings=None
    ):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)

        # Identify all grid mapping variables.
        for nc_var_name, nc_var in target.items():
            # Check for a grid mapping variable reference.
            nc_var_att = getattr(nc_var, cls.cf_identity, None)

            if nc_var_att is not None:
                # All `grid_mapping` attributes will already have been parsed prior
                # to `identify` being called and passed in as an argument. We can
                # ignore the attribute here (it's just used to identify that a grid
                # mapping exists for this data variable) and get the pre-parsed
                # mapping from the `coord_mapping_systems` keyword:
                cs_mappings = None
                if coord_system_mappings:
                    cs_mappings = coord_system_mappings.get(nc_var_name, None)

                if not cs_mappings:
                    # If cs_mappings is None, some parse error must have occurred and the
                    # user will have already been warned by `_parse_extended_grid_mappings`
                    continue

                # group the cs_mappings by coordinate system, as we want to iterate over coord systems:
                uniq_cs = set(cs_mappings.values())
                cs_coord_mappings = {
                    cs: [
                        coord
                        for coord, coord_cs in cs_mappings.items()
                        if cs == coord_cs
                    ]
                    for cs in uniq_cs
                }

                for name, coords in cs_coord_mappings.items():
                    if name not in ignore:
                        if name not in variables:
                            if warn:
                                message = "Missing CF-netCDF grid mapping variable %r, referenced by netCDF variable %r"
                                warnings.warn(
                                    message % (name, nc_var_name),
                                    category=iris.warnings.IrisCfMissingVarWarning,
                                )
                        else:
                            # For extended grid_mapping, also check coord references exist:
                            has_a_valid_coord = False
                            if coords:
                                for coord_name in coords:
                                    # coord_name could be None if simple grid_mapping is used.
                                    if coord_name is None or (
                                        coord_name and coord_name in variables
                                    ):
                                        has_a_valid_coord = True
                                    else:
                                        message = "Missing CF-netCDF coordinate variable %r (associated with grid mapping variable %r), referenced by netCDF variable %r"
                                        warnings.warn(
                                            message % (coord_name, name, nc_var_name),
                                            category=iris.warnings.IrisCfMissingVarWarning,
                                        )
                            #  Only add as a CFGridMappingVariable if at least one of its referenced coords exists:
                            if has_a_valid_coord:
                                result[name] = CFGridMappingVariable(
                                    name, variables[name]
                                )
        return result


class CFLabelVariable(CFVariable):
    """Cariable is any netCDF variable that contain string textual information, or labels.

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
                                warnings.warn(
                                    message % (name, nc_var_name),
                                    category=iris.warnings.IrisCfMissingVarWarning,
                                )
                        else:
                            # Register variable, but only allow string type.
                            var = variables[name]
                            if _is_str_dtype(var):
                                result[name] = CFLabelVariable(name, var)

        return result

    def cf_label_dimensions(self, cf_data_var):
        """Return the name of the associated CF-netCDF label variable data dimensions.

        Parameters
        ----------
        cf_data_var : :class:`iris.fileformats.cf.CFDataVariable`
            The CF-netCDF data variable which the CF-netCDF label variable
            describes.

        Returns
        -------
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
        """Determine dimensionality coverage.

        Determine whether the dimensionality of this variable
        is a subset of the specified target variable.

        Note that, by default scalar variables always span the
        dimensionality of the target variable.

        Parameters
        ----------
        cf_variable :
            Compare dimensionality with the :class:`CFVariable`.

        Returns
        -------
        bool

        """
        # Scalar variables always span the target variable.
        result = True
        if self.dimensions:
            source = self.dimensions
            target = cf_variable.dimensions
            # Ignore label string length dimension.
            result = set(source[:-1]).issubset(target) or set(source[1:]).issubset(
                target
            )
        return result


class CFMeasureVariable(CFVariable):
    """A CF-netCDF measure variable is a variable that contains cell areas or volumes.

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
                                    message % (variable_name, nc_var_name),
                                    category=iris.warnings.IrisCfMissingVarWarning,
                                )
                        else:
                            result[variable_name] = CFMeasureVariable(
                                variable_name,
                                variables[variable_name],
                                measure,
                            )

        return result


class CFUGridConnectivityVariable(CFVariable):
    """A CF_UGRID connectivity variable.

    A CF_UGRID connectivity variable points to an index variable identifying
    for every element (edge/face/volume) the indices of its corner nodes. The
    connectivity array will thus be a matrix of size n-elements x n-corners.
    For the indexing one may use either 0- or 1-based indexing; the convention
    used should be specified using a ``start_index`` attribute to the index
    variable.

    For face elements: the corner nodes should be specified in anticlockwise
    direction as viewed from above. For volume elements: use the
    additional attribute ``volume_shape_type`` which points to a flag variable
    that specifies for every volume its shape.

    Identified by a CF-netCDF variable attribute equal to any one of the values
    in :attr:`~iris.mesh.Connectivity.UGRID_CF_ROLES`.

    .. seealso::

        The UGRID Conventions, https://ugrid-conventions.github.io/ugrid-conventions/

    """

    cf_identity = NotImplemented
    cf_identities = Connectivity.UGRID_CF_ROLES

    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)

        # Identify all CF-UGRID connectivity variables.
        for nc_var_name, nc_var in target.items():
            # Check for connectivity variable references, iterating through
            # the valid cf roles.
            for identity in cls.cf_identities:
                nc_var_att = getattr(nc_var, identity, None)

                if nc_var_att is not None:
                    # UGRID only allows for one of each connectivity cf role.
                    name = nc_var_att.strip()
                    if name not in ignore:
                        if name not in variables:
                            message = (
                                f"Missing CF-UGRID connectivity variable "
                                f"{name}, referenced by netCDF variable "
                                f"{nc_var_name}"
                            )
                            if warn:
                                warnings.warn(message, category=IrisCfMissingVarWarning)
                        else:
                            # Restrict to non-string type i.e. not a
                            # CFLabelVariable.
                            if not _is_str_dtype(variables[name]):
                                result[name] = CFUGridConnectivityVariable(
                                    name, variables[name]
                                )
                            else:
                                message = (
                                    f"Ignoring variable {name}, identified "
                                    f"as a CF-UGRID connectivity - is a "
                                    f"CF-netCDF label variable."
                                )
                                if warn:
                                    warnings.warn(
                                        message, category=IrisCfLabelVarWarning
                                    )

        return result


class CFUGridAuxiliaryCoordinateVariable(CFVariable):
    """A CF-UGRID auxiliary coordinate variable.

    A CF-UGRID auxiliary coordinate variable is a CF-netCDF auxiliary
    coordinate variable representing the element (node/edge/face/volume)
    locations (latitude, longitude or other spatial coordinates, and optional
    elevation or other coordinates). These auxiliary coordinate variables will
    have length n-elements.

    For elements other than nodes, these auxiliary coordinate variables may
    have in turn a ``bounds`` attribute that specifies the bounding coordinates
    of the element (thereby duplicating the data in the ``node_coordinates``
    variables).

    Identified by the CF-netCDF variable attribute
    ``node_``/``edge_``/``face_``/``volume_coordinates``.

    .. seealso::

        The UGRID Conventions, https://ugrid-conventions.github.io/ugrid-conventions/

    """

    cf_identity = NotImplemented
    cf_identities = [
        "node_coordinates",
        "edge_coordinates",
        "face_coordinates",
        "volume_coordinates",
    ]

    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)

        # Identify any CF-UGRID-relevant auxiliary coordinate variables.
        for nc_var_name, nc_var in target.items():
            # Check for UGRID auxiliary coordinate variable references.
            for identity in cls.cf_identities:
                nc_var_att = getattr(nc_var, identity, None)

                if nc_var_att is not None:
                    for name in nc_var_att.split():
                        if name not in ignore:
                            if name not in variables:
                                message = (
                                    f"Missing CF-netCDF auxiliary coordinate "
                                    f"variable {name}, referenced by netCDF "
                                    f"variable {nc_var_name}"
                                )
                                if warn:
                                    warnings.warn(
                                        message,
                                        category=IrisCfMissingVarWarning,
                                    )
                            else:
                                # Restrict to non-string type i.e. not a
                                # CFLabelVariable.
                                if not _is_str_dtype(variables[name]):
                                    result[name] = CFUGridAuxiliaryCoordinateVariable(
                                        name, variables[name]
                                    )
                                else:
                                    message = (
                                        f"Ignoring variable {name}, "
                                        f"identified as a CF-netCDF "
                                        f"auxiliary coordinate - is a "
                                        f"CF-netCDF label variable."
                                    )
                                    if warn:
                                        warnings.warn(
                                            message,
                                            category=IrisCfLabelVarWarning,
                                        )

        return result


class CFUGridMeshVariable(CFVariable):
    """A CF-UGRID mesh variable is a dummy variable for storing topology information as attributes.

    A CF-UGRID mesh variable is a dummy variable for storing topology
    information as attributes. The mesh variable has the ``cf_role``
    'mesh_topology'.

    The UGRID conventions describe define the mesh topology as the
    interconnection of various geometrical elements of the mesh. The pure
    interconnectivity is independent of georeferencing the individual
    geometrical elements, but for the practical applications for which the
    UGRID CF extension is defined, coordinate data will always be added.

    Identified by the CF-netCDF variable attribute 'mesh'.

    .. seealso::

        The UGRID Conventions, https://ugrid-conventions.github.io/ugrid-conventions/

    """

    cf_identity = "mesh"

    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)

        # Identify all CF-UGRID mesh variables.
        all_vars = target == variables
        for nc_var_name, nc_var in target.items():
            if all_vars:
                # SPECIAL BEHAVIOUR FOR MESH VARIABLES.
                # We are looking for all mesh variables. Check if THIS variable
                #  is a mesh using its own attributes.
                if getattr(nc_var, "cf_role", "") == "mesh_topology":
                    result[nc_var_name] = CFUGridMeshVariable(nc_var_name, nc_var)

            # Check for mesh variable references.
            nc_var_att = getattr(nc_var, cls.cf_identity, None)

            if nc_var_att is not None:
                # UGRID only allows for 1 mesh per variable.
                name = nc_var_att.strip()
                if name not in ignore:
                    if name not in variables:
                        message = (
                            f"Missing CF-UGRID mesh variable {name}, "
                            f"referenced by netCDF variable {nc_var_name}"
                        )
                        if warn:
                            warnings.warn(message, category=IrisCfMissingVarWarning)
                    else:
                        # Restrict to non-string type i.e. not a
                        # CFLabelVariable.
                        if not _is_str_dtype(variables[name]):
                            result[name] = CFUGridMeshVariable(name, variables[name])
                        else:
                            message = (
                                f"Ignoring variable {name}, identified as a "
                                f"CF-UGRID mesh - is a CF-netCDF label "
                                f"variable."
                            )
                            if warn:
                                warnings.warn(message, category=IrisCfLabelVarWarning)

        return result
