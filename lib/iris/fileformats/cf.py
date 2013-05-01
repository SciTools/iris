# (C) British Crown Copyright 2010 - 2013, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
Provides the capability to load netCDF files and interprete them 
according to the 'NetCDF Climate and Forecast (CF) Metadata Conventions'. 

References:

[CF]  NetCDF Climate and Forecast (CF) Metadata conventions, Version 1.5, October, 2010.
[NUG] NetCDF User's Guide, http://www.unidata.ucar.edu/software/netcdf/docs/netcdf.html

"""

from abc import ABCMeta, abstractmethod
import os
import re
import UserDict
import warnings

import netCDF4
import numpy as np
import numpy.ma as ma

import iris.util


#
# CF parse pattern common to both formula terms and measure CF variables.
#
_CF_PARSE = re.compile(r'''
                           \s*
                           (?P<lhs>[\w_]+)
                           \s*:\s*
                           (?P<rhs>[\w_]+)
                           \s*
                        ''', re.VERBOSE)

# NetCDF variable attributes handled by the netCDF4 module and
# therefore automatically classed as "used" attributes.
_CF_ATTRS_IGNORE = set(['_FillValue', 'add_offset', 'missing_value', 'scale_factor', ])


################################################################################
class CFVariable(object):
    """Abstract base class wrapper for a CF-netCDF variable."""

    __metaclass__ = ABCMeta

    cf_identity = None
    '''Name of the netCDF variable attribute that identifies this CF-netCDF variable'''
    
    def __init__(self, name, data):
        # Accessing the list of netCDF attributes is surprisingly slow.
        # Since it's used repeatedly, caching the list makes things
        # quite a bit faster.
        self._nc_attrs = data.ncattrs()

        self.cf_name = name
        '''NetCDF variable name'''

        self.cf_data = data
        '''NetCDF4 Variable data instance'''

        self.cf_group = None
        '''Collection of CF-netCDF variables associated with this variable'''

        self.cf_terms_by_root = {}
        '''CF-netCDF formula terms that his variable participates in'''

        self.cf_attrs_reset()

    @staticmethod
    def _identify_common(variables, ignore, target):
        if ignore is None:
            ignore = []
            
        if target is None:
            target = variables
        elif isinstance(target, basestring):
            if target not in variables:
                raise ValueError('Cannot identify unknown target CF-netCDF variable %r' % target)
            target = {target: variables[target]}
        else:
            raise TypeError('Expect a target CF-netCDF variable name')
    
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

    def __eq__(self, other):
        # CF variable names are unique.
        return self.cf_name == other.cf_name

    def __ne__(self, other):
        # CF variable names are unique.
        return self.cf_name != other.cf_name

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
        return '%s(%r, %r)' % (self.__class__.__name__, self.cf_name, self.cf_data)

    def cf_attrs(self):
        """Return a list of all attribute name and value pairs of the CF-netCDF variable."""
        return tuple((attr, self.getncattr(attr))
                        for attr in sorted(self._nc_attrs))

    def cf_attrs_ignored(self):
        """Return a list of all ignored attribute name and value pairs of the CF-netCDF variable."""
        return tuple((attr, self.getncattr(attr)) for attr in
                        sorted(set(self._nc_attrs) & _CF_ATTRS_IGNORE))

    def cf_attrs_used(self):
        """Return a list of all accessed attribute name and value pairs of the CF-netCDF variable."""
        return tuple((attr, self.getncattr(attr)) for attr in
                        sorted(self._cf_attrs))

    def cf_attrs_unused(self):
        """Return a list of all non-accessed attribute name and value pairs of the CF-netCDF variable."""
        return tuple((attr, self.getncattr(attr)) for attr in
                        sorted(set(self._nc_attrs) - self._cf_attrs))

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
    cf_identity = 'ancillary_variables'
    
    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)
        netcdf_variable_names = variables.keys()

        # Identify all CF ancillary data variables.
        for nc_var_name, nc_var in target.iteritems():
            # Check for ancillary data variable references.
            nc_var_att = getattr(nc_var, cls.cf_identity, None)

            if nc_var_att is not None:
                for name in nc_var_att.split():
                    if name not in ignore:
                        if name not in netcdf_variable_names:
                            if warn:
                                message = 'Missing CF-netCDF ancillary data variable %r, referenced by netCDF variable %r'
                                warnings.warn(message % (name, nc_var_name))
                        else:
                            result[name] = CFAncillaryDataVariable(name, variables[name])

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
    cf_identity = 'coordinates'

    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)
        netcdf_variable_names = variables.keys()

        # Identify all CF auxiliary coordinate variables.
        for nc_var_name, nc_var in target.iteritems():
            # Check for auxiliary coordinate variable references.
            nc_var_att = getattr(nc_var, cls.cf_identity, None)

            if nc_var_att is not None:
                for name in nc_var_att.split():
                    if name not in ignore:
                        if name not in netcdf_variable_names:
                            if warn:
                                message = 'Missing CF-netCDF auxiliary coordinate variable %r, referenced by netCDF variable %r'
                                warnings.warn(message % (name, nc_var_name))
                        else:
                            # Restrict to non-string type i.e. not a CFLabelVariable.
                            if not np.issubdtype(variables[name].dtype, np.str):
                                result[name] = CFAuxiliaryCoordinateVariable(name, variables[name])

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
    cf_identity = 'bounds'

    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)
        netcdf_variable_names = variables.keys()

        # Identify all CF boundary variables.
        for nc_var_name, nc_var in target.iteritems():
            # Check for a boundary variable reference.
            nc_var_att = getattr(nc_var, cls.cf_identity, None)

            if nc_var_att is not None:
                name = nc_var_att.strip()

                if name not in ignore:
                    if name not in netcdf_variable_names:
                        if warn:
                            message = 'Missing CF-netCDF boundary variable %r, referenced by netCDF variable %r'
                            warnings.warn(message % (name, nc_var_name))
                    else:
                        result[name] = CFBoundaryVariable(name, variables[name])

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
    cf_identity = 'climatology'

    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)
        netcdf_variable_names = variables.keys()

        # Identify all CF climatology variables.
        for nc_var_name, nc_var in target.iteritems():
            # Check for a climatology variable reference.
            nc_var_att = getattr(nc_var, cls.cf_identity, None)

            if nc_var_att is not None:
                name = nc_var_att.strip()

                if name not in ignore:
                    if name not in netcdf_variable_names:
                        if warn:
                            message = 'Missing CF-netCDF climatology variable %r, referenced by netCDF variable %r'
                            warnings.warn(message % (name, nc_var_name))
                    else:
                        result[name] = CFClimatologyVariable(name, variables[name])

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
    def identify(cls, variables, ignore=None, target=None, warn=True, monotonic=False):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)

        # Identify all CF coordinate variables.
        for nc_var_name, nc_var in target.iteritems():
            if nc_var_name in ignore:
                continue
            # String variables can't be coordinates
            if np.issubdtype(nc_var.dtype, np.str):
                continue
            # Restrict to one-dimensional with name as dimension OR zero-dimensional scalar
            if not ((nc_var.ndim == 1 and nc_var_name in nc_var.dimensions) or (nc_var.ndim == 0)):
                continue
            # Restrict to monotonic?
            if monotonic:
                data = nc_var[:]
                # Gracefully fill a masked coordinate.
                if ma.isMaskedArray(data):
                    data = ma.filled(data)
                if nc_var.shape == () or nc_var.shape == (1,) or iris.util.monotonic(data):
                    result[nc_var_name] = CFCoordinateVariable(nc_var_name, nc_var)
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
    cf_identity = 'formula_terms'

    def __init__(self, name, data, formula_root, formula_term):
        CFVariable.__init__(self, name, data)
        self.cf_root = formula_root
        '''CF-netCDF variable name that defines the formula terms'''
        self.cf_term = formula_term
        '''Formula term of the associated formula variable'''

    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)
        netcdf_variable_names = variables.keys()

        # Identify all CF formula terms variables.
        for nc_var_name, nc_var in target.iteritems():
            # Check for formula terms variable references.
            nc_var_att = getattr(nc_var, cls.cf_identity, None)

            if nc_var_att is not None:
                for match_item in _CF_PARSE.finditer(nc_var_att):
                    match_group = match_item.groupdict()
                    term_name = match_group['lhs']
                    variable_name = match_group['rhs']

                    if variable_name not in ignore:
                        if variable_name not in netcdf_variable_names:
                            if warn:
                                message = 'Missing CF-netCDF formula term variable %r, referenced by netCDF variable %r'
                                warnings.warn(message % (variable_name, nc_var_name))
                        else:
                            result[variable_name] = _CFFormulaTermsVariable(variable_name, 
                                                                            variables[variable_name], 
                                                                            nc_var_name, term_name)

        return result
    
    def __repr__(self):
        return '%s(%r, %r, %r, %r)' % (self.__class__.__name__, 
                                       self.cf_name, self.cf_data, 
                                       self.cf_root, self.cf_term)


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
    cf_identity = 'grid_mapping'

    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)
        netcdf_variable_names = variables.keys()

        # Identify all grid mapping variables.
        for nc_var_name, nc_var in target.iteritems():
            # Check for a grid mapping variable reference.
            nc_var_att = getattr(nc_var, cls.cf_identity, None)

            if nc_var_att is not None:
                name = nc_var_att.strip()
                
                if name not in ignore:
                    if name not in netcdf_variable_names:
                        if warn:
                            message = 'Missing CF-netCDF grid mapping variable %r, referenced by netCDF variable %r'
                            warnings.warn(message % (name, nc_var_name))
                    else:
                        result[name] = CFGridMappingVariable(name, variables[name])

        return result


class CFLabelVariable(CFVariable):
    """
    A CF-netCDF CF label variable is any netCDF variable that contain string
    textual information, or labels.

    Identified by the CF-netCDF variable attribute 'coordinates'.
    Also see :class:`iris.fileformats.cf.CFAuxiliaryCoordinateVariable`.

    Ref: [CF] Section 6.1. Labels.

    """
    cf_identity = 'coordinates'

    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)
        netcdf_variable_names = variables.keys()

        # Identify all CF label variables.
        for nc_var_name, nc_var in target.iteritems():
            # Check for label variable references.
            nc_var_att = getattr(nc_var, cls.cf_identity, None)

            if nc_var_att is not None:
                for name in nc_var_att.split():
                    if name not in ignore:
                        if name not in netcdf_variable_names:
                            if warn:
                                message = 'Missing CF-netCDF label variable %r, referenced by netCDF variable %r'
                                warnings.warn(message % (name, nc_var_name))
                        else:
                            # Restrict to only string type.
                            if np.issubdtype(variables[name].dtype, np.str):
                                result[name] = CFLabelVariable(name, variables[name])

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
            raise TypeError('cf_data_var argument should be of type CFDataVariable. Got %r.' % type(cf_data_var))

        # Determine the name of the label string (or length) dimension by
        # finding the dimension name that doesn't exist within the data dimensions. 
        str_dim_name = list(set(self.dimensions) - set(cf_data_var.dimensions))
        
        if len(str_dim_name) != 1:
            raise ValueError('Invalid string dimensions for CF-netCDF label variable %r' % self.cf_name)

        str_dim_name = str_dim_name[0]
        label_data = self[:]
        
        if isinstance(label_data, ma.MaskedArray):
            label_data = label_data.filled()

        # Determine whether we have a string-valued scalar label
        # i.e. a character variable that only has one dimension (the length of the string).
        if self.ndim == 1:
            data = np.array([''.join(label_data).strip()])
        else:
            # Determine the index of the string dimension.
            str_dim = self.dimensions.index(str_dim_name)
    
            # Calculate new label data shape (without string dimension) and create payload array.
            new_shape = tuple(dim_len for i, dim_len in enumerate(self.shape) if i != str_dim)
            data = np.empty(new_shape, dtype='|S%d' % self.shape[str_dim])
    
            for index in np.ndindex(new_shape):
                # Create the slice for the label data.
                if str_dim == 0:
                    label_index = (slice(None, None),) + index
                else:
                    label_index = index + (slice(None, None),)
                    
                data[index] = ''.join(label_data[label_index]).strip()

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
            raise TypeError('cf_data_var argument should be of type CFDataVariable. Got %r.' % type(cf_data_var))

        return tuple([dim_name for dim_name in self.dimensions if dim_name in cf_data_var.dimensions])


class CFMeasureVariable(CFVariable):
    """
    A CF-netCDF measure variable is a variable that contains cell areas or volumes.

    Identified by the CF-netCDF variable attribute 'cell_measures'.

    Ref: [CF] Section 7.2. Cell Measures.

    """
    cf_identity = 'cell_measures'

    def __init__(self, name, data, measure):
        CFVariable.__init__(self, name, data)
        self.cf_measure = measure
        '''Associated cell measure of the cell variable'''
        
    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)
        netcdf_variable_names = variables.keys()

        # Identify all CF measure variables.
        for nc_var_name, nc_var in target.iteritems():
            # Check for measure variable references.
            nc_var_att = getattr(nc_var, cls.cf_identity, None)
            
            if nc_var_att is not None:
                for match_item in _CF_PARSE.finditer(nc_var_att):
                    match_group = match_item.groupdict()
                    measure = match_group['lhs']
                    variable_name = match_group['rhs']
                    
                    if variable_name not in ignore:
                        if variable_name not in netcdf_variable_names:
                            if warn:
                                message = 'Missing CF-netCDF measure variable %r, referenced by netCDF variable %r'
                                warnings.warn(message % (variable_name, nc_var_name))
                        else:
                            result[variable_name] = CFMeasureVariable(variable_name, variables[variable_name], measure)

        return result
            
            
################################################################################
class CFGroup(object, UserDict.DictMixin):
    """
    Represents a collection of 'NetCDF Climate and Forecast (CF) Metadata
    Conventions' variables and netCDF global attributes.
    
    """
    def __init__(self):       
        self._cf_variables = {}
        '''Collection of CF-netCDF variables'''
        self.global_attributes = {}
        '''Collection of netCDF global attributes'''

    def _cf_getter(self, cls):
        # Generate dictionary with dictionary comprehension.
        return {cf_name:cf_var for cf_name, cf_var in self._cf_variables.iteritems() if isinstance(cf_var, cls)}

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
        return {cf_name:cf_var for cf_name, cf_var in self._cf_variables.iteritems() if cf_var.has_formula_terms()}
    
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

    def keys(self):
        """Return the names of all the CF-netCDF variables in the group."""
        return self._cf_variables.keys()

    def __setitem__(self, name, variable):
        if not isinstance(variable, CFVariable):
            raise TypeError('Attempted to add an invalid CF-netCDF variable to the %s' % self.__class__.__name__)

        if name != variable.cf_name:
            raise ValueError('Mismatch between key name %r and CF-netCDF variable name %r' % (str(name), variable.cf_name))

        self._cf_variables[name] = variable

    def __getitem__(self, name):
        if name not in self._cf_variables:
            raise KeyError('Cannot get unknown CF-netCDF variable name %r' % str(name))
        
        return self._cf_variables[name]
    
    def __delitem__(self, name):
        if name not in self._cf_variables:
            raise KeyError('Cannot delete unknown CF-netcdf variable name %r' % str(name))

        del self._cf_variables[name]

    def __repr__(self):
        result = []
        result.append('variables:%d' % len(self._cf_variables))
        result.append('global_attributes:%d' % len(self.global_attributes))
            
        return '<%s of %s>' % (self.__class__.__name__, ', '.join(result))


################################################################################
class CFReader(object):
    """
    This class allows the contents of a netCDF file to be interpreted according 
    to the 'NetCDF Climate and Forecast (CF) Metadata Conventions'.

    """
    def __init__(self, filename, warn=False, monotonic=False):
        self._filename = os.path.expanduser(filename)
        # All CF variable types EXCEPT for the "special cases" of
        # CFDataVariable, CFCoordinateVariable and _CFFormulaTermsVariable.
        self._variable_types = (CFAncillaryDataVariable, CFAuxiliaryCoordinateVariable, 
                                CFBoundaryVariable, CFClimatologyVariable, 
                                CFGridMappingVariable, CFLabelVariable, CFMeasureVariable)
        
        self.cf_group = CFGroup()
        '''Collection of CF-netCDF variables associated with this netCDF file'''

        self._dataset = netCDF4.Dataset(self._filename, mode='r')

        # Issue load optimisation warning.
        if warn and self._dataset.file_format in ['NETCDF3_CLASSIC', 'NETCDF3_64BIT']:
            warnings.warn('Optimise CF-netCDF loading by converting data from NetCDF3 ' \
                          'to NetCDF4 file format using the "nccopy" command.')
        
        self._check_monotonic = monotonic

        self._translate()
        self._build_cf_groups()
        self._reset()

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self._filename)

    def _translate(self):
        """Classify the netCDF variables into CF-netCDF variables."""
        
        netcdf_variable_names = self._dataset.variables.keys()

        # Identify all CF coordinate variables first. This must be done
        # first as, by CF convention, the definition of a CF auxiliary
        # coordinate variable may include a scalar CF coordinate variable,
        # whereas we want these two types of variables to be mutually exclusive.
        coords = CFCoordinateVariable.identify(self._dataset.variables,
                                               monotonic=self._check_monotonic)
        self.cf_group.update(coords)
        coordinate_names = self.cf_group.coordinates.keys()

        # Identify all CF variables EXCEPT for the "special cases".
        for variable_type in self._variable_types:
            # Prevent grid mapping variables being mis-identified as CF coordinate variables.
            ignore = None if issubclass(variable_type, CFGridMappingVariable) else coordinate_names
            self.cf_group.update(variable_type.identify(self._dataset.variables, ignore=ignore))

        # Identify global netCDF attributes.
        attr_dict = {attr_name: getattr(self._dataset, attr_name, '') for
                        attr_name in self._dataset.ncattrs()}
        self.cf_group.global_attributes.update(attr_dict)

        # Determine the CF data variables.
        data_variable_names = set(netcdf_variable_names) - set(self.cf_group.ancillary_variables) - \
                              set(self.cf_group.auxiliary_coordinates) - set(self.cf_group.bounds) - \
                              set(self.cf_group.climatology) - set(self.cf_group.coordinates) - \
                              set(self.cf_group.grid_mappings) - set(self.cf_group.labels) - \
                              set(self.cf_group.cell_measures)

        for name in data_variable_names:
            self.cf_group[name] = CFDataVariable(name, self._dataset.variables[name])

        # Identify and register all CF formula terms with the relevant CF variables.
        formula_terms = _CFFormulaTermsVariable.identify(self._dataset.variables)
        for cf_var in formula_terms.itervalues():
            if cf_var.cf_name in self.cf_group:
                self.cf_group[cf_var.cf_name].add_formula_term(cf_var.cf_root, cf_var.cf_term)

    def _build_cf_groups(self):
        """Build the first order relationships between CF-netCDF variables."""
        
        coordinate_names = self.cf_group.coordinates.keys()

        for cf_variable in self.cf_group.itervalues():
            cf_group = CFGroup()

            # Build CF variable relationships.
            for variable_type in self._variable_types:
                # Prevent grid mapping variables being mis-identified as
                # CF coordinate variables.
                ignore = None if issubclass(variable_type, CFGridMappingVariable) else coordinate_names
                match = variable_type.identify(self._dataset.variables, ignore=ignore,
                                               target=cf_variable.cf_name, warn=False)
                cf_group.update({name: self.cf_group[name] for name in match.iterkeys()})

            # Build CF data variable relationships.
            if isinstance(cf_variable, CFDataVariable):
                # Add global netCDF attributes.
                cf_group.global_attributes.update(self.cf_group.global_attributes)
                # Add appropriate "dimensioned" CF coordinate variables.
                cf_group.update({cf_name: self.cf_group[cf_name] for cf_name
                                    in cf_variable.dimensions if cf_name in
                                    self.cf_group.coordinates})
                # Add appropriate "dimensionless" CF coordinate variables.
                coordinates_attr = getattr(cf_variable, 'coordinates', '')
                cf_group.update({cf_name: self.cf_group[cf_name] for cf_name
                                    in coordinates_attr.split() if cf_name in
                                    self.cf_group.coordinates})

            # Add the CF group to the variable.
            cf_variable.cf_group = cf_group

    def _reset(self):
        """Reset the attribute touch history of each variable."""
        
        for nc_var_name in self._dataset.variables.iterkeys():
            self.cf_group[nc_var_name].cf_attrs_reset()

    def __del__(self):
        # Explicitly close dataset to prevent file remaining open.
        self._dataset.close()


