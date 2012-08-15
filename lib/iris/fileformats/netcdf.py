# (C) British Crown Copyright 2010 - 2012, Met Office
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
Module to support the loading of a NetCDF file into an Iris cube.

See also: `netCDF4 python <http://code.google.com/p/netcdf4-python/>`_.

Also refer to document 'NetCDF Climate and Forecast (CF) Metadata Conventions', Version 1.4, 27 February 2009.

"""

import collections
import itertools
import os
import os.path
import warnings

import iris.proxy
iris.proxy.apply_proxy('netCDF4', globals())
import numpy as np
from pyke import knowledge_engine

import iris.analysis
import iris.coord_systems
import iris.coords
import iris.cube
import iris.exceptions
import iris.fileformats.cf
import iris.fileformats.manager
import iris.fileformats._pyke_rules
import iris.io
import iris.unit
import iris.util


# Show Pyke inference engine statistics.
DEBUG = False

# Pyke CF related file names.
_PYKE_RULE_BASE = 'fc_rules_cf'
_PYKE_FACT_BASE = 'facts_cf'

# Standard CML spatio-temporal axis names.
SPATIO_TEMPORAL_AXES = ['t', 'z', 'y', 'x']

# Pass through CF attributes:
#  - comment
#  - Conventions
#  - history
#  - institution
#  - reference
#  - source
#  - title
#
_CF_ATTRS = ['add_offset', 'ancillary_variables', 'axis', 'bounds', 'calendar', 
            'cell_measures', 'cell_methods', 'climatology', 'compress',
            'coordinates', '_FillValue', 'flag_masks', 'flag_meanings', 'flag_values',
            'formula_terms', 'grid_mapping', 'leap_month', 'leap_year', 'long_name',
            'missing_value', 'month_lengths', 'positive', 'scale_factor',
            'standard_error_multiplier', 'standard_name', 'units', 'valid_max',
            'valid_min', 'valid_range']

_CF_CONVENTIONS_VERSION = 'CF-1.5'


_FactoryDefn = collections.namedtuple('_FactoryDefn', ('primary', 'std_name',
                                                     'formula_terms_format'))
_FACTORY_DEFNS = {
    iris.aux_factory.HybridHeightFactory: _FactoryDefn(
        primary='delta',
        std_name='atmosphere_hybrid_height_coordinate',
        formula_terms_format='a: {delta} b: {sigma} orog: {orography}'
    ),
}


def _pyke_kb_engine():
    """Return the PyKE knowledge engine for CF->cube conversion."""

    pyke_dir = os.path.join(os.path.dirname(__file__), '_pyke_rules')
    compile_dir = os.path.join(pyke_dir, 'compiled_krb')
    engine = None

    if os.path.exists(compile_dir):
        oldest_pyke_compile_file = min([os.path.getmtime(os.path.join(compile_dir, fname)) for fname in os.listdir(compile_dir) if not fname.startswith('_')])
        rule_age = os.path.getmtime(os.path.join(pyke_dir, _PYKE_RULE_BASE + '.krb'))

        if oldest_pyke_compile_file > rule_age:
            # Initialise the pyke inference engine.
            engine = knowledge_engine.engine((None, 'iris.fileformats._pyke_rules.compiled_krb'))

    if engine is None:
        engine = knowledge_engine.engine(iris.fileformats._pyke_rules)

    return engine


class NetCDFDataProxy(object):
    """A reference to the data payload of a single NetCDF file variable."""

    __slots__ = ('path', 'variable_name')

    def __init__(self, path, variable_name):
        self.path = path
        self.variable_name = variable_name

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.path, self.variable_name)

    def load(self, data_shape, data_type, mdi, deferred_slice):
        """
        Load the corresponding proxy data item and perform any deferred slicing.
        
        Args:
        
        * data_shape (tuple of int):
            The data shape of the proxy data item.
        * data_type (:class:`numpy.dtype`):
            The data type of the proxy data item.
        * mdi (float):
            The missing data indicator value.
        * deferred_slice (tuple):
            The deferred slice to be applied to the proxy data item.
        
        Returns:
            :class:`numpy.ndarray`
        
        """
        dataset = netCDF4.Dataset(self.path)
        variable = dataset.variables[self.variable_name]
        # Get the NetCDF variable data and slice.
        payload = variable[deferred_slice]
        dataset.close()

        return payload


def _assert_case_specific_facts(engine, cf, cf_group):
    # Initialise pyke engine "provides" hooks.
    engine.provides['coordinates'] = []

    # Assert facts for CF coordinates.
    for cf_name in cf_group.coordinates.iterkeys():
        engine.add_case_specific_fact(_PYKE_FACT_BASE, 'coordinate', (cf_name,))

    # Assert facts for CF auxiliary coordinates.
    for cf_name in cf_group.auxiliary_coordinates.iterkeys():
        engine.add_case_specific_fact(_PYKE_FACT_BASE, 'auxiliary_coordinate', (cf_name,))

    # Assert facts for CF grid_mappings.
    for cf_name in cf_group.grid_mappings.iterkeys():
        engine.add_case_specific_fact(_PYKE_FACT_BASE, 'grid_mapping', (cf_name,))

    # Assert facts for CF labels.
    for cf_name in cf_group.labels.iterkeys():
        engine.add_case_specific_fact(_PYKE_FACT_BASE, 'label', (cf_name,))
        
    # Assert facts for CF formula terms associated with the cf_group
    # of the CF data variable.
    formula_root = set()
    for cf_var in cf.cf_group.formula_terms.itervalues():
        for cf_root, cf_term in cf_var.cf_terms_by_root.iteritems():
            # Only assert this fact if the formula root variable is
            # defined in the CF group of the CF data variable.
            if cf_root in cf_group:
                formula_root.add(cf_root)
                engine.add_case_specific_fact(_PYKE_FACT_BASE, 'formula_term', (cf_var.cf_name, cf_root, cf_term))
        
    for cf_root in formula_root:
        engine.add_case_specific_fact(_PYKE_FACT_BASE, 'formula_root', (cf_root,)) 


def _pyke_stats(engine, cf_name):
    if DEBUG:
        print '-'*80
        print 'CF Data Variable: %r' % cf_name
        
        engine.print_stats()

        print 'Rules Triggered:'

        for rule in sorted(list(engine.rule_triggered)):
            print '\t%s' % rule

        print 'Case Specific Facts:'
        kb_facts = engine.get_kb(_PYKE_FACT_BASE)

        for key in kb_facts.entity_lists.iterkeys():
            for arg in kb_facts.entity_lists[key].case_specific_facts:
                print '\t%s%s' % (key, arg)


def _set_attributes(attributes, key, value):
    """Set the attributes dictionary, converting unicode strings appropriately."""

    if isinstance(value, unicode):
        try:
            attributes[str(key)] = str(value)
        except UnicodeEncodeError:
            attributes[str(key)] = value
    else:
        attributes[str(key)] = value


def _load_cube(engine, cf, cf_var, filename):
    """Create the cube associated with the CF-netCDF data variable."""
     
    # Figure out what the eventual data type will be after any scale/offset transforms.
    dummy_data = np.zeros(1, dtype=cf_var.dtype)
    if hasattr(cf_var, 'scale_factor'):
        dummy_data = cf_var.scale_factor * dummy_data
    if hasattr(cf_var, 'add_offset'):
        dummy_data = cf_var.add_offset + dummy_data

    # Create cube with data (not yet deferred), but no metadata
    data_proxies = np.array(NetCDFDataProxy(filename, cf_var.cf_name))
    data_manager = iris.fileformats.manager.DataManager(cf_var.shape, dummy_data.dtype, None)
    cube = iris.cube.Cube(data_proxies, data_manager=data_manager)
    
    # Reset the pyke inference engine.
    engine.reset()

    # Initialise pyke engine rule processing hooks.
    engine.cf_var = cf_var
    engine.cube = cube
    engine.provides = {}
    engine.requires = {}
    engine.rule_triggered = set()

    # Assert any case-specific facts.
    _assert_case_specific_facts(engine, cf, cf_var.cf_group)

    # Run pyke inference engine with forward chaining rules.
    engine.activate(_PYKE_RULE_BASE)

    # Populate coordinate attributes with the untouched attributes from the associated CF-netCDF variable.
    coordinates = engine.provides.get('coordinates', [])
    attribute_predicate = lambda item: item[0] not in _CF_ATTRS
 
    for coord, cf_var_name in coordinates:
        for attr_name, attr_value in itertools.ifilter(attribute_predicate, cf.cf_group[cf_var_name].cf_attrs_unused()):
            _set_attributes(coord.attributes, attr_name, attr_value)
                
    # Attach untouched attributes of the associated CF-netCDF data variable to the cube.
    for attr_name, attr_value in itertools.ifilter(attribute_predicate, cf_var.cf_attrs_unused()):
        _set_attributes(cube.attributes, attr_name, attr_value)

    # Show pyke session statistics.
    _pyke_stats(engine, cf_var.cf_name)
    
    return cube


def _load_aux_factory(engine, cf, filename, cube):
    """
    Convert any CF-netCDF dimensionless coordinate to an AuxCoordFactory.
    
    """
    formula_type = engine.requires.get('formula_type')
    
    if formula_type == 'atmosphere_hybrid_height_coordinate':
        def coord_from_var_name(name):
            mapping = engine.provides['coordinates']
            for coord, cf_var_name in engine.provides['coordinates']:
                if cf_var_name == name:
                    return coord
            raise ValueError('Unable to find coordinate for variable {!r}'.format(name))
        # Convert term names to coordinates (via netCDF variable names).
        terms_to_var_names = engine.requires['formula_terms']
        delta = coord_from_var_name(terms_to_var_names['a'])
        sigma = coord_from_var_name(terms_to_var_names['b'])
        orography = coord_from_var_name(terms_to_var_names['orog'])
        factory = iris.aux_factory.HybridHeightFactory(delta, sigma, orography)
        cube.add_aux_factory(factory)


def load_cubes(filenames, callback=None):
    """
    Loads cubes from a list of NetCDF filenames.
    
    Args:
    
    * filenames (string/list):
        One or more NetCDF filenames to load.
    
    Kwargs:

    * callback (callable function):
        Function which can be passed on to :func:`iris.io.run_callback`.

    Returns:
        Generator of loaded NetCDF :class:`iris.cubes.Cube`.
    
    """
    # Initialise the pyke inference engine.
    engine = _pyke_kb_engine()

    if isinstance(filenames, basestring):
        filenames = [filenames]

    for filename in filenames:
        # Ingest the netCDF file.
        cf = iris.fileformats.cf.CFReader(filename)

        # Process each CF data variable.
        for cf_var in cf.cf_group.data_variables.itervalues():
            # Only process CF data variables that do not participate in a formula term.
            if not cf_var.has_formula_terms():
                cube = _load_cube(engine, cf, cf_var, filename)
                
                # Process any associated formula terms and attach
                # the corresponding AuxCoordFactory.
                _load_aux_factory(engine, cf, filename, cube)

                # Perform any user registered callback function.
                cube = iris.io.run_callback(callback, cube, engine.cf_var, filename)

                # Callback mechanism may return None, which must not be yielded. 
                if cube is None:
                    continue

                yield cube


def _cf_coord_identity(coord):
    """Return (standard_name, long_name, unit) of the given :class:`iris.coords.Coord` instance."""
    # Default behaviour
    standard_name = coord.standard_name
    long_name = coord.long_name
    units = str(coord.units)
    
    # Special cases
    # i) Rotated pole
    if isinstance(coord.coord_system, iris.coord_systems.LatLonCS):
        if coord.name() in ['latitude', 'grid_latitude']:
            if coord.coord_system.has_rotated_pole():
                standard_name = 'grid_latitude'
                units = 'degrees'
            else:
                units = 'degrees_north'

        if coord.name() in ['longitude', 'grid_longitude']:
            if coord.coord_system.has_rotated_pole():
                standard_name = 'grid_longitude'
                units = 'degrees'
            else:
                units = 'degrees_east'

    return standard_name, long_name, units


def _create_bounds(dataset, coord, cf_var, cf_name):
    if coord.has_bounds():
        n_bounds = coord.bounds.shape[-1]

        if n_bounds == 2:
            bounds_dimension_name = 'bnds'
        else:
            bounds_dimension_name = 'bnds_%s' % n_bounds

        if bounds_dimension_name not in dataset.dimensions:
            # Create the bounds dimension with the appropriate extent.
            dataset.createDimension(bounds_dimension_name, n_bounds)

        cf_var.bounds = cf_name + '_bnds'
        cf_var_bounds = dataset.createVariable(cf_var.bounds, coord.bounds.dtype, cf_var.dimensions + (bounds_dimension_name,))
        cf_var_bounds[:] = coord.bounds


def _create_cf_variable(dataset, cube, dimension_names, coord, factory_defn):
    """
    Create the associated CF-netCDF variable in the netCDF dataset for the 
    given coordinate. If required, also create the CF-netCDF bounds variable
    and associated dimension. 
    
    Args:

    * dataset (:class:`netCDF4.Dataset`):
        The CF-netCDF data file being created.
    * cube (:class:`iris.cube.Cube`):
        The associated cube being saved to CF-netCDF file.
    * dimension_names:
        List of string names for each dimension of the cube.
    * coord (:class:`iris.coords.Coord`):
        The coordinate to be saved to CF-netCDF file.
    * factory_defn (:class:`_FactoryDefn`):
        An optional description of the AuxCoordFactory relevant to this
        cube.

    Returns:
        The string name of the associated CF-netCDF variable saved.
    
    """
    cf_name = coord.name()

    # Derive the data dimension names for the coordinate.
    cf_dimensions = [dimension_names[dim] for dim in cube.coord_dims(coord)]

    if np.issubdtype(coord.points.dtype, np.str):
        string_dimension_depth = coord.points.dtype.itemsize
        string_dimension_name = 'string%d' % string_dimension_depth

        # Determine whether to create the string length dimension.
        if string_dimension_name not in dataset.dimensions:
            dataset.createDimension(string_dimension_name, string_dimension_depth)

        # Add the string length dimension to dimension names.
        cf_dimensions.append(string_dimension_name)

        # Create the label coordinate variable.
        cf_var = dataset.createVariable(cf_name, '|S1', cf_dimensions)

        # Add the payload to the label coordinate variable.
        if len(cf_dimensions) == 1:
            cf_var[:] = list('%- *s' % (string_dimension_depth, coord.points[0]))
        else:
            for index in np.ndindex(coord.points.shape):
                index_slice = tuple(list(index) + [slice(None, None)])
                cf_var[index_slice] = list('%- *s' % (string_dimension_depth, coord.points[index]))
    else:
        # Identify the collection of coordinates that represent CF-netCDF coordinate variables.
        cf_coordinates = cube.dim_coords

        if coord in cf_coordinates:
            # By definition of a CF-netCDF coordinate variable this coordinate must be 1-D
            # and the name of the CF-netCDF variable must be the same as its dimension name.
            cf_name = cf_dimensions[0]

        # Create the CF-netCDF variable.
        cf_var = dataset.createVariable(cf_name, coord.points.dtype, cf_dimensions)

        # Add the axis attribute for spatio-temporal CF-netCDF coordinates.
        if coord in cf_coordinates:
            axis = iris.util.guess_coord_axis(coord)
            if axis is not None and axis.lower() in SPATIO_TEMPORAL_AXES:
                cf_var.axis = axis.upper()

        # Add the data to the CF-netCDF variable.
        cf_var[:] = coord.points

        # Create the associated CF-netCDF bounds variable.
        _create_bounds(dataset, coord, cf_var, cf_name)

    # Deal with CF-netCDF units and standard name.
    standard_name, long_name, units = _cf_coord_identity(coord)

    # If this coordinate should describe a dimensionless vertical
    # coordinate, then override `standard_name`, `long_name`, and `axis`,
    # and also set the `formula_terms` attribute.
    if factory_defn:
        dependencies = cube.aux_factories[0].dependencies
        if coord is dependencies[factory_defn.primary]:
            standard_name = factory_defn.std_name
            cf_var.axis = 'Z'

            fmt = factory_defn.formula_terms_format
            names = {key: coord.name() for key, coord in
                            dependencies.iteritems()}
            formula_terms = fmt.format(**names)
            cf_var.formula_terms = formula_terms

    if units != 'unknown':
        cf_var.units = units

    if standard_name is not None:
        cf_var.standard_name = standard_name

    if long_name is not None:
        cf_var.long_name = long_name

    # Add the CF-netCDF calendar attribute.
    if coord.units.calendar:
        cf_var.calendar = coord.units.calendar

    # Add any other custom coordinate attributes.
    for name in sorted(coord.attributes):
        # Don't clobber existing attributes.
        if not hasattr(cf_var, name):
            setattr(cf_var, name, coord.attributes[name])

    return cf_name


def _create_cf_cell_methods(cube, dimension_names):
    """Create CF-netCDF string representation of a cube cell methods."""
    cell_methods = []

    # Identify the collection of coordinates that represent CF-netCDF coordinate variables.
    cf_coordinates = cube.dim_coords
    
    for cm in cube.cell_methods:
        names = ''

        for name in cm.coord_names:
            coord = cube.coords(name)

            if coord:
                coord = coord[0]
                if coord in cf_coordinates:
                    name = dimension_names[cube.coord_dims(coord)[0]]

            names += '%s: ' % name
        
        interval = ' '.join(['interval: %s' % interval for interval in cm.intervals or []])
        comment = ' '.join(['comment: %s' % comment for comment in cm.comments or []])
        extra = ' '.join([interval, comment]).strip()
        
        if extra:
            extra = ' (%s)' % extra
            
        cell_methods.append(names + cm.method + extra)
            
    return ' '.join(cell_methods)


def _create_cf_grid_mapping(dataset, cube, cf_var):
    """
    Create CF-netCDF grid mapping variable and associated CF-netCDF
    data variable grid mapping attribute. 
    
    """
    cs = cube.coord_system('HorizontalCS')
    
    if cs is not None:
        if isinstance(cs, iris.coord_systems.LatLonCS):
            cf_grid_name = 'rotated_latitude_longitude' if cs.has_rotated_pole() else 'latitude_longitude'
            
            if cf_grid_name not in dataset.variables:
                cf_var.grid_mapping = cf_grid_name
                cf_var_grid = dataset.createVariable(cf_grid_name, np.int32)
                cf_var_grid.grid_mapping_name = cf_grid_name
                cf_var_grid.longitude_of_prime_meridian = 0.0
    
                if cs.datum:
                    cf_var_grid.semi_major_axis = cs.datum.semi_major_axis
                    cf_var_grid.semi_minor_axis = cs.datum.semi_minor_axis
                
                if cs.has_rotated_pole():
                    if cs.n_pole:
                        cf_var_grid.grid_north_pole_latitude = cs.n_pole.latitude
                        cf_var_grid.grid_north_pole_longitude = cs.n_pole.longitude
                        
                    cf_var_grid.north_pole_grid_longitude = cs.reference_longitude
            else:
                # Reference previously created grid mapping
                cf_var.grid_mapping = cf_grid_name
        else:
            warnings.warn('Unable to represent the horizontal coordinate system. The coordinate system type %r is not yet implemented.' % type(cs))


def _create_cf_data_variable(dataset, cube, dimension_names):
    """
    Create CF-netCDF data variable for the cube and any associated grid mapping.
    
    Args:
    
    * dataset (:class:`netCDF4.Dataset`):
        The CF-netCDF data file being created.
    * cube (:class:`iris.cube.Cube`):
        The associated cube being saved to CF-netCDF file.
    * dimension_names:
        List of string names for each dimension of the cube.
        
    Returns:
        The newly created CF-netCDF data variable. 
    
    """
    cf_name = cube.name()
    
    # Determine whether there is a cube MDI value.
    fill_value = None
    if isinstance(cube.data, np.ma.core.MaskedArray):
        fill_value = cube.data.fill_value
        
    # Create the cube CF-netCDF data variable with data payload.
    cf_var = dataset.createVariable(cf_name, cube.data.dtype, dimension_names, fill_value=fill_value)
    cf_var[:] = cube.data
    
    if cube.standard_name:
    	cf_var.standard_name = cube.standard_name

    if cube.long_name:
        cf_var.long_name = cube.long_name

    if cube.units != 'unknown':
        cf_var.units = str(cube.units)

    # Add any other cube attributes as CF-netCDF data variable attributes.
    for attr_name in sorted(cube.attributes):
        value = cube.attributes[attr_name]
        
        if attr_name == 'STASH':
            # Adopting provisional Metadata Conventions for representing MO Scientific Data encoded in NetCDF Format.
            attr_name = 'ukmo__um_stash_source'
            value = str(value)

        if attr_name == "ukmo__process_flags":
            value = " ".join([x.replace(" ", "_") for x in value])

        if attr_name.lower() != 'conventions':
           setattr(cf_var, attr_name, value)

    # Declare the CF conventions versions.
    dataset.Conventions = _CF_CONVENTIONS_VERSION

    # Create the CF-netCDF data variable cell method attribute.
    cell_methods = _create_cf_cell_methods(cube, dimension_names)
    
    if cell_methods:
        cf_var.cell_methods = cell_methods
    
    # Create the CF-netCDF grid mapping.
    _create_cf_grid_mapping(dataset, cube, cf_var)

    return cf_var


def save(cube, filename, netcdf_format='NETCDF4'):
    """
    Save a cube to a netCDF file, given the cube and the filename.
    
    Args:
    
    * cube (:class:`iris.cube.Cube`):
        The :class:`iris.cube.Cube` to be saved to a netCDF file.

    * filename (string):
        Name of the netCDF file to save the cube.

    * netcdf_format (string):
        Underlying netCDF file format, one of 'NETCDF4', 'NETCDF4_CLASSIC', 
        'NETCDF3_CLASSIC' or 'NETCDF3_64BIT'. Default is 'NETCDF4' format.

    Returns:
        None.
    
    """
    if not isinstance(cube, iris.cube.Cube):
        raise TypeError('Expecting a single cube instance, got %r.' % type(cube))

    if netcdf_format not in ['NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_CLASSIC', 'NETCDF3_64BIT']:
        raise ValueError('Unknown netCDF file format, got %r' % netcdf_format)

    if len(cube.aux_factories) > 1:
        raise ValueError('Multiple auxiliary factories are not supported.')

    dataset = netCDF4.Dataset(filename, mode='w', format=netcdf_format)
    
    # Create the CF-netCDF data dimension names.
    dimension_names = []
    for dim in xrange(cube.ndim):
        coords = cube.coords(dimensions=dim, dim_coords=True)
        if coords is not None:
            if len(coords) != 1:
                raise iris.exceptions.IrisError('Cube appears to have multiple dimension coordinates on dimension %d' % dim)
            dimension_names.append(coords[0].name())
        else:
             # There are no CF-netCDF coordinates describing this data dimension.
            dimension_names.append('dim%d' % dim)

    # Create the CF-netCDF data dimensions.
    for dim_name, dim_len in zip(dimension_names, cube.shape):
        dataset.createDimension(dim_name, dim_len)

    # Identify the collection of coordinates that represent CF-netCDF coordinate variables.
    cf_coordinates = cube.dim_coords

    # Create the associated cube CF-netCDF data variable.
    cf_var_cube = _create_cf_data_variable(dataset, cube, dimension_names)

    factory_defn = None
    if cube.aux_factories:
        factory = cube.aux_factories[0]
        factory_defn = _FACTORY_DEFNS.get(type(factory), None)

    # Ensure we create the netCDF coordinate variables first.
    for coord in cf_coordinates:
        # Create the associated coordinate CF-netCDF variable.
        _create_cf_variable(dataset, cube, dimension_names, coord, factory_defn)
    
    # List of CF-netCDF auxiliary coordinate variable names.
    auxiliary_coordinate_names = []
    for coord in sorted(cube.aux_coords, key=lambda coord: coord.name()):
        # Create the associated coordinate CF-netCDF variable.
        cf_name = _create_cf_variable(dataset, cube, dimension_names, coord, factory_defn)

        if cf_name is not None:
            auxiliary_coordinate_names.append(cf_name)

    # Add CF-netCDF auxiliary coordinate variable references to the CF-netCDF data variable.
    if auxiliary_coordinate_names:
        cf_var_cube.coordinates = ' '.join(sorted(auxiliary_coordinate_names))

    # Flush any buffered data to the CF-netCDF file before closing.
    dataset.sync()
    dataset.close()
