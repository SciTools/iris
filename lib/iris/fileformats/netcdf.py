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
Module to support the loading of a NetCDF file into an Iris cube.

See also: `netCDF4 python <http://code.google.com/p/netcdf4-python/>`_.

Also refer to document 'NetCDF Climate and Forecast (CF) Metadata Conventions',
Version 1.4, 27 February 2009.

"""

import collections
import itertools
import os
import os.path
import string
import warnings

import iris.proxy
iris.proxy.apply_proxy('netCDF4', globals())
import numpy as np
import numpy.ma as ma
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
#  - positive
#
_CF_ATTRS = ['add_offset', 'ancillary_variables', 'axis', 'bounds', 'calendar',
             'cell_measures', 'cell_methods', 'climatology', 'compress',
             'coordinates', '_FillValue', 'flag_masks', 'flag_meanings',
             'flag_values', 'formula_terms', 'grid_mapping', 'leap_month',
             'leap_year', 'long_name', 'missing_value', 'month_lengths',
             'scale_factor', 'standard_error_multiplier',
             'standard_name', 'units', 'valid_max', 'valid_min', 'valid_range']

_CF_CONVENTIONS_VERSION = 'CF-1.5'

_FactoryDefn = collections.namedtuple('_FactoryDefn', ('primary', 'std_name',
                                                       'formula_terms_format'))
_FACTORY_DEFNS = {
    iris.aux_factory.HybridHeightFactory: _FactoryDefn(
        primary='delta',
        std_name='atmosphere_hybrid_height_coordinate',
        formula_terms_format='a: {delta} b: {sigma} orog: {orography}'), }


class CFNameCoordMap(object):
    """Provide a simple CF name to CF coordinate mapping."""

    _Map = collections.namedtuple('_Map', ['name', 'coord'])

    def __init__(self):
        self._map = []

    def append(self, name, coord):
        """
        Append the given name and coordinate pair to the mapping.

        Args:

        * name:
            CF name of the associated coordinate.

        * coord:
            The coordinate of the associated CF name.

        Returns:
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
        """
        Return the CF name, given a coordinate

        Args:

        * coord:
            The coordinate of the associated CF name.

        Returns:
            Coordinate.

        """
        result = None
        for pair in self._map:
            if coord == pair.coord:
                result = pair.name
                break
        if result is None:
            msg = 'Coordinate is not mapped, {!r}'.format(coord)
            raise KeyError(msg)
        return result

    def coord(self, name):
        """
        Return the coordinate, given a CF name.

        Args:

        * name:
            CF name of the associated coordinate.

        Returns:
            CF name.

        """
        result = None
        for pair in self._map:
            if name == pair.name:
                result = pair.coord
                break
        if result is None:
            msg = 'Name is not mapped, {!r}'.format(name)
            raise KeyError(msg)
        return result


def _pyke_kb_engine():
    """Return the PyKE knowledge engine for CF->cube conversion."""

    pyke_dir = os.path.join(os.path.dirname(__file__), '_pyke_rules')
    compile_dir = os.path.join(pyke_dir, 'compiled_krb')
    engine = None

    if os.path.exists(compile_dir):
        tmpvar = [os.path.getmtime(os.path.join(compile_dir, fname)) for
                  fname in os.listdir(compile_dir) if not
                  fname.startswith('_')]
        if tmpvar:
            oldest_pyke_compile_file = min(tmpvar)
            rule_age = os.path.getmtime(
                os.path.join(pyke_dir, _PYKE_RULE_BASE + '.krb'))

            if oldest_pyke_compile_file >= rule_age:
                # Initialise the pyke inference engine.
                engine = knowledge_engine.engine(
                    (None, 'iris.fileformats._pyke_rules.compiled_krb'))

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
        return '%s(%r, %r)' % (self.__class__.__name__, self.path,
                               self.variable_name)

    def __getstate__(self):
        return {attr: getattr(self, attr) for attr in self.__slots__}

    def __setstate__(self, state):
        for key, value in state.iteritems():
            setattr(self, key, value)

    def load(self, data_shape, data_type, mdi, deferred_slice):
        """
        Load the corresponding proxy data item and perform any deferred
        slicing.

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
        engine.add_case_specific_fact(_PYKE_FACT_BASE, 'coordinate',
                                      (cf_name,))

    # Assert facts for CF auxiliary coordinates.
    for cf_name in cf_group.auxiliary_coordinates.iterkeys():
        engine.add_case_specific_fact(_PYKE_FACT_BASE, 'auxiliary_coordinate',
                                      (cf_name,))

    # Assert facts for CF grid_mappings.
    for cf_name in cf_group.grid_mappings.iterkeys():
        engine.add_case_specific_fact(_PYKE_FACT_BASE, 'grid_mapping',
                                      (cf_name,))

    # Assert facts for CF labels.
    for cf_name in cf_group.labels.iterkeys():
        engine.add_case_specific_fact(_PYKE_FACT_BASE, 'label',
                                      (cf_name,))

    # Assert facts for CF formula terms associated with the cf_group
    # of the CF data variable.
    formula_root = set()
    for cf_var in cf.cf_group.formula_terms.itervalues():
        for cf_root, cf_term in cf_var.cf_terms_by_root.iteritems():
            # Only assert this fact if the formula root variable is
            # defined in the CF group of the CF data variable.
            if cf_root in cf_group:
                formula_root.add(cf_root)
                engine.add_case_specific_fact(_PYKE_FACT_BASE, 'formula_term',
                                              (cf_var.cf_name, cf_root,
                                               cf_term))

    for cf_root in formula_root:
        engine.add_case_specific_fact(_PYKE_FACT_BASE, 'formula_root',
                                      (cf_root,))


def _pyke_stats(engine, cf_name):
    if DEBUG:
        print '-' * 80
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
    """Set attributes dictionary, converting unicode strings appropriately."""

    if isinstance(value, unicode):
        try:
            attributes[str(key)] = str(value)
        except UnicodeEncodeError:
            attributes[str(key)] = value
    else:
        attributes[str(key)] = value


def _load_cube(engine, cf, cf_var, filename):
    """Create the cube associated with the CF-netCDF data variable."""

    # Figure out what the eventual data type will be after any scale/offset
    # transforms.
    dummy_data = np.zeros(1, dtype=cf_var.dtype)
    if hasattr(cf_var, 'scale_factor'):
        dummy_data = cf_var.scale_factor * dummy_data
    if hasattr(cf_var, 'add_offset'):
        dummy_data = cf_var.add_offset + dummy_data

    # Create cube with data (not yet deferred), but no metadata
    data_proxies = np.array(NetCDFDataProxy(filename, cf_var.cf_name))
    data_manager = iris.fileformats.manager.DataManager(cf_var.shape,
                                                        dummy_data.dtype,
                                                        None)
    cube = iris.cube.Cube(data_proxies, data_manager=data_manager)

    # Reset the pyke inference engine.
    engine.reset()

    # Initialise pyke engine rule processing hooks.
    engine.cf_var = cf_var
    engine.cube = cube
    engine.provides = {}
    engine.requires = {}
    engine.rule_triggered = set()
    engine.filename = filename

    # Assert any case-specific facts.
    _assert_case_specific_facts(engine, cf, cf_var.cf_group)

    # Run pyke inference engine with forward chaining rules.
    engine.activate(_PYKE_RULE_BASE)

    # Populate coordinate attributes with the untouched attributes from the
    # associated CF-netCDF variable.
    coordinates = engine.provides.get('coordinates', [])
    attribute_predicate = lambda item: item[0] not in _CF_ATTRS

    for coord, cf_var_name in coordinates:
        tmpvar = itertools.ifilter(attribute_predicate,
                                   cf.cf_group[cf_var_name].cf_attrs_unused())
        for attr_name, attr_value in tmpvar:
            _set_attributes(coord.attributes, attr_name, attr_value)

    tmpvar = itertools.ifilter(attribute_predicate, cf_var.cf_attrs_unused())
    # Attach untouched attributes of the associated CF-netCDF data variable to
    # the cube.
    for attr_name, attr_value in tmpvar:
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
            raise ValueError('Unable to find coordinate for variable '
                             '{!r}'.format(name))
        # Convert term names to coordinates (via netCDF variable names).
        terms_to_var_names = engine.requires['formula_terms']
        delta = coord_from_var_name(terms_to_var_names['a'])
        sigma = coord_from_var_name(terms_to_var_names['b'])
        orography = coord_from_var_name(terms_to_var_names['orog'])
        factory = iris.aux_factory.HybridHeightFactory(delta, sigma, orography)
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
            # Only process CF data variables that do not participate in a
            # formula term.
            if not cf_var.has_formula_terms():
                cube = _load_cube(engine, cf, cf_var, filename)

                # Process any associated formula terms and attach
                # the corresponding AuxCoordFactory.
                _load_aux_factory(engine, cf, filename, cube)

                # Perform any user registered callback function.
                cube = iris.io.run_callback(callback, cube, engine.cf_var,
                                            filename)

                # Callback mechanism may return None, which must not be yielded
                if cube is None:
                    continue

                yield cube


class Saver(object):
    """A manager for saving netcdf files."""

    def __init__(self, filename, netcdf_format):
        """
        A manager for saving netcdf files.

        Args:

        * filename (string):
            Name of the netCDF file to save the cube.

        * netcdf_format (string):
            Underlying netCDF file format, one of 'NETCDF4', 'NETCDF4_CLASSIC',
            'NETCDF3_CLASSIC' or 'NETCDF3_64BIT'. Default is 'NETCDF4' format.

        Returns:
            None.

        For example::

            # Initialise Manager for saving
            with Saver(filename, netcdf_format) as sman:
                # Iterate through the cubelist.
                for cube in cubes:
                    sman.write(cube)

        """
        if netcdf_format not in ['NETCDF4', 'NETCDF4_CLASSIC',
                                 'NETCDF3_CLASSIC', 'NETCDF3_64BIT']:
            raise ValueError('Unknown netCDF file format, got %r' %
                             netcdf_format)

        # All persistent variables
        #: CF name mapping with iris coordinates
        self._name_coord_map = CFNameCoordMap()
        #: List of dimension coordinates added to the file
        self._dim_coords = []
        #: List of grid mappings added to the file
        self._coord_systems = []
        #: A dictionary, listing dimension names and corresponding length
        self._existing_dim = {}
        #: NetCDF dataset
        self._dataset = netCDF4.Dataset(filename, mode='w',
                                        format=netcdf_format)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        """Flush any buffered data to the CF-netCDF file before closing."""

        self._dataset.sync()
        self._dataset.close()

    def write(self, cube):
        """
        Wrapper for saving cubes to a NetCDF file.

        Args:

        * cube (:class:`iris.cube.Cube`):
            A :class:`iris.cube.Cube` to be saved to a netCDF file.

        Returns:
            None.

        """
        if len(cube.aux_factories) > 1:
            raise ValueError('Multiple auxiliary factories are not supported.')

        cf_profile_available = (
            'cf_profile' in iris.site_configuration and
            iris.site_configuration['cf_profile'] not in [None, False])

        if cf_profile_available:
            # Perform a CF profile of the cube. This may result in an exception
            # being raised if mandatory requirements are not satisfied.
            profile = iris.site_configuration['cf_profile'](cube)

        # Get suitable dimension names.
        dimension_names = self._get_dim_names(cube)

        # Create the CF-netCDF data dimensions.
        self._create_cf_dimensions(dimension_names)

        # Create the associated cube CF-netCDF data variable.
        cf_var_cube = self._create_cf_data_variable(cube, dimension_names)

        # Add coordinate variables and return factory definitions
        factory_defn = self._add_dim_coords(cube, dimension_names)

        # Add the auxiliary coordinate variable names and associate the data
        # variable to them
        cf_var_cube = self._add_aux_coords(cube, cf_var_cube, dimension_names,
                                           factory_defn)

        if cf_profile_available:
            # Perform a CF patch of the dataset.
            iris.site_configuration['cf_patch'](profile, self._dataset,
                                                cf_var_cube)

    def _create_cf_dimensions(self, dimension_names):
        """
        Create the CF-netCDF data dimensions.

        Create the CF-netCDF data dimensions, making the outermost dimension
        an unlimited dimension.

        Args:

        * dimension_names (list):
            Names associated with the dimensions of the cube.

        Returns:
            None.

        """
        if dimension_names:
            if dimension_names[0] not in self._dataset.dimensions:
                self._dataset.createDimension(dimension_names[0], None)
        for dim_name in dimension_names[1:]:
            if dim_name not in self._dataset.dimensions:
                self._dataset.createDimension(dim_name,
                                              self._existing_dim[dim_name])

    def _add_aux_coords(self, cube, cf_var_cube, dimension_names,
                        factory_defn):
        """
        Add aux. coordinate to the dataset and associate with the data variable

        Args:

        * cube (:class:`iris.cube.Cube`) or cubelist
          (:class:`iris.cube.CubeList`):
            A :class:`iris.cube.Cube`, :class:`iris.cube.CubeList` or list of
            cubes to be saved to a netCDF file.
        * cf_var_cube (:class:`netcdf.netcdf_variable`):
            cf variable cube representation.
        * dimension_names (list):
            Names associated with the dimensions of the cube.
        * factory_defn (:class:`_FactoryDefn`):
            An optional description of the AuxCoordFactory relevant to this
            cube.

        Returns:
            Updated cf_var_cube with coordinates added.

        """
        auxiliary_coordinate_names = []
        # Add CF-netCDF variables for the associated auxiliary coordinates.
        for coord in sorted(cube.aux_coords, key=lambda coord: coord.name()):
            # Create the associated coordinate CF-netCDF variable.
            if coord not in self._name_coord_map.coords:
                cf_name = self._create_cf_variable(cube, dimension_names,
                                                   coord, factory_defn)
                self._name_coord_map.append(cf_name, coord)
            else:
                cf_name = self._name_coord_map.name(coord)

            if cf_name is not None:
                auxiliary_coordinate_names.append(cf_name)

        # Add CF-netCDF auxiliary coordinate variable references to the
        # CF-netCDF data variable.
        if auxiliary_coordinate_names:
            cf_var_cube.coordinates = ' '.join(
                sorted(auxiliary_coordinate_names))
        return cf_var_cube

    def _add_dim_coords(self, cube, dimension_names):
        """
        Add coordinate variables to NetCDF dataset.

        Args:

        * cube (:class:`iris.cube.Cube`) or cubelist
          (:class:`iris.cube.CubeList`):
            A :class:`iris.cube.Cube`, :class:`iris.cube.CubeList` or list of
            cubes to be saved to a netCDF file.
        * dimension_names (list):
            Names associated with the dimensions of the cube.

        Returns:
            Factory definitions, a description of the AuxCoordFactory relevant
            to this cube.

        """
        factory_defn = None
        if cube.aux_factories:
            factory = cube.aux_factories[0]
            factory_defn = _FACTORY_DEFNS.get(type(factory), None)

        # Ensure we create the netCDF coordinate variables first.
        for coord in cube.dim_coords:
            # Create the associated coordinate CF-netCDF variable.
            if coord not in self._name_coord_map.coords:
                cf_name = self._create_cf_variable(cube, dimension_names,
                                                   coord, factory_defn)
                self._name_coord_map.append(cf_name, coord)
        return factory_defn

    def _get_dim_names(self, cube):
        """
        Determine suitable CF-netCDF data dimension names.

        Args:

        * cube (:class:`iris.cube.Cube`) or cubelist
          (:class:`iris.cube.CubeList`):
            A :class:`iris.cube.Cube`, :class:`iris.cube.CubeList` or list of
            cubes to be saved to a netCDF file.

        Returns:
            List of dimension names with length equal the number of dimensions
            in the cube.

        """
        dimension_names = []
        for dim in xrange(cube.ndim):
            coords = cube.coords(dimensions=dim, dim_coords=True)
            if coords:
                coord = coords[0]

                dim_name = self._get_coord_variable_name(cube, coord)
                # Add only dimensions that have not already been added.
                if coord not in self._dim_coords:
                    # Determine unique dimension name
                    while (dim_name in self._existing_dim or
                           dim_name in self._name_coord_map.names):
                        dim_name = self._increment_name(dim_name)

                    # Update names added, current cube dim names used and
                    # unique coordinates added.
                    self._existing_dim[dim_name] = coord.shape[0]
                    dimension_names.append(dim_name)
                    self._dim_coords.append(coord)
                else:
                    # Return the dim_name associated with the existing
                    # coordinate.
                    dim_name = self._name_coord_map.name(coord)
                    dimension_names.append(dim_name)

            else:
                # No CF-netCDF coordinates describe this data dimension.
                dim_name = 'dim%d' % dim
                if dim_name in self._existing_dim:
                    # Increment name if conflicted with one already existing.
                    if self._existing_dim[dim_name] != cube.shape[dim]:
                        while (dim_name in self._existing_dim and
                               self._existing_dim[dim_name] !=
                               cube.shape[dim] or
                               dim_name in self._name_coord_map.names):
                            dim_name = self._increment_name(dim_name)
                        # Update dictionary with new entry
                        self._existing_dim[dim_name] = cube.shape[dim]
                else:
                    # Update dictionary with new entry
                    self._existing_dim[dim_name] = cube.shape[dim]

                dimension_names.append(dim_name)
        return dimension_names

    def _cf_coord_identity(self, coord):
        """
        Determine a suitable units from a given coordinate.

        Args:

        * coord (:class:`iris.coords.Coord`):
            A coordinate of a cube.

        Returns:
            The (standard_name, long_name, unit) of the given
            :class:`iris.coords.Coord` instance.

        """

        units = str(coord.units)

        # TODO: Use #61 to get the units.
        if isinstance(coord.coord_system, iris.coord_systems.GeogCS):
            if "latitude" in coord.standard_name:
                units = 'degrees_north'
            elif "longitude" in coord.standard_name:
                units = 'degrees_east'

        elif isinstance(coord.coord_system, iris.coord_systems.RotatedGeogCS):
            units = 'degrees'

        elif isinstance(coord.coord_system,
                        iris.coord_systems.TransverseMercator):
            units = 'm'

        return coord.standard_name, coord.long_name, units

    def _create_cf_bounds(self, coord, cf_var, cf_name):
        """
        Create the associated CF-netCDF bounds variable.

        Args:

        * coord (:class:`iris.coords.Coord`):
            A coordinate of a cube.
        * cf_var:
            CF-netCDF variable
        * cf_name (string):
            name of the CF-NetCDF variable.

        Returns:
            None

        """
        if coord.has_bounds():
            n_bounds = coord.bounds.shape[-1]

            if n_bounds == 2:
                bounds_dimension_name = 'bnds'
            else:
                bounds_dimension_name = 'bnds_%s' % n_bounds

            if bounds_dimension_name not in self._dataset.dimensions:
                # Create the bounds dimension with the appropriate extent.
                self._dataset.createDimension(bounds_dimension_name, n_bounds)

            cf_var.bounds = cf_name + '_bnds'
            cf_var_bounds = self._dataset.createVariable(
                cf_var.bounds, coord.bounds.dtype,
                cf_var.dimensions + (bounds_dimension_name,))
            cf_var_bounds[:] = coord.bounds

    def _get_cube_variable_name(self, cube):
        """
        Returns a CF-netCDF variable name for the given cube.

        Args:

        * cube (class:`iris.cube.Cube`):
            An instance of a cube for which a CF-netCDF variable
            name is required.

        Returns:
            A CF-netCDF variable name as a string.

        """
        if cube.var_name is not None:
            cf_name = cube.var_name
        else:
            # Convert to lower case and replace whitespace by underscores.
            cf_name = '_'.join(cube.name().lower().split())

        return cf_name

    def _get_coord_variable_name(self, cube, coord):
        """
        Returns a CF-netCDF variable name for the given coordinate.

        Args:

        * cube (:class:`iris.cube.Cube`):
            The cube that contains the given coordinate.
        * coord (:class:`iris.coords.Coord`):
            An instance of a coordinate for which a CF-netCDF variable
            name is required.

        Returns:
            A CF-netCDF variable name as a string.

        """
        if coord.var_name is not None:
            cf_name = coord.var_name
        else:
            name = coord.standard_name or coord.long_name
            if not name or set(name).intersection(string.whitespace):
                # Auto-generate name based on associated dimensions.
                name = ''
                for dim in cube.coord_dims(coord):
                    name += 'dim{}'.format(dim)
                # Handle scalar coordinate (dims == ()).
                if not name:
                    name = 'unknown_scalar'
            # Convert to lower case and replace whitespace by underscores.
            cf_name = '_'.join(name.lower().split())

        return cf_name

    def _create_cf_variable(self, cube, dimension_names, coord, factory_defn):
        """
        Create the associated CF-netCDF variable in the netCDF dataset for the
        given coordinate. If required, also create the CF-netCDF bounds
        variable and associated dimension.

        Args:

        * dataset (:class:`netCDF4.Dataset`):
            The CF-netCDF data file being created.
        * cube (:class:`iris.cube.Cube`):
            The associated cube being saved to CF-netCDF file.
        * dimension_names (list):
            Names for each dimension of the cube.
        * coord (:class:`iris.coords.Coord`):
            The coordinate to be saved to CF-netCDF file.
        * factory_defn (:class:`_FactoryDefn`):
            An optional description of the AuxCoordFactory relevant to this
            cube.

        Returns:
            The string name of the associated CF-netCDF variable saved.

        """
        cf_name = self._get_coord_variable_name(cube, coord)
        while cf_name in self._dataset.variables:
            cf_name = self._increment_name(cf_name)

        # Derive the data dimension names for the coordinate.
        cf_dimensions = [dimension_names[dim] for dim in
                         cube.coord_dims(coord)]

        if np.issubdtype(coord.points.dtype, np.str):
            string_dimension_depth = coord.points.dtype.itemsize
            string_dimension_name = 'string%d' % string_dimension_depth

            # Determine whether to create the string length dimension.
            if string_dimension_name not in self._dataset.dimensions:
                self._dataset.createDimension(string_dimension_name,
                                              string_dimension_depth)

            # Add the string length dimension to dimension names.
            cf_dimensions.append(string_dimension_name)

            # Create the label coordinate variable.
            cf_var = self._dataset.createVariable(cf_name, '|S1',
                                                  cf_dimensions)

            # Add the payload to the label coordinate variable.
            if len(cf_dimensions) == 1:
                cf_var[:] = list('%- *s' % (string_dimension_depth,
                                            coord.points[0]))
            else:
                for index in np.ndindex(coord.points.shape):
                    index_slice = tuple(list(index) + [slice(None, None)])
                    cf_var[index_slice] = list('%- *s' %
                                               (string_dimension_depth,
                                                coord.points[index]))
        else:
            # Identify the collection of coordinates that represent CF-netCDF
            # coordinate variables.
            cf_coordinates = cube.dim_coords

            if coord in cf_coordinates:
                # By definition of a CF-netCDF coordinate variable this
                # coordinate must be 1-D and the name of the CF-netCDF variable
                # must be the same as its dimension name.
                cf_name = cf_dimensions[0]

            # Create the CF-netCDF variable.
            cf_var = self._dataset.createVariable(cf_name, coord.points.dtype,
                                                  cf_dimensions)

            # Add the axis attribute for spatio-temporal CF-netCDF coordinates.
            if coord in cf_coordinates:
                axis = iris.util.guess_coord_axis(coord)
                if axis is not None and axis.lower() in SPATIO_TEMPORAL_AXES:
                    cf_var.axis = axis.upper()

            # Add the data to the CF-netCDF variable.
            cf_var[:] = coord.points

            # Create the associated CF-netCDF bounds variable.
            self._create_cf_bounds(coord, cf_var, cf_name)

        # Deal with CF-netCDF units and standard name.
        standard_name, long_name, units = self._cf_coord_identity(coord)

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
            value = coord.attributes[name]

            if name == 'STASH':
                # Adopting provisional Metadata Conventions for representing MO
                # Scientific Data encoded in NetCDF Format.
                name = 'ukmo__um_stash_source'
                value = str(value)

            # Don't clobber existing attributes.
            if not hasattr(cf_var, name):
                setattr(cf_var, name, value)

        return cf_name

    def _create_cf_cell_methods(self, cube, dimension_names):
        """
        Create CF-netCDF string representation of a cube cell methods.

        Args:

        * cube (:class:`iris.cube.Cube`) or cubelist
          (:class:`iris.cube.CubeList`):
            A :class:`iris.cube.Cube`, :class:`iris.cube.CubeList` or list of
            cubes to be saved to a netCDF file.
        * dimension_names (list):
            Names associated with the dimensions of the cube.

        Returns:
            CF-netCDF string representation of a cube cell methods.

        """
        cell_methods = []

        # Identify the collection of coordinates that represent CF-netCDF
        # coordinate variables.
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

            interval = ' '.join(['interval: %s' % interval for interval in
                                 cm.intervals or []])
            comment = ' '.join(['comment: %s' % comment for comment in
                                cm.comments or []])
            extra = ' '.join([interval, comment]).strip()

            if extra:
                extra = ' (%s)' % extra

            cell_methods.append(names + cm.method + extra)

        return ' '.join(cell_methods)

    def _create_cf_grid_mapping(self, cube, cf_var_cube):
        """
        Create CF-netCDF grid mapping variable and associated CF-netCDF
        data variable grid mapping attribute.

        Args:

        * cube (:class:`iris.cube.Cube`) or cubelist
          (:class:`iris.cube.CubeList`):
            A :class:`iris.cube.Cube`, :class:`iris.cube.CubeList` or list of
            cubes to be saved to a netCDF file.
        * cf_var_cube (:class:`netcdf.netcdf_variable`):
            cf variable cube representation.

        Returns:
            None

        """
        cs = cube.coord_system('CoordSystem')
        if cs is not None:
            # Grid var not yet created?
            if cs not in self._coord_systems:
                while cs.grid_mapping_name in self._dataset.variables:
                    cs.grid_mapping_name = (
                        self._increment_name(cs.grid_mapping_name))

                cf_var_grid = self._dataset.createVariable(
                    cs.grid_mapping_name, np.int32)
                cf_var_grid.grid_mapping_name = cs.grid_mapping_name

                # latlon
                if isinstance(cs, iris.coord_systems.GeogCS):
                    cf_var_grid.longitude_of_prime_meridian = (
                        cs.longitude_of_prime_meridian)
                    cf_var_grid.semi_major_axis = cs.semi_major_axis
                    cf_var_grid.semi_minor_axis = cs.semi_minor_axis

                # rotated latlon
                elif isinstance(cs, iris.coord_systems.RotatedGeogCS):
                    if cs.ellipsoid:
                        cf_var_grid.longitude_of_prime_meridian = (
                            cs.ellipsoid.longitude_of_prime_meridian)
                        cf_var_grid.semi_major_axis = (
                            cs.ellipsoid.semi_major_axis)
                        cf_var_grid.semi_minor_axis = (
                            cs.ellipsoid.semi_minor_axis)
                    cf_var_grid.grid_north_pole_latitude = (
                        cs.grid_north_pole_latitude)
                    cf_var_grid.grid_north_pole_longitude = (
                        cs.grid_north_pole_longitude)
                    cf_var_grid.north_pole_grid_longitude = (
                        cs.north_pole_grid_longitude)

                # tmerc
                elif isinstance(cs, iris.coord_systems.TransverseMercator):
                    warnings.warn('TransverseMercator coordinate system not '
                                  'yet handled')

                # osgb (a specific tmerc)
                elif isinstance(cs, iris.coord_systems.OSGB):
                    warnings.warn('OSGB coordinate system not yet handled')

                # other
                else:
                    warnings.warn('Unable to represent the horizontal '
                                  'coordinate system. The coordinate system '
                                  'type %r is not yet implemented.' % type(cs))

                self._coord_systems.append(cs)

            # Refer to grid var
            cf_var_cube.grid_mapping = cs.grid_mapping_name

    def _create_cf_data_variable(self, cube, dimension_names):
        """
        Create CF-netCDF data variable for the cube and any associated grid
        mapping.

        Args:

        * dataset (:class:`netCDF4.Dataset`):
            The CF-netCDF data file being created.
        * cube (:class:`iris.cube.Cube`):
            The associated cube being saved to CF-netCDF file.
        * dimension_names (list):
            String names for each dimension of the cube.

        Returns:
            The newly created CF-netCDF data variable.

        """
        cf_name = self._get_cube_variable_name(cube)
        while cf_name in self._dataset.variables:
            cf_name = self._increment_name(cf_name)

        # Determine whether there is a cube MDI value.
        fill_value = None
        if isinstance(cube.data, ma.core.MaskedArray):
            fill_value = cube.data.fill_value

        # Create the cube CF-netCDF data variable with data payload.
        cf_var = self._dataset.createVariable(cf_name, cube.data.dtype,
                                              dimension_names,
                                              fill_value=fill_value)
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
                # Adopting provisional Metadata Conventions for representing MO
                # Scientific Data encoded in NetCDF Format.
                attr_name = 'ukmo__um_stash_source'
                value = str(value)

            if attr_name == "ukmo__process_flags":
                value = " ".join([x.replace(" ", "_") for x in value])

            if attr_name.lower() != 'conventions':
                setattr(cf_var, attr_name, value)

        self._dataset.Conventions = _CF_CONVENTIONS_VERSION

        # Create the CF-netCDF data variable cell method attribute.
        cell_methods = self._create_cf_cell_methods(cube, dimension_names)

        if cell_methods:
            cf_var.cell_methods = cell_methods

        # Create the CF-netCDF grid mapping.
        self._create_cf_grid_mapping(cube, cf_var)

        return cf_var

    def _increment_name(self, varname):
        """
        Increment string name or begin increment.

        Avoidance of conflicts between variable names, where the name is
        incremented to distinguish it from others.

        Args:

        * varname (string):
            Variable name to increment.

        Returns:
            Incremented varname.

        """
        num = 0
        try:
            name, endnum = varname.rsplit('_', 1)
            if endnum.isdigit():
                num = int(endnum) + 1
                varname = name
        except ValueError:
            pass

        return '{}_{}'.format(varname, num)


def save(cube, filename, netcdf_format='NETCDF4'):
    """
    Save cube(s) to a netCDF file, given the cube and the filename.

    Args:

    * cube (:class:`iris.cube.Cube` or :class:`iris.cube.CubeList`):
        A :class:`iris.cube.Cube`, :class:`iris.cube.CubeList` or other
        iterable of cubes to be saved to a netCDF file.

    * filename (string):
        Name of the netCDF file to save the cube(s).

    Kwargs:

    * netcdf_format (string):
        Underlying netCDF file format, one of 'NETCDF4', 'NETCDF4_CLASSIC',
        'NETCDF3_CLASSIC' or 'NETCDF3_64BIT'. Default is 'NETCDF4' format.

    Returns:
        None.

    .. seealso::

        NetCDF Context manager (:class:`~Saver`).

    """
    if isinstance(cube, iris.cube.Cube):
        cubes = iris.cube.CubeList()
        cubes.append(cube)
    else:
        cubes = cube

    # Initialise Manager for saving
    with Saver(filename, netcdf_format) as sman:
        # Iterate through the cubelist.
        for cube in cubes:
            sman.write(cube)
