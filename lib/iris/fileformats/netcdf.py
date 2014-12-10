# (C) British Crown Copyright 2010 - 2014, Met Office
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

from __future__ import (absolute_import, division, print_function)

import collections
import itertools
import os
import os.path
import string
import warnings

import biggus
import iris.proxy
iris.proxy.apply_proxy('netCDF4', globals())
import numpy as np
import numpy.ma as ma
from pyke import knowledge_engine

import iris.analysis
from iris.aux_factory import HybridHeightFactory, HybridPressureFactory, \
    OceanSigmaZFactory
import iris.coord_systems
import iris.coords
import iris.cube
import iris.exceptions
import iris.fileformats.cf
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

# CF attributes that should not be global.
_CF_DATA_ATTRS = ['flag_masks', 'flag_meanings', 'flag_values',
                  'instance_dimension', 'sample_dimension',
                  'standard_error_multiplier']

# CF attributes that should only be global.
_CF_GLOBAL_ATTRS = ['conventions', 'featureType', 'history', 'title']

# UKMO specific attributes that should not be global.
_UKMO_DATA_ATTRS = ['STASH', 'um_stash_source', 'ukmo__process_flags']

CF_CONVENTIONS_VERSION = 'CF-1.5'

_FactoryDefn = collections.namedtuple('_FactoryDefn', ('primary', 'std_name',
                                                       'formula_terms_format'))
_FACTORY_DEFNS = {
    HybridHeightFactory: _FactoryDefn(
        primary='delta',
        std_name='atmosphere_hybrid_height_coordinate',
        formula_terms_format='a: {delta} b: {sigma} orog: {orography}'),
    HybridPressureFactory: _FactoryDefn(
        primary='delta',
        std_name='atmosphere_hybrid_sigma_pressure_coordinate',
        formula_terms_format='ap: {delta} b: {sigma} '
        'ps: {surface_air_pressure}'),
    OceanSigmaZFactory: _FactoryDefn(
        primary='zlev',
        std_name='ocean_sigma_z_coordinate',
        formula_terms_format='sigma: {sigma} eta: {eta} depth: {depth} '
        'depth_c: {depth_c} nsigma: {nsigma} zlev: {zlev}')
}


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

    __slots__ = ('shape', 'dtype', 'path', 'variable_name', 'fill_value')

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
            data = variable[keys]
        finally:
            dataset.close()
        return data

    def __repr__(self):
        fmt = '<{self.__class__.__name__} shape={self.shape}' \
              ' dtype={self.dtype!r} path={self.path!r}' \
              ' variable_name={self.variable_name!r}>'
        return fmt.format(self=self)

    def __getstate__(self):
        return {attr: getattr(self, attr) for attr in self.__slots__}

    def __setstate__(self, state):
        for key, value in state.iteritems():
            setattr(self, key, value)


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
        print('-' * 80)
        print('CF Data Variable: %r' % cf_name)

        engine.print_stats()

        print('Rules Triggered:')

        for rule in sorted(list(engine.rule_triggered)):
            print('\t%s' % rule)

        print('Case Specific Facts:')
        kb_facts = engine.get_kb(_PYKE_FACT_BASE)

        for key in kb_facts.entity_lists.iterkeys():
            for arg in kb_facts.entity_lists[key].case_specific_facts:
                print('\t%s%s' % (key, arg))


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

    # Create cube with deferred data, but no metadata
    fill_value = getattr(cf_var.cf_data, '_FillValue',
                         netCDF4.default_fillvals[cf_var.dtype.str[1:]])
    proxy = NetCDFDataProxy(cf_var.shape, dummy_data.dtype,
                            filename, cf_var.cf_name, fill_value)
    data = biggus.OrthoArrayAdapter(proxy)
    cube = iris.cube.Cube(data)

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


def _load_aux_factory(engine, cube):
    """
    Convert any CF-netCDF dimensionless coordinate to an AuxCoordFactory.

    """
    formula_type = engine.requires.get('formula_type')
    if formula_type in ['atmosphere_hybrid_height_coordinate',
                        'atmosphere_hybrid_sigma_pressure_coordinate',
                        'ocean_sigma_z_coordinate']:
        def coord_from_term(term):
            # Convert term names to coordinates (via netCDF variable names).
            name = engine.requires['formula_terms'][term]
            for coord, cf_var_name in engine.provides['coordinates']:
                if cf_var_name == name:
                    return coord
            raise ValueError('Unable to find coordinate for variable '
                             '{!r}'.format(name))

        if formula_type == 'atmosphere_hybrid_height_coordinate':
            delta = coord_from_term('a')
            sigma = coord_from_term('b')
            orography = coord_from_term('orog')
            factory = HybridHeightFactory(delta, sigma, orography)
        elif formula_type == 'atmosphere_hybrid_sigma_pressure_coordinate':
            # Hybrid pressure has two valid versions of its formula terms:
            # "p0: var1 a: var2 b: var3 ps: var4" or
            # "ap: var1 b: var2 ps: var3" where "ap = p0 * a"
            try:
                # Attempt to get the "ap" term.
                delta = coord_from_term('ap')
            except (KeyError, ValueError):
                # The "ap" term is unavailable, so try getting terms "p0"
                # and "a" terms in order to derive an "ap" equivalent term.
                coord_p0 = coord_from_term('p0')
                if coord_p0.shape != (1,):
                    msg = 'Expecting {!r} to be a scalar reference pressure ' \
                        'coordinate, got shape {!r}'.format(coord_p0.var_name,
                                                            coord_p0.shape)
                    raise ValueError(msg)
                if coord_p0.has_bounds():
                    msg = 'Ignoring atmosphere hybrid sigma pressure scalar ' \
                        'coordinate {!r} bounds.'.format(coord_p0.name())
                    warnings.warn(msg)
                coord_a = coord_from_term('a')
                delta = coord_a * coord_p0.points[0]
                delta.units = coord_a.units * coord_p0.units
                delta.rename('vertical pressure')
                delta.var_name = 'ap'
                cube.add_aux_coord(delta, cube.coord_dims(coord_a))

            sigma = coord_from_term('b')
            surface_air_pressure = coord_from_term('ps')
            factory = HybridPressureFactory(delta, sigma, surface_air_pressure)
        elif formula_type == 'ocean_sigma_z_coordinate':
            sigma = coord_from_term('sigma')
            eta = coord_from_term('eta')
            depth = coord_from_term('depth')
            depth_c = coord_from_term('depth_c')
            nsigma = coord_from_term('nsigma')
            zlev = coord_from_term('zlev')
            factory = OceanSigmaZFactory(sigma, eta, depth,
                                         depth_c, nsigma, zlev)
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
        data_variables = cf.cf_group.data_variables.values() + \
            cf.cf_group.promoted.values()
        for cf_var in data_variables:
            cube = _load_cube(engine, cf, cf_var, filename)

            # Process any associated formula terms and attach
            # the corresponding AuxCoordFactory.
            _load_aux_factory(engine, cube)

            # Perform any user registered callback function.
            cube = iris.io.run_callback(callback, cube, cf_var, filename)

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
        try:
            self._dataset = netCDF4.Dataset(filename, mode='w',
                                            format=netcdf_format)
        except RuntimeError:
            dir_name = os.path.dirname(filename)
            if not os.path.isdir(dir_name):
                msg = 'No such file or directory: {}'.format(dir_name)
                raise IOError(msg)
            if not os.access(dir_name, os.R_OK | os.W_OK):
                msg = 'Permission denied: {}'.format(filename)
                raise IOError(msg)
            else:
                raise

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        """Flush any buffered data to the CF-netCDF file before closing."""

        self._dataset.sync()
        self._dataset.close()

    def write(self, cube, local_keys=None, unlimited_dimensions=None,
              zlib=False, complevel=4, shuffle=True, fletcher32=False,
              contiguous=False, chunksizes=None, endian='native',
              least_significant_digit=None):
        """
        Wrapper for saving cubes to a NetCDF file.

        Args:

        * cube (:class:`iris.cube.Cube`):
            A :class:`iris.cube.Cube` to be saved to a netCDF file.

        Kwargs:

        * local_keys (iterable of strings):
            An interable of cube attribute keys. Any cube attributes with
            matching keys will become attributes on the data variable rather
            than global attributes.

        * unlimited_dimensions (iterable of strings and/or
          :class:`iris.coords.Coord` objects):
            Explicit list of coordinate names (or coordinate objects)
            corresponding to coordinate dimensions of `cube` to save with the
            NetCDF dimension variable length 'UNLIMITED'. By default, the
            outermost (first) dimension for each cube is used. Only the
            'NETCDF4' format supports multiple 'UNLIMITED' dimensions. To save
            no unlimited dimensions, use `unlimited_dimensions=[]` (an empty
            list).

        * zlib (bool):
            If `True`, the data will be compressed in the netCDF file using
            gzip compression (default `False`).

        * complevel (int):
            An integer between 1 and 9 describing the level of compression
            desired (default 4). Ignored if `zlib=False`.

        * shuffle (bool):
            If `True`, the HDF5 shuffle filter will be applied before
            compressing the data (default `True`). This significantly improves
            compression. Ignored if `zlib=False`.

        * fletcher32 (bool):
            If `True`, the Fletcher32 HDF5 checksum algorithm is activated to
            detect errors. Default `False`.

        * contiguous (bool):
            If `True`, the variable data is stored contiguously on disk.
            Default `False`. Setting to `True` for a variable with an unlimited
            dimension will trigger an error.

        * chunksizes (tuple of int):
            Used to manually specify the HDF5 chunksizes for each dimension of
            the variable. A detailed discussion of HDF chunking and I/O
            performance is available here:
            http://www.hdfgroup.org/HDF5/doc/H5.user/Chunking.html. Basically,
            you want the chunk size for each dimension to match as closely as
            possible the size of the data block that users will read from the
            file. `chunksizes` cannot be set if `contiguous=True`.

        * endian (string):
            Used to control whether the data is stored in little or big endian
            format on disk. Possible values are 'little', 'big' or 'native'
            (default). The library will automatically handle endian conversions
            when the data is read, but if the data is always going to be read
            on a computer with the opposite format as the one used to create
            the file, there may be some performance advantage to be gained by
            setting the endian-ness.

        * least_significant_digit (int):
            If `least_significant_digit` is specified, variable data will be
            truncated (quantized). In conjunction with `zlib=True` this
            produces 'lossy', but significantly more efficient compression. For
            example, if `least_significant_digit=1`, data will be quantized
            using `numpy.around(scale*data)/scale`, where `scale = 2**bits`,
            and `bits` is determined so that a precision of 0.1 is retained (in
            this case `bits=4`). From
            http://www.cdc.noaa.gov/cdc/conventions/cdc_netcdf_standard.shtml:
            "least_significant_digit -- power of ten of the smallest decimal
            place in unpacked data that is a reliable value". Default is
            `None`, or no quantization, or 'lossless' compression.

        Returns:
            None.

        .. note::

            The `zlib`, `complevel`, `shuffle`, `fletcher32`, `contiguous`,
            `chunksizes` and `endian` keywords are silently ignored for netCDF
            3 files that do not use HDF5.

        """
        cf_profile_available = (iris.site_configuration.get('cf_profile') not
                                in [None, False])
        if cf_profile_available:
            # Perform a CF profile of the cube. This may result in an exception
            # being raised if mandatory requirements are not satisfied.
            profile = iris.site_configuration['cf_profile'](cube)

        # Get suitable dimension names.
        dimension_names = self._get_dim_names(cube)

        # Create the CF-netCDF data dimensions.
        self._create_cf_dimensions(cube, dimension_names, unlimited_dimensions)

        # Create the associated cube CF-netCDF data variable.
        cf_var_cube = self._create_cf_data_variable(
            cube, dimension_names, local_keys, zlib=zlib, complevel=complevel,
            shuffle=shuffle, fletcher32=fletcher32, contiguous=contiguous,
            chunksizes=chunksizes, endian=endian,
            least_significant_digit=least_significant_digit)

        # Add coordinate variables.
        self._add_dim_coords(cube, dimension_names)

        # Add the auxiliary coordinate variable names and associate the data
        # variable to them
        self._add_aux_coords(cube, cf_var_cube, dimension_names)

        # Add the formula terms to the appropriate cf variables for each
        # aux factory in the cube.
        self._add_aux_factories(cube)

        # Add data variable-only attribute names to local_keys.
        if local_keys is None:
            local_keys = set()
        else:
            local_keys = set(local_keys)
        local_keys.update(_CF_DATA_ATTRS, _UKMO_DATA_ATTRS)

        # Add global attributes taking into account local_keys.
        global_attributes = {k: v for k, v in cube.attributes.iteritems() if k
                             not in local_keys and k.lower() != 'conventions'}
        self.update_global_attributes(global_attributes)

        if cf_profile_available:
            cf_patch = iris.site_configuration.get('cf_patch')
            if cf_patch is not None:
                # Perform a CF patch of the dataset.
                cf_patch(profile, self._dataset, cf_var_cube)
            else:
                msg = 'cf_profile is available but no {} defined.'.format(
                    'cf_patch')
                warnings.warn(msg)

    def update_global_attributes(self, attributes=None, **kwargs):
        """
        Update the CF global attributes based on the provided
        iterable/dictionary and/or keyword arguments.

        Args:

        * attributes (dict or iterable of key, value pairs):
            CF global attributes to be updated.

        """
        if attributes is not None:
            # Handle sequence e.g. [('fruit', 'apple'), ...].
            if not hasattr(attributes, 'keys'):
                attributes = dict(attributes)

            for attr_name in sorted(attributes):
                self._dataset.setncattr(attr_name, attributes[attr_name])

        for attr_name in sorted(kwargs):
            self._dataset.setncattr(attr_name, kwargs[attr_name])

    def _create_cf_dimensions(self, cube, dimension_names,
                              unlimited_dimensions=None):
        """
        Create the CF-netCDF data dimensions.

        Args:

        * cube (:class:`iris.cube.Cube`):
            A :class:`iris.cube.Cube` in which to lookup coordinates.

        Kwargs:

        * unlimited_dimensions (iterable of strings and/or
          :class:`iris.coords.Coord` objects):
            List of coordinates to make unlimited. By default, the
            outermost dimension is made unlimited.

        Returns:
            None.

        """
        unlimited_dim_names = []
        if unlimited_dimensions is None:
            if dimension_names:
                unlimited_dim_names.append(dimension_names[0])
        else:
            for coord in unlimited_dimensions:
                try:
                    coord = cube.coord(name_or_coord=coord, dim_coords=True)
                except iris.exceptions.CoordinateNotFoundError:
                    # coordinate isn't used for this cube, but it might be
                    # used for a different one
                    pass
                else:
                    dim_name = self._get_coord_variable_name(cube, coord)
                    unlimited_dim_names.append(dim_name)

        for dim_name in dimension_names:
            if dim_name not in self._dataset.dimensions:
                if dim_name in unlimited_dim_names:
                    size = None
                else:
                    size = self._existing_dim[dim_name]
                self._dataset.createDimension(dim_name, size)

    def _add_aux_coords(self, cube, cf_var_cube, dimension_names):
        """
        Add aux. coordinate to the dataset and associate with the data variable

        Args:

        * cube (:class:`iris.cube.Cube`):
            A :class:`iris.cube.Cube` to be saved to a netCDF file.
        * cf_var_cube (:class:`netcdf.netcdf_variable`):
            cf variable cube representation.
        * dimension_names (list):
            Names associated with the dimensions of the cube.

        """
        auxiliary_coordinate_names = []
        # Add CF-netCDF variables for the associated auxiliary coordinates.
        for coord in sorted(cube.aux_coords, key=lambda coord: coord.name()):
            # Create the associated coordinate CF-netCDF variable.
            if coord not in self._name_coord_map.coords:
                cf_name = self._create_cf_variable(cube, dimension_names,
                                                   coord)
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

    def _add_dim_coords(self, cube, dimension_names):
        """
        Add coordinate variables to NetCDF dataset.

        Args:

        * cube (:class:`iris.cube.Cube`):
            A :class:`iris.cube.Cube` to be saved to a netCDF file.
        * dimension_names (list):
            Names associated with the dimensions of the cube.

        """
        # Ensure we create the netCDF coordinate variables first.
        for coord in cube.dim_coords:
            # Create the associated coordinate CF-netCDF variable.
            if coord not in self._name_coord_map.coords:
                cf_name = self._create_cf_variable(cube, dimension_names,
                                                   coord)
                self._name_coord_map.append(cf_name, coord)

    def _add_aux_factories(self, cube):
        """
        Modifies the variables of the NetCDF dataset to represent
        the presence of dimensionless vertical coordinates based on
        the aux factories of the cube (if any).

        Args:

        * cube (:class:`iris.cube.Cube`):
            A :class:`iris.cube.Cube` to be saved to a netCDF file.

        """
        primaries = []
        for factory in cube.aux_factories:
            factory_defn = _FACTORY_DEFNS.get(type(factory), None)
            if factory_defn is None:
                msg = 'Unable to determine formula terms ' \
                      'for AuxFactory: {!r}'.format(factory)
                warnings.warn(msg)
            else:
                # Override `standard_name`, `long_name`, and `axis` of the
                # primary coord that signals the presense of a dimensionless
                # vertical coord, then set the `formula_terms` attribute.
                primary_coord = factory.dependencies[factory_defn.primary]
                if primary_coord in primaries:
                    msg = 'Cube {!r} has multiple aux factories that share ' \
                          'a common primary coordinate {!r}. Unable to save ' \
                          'to netCDF as having multiple formula terms on a ' \
                          'single coordinate is not supported.'
                    raise ValueError(msg.format(cube, primary_coord.name()))
                primaries.append(primary_coord)
                cf_name = self._name_coord_map.name(primary_coord)
                cf_var = self._dataset.variables[cf_name]
                cf_var.standard_name = factory_defn.std_name
                cf_var.axis = 'Z'
                names = {key: self._name_coord_map.name(coord) for
                         key, coord in factory.dependencies.iteritems()}
                formula_terms = factory_defn.formula_terms_format.format(
                    **names)
                cf_var.formula_terms = formula_terms

    def _get_dim_names(self, cube):
        """
        Determine suitable CF-netCDF data dimension names.

        Args:

        * cube (:class:`iris.cube.Cube`):
            A :class:`iris.cube.Cube` to be saved to a netCDF file.

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

    def _ensure_valid_dtype(self, values, src_name, src_object):
        # NetCDF3 does not support int64 or unsigned ints, so we check
        # if we can store them as int32 instead.
        if ((np.issubdtype(values.dtype, np.int64) or
                np.issubdtype(values.dtype, np.unsignedinteger)) and
                self._dataset.file_format in ('NETCDF3_CLASSIC',
                                              'NETCDF3_64BIT')):
            # Cast to an integer type supported by netCDF3.
            if not np.can_cast(values.max(), np.int32) or \
                    not np.can_cast(values.min(), np.int32):
                msg = 'The data type of {} {!r} is not supported by {} and' \
                      ' its values cannot be safely cast to a supported' \
                      ' integer type.'
                msg = msg.format(src_name, src_object,
                                 self._dataset.file_format)
                raise ValueError(msg)
            values = values.astype(np.int32)
        return values

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
            # Get the values in a form which is valid for the file format.
            bounds = self._ensure_valid_dtype(coord.bounds,
                                              'the bounds of coordinate',
                                              coord)
            n_bounds = bounds.shape[-1]

            if n_bounds == 2:
                bounds_dimension_name = 'bnds'
            else:
                bounds_dimension_name = 'bnds_%s' % n_bounds

            if bounds_dimension_name not in self._dataset.dimensions:
                # Create the bounds dimension with the appropriate extent.
                self._dataset.createDimension(bounds_dimension_name, n_bounds)

            cf_var.bounds = cf_name + '_bnds'
            cf_var_bounds = self._dataset.createVariable(
                cf_var.bounds, bounds.dtype.newbyteorder('='),
                cf_var.dimensions + (bounds_dimension_name,))
            cf_var_bounds[:] = bounds

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

    def _create_cf_variable(self, cube, dimension_names, coord):
        """
        Create the associated CF-netCDF variable in the netCDF dataset for the
        given coordinate. If required, also create the CF-netCDF bounds
        variable and associated dimension.

        Args:

        * cube (:class:`iris.cube.Cube`):
            The associated cube being saved to CF-netCDF file.
        * dimension_names (list):
            Names for each dimension of the cube.
        * coord (:class:`iris.coords.Coord`):
            The coordinate to be saved to CF-netCDF file.

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

            # Get the values in a form which is valid for the file format.
            points = self._ensure_valid_dtype(coord.points, 'coordinate',
                                              coord)

            # Create the CF-netCDF variable.
            cf_var = self._dataset.createVariable(
                cf_name, points.dtype.newbyteorder('='), cf_dimensions)

            # Add the axis attribute for spatio-temporal CF-netCDF coordinates.
            if coord in cf_coordinates:
                axis = iris.util.guess_coord_axis(coord)
                if axis is not None and axis.lower() in SPATIO_TEMPORAL_AXES:
                    cf_var.axis = axis.upper()

            # Add the data to the CF-netCDF variable.
            cf_var[:] = points

            # Create the associated CF-netCDF bounds variable.
            self._create_cf_bounds(coord, cf_var, cf_name)

        # Deal with CF-netCDF units and standard name.
        standard_name, long_name, units = self._cf_coord_identity(coord)

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
                name = 'um_stash_source'
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

                def add_ellipsoid(ellipsoid):
                    cf_var_grid.longitude_of_prime_meridian = (
                        ellipsoid.longitude_of_prime_meridian)
                    semi_major = ellipsoid.semi_major_axis
                    semi_minor = ellipsoid.semi_minor_axis
                    if semi_minor == semi_major:
                        cf_var_grid.earth_radius = semi_major
                    else:
                        cf_var_grid.semi_major_axis = semi_major
                        cf_var_grid.semi_minor_axis = semi_minor

                # latlon
                if isinstance(cs, iris.coord_systems.GeogCS):
                    add_ellipsoid(cs)

                # rotated latlon
                elif isinstance(cs, iris.coord_systems.RotatedGeogCS):
                    if cs.ellipsoid:
                        add_ellipsoid(cs.ellipsoid)
                    cf_var_grid.grid_north_pole_latitude = (
                        cs.grid_north_pole_latitude)
                    cf_var_grid.grid_north_pole_longitude = (
                        cs.grid_north_pole_longitude)
                    cf_var_grid.north_pole_grid_longitude = (
                        cs.north_pole_grid_longitude)

                # tmerc
                elif isinstance(cs, iris.coord_systems.TransverseMercator):
                    if cs.ellipsoid:
                        add_ellipsoid(cs.ellipsoid)
                    cf_var_grid.longitude_of_central_meridian = (
                        cs.longitude_of_central_meridian)
                    cf_var_grid.latitude_of_projection_origin = (
                        cs.latitude_of_projection_origin)
                    cf_var_grid.false_easting = cs.false_easting
                    cf_var_grid.false_northing = cs.false_northing
                    cf_var_grid.scale_factor_at_central_meridian = (
                        cs.scale_factor_at_central_meridian)

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

    def _create_cf_data_variable(self, cube, dimension_names, local_keys=None,
                                 **kwargs):
        """
        Create CF-netCDF data variable for the cube and any associated grid
        mapping.

        Args:

        * cube (:class:`iris.cube.Cube`):
            The associated cube being saved to CF-netCDF file.
        * dimension_names (list):
            String names for each dimension of the cube.

        Kwargs:

        * local_keys (iterable of strings):
            An interable of cube attribute keys. Any cube attributes
            with matching keys will become attributes on the data variable.

        All other keywords are passed through to the dataset's `createVariable`
        method.

        Returns:
            The newly created CF-netCDF data variable.

        """
        cf_name = self._get_cube_variable_name(cube)
        while cf_name in self._dataset.variables:
            cf_name = self._increment_name(cf_name)

        # if netcdf3 avoid streaming due to dtype handling
        if self._dataset.file_format in ('NETCDF3_CLASSIC',
                                         'NETCDF3_64BIT'):
            # Determine whether there is a cube MDI value.
            fill_value = None
            if isinstance(cube.data, ma.core.MaskedArray):
                fill_value = cube.data.fill_value

            # Get the values in a form which is valid for the file format.
            data = self._ensure_valid_dtype(cube.data, 'cube', cube)

            # Create the cube CF-netCDF data variable with data payload.
            cf_var = self._dataset.createVariable(
                cf_name, data.dtype.newbyteorder('='), dimension_names,
                fill_value=fill_value, **kwargs)
            cf_var[:] = data

        else:
            # Create the cube CF-netCDF data variable.
            cf_var = self._dataset.createVariable(
                cf_name, cube.lazy_data().dtype.newbyteorder('='), 
                dimension_names, **kwargs)
            # stream the data
            biggus.save([cube.lazy_data()], [cf_var], masked=True)

        if cube.standard_name:
            cf_var.standard_name = cube.standard_name

        if cube.long_name:
            cf_var.long_name = cube.long_name

        if cube.units != 'unknown':
            cf_var.units = str(cube.units)

        # Add data variable-only attribute names to local_keys.
        if local_keys is None:
            local_keys = set()
        else:
            local_keys = set(local_keys)
        local_keys.update(_CF_DATA_ATTRS, _UKMO_DATA_ATTRS)

        # Add any cube attributes whose keys are in local_keys as
        # CF-netCDF data variable attributes.
        attr_names = set(cube.attributes).intersection(local_keys)
        for attr_name in sorted(attr_names):
            # Do not output 'conventions' attribute.
            if attr_name.lower() == 'conventions':
                continue

            value = cube.attributes[attr_name]

            if attr_name == 'STASH':
                # Adopting provisional Metadata Conventions for representing MO
                # Scientific Data encoded in NetCDF Format.
                attr_name = 'um_stash_source'
                value = str(value)

            if attr_name == "ukmo__process_flags":
                value = " ".join([x.replace(" ", "_") for x in value])

            if attr_name in _CF_GLOBAL_ATTRS:
                msg = '{attr_name!r} is being added as CF data variable ' \
                      'attribute, but {attr_name!r} should only be a CF ' \
                      'global attribute.'.format(attr_name=attr_name)
                warnings.warn(msg)

            setattr(cf_var, attr_name, value)

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


def save(cube, filename, netcdf_format='NETCDF4', local_keys=None,
         unlimited_dimensions=None, zlib=False, complevel=4, shuffle=True,
         fletcher32=False, contiguous=False, chunksizes=None, endian='native',
         least_significant_digit=None):
    """
    Save cube(s) to a netCDF file, given the cube and the filename.

    * Iris will write CF 1.5 compliant NetCDF files.
    * The attributes dictionaries on each cube in the saved cube list
      will be compared and common attributes saved as NetCDF global
      attributes where appropriate.
    * Keyword arguments specifying how to save the data are applied
      to each cube. To use different settings for different cubes, use
      the NetCDF Context manager (:class:`~Saver`) directly.

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

    * local_keys (iterable of strings):
        An interable of cube attribute keys. Any cube attributes with
        matching keys will become attributes on the data variable rather
        than global attributes.

    * unlimited_dimensions (iterable of strings and/or
      :class:`iris.coords.Coord` objects):
        Explicit list of coordinate names (or coordinate objects) corresponding
        to coordinate dimensions of `cube` to save with the NetCDF dimension
        variable length 'UNLIMITED'. By default, the outermost (first)
        dimension for each cube is used. Only the 'NETCDF4' format supports
        multiple 'UNLIMITED' dimensions. To save no unlimited dimensions, use
        `unlimited_dimensions=[]` (an empty list).

    * zlib (bool):
        If `True`, the data will be compressed in the netCDF file using gzip
        compression (default `False`).

    * complevel (int):
        An integer between 1 and 9 describing the level of compression desired
        (default 4). Ignored if `zlib=False`.

    * shuffle (bool):
        If `True`, the HDF5 shuffle filter will be applied before compressing
        the data (default `True`). This significantly improves compression.
        Ignored if `zlib=False`.

    * fletcher32 (bool):
        If `True`, the Fletcher32 HDF5 checksum algorithm is activated to
        detect errors. Default `False`.

    * contiguous (bool):
        If `True`, the variable data is stored contiguously on disk. Default
        `False`. Setting to `True` for a variable with an unlimited dimension
        will trigger an error.

    * chunksizes (tuple of int):
        Used to manually specify the HDF5 chunksizes for each dimension of the
        variable. A detailed discussion of HDF chunking and I/O performance is
        available here: http://www.hdfgroup.org/HDF5/doc/H5.user/Chunking.html.
        Basically, you want the chunk size for each dimension to match as
        closely as possible the size of the data block that users will read
        from the file. `chunksizes` cannot be set if `contiguous=True`.

    * endian (string):
        Used to control whether the data is stored in little or big endian
        format on disk. Possible values are 'little', 'big' or 'native'
        (default). The library will automatically handle endian conversions
        when the data is read, but if the data is always going to be read on a
        computer with the opposite format as the one used to create the file,
        there may be some performance advantage to be gained by setting the
        endian-ness.

    * least_significant_digit (int):
        If `least_significant_digit` is specified, variable data will be
        truncated (quantized). In conjunction with `zlib=True` this produces
        'lossy', but significantly more efficient compression. For example, if
        `least_significant_digit=1`, data will be quantized using
        `numpy.around(scale*data)/scale`, where `scale = 2**bits`, and `bits`
        is determined so that a precision of 0.1 is retained (in this case
        `bits=4`). From
        http://www.cdc.noaa.gov/cdc/conventions/cdc_netcdf_standard.shtml:
        "least_significant_digit -- power of ten of the smallest decimal place
        in unpacked data that is a reliable value". Default is `None`, or no
        quantization, or 'lossless' compression.

    Returns:
        None.

    .. note::

        The `zlib`, `complevel`, `shuffle`, `fletcher32`, `contiguous`,
        `chunksizes` and `endian` keywords are silently ignored for netCDF 3
        files that do not use HDF5.

    .. seealso::

        NetCDF Context manager (:class:`~Saver`).

    """
    if isinstance(cube, iris.cube.Cube):
        cubes = iris.cube.CubeList()
        cubes.append(cube)
    else:
        cubes = cube

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
            if np.any(attributes[key] != cube.attributes[key]):
                different_value_keys.append(key)
        common_keys.difference_update(different_value_keys)
        local_keys.update(different_value_keys)

    # Initialise Manager for saving
    with Saver(filename, netcdf_format) as sman:
        # Iterate through the cubelist.
        for cube in cubes:
            sman.write(cube, local_keys, unlimited_dimensions, zlib, complevel,
                       shuffle, fletcher32, contiguous, chunksizes, endian,
                       least_significant_digit)

        conventions = CF_CONVENTIONS_VERSION

        # Perform a CF patch of the conventions attribute.
        cf_profile_available = (iris.site_configuration.get('cf_profile') not
                                in [None, False])
        if cf_profile_available:
            conventions_patch = iris.site_configuration.get(
                'cf_patch_conventions')
            if conventions_patch is not None:
                conventions = conventions_patch(conventions)
            else:
                msg = 'cf_profile is available but no {} defined.'.format(
                    'cf_patch_conventions')
                warnings.warn(msg)

        # Add conventions attribute.
        sman.update_global_attributes(Conventions=conventions)
