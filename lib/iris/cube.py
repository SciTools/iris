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
Classes for representing multi-dimensional data with metadata.

"""

from xml.dom.minidom import Document
import collections
import copy
import datetime
import itertools
import operator
import re
import UserDict
import warnings
import zlib

import numpy

import iris.analysis
import iris.analysis.maths
import iris.analysis.interpolate
import iris.analysis.trajectory
import iris.aux_factory
import iris.coord_systems
import iris.coords
import iris._constraints
import iris._merge
import iris.exceptions
import iris.fileformats.rules
import iris.util

from iris._cube_coord_common import CFVariableMixin, LimitedAttributeDict


__all__ = ['Cube', 'CubeList', 'CubeMetadata']


class CubeMetadata(collections.namedtuple('CubeMetadata', 
                                          ['standard_name', 'long_name',
                                           'units', 'attributes',
                                           'cell_methods'])):
    """
    Represents the phenomenon metadata for a single :class:`Cube`.

    """
    def name(self, default='unknown'):
        """
        Returns a human-readable name.
        
        First it tries self.standard_name, then it tries the 'long_name'
        attributes, before falling back to the value of `default` (which
        itself defaults to 'unknown').
        
        """
        return self.standard_name or self.long_name or default


# The XML namespace to use for CubeML documents
XML_NAMESPACE_URI = "urn:x-iris:cubeml-0.2"


class _CubeFilter(object):
    """
    A constraint, paired with a list of cubes matching that constraint.

    """
    def __init__(self, constraint, cubes=None):
        self.constraint = constraint
        if cubes is None:
            cubes = CubeList()
        self.cubes = cubes

    def __len__(self):
        return len(self.cubes)

    def add(self, cube):
        """
        Adds the appropriate (sub)cube to the list of cubes where it
        matches the constraint.

        """
        sub_cube = self.constraint.extract(cube)
        if sub_cube is not None:
            self.cubes.append(sub_cube)

    def merged(self, unique=False):
        """
        Returns a new :class:`_CubeFilter` by merging the list of
        cubes.

        Kwargs:

        * unique:
            If True, raises `iris.exceptions.DuplicateDataError` if
            duplicate cubes are detected.

        """
        return _CubeFilter(self.constraint, self.cubes.merge(unique))


class _CubeFilterCollection(object):
    """
    A list of _CubeFilter instances.

    """
    @staticmethod
    def from_cubes(cubes, constraints=None):
        """
        Creates a new collection from an iterable of cubes, and some
        optional constraints.
        
        """
        constraints = iris._constraints.list_of_constraints(constraints)
        pairs = [_CubeFilter(constraint) for constraint in constraints]
        collection = _CubeFilterCollection(pairs)
        for cube in cubes:
            collection.add_cube(cube)
        return collection

    def __init__(self, pairs):
        self.pairs = pairs

    def add_cube(self, cube):
        """
        Adds the given :class:`~iris.cube.Cube` to all of the relevant
        constraint pairs.

        """
        for pair in self.pairs:
            pair.add(cube)

    def cubes(self):
        """
        Returns all the cubes in this collection concatenated into a
        single :class:`CubeList`.

        """
        result = CubeList()
        for pair in self.pairs:
            result.extend(pair.cubes)
        return result

    def merged(self, unique=False):
        """
        Returns a new :class:`_CubeFilterCollection` by merging all the cube
        lists of this collection.

        Kwargs:

        * unique:
            If True, raises `iris.exceptions.DuplicateDataError` if
            duplicate cubes are detected.

        """
        return _CubeFilterCollection([pair.merged(unique) for pair in self.pairs])


class CubeList(list):
    """All the functionality of a standard :class:`list` with added "Cube" context."""

    def __new__(cls, list_of_cubes=None):
        """Given a :class:`list` of cubes, return a CubeList instance."""
        cube_list = list.__new__(cls, list_of_cubes)
        
        # Check that all items in the incoming list are cubes. Note that this checking
        # does not guarantee that a CubeList instance *always* has just cubes in its list as
        # the append & __getitem__ methods have not been overridden.
        if not all([isinstance(cube, Cube) for cube in cube_list]):
            raise ValueError('All items in list_of_cubes must be Cube instances.')
        return cube_list 

    def __str__(self):
        """Runs short :method:`Cube.summary` on every cube."""
        result = ['%s: %s' % (i, cube.summary(shorten=True)) for i, cube in enumerate(self)]
        if result:
            result = '\n'.join(result)
        else:
            result = '< No cubes >'
        return result 

    def __repr__(self):
        """Runs repr on every cube."""
        return '[%s]' % ',\n'.join([repr(cube) for cube in self])

    # TODO #370 Which operators need overloads?
    def __add__(self, other):
        return CubeList(list.__add__(self, other))

    def xml(self, checksum=False):
        """Return a string of the XML that this list of cubes represents."""
        doc = Document()
        cubes_xml_element = doc.createElement("cubes")
        cubes_xml_element.setAttribute("xmlns", XML_NAMESPACE_URI)

        for cube_obj in self:
            cubes_xml_element.appendChild(cube_obj._xml_element(doc, checksum=checksum))

        doc.appendChild(cubes_xml_element)

        # return our newly created XML string
        return doc.toprettyxml(indent="  ")

    def extract(self, constraints, strict=False):
        """
        Filter each of the cubes which can be filtered by the given constraints.

        This method iterates over each constraint given, and subsets each of the cubes
        in this CubeList where possible. Thus, a CubeList of length **n** when filtered
        with **m** constraints can generate a maximum of **m * n** cubes.

        Keywords:

        * strict - boolean
            If strict is True, then there must be exactly one cube which is filtered per
            constraint.        

        """
        return self._extract_and_merge(self, constraints, strict, merge_unique=None)

    @staticmethod
    def _extract_and_merge(cubes, constraints, strict, merge_unique=False):
        # * merge_unique - if None: no merging, if false: non unique merging, else unique merging (see merge)

        constraints = iris._constraints.list_of_constraints(constraints)

        # group the resultant cubes by constraints in a dictionary
        constraint_groups = dict([(constraint, CubeList()) for constraint in constraints])
        for cube in cubes:
            for constraint, cube_list in constraint_groups.iteritems():
                sub_cube = constraint.extract(cube)
                if sub_cube is not None:
                    cube_list.append(sub_cube)

        if merge_unique is not None:
            for constraint, cubelist in constraint_groups.iteritems():
                constraint_groups[constraint] = cubelist.merge(merge_unique)

        result = CubeList()
        for constraint in constraints:
            constraint_cubes = constraint_groups[constraint]
            if strict and len(constraint_cubes) != 1:
                raise iris.exceptions.ConstraintMismatchError('Got %s cubes for constraint %r, '
                                                        'expecting 1.' % (len(constraint_cubes), constraint))
            result.extend(constraint_cubes)

        if strict and len(constraints) == 1:
            result = result[0]

        return result

    def extract_strict(self, constraints):
        """
        Calls :meth:`CubeList.extract` with the strict keyword set to True.
        """        
        return self.extract(constraints, strict=True)

    def merge(self, unique=True):
        """
        Returns the :class:`CubeList` resulting from merging this
        :class:`CubeList`.

        Kwargs:

        * unique:
            If True, raises `iris.exceptions.DuplicateDataError` if
            duplicate cubes are detected.

        """
        # Register each of our cubes with its appropriate ProtoCube.
        proto_cubes_by_name = {}
        for cube in self:
            name = cube.standard_name
            proto_cubes = proto_cubes_by_name.setdefault(name, [])
            proto_cube = None

            for target_proto_cube in proto_cubes:
                if target_proto_cube.register(cube):
                    proto_cube = target_proto_cube
                    break

            if proto_cube is None:
                proto_cube = iris._merge.ProtoCube(cube)
                proto_cubes.append(proto_cube)

        # Extract all the merged cubes from the ProtoCubes.
        merged_cubes = CubeList()
        for name in sorted(proto_cubes_by_name):
            for proto_cube in proto_cubes_by_name[name]:
                merged_cubes.extend(proto_cube.merge(unique=unique))

        return merged_cubes


class Cube(CFVariableMixin):
    """
    A single Iris cube of data and metadata.

    Typically obtained from :func:`iris.load`, :func:`iris.load_cube`,
    :func:`iris.load_cubes`, or from the manipulation of existing cubes.

    For example:

        >>> cube = iris.load_cube(iris.sample_data_path('air_temp.pp'))
        >>> print cube
        air_temperature                     (latitude: 73; longitude: 96)
             Dimension coordinates:
                  latitude                           x              -
                  longitude                          -              x
             Scalar coordinates:
                  forecast_period: 6477 hours
                  forecast_reference_time: 243363.0 hours since 1970-01-01 00:00:00
                  pressure: 1000.0 hPa
                  time: 232560.0 hours since 1970-01-01 00:00:00, bound=(215280.0, 249840.0) hours since 1970-01-01 00:00:00
             Attributes:
                  STASH: m01s16i203
                  source: Data from Met Office Unified Model
             Cell methods:
                  mean: time


    See the :doc:`user guide</userguide/index>` for more information.

    """
    def __init__(self, data, standard_name=None, long_name=None, units=None,
                 attributes=None, cell_methods=None, dim_coords_and_dims=None,
                 aux_coords_and_dims=None, aux_factories=None, data_manager=None):        
        """
        Creates a cube with data and optional metadata. 

        Not typically used - normally cubes are obtained by loading data
        (e.g. :func:`iris.load`) or from manipulating existing cubes.
        
        Args:
        
        * data 
            A numpy array containing the phenomenon values or a data manager object. This object defines the 
            shape of the cube and the value in each cell. 
            
            See :attr:`Cube.data<iris.cube.Cube.data>` and :class:`iris.fileformats.manager.DataManager` 

        Kwargs:

        * standard_name
            The standard name for the Cube's data.
        * long_name
            An unconstrained description of the cube. 
        * units
            The unit of the cube, e.g. ``"m s-1"`` or ``"kelvin"``
        * attributes
            A dictionary of cube attributes
        * cell_methods
            A tuple of CellMethod objects, generally set by Iris, e.g. ``(CellMethod("mean", coords='latitude'), )``
        * dim_coords_and_dims
            A list of coordinates with scalar dimension mappings, e.g ``[(lat_coord, 0), (lon_coord, 1)]``.
        * aux_coords_and_dims
            A list of coordinates with dimension mappings, e.g ``[(lat_coord, 0), (lon_coord, (0, 1))]``.
            See also :meth:`Cube.add_dim_coord()<iris.cube.Cube.add_dim_coord>` and :meth:`Cube.add_aux_coord()<iris.cube.Cube.add_aux_coord>`.
        * aux_factories
            A list of auxiliary coordinate factories. See :mod:`iris.aux_factory`.
        * data_manager
            A :class:`iris.fileformats.manager.DataManager` instance. If a data manager is provided, then
            the data should be a numpy array of data proxy instances. See :class:`iris.fileformats.pp.PPDataProxy` or
            :class:`iris.fileformats.netcdf.NetCDFDataProxy`.

        For example::

            latitude = DimCoord(range(-85, 105, 10), standard_name='latitude', units='degrees')
            longitude = DimCoord(range(0, 360, 10), standard_name='longitude', units='degrees')
            cube = Cube(numpy.zeros((18, 36), numpy.float32),
                        dim_coords_and_dims=[(latitude, 0), (longitude, 1)])

        """
        # Temporary error while we transition the API.
        if isinstance(data, basestring):
            raise TypeError('Invalid data type: {!r}.'.format(data))

        if data_manager is not None:
            self._data = data
            self._data_manager = data_manager
        else:
            if isinstance(data, numpy.ndarray):
                self._data = data
            else:
                self._data = numpy.asarray(data)                
            self._data_manager = None

        self.standard_name = standard_name
        """The "standard name" for the Cube's phenomenon."""

        self.units = units
        """An instance of :class:`iris.unit.Unit` describing the Cube's data."""
        
        self.long_name = long_name
        """The "long name" for the Cube's phenomenon."""
        
        self.cell_methods = cell_methods
        
        self.attributes = attributes
        """A dictionary, with a few restricted keys, for arbitrary Cube metadata."""

        # Coords
        self._dim_coords_and_dims = []
        self._aux_coords_and_dims = []
        self._aux_factories = []
        
        if dim_coords_and_dims:
            for coord, dim in dim_coords_and_dims:
                self.add_dim_coord(coord, dim)

        if aux_coords_and_dims:
            for coord, dims in aux_coords_and_dims:
                self.add_aux_coord(coord, dims)

        if aux_factories:
            for factory in aux_factories:
                self.add_aux_factory(factory)

    @property
    def metadata(self):
        """
        An instance of :class:`CubeMetadata` describing the phenomenon.
        
        This property can be updated with any of:
         - another :class:`CubeMetadata` instance,
         - a tuple/dict which can be used to make a :class:`CubeMetadata`,
         - or any object providing the attributes exposed by
           :class:`CubeMetadata`.

        """
        return CubeMetadata(self.standard_name, self.long_name, self.units,
                            self.attributes, self.cell_methods)

    @metadata.setter
    def metadata(self, value):
        try:
            value = CubeMetadata(**value)
        except TypeError:
            try:
                value = CubeMetadata(*value)
            except TypeError:
                attr_check = lambda name: not hasattr(value, name)
                missing_attrs = filter(attr_check, CubeMetadata._fields)
                if missing_attrs:
                    raise TypeError('Invalid/incomplete metadata')
        for name in CubeMetadata._fields:
            setattr(self, name, getattr(value, name))

    @property
    def units(self):
        """The :mod:`~iris.unit.Unit` instance of the phenomenon."""
        return self._units

    @units.setter
    def units(self, unit):
        unit = iris.unit.as_unit(unit)
        # If the cube has units and the desired unit is valid convert
        # the data.
        if (hasattr(self, '_units') and
                not (self.units.unknown or
                     self.units.no_unit or
                     unit.unknown or
                     unit.no_unit)):
            self.data = self.units.convert(self.data, unit)
        self._units = unit

    def clear_units(self):
        """Sets the cube's units to 'unknown'."""
        self.units = None

    def replace_units(self, unit):
        """
        Changes the cube's units to a given value without modifying
        its data array.

        .. note::

            To convert a cube from one unit to another (e.g. kelvin
            to celsius) assign to the :attr:`units <iris.cube.Cube.units>`
            attribute directly. For example::

                cube.units = 'celsius'

        """
        self.clear_units()
        self.units = unit

    def unit_converted(self, new_unit):
        """Return a cube converted to a given unit."""
        new_cube = self.copy()
        new_cube.units = new_unit
        return new_cube

    def add_cell_method(self, cell_method):
        """Add a CellMethod to the Cube."""
        self.cell_methods += (cell_method, )

    def add_aux_coord(self, coord, data_dims=None):
        """
        Adds a CF auxiliary coordinate to the cube.

        Args:

        * coord
            The :class:`iris.coords.DimCoord` or :class:`iris.coords.AuxCoord`
            instance to add to the cube.

        Kwargs:

        * data_dims
            Integer or iterable of integers giving the data dimensions spanned by the coordinate.
            
        Raises a ValueError if a coordinate with identical metadata already exists
        on the cube.

        See also :meth:`Cube.remove_coord()<iris.cube.Cube.remove_coord>`.

        """

        # Convert to a tuple of integers
        if data_dims is None:
            data_dims = tuple()
        elif isinstance(data_dims, collections.Container):
            data_dims = tuple(int(d) for d in data_dims)
        else:
            data_dims = (int(data_dims),)

        if data_dims:
            if len(data_dims) != coord.ndim:
                msg = 'Invalid data dimensions: {} given, {} expected for {!r}.'
                msg = msg.format(len(data_dims), coord.ndim, coord.name())
                raise ValueError(msg)
            # Check compatibility with the shape of the data
            for i, dim in enumerate(data_dims):
                if coord.shape[i] != self.shape[dim]:
                    msg = 'Unequal lengths. Cube dimension {} => {};' \
                          ' coord {!r} dimension {} => {}.'
                    raise ValueError(msg.format(dim, self.shape[dim],
                                                coord.name(), i,
                                                coord.shape[i]))
        elif coord.shape != (1,):
            raise ValueError('You must supply the data-dimensions for multi-valued coordinates.')

        if self.coords(coord=coord):  # TODO: just fail on duplicate object
            raise ValueError('Duplicate coordinates are not permitted.')
        self._aux_coords_and_dims.append([coord, data_dims])

    def add_aux_factory(self, aux_factory):
        """
        Adds an auxiliary coordinate factory to the cube.

        Args:

        * aux_factory
            The :class:`iris.aux_factory.AuxCoordFactory` instance to add.

        """
        if not isinstance(aux_factory, iris.aux_factory.AuxCoordFactory):
            raise TypeError('Factory must be a subclass of iris.aux_factory.AuxCoordFactory.')
        self._aux_factories.append(aux_factory)

    def add_dim_coord(self, dim_coord, data_dim):
        """
        Add a CF coordinate to the cube.

        Args:

        * dim_coord
            The :class:`iris.coords.DimCoord` instance to add to the cube.
        * data_dim
            Integer giving the data dimension spanned by the coordinate.
            
        Raises a ValueError if a coordinate with identical metadata already exists
        on the cube or if a coord already exists for the given dimension.

        See also :meth:`Cube.remove_coord()<iris.cube.Cube.remove_coord>`.

        """
        if self.coords(coord=dim_coord):
            raise ValueError('Duplicate coordinates are not permitted.')
        if isinstance(data_dim, collections.Container) and len(data_dim) != 1:
            raise ValueError('The supplied data dimension must be a single number')

        # Convert data_dim to a single integer
        if isinstance(data_dim, collections.Container):
            data_dim = int(list(data_dim)[0])
        else:
            data_dim = int(data_dim)

        # Check data_dim value is valid
        if data_dim < 0 or data_dim >= self.ndim:
            raise ValueError('The cube does not have the specified dimension (%d)' % data_dim)

        # Check dimension is available
        if self.coords(dimensions=data_dim, dim_coords=True):
            raise ValueError('A dim_coord is already associated with dimension %d.' % data_dim)

        # Check compatibility with the shape of the data
        if dim_coord.shape[0] != self.shape[data_dim]:
            msg = 'Unequal lengths. Cube dimension {} => {}; coord {!r} => {}.'
            raise ValueError(msg.format(data_dim, self.shape[data_dim],
                                        dim_coord.name(),
                                        len(dim_coord.points)))

        self._dim_coords_and_dims.append([dim_coord, int(data_dim)])

    def remove_aux_factory(self, aux_factory):
        """Removes the given auxiliary coordinate factory from the cube."""
        self._aux_factories.remove(aux_factory)

    def _remove_coord(self, coord):
        self._dim_coords_and_dims = [(coord_, dim) for coord_, dim in self._dim_coords_and_dims if coord_ is not coord]
        self._aux_coords_and_dims = [(coord_, dims) for coord_, dims in self._aux_coords_and_dims if coord_ is not coord]

    def remove_coord(self, coord):
        """
        Removes a coordinate from the cube.

        Args:

        * coord (string or coord)
            The (name of the) coordinate to remove from the cube.

        See also :meth:`Cube.add_coord()<iris.cube.Cube.add_coord>`.

        """
        if isinstance(coord, basestring):
            coord = self.coord(name=coord)
        else:
            coord = self.coord(coord=coord)

        self._remove_coord(coord)

        for factory in self.aux_factories:
            factory.update(coord)

    def replace_coord(self, new_coord):
        """Replace the coordinate whose metadata matches the given coordinate."""
        old_coord = self.coord(coord=new_coord)
        dims = self.coord_dims(old_coord) 
        was_dimensioned = old_coord in self.dim_coords
        self._remove_coord(old_coord) 
        if was_dimensioned and isinstance(new_coord, iris.coords.DimCoord): 
            self.add_dim_coord(new_coord, dims[0]) 
        else: 
            self.add_aux_coord(new_coord, dims) 

        for factory in self.aux_factories:
            factory.update(old_coord, new_coord)

    def coord_dims(self, coord):
        """
        Returns a tuple of the data dimensions relevant to the given
        coordinate.

        When searching for the given coordinate in the cube the comparison is
        made using coordinate metadata equality. Hence the given coordinate
        instance need not exist on the cube, and may contain different
        coordinate values.

        Args:

        * coord
            The :class:`iris.coords.Coord` instance to look for.

        """
        target_defn = coord._as_defn()

        ### Search by coord definition first

        # Search dim coords first
        matches = [(dim,) for coord_, dim in self._dim_coords_and_dims if coord_._as_defn() == target_defn]

        # Search aux coords
        if not matches:
            matches = [dims for coord_, dims in self._aux_coords_and_dims if coord_._as_defn() == target_defn]

        # Search derived aux coords
        if not matches:
            match = lambda factory: factory._as_defn() == target_defn
            factories = filter(match, self._aux_factories)
            matches = [factory.derived_dims(self.coord_dims) for factory in factories]

        ### Search by coord name, if have no match
        # XXX Where did this come from? And why isn't it reflected in the docstring?
        
        if not matches:
            matches = [(dim,) for coord_, dim in self._dim_coords_and_dims if coord_.name() == coord.name()]

        # Search aux coords
        if not matches:
            matches = [dims for coord_, dims in self._aux_coords_and_dims if coord_.name() == coord.name()]

#        # Search derived aux coords
#        if not matches:
#            match = lambda factory: factory.name() == coord.name()
#            factories = filter(match, self._aux_factories)
#            matches = [factory.derived_dims(self.coord_dims) for factory in factories]

        if not matches:
            raise iris.exceptions.CoordinateNotFoundError(coord.name())

        return matches[0]

    def aux_factory(self, name=None, standard_name=None, long_name=None):
        """
        Returns the single coordinate factory that matches the criteria,
        or raises an error if not found.
        
        Kwargs:

        * name 
            If not None, matches against factory.name().
        * standard_name 
            The CF standard name of the desired coordinate factory.
            If None, does not check for standard name. 
        * long_name 
            An unconstrained description of the coordinate factory.
            If None, does not check for long_name. 

        .. note::

            If the arguments given do not result in precisely 1 coordinate
            factory being matched, an
            :class:`iris.exceptions.CoordinateNotFoundError` is raised.
            
        """
        factories = self.aux_factories

        if name is not None:
            if not isinstance(name, basestring):
                raise ValueError('The name keyword is expecting a string type only. Got %s.' % type(name))
            factories = filter(lambda factory: factory.name() == name, factories)

        if standard_name is not None:
            if not isinstance(standard_name, basestring):
                raise ValueError('The standard_name keyword is expecting a string type only. Got %s.' % type(standard_name))
            factories = filter(lambda factory: factory.standard_name == standard_name, factories)

        if long_name is not None:
            if not isinstance(long_name, basestring):
                raise ValueError('The long_name keyword is expecting a string type only. Got %s.' % type(long_name))
            factories = filter(lambda factory: factory.long_name == long_name, factories)

        if len(factories) > 1:
            msg = 'Expected to find exactly 1 coordinate factory, but found %s. They were: %s.' \
                    % (len(coords), ', '.join(factory.name() for factory in factories))
            raise iris.exceptions.CoordinateNotFoundError(msg)
        elif len(factories) == 0:
            bad_name = name or standard_name or long_name
            msg = 'Expected to find exactly 1 %s coordinate factory, but found none.' % bad_name
            raise iris.exceptions.CoordinateNotFoundError(msg)

        return factories[0]


    def coords(self, name=None, standard_name=None, long_name=None, attributes=None, 
               axis=None, contains_dimension=None, dimensions=None, 
               coord=None, coord_system=None, dim_coords=None): 
        """
        Return a list of coordinates in this cube fitting the given criteria.

        Kwargs:
        
        * name
            The standard name or long name or default name of the desired coordinate. 
            If None, does not check for name. Also see, :attr:`Cube.name`.
        * standard_name 
            The CF standard name of the desired coordinate. If None, does not check for standard name. 
        * long_name 
            An unconstrained description of the coordinate. If None, does not check for long_name. 
        * attributes
            A dictionary of attributes desired on the coordinates. If None, does not check for attributes.
        * axis
            The desired coordinate axis, see :func:`iris.util.guess_coord_axis`. If None, does not check for axis.
            Accepts the values 'X', 'Y', 'Z' and 'T' (case-insensitive).
        * contains_dimension
            The desired coordinate contains the data dimension. If None, does not check for the dimension.
        * dimensions
            The exact data dimensions of the desired coordinate. Coordinates with no data dimension can be found with an
            empty tuple or list (i.e. ``()`` or ``[]``). If None, does not check for dimensions.
        * coord
            Whether the desired coordinates have metadata equal to the given coordinate instance. If None, no check is done.
            Accepts either a :class:'iris.coords.DimCoord`, :class:`iris.coords.AuxCoord` or :class:`iris.coords.CoordDefn`.
        * coord_system
            Whether the desired coordinates have coordinate systems equal to the given coordinate system. If None, no check is done.
        * dim_coords
            Set to True to only return coordinates that are the cube's dimension coordinates. Set to False to only
            return coordinates that are the cube's auxiliary and derived coordinates. If None, returns all coordinates.
        
        See also :meth:`Cube.coord()<iris.cube.Cube.coord>`.

        """
        coords_and_factories = []

        if dim_coords in [True, None]:
            coords_and_factories += list(self.dim_coords)

        if dim_coords in [False, None]:
            coords_and_factories += list(self.aux_coords)
            coords_and_factories += list(self.aux_factories)

        if name is not None:
            coords_and_factories = filter(lambda coord_: coord_.name() == name, coords_and_factories)

        if standard_name is not None:
            coords_and_factories = filter(lambda coord_: coord_.standard_name == standard_name, coords_and_factories)

        if long_name is not None:
            coords_and_factories = filter(lambda coord_: coord_.long_name == long_name, coords_and_factories)

        if axis is not None:
            axis = axis.upper()
            coords_and_factories = filter(lambda coord_: iris.util.guess_coord_axis(coord_) == axis, coords_and_factories)

        if attributes is not None:
            if not isinstance(attributes, collections.Mapping):
                raise ValueError('The attributes keyword was expecting a dictionary type, but got a %s instead.' % type(attributes))
            filter_func = lambda coord_: all(k in coord_.attributes and coord_.attributes[k] == v for k, v in attributes.iteritems())
            coords_and_factories = filter(filter_func, coords_and_factories)

        if coord_system is not None:
            coords_and_factories = filter(lambda coord_: coord_.coord_system == coord_system, coords_and_factories)
        
        if coord is not None:
            if isinstance(coord, iris.coords.CoordDefn):
                defn = coord
            else:
                defn = coord._as_defn()
            coords_and_factories = filter(lambda coord_: coord_._as_defn() == defn, coords_and_factories)
        
        if contains_dimension is not None:
            coords_and_factories = filter(lambda coord_: contains_dimension in self.coord_dims(coord_), coords_and_factories)

        if dimensions is not None:
            if not isinstance(dimensions, collections.Container):
                dimensions = [dimensions]
            coords_and_factories = filter(lambda coord_: tuple(dimensions) == self.coord_dims(coord_), coords_and_factories)

        # If any factories remain after the above filters we have to make the coords so they can be returned
        def extract_coord(coord_or_factory):
            if isinstance(coord_or_factory, iris.aux_factory.AuxCoordFactory):
                coord = coord_or_factory.make_coord(self.coord_dims)
            elif isinstance(coord_or_factory, iris.coords.Coord):
                coord = coord_or_factory
            else:
                raise ValueError('Expected Coord or AuxCoordFactory, got %r.' % type(coord_or_factory))
            return coord
        coords = [extract_coord(coord_or_factory) for coord_or_factory in coords_and_factories]

        return coords
    
    def coord(self, name=None, standard_name=None, long_name=None, attributes=None, 
               axis=None, contains_dimension=None, dimensions=None, 
               coord=None, coord_system=None, dim_coords=None):
        """
        Return a single coord given the same arguments as :meth:`Cube.coords`.
        
        .. note::
            If the arguments given do not result in precisely 1 coordinate being matched,
            an :class:`iris.exceptions.CoordinateNotFoundError` is raised.
            
        .. seealso:: :meth:`Cube.coords()<iris.cube.Cube.coords>` for full keyword documentation.

        """
        coords = self.coords(name=name, standard_name=standard_name, long_name=long_name, attributes=attributes,
                             axis=axis, contains_dimension=contains_dimension, dimensions=dimensions,
                             coord=coord, coord_system=coord_system, dim_coords=dim_coords)

        if len(coords) > 1:
            msg = 'Expected to find exactly 1 coordinate, but found %s. They were: %s.' \
                    % (len(coords), ', '.join(coord.name() for coord in coords))
            raise iris.exceptions.CoordinateNotFoundError(msg)
        elif len(coords) == 0:
            bad_name = name or standard_name or long_name or (coord and coord.name()) or ''
            msg = 'Expected to find exactly 1 %s coordinate, but found none.' % bad_name
            raise iris.exceptions.CoordinateNotFoundError(msg)

        return coords[0]
    
    def coord_system(self, spec):
        """Return the CoordSystem of the given type - or None.

        Args:

        * spec
            The the name or type of a CoordSystem subclass. E.g ::
            
                cube.coord_system("GeogCS")
                cube.coord_system(iris.coord_systems.GeogCS)

        If spec is provided as a type it can be a superclass of any CoordSystems found. 

        """
        # Was a string or a type provided?
        if isinstance(spec, basestring):
            spec_name = spec
        else:
            assert issubclass(spec, iris.coord_systems.CoordSystem), "type %s is not a subclass of CoordSystem" % spec 
            spec_name = spec.__name__

        # Gather a temporary list of our unique CoordSystems.
        coord_systems = ClassDict(iris.coord_systems.CoordSystem) 
        for coord in self.coords():
            if coord.coord_system:
                coord_systems.add(coord.coord_system, replace=True)

        return coord_systems.get(spec_name)

    @property
    def cell_methods(self):
        """Tuple of :class:`iris.coords.CellMethod` representing the processing done on the phenomenon.""" 
        return self._cell_methods

    @cell_methods.setter
    def cell_methods(self, cell_methods):
        self._cell_methods = tuple(cell_methods) if cell_methods else tuple()

    @property
    def shape(self):
        """The shape of the data of this cube."""
        if self._data_manager is None:
            if self._data is None:
                shape = ()
            else:
                shape = self._data.shape
        else:
            shape = self._data_manager.shape(self._data)
        return shape
    
    @property
    def ndim(self):
        """The number of dimensions in the data of this cube."""
        return len(self.shape)    

    @property
    def data(self):
        """
        The :class:`numpy.ndarray` representing the multi-dimensional data of the cube.
        
        .. note::
            Cubes obtained from netCDF, PP, and FieldsFile files will only populate this attribute on
            its first use.

            To obtain the shape of the data without causing it to be loaded, use the Cube.shape attribute.

        Example::
            >>> fname = iris.sample_data_path('air_temp.pp')
            >>> cube = iris.load_cube(fname, 'air_temperature')  # cube.data does not yet have a value.
            >>> print cube.shape                                # cube.data still does not have a value.
            (73, 96)
            >>> cube = cube[:10, :20]                           # cube.data still does not have a value.
            >>> data = cube.data                                # Only now is the data loaded.
            >>> print data.shape
            (10, 20)
        
        """
        # Cache the real data on first use
        if self._data_manager is not None:
            try:
                self._data = self._data_manager.load(self._data)

            except (MemoryError, self._data_manager.ArrayTooBigForAddressSpace), error:
                dm_shape = self._data_manager.pre_slice_array_shape(self._data)
                # if the data manager shape is not the same as the cube's shape, it is because there must be deferred
                # indexing pending once the data has been read into memory. Make the error message for this nice.
                if self.shape != dm_shape:
                    contextual_message = ("The cube's data array shape would have been %r, with the data manager " +
                                 "needing a data shape of %r (before it can reduce to the cube's required size); "
                                 "the data type is %s.\n") % (dm_shape, self.shape, self._data_manager.data_type)
                else:
                    contextual_message = 'The array shape would have been %r and the data type %s.\n' % (self.shape, self._data_manager.data_type)

                if isinstance(error, MemoryError):
                    raise MemoryError(
                      "Failed to create the cube's data as there was not enough memory available.\n" +
                      contextual_message +
                      'Consider freeing up variables or indexing the cube before getting its data.'
                      )
                else:
                    raise ValueError(
                      "Failed to create the cube's data as there is not enough address space to represent the array.\n" +
                      contextual_message +
                      'The cube will need to be reduced in order to load the data.'
                      )

            self._data_manager = None
        return self._data

    @data.setter
    def data(self, value):
        data = numpy.asanyarray(value)

        if self.shape != data.shape:
            # The _ONLY_ data reshape permitted is converting a 0-dimensional array i.e. self.shape == ()
            # into a 1-dimensional array of length one i.e. data.shape == (1,)
            if self.shape or data.shape != (1,):
                raise ValueError('Require cube data with shape %r, got %r.' % (self.shape, data.shape))

        self._data = data
        self._data_manager = None
    
    @property
    def dim_coords(self):
        """Return a tuple of all the dim_coords, ordered by dimension"""
        return tuple((coord for coord, dim in sorted(self._dim_coords_and_dims, key=lambda (coord, dim): (dim, coord.name()))))

    @property
    def aux_coords(self):
        """Return a tuple of all the aux_coords, ordered by dimension(s)"""
        return tuple((coord for coord, dims in sorted(self._aux_coords_and_dims, key=lambda (coord, dims): (dims, coord.name()))))

    @property
    def derived_coords(self):
        """Returns a tuple of all the AuxCoords generated from the coordinate factories."""
        return tuple(factory.make_coord(self.coord_dims) for factory in sorted(self.aux_factories, key=lambda factory: factory.name()))

    @property
    def aux_factories(self):
        """Return a tuple of all the coordinate factories."""
        return tuple(self._aux_factories)

    def _summary_coord_extra(self, coord, indent):
        # Returns the text needed to ensure this coordinate can be distinguished
        # from all others with the same name.
        extra = ''
        similar_coords = self.coords(coord.name())
        if len(similar_coords) > 1:
            # Find all the attribute keys
            keys = set()
            for similar_coord in similar_coords:
                keys.update(similar_coord.attributes.iterkeys())
            # Look for any attributes that vary
            vary = set()
            attributes = {}
            for key in keys:
                for similar_coord in similar_coords:
                    if key not in similar_coord.attributes:
                        vary.add(key)
                        break
                    value = similar_coord.attributes[key]
                    if attributes.setdefault(key, value) != value:
                        vary.add(key)
                        break
            keys = sorted(vary & coord.attributes.viewkeys())
            bits = ['{}={!r}'.format(key, coord.attributes[key]) for key in keys]
            if bits:
                extra = indent + ', '.join(bits)
        return extra

    def _summary_extra(self, coords, summary, indent):
        # Where necessary, inserts extra lines into the summary to ensure
        # coordinates can be distinguished.
        new_summary = []
        for coord, summary in zip(coords, summary):
            new_summary.append(summary)
            extra = self._summary_coord_extra(coord, indent)
            if extra:
                new_summary.append(extra)
        return new_summary

    def summary(self, shorten=False, name_padding=35):
        """
        String summary of the Cube with name, a list of dim coord names versus length 
        and optionally relevant coordinate information.
        
        """
        # Create a set to contain the axis names for each data dimension.
        dim_names = [set() for dim in xrange(len(self.shape))]
        
        # Add the dim_coord names that participate in the associated data dimensions. 
        for dim in xrange(len(self.shape)):
            for coord in self.coords(contains_dimension=dim, dim_coords=True):
                dim_names[dim].add(coord.name())

        # Convert axes sets to lists and sort.
        dim_names = [sorted(names, key=sorted_axes) for names in dim_names]
        
        # Generate textual summary of the cube dimensionality.
        if self.shape == ():
            dimension_header = 'scalar cube'
        else:
            dimension_header = '; '.join([', '.join(dim_names[dim] or ['*ANONYMOUS*']) + 
                                         ': %d' % dim_shape for dim, dim_shape in enumerate(self.shape)])

        cube_header = '%-*s (%s)' % (name_padding, self.name() or 'unknown', dimension_header)
        summary = ''
        
        # Generate full cube textual summary.
        if not shorten:
            indent = 10
            extra_indent = ' ' * 13
            
            # Cache the derived coords so we can rely on consistent
            # object IDs.
            derived_coords = self.derived_coords
            # Determine the cube coordinates that are scalar (single-valued) AND non-dimensioned.
            dim_coords = self.dim_coords
            aux_coords = self.aux_coords
            all_coords = dim_coords + aux_coords + derived_coords
            scalar_coords = [coord for coord in all_coords if not self.coord_dims(coord) and coord.shape == (1,)]
            # Determine the cube coordinates that are not scalar BUT dimensioned.
            scalar_coord_ids = set(map(id, scalar_coords))
            vector_dim_coords = [coord for coord in dim_coords if id(coord) not in scalar_coord_ids]
            vector_aux_coords = [coord for coord in aux_coords if id(coord) not in scalar_coord_ids]
            vector_derived_coords = [coord for coord in derived_coords if id(coord) not in scalar_coord_ids]
            
            # Determine the cube coordinates that don't describe the cube and are most likely erroneous.
            vector_coords = vector_dim_coords + vector_aux_coords + vector_derived_coords
            ok_coord_ids = scalar_coord_ids.union(set(map(id, vector_coords)))
            invalid_coords = [coord for coord in all_coords if id(coord) not in ok_coord_ids]
            
            # Sort scalar coordinates by name.
            scalar_coords.sort(key=lambda coord: coord.name())
            # Sort vector coordinates by data dimension and name.
            vector_dim_coords.sort(key=lambda coord: (self.coord_dims(coord), coord.name()))
            vector_aux_coords.sort(key=lambda coord: (self.coord_dims(coord), coord.name()))
            vector_derived_coords.sort(key=lambda coord: (self.coord_dims(coord), coord.name()))
            # Sort other coordinates by name.
            invalid_coords.sort(key=lambda coord: coord.name())
            
            #
            # Generate textual summary of cube vector coordinates.
            #
            def vector_summary(vector_coords, cube_header, max_line_offset):
                """
                Generates a list of suitably aligned strings containing coord names 
                and dimensions indicated by one or more 'x' symbols.

                .. note:: 
                    The function may need to update the cube header so this is returned with the list of strings. 
                
                """
                vector_summary = []
                if vector_coords:
                    # Identify offsets for each dimension text marker.
                    alignment = numpy.array([index for index, value in enumerate(cube_header) if value == ':'])
                    
                    # Generate basic textual summary for each vector coordinate - WITHOUT dimension markers.
                    for coord in vector_coords:
                        vector_summary.append('%*s%s' % (indent, ' ', iris.util.clip_string(str(coord.name()))))
                        
                    min_alignment = min(alignment)
                    
                    # Determine whether the cube header requires realignment due to
                    # one or more longer vector coordinate summaries.
                    if max_line_offset >= min_alignment:
                        delta = max_line_offset - min_alignment + 5
                        cube_header = '%-*s (%s)' % (int(name_padding + delta), self.name() or 'unknown', dimension_header)
                        alignment += delta
        
                    # Generate full textual summary for each vector coordinate - WITH dimension markers.
                    for index, coord in enumerate(vector_coords):
                        for dim in xrange(len(self.shape)):
                            format = '%*sx' if dim in self.coord_dims(coord) else '%*s-'
                            vector_summary[index] += format % (int(alignment[dim] - len(vector_summary[index])), ' ')

                    # Interleave any extra lines that are needed to distinguish the coordinates.
                    vector_summary = self._summary_extra(vector_coords, vector_summary, extra_indent)

                return vector_summary, cube_header
           
            # Calculate the maximum line offset.
            max_line_offset = 0
            for coord in all_coords:
                max_line_offset = max(max_line_offset, len('%*s%s' % (indent, ' ', iris.util.clip_string(str(coord.name())))))

            if vector_dim_coords:
                dim_coord_summary, cube_header = vector_summary(vector_dim_coords, cube_header, max_line_offset)
                summary += '\n     Dimension coordinates:\n' + '\n'.join(dim_coord_summary)

            if vector_aux_coords:
                aux_coord_summary, cube_header = vector_summary(vector_aux_coords, cube_header, max_line_offset)
                summary += '\n     Auxiliary coordinates:\n' + '\n'.join(aux_coord_summary)

            if vector_derived_coords:
                derived_coord_summary, cube_header = vector_summary(vector_derived_coords, cube_header, max_line_offset)
                summary += '\n     Derived coordinates:\n' + '\n'.join(derived_coord_summary)
                        
            #
            # Generate textual summary of cube scalar coordinates.
            #
            scalar_summary = []

            if scalar_coords:
                for coord in scalar_coords:
                    if coord.units in ['1', 'no_unit', 'unknown']:
                        unit = ''
                    else:
                        unit = ' {!s}'.format(coord.units)

                    # Format cell depending on type of point and whether it
                    # has a bound
                    coord_cell = coord.cell(0)
                    if isinstance(coord_cell.point, basestring):
                        # indent string type coordinates
                        coord_cell_split = [iris.util.clip_string(str(item)) for
                                            item in coord_cell.point.split('\n')]
                        line_sep = '\n{pad:{width}}'.format(pad=' ', width=indent +
                                                            len(coord.name()) + 2)
                        coord_cell_str = line_sep.join(coord_cell_split) + unit
                    else:
                        coord_cell_str = '{!s}{}'.format(coord_cell.point, unit)
                        if coord_cell.bound is not None:
                            bound = '({})'.format(', '.join(str(val) for val in
                                                            coord_cell.bound))
                            coord_cell_str += ', bound={}{}'.format(bound, unit)

                    scalar_summary.append('{pad:{width}}{name}: {cell}'.format(pad=' ',
                                                                               width=indent,
                                                                               name=coord.name(),
                                                                               cell=coord_cell_str))

                # Interleave any extra lines that are needed to distinguish the coordinates.
                scalar_summary = self._summary_extra(scalar_coords, scalar_summary, extra_indent)

                summary += '\n     Scalar coordinates:\n' + '\n'.join(scalar_summary)

            #
            # Generate summary of cube's invalid coordinates.
            #
            if invalid_coords:
                invalid_summary = []
                
                for coord in invalid_coords:
                    invalid_summary.append('%*s%s' % (indent, ' ', coord.name()))
                
                # Interleave any extra lines that are needed to distinguish the coordinates.
                invalid_summary = self._summary_extra(invalid_coords, invalid_summary, extra_indent)

                summary += '\n     Invalid coordinates:\n' + '\n'.join(invalid_summary)

            #
            # Generate summary of cube attributes.
            #
            if self.attributes:
                attribute_summary = []
                for name, value in sorted(self.attributes.iteritems()):
                    if name == 'history':
                        value = re.sub("[\d\/]{8} [\d\:]{8} Iris\: ", '', str(value))
                    else:
                        value = str(value)
                    attribute_summary.append('%*s%s: %s' % (indent, ' ', name, iris.util.clip_string(value)))
                summary += '\n     Attributes:\n' + '\n'.join(attribute_summary)
        
            #
            # Generate summary of cube cell methods
            #
            if self.cell_methods:
                summary += '\n     Cell methods:\n'
                cm_lines = []
                
                for cm in self.cell_methods:
                    cm_lines.append('%*s%s' % (indent, ' ', str(cm)))
                summary += '\n'.join(cm_lines)

        # Construct the final cube summary.
        summary = cube_header + summary

        return summary

    def assert_valid(self):
        """Raise an exception if the cube is invalid; otherwise return None."""
        
        warnings.warn('Cube.assert_valid() has been deprecated.')

    def __str__(self):
        return self.summary()

    def __repr__(self):
        return "<iris 'Cube' of %s>" % self.summary(shorten=True, name_padding=1)

    def __iter__(self):
        # Emit a warning about iterating over the cube.
        # __getitem__ makes this possible, but now deprecated as confusing.
        warnings.warn('Cube iteration has been deprecated: '
                      'please use Cube.slices() instead.')

        # Return a simple first-index iterator, equivalent to that produced
        # with __getitem__, if __iter__ was not defined.
        return (self[i] for i in range(self.shape[0]))

    def __getitem__(self, keys):
        """
        Cube indexing (through use of square bracket notation) has been implemented at the data level. That is,
        the indices provided to this method should be aligned to the data of the cube, and thus the indices requested
        must be applicable directly to the cube.data attribute. All metadata will be subsequently indexed appropriately.

        """ 
        # turn the keys into a full slice spec (all dims) 
        full_slice = iris.util._build_full_slice_given_keys(keys, len(self.shape))
        
        # make indexing on the cube column based by using the column_slices_generator 
        # (potentially requires slicing the data multiple times)
        dimension_mapping, slice_gen = iris.util.column_slices_generator(full_slice, len(self.shape))
        new_coord_dims = lambda coord_: [dimension_mapping[d] for d in self.coord_dims(coord_) if dimension_mapping[d] is not None]

        try:
            first_slice = slice_gen.next()
        except StopIteration:
            first_slice = None
        
        # handle unloaded data
        data_manager = None
        use_data_proxy = self._data_manager is not None

        if first_slice is not None:
            if use_data_proxy:
                data, data_manager = self._data_manager.getitem(self._data, first_slice)
            else:
                data = self.data[first_slice]
        else:
            if use_data_proxy:
                data, data_manager = copy.deepcopy(self._data), copy.deepcopy(self._data_manager)
            else:
                data = copy.deepcopy(self.data)

        for other_slice in slice_gen:
            if use_data_proxy:
                data, data_manager = data_manager.getitem(data, other_slice)
            else:
                data = data[other_slice]
                
        # We don't want a view of the numpy array, so take a copy of it if it's not our own
        # (this applies to proxy "empty data" arrays too)
        if not data.flags['OWNDATA']:
            data = data.copy()
            
        # We can turn a masked array into a normal array if it's full.
        if isinstance(data, numpy.ma.core.MaskedArray):  
            if numpy.ma.count_masked(data) == 0:
                data = data.filled() 

        # Make the new cube slice            
        cube = Cube(data, data_manager=data_manager)
        cube.metadata = copy.deepcopy(self.metadata)

        # Record a mapping from old coordinate IDs to new coordinates,
        # for subsequent use in creating updated aux_factories.
        coord_mapping = {}

        # Slice the coords
        for coord in self.aux_coords:
            coord_keys = tuple([full_slice[dim] for dim in self.coord_dims(coord)])
            try:
                new_coord = coord[coord_keys]
            except ValueError:  # TODO make this exception more specific to catch monotonic error
                # Attempt to slice it by converting to AuxCoord first
                new_coord = iris.coords.AuxCoord.from_coord(coord)[coord_keys]
            cube.add_aux_coord(new_coord, new_coord_dims(coord))
            coord_mapping[id(coord)] = new_coord

        for coord in self.dim_coords:
            coord_keys = tuple([full_slice[dim] for dim in self.coord_dims(coord)])
            new_dims = new_coord_dims(coord)
            # Try/Catch to handle slicing that makes the points/bounds non-monotonic
            try:
                new_coord = coord[coord_keys]
                if not new_dims:
                    # If the associated dimension has been sliced so the coord is a scalar move the 
                    # coord to the aux_coords container
                    cube.add_aux_coord(new_coord, new_dims)
                else:
                    cube.add_dim_coord(new_coord, new_dims)
            except ValueError:  # TODO make this exception more specific to catch monotonic error
                # Attempt to slice it by converting to AuxCoord first
                new_coord = iris.coords.AuxCoord.from_coord(coord)[coord_keys]
                cube.add_aux_coord(new_coord, new_dims)
            coord_mapping[id(coord)] = new_coord

        for factory in self.aux_factories:
            cube.add_aux_factory(factory.updated(coord_mapping))

        return cube

    def subset(self, coord):
        """
        Get a subset of the cube by providing the desired resultant coordinate.

        """
        if not isinstance(coord, iris.coords.Coord):
            raise ValueError('coord_to_extract must be a valid Coord.')

        # Get the coord to extract from the cube
        coord_to_extract = self.coord(coord=coord)
        if len(self.coord_dims(coord_to_extract)) > 1:
            raise iris.exceptions.CoordinateMultiDimError("Currently, only 1D coords can be used to subset a cube")
        # Identify the dimension of the cube which this coordinate references
        coord_to_extract_dim = self.coord_dims(coord_to_extract)[0]

        # Identify the indices which intersect the requested coord and coord_to_extract
        coordinate_indices = coord_to_extract.intersect(coord, return_indices=True)

        # Build up a slice which spans the whole of the cube
        full_slice = [slice(None, None)] * len(self.shape)
        # Update the full slice to only extract specific indices which were identified above
        full_slice[coord_to_extract_dim] = coordinate_indices
        full_slice = tuple(full_slice)
        return self[full_slice]

    def extract(self, constraint):
        """
        Filter the cube by the given constraint using :meth:`iris.Constraint.extract` method.
        """
        # Cast the constraint into a proper constraint if it is not so already
        constraint = iris._constraints.as_constraint(constraint)
        return constraint.extract(self)

    def extract_by_trajectory(self, points):
        """
        Extract a sub-cube at the given n-dimensional points.
        
        .. deprecated::
            Please use :func:`iris.analysis.trajectory.interpolate` instead of this method.
        
        Args:

        * points 
            **Either** Array of dicts with identical keys, for example: 
            ``[ {'latitude':0, 'longitude':0}, {'latitude':1, 'longitude':1} ]``
            **Or** a :class:`iris.analysis.trajectory.Trajectory` instance.

        The coordinates specified by the points will be boiled down into a single data dimension.
        (This is so we can make a cube from points [(0, 0), (1, 0), (1, 1)] for example)

        """
        warnings.warn('Cube.extract_by_trajectory() has been deprecated - please use iris.analysis.trajectory.interpolate().')

        if isinstance(points, iris.analysis.trajectory.Trajectory):
            points = points.sampled_points
        else:
            # Do all points have matching coord names?
            coord_names = points[0].viewkeys()
            for point in points[1:]:
                if point.viewkeys() ^ coord_names:
                    raise ValueError('Point coordinates are inconsistent.')

        # Re-write as a sequence of coordinate-values pairs.
        sample_points = []
        for coord_name in points[0].keys():
            coord = self.coord(coord_name)
            values = [point[coord_name] for point in points]
            sample_points.append((coord, values))

        trajectory_cube = iris.analysis.trajectory.interpolate(self, sample_points)
        return trajectory_cube

    def _as_list_of_coords(self, names_or_coords):
        """
        Convert a name, coord, or list of names/coords to a list of coords.
        """
        # If not iterable, convert to list of a single item
        if not hasattr(names_or_coords, '__iter__'):
            names_or_coords = [names_or_coords]
            
        coords = []
        for name_or_coord in names_or_coords:
            if isinstance(name_or_coord, basestring):
                coords.append(self.coord(name_or_coord))
            elif isinstance(name_or_coord, iris.coords.Coord):
                coords.append(self.coord(coord=name_or_coord))
            else:
                # Don't know how to handle this type
                raise TypeError("Don't know how to handle coordinate of type %s. Ensure all coordinates are of type basestring or iris.coords.Coord." % type(name_or_coord))
        return coords

    def slices(self, coords_to_slice, ordered=True):
        """
        Return an iterator of all cubes given the coordinates desired.

        Parameters:

        * coords_to_slice (string, coord or a list of strings/coords) :
            Coordinate names/coordinates to iterate over. They must all be orthogonal
            (i.e. point to different dimensions).

        Kwargs:

        * ordered: if True, the order which the coords to slice are given will be the order in which
                     they represent the data in the resulting cube slices

        Returns:
            An iterator of sub cubes.

        For example, to get all 2d longitude/latitude cubes from a multi-dimensional cube::

            for sub_cube in cube.slices(['longitude', 'latitude']):
                print sub_cube

        """
        if not isinstance(ordered, bool):
            raise TypeError('Second argument to slices must be boolean. See documentation.')

        # Convert any coordinate names to coordinates
        coords = self._as_list_of_coords(coords_to_slice)

        # Get the union of all describing dimensions for the resulting cube(s)
        requested_dims = set()
        for coord in coords:
            #if coord.ndim > 1:
            #    raise iris.exceptions.CoordinateMultiDimError(coord)
            dims = self.coord_dims(coord)
            # TODO determine if we want to bother catching this as it has no effect
            if not dims:
                raise ValueError("Requested an iterator over a coordinate (%s) which "
                                 'does not describe a dimension.' % coord.name())

            requested_dims.update(dims)

        if len(requested_dims) != sum((len(self.coord_dims(coord)) for coord in coords)):
            raise ValueError('The requested coordinates are not orthogonal.')

        # Create a list with of the shape of our data
        dims_index = list(self.shape)

        # Set the dimensions which have been requested to length 1
        for d in requested_dims:
            dims_index[d] = 1

        return _SliceIterator(self, dims_index, requested_dims, ordered, coords)

    # TODO: This is not used anywhere. Remove.
    @property
    def title(self):
        title = '%s with ' % self.name().replace('_', ' ').capitalize()
        attribute_str_list = []
        for coord in self.coords():
            if coord.shape == (1,):
                cell = coord.cell(0)
                if coord.has_points():
                    attribute_str_list.append('%s: %s' % (coord.name(), cell.point))
                elif coord.has_bounds():
                    attribute_str_list.append('%s: between %s & %s' % (coord.name(), cell.bound[0], cell.bound[1]))

        current_len = len(title)
        for i, line in enumerate(attribute_str_list):
            if (current_len + len(line)) > 90:
                attribute_str_list[i] = '\n' + line
                current_len = len(line)
            else:
                current_len += len(line)

        title = title + ', '.join(attribute_str_list)
        return title

    def transpose(self, new_order=None):
        """
        Re-order the data dimensions of the cube in-place.
        
        new_order - list of ints, optional
                    By default, reverse the dimensions, otherwise permute the axes according
                    to the values given.

        .. note:: If defined, new_order must span all of the data dimensions.
        
        Example usage::
        
            # put the second dimension first, followed by the third dimension, and finally put the first dimension third
            cube.transpose([1, 2, 0])

        """
        if new_order is None:
            new_order = numpy.arange(self.data.ndim)[::-1]
        elif len(new_order) != self.data.ndim:
            raise ValueError('Incorrect number of dimensions.')

        # The data needs to be copied, otherwise this view of the transposed data will not be contiguous.
        # Ensure not to assign via the cube.data setter property since we are reshaping the cube payload in-place.
        self._data = numpy.transpose(self.data, new_order).copy()

        dim_mapping = {src: dest for dest, src in enumerate(new_order)}

        def remap_dim_coord(coord_and_dim):
            coord, dim = coord_and_dim
            return coord, dim_mapping[dim]
        self._dim_coords_and_dims = map(remap_dim_coord, self._dim_coords_and_dims)

        def remap_aux_coord(coord_and_dims):
            coord, dims = coord_and_dims
            return coord, tuple(dim_mapping[dim] for dim in dims)
        self._aux_coords_and_dims = map(remap_aux_coord, self._aux_coords_and_dims)

    def xml(self, checksum=False):
        """
        Returns a fully valid CubeML string representation of the Cube.

        """
        doc = Document()

        cube_xml_element = self._xml_element(doc, checksum=checksum)
        cube_xml_element.setAttribute("xmlns", XML_NAMESPACE_URI)
        doc.appendChild(cube_xml_element)

        # Print our newly created XML
        return doc.toprettyxml(indent="  ")
    
    def _xml_element(self, doc, checksum=False):
        cube_xml_element = doc.createElement("cube")

        if self.standard_name:
            cube_xml_element.setAttribute('standard_name', self.standard_name)
        if self.long_name:
            cube_xml_element.setAttribute('long_name', self.long_name)
        cube_xml_element.setAttribute('units', str(self.units))

        if self.attributes:
            attributes_element = doc.createElement('attributes')
            for name in sorted(self.attributes.iterkeys()):
                attribute_element = doc.createElement('attribute')
                attribute_element.setAttribute('name', name)
                if name == 'history':
                    value = re.sub("[\d\/]{8} [\d\:]{8} Iris\: ", '', str(self.attributes[name]))
                else:
                    value = str(self.attributes[name])
                attribute_element.setAttribute('value', value)
                attributes_element.appendChild(attribute_element)
            cube_xml_element.appendChild(attributes_element)

        coords_xml_element = doc.createElement("coords")
        for coord in sorted(self.coords(), key=lambda coord: coord.name()):
            # make a "cube coordinate" element which holds the dimensions (if appropriate)
            # which itself will have a sub-element of the coordinate instance itself. 
            cube_coord_xml_element = doc.createElement("coord")
            coords_xml_element.appendChild(cube_coord_xml_element)
            
            dims = list(self.coord_dims(coord))
            if dims:
                cube_coord_xml_element.setAttribute("datadims", repr(dims))
            
            coord_xml_element = coord.xml_element(doc)
            cube_coord_xml_element.appendChild(coord_xml_element)
        cube_xml_element.appendChild(coords_xml_element)

        # cell methods (no sorting!)
        cell_methods_xml_element = doc.createElement("cellMethods")
        for cm in self.cell_methods:
            cell_method_xml_element = cm.xml_element(doc)
            cell_methods_xml_element.appendChild(cell_method_xml_element)
        cube_xml_element.appendChild(cell_methods_xml_element)

        if self._data is not None:
            data_xml_element = doc.createElement("data")

            data_xml_element.setAttribute("shape", str(self.shape))

            # Add the datatype
            if self._data_manager is not None:
                data_xml_element.setAttribute("dtype", self._data_manager.data_type.name)
            else:
                data_xml_element.setAttribute("dtype", self._data.dtype.name)

            # getting a checksum triggers any deferred loading
            if checksum:
                data = self.data
                # Ensure data is row-major contiguous for crc32 computation.
                if not data.flags['C_CONTIGUOUS'] and \
                        not data.flags['F_CONTIGUOUS']:
                    data = numpy.ascontiguousarray(data)
                crc = hex(zlib.crc32(data))
                data_xml_element.setAttribute("checksum", crc)
                if isinstance(data, numpy.ma.core.MaskedArray):
                    crc = hex(zlib.crc32(data.mask))
                    data_xml_element.setAttribute("mask_checksum", crc)
            elif self._data_manager is not None:
                data_xml_element.setAttribute("state", "deferred")
            else:
                data_xml_element.setAttribute("state", "loaded")

            cube_xml_element.appendChild(data_xml_element)

        return cube_xml_element

    def copy(self, data=None):
        """
        Returns a deep copy of this cube.

        Kwargs:
            
        * data:
            Replace the data of the cube copy with provided data payload.

        Returns:
            A copy instance of the :class:`Cube`.

        """
        return self._deepcopy({}, data)

    def __copy__(self):
        """Shallow copying is disallowed for Cubes."""
        raise copy.Error("Cube shallow-copy not allowed. Use deepcopy() or Cube.copy()")

    def __deepcopy__(self, memo):
        return self._deepcopy(memo)

    def _deepcopy(self, memo, data=None):
        # TODO FIX this with deferred loading and investiaget data=False,...
        if data is None:
            if self._data is not None and self._data.ndim == 0:
                # Cope with NumPy's asymmetric (aka. "annoying!") behaviour of deepcopy on 0-d arrays.
                new_cube_data = numpy.asanyarray(self._data)
            else:
                new_cube_data = copy.deepcopy(self._data, memo)

            new_cube_data_manager = copy.deepcopy(self._data_manager, memo)
        else:
            data = numpy.asanyarray(data)

            if data.shape != self.shape:
                raise ValueError('Cannot copy cube with new data of a different shape (slice or subset the cube first).')

            new_cube_data = data
            new_cube_data_manager = None

        new_dim_coords_and_dims = copy.deepcopy(self._dim_coords_and_dims, memo)
        new_aux_coords_and_dims = copy.deepcopy(self._aux_coords_and_dims, memo)

        # Record a mapping from old coordinate IDs to new coordinates,
        # for subsequent use in creating updated aux_factories.
        coord_mapping = {}
        for old_pair, new_pair in zip(self._dim_coords_and_dims, new_dim_coords_and_dims):
            coord_mapping[id(old_pair[0])] = new_pair[0]
        for old_pair, new_pair in zip(self._aux_coords_and_dims, new_aux_coords_and_dims):
            coord_mapping[id(old_pair[0])] = new_pair[0]

        new_cube = Cube(new_cube_data,
                        dim_coords_and_dims = new_dim_coords_and_dims,
                        aux_coords_and_dims = new_aux_coords_and_dims,
                        data_manager=new_cube_data_manager
                        )
        new_cube.metadata = copy.deepcopy(self.metadata, memo)

        for factory in self.aux_factories:
            new_cube.add_aux_factory(factory.updated(coord_mapping))
        
        return new_cube

    # START OPERATOR OVERLOADS
    def __eq__(self, other):
        result = NotImplemented
        
        if isinstance(other, Cube):
            result = self.metadata == other.metadata
            
            # having checked the metadata, now check the coordinates
            if result:
                coord_comparison = iris.analysis.coord_comparison(self, other)
                # if there are any coordinates which are not equal
                result = not (bool(coord_comparison['not_equal']) or bool(coord_comparison['non_equal_data_dimension']))
                
            # having checked everything else, check approximate data equality - loading the data if has not already been loaded.
            if result:
                result = numpy.all(numpy.abs(self.data - other.data) < 1e-8)
            
        return result
        
    # Must supply __ne__, Python does not defer to __eq__ for negative equality
    def __ne__(self, other):
        return not self == other         
            
    def __add__(self, other):
        return iris.analysis.maths.add(self, other, ignore=True)
    __radd__ = __add__
    
    def __sub__(self, other):
        return iris.analysis.maths.subtract(self, other, ignore=True)

    __mul__ = iris.analysis.maths.multiply
    __rmul__ = iris.analysis.maths.multiply
    __div__ = iris.analysis.maths.divide
    __truediv__ = iris.analysis.maths.divide
    __pow__ = iris.analysis.maths.exponentiate
    # END OPERATOR OVERLOADS

    def add_history(self, string):
        """
        Add the given string to the cube's history.
        If the history coordinate does not exist, then one will be created.

        """
        string = '%s Iris: %s' % (datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"), string)
 
        try:
            history = self.attributes['history']
            self.attributes['history'] = '%s\n%s' % (history, string)
        except KeyError:
            self.attributes['history'] = string
       

    # START ANALYSIS ROUTINES

    regridded = iris.util._wrap_function_for_method(iris.analysis.interpolate.regrid,
        """
        Returns a new cube with values derived from this cube on the horizontal grid specified
        by the grid_cube.

        """)
    # END ANALYSIS ROUTINES

    def collapsed(self, coords, aggregator, **kwargs):
        """
        Collapse one or more dimensions over the cube given the coordinate/s and an aggregation.

        Args:

        * coords (string, coord or a list of strings/coords) :
            Coordinate names/coordinates over which the cube should be collapsed.

        * aggregator (:class:`iris.analysis.Aggregator`):
            Aggregator to be applied for collapse operation.

        Kwargs:

        * kwargs:
            Aggregation function keyword arguments.

        Returns:
            Collapsed cube.

        For example:

            >>> import iris
            >>> import iris.analysis
            >>> cube = iris.load_cube(iris.sample_data_path('ostia_monthly.nc'))
            >>> new_cube = cube.collapsed('longitude', iris.analysis.MEAN)
            >>> print new_cube
            surface_temperature                 (time: 54; latitude: 18)
                 Dimension coordinates:
                      time                           x             -
                      latitude                       -             x
                 Auxiliary coordinates:
                      forecast_reference_time        x             -
                 Scalar coordinates:
                      forecast_period: 0 hours
                      longitude: 180.0 degrees, bound=(0.0, 360.0) degrees
                 Attributes:
                      Conventions: CF-1.5
                      STASH: m01s00i024
                      history: Mean of surface_temperature aggregated over month, year
            Mean of surface_temperature...
                 Cell methods:
                      mean: month, year
                      mean: longitude


        .. note::
            Some aggregations are not commutative and hence the order of processing is important i.e.::
            
                cube.collapsed('realization', iris.analysis.VARIANCE).collapsed('height', iris.analysis.VARIANCE)
                
            is not necessarily the same result as::
            
                result2 = cube.collapsed('height', iris.analysis.VARIANCE).collapsed('realization', iris.analysis.VARIANCE)
            
            Conversely operations which operate on more than one coordinate at the same time are commutative as they are combined
            internally into a single operation. Hence the order of the coordinates supplied in the list does not matter::
            
                cube.collapsed(['longitude', 'latitude'], iris.analysis.VARIANCE)
            
            is the same (apart from the logically equivalent cell methods that may be created etc.) as::
            
                cube.collapsed(['latitude', 'longitude'], iris.analysis.VARIANCE)

        """
        # Convert any coordinate names to coordinates
        coords = self._as_list_of_coords(coords)

        # Determine the dimensions we need to collapse (and those we don't)
        dimensions_to_collapse = set()
        for coord in coords:
            dimensions_to_collapse.update(self.coord_dims(coord)) 
        
        if not dimensions_to_collapse:
            raise iris.exceptions.CoordinateCollapseError('Cannot collapse a dimension which does not describe any data.')
        
        untouched_dimensions = set(range(self.ndim)) - dimensions_to_collapse
        
        # Remove the collapsed dimension(s) from the metadata
        indices = [slice(None, None)] * self.ndim
        for dim in dimensions_to_collapse:
            indices[dim] = 0
        collapsed_cube = self[tuple(indices)]

        # Collapse any coords that span the dimension(s) being collapsed
        for coord in self.dim_coords + self.aux_coords:
            coord_dims = self.coord_dims(coord)
            if dimensions_to_collapse.intersection(coord_dims):
                local_dims = [coord_dims.index(dim) for dim in dimensions_to_collapse if dim in coord_dims]
                collapsed_cube.replace_coord(coord.collapsed(local_dims))

        # Perform the aggregation over the cube data
        # First reshape the data so that the dimensions being aggregated over are grouped 'at the end'.
        new_shape = [self.shape[dim] for dim in sorted(untouched_dimensions)] + [reduce(operator.mul, (self.shape[dim] for dim in dimensions_to_collapse))]
        unrolled_data = numpy.transpose(self.data, sorted(untouched_dimensions) + sorted(dimensions_to_collapse)).reshape(new_shape)
        # Perform the same operation on the weights if applicable
        if kwargs.get("weights") is not None:
            weights = kwargs["weights"].view() 
            kwargs["weights"] = numpy.transpose(weights, sorted(untouched_dimensions) + sorted(dimensions_to_collapse)).reshape(new_shape)

        data_result = aggregator.aggregate(unrolled_data, axis=-1, **kwargs)
        aggregator.update_metadata(collapsed_cube, coords, axis=-1, **kwargs)
        result = aggregator.post_process(collapsed_cube, data_result, **kwargs)
        return result

    def aggregated_by(self, coords, aggregator, **kwargs):
        """
        Perform aggregation over the cube given one or more "group coordinates".
        
        A "group coordinate" is a coordinate where repeating values represent a single group,
        such as a month coordinate on a daily time slice.
        TODO: It is not clear if repeating values must be consecutive to form a group.

        The group coordinates must all be over the same cube dimension. Each common value group 
        identified over all the group-by coordinates is collapsed using the provided aggregator.

        Args:

        * coords (list of either coord names or :class:`iris.coords.Coord` instances):
            One or more coordinates over which group aggregation is to be performed.
        * aggregator (:class:`iris.analysis.Aggregator`):
            Aggregator to be applied to each group.

        Kwargs:

        * kwargs:
            Aggregator and aggregation function keyword arguments.

        Returns:
            :class:`iris.cube.Cube`.

        For example:

            >>> import iris
            >>> import iris.analysis
            >>> import iris.coord_categorisation as cat
            >>> fname = iris.sample_data_path('ostia_monthly.nc')
            >>> cube = iris.load_cube(fname, 'surface_temperature')
            >>> cat.add_year(cube, 'time', name='year')
            >>> new_cube = cube.aggregated_by('year', iris.analysis.MEAN)
            >>> print new_cube
            surface_temperature                 (*ANONYMOUS*: 5; latitude: 18; longitude: 432)
                 Dimension coordinates:
                      latitude                              -            x              -
                      longitude                             -            -              x
                 Auxiliary coordinates:
                      forecast_reference_time               x            -              -
                      time                                  x            -              -
                      year                                  x            -              -
                 Scalar coordinates:
                      forecast_period: 0 hours
                 Attributes:
                      Conventions: CF-1.5
                      STASH: m01s00i024
                      history: Mean of surface_temperature aggregated over month, year
            Mean of surface_temperature...
                 Cell methods:
                      mean: month, year
                      mean: year

        """
        groupby_coords = []
        dimension_to_groupby = None

        # We can't handle weights
        if isinstance(aggregator, iris.analysis.WeightedAggregator) and aggregator.uses_weighting(**kwargs):
            raise ValueError('Invalid Aggregation, aggergated_by() cannot use weights.')
        
        for coord in sorted(self._as_list_of_coords(coords), key=lambda coord: coord._as_defn()): 
            if coord.ndim > 1:
                raise iris.exceptions.CoordinateMultiDimError('Cannot aggregate_by coord %s as it is multidimensional.' % coord.name())
            dimension = self.coord_dims(coord)
            if not dimension:
                raise iris.exceptions.CoordinateCollapseError('Cannot group-by the coordinate "%s", as '
                                                              'its dimension does not describe any data.' % coord.name())
            if dimension_to_groupby is None:
                dimension_to_groupby = dimension[0]
            if dimension_to_groupby != dimension[0]:
                raise iris.exceptions.CoordinateCollapseError('Cannot group-by coordinates over different dimensions.')
            groupby_coords.append(coord)

        # Determine the other coordinates that share the same group-by coordinate dimension.
        shared_coords = filter(lambda coord_: coord_ not in groupby_coords, self.coords(dimensions=dimension_to_groupby))

        # Create the aggregation group-by instance.
        groupby = iris.analysis._Groupby(groupby_coords, shared_coords)
        
        # Create the resulting aggregate-by cube and remove the original coordinates which are going to be groupedby.
        key = [slice(None, None)] * self.ndim
        key[dimension_to_groupby] = (0,) * len(groupby)
        key = tuple(key)
        aggregateby_cube = self[key]
        for coord in groupby_coords + shared_coords:
            aggregateby_cube.remove_coord(coord) 
        
        # Determine the group-by cube data shape.
        data_shape = list(self.shape)
        data_shape[dimension_to_groupby] = len(groupby)

        # Aggregate the group-by data.
        cube_slice = [slice(None, None)] * len(data_shape)

        for i, groupby_slice in enumerate(groupby.group()):
            # Slice the cube with the group-by slice to create a group-by sub-cube.
            cube_slice[dimension_to_groupby] = groupby_slice
            groupby_sub_cube = self[tuple(cube_slice)]
            # Perform the aggregation over the group-by sub-cube and
            # repatriate the aggregated data into the aggregate-by cube data.
            cube_slice[dimension_to_groupby] = i
            # Determine aggregation result data type for the aggregate-by cube data on first pass.
            if i == 0:
                result = aggregator.aggregate(groupby_sub_cube.data, axis=dimension_to_groupby, **kwargs)
                aggregateby_data = numpy.zeros(data_shape, dtype=result.dtype)
                aggregateby_data[tuple(cube_slice)] = result
            else:
                aggregateby_data[tuple(cube_slice)] = aggregator.aggregate(groupby_sub_cube.data, axis=dimension_to_groupby, **kwargs)

        # Add the aggregation meta data to the aggregate-by cube.
        aggregator.update_metadata(aggregateby_cube, groupby_coords, aggregate=True, **kwargs)
        # Replace the appropriate coordinates within the aggregate-by cube.
        for coord in groupby.coords:
            aggregateby_cube.add_aux_coord(coord.copy(), dimension_to_groupby)
        # Attatch the aggregate-by data into the aggregate-by cube.
        aggregateby_cube.data = aggregateby_data

        return aggregateby_cube

    def rolling_window(self, coord, aggregator, window, **kwargs):
        """
        Perform rolling window aggregation on a cube given a coordinate, an 
        aggregation method and a window size.

        Args:

        * coord (string/:class:`iris.coords.Coord`):
            The coordinate over which to perform the rolling window aggregation.
        * aggregator (:class:`iris.analysis.Aggregator`):
            Aggregator to be applied to the data.
        * window (int):
            Size of window to use.

        Kwargs:

        * kwargs:
            Aggregator and aggregation function keyword arguments. The weights
            argument to the aggregator, if any, should be a 1d array with the
            same length as the chosen window.

        Returns:
            :class:`iris.cube.Cube`.

        For example:

            >>> import iris, iris.analysis
            >>> fname = iris.sample_data_path('GloSea4', 'ensemble_010.pp')
            >>> air_press = iris.load_cube(fname, 'surface_temperature')
            >>> print air_press
            surface_temperature                 (time: 6; latitude: 145; longitude: 192)
                 Dimension coordinates:
                      time                           x            -               -
                      latitude                       -            x               -
                      longitude                      -            -               x
                 Auxiliary coordinates:
                      forecast_period                x            -               -
                 Scalar coordinates:
                      forecast_reference_time: 364272.0 hours since 1970-01-01 00:00:00
                      realization: 10
                 Attributes:
                      STASH: m01s00i024
                      source: Data from Met Office Unified Model 7.06
                 Cell methods:
                      mean: time (1 hour)


            >>> print air_press.rolling_window('time', iris.analysis.MEAN, 3)
            surface_temperature                 (time: 4; latitude: 145; longitude: 192)
                 Dimension coordinates:
                      time                           x            -               -
                      latitude                       -            x               -
                      longitude                      -            -               x
                 Auxiliary coordinates:
                      forecast_period                x            -               -
                 Scalar coordinates:
                      forecast_reference_time: 364272.0 hours since 1970-01-01 00:00:00
                      realization: 10
                 Attributes:
                      STASH: m01s00i024
                      history: Mean of surface_temperature with a rolling window of length 3 over tim...
                      source: Data from Met Office Unified Model 7.06
                 Cell methods:
                      mean: time (1 hour)
                      mean: time


            Notice that the forecast_period dimension now represents the 4
            possible windows of size 3 from the original cube. 

        """
        coord = self._as_list_of_coords(coord)[0]

        if getattr(coord, 'circular', False):
            raise iris.exceptions.NotYetImplementedError(
                'Rolling window over a circular coordinate.')

        if window < 2:
            raise ValueError('Cannot perform rolling window '
                             'with a window size less than 2.')

        if coord.ndim > 1:
            raise iris.exceptions.CoordinateMultiDimError(coord)

        dimension = self.coord_dims(coord)
        if len(dimension) != 1:
            raise iris.exceptions.CoordinateCollapseError(
                'Cannot perform rolling window with coordinate "%s", '
                'must map to one data dimension.'  % coord.name())
        dimension = dimension[0]

        # Use indexing to get a result-cube of the correct shape.
        # NB. This indexes the data array which is wasted work.
        # As index-to-get-shape-then-fiddle is a common pattern, perhaps
        # some sort of `cube.prepare()` method would be handy to allow
        # re-shaping with given data, and returning a mapping of
        # old-to-new-coords (to avoid having to use metadata identity)?
        key = [slice(None, None)] * self.ndim
        key[dimension] = slice(None, self.shape[dimension] - window + 1)
        new_cube = self[tuple(key)]

        # take a view of the original data using the rolling_window function
        # this will add an extra dimension to the data at dimension + 1 which
        # represents the rolled window (i.e. will have a length of window)
        rolling_window_data = iris.util.rolling_window(self.data,
                                                       window=window,
                                                       axis=dimension)

        # now update all of the coordinates to reflect the aggregation
        for coord_ in self.coords(dimensions=dimension):
            if coord_.has_bounds():
                warnings.warn('The bounds of coordinate %r were ignored in '
                              'the rolling window operation.' % coord_.name())

            if coord_.ndim != 1:
                raise ValueError('Cannot calculate the rolling '
                                 'window of %s as it is a multidimensional '
                                 'coordinate.' % coord_.name())

            new_bounds = iris.util.rolling_window(coord_.points, window)

            # Take the first and last element of the rolled window (i.e. the bounds)
            new_bounds = new_bounds[:, (0, -1)]
            new_points = numpy.mean(new_bounds, axis=-1)

            # wipe the coords points and set the bounds
            new_coord = new_cube.coord(coord=coord_)
            new_coord.points = new_points
            new_coord.bounds = new_bounds

        # update the metadata of the cube itself
        aggregator.update_metadata(
            new_cube, [coord],
            action='with a rolling window of length %s over' % window,
            **kwargs)
        # and perform the data transformation, generating weights first if
        # needed
        newkwargs = {}
        if isinstance(aggregator, iris.analysis.WeightedAggregator) and \
                aggregator.uses_weighting(**kwargs):
            if 'weights' in kwargs.keys():
                weights = kwargs['weights']
                if weights.ndim > 1 or weights.shape[0] != window:
                    raise ValueError('Weights for rolling window aggregation '
                                     'must be a 1d array with the same length '
                                     'as the window.')
                newkwargs['weights'] = iris.util.broadcast_weights(
                    weights, rolling_window_data, [dimension+1])
        new_cube.data = aggregator.aggregate(rolling_window_data,
                                            axis=dimension + 1,
                                            **newkwargs)

        return new_cube
    

class ClassDict(object, UserDict.DictMixin):
    """
    A mapping that stores objects keyed on their superclasses and their names.

    The mapping has a root class, all stored objects must be a subclass of the root class.
    The superclasses used for an object include the class of the object, but do not include the root class.
    Only one object is allowed for any key.

    """
    def __init__(self, superclass):
        if not isinstance(superclass, type):
            raise TypeError("The superclass must be a Python type or new style class.")
        self._superclass = superclass
        self._basic_map = {}
        self._retrieval_map = {}

    def add(self, object_, replace=False):
        '''Add an object to the dictionary.'''
        if not isinstance(object_, self._superclass):
            raise TypeError("Only subclasses of '%s' are allowed as values." % self._superclass.__name__)
        # Find all the superclasses of the given object, starting with the object's class.
        superclasses = type.mro(type(object_))
        if not replace:
            # Ensure nothing else is already registered against those superclasses.
            # NB. This implies the _basic_map will also be empty for this object.
            for key_class in superclasses:
                if key_class in self._retrieval_map:
                    raise ValueError("Cannot add instance of '%s' because instance of '%s' already added." % (
                            type(object_).__name__, key_class.__name__))
        # Register the given object against those superclasses.
        for key_class in superclasses:
            self._retrieval_map[key_class] = object_
            self._retrieval_map[key_class.__name__] = object_
        self._basic_map[type(object_)] = object_

    def __getitem__(self, class_):
        try:
            return self._retrieval_map[class_]
        except KeyError:
            raise KeyError('Coordinate system %r does not exist.' % class_)

    def __delitem__(self, class_):
        cs = self[class_]
        keys = [k for k, v in self._retrieval_map.iteritems() if v == cs]
        for key in keys:
            del self._retrieval_map[key]
        del self._basic_map[type(cs)]
        return cs

    def keys(self):
        '''Return the keys of the dictionary mapping.'''
        return self._basic_map.keys()


def sorted_axes(axes):
    """Returns the axis names sorted alphabetically, with the exception that
    't', 'z', 'y', and, 'x' are sorted to the end."""
    return sorted(axes, key=lambda name: ({'x':4, 'y':3, 'z':2, 't':1}.get(name, 0), name))


# See Cube.slice() for the definition/context.
class _SliceIterator(collections.Iterator):
    def __init__(self, cube, dims_index, requested_dims, ordered, coords):
        self._cube = cube

        # Let Numpy do some work in providing all of the permutations of our data shape.
        # This functionality is something like:
        # ndindex(2, 1, 3) -> [(0, 0, 0), (0, 0, 1), (0, 0, 2), (1, 0, 0), (1, 0, 1), (1, 0, 2)]
        self._ndindex = numpy.ndindex(*dims_index)

        self._requested_dims = requested_dims
        self._ordered = ordered
        self._coords = coords

    def next(self):
        # NB. When self._ndindex runs out it will raise StopIteration for us.
        index_tuple = self._ndindex.next()

        # Turn the given tuple into a list so that we can do something with it
        index_list = list(index_tuple)

        # For each of the spanning dimensions requested, replace the 0 with a spanning slice
        for d in self._requested_dims:
            index_list[d] = slice(None, None)

        # Request the slice
        cube = self._cube[tuple(index_list)]

        if self._ordered:
            transpose_order = []
            for coord in self._coords:
                transpose_order += sorted(cube.coord_dims(coord))
            if transpose_order != range(len(cube.shape)):
                cube.transpose(transpose_order)

        return cube
