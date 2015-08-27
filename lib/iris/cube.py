# (C) British Crown Copyright 2010 - 2015, Met Office
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

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

from xml.dom.minidom import Document
import collections
import copy
import datetime
import operator
import warnings
import zlib

import biggus
import numpy as np
import numpy.ma as ma

import iris.analysis
from iris.analysis.cartography import wrap_lons
import iris.analysis.maths
import iris.analysis.interpolate
import iris.aux_factory
import iris.coord_systems
import iris.coords
import iris._concatenate
import iris._constraints
import iris._merge
import iris.exceptions
import iris.util

from iris._cube_coord_common import CFVariableMixin
from functools import reduce


__all__ = ['Cube', 'CubeList', 'CubeMetadata']


class CubeMetadata(collections.namedtuple('CubeMetadata',
                                          ['standard_name',
                                           'long_name',
                                           'var_name',
                                           'units',
                                           'attributes',
                                           'cell_measures',
                                           'cell_methods'])):
    """
    Represents the phenomenon metadata for a single :class:`Cube`.

    """
    def name(self, default='unknown'):
        """
        Returns a human-readable name.

        First it tries self.standard_name, then it tries the 'long_name'
        attribute, then the 'var_name' attribute, before falling back to
        the value of `default` (which itself defaults to 'unknown').

        """
        return self.standard_name or self.long_name or self.var_name or default


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
        return _CubeFilterCollection([pair.merged(unique) for pair in
                                      self.pairs])


class CubeList(list):
    """
    All the functionality of a standard :class:`list` with added "Cube"
    context.

    """

    def __new__(cls, list_of_cubes=None):
        """Given a :class:`list` of cubes, return a CubeList instance."""
        cube_list = list.__new__(cls, list_of_cubes)

        # Check that all items in the incoming list are cubes. Note that this
        # checking does not guarantee that a CubeList instance *always* has
        # just cubes in its list as the append & __getitem__ methods have not
        # been overridden.
        if not all([isinstance(cube, Cube) for cube in cube_list]):
            raise ValueError('All items in list_of_cubes must be Cube '
                             'instances.')
        return cube_list

    def __str__(self):
        """Runs short :meth:`Cube.summary` on every cube."""
        result = ['%s: %s' % (i, cube.summary(shorten=True)) for i, cube in
                  enumerate(self)]
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

    def __getitem__(self, keys):
        """x.__getitem__(y) <==> x[y]"""
        result = super(CubeList, self).__getitem__(keys)
        if isinstance(result, list):
            result = CubeList(result)
        return result

    def __getslice__(self, start, stop):
        """
        x.__getslice__(i, j) <==> x[i:j]

        Use of negative indices is not supported.

        """
        result = super(CubeList, self).__getslice__(start, stop)
        result = CubeList(result)
        return result

    def xml(self, checksum=False, order=True, byteorder=True):
        """Return a string of the XML that this list of cubes represents."""
        doc = Document()
        cubes_xml_element = doc.createElement("cubes")
        cubes_xml_element.setAttribute("xmlns", XML_NAMESPACE_URI)

        for cube_obj in self:
            cubes_xml_element.appendChild(
                cube_obj._xml_element(
                    doc, checksum=checksum, order=order, byteorder=byteorder))

        doc.appendChild(cubes_xml_element)

        # return our newly created XML string
        return doc.toprettyxml(indent="  ")

    def extract(self, constraints, strict=False):
        """
        Filter each of the cubes which can be filtered by the given
        constraints.

        This method iterates over each constraint given, and subsets each of
        the cubes in this CubeList where possible. Thus, a CubeList of length
        **n** when filtered with **m** constraints can generate a maximum of
        **m * n** cubes.

        Keywords:

        * strict - boolean
            If strict is True, then there must be exactly one cube which is
            filtered per constraint.

        """
        return self._extract_and_merge(self, constraints, strict,
                                       merge_unique=None)

    @staticmethod
    def _extract_and_merge(cubes, constraints, strict, merge_unique=False):
        # * merge_unique - if None: no merging, if false: non unique merging,
        # else unique merging (see merge)

        constraints = iris._constraints.list_of_constraints(constraints)

        # group the resultant cubes by constraints in a dictionary
        constraint_groups = dict([(constraint, CubeList()) for constraint in
                                 constraints])
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
                msg = 'Got %s cubes for constraint %r, ' \
                      'expecting 1.' % (len(constraint_cubes), constraint)
                raise iris.exceptions.ConstraintMismatchError(msg)
            result.extend(constraint_cubes)

        if strict and len(constraints) == 1:
            result = result[0]

        return result

    def extract_strict(self, constraints):
        """
        Calls :meth:`CubeList.extract` with the strict keyword set to True.

        """
        return self.extract(constraints, strict=True)

    def extract_overlapping(self, coord_names):
        """
        Returns a :class:`CubeList` of cubes extracted over regions
        where the coordinates overlap, for the coordinates
        in coord_names.

        Args:

        * coord_names:
           A string or list of strings of the names of the coordinates
           over which to perform the extraction.

        """
        if isinstance(coord_names, basestring):
            coord_names = [coord_names]

        def make_overlap_fn(coord_name):
            def overlap_fn(cell):
                return all(cell in cube.coord(coord_name).cells()
                           for cube in self)
            return overlap_fn

        coord_values = {coord_name: make_overlap_fn(coord_name)
                        for coord_name in coord_names}

        return self.extract(iris.Constraint(coord_values=coord_values))

    def merge_cube(self):
        """
        Return the merged contents of the :class:`CubeList` as a single
        :class:`Cube`.

        If it is not possible to merge the `CubeList` into a single
        `Cube`, a :class:`~iris.exceptions.MergeError` will be raised
        describing the reason for the failure.

        For example:

            >>> cube_1 = iris.cube.Cube([1, 2])
            >>> cube_1.add_aux_coord(iris.coords.AuxCoord(0, long_name='x'))
            >>> cube_2 = iris.cube.Cube([3, 4])
            >>> cube_2.add_aux_coord(iris.coords.AuxCoord(1, long_name='x'))
            >>> cube_2.add_dim_coord(
            ...     iris.coords.DimCoord([0, 1], long_name='z'), 0)
            >>> single_cube = iris.cube.CubeList([cube_1, cube_2]).merge_cube()
            Traceback (most recent call last):
            ...
            iris.exceptions.MergeError: failed to merge into a single cube.
              Coordinates in cube.dim_coords differ: z.
              Coordinate-to-dimension mapping differs for cube.dim_coords.

        """
        if not self:
            raise ValueError("can't merge an empty CubeList")

        # Register each of our cubes with a single ProtoCube.
        proto_cube = iris._merge.ProtoCube(self[0])
        for cube in self[1:]:
            proto_cube.register(cube, error_on_mismatch=True)

        # Extract the merged cube from the ProtoCube.
        merged_cube, = proto_cube.merge()
        return merged_cube

    def merge(self, unique=True):
        """
        Returns the :class:`CubeList` resulting from merging this
        :class:`CubeList`.

        Kwargs:

        * unique:
            If True, raises `iris.exceptions.DuplicateDataError` if
            duplicate cubes are detected.

        This combines cubes with different values of an auxiliary scalar
        coordinate, by constructing a new dimension.

        .. testsetup::

            import iris
            c1 = iris.cube.Cube([0,1,2], long_name='some_parameter')
            xco = iris.coords.DimCoord([11, 12, 13], long_name='x_vals')
            c1.add_dim_coord(xco, 0)
            c1.add_aux_coord(iris.coords.AuxCoord([100], long_name='y_vals'))
            c2 = c1.copy()
            c2.coord('y_vals').points = [200]

        For example::

            >>> print(c1)
            some_parameter / (unknown)          (x_vals: 3)
                 Dimension coordinates:
                      x_vals                           x
                 Scalar coordinates:
                      y_vals: 100
            >>> print(c2)
            some_parameter / (unknown)          (x_vals: 3)
                 Dimension coordinates:
                      x_vals                           x
                 Scalar coordinates:
                      y_vals: 200
            >>> cube_list = iris.cube.CubeList([c1, c2])
            >>> new_cube = cube_list.merge()[0]
            >>> print(new_cube)
            some_parameter / (unknown)          (y_vals: 2; x_vals: 3)
                 Dimension coordinates:
                      y_vals                           x          -
                      x_vals                           -          x
            >>> print(new_cube.coord('y_vals').points)
            [100 200]
            >>>

        Contrast this with :meth:`iris.cube.CubeList.concatenate`, which joins
        cubes along an existing dimension.

        .. note::

            If time coordinates in the list of cubes have differing epochs then
            the cubes will not be able to be merged. If this occurs, use
            :func:`iris.util.unify_time_units` to normalise the epochs of the
            time coordinates so that the cubes can be merged.

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

    def concatenate_cube(self):
        """
        Return the concatenated contents of the :class:`CubeList` as a single
        :class:`Cube`.

        If it is not possible to concatenate the `CubeList` into a single
        `Cube`, a :class:`~iris.exceptions.ConcatenateError` will be raised
        describing the reason for the failure.

        """
        if not self:
            raise ValueError("can't concatenate an empty CubeList")

        names = [cube.metadata.name() for cube in self]
        unique_names = list(collections.OrderedDict.fromkeys(names))
        if len(unique_names) == 1:
            res = iris._concatenate.concatenate(self, error_on_mismatch=True)
            n_res_cubes = len(res)
            if n_res_cubes == 1:
                return res[0]
            else:
                msgs = []
                msgs.append('An unexpected problem prevented concatenation.')
                msgs.append('Expected only a single cube, '
                            'found {}.'.format(n_res_cubes))
                raise iris.exceptions.ConcatenateError(msgs)
        else:
            msgs = []
            msgs.append('Cube names differ: {} != {}'.format(names[0],
                                                             names[1]))
            raise iris.exceptions.ConcatenateError(msgs)

    def concatenate(self):
        """
        Concatenate the cubes over their common dimensions.

        Returns:
            A new :class:`iris.cube.CubeList` of concatenated
            :class:`iris.cube.Cube` instances.

        This combines cubes with a common dimension coordinate, but occupying
        different regions of the coordinate value.  The cubes are joined across
        that dimension.

        .. testsetup::

            import iris
            import numpy as np
            xco = iris.coords.DimCoord([11, 12, 13, 14], long_name='x_vals')
            yco1 = iris.coords.DimCoord([4, 5], long_name='y_vals')
            yco2 = iris.coords.DimCoord([7, 9, 10], long_name='y_vals')
            c1 = iris.cube.Cube(np.zeros((2,4)), long_name='some_parameter')
            c1.add_dim_coord(xco, 1)
            c1.add_dim_coord(yco1, 0)
            c2 = iris.cube.Cube(np.zeros((3,4)), long_name='some_parameter')
            c2.add_dim_coord(xco, 1)
            c2.add_dim_coord(yco2, 0)

        For example::

            >>> print(c1)
            some_parameter / (unknown)          (y_vals: 2; x_vals: 4)
                 Dimension coordinates:
                      y_vals                           x          -
                      x_vals                           -          x
            >>> print(c1.coord('y_vals').points)
            [4 5]
            >>> print(c2)
            some_parameter / (unknown)          (y_vals: 3; x_vals: 4)
                 Dimension coordinates:
                      y_vals                           x          -
                      x_vals                           -          x
            >>> print(c2.coord('y_vals').points)
            [ 7  9 10]
            >>> cube_list = iris.cube.CubeList([c1, c2])
            >>> new_cube = cube_list.concatenate()[0]
            >>> print(new_cube)
            some_parameter / (unknown)          (y_vals: 5; x_vals: 4)
                 Dimension coordinates:
                      y_vals                           x          -
                      x_vals                           -          x
            >>> print(new_cube.coord('y_vals').points)
            [ 4  5  7  9 10]
            >>>

        Contrast this with :meth:`iris.cube.CubeList.merge`, which makes a new
        dimension from values of an auxiliary scalar coordinate.

        .. note::

            If time coordinates in the list of cubes have differing epochs then
            the cubes will not be able to be concatenated. If this occurs, use
            :func:`iris.util.unify_time_units` to normalise the epochs of the
            time coordinates so that the cubes can be concatenated.

        .. warning::

            This routine will load your data payload!

        """
        return iris._concatenate.concatenate(self)


class Cube(CFVariableMixin):
    """
    A single Iris cube of data and metadata.

    Typically obtained from :func:`iris.load`, :func:`iris.load_cube`,
    :func:`iris.load_cubes`, or from the manipulation of existing cubes.

    For example:

        >>> cube = iris.load_cube(iris.sample_data_path('air_temp.pp'))
        >>> print(cube)
        air_temperature / (K)               (latitude: 73; longitude: 96)
             Dimension coordinates:
                  latitude                           x              -
                  longitude                          -              x
             Scalar coordinates:
                  forecast_period: 6477 hours, bound=(-28083.0, 6477.0) hours
                  forecast_reference_time: 1998-03-01 03:00:00
                  pressure: 1000.0 hPa
                  time: 1998-12-01 00:00:00, \
bound=(1994-12-01 00:00:00, 1998-12-01 00:00:00)
             Attributes:
                  STASH: m01s16i203
                  source: Data from Met Office Unified Model
             Cell methods:
                  mean within years: time
                  mean over years: time


    See the :doc:`user guide</userguide/index>` for more information.

    """

    #: Indicates to client code that the object supports
    #: "orthogonal indexing", which means that slices that are 1d arrays
    #: or lists slice along each dimension independently. This behavior
    #: is similar to Fortran or Matlab, but different than numpy.
    __orthogonal_indexing__ = True

    def __init__(self, data, standard_name=None, long_name=None,
                 var_name=None, units=None, attributes=None,
                 cell_measures=None, cell_methods=None,
                 dim_coords_and_dims=None, aux_coords_and_dims=None,
                 aux_factories=None):
        """
        Creates a cube with data and optional metadata.

        Not typically used - normally cubes are obtained by loading data
        (e.g. :func:`iris.load`) or from manipulating existing cubes.

        Args:

        * data
            This object defines the shape of the cube and the phenomenon
            value in each cell.

            It can be a biggus array, a numpy array, a numpy array
            subclass (such as :class:`numpy.ma.MaskedArray`), or an
            *array_like* as described in :func:`numpy.asarray`.

            See :attr:`Cube.data<iris.cube.Cube.data>`.

        Kwargs:

        * standard_name
            The standard name for the Cube's data.
        * long_name
            An unconstrained description of the cube.
        * var_name
            The CF variable name for the cube.
        * units
            The unit of the cube, e.g. ``"m s-1"`` or ``"kelvin"``.
        * attributes
            A dictionary of cube attributes
        * cell_methods
            A tuple of CellMethod objects, generally set by Iris, e.g.
            ``(CellMethod("mean", coords='latitude'), )``.
        * dim_coords_and_dims
            A list of coordinates with scalar dimension mappings, e.g
            ``[(lat_coord, 0), (lon_coord, 1)]``.
        * aux_coords_and_dims
            A list of coordinates with dimension mappings,
            e.g ``[(lat_coord, 0), (lon_coord, (0, 1))]``.
            See also :meth:`Cube.add_dim_coord()<iris.cube.Cube.add_dim_coord>`
            and :meth:`Cube.add_aux_coord()<iris.cube.Cube.add_aux_coord>`.
        * aux_factories
            A list of auxiliary coordinate factories. See
            :mod:`iris.aux_factory`.

        For example::
            >>> from iris.coords import DimCoord
            >>> from iris.cube import Cube
            >>> latitude = DimCoord(np.linspace(-90, 90, 4),
            ...                     standard_name='latitude',
            ...                     units='degrees')
            >>> longitude = DimCoord(np.linspace(45, 360, 8),
            ...                      standard_name='longitude',
            ...                      units='degrees')
            >>> cube = Cube(np.zeros((4, 8), np.float32),
            ...             dim_coords_and_dims=[(latitude, 0),
            ...                                  (longitude, 1)])

        """
        # Temporary error while we transition the API.
        if isinstance(data, basestring):
            raise TypeError('Invalid data type: {!r}.'.format(data))

        if not isinstance(data, (biggus.Array, ma.MaskedArray)):
            data = np.asarray(data)
        self._my_data = data

        #: The "standard name" for the Cube's phenomenon.
        self.standard_name = standard_name

        #: An instance of :class:`iris.unit.Unit` describing the Cube's data.
        self.units = units

        #: The "long name" for the Cube's phenomenon.
        self.long_name = long_name

        #: The CF variable name for the Cube.
        self.var_name = var_name

        self.cell_methods = cell_methods

        #: A dictionary, with a few restricted keys, for arbitrary
        #: Cube metadata.
        self.attributes = attributes

        # Coords
        self._dim_coords_and_dims = []
        self._aux_coords_and_dims = []
        self._aux_factories = []

        identities = set()
        if dim_coords_and_dims:
            dims = set()
            for coord, dim in dim_coords_and_dims:
                identity = coord.standard_name, coord.long_name
                if identity not in identities and dim not in dims:
                    self._add_unique_dim_coord(coord, dim)
                else:
                    self.add_dim_coord(coord, dim)
                identities.add(identity)
                dims.add(dim)

        if aux_coords_and_dims:
            for coord, dims in aux_coords_and_dims:
                identity = coord.standard_name, coord.long_name
                if identity not in identities:
                    self._add_unique_aux_coord(coord, dims)
                else:
                    self.add_aux_coord(coord, dims)
                identities.add(identity)

        if aux_factories:
            for factory in aux_factories:
                self.add_aux_factory(factory)
                
        self.cell_measures = cell_measures

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
        return CubeMetadata(self.standard_name, self.long_name, self.var_name,
                            self.units, self.attributes, self.cell_measures,
                            self.cell_methods)

    @metadata.setter
    def metadata(self, value):
        try:
            value = CubeMetadata(**value)
        except TypeError:
            try:
                value = CubeMetadata(*value)
            except TypeError:
                missing_attrs = [field for field in CubeMetadata._fields
                                 if not hasattr(value, field)]
                if missing_attrs:
                    raise TypeError('Invalid/incomplete metadata')
        for name in CubeMetadata._fields:
            setattr(self, name, getattr(value, name))

    def is_compatible(self, other, ignore=None):
        """
        Return whether the cube is compatible with another.

        Compatibility is determined by comparing :meth:`iris.cube.Cube.name()`,
        :attr:`iris.cube.Cube.units`, :attr:`iris.cube.Cube.cell_methods` and
        :attr:`iris.cube.Cube.attributes` that are present in both objects.

        Args:

        * other:
            An instance of :class:`iris.cube.Cube` or
            :class:`iris.cube.CubeMetadata`.
        * ignore:
           A single attribute key or iterable of attribute keys to ignore when
           comparing the cubes. Default is None. To ignore all attributes set
           this to other.attributes.

        Returns:
           Boolean.

        .. seealso::

            :meth:`iris.util.describe_diff()`

        .. note::

            This function does not indicate whether the two cubes can be
            merged, instead it checks only the four items quoted above for
            equality. Determining whether two cubes will merge requires
            additional logic that is beyond the scope of this method.

        """
        compatible = (self.name() == other.name() and
                      self.units == other.units and
                      self.cell_methods == other.cell_methods)

        if compatible:
            common_keys = set(self.attributes).intersection(other.attributes)
            if ignore is not None:
                if isinstance(ignore, basestring):
                    ignore = (ignore,)
                common_keys = common_keys.difference(ignore)
            for key in common_keys:
                if np.any(self.attributes[key] != other.attributes[key]):
                    compatible = False
                    break

        return compatible

    def convert_units(self, unit):
        """
        Change the cube's units, converting the values in the data array.

        For example, if a cube's :attr:`~iris.cube.Cube.units` are
        kelvin then::

            cube.convert_units('celsius')

        will change the cube's :attr:`~iris.cube.Cube.units` attribute to
        celsius and subtract 273.15 from each value in
        :attr:`~iris.cube.Cube.data`.

        .. warning::
            Calling this method will trigger any deferred loading, causing
            the cube's data array to be loaded into memory.

        """
        # If the cube has units convert the data.
        if not self.units.is_unknown():
            self.data = self.units.convert(self.data, unit)
        self.units = unit

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
            Integer or iterable of integers giving the data dimensions spanned
            by the coordinate.

        Raises a ValueError if a coordinate with identical metadata already
        exists on the cube.

        See also :meth:`Cube.remove_coord()<iris.cube.Cube.remove_coord>`.

        """
        if self.coords(coord):  # TODO: just fail on duplicate object
            raise ValueError('Duplicate coordinates are not permitted.')
        self._add_unique_aux_coord(coord, data_dims)

    def _add_unique_aux_coord(self, coord, data_dims):
        # Convert to a tuple of integers
        if data_dims is None:
            data_dims = tuple()
        elif isinstance(data_dims, collections.Container):
            data_dims = tuple(int(d) for d in data_dims)
        else:
            data_dims = (int(data_dims),)

        if data_dims:
            if len(data_dims) != coord.ndim:
                msg = 'Invalid data dimensions: {} given, {} expected for ' \
                      '{!r}.'.format(len(data_dims), coord.ndim, coord.name())
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
            raise ValueError('Missing data dimensions for multi-valued'
                             ' coordinate {!r}'.format(coord.name()))

        self._aux_coords_and_dims.append([coord, data_dims])

    def add_aux_factory(self, aux_factory):
        """
        Adds an auxiliary coordinate factory to the cube.

        Args:

        * aux_factory
            The :class:`iris.aux_factory.AuxCoordFactory` instance to add.

        """
        if not isinstance(aux_factory, iris.aux_factory.AuxCoordFactory):
            raise TypeError('Factory must be a subclass of '
                            'iris.aux_factory.AuxCoordFactory.')
        self._aux_factories.append(aux_factory)

    def add_dim_coord(self, dim_coord, data_dim):
        """
        Add a CF coordinate to the cube.

        Args:

        * dim_coord
            The :class:`iris.coords.DimCoord` instance to add to the cube.
        * data_dim
            Integer giving the data dimension spanned by the coordinate.

        Raises a ValueError if a coordinate with identical metadata already
        exists on the cube or if a coord already exists for the
        given dimension.

        See also :meth:`Cube.remove_coord()<iris.cube.Cube.remove_coord>`.

        """
        if self.coords(dim_coord):
            raise ValueError('The coordinate already exists on the cube. '
                             'Duplicate coordinates are not permitted.')
        # Check dimension is available
        if self.coords(dimensions=data_dim, dim_coords=True):
            raise ValueError('A dim_coord is already associated with '
                             'dimension %d.' % data_dim)
        self._add_unique_dim_coord(dim_coord, data_dim)

    def _add_unique_dim_coord(self, dim_coord, data_dim):
        if isinstance(dim_coord, iris.coords.AuxCoord):
            raise ValueError('The dim_coord may not be an AuxCoord instance.')

        # Convert data_dim to a single integer
        if isinstance(data_dim, collections.Container):
            if len(data_dim) != 1:
                raise ValueError('The supplied data dimension must be a'
                                 ' single number.')
            data_dim = int(list(data_dim)[0])
        else:
            data_dim = int(data_dim)

        # Check data_dim value is valid
        if data_dim < 0 or data_dim >= self.ndim:
            raise ValueError('The cube does not have the specified dimension '
                             '(%d)' % data_dim)

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
        self._dim_coords_and_dims = [(coord_, dim) for coord_, dim in
                                     self._dim_coords_and_dims if coord_
                                     is not coord]
        self._aux_coords_and_dims = [(coord_, dims) for coord_, dims in
                                     self._aux_coords_and_dims if coord_
                                     is not coord]

    def remove_coord(self, coord):
        """
        Removes a coordinate from the cube.

        Args:

        * coord (string or coord)
            The (name of the) coordinate to remove from the cube.

        See also :meth:`Cube.add_dim_coord()<iris.cube.Cube.add_dim_coord>`
        and :meth:`Cube.add_aux_coord()<iris.cube.Cube.add_aux_coord>`.

        """
        coord = self.coord(coord)
        self._remove_coord(coord)

        for factory in self.aux_factories:
            factory.update(coord)

    def replace_coord(self, new_coord):
        """
        Replace the coordinate whose metadata matches the given coordinate.

        """
        old_coord = self.coord(new_coord)
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

        * coord (string or coord)
            The (name of the) coord to look for.

        """

        coord = self.coord(coord)

        # Search for existing coordinate (object) on the cube, faster lookup
        # than equality - makes no functional difference.
        matches = [(dim,) for coord_, dim in self._dim_coords_and_dims if
                   coord_ is coord]
        if not matches:
            matches = [dims for coord_, dims in self._aux_coords_and_dims if
                       coord_ is coord]

        # Search derived aux coords
        target_defn = coord._as_defn()
        if not matches:
            match = lambda factory: factory._as_defn() == target_defn
            factories = filter(match, self._aux_factories)
            matches = [factory.derived_dims(self.coord_dims) for factory in
                       factories]

        # Deprecate name based searching
        # -- Search by coord name, if have no match
        # XXX Where did this come from? And why isn't it reflected in the
        # docstring?
        if not matches:
            warnings.warn('name based coord matching is deprecated and will '
                          'be removed in a future release.',
                          stacklevel=2)
            matches = [(dim,) for coord_, dim in self._dim_coords_and_dims if
                       coord_.name() == coord.name()]
        # Finish deprecate name based searching

        if not matches:
            raise iris.exceptions.CoordinateNotFoundError(coord.name())

        return matches[0]

    def aux_factory(self, name=None, standard_name=None, long_name=None,
                    var_name=None):
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
        * var_name
            The CF variable name of the desired coordinate factory.
            If None, does not check for var_name.

        .. note::

            If the arguments given do not result in precisely 1 coordinate
            factory being matched, an
            :class:`iris.exceptions.CoordinateNotFoundError` is raised.

        """
        factories = self.aux_factories

        if name is not None:
            factories = [factory for factory in factories if
                         factory.name() == name]

        if standard_name is not None:
            factories = [factory for factory in factories if
                         factory.standard_name == standard_name]

        if long_name is not None:
            factories = [factory for factory in factories if
                         factory.long_name == long_name]

        if var_name is not None:
            factories = [factory for factory in factories if
                         factory.var_name == var_name]

        if len(factories) > 1:
            factory_names = (factory.name() for factory in factories)
            msg = 'Expected to find exactly one coordinate factory, but ' \
                  'found {}. They were: {}.'.format(len(factories),
                                                    ', '.join(factory_names))
            raise iris.exceptions.CoordinateNotFoundError(msg)
        elif len(factories) == 0:
            msg = 'Expected to find exactly one coordinate factory, but ' \
                  'found none.'
            raise iris.exceptions.CoordinateNotFoundError(msg)

        return factories[0]

    def coords(self, name_or_coord=None, standard_name=None,
               long_name=None, var_name=None, attributes=None, axis=None,
               contains_dimension=None, dimensions=None, coord=None,
               coord_system=None, dim_coords=None, name=None):
        """
        Return a list of coordinates in this cube fitting the given criteria.

        Kwargs:

        * name_or_coord
            Either

            (a) a :attr:`standard_name`, :attr:`long_name`, or
            :attr:`var_name`. Defaults to value of `default`
            (which itself defaults to `unknown`) as defined in
            :class:`iris._cube_coord_common.CFVariableMixin`.

            (b) a coordinate instance with metadata equal to that of
            the desired coordinates. Accepts either a
            :class:`iris.coords.DimCoord`, :class:`iris.coords.AuxCoord`,
            :class:`iris.aux_factory.AuxCoordFactory`
            or :class:`iris.coords.CoordDefn`.
        * name
            .. deprecated:: 1.6. Please use the name_or_coord kwarg.
        * standard_name
            The CF standard name of the desired coordinate. If None, does not
            check for standard name.
        * long_name
            An unconstrained description of the coordinate. If None, does not
            check for long_name.
        * var_name
            The CF variable name of the desired coordinate. If None, does not
            check for var_name.
        * attributes
            A dictionary of attributes desired on the coordinates. If None,
            does not check for attributes.
        * axis
            The desired coordinate axis, see
            :func:`iris.util.guess_coord_axis`. If None, does not check for
            axis. Accepts the values 'X', 'Y', 'Z' and 'T' (case-insensitive).
        * contains_dimension
            The desired coordinate contains the data dimension. If None, does
            not check for the dimension.
        * dimensions
            The exact data dimensions of the desired coordinate. Coordinates
            with no data dimension can be found with an empty tuple or list
            (i.e. ``()`` or ``[]``). If None, does not check for dimensions.
        * coord
            .. deprecated:: 1.6. Please use the name_or_coord kwarg.
        * coord_system
            Whether the desired coordinates have coordinate systems equal to
            the given coordinate system. If None, no check is done.
        * dim_coords
            Set to True to only return coordinates that are the cube's
            dimension coordinates. Set to False to only return coordinates
            that are the cube's auxiliary and derived coordinates. If None,
            returns all coordinates.

        See also :meth:`Cube.coord()<iris.cube.Cube.coord>`.

        """
        # Handle deprecated kwargs
        if name is not None:
            name_or_coord = name
            warnings.warn('the name kwarg is deprecated and will be removed '
                          'in a future release. Consider converting '
                          'existing code to use the name_or_coord '
                          'kwarg as a replacement.',
                          stacklevel=2)
        if coord is not None:
            name_or_coord = coord
            warnings.warn('the coord kwarg is deprecated and will be removed '
                          'in a future release. Consider converting '
                          'existing code to use the name_or_coord '
                          'kwarg as a replacement.',
                          stacklevel=2)
        # Finish handling deprecated kwargs

        name = None
        coord = None

        if isinstance(name_or_coord, basestring):
            name = name_or_coord
        else:
            coord = name_or_coord

        coords_and_factories = []

        if dim_coords in [True, None]:
            coords_and_factories += list(self.dim_coords)

        if dim_coords in [False, None]:
            coords_and_factories += list(self.aux_coords)
            coords_and_factories += list(self.aux_factories)

        if name is not None:
            coords_and_factories = [coord_ for coord_ in coords_and_factories
                                    if coord_.name() == name]

        if standard_name is not None:
            coords_and_factories = [coord_ for coord_ in coords_and_factories
                                    if coord_.standard_name == standard_name]

        if long_name is not None:
            coords_and_factories = [coord_ for coord_ in coords_and_factories
                                    if coord_.long_name == long_name]

        if var_name is not None:
            coords_and_factories = [coord_ for coord_ in coords_and_factories
                                    if coord_.var_name == var_name]

        if axis is not None:
            axis = axis.upper()
            guess_axis = iris.util.guess_coord_axis
            coords_and_factories = [coord_ for coord_ in coords_and_factories
                                    if guess_axis(coord_) == axis]

        if attributes is not None:
            if not isinstance(attributes, collections.Mapping):
                msg = 'The attributes keyword was expecting a dictionary ' \
                      'type, but got a %s instead.' % type(attributes)
                raise ValueError(msg)
            attr_filter = lambda coord_: all(k in coord_.attributes and
                                             coord_.attributes[k] == v for
                                             k, v in attributes.iteritems())
            coords_and_factories = [coord_ for coord_ in coords_and_factories
                                    if attr_filter(coord_)]

        if coord_system is not None:
            coords_and_factories = [coord_ for coord_ in coords_and_factories
                                    if coord_.coord_system == coord_system]

        if coord is not None:
            if isinstance(coord, iris.coords.CoordDefn):
                defn = coord
            else:
                defn = coord._as_defn()
            coords_and_factories = [coord_ for coord_ in coords_and_factories
                                    if coord_._as_defn() == defn]

        if contains_dimension is not None:
            coords_and_factories = [coord_ for coord_ in coords_and_factories
                                    if contains_dimension in
                                    self.coord_dims(coord_)]

        if dimensions is not None:
            if not isinstance(dimensions, collections.Container):
                dimensions = [dimensions]
            dimensions = tuple(dimensions)
            coords_and_factories = [coord_ for coord_ in coords_and_factories
                                    if self.coord_dims(coord_) == dimensions]

        # If any factories remain after the above filters we have to make the
        # coords so they can be returned
        def extract_coord(coord_or_factory):
            if isinstance(coord_or_factory, iris.aux_factory.AuxCoordFactory):
                coord = coord_or_factory.make_coord(self.coord_dims)
            elif isinstance(coord_or_factory, iris.coords.Coord):
                coord = coord_or_factory
            else:
                msg = 'Expected Coord or AuxCoordFactory, got ' \
                      '{!r}.'.format(type(coord_or_factory))
                raise ValueError(msg)
            return coord
        coords = [extract_coord(coord_or_factory) for coord_or_factory in
                  coords_and_factories]

        return coords

    def coord(self, name_or_coord=None, standard_name=None,
              long_name=None, var_name=None, attributes=None, axis=None,
              contains_dimension=None, dimensions=None, coord=None,
              coord_system=None, dim_coords=None, name=None):
        """
        Return a single coord given the same arguments as :meth:`Cube.coords`.

        .. note::

            If the arguments given do not result in precisely 1 coordinate
            being matched, an :class:`iris.exceptions.CoordinateNotFoundError`
            is raised.

        .. seealso::

            :meth:`Cube.coords()<iris.cube.Cube.coords>` for full keyword
            documentation.

        """
        # Handle deprecated kwargs
        if name is not None:
            name_or_coord = name
            warnings.warn('the name kwarg is deprecated and will be removed '
                          'in a future release. Consider converting '
                          'existing code to use the name_or_coord '
                          'kwarg as a replacement.',
                          stacklevel=2)
        if coord is not None:
            name_or_coord = coord
            warnings.warn('the coord kwarg is deprecated and will be removed '
                          'in a future release. Consider converting '
                          'existing code to use the name_or_coord '
                          'kwarg as a replacement.',
                          stacklevel=2)
        # Finish handling deprecated kwargs

        coords = self.coords(name_or_coord=name_or_coord,
                             standard_name=standard_name,
                             long_name=long_name, var_name=var_name,
                             attributes=attributes, axis=axis,
                             contains_dimension=contains_dimension,
                             dimensions=dimensions,
                             coord_system=coord_system,
                             dim_coords=dim_coords)

        if len(coords) > 1:
            msg = 'Expected to find exactly 1 coordinate, but found %s. ' \
                  'They were: %s.' % (len(coords), ', '.join(coord.name() for
                                                             coord in coords))
            raise iris.exceptions.CoordinateNotFoundError(msg)
        elif len(coords) == 0:
            bad_name = name or standard_name or long_name or \
                (coord and coord.name()) or ''
            msg = 'Expected to find exactly 1 %s coordinate, but found ' \
                  'none.' % bad_name
            raise iris.exceptions.CoordinateNotFoundError(msg)

        return coords[0]

    def coord_system(self, spec=None):
        """
        Find the coordinate system of the given type.

        If no target coordinate system is provided then find
        any available coordinate system.

        Kwargs:

        * spec:
            The the name or type of a coordinate system subclass.
            E.g. ::

                cube.coord_system("GeogCS")
                cube.coord_system(iris.coord_systems.GeogCS)

            If spec is provided as a type it can be a superclass of
            any coordinate system found.

            If spec is None, then find any available coordinate
            systems within the :class:`iris.cube.Cube`.

        Returns:
            The :class:`iris.coord_systems.CoordSystem` or None.

        """
        if isinstance(spec, basestring) or spec is None:
            spec_name = spec
        else:
            msg = "type %s is not a subclass of CoordSystem" % spec
            assert issubclass(spec, iris.coord_systems.CoordSystem), msg
            spec_name = spec.__name__

        # Gather a temporary list of our unique CoordSystems.
        coord_systems = ClassDict(iris.coord_systems.CoordSystem)
        for coord in self.coords():
            if coord.coord_system:
                coord_systems.add(coord.coord_system, replace=True)

        result = None
        if spec_name is None:
            for key in sorted(coord_systems.keys()):
                result = coord_systems[key]
                break
        else:
            result = coord_systems.get(spec_name)

        return result

    @property
    def cell_methods(self):
        """
        Tuple of :class:`iris.coords.CellMethod` representing the processing
        done on the phenomenon.

        """
        return self._cell_methods

    @cell_methods.setter
    def cell_methods(self, cell_methods):
        self._cell_methods = tuple(cell_methods) if cell_methods else tuple()

    @property
    def cell_measures(self):

        return self._cell_measures

    @cell_measures.setter
    def cell_measures(self, cell_measures):
        
        if cell_measures:
            if not isinstance(cell_measures, iris.coords.CellMeasures):

                raise TypeError("cell_measures must be an instance of iris."
                                "coords.CellMeasures or None")
            

            # this is not quite right, I've not been able to find an easy way
            # to get the spatial shape of the cube (i.e. the cell structures
            # shape)
            if not set(cell_measures.shape).issubset(set(self.shape)):
                
               raise TypeError("Cell Measures shape %r must match Cube shape %r"
                               % (cell_measures.shape, self.shape))

        self._cell_measures = cell_measures

    @property
    def shape(self):
        """The shape of the data of this cube."""
        shape = self.lazy_data().shape
        return shape

    @property
    def dtype(self):
        """The :class:`numpy.dtype` of the data of this cube."""
        return self.lazy_data().dtype

    @property
    def ndim(self):
        """The number of dimensions in the data of this cube."""
        return len(self.shape)

    def lazy_data(self, array=None):
        """
        Return a :class:`biggus.Array` representing the
        multi-dimensional data of the Cube, and optionally provide a
        new array of values.

        Accessing this method will never cause the data to be loaded.
        Similarly, calling methods on, or indexing, the returned Array
        will not cause the Cube to have loaded data.

        If the data have already been loaded for the Cube, the returned
        Array will be a :class:`biggus.NumpyArrayAdapter` which wraps
        the numpy array from `self.data`.

        Kwargs:

        * array (:class:`biggus.Array` or None):
            When this is not None it sets the multi-dimensional data of
            the cube to the given value.

        Returns:
            A :class:`biggus.Array` representing the multi-dimensional
            data of the Cube.

        """
        if array is not None:
            if not isinstance(array, biggus.Array):
                raise TypeError('new values must be a biggus.Array')
            if self.shape != array.shape:
                # The _ONLY_ data reshape permitted is converting a
                # 0-dimensional array into a 1-dimensional array of
                # length one.
                # i.e. self.shape = () and array.shape == (1,)
                if self.shape or array.shape != (1,):
                    raise ValueError('Require cube data with shape %r, got '
                                     '%r.' % (self.shape, array.shape))
            self._my_data = array
        else:
            array = self._my_data
            if not isinstance(array, biggus.Array):
                array = biggus.NumpyArrayAdapter(array)
        return array

    @property
    def data(self):
        """
        The :class:`numpy.ndarray` representing the multi-dimensional data of
        the cube.

        .. note::

            Cubes obtained from netCDF, PP, and FieldsFile files will only
            populate this attribute on its first use.

            To obtain the shape of the data without causing it to be loaded,
            use the Cube.shape attribute.

        Example::
            >>> fname = iris.sample_data_path('air_temp.pp')
            >>> cube = iris.load_cube(fname, 'air_temperature')
            >>> # cube.data does not yet have a value.
            ...
            >>> print(cube.shape)
            (73, 96)
            >>> # cube.data still does not have a value.
            ...
            >>> cube = cube[:10, :20]
            >>> # cube.data still does not have a value.
            ...
            >>> data = cube.data
            >>> # Only now is the data loaded.
            ...
            >>> print(data.shape)
            (10, 20)

        """
        data = self._my_data
        if not isinstance(data, np.ndarray):
            try:
                data = data.masked_array()
            except MemoryError:
                msg = "Failed to create the cube's data as there was not" \
                      " enough memory available.\n" \
                      "The array shape would have been {0!r} and the data" \
                      " type {1}.\n" \
                      "Consider freeing up variables or indexing the cube" \
                      " before getting its data."
                msg = msg.format(self.shape, data.dtype)
                raise MemoryError(msg)
            # Unmask the array only if it is filled.
            if ma.count_masked(data) == 0:
                data = data.data
            self._my_data = data
        return data

    @data.setter
    def data(self, value):
        data = np.asanyarray(value)

        if self.shape != data.shape:
            # The _ONLY_ data reshape permitted is converting a 0-dimensional
            # array i.e. self.shape == () into a 1-dimensional array of length
            # one i.e. data.shape == (1,)
            if self.shape or data.shape != (1,):
                raise ValueError('Require cube data with shape %r, got '
                                 '%r.' % (self.shape, data.shape))

        self._my_data = data

    def has_lazy_data(self):
        return isinstance(self._my_data, biggus.Array)

    @property
    def dim_coords(self):
        """
        Return a tuple of all the dimension coordinates, ordered by dimension.

        .. note::

            The length of the returned tuple is not necessarily the same as
            :attr:`Cube.ndim` as there may be dimensions on the cube without
            dimension coordinates. It is therefore unreliable to use the
            resulting tuple to identify the dimension coordinates for a given
            dimension - instead use the :meth:`Cube.coord` method with the
            ``dimensions`` and ``dim_coords`` keyword arguments.

        """
        return tuple((coord for coord, dim in
                      sorted(self._dim_coords_and_dims,
                             key=lambda co_di: (co_di[1], co_di[0].name()))))

    @property
    def aux_coords(self):
        """
        Return a tuple of all the auxiliary coordinates, ordered by
        dimension(s).

        """
        return tuple((coord for coord, dims in
                      sorted(self._aux_coords_and_dims,
                             key=lambda co_di: (co_di[1], co_di[0].name()))))

    @property
    def derived_coords(self):
        """
        Return a tuple of all the coordinates generated by the coordinate
        factories.

        """
        return tuple(factory.make_coord(self.coord_dims) for factory in
                     sorted(self.aux_factories,
                            key=lambda factory: factory.name()))

    @property
    def aux_factories(self):
        """Return a tuple of all the coordinate factories."""
        return tuple(self._aux_factories)

    def _summary_coord_extra(self, coord, indent):
        # Returns the text needed to ensure this coordinate can be
        # distinguished from all others with the same name.
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
            bits = ['{}={!r}'.format(key, coord.attributes[key]) for key in
                    keys]
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
        Unicode string summary of the Cube with name, a list of dim coord names
        versus length and optionally relevant coordinate information.

        """
        # Create a set to contain the axis names for each data dimension.
        dim_names = [set() for dim in range(len(self.shape))]

        # Add the dim_coord names that participate in the associated data
        # dimensions.
        for dim in range(len(self.shape)):
            dim_coords = self.coords(contains_dimension=dim, dim_coords=True)
            if dim_coords:
                dim_names[dim].add(dim_coords[0].name())
            else:
                dim_names[dim].add('-- ')

        # Convert axes sets to lists and sort.
        dim_names = [sorted(names, key=sorted_axes) for names in dim_names]

        # Generate textual summary of the cube dimensionality.
        if self.shape == ():
            dimension_header = 'scalar cube'
        else:
            dimension_header = '; '.join(
                [', '.join(dim_names[dim]) +
                 ': %d' % dim_shape for dim, dim_shape in
                 enumerate(self.shape)])

        nameunit = '{name} / ({units})'.format(name=self.name(),
                                               units=self.units)
        cube_header = '{nameunit!s:{length}} ({dimension})'.format(
            length=name_padding,
            nameunit=nameunit,
            dimension=dimension_header)
        summary = ''

        # Generate full cube textual summary.
        if not shorten:
            indent = 10
            extra_indent = ' ' * 13

            # Cache the derived coords so we can rely on consistent
            # object IDs.
            derived_coords = self.derived_coords
            # Determine the cube coordinates that are scalar (single-valued)
            # AND non-dimensioned.
            dim_coords = self.dim_coords
            aux_coords = self.aux_coords
            all_coords = dim_coords + aux_coords + derived_coords
            scalar_coords = [coord for coord in all_coords if not
                             self.coord_dims(coord) and coord.shape == (1,)]
            # Determine the cube coordinates that are not scalar BUT
            # dimensioned.
            scalar_coord_ids = set(map(id, scalar_coords))
            vector_dim_coords = [coord for coord in dim_coords if id(coord) not
                                 in scalar_coord_ids]
            vector_aux_coords = [coord for coord in aux_coords if id(coord) not
                                 in scalar_coord_ids]
            vector_derived_coords = [coord for coord in derived_coords if
                                     id(coord) not in scalar_coord_ids]

            # Determine the cube coordinates that don't describe the cube and
            # are most likely erroneous.
            vector_coords = vector_dim_coords + vector_aux_coords + \
                vector_derived_coords
            ok_coord_ids = scalar_coord_ids.union(set(map(id, vector_coords)))
            invalid_coords = [coord for coord in all_coords if id(coord) not
                              in ok_coord_ids]

            # Sort scalar coordinates by name.
            scalar_coords.sort(key=lambda coord: coord.name())
            # Sort vector coordinates by data dimension and name.
            vector_dim_coords.sort(
                key=lambda coord: (self.coord_dims(coord), coord.name()))
            vector_aux_coords.sort(
                key=lambda coord: (self.coord_dims(coord), coord.name()))
            vector_derived_coords.sort(
                key=lambda coord: (self.coord_dims(coord), coord.name()))
            # Sort other coordinates by name.
            invalid_coords.sort(key=lambda coord: coord.name())

            #
            # Generate textual summary of cube vector coordinates.
            #
            def vector_summary(vector_coords, cube_header, max_line_offset):
                """
                Generates a list of suitably aligned strings containing coord
                names and dimensions indicated by one or more 'x' symbols.

                .. note::

                    The function may need to update the cube header so this is
                    returned with the list of strings.

                """
                vector_summary = []
                if vector_coords:
                    # Identify offsets for each dimension text marker.
                    alignment = np.array([index for index, value in
                                          enumerate(cube_header) if
                                          value == ':'])

                    # Generate basic textual summary for each vector coordinate
                    # - WITHOUT dimension markers.
                    for coord in vector_coords:
                        vector_summary.append('%*s%s' % (
                            indent, ' ', iris.util.clip_string(coord.name())))
                    min_alignment = min(alignment)

                    # Determine whether the cube header requires realignment
                    # due to one or more longer vector coordinate summaries.
                    if max_line_offset >= min_alignment:
                        delta = max_line_offset - min_alignment + 5
                        cube_header = '%-*s (%s)' % (int(name_padding + delta),
                                                     self.name() or 'unknown',
                                                     dimension_header)
                        alignment += delta

                    # Generate full textual summary for each vector coordinate
                    # - WITH dimension markers.
                    for index, coord in enumerate(vector_coords):
                        dims = self.coord_dims(coord)
                        for dim in range(len(self.shape)):
                            width = alignment[dim] - len(vector_summary[index])
                            char = 'x' if dim in dims else '-'
                            line = '{pad:{width}}{char}'.format(pad=' ',
                                                                width=width,
                                                                char=char)
                            vector_summary[index] += line
                    # Interleave any extra lines that are needed to distinguish
                    # the coordinates.
                    vector_summary = self._summary_extra(vector_coords,
                                                         vector_summary,
                                                         extra_indent)

                return vector_summary, cube_header

            # Calculate the maximum line offset.
            max_line_offset = 0
            for coord in all_coords:
                max_line_offset = max(max_line_offset, len('%*s%s' % (
                    indent, ' ', iris.util.clip_string(str(coord.name())))))

            if vector_dim_coords:
                dim_coord_summary, cube_header = vector_summary(
                    vector_dim_coords, cube_header, max_line_offset)
                summary += '\n     Dimension coordinates:\n' + \
                    '\n'.join(dim_coord_summary)

            if vector_aux_coords:
                aux_coord_summary, cube_header = vector_summary(
                    vector_aux_coords, cube_header, max_line_offset)
                summary += '\n     Auxiliary coordinates:\n' + \
                    '\n'.join(aux_coord_summary)

            if vector_derived_coords:
                derived_coord_summary, cube_header = vector_summary(
                    vector_derived_coords, cube_header, max_line_offset)
                summary += '\n     Derived coordinates:\n' + \
                    '\n'.join(derived_coord_summary)
                            
            #
            # Generate summary of cube cell measures attribute
            #

            if self.cell_measures:
                summary += '\n     Cell Measures:\n'
                
                summary += '%*s%s' % (indent, ' ', str(self.cell_measures))

            #
            # Generate textual summary of cube scalar coordinates.
            #
            scalar_summary = []

            if scalar_coords:
                for coord in scalar_coords:
                    if (coord.units in ['1', 'no_unit', 'unknown'] or
                            coord.units.is_time_reference()):
                        unit = ''
                    else:
                        unit = ' {!s}'.format(coord.units)

                    # Format cell depending on type of point and whether it
                    # has a bound
                    with iris.FUTURE.context(cell_datetime_objects=False):
                        coord_cell = coord.cell(0)
                    if isinstance(coord_cell.point, basestring):
                        # Indent string type coordinates
                        coord_cell_split = [iris.util.clip_string(str(item))
                                            for item in
                                            coord_cell.point.split('\n')]
                        line_sep = '\n{pad:{width}}'.format(
                            pad=' ', width=indent + len(coord.name()) + 2)
                        coord_cell_str = line_sep.join(coord_cell_split) + unit
                    else:
                        # Human readable times
                        if coord.units.is_time_reference():
                            coord_cell_cpoint = coord.units.num2date(
                                coord_cell.point)
                            if coord_cell.bound is not None:
                                coord_cell_cbound = coord.units.num2date(
                                    coord_cell.bound)
                        else:
                            coord_cell_cpoint = coord_cell.point
                            coord_cell_cbound = coord_cell.bound

                        coord_cell_str = '{!s}{}'.format(coord_cell_cpoint,
                                                         unit)
                        if coord_cell.bound is not None:
                            bound = '({})'.format(', '.join(str(val) for
                                                  val in coord_cell_cbound))
                            coord_cell_str += ', bound={}{}'.format(bound,
                                                                    unit)

                    scalar_summary.append('{pad:{width}}{name}: {cell}'.format(
                        pad=' ', width=indent, name=coord.name(),
                        cell=coord_cell_str))

                # Interleave any extra lines that are needed to distinguish
                # the coordinates.
                scalar_summary = self._summary_extra(scalar_coords,
                                                     scalar_summary,
                                                     extra_indent)

                summary += '\n     Scalar coordinates:\n' + '\n'.join(
                    scalar_summary)

            #
            # Generate summary of cube's invalid coordinates.
            #
            if invalid_coords:
                invalid_summary = []

                for coord in invalid_coords:
                    invalid_summary.append(
                        '%*s%s' % (indent, ' ', coord.name()))

                # Interleave any extra lines that are needed to distinguish the
                # coordinates.
                invalid_summary = self._summary_extra(
                    invalid_coords, invalid_summary, extra_indent)

                summary += '\n     Invalid coordinates:\n' + \
                    '\n'.join(invalid_summary)


            #
            # Generate summary of cube attributes.
            #
            if self.attributes:
                attribute_lines = []
                for name, value in sorted(self.attributes.iteritems()):
                    value = iris.util.clip_string(unicode(value))
                    line = u'{pad:{width}}{name}: {value}'.format(pad=' ',
                                                                  width=indent,
                                                                  name=name,
                                                                  value=value)
                    attribute_lines.append(line)
                summary += '\n     Attributes:\n' + '\n'.join(attribute_lines)


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
        return self.summary().encode(errors='replace')

    def __unicode__(self):
        return self.summary()

    def __repr__(self):
        return "<iris 'Cube' of %s>" % self.summary(shorten=True,
                                                    name_padding=1)

    def __iter__(self):
        raise TypeError('Cube is not iterable')

    def __getitem__(self, keys):
        """
        Cube indexing (through use of square bracket notation) has been
        implemented at the data level. That is, the indices provided to this
        method should be aligned to the data of the cube, and thus the indices
        requested must be applicable directly to the cube.data attribute. All
        metadata will be subsequently indexed appropriately.

        """
        # turn the keys into a full slice spec (all dims)
        full_slice = iris.util._build_full_slice_given_keys(keys,
                                                            len(self.shape))

        # make indexing on the cube column based by using the
        # column_slices_generator (potentially requires slicing the data
        # multiple times)
        dimension_mapping, slice_gen = iris.util.column_slices_generator(
            full_slice, len(self.shape))
        new_coord_dims = lambda coord_: [dimension_mapping[d] for d in
                                         self.coord_dims(coord_) if
                                         dimension_mapping[d] is not None]

        try:
            first_slice = next(slice_gen)
        except StopIteration:
            first_slice = None

        if first_slice is not None:
            data = self._my_data[first_slice]
        else:
            data = copy.deepcopy(self._my_data)

        for other_slice in slice_gen:
            data = data[other_slice]

        # We don't want a view of the data, so take a copy of it if it's
        # not already our own.
        if isinstance(data, biggus.Array) or not data.flags['OWNDATA']:
            data = copy.deepcopy(data)

        # We can turn a masked array into a normal array if it's full.
        if isinstance(data, ma.core.MaskedArray):
            if ma.count_masked(data) == 0:
                data = data.filled()

        # Make the new cube slice
        cube = Cube(data)
        cube.metadata = copy.deepcopy(self.metadata)

        # Record a mapping from old coordinate IDs to new coordinates,
        # for subsequent use in creating updated aux_factories.
        coord_mapping = {}

        # Slice the coords
        for coord in self.aux_coords:
            coord_keys = tuple([full_slice[dim] for dim in
                                self.coord_dims(coord)])
            try:
                new_coord = coord[coord_keys]
            except ValueError:
                # TODO make this except more specific to catch monotonic error
                # Attempt to slice it by converting to AuxCoord first
                new_coord = iris.coords.AuxCoord.from_coord(coord)[coord_keys]
            cube.add_aux_coord(new_coord, new_coord_dims(coord))
            coord_mapping[id(coord)] = new_coord

        for coord in self.dim_coords:
            coord_keys = tuple([full_slice[dim] for dim in
                                self.coord_dims(coord)])
            new_dims = new_coord_dims(coord)
            # Try/Catch to handle slicing that makes the points/bounds
            # non-monotonic
            try:
                new_coord = coord[coord_keys]
                if not new_dims:
                    # If the associated dimension has been sliced so the coord
                    # is a scalar move the coord to the aux_coords container
                    cube.add_aux_coord(new_coord, new_dims)
                else:
                    cube.add_dim_coord(new_coord, new_dims)
            except ValueError:
                # TODO make this except more specific to catch monotonic error
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
        coord_to_extract = self.coord(coord)
        if len(self.coord_dims(coord_to_extract)) > 1:
            msg = "Currently, only 1D coords can be used to subset a cube"
            raise iris.exceptions.CoordinateMultiDimError(msg)
        # Identify the dimension of the cube which this coordinate references
        coord_to_extract_dim = self.coord_dims(coord_to_extract)[0]

        # Identify the indices which intersect the requested coord and
        # coord_to_extract
        coordinate_indices = coord_to_extract.intersect(coord,
                                                        return_indices=True)

        # Build up a slice which spans the whole of the cube
        full_slice = [slice(None, None)] * len(self.shape)
        # Update the full slice to only extract specific indices which were
        # identified above
        full_slice[coord_to_extract_dim] = coordinate_indices
        full_slice = tuple(full_slice)
        return self[full_slice]

    def extract(self, constraint):
        """
        Filter the cube by the given constraint using
        :meth:`iris.Constraint.extract` method.

        """
        # Cast the constraint into a proper constraint if it is not so already
        constraint = iris._constraints.as_constraint(constraint)
        return constraint.extract(self)

    def intersection(self, *args, **kwargs):
        """
        Return the intersection of the cube with specified coordinate
        ranges.

        Coordinate ranges can be specified as:

        (a) instances of :class:`iris.coords.CoordExtent`.

        (b) keyword arguments, where the keyword name specifies the name
            of the coordinate (as defined in :meth:`iris.cube.Cube.coords()`)
            and the value defines the corresponding range of coordinate
            values as a tuple. The tuple must contain two, three, or four
            items corresponding to: (minimum, maximum, min_inclusive,
            max_inclusive). Where the items are defined as:

            * minimum
                The minimum value of the range to select.

            * maximum
                The maximum value of the range to select.

            * min_inclusive
                If True, coordinate values equal to `minimum` will be included
                in the selection. Default is True.

            * max_inclusive
                If True, coordinate values equal to `maximum` will be included
                in the selection. Default is True.

        To perform an intersection that ignores any bounds on the coordinates,
        set the optional keyword argument *ignore_bounds* to True. Defaults to
        False.

        .. note::

            For ranges defined over "circular" coordinates (i.e. those
            where the `units` attribute has a modulus defined) the cube
            will be "rolled" to fit where neccesary.

        .. warning::

            Currently this routine only works with "circular"
            coordinates (as defined in the previous note.)

        For example::

            >>> import iris
            >>> cube = iris.load_cube(iris.sample_data_path('air_temp.pp'))
            >>> print(cube.coord('longitude').points[::10])
            [   0.           37.49999237   74.99998474  112.49996948  \
149.99996948
              187.49995422  224.99993896  262.49993896  299.99993896  \
337.49990845]
            >>> subset = cube.intersection(longitude=(30, 50))
            >>> print(subset.coord('longitude').points)
            [ 33.74999237  37.49999237  41.24998856  44.99998856  48.74998856]
            >>> subset = cube.intersection(longitude=(-10, 10))
            >>> print(subset.coord('longitude').points)
            [-7.50012207 -3.75012207  0.          3.75        7.5       ]

        Returns:
            A new :class:`~iris.cube.Cube` giving the subset of the cube
            which intersects with the requested coordinate intervals.

        """
        result = self
        ignore_bounds = kwargs.pop('ignore_bounds', False)
        for arg in args:
            result = result._intersect(*arg, ignore_bounds=ignore_bounds)
        for name, value in kwargs.iteritems():
            result = result._intersect(name, *value,
                                       ignore_bounds=ignore_bounds)
        return result

    def _intersect(self, name_or_coord, minimum, maximum,
                   min_inclusive=True, max_inclusive=True,
                   ignore_bounds=False):
        coord = self.coord(name_or_coord)
        if coord.ndim != 1:
            raise iris.exceptions.CoordinateMultiDimError(coord)
        if coord.nbounds not in (0, 2):
            raise ValueError('expected 0 or 2 bound values per cell')
        if minimum > maximum:
            raise ValueError('minimum greater than maximum')
        modulus = coord.units.modulus
        if modulus is None:
            raise ValueError('coordinate units with no modulus are not yet'
                             ' supported')
        subsets, points, bounds = self._intersect_modulus(coord,
                                                          minimum, maximum,
                                                          min_inclusive,
                                                          max_inclusive,
                                                          ignore_bounds)

        # By this point we have either one or two subsets along the relevant
        # dimension. If it's just one subset (which might be a slice or an
        # unordered collection of indices) we can simply index the cube
        # and we're done. If it's two subsets we need to stitch the two
        # pieces together.
        def make_chunk(key):
            chunk = self[key_tuple_prefix + (key,)]
            chunk_coord = chunk.coord(coord)
            chunk_coord.points = points[(key,)]
            if chunk_coord.has_bounds():
                chunk_coord.bounds = bounds[(key,)]
            return chunk

        dim, = self.coord_dims(coord)
        key_tuple_prefix = (slice(None),) * dim
        chunks = [make_chunk(key) for key in subsets]
        if len(chunks) == 1:
            result = chunks[0]
        else:
            if self.has_lazy_data():
                data = biggus.LinearMosaic([chunk.lazy_data()
                                            for chunk in chunks],
                                           dim)
            else:
                module = ma if ma.isMaskedArray(self.data) else np
                data = module.concatenate([chunk.data for chunk in chunks],
                                          dim)
            result = iris.cube.Cube(data)
            result.metadata = copy.deepcopy(self.metadata)

            # Record a mapping from old coordinate IDs to new coordinates,
            # for subsequent use in creating updated aux_factories.
            coord_mapping = {}

            def create_coords(src_coords, add_coord):
                # Add copies of the source coordinates, selecting
                # the appropriate subsets out of coordinates which
                # share the intersection dimension.
                preserve_circular = (min_inclusive and max_inclusive and
                                     abs(maximum - minimum) == modulus)
                for src_coord in src_coords:
                    dims = self.coord_dims(src_coord)
                    if dim in dims:
                        dim_within_coord = dims.index(dim)
                        points = np.concatenate([chunk.coord(src_coord).points
                                                 for chunk in chunks],
                                                dim_within_coord)
                        if src_coord.has_bounds():
                            bounds = np.concatenate(
                                [chunk.coord(src_coord).bounds
                                 for chunk in chunks],
                                dim_within_coord)
                        else:
                            bounds = None
                        result_coord = src_coord.copy(points=points,
                                                      bounds=bounds)

                        circular = getattr(result_coord, 'circular', False)
                        if circular and not preserve_circular:
                            result_coord.circular = False
                    else:
                        result_coord = src_coord.copy()
                    add_coord(result_coord, dims)
                    coord_mapping[id(src_coord)] = result_coord

            create_coords(self.dim_coords, result.add_dim_coord)
            create_coords(self.aux_coords, result.add_aux_coord)
            for factory in self.aux_factories:
                result.add_aux_factory(factory.updated(coord_mapping))
        return result

    def _intersect_modulus(self, coord, minimum, maximum, min_inclusive,
                           max_inclusive, ignore_bounds):
        modulus = coord.units.modulus
        if maximum > minimum + modulus:
            raise ValueError("requested range greater than coordinate's"
                             " unit's modulus")
        if coord.has_bounds():
            values = coord.bounds
        else:
            values = coord.points
        if values.max() > values.min() + modulus:
            raise ValueError("coordinate's range greater than coordinate's"
                             " unit's modulus")
        min_comp = np.less_equal if min_inclusive else np.less
        max_comp = np.less_equal if max_inclusive else np.less

        if coord.has_bounds():
            bounds = wrap_lons(coord.bounds, minimum, modulus)
            if ignore_bounds:
                points = wrap_lons(coord.points, minimum, modulus)
                inside_indices, = np.where(
                    np.logical_and(min_comp(minimum, points),
                                   max_comp(points, maximum)))
            else:
                inside = np.logical_and(min_comp(minimum, bounds),
                                        max_comp(bounds, maximum))
                inside_indices, = np.where(np.any(inside, axis=1))

            # To ensure that bounds (and points) of matching cells aren't
            # "scrambled" by the wrap operation we detect split cells that
            # straddle the wrap point and choose a new wrap point which avoids
            # split cells.
            # For example: the cell [349.875, 350.4375] wrapped at -10 would
            # become [349.875, -9.5625] which is no longer valid. The lower
            # cell bound value (and possibly associated point) are
            # recalculated so that they are consistent with the extended
            # wrapping scheme which moves the wrap point to the correct lower
            # bound value (-10.125) thus resulting in the cell no longer
            # being split. For bounds which may extend exactly the length of
            # the modulus, we simply preserve the point to bound difference,
            # and call the new bounds = the new points + the difference.
            pre_wrap_delta = np.diff(coord.bounds[inside_indices])
            post_wrap_delta = np.diff(bounds[inside_indices])
            close_enough = np.allclose(pre_wrap_delta, post_wrap_delta)
            if not close_enough:
                split_cell_indices, _ = np.where(pre_wrap_delta !=
                                                 post_wrap_delta)

                # Recalculate the extended minimum.
                indices = inside_indices[split_cell_indices]
                cells = bounds[indices]
                cells_delta = np.diff(coord.bounds[indices])

                # Watch out for ascending/descending bounds
                if cells_delta[0, 0] > 0:
                    cells[:, 0] = cells[:, 1] - cells_delta[:, 0]
                    minimum = np.min(cells[:, 0])
                else:
                    cells[:, 1] = cells[:, 0] + cells_delta[:, 0]
                    minimum = np.min(cells[:, 1])

            points = wrap_lons(coord.points, minimum, modulus)

            bound_diffs = coord.points[:, np.newaxis] - coord.bounds
            bounds = points[:, np.newaxis] - bound_diffs
        else:
            points = wrap_lons(coord.points, minimum, modulus)
            bounds = None
            inside_indices, = np.where(
                np.logical_and(min_comp(minimum, points),
                               max_comp(points, maximum)))
        if isinstance(coord, iris.coords.DimCoord):
            delta = coord.points[inside_indices] - points[inside_indices]
            step = np.rint(np.diff(delta) / modulus)
            non_zero_step_indices = np.nonzero(step)[0]
            if non_zero_step_indices.size:
                # A contiguous block at the start and another at the
                # end. (NB. We can't have more than two blocks
                # because we've already restricted the coordinate's
                # range to its modulus).
                end_of_first_chunk = non_zero_step_indices[0]
                subsets = [slice(inside_indices[end_of_first_chunk + 1], None),
                           slice(None, inside_indices[end_of_first_chunk] + 1)]
            else:
                # A single, contiguous block.
                subsets = [slice(inside_indices[0], inside_indices[-1] + 1)]
        else:
            # An AuxCoord could have its values in an arbitrary
            # order, and hence a range of values can select an
            # arbitrary subset. Also, we want to preserve the order
            # from the original AuxCoord. So we just use the indices
            # directly.
            subsets = [inside_indices]
        return subsets, points, bounds

    def _as_list_of_coords(self, names_or_coords):
        """
        Convert a name, coord, or list of names/coords to a list of coords.
        """
        # If not iterable, convert to list of a single item
        if not hasattr(names_or_coords, '__iter__'):
            names_or_coords = [names_or_coords]

        coords = []
        for name_or_coord in names_or_coords:
            if (isinstance(name_or_coord, basestring) or
                    isinstance(name_or_coord, iris.coords.Coord)):
                coords.append(self.coord(name_or_coord))
            else:
                # Don't know how to handle this type
                msg = "Don't know how to handle coordinate of type %s. " \
                      "Ensure all coordinates are of type basestring or " \
                      "iris.coords.Coord." % type(name_or_coord)
                raise TypeError(msg)
        return coords

    def slices_over(self, ref_to_slice):
        """
        Return an iterator of all subcubes along a given coordinate or
        dimension index, or multiple of these.

        Args:

        * ref_to_slice (string, coord, dimension index or a list of these):
            Determines which dimensions will be iterated along (i.e. the
            dimensions that are not returned in the subcubes).
            A mix of input types can also be provided.

        Returns:
            An iterator of subcubes.

        For example, to get all subcubes along the time dimension::

            for sub_cube in cube.slices_over('time'):
                print(sub_cube)

        .. seealso:: :meth:`iris.cube.Cube.slices`.

        .. note::

            The order of dimension references to slice along does not affect
            the order of returned items in the iterator; instead the ordering
            is based on the fastest-changing dimension.

        """
        # Required to handle a mix between types.
        if not hasattr(ref_to_slice, '__iter__'):
            ref_to_slice = [ref_to_slice]

        slice_dims = set()
        for ref in ref_to_slice:
            try:
                coord, = self._as_list_of_coords(ref)
            except TypeError:
                dim = int(ref)
                if dim < 0 or dim > self.ndim:
                    msg = ('Requested an iterator over a dimension ({}) '
                           'which does not exist.'.format(dim))
                    raise ValueError(msg)
                # Convert coord index to a single-element list to prevent a
                # TypeError when `slice_dims.update` is called with it.
                dims = [dim]
            else:
                dims = self.coord_dims(coord)
            slice_dims.update(dims)

        all_dims = set(range(self.ndim))
        opposite_dims = list(all_dims - slice_dims)
        return self.slices(opposite_dims, ordered=False)

    def slices(self, ref_to_slice, ordered=True):
        """
        Return an iterator of all subcubes given the coordinates or dimension
        indices desired to be present in each subcube.

        Args:

        * ref_to_slice (string, coord, dimension index or a list of these):
            Determines which dimensions will be returned in the subcubes (i.e.
            the dimensions that are not iterated over).
            A mix of input types can also be provided. They must all be
            orthogonal (i.e. point to different dimensions).

        Kwargs:

        * ordered: if True, the order which the coords to slice or data_dims
            are given will be the order in which they represent the data in
            the resulting cube slices.  If False, the order will follow that of
            the source cube.  Default is True.

        Returns:
            An iterator of subcubes.

        For example, to get all 2d longitude/latitude subcubes from a
        multi-dimensional cube::

            for sub_cube in cube.slices(['longitude', 'latitude']):
                print(sub_cube)

        .. seealso:: :meth:`iris.cube.Cube.slices_over`.

        """
        if not isinstance(ordered, bool):
            raise TypeError("'ordered' argument to slices must be boolean.")

        # Required to handle a mix between types
        if not hasattr(ref_to_slice, '__iter__'):
            ref_to_slice = [ref_to_slice]

        dim_to_slice = []
        for ref in ref_to_slice:
            try:
                # attempt to handle as coordinate
                coord = self._as_list_of_coords(ref)[0]
                dims = self.coord_dims(coord)
                if not dims:
                    msg = ('Requested an iterator over a coordinate ({}) '
                           'which does not describe a dimension.')
                    msg = msg.format(coord.name())
                    raise ValueError(msg)
                dim_to_slice.extend(dims)

            except TypeError:
                try:
                    # attempt to handle as dimension index
                    dim = int(ref)
                except ValueError:
                    raise ValueError('{} Incompatible type {} for '
                                     'slicing'.format(ref, type(ref)))
                if dim < 0 or dim > self.ndim:
                    msg = ('Requested an iterator over a dimension ({}) '
                           'which does not exist.'.format(dim))
                    raise ValueError(msg)
                dim_to_slice.append(dim)

        if len(set(dim_to_slice)) != len(dim_to_slice):
            msg = 'The requested coordinates are not orthogonal.'
            raise ValueError(msg)

        # Create a list with of the shape of our data
        dims_index = list(self.shape)

        # Set the dimensions which have been requested to length 1
        for d in dim_to_slice:
            dims_index[d] = 1

        return _SliceIterator(self, dims_index, dim_to_slice, ordered)

    def transpose(self, new_order=None):
        """
        Re-order the data dimensions of the cube in-place.

        new_order - list of ints, optional
                    By default, reverse the dimensions, otherwise permute the
                    axes according to the values given.

        .. note:: If defined, new_order must span all of the data dimensions.

        Example usage::

            # put the second dimension first, followed by the third dimension,
            and finally put the first dimension third cube.transpose([1, 2, 0])

        """
        if new_order is None:
            new_order = np.arange(self.data.ndim)[::-1]
        elif len(new_order) != self.data.ndim:
            raise ValueError('Incorrect number of dimensions.')

        # The data needs to be copied, otherwise this view of the transposed
        # data will not be contiguous. Ensure not to assign via the cube.data
        # setter property since we are reshaping the cube payload in-place.
        self._my_data = np.transpose(self.data, new_order).copy()

        dim_mapping = {src: dest for dest, src in enumerate(new_order)}

        def remap_dim_coord(coord_and_dim):
            coord, dim = coord_and_dim
            return coord, dim_mapping[dim]
        self._dim_coords_and_dims = list(map(remap_dim_coord,
                                             self._dim_coords_and_dims))

        def remap_aux_coord(coord_and_dims):
            coord, dims = coord_and_dims
            return coord, tuple(dim_mapping[dim] for dim in dims)
        self._aux_coords_and_dims = list(map(remap_aux_coord,
                                             self._aux_coords_and_dims))

    def xml(self, checksum=False, order=True, byteorder=True):
        """
        Returns a fully valid CubeML string representation of the Cube.

        """
        doc = Document()

        cube_xml_element = self._xml_element(doc, checksum=checksum,
                                             order=order,
                                             byteorder=byteorder)
        cube_xml_element.setAttribute("xmlns", XML_NAMESPACE_URI)
        doc.appendChild(cube_xml_element)

        # Print our newly created XML
        return doc.toprettyxml(indent="  ")

    def _xml_element(self, doc, checksum=False, order=True, byteorder=True):
        cube_xml_element = doc.createElement("cube")

        if self.standard_name:
            cube_xml_element.setAttribute('standard_name', self.standard_name)
        if self.long_name:
            cube_xml_element.setAttribute('long_name', self.long_name)
        if self.var_name:
            cube_xml_element.setAttribute('var_name', self.var_name)
        cube_xml_element.setAttribute('units', str(self.units))

        if self.attributes:
            attributes_element = doc.createElement('attributes')
            for name in sorted(self.attributes.iterkeys()):
                attribute_element = doc.createElement('attribute')
                attribute_element.setAttribute('name', name)
                value = str(self.attributes[name])
                attribute_element.setAttribute('value', value)
                attributes_element.appendChild(attribute_element)
            cube_xml_element.appendChild(attributes_element)

        coords_xml_element = doc.createElement("coords")
        for coord in sorted(self.coords(), key=lambda coord: coord.name()):
            # make a "cube coordinate" element which holds the dimensions (if
            # appropriate) which itself will have a sub-element of the
            # coordinate instance itself.
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

        data_xml_element = doc.createElement("data")

        data_xml_element.setAttribute("shape", str(self.shape))

        # NB. Getting a checksum triggers any deferred loading,
        # in which case it also has the side-effect of forcing the
        # byte order to be native.
        if checksum:
            data = self.data

            # Ensure consistent memory layout for checksums.
            def normalise(data):
                data = np.ascontiguousarray(data)
                if data.dtype.newbyteorder('<') != data.dtype:
                    data = data.byteswap(False)
                    data.dtype = data.dtype.newbyteorder('<')
                return data

            if isinstance(data, ma.MaskedArray):
                # Fill in masked values to avoid the checksum being
                # sensitive to unused numbers. Use a fixed value so
                # a change in fill_value doesn't affect the
                # checksum.
                crc = '0x%08x' % (
                    zlib.crc32(normalise(data.filled(0))) & 0xffffffff, )
                data_xml_element.setAttribute("checksum", crc)
                if ma.is_masked(data):
                    crc = '0x%08x' % (
                        zlib.crc32(normalise(data.mask)) & 0xffffffff, )
                else:
                    crc = 'no-masked-elements'
                data_xml_element.setAttribute("mask_checksum", crc)
                data_xml_element.setAttribute('fill_value',
                                              str(data.fill_value))
            else:
                crc = '0x%08x' % (zlib.crc32(normalise(data)) & 0xffffffff, )
                data_xml_element.setAttribute("checksum", crc)
        elif self.has_lazy_data():
            data_xml_element.setAttribute("state", "deferred")
        else:
            data_xml_element.setAttribute("state", "loaded")

        # Add the dtype, and also the array and mask orders if the
        # data is loaded.
        if not self.has_lazy_data():
            data = self.data
            dtype = data.dtype

            def _order(array):
                order = ''
                if array.flags['C_CONTIGUOUS']:
                    order = 'C'
                elif array.flags['F_CONTIGUOUS']:
                    order = 'F'
                return order
            if order:
                data_xml_element.setAttribute('order', _order(data))

            # NB. dtype.byteorder can return '=', which is bad for
            # cross-platform consistency - so we use dtype.str
            # instead.
            if byteorder:
                array_byteorder = {'>': 'big', '<': 'little'}.get(dtype.str[0])
                if array_byteorder is not None:
                    data_xml_element.setAttribute('byteorder', array_byteorder)

            if order and isinstance(data, ma.core.MaskedArray):
                data_xml_element.setAttribute('mask_order',
                                              _order(data.mask))
        else:
            dtype = self.lazy_data().dtype
        data_xml_element.setAttribute('dtype', dtype.name)

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
        raise copy.Error("Cube shallow-copy not allowed. Use deepcopy() or "
                         "Cube.copy()")

    def __deepcopy__(self, memo):
        return self._deepcopy(memo)

    def _deepcopy(self, memo, data=None):
        if data is None:
            # Use a copy of the source cube data.
            if self.has_lazy_data():
                # Use copy.copy, as lazy arrays don't have a copy method.
                new_cube_data = copy.copy(self.lazy_data())
            else:
                # Do *not* use copy.copy, as NumPy 0-d arrays do that wrong.
                new_cube_data = self.data.copy()
        else:
            # Use the provided data (without copying it).
            if not isinstance(data, biggus.Array):
                data = np.asanyarray(data)

            if data.shape != self.shape:
                msg = 'Cannot copy cube with new data of a different shape ' \
                      '(slice or subset the cube first).'
                raise ValueError(msg)

            new_cube_data = data

        new_dim_coords_and_dims = copy.deepcopy(self._dim_coords_and_dims,
                                                memo)
        new_aux_coords_and_dims = copy.deepcopy(self._aux_coords_and_dims,
                                                memo)

        # Record a mapping from old coordinate IDs to new coordinates,
        # for subsequent use in creating updated aux_factories.
        coord_mapping = {}
        for old_pair, new_pair in zip(self._dim_coords_and_dims,
                                      new_dim_coords_and_dims):
            coord_mapping[id(old_pair[0])] = new_pair[0]
        for old_pair, new_pair in zip(self._aux_coords_and_dims,
                                      new_aux_coords_and_dims):
            coord_mapping[id(old_pair[0])] = new_pair[0]

        new_cube = Cube(new_cube_data,
                        dim_coords_and_dims=new_dim_coords_and_dims,
                        aux_coords_and_dims=new_aux_coords_and_dims)
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
                result = not (coord_comparison['not_equal'] or
                              coord_comparison['non_equal_data_dimension'])

            # having checked everything else, check approximate data
            # equality - loading the data if has not already been loaded.
            if result:
                result = np.all(np.abs(self.data - other.data) < 1e-8)

        return result

    # Must supply __ne__, Python does not defer to __eq__ for negative equality
    def __ne__(self, other):
        result = self.__eq__(other)
        if result is not NotImplemented:
            result = not result
        return result

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

        .. deprecated:: 1.6
            Add/modify history metadata within
            attr:`~iris.cube.Cube.attributes` as needed.

        """
        warnings.warn("Cube.add_history() has been deprecated - "
                      "please modify/create cube.attributes['history'] "
                      "as needed.")

        timestamp = datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")
        string = '%s Iris: %s' % (timestamp, string)

        try:
            history = self.attributes['history']
            self.attributes['history'] = '%s\n%s' % (history, string)
        except KeyError:
            self.attributes['history'] = string

    # START ANALYSIS ROUTINES

    regridded = iris.util._wrap_function_for_method(
        iris.analysis.interpolate.regrid,
        """
        Returns a new cube with values derived from this cube on the
        horizontal grid specified by the grid_cube.

        """)

    # END ANALYSIS ROUTINES

    def collapsed(self, coords, aggregator, **kwargs):
        """
        Collapse one or more dimensions over the cube given the coordinate/s
        and an aggregation.

        Examples of aggregations that may be used include
        :data:`~iris.analysis.COUNT` and :data:`~iris.analysis.MAX`.

        Weighted aggregations (:class:`iris.analysis.WeightedAggregator`) may
        also be supplied. These include :data:`~iris.analysis.MEAN` and
        sum :data:`~iris.analysis.SUM`.

        Weighted aggregations support an optional *weights* keyword argument.
        If set, this should be supplied as an array of weights whose shape
        matches the cube. Values for latitude-longitude area weights may be
        calculated using :func:`iris.analysis.cartography.area_weights`.

        Some Iris aggregators support "lazy" evaluation, meaning that
        cubes resulting from this method may represent data arrays which are
        not computed until the data is requested (e.g. via ``cube.data`` or
        ``iris.save``). If lazy evaluation exists for the given aggregator
        it will be used wherever possible when this cube's data is itself
        a deferred array.

        Args:

        * coords (string, coord or a list of strings/coords):
            Coordinate names/coordinates over which the cube should be
            collapsed.

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
            >>> path = iris.sample_data_path('ostia_monthly.nc')
            >>> cube = iris.load_cube(path)
            >>> new_cube = cube.collapsed('longitude', iris.analysis.MEAN)
            >>> print(new_cube)
            surface_temperature / (K)           (time: 54; latitude: 18)
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
                 Cell methods:
                      mean: month, year
                      mean: longitude


        .. note::

            Some aggregations are not commutative and hence the order of
            processing is important i.e.::

                tmp = cube.collapsed('realization', iris.analysis.VARIANCE)
                result = tmp.collapsed('height', iris.analysis.VARIANCE)

            is not necessarily the same result as::

                tmp = cube.collapsed('height', iris.analysis.VARIANCE)
                result2 = tmp.collapsed('realization', iris.analysis.VARIANCE)

            Conversely operations which operate on more than one coordinate
            at the same time are commutative as they are combined internally
            into a single operation. Hence the order of the coordinates
            supplied in the list does not matter::

                cube.collapsed(['longitude', 'latitude'],
                               iris.analysis.VARIANCE)

            is the same (apart from the logically equivalent cell methods that
            may be created etc.) as::

                cube.collapsed(['latitude', 'longitude'],
                               iris.analysis.VARIANCE)

        """
        # Convert any coordinate names to coordinates
        coords = self._as_list_of_coords(coords)

        if (isinstance(aggregator, iris.analysis.WeightedAggregator) and
                not aggregator.uses_weighting(**kwargs)):
            msg = "Collapsing spatial coordinate {!r} without weighting"
            lat_match = [coord for coord in coords
                         if 'latitude' in coord.name()]
            if lat_match:
                for coord in lat_match:
                    warnings.warn(msg.format(coord.name()))

        # Determine the dimensions we need to collapse (and those we don't)
        if aggregator.cell_method == 'peak':
            dims_to_collapse = [list(self.coord_dims(coord))
                                for coord in coords]

            # Remove duplicate dimensions.
            new_dims = collections.OrderedDict.fromkeys(
                d for dim in dims_to_collapse for d in dim)
            # Reverse the dimensions so the order can be maintained when
            # reshaping the data.
            dims_to_collapse = list(new_dims)[::-1]
        else:
            dims_to_collapse = set()
            for coord in coords:
                dims_to_collapse.update(self.coord_dims(coord))

        if not dims_to_collapse:
            msg = 'Cannot collapse a dimension which does not describe any ' \
                  'data.'
            raise iris.exceptions.CoordinateCollapseError(msg)

        untouched_dims = set(range(self.ndim)) - set(dims_to_collapse)

        # Remove the collapsed dimension(s) from the metadata
        indices = [slice(None, None)] * self.ndim
        for dim in dims_to_collapse:
            indices[dim] = 0
        collapsed_cube = self[tuple(indices)]

        # Collapse any coords that span the dimension(s) being collapsed
        for coord in self.dim_coords + self.aux_coords:
            coord_dims = self.coord_dims(coord)
            if set(dims_to_collapse).intersection(coord_dims):
                local_dims = [coord_dims.index(dim) for dim in
                              dims_to_collapse if dim in coord_dims]
                collapsed_cube.replace_coord(coord.collapsed(local_dims))

        untouched_dims = sorted(untouched_dims)

        # Record the axis(s) argument passed to 'aggregation', so the same is
        # passed to the 'update_metadata' function.
        collapse_axis = -1

        data_result = None

        # Perform the actual aggregation.
        if aggregator.cell_method == 'peak':
            # The PEAK aggregator must collapse each coordinate separately.
            untouched_shape = [self.shape[d] for d in untouched_dims]
            collapsed_shape = [self.shape[d] for d in dims_to_collapse]
            new_shape = untouched_shape + collapsed_shape

            array_dims = untouched_dims + dims_to_collapse
            unrolled_data = np.transpose(
                self.data, array_dims).reshape(new_shape)

            for dim in dims_to_collapse:
                unrolled_data = aggregator.aggregate(unrolled_data,
                                                     axis=-1,
                                                     **kwargs)
            data_result = unrolled_data

        # Perform the aggregation in lazy form if possible.
        elif (aggregator.lazy_func is not None
                and len(dims_to_collapse) == 1 and self.has_lazy_data()):
            # Use a lazy operation separately defined by the aggregator, based
            # on the cube lazy array.
            # NOTE: do not reform the data in this case, as 'lazy_aggregate'
            # accepts multiple axes (unlike 'aggregate').
            collapse_axis = dims_to_collapse
            try:
                data_result = aggregator.lazy_aggregate(self.lazy_data(),
                                                        collapse_axis,
                                                        **kwargs)
            except TypeError:
                # TypeError - when unexpected keywords passed through (such as
                # weights to mean)
                pass

        # If we weren't able to complete a lazy aggregation, compute it
        # directly now.
        if data_result is None:
            # Perform the (non-lazy) aggregation over the cube data
            # First reshape the data so that the dimensions being aggregated
            # over are grouped 'at the end' (i.e. axis=-1).
            dims_to_collapse = sorted(dims_to_collapse)

            end_size = reduce(operator.mul, (self.shape[dim] for dim in
                                             dims_to_collapse))
            untouched_shape = [self.shape[dim] for dim in untouched_dims]
            new_shape = untouched_shape + [end_size]
            dims = untouched_dims + dims_to_collapse
            unrolled_data = np.transpose(self.data, dims).reshape(new_shape)

            # Perform the same operation on the weights if applicable
            if kwargs.get("weights") is not None:
                weights = kwargs["weights"].view()
                kwargs["weights"] = np.transpose(weights,
                                                 dims).reshape(new_shape)

            data_result = aggregator.aggregate(unrolled_data,
                                               axis=-1,
                                               **kwargs)
        aggregator.update_metadata(collapsed_cube, coords, axis=collapse_axis,
                                   **kwargs)
        result = aggregator.post_process(collapsed_cube, data_result, coords,
                                         **kwargs)
        return result

    def aggregated_by(self, coords, aggregator, **kwargs):
        """
        Perform aggregation over the cube given one or more "group
        coordinates".

        A "group coordinate" is a coordinate where repeating values represent a
        single group, such as a month coordinate on a daily time slice.
        TODO: It is not clear if repeating values must be consecutive to form a
        group.

        The group coordinates must all be over the same cube dimension. Each
        common value group identified over all the group-by coordinates is
        collapsed using the provided aggregator.

        Args:

        * coords (list of coord names or :class:`iris.coords.Coord` instances):
            One or more coordinates over which group aggregation is to be
            performed.
        * aggregator (:class:`iris.analysis.Aggregator`):
            Aggregator to be applied to each group.

        Kwargs:

        * kwargs:
            Aggregator and aggregation function keyword arguments.

        Returns:
            :class:`iris.cube.Cube`.

        .. note::

            This operation does not yet have support for lazy evaluation.

        For example:

            >>> import iris
            >>> import iris.analysis
            >>> import iris.coord_categorisation as cat
            >>> fname = iris.sample_data_path('ostia_monthly.nc')
            >>> cube = iris.load_cube(fname, 'surface_temperature')
            >>> cat.add_year(cube, 'time', name='year')
            >>> new_cube = cube.aggregated_by('year', iris.analysis.MEAN)
            >>> print(new_cube)
            surface_temperature / (K)           \
(time: 5; latitude: 18; longitude: 432)
                 Dimension coordinates:
                      time                      \
     x            -              -
                      latitude                  \
     -            x              -
                      longitude                 \
     -            -              x
                 Auxiliary coordinates:
                      forecast_reference_time   \
     x            -              -
                      year                      \
     x            -              -
                 Scalar coordinates:
                      forecast_period: 0 hours
                 Attributes:
                      Conventions: CF-1.5
                      STASH: m01s00i024
                 Cell methods:
                      mean: month, year
                      mean: year

        """
        groupby_coords = []
        dimension_to_groupby = None

        # We can't handle weights
        if isinstance(aggregator, iris.analysis.WeightedAggregator) and \
                aggregator.uses_weighting(**kwargs):
            raise ValueError('Invalid Aggregation, aggregated_by() cannot use'
                             ' weights.')

        coords = self._as_list_of_coords(coords)
        for coord in sorted(coords, key=lambda coord: coord._as_defn()):
            if coord.ndim > 1:
                msg = 'Cannot aggregate_by coord %s as it is ' \
                      'multidimensional.' % coord.name()
                raise iris.exceptions.CoordinateMultiDimError(msg)
            dimension = self.coord_dims(coord)
            if not dimension:
                msg = 'Cannot group-by the coordinate "%s", as its ' \
                      'dimension does not describe any data.' % coord.name()
                raise iris.exceptions.CoordinateCollapseError(msg)
            if dimension_to_groupby is None:
                dimension_to_groupby = dimension[0]
            if dimension_to_groupby != dimension[0]:
                msg = 'Cannot group-by coordinates over different dimensions.'
                raise iris.exceptions.CoordinateCollapseError(msg)
            groupby_coords.append(coord)

        # Determine the other coordinates that share the same group-by
        # coordinate dimension.
        shared_coords = list(filter(
            lambda coord_: coord_ not in groupby_coords,
            self.coords(dimensions=dimension_to_groupby)))

        # Create the aggregation group-by instance.
        groupby = iris.analysis._Groupby(groupby_coords, shared_coords)

        # Create the resulting aggregate-by cube and remove the original
        # coordinates that are going to be groupedby.
        key = [slice(None, None)] * self.ndim
        # Generate unique index tuple key to maintain monotonicity.
        key[dimension_to_groupby] = tuple(range(len(groupby)))
        key = tuple(key)
        aggregateby_cube = self[key]
        for coord in groupby_coords + shared_coords:
            aggregateby_cube.remove_coord(coord)

        # Determine the group-by cube data shape.
        data_shape = list(self.shape + aggregator.aggregate_shape(**kwargs))
        data_shape[dimension_to_groupby] = len(groupby)

        # Aggregate the group-by data.
        cube_slice = [slice(None, None)] * len(data_shape)

        for i, groupby_slice in enumerate(groupby.group()):
            # Slice the cube with the group-by slice to create a group-by
            # sub-cube.
            cube_slice[dimension_to_groupby] = groupby_slice
            groupby_sub_cube = self[tuple(cube_slice)]
            # Perform the aggregation over the group-by sub-cube and
            # repatriate the aggregated data into the aggregate-by cube data.
            cube_slice[dimension_to_groupby] = i
            result = aggregator.aggregate(groupby_sub_cube.data,
                                          axis=dimension_to_groupby,
                                          **kwargs)

            # Determine aggregation result data type for the aggregate-by cube
            # data on first pass.
            if i == 0:
                if isinstance(self.data, ma.MaskedArray):
                    aggregateby_data = ma.zeros(data_shape, dtype=result.dtype)
                else:
                    aggregateby_data = np.zeros(data_shape, dtype=result.dtype)

            aggregateby_data[tuple(cube_slice)] = result

        # Add the aggregation meta data to the aggregate-by cube.
        aggregator.update_metadata(aggregateby_cube,
                                   groupby_coords,
                                   aggregate=True, **kwargs)
        # Replace the appropriate coordinates within the aggregate-by cube.
        dim_coord, = self.coords(dimensions=dimension_to_groupby,
                                 dim_coords=True) or [None]
        for coord in groupby.coords:
            if dim_coord is not None and \
                    dim_coord._as_defn() == coord._as_defn() and \
                    isinstance(coord, iris.coords.DimCoord):
                aggregateby_cube.add_dim_coord(coord.copy(),
                                               dimension_to_groupby)
            else:
                aggregateby_cube.add_aux_coord(coord.copy(),
                                               dimension_to_groupby)

        # Attach the aggregate-by data into the aggregate-by cube.
        aggregateby_cube = aggregator.post_process(aggregateby_cube,
                                                   aggregateby_data,
                                                   coords, **kwargs)

        return aggregateby_cube

    def rolling_window(self, coord, aggregator, window, **kwargs):
        """
        Perform rolling window aggregation on a cube given a coordinate, an
        aggregation method and a window size.

        Args:

        * coord (string/:class:`iris.coords.Coord`):
            The coordinate over which to perform the rolling window
            aggregation.
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

        .. note::

            This operation does not yet have support for lazy evaluation.

        For example:

            >>> import iris, iris.analysis
            >>> fname = iris.sample_data_path('GloSea4', 'ensemble_010.pp')
            >>> air_press = iris.load_cube(fname, 'surface_temperature')
            >>> print(air_press)
            surface_temperature / (K)           \
(time: 6; latitude: 145; longitude: 192)
                 Dimension coordinates:
                      time                      \
     x            -               -
                      latitude                  \
     -            x               -
                      longitude                 \
     -            -               x
                 Auxiliary coordinates:
                      forecast_period           \
     x            -               -
                 Scalar coordinates:
                      forecast_reference_time: 2011-07-23 00:00:00
                      realization: 10
                 Attributes:
                      STASH: m01s00i024
                      source: Data from Met Office Unified Model
                      um_version: 7.6
                 Cell methods:
                      mean: time (1 hour)


            >>> print(air_press.rolling_window('time', iris.analysis.MEAN, 3))
            surface_temperature / (K)           \
(time: 4; latitude: 145; longitude: 192)
                 Dimension coordinates:
                      time                      \
     x            -               -
                      latitude                  \
     -            x               -
                      longitude                 \
     -            -               x
                 Auxiliary coordinates:
                      forecast_period           \
     x            -               -
                 Scalar coordinates:
                      forecast_reference_time: 2011-07-23 00:00:00
                      realization: 10
                 Attributes:
                      STASH: m01s00i024
                      source: Data from Met Office Unified Model
                      um_version: 7.6
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
                'must map to one data dimension.' % coord.name())
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

            if np.issubdtype(new_bounds.dtype, np.str):
                # Handle case where the AuxCoord contains string. The points
                # are the serialized form of the points contributing to each
                # window and the bounds are the first and last points in the
                # window as with numeric coordinates.
                new_points = np.apply_along_axis(lambda x: '|'.join(x), -1,
                                                 new_bounds)
                new_bounds = new_bounds[:, (0, -1)]
            else:
                # Take the first and last element of the rolled window (i.e.
                # the bounds) and the new points are the midpoints of these
                # bounds.
                new_bounds = new_bounds[:, (0, -1)]
                new_points = np.mean(new_bounds, axis=-1)

            # wipe the coords points and set the bounds
            new_coord = new_cube.coord(coord_)
            new_coord.points = new_points
            new_coord.bounds = new_bounds

        # update the metadata of the cube itself
        aggregator.update_metadata(
            new_cube, [coord],
            action='with a rolling window of length %s over' % window,
            **kwargs)
        # and perform the data transformation, generating weights first if
        # needed
        if isinstance(aggregator, iris.analysis.WeightedAggregator) and \
                aggregator.uses_weighting(**kwargs):
            if 'weights' in kwargs:
                weights = kwargs['weights']
                if weights.ndim > 1 or weights.shape[0] != window:
                    raise ValueError('Weights for rolling window aggregation '
                                     'must be a 1d array with the same length '
                                     'as the window.')
                kwargs = dict(kwargs)
                kwargs['weights'] = iris.util.broadcast_to_shape(
                    weights, rolling_window_data.shape, (dimension + 1,))
        data_result = aggregator.aggregate(rolling_window_data,
                                           axis=dimension + 1,
                                           **kwargs)
        result = aggregator.post_process(new_cube, data_result, [coord],
                                         **kwargs)
        return result

    def interpolate(self, sample_points, scheme, collapse_scalar=True):
        """
        Interpolate from this :class:`~iris.cube.Cube` to the given
        sample points using the given interpolation scheme.

        Args:

        * sample_points:
            A sequence of (coordinate, points) pairs over which to
            interpolate. The values for coordinates that correspond to
            dates or times may optionally be supplied as datetime.datetime or
            netcdftime.datetime instances.
        * scheme:
            The type of interpolation to use to interpolate from this
            :class:`~iris.cube.Cube` to the given sample points. The
            interpolation schemes currently available in Iris are:
                * :class:`iris.analysis.Linear`, and
                * :class:`iris.analysis.Nearest`.

        Kwargs:

        * collapse_scalar:
            Whether to collapse the dimension of scalar sample points
            in the resulting cube. Default is True.

        Returns:
            A cube interpolated at the given sample points.
            If `collapse_scalar` is True then the dimensionality of the cube
            will be the number of original cube dimensions minus
            the number of scalar coordinates.

        For example:

            >>> import datetime
            >>> import iris
            >>> path = iris.sample_data_path('uk_hires.pp')
            >>> cube = iris.load_cube(path, 'air_potential_temperature')
            >>> print(cube.summary(shorten=True))
            air_potential_temperature / (K)     \
(time: 3; model_level_number: 7; grid_latitude: 204; grid_longitude: 187)
            >>> print(cube.coord('time'))
            DimCoord([2009-11-19 10:00:00, 2009-11-19 11:00:00, \
2009-11-19 12:00:00], standard_name='time', calendar='gregorian')
            >>> print(cube.coord('time').points)
            [ 349618.  349619.  349620.]
            >>> samples = [('time', 349618.5)]
            >>> result = cube.interpolate(samples, iris.analysis.Linear())
            >>> print(result.summary(shorten=True))
            air_potential_temperature / (K)     \
(model_level_number: 7; grid_latitude: 204; grid_longitude: 187)
            >>> print(result.coord('time'))
            DimCoord([2009-11-19 10:30:00], standard_name='time', \
calendar='gregorian')
            >>> print(result.coord('time').points)
            [ 349618.5]
            >>> # For datetime-like coordinates, we can also use
            >>> # datetime-like objects.
            >>> samples = [('time', datetime.datetime(2009, 11, 19, 10, 30))]
            >>> result2 = cube.interpolate(samples, iris.analysis.Linear())
            >>> print(result2.summary(shorten=True))
            air_potential_temperature / (K)     \
(model_level_number: 7; grid_latitude: 204; grid_longitude: 187)
            >>> print(result2.coord('time'))
            DimCoord([2009-11-19 10:30:00], standard_name='time', \
calendar='gregorian')
            >>> print(result2.coord('time').points)
            [ 349618.5]
            >>> print(result == result2)
            True

        """
        coords, points = zip(*sample_points)
        interp = scheme.interpolator(self, coords)
        return interp(points, collapse_scalar=collapse_scalar)

    def regrid(self, grid, scheme):
        """
        Regrid this :class:`~iris.cube.Cube` on to the given target `grid`
        using the given regridding `scheme`.

        Args:

        * grid:
            A :class:`~iris.cube.Cube` that defines the target grid.
        * scheme:
            The type of regridding to use to regrid this cube onto the
            target grid. The regridding schemes currently available
            in Iris are:
                * :class:`iris.analysis.Linear`,
                * :class:`iris.analysis.Nearest`, and
                * :class:`iris.analysis.AreaWeighted`.

        Returns:
            A cube defined with the horizontal dimensions of the target grid
            and the other dimensions from this cube. The data values of
            this cube will be converted to values on the new grid
            according to the given regridding scheme.

        """
        regridder = scheme.regridder(self, grid)
        return regridder(self)


class ClassDict(collections.MutableMapping, object):
    """
    A mapping that stores objects keyed on their superclasses and their names.

    The mapping has a root class, all stored objects must be a subclass of the
    root class. The superclasses used for an object include the class of the
    object, but do not include the root class. Only one object is allowed for
    any key.

    """
    def __init__(self, superclass):
        if not isinstance(superclass, type):
            raise TypeError("The superclass must be a Python type or new "
                            "style class.")
        self._superclass = superclass
        self._basic_map = {}
        self._retrieval_map = {}

    def add(self, object_, replace=False):
        '''Add an object to the dictionary.'''
        if not isinstance(object_, self._superclass):
            msg = "Only subclasses of {!r} are allowed as values.".format(
                self._superclass.__name__)
            raise TypeError(msg)
        # Find all the superclasses of the given object, starting with the
        # object's class.
        superclasses = type.mro(type(object_))
        if not replace:
            # Ensure nothing else is already registered against those
            # superclasses.
            # NB. This implies the _basic_map will also be empty for this
            # object.
            for key_class in superclasses:
                if key_class in self._retrieval_map:
                    msg = "Cannot add instance of '%s' because instance of " \
                          "'%s' already added." % (type(object_).__name__,
                                                   key_class.__name__)
                    raise ValueError(msg)
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

    def __setitem__(self, key, value):
        raise NotImplementedError('You must call the add method instead.')

    def __delitem__(self, class_):
        cs = self[class_]
        keys = [k for k, v in self._retrieval_map.iteritems() if v == cs]
        for key in keys:
            del self._retrieval_map[key]
        del self._basic_map[type(cs)]
        return cs

    def __len__(self):
        return len(self._basic_map)

    def __iter__(self):
        for item in self._basic_map:
            yield item

    def keys(self):
        '''Return the keys of the dictionary mapping.'''
        return self._basic_map.keys()


def sorted_axes(axes):
    """
    Returns the axis names sorted alphabetically, with the exception that
    't', 'z', 'y', and, 'x' are sorted to the end.

    """
    return sorted(axes, key=lambda name: ({'x': 4,
                                           'y': 3,
                                           'z': 2,
                                           't': 1}.get(name, 0), name))


# See Cube.slice() for the definition/context.
class _SliceIterator(collections.Iterator):
    def __init__(self, cube, dims_index, requested_dims, ordered):
        self._cube = cube

        # Let Numpy do some work in providing all of the permutations of our
        # data shape. This functionality is something like:
        # ndindex(2, 1, 3) -> [(0, 0, 0), (0, 0, 1), (0, 0, 2),
        #                      (1, 0, 0), (1, 0, 1), (1, 0, 2)]
        self._ndindex = np.ndindex(*dims_index)

        self._requested_dims = requested_dims
        # indexing relating to sliced cube
        self._mod_requested_dims = np.argsort(requested_dims)
        self._ordered = ordered

    def next(self):
        # NB. When self._ndindex runs out it will raise StopIteration for us.
        index_tuple = next(self._ndindex)

        # Turn the given tuple into a list so that we can do something with it
        index_list = list(index_tuple)

        # For each of the spanning dimensions requested, replace the 0 with a
        # spanning slice
        for d in self._requested_dims:
            index_list[d] = slice(None, None)

        # Request the slice
        cube = self._cube[tuple(index_list)]

        if self._ordered:
            if any(self._mod_requested_dims != list(range(len(cube.shape)))):
                cube.transpose(self._mod_requested_dims)

        return cube
