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
Definitions of derived coordinates.

"""

from abc import ABCMeta, abstractmethod, abstractproperty
import warnings
import zlib

import numpy as np

from iris._cube_coord_common import CFVariableMixin
import iris.coords
import iris.util


class LazyArray(object):
    """
    Represents a simplified NumPy array which is only computed on demand.

    It provides the :meth:`view()` and :meth:`reshape()` methods so it
    can be used in place of a standard NumPy array under some
    circumstances.

    The first use of either of these methods causes the array to be
    computed and cached for any subsequent access.
    
    """

    def __init__(self, shape, func):
        """
        Args:
        
        * shape (tuple):
            The shape of the array which will be created.
        * func:
            The function which will be called to supply the real array.

        """
        self.shape = tuple(shape)
        self._func = func
        self._array = None

    def __repr__(self):
        return '<LazyArray(shape={})>'.format(self.shape)

    def _cached_array(self):
        if self._array is None:
            self._array = self._func()
            del self._func
        return self._array

    def reshape(self, *args, **kwargs):
        """
        Returns a view of this array with the given shape.

        See :meth:`numpy.ndarray.reshape()` for argument details.

        """
        return self._cached_array().reshape(*args, **kwargs)

    def to_xml_attr(self):
        """
        Returns a string describing this array, suitable for use in CML.

        """
        crc = zlib.crc32(np.array(self._cached_array(), order='C'))
        return 'LazyArray(shape={}, checksum={})'.format(self.shape, crc)

    def view(self, *args, **kwargs):
        """
        Returns a view of this array.

        See :meth:`numpy.ndarray.view()` for argument details.

        """
        return self._cached_array().view(*args, **kwargs)


class AuxCoordFactory(CFVariableMixin):
    """
    Represents a "factory" which can manufacture an additional auxiliary
    coordinate on demand, by combining the values of other coordinates.

    Each concrete subclass represents a specific formula for deriving
    values from other coordinates.

    The `standard_name`, `long_name`, `var_name`, `units`, `attributes` and
    `coord_system` of the factory are used to set the corresponding
    properties of the resulting auxiliary coordinates.
    
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        #: Descriptive name of the coordinate made by the factory
        self.long_name = None

        #: CF variable name of the coordinate made by the factory
        self.var_name = None

        #: Coordinate system (if any) of the coordinate made by the factory
        self.coord_system = None

    @abstractproperty
    def dependencies(self):
        """
        Returns a dictionary mapping from constructor argument names to
        the corresponding coordinates.

        """

    def _as_defn(self):
        defn = iris.coords.CoordDefn(self.standard_name, self.long_name,
                                     self.var_name, self.units,
                                     self.attributes, self.coord_system)
        return defn

    @abstractmethod
    def make_coord(self, coord_dims_func):
        """
        Returns a new :class:`iris.coords.AuxCoord` as defined by this
        factory.

        Args:

        * coord_dims_func:
            A callable which can return the list of dimensions relevant
            to a given coordinate. 
            See :meth:`iris.cube.Cube.coord_dims()`.
        
        """

    @abstractmethod
    def update(self, old_coord, new_coord=None):
        """
        Notifies the factory of a removal/replacement of a dependency.

        Args:

        * old_coord:
            The dependency coordinate to be removed/replaced.
        * new_coord:
            If None, the dependency using old_coord is removed, othewise
            the dependency is updated to use new_coord.

        """

    def __repr__(self):
        def arg_text(item):
            key, coord = item
            return '{}={}'.format(key, str(coord and repr(coord.name())))
        items = self.dependencies.items()
        items.sort(key=lambda item: item[0])
        args = map(arg_text, items)
        return '<{}({})>'.format(type(self).__name__, ', '.join(args))

    def derived_dims(self, coord_dims_func):
        """
        Returns the virtual dim-mapping for the derived coordinate.

        Args:

        * coord_dims_func:
            A callable which can return the list of dimensions relevant
            to a given coordinate. 
            See :meth:`iris.cube.Cube.coord_dims()`.
        
        """
        # Which dimensions are relevant?
        # e.g. If sigma -> [1] and orog -> [2, 3] then result = [1, 2, 3]
        derived_dims = set()
        for coord in self.dependencies.itervalues():
            if coord:
                derived_dims.update(coord_dims_func(coord))

        # Apply a fixed order so we know how to map dependency dims to
        # our own dims (and so the Cube can map them to Cube dims).
        derived_dims = tuple(sorted(derived_dims))
        return derived_dims

    def updated(self, new_coord_mapping):
        """
        Creates a new instance of this factory where the dependencies
        are replaced according to the given mapping.

        Args:

        * new_coord_mapping:
            A dictionary mapping from the object IDs potentially used
            by this factory, to the coordinate objects that should be
            used instead.

        """
        new_dependencies = {}
        for key, coord in self.dependencies.iteritems():
            if coord:
                coord = new_coord_mapping[id(coord)]
            new_dependencies[key] = coord
        return type(self)(**new_dependencies)

    def xml_element(self, doc):
        """
        Returns a DOM element describing this coordinate factory.

        """
        element = doc.createElement('coordFactory')
        for key, coord in self.dependencies.iteritems():
            element.setAttribute(key, coord._xml_id())
        element.appendChild(self.make_coord().xml_element(doc))
        return element

    def _dependency_dims(self, coord_dims_func):
        dependency_dims = {}
        for key, coord in self.dependencies.iteritems():
            if coord:
                dependency_dims[key] = coord_dims_func(coord)
        return dependency_dims

    def _nd_bounds(self, coord, dims, ndim):
        """
        Returns the coord's bounds in Cube-orientation and
        broadcastable to N dimensions.
        
        Example:
            coord.shape == (70,)
            coord.nbounds = 2
            dims == [3]
            ndim == 5
        results in:
            nd_bounds.shape == (1, 1, 1, 70, 1, 2)

        """
        # Transpose to be consistent with the Cube.
        sorted_pairs = sorted(enumerate(dims), key=lambda pair: pair[1])
        transpose_order = [pair[0] for pair in sorted_pairs] + [len(dims)]
        bounds = coord.bounds
        if dims:
            bounds = bounds.transpose(transpose_order)

        # Figure out the n-dimensional shape.
        nd_shape = [1] * ndim + [coord.nbounds]
        for dim, size in zip(dims, coord.shape):
            nd_shape[dim] = size
        bounds.shape = tuple(nd_shape)
        return bounds

    def _nd_points(self, coord, dims, ndim):
        """
        Returns the coord's points in Cube-orientation and
        broadcastable to N dimensions.
        
        Example:
            coord.shape == (4, 3)
            dims == [3, 2]
            ndim == 5
        results in:
            nd_points.shape == (1, 1, 3, 4, 1)

        """
        # Transpose to be consistent with the Cube.
        sorted_pairs = sorted(enumerate(dims), key=lambda pair: pair[1])
        transpose_order = [pair[0] for pair in sorted_pairs]
        points = coord.points
        if dims:
            points = points.transpose(transpose_order)

        # Figure out the n-dimensional shape.
        nd_shape = [1] * ndim
        for dim, size in zip(dims, coord.shape):
            nd_shape[dim] = size
        points.shape = tuple(nd_shape)
        return points

    def _remap(self, dependency_dims, derived_dims):
        if derived_dims:
            ndim = max(derived_dims) + 1
        else:
            ndim = 1

        nd_points_by_key = {}
        for key, coord in self.dependencies.iteritems():
            if coord:
                # Get the points as consistent with the Cube.
                nd_points = self._nd_points(coord, dependency_dims[key], ndim)

                # Restrict to just the dimensions relevant to the derived coord.
                # NB. These are always in Cube-order, so no transpose is needed.
                shape = []
                for dim in derived_dims:
                    shape.append(nd_points.shape[dim])
                # Ensure the array always has at least one dimension to be
                # compatible with normal coordinates.
                if not derived_dims:
                    shape.append(1)
                nd_points.shape = shape
            else:
                # If no coord, treat value as zero.
                # Use a float16 to provide `shape` attribute and avoid
                # promoting other arguments to a higher precision.
                nd_points = np.float16(0)

            nd_points_by_key[key] = nd_points
        return nd_points_by_key

    def _remap_with_bounds(self, dependency_dims, derived_dims):
        if derived_dims:
            ndim = max(derived_dims) + 1
        else:
            ndim = 1

        nd_values_by_key = {}
        for key, coord in self.dependencies.iteritems():
            if coord:
                # Get the bounds or points as consistent with the Cube.
                if coord.nbounds:
                    nd_values = self._nd_bounds(coord, dependency_dims[key], ndim)
                else:
                    nd_values = self._nd_points(coord, dependency_dims[key], ndim)

                # Restrict to just the dimensions relevant to the derived coord.
                # NB. These are always in Cube-order, so no transpose is needed.
                shape = []
                for dim in derived_dims:
                    shape.append(nd_values.shape[dim])
                # Ensure the array always has at least one dimension to be
                # compatible with normal coordinates.
                if not derived_dims:
                    shape.append(1)
                # Add on the N-bounds dimension
                if coord.nbounds:
                    shape.append(nd_values.shape[-1])
                else:
                    # NB. For a non-bounded coordinate we still need an
                    # extra dimension to make the shape compatible, so
                    # we just add an extra 1.
                    shape.append(1)
                nd_values.shape = shape
            else:
                # If no coord, treat value as zero.
                # Use a float16 to provide `shape` attribute and avoid
                # promoting other arguments to a higher precision.
                nd_values = np.float16(0)

            nd_values_by_key[key] = nd_values
        return nd_values_by_key

    def _shape(self, nd_values_by_key):
        nd_values = nd_values_by_key.values()
        shape = list(nd_values.pop().shape)
        for array in nd_values:
            for i, size in enumerate(array.shape):
                if size > 1:
                    # NB. If there's an inconsistency it can only come
                    # from a mismatch in the number of bounds (the Cube
                    # ensures the other dimensions must match).
                    # But we can't afford to raise an error now - it'd 
                    # break Cube.derived_coords. Instead, we let the
                    # error happen when the derived coordinate's bounds
                    # are accessed.
                    shape[i] = size
        return shape


class HybridHeightFactory(AuxCoordFactory):
    """
    Defines a hybrid-height coordinate factory with the formula:
        z = a + b * orog

    """

    def __init__(self, delta=None, sigma=None, orography=None):
        """
        Creates a hybrid-height coordinate factory with the formula:
            z = a + b * orog

        At least one of `delta` or `orography` must be provided.

        Args:

        * delta: Coord
            The coordinate providing the `a` term.
        * sigma: Coord
            The coordinate providing the `b` term.
        * orography: Coord
            The coordinate providing the `orog` term.

        """
        super(HybridHeightFactory, self).__init__()

        if delta and delta.nbounds not in (0, 2):
            raise ValueError('Invalid delta coordinate: must have either 0 or'
                             ' 2 bounds.')
        if sigma and sigma.nbounds not in (0, 2):
            raise ValueError('Invalid sigma coordinate: must have either 0 or'
                             ' 2 bounds.')
        if orography and orography.nbounds:
            warnings.warn('Orography coordinate {!r} has bounds.'
                          'These will be disregarded.'.format(orography.name()), 
                          UserWarning, stacklevel=2)

        self.delta = delta
        self.sigma = sigma
        self.orography = orography

        self.standard_name = 'altitude'
        if delta is None and orography is None:
            raise ValueError('Unable to determine units: no delta or orography'
                             ' available.')
        if delta and orography and delta.units != orography.units:
            raise ValueError('Incompatible units: delta and orography must have'
                             ' the same units.')
        self.units = (delta and delta.units) or orography.units
        if not self.units.is_convertible('m'):
            raise ValueError('Invalid units: delta and/or orography'
                             ' must be expressed in length units.')
        self.attributes = {'positive': 'up'}

    @property
    def dependencies(self):
        """
        Returns a dictionary mapping from constructor argument names to
        the corresponding coordinates.

        """
        return {'delta': self.delta, 'sigma': self.sigma, 'orography': self.orography}

    def _derive(self, delta, sigma, surface_pressure):
        temp = delta + sigma * surface_pressure
        return temp

    def make_coord(self, coord_dims_func):
        """
        Returns a new :class:`iris.coords.AuxCoord` as defined by this
        factory.

        Args:

        * coord_dims_func:
            A callable which can return the list of dimensions relevant
            to a given coordinate. 
            See :meth:`iris.cube.Cube.coord_dims()`.
        
        """
        # Which dimensions are relevant?
        derived_dims = self.derived_dims(coord_dims_func)

        dependency_dims = self._dependency_dims(coord_dims_func)

        # Build a "lazy" points array.
        nd_points_by_key = self._remap(dependency_dims, derived_dims)
        # Define the function here to obtain a closure.
        def calc_points():
            return self._derive(nd_points_by_key['delta'],
                                nd_points_by_key['sigma'],
                                nd_points_by_key['orography'])
        shape = self._shape(nd_points_by_key)
        points = LazyArray(shape, calc_points)

        bounds = None
        if ((self.delta and self.delta.nbounds) or
                (self.sigma and self.sigma.nbounds)):
            # Build a "lazy" bounds array.
            nd_values_by_key = self._remap_with_bounds(dependency_dims, derived_dims)
            # Define the function here to obtain a closure.
            def calc_bounds():
                delta = nd_values_by_key['delta']
                sigma = nd_values_by_key['sigma']
                orography = nd_values_by_key['orography']
                ok_bound_shapes = [(), (1,), (2,)]
                if delta.shape[-1:] not in ok_bound_shapes:
                    raise ValueError('Invalid delta coordinate bounds.')
                if sigma.shape[-1:] not in ok_bound_shapes:
                    raise ValueError('Invalid sigma coordinate bounds.')
                if orography.shape[-1:] not in [(), (1,)]:
                    warnings.warn('Orography coordinate has bounds. '
                                  'These are being disregarded.', UserWarning, stacklevel=2)
                    orography_pts_shape = list(nd_points_by_key['orography'].shape)
                    orography = nd_points_by_key['orography'].reshape(orography_pts_shape.append(1))
                return self._derive(delta, sigma, orography)
            b_shape = self._shape(nd_values_by_key)
            bounds = LazyArray(b_shape, calc_bounds)

        hybrid_height = iris.coords.AuxCoord(points,
                                             standard_name=self.standard_name,
                                             long_name=self.long_name,
                                             var_name=self.var_name,
                                             units=self.units,
                                             bounds=bounds,
                                             attributes=self.attributes,
                                             coord_system=self.coord_system)
        return hybrid_height

    def update(self, old_coord, new_coord=None):
        """
        Notifies the factory of the removal/replacement of a coordinate
        which might be a dependency.

        Args:

        * old_coord:
            The coordinate to be removed/replaced.
        * new_coord:
            If None, any dependency using old_coord is removed, othewise
            any dependency using old_coord is updated to use new_coord.

        """
        if self.delta is old_coord:
            if new_coord and new_coord.nbounds not in (0, 2):
                raise ValueError('Invalid delta coordinate: must have either 0 or 2 bounds.')
            self.delta = new_coord
        elif self.sigma is old_coord:
            if new_coord and new_coord.nbounds not in (0, 2):
                raise ValueError('Invalid sigma coordinate: must have either 0 or 2 bounds.')
            self.sigma = new_coord
        elif self.orography is old_coord:
            if new_coord and new_coord.nbounds:
                warnings.warn('Orography coordinate {!r} has bounds. '
                              'These will be disregarded.'.format(new_coord.name()), 
                              UserWarning, stacklevel=2)
            self.orography = new_coord


class HybridPressureFactory(AuxCoordFactory):
    """
    Defines a hybrid-pressure coordinate factory with the formula:
        p = ap + b * ps

    """

    def __init__(self, delta=None, sigma=None, surface_pressure=None):
        """
        Creates a hybrid-height coordinate factory with the formula:
            p = ap + b * ps

        At least one of `delta` or `surface_pressure` must be provided.

        Args:

        * delta: Coord
            The coordinate providing the `ap` term.
        * sigma: Coord
            The coordinate providing the `b` term.
        * surface_pressure: Coord
            The coordinate providing the `ps` term.

        """
        super(HybridPressureFactory, self).__init__()

        if delta and delta.nbounds not in (0, 2):
            raise ValueError('Invalid delta coordinate: must have either 0 or'
                             ' 2 bounds.')
        if sigma and sigma.nbounds not in (0, 2):
            raise ValueError('Invalid sigma coordinate: must have either 0 or'
                             ' 2 bounds.')
        if surface_pressure and surface_pressure.nbounds:
            warnings.warn('Surface pressure coordinate {!r} has bounds.'
                          'These will be disregarded.'.format(surface_pressure.name()),
                          UserWarning, stacklevel=2)

        self.delta = delta
        self.sigma = sigma
        self.surface_pressure = surface_pressure

        self.standard_name = 'air_pressure'
        if delta is None and surface_pressure is None:
            raise ValueError('Unable to determine units: no delta or'
                             ' surface_pressure available.')
        if delta and surface_pressure and delta.units != surface_pressure.units:
            raise ValueError('Incompatible units: delta and surface_pressure'
                             ' must have the same units.')
        self.units = (delta and delta.units) or surface_pressure.units
        if not self.units.is_convertible('Pa'):
            raise ValueError('Invalid units: delta and/or surface_pressure'
                             ' must be expressed in pressure units.')
        self.attributes = {}

    @property
    def dependencies(self):
        """
        Returns a dictionary mapping from constructor argument names to
        the corresponding coordinates.

        """
        return {'delta': self.delta, 'sigma': self.sigma,
                'surface_pressure': self.surface_pressure}

    def _derive(self, delta, sigma, surface_pressure):
        temp = delta + sigma * surface_pressure
        return temp

    def make_coord(self, coord_dims_func):
        """
        Returns a new :class:`iris.coords.AuxCoord` as defined by this
        factory.

        Args:

        * coord_dims_func:
            A callable which can return the list of dimensions relevant
            to a given coordinate. 
            See :meth:`iris.cube.Cube.coord_dims()`.
        
        """
        # Which dimensions are relevant?
        derived_dims = self.derived_dims(coord_dims_func)

        dependency_dims = self._dependency_dims(coord_dims_func)

        # Build a "lazy" points array.
        nd_points_by_key = self._remap(dependency_dims, derived_dims)
        # Define the function here to obtain a closure.
        def calc_points():
            return self._derive(nd_points_by_key['delta'],
                                nd_points_by_key['sigma'],
                                nd_points_by_key['surface_pressure'])
        shape = self._shape(nd_points_by_key)
        points = LazyArray(shape, calc_points)

        bounds = None
        if ((self.delta and self.delta.nbounds) or
                (self.sigma and self.sigma.nbounds)):
            # Build a "lazy" bounds array.
            nd_values_by_key = self._remap_with_bounds(dependency_dims, derived_dims)
            # Define the function here to obtain a closure.
            def calc_bounds():
                delta = nd_values_by_key['delta']
                sigma = nd_values_by_key['sigma']
                surface_pressure = nd_values_by_key['surface_pressure']
                ok_bound_shapes = [(), (1,), (2,)]
                if delta.shape[-1:] not in ok_bound_shapes:
                    raise ValueError('Invalid delta coordinate bounds.')
                if sigma.shape[-1:] not in ok_bound_shapes:
                    raise ValueError('Invalid sigma coordinate bounds.')
                if surface_pressure.shape[-1:] not in [(), (1,)]:
                    warnings.warn('Surface pressure coordinate has bounds. '
                                  'These are being disregarded.')
                    surface_pressure_pts_shape = list(nd_points_by_key['surface_pressure'].shape)
                    surface_pressure = nd_points_by_key['surface_pressure'].reshape(surface_pressure_pts_shape.append(1))
                return self._derive(delta, sigma, surface_pressure)
            b_shape = self._shape(nd_values_by_key)
            bounds = LazyArray(b_shape, calc_bounds)

        hybrid_pressure = iris.coords.AuxCoord(points,
                                               standard_name=self.standard_name,
                                               long_name=self.long_name,
                                               var_name=self.var_name,
                                               units=self.units,
                                               bounds=bounds,
                                               attributes=self.attributes,
                                               coord_system=self.coord_system)
        return hybrid_pressure

    def update(self, old_coord, new_coord=None):
        """
        Notifies the factory of the removal/replacement of a coordinate
        which might be a dependency.

        Args:

        * old_coord:
            The coordinate to be removed/replaced.
        * new_coord:
            If None, any dependency using old_coord is removed, othewise
            any dependency using old_coord is updated to use new_coord.

        """
        if self.delta is old_coord:
            if new_coord and new_coord.nbounds not in (0, 2):
                raise ValueError('Invalid delta coordinate: must have either 0 or 2 bounds.')
            self.delta = new_coord
        elif self.sigma is old_coord:
            if new_coord and new_coord.nbounds not in (0, 2):
                raise ValueError('Invalid sigma coordinate: must have either 0 or 2 bounds.')
            self.sigma = new_coord
        elif self.surface_pressure is old_coord:
            if new_coord and new_coord.nbounds:
                warnings.warn('Surface pressure coordinate {!r} has bounds. '
                              'These will be disregarded.'.format(new_coord.name()),
                              UserWarning, stacklevel=2)
            self.surface_pressure = new_coord
