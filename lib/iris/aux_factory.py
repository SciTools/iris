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
Definitions of derived coordinates.

"""

from __future__ import (absolute_import, division, print_function)

from abc import ABCMeta, abstractmethod, abstractproperty
import warnings
import zlib

import numpy as np

from iris._cube_coord_common import CFVariableMixin
import iris.coords
import iris.unit
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
    def __init__(self, shape, func, dtype=None):
        """
        Args:

        * shape (tuple):
            The shape of the array which will be created.
        * func:
            The function which will be called to supply the real array.

        Kwargs:

        * dtype (np.dtype):
            The numpy dtype of the array which will be created.
            Defaults to None to signify the dtype is unknown.

        """
        self.shape = tuple(shape)
        self._func = func
        self.dtype = dtype
        self._array = None

    def __repr__(self):
        return '<LazyArray(shape={}, dtype={!r})>'.format(self.shape,
                                                          self.dtype)

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
            If None, the dependency using old_coord is removed, otherwise
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

                # Restrict to just the dimensions relevant to the
                # derived coord. NB. These are always in Cube-order, so
                # no transpose is needed.
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
                    nd_values = self._nd_bounds(coord, dependency_dims[key],
                                                ndim)
                else:
                    nd_values = self._nd_points(coord, dependency_dims[key],
                                                ndim)

                # Restrict to just the dimensions relevant to the
                # derived coord. NB. These are always in Cube-order, so
                # no transpose is needed.
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
        nd_values = sorted(nd_values_by_key.values(),
                           key=lambda value: value.ndim)
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

    def _dtype(self, arrays_by_key, **other_args):
        dummy_args = {}
        for key, array in arrays_by_key.iteritems():
            dummy_args[key] = np.zeros(1, dtype=array.dtype)
        dummy_args.update(other_args)
        dummy_data = self._derive(**dummy_args)
        return dummy_data.dtype


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
            msg = 'Orography coordinate {!r} has bounds.' \
                  ' These will be disregarded.'.format(orography.name())
            warnings.warn(msg, UserWarning, stacklevel=2)

        self.delta = delta
        self.sigma = sigma
        self.orography = orography

        self.standard_name = 'altitude'
        if delta is None and orography is None:
            raise ValueError('Unable to determine units: no delta or orography'
                             ' available.')
        if delta and orography and delta.units != orography.units:
            raise ValueError('Incompatible units: delta and orography must'
                             ' have the same units.')
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
        return {'delta': self.delta, 'sigma': self.sigma,
                'orography': self.orography}

    def _derive(self, delta, sigma, orography):
        temp = delta + sigma * orography
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
        dtype = self._dtype(nd_points_by_key)
        points = LazyArray(shape, calc_points, dtype)

        bounds = None
        if ((self.delta and self.delta.nbounds) or
                (self.sigma and self.sigma.nbounds)):
            # Build a "lazy" bounds array.
            nd_values_by_key = self._remap_with_bounds(dependency_dims,
                                                       derived_dims)

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
                                  'These are being disregarded.',
                                  UserWarning, stacklevel=2)
                    orography_pts = nd_points_by_key['orography']
                    orography_pts_shape = list(orography_pts.shape)
                    orography = orography_pts.reshape(
                        orography_pts_shape.append(1))
                return self._derive(delta, sigma, orography)
            b_shape = self._shape(nd_values_by_key)
            b_dtype = self._dtype(nd_values_by_key)
            bounds = LazyArray(b_shape, calc_bounds, b_dtype)

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
            If None, any dependency using old_coord is removed, otherwise
            any dependency using old_coord is updated to use new_coord.

        """
        if self.delta is old_coord:
            if new_coord and new_coord.nbounds not in (0, 2):
                raise ValueError('Invalid delta coordinate:'
                                 ' must have either 0 or 2 bounds.')
            self.delta = new_coord
        elif self.sigma is old_coord:
            if new_coord and new_coord.nbounds not in (0, 2):
                raise ValueError('Invalid sigma coordinate:'
                                 ' must have either 0 or 2 bounds.')
            self.sigma = new_coord
        elif self.orography is old_coord:
            if new_coord and new_coord.nbounds:
                msg = 'Orography coordinate {!r} has bounds.' \
                      ' These will be disregarded.'.format(new_coord.name())
                warnings.warn(msg, UserWarning, stacklevel=2)
            self.orography = new_coord


class HybridPressureFactory(AuxCoordFactory):
    """
    Defines a hybrid-pressure coordinate factory with the formula:
        p = ap + b * ps

    """
    def __init__(self, delta=None, sigma=None, surface_air_pressure=None):
        """
        Creates a hybrid-height coordinate factory with the formula:
            p = ap + b * ps

        At least one of `delta` or `surface_air_pressure` must be provided.

        Args:

        * delta: Coord
            The coordinate providing the `ap` term.
        * sigma: Coord
            The coordinate providing the `b` term.
        * surface_air_pressure: Coord
            The coordinate providing the `ps` term.

        """
        super(HybridPressureFactory, self).__init__()

        # Check that provided coords meet necessary conditions.
        self._check_dependencies(delta, sigma, surface_air_pressure)

        self.delta = delta
        self.sigma = sigma
        self.surface_air_pressure = surface_air_pressure

        self.standard_name = 'air_pressure'
        self.attributes = {}

    @property
    def units(self):
        if self.delta is not None:
            units = self.delta.units
        else:
            units = self.surface_air_pressure.units
        return units

    @staticmethod
    def _check_dependencies(delta, sigma,
                            surface_air_pressure):
        # Check for sufficient coordinates.
        if (delta is None and (sigma is None or
                               surface_air_pressure is None)):
            msg = 'Unable to contruct hybrid pressure coordinate factory ' \
                  'due to insufficient source coordinates.'
            raise ValueError(msg)

        # Check bounds.
        if delta and delta.nbounds not in (0, 2):
            raise ValueError('Invalid delta coordinate: must have either 0 or'
                             ' 2 bounds.')
        if sigma and sigma.nbounds not in (0, 2):
            raise ValueError('Invalid sigma coordinate: must have either 0 or'
                             ' 2 bounds.')
        if surface_air_pressure and surface_air_pressure.nbounds:
            msg = 'Surface pressure coordinate {!r} has bounds. These will' \
                  ' be disregarded.'.format(surface_air_pressure.name())
            warnings.warn(msg, UserWarning, stacklevel=2)

        # Check units.
        if sigma is not None and not sigma.units.is_dimensionless():
            raise ValueError('Invalid units: sigma must be dimensionless.')
        if delta is not None and surface_air_pressure is not None and \
                delta.units != surface_air_pressure.units:
            msg = 'Incompatible units: delta and ' \
                  'surface_air_pressure must have the same units.'
            raise ValueError(msg)

        if delta is not None:
            units = delta.units
        else:
            units = surface_air_pressure.units

        if not units.is_convertible('Pa'):
            msg = 'Invalid units: delta and ' \
                'surface_air_pressure must have units of pressure.'
            raise ValueError(msg)

    @property
    def dependencies(self):
        """
        Returns a dictionary mapping from constructor argument names to
        the corresponding coordinates.

        """
        return {'delta': self.delta, 'sigma': self.sigma,
                'surface_air_pressure': self.surface_air_pressure}

    def _derive(self, delta, sigma, surface_air_pressure):
        temp = delta + sigma * surface_air_pressure
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
                                nd_points_by_key['surface_air_pressure'])
        shape = self._shape(nd_points_by_key)
        dtype = self._dtype(nd_points_by_key)
        points = LazyArray(shape, calc_points, dtype)

        bounds = None
        if ((self.delta and self.delta.nbounds) or
                (self.sigma and self.sigma.nbounds)):
            # Build a "lazy" bounds array.
            nd_values_by_key = self._remap_with_bounds(dependency_dims,
                                                       derived_dims)

            # Define the function here to obtain a closure.
            def calc_bounds():
                delta = nd_values_by_key['delta']
                sigma = nd_values_by_key['sigma']
                surface_air_pressure = nd_values_by_key['surface_air_pressure']
                ok_bound_shapes = [(), (1,), (2,)]
                if delta.shape[-1:] not in ok_bound_shapes:
                    raise ValueError('Invalid delta coordinate bounds.')
                if sigma.shape[-1:] not in ok_bound_shapes:
                    raise ValueError('Invalid sigma coordinate bounds.')
                if surface_air_pressure.shape[-1:] not in [(), (1,)]:
                    warnings.warn('Surface pressure coordinate has bounds. '
                                  'These are being disregarded.')
                    surface_air_pressure_pts = nd_points_by_key[
                        'surface_air_pressure']
                    surface_air_pressure_pts_shape = list(
                        surface_air_pressure_pts.shape)
                    surface_air_pressure = surface_air_pressure_pts.reshape(
                        surface_air_pressure_pts_shape.append(1))
                return self._derive(delta, sigma, surface_air_pressure)
            b_shape = self._shape(nd_values_by_key)
            b_dtype = self._dtype(nd_values_by_key)
            bounds = LazyArray(b_shape, calc_bounds, b_dtype)

        hybrid_pressure = iris.coords.AuxCoord(
            points, standard_name=self.standard_name, long_name=self.long_name,
            var_name=self.var_name, units=self.units, bounds=bounds,
            attributes=self.attributes, coord_system=self.coord_system)
        return hybrid_pressure

    def update(self, old_coord, new_coord=None):
        """
        Notifies the factory of the removal/replacement of a coordinate
        which might be a dependency.

        Args:

        * old_coord:
            The coordinate to be removed/replaced.
        * new_coord:
            If None, any dependency using old_coord is removed, otherwise
            any dependency using old_coord is updated to use new_coord.

        """
        new_dependencies = self.dependencies
        for name, coord in self.dependencies.items():
            if old_coord is coord:
                new_dependencies[name] = new_coord
                try:
                    self._check_dependencies(**new_dependencies)
                except ValueError as e:
                    msg = 'Failed to update dependencies. ' + e.message
                    raise ValueError(msg)
                else:
                    setattr(self, name, new_coord)
                break


class OceanSigmaZFactory(AuxCoordFactory):
    """Defines an ocean sigma over z coordinate factory."""

    def __init__(self, sigma=None, eta=None, depth=None,
                 depth_c=None, nsigma=None, zlev=None):
        """
        Creates a ocean sigma over z coordinate factory with the formula:

        if k < nsigma:
            z(n, k, j, i) = eta(n, j, i) + sigma(k) *
                             (min(depth_c, depth(j, i)) + eta(n, j, i))

        if k >= nsigma:
            z(n, k, j, i) = zlev(k)

        The `zlev` and 'nsigma' coordinates must be provided, and at least
        either `eta`, or 'sigma' and `depth` and `depth_c` coordinates.

        """
        super(OceanSigmaZFactory, self).__init__()

        # Check that provided coordinates meet necessary conditions.
        self._check_dependencies(sigma, eta, depth, depth_c, nsigma, zlev)

        self.sigma = sigma
        self.eta = eta
        self.depth = depth
        self.depth_c = depth_c
        self.nsigma = nsigma
        self.zlev = zlev

        self.standard_name = 'sea_surface_height_above_reference_ellipsoid'
        self.attributes = {'positive': 'up'}

    @property
    def units(self):
        return self.zlev.units

    @staticmethod
    def _check_dependencies(sigma, eta, depth, depth_c, nsigma, zlev):
        # Check for sufficient factory coordinates.
        if zlev is None:
            raise ValueError('Unable to determine units: '
                             'no zlev coordinate available.')
        if nsigma is None:
            raise ValueError('Missing nsigma coordinate.')

        if eta is None and (sigma is None or depth_c is None or
                            depth is None):
            msg = 'Unable to construct ocean sigma over z coordinate ' \
                'factory due to insufficient source coordinates.'
            raise ValueError(msg)

        # Check bounds and shape.
        for coord, term in ((sigma, 'sigma'), (zlev, 'zlev')):
            if coord is not None and coord.nbounds not in (0, 2):
                msg = 'Invalid {} coordinate {!r}: must have either ' \
                    '0 or 2 bounds.'.format(term, coord.name())
                raise ValueError(msg)

        if sigma and sigma.nbounds != zlev.nbounds:
            msg = 'The sigma coordinate {!r} and zlev coordinate {!r} ' \
                'must be equally bounded.'.format(sigma.name(), zlev.name())
            raise ValueError(msg)

        coords = ((eta, 'eta'), (depth, 'depth'),
                  (depth_c, 'depth_c'), (nsigma, 'nsigma'))
        for coord, term in coords:
            if coord is not None and coord.nbounds:
                msg = 'The {} coordinate {!r} has bounds. ' \
                    'These are being disregarded.'.format(term, coord.name())
                warnings.warn(msg, UserWarning, stacklevel=2)

        for coord, term in ((depth_c, 'depth_c'), (nsigma, 'nsigma')):
            if coord is not None and coord.shape != (1,):
                msg = 'Expected scalar {} coordinate {!r}: ' \
                    'got shape {!r}.'.format(term, coord.name(), coord.shape)
                raise ValueError(msg)

        # Check units.
        if not zlev.units.is_convertible('m'):
            msg = 'Invalid units: zlev coordinate {!r} ' \
                'must have units of distance.'.format(zlev.name())
            raise ValueError(msg)

        if sigma is not None and not sigma.units.is_dimensionless():
            msg = 'Invalid units: sigma coordinate {!r} ' \
                'must be dimensionless.'.format(sigma.name())
            raise ValueError(msg)

        coords = ((eta, 'eta'), (depth_c, 'depth_c'), (depth, 'depth'))
        for coord, term in coords:
            if coord is not None and coord.units != zlev.units:
                msg = 'Incompatible units: {} coordinate {!r} and zlev ' \
                    'coordinate {!r} must have ' \
                    'the same units.'.format(term, coord.name(), zlev.name())
                raise ValueError(msg)

    @property
    def dependencies(self):
        """
        Returns a dictionary mapping from constructor argument names to
        the corresponding coordinates.

        """
        return dict(sigma=self.sigma, eta=self.eta, depth=self.depth,
                    depth_c=self.depth_c, nsigma=self.nsigma, zlev=self.zlev)

    def _derive(self, sigma, eta, depth, depth_c,
                nsigma, zlev, shape, nsigma_slice):
        # Perform the ocean sigma over z coordinate nsigma slice.
        if eta.ndim:
            eta = eta[nsigma_slice]
        if sigma.ndim:
            sigma = sigma[nsigma_slice]
        if depth.ndim:
            depth = depth[nsigma_slice]
        # Note that, this performs a point-wise minimum.
        temp = eta + sigma * (np.minimum(depth_c, depth) + eta)
        # Calculate the final derived result.
        result = np.ones(shape, dtype=temp.dtype) * zlev
        result[nsigma_slice] = temp

        return result

    def make_coord(self, coord_dims_func):
        """
        Returns a new :class:`iris.coords.AuxCoord` as defined by this factory.

        Args:

        * coord_dims_func:
            A callable which can return the list of dimesions relevant
            to a given coordinate. See :meth:`iris.cube.Cube.coord_dims()`.

        """
        # Determine the relevant dimensions.
        derived_dims = self.derived_dims(coord_dims_func)
        dependency_dims = self._dependency_dims(coord_dims_func)

        # Build a "lazy" points array.
        nd_points_by_key = self._remap(dependency_dims, derived_dims)
        points_shape = self._shape(nd_points_by_key)
        points_dtype = self._dtype(nd_points_by_key, shape=(), nsigma_slice=())

        # Calculate the nsigma slice.
        nsigma_slice = [slice(None)] * len(derived_dims)
        dim, = dependency_dims['zlev']
        index = derived_dims.index(dim)
        nsigma_slice[index] = slice(0, int(nd_points_by_key['nsigma']))

        # Define the function here to obtain a closure.
        def calc_points():
            return self._derive(nd_points_by_key['sigma'],
                                nd_points_by_key['eta'],
                                nd_points_by_key['depth'],
                                nd_points_by_key['depth_c'],
                                nd_points_by_key['nsigma'],
                                nd_points_by_key['zlev'],
                                points_shape,
                                nsigma_slice)

        points = LazyArray(points_shape, calc_points, points_dtype)

        bounds = None
        if self.zlev.nbounds or (self.sigma and self.sigma.nbounds):
            # Build a "lazy" bounds array.
            nd_values_by_key = self._remap_with_bounds(dependency_dims,
                                                       derived_dims)
            bounds_shape = self._shape(nd_values_by_key)
            bounds_dtype = self._dtype(nd_values_by_key, shape=(),
                                       nsigma_slice=())
            nsigma_slice_bounds = nsigma_slice + [slice(None)]

            # Define the function here to obtain a closure.
            def calc_bounds():
                valid_shapes = [(), (1,), (2,)]
                for key in ('sigma', 'zlev'):
                    if nd_values_by_key[key].shape[-1:] not in valid_shapes:
                        name = self.dependencies[key].name()
                        msg = 'Invalid bounds for {} ' \
                            'coordinate {!r}.'.format(key, name)
                        raise ValueError(msg)
                valid_shapes.pop()
                for key in ('eta', 'depth', 'depth_c', 'nsigma'):
                    if nd_values_by_key[key].shape[-1:] not in valid_shapes:
                        name = self.dependencies[key].name()
                        msg = 'The {} coordinate {!r} has bounds. ' \
                            'These are being disregarded.'.format(key, name)
                        warnings.warn(msg, UserWarning, stacklevel=2)
                        # Swap bounds with points.
                        shape = list(nd_points_by_key[key].shape)
                        bounds = nd_points_by_key[key].reshape(shape.append(1))
                        nd_values_by_key[key] = bounds
                return self._derive(nd_values_by_key['sigma'],
                                    nd_values_by_key['eta'],
                                    nd_values_by_key['depth'],
                                    nd_values_by_key['depth_c'],
                                    nd_values_by_key['nsigma'],
                                    nd_values_by_key['zlev'],
                                    bounds_shape,
                                    nsigma_slice_bounds)

            bounds = LazyArray(bounds_shape, calc_bounds, bounds_dtype)

        coord = iris.coords.AuxCoord(points,
                                     standard_name=self.standard_name,
                                     long_name=self.long_name,
                                     var_name=self.var_name,
                                     units=self.units,
                                     bounds=bounds,
                                     attributes=self.attributes,
                                     coord_system=self.coord_system)
        return coord

    def update(self, old_coord, new_coord=None):
        """
        Notifies the factory of the removal/replacement of a coordinate
        which might be a dependency.

        Args:

        * old_coord:
            The coordinate to be removed/replaced.
        * new_coord:
            If None, any dependency using old_coord is removed, otherwise
            any dependency using old_coord is updated to use new_coord.

        """
        new_dependencies = self.dependencies
        for name, coord in self.dependencies.items():
            if old_coord is coord:
                new_dependencies[name] = new_coord
                try:
                    self._check_dependencies(**new_dependencies)
                except ValueError as e:
                    msg = 'Failed to update dependencies. ' + e.message
                    raise ValueError(msg)
                else:
                    setattr(self, name, new_coord)
                break


class OceanSigmaFactory(AuxCoordFactory):
    """Defines an ocean sigma coordinate factory."""

    def __init__(self, sigma=None, eta=None, depth=None):
        """
        Creates an ocean sigma coordinate factory with the formula:

        z(n, k, j, i) = eta(n, j, i) + sigma(k) *
                        (depth(j, i) + eta(n, j, i))

        """
        super(OceanSigmaFactory, self).__init__()

        # Check that provided coordinates meet necessary conditions.
        self._check_dependencies(sigma, eta, depth)

        self.sigma = sigma
        self.eta = eta
        self.depth = depth

        self.standard_name = 'sea_surface_height_above_reference_ellipsoid'
        self.attributes = {'positive': 'up'}

    @property
    def units(self):
        return self.depth.units

    @staticmethod
    def _check_dependencies(sigma, eta, depth):
        # Check for sufficient factory coordinates.
        if eta is None or sigma is None or depth is None:
            msg = 'Unable to construct ocean sigma coordinate ' \
                'factory due to insufficient source coordinates.'
            raise ValueError(msg)

        # Check bounds and shape.
        coord, term = (sigma, 'sigma')
        if coord is not None and coord.nbounds not in (0, 2):
            msg = 'Invalid {} coordinate {!r}: must have either ' \
                  '0 or 2 bounds.'.format(term, coord.name())
            raise ValueError(msg)

        coords = ((eta, 'eta'), (depth, 'depth'))
        for coord, term in coords:
            if coord is not None and coord.nbounds:
                msg = 'The {} coordinate {!r} has bounds. ' \
                    'These are being disregarded.'.format(term, coord.name())
                warnings.warn(msg, UserWarning, stacklevel=2)

        # Check units.
        if sigma is not None and not sigma.units.is_dimensionless():
            msg = 'Invalid units: sigma coordinate {!r} ' \
                'must be dimensionless.'.format(sigma.name())
            raise ValueError(msg)

        coords = ((eta, 'eta'), (depth, 'depth'))
        for coord, term in coords:
            if coord is not None and coord.units != depth.units:
                msg = 'Incompatible units: {} coordinate {!r} and depth ' \
                    'coordinate {!r} must have ' \
                    'the same units.'.format(term, coord.name(), depth.name())
                raise ValueError(msg)

    @property
    def dependencies(self):
        """
        Returns a dictionary mapping from constructor argument names to
        the corresponding coordinates.

        """
        return dict(sigma=self.sigma, eta=self.eta, depth=self.depth)

    def _derive(self, sigma, eta, depth):
        result = eta + sigma * (depth + eta)
        return result

    def make_coord(self, coord_dims_func):
        """
        Returns a new :class:`iris.coords.AuxCoord` as defined by this factory.

        Args:

        * coord_dims_func:
            A callable which can return the list of dimensions relevant
            to a given coordinate. See :meth:`iris.cube.Cube.coord_dims()`.

        """
        # Determine the relevant dimensions.
        derived_dims = self.derived_dims(coord_dims_func)
        dependency_dims = self._dependency_dims(coord_dims_func)

        # Build a "lazy" points array.
        nd_points_by_key = self._remap(dependency_dims, derived_dims)
        points_shape = self._shape(nd_points_by_key)

        # Define the function here to obtain a closure.
        def calc_points():
            return self._derive(nd_points_by_key['sigma'],
                                nd_points_by_key['eta'],
                                nd_points_by_key['depth'])

        points = LazyArray(points_shape, calc_points)

        bounds = None
        if self.sigma and self.sigma.nbounds:
            # Build a "lazy" bounds array.
            nd_values_by_key = self._remap_with_bounds(dependency_dims,
                                                       derived_dims)
            bounds_shape = self._shape(nd_values_by_key)

            # Define the function here to obtain a closure.
            def calc_bounds():
                valid_shapes = [(), (1,), (2,)]
                key = 'sigma'
                if nd_values_by_key[key].shape[-1:] not in valid_shapes:
                    name = self.dependencies[key].name()
                    msg = 'Invalid bounds for {} ' \
                        'coordinate {!r}.'.format(key, name)
                    raise ValueError(msg)
                valid_shapes.pop()
                for key in ('eta', 'depth'):
                    if nd_values_by_key[key].shape[-1:] not in valid_shapes:
                        name = self.dependencies[key].name()
                        msg = 'The {} coordinate {!r} has bounds. ' \
                            'These are being disregarded.'.format(key, name)
                        warnings.warn(msg, UserWarning, stacklevel=2)
                        # Swap bounds with points.
                        shape = list(nd_points_by_key[key].shape)
                        bounds = nd_points_by_key[key].reshape(shape.append(1))
                        nd_values_by_key[key] = bounds
                return self._derive(nd_values_by_key['sigma'],
                                    nd_values_by_key['eta'],
                                    nd_values_by_key['depth'],
                                    bounds_shape)

            bounds = LazyArray(bounds_shape, calc_bounds)

        coord = iris.coords.AuxCoord(points,
                                     standard_name=self.standard_name,
                                     long_name=self.long_name,
                                     var_name=self.var_name,
                                     units=self.units,
                                     bounds=bounds,
                                     attributes=self.attributes,
                                     coord_system=self.coord_system)
        return coord

    def update(self, old_coord, new_coord=None):
        """
        Notifies the factory of the removal/replacement of a coordinate
        which might be a dependency.

        Args:

        * old_coord:
            The coordinate to be removed/replaced.
        * new_coord:
            If None, any dependency using old_coord is removed, otherwise
            any dependency using old_coord is updated to use new_coord.

        """
        new_dependencies = self.dependencies
        for name, coord in self.dependencies.items():
            if old_coord is coord:
                new_dependencies[name] = new_coord
                try:
                    self._check_dependencies(**new_dependencies)
                except ValueError as e:
                    msg = 'Failed to update dependencies. ' + e.message
                    raise ValueError(msg)
                else:
                    setattr(self, name, new_coord)
                break


class OceanSg1Factory(AuxCoordFactory):
    """Defines an Ocean s-coordinate, generic form 1 factory."""

    def __init__(self, s=None, c=None, eta=None, depth=None, depth_c=None):
        """
        Creates an Ocean s-coordinate, generic form 1 factory with the formula:

        z(n,k,j,i) = S(k,j,i) + eta(n,j,i) * (1 + S(k,j,i) / depth(j,i))

        where:
            S(k,j,i) = depth_c * s(k) + (depth(j,i) - depth_c) * C(k)

        """
        super(OceanSg1Factory, self).__init__()

        # Check that provided coordinates meet necessary conditions.
        self._check_dependencies(s, c, eta, depth, depth_c)

        self.s = s
        self.c = c
        self.eta = eta
        self.depth = depth
        self.depth_c = depth_c

        self.standard_name = 'sea_surface_height_above_reference_ellipsoid'
        self.attributes = {'positive': 'up'}

    @property
    def units(self):
        return self.depth.units

    @staticmethod
    def _check_dependencies(s, c, eta, depth, depth_c):
        # Check for sufficient factory coordinates.
        if (eta is None or s is None or c is None or
           depth is None or depth_c is None):
            msg = 'Unable to construct Ocean s-coordinate, generic form 1 ' \
                'factory due to insufficient source coordinates.'
            raise ValueError(msg)

        # Check bounds and shape.
        coords = ((s, 's'), (c, 'c'))
        for coord, term in coords:
            if coord is not None and coord.nbounds not in (0, 2):
                msg = 'Invalid {} coordinate {!r}: must have either ' \
                    '0 or 2 bounds.'.format(term, coord.name())
                raise ValueError(msg)

        if s and s.nbounds != c.nbounds:
            msg = 'The s coordinate {!r} and c coordinate {!r} ' \
                'must be equally bounded.'.format(s.name(), c.name())
            raise ValueError(msg)

        coords = ((eta, 'eta'), (depth, 'depth'))
        for coord, term in coords:
            if coord is not None and coord.nbounds:
                msg = 'The {} coordinate {!r} has bounds. ' \
                    'These are being disregarded.'.format(term, coord.name())
                warnings.warn(msg, UserWarning, stacklevel=2)

        if depth_c is not None and depth_c.shape != (1,):
            msg = 'Expected scalar {} coordinate {!r}: ' \
                'got shape {!r}.'.format(term, coord.name(), coord.shape)
            raise ValueError(msg)

        # Check units.
        coords = ((s, 's'), (c, 'c'))
        for coord, term in coords:
            if coord is not None and not coord.units.is_dimensionless():
                msg = 'Invalid units: {} coordinate {!r} ' \
                    'must be dimensionless.'.format(term, coord.name())
                raise ValueError(msg)

        coords = ((eta, 'eta'), (depth, 'depth'), (depth_c, 'depth_c'))
        for coord, term in coords:
            if coord is not None and coord.units != depth.units:
                msg = 'Incompatible units: {} coordinate {!r} and depth ' \
                    'coordinate {!r} must have ' \
                    'the same units.'.format(term, coord.name(), depth.name())
                raise ValueError(msg)

    @property
    def dependencies(self):
        """
        Returns a dictionary mapping from constructor argument names to
        the corresponding coordinates.

        """
        return dict(s=self.s, c=self.c, eta=self.eta, depth=self.depth,
                    depth_c=self.depth_c)

    def _derive(self, s, c, eta, depth, depth_c):
        S = depth_c * s + (depth - depth_c) * c
        result = S + eta * (1 + S / depth)
        return result

    def make_coord(self, coord_dims_func):
        """
        Returns a new :class:`iris.coords.AuxCoord` as defined by this factory.

        Args:

        * coord_dims_func:
            A callable which can return the list of dimensions relevant
            to a given coordinate. See :meth:`iris.cube.Cube.coord_dims()`.

        """
        # Determine the relevant dimensions.
        derived_dims = self.derived_dims(coord_dims_func)
        dependency_dims = self._dependency_dims(coord_dims_func)

        # Build a "lazy" points array.
        nd_points_by_key = self._remap(dependency_dims, derived_dims)
        points_shape = self._shape(nd_points_by_key)

        # Define the function here to obtain a closure.
        def calc_points():
            return self._derive(nd_points_by_key['s'],
                                nd_points_by_key['c'],
                                nd_points_by_key['eta'],
                                nd_points_by_key['depth'],
                                nd_points_by_key['depth_c'])

        points = LazyArray(points_shape, calc_points)

        bounds = None
        if self.s.nbounds or (self.c and self.c.nbounds):
            # Build a "lazy" bounds array.
            nd_values_by_key = self._remap_with_bounds(dependency_dims,
                                                       derived_dims)
            bounds_shape = self._shape(nd_values_by_key)

            # Define the function here to obtain a closure.
            def calc_bounds():
                valid_shapes = [(), (1,), (2,)]
                key = 's'
                if nd_values_by_key[key].shape[-1:] not in valid_shapes:
                    name = self.dependencies[key].name()
                    msg = 'Invalid bounds for {} ' \
                        'coordinate {!r}.'.format(key, name)
                    raise ValueError(msg)
                valid_shapes.pop()
                for key in ('eta', 'depth', 'depth_c'):
                    if nd_values_by_key[key].shape[-1:] not in valid_shapes:
                        name = self.dependencies[key].name()
                        msg = 'The {} coordinate {!r} has bounds. ' \
                            'These are being disregarded.'.format(key, name)
                        warnings.warn(msg, UserWarning, stacklevel=2)
                        # Swap bounds with points.
                        shape = list(nd_points_by_key[key].shape)
                        bounds = nd_points_by_key[key].reshape(shape.append(1))
                        nd_values_by_key[key] = bounds
                return self._derive(nd_values_by_key['s'],
                                    nd_values_by_key['c'],
                                    nd_values_by_key['eta'],
                                    nd_values_by_key['depth'],
                                    nd_values_by_key['depth_c'],
                                    bounds_shape)

            bounds = LazyArray(bounds_shape, calc_bounds)

        coord = iris.coords.AuxCoord(points,
                                     standard_name=self.standard_name,
                                     long_name=self.long_name,
                                     var_name=self.var_name,
                                     units=self.units,
                                     bounds=bounds,
                                     attributes=self.attributes,
                                     coord_system=self.coord_system)
        return coord

    def update(self, old_coord, new_coord=None):
        """
        Notifies the factory of the removal/replacement of a coordinate
        which might be a dependency.

        Args:

        * old_coord:
            The coordinate to be removed/replaced.
        * new_coord:
            If None, any dependency using old_coord is removed, otherwise
            any dependency using old_coord is updated to use new_coord.

        """
        new_dependencies = self.dependencies
        for name, coord in self.dependencies.items():
            if old_coord is coord:
                new_dependencies[name] = new_coord
                try:
                    self._check_dependencies(**new_dependencies)
                except ValueError as e:
                    msg = 'Failed to update dependencies. ' + e.message
                    raise ValueError(msg)
                else:
                    setattr(self, name, new_coord)
                break


class OceanSFactory(AuxCoordFactory):
    """Defines an Ocean s-coordinate factory."""

    def __init__(self, s=None, eta=None, depth=None, a=None, b=None,
                 depth_c=None):
        """
        Creates an Ocean s-coordinate factory with the formula:

        z(n,k,j,i) = eta(n,j,i)*(1+s(k)) + depth_c*s(k) +
                     (depth(j,i)-depth_c)*C(k)

        where:
            C(k) = (1-b) * sinh(a*s(k)) / sinh(a) +
                   b * [tanh(a * (s(k) + 0.5)) / (2 * tanh(0.5*a)) - 0.5]

        """
        super(OceanSFactory, self).__init__()

        # Check that provided coordinates meet necessary conditions.
        self._check_dependencies(s, eta, depth, a, b, depth_c)

        self.s = s
        self.eta = eta
        self.depth = depth
        self.a = a
        self.b = b
        self.depth_c = depth_c

        self.standard_name = 'sea_surface_height_above_reference_ellipsoid'
        self.attributes = {'positive': 'up'}

    @property
    def units(self):
        return self.depth.units

    @staticmethod
    def _check_dependencies(s, eta, depth, a, b, depth_c):
        # Check for sufficient factory coordinates.
        if (eta is None or s is None or depth is None or
           a is None or b is None or depth_c is None):
            msg = 'Unable to construct Ocean s-coordinate ' \
                'factory due to insufficient source coordinates.'
            raise ValueError(msg)

        # Check bounds and shape.
        if s is not None and s.nbounds not in (0, 2):
            msg = 'Invalid s coordinate {!r}: must have either ' \
                '0 or 2 bounds.'.format(s.name())
            raise ValueError(msg)

        coords = ((eta, 'eta'), (depth, 'depth'))
        for coord, term in coords:
            if coord is not None and coord.nbounds:
                msg = 'The {} coordinate {!r} has bounds. ' \
                    'These are being disregarded.'.format(term, coord.name())
                warnings.warn(msg, UserWarning, stacklevel=2)

        coords = ((a, 'a'), (b, 'b'), (depth_c, 'depth_c'))
        for coord, term in coords:
            if coord is not None and coord.shape != (1,):
                msg = 'Expected scalar {} coordinate {!r}: ' \
                    'got shape {!r}.'.format(term, coord.name(), coord.shape)
                raise ValueError(msg)

        # Check units.
        if s is not None and not s.units.is_dimensionless():
            msg = 'Invalid units: s coordinate {!r} ' \
                'must be dimensionless.'.format(s.name())
            raise ValueError(msg)

        coords = ((eta, 'eta'), (depth, 'depth'), (depth_c, 'depth_c'))
        for coord, term in coords:
            if coord is not None and coord.units != depth.units:
                msg = 'Incompatible units: {} coordinate {!r} and depth ' \
                    'coordinate {!r} must have ' \
                    'the same units.'.format(term, coord.name(), depth.name())
                raise ValueError(msg)

    @property
    def dependencies(self):
        """
        Returns a dictionary mapping from constructor argument names to
        the corresponding coordinates.

        """
        return dict(s=self.s, eta=self.eta, depth=self.depth, a=self.a,
                    b=self.b, depth_c=self.depth_c)

    def _derive(self, s, eta, depth, a, b, depth_c):
        c = ((1 - b) * np.sinh(a * s) / np.sinh(a) + b *
             (np.tanh(a * (s + 0.5)) / (2 * np.tanh(0.5 * a)) - 0.5))
        result = eta * (1 + s) + depth_c * s + (depth - depth_c) * c
        return result

    def make_coord(self, coord_dims_func):
        """
        Returns a new :class:`iris.coords.AuxCoord` as defined by this factory.

        Args:

        * coord_dims_func:
            A callable which can return the list of dimensions relevant
            to a given coordinate. See :meth:`iris.cube.Cube.coord_dims()`.

        """
        # Determine the relevant dimensions.
        derived_dims = self.derived_dims(coord_dims_func)
        dependency_dims = self._dependency_dims(coord_dims_func)

        # Build a "lazy" points array.
        nd_points_by_key = self._remap(dependency_dims, derived_dims)
        points_shape = self._shape(nd_points_by_key)

        # Define the function here to obtain a closure.
        def calc_points():
            return self._derive(nd_points_by_key['s'],
                                nd_points_by_key['eta'],
                                nd_points_by_key['depth'],
                                nd_points_by_key['a'],
                                nd_points_by_key['b'],
                                nd_points_by_key['depth_c'])

        points = LazyArray(points_shape, calc_points)

        bounds = None
        if self.s.nbounds:
            # Build a "lazy" bounds array.
            nd_values_by_key = self._remap_with_bounds(dependency_dims,
                                                       derived_dims)
            bounds_shape = self._shape(nd_values_by_key)

            # Define the function here to obtain a closure.
            def calc_bounds():
                valid_shapes = [(), (1,), (2,)]
                key = 's'
                if nd_values_by_key[key].shape[-1:] not in valid_shapes:
                    name = self.dependencies[key].name()
                    msg = 'Invalid bounds for {} ' \
                        'coordinate {!r}.'.format(key, name)
                    raise ValueError(msg)
                valid_shapes.pop()
                for key in ('eta', 'depth', 'a', 'b', 'depth_c'):
                    if nd_values_by_key[key].shape[-1:] not in valid_shapes:
                        name = self.dependencies[key].name()
                        msg = 'The {} coordinate {!r} has bounds. ' \
                            'These are being disregarded.'.format(key, name)
                        warnings.warn(msg, UserWarning, stacklevel=2)
                        # Swap bounds with points.
                        shape = list(nd_points_by_key[key].shape)
                        bounds = nd_points_by_key[key].reshape(shape.append(1))
                        nd_values_by_key[key] = bounds
                return self._derive(nd_values_by_key['s'],
                                    nd_values_by_key['eta'],
                                    nd_values_by_key['depth'],
                                    nd_values_by_key['a'],
                                    nd_values_by_key['b'],
                                    nd_values_by_key['depth_c'],
                                    bounds_shape)

            bounds = LazyArray(bounds_shape, calc_bounds)

        coord = iris.coords.AuxCoord(points,
                                     standard_name=self.standard_name,
                                     long_name=self.long_name,
                                     var_name=self.var_name,
                                     units=self.units,
                                     bounds=bounds,
                                     attributes=self.attributes,
                                     coord_system=self.coord_system)
        return coord

    def update(self, old_coord, new_coord=None):
        """
        Notifies the factory of the removal/replacement of a coordinate
        which might be a dependency.

        Args:

        * old_coord:
            The coordinate to be removed/replaced.
        * new_coord:
            If None, any dependency using old_coord is removed, otherwise
            any dependency using old_coord is updated to use new_coord.

        """
        new_dependencies = self.dependencies
        for name, coord in self.dependencies.items():
            if old_coord is coord:
                new_dependencies[name] = new_coord
                try:
                    self._check_dependencies(**new_dependencies)
                except ValueError as e:
                    msg = 'Failed to update dependencies. ' + e.message
                    raise ValueError(msg)
                else:
                    setattr(self, name, new_coord)
                break


class OceanSg2Factory(AuxCoordFactory):
    """Defines an Ocean s-coordinate, generic form 2 factory."""

    def __init__(self, s=None, c=None, eta=None, depth=None, depth_c=None):
        """
        Creates an Ocean s-coordinate, generic form 2 factory with the formula:

        z(n,k,j,i) = eta(n,j,i) + (eta(n,j,i) + depth(j,i)) * S(k,j,i)

        where:
            S(k,j,i) = (depth_c * s(k) + depth(j,i) * C(k)) /
                       (depth_c + depth(j,i))

        """
        super(OceanSg2Factory, self).__init__()

        # Check that provided coordinates meet necessary conditions.
        self._check_dependencies(s, c, eta, depth, depth_c)

        self.s = s
        self.c = c
        self.eta = eta
        self.depth = depth
        self.depth_c = depth_c

        self.standard_name = 'sea_surface_height_above_reference_ellipsoid'
        self.attributes = {'positive': 'up'}

    @property
    def units(self):
        return self.depth.units

    @staticmethod
    def _check_dependencies(s, c, eta, depth, depth_c):
        # Check for sufficient factory coordinates.
        if (eta is None or s is None or c is None or
           depth is None or depth_c is None):
            msg = 'Unable to construct Ocean s-coordinate, generic form 2 ' \
                'factory due to insufficient source coordinates.'
            raise ValueError(msg)

        # Check bounds and shape.
        coords = ((s, 's'), (c, 'c'))
        for coord, term in coords:
            if coord is not None and coord.nbounds not in (0, 2):
                msg = 'Invalid {} coordinate {!r}: must have either ' \
                    '0 or 2 bounds.'.format(term, coord.name())
                raise ValueError(msg)

        if s and s.nbounds != c.nbounds:
            msg = 'The s coordinate {!r} and c coordinate {!r} ' \
                'must be equally bounded.'.format(s.name(), c.name())
            raise ValueError(msg)

        coords = ((eta, 'eta'), (depth, 'depth'))
        for coord, term in coords:
            if coord is not None and coord.nbounds:
                msg = 'The {} coordinate {!r} has bounds. ' \
                    'These are being disregarded.'.format(term, coord.name())
                warnings.warn(msg, UserWarning, stacklevel=2)

        if depth_c is not None and depth_c.shape != (1,):
            msg = 'Expected scalar depth_c coordinate {!r}: ' \
                'got shape {!r}.'.format(depth_c.name(), depth_c.shape)
            raise ValueError(msg)

        # Check units.
        coords = ((s, 's'), (c, 'c'))
        for coord, term in coords:
            if coord is not None and not coord.units.is_dimensionless():
                msg = 'Invalid units: {} coordinate {!r} ' \
                    'must be dimensionless.'.format(term, coord.name())
                raise ValueError(msg)

        coords = ((eta, 'eta'), (depth, 'depth'), (depth_c, 'depth_c'))
        for coord, term in coords:
            if coord is not None and coord.units != depth.units:
                msg = 'Incompatible units: {} coordinate {!r} and depth ' \
                    'coordinate {!r} must have ' \
                    'the same units.'.format(term, coord.name(), depth.name())
                raise ValueError(msg)

    @property
    def dependencies(self):
        """
        Returns a dictionary mapping from constructor argument names to
        the corresponding coordinates.

        """
        return dict(s=self.s, c=self.c, eta=self.eta, depth=self.depth,
                    depth_c=self.depth_c)

    def _derive(self, s, c, eta, depth, depth_c):
        S = (depth_c * s + depth * c) / (depth_c + depth)
        result = eta + (eta + depth) * S
        return result

    def make_coord(self, coord_dims_func):
        """
        Returns a new :class:`iris.coords.AuxCoord` as defined by this factory.

        Args:

        * coord_dims_func:
            A callable which can return the list of dimensions relevant
            to a given coordinate. See :meth:`iris.cube.Cube.coord_dims()`.

        """
        # Determine the relevant dimensions.
        derived_dims = self.derived_dims(coord_dims_func)
        dependency_dims = self._dependency_dims(coord_dims_func)

        # Build a "lazy" points array.
        nd_points_by_key = self._remap(dependency_dims, derived_dims)
        points_shape = self._shape(nd_points_by_key)

        # Define the function here to obtain a closure.
        def calc_points():
            return self._derive(nd_points_by_key['s'],
                                nd_points_by_key['c'],
                                nd_points_by_key['eta'],
                                nd_points_by_key['depth'],
                                nd_points_by_key['depth_c'])

        points = LazyArray(points_shape, calc_points)

        bounds = None
        if self.s.nbounds or (self.c and self.c.nbounds):
            # Build a "lazy" bounds array.
            nd_values_by_key = self._remap_with_bounds(dependency_dims,
                                                       derived_dims)
            bounds_shape = self._shape(nd_values_by_key)

            # Define the function here to obtain a closure.
            def calc_bounds():
                valid_shapes = [(), (1,), (2,)]
                key = 's'
                if nd_values_by_key[key].shape[-1:] not in valid_shapes:
                    name = self.dependencies[key].name()
                    msg = 'Invalid bounds for {} ' \
                        'coordinate {!r}.'.format(key, name)
                    raise ValueError(msg)
                valid_shapes.pop()
                for key in ('eta', 'depth', 'depth_c'):
                    if nd_values_by_key[key].shape[-1:] not in valid_shapes:
                        name = self.dependencies[key].name()
                        msg = 'The {} coordinate {!r} has bounds. ' \
                            'These are being disregarded.'.format(key, name)
                        warnings.warn(msg, UserWarning, stacklevel=2)
                        # Swap bounds with points.
                        shape = list(nd_points_by_key[key].shape)
                        bounds = nd_points_by_key[key].reshape(shape.append(1))
                        nd_values_by_key[key] = bounds
                return self._derive(nd_values_by_key['s'],
                                    nd_values_by_key['c'],
                                    nd_values_by_key['eta'],
                                    nd_values_by_key['depth'],
                                    nd_values_by_key['depth_c'],
                                    bounds_shape)

            bounds = LazyArray(bounds_shape, calc_bounds)

        coord = iris.coords.AuxCoord(points,
                                     standard_name=self.standard_name,
                                     long_name=self.long_name,
                                     var_name=self.var_name,
                                     units=self.units,
                                     bounds=bounds,
                                     attributes=self.attributes,
                                     coord_system=self.coord_system)
        return coord

    def update(self, old_coord, new_coord=None):
        """
        Notifies the factory of the removal/replacement of a coordinate
        which might be a dependency.

        Args:

        * old_coord:
            The coordinate to be removed/replaced.
        * new_coord:
            If None, any dependency using old_coord is removed, otherwise
            any dependency using old_coord is updated to use new_coord.

        """
        new_dependencies = self.dependencies
        for name, coord in self.dependencies.items():
            if old_coord is coord:
                new_dependencies[name] = new_coord
                try:
                    self._check_dependencies(**new_dependencies)
                except ValueError as e:
                    msg = 'Failed to update dependencies. ' + e.message
                    raise ValueError(msg)
                else:
                    setattr(self, name, new_coord)
                break
