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
import collections

import numpy as np
from numpy.lib.stride_tricks import as_strided

import iris.cube
import iris.coords

from iris.experimental.regrid import _RegularGridInterpolator, _ndim_coords_from_arrays
from iris.analysis.interpolate import _extend_circular_coord_and_data, _extend_circular_data


class LinearInterpolator(object):
    def __init__(self, cube, interp_coords, extrapolation_mode='linear'):
        # Cube is "data_cube".
        self.cube = cube
        # XXX use _set_extrapolation_mode.
        self.extrapolation_mode = extrapolation_mode
        self._bounds_error = (extrapolation_mode == 'error')
        
        self._interp_coords = interp_coords
        self.ndims = len(interp_coords)
        data_coords = [cube.coord(coord) for coord in interp_coords]
        
        # Triggers the loading - is that really necessary...?
        data = cube.data
        coord_points_list = []
        
        #: Contains tuples of (interpolation dimension, coordinate dimension in cube, min coord point
        #:                     max_coord_point, coord modulus or 0)
        self._circulars = []
        for interp_dim, coord in enumerate(data_coords):
            if coord.ndim != 1:
                raise ValueError('Interpolation coords must be 1-d for '
                                 'non-triangulation based interpolation.')
            # xxx What about 1dim coords not bound to a dimension?
            coord_dims = cube.coord_dims(coord)

            is_circular = getattr(coord, 'circular', False)

            if is_circular:
                coord_points, data = _extend_circular_coord_and_data(coord, data, coord_dims[0])
                self._circulars.append((interp_dim, coord_dims[0],
                                        coord_points.min(),
                                        coord_points.max(),
                                        getattr(coord.units, 'modulus', 0)))
            else:
                coord_points = coord.points
                data = data
            coord_points_list.append([coord_points, coord_dims])
        
#        coord_points_list, coord_dims = zip(*sorted(coord_points_list, key=lambda (points, dim): dim))
        coord_points, self.coord_dims = zip(*coord_points_list)
        self.coord_dims = list(np.concatenate(self.coord_dims))

        # XXX Assuming it is monotonic at all - not safe...
        self.coord_decreasing = [np.all(np.diff(points) < 0) for points in coord_points]
        if np.any(self.coord_decreasing):
            coord_points = [points[::-1] if is_decreasing else points
                            for is_decreasing, points in zip(self.coord_decreasing, coord_points)]

        if len(interp_coords) != len(set(self.coord_dims)):
            raise ValueError('Coordinates repeat a data dimension - the '
                             'interpolation would be over-specified.')
        shape = [data.shape[ind] for ind in self.coord_dims]
        
        
        # Create some data which has the simplest possible float dtype.
        
        mock_data = as_strided(np.array(0, dtype=self.interpolated_dtype(cube.data.dtype)),
                               shape=shape, strides=[0] * len(self.coord_dims))
        self._interpolator = self._build_interpolator(coord_points, mock_data)

    def _build_interpolator(self, coord_points, mock_data):
        return _RegularGridInterpolator(coord_points, mock_data,
                                        fill_value=None, bounds_error=False)


    def _update_extrapolation_mode(self, mode):
        if mode == 'linear':
            self._interpolator.bounds_error = False
            self._interpolator.fill_value = None
        elif mode == 'error':
            self._interpolator.bounds_error = True
        elif mode == 'nan':
            self._interpolator.bounds_error = False
            # This is the check in the _RegularGridInterpolator, but I do not
            # agree with it - I can put NaNs in a float32 array...
#            if not np.can_cast(np.nan, self._interpolator.values.dtype):
#                raise ValueError("The interpolation array's dtype does not "
#                                 "support NaNs for extrapolation.")
            self._interpolator.fill_value = np.nan
            
        else:
            # TODO - implement mask extrapolation...
            raise ValueError('Extrapolation mode {!r} not supported.'.format(mode))

    def interpolated_dtype(self, dtype):
        """Return the dtype resulting from interpolating data of the given dtype."""
        return np.result_type(np.float16, dtype)

    def interpolate_data(self, coord_points, data, extrapolation_mode='linear',
                         data_dims=None):
        # Note: Dimensions are never lost - at a minimum they will be of length one. The exception to this is the dimensions -1 
        # that are being interpolated - they currently are all squeezed into a single dimension.
        # Note: Interpolated coordinates are squeezed and pushed to the last dimension.
        
        # data_dims is the dimensions that the given array map to on the originally given cube.
        # If None, range(cube.ndim) is equivalent.
        
        data_dims = data_dims or range(self.cube.ndim)
        if len(data_dims) != data.ndim:
            raise ValueError('data being interpolated is not consistent with the data passed through.')

        # Put the given array into the shape of the original cube's array.
        strides = list(data.strides)
        for dim in range(self.cube.ndim):
            if dim not in data_dims:
                strides.insert(dim, 0)
        data = as_strided(data, strides=strides, shape=self.cube.shape)

        if self._circulars:
            for _, data_dim, _, _, _ in self._circulars:
                data = _extend_circular_data(data, data_dim)

        try:
            coord_points = _ndim_coords_from_arrays(coord_points, self.ndims)
            if coord_points.shape[-1] != self.ndims:
                raise ValueError()
        except ValueError:
            raise ValueError('The given coordinates are not appropriate for'
                             ' the interpolator. There are {0} dimension(s)'
                             ' and ideally the given coordinate array should'
                             ' have a shape of (npts, {0}). Got an array of'
                             ' shape {1}.'
                             ''.format(self.ndims,
                                       np.asanyarray(coord_points).shape))

        if coord_points.dtype == object:
            raise ValueError('Perhaps inconsistently shaped arrays were passed as '
                             'coordinate points. The resulting numpy array has '
                             '"object" as its type.')
        elif issubclass(coord_points.dtype.type, np.integer):
            coord_points = coord_points.astype(np.float)

        if any(self.coord_decreasing):
            data_indexer = [slice(None)] * data.ndim
            for is_decreasing, dim in zip(self.coord_decreasing, self.coord_dims):
                if is_decreasing:
                    data_indexer[dim] = slice(None, None, -1)
            data = data[tuple(data_indexer)]

        # Map all the requested values into the range of the source
        # data (centred over the centre of the source data to allow
        # extrapolation where required).
        if any(self._circulars):
            for (interp_dim, _, src_min, src_max, 
                    src_modulus) in self._circulars:
                offset = (src_max + src_min - src_modulus) * 0.5
                coord_points[:, interp_dim] -= offset
                coord_points[:, interp_dim] = (coord_points[:, interp_dim] %
                                               src_modulus) + offset

        self._update_extrapolation_mode(extrapolation_mode)

        # Build up a shape suitable for passing to ndindex, inside the loop we
        # will insert slice(None) on the data indices.
        iter_shape = list(length if index not in self.coord_dims
                          else 1
                          for index, length in enumerate(data.shape))

        result_shape = list(length for index, length in enumerate(data.shape)
                            if index not in self.coord_dims)
        #; Keep track of the dimension to put the interpolated data into.
        index_dimension = len(result_shape)# min(self.coord_dims)
        result_shape.insert(index_dimension, coord_points.shape[0])

        masked = isinstance(data, np.ma.MaskedArray)
        dtype = self.interpolated_dtype(data.dtype)
        if masked:
            result_data = np.ma.empty(result_shape, dtype=dtype)
            result_data.mask = np.zeros(result_shape, dtype=np.bool)
        else:
            result_data = np.empty(result_shape, dtype=dtype)

        # Iterate through each 2d slice of the data, updating the interpolator
        # with the new data as we go.
        for ndindex in np.ndindex(tuple(iter_shape)):
            # TODO: masked arrays...
            interpolant_index = [position for index, position in enumerate(ndindex)
                                      if index not in self.coord_dims]
            interpolant_index.insert(index_dimension, slice(None))
            index = tuple(position if index not in self.coord_dims
                          else slice(None, None)
                          for index, position in enumerate(ndindex))
            sub_data = data[index]
            
            trans, _ = zip(*sorted(enumerate(self.coord_dims), key=lambda (i, dim): dim))
            sub_data = np.transpose(sub_data, trans).copy()

            r = self._interpolate_data_at_coord_points(sub_data, coord_points)
            result_data[interpolant_index] = r
            if masked:
                r = self._interpolate_data_at_coord_points(sub_data.mask, coord_points)
                result_data.mask[interpolant_index] = r > 0
        return result_data
    
    def _interpolate_data_at_coord_points(self, data, coord_points):
        self._interpolator.values = data
        return self._interpolator(coord_points)

    def orthogonal_points(self, sample_points, data, extrapolation_mode='linear', data_dims=None):
        data_dims = data_dims or range(self.cube.ndim)

        coord_points = [None] * len(self.coord_dims)
        #: The shape of the array after full interpolation of each dimension.
        interpolated_shape = list(self.cube.shape)
        cube_dim_to_result_dim = {}

        interp_coords_seen = 0
        n_non_interp_coords = self.cube.ndim - len(self.coord_dims)

        for dim in range(self.cube.ndim):
            if dim not in self.coord_dims:
                cube_dim_to_result_dim[dim] = dim - interp_coords_seen
            else:
                cube_dim_to_result_dim[dim] = n_non_interp_coords + self.coord_dims.index(dim)
                interp_coords_seen += 1

        result_to_cube_dim = {v: k for k, v in cube_dim_to_result_dim.items()}

        for orig_coord, points in sample_points:
            if orig_coord not in self._interp_coords:
                raise ValueError('Coord {} was not one of those passed to the '
                                 'constructor.'.format(orig_coord))
            coord = self.cube.coord(orig_coord)
            points = np.asanyarray(points)

            order_in_constructor = self._interp_coords.index(orig_coord)

            coord_points[order_in_constructor] = points
            for coord_dim in self.cube.coord_dims(coord):
                interpolated_shape[self.coord_dims[order_in_constructor]] = points.size

        # Given an expected shape for the final array, compute the shape of
        # the array which puts the interpolated dimension last. 
        _, target_shape = zip(*sorted(enumerate(interpolated_shape),
                                          key=lambda (dim, length): cube_dim_to_result_dim[dim]))

        # Now compute the transpose array which needs to be applied to the
        # previously computed shape to get back to the expected interpolated
        # shape.
        transpose_order, _ = zip(*sorted(enumerate(interpolated_shape),
                                          key=lambda (dim, length): result_to_cube_dim[dim]))
        
        # Turn the coord points into one dimensional cross-products.
        if len(coord_points) > 1:
            coord_points = [arr.flatten() for arr in np.meshgrid(*coord_points[::-1])]
        
        # Now turn the list of cross-product 1-d arrays into an array of
        # shape (n_interp_points, n_dims).
        coord_points = np.asanyarray(coord_points).reshape(len(self.coord_dims), -1).T[:, ::-1]
        
        data = self.interpolate_data(coord_points, data, extrapolation_mode, data_dims=data_dims)
        # Turn the interpolated data back into the order that it was given to
        # us in the first place.
        return np.transpose(data.reshape(target_shape), transpose_order)

    def _orthogonal_points_preserve_dimensionality(self, sample_points, data,
                                                   extrapolation_mode='linear', data_dims=None):
        data = self.orthogonal_points(sample_points, data, extrapolation_mode, data_dims)
        index = tuple(0 if dim not in data_dims else slice(None)
                      for dim in range(self.cube.ndim))
        r = data[index]
        return r
    
    def _resample_coord(self, sample_points, coord, coord_dims, extrapolation_mode):
        coord_points = coord.points
        
        new_points = self._orthogonal_points_preserve_dimensionality(sample_points, coord_points,
                                                                     extrapolation_mode, coord_dims)

        # Watch out for DimCoord instances that are no longer monotonic
        # after the resampling.
        try:
            new_coord = coord.copy(new_points)
        except ValueError:
            new_coord = iris.coords.AuxCoord.from_coord(coord).copy(new_points)
        return new_coord

    def orthogonal_cube(self, sample_points, extrapolation_mode='linear', collapse_scalar=True):
        data = self.cube.data
        data = self.orthogonal_points(sample_points,
                                      data, extrapolation_mode)

        # Get hold of the original interpolation coordinates in terms of the given cube.
        interp_coords = [self.cube.coord(coord) for coord in self._interp_coords]
        sample_point_order = [self.cube.coord(coord) for coord, _ in sample_points]

        _new_scalar_dims = set()
        if collapse_scalar:
            for coord, points in sample_points:
                coord = self.cube.coord(coord)
                if np.array(points).ndim == 0:
                    _new_scalar_dims.add(self.coord_dims[interp_coords.index(coord)])
#                    _new_scalar_dims.update(self.cube.coord_dims(coord))

            # Map old dimensions to new (old dimensions no longer existing are defined as None).
            _dimension_mapping = {}
            skipped = 0
            for dim in range(data.ndim):
                if dim in _new_scalar_dims:
                    _dimension_mapping[dim] = None
                    skipped += 1
                else:
                    _dimension_mapping[dim] = dim - skipped

            # Remove the *specific* scalar axes from the data.
            data_index = [slice(None)] * self.cube.data.ndim
            for scalar_dim in _new_scalar_dims:
                data_index[scalar_dim] = 0
             
            data = data[tuple(data_index)]

        cube = self.cube
        new_cube = iris.cube.Cube(data)
        new_cube.metadata = cube.metadata

        def construct_new_coord_given_points(coord, points):
            # Handle what was previously a DimCoord which would no longer be
            # monotonic. 
            try:
                return iris.coords.DimCoord.from_coord(coord.copy(points))
            except ValueError:
                return iris.coords.AuxCoord.from_coord(coord).copy(points)
        
        dims_with_dim_coords = []
        
        # 2) Copy/interpolate the coordinates.
        for dim_coord in cube.dim_coords:
            dim, = cube.coord_dims(dim_coord)
            if dim_coord in interp_coords:
                new_points = sample_points[sample_point_order.index(dim_coord)][1]
                new_coord = construct_new_coord_given_points(dim_coord, new_points)
            elif set([dim]).intersection(set(self.coord_dims)):
                new_coord = self._resample_coord(sample_points, dim_coord, [dim], extrapolation_mode)
            else:
                new_coord = dim_coord.copy()

            if collapse_scalar:
                dim = _dimension_mapping[dim]

            # new_coord may no longer be a dim coord, so check we don't need
            # to add it as an aux coord (thus leaving the dim anonymous).
            if isinstance(new_coord, iris.coords.DimCoord) and dim is not None:
                new_cube._add_unique_dim_coord(new_coord, dim)
                dims_with_dim_coords.append(dim)
            else:
                new_cube._add_unique_aux_coord(new_coord, dim)

        for coord in cube.aux_coords:
            dims = cube.coord_dims(coord)
            if coord in interp_coords:
                new_points = sample_points[sample_point_order.index(coord)][1]
                new_coord = construct_new_coord_given_points(coord, new_points)
                orig = dims
#                dims = [self.coord_dims[self.ndims - interp_coords.index(coord) - 1]]
                dims = [self.coord_dims[interp_coords.index(coord)]] 
            elif set(dims).intersection(set(self.coord_dims)):
                new_coord = self._resample_coord(sample_points, coord, dims, extrapolation_mode)
            else:
                new_coord = coord.copy()
            
            if not collapse_scalar:
                new_dims = dims
            else:
                subsetter = [slice(None)] * new_coord.ndim
                new_dims = []
                for index, dim in enumerate(dims):
                    if _dimension_mapping[dim] is None:
                        subsetter[index] = 0
                    else:
                        new_dims.append(_dimension_mapping[dim])
                if len(new_dims) != len(dims):
                    new_coord = new_coord[tuple(subsetter)]
            
            if isinstance(new_coord, iris.coords.DimCoord) and len(new_dims) > 0 and new_dims[0] not in dims_with_dim_coords:
                new_cube._add_unique_dim_coord(new_coord, new_dims)
                dims_with_dim_coords.append(new_dims[0])
            else:
                new_cube._add_unique_aux_coord(new_coord, new_dims)

        return new_cube


def linear(cube, sample_points, extrapolation_mode='linear'):
    coords = []
    if isinstance(sample_points, dict):
        sample_points = sample_points.items()
    
    # catch the case where a user passes a single (coord/name, value) pair rather than a list of pairs
    if sample_points and not (isinstance(sample_points[0], collections.Container) and not isinstance(sample_points[0], basestring)):
        raise TypeError('Expecting the sample points to be a list of tuple pairs representing (coord, points), got a list of %s.' % type(sample_points[0]))

    for coord, _ in sample_points:
        coords.append(coord)
    interp = LinearInterpolator(cube, coords, extrapolation_mode)
    return interp.orthogonal_cube(sample_points, extrapolation_mode)


    