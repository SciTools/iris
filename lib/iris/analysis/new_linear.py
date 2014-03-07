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
Interpolation and re-gridding routines.

See also: :mod:`NumPy <numpy>`, and :ref:`SciPy <scipy:modindex>`.

"""
import collections
import warnings

import numpy as np
import numpy.ma as ma
from numpy.lib.stride_tricks import as_strided
import scipy
import scipy.spatial
from scipy.interpolate.interpolate import interp1d

import iris.cube
import iris.coord_systems
import iris.coords
import iris.exceptions

from iris.experimental.regrid import _RegularGridInterpolator, _ndim_coords_from_arrays
from iris.analysis.interpolate import _resample_coord, Linear1dExtrapolator, _extend_circular_coord_and_data


class LinearInterpolator(object):
    def __init__(self, cube, interp_coords, extrapolation_mode='linear'):
        # Cube is "data_cube".
        self.cube = cube
        self.extrapolation_mode = extrapolation_mode
        self._bounds_error = (extrapolation_mode == 'error')
        
        self._interp_coords = interp_coords
        self.ndims = len(interp_coords)
        data_coords = [cube.coord(coord) for coord in interp_coords]
        
        # Triggers the loading - is that really necessary...?
        data = cube.data
        coord_points_list = []

        for coord in data_coords:
            if coord.ndim != 1:
                raise ValueError('Interpolation coords must be 1-d for '
                                 'non-triangulation based interpolation.')
            # xxx What about 1dim coords not bound to a dimension?
            coord_dims = cube.coord_dims(coord)

            if getattr(coord, 'circular', False):
                assert coord.ndim == 1
                coord_points, data = _extend_circular_coord_and_data(coord, data, coord_dims[0])
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
        shape = [cube.shape[ind] for ind in self.coord_dims]
        
        mock_data = as_strided(np.array(0, dtype=float),
                               shape=shape, strides=[0] * len(self.coord_dims))

        self._interpolator = _RegularGridInterpolator(coord_points, mock_data,
                                                      fill_value=None,
                                                      bounds_error=False) 

    def _update_extrapolation_mode(self, mode):
        if mode == 'linear':
            self._interpolator.bounds_error = False
            self._interpolator.fill_value = None
        elif mode == 'error':
            self._interpolator.bounds_error = True
        elif mode == 'nan':
            self._interpolator.bounds_error = False
            if not np.can_cast(np.nan, self._interpolator.values.dtype):
                raise ValueError("The interpolation array's dtype does not "
                                 "support NaNs for extrapolation.")
            self._interpolator.fill_value = np.nan
            
        else:
            # TODO - implement mask extrapolation...
            raise ValueError('Extrapolation mode {!r} not supported.'.format(mode))

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

        if any(self.coord_decreasing):
            data_indexer = [slice(None)] * data.ndim
            for is_decreasing, dim in zip(self.coord_decreasing, self.coord_dims):
                if is_decreasing:
                    data_indexer[dim] = slice(None, None, -1)
            data = data[tuple(data_indexer)]

        self._update_extrapolation_mode(extrapolation_mode)

        # TODO: Move interpolator coordinates to mirror the centre of the given coord_points.

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
        
        result_data = np.empty(result_shape, dtype=self._interpolator.values.dtype)

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
#                interpolated_shape[coord_dim] = points.size
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
        print 'Data:', data.shape, target_shape
        return np.transpose(data.reshape(target_shape), transpose_order)

#    def _resample_coord(self, sample_points, coord, ):

    def _orthogonal_points_preserve_dimensionality(self, sample_points, data,
                                                   extrapolation_mode='linear', data_dims=None):
        data = self.orthogonal_points(sample_points, data, extrapolation_mode, data_dims)
        index = tuple(0 if dim not in data_dims else slice(None)
                      for dim in range(self.cube.ndim))
        r = data[index]
        return r
    
    def _resample_coord(self, sample_points, coord, coord_dims, extrapolation_mode):
        coord_points = coord.points
        if getattr(coord, 'circular', False):
            assert coord.ndim == 1, 'Only DimCoords are circular'
            modulus = np.array(coord.units.modulus or 0,
                               dtype=coord.dtype)
            coord_points = np.append(coord.points,
                                     coord.points[0] + modulus)

        new_points = self._orthogonal_points_preserve_dimensionality(sample_points, coord_points, extrapolation_mode, coord_dims)

        # Watch out for DimCoord instances that are no longer monotonic
        # after the resampling.
        try:
            new_coord = coord.copy(new_points)
        except ValueError:
            new_coord = iris.coords.AuxCoord.from_coord(coord).copy(new_points)
        return new_coord

    def orthogonal_cube(self, sample_points, extrapolation_mode='linear'):
        data = self.cube.data
        data = self.orthogonal_points(sample_points,
                                      data, extrapolation_mode)

        cube = self.cube
        new_cube = iris.cube.Cube(data)
        new_cube.metadata = cube.metadata

        # 2) Copy/interpolate the coordinates.
        for dim_coord in cube.dim_coords:
            dim, = cube.coord_dims(dim_coord)
            if set([dim]).intersection(set(self.coord_dims)):
                new_coord = self._resample_coord(sample_points, dim_coord, [dim], extrapolation_mode)
            else:
                new_coord = dim_coord.copy()
            # new_coord may no longer be a dim coord, so check we don't need
            # to add it as an aux coord (thus leaving the dim anonymous). 
            if isinstance(new_coord, iris.coords.DimCoord):
                new_cube.add_dim_coord(new_coord, dim)
            else:
                new_cube.add_aux_coord(new_coord, dim)

        for coord in cube.aux_coords:
            dims = cube.coord_dims(coord)
            if set(dims).intersection(set(self.coord_dims)):
                new_coord = self._resample_coord(sample_points, coord, dims, extrapolation_mode)
            else:
                new_coord = coord.copy()
            new_cube.add_aux_coord(new_coord, dims)

        return new_cube


def linear(cube, sample_points, extrapolation_mode='linear'):
    coords = []
    if isinstance(sample_points, dict):
        sample_points = sample_points.items()
    for coord, _ in sample_points:
        coords.append(coord)
    interp = LinearInterpolator(cube, coords, extrapolation_mode)
    return interp.orthogonal_cube(sample_points, extrapolation_mode)


class TriangulatedLinearInterpolator(LinearInterpolator):
    def __init__(self, cube, interp_coords, extrapolation_mode='linear'):
        # Cube is "data_cube".
        self.cube = cube
        self.extrapolation_mode = extrapolation_mode
        self._bounds_error = (extrapolation_mode == 'error')
        
        self._interp_coords = interp_coords
        self.ndims = len(interp_coords)
        data_coords = [cube.coord(coord) for coord in interp_coords]

        # Triggers the loading - is that really necessary...?
        data = cube.data
        coord_points_list = []

        unique_dims = set()

        for coord in data_coords:
            # xxx What about 1dim coords not bound to a dimension?
            coord_dims = cube.coord_dims(coord)

            if getattr(coord, 'circular', False):
                assert coord.ndim == 1
                coord_points, data = _extend_circular_coord_and_data(coord, data, coord_dims[0])
            else:
                coord_points = coord.points
                data = data
            unique_dims.update(coord_dims)
            coord_points_list.append([coord_points, coord_dims])

        shape = [cube.shape[ind] for ind in unique_dims]
        
        i = 0
        cube_dim_to_interp_dim = {}
        for dim in range(cube.ndim):
            if dim in unique_dims:
                cube_dim_to_interp_dim[dim] = i
                i += 1
        
        for coord_points_pair in coord_points_list:
            points, dims = coord_points_pair
            dims_of_interest = [cube_dim_to_interp_dim[dim] for dim in dims
                                if dim in unique_dims]
            coord_points_pair[0] = _array1_in_terms_of_2(points, dims_of_interest, shape).flatten()
        
        shape = [cube.shape[ind] for ind in unique_dims]
        
        
#        coord_points_list, coord_dims = zip(*sorted(coord_points_list, key=lambda (points, dim): dim))
        coord_points, self.coord_dims = zip(*coord_points_list)
        self.coord_dims = list(set(list(np.concatenate(self.coord_dims))))

        #: Coord decreasing - doesn't matter for this interpolation.
        self.coord_decreasing = [False] * len(interp_coords)
        
        coord_points = np.array(coord_points).T
        
        if len(interp_coords) != len(set(self.coord_dims)):
            raise ValueError('Coordinates repeat a data dimension - the '
                             'interpolation would be over-specified.')
        shape = [cube.shape[ind] for ind in set(self.coord_dims)]
        
        mock_data = as_strided(np.array(0, dtype=float),
                               shape=(np.product(shape), ), strides=[0])
        
        from scipy.interpolate import LinearNDInterpolator
        
        self._interpolator = LinearNDInterpolator(coord_points, mock_data) 
        
    def _interpolate_data_at_coord_points(self, data, coord_points):
        # XXX Do we have to force it into float64?
        self._interpolator.values = data.reshape(-1, 1).astype(np.float64)
        return self._interpolator(coord_points)


def _array1_in_terms_of_2(array, array_dims_into_2, array_2_shape, expand=True):
    """
    Take any array, which is a subset of array 2 in shape, and for which
    each dimension of 1 can be mapped to a dimension of 2, and return a full
    array which is of the same shape as array 2.
    
    XXX Ugly docstring!
    
    """
    assert array.ndim == len(array_dims_into_2)

    array1_dims_to_array2 = {i: dim for i, dim in enumerate(array_dims_into_2)}
    array2_dims_to_array1 = {dim: array_dims_into_2.index(dim)
                             for dim in range(len(array_2_shape))
                             if dim in array_dims_into_2}

    # Re-order the array to match the order of array 2.
    transpose = sorted(range(array.ndim), key=array1_dims_to_array2.get)
    array = np.transpose(array, transpose)

    # Extend the dimensions of the array to match the number of dimensions
    # in array 2.
    new_shape = [1 if dim not in array2_dims_to_array1 else length
                 for dim, length in enumerate(array_2_shape)]
    array = array.reshape(new_shape)
    
    if expand:
        new_strides = [0 if dim not in array2_dims_to_array1 else stride
                       for dim, stride in enumerate(array.strides)]
        array = as_strided(array, strides=new_strides, shape=array_2_shape)
        
    return array


if __name__ == '__main__':
    a1 = np.arange(12).reshape(4, 3)
    a2 = np.arange(24).reshape(2, 3, 4)
    r = _array1_in_terms_of_2(a1, [2, 1], a2.shape, False)
    assert np.all(r[0] == a1.T)
    
    r = _array1_in_terms_of_2(a1, [2, 1], a2.shape, True)
    assert r.shape == a2.shape
    

if __name__ == '__main__':
    import iris.tests.stock as stock
    from iris.analysis.new_linear import linear, LinearInterpolator, TriangulatedLinearInterpolator

    from nose.tools import assert_raises_regexp
    
    cube = stock.simple_3d_w_multidim_coords()
    cube.add_aux_coord(iris.coords.DimCoord(range(2), 'height'), 0)
    cube.add_dim_coord(iris.coords.DimCoord(range(3), 'latitude'), 1)
    cube.add_dim_coord(iris.coords.DimCoord(range(4), 'longitude'), 2)
    cube.data = cube.data.astype(np.float)

    data = np.arange(24).reshape(2, 3, 4)

#    interpolator = TriangulatedLinearInterpolator(cube, ['latitude', 'longitude'])
#    print interpolator.interpolate_data([0, 0], cube.data, 'nan')
    
    interpolator = TriangulatedLinearInterpolator(cube, ['bar', 'foo'])
    print interpolator.interpolate_data([0, 0], cube.data, 'nan')
    print interpolator.interpolate_data([2.5, -7.5], cube.data, 'nan')
    print interpolator.orthogonal_points([['foo', -7.5], ['bar', 2.5]], cube.data, 'nan')
    print interpolator.orthogonal_points([['foo', -7.5], ['bar', 4]], cube.data, 'nan')
    
    print interpolator.orthogonal_points([['foo', np.linspace(-7.5, -5, 10)], ['bar', 4]],
                                         cube.data, 'nan')
    r = interpolator.orthogonal_cube([['foo', np.linspace(-7.5, -5, 10)], ['bar', 4]],
                                     'nan')
    print r.coord('bar')
    print r.coord('foo')


if __name__ == '__mains__':
    fname = '/data/local/dataZoo/NetCDF/ORCA1/NEMO_ORCA1_CF.nc'
    sst = iris.load_cube(fname, 'sea_surface_temperature')
    
    import iris.quickplot as qplt
    import matplotlib.pyplot as plt
    
    plt.switch_backend('tkagg')
    
    interpolator = TriangulatedLinearInterpolator(sst, ['latitude', 'longitude'])
    arctic_circle = interpolator.orthogonal_cube([['longitude', np.linspace(-180, 180, 180)],
                                                  ['latitude', 60]], extrapolation_mode='nan')
    arctic_circle.attributes.clear()
    print sst
    print arctic_circle
    arctic_circle = arctic_circle[0, 0, :]
    
    print arctic_circle.data
    
    
    qplt.plot(arctic_circle.coord('longitude'), arctic_circle)
    plt.show()
    import matplotlib
    print matplotlib.get_backend()