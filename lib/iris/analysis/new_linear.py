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
import scipy
import scipy.spatial
from scipy.interpolate.interpolate import interp1d

import iris.cube
import iris.coord_systems
import iris.coords
import iris.exceptions

from iris.experimental.regrid import _RegularGridInterpolator, _ndim_coords_from_arrays
from iris.analysis.interpolate import _resample_coord, Linear1dExtrapolator, _extend_circular_coord_and_data

import numpy.lib.stride_tricks as stride_tricks



class LinearInterpolator(object):
    def __init__(self, cube, interp_coords, extrapolation_mode='linear'):
        # Cube is "data_cube".
        self.cube = cube
        self.extrapolation_mode = extrapolation_mode
        self._bounds_error = (extrapolation_mode == 'error')
        
        self._interp_coords = interp_coords
        data_coords = [cube.coord(coord) for coord in interp_coords]
        
        # Triggers the loading - is that really necessary...?
        data = cube.data
        coord_points_list = []

        for coord in data_coords:
            if coord.ndim != 1:
                raise ValueError('Interpolation coords must be 1-d for '
                                 'non-triangulation based interpolation.')
            # xxx What about 1dim coords not bound to a dimension?
            coord_dim, = cube.coord_dims(coord)

            if getattr(coord, 'circular', False):
                coord_points, data = _extend_circular_coord_and_data(coord, data, coord_dim)
            else:
                coord_points = coord.points
                data = data
            coord_points_list.append([coord_points, coord_dim])
        
#        coord_points_list, coord_dims = zip(*sorted(coord_points_list, key=lambda (points, dim): dim))
        coord_points, self.coord_dims = zip(*coord_points_list)

        self.coord_decreasing = [np.all(np.diff(points) < 0) for points in coord_points]
        if np.any(self.coord_decreasing):
            coord_points = [points[::-1] if is_decreasing else points
                            for is_decreasing, points in zip(self.coord_decreasing, coord_points)]

        if list(self.coord_dims) != sorted(self.coord_dims):
            raise NotImplementedError("Haven't yet implemented the transpose problem. "
                                      "Should be easy enough...")

        if len(self.coord_dims) != len(set(self.coord_dims)):
            raise ValueError('Coordinates repeat a data dimension - the '
                             'interpolation would be over-specified.')
        shape = [cube.shape[ind] for ind in self.coord_dims]
        
        mock_data = stride_tricks.as_strided(np.array(0, dtype=float),
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

    def interpolate_data(self, coord_points, data, extrapolation_mode='linear'):
        # Note: Dimensions are never lost - at a minimum they will be of length one. The exception to this is the dimensions -1 
        # that are being interpolated - they currently are all squeezed into a single dimension.
        # Note: Interpolated coordinates are squeezed and pushed to the last dimension.
        
        if data.shape != self.cube.shape:
            raise ValueError('data being interpolated is not consistent with the data passed through.')
        try:
            coord_points = _ndim_coords_from_arrays(coord_points, len(self.coord_dims))
            if coord_points.shape[-1] != len(self.coord_dims):
                raise ValueError()
        except ValueError:
            raise ValueError('The given coordinates are not appropriate for'
                             ' the interpolator. There are {0} dimension(s)'
                             ' and ideally the given coordinate array should'
                             ' have a shape of (npts, {0}). Got an array of'
                             ' shape {1}.'
                             ''.format(len(self.coord_dims),
                                       np.asanyarray(coord_points).shape))

        if coord_points.dtype == object:
            raise ValueError('Perhaps inconsistently shaped arrays were passed as '
                             'coordinate points. The resulting numpy array has '
                             '"object" as its type.')

        if any(self.coord_decreasing):
            # XXX Needs to handle different shaped array.
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
            self._interpolator.values = sub_data
            r = self._interpolator(coord_points)
            result_data[interpolant_index] = r
        return result_data

    def orthogonal_points(self, sample_points, extrapolation_mode='linear'):
        # sample points are orthogonal...

        # catch the case where a user passes a single (coord/name, value) pair rather than a list of pairs
        if sample_points and not (isinstance(sample_points[0], collections.Container) and not isinstance(sample_points[0], basestring)):
            raise TypeError('Expecting the sample points to be a list of tuple pairs representing (coord, points), got a list of %s.' % type(sample_points[0]))

        coord_points = [None] * len(self.coord_dims)
        for coord, points in sample_points:
            if coord not in self._interp_coords:
                raise ValueError('Coord {} was not one of those passed to the '
                                 'constructor.'.format(coord))
            coord = self.cube.coord(coord)
            coord_dim, = self.cube.coord_dims(coord)
            coord_points[self.coord_dims.index(coord_dim)] = points
        if len(coord_points) > 1:
            coord_points = [arr.flatten() for arr in np.meshgrid(*coord_points)]
        coord_points = np.asanyarray(coord_points).T.reshape(-1, len(self.coord_dims))
        return self.interpolate_data(coord_points, self.cube.data, extrapolation_mode)
    
    def points(self, sample_points, extrapolation_mode='linear'):
        # Same as orthogonal.
        
        coord_points_dims = []
        coord_points_posn = []
        coord_points = [None] * len(self.coord_dims)
        shape_minus_interp = [length for dim, length in enumerate(self.cube.shape)
                              if dim not in self.coord_dims]
        sample_shape = [None] * len(sample_points)
        transpose = range(self.cube.ndim)

        for coord, points in sample_points:
            if coord not in self._interp_coords:
                raise ValueError('Coord {} was not one of those passed to the '
                                 'constructor.'.format(coord))
            coord = self.cube.coord(coord)
            coord_dim, = self.cube.coord_dims(coord)
            points = np.asanyarray(points)
            order_in_constructor = self.coord_dims.index(coord_dim)
            position_in_result = len(shape_minus_interp) + order_in_constructor
            transpose.insert(coord_dim, transpose.pop(position_in_result))
            sample_shape[order_in_constructor] = points.size
            coord_points[order_in_constructor] = points
            coord_points_posn.append(self.coord_dims.index(coord_dim))
            coord_points_dims.append(coord_dim)

        target_shape = shape_minus_interp + sample_shape

        if len(coord_points) > 1:
            coord_points = [arr.flatten() for arr in np.meshgrid(*coord_points)]
        coord_points = np.asanyarray(coord_points).T.reshape(-1, len(self.coord_dims))
        
        coord_points = _ndim_coords_from_arrays(coord_points, self.cube.ndim)
        
        print 'Posn: ', coord_points_posn
        print 'Target: ', target_shape
        print 'Transpose:', transpose
        
        data = self.interpolate_data(coord_points, self.cube.data, extrapolation_mode)
        data = data.reshape(target_shape)
        data = np.transpose(data, transpose)
        
        return data

    def construct_cube(self, new_data, orig_cube):
        new_cube = iris.cube.Cube(new_data)
        new_cube.metadata = cube.metadata

        # If requested_points is an array scalar then `new_cube` will
        # have one less dimension than `cube`. (The `sample_dim`
        # dimension will vanish.) In which case we build a mapping from
        # `cube` dimensions to `new_cube` dimensions.
        dim_mapping = None
        if new_cube.ndim != cube.ndim:
            dim_mapping = {i: i for i in range(sample_dim)}
            dim_mapping[sample_dim] = None
            for i in range(sample_dim + 1, cube.ndim):
                dim_mapping[i] = i - 1

        # 2) Copy/interpolate the coordinates.
        for dim_coord in cube.dim_coords:
            dims = cube.coord_dims(dim_coord)
            if sample_dim in dims:
                new_coord = _resample_coord(dim_coord, src_coord, direction,
                                            requested_points, interpolate)
            else:
                new_coord = dim_coord.copy()
            if dim_mapping:
                dims = [dim_mapping[dim] for dim in dims
                            if dim_mapping[dim] is not None]
            if isinstance(new_coord, iris.coords.DimCoord) and dims:
                new_cube.add_dim_coord(new_coord, dims)
            else:
                new_cube.add_aux_coord(new_coord, dims)

        for coord in cube.aux_coords:
            dims = cube.coord_dims(coord)
            if sample_dim in dims:
                new_coord = _resample_coord(coord, src_coord, direction,
                                            requested_points, interpolate)
            else:
                new_coord = coord.copy()
            if dim_mapping:
                dims = [dim_mapping[dim] for dim in dims
                            if dim_mapping[dim] is not None]
            new_cube.add_aux_coord(new_coord, dims)

        return new_cube


def linear(cube, sample_points, extrapolation_mode='linear'):
    # Iterate over all of the requested keys in the given points_dict calling this routine repeatedly.
    if len(sample_points) > 1:
        result = cube
        for coord, cells in sample_points:
            result = linear(result, [(coord, cells)], extrapolation_mode=extrapolation_mode)
        return result

    else:
        # Now we must be down to a single sample coordinate and its
        # values.
        src_coord, requested_points = sample_points[0]
        sample_values = np.array(requested_points)

        # 1) Define the interpolation characteristics.

        # Get the sample dimension (which we have already tested is not None)
        sample_dim = cube.coord_dims(src_coord)[0]


        # Map all the requested values into the range of the source
        # data (centered over the centre of the source data to allow
        # extrapolation where required).
        src_axis = iris.util.guess_coord_axis(src_coord)
        if src_axis == 'X' and src_coord.units.modulus:
            modulus = src_coord.units.modulus
            offset = (src_points.max() + src_points.min() - modulus) * 0.5
            sample_values = ((sample_values - offset) % modulus) + offset

        if len(src_points) == 1:
            if extrapolation_mode == 'error' and \
                    np.any(sample_values != src_points):
                raise ValueError('Attempting to extrapolate from a single '
                                 'point with extrapolation mode set '
                                 'to {!r}.'.format(extrapolation_mode))
            direction = 0

            def interpolate(fx, new_x, axis=None, **kwargs):
                # All kwargs other than axis are ignored.
                if axis is None:
                    axis = -1
                new_x = np.array(new_x)
                new_shape = list(fx.shape)
                new_shape[axis] = new_x.size
                fx = np.broadcast_arrays(fx, np.empty(new_shape))[0].copy()
                if extrapolation_mode == 'nan':
                    indices = [slice(None)] * fx.ndim
                    indices[axis] = new_x != src_points
                    fx[tuple(indices)] = np.nan
                # If new_x is a scalar, then remove the dimension from fx.
                if not new_x.shape:
                    del new_shape[axis]
                    fx.shape = new_shape
                return fx
        else:
            monotonic, direction = iris.util.monotonic(src_points,
                                                       return_direction=True)
            if not monotonic:
                raise ValueError('Unable to linearly interpolate this '
                                 'cube as the coordinate {!r} is not '
                                 'monotonic'.format(src_coord.name()))

            # SciPy's interp1d requires monotonic increasing coord values.
            if direction == -1:
                src_points = iris.util.reverse(src_points, axes=0)
                data = iris.util.reverse(data, axes=sample_dim)

            # Wrap it all up in a function which makes the right kind of
            # interpolator/extrapolator.
            # NB. This uses a closure to capture the values of src_points,
            # bounds_error, and extrapolation_mode.
            def interpolate(fx, new_x, **kwargs):
                # SciPy's interp1d needs float values, so if we're given
                # integer values, convert them to the smallest possible
                # float dtype that can accurately preserve the values.
                if fx.dtype.kind == 'i':
                    fx = fx.astype(np.promote_types(fx.dtype, np.float16))
                x = src_points.astype(fx.dtype)
                interpolator = interp1d(x, fx, kind='linear',
                                        bounds_error=bounds_error, **kwargs)
                if extrapolation_mode == 'linear':
                    interpolator = Linear1dExtrapolator(interpolator)
                new_fx = interpolator(np.array(new_x, dtype=fx.dtype))
                return new_fx

        # 2) Interpolate the data and produce our new Cube.
        if isinstance(data, ma.MaskedArray):
            # interpolate data, ignoring the mask
            new_data = interpolate(data.data, sample_values, axis=sample_dim,
                                   copy=False)
            # Mask out any results which contain a non-zero contribution
            # from a masked value when interpolated from mask cast as 1,0.
            mask_dataset = ma.getmaskarray(data).astype(float)
            new_mask = interpolate(mask_dataset, sample_values,
                                   axis=sample_dim, copy=False) > 0
            # create new_data masked array
            new_data = ma.MaskedArray(new_data, mask=new_mask)
        else:
            new_data = interpolate(data, sample_values, axis=sample_dim,
                                   copy=False)
        new_cube = iris.cube.Cube(new_data)
        new_cube.metadata = cube.metadata

        # If requested_points is an array scalar then `new_cube` will
        # have one less dimension than `cube`. (The `sample_dim`
        # dimension will vanish.) In which case we build a mapping from
        # `cube` dimensions to `new_cube` dimensions.
        dim_mapping = None
        if new_cube.ndim != cube.ndim:
            dim_mapping = {i: i for i in range(sample_dim)}
            dim_mapping[sample_dim] = None
            for i in range(sample_dim + 1, cube.ndim):
                dim_mapping[i] = i - 1

        # 2) Copy/interpolate the coordinates.
        for dim_coord in cube.dim_coords:
            dims = cube.coord_dims(dim_coord)
            if sample_dim in dims:
                new_coord = _resample_coord(dim_coord, src_coord, direction,
                                            requested_points, interpolate)
            else:
                new_coord = dim_coord.copy()
            if dim_mapping:
                dims = [dim_mapping[dim] for dim in dims
                            if dim_mapping[dim] is not None]
            if isinstance(new_coord, iris.coords.DimCoord) and dims:
                new_cube.add_dim_coord(new_coord, dims)
            else:
                new_cube.add_aux_coord(new_coord, dims)

        for coord in cube.aux_coords:
            dims = cube.coord_dims(coord)
            if sample_dim in dims:
                new_coord = _resample_coord(coord, src_coord, direction,
                                            requested_points, interpolate)
            else:
                new_coord = coord.copy()
            if dim_mapping:
                dims = [dim_mapping[dim] for dim in dims
                            if dim_mapping[dim] is not None]
            new_cube.add_aux_coord(new_coord, dims)

        return new_cube


if __name__ == '__main__':
    import iris.tests.stock as stock
    from iris.analysis.new_linear import linear, LinearInterpolator
    
    from nose.tools import assert_raises_regexp
    
    cube = stock.simple_3d_w_multidim_coords()
    cube.add_aux_coord(iris.coords.DimCoord(range(2), 'height'), 0)
    cube.add_dim_coord(iris.coords.DimCoord(range(3), 'latitude'), 1)
    cube.add_dim_coord(iris.coords.DimCoord(range(4), 'longitude'), 2)
    cube.data = cube.data.astype(np.float)

    data = np.arange(24).reshape(2, 3, 4)

    interpolator = LinearInterpolator(cube, ['latitude'])
    assert np.all(interpolator.interpolate_data([1, 2], data) ==
                  np.transpose(data[:, 1:3, :], [0, 2, 1])), 'Wrong result'
    assert np.all(interpolator.interpolate_data([1.5], data) ==
                  data[:, 1:3, :].mean(axis=1)[:, :, np.newaxis]), 'Wrong result'
    # DO we care about this case?... I think so
#    interpolator.interpolate_data([-1.5], data[0])
    
    # Extrapolation (linear)
    expected = np.transpose(data[:, 0:1] - (data[:, 1:2] - data[:, 0:1]), [0, 2, 1])
    assert np.all(expected == interpolator.interpolate_data([-1], data)), 'Wrong result'
    
    interpolator = LinearInterpolator(cube, ['latitude'])
    interpolator._interpolator.values = interpolator._interpolator.values.astype(np.float64)
    assert np.all(np.isnan(interpolator.interpolate_data([-1], data.astype(np.float64), extrapolation_mode='nan'))), 'Wrong result'
    
    with assert_raises_regexp(ValueError, 'The resulting numpy array has "object" as its type'):
        interpolator.interpolate_data([[1, 2], [1]], cube.data)
    
    with assert_raises_regexp(ValueError, 'One of the requested xi is out of bounds in dimension 0'):
        assert np.all(np.isnan(interpolator.interpolate_data([-1], data.astype(np.float64), extrapolation_mode='error'))), 'Wrong result'
        

    interpolator = LinearInterpolator(cube, ['latitude', 'longitude'])
    assert np.all(interpolator.interpolate_data([[1, 2], [2, 2]], data) ==
                  data[:, 1:3, 2:3].reshape(-1, 2)), 'Wrong result'
    
    interpolator = LinearInterpolator(cube, ['height', 'longitude'])
    assert np.all(interpolator.interpolate_data([[1, 1], [1, 2]], data) == 
                  data[1:2, :, 1:3]), 'Wrong result'
    
    assert np.all(interpolator.interpolate_data([[1, 1]], data) == 
                  interpolator.interpolate_data([1, 1], data))
    with assert_raises_regexp(ValueError, 'coordinates are not appropriate for the interpolator'):
        print interpolator.interpolate_data([1], data)
    
    with assert_raises_regexp(ValueError, ('There are 2 '
                                           r'dimension\(s\) and ideally the given coordinate array '
                                           r'should have a shape of \(npts\, 2\). Got an array of shape '
                                           r'\(2\, 4\)')):
        interpolator.interpolate_data([[1, 1, 2, 2], [1, 2, 1, 2]], cube.data)
    
    with assert_raises_regexp(ValueError, 'data being interpolated is not consistent with the data passed through'):
        print interpolator.interpolate_data([1], data[0])

    cube.data = cube.data.astype(int)
    interpolator = LinearInterpolator(cube, ['latitude'])
    assert interpolator.interpolate_data([0.125], data).dtype == np.float64, 'Wrong type'
    cube.data = cube.data.astype(np.float)

    # Points interface.
    r = interpolator.orthogonal_points([['latitude', [1]]]) # TODO test it calls interpolate_data appropriately.
    
    interpolator = LinearInterpolator(cube, ['latitude', 'longitude'])
    assert np.all(interpolator.orthogonal_points([['longitude', [1, 2]], ['latitude', [1, 2]]]) == 
                  interpolator.interpolate_data([[1, 1], [2, 1], [1, 2], [2, 2]],
                                                cube.data)), 'Wrong type'
    
    interpolator = LinearInterpolator(cube[:, 0:1, 0], ['latitude'])
    # Linear extrapolation of a single valued element.
    assert np.all(interpolator.interpolate_data([1001], cube[:, 0:1, 0].data) ==
                  cube[:, 0:1, 0].data)
    assert np.all(np.isnan(interpolator.interpolate_data([1001], cube[:, 0:1, 0].data,
                                                         extrapolation_mode='nan')))
    # No extrapolation for a single length dimension.
    assert np.all(interpolator.interpolate_data([0], cube[:, 0:1, 0].data,
                                                         extrapolation_mode='nan') ==
                  cube[:, 0:1, 0].data)
    
    # Monotonic.
    interpolator = LinearInterpolator(cube[:, ::-1], ['latitude'])
    assert np.all(interpolator.interpolate_data([0], cube[:, ::-1].data) ==
                  np.transpose(cube.data[:, 0:1], [0, 2, 1])), 'Wrong result'
    
    
    interpolator = LinearInterpolator(cube, ['height', 'longitude'])
    r = interpolator.points([['longitude', [0]], ['height', [0, 1]]])
    r2 = interpolator.points([['height', [0, 1]], ['longitude', [0]]])
    expected = cube.data[0:2, :, 0:1]
    assert np.all(r == expected), 'Wrong result'
    assert np.all(r == r2), 'Wrong result'
    interpolator = LinearInterpolator(cube, ['longitude', 'height'])
    r3 = interpolator.points([['longitude', [0]], ['height', [0, 1]]])
    assert np.all(r == r3), 'Wrong result'
    r4 = interpolator.points([['height', [0, 1]], ['longitude', [0]]])
    assert np.all(r == r4), 'Wrong result'

    print 'Done'