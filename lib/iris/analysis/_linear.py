# (C) British Crown Copyright 2014, Met Office
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
from collections import namedtuple
from itertools import product
import operator

import numpy as np
import numpy.ma as ma
from numpy.lib.stride_tricks import as_strided

from iris.analysis._interpolation import get_xy_dim_coords, snapshot_grid
from iris.analysis._scipy_interpolate import _RegularGridInterpolator
from iris.analysis.cartography import wrap_lons as wrap_circular_points
from iris.coords import DimCoord, AuxCoord
import iris.cube
import iris.experimental.regrid as eregrid


_DEFAULT_DTYPE = np.float16

_LinearExtrapMode = namedtuple('_LinearExtrapMode', ['bounds_error',
                                                     'fill_value',
                                                     'mask_fill_value',
                                                     'force_mask'])
_LINEAR_EXTRAPOLATION_MODES = {
    'linear': _LinearExtrapMode(False, None, None, False),
    'error': _LinearExtrapMode(True, 0, 0, False),
    'nan': _LinearExtrapMode(False, np.nan, 0, False),
    'mask': _LinearExtrapMode(False, np.nan, 1, True),
    'nanmask': _LinearExtrapMode(False, np.nan, 1, False)
}


def _extend_circular_coord(coord, points):
    """
    Return coordinates points with a shape extended by one
    This is common when dealing with circular coordinates.

    """
    modulus = np.array(coord.units.modulus or 0,
                       dtype=coord.dtype)
    points = np.append(points, points[0] + modulus)
    return points


def _extend_circular_coord_and_data(coord, data, coord_dim):
    """
    Return coordinate points and a data array with a shape extended by one
    in the coord_dim axis. This is common when dealing with circular
    coordinates.

    """
    points = _extend_circular_coord(coord, coord.points)
    data = _extend_circular_data(data, coord_dim)
    return points, data


def _extend_circular_data(data, coord_dim):
    coord_slice_in_cube = [slice(None)] * data.ndim
    coord_slice_in_cube[coord_dim] = slice(0, 1)

    # TODO: Restore this code after resolution of the following issue:
    # https://github.com/numpy/numpy/issues/478
    # data = np.append(cube.data,
    #                  cube.data[tuple(coord_slice_in_cube)],
    #                  axis=sample_dim)
    # This is the alternative, temporary workaround.
    # It doesn't use append on an nD mask.
    if not (isinstance(data, ma.MaskedArray) and
            not isinstance(data.mask, np.ndarray)) or \
            len(data.mask.shape) == 0:
        data = np.append(data,
                         data[tuple(coord_slice_in_cube)],
                         axis=coord_dim)
    else:
        new_data = np.append(data.data,
                             data.data[tuple(coord_slice_in_cube)],
                             axis=coord_dim)
        new_mask = np.append(data.mask,
                             data.mask[tuple(coord_slice_in_cube)],
                             axis=coord_dim)
        data = ma.array(new_data, mask=new_mask)
    return data


def _canonical_sample_points(coords, sample_points):
    """
    Return the canonical form of the points values.

    Ensures that any points supplied as datetime objects, or similar,
    are converted to their numeric form.

    """
    canonical_sample_points = []
    for coord, points in zip(coords, sample_points):
        if coord.units.is_time_reference():
            def convert_date(date):
                try:
                    date = coord.units.date2num(date)
                except AttributeError:
                    pass
                return date
            convert_dates = np.vectorize(convert_date, [np.dtype(float)])
            points = convert_dates(points)
        canonical_sample_points.append(points)
    return canonical_sample_points


class LinearInterpolator(object):
    """
    This class provides support for performing linear interpolation over
    one or more orthogonal dimensions.

    """
    def __init__(self, src_cube, coords, extrapolation_mode='linear'):
        """
        Perform linear interpolation over one or more orthogonal coordinates.

        Args:

        * src_cube:
            The :class:`iris.cube.Cube` which is to be interpolated.
        * coords:
            The names or coordinate instances which are to be
            interpolated over

        Kwargs:

        * extrapolation_mode:
            Must be one of the following strings:

              * 'linear' - The extrapolation points will be calculated by
                extending the gradient of the closest two points.
              * 'nan' - The extrapolation points will be be set to NaN.
              * 'error' - A ValueError exception will be raised, notifying an
                attempt to extrapolate.
              * 'mask' - The extrapolation points will always be masked, even
                if the source data is not a MaskedArray.
              * 'nanmask' - If the source data is a MaskedArray the
                extrapolation points will be masked. Otherwise they will be
                set to NaN.

            Default mode of extrapolation is 'linear'

        """
        # Trigger any deferred loading of the source cube's data and snapshot
        # its state to ensure that the interpolator is impervious to external
        # changes to the original source cube. The data is loaded to prevent
        # the snaphot having lazy data, avoiding the potential for the
        # same data to be loaded again and again.
        if src_cube.has_lazy_data():
            src_cube.data
        self._src_cube = src_cube.copy()
        # Coordinates defining the dimensions to be interpolated.
        self._src_coords = [self._src_cube.coord(coord) for coord in coords]
        # The extrapolation mode.
        if extrapolation_mode not in _LINEAR_EXTRAPOLATION_MODES:
            msg = 'Extrapolation mode {!r} not supported.'
            raise ValueError(msg.format(extrapolation_mode))
        self._mode = extrapolation_mode
        # The point values defining the dimensions to be interpolated.
        self._src_points = []
        # A list of flags indicating dimensions that need to be reversed.
        self._coord_decreasing = []
        # The cube dimensions to be interpolated over.
        self._interp_dims = []
        # meta-data to support circular data-sets.
        self._circulars = []
        # Instance of the interpolator that performs the actual interpolation.
        self._interpolator = None

        # Perform initial start-up configuration and validation.
        self._setup()

    @property
    def cube(self):
        return self._src_cube

    @property
    def coords(self):
        return self._src_coords

    @property
    def extrapolation_mode(self):
        return self._mode

    def _account_for_circular(self, points, data):
        """
        Extend the given data array, and re-centralise coordinate points
        for circular (1D) coordinates.

        """
        for (circular, modulus, index, dim, offset) in self._circulars:
            if modulus:
                # Map all the requested values into the range of the source
                # data (centred over the centre of the source data to allow
                # extrapolation where required).
                points[:, index] = wrap_circular_points(points[:, index],
                                                        offset, modulus)

            # Also extend data if circular (to match the coord points, which
            # 'setup' already extended).
            if circular:
                data = _extend_circular_data(data, dim)

        return points, data

    def _account_for_inverted(self, data):
        if np.any(self._coord_decreasing):
            dim_slices = [slice(None)] * data.ndim
            for interp_dim, flip in zip(self._interp_dims,
                                        self._coord_decreasing):
                if flip:
                    dim_slices[interp_dim] = slice(-1, None, -1)
            data = data[dim_slices]
        return data

    def _interpolate(self, data, interp_points):
        """
        Interpolate a data array over N dimensions.

        Create and cache the underlying interpolator instance before invoking
        it to perform interpolation over the data at the given coordinate point
        values.

        * data (ndarray):
            A data array, to be interpolated in its first 'N' dimensions.

        * interp_points (ndarray):
            An array of interpolation coordinate values.
            Its shape is (..., N) where N is the number of interpolation
            dimensions.
            "interp_points[..., i]" are interpolation point values for the i'th
            coordinate, which is mapped to the i'th data dimension.
            The other (leading) dimensions index over the different required
            sample points.

        Returns:

            A :class:`np.ndarray`.  Its shape is "points_shape + extra_shape",
            where "extra_shape" is the remaining non-interpolated dimensions of
            the data array (i.e. 'data.shape[N:]'), and "points_shape" is the
            leading dimensions of interp_points,
            (i.e. 'interp_points.shape[:-1]').

        """
        dtype = self._interpolated_dtype(data.dtype)
        if data.dtype != dtype:
            # Perform dtype promotion.
            data = data.astype(dtype)

        mode = _LINEAR_EXTRAPOLATION_MODES[self._mode]
        if self._interpolator is None:
            # Cache the interpolator instance.
            # NB. The constructor of the _RegularGridInterpolator class does
            # some unnecessary checks on the fill_value parameter,
            # so we set it afterwards instead. Sneaky. ;-)
            self._interpolator = _RegularGridInterpolator(
                self._src_points, data, bounds_error=mode.bounds_error,
                fill_value=None)
        else:
            self._interpolator.values = data

        # We may be re-using a cached interpolator, so ensure the fill
        # value is set appropriately for extrapolating data values.
        self._interpolator.fill_value = mode.fill_value
        result = self._interpolator(interp_points)

        if result.dtype != data.dtype:
            # Cast the data dtype to be as expected. Note that, the dtype
            # of the interpolated result is influenced by the dtype of the
            # interpolation points.
            result = result.astype(data.dtype)

        if np.ma.isMaskedArray(data) or mode.force_mask:
            # NB. np.ma.getmaskarray returns an array of `False` if
            # `data` is not a masked array.
            src_mask = np.ma.getmaskarray(data)
            # Switch the extrapolation to work with mask values.
            self._interpolator.fill_value = mode.mask_fill_value
            self._interpolator.values = src_mask
            mask_fraction = self._interpolator(interp_points)
            new_mask = (mask_fraction > 0)
            if isinstance(data, ma.MaskedArray) or np.any(new_mask):
                result = np.ma.MaskedArray(result, new_mask)

        return result

    def _resample_coord(self, sample_points, coord, coord_dims):
        """
        Interpolate the given coordinate at the provided sample points.

        """
        # NB. This section is ripe for improvement:
        # - Internally self._points() expands coord.points to the same
        #   N-dimensional shape as the cube's data, but it doesn't
        #   collapse it again before returning so we have to do that
        #   here.
        # - By expanding to N dimensions self._points() is doing
        #   unnecessary work.
        data = self._points(sample_points, coord.points, coord_dims)
        index = tuple(0 if dim not in coord_dims else slice(None)
                      for dim in range(self._src_cube.ndim))
        new_points = data[index]
        # Watch out for DimCoord instances that are no longer monotonic
        # after the resampling.
        try:
            new_coord = coord.copy(new_points)
        except ValueError:
            aux_coord = AuxCoord.from_coord(coord)
            new_coord = aux_coord.copy(new_points)
        return new_coord

    def _setup(self):
        """
        Perform initial start-up configuration and validation based on the
        cube and the specified coordinates to be interpolated over.

        """
        # Pre-calculate control data for each interpolation coordinate.
        self._src_points = []
        self._coord_decreasing = []
        self._circulars = []
        self._interp_dims = []
        for index, coord in enumerate(self._src_coords):
            coord_dims = self._src_cube.coord_dims(coord)
            coord_points = coord.points

            # Record if coord is descending-order, and adjust points.
            # (notes copied from pelson :-
            #    Force all coordinates to be monotonically increasing.
            #    Generally this isn't always necessary for a rectilinear
            #    interpolator, but it is a common requirement.)
            decreasing = (coord.ndim == 1 and
                          # NOTE: this clause avoids an error when > 1D,
                          # as '_validate' raises a more readable error.
                          coord_points.size > 1 and
                          coord_points[1] < coord_points[0])
            self._coord_decreasing.append(decreasing)
            if decreasing:
                coord_points = coord_points[::-1]

            # Record info if coord is circular, and adjust points.
            circular = getattr(coord, 'circular', False)
            modulus = getattr(coord.units, 'modulus', 0)
            if circular or modulus:
                # Only DimCoords can be circular.
                if circular:
                    coord_points = _extend_circular_coord(coord, coord_points)
                offset = ((coord_points.max() + coord_points.min() - modulus)
                          * 0.5)
                self._circulars.append((circular, modulus,
                                        index, coord_dims[0],
                                        offset))

            self._src_points.append(coord_points)

            # Record any interpolation cube dims we haven't already seen.
            coord_dims = [c for c in coord_dims
                          if c not in self._interp_dims]
            self._interp_dims += coord_dims

        self._validate()

    def _validate(self):
        """
        Perform all sanity checks to ensure that the interpolation request
        over the cube with the specified coordinates is valid and can be
        performed.

        """
        if len(set(self._interp_dims)) != len(self._src_coords):
            raise ValueError('Coordinates repeat a data dimension - the '
                             'interpolation would be over-specified.')

        for coord in self._src_coords:
            if coord.ndim != 1:
                raise ValueError('Interpolation coords must be 1-d for '
                                 'rectilinear interpolation.')

            if not isinstance(coord, DimCoord):
                # Check monotonic.
                if not iris.util.monotonic(coord.points, strict=True):
                    msg = 'Cannot interpolate over the non-' \
                        'monotonic coordinate {}.'
                    raise ValueError(msg.format(coord.name()))

    def _interpolated_dtype(self, dtype):
        """
        Determine the minimum base dtype required by the
        underlying interpolator.

        """
        return np.result_type(_DEFAULT_DTYPE, dtype)

    def _points(self, sample_points, data, data_dims=None):
        """
        Interpolate the given data values at the specified list of orthogonal
        (coord, points) pairs.

        Args:

        * sample_points:
            A list of N iterables, where N is the number of coordinates
            passed to the constructor.
            [sample_values_for_coord_0, sample_values_for_coord_1, ...]
        * data:
            The data to interpolate - not necessarily the data from the cube
            that was used to construct this interpolator. If the data has
            fewer dimensions, then data_dims must be defined.

        Kwargs:

        * data_dims:
            The dimensions of the given data array in terms of the original
            cube passed through to this interpolator's constructor. If None,
            the data dimensions must map one-to-one onto the increasing
            dimension order of the cube.

        Returns:
            An :class:`~numpy.ndarray` or :class:`~numpy.ma.MaskedArray`
            instance of the interpolated data.

        """
        dims = range(self._src_cube.ndim)
        data_dims = data_dims or dims

        if len(data_dims) != data.ndim:
            msg = 'Data being interpolated is not consistent with ' \
                'the data passed through.'
            raise ValueError(msg)

        if sorted(data_dims) != list(data_dims):
            # To do this, a pre & post transpose will be necessary.
            msg = 'Currently only increasing data_dims is supported.'
            raise NotImplementedError(msg)

        # Broadcast the data into the shape of the original cube.
        if data_dims != range(self._src_cube.ndim):
            strides = list(data.strides)
            for dim in range(self._src_cube.ndim):
                if dim not in data_dims:
                    strides.insert(dim, 0)
            data = as_strided(data, strides=strides,
                              shape=self._src_cube.shape)

        data = self._account_for_inverted(data)
        # Calculate the transpose order to shuffle the interpolated dimensions
        # to the lower dimensions for the interpolation algorithm. Then the
        # transpose order to restore the dimensions to their original
        # positions.
        di = self._interp_dims
        ds = sorted(dims, key=lambda d: d not in di)
        dmap = {d: di.index(d) if d in di else ds.index(d) for d in dims}
        interp_order, _ = zip(*sorted(dmap.items(),
                                      key=operator.itemgetter(1)))
        _, src_order = zip(*sorted(dmap.items(), key=operator.itemgetter(0)))

        # Prepare the sample points for interpolation and calculate the
        # shape of the interpolated result.
        interp_points = []
        interp_shape = []
        for index, points in enumerate(sample_points):
            dtype = self._interpolated_dtype(self._src_points[index].dtype)
            points = np.array(points, dtype=dtype, ndmin=1)
            interp_points.append(points)
            interp_shape.append(points.size)

        interp_shape.extend(length for dim, length in enumerate(data.shape) if
                            dim not in di)

        # Convert the interpolation points into a cross-product array
        # with shape (n_cross_points, n_dims)
        interp_points = np.asarray([pts for pts in product(*interp_points)])

        # Adjust for circularity.
        interp_points, data = self._account_for_circular(interp_points, data)

        if interp_order != dims:
            # Transpose data in preparation for interpolation.
            data = np.transpose(data, interp_order)

        # Interpolate and reshape the data ...
        result = self._interpolate(data, interp_points)
        result = result.reshape(interp_shape)

        if src_order != dims:
            # Restore the interpolated result to the original
            # source cube dimensional order.
            result = np.transpose(result, src_order)

        return result

    def __call__(self, sample_points, collapse_scalar=True):
        """
        Construct a cube from the specified orthogonal interpolation points.

        Args:

        * sample_points:
            A list of N iterables, where N is the number of coordinates
            passed to the constructor.
            [sample_values_for_coord_0, sample_values_for_coord_1, ...]

        Kwargs:

        * collapse_scalar:
            Whether to collapse the dimension of the scalar sample points
            in the resulting cube. Default is True.

        Returns:
            A cube interpolated at the given sample points. The dimensionality
            of the cube will be the number of original cube dimensions minus
            the number of scalar coordinates, if collapse_scalar is True.

        """
        if len(sample_points) != len(self._src_coords):
            msg = 'Expected sample points for {} coordinates, got {}.'
            raise ValueError(msg.format(len(self._src_coords),
                                        len(sample_points)))

        sample_points = _canonical_sample_points(self._src_coords,
                                                 sample_points)

        data = self._src_cube.data
        # Interpolate the cube payload.
        interpolated_data = self._points(sample_points, data)

        if collapse_scalar:
            # When collapse_scalar is True, keep track of the dimensions for
            # which sample points is scalar : We will remove these dimensions
            # later on.
            _new_scalar_dims = []
            for dim, points in zip(self._interp_dims, sample_points):
                if np.array(points).ndim == 0:
                    _new_scalar_dims.append(dim)

        cube = self._src_cube
        new_cube = iris.cube.Cube(interpolated_data)
        new_cube.metadata = cube.metadata

        def construct_new_coord_given_points(coord, points):
            # Handle what was previously a DimCoord which may no longer be
            # monotonic.
            try:
                return coord.copy(points)
            except ValueError:
                return AuxCoord.from_coord(coord).copy(points)

        # Keep track of id(coord) -> new_coord for aux factory construction
        # later on.
        coord_mapping = {}
        dims_with_dim_coords = []

        def construct_new_coord(coord):
            dims = cube.coord_dims(coord)
            if coord in self._src_coords:
                index = self._src_coords.index(coord)
                new_points = sample_points[index]
                new_coord = construct_new_coord_given_points(coord, new_points)
                # isinstance not possible here as a dimension coordinate can be
                # mapped to the aux coordinates of a cube.
                if coord in cube.aux_coords:
                    dims = [self._interp_dims[index]]
            else:
                if set(dims).intersection(set(self._interp_dims)):
                    # Interpolate the coordinate payload.
                    new_coord = self._resample_coord(sample_points, coord,
                                                     dims)
                else:
                    new_coord = coord.copy()
            return new_coord, dims

        def gen_new_cube():
            if (isinstance(new_coord, DimCoord) and len(dims) > 0
                    and dims[0] not in dims_with_dim_coords):
                new_cube._add_unique_dim_coord(new_coord, dims)
                dims_with_dim_coords.append(dims[0])
            else:
                new_cube._add_unique_aux_coord(new_coord, dims)
            coord_mapping[id(coord)] = new_coord

        # Copy/interpolate the coordinates.
        for coord in (cube.dim_coords + cube.aux_coords):
            new_coord, dims = construct_new_coord(coord)
            gen_new_cube()

        for factory in self._src_cube.aux_factories:
            new_cube.add_aux_factory(factory.updated(coord_mapping))

        if collapse_scalar and _new_scalar_dims:
            dim_slices = [0 if dim in _new_scalar_dims else slice(None)
                          for dim in range(new_cube.ndim)]
            new_cube = new_cube[tuple(dim_slices)]

        return new_cube


class LinearRegridder(object):
    """
    This class provides support for performing regridding via linear
    interpolation.

    """
    def __init__(self, src_grid_cube, target_grid_cube,
                 extrapolation_mode='linear'):
        """
        Create a linear regridder for conversions between the source
        and target grids.

        Args:

        * src_grid_cube:
            The :class:`~iris.cube.Cube` providing the source grid.
        * target_grid_cube:
            The :class:`~iris.cube.Cube` providing the target grid.

        Kwargs:

        * extrapolation_mode:
            Must be one of the following strings:

              * 'linear' - The extrapolation points will be calculated by
                extending the gradient of closest two points.
              * 'nan' - The extrapolation points will be be set to NaN.
              * 'error' - An exception will be raised, notifying an
                attempt to extrapolate.
              * 'mask' - The extrapolation points will always be masked, even
                if the source data is not a MaskedArray.
              * 'nanmask' - If the source data is a MaskedArray the
                extrapolation points will be masked. Otherwise they will be
                set to NaN.

            Default mode of extrapolation is 'linear'

        """
        # Snapshot the state of the cubes to ensure that the regridder
        # is impervious to external changes to the original source cubes.
        self._src_grid = snapshot_grid(src_grid_cube)
        self._target_grid = snapshot_grid(target_grid_cube)
        # The extrapolation mode.
        if extrapolation_mode not in _LINEAR_EXTRAPOLATION_MODES:
            msg = 'Extrapolation mode {!r} not supported.'
            raise ValueError(msg.format(extrapolation_mode))
        self._extrapolation_mode = extrapolation_mode

        # The need for an actual Cube is an implementation quirk
        # caused by the current usage of the experimental regrid
        # function.
        self._target_grid_cube_cache = None

    @property
    def _target_grid_cube(self):
        if self._target_grid_cube_cache is None:
            x, y = self._target_grid
            data = np.empty((y.points.size, x.points.size))
            cube = iris.cube.Cube(data)
            cube.add_dim_coord(y, 0)
            cube.add_dim_coord(x, 1)
            self._target_grid_cube_cache = cube
        return self._target_grid_cube_cache

    def __call__(self, cube):
        """
        Regrid this :class:`~iris.cube.Cube` on to the target grid of
        this :class:`LinearRegridder`.

        The given cube must be defined with the same grid as the source
        grid used to create this :class:`LinearRegridder`.

        Args:

        * cube:
            A :class:`~iris.cube.Cube` to be regridded.

        Returns:
            A cube defined with the horizontal dimensions of the target
            and the other dimensions from this cube. The data values of
            this cube will be converted to values on the new grid using
            linear interpolation.

        """
        if get_xy_dim_coords(cube) != self._src_grid:
            raise ValueError('The given cube is not defined on the same '
                             'source grid as this regridder.')
        return eregrid.regrid_bilinear_rectilinear_src_and_grid(
            cube, self._target_grid_cube,
            extrapolation_mode=self._extrapolation_mode)
