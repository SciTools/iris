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
from itertools import izip

import numpy as np
from numpy.lib.stride_tricks import as_strided

import iris.cube
from iris.coords import DimCoord, AuxCoord
from iris.experimental.regrid import (_RegularGridInterpolator,
                                      _ndim_coords_from_arrays)
from iris.analysis.interpolate import (_extend_circular_coord_and_data,
                                       _extend_circular_data)


class Interpolator(object):

    """
    Provides a superclass suitable for adaptation and use with many of
    the interpolation methods found in ``scipy.interpolate``.

    """

    def __init__(self, cube, interp_coords, extrapolation_mode=None):
        """
        Args:

         * cube - the cube which represents the locations of the data to be
                  interpolated from.
         * interp_coords - the names or coord instances which are to be
                           interpolated over. Actual interpolation points are
                           provided upon calling specific methods on an
                           Interpolator instance.
         * extrapolation_mode - The extrapolation mode to use with this
                                interpolator. See the specific interpolator
                                class for supported modes.

        .. note::

            The attributes of the Interpolator should be considered read-only.
            Much of the setup of this Interpolator is necessarily done in
            ``__init__``.

        """
        #: The "source" or "data" coordinate of this interpolator. Many of the
        #: interpolation methods will accept data of a compatible shape as
        #: this cube.
        self.cube = cube

        #: A list of the point values for the given interp_coords. The shape
        #: of the coordinate arrays depends on the Interpolator.
        self.coord_points = []

        #: The dimensions over which the interpolation will take place. For
        #: structured interpolation such as NDLinear there will be as many
        #: coord_dims are there are interpolation coordinates, though this
        #: may not hold true for scattered/triangulation based interpolations.
        self.coord_dims = []

        #: The number of dimensions that this interpolator interpolates over.
        self.ndims = None

        #: The interpolation coordinates for this interpolator as given in
        #: the constructor.
        self._interp_coords = []

        #: The interpolation coordinates for this interpolator acquired from
        #: :attr:`self.cube`.
        self.coords = []

        #: The shape of the array passed to this Interpolator once circular
        #: coordinates are accounted for.
        self._data_shape = None

        #: List of tuples representing:
        #:   (interp_dim, coord_dim, min_value, max_value, coord modulus or 0)
        self._circulars = []

        self._define_interpolation_coords(interp_coords)
        self._validate_coordinates()

        #: The simplest shape of data being passed to the (scipy) interpolator
        #: constructor.
        interpolation_shape = [self._data_shape[ind]
                               for ind in self.coord_dims]

        # Create some data which has the simplest possible float dtype.
        dtype = self.interpolated_dtype(cube.data.dtype)
        mock_data = as_strided(np.array(0, dtype=dtype),
                               shape=interpolation_shape,
                               strides=[0] * len(self.coord_dims))
        self._interpolator = self._build_interpolator(self.coord_points,
                                                      mock_data,
                                                      extrapolation_mode)

    def _validate_coordinates(self):
        """
        A hook to validate the coordinates for this interpolator. Types of
        validation may include checking that the coordinates are monotonic,
        1-dimensional etc. - it is expected that any invalid inputs raise
        exceptions at this point.

        The default implementation is to do no validation.

        """

    def _define_interpolation_coords(self, interp_coords):
        """
        Sets up the interpolator's interpolation coordinates.

        Attributes defined by this method:

         * _interp_coords
         * ndims
         * _circulars
         * _data_shape
         * coord_points
         * coord_dims
         * coords

        """
        cube = self.cube
        # Triggers the loading - is that really necessary...?
        data = self.cube.data
        self._interp_coords = interp_coords
        self.ndims = len(interp_coords)
        self.coords = [self.cube.coord(coord) for coord in interp_coords]

        coord_points_list = []

        for interp_dim, coord in enumerate(self.coords):
            coord_dims = cube.coord_dims(coord)

            if getattr(coord, 'circular', False):
                # Only DimCoords can be circular.
                coord_points, data = _extend_circular_coord_and_data(
                    coord, data, coord_dims[0])
                modulus = getattr(coord.units, 'modulus', 0)
                self._circulars.append((interp_dim, coord_dims[0],
                                        coord_points.min(),
                                        coord_points.max(), modulus))
            else:
                coord_points = coord.points
            coord_points_list.append([coord_points, coord_dims])

        self._data_shape = data.shape

        self.coord_points, coord_dims_lists = zip(*coord_points_list)

        for dim_list in coord_dims_lists:
            for dim in dim_list:
                if dim not in self.coord_dims:
                    self.coord_dims.append(dim)

    def _build_interpolator(self, coord_points, mock_data,
                            extrapolation_mode=None):
        """
        Returns the interpolator to be put in ``self._interpolator`` for
        this interpolation scheme. The interpolator may then be used by the
        :meth:`_interpolate_data_at_coord_points` method.

        """
        raise NotImplementedError('Subclass must implement.')

    def _interpolate_data_at_coord_points(self, data, coord_points):
        """
        Do the actual interpolation. Data is an array of
        :data:`.ndim` dimensions.
        """
        raise NotImplementedError('Subclass must implement.')

    def interpolated_dtype(self, dtype):
        """
        Return the dtype that the underlying interpolator is expecting as
        its input data.

        .. note::

            By default the interpolated dtype is a floating point value,
            however some subclasses may want to relax this requirement.

        """
        # Default to the interpolator only being able to return float values,
        # though subclasses may define their own rules (fixed dtype,
        # int support etc.).
        return np.result_type(np.float16, dtype)

    def interpolate_data(self, coord_points, data, data_dims=None):
        """
        Given an array of coordinate points with shape (npts, ndim), return
        an array computed by interpolating the given data.

        Args:

        * coord_points - The coordinate values to interpolate over. The array
                         should have a shape of (npts, ndim).

        * data - The data to interpolate - not necessarily the data that was
                 in the cube which was used to construct this interpolator.
                 If the data has fewer dimensions, data_dims should be
                 defined.

        Kwargs:

        * data_dims - The dimensions of the given data array in terms of the
                      original cube passed through to this Interpolator's
                      constructor. If None the data dimensions must map
                      one-to-one on :data:`.cube`.

        Returns:

        * interpolated_data - The ``data`` array interpolated using the given
                              ``coord_points``. The shape of
                              ``interpolated_data`` will be the shape of the
                              *original cube's* array with :data:`.coord_dims`
                              removed and a trailing dimension of length
                              ``npts``.

        .. note::

            The implementation of this method means that even for small
            subsets of the original cube's data, the data to be interpolated
            will be broadcast into the orginal cube's shape - thus resulting
            in more interpolation calls than are optimally needed. This has
            been done for implementation simplification, but there is no
            fundamental reason this must be the case.

        """
        coord_points = self._interpolate_data_prepare_coord_points(
            coord_points)

        data_dims = data_dims or range(self.cube.ndim)

        if len(data_dims) != data.ndim:
            raise ValueError('data being interpolated is not consistent with '
                             'the data passed through.')

        if sorted(data_dims) != list(data_dims):
            # To do this, a pre & post transpose will be necessary.
            raise NotImplementedError('Currently only increasing data_dims '
                                      'has been implemented.')

        if data_dims != range(self.cube.ndim):
            # Put the given array into the shape of the original cube's array.
            strides = list(data.strides)
            for dim in range(self.cube.ndim):
                if dim not in data_dims:
                    strides.insert(dim, 0)

            data = as_strided(data, strides=strides, shape=self.cube.shape)

        coord_points, data = self._account_for_circular(coord_points, data)

        # Build up a shape suitable for passing to ndindex, inside the loop we
        # will insert slice(None) on the data indices.
        iter_shape = list(length if index not in self.coord_dims
                          else 1
                          for index, length in enumerate(data.shape))

        result_shape = list(length for index, length in enumerate(data.shape)
                            if index not in self.coord_dims)
        # Keep track of the dimension to put the interpolated data into.
        index_dimension = len(result_shape)
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
            interpolant_index = [position
                                 for index, position in enumerate(ndindex)
                                 if index not in self.coord_dims]
            interpolant_index.insert(index_dimension, slice(None))
            index = tuple(position if index not in self.coord_dims
                          else slice(None)
                          for index, position in enumerate(ndindex))
            sub_data = data[index]

            trans, _ = zip(*sorted(enumerate(self.coord_dims),
                                   key=lambda (i, dim): dim))
            sub_data = np.transpose(sub_data, trans).copy()

            r = self._interpolate_data_at_coord_points(sub_data, coord_points)
            result_data[interpolant_index] = r
            if masked:
                r = self._interpolate_data_at_coord_points(sub_data.mask,
                                                           coord_points)
                result_data.mask[interpolant_index] = r > 0
        return result_data

    def _interpolate_data_prepare_coord_points(self, coord_points):
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
            raise ValueError('Perhaps inconsistently shaped arrays were '
                             'passed as coordinate points. The resulting '
                             'numpy array has "object" as its type.')
        return coord_points

    def _account_for_circular(self, coord_points, data):
        """
        Extend the given data array, and re-centralise coordinate
        points for circular (1D) coordinates.

        """
        # Map all the requested values into the range of the source
        # data (centred over the centre of the source data to allow
        # extrapolation where required).
        if self._circulars:
            for _, data_dim, _, _, _ in self._circulars:
                data = _extend_circular_data(data, data_dim)

            for (interp_dim, _, src_min, src_max,
                    src_modulus) in self._circulars:
                offset = (src_max + src_min - src_modulus) * 0.5
                coord_points[:, interp_dim] -= offset
                coord_points[:, interp_dim] = (coord_points[:, interp_dim] %
                                               src_modulus) + offset
        return coord_points, data

    def points(self, sample_points, data, data_dims=None):
        """
        Interpolate the given data values at the given list of
        orthogonal (coord, points) pairs.

        Args

        * sample_points - list of (coord, points) pairs. Order of coordinates
                          needn't be the order of the coordinates passed to
                          this interpolator's constructor.
        * data - The data to interpolate - not necessarily the data that was
                 in the cube which was used to construct this interpolator.
                 If the data has fewer dimensions, data_dims should be
                 defined.

        Kwargs:

        * data_dims - The dimensions of the given data array in terms of the
                      original cube passed through to this Interpolator's
                      constructor. If None the data dimensions must map
                      one-to-one on :data:`.cube`.

        Returns

        * interpolated_data - An array of shape

        """
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
                cube_dim_to_result_dim[dim] = n_non_interp_coords + \
                    self.coord_dims.index(dim)
                interp_coords_seen += 1

        result_to_cube_dim = {v: k for k, v in cube_dim_to_result_dim.items()}

        for orig_coord, points in sample_points:
            if orig_coord not in self._interp_coords:
                raise ValueError('Coord {!r} was not one of those passed to '
                                 'the constructor.'.format(orig_coord))
            coord = self.cube.coord(orig_coord)
            points = np.asanyarray(points)

            order_in_constructor = self._interp_coords.index(orig_coord)

            coord_points[order_in_constructor] = points
            for coord_dim in self.cube.coord_dims(coord):
                interpolated_shape[self.coord_dims[order_in_constructor]] = \
                    points.size

        # Given an expected shape for the final array, compute the shape of
        # the array which puts the interpolated dimensions last.
        new_dimension_order = (lambda (dim, length):
                               cube_dim_to_result_dim[dim])
        _, target_shape = zip(*sorted(enumerate(interpolated_shape),
                                      key=new_dimension_order))

        # Now compute the transpose array which needs to be applied to the
        # previously computed shape to get back to the expected interpolated
        # shape.
        old_dimension_order = lambda (dim, length): result_to_cube_dim[dim]
        transpose_order, _ = zip(*sorted(enumerate(interpolated_shape),
                                         key=old_dimension_order))

        # Turn the coord points into one dimensional cross-products.
        if len(coord_points) > 1:
            coord_points = [arr.flatten()
                            for arr in np.meshgrid(*coord_points[::-1])]

        # Now turn the list of cross-product 1-d arrays into an array of
        # shape (n_interp_points, n_dims).
        coord_points = np.asanyarray(coord_points).reshape(self.ndims,
                                                           -1).T[:, ::-1]

        data = self.interpolate_data(coord_points, data, data_dims=data_dims)
        # Turn the interpolated data back into the order that it was given to
        # us in the first place.
        return np.transpose(data.reshape(target_shape), transpose_order)

    def _orthogonal_points_preserve_dimensionality(self, sample_points, data,
                                                   data_dims=None):
        data = self.points(sample_points, data, data_dims)
        index = tuple(0 if dim not in data_dims else slice(None)
                      for dim in range(self.cube.ndim))
        return data[index]

    def _resample_coord(self, sample_points, coord, coord_dims):
        coord_points = coord.points

        new_points = self._orthogonal_points_preserve_dimensionality(
            sample_points, coord_points, coord_dims)

        # Watch out for DimCoord instances that are no longer monotonic
        # after the resampling.
        try:
            new_coord = coord.copy(new_points)
        except ValueError:
            aux_coord = AuxCoord.from_coord(coord)
            new_coord = aux_coord.copy(new_points)
        return new_coord

    def __call__(self, sample_points, collapse_scalar=True):
        """
        Construct a cube from the specified orthogonal interpolation points.

        Args

        * sample_points - list of (coord, points) pairs. Order of coordinates
                          needn't be the order of the coordinates passed to
                          this interpolator's constructor.

        Kwargs

        * collapse_scalar - whether to collapse the dimension of the scalar
                            sample points in the resulting cube. Default is
                            True.

        Returns:

        * interpolated_cube - a cube interpolated at the given sample points.
                              The dimensionality of the cube will be the
                              number of dimensions in :data:`.cube` minus the
                              number of scalar coordinates if
                              ``collapse_scalar`` is True.

        """
        data = self.cube.data
        interpolated_data = self.points(sample_points, data)
        if interpolated_data.ndim == 0:
            interpolated_data = np.asanyarray(interpolated_data, ndmin=1)

        # Get hold of the original interpolation coordinates in terms of the
        # given cube.
        interp_coords = self.coords
        sample_point_order = [self.cube.coord(coord)
                              for coord, _ in sample_points]

        # Keep track of the dimensions for which sample points is scalar when
        # collapse_scalar is True - we will remove these scalar dimensions
        # later on.
        _new_scalar_dims = []
        if collapse_scalar:
            for coord, points in sample_points:
                coord = self.cube.coord(coord)
                if np.array(points).ndim == 0:
                    new_dim = self.coord_dims[interp_coords.index(coord)]
                    _new_scalar_dims.append(new_dim)

        cube = self.cube
        new_cube = iris.cube.Cube(interpolated_data)
        new_cube.metadata = cube.metadata

        def construct_new_coord_given_points(coord, points):
            # Handle what was previously a DimCoord which may no longer be
            # monotonic.
            try:
                return DimCoord.from_coord(coord).copy(points)
            except ValueError:
                return AuxCoord.from_coord(coord).copy(points)

        # Keep track of id(coord) -> new_coord for aux factory construction
        # later on.
        coord_mapping = {}

        dims_with_dim_coords = []

        # Copy/interpolate the coordinates.
        for coord in cube.dim_coords:
            dim, = cube.coord_dims(coord)
            if coord in interp_coords:
                new_points = sample_points[sample_point_order.index(coord)][1]
                new_coord = construct_new_coord_given_points(coord,
                                                             new_points)
            elif set([dim]).intersection(set(self.coord_dims)):
                new_coord = self._resample_coord(sample_points, coord,
                                                 [dim])
            else:
                new_coord = coord.copy()

            # new_coord may no longer be a dim coord, so check we don't need
            # to add it as an aux coord (thus leaving the dim anonymous).
            if isinstance(new_coord, DimCoord) and dim is not None:
                new_cube._add_unique_dim_coord(new_coord, dim)
                dims_with_dim_coords.append(dim)
            else:
                new_cube._add_unique_aux_coord(new_coord, dim)
            coord_mapping[id(coord)] = new_coord

        for coord in cube.aux_coords:
            dims = cube.coord_dims(coord)
            if coord in interp_coords:
                new_points = sample_points[sample_point_order.index(coord)][1]
                new_coord = construct_new_coord_given_points(coord,
                                                             new_points)
                dims = [self.coord_dims[interp_coords.index(coord)]]
            elif set(dims).intersection(set(self.coord_dims)):
                new_coord = self._resample_coord(sample_points, coord, dims)
            else:
                new_coord = coord.copy()

            new_dims = dims

            if (isinstance(new_coord, DimCoord) and len(new_dims) > 0
                    and new_dims[0] not in dims_with_dim_coords):
                new_cube._add_unique_dim_coord(new_coord, new_dims)
                dims_with_dim_coords.append(new_dims[0])
            else:
                new_cube._add_unique_aux_coord(new_coord, new_dims)
            coord_mapping[id(coord)] = new_coord

        for factory in self.cube.aux_factories:
            new_cube.add_aux_factory(factory.updated(coord_mapping))

        if _new_scalar_dims:
            iris.util.remap_cube_dimensions(new_cube,
                                            remove_axes=_new_scalar_dims)

        return new_cube


class RectilinearInterpolator(Interpolator):

    """
    The baseclass for interpolations which expect 1D monotonically
    increasing coordinate(s).

    """

    def _validate_coordinates(self):
        Interpolator._validate_coordinates(self)
        if len(set(self.coord_dims)) != len(self._interp_coords):
            raise ValueError('Coordinates repeat a data dimension - the '
                             'interpolation would be over-specified.')

        for coord in self.coords:
            if coord.ndim != 1:
                raise ValueError('Interpolation coords must be 1-d for '
                                 'rectilinear interpolation.')

            if not isinstance(coord, DimCoord):
                # check monotonic.
                if not iris.util.monotonic(coord.points, strict=True):
                    msg = 'Cannot interpolate over the non-' \
                        'monotonic coordinate {}.'
                    raise ValueError(msg.format(coord.name()))

        self._reorder_non_increasing_axes()

    def _reorder_non_increasing_axes(self):
        # Force all coordinates to be monotonically increasing. Generally this
        # isn't always necessary for a rectilinear interpolator, but it is a
        # common requirement.

        # We've already checked that each coordinate is strictly monotonic in
        # _validate_coord, so we just need to get the direction from the
        # difference of the first two values (if more than one exists!).
        self.coord_decreasing = [np.all(np.diff(points[:2]) < 0)
                                 for points in self.coord_points]
        if np.any(self.coord_decreasing):
            pairs = izip(self.coord_decreasing, self.coord_points)
            self.coord_points = [points[::-1] if is_decreasing else points
                                 for is_decreasing, points in pairs]


class LinearInterpolator(RectilinearInterpolator):

    def extrapolation_mode(self, mode):
        """
        Set the extrapolation mode for this LinearInterpolator.
        Valid modes are 'linear', 'error', 'nan'.

        """
        self._update_extrapolation_mode(mode)
        self.extrapolation_mode = mode

    def _interpolate_data_at_coord_points(self, data, coord_points):
        self._interpolator.values = data
        return self._interpolator(coord_points)

    def _interpolate_data_prepare_coord_points(self, coord_points):
        cls = RectilinearInterpolator
        coord_points = cls._interpolate_data_prepare_coord_points(self,
                                                                  coord_points)
        if issubclass(coord_points.dtype.type, np.integer):
            coord_points = coord_points.astype(np.float)
        return coord_points

    def _update_extrapolation_mode(self, mode=None):
        if mode is None:
            mode = 'linear'

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
            raise ValueError(
                'Extrapolation mode {!r} not supported.'.format(mode))

    def _build_interpolator(self, coord_points, mock_data,
                            extrapolation_mode=None):
        self._interpolator = _RegularGridInterpolator(coord_points, mock_data,
                                                      fill_value=None,
                                                      bounds_error=False)
        self._update_extrapolation_mode(extrapolation_mode)
        return self._interpolator
