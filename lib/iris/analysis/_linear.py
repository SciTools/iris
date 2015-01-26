# (C) British Crown Copyright 2014 - 2015, Met Office
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

from __future__ import (absolute_import, division, print_function)

import copy
import functools
import warnings

import numpy as np
import numpy.ma as ma

from iris.analysis._interpolation import (EXTRAPOLATION_MODES,
                                          extend_circular_coord_and_data,
                                          get_xy_dim_coords, snapshot_grid)
from iris.analysis._scipy_interpolate import _RegularGridInterpolator
import iris.cube


class RectilinearRegridder(object):
    """
    This class provides support for performing regridding via linear
    interpolation.

    """
    def __init__(self, src_grid_cube, tgt_grid_cube, extrapolation_mode):
        """
        Create a linear regridder for conversions between the source
        and target grids.

        Args:

        * src_grid_cube:
            The :class:`~iris.cube.Cube` providing the source grid.
        * tgt_grid_cube:
            The :class:`~iris.cube.Cube` providing the target grid.
        * extrapolation_mode:
            Must be one of the following strings:

              * 'extrapolate' - The extrapolation points will be
                calculated by extending the gradient of the closest two
                points.
              * 'nan' - The extrapolation points will be be set to NaN.
              * 'error' - An exception will be raised, notifying an
                attempt to extrapolate.
              * 'mask' - The extrapolation points will always be masked, even
                if the source data is not a MaskedArray.
              * 'nanmask' - If the source data is a MaskedArray the
                extrapolation points will be masked. Otherwise they will be
                set to NaN.

        """
        # Validity checks.
        if not isinstance(src_grid_cube, iris.cube.Cube):
            raise TypeError("'src_grid_cube' must be a Cube")
        if not isinstance(tgt_grid_cube, iris.cube.Cube):
            raise TypeError("'tgt_grid_cube' must be a Cube")
        # Snapshot the state of the cubes to ensure that the regridder
        # is impervious to external changes to the original source cubes.
        self._src_grid = snapshot_grid(src_grid_cube)
        self._tgt_grid = snapshot_grid(tgt_grid_cube)
        # Check the target grid units.
        for coord in self._tgt_grid:
            self._check_units(coord)
        # The extrapolation mode.
        if extrapolation_mode not in EXTRAPOLATION_MODES:
            msg = 'Invalid extrapolation mode {!r}'
            raise ValueError(msg.format(extrapolation_mode))
        self._extrapolation_mode = extrapolation_mode

    @staticmethod
    def _sample_grid(src_coord_system, grid_x_coord, grid_y_coord):
        """
        Convert the rectilinear grid coordinates to a curvilinear grid in
        the source coordinate system.

        The `grid_x_coord` and `grid_y_coord` must share a common coordinate
        system.

        Args:

        * src_coord_system:
            The :class:`iris.coord_system.CoordSystem` for the grid of the
            source Cube.
        * grid_x_coord:
            The :class:`iris.coords.DimCoord` for the X coordinate.
        * grid_y_coord:
            The :class:`iris.coords.DimCoord` for the Y coordinate.

        Returns:
            A tuple of the X and Y coordinate values as 2-dimensional
            arrays.

        """
        grid_x, grid_y = np.meshgrid(grid_x_coord.points, grid_y_coord.points)
        # Skip the CRS transform if we can to avoid precision problems.
        if src_coord_system == grid_x_coord.coord_system:
            sample_grid_x = grid_x
            sample_grid_y = grid_y
        else:
            src_crs = src_coord_system.as_cartopy_crs()
            grid_crs = grid_x_coord.coord_system.as_cartopy_crs()
            sample_xyz = src_crs.transform_points(grid_crs, grid_x, grid_y)
            sample_grid_x = sample_xyz[..., 0]
            sample_grid_y = sample_xyz[..., 1]
        return sample_grid_x, sample_grid_y

    @staticmethod
    def _regrid_bilinear_array(src_data, x_dim, y_dim,
                               src_x_coord, src_y_coord,
                               sample_grid_x, sample_grid_y,
                               extrapolation_mode='nanmask'):
        """
        Regrid the given data from the src grid to the sample grid.

        The result will be a MaskedArray if either/both of:
         - the source array is a MaskedArray,
         - the extrapolation_mode is 'mask' and the result requires
           extrapolation.

        If the result is a MaskedArray the mask for each element will be set
        if either/both of:
         - there is a non-zero contribution from masked items in the input data
         - the element requires extrapolation and the extrapolation_mode
           dictates a masked value.

        Args:

        * src_data:
            An N-dimensional NumPy array or MaskedArray.
        * x_dim:
            The X dimension within `src_data`.
        * y_dim:
            The Y dimension within `src_data`.
        * src_x_coord:
            The X :class:`iris.coords.DimCoord`.
        * src_y_coord:
            The Y :class:`iris.coords.DimCoord`.
        * sample_grid_x:
            A 2-dimensional array of sample X values.
        * sample_grid_y:
            A 2-dimensional array of sample Y values.

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

            The default mode of extrapolation is 'nanmask'.

        Returns:
            The regridded data as an N-dimensional NumPy array. The lengths
            of the X and Y dimensions will now match those of the sample
            grid.

        """
        if sample_grid_x.shape != sample_grid_y.shape:
            raise ValueError('Inconsistent sample grid shapes.')
        if sample_grid_x.ndim != 2:
            raise ValueError('Sample grid must be 2-dimensional.')

        # Prepare the result data array
        shape = list(src_data.shape)
        assert shape[x_dim] == src_x_coord.shape[0]
        assert shape[y_dim] == src_y_coord.shape[0]

        shape[y_dim] = sample_grid_x.shape[0]
        shape[x_dim] = sample_grid_x.shape[1]

        # If we're given integer values, convert them to the smallest
        # possible float dtype that can accurately preserve the values.
        dtype = src_data.dtype
        if dtype.kind == 'i':
            dtype = np.promote_types(dtype, np.float16)

        if isinstance(src_data, ma.MaskedArray):
            data = ma.empty(shape, dtype=dtype)
            data.mask = np.zeros(data.shape, dtype=np.bool)
        else:
            data = np.empty(shape, dtype=dtype)

        # The interpolation class requires monotonically increasing
        # coordinates, so flip the coordinate(s) and data if the aren't.
        reverse_x = src_x_coord.points[0] > src_x_coord.points[1]
        reverse_y = src_y_coord.points[0] > src_y_coord.points[1]
        flip_index = [slice(None)] * src_data.ndim
        if reverse_x:
            src_x_coord = src_x_coord[::-1]
            flip_index[x_dim] = slice(None, None, -1)
        if reverse_y:
            src_y_coord = src_y_coord[::-1]
            flip_index[y_dim] = slice(None, None, -1)
        src_data = src_data[tuple(flip_index)]

        if src_x_coord.circular:
            x_points, src_data = extend_circular_coord_and_data(src_x_coord,
                                                                src_data,
                                                                x_dim)
        else:
            x_points = src_x_coord.points

        # Slice out the first full 2D piece of data for construction of the
        # interpolator.
        index = [0] * src_data.ndim
        index[x_dim] = index[y_dim] = slice(None)
        initial_data = src_data[tuple(index)]
        if y_dim < x_dim:
            initial_data = initial_data.T

        # Construct the interpolator, we will fill in any values out of bounds
        # manually.
        interpolator = _RegularGridInterpolator([x_points,
                                                 src_y_coord.points],
                                                initial_data, fill_value=None,
                                                bounds_error=False)
        # The constructor of the _RegularGridInterpolator class does
        # some unnecessary checks on these values, so we set them
        # afterwards instead. Sneaky. ;-)
        try:
            mode = EXTRAPOLATION_MODES[extrapolation_mode]
        except KeyError:
            raise ValueError('Invalid extrapolation mode.')
        interpolator.bounds_error = mode.bounds_error
        interpolator.fill_value = mode.fill_value

        # Construct the target coordinate points array, suitable for passing to
        # the interpolator multiple times.
        interp_coords = [sample_grid_x.astype(np.float64)[..., np.newaxis],
                         sample_grid_y.astype(np.float64)[..., np.newaxis]]

        # Map all the requested values into the range of the source
        # data (centred over the centre of the source data to allow
        # extrapolation where required).
        min_x, max_x = x_points.min(), x_points.max()
        min_y, max_y = src_y_coord.points.min(), src_y_coord.points.max()
        if src_x_coord.units.modulus:
            modulus = src_x_coord.units.modulus
            offset = (max_x + min_x - modulus) * 0.5
            interp_coords[0] -= offset
            interp_coords[0] = (interp_coords[0] % modulus) + offset

        interp_coords = np.dstack(interp_coords)

        def interpolate(data):
            # Update the interpolator for this data slice.
            data = data.astype(interpolator.values.dtype)
            if y_dim < x_dim:
                data = data.T
            interpolator.values = data
            data = interpolator(interp_coords)
            if y_dim > x_dim:
                data = data.T
            return data

        # Build up a shape suitable for passing to ndindex, inside the loop we
        # will insert slice(None) on the data indices.
        iter_shape = list(shape)
        iter_shape[x_dim] = iter_shape[y_dim] = 1

        # Iterate through each 2d slice of the data, updating the interpolator
        # with the new data as we go.
        for index in np.ndindex(tuple(iter_shape)):
            index = list(index)
            index[x_dim] = index[y_dim] = slice(None)

            src_subset = src_data[tuple(index)]
            interpolator.fill_value = mode.fill_value
            data[tuple(index)] = interpolate(src_subset)

            if isinstance(data, ma.MaskedArray) or mode.force_mask:
                # NB. np.ma.getmaskarray returns an array of `False` if
                # `src_subset` is not a masked array.
                src_mask = np.ma.getmaskarray(src_subset)
                interpolator.fill_value = mode.mask_fill_value
                mask_fraction = interpolate(src_mask)
                new_mask = (mask_fraction > 0)

                if np.ma.isMaskedArray(data):
                    data.mask[tuple(index)] = new_mask
                elif np.any(new_mask):
                    # Set mask=False to ensure we have an expanded mask array.
                    data = np.ma.MaskedArray(data, mask=False)
                    data.mask[tuple(index)] = new_mask

        return data

    @staticmethod
    def _create_cube(data, src, x_dim, y_dim, src_x_coord, src_y_coord,
                     grid_x_coord, grid_y_coord, sample_grid_x, sample_grid_y,
                     regrid_callback):
        """
        Return a new Cube for the result of regridding the source Cube onto
        the new grid.

        All the metadata and coordinates of the result Cube are copied from
        the source Cube, with two exceptions:
            - Grid dimension coordinates are copied from the grid Cube.
            - Auxiliary coordinates which span the grid dimensions are
              ignored, except where they provide a reference surface for an
              :class:`iris.aux_factory.AuxCoordFactory`.

        Args:

        * data:
            The regridded data as an N-dimensional NumPy array.
        * src:
            The source Cube.
        * x_dim:
            The X dimension within the source Cube.
        * y_dim:
            The Y dimension within the source Cube.
        * src_x_coord:
            The X :class:`iris.coords.DimCoord`.
        * src_y_coord:
            The Y :class:`iris.coords.DimCoord`.
        * grid_x_coord:
            The :class:`iris.coords.DimCoord` for the new grid's X
            coordinate.
        * grid_y_coord:
            The :class:`iris.coords.DimCoord` for the new grid's Y
            coordinate.
        * sample_grid_x:
            A 2-dimensional array of sample X values.
        * sample_grid_y:
            A 2-dimensional array of sample Y values.
        * regrid_callback:
            The routine that will be used to calculate the interpolated
            values of any reference surfaces.

        Returns:
            The new, regridded Cube.

        """
        # Create a result cube with the appropriate metadata
        result = iris.cube.Cube(data)
        result.metadata = copy.deepcopy(src.metadata)

        # Copy across all the coordinates which don't span the grid.
        # Record a mapping from old coordinate IDs to new coordinates,
        # for subsequent use in creating updated aux_factories.
        coord_mapping = {}

        def copy_coords(src_coords, add_method):
            for coord in src_coords:
                dims = src.coord_dims(coord)
                if coord is src_x_coord:
                    coord = grid_x_coord
                elif coord is src_y_coord:
                    coord = grid_y_coord
                elif x_dim in dims or y_dim in dims:
                    continue
                result_coord = coord.copy()
                add_method(result_coord, dims)
                coord_mapping[id(coord)] = result_coord

        copy_coords(src.dim_coords, result.add_dim_coord)
        copy_coords(src.aux_coords, result.add_aux_coord)

        def regrid_reference_surface(src_surface_coord, surface_dims,
                                     x_dim, y_dim, src_x_coord, src_y_coord,
                                     sample_grid_x, sample_grid_y,
                                     regrid_callback):
            # Determine which of the reference surface's dimensions span the X
            # and Y dimensions of the source cube.
            surface_x_dim = surface_dims.index(x_dim)
            surface_y_dim = surface_dims.index(y_dim)
            surface = regrid_callback(src_surface_coord.points,
                                      surface_x_dim, surface_y_dim,
                                      src_x_coord, src_y_coord,
                                      sample_grid_x, sample_grid_y)
            surface_coord = src_surface_coord.copy(surface)
            return surface_coord

        # Copy across any AuxFactory instances, and regrid their reference
        # surfaces where required.
        for factory in src.aux_factories:
            for coord in factory.dependencies.itervalues():
                if coord is None:
                    continue
                dims = src.coord_dims(coord)
                if x_dim in dims and y_dim in dims:
                    result_coord = regrid_reference_surface(coord, dims,
                                                            x_dim, y_dim,
                                                            src_x_coord,
                                                            src_y_coord,
                                                            sample_grid_x,
                                                            sample_grid_y,
                                                            regrid_callback)
                    result.add_aux_coord(result_coord, dims)
                    coord_mapping[id(coord)] = result_coord
            try:
                result.add_aux_factory(factory.updated(coord_mapping))
            except KeyError:
                msg = 'Cannot update aux_factory {!r} because of dropped' \
                      ' coordinates.'.format(factory.name())
                warnings.warn(msg)
        return result

    def _check_units(self, coord):
        if coord.coord_system is None:
            # No restriction on units.
            pass
        elif isinstance(coord.coord_system,
                        (iris.coord_systems.GeogCS,
                         iris.coord_systems.RotatedGeogCS)):
            # Units for lat-lon or rotated pole must be 'degrees'. Note
            # that 'degrees_east' etc. are equal to 'degrees'.
            if coord.units != 'degrees':
                msg = "Unsupported units for coordinate system. " \
                      "Expected 'degrees' got {!r}.".format(coord.units)
                raise ValueError(msg)
        else:
            # Units for other coord systems must be equal to metres.
            if coord.units != 'm':
                msg = "Unsupported units for coordinate system. " \
                      "Expected 'metres' got {!r}.".format(coord.units)
                raise ValueError(msg)

    def __call__(self, src):
        """
        Regrid this :class:`~iris.cube.Cube` on to the target grid of
        this :class:`RectilinearRegridder`.

        The given cube must be defined with the same grid as the source
        grid used to create this :class:`RectilinearRegridder`.

        Args:

        * src:
            A :class:`~iris.cube.Cube` to be regridded.

        Returns:
            A cube defined with the horizontal dimensions of the target
            and the other dimensions from this cube. The data values of
            this cube will be converted to values on the new grid using
            linear interpolation.

        """
        # Validity checks.
        if not isinstance(src, iris.cube.Cube):
            raise TypeError("'src' must be a Cube")
        if get_xy_dim_coords(src) != self._src_grid:
            raise ValueError('The given cube is not defined on the same '
                             'source grid as this regridder.')

        src_x_coord, src_y_coord = get_xy_dim_coords(src)
        grid_x_coord, grid_y_coord = self._tgt_grid
        src_cs = src_x_coord.coord_system
        grid_cs = grid_x_coord.coord_system

        if src_cs is None and grid_cs is None:
            if not (src_x_coord.is_compatible(grid_x_coord) and
                    src_y_coord.is_compatible(grid_y_coord)):
                raise ValueError("The rectilinear grid coordinates of the "
                                 "given cube and target grid have no "
                                 "coordinate system but they do not have "
                                 "matching coordinate metadata.")
        elif src_cs is None or grid_cs is None:
            raise ValueError("The rectilinear grid coordinates of the given "
                             "cube and target grid must either both have "
                             "coordinate systems or both have no coordinate "
                             "system but with matching coordinate metadata.")

        # Check the source grid units.
        for coord in (src_x_coord, src_y_coord):
            self._check_units(coord)

        # Convert the grid to a 2D sample grid in the src CRS.
        sample_grid = self._sample_grid(src_cs, grid_x_coord, grid_y_coord)
        sample_grid_x, sample_grid_y = sample_grid

        # Compute the interpolated data values.
        x_dim = src.coord_dims(src_x_coord)[0]
        y_dim = src.coord_dims(src_y_coord)[0]
        data = self._regrid_bilinear_array(src.data, x_dim, y_dim,
                                           src_x_coord, src_y_coord,
                                           sample_grid_x, sample_grid_y,
                                           self._extrapolation_mode)

        # Wrap up the data as a Cube.
        regrid_callback = functools.partial(self._regrid_bilinear_array,
                                            extrapolation_mode='nan')
        result = self._create_cube(data, src, x_dim, y_dim,
                                   src_x_coord, src_y_coord,
                                   grid_x_coord, grid_y_coord,
                                   sample_grid_x, sample_grid_y,
                                   regrid_callback)
        return result
