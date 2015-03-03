# (C) British Crown Copyright 2013 - 2015, Met Office
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

# Historically this was auto-generated from
# SciTools/iris-code-generators:tools/gen_rules.py

import numpy as np

from iris.aux_factory import HybridHeightFactory, HybridPressureFactory
from iris.coords import AuxCoord, CellMethod, DimCoord
from iris.fileformats.rules import (ConversionMetadata, Factory, Reference,
                                    ReferenceTarget)
from iris.fileformats.um_cf_map import (LBFC_TO_CF, STASH_TO_CF,
                                        _STASHCODE_IMPLIED_HEIGHTS)
from iris.unit import Unit
import iris.fileformats.pp
import iris.unit


###############################################################################
#
# Convert vectorisation routines.
#

def _dim_or_aux(*args, **kwargs):
    try:
        result = DimCoord(*args, **kwargs)
    except ValueError:
        attr = kwargs.get('attributes')
        if attr is not None and 'positive' in attr:
            del attr['positive']
        result = AuxCoord(*args, **kwargs)
    return result


def _convert_vertical_coords(lbcode, lbvc, blev, lblev, stash,
                             bhlev, bhrlev, brsvd1, brsvd2, brlev,
                             dim=None):
    """
    Encode scalar or vector vertical level values from PP headers as CM data
    components.

    Args:

    * lbcode:
        Scalar field :class:`iris.fileformats.pp.SplittableInt` value.

    * lbvc:
        Scalar field value.

    * blev:
        Scalar field value or :class:`numpy.ndarray` vector of field values.

    * lblev:
        Scalar field value or :class:`numpy.ndarray` vector of field values.

    * stash:
        Scalar field :class:`iris.fileformats.pp.STASH` value.

    * bhlev:
        Scalar field value or :class:`numpy.ndarray` vector of field values.

    * bhrlev:
        Scalar field value or :class:`numpy.ndarray` vector of field values.

    * brsvd1:
        Scalar field value or :class:`numpy.ndarray` vector of field values.

    * brsvd2:
        Scalar field value or :class:`numpy.ndarray` vector of field values.

    * brlev:
        Scalar field value or :class:`numpy.ndarray` vector of field values.

    Kwargs:

    * dim:
        Associated dimension of the vertical coordinate. Defaults to None.

    Returns:
        A tuple containing a list of coords_and_dims, and a list of factories.

    """
    factories = []
    coords_and_dims = []

    # See Word no. 33 (LBLEV) in section 4 of UM Model Docs (F3).
    BASE_RHO_LEVEL_LBLEV = 9999
    model_level_number  = np.atleast_1d(lblev)
    model_level_number[model_level_number == BASE_RHO_LEVEL_LBLEV] = 0

    # Ensure to vectorise these arguments as arrays, as they participate
    # in the conditions of convert rules.
    blev = np.atleast_1d(blev)
    brsvd1 = np.atleast_1d(brsvd1)
    brlev = np.atleast_1d(brlev)

    # Height.
    if (lbvc == 1) and \
            (not (str(stash) in _STASHCODE_IMPLIED_HEIGHTS)) and \
            (np.all(blev != -1)):
        coord = _dim_or_aux(blev, standard_name='height', units='m',
                            attributes={'positive': 'up'})
        coords_and_dims.append((coord, dim))

    if str(stash) in _STASHCODE_IMPLIED_HEIGHTS:
        height = _STASHCODE_IMPLIED_HEIGHTS[str(stash)]
        coord = DimCoord(height, standard_name='height', units='m',
                         attributes={'positive': 'up'})
        coords_and_dims.append((coord, None))

    # Model level number.
    if (len(lbcode) != 5) and \
            (lbvc == 2):
        coord = _dim_or_aux(model_level_number, standard_name='model_level_number',
                            attributes={'positive': 'down'})
        coords_and_dims.append((coord, dim))

    # Depth - unbound.
    if (len(lbcode) != 5) and \
            (lbvc == 2) and \
            np.all(brsvd1 == brlev):
        coord = _dim_or_aux(blev, standard_name='depth', units='m',
                            attributes={'positive': 'down'})
        coords_and_dims.append((coord, dim))

    # Depth - bound.
    if (len(lbcode) != 5) and \
            (lbvc == 2) and \
            np.all(brsvd1 != brlev):
        coord = _dim_or_aux(blev, standard_name='depth', units='m',
                            bounds=np.vstack((brsvd1, brlev)).T,
                            attributes={'positive': 'down'})
        coords_and_dims.append((coord, dim))

    # Depth - unbound and bound (mixed).
    if (len(lbcode) != 5) and \
            (lbvc == 2) and \
            (np.any(brsvd1 == brlev) and np.any(brsvd1 != brlev)):
        lower = np.where(brsvd1 == brlev, blev, brsvd1)
        upper = np.where(brsvd1 == brlev, blev, brlev)
        coord = _dim_or_aux(blev, standard_name='depth', units='m',
                            bounds=np.vstack((lower, upper)).T,
                            attributes={'positive': 'down'})
        coords_and_dims.append((coord, dim))

    # Soil level.
    if len(lbcode) != 5 and \
            lbvc == 6:
        coord = _dim_or_aux(model_level_number, long_name='soil_model_level_number',
                            attributes={'positive': 'down'})
        coords_and_dims.append((coord, dim))

    # Pressure.
    if (lbvc == 8) and \
            (len(lbcode) != 5 or (len(lbcode) == 5 and 1 not in [lbcode.ix, lbcode.iy])):
        coord = _dim_or_aux(blev, long_name='pressure', units='hPa')
        coords_and_dims.append((coord, dim))

    # Air potential temperature.
    if (len(lbcode) != 5) and \
            (lbvc == 19):
        coord = _dim_or_aux(blev, standard_name='air_potential_temperature', units='K',
                            attributes={'positive': 'up'})
        coords_and_dims.append((coord, dim))

    # Hybrid pressure levels.
    if lbvc == 9:
        model_level_number = _dim_or_aux(model_level_number,
                                         standard_name='model_level_number',
                                         attributes={'positive': 'up'})
        level_pressure = _dim_or_aux(bhlev,
                                     long_name='level_pressure',
                                     units='Pa',
                                     bounds=np.vstack((bhrlev, brsvd2)).T)
        sigma = AuxCoord(blev,
                         long_name='sigma',
                         bounds=np.vstack((brlev, brsvd1)).T)
        coords_and_dims.extend([(model_level_number, dim),
                                (level_pressure, dim),
                                (sigma, dim)])
        factories.append(Factory(HybridPressureFactory,
                                 [{'long_name': 'level_pressure'},
                                  {'long_name': 'sigma'},
                                  Reference('surface_air_pressure')]))

    # Hybrid height levels.
    if lbvc == 65:
        model_level_number = _dim_or_aux(model_level_number,
                                         standard_name='model_level_number',
                                         attributes={'positive': 'up'})
        level_height = _dim_or_aux(blev,
                                   long_name='level_height',
                                   units='m',
                                   bounds=np.vstack((brlev, brsvd1)).T,
                                   attributes={'positive': 'up'})
        sigma = AuxCoord(bhlev,
                         long_name='sigma',
                         bounds=np.vstack((bhrlev, brsvd2)).T)
        coords_and_dims.extend([(model_level_number, dim),
                                (level_height, dim),
                                (sigma, dim)])
        factories.append(Factory(HybridHeightFactory,
                                 [{'long_name': 'level_height'},
                                  {'long_name': 'sigma'},
                                  Reference('orography')]))

    return coords_and_dims, factories


def _reshape_vector_args(values_and_dims):
    """
    Reshape a group of (array, dimensions-mapping) onto all dimensions.

    The resulting arrays are all mapped over the same dimensions; as many as
    the maximum dimension number found in the inputs.  Those dimensions not
    mapped by a given input appear as length-1 dimensions in the output array.
    The resulting arrays are thus all mutually compatible in arithmetic -- i.e.
    can combine without broadcasting errors (provided that all inputs mapping
    to a dimension define the same associated length).

    Args:

    * values_and_dims (iterable of (array-like, iterable of int)):
        Input arrays with associated mapping dimension numbers.
        The length of each 'dims' must match the ndims of the 'value'.

    Returns:

    * reshaped_arrays (iterable of arrays).
        The inputs, transposed and reshaped onto common target dimensions.

    """
    # Find maximum dimension index, which sets ndim of results.
    max_dims = [max(dims) if dims else -1 for _, dims in values_and_dims]
    max_dim = max(max_dims) if max_dims else -1
    result = []
    for value, dims in values_and_dims:
        value = np.asarray(value)
        if len(dims) != value.ndim:
            raise ValueError('Lengths of dimension-mappings must match '
                             'input array dimensions.')
        # Save dim sizes in original order.
        original_shape = value.shape
        if dims:
            # Transpose values to put its dims in the target order.
            dims_order = sorted(range(len(dims)),
                                key=lambda i_dim: dims[i_dim])
            value = value.transpose(dims_order)
        if max_dim != -1:
            # Reshape to add any extra *1 dims.
            shape = [1] * (max_dim + 1)
            for i_dim, dim in enumerate(dims):
                shape[dim] = original_shape[i_dim]
            value = value.reshape(shape)
        result.append(value)
    return result


def _collapse_degenerate_points_and_bounds(points, bounds=None, rtol=1.0e-7):
    """
    Collapse points (and optionally bounds) in any dimensions over which all
    values are the same.

    All dimensions are tested, and if degenerate are reduced to length 1.

    Value equivalence is controlled by a tolerance, to avoid problems with
    numbers from netcdftime.date2num, which has limited precision because of
    the way it calculates with floats of days.

    Args:

    * points (:class:`numpy.ndarray`)):
        Array of points values.

    Kwargs:

    * bounds (:class:`numpy.ndarray`)
        Array of bounds values. This array should have an additional vertex
        dimension (typically of length 2) when compared to the  points array
        i.e. bounds.shape = points.shape + (nvertex,)

    Returns:

        A (points, bounds) tuple.

    """
    array = points
    if bounds is not None:
        array = np.vstack((points, bounds.T)).T

    for i_dim in range(points.ndim):
        if array.shape[i_dim] > 1:
            slice_inds = [slice(None)] * points.ndim
            slice_inds[i_dim] = slice(0, 1)
            slice_0 = array[slice_inds]
            if np.allclose(array, slice_0, rtol):
                array = slice_0

    points = array
    if bounds is not None:
        points = array[..., 0]
        bounds = array[..., 1:]

    return points, bounds


def _reduce_points_and_bounds(points, lower_and_upper_bounds=None):
    """
    Reduce the dimensionality of arrays of coordinate points (and optionally
    bounds).

    Dimensions over which all values are the same are reduced to size 1, using
    :func:`_collapse_degenerate_points_and_bounds`.
    All size-1 dimensions are then removed.
    If the bounds arrays are also passed in, then all three arrays must have
    the same shape or be capable of being broadcast to match.

    Args:

    * points (array-like):
        Coordinate point values.

    * lower_and_upper_bounds (pair of array-like, or None):
        Corresponding bounds values (lower, upper), if any.

    Returns:
        dims (iterable of ints), points(array), bounds(array)

        * 'dims' is the mapping from the result array dimensions to the
            original dimensions.  However, when 'array' is scalar, 'dims' will
            be None (rather than an empty tuple).
        * 'points' and 'bounds' are the reduced arrays.
            If no bounds were passed, None is returned.

    """
    orig_points_dtype = np.asarray(points).dtype
    bounds = None
    if lower_and_upper_bounds is not None:
        lower_bounds, upper_bounds = np.broadcast_arrays(
            *lower_and_upper_bounds)
        orig_bounds_dtype = lower_bounds.dtype
        bounds = np.vstack((lower_bounds, upper_bounds)).T

    # Attempt to broadcast points to match bounds to handle scalars.
    if bounds is not None and points.shape != bounds.shape[:-1]:
        points, _ = np.broadcast_arrays(points, bounds[..., 0])

    points, bounds = _collapse_degenerate_points_and_bounds(points, bounds)

    used_dims = tuple(i_dim for i_dim in range(points.ndim)
                      if points.shape[i_dim] > 1)
    reshape_inds = tuple([points.shape[dim] for dim in used_dims])
    points = points.reshape(reshape_inds)
    points = points.astype(orig_points_dtype)
    if bounds is not None:
        bounds = bounds.reshape(reshape_inds + (2,))
        bounds = bounds.astype(orig_bounds_dtype)

    if not used_dims:
        used_dims = None

    return used_dims, points, bounds


def _convert_time_coords(lbcode, lbtim, epoch_hours_unit,
                         t1, t2, lbft,
                         t1_dims=(), t2_dims=(), lbft_dims=()):
    """
    Make time coordinates from the time metadata.

    Args:

    * lbcode(:class:`iris.fileformats.pp.SplittableInt`):
        Scalar field value.
    * lbtim (:class:`iris.fileformats.pp.SplittableInt`):
        Scalar field value.
    * epoch_hours_unit (:class:`iris.units.Unit`):
        Epoch time reference unit.
    * t1 (array-like or scalar):
        Scalar field value or an array of values.
    * t2 (array-like or scalar):
        Scalar field value or an array of values.
    * lbft (array-like or scalar):
        Scalar field value or an array of values.

    Kwargs:

    * t1_dims, t2_dims, lbft_dims (tuples of int):
        Cube dimension mappings for the array metadata. Each default to
        to (). The length of each dims tuple should equal the dimensionality
        of the corresponding array of values.

    Returns:

        A list of (coordinate, dims) tuples. The coordinates are instance of
        :class:`iris.coords.DimCoord` if possible, otherwise they are instance
        of :class:`iris.coords.AuxCoord`. When the coordinate is of length one,
        the `dims` value is None rather than an empty tuple.

    """
    # Reform the input values so they have all the same number of dimensions,
    # transposing where necessary (based on the dimension mappings) so the
    # dimensions are common across each array.
    # Note that this does not guarantee that the arrays are broadcastable,
    # but subsequent arithmetic makes this assumption.
    t1, t2, lbft = _reshape_vector_args([(t1, t1_dims), (t2, t2_dims),
                                         (lbft, lbft_dims)])

    def date2hours(times_array):
        """Convert datetime values to hours-since-epoch values."""
        # Use a minimum-1D form, as netcdftime.date2num doesn't like scalars.
        times_array = np.asarray(times_array)
        return epoch_hours_unit.date2num(
            np.atleast_1d(times_array)).reshape(times_array.shape)

    t1_epoch_hours = date2hours(t1)
    t2_epoch_hours = date2hours(t2)
    hours_from_t1_to_t2 = t2_epoch_hours - t1_epoch_hours
    hours_from_t2_to_t1 = t1_epoch_hours - t2_epoch_hours
    coords_and_dims = []

    if ((lbtim.ia == 0) and
        (lbtim.ib == 0) and
        (lbtim.ic in [1, 2, 3, 4]) and
        (len(lbcode) != 5 or (len(lbcode) == 5 and
                              lbcode.ix not in [20, 21, 22, 23] and
                              lbcode.iy not in [20, 21, 22, 23]))):
        dims, points, _ = _reduce_points_and_bounds(t1_epoch_hours)
        coords_and_dims.append(
            (_dim_or_aux(points, standard_name='time', units=epoch_hours_unit),
             dims))

    if ((lbtim.ia == 0) and
        (lbtim.ib == 1) and
        (lbtim.ic in [1, 2, 3, 4]) and
        (len(lbcode) != 5 or (len(lbcode) == 5
                              and lbcode.ix not in [20, 21, 22, 23]
                              and lbcode.iy not in [20, 21, 22, 23]))):
        dims, points, _ = _reduce_points_and_bounds(hours_from_t2_to_t1)
        coords_and_dims.append(
            (_dim_or_aux(points, standard_name='forecast_period',
                         units='hours'),
             dims))
        dims, points, _ = _reduce_points_and_bounds(t1_epoch_hours)
        coords_and_dims.append(
            (_dim_or_aux(points, standard_name='time', units=epoch_hours_unit),
             dims))
        dims, points, _ = _reduce_points_and_bounds(t2_epoch_hours)
        coords_and_dims.append(
            (_dim_or_aux(points, standard_name='forecast_reference_time',
                         units=epoch_hours_unit),
             dims))

    if ((lbtim.ib == 2) and
        (lbtim.ic in [1, 2, 4]) and
        ((len(lbcode) != 5) or (len(lbcode) == 5 and
                                lbcode.ix not in [20, 21, 22, 23]
                                and lbcode.iy not in [20, 21, 22, 23]))):
        dims, points, bounds = _reduce_points_and_bounds(
            lbft - 0.5 * hours_from_t1_to_t2,
            (lbft - hours_from_t1_to_t2, lbft))
        coords_and_dims.append(
            (_dim_or_aux(standard_name='forecast_period', units='hours',
                         points=points, bounds=bounds),
             dims))

        dims, points, bounds = _reduce_points_and_bounds(
            0.5 * (t1_epoch_hours + t2_epoch_hours),
            (t1_epoch_hours, t2_epoch_hours))
        coords_and_dims.append(
            (_dim_or_aux(standard_name='time', units=epoch_hours_unit,
                         points=points, bounds=bounds),
             dims))

        dims, points, _ = _reduce_points_and_bounds(t2_epoch_hours - lbft)
        coords_and_dims.append(
            (_dim_or_aux(points, standard_name='forecast_reference_time',
                         units=epoch_hours_unit),
             dims))

    if ((lbtim.ib == 3) and
        (lbtim.ic in [1, 2, 4]) and
        ((len(lbcode) != 5) or (len(lbcode) == 5 and
                                lbcode.ix not in [20, 21, 22, 23] and
                                lbcode.iy not in [20, 21, 22, 23]))):
        dims, points, bounds = _reduce_points_and_bounds(
            lbft, (lbft - hours_from_t1_to_t2, lbft))
        coords_and_dims.append(
            (_dim_or_aux(standard_name='forecast_period', units='hours',
                         points=points, bounds=bounds),
             dims))

        dims, points, bounds = _reduce_points_and_bounds(
            t2_epoch_hours, (t1_epoch_hours, t2_epoch_hours))
        coords_and_dims.append(
            (_dim_or_aux(standard_name='time', units=epoch_hours_unit,
                         points=points, bounds=bounds),
             dims))

        dims, points, _ = _reduce_points_and_bounds(t2_epoch_hours - lbft)
        coords_and_dims.append(
            (_dim_or_aux(points, standard_name='forecast_reference_time',
                         units=epoch_hours_unit),
             dims))

    return coords_and_dims


###############################################################################


def _model_level_number(lblev):
    """
    Return model level number for an LBLEV value.

    Args:

    * lblev (int):
        PP field LBLEV value.

    Returns:
        Model level number (integer).

    """
    # See Word no. 33 (LBLEV) in section 4 of UM Model Docs (F3).
    SURFACE_AND_ZEROTH_RHO_LEVEL_LBLEV = 9999

    if lblev == SURFACE_AND_ZEROTH_RHO_LEVEL_LBLEV:
        model_level_number = 0
    else:
        model_level_number = lblev

    return model_level_number


def _convert_scalar_time_coords(lbcode, lbtim, epoch_hours_unit, t1, t2, lbft):
    """
    Encode scalar time values from PP headers as CM data components.

    Returns a list of coords_and_dims.

    """
    t1_epoch_hours = epoch_hours_unit.date2num(t1)
    t2_epoch_hours = epoch_hours_unit.date2num(t2)
    hours_from_t1_to_t2 = t2_epoch_hours - t1_epoch_hours
    hours_from_t2_to_t1 = t1_epoch_hours - t2_epoch_hours
    coords_and_dims = []

    if \
            (lbtim.ia == 0) and \
            (lbtim.ib == 0) and \
            (lbtim.ic in [1, 2, 3, 4]) and \
            (len(lbcode) != 5 or (len(lbcode) == 5 and lbcode.ix not in [20, 21, 22, 23] and lbcode.iy not in [20, 21, 22, 23])):
        coords_and_dims.append((DimCoord(t1_epoch_hours, standard_name='time', units=epoch_hours_unit), None))

    if \
            (lbtim.ia == 0) and \
            (lbtim.ib == 1) and \
            (lbtim.ic in [1, 2, 3, 4]) and \
            (len(lbcode) != 5 or (len(lbcode) == 5 and lbcode.ix not in [20, 21, 22, 23] and lbcode.iy not in [20, 21, 22, 23])):
        coords_and_dims.append((DimCoord(hours_from_t2_to_t1, standard_name='forecast_period', units='hours'), None))
        coords_and_dims.append((DimCoord(t1_epoch_hours, standard_name='time', units=epoch_hours_unit), None))
        coords_and_dims.append((DimCoord(t2_epoch_hours, standard_name='forecast_reference_time', units=epoch_hours_unit), None))

    if \
            (lbtim.ib == 2) and \
            (lbtim.ic in [1, 2, 4]) and \
            ((len(lbcode) != 5) or (len(lbcode) == 5 and lbcode.ix not in [20, 21, 22, 23] and lbcode.iy not in [20, 21, 22, 23])):
        coords_and_dims.append((
            DimCoord(standard_name='forecast_period', units='hours',
                     points=lbft - 0.5 * hours_from_t1_to_t2,
                     bounds=[lbft - hours_from_t1_to_t2, lbft]),
            None))
        coords_and_dims.append((
            DimCoord(standard_name='time', units=epoch_hours_unit,
                     points=0.5 * (t1_epoch_hours + t2_epoch_hours),
                     bounds=[t1_epoch_hours, t2_epoch_hours]),
            None))
        coords_and_dims.append((DimCoord(t2_epoch_hours - lbft, standard_name='forecast_reference_time', units=epoch_hours_unit), None))

    if \
            (lbtim.ib == 3) and \
            (lbtim.ic in [1, 2, 4]) and \
            ((len(lbcode) != 5) or (len(lbcode) == 5 and lbcode.ix not in [20, 21, 22, 23] and lbcode.iy not in [20, 21, 22, 23])):
        coords_and_dims.append((
            DimCoord(standard_name='forecast_period', units='hours',
                     points=lbft, bounds=[lbft - hours_from_t1_to_t2, lbft]),
            None))
        coords_and_dims.append((
            DimCoord(standard_name='time', units=epoch_hours_unit,
                     points=t2_epoch_hours, bounds=[t1_epoch_hours, t2_epoch_hours]),
            None))
        coords_and_dims.append((DimCoord(t2_epoch_hours - lbft, standard_name='forecast_reference_time', units=epoch_hours_unit), None))

    return coords_and_dims


def _convert_scalar_vertical_coords(lbcode, lbvc, blev, lblev, stash,
                                    bhlev, bhrlev, brsvd1, brsvd2, brlev):
    """
    Encode scalar vertical level values from PP headers as CM data components.

    Returns (<list of coords_and_dims>, <list of factories>)

    """
    factories = []
    coords_and_dims = []
    model_level_number = _model_level_number(lblev)

    if \
            (lbvc == 1) and \
            (not (str(stash) in _STASHCODE_IMPLIED_HEIGHTS.keys())) and \
            (blev != -1):
        coords_and_dims.append(
            (DimCoord(blev, standard_name='height', units='m',
                      attributes={'positive': 'up'}),
             None))

    if str(stash) in _STASHCODE_IMPLIED_HEIGHTS.keys():
        coords_and_dims.append(
            (DimCoord(_STASHCODE_IMPLIED_HEIGHTS[str(stash)],
                      standard_name='height', units='m',
                      attributes={'positive': 'up'}),
             None))

    if \
            (len(lbcode) != 5) and \
            (lbvc == 2):
        coords_and_dims.append((DimCoord(model_level_number, standard_name='model_level_number', attributes={'positive': 'down'}), None))

    if \
            (len(lbcode) != 5) and \
            (lbvc == 2) and \
            (brsvd1 == brlev):
        coords_and_dims.append((DimCoord(blev, standard_name='depth', units='m', attributes={'positive': 'down'}), None))

    if \
            (len(lbcode) != 5) and \
            (lbvc == 2) and \
            (brsvd1 != brlev):
        coords_and_dims.append((DimCoord(blev, standard_name='depth', units='m', bounds=[brsvd1, brlev], attributes={'positive': 'down'}), None))

    # soil level
    if len(lbcode) != 5 and lbvc == 6:
        coords_and_dims.append((DimCoord(model_level_number, long_name='soil_model_level_number', attributes={'positive': 'down'}), None))

    if \
            (lbvc == 8) and \
            (len(lbcode) != 5 or (len(lbcode) == 5 and 1 not in [lbcode.ix, lbcode.iy])):
        coords_and_dims.append((DimCoord(blev, long_name='pressure', units='hPa'), None))

    if \
            (len(lbcode) != 5) and \
            (lbvc == 19):
        coords_and_dims.append((DimCoord(blev, standard_name='air_potential_temperature', units='K', attributes={'positive': 'up'}), None))

    # Hybrid pressure levels (--> scalar coordinates)
    if lbvc == 9:
        model_level_number = DimCoord(model_level_number,
                                      standard_name='model_level_number',
                                      attributes={'positive': 'up'})
        # The following match the hybrid height scheme, but data has the
        # blev and bhlev values the other way around.
        #level_pressure = DimCoord(blev,
        #                          long_name='level_pressure',
        #                          units='Pa',
        #                          bounds=[brlev, brsvd1])
        #sigma = AuxCoord(bhlev,
        #                 long_name='sigma',
        #                 bounds=[bhrlev, brsvd2])
        level_pressure = DimCoord(bhlev,
                                  long_name='level_pressure',
                                  units='Pa',
                                  bounds=[bhrlev, brsvd2])
        sigma = AuxCoord(blev,
                         long_name='sigma',
                         bounds=[brlev, brsvd1])
        coords_and_dims.extend([(model_level_number, None),
                                    (level_pressure, None),
                                    (sigma, None)])
        factories.append(Factory(HybridPressureFactory,
                                 [{'long_name': 'level_pressure'},
                                  {'long_name': 'sigma'},
                                  Reference('surface_air_pressure')]))

    # Hybrid height levels (--> scalar coordinates + factory)
    if lbvc == 65:
        coords_and_dims.append((DimCoord(model_level_number, standard_name='model_level_number', attributes={'positive': 'up'}), None))
        coords_and_dims.append((DimCoord(blev, long_name='level_height', units='m', bounds=[brlev, brsvd1], attributes={'positive': 'up'}), None))
        coords_and_dims.append((AuxCoord(bhlev, long_name='sigma', bounds=[bhrlev, brsvd2]), None))
        factories.append(Factory(HybridHeightFactory, [{'long_name': 'level_height'}, {'long_name': 'sigma'}, Reference('orography')]))

    return coords_and_dims, factories


def _convert_scalar_realization_coords(lbrsvd4):
    """
    Encode scalar 'realization' (aka ensemble) numbers as CM data.

    Returns a list of coords_and_dims.

    """
    # Realization (aka ensemble) (--> scalar coordinates)
    coords_and_dims = []
    if lbrsvd4 != 0:
        coords_and_dims.append(
            (DimCoord(lbrsvd4, standard_name='realization'), None))
    return coords_and_dims


def _convert_scalar_pseudo_level_coords(lbuser5):
    """
    Encode scalar pseudo-level values as CM data.

    Returns a list of coords_and_dims.

    """
    coords_and_dims = []
    if lbuser5 != 0:
        coords_and_dims.append(
            (DimCoord(lbuser5, long_name='pseudo_level', units='1'), None))
    return coords_and_dims


def convert(f):
    """
    Converts a PP field into the corresponding items of Cube metadata.

    Args:

    * f:
        A :class:`iris.fileformats.pp.PPField` object.

    Returns:
        A :class:`iris.fileformats.rules.ConversionMetadata` object.

    """
    factories = []
    aux_coords_and_dims = []

    # "Normal" (non-cross-sectional) Time values (--> scalar coordinates)
    time_coords_and_dims = _convert_scalar_time_coords(
        lbcode=f.lbcode, lbtim=f.lbtim,
        epoch_hours_unit=f.time_unit('hours'),
        t1=f.t1, t2=f.t2, lbft=f.lbft)
    aux_coords_and_dims.extend(time_coords_and_dims)

    # "Normal" (non-cross-sectional) Vertical levels
    #    (--> scalar coordinates and factories)
    vertical_coords_and_dims, vertical_factories = \
        _convert_scalar_vertical_coords(
            lbcode=f.lbcode,
            lbvc=f.lbvc,
            blev=f.blev,
            lblev=f.lblev,
            stash=f.stash,
            bhlev=f.bhlev,
            bhrlev=f.bhrlev,
            brsvd1=f.brsvd[0],
            brsvd2=f.brsvd[1],
            brlev=f.brlev)
    aux_coords_and_dims.extend(vertical_coords_and_dims)
    factories.extend(vertical_factories)

    # Realization (aka ensemble) (--> scalar coordinates)
    aux_coords_and_dims.extend(_convert_scalar_realization_coords(
        lbrsvd4=f.lbrsvd[3]))

    # Pseudo-level coordinate (--> scalar coordinates)
    aux_coords_and_dims.extend(_convert_scalar_pseudo_level_coords(
        lbuser5=f.lbuser[4]))

    # All the other rules.
    references, standard_name, long_name, units, attributes, cell_methods, \
        dim_coords_and_dims, other_aux_coords_and_dims = _all_other_rules(f)
    aux_coords_and_dims.extend(other_aux_coords_and_dims)

    return ConversionMetadata(factories, references, standard_name, long_name,
                              units, attributes, cell_methods,
                              dim_coords_and_dims, aux_coords_and_dims)


def _all_other_rules(f):
    """
    This deals with all the other rules that have not been factored into any of
    the other convert_scalar_coordinate functions above.

    """
    references = []
    standard_name = None
    long_name = None
    units = None
    attributes = {}
    cell_methods = []
    dim_coords_and_dims = []
    aux_coords_and_dims = []

    # Season coordinates (--> scalar coordinates)
    if (f.lbtim.ib == 3 and f.lbtim.ic in [1, 2, 4] and
            (len(f.lbcode) != 5 or
             (len(f.lbcode) == 5 and
              (f.lbcode.ix not in [20, 21, 22, 23] and
               f.lbcode.iy not in [20, 21, 22, 23]))) and
            f.lbmon == 12 and f.lbdat == 1 and f.lbhr == 0 and f.lbmin == 0 and
            f.lbmond == 3 and f.lbdatd == 1 and f.lbhrd == 0 and
            f.lbmind == 0):
        aux_coords_and_dims.append(
            (AuxCoord('djf', long_name='season', units='no_unit'),
             None))

    if (f.lbtim.ib == 3 and f.lbtim.ic in [1, 2, 4] and
            ((len(f.lbcode) != 5) or
             (len(f.lbcode) == 5 and
              f.lbcode.ix not in [20, 21, 22, 23]
              and f.lbcode.iy not in [20, 21, 22, 23])) and
            f.lbmon == 3 and f.lbdat == 1 and f.lbhr == 0 and f.lbmin == 0 and
            f.lbmond == 6 and f.lbdatd == 1 and f.lbhrd == 0 and
            f.lbmind == 0):
        aux_coords_and_dims.append(
            (AuxCoord('mam', long_name='season', units='no_unit'),
             None))

    if (f.lbtim.ib == 3 and f.lbtim.ic in [1, 2, 4] and
            ((len(f.lbcode) != 5) or
             (len(f.lbcode) == 5 and
              f.lbcode.ix not in [20, 21, 22, 23] and
              f.lbcode.iy not in [20, 21, 22, 23])) and
            f.lbmon == 6 and f.lbdat == 1 and f.lbhr == 0 and f.lbmin == 0 and
            f.lbmond == 9 and f.lbdatd == 1 and f.lbhrd == 0 and
            f.lbmind == 0):
        aux_coords_and_dims.append(
            (AuxCoord('jja', long_name='season', units='no_unit'),
             None))

    if (f.lbtim.ib == 3 and f.lbtim.ic in [1, 2, 4] and
            ((len(f.lbcode) != 5) or
             (len(f.lbcode) == 5 and
              f.lbcode.ix not in [20, 21, 22, 23] and
              f.lbcode.iy not in [20, 21, 22, 23])) and
            f.lbmon == 9 and f.lbdat == 1 and f.lbhr == 0 and f.lbmin == 0 and
            f.lbmond == 12 and f.lbdatd == 1 and f.lbhrd == 0 and
            f.lbmind == 0):
        aux_coords_and_dims.append(
            (AuxCoord('son', long_name='season', units='no_unit'),
             None))

    # "Normal" (i.e. not cross-sectional) lats+lons (--> vector coordinates)
    if (f.bdx != 0.0 and f.bdx != f.bmdi and len(f.lbcode) != 5 and
            f.lbcode[0] == 1):
        dim_coords_and_dims.append(
            (DimCoord.from_regular(f.bzx, f.bdx, f.lbnpt,
                                   standard_name=f._x_coord_name(),
                                   units='degrees',
                                   circular=(f.lbhem in [0, 4]),
                                   coord_system=f.coord_system()),
             1))

    if (f.bdx != 0.0 and f.bdx != f.bmdi and len(f.lbcode) != 5 and
            f.lbcode[0] == 2):
        dim_coords_and_dims.append(
            (DimCoord.from_regular(f.bzx, f.bdx, f.lbnpt,
                                   standard_name=f._x_coord_name(),
                                   units='degrees',
                                   circular=(f.lbhem in [0, 4]),
                                   coord_system=f.coord_system(),
                                   with_bounds=True),
             1))

    if (f.bdy != 0.0 and f.bdy != f.bmdi and len(f.lbcode) != 5 and
            f.lbcode[0] == 1):
        dim_coords_and_dims.append(
            (DimCoord.from_regular(f.bzy, f.bdy, f.lbrow,
                                   standard_name=f._y_coord_name(),
                                   units='degrees',
                                   coord_system=f.coord_system()),
             0))

    if (f.bdy != 0.0 and f.bdy != f.bmdi and len(f.lbcode) != 5 and
            f.lbcode[0] == 2):
        dim_coords_and_dims.append(
            (DimCoord.from_regular(f.bzy, f.bdy, f.lbrow,
                                   standard_name=f._y_coord_name(),
                                   units='degrees',
                                   coord_system=f.coord_system(),
                                   with_bounds=True),
             0))

    if ((f.bdy == 0.0 or f.bdy == f.bmdi) and
            (len(f.lbcode) != 5 or
             (len(f.lbcode) == 5 and f.lbcode.iy == 10))):
        dim_coords_and_dims.append(
            (DimCoord(f.y, standard_name=f._y_coord_name(), units='degrees',
                      bounds=f.y_bounds, coord_system=f.coord_system()),
             0))

    if ((f.bdx == 0.0 or f.bdx == f.bmdi) and
            (len(f.lbcode) != 5 or
             (len(f.lbcode) == 5 and f.lbcode.ix == 11))):
        dim_coords_and_dims.append(
            (DimCoord(f.x, standard_name=f._x_coord_name(),  units='degrees',
                      bounds=f.x_bounds, circular=(f.lbhem in [0, 4]),
                      coord_system=f.coord_system()),
             1))

    # Cross-sectional vertical level types (--> vector coordinates)
    if (len(f.lbcode) == 5 and f.lbcode.iy == 2 and
            (f.bdy == 0 or f.bdy == f.bmdi)):
        dim_coords_and_dims.append(
            (DimCoord(f.y, standard_name='height', units='km',
                      bounds=f.y_bounds, attributes={'positive': 'up'}),
             0))

    if (len(f.lbcode) == 5 and f.lbcode[-1] == 1 and f.lbcode.iy == 4):
        dim_coords_and_dims.append(
            (DimCoord(f.y, standard_name='depth', units='m',
                      bounds=f.y_bounds, attributes={'positive': 'down'}),
             0))

    if (len(f.lbcode) == 5 and f.lbcode.ix == 10 and f.bdx != 0 and
            f.bdx != f.bmdi):
        dim_coords_and_dims.append(
            (DimCoord.from_regular(f.bzx, f.bdx, f.lbnpt,
                                   standard_name=f._y_coord_name(),
                                   units='degrees',
                                   coord_system=f.coord_system()),
             1))

    if (len(f.lbcode) == 5 and
            f.lbcode.iy == 1 and
            (f.bdy == 0 or f.bdy == f.bmdi)):
        dim_coords_and_dims.append(
            (DimCoord(f.y, long_name='pressure', units='hPa',
                      bounds=f.y_bounds),
             0))

    if (len(f.lbcode) == 5 and f.lbcode.ix == 1 and
            (f.bdx == 0 or f.bdx == f.bmdi)):
        dim_coords_and_dims.append((DimCoord(f.x, long_name='pressure',
                                             units='hPa', bounds=f.x_bounds),
                                    1))

    # Cross-sectional time values (--> vector coordinates)
    if (len(f.lbcode) == 5 and f.lbcode[-1] == 1 and f.lbcode.iy == 23):
        dim_coords_and_dims.append(
            (DimCoord(
                f.y,
                standard_name='time',
                units=Unit('days since 0000-01-01 00:00:00',
                           calendar=iris.unit.CALENDAR_360_DAY),
                bounds=f.y_bounds),
             0))

    if (len(f.lbcode) == 5 and f.lbcode[-1] == 1 and f.lbcode.ix == 23):
        dim_coords_and_dims.append(
            (DimCoord(
                f.x,
                standard_name='time',
                units=Unit('days since 0000-01-01 00:00:00',
                           calendar=iris.unit.CALENDAR_360_DAY),
                bounds=f.x_bounds),
             1))

    # Site number (--> scalar coordinate)
    if (len(f.lbcode) == 5 and f.lbcode[-1] == 1 and f.lbcode.ix == 13 and
            f.bdx != 0):
        dim_coords_and_dims.append(
            (DimCoord.from_regular(f.bzx, f.bdx, f.lbnpt,
                                   long_name='site_number', units='1'),
             1))

    # Site number cross-sections (???)
    if (len(f.lbcode) == 5 and
            13 in [f.lbcode.ix, f.lbcode.iy] and
            11 not in [f.lbcode.ix, f.lbcode.iy] and
            hasattr(f, 'lower_x_domain') and
            hasattr(f, 'upper_x_domain') and
            all(f.lower_x_domain != -1.e+30) and
            all(f.upper_x_domain != -1.e+30)):
        aux_coords_and_dims.append(
            (AuxCoord((f.lower_x_domain + f.upper_x_domain) / 2.0,
                      standard_name=f._x_coord_name(), units='degrees',
                      bounds=np.array([f.lower_x_domain, f.upper_x_domain]).T,
                      coord_system=f.coord_system()),
             1 if f.lbcode.ix == 13 else 0))

    if (len(f.lbcode) == 5 and
            13 in [f.lbcode.ix, f.lbcode.iy] and
            10 not in [f.lbcode.ix, f.lbcode.iy] and
            hasattr(f, 'lower_y_domain') and
            hasattr(f, 'upper_y_domain') and
            all(f.lower_y_domain != -1.e+30) and
            all(f.upper_y_domain != -1.e+30)):
        aux_coords_and_dims.append(
            (AuxCoord((f.lower_y_domain + f.upper_y_domain) / 2.0,
                      standard_name=f._y_coord_name(), units='degrees',
                      bounds=np.array([f.lower_y_domain, f.upper_y_domain]).T,
                      coord_system=f.coord_system()),
             1 if f.lbcode.ix == 13 else 0))

    # LBPROC codings (--> cell methods + attributes)
    if f.lbproc == 128:
        method = 'mean'
    elif f.lbproc == 4096:
        method = 'minimum'
    elif f.lbproc == 8192:
        method = 'maximum'
    else:
        method = None

    if method is not None:
        if f.lbtim.ia != 0:
            intervals = '{} hour'.format(f.lbtim.ia)
        else:
            intervals = None

        if f.lbtim.ib == 2:
            # Aggregation over a period of time.
            cell_methods.append(CellMethod(method,
                                           coords='time',
                                           intervals=intervals))
        elif f.lbtim.ib == 3 and f.lbproc == 128:
            # Aggregation over a period of time within a year, over a number
            # of years.
            # Only mean (lbproc of 128) is handled as the min/max
            # interpretation is ambiguous e.g. decadal mean of daily max,
            # decadal max of daily mean, decadal mean of max daily mean etc.
            cell_methods.append(CellMethod('{} within years'.format(method),
                                           coords='time',
                                           intervals=intervals))
            cell_methods.append(CellMethod('{} over years'.format(method),
                                           coords='time'))
        else:
            # Generic cell method to indicate a time aggregation.
            cell_methods.append(CellMethod(method,
                                           coords='time'))

    if f.lbproc not in [0, 128, 4096, 8192]:
        attributes["ukmo__process_flags"] = tuple(sorted(
            [name for value, name in iris.fileformats.pp.lbproc_map.iteritems()
             if isinstance(value, int) and f.lbproc & value]))

    if (f.lbsrce % 10000) == 1111:
        attributes['source'] = 'Data from Met Office Unified Model'
        # Also define MO-netCDF compliant UM version.
        um_major = (f.lbsrce // 10000) // 100
        if um_major != 0:
            um_minor = (f.lbsrce // 10000) % 100
            attributes['um_version'] = '{:d}.{:d}'.format(um_major, um_minor)

    if (f.lbuser[6] != 0 or
            (f.lbuser[3] // 1000) != 0 or
            (f.lbuser[3] % 1000) != 0):
        attributes['STASH'] = f.stash

    if str(f.stash) in STASH_TO_CF:
        standard_name = STASH_TO_CF[str(f.stash)].standard_name
        units = STASH_TO_CF[str(f.stash)].units
        long_name = STASH_TO_CF[str(f.stash)].long_name

    if (not f.stash.is_valid and f.lbfc in LBFC_TO_CF):
        standard_name = LBFC_TO_CF[f.lbfc].standard_name
        units = LBFC_TO_CF[f.lbfc].units
        long_name = LBFC_TO_CF[f.lbfc].long_name

    # Orography reference field (--> reference target)
    if f.lbuser[3] == 33:
        references.append(ReferenceTarget('orography', None))

    # Surface pressure reference field (--> reference target)
    if f.lbuser[3] == 409 or f.lbuser[3] == 1:
        references.append(ReferenceTarget('surface_air_pressure', None))

    return (references, standard_name, long_name, units, attributes,
            cell_methods, dim_coords_and_dims, aux_coords_and_dims)
