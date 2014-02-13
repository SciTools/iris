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
A package providing :class:`iris.cube.Cube` analysis support.

This module defines a suite of :class:`~iris.analysis.Aggregator` instances,
which are used to specify the statistical measure to calculate over a
:class:`~iris.cube.Cube`, using methods such as
:meth:`~iris.cube.Cube.aggregated_by` and :meth:`~iris.cube.Cube.collapsed`.

The :class:`~iris.analysis.Aggregator` is a convenience class that allows
specific statistical aggregation operators to be defined and instantiated.
These operators can then be used to collapse, or partially collapse, one or
more dimensions of a :class:`~iris.cube.Cube`, as discussed in
:ref:`cube-statistics`.

In particular, :ref:`cube-statistics-collapsing` discusses how to use
:const:`MEAN` to average over one dimension of a :class:`~iris.cube.Cube`,
and also how to perform weighted :ref:`cube-statistics-collapsing-average`.
While :ref:`cube-statistics-aggregated-by` shows how to aggregate similar
groups of data points along a single dimension, to result in fewer points
in that dimension.

The gallery contains several interesting worked examples of how an
:class:`~iris.analysis.Aggregator` may be used, including:
 * :ref:`graphics-COP_1d_plot`
 * :ref:`graphics-SOI_filtering`
 * :ref:`graphics-hovmoller`
 * :ref:`graphics-lagged_ensemble`

"""
from __future__ import division

import collections
from copy import deepcopy

import biggus
import numpy as np
import numpy.ma as ma
import scipy.interpolate
import scipy.stats.mstats

import iris.coords
from iris.exceptions import LazyAggregatorError


__all__ = ('COUNT', 'GMEAN', 'HMEAN', 'MAX', 'MEAN', 'MEDIAN', 'MIN',
           'PEAK', 'PERCENTILE', 'PROPORTION', 'RMS', 'STD_DEV', 'SUM',
           'VARIANCE', 'coord_comparison', 'Aggregator', 'WeightedAggregator',
           'clear_phenomenon_identity')


class _CoordGroup(object):
    """
    Represents a list of coordinates, one for each given cube. Which can be
    operated on conveniently.

    """
    def __init__(self, coords, cubes):
        self.coords = coords
        self.cubes = cubes

    def __iter__(self):
        return iter(self.coords)

    def __getitem__(self, key):
        return list(self).__getitem__(key)

    def _first_coord_w_cube(self):
        """
        Return the first none None coordinate, and its associated cube
        as (cube, coord).

        """
        return filter(lambda cube_coord: cube_coord[1] is not None,
                      zip(self.cubes, self.coords))[0]

    def __repr__(self):
        # No exact repr, so a helpful string is given instead
        return '[' + ', '.join([coord.name() if coord is not None
                                else 'None' for coord in self]) + ']'

    def name(self):
        _, first_coord = self._first_coord_w_cube()
        return first_coord.name()

    def _oid_tuple(self):
        """Return a tuple of object ids for this _CoordGroup's coordinates"""
        return tuple((id(coord) for coord in self))

    def __hash__(self):
        return hash(self._oid_tuple())

    def __eq__(self, other):
        # equals is overridden to guarantee that two _CoordGroups are only
        # equal if their coordinates are the same objects (by object id)
        # this is useful in the context of comparing _CoordGroups if they are
        # part of a set operation such as that in coord_compare, but
        # not useful in many other circumstances (i.e. deepcopying a
        # _CoordGroups instance would mean that copy != original)
        result = NotImplemented
        if isinstance(other, _CoordGroup):
            result = self._oid_tuple() == other._oid_tuple()
        return result

    def matches(self, predicate, default_val=True):
        """
        Apply a function to a coord group returning a list of bools
        for each coordinate.

        The predicate function should take exactly 2 arguments (cube, coord)
        and return a boolean.

        If None is in the coord group then return True.

        """
        for cube, coord in zip(self.cubes, self.coords):
            if coord is None:
                yield default_val
            else:
                yield predicate(cube, coord)

    def matches_all(self, predicate):
        """
        Return whether all coordinates match the given function after running
        it through :meth:`matches`.

        If None is in the coord group then return True.

        """
        return all(self.matches(predicate))

    def matches_any(self, predicate):
        """
        Return whether any coordinates match the given function after running
        it through :meth:`matches`.

        If None is in the coord group then return True.

        """
        return any(self.matches(predicate))


def coord_comparison(*cubes):
    """
    Convenience function to help compare coordinates on one or more cubes
    by their metadata.

    Return a dictionary where the key represents the statement,
    "Given these cubes list the coordinates which,
    when grouped by metadata, are/have..."

    Keys:

    * grouped_coords
       A list of coordinate groups of all the coordinates grouped together
       by their coordinate definition
    * ungroupable
       A list of coordinate groups which contain at least one None,
       meaning not all Cubes provide an equivalent coordinate
    * not_equal
       A list of coordinate groups of which not all are equal
       (superset of ungroupable)
    * no_data_dimension
       A list of coordinate groups of which all have no data dimensions on
       their respective cubes
    * scalar
       A list of coordinate groups of which all have shape (1, )
    * non_equal_data_dimension
       A list of coordinate groups of which not all have the same
       data dimension on their respective cubes
    * non_equal_shape
       A list of coordinate groups of which not all have the same shape
    * equal_data_dimension
       A list of coordinate groups of which all have the same data dimension
       on their respective cubes
    * equal
       A list of coordinate groups of which all are equal
    * ungroupable_and_dimensioned
       A list of coordinate groups of which not all cubes had an equivalent
       (in metadata) coordinate which also describe a data dimension
    * dimensioned
       A list of coordinate groups of which all describe a data dimension on
       their respective cubes
    * ignorable
       A list of scalar, ungroupable non_equal coordinate groups
    * resamplable
        A list of equal, different data dimensioned coordinate groups
    * transposable
       A list of non equal, same data dimensioned, non scalar coordinate groups

    Example usage::

        result = coord_comparison(cube1, cube2)
        print 'All equal coordinates: ', result['equal']

    """
    all_coords = [cube.coords() for cube in cubes]
    grouped_coords = []

    # set of coordinates id()s of coordinates which have been processed
    processed_coords = set()

    # iterate through all cubes, then by each coordinate in the cube looking
    # for coordinate groups
    for cube, coords in zip(cubes, all_coords):
        for coord in coords:

            # if this coordinate has already been processed, then continue on
            # to the next one
            if id(coord) in processed_coords:
                continue

            # setup a list to hold the coordinates which will be turned into a
            # coordinate group and added to the grouped_coords list
            this_coords_coord_group = []

            for other_cube_i, other_cube in enumerate(cubes):
                # setup a variable to hold the coordinate which will be added
                # to the coordinate group for this cube
                coord_to_add_to_group = None

                # don't bother checking if the current cube is the one we are
                # trying to match coordinates too
                if other_cube is cube:
                    coord_to_add_to_group = coord
                else:
                    # iterate through all coordinates in this cube
                    for other_coord in all_coords[other_cube_i]:
                        # for optimisation, check that the name is equivalent
                        # *before* checking all of the metadata is equivalent
                        eq = (id(other_coord) not in processed_coords and
                              other_coord.name() == coord.name() and
                              other_coord._as_defn() == coord._as_defn())
                        if eq:
                            coord_to_add_to_group = other_coord
                            break

                # add the coordinate to the group
                if coord_to_add_to_group is None:
                    this_coords_coord_group.append(None)
                else:
                    this_coords_coord_group.append(coord_to_add_to_group)
                    # add the object id of the coordinate which is being added
                    # to the group to the processed coordinate list
                    processed_coords.add(id(coord_to_add_to_group))

            # add the group to the list of groups
            grouped_coords.append(_CoordGroup(this_coords_coord_group, cubes))

    # define some sets which will be populated in the subsequent loop
    ungroupable = set()
    different_shaped_coords = set()
    different_data_dimension = set()
    no_data_dimension = set()
    scalar_coords = set()
    not_equal = set()

    for coord_group in grouped_coords:
        first_cube, first_coord = coord_group._first_coord_w_cube()

        # Get all coordinate groups which aren't complete (i.e. there is a
        # None in the group)
        coord_is_None_fn = lambda cube, coord: coord is None
        if coord_group.matches_any(coord_is_None_fn):
            ungroupable.add(coord_group)

        # Get all coordinate groups which don't all equal one another
        # (None -> group not all equal)
        not_equal_fn = lambda cube, coord: coord != first_coord
        if coord_group.matches_any(not_equal_fn):
            not_equal.add(coord_group)

        # Get all coordinate groups which don't all share the same shape
        # (None -> group has different shapes)
        diff_shape_fn = lambda cube, coord: coord.shape != first_coord.shape
        if coord_group.matches_any(diff_shape_fn):
            different_shaped_coords.add(coord_group)

        # Get all coordinate groups which don't all share the same data
        # dimension on their respective cubes
        # (None -> group describes a different dimension)
        diff_data_dim_fn = lambda cube, coord: \
            cube.coord_dims(coord) != first_cube.coord_dims(first_coord)
        if coord_group.matches_any(diff_data_dim_fn):
            different_data_dimension.add(coord_group)

        # get all coordinate groups which don't describe a dimension
        # (None -> doesn't describe a dimension)
        no_data_dim_fn = lambda cube, coord: cube.coord_dims(coord) == ()
        if coord_group.matches_all(no_data_dim_fn):
            no_data_dimension.add(coord_group)

        # get all coordinate groups which don't describe a dimension
        # (None -> not a scalar coordinate)
        no_data_dim_fn = lambda cube, coord: coord.shape == (1, )
        if coord_group.matches_all(no_data_dim_fn):
            scalar_coords.add(coord_group)

    result = {}
    result['grouped_coords'] = set(grouped_coords)
    result['not_equal'] = not_equal
    result['ungroupable'] = ungroupable
    result['no_data_dimension'] = no_data_dimension
    result['scalar'] = scalar_coords
    result['non_equal_data_dimension'] = different_data_dimension
    result['non_equal_shape'] = different_shaped_coords

    result['equal_data_dimension'] = (result['grouped_coords'] -
                                      result['non_equal_data_dimension'])
    result['equal'] = result['grouped_coords'] - result['not_equal']
    result['dimensioned'] = (result['grouped_coords'] -
                             result['no_data_dimension'])
    result['ungroupable_and_dimensioned'] = (result['ungroupable'] &
                                             result['dimensioned'])
    result['ignorable'] = ((result['not_equal'] | result['ungroupable']) &
                           result['no_data_dimension'])
    result['resamplable'] = (result['not_equal'] &
                             result['equal_data_dimension'] - result['scalar'])
    result['transposable'] = (result['equal'] &
                              result['non_equal_data_dimension'])

    # for convenience, turn all of the sets in the dictionary into lists,
    # sorted by the name of the group
    for key, groups in result.iteritems():
        result[key] = sorted(groups, key=lambda group: group.name())

    return result


class Aggregator(object):
    """
    The :class:`Aggregator` class provides common aggregation functionality.

    """
    def __init__(self, cell_method, call_func, units_func=None,
                 lazy_func=None, **kwargs):
        """
        Create an aggregator for the given :data:`call_func`.

        Args:

        * cell_method (string):
            Cell method name. Supports string format substitution so
            keywords can be included in the cell method.
            For example, the :data:`PERCENTILE` aggregator uses
            *'percentile ({percent}%)'*, where the *'{percent}'* is replaced
            with *10* when calling *cube.collapsed(PERCENTILE, percent=10)*.  
        * call_func (callable):
            Data aggregation function. Call signature: (data, axis, **kwargs).

        Kwargs:

        * units_func (callable):
            If provided, called to convert a cube's units.
            Call signature: (units) (returns the converted unit).
        * lazy_func (callable):
            An alternative to :data:`call_func` implementing a lazy
            aggregation. Note that, it need not support all features of the
            main operation, but should raise an error in unhandled cases.
        * kwargs:
            Passed through to :data:`call_func`.

        Aggregators are used by cube aggregation functions;
        :meth:`~iris.cube.Cube.collapsed`,
        :meth:`~iris.cube.Cube.aggregated_by` and
        :meth:`~iris.cube.Cube.rolling_window`. A list of aggregators, such as
        :data:`~iris.analysis.MAX` and :data:`~iris.analysis.MEAN`, can be
        seen at the top of this page. Each of these is an :class:`Aggregator`
        object that is passed to Iris::

            result = cube.collapsed('longitude', iris.analysis.MEAN)

        An aggregator can be defined with custom functionality::

            skew = Aggregator('skew', scipy.stats.skew)

        The following example shows how this might be used::

            # Setup.
            import scipy.stats
            from iris.analysis import Aggregator
            from iris.coord_categorisation import add_categorised_coord
            from iris.tests.stock import global_pp
            cube = global_pp()

            # Create a custom aggregator and a coordinate to aggregate.
            skew = Aggregator('skew', scipy.stats.skew)
            add_categorised_coord(
                cube, name='quadrant', from_coord='longitude',
                category_function=lambda crd, val: int(val / 90))

            # Aggregate.
            quad_skew = cube.aggregated_by('quadrant', skew)

        """
        #: Cube cell method string.
        self.cell_method = cell_method
        #: Data aggregation function.
        self.call_func = call_func
        #: Unit conversion function.
        self.units_func = units_func
        #: Lazy aggregation function.
        self.lazy_func = lazy_func

        self._kwargs = kwargs

    def lazy_aggregate(self, data, axis, **kwargs):
        """
        Peform aggregation over the data with a lazy operation, analagous to
        the 'aggregate' result.

        This method is only expected to be used by Iris.

        Args:

        * data (array):
            A lazy array (:class:`biggus.Array`).

        * axis (int or list of int):
            The dimensions to aggregate over -- note that this is defined
            differently to the 'aggregate' method 'axis' argument, which only
            accepts a single dimension index.

        Kwargs:

        * mdtol (float):
            Tolerance of missing data. The value returned will be masked if
            the fraction of data to missing data is less than or equal to
            mdtol.  mdtol=0 means no missing data is tolerated while mdtol=1
            will return the resulting value from the aggregation function.
            Defaults to 1.

            .. warning::

                This may not be supported by all lazy operations.  Where it is
                not supported, *any* mention of this keyword will produce an
                error.

        * kwargs:
            All keyword arguments apart from those specified above, are
            passed through to the aggregation function.
            This function is usually used in conjunction with update_metadata,
            which should be passed the same keyword arguments.

        Returns:
            A lazy array representing the aggregation operation
            (:class:`biggus.Array`).

        """
        if not self.lazy_func:
            raise LazyAggregatorError(
                '{} aggregator does not support lazy operation.'.format(
                    self.cell_method))
        return self.lazy_func(data, axis, **kwargs)

    def aggregate(self, data, axis, **kwargs):
        """
        Perform the aggregation function given the data.

        This method is only expected to be used by Iris.

        Args:

        * data (array):
            Data array.

        * axis (int):
            Axis to aggregate over.

        Kwargs:

        * mdtol (float):
            Tolerance of missing data. The value returned will be masked if
            the fraction of data to missing data is less than or equal to
            mdtol.  mdtol=0 means no missing data is tolerated while mdtol=1
            will return the resulting value from the aggregation function.
            Defaults to 1.

        * kwargs:
            All keyword arguments apart from those specified above, are
            passed through to the aggregation function.
            This function is usually used in conjunction with update_metadata,
            which should be passed the same keyword arguments.

        Returns:
            The aggregated data.

        """
        kwargs = dict(self._kwargs.items() + kwargs.items())
        mdtol = kwargs.pop('mdtol', None)

        result = self.call_func(data, axis=axis, **kwargs)
        if (mdtol is not None and ma.isMaskedArray(data)):
            fraction_not_missing = data.count(axis=axis) / data.shape[axis]
            mask_update = 1 - mdtol > fraction_not_missing
            if ma.isMaskedArray(result):
                result.mask = result.mask | mask_update
            else:
                result = ma.array(result, mask=mask_update)
                if result.ndim is 0:
                    result = result * np.array([1])
                    result.mask = result.mask[np.newaxis]

        return result

    def update_metadata(self, cube, coords, **kwargs):
        """
        Update cube cell method metadata w.r.t the aggregation function.

        This method is only expected to be used by Iris.

        Args:

        * cube (:class:`iris.cube.Cube`):
            Source cube that requires metadata update.
        * coords (:class:`iris.coords.Coord`):
            The coords that were aggregated.

        Kwargs:

        * kwargs:
            This function is usually used in conjunction with aggregate,
            which should be passed the same keyword arguments.

        """
        kwargs = dict(self._kwargs.items() + kwargs.items())

        if not isinstance(coords, (list, tuple)):
            coords = [coords]

        coord_names = []
        for coord in coords:
            if not isinstance(coord, iris.coords.Coord):
                raise TypeError('Coordinate instance expected to the '
                                'Aggregator object.')
            coord_names.append(coord.name())

        # Add a cell method.
        method_name = self.cell_method.format(**kwargs)
        cell_method = iris.coords.CellMethod(method_name, coord_names)
        cube.add_cell_method(cell_method)

        # Update the units if required.
        if self.units_func is not None:
            cube.units = self.units_func(cube.units)

    def post_process(self, collapsed_cube, data_result, **kwargs):
        """
        Ensure the aggregated data is an array if collapsed to a single value.

        This method is only expected to be used by Iris.

        Args:

        * collapsed_cube
            A :class:`iris.cube.Cube`.
        * data_result
            Result from :func:`iris.analysis.Aggregator.aggregate`

        """
        if isinstance(data_result, biggus.Array):
            collapsed_cube.lazy_data(data_result)
        else:
            collapsed_cube.data = iris.util.ensure_array(data_result)
        return collapsed_cube


class WeightedAggregator(Aggregator):
    """
    Convenience class that supports common weighted aggregation functionality.

    """
    def __init__(self, cell_method, call_func, units_func=None,
                 lazy_func=None, **kwargs):
        """
        Create a weighted aggregator for the given :data:`call_func`.

        Args:

        * cell_method (string):
            Cell method string that supports string format substitution.
        * call_func (callable):
            Data aggregation function. Call signature (data, axis, **kwargs).

        Kwargs:

        * units_func (callable):
            Units conversion function.
        * lazy_func (callable):
            An alternative to :data:`call_func` implementing a lazy
            aggregation. Note that, it need not support all features of the
            main operation, but should raise an error in unhandled cases.
        * kwargs:
            Passed through to :data:`call_func`.

        """
        Aggregator.__init__(self, cell_method, call_func,
                            units_func=units_func, lazy_func=lazy_func,
                            **kwargs)

        #: A list of keywords that trigger weighted behaviour.
        self._weighting_keywords = ["returned", "weights"]

    def uses_weighting(self, **kwargs):
        """
        Determine whether this aggregator uses weighting.

        Kwargs:

        * kwargs:
            Arguments to filter of weighted keywords.

        Returns:
            Boolean.

        """
        result = False
        for kwarg in kwargs.keys():
            if kwarg in self._weighting_keywords:
                result = True
                break
        return result

    def post_process(self, collapsed_cube, data_result, **kwargs):
        """
        Process the result from :func:`iris.analysis.Aggregator.aggregate`.

        Ensures data is an array, when collapsed to a single value.
        Returns a tuple(cube, weights) if a tuple(data, weights) was returned
        from :func:`iris.analysis.Aggregator.aggregate`.

        Args:

        * collapsed_cube
            A :class:`iris.cube.Cube`.
        * data_result
            Result from :func:`iris.analysis.Aggregator.aggregate`

        """
        if kwargs.get('returned', False):
            # Package the data into the cube and return a tuple
            collapsed_cube.data, collapsed_weights = data_result
            collapsed_cube.data = iris.util.ensure_array(collapsed_cube.data)
            result = (collapsed_cube, collapsed_weights)
        else:
            result = Aggregator.post_process(self, collapsed_cube,
                                             data_result, **kwargs)

        return result


def _percentile(data, axis, percent, **kwargs):
    # NB. scipy.stats.mstats.scoreatpercentile always works across just the
    # first dimension of its input data, and  returns a result that has one
    # fewer dimension than the input.
    # So shape=(3, 4, 5) -> shape(4, 5)
    data = np.rollaxis(data, axis)
    shape = data.shape[1:]
    if shape:
        data = data.reshape([data.shape[0], np.prod(shape)])
    result = scipy.stats.mstats.scoreatpercentile(data, percent, **kwargs)
    if not ma.isMaskedArray(data) and not ma.is_masked(result):
        result = np.asarray(result)
    if shape:
        result = result.reshape(shape)
    return result


def _count(array, function, axis, **kwargs):
    if not callable(function):
        raise ValueError('function must be a callable. Got %s.'
                         % type(function))
    return ma.sum(function(array), axis=axis, **kwargs)


def _proportion(array, function, axis, **kwargs):
    # if the incoming array is masked use that to count the total number of
    # values
    if isinstance(array, ma.MaskedArray):
        # calculate the total number of non-masked values across the given axis
        total_non_masked = _count(array.mask, np.logical_not,
                                  axis=axis, **kwargs)
        total_non_masked = ma.masked_equal(total_non_masked, 0)
    else:
        total_non_masked = array.shape[axis]

    return _count(array, function, axis=axis, **kwargs) / total_non_masked


def _rms(array, axis, **kwargs):
    rval = np.sqrt(ma.average(np.square(array), axis=axis, **kwargs))
    if not ma.isMaskedArray(array):
        rval = np.asarray(rval)
    return rval


def _sum(array, **kwargs):
    # weighted or scaled sum
    axis_in = kwargs.get('axis', None)
    weights_in = kwargs.pop('weights', None)
    returned_in = kwargs.pop('returned', False)
    if weights_in is not None:
        wsum = ma.sum(weights_in * array, **kwargs)
    else:
        wsum = ma.sum(array, **kwargs)
    if returned_in:
        if weights_in is None:
            weights = np.ones_like(array)
        else:
            weights = weights_in
        rvalue = (wsum, ma.sum(weights, axis=axis_in))
    else:
        rvalue = wsum
    return rvalue


def _peak(array, **kwargs):
    def column_segments(column):
        nan_indices = np.where(np.isnan(column))[0]
        columns = []

        if len(nan_indices) == 0:
            columns.append(column)
        else:
            for index, nan_index in enumerate(nan_indices):
                if index == 0:
                    if index != nan_index:
                        columns.append(column[:nan_index])
                elif nan_indices[index - 1] != (nan_index - 1):
                    columns.append(column[nan_indices[index - 1] + 1:
                                   nan_index])
            if nan_indices[-1] != len(column) - 1:
                columns.append(column[nan_indices[-1] + 1:])
        return columns

    def interp_order(length):
        if length == 1:
            k = None
        elif length > 5:
            k = 5
        else:
            k = length - 1
        return k

    # Collapse array to its final data shape.
    slices = [slice(None)] * array.ndim
    slices[-1] = 0

    if isinstance(array.dtype, np.float):
        data = array[slices]
    else:
        # Cast non-float data type.
        data = array.astype('float32')[slices]

    # Generate nd-index iterator over array.
    shape = list(array.shape)
    shape[-1] = 1
    ndindices = np.ndindex(*shape)

    for ndindex in ndindices:
        ndindex_slice = list(ndindex)
        ndindex_slice[-1] = slice(None)
        column_slice = array[tuple(ndindex_slice)]

        # Check if the column slice contains a single value, nans only,
        # masked values only or if the values are all equal.
        equal_slice = np.ones(column_slice.size,
                              dtype=column_slice.dtype) * column_slice[0]
        if column_slice.size == 1 or \
                all(np.isnan(column_slice)) or \
                ma.count(column_slice) == 0 or \
                np.all(np.equal(equal_slice, column_slice)):
            continue

        # Check if the column slice is masked.
        if ma.isMaskedArray(column_slice):
            # Check if the column slice contains only nans, without inf
            # or -inf values, regardless of the mask.
            if not np.any(np.isfinite(column_slice)) and \
                    not np.any(np.isinf(column_slice)):
                data[ndindex[:-1]] = np.nan
                continue

            # Replace masked values with nans.
            column_slice = column_slice.filled(np.nan)

        # Determine the column segments that require a fitted spline.
        columns = column_segments(column_slice)
        column_peaks = []

        for column in columns:
            # Determine the interpolation order for the spline fit.
            k = interp_order(column.size)

            if k is None:
                column_peaks.append(column[0])
                continue

            tck = scipy.interpolate.splrep(range(column.size), column, k=k)
            npoints = column.size * 100
            points = np.linspace(0, column.size - 1, npoints)
            spline = scipy.interpolate.splev(points, tck)

            column_max = np.max(column)
            spline_max = np.max(spline)
            # Check if the max value of the spline is greater than the
            # max value of the column.
            if spline_max > column_max:
                column_peaks.append(spline_max)
            else:
                column_peaks.append(column_max)

        data[ndindex[:-1]] = np.max(column_peaks)

    return data


#
# Common partial Aggregation class constructors.
#
COUNT = Aggregator('count', _count, lambda units: 1)
"""
An :class:`~iris.analysis.Aggregator` instance that counts the number
of :class:`~iris.cube.Cube` data occurrences that satisfy a particular
criterion, as defined by a user supplied *function*.

**Required** kwargs associated with the use of this aggregator:

* function (callable):
    A function which converts an array of data values into a corresponding
    array of True/False values.

**For example**:

To compute the number of *ensemble members* with precipitation exceeding 10
(in cube data units) could be calculated with::

    result = precip_cube.collapsed('ensemble_member', iris.analysis.COUNT,
                                   function=lambda values: values > 10)

.. seealso:: The :func:`~iris.analysis.PROPORTION` aggregator.

This aggregator handles masked data.

"""


GMEAN = Aggregator('geometric_mean', scipy.stats.mstats.gmean)
"""
An :class:`~iris.analysis.Aggregator` instance that calculates the
geometric mean over a :class:`~iris.cube.Cube`, as computed by
:func:`scipy.stats.mstats.gmean`.

**For example**:

To compute zonal geometric means over the *longitude* axis of a cube::

    result = cube.collapsed('longitude', iris.analysis.GMEAN)

This aggregator handles masked data.

"""


HMEAN = Aggregator('harmonic_mean', scipy.stats.mstats.hmean)
"""
An :class:`~iris.analysis.Aggregator` instance that calculates the
harmonic mean over a :class:`~iris.cube.Cube`, as computed by
:func:`scipy.stats.mstats.hmean`.

**For example**:

To compute zonal harmonic mean over the *longitude* axis of a cube::

    result = cube.collapsed('longitude', iris.analysis.HMEAN)

.. note::

    The harmonic mean is only valid if all data values are greater
    than zero.

This aggregator handles masked data.

"""


MAX = Aggregator('maximum', ma.max)
"""
An :class:`~iris.analysis.Aggregator` instance that calculates
the maximum over a :class:`~iris.cube.Cube`, as computed by
:func:`numpy.ma.max`.

**For example**:

To compute zonal maximums over the *longitude* axis of a cube::

    result = cube.collapsed('longitude', iris.analysis.MAX)

This aggregator handles masked data.

"""


MEAN = WeightedAggregator('mean', ma.average, lazy_func=biggus.mean)
"""
An :class:`~iris.analysis.Aggregator` instance that calculates
the mean over a :class:`~iris.cube.Cube`, as computed by
:func:`numpy.ma.average`.

Additional kwargs associated with the use of this aggregator:

* weights (float ndarray):
    Weights matching the shape of the cube or the length of the window
    for rolling window operations. Note that, latitude/longitude area
    weights can be calculated using
    :func:`iris.analysis.cartography.area_weights`.
* returned (boolean):
    Set this to True to indicate that the collapsed weights are to be
    returned along with the collapsed data. Defaults to False.

**For example**:

To compute zonal means over the *longitude* axis of a cube::

    result = cube.collapsed('longitude', iris.analysis.MEAN)

To compute a weighted area average::

    coords = ('longitude', 'latitude')
    collapsed_cube, collapsed_weights = cube.collapsed(coords,
                                                       iris.analysis.MEAN,
                                                       weights=weights,
                                                       returned=True)

.. note::

    Lazy operation is supported, via :func:`biggus.mean`::

        result = cube.collapsed('longitude', iris.analysis.MEAN, lazy=True)

This aggregator handles masked data.

"""


MEDIAN = Aggregator('median', ma.median)
"""
An :class:`~iris.analysis.Aggregator` instance that calculates
the median over a :class:`~iris.cube.Cube`, as computed by
:func:`numpy.ma.median`.

**For example**:

To compute zonal medians over the *longitude* axis of a cube::

    result = cube.collapsed('longitude', iris.analysis.MEDIAN)

This aggregator handles masked data.

"""


MIN = Aggregator('minimum', ma.min)
"""
An :class:`~iris.analysis.Aggregator` instance that calculates
the minimum over a :class:`~iris.cube.Cube`, as computed by
:func:`numpy.ma.min`.

**For example**:

To compute zonal minimums over the *longitude* axis of a cube::

    result = cube.collapsed('longitude', iris.analysis.MIN)

This aggregator handles masked data.

"""


PEAK = Aggregator('peak', _peak)
"""
An :class:`~iris.analysis.Aggregator` instance that calculates
the peak value derived from a spline interpolation over a
:class:`~iris.cube.Cube`.

The peak calculation takes into account nan values. Therefore, if the number
of non-nan values is zero the result itself will be an array of nan values.

The peak calculation also takes into account masked values. Therefore, if the
number of non-masked values is zero the result itself will be a masked array.

If multiple coordinates are specified, then the peak calculations are
performed individually, in sequence, for each coordinate specified.

**For example**:

To compute the peak over the *time* axis of a cube::

    result = cube.collapsed('time', iris.analysis.PEAK)

This aggregator handles masked data.

"""


PERCENTILE = Aggregator('percentile ({percent}%)',
                        _percentile,
                        alphap=1,
                        betap=1)
"""
An :class:`~iris.analysis.Aggregator` instance that calculates the
percentile over a :class:`~iris.cube.Cube`, as computed by
:func:`scipy.stats.mstats.scoreatpercentile`.

**Required** kwargs associated with the use of this aggregator:

* percent (float):
    Percentile rank at which to extract value.

Additional kwargs associated with the use of this aggregator:

* alphap (float):
    Plotting positions parameter, see :func:`scipy.stats.mstats.mquantiles`.
    Defaults to 1.
* betap (float):
    Plotting positions parameter, see :func:`scipy.stats.mstats.mquantiles`.
    Defaults to 1.

**For example**:

To compute the 90th percentile over *time*::

    result = cube.collapsed('time', iris.analysis.PERCENTILE, percent=90)

This aggregator handles masked data.

"""


PROPORTION = Aggregator('proportion', _proportion, lambda units: 1)
"""
An :class:`~iris.analysis.Aggregator` instance that calculates the
proportion, as a fraction, of :class:`~iris.cube.Cube` data occurrences
that satisfy a particular criterion, as defined by a user supplied
*function*.

**Required** kwargs associated with the use of this aggregator:

* function (callable):
    A function which converts an array of data values into a corresponding
    array of True/False values.

**For example**:

To compute the probability of precipitation exceeding 10
(in cube data units) across *ensemble members* could be calculated with::

    result = precip_cube.collapsed('ensemble_member', iris.analysis.PROPORTION,
                                   function=lambda values: values > 10)

Similarly, the proportion of *time* precipitation exceeded 10
(in cube data units) could be calculated with::

    result = precip_cube.collapsed('time', iris.analysis.PROPORTION,
                                   function=lambda values: values > 10)

.. seealso:: The :func:`~iris.analysis.COUNT` aggregator.

This aggregator handles masked data.

"""


RMS = WeightedAggregator('root mean square', _rms)
"""
An :class:`~iris.analysis.Aggregator` instance that calculates
the root mean square over a :class:`~iris.cube.Cube`, as computed by
((x0**2 + x1**2 + ... + xN-1**2) / N) ** 0.5.

Additional kwargs associated with the use of this aggregator:

* weights (float ndarray):
    Weights matching the shape of the cube or the length of the window for
    rolling window operations. The weights are applied to the squares when
    taking the mean.

**For example**:

To compute the zonal root mean square over the *longitude* axis of a cube::

    result = cube.collapsed('longitude', iris.analysis.RMS)

This aggregator handles masked data.

"""


STD_DEV = Aggregator('standard_deviation', ma.std, ddof=1)
"""
An :class:`~iris.analysis.Aggregator` instance that calculates
the standard deviation over a :class:`~iris.cube.Cube`, as
computed by :func:`numpy.ma.std`.

Additional kwargs associated with the use of this aggregator:

* ddof (integer):
    Delta degrees of freedom. The divisor used in calculations is N - ddof,
    where N represents the number of elements. Defaults to 1.

**For example**:

To compute zonal standard deviations over the *longitude* axis of a cube::

    result = cube.collapsed('longitude', iris.analysis.STD_DEV)

To obtain the biased standard deviation::

    result = cube.collapsed('longitude', iris.analysis.STD_DEV, ddof=0)

This aggregator handles masked data.

"""


SUM = WeightedAggregator('sum', _sum)
"""
An :class:`~iris.analysis.Aggregator` instance that calculates
the sum over a :class:`~iris.cube.Cube`, as computed by :func:`numpy.ma.sum`.

Additional kwargs associated with the use of this aggregator:

* weights (float ndarray):
    Weights matching the shape of the cube, or the length of
    the window for rolling window operations.
* returned (boolean):
    Set this to True to indicate the collapsed weights are to be returned
    along with the collapsed data. Defaults to False.

**For example**:

To compute an accumulation over the *time* axis of a cube::

    result = cube.collapsed('time', iris.analysis.SUM)

To compute a weighted rolling sum e.g. to apply a digital filter::

    weights = np.array([.1, .2, .4, .2, .1])
    result = cube.rolling_window('time', iris.analysis.SUM,
                                 len(weights), weights=weights)

This aggregator handles masked data.

"""


VARIANCE = Aggregator('variance', ma.var, lambda units: units * units,
                      lazy_func=biggus.var, ddof=1)
"""
An :class:`~iris.analysis.Aggregator` instance that calculates
the variance over a :class:`~iris.cube.Cube`, as computed by
:func:`numpy.ma.var`.

Additional kwargs associated with the use of this aggregator:

* ddof (integer):
    Delta degrees of freedom. The divisor used in calculations is N - ddof,
    where N represents the number of elements. Defaults to 1.

**For example**:

To compute zonal variance over the *longitude* axis of a cube::

    result = cube.collapsed('longitude', iris.analysis.VARIANCE)

To obtain the biased variance::

    result = cube.collapsed('longitude', iris.analysis.VARIANCE, ddof=0)

.. note::

    Lazy operation is supported, via :func:`biggus.var`::

        result = cube.collapsed('longitude', iris.analysis.VARIANCE, lazy=True)

This aggregator handles masked data.

"""


class _Groupby(object):
    """
    Convenience class to determine group slices over one or more group-by
    coordinates.

    Generate the coordinate slices for the groups and calculate the
    new group-by coordinates and the new shared coordinates given the
    group slices. Note that, new shared coordinates will be bounded
    coordinates.

    Assumes that all the coordinates share the same axis, therefore all
    of the coordinates must be of the same length.

    Group-by coordinates are those coordinates over which value groups
    are to be determined.

    Shared coordinates are those coordinates which share the same axis
    as group-by coordinates, but which are not to be included in the
    group-by analysis.

    """
    def __init__(self, groupby_coords, shared_coords=None):
        """
        Determine the group slices over the group-by coordinates.

        Args:

        * groupby_coords (list :class:`iris.coords.Coord` instances):
            One or more coordinates from the same axis over which to group-by.

        Kwargs:

        * shared_coords (list of :class:`iris.coords.Coord` instances):
            One or more coordinates that share the same group-by
            coordinate axis.

        """
        #: Group-by and shared coordinates that have been grouped.
        self.coords = []
        self._groupby_coords = []
        self._shared_coords = []
        self._slices_by_key = collections.OrderedDict()
        self._stop = None
        # Ensure group-by coordinates are iterable.
        if not isinstance(groupby_coords, collections.Iterable):
            raise TypeError('groupby_coords must be a '
                            '`collections.Iterable` type.')

        # Add valid group-by coordinates.
        for coord in groupby_coords:
            self._add_groupby_coord(coord)
        # Add the coordinates sharing the same axis as the group-by
        # coordinates.
        if shared_coords is not None:
            # Ensure shared coordinates are iterable.
            if not isinstance(shared_coords, collections.Iterable):
                raise TypeError('shared_coords must be a '
                                '`collections.Iterable` type.')
            # Add valid shared coordinates.
            for coord in shared_coords:
                self._add_shared_coord(coord)

    def _add_groupby_coord(self, coord):
        if coord.ndim != 1:
            raise iris.exceptions.CoordinateMultiDimError(coord)
        if self._stop is None:
            self._stop = coord.shape[0]
        if coord.shape[0] != self._stop:
            raise ValueError('Group-by coordinates have different lengths.')
        self._groupby_coords.append(coord)

    def _add_shared_coord(self, coord):
        if coord.ndim != 1:
            raise iris.exceptions.CoordinateMultiDimError(coord)
        if coord.shape[0] != self._stop and self._stop is not None:
            raise ValueError('Shared coordinates have different lengths.')
        self._shared_coords.append(coord)

    def group(self):
        """
        Calculate the groups and associated slices over one or more group-by
        coordinates.

        Also creates new group-by and shared coordinates given the calculated
        group slices.

        Returns:
            A generator of the coordinate group slices.

        """
        if self._groupby_coords:
            if not self._slices_by_key:
                items = []
                groups = []

                for coord in self._groupby_coords:
                    groups.append(iris.coords._GroupIterator(coord.points))
                    items.append(groups[-1].next())

                # Construct the group slice for each group over the group-by
                # coordinates. Keep constructing until all group-by coordinate
                # groups are exhausted.
                while any([item is not None for item in items]):
                    # Determine the extent (start, stop) of the group given
                    # each current group-by coordinate group.
                    start = max([item.groupby_slice.start for item in items
                                 if item is not None])
                    stop = min([item.groupby_slice.stop for item in items
                                if item is not None])
                    # Construct composite group key for the group using the
                    # start value from each group-by coordinate.
                    key = tuple([coord._points[start] for coord
                                 in self._groupby_coords])
                    # Associate group slice with group key within the ordered
                    # dictionary.
                    self._slices_by_key.setdefault(key, []).append(slice(start,
                                                                         stop))
                    # Prepare for the next group slice construction over the
                    # group-by coordinates.
                    for item_index, item in enumerate(items):
                        if item is None:
                            continue
                        # Get coordinate current group slice.
                        groupby_slice = item.groupby_slice
                        # Determine whether coordinate has spanned all its
                        # groups i.e. its full length
                        # or whether we need to get the coordinates next group.
                        if groupby_slice.stop == self._stop:
                            # This coordinate has exhausted all its groups,
                            # so remove it.
                            items[item_index] = None
                        elif groupby_slice.stop == stop:
                            # The current group of this coordinate is
                            # exhausted, so get the next one.
                            items[item_index] = groups[item_index].next()

                # Merge multiple slices together into one tuple.
                self._slice_merge()
                # Calculate the new group-by coordinates.
                self._compute_groupby_coords()
                # Calculate the new shared coordinates.
                self._compute_shared_coords()
            # Generate the group-by slices/groups.
            for groupby_slice in self._slices_by_key.itervalues():
                yield groupby_slice

        return

    def _slice_merge(self):
        """
        Merge multiple slices into one tuple and collapse items from
        containing list.

        """
        # Iterate over the ordered dictionary in order to reduce
        # multiple slices into a single tuple and collapse
        # all items from containing list.
        for key, groupby_slices in self._slices_by_key.iteritems():
            if len(groupby_slices) > 1:
                # Compress multiple slices into tuple representation.
                groupby_indicies = []

                for groupby_slice in groupby_slices:
                    groupby_indicies.extend(range(groupby_slice.start,
                                                  groupby_slice.stop))

                self._slices_by_key[key] = tuple(groupby_indicies)
            else:
                # Remove single inner slice from list.
                self._slices_by_key[key] = groupby_slices[0]

    def _compute_groupby_coords(self):
        """Create new group-by coordinates given the group slices."""

        groupby_slice = []

        # Iterate over the ordered dictionary in order to construct
        # a group-by slice that samples the first element from each group.
        for key_slice in self._slices_by_key.itervalues():
            if isinstance(key_slice, tuple):
                groupby_slice.append(key_slice[0])
            else:
                groupby_slice.append(key_slice.start)

        groupby_slice = np.array(groupby_slice)

        # Create new group-by coordinates from the group-by slice.
        self.coords = [coord[groupby_slice] for coord in self._groupby_coords]

    def _compute_shared_coords(self):
        """Create the new shared coordinates given the group slices."""

        groupby_bounds = []

        # Iterate over the ordered dictionary in order to construct
        # a list of tuple group boundary indexes.
        for key_slice in self._slices_by_key.itervalues():
            if isinstance(key_slice, tuple):
                groupby_bounds.append((key_slice[0], key_slice[-1]))
            else:
                groupby_bounds.append((key_slice.start, key_slice.stop-1))

        # Create new shared bounded coordinates.
        for coord in self._shared_coords:
            if coord.points.dtype.kind == 'S':
                if coord.bounds is None:
                    new_points = []
                    new_bounds = None
                    for key_slice in self._slices_by_key.itervalues():
                        new_pt = '|'.join(coord.points[i] for i in key_slice)
                        new_points.append(new_pt)
                else:
                    msg = ('collapsing the bounded string coordinate {0!r}'
                           ' is not supported'.format(coord.name()))
                    raise ValueError(msg)
            else:
                new_bounds = []

                # Construct list of coordinate group boundary pairs.
                for start, stop in groupby_bounds:
                    if coord.has_bounds():
                        # Collapse group bounds into bounds.
                        if (getattr(coord, 'circular', False) and
                                (stop + 1) == len(coord.points)):
                            new_bounds.append([coord.bounds[start, 0],
                                              coord.bounds[0, 0] +
                                              coord.units.modulus])
                        else:
                            new_bounds.append([coord.bounds[start, 0],
                                              coord.bounds[stop, 1]])
                    else:
                        # Collapse group points into bounds.
                        if (getattr(coord, 'circular', False) and
                                (stop + 1) == len(coord.points)):
                            new_bounds.append([coord.points[start],
                                              coord.points[0] +
                                              coord.units.modulus])
                        else:
                            new_bounds.append([coord.points[start],
                                              coord.points[stop]])

                # Now create the new bounded group shared coordinate.
                try:
                    new_points = np.array(new_bounds).mean(-1)
                except TypeError:
                    msg = 'The {0!r} coordinate on the collapsing dimension' \
                          ' cannot be collapsed.'.format(coord.name())
                    raise ValueError(msg)

            try:
                self.coords.append(coord.copy(points=new_points,
                                              bounds=new_bounds))
            except ValueError:
                # non monotonic points/bounds
                self.coords.append(iris.coords.AuxCoord.from_coord(coord).copy(
                    points=new_points, bounds=new_bounds))

    def __len__(self):
        """Calculate the number of groups given the group-by coordinates."""

        if self._slices_by_key:
            value = len(self._slices_by_key)
        else:
            value = len([s for s in self.group()])

        return value

    def __repr__(self):
        groupby_coords = [coord.name() for coord in self._groupby_coords]

        if self._shared_coords_by_name:
            shared_coords = [coord.name() for coord in self._shared_coords]
            shared_string = ', shared_coords=%r)' % shared_coords
        else:
            shared_string = ')'

        return '%s(%r%s' % (self.__class__.__name__, groupby_coords,
                            shared_string)


def clear_phenomenon_identity(cube):
    """
    Helper function to clear the standard_name, attributes, and
    cell_methods of a cube.

    """
    cube.rename(None)
    cube.attributes.clear()
    cube.cell_methods = tuple()
