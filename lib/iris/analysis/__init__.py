# (C) British Crown Copyright 2010 - 2012, Met Office
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
A package providing various analysis facilities.

Primarily, this module provides definitions of statistical operations, such as
:const:`MEAN` and :const:`STD_DEV`, that can be applied to Cubes via methods
such as: :meth:`iris.cube.Cube.collapsed` and
:meth:`iris.cube.Cube.aggregated_by`.

 .. note:: These statistical operations define how to transform both the
           metadata and the data.

"""
from __future__ import division

import collections
from copy import deepcopy
from functools import partial

import numpy
import scipy.stats.mstats

import iris.coords


__all__ = ('COUNT', 'GMEAN', 'HMEAN', 'MAX', 'MEAN', 'MEDIAN', 'MIN', 'PERCENTILE', 'PROPORTION', 'STD_DEV', 'SUM', 'VARIANCE',
            'coord_comparison', 'Aggregator', 'clear_phenomenon_identity')


class _CoordGroup(object):
    """Represents a list of coordinates, one for each given cube. Which can be operated on conveniently."""
    def __init__(self, coords, cubes):
        self.coords = coords
        self.cubes = cubes

    def __iter__(self):
        return iter(self.coords)

    def __getitem__(self, key):
        return list(self).__getitem__(key)

    def _first_coord_w_cube(self):
        """Return the first none None coordinate, and its associated cube as (cube, coord)."""
        return filter(lambda cube_coord: cube_coord[1] is not None, zip(self.cubes, self.coords))[0]

    def __repr__(self):
        # No exact repr, so a helpful string is given instead
        return '[' + ', '.join([coord.name() if coord is not None else 'None' for coord in self])  + ']'
        
    def name(self):
        _, first_coord = self._first_coord_w_cube()
        return first_coord.name()
    
    def _oid_tuple(self):
        """Return a tuple of object ids for this _CoordGroup's coordinates"""
        return tuple((id(coord) for coord in self))
    
    def __hash__(self):
        return hash(self._oid_tuple())
    
    def __eq__(self, other):
        # equals is overridden to guarantee that two _CoordGroups are only equal if their coordinates are the same objects (by object id)
        # this is useful in the context of comparing _CoordGroups if they are part of a set operation such as that in coord_compare, but
        # not useful in many other circumstances (i.e. deepcopying a _CoordGroups instance would mean that copy != original) 
        result = NotImplemented
        if isinstance(other, _CoordGroup):
            result = self._oid_tuple() == other._oid_tuple()
        return result
    
    def matches(self, predicate, default_val=True):
        """Apply a function to a coord group returning a list of bools for each coordinate.
        
        The predicate function should take exactly 2 arguments (cube, coord) and return a boolean.
        
        If None is in the coord group then return True.
        
        """        
        for cube, coord in zip(self.cubes, self.coords):
            if coord is None:
                yield default_val
            else:
                yield predicate(cube, coord)
                
    def matches_all(self, predicate):
        """
        Return whether all coordinates match the given function after running it through :meth:`matches`.
        
        If None is in the coord group then return True.
                
        """
        return all(self.matches(predicate))
    
    def matches_any(self, predicate):
        """
        Return whether any coordinates match the given function after running it through :meth:`matches`. 
        
        If None is in the coord group then return True.
        
        """
        return any(self.matches(predicate))
        

def coord_comparison(*cubes):
    """
    Convenience function to help compare coordinates on one or more cubes by their metadata.
    
    Return a dictionary where the key represents the statement, "Given these cubes list the coordinates 
    which, when grouped by metadata, are/have..."
    
    Keys:
    
    * grouped_coords
       A list of coordinate groups of all the coordinates grouped together by their coordinate definition
    * ungroupable
       A list of coordinate groups which contain at least one None, meaning not all Cubes provide an equivalent coordinate  
    * not_equal
       A list of coordinate groups of which not all are equal (superset of ungroupable)
    * no_data_dimension
       A list of coordinate groups of which all have no data dimensions on their respective cubes 
    * scalar
       A list of coordinate groups of which all have shape (1, )
    * non_equal_data_dimension
       A list of coordinate groups of which not all have the same data dimension on their respective cubes
    * non_equal_shape
       A list of coordinate groups of which not all have the same shape
    * equal_data_dimension
       A list of coordinate groups of which all have the same data dimension on their respective cubes
    * equal
       A list of coordinate groups of which all are equal
    * ungroupable_and_dimensioned
       A list of coordinate groups of which not all cubes had an equivalent (in metadata) coordinate which also describe a data dimension
    * dimensioned
       A list of coordinate groups of which all describe a data dimension on their respective cubes
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
    
    # iterate through all cubes, then by each coordinate in the cube looking for coordinate groups
    for cube, coords in zip(cubes, all_coords):
        for coord in coords:
            
            # if this coordinate has already been processed, then continue on to the next one
            if id(coord) in processed_coords:
                continue
        
            # setup a list to hold the coordinates which will be turned into a coordinate group and added to the grouped_coords list
            this_coords_coord_group = []
            
            for other_cube_i, other_cube in enumerate(cubes):
                # setup a variable to hold the coordinate which will be added to the coordinate group for this cube
                coord_to_add_to_group = None
                
                # don't bother checking if the current cube is the one we are trying to match coordinates too
                if other_cube is cube:
                    coord_to_add_to_group = coord
                else:
                    # iterate through all coordinates in this cube
                    for other_coord in all_coords[other_cube_i]:
                        # for optimisation, check that the name is equivalent *before* checking all of the metadata is equivalent
                        if (id(other_coord) not in processed_coords and 
                                                other_coord.name() == coord.name() and 
                                                other_coord._as_defn() == coord._as_defn()):
                            coord_to_add_to_group = other_coord
                            break
                
                # add the coordinate to the group
                if coord_to_add_to_group is None:
                    this_coords_coord_group.append(None)
                else:
                    this_coords_coord_group.append(coord_to_add_to_group)
                    # add the object id of the coordinate which is being added to the group to the processed coordinate list
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
        
        # Get all coordinate groups which aren't complete (i.e. there is a None in the group)
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
        
        # Get all coordinate groups which don't all share the same data dimension on their respective cubes
        # (None -> group describes a different dimension)    
        diff_data_dim_fn = lambda cube, coord: cube.coord_dims(coord=coord) != first_cube.coord_dims(first_coord)
        if coord_group.matches_any(diff_data_dim_fn):
            different_data_dimension.add(coord_group)
        
        # get all coordinate groups which don't describe a dimension 
        # (None -> doesn't describe a dimension)
        no_data_dim_fn = lambda cube, coord: cube.coord_dims(coord=coord) == []
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
    
    result['equal_data_dimension'] = result['grouped_coords'] - result['non_equal_data_dimension']
    result['equal'] = result['grouped_coords'] - result['not_equal']
    result['dimensioned'] = result['grouped_coords'] - result['no_data_dimension']
    result['ungroupable_and_dimensioned'] = result['ungroupable'] & result['dimensioned']
    result['ignorable'] = (result['not_equal'] | result['ungroupable']) & result['no_data_dimension']
    result['resamplable'] = result['not_equal'] & result['equal_data_dimension'] - result['scalar']
    result['transposable'] = result['equal'] & result['non_equal_data_dimension']

    # for convenience, turn all of the sets in the dictionary into lists, sorted by the name of the group  
    for key, groups in result.iteritems():
        result[key] = sorted(groups, key=lambda group: group.name())
        
    return result


class Aggregator(object):
    """Convenience class that supports common aggregation functionality."""
    
    def __init__(self, history, cell_method, call_func, **kwargs):
        """
        Create an aggregator for the given call_func.
        
        Args:
        
        * history (string):
            History string that supports string format substitution.
        * cell_method (string):
            Cell method string that supports string format substitution.
        * call_func (callable):
            Data aggregation function.
        
        Kwargs:
        
        * kwargs:
            Passed through to call_func.
        
        """
        self.history = history
        '''Cube history string that supports string format substitution.'''
        self.cell_method = cell_method
        '''Cube cell method string.'''
        self.call_func = call_func
        '''Data aggregation function.'''

        self._kwargs = kwargs

    def aggregate(self, data, axis, **kwargs):
        """
        Perform the aggregation function given the data.
        
        Keyword arguments are passed through to the data aggregation function
        (for example, the "percent" keyword for a percentile aggregator).
        This function is usually used in conjunction with update_metadata(),
        which should be passed the same keyword arguments.
        
        Returns:
            The aggregated data.
            
        """
        kwargs = dict(self._kwargs.items() + kwargs.items())
        return self.call_func(data, axis=axis, **kwargs)
        
    def update_metadata(self, cube, coords, **kwargs):
        """
        Update cube history and cell method metadata w.r.t the aggregation function.
        
        Args:
        
        * cube (:class:`iris.cube.Cube`):
            Source cube that requires metadata update.
        * coords (:class:`iris.coords.Coord`):
            The coords that were aggregated.
            
        Kwargs:

        * This function is intended to be used in conjuction with aggregate() and should be
          passed the same keywords (for example, the "percent" keyword for a percentile aggregator).

        """
        kwargs = dict(self._kwargs.items() + kwargs.items())
        
        if not isinstance(coords, (list, tuple)):
            coords = [coords]
           
        coord_names = []
        for coord in coords:
            if not isinstance(coord, iris.coords.Coord):
                raise TypeError('Coordinate instance expected to the Aggregator object.')
            coord_names.append(coord.name())
            
        action = kwargs.pop('action', 'over')
        if kwargs.pop('aggregate', False):
            action = 'aggregated ' + action

        # Add standard items to the history dictionary.
        history_dict = deepcopy(self._kwargs)
        history_dict['standard_name'] = cube.name()
        history_dict['coord_names'] = ', '.join(coord_names)
        history_dict['action'] = action
        history_dict.update(kwargs)
        # Add history in-place.
        cube.add_history(self.history.format(**history_dict))
        # Add a cell method.
        method_name = self.cell_method.format(**kwargs)
        cell_method = iris.coords.CellMethod(method_name, coord_names)
        cube.add_cell_method(cell_method)        

    def post_process(self, collapsed_cube, data_result, **kwargs):
        """
        Process the result from :func:`iris.analysis.Aggregator.aggregate`.
        
        Ensures data is an array, when collapsed to a single value.
        
        Args:
        
        * collapsed_cube
            A :class:`iris.cube.Cube`.
        * data_result
            Result from :func:`iris.analysis.Aggregator.aggregate`

        """
        collapsed_cube.data = iris.util.ensure_array(data_result)
        return collapsed_cube


class WeightedAggregator(Aggregator):

    def __init__(self, history, cell_method, call_func, **kwargs):
        Aggregator.__init__(self, history, cell_method, call_func, **kwargs)

        self._weighting_keywords = ["returned", "weights"]
        '''A list of keywords that trigger weighted behaviour.'''

    def uses_weighting(self, **kwargs):
        """Does this aggregator use weighting with the given keywords?"""
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
        Returns a tuple(cube, weights) if a tuple(data, weights) was returned from :func:`iris.analysis.Aggregator.aggregate`.
        
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
            result = Aggregator.post_process(self, collapsed_cube, data_result, **kwargs)

        return result


def _percentile(data, axis, percent, **kwargs):
    # NB. scipy.stats.mstats.scoreatpercentile always works across just the first
    # dimension of its input data, and  returns a result that has one fewer
    # dimension than the input.
    # So shape=(3, 4, 5) -> shape(4, 5)
    if axis != 0:
        data = data.swapaxes(0, axis)
    return numpy.array(scipy.stats.mstats.scoreatpercentile(data, percent, **kwargs), ndmin=1)


def _count(array, function, axis, **kwargs):
    if not callable(function):
        raise ValueError('function must be a callable. Got %s.' % type(function))
    return numpy.ma.sum(function(array), axis=axis, **kwargs)


def _proportion(array, function, axis, **kwargs):
    # if the incoming array is masked use that to count the total number of values
    if isinstance(array, numpy.ma.MaskedArray):
        # calculate the total number of non-masked values across the given axis
        total_non_masked = _count(array.mask, lambda v: v == False, axis=axis, **kwargs)
        total_non_masked = numpy.ma.masked_equal(total_non_masked, 0)
    else:
        total_non_masked = array.shape[axis] 
        
    return _count(array, function, axis=axis, **kwargs) / total_non_masked


#
# Common partial Aggregation class constructors.
#
COUNT = Aggregator('Count of {standard_name:s} {action:s} {coord_names:s}',
                                       'count',
                                       _count)
"""
The number of data that match the given function.

Args:

* function:
    A function which returns True or False given a value in the data array. 

For example, the number of ensembles with precipitation exceeding 10 (in cube data units) could be calculated with::

    result = precip_cube.collapsed('ensemble_member', iris.analysis.COUNT, 
                                   function=lambda data_value: data_value > 10)

.. seealso:: :func:`iris.analysis.PROPORTION`
                                   
"""


GMEAN = Aggregator('Geometric mean of {standard_name:s} {action:s} {coord_names:s}',
                'geometric_mean',
                scipy.stats.mstats.gmean)
"""
The geometric mean, as computed by :func:`scipy.stats.mstats.gmean`.

For example, to compute zonal geometric means::

    result = cube.collapsed('longitude', iris.analysis.GMEAN)

"""


HMEAN = Aggregator('Harmonic mean of {standard_name:s} {action:s} {coord_names:s}',
                'harmonic_mean',
                scipy.stats.mstats.hmean)
"""
The harmonic mean, as computed by :func:`scipy.stats.mstats.hmean`.

For example, to compute zonal harmonic means::

    result = cube.collapsed('longitude', iris.analysis.HMEAN)

.. note:: The harmonic mean is only valid if all data values are greater than zero.

"""


MAX = Aggregator('Maximum of {standard_name:s} {action:s} {coord_names:s}',
              'maximum',
              numpy.ma.max)
"""
The maximum, as computed by :func:`numpy.ma.max`.

For example, to compute zonal maximums::

    result = cube.collapsed('longitude', iris.analysis.MAX)

"""


MEAN = WeightedAggregator('Mean of {standard_name:s} {action:s} {coord_names:s}',
               'mean',
               numpy.ma.average)
"""
The mean, as computed by :func:`numpy.ma.average`.

For example, to compute zonal means::

    result = cube.collapsed('longitude', iris.analysis.MEAN)
    
Additional kwargs available:

* weights
    Optional array of floats. If supplied, the shape must match the cube.

    LatLon area weights can be calculated using :func:`iris.analysis.cartography.area_weights`.
* returned
    Set this to True to indicate the collapsed weights are to be returned
    along with the collapsed data. Defaults to False.

For example::

    cube_out, weights_out = cube_in.collapsed(coord_names, iris.analysis.MEAN, weights=weights_in, returned=True)

"""


MEDIAN = Aggregator('Median of {standard_name:s} {action:s} {coord_names:s}',
                 'median',
                 numpy.ma.median)
"""
The median, as computed by :func:`numpy.ma.median`.

For example, to compute zonal medians::

    result = cube.collapsed('longitude', iris.analysis.MEDIAN)

"""


MIN = Aggregator('Minimum of {standard_name:s} {action:s} {coord_names:s}',
              'minimum',
              numpy.ma.min)
"""
The minimum, as computed by :func:`numpy.ma.min`.

For example, to compute zonal minimums::

    result = cube.collapsed('longitude', iris.analysis.MIN)

"""


PERCENTILE = Aggregator('Percentile ({percent}%) of {standard_name:s} {action:s} {coord_names:s}',
                  'percentile ({percent}%)',
                  _percentile,
                  alphap=1, 
                  betap=1,
                  )
"""
The percentile, as computed by :func:`scipy.stats.mstats.scoreatpercentile`.

Required kwargs:

* percent:
    Percentile rank at which to extract value. No default.

For example, to compute the 90th percentile over time::

    result = cube.collapsed('time', iris.analysis.PERCENTILE, percent=90)

.. note::
    The default values of ``alphap`` and ``betap`` are both 1. For detailed meanings on
    these values see :func:`scipy.stats.mstats.mquantiles`.

"""


PROPORTION = Aggregator('Proportion of {standard_name:s} {action:s} {coord_names:s}',
                                       'proportion',
                                       _proportion)
"""
The proportion, as a decimal, of data that match the given function.

The proportion calculation takes into account masked values, therefore if the number of non-masked values is 
zero the result itself will be a masked array.

Args:

* function:
    A function which returns True or False given a value in the data array. 

For example, the probability of precipitation exceeding 10 (in cube data units) across ensemble members
could be calculated with::

    result = precip_cube.collapsed('ensemble_member', iris.analysis.PROPORTION, 
                                   function=lambda data_value: data_value > 10)

Similarly, the proportion of times precipitation exceeded 10 (in cube data units) could be calculated with::

    result = precip_cube.collapsed('time', iris.analysis.PROPORTION, 
                                   function=lambda data_value: data_value > 10)

.. seealso:: :func:`iris.analysis.COUNT`
                                
"""


STD_DEV = Aggregator('Standard deviation of {standard_name:s} {action:s} {coord_names:s} (delta degrees of freedom: {ddof:d})',
                  'standard_deviation',
                  numpy.ma.std,
                  ddof=1)
"""
The standard deviation, as computed by :func:`numpy.ma.std`.

For example, to compute zonal standard deviations::

    result = cube.collapsed('longitude', iris.analysis.STD_DEV)

Additional kwargs available:

* ddof:
    Delta degrees of freedom. The divisor used in calculations is N - ddof,
    where N represents the number of elements. By default ddof is one.

For example, to obtain the biased standard deviation::

    result = cube.collapsed(coord_to_collapse, iris.analysis.STD_DEV, ddof=0)

"""


SUM = Aggregator('Sum of {standard_name:s} {action:s} {coord_names:s}',
              'sum',
              numpy.ma.sum)
"""
The sum of a dataset, as computed by :func:`numpy.ma.sum`.

For example, to compute an accumulation over time::

    result = cube.collapsed('time', iris.analysis.SUM)

"""


VARIANCE = Aggregator('Variance of {standard_name:s} {action:s} {coord_names:s} (delta degrees of freedom: {ddof:d})',
                   'variance',
                   numpy.ma.var,
                   ddof=1)
"""
The variance, as computed by :func:`numpy.ma.var`.

For example, to compute zonal variance::

    result = cube.collapsed('longitude', iris.analysis.VARIANCE)

Additional kwargs available:

* ddof:
    Delta degrees of freedom. The divisor used in calculations is N - ddof,
    where N represents the number of elements. By default ddof is one.

For example, to obtain the biased variance::

    result = cube.collapsed(coord_to_collapse, iris.analysis.VARIANCE, ddof=0)

"""


class _Groupby(object):
    """
    Convenience class to determine group slices over one or more group-by coordinates.
    
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
            One or more coordinates that share the same group-by coordinate axis.
            
        """
        self.coords = []
        '''Group-by and shared coordinates that have been grouped.'''
        self._groupby_coords = []
        self._shared_coords = []
        self._slices_by_key = collections.OrderedDict()
        self._stop = None
        # Ensure group-by coordinates are iterable.
        if not isinstance(groupby_coords, collections.Iterable):
            raise TypeError('groupby_coords must be a `collections.Iterable` type.')

        # Add valid group-by coordinates.
        for coord in groupby_coords:
            self._add_groupby_coord(coord)
        # Add the coordinates sharing the same axis as the group-by coordinates.
        if shared_coords is not None:
            # Ensure shared coordinates are iterable.
            if not isinstance(shared_coords, collections.Iterable):
                raise TypeError('shared_coords must be a `collections.Iterable` type.')
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
        Calculate the groups and associated slices over one or more group-by coordinates.
        
        Also creates new group-by and shared coordinates given the calculated group slices.
        
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
                    
                # Construct the group slice for each group over the group-by coordinates.
                # Keep constructing until all group-by coordinate groups are exhausted.
                while any([item is not None for item in items]):
                    # Determine the extent (start, stop) of the group given each current group-by coordinate group.
                    start = max([item.groupby_slice.start for item in items if item is not None])
                    stop = min([item.groupby_slice.stop for item in items if item is not None])
                    # Construct composite group key for the group using the start value from each group-by coordinate.
                    key = tuple([coord._points[start] for coord in self._groupby_coords])
                    # Associate group slice with group key within the ordered dictionary.
                    self._slices_by_key.setdefault(key, []).append(slice(start, stop))
                    # Prepare for the next group slice construction over the group-by coordinates.
                    for item_index, item in enumerate(items):
                        # Get coordinate current group slice.
                        groupby_slice = item.groupby_slice
                        # Determine whether coordinate has spanned all its groups i.e. its full length
                        # or whether we need to get the coordinates next group.
                        if groupby_slice.stop == self._stop:
                            # This coordinate has exhausted all its groups, so remove it.
                            items[item_index] = None
                        elif groupby_slice.stop == stop:
                            # The current group of this coordinate is exhausted, so get the next one.
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
        """Merge multiple slices into one tuple and collapse items from containing list."""

        # Iterate over the ordered dictionary in order to reduce
        # multiple slices into a single tuple and collapse
        # all items from containing list.
        for key, groupby_slices in self._slices_by_key.iteritems():
            if len(groupby_slices) > 1:
                # Compress multiple slices into tuple representation.
                groupby_indicies = []
                
                for groupby_slice in groupby_slices:
                    groupby_indicies.extend(range(groupby_slice.start, groupby_slice.stop))
                    
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
                
        groupby_slice = numpy.array(groupby_slice)
        
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
            new_bounds = []
            
            # Construct list of coordinate group boundary pairs.
            for start, stop in groupby_bounds:
                if coord.has_bounds():
                    # Collapse group bounds into bounds.
                    if getattr(coord, 'circular', False) and (stop+1) == len(coord):
                        new_bounds.append([coord.bounds[start, 0], coord.bounds[0, 0] + coord.units.modulus])
                    else:
                        new_bounds.append([coord.bounds[start, 0], coord.bounds[stop, 1]])
                else:
                    # Collapse group points into bounds.
                    if getattr(coord, 'circular', False) and (stop+1) == len(coord):
                        new_bounds.append([coord.points[start], coord.points[0] + coord.units.modulus])
                    else:
                        new_bounds.append([coord.points[start], coord.points[stop]])
            
            # Now create the new bounded group shared coordinate.
            new_points = numpy.array(new_bounds).mean(-1)
            self.coords.append(coord.copy(points=new_points, bounds=new_bounds))

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
            shared_string =  ', shared_coords=%r)' % shared_coords
        else:
            shared_string = ')'
        
        return '%s(%r%s' % (self.__class__.__name__, groupby_coords, shared_string)


def clear_phenomenon_identity(cube):
    """Helper function to clear the standard_name, attributes, and cell_methods of a cube."""
    cube.rename(None)
    cube.attributes.clear()
    cube.cell_methods = tuple()
