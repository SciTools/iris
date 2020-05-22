# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
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

 * :ref:`Meteorology-COP_1d_plot`
 * :ref:`General-SOI_filtering`
 * :ref:`Meteorology-hovmoller`
 * :ref:`Meteorology-lagged_ensemble`
 * :ref:`General-custom_aggregation`

"""

from collections import OrderedDict
from collections.abc import Iterable
from functools import wraps

import dask.array as da
import numpy as np
import numpy.ma as ma
import scipy.interpolate
import scipy.stats.mstats

from iris.analysis._area_weighted import AreaWeightedRegridder
from iris.analysis._interpolation import (
    EXTRAPOLATION_MODES,
    RectilinearInterpolator,
)
from iris.analysis._regrid import RectilinearRegridder, CurvilinearRegridder
import iris.coords
from iris.exceptions import LazyAggregatorError
import iris._lazy_data

__all__ = (
    "COUNT",
    "GMEAN",
    "HMEAN",
    "MAX",
    "MEAN",
    "MEDIAN",
    "MIN",
    "PEAK",
    "PERCENTILE",
    "PROPORTION",
    "RMS",
    "STD_DEV",
    "SUM",
    "VARIANCE",
    "WPERCENTILE",
    "Aggregator",
    "WeightedAggregator",
    "clear_phenomenon_identity",
    "Linear",
    "AreaWeighted",
    "Nearest",
    "UnstructuredNearest",
    "PointInCell",
)


class _CoordGroup:
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
        return next(
            filter(
                lambda cube_coord: cube_coord[1] is not None,
                zip(self.cubes, self.coords),
            )
        )

    def __repr__(self):
        # No exact repr, so a helpful string is given instead
        return (
            "["
            + ", ".join(
                [
                    coord.name() if coord is not None else "None"
                    for coord in self
                ]
            )
            + "]"
        )

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


def _dimensional_metadata_comparison(*cubes, object_get=None):
    """
    Convenience function to help compare coordinates, cell-measures or
    ancillary-variables, on one or more cubes, by their metadata.

    .. Note::

        Up to Iris 2.x, this _used_ to be the public API method
        "iris.analysis.coord_comparison".
        It has since been generalised, and made private.
        However, the cube elements handled are still mostly referred to as 'coords' /
        'coordinates' throughout, for simplicity :  In fact, they will all be either
        `iris.coords.Coord`, `iris.coords.CellMeasure` or
        `iris.coords.AncillaryVariable`, the cube element type being controlled by the
        'object_get' keyword.

    Args:

    * cubes (iterable of `iris.cube.Cube`):
        a set of cubes whose coordinates, cell-measures or ancillary-variables are to
        be compared.

    Kwargs:

    * object_get (callable(cube) or None):
        If not None, this must be a cube method returning a list of all cube elements
        of the required type, i.e. one of `iris.cube.Cube.coords`,
        `iris.cube.Cube.cell_measures`, or `iris.cube.Cube.ancillary_variables`.
        If not specified, defaults to `iris.cube.Cube.coords`

    Returns:

        result (dict mapping string: list of _CoordGroup):
            A dictionary whose keys are match categories and values are groups of
            coordinates, cell-measures or ancillary-variables.

    The values of the returned dictionary are lists of _CoordGroup representing
    grouped coordinates.  Each _CoordGroup contains all the input 'cubes', and a
    matching list of the coord within each cube that matches some specific CoordDefn
    (or maybe None).

    The keys of the returned dictionary are strings naming 'categories' :  Each
    represents a statement,
    "Given these cubes list the coordinates which,
    when grouped by metadata, are/have..."

    Returned Keys:

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

        result = _dimensional_metadata_comparison(cube1, cube2)
        print('All equal coordinates: ', result['equal'])

    """
    if object_get is None:
        from iris.cube import Cube

        object_get = Cube.coords

    all_coords = [object_get(cube) for cube in cubes]
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
                        eq = (
                            other_coord is coord
                            or other_coord.name() == coord.name()
                            and other_coord._as_defn() == coord._as_defn()
                        )
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
        def coord_is_None_fn(cube, coord):
            return coord is None

        if coord_group.matches_any(coord_is_None_fn):
            ungroupable.add(coord_group)

        # Get all coordinate groups which don't all equal one another
        # (None -> group not all equal)
        def not_equal_fn(cube, coord):
            return coord != first_coord

        if coord_group.matches_any(not_equal_fn):
            not_equal.add(coord_group)

        # Get all coordinate groups which don't all share the same shape
        # (None -> group has different shapes)
        def diff_shape_fn(cube, coord):
            return coord.shape != first_coord.shape

        if coord_group.matches_any(diff_shape_fn):
            different_shaped_coords.add(coord_group)

        # Get all coordinate groups which don't all share the same data
        # dimension on their respective cubes
        # (None -> group describes a different dimension)
        def diff_data_dim_fn(cube, coord):
            return coord.cube_dims(cube) != first_coord.cube_dims(first_cube)

        if coord_group.matches_any(diff_data_dim_fn):
            different_data_dimension.add(coord_group)

        # get all coordinate groups which don't describe a dimension
        # (None -> doesn't describe a dimension)
        def no_data_dim_fn(cube, coord):
            return coord.cube_dims(cube) == ()

        if coord_group.matches_all(no_data_dim_fn):
            no_data_dimension.add(coord_group)

        # get all coordinate groups which don't describe a dimension
        # (None -> not a scalar coordinate)
        def no_data_dim_fn(cube, coord):
            return coord.shape == (1,)

        if coord_group.matches_all(no_data_dim_fn):
            scalar_coords.add(coord_group)

    result = {}
    result["grouped_coords"] = set(grouped_coords)
    result["not_equal"] = not_equal
    result["ungroupable"] = ungroupable
    result["no_data_dimension"] = no_data_dimension
    result["scalar"] = scalar_coords
    result["non_equal_data_dimension"] = different_data_dimension
    result["non_equal_shape"] = different_shaped_coords

    result["equal_data_dimension"] = (
        result["grouped_coords"] - result["non_equal_data_dimension"]
    )
    result["equal"] = result["grouped_coords"] - result["not_equal"]
    result["dimensioned"] = (
        result["grouped_coords"] - result["no_data_dimension"]
    )
    result["ungroupable_and_dimensioned"] = (
        result["ungroupable"] & result["dimensioned"]
    )
    result["ignorable"] = (
        result["not_equal"] | result["ungroupable"]
    ) & result["no_data_dimension"]
    result["resamplable"] = (
        result["not_equal"] & result["equal_data_dimension"] - result["scalar"]
    )
    result["transposable"] = (
        result["equal"] & result["non_equal_data_dimension"]
    )

    # for convenience, turn all of the sets in the dictionary into lists,
    # sorted by the name of the group
    for key, groups in result.items():
        result[key] = sorted(groups, key=lambda group: group.name())

    return result


class _Aggregator:
    """
    The :class:`_Aggregator` base class provides common aggregation
    functionality.

    """

    def __init__(
        self, cell_method, call_func, units_func=None, lazy_func=None, **kwargs
    ):
        r"""
        Create an aggregator for the given :data:`call_func`.

        Args:

        * cell_method (string):
            Cell method definition formatter.  Used in the fashion
            "cell_method.format(\**kwargs)", to produce a cell-method string
            which can include keyword values.

        * call_func (callable):
            | *Call signature*: (data, axis=None, \**kwargs)

            Data aggregation function.
            Returns an aggregation result, collapsing the 'axis' dimension of
            the 'data' argument.

        Kwargs:

        * units_func (callable):
            | *Call signature*: (units)

            If provided, called to convert a cube's units.
            Returns an :class:`cf_units.Unit`, or a
            value that can be made into one.

        * lazy_func (callable or None):
            An alternative to :data:`call_func` implementing a lazy
            aggregation. Note that, it need not support all features of the
            main operation, but should raise an error in unhandled cases.

        Additional kwargs::
            Passed through to :data:`call_func` and :data:`lazy_func`.

        Aggregators are used by cube aggregation methods such as
        :meth:`~iris.cube.Cube.collapsed` and
        :meth:`~iris.cube.Cube.aggregated_by`.  For example::

            result = cube.collapsed('longitude', iris.analysis.MEAN)

        A variety of ready-made aggregators are provided in this module, such
        as :data:`~iris.analysis.MEAN` and :data:`~iris.analysis.MAX`.  Custom
        aggregators can also be created for special purposes, see
        :ref:`General-custom_aggregation` for a worked example.

        """
        #: Cube cell method string.
        self.cell_method = cell_method
        #: Data aggregation function.
        self.call_func = call_func
        #: Unit conversion function.
        self.units_func = units_func
        #: Lazy aggregation function, may be None to indicate that a lazy
        #: operation is not available.
        self.lazy_func = lazy_func

        self._kwargs = kwargs

    def lazy_aggregate(self, data, axis, **kwargs):
        """
        Perform aggregation over the data with a lazy operation, analogous to
        the 'aggregate' result.

        Keyword arguments are passed through to the data aggregation function
        (for example, the "percent" keyword for a percentile aggregator).
        This function is usually used in conjunction with update_metadata(),
        which should be passed the same keyword arguments.

        Args:

        * data (array):
            A lazy array (:class:`dask.array.Array`).

        * axis (int or list of int):
            The dimensions to aggregate over -- note that this is defined
            differently to the 'aggregate' method 'axis' argument, which only
            accepts a single dimension index.

        Kwargs:

        * kwargs:
            All keyword arguments are passed through to the data aggregation
            function.

        Returns:
            A lazy array representing the aggregation operation
            (:class:`dask.array.Array`).

        """
        if self.lazy_func is None:
            msg = "{} aggregator does not support lazy operation."
            raise LazyAggregatorError(msg.format(self.name()))

        # Combine keyword args with `kwargs` taking priority over those
        # provided to __init__.
        kwargs = dict(list(self._kwargs.items()) + list(kwargs.items()))

        return self.lazy_func(data, axis=axis, **kwargs)

    def aggregate(self, data, axis, **kwargs):
        """
        Perform the aggregation function given the data.

        Keyword arguments are passed through to the data aggregation function
        (for example, the "percent" keyword for a percentile aggregator).
        This function is usually used in conjunction with update_metadata(),
        which should be passed the same keyword arguments.

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
            passed through to the data aggregation function.

        Returns:
            The aggregated data.

        """
        kwargs = dict(list(self._kwargs.items()) + list(kwargs.items()))
        mdtol = kwargs.pop("mdtol", None)

        result = self.call_func(data, axis=axis, **kwargs)
        if mdtol is not None and ma.isMaskedArray(data):
            fraction_not_missing = data.count(axis=axis) / data.shape[axis]
            mask_update = 1 - mdtol > fraction_not_missing
            if ma.isMaskedArray(result):
                result.mask = result.mask | mask_update
            else:
                result = ma.array(result, mask=mask_update)

        return result

    def update_metadata(self, cube, coords, **kwargs):
        """
        Update common cube metadata w.r.t the aggregation function.

        Args:

        * cube (:class:`iris.cube.Cube`):
            Source cube that requires metadata update.
        * coords (:class:`iris.coords.Coord`):
            The one or more coordinates that were aggregated.

        Kwargs:

        * This function is intended to be used in conjuction with aggregate()
          and should be passed the same keywords (for example, the "ddof"
          keyword for a standard deviation aggregator).

        """
        # Update the units if required.
        if self.units_func is not None:
            cube.units = self.units_func(cube.units)

    def post_process(self, collapsed_cube, data_result, coords, **kwargs):
        """
        Process the result from :func:`iris.analysis.Aggregator.aggregate`.

        Args:

        * collapsed_cube:
            A :class:`iris.cube.Cube`.
        * data_result:
            Result from :func:`iris.analysis.Aggregator.aggregate`
        * coords:
            The one or more coordinates that were aggregated over.

        Kwargs:

        * This function is intended to be used in conjunction with aggregate()
          and should be passed the same keywords (for example, the "ddof"
          keyword from a standard deviation aggregator).

        Returns:
            The collapsed cube with its aggregated data payload.

        """
        collapsed_cube.data = data_result
        return collapsed_cube

    def aggregate_shape(self, **kwargs):
        """
        The shape of the new dimension/s created by the aggregator.

        Kwargs:

        * This function is intended to be used in conjunction with aggregate()
          and should be passed the same keywords.

        Returns:
            A tuple of the aggregate shape.

        """
        return ()

    def name(self):
        """
        Returns the name of the aggregator.

        """
        try:
            name = "_".join(self.cell_method.split())
        except AttributeError:
            name = "unknown"
        return name


class PercentileAggregator(_Aggregator):
    """
    The :class:`PercentileAggregator` class provides percentile aggregation
    functionality.

    This aggregator *may* introduce a new dimension to the data for the
    statistic being calculated, but only if more than one quantile is required.
    For example, calculating the 50th and 90th percentile will result in a new
    data dimension with an extent of 2, for each of the quantiles calculated.

    """

    def __init__(self, units_func=None, lazy_func=None, **kwargs):
        """
        Create a percentile aggregator.

        Kwargs:

        * units_func (callable):
            | *Call signature*: (units)

            If provided, called to convert a cube's units.
            Returns an :class:`cf_units.Unit`, or a
            value that can be made into one.

        * lazy_func (callable or None):
            An alternative to :data:`call_func` implementing a lazy
            aggregation. Note that, it need not support all features of the
            main operation, but should raise an error in unhandled cases.

        Additional kwargs::
            Passed through to :data:`call_func` and :data:`lazy_func`.

        This aggregator can used by cube aggregation methods such as
        :meth:`~iris.cube.Cube.collapsed` and
        :meth:`~iris.cube.Cube.aggregated_by`.  For example::

            cube.collapsed('longitude', iris.analysis.PERCENTILE, percent=50)

        """
        self._name = "percentile"
        self._args = ["percent"]
        _Aggregator.__init__(
            self,
            None,
            _percentile,
            units_func=units_func,
            lazy_func=lazy_func,
            **kwargs,
        )

    def aggregate(self, data, axis, **kwargs):
        """
        Perform the percentile aggregation over the given data.

        Keyword arguments are passed through to the data aggregation function
        (for example, the "percent" keyword for a percentile aggregator).
        This function is usually used in conjunction with update_metadata(),
        which should be passed the same keyword arguments.

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
            passed through to the data aggregation function.

        Returns:
            The aggregated data.

        """

        msg = "{} aggregator requires the mandatory keyword argument {!r}."
        for arg in self._args:
            if arg not in kwargs:
                raise ValueError(msg.format(self.name(), arg))

        return _Aggregator.aggregate(self, data, axis, **kwargs)

    def post_process(self, collapsed_cube, data_result, coords, **kwargs):
        """
        Process the result from :func:`iris.analysis.Aggregator.aggregate`.

        Args:

        * collapsed_cube:
            A :class:`iris.cube.Cube`.
        * data_result:
            Result from :func:`iris.analysis.Aggregator.aggregate`
        * coords:
            The one or more coordinates that were aggregated over.

        Kwargs:

        * This function is intended to be used in conjunction with aggregate()
          and should be passed the same keywords (for example, the "percent"
          keywords from a percentile aggregator).

        Returns:
            The collapsed cube with it's aggregated data payload.

        """
        cubes = iris.cube.CubeList()
        # The additive aggregator requires a mandatory keyword.
        msg = "{} aggregator requires the mandatory keyword argument {!r}."
        for arg in self._args:
            if arg not in kwargs:
                raise ValueError(msg.format(self.name(), arg))

        points = kwargs[self._args[0]]
        # Derive the name of the additive coordinate.
        names = [coord.name() for coord in coords]
        coord_name = "{}_over_{}".format(self.name(), "_".join(names))

        if not isinstance(points, Iterable):
            points = [points]

        # Decorate a collapsed cube with a scalar additive coordinate
        # for each of the additive points, to result in a possibly higher
        # order cube.
        for point in points:
            cube = collapsed_cube.copy()
            coord = iris.coords.AuxCoord(
                point, long_name=coord_name, units="percent"
            )
            cube.add_aux_coord(coord)
            cubes.append(cube)

        collapsed_cube = cubes.merge_cube()

        # Ensure to roll the data payload additive dimension, which should
        # be the last dimension for an additive operation with more than
        # one point, to be the first dimension, thus matching the collapsed
        # cube.
        if self.aggregate_shape(**kwargs):
            # Roll the last additive dimension to be the first.
            data_result = np.rollaxis(data_result, -1)

        # Marry the collapsed cube and the data payload together.
        result = _Aggregator.post_process(
            self, collapsed_cube, data_result, coords, **kwargs
        )
        return result

    def aggregate_shape(self, **kwargs):
        """
        The shape of the additive dimension created by the aggregator.

        Kwargs:

        * This function is intended to be used in conjunction with aggregate()
          and should be passed the same keywords.

        Returns:
            A tuple of the additive dimension shape.

        """

        msg = "{} aggregator requires the mandatory keyword argument {!r}."
        for arg in self._args:
            if arg not in kwargs:
                raise ValueError(msg.format(self.name(), arg))

        points = kwargs[self._args[0]]
        shape = ()

        if not isinstance(points, Iterable):
            points = [points]

        points = np.array(points)

        if points.shape > (1,):
            shape = points.shape

        return shape

    def name(self):
        """
        Returns the name of the aggregator.

        """
        return self._name


class WeightedPercentileAggregator(PercentileAggregator):
    """
    The :class:`WeightedPercentileAggregator` class provides percentile
    aggregation functionality.

    This aggregator *may* introduce a new dimension to the data for the
    statistic being calculated, but only if more than one quantile is required.
    For example, calculating the 50th and 90th percentile will result in a new
    data dimension with an extent of 2, for each of the quantiles calculated.

    """

    def __init__(self, units_func=None, lazy_func=None, **kwargs):
        """
        Create a weighted percentile aggregator.

        Kwargs:

        * units_func (callable):
            | *Call signature*: (units)

            If provided, called to convert a cube's units.
            Returns an :class:`cf_units.Unit`, or a
            value that can be made into one.

        * lazy_func (callable or None):
            An alternative to :data:`call_func` implementing a lazy
            aggregation. Note that, it need not support all features of the
            main operation, but should raise an error in unhandled cases.

        Additional kwargs::
            Passed through to :data:`call_func` and :data:`lazy_func`.

        This aggregator can used by cube aggregation methods such as
        :meth:`~iris.cube.Cube.collapsed` and
        :meth:`~iris.cube.Cube.aggregated_by`.  For example::

            cube.collapsed('longitude', iris.analysis.WPERCENTILE, percent=50,
                             weights=iris.analysis.cartography.area_weights(cube))

        """
        _Aggregator.__init__(
            self,
            None,
            _weighted_percentile,
            units_func=units_func,
            lazy_func=lazy_func,
            **kwargs,
        )

        self._name = "weighted_percentile"
        self._args = ["percent", "weights"]

        #: A list of keywords associated with weighted behaviour.
        self._weighting_keywords = ["returned", "weights"]

    def post_process(self, collapsed_cube, data_result, coords, **kwargs):
        """
        Process the result from :func:`iris.analysis.Aggregator.aggregate`.

        Returns a tuple(cube, weights) if a tuple(data, weights) was returned
        from :func:`iris.analysis.Aggregator.aggregate`.

        Args:

        * collapsed_cube:
            A :class:`iris.cube.Cube`.
        * data_result:
            Result from :func:`iris.analysis.Aggregator.aggregate`
        * coords:
            The one or more coordinates that were aggregated over.

        Kwargs:

        * This function is intended to be used in conjunction with aggregate()
          and should be passed the same keywords (for example, the "weights"
          keyword).

        Returns:
            The collapsed cube with it's aggregated data payload. Or a tuple
            pair of (cube, weights) if the keyword "returned" is specified
            and True.

        """
        if kwargs.get("returned", False):
            # Package the data into the cube and return a tuple
            collapsed_cube = PercentileAggregator.post_process(
                self, collapsed_cube, data_result[0], coords, **kwargs
            )

            result = (collapsed_cube, data_result[1])
        else:
            result = PercentileAggregator.post_process(
                self, collapsed_cube, data_result, coords, **kwargs
            )

        return result


class Aggregator(_Aggregator):
    """
    The :class:`Aggregator` class provides common aggregation functionality.

    """

    def update_metadata(self, cube, coords, **kwargs):
        """
        Update cube cell method metadata w.r.t the aggregation function.

        Args:

        * cube (:class:`iris.cube.Cube`):
            Source cube that requires metadata update.
        * coords (:class:`iris.coords.Coord`):
            The one or more coordinates that were aggregated.

        Kwargs:

        * This function is intended to be used in conjuction with aggregate()
          and should be passed the same keywords (for example, the "ddof"
          keyword for a standard deviation aggregator).

        """
        _Aggregator.update_metadata(self, cube, coords, **kwargs)

        kwargs = dict(list(self._kwargs.items()) + list(kwargs.items()))

        if not isinstance(coords, (list, tuple)):
            coords = [coords]

        coord_names = []
        for coord in coords:
            if not isinstance(coord, iris.coords.Coord):
                raise TypeError(
                    "Coordinate instance expected to the " "Aggregator object."
                )
            coord_names.append(coord.name())

        # Add a cell method.
        method_name = self.cell_method.format(**kwargs)
        cell_method = iris.coords.CellMethod(method_name, coord_names)
        cube.add_cell_method(cell_method)


class WeightedAggregator(Aggregator):
    """
    Convenience class that supports common weighted aggregation functionality.

    """

    def __init__(
        self, cell_method, call_func, units_func=None, lazy_func=None, **kwargs
    ):
        """
        Create a weighted aggregator for the given :data:`call_func`.

        Args:

        * cell_method (string):
            Cell method string that supports string format substitution.

        * call_func (callable):
            Data aggregation function. Call signature `(data, axis, **kwargs)`.

        Kwargs:

        * units_func (callable):
            Units conversion function.

        * lazy_func (callable or None):
            An alternative to :data:`call_func` implementing a lazy
            aggregation. Note that, it need not support all features of the
            main operation, but should raise an error in unhandled cases.

        Additional kwargs:
            Passed through to :data:`call_func` and :data:`lazy_func`.

        """
        Aggregator.__init__(
            self,
            cell_method,
            call_func,
            units_func=units_func,
            lazy_func=lazy_func,
            **kwargs,
        )

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

    def post_process(self, collapsed_cube, data_result, coords, **kwargs):
        """
        Process the result from :func:`iris.analysis.Aggregator.aggregate`.

        Returns a tuple(cube, weights) if a tuple(data, weights) was returned
        from :func:`iris.analysis.Aggregator.aggregate`.

        Args:

        * collapsed_cube:
            A :class:`iris.cube.Cube`.
        * data_result:
            Result from :func:`iris.analysis.Aggregator.aggregate`
        * coords:
            The one or more coordinates that were aggregated over.

        Kwargs:

        * This function is intended to be used in conjunction with aggregate()
          and should be passed the same keywords (for example, the "weights"
          keywords from a mean aggregator).

        Returns:
            The collapsed cube with it's aggregated data payload. Or a tuple
            pair of (cube, weights) if the keyword "returned" is specified
            and True.

        """
        if kwargs.get("returned", False):
            # Package the data into the cube and return a tuple
            collapsed_cube.data, collapsed_weights = data_result
            result = (collapsed_cube, collapsed_weights)
        else:
            result = Aggregator.post_process(
                self, collapsed_cube, data_result, coords, **kwargs
            )

        return result


def _build_dask_mdtol_function(dask_stats_function):
    """
    Make a wrapped dask statistic function that supports the 'mdtol' keyword.

    'dask_function' must be a dask statistical function, compatible with the
    call signature : "dask_stats_function(data, axis=axis, **kwargs)".
    It must be masked-data tolerant, i.e. it ignores masked input points and
    performs a calculation on only the unmasked points.
    For example, mean([1, --, 2]) = (1 + 2) / 2 = 1.5.

    The returned value is a new function operating on dask arrays.
    It has the call signature `stat(data, axis=-1, mdtol=None, **kwargs)`.

    """

    @wraps(dask_stats_function)
    def inner_stat(array, axis=-1, mdtol=None, **kwargs):
        # Call the statistic to get the basic result (missing-data tolerant).
        dask_result = dask_stats_function(array, axis=axis, **kwargs)
        if mdtol is None or mdtol >= 1.0:
            result = dask_result
        else:
            # Build a lazy computation to compare the fraction of missing
            # input points at each output point to the 'mdtol' threshold.
            point_mask_counts = da.sum(da.ma.getmaskarray(array), axis=axis)
            points_per_calc = array.size / dask_result.size
            masked_point_fractions = point_mask_counts / points_per_calc
            boolean_mask = masked_point_fractions > mdtol
            # Return an mdtol-masked version of the basic result.
            result = da.ma.masked_array(
                da.ma.getdata(dask_result), boolean_mask
            )
        return result

    return inner_stat


def _percentile(data, axis, percent, fast_percentile_method=False, **kwargs):
    """
    The percentile aggregator is an additive operation. This means that
    it *may* introduce a new dimension to the data for the statistic being
    calculated, but only if more than one percentile point is requested.

    If a new additive dimension is formed, then it will always be the last
    dimension of the resulting percentile data payload.

    Kwargs:

    * fast_percentile_method (boolean) :
        When set to True, uses the numpy.percentiles method as a faster
        alternative to the scipy.mstats.mquantiles method. Does not handle
        masked arrays.

    """
    # Ensure that the target axis is the last dimension.
    data = np.rollaxis(data, axis, start=data.ndim)
    shape = data.shape[:-1]
    # Flatten any leading dimensions.
    if shape:
        data = data.reshape([np.prod(shape), data.shape[-1]])
    # Perform the percentile calculation.
    if fast_percentile_method:
        msg = "Cannot use fast np.percentile method with masked array."
        if ma.is_masked(data):
            raise TypeError(msg)
        result = np.percentile(data, percent, axis=-1)
        result = result.T
    else:
        quantiles = np.array(percent) / 100.0
        result = scipy.stats.mstats.mquantiles(
            data, quantiles, axis=-1, **kwargs
        )
    if not ma.isMaskedArray(data) and not ma.is_masked(result):
        result = np.asarray(result)
    else:
        result = ma.MaskedArray(result)

    # Ensure to unflatten any leading dimensions.
    if shape:
        if not isinstance(percent, Iterable):
            percent = [percent]
        percent = np.array(percent)
        # Account for the additive dimension.
        if percent.shape > (1,):
            shape += percent.shape
        result = result.reshape(shape)
    # Check whether to reduce to a scalar result, as per the behaviour
    # of other aggregators.
    if result.shape == (1,) and quantiles.ndim == 0:
        result = result[0]

    return result


def _weighted_quantile_1D(data, weights, quantiles, **kwargs):
    """
    Compute the weighted quantile of a 1D numpy array.

    Adapted from `wquantiles <https://github.com/nudomarinero/wquantiles/>`_

    Args:

    * data (array)
        One dimensional data array
    * weights (array)
        Array of the same size of `data`.  If data is masked, weights must have
        matching mask.
    * quantiles : (float or sequence of floats)
        Quantile(s) to compute. Must have a value between 0 and 1.

    **kwargs
        passed to `scipy.interpolate.interp1d`

    Returns:
        array or float.  Calculated quantile values (set to np.nan wherever sum
        of weights is zero or masked)
    """
    # Return np.nan if no useable points found
    if np.isclose(weights.sum(), 0.0) or ma.is_masked(weights.sum()):
        return np.resize(np.array(np.nan), len(quantiles))
    # Sort the data
    ind_sorted = ma.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]
    # Compute the auxiliary arrays
    Sn = np.cumsum(sorted_weights)
    Pn = (Sn - 0.5 * sorted_weights) / np.sum(sorted_weights)
    # Get the value of the weighted quantiles
    interpolator = scipy.interpolate.interp1d(
        Pn, sorted_data, bounds_error=False, **kwargs
    )
    result = interpolator(quantiles)
    # Set cases where quantile falls outside data range to min or max
    np.place(result, Pn.min() > quantiles, sorted_data.min())
    np.place(result, Pn.max() < quantiles, sorted_data.max())

    return result


def _weighted_percentile(
    data, axis, weights, percent, returned=False, **kwargs
):
    """
    The weighted_percentile aggregator is an additive operation. This means
    that it *may* introduce a new dimension to the data for the statistic being
    calculated, but only if more than one percentile point is requested.

    If a new additive dimension is formed, then it will always be the last
    dimension of the resulting percentile data payload.

    Args:

    * data: ndarray or masked array

    * axis: int
         axis to calculate percentiles over

    * weights: ndarray
         array with the weights.  Must have same shape as data

    * percent: float or sequence of floats
         Percentile rank/s at which to extract value/s.

    * returned: bool, optional
         Default False.  If True, returns a tuple with the percentiles as the
         first element and the sum of the weights as the second element.

    """
    # Ensure that data and weights arrays are same shape.
    if data.shape != weights.shape:
        raise ValueError("_weighted_percentile: weights wrong shape.")
    # Ensure that the target axis is the last dimension.
    data = np.rollaxis(data, axis, start=data.ndim)
    weights = np.rollaxis(weights, axis, start=data.ndim)
    quantiles = np.array(percent) / 100.0
    # Add data mask to weights if necessary.
    if ma.isMaskedArray(data):
        weights = ma.array(weights, mask=data.mask)
    shape = data.shape[:-1]
    # Flatten any leading dimensions and loop over them
    if shape:
        data = data.reshape([np.prod(shape), data.shape[-1]])
        weights = weights.reshape([np.prod(shape), data.shape[-1]])
        result = np.empty((np.prod(shape), quantiles.size))
        # Perform the percentile calculation.
        for res, dat, wt in zip(result, data, weights):
            res[:] = _weighted_quantile_1D(dat, wt, quantiles, **kwargs)
    else:
        # Data is 1D
        result = _weighted_quantile_1D(data, weights, quantiles, **kwargs)

    if np.any(np.isnan(result)):
        result = ma.masked_invalid(result)

    if not ma.isMaskedArray(data) and not ma.is_masked(result):
        result = np.asarray(result)

    # Ensure to unflatten any leading dimensions.
    if shape:
        if not isinstance(percent, Iterable):
            percent = [percent]
        percent = np.array(percent)
        # Account for the additive dimension.
        if percent.shape > (1,):
            shape += percent.shape
        result = result.reshape(shape)
    # Check whether to reduce to a scalar result, as per the behaviour
    # of other aggregators.
    if result.shape == (1,) and quantiles.ndim == 0:
        result = result[0]

    if returned:
        return result, weights.sum(axis=-1)
    else:
        return result


@_build_dask_mdtol_function
def _lazy_count(array, **kwargs):
    array = iris._lazy_data.as_lazy_data(array)
    func = kwargs.pop("function", None)
    if not callable(func):
        emsg = "function must be a callable. Got {}."
        raise TypeError(emsg.format(type(func)))
    return da.sum(func(array), **kwargs)


def _proportion(array, function, axis, **kwargs):
    count = iris._lazy_data.non_lazy(_lazy_count)
    # if the incoming array is masked use that to count the total number of
    # values
    if ma.isMaskedArray(array):
        # calculate the total number of non-masked values across the given axis
        if array.mask is np.bool_(False):
            # numpy will return a single boolean as a mask if the mask
            # was not explicitly specified on array construction, so in this
            # case pass the array shape instead of the mask:
            total_non_masked = array.shape[axis]
        else:
            total_non_masked = count(
                array.mask, axis=axis, function=np.logical_not, **kwargs
            )
            total_non_masked = ma.masked_equal(total_non_masked, 0)
    else:
        total_non_masked = array.shape[axis]

    # Sanitise the result of this operation thru ma.asarray to ensure that
    # the dtype of the fill-value and the dtype of the array are aligned.
    # Otherwise, it is possible for numpy to return a masked array that has
    # a dtype for its data that is different to the dtype of the fill-value,
    # which can cause issues outside this function.
    # Reference - tests/unit/analyis/test_PROPORTION.py Test_masked.test_ma
    numerator = count(array, axis=axis, function=function, **kwargs)
    result = ma.asarray(numerator / total_non_masked)

    return result


def _rms(array, axis, **kwargs):
    # XXX due to the current limitations in `da.average` (see below), maintain
    # an explicit non-lazy aggregation function for now.
    # Note: retaining this function also means that if weights are passed to
    # the lazy aggregator, the aggregation will fall back to using this
    # non-lazy aggregator.
    rval = np.sqrt(ma.average(np.square(array), axis=axis, **kwargs))
    if not ma.isMaskedArray(array):
        rval = np.asarray(rval)
    return rval


@_build_dask_mdtol_function
def _lazy_rms(array, axis, **kwargs):
    # XXX This should use `da.average` and not `da.mean`, as does the above.
    # However `da.average` current doesn't handle masked weights correctly
    # (see https://github.com/dask/dask/issues/3846).
    # To work around this we use da.mean, which doesn't support weights at
    # all. Thus trying to use this aggregator with weights will currently
    # raise an error in dask due to the unexpected keyword `weights`,
    # rather than silently returning the wrong answer.
    return da.sqrt(da.mean(array ** 2, axis=axis, **kwargs))


@_build_dask_mdtol_function
def _lazy_sum(array, **kwargs):
    array = iris._lazy_data.as_lazy_data(array)
    # weighted or scaled sum
    axis_in = kwargs.get("axis", None)
    weights_in = kwargs.pop("weights", None)
    returned_in = kwargs.pop("returned", False)
    if weights_in is not None:
        wsum = da.sum(weights_in * array, **kwargs)
    else:
        wsum = da.sum(array, **kwargs)
    if returned_in:
        if weights_in is None:
            weights = iris._lazy_data.as_lazy_data(np.ones_like(array))
        else:
            weights = weights_in
        rvalue = (wsum, da.sum(weights, axis=axis_in))
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
                    columns.append(
                        column[nan_indices[index - 1] + 1 : nan_index]
                    )
            if nan_indices[-1] != len(column) - 1:
                columns.append(column[nan_indices[-1] + 1 :])
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
    endslice = slice(0, 1) if len(slices) == 1 else 0
    slices[-1] = endslice
    slices = tuple(slices)  # Numpy>=1.16 : index with tuple, *not* list.

    if isinstance(array.dtype, np.float):
        data = array[slices]
    else:
        # Cast non-float data type.
        data = array.astype("float32")[slices]

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
        equal_slice = (
            np.ones(column_slice.size, dtype=column_slice.dtype)
            * column_slice[0]
        )
        if (
            column_slice.size == 1
            or all(np.isnan(column_slice))
            or ma.count(column_slice) == 0
            or np.all(np.equal(equal_slice, column_slice))
        ):
            continue

        # Check if the column slice is masked.
        if ma.isMaskedArray(column_slice):
            # Check if the column slice contains only nans, without inf
            # or -inf values, regardless of the mask.
            if not np.any(np.isfinite(column_slice)) and not np.any(
                np.isinf(column_slice)
            ):
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

            tck = scipy.interpolate.splrep(np.arange(column.size), column, k=k)
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
COUNT = Aggregator(
    "count",
    iris._lazy_data.non_lazy(_lazy_count),
    units_func=lambda units: 1,
    lazy_func=_lazy_count,
)
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


GMEAN = Aggregator("geometric_mean", scipy.stats.mstats.gmean)
"""
An :class:`~iris.analysis.Aggregator` instance that calculates the
geometric mean over a :class:`~iris.cube.Cube`, as computed by
:func:`scipy.stats.mstats.gmean`.

**For example**:

To compute zonal geometric means over the *longitude* axis of a cube::

    result = cube.collapsed('longitude', iris.analysis.GMEAN)

This aggregator handles masked data.

"""


HMEAN = Aggregator("harmonic_mean", scipy.stats.mstats.hmean)
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


MEAN = WeightedAggregator(
    "mean", ma.average, lazy_func=_build_dask_mdtol_function(da.ma.average)
)
"""
An :class:`~iris.analysis.Aggregator` instance that calculates
the mean over a :class:`~iris.cube.Cube`, as computed by
:func:`numpy.ma.average`.

Additional kwargs associated with the use of this aggregator:

* mdtol (float):
    Tolerance of missing data. The value returned in each element of the
    returned array will be masked if the fraction of masked data contributing
    to that element exceeds mdtol. This fraction is calculated based on the
    number of masked elements. mdtol=0 means no missing data is tolerated
    while mdtol=1 means the resulting element will be masked if and only if
    all the contributing elements are masked. Defaults to 1.
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

    Lazy operation is supported, via :func:`dask.array.ma.average`.

This aggregator handles masked data.

"""


MEDIAN = Aggregator("median", ma.median)
"""
An :class:`~iris.analysis.Aggregator` instance that calculates
the median over a :class:`~iris.cube.Cube`, as computed by
:func:`numpy.ma.median`.

**For example**:

To compute zonal medians over the *longitude* axis of a cube::

    result = cube.collapsed('longitude', iris.analysis.MEDIAN)

This aggregator handles masked data.

"""


MIN = Aggregator(
    "minimum", ma.min, lazy_func=_build_dask_mdtol_function(da.min)
)
"""
An :class:`~iris.analysis.Aggregator` instance that calculates
the minimum over a :class:`~iris.cube.Cube`, as computed by
:func:`numpy.ma.min`.

**For example**:

To compute zonal minimums over the *longitude* axis of a cube::

    result = cube.collapsed('longitude', iris.analysis.MIN)

This aggregator handles masked data.

"""


MAX = Aggregator(
    "maximum", ma.max, lazy_func=_build_dask_mdtol_function(da.max)
)
"""
An :class:`~iris.analysis.Aggregator` instance that calculates
the maximum over a :class:`~iris.cube.Cube`, as computed by
:func:`numpy.ma.max`.

**For example**:

To compute zonal maximums over the *longitude* axis of a cube::

    result = cube.collapsed('longitude', iris.analysis.MAX)

This aggregator handles masked data.

"""


PEAK = Aggregator("peak", _peak)
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


PERCENTILE = PercentileAggregator(alphap=1, betap=1)
"""
An :class:`~iris.analysis.PercentileAggregator` instance that calculates the
percentile over a :class:`~iris.cube.Cube`, as computed by
:func:`scipy.stats.mstats.mquantiles`.

**Required** kwargs associated with the use of this aggregator:

* percent (float or sequence of floats):
    Percentile rank/s at which to extract value/s.

Additional kwargs associated with the use of this aggregator:

* alphap (float):
    Plotting positions parameter, see :func:`scipy.stats.mstats.mquantiles`.
    Defaults to 1.
* betap (float):
    Plotting positions parameter, see :func:`scipy.stats.mstats.mquantiles`.
    Defaults to 1.

**For example**:

To compute the 10th and 90th percentile over *time*::

    result = cube.collapsed('time', iris.analysis.PERCENTILE, percent=[10, 90])

This aggregator handles masked data.

"""


PROPORTION = Aggregator("proportion", _proportion, units_func=lambda units: 1)
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


RMS = WeightedAggregator(
    "root mean square", _rms, lazy_func=_build_dask_mdtol_function(_lazy_rms)
)
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


STD_DEV = Aggregator(
    "standard_deviation",
    ma.std,
    ddof=1,
    lazy_func=_build_dask_mdtol_function(da.std),
)
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

.. note::

    Lazy operation is supported, via :func:`dask.array.nanstd`.

This aggregator handles masked data.

"""


SUM = WeightedAggregator(
    "sum",
    iris._lazy_data.non_lazy(_lazy_sum),
    lazy_func=_build_dask_mdtol_function(_lazy_sum),
)
"""
An :class:`~iris.analysis.Aggregator` instance that calculates
the sum over a :class:`~iris.cube.Cube`, as computed by :func:`numpy.ma.sum`.

Additional kwargs associated with the use of this aggregator:

* weights (float ndarray):
    Weights matching the shape of the cube, or the length of
    the window for rolling window operations. Weights should be
    normalized before using them with this aggregator if scaling
    is not intended.
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


VARIANCE = Aggregator(
    "variance",
    ma.var,
    units_func=lambda units: units * units,
    lazy_func=_build_dask_mdtol_function(da.var),
    ddof=1,
)
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

    Lazy operation is supported, via :func:`dask.array.nanvar`.

This aggregator handles masked data.

"""


WPERCENTILE = WeightedPercentileAggregator()
"""
An :class:`~iris.analysis.WeightedPercentileAggregator` instance that
calculates the weighted percentile over a :class:`~iris.cube.Cube`.

**Required** kwargs associated with the use of this aggregator:

* percent (float or sequence of floats):
    Percentile rank/s at which to extract value/s.

* weights (float ndarray):
    Weights matching the shape of the cube or the length of the window
    for rolling window operations. Note that, latitude/longitude area
    weights can be calculated using
    :func:`iris.analysis.cartography.area_weights`.

Additional kwargs associated with the use of this aggregator:

* returned (boolean):
    Set this to True to indicate that the collapsed weights are to be
    returned along with the collapsed data. Defaults to False.

* kind (string or int):
    Specifies the kind of interpolation used, see
    :func:`scipy.interpolate.interp1d` Defaults to "linear", which is
    equivalent to alphap=0.5, betap=0.5 in `iris.analysis.PERCENTILE`

"""


class _Groupby:
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

        * shared_coords (list of (:class:`iris.coords.Coord`, `int`) pairs):
            One or more coordinates (including multidimensional coordinates)
            that share the same group-by coordinate axis.  The `int` identifies
            which dimension of the coord is on the group-by coordinate axis.

        """
        #: Group-by and shared coordinates that have been grouped.
        self.coords = []
        self._groupby_coords = []
        self._shared_coords = []
        self._slices_by_key = OrderedDict()
        self._stop = None
        # Ensure group-by coordinates are iterable.
        if not isinstance(groupby_coords, Iterable):
            raise TypeError(
                "groupby_coords must be a " "`collections.Iterable` type."
            )

        # Add valid group-by coordinates.
        for coord in groupby_coords:
            self._add_groupby_coord(coord)
        # Add the coordinates sharing the same axis as the group-by
        # coordinates.
        if shared_coords is not None:
            # Ensure shared coordinates are iterable.
            if not isinstance(shared_coords, Iterable):
                raise TypeError(
                    "shared_coords must be a " "`collections.Iterable` type."
                )
            # Add valid shared coordinates.
            for coord, dim in shared_coords:
                self._add_shared_coord(coord, dim)

    def _add_groupby_coord(self, coord):
        if coord.ndim != 1:
            raise iris.exceptions.CoordinateMultiDimError(coord)
        if self._stop is None:
            self._stop = coord.shape[0]
        if coord.shape[0] != self._stop:
            raise ValueError("Group-by coordinates have different lengths.")
        self._groupby_coords.append(coord)

    def _add_shared_coord(self, coord, dim):
        if coord.shape[dim] != self._stop and self._stop is not None:
            raise ValueError("Shared coordinates have different lengths.")
        self._shared_coords.append((coord, dim))

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
                    items.append(next(groups[-1]))

                # Construct the group slice for each group over the group-by
                # coordinates. Keep constructing until all group-by coordinate
                # groups are exhausted.
                while any([item is not None for item in items]):
                    # Determine the extent (start, stop) of the group given
                    # each current group-by coordinate group.
                    start = max(
                        [
                            item.groupby_slice.start
                            for item in items
                            if item is not None
                        ]
                    )
                    stop = min(
                        [
                            item.groupby_slice.stop
                            for item in items
                            if item is not None
                        ]
                    )
                    # Construct composite group key for the group using the
                    # start value from each group-by coordinate.
                    key = tuple(
                        [coord.points[start] for coord in self._groupby_coords]
                    )
                    # Associate group slice with group key within the ordered
                    # dictionary.
                    self._slices_by_key.setdefault(key, []).append(
                        slice(start, stop)
                    )
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
                            items[item_index] = next(groups[item_index])

                # Merge multiple slices together into one tuple.
                self._slice_merge()
                # Calculate the new group-by coordinates.
                self._compute_groupby_coords()
                # Calculate the new shared coordinates.
                self._compute_shared_coords()
            # Generate the group-by slices/groups.
            for groupby_slice in self._slices_by_key.values():
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
        for key, groupby_slices in self._slices_by_key.items():
            if len(groupby_slices) > 1:
                # Compress multiple slices into tuple representation.
                groupby_indicies = []

                for groupby_slice in groupby_slices:
                    groupby_indicies.extend(
                        range(groupby_slice.start, groupby_slice.stop)
                    )

                self._slices_by_key[key] = tuple(groupby_indicies)
            else:
                # Remove single inner slice from list.
                self._slices_by_key[key] = groupby_slices[0]

    def _compute_groupby_coords(self):
        """Create new group-by coordinates given the group slices."""

        groupby_slice = []

        # Iterate over the ordered dictionary in order to construct
        # a group-by slice that samples the first element from each group.
        for key_slice in self._slices_by_key.values():
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
        for key_slice in self._slices_by_key.values():
            if isinstance(key_slice, tuple):
                groupby_bounds.append((key_slice[0], key_slice[-1]))
            else:
                groupby_bounds.append((key_slice.start, key_slice.stop - 1))

        # Create new shared bounded coordinates.
        for coord, dim in self._shared_coords:
            if coord.points.dtype.kind in "SU":
                if coord.bounds is None:
                    new_points = []
                    new_bounds = None
                    # np.apply_along_axis does not work with str.join, so we
                    # need to loop through the array directly. First move axis
                    # of interest to trailing dim and flatten the others.
                    work_arr = np.moveaxis(coord.points, dim, -1)
                    shape = work_arr.shape
                    work_shape = (-1, shape[-1])
                    new_shape = (len(self),)
                    if coord.ndim > 1:
                        new_shape += shape[:-1]
                    work_arr = work_arr.reshape(work_shape)

                    for key_slice in self._slices_by_key.values():
                        if isinstance(key_slice, slice):
                            indices = key_slice.indices(
                                coord.points.shape[dim]
                            )
                            key_slice = range(*indices)

                        for arr in work_arr:
                            new_points.append("|".join(arr.take(key_slice)))

                    # Reinstate flattened dimensions. Aggregated dim now leads.
                    new_points = np.array(new_points).reshape(new_shape)

                    # Move aggregated dimension back to position it started in.
                    new_points = np.moveaxis(new_points, 0, dim)
                else:
                    msg = (
                        "collapsing the bounded string coordinate {0!r}"
                        " is not supported".format(coord.name())
                    )
                    raise ValueError(msg)
            else:
                new_bounds = []

                # Construct list of coordinate group boundary pairs.
                for start, stop in groupby_bounds:
                    if coord.has_bounds():
                        # Collapse group bounds into bounds.
                        if (
                            getattr(coord, "circular", False)
                            and (stop + 1) == coord.shape[dim]
                        ):
                            new_bounds.append(
                                [
                                    coord.bounds.take(start, dim).take(0, -1),
                                    coord.bounds.take(0, dim).take(0, -1)
                                    + coord.units.modulus,
                                ]
                            )
                        else:
                            new_bounds.append(
                                [
                                    coord.bounds.take(start, dim).take(0, -1),
                                    coord.bounds.take(stop, dim).take(1, -1),
                                ]
                            )
                    else:
                        # Collapse group points into bounds.
                        if getattr(coord, "circular", False) and (
                            stop + 1
                        ) == len(coord.points):
                            new_bounds.append(
                                [
                                    coord.points.take(start, dim),
                                    coord.points.take(0, dim)
                                    + coord.units.modulus,
                                ]
                            )
                        else:
                            new_bounds.append(
                                [
                                    coord.points.take(start, dim),
                                    coord.points.take(stop, dim),
                                ]
                            )

                # Bounds needs to be an array with the length 2 start-stop
                # dimension last, and the aggregated dimension back in its
                # original position.
                new_bounds = np.moveaxis(
                    np.array(new_bounds), (0, 1), (dim, -1)
                )

                # Now create the new bounded group shared coordinate.
                try:
                    new_points = new_bounds.mean(-1)
                except TypeError:
                    msg = (
                        "The {0!r} coordinate on the collapsing dimension"
                        " cannot be collapsed.".format(coord.name())
                    )
                    raise ValueError(msg)

            try:
                self.coords.append(
                    coord.copy(points=new_points, bounds=new_bounds)
                )
            except ValueError:
                # non monotonic points/bounds
                self.coords.append(
                    iris.coords.AuxCoord.from_coord(coord).copy(
                        points=new_points, bounds=new_bounds
                    )
                )

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
            shared_string = ", shared_coords=%r)" % shared_coords
        else:
            shared_string = ")"

        return "%s(%r%s" % (
            self.__class__.__name__,
            groupby_coords,
            shared_string,
        )


def clear_phenomenon_identity(cube):
    """
    Helper function to clear the standard_name, attributes, and
    cell_methods of a cube.

    """
    cube.rename(None)
    cube.attributes.clear()
    cube.cell_methods = tuple()


###############################################################################
#
# Interpolation API
#
###############################################################################


class Linear:
    """
    This class describes the linear interpolation and regridding scheme for
    interpolating or regridding over one or more orthogonal coordinates,
    typically for use with :meth:`iris.cube.Cube.interpolate()` or
    :meth:`iris.cube.Cube.regrid()`.

    """

    LINEAR_EXTRAPOLATION_MODES = list(EXTRAPOLATION_MODES.keys()) + ["linear"]

    def __init__(self, extrapolation_mode="linear"):
        """
        Linear interpolation and regridding scheme suitable for interpolating
        or regridding over one or more orthogonal coordinates.

        Kwargs:

        * extrapolation_mode:
            Must be one of the following strings:

              * 'extrapolate' or 'linear' - The extrapolation points
                will be calculated by extending the gradient of the
                closest two points.
              * 'nan' - The extrapolation points will be be set to NaN.
              * 'error' - A ValueError exception will be raised, notifying an
                attempt to extrapolate.
              * 'mask' - The extrapolation points will always be masked, even
                if the source data is not a MaskedArray.
              * 'nanmask' - If the source data is a MaskedArray the
                extrapolation points will be masked. Otherwise they will be
                set to NaN.

            The default mode of extrapolation is 'linear'.

        """
        if extrapolation_mode not in self.LINEAR_EXTRAPOLATION_MODES:
            msg = "Extrapolation mode {!r} not supported."
            raise ValueError(msg.format(extrapolation_mode))
        self.extrapolation_mode = extrapolation_mode

    def __repr__(self):
        return "Linear({!r})".format(self.extrapolation_mode)

    def _normalised_extrapolation_mode(self):
        mode = self.extrapolation_mode
        if mode == "linear":
            mode = "extrapolate"
        return mode

    def interpolator(self, cube, coords):
        """
        Creates a linear interpolator to perform interpolation over the
        given :class:`~iris.cube.Cube` specified by the dimensions of
        the given coordinates.

        Typically you should use :meth:`iris.cube.Cube.interpolate` for
        interpolating a cube. There are, however, some situations when
        constructing your own interpolator is preferable. These are detailed
        in the :ref:`user guide <caching_an_interpolator>`.

        Args:

        * cube:
            The source :class:`iris.cube.Cube` to be interpolated.
        * coords:
            The names or coordinate instances that are to be
            interpolated over.

        Returns:
            A callable with the interface:

                `callable(sample_points, collapse_scalar=True)`

            where `sample_points` is a sequence containing an array of values
            for each of the coordinates passed to this method, and
            `collapse_scalar` determines whether to remove length one
            dimensions in the result cube caused by scalar values in
            `sample_points`.

            The values for coordinates that correspond to date/times
            may optionally be supplied as datetime.datetime or
            cftime.datetime instances.

            For example, for the callable returned by:
            `Linear().interpolator(cube, ['latitude', 'longitude'])`,
            sample_points must have the form
            `[new_lat_values, new_lon_values]`.

        """
        return RectilinearInterpolator(
            cube, coords, "linear", self._normalised_extrapolation_mode()
        )

    def regridder(self, src_grid, target_grid):
        """
        Creates a linear regridder to perform regridding from the source
        grid to the target grid.

        Typically you should use :meth:`iris.cube.Cube.regrid` for
        regridding a cube. There are, however, some situations when
        constructing your own regridder is preferable. These are detailed in
        the :ref:`user guide <caching_a_regridder>`.

        Args:

        * src_grid:
            The :class:`~iris.cube.Cube` defining the source grid.
        * target_grid:
            The :class:`~iris.cube.Cube` defining the target grid.

        Returns:
            A callable with the interface:

                `callable(cube)`

            where `cube` is a cube with the same grid as `src_grid`
            that is to be regridded to the `target_grid`.

        """
        return RectilinearRegridder(
            src_grid,
            target_grid,
            "linear",
            self._normalised_extrapolation_mode(),
        )


class AreaWeighted:
    """
    This class describes an area-weighted regridding scheme for regridding
    between 'ordinary' horizontal grids with separated X and Y coordinates in a
    common coordinate system.
    Typically for use with :meth:`iris.cube.Cube.regrid()`.

    """

    def __init__(self, mdtol=1):
        """
        Area-weighted regridding scheme suitable for regridding between
        different orthogonal XY grids in the same coordinate system.

        Kwargs:

        * mdtol (float):
            Tolerance of missing data. The value returned in each element of
            the returned array will be masked if the fraction of missing data
            exceeds mdtol. This fraction is calculated based on the area of
            masked cells within each target cell. mdtol=0 means no masked
            data is tolerated while mdtol=1 will mean the resulting element
            will be masked if and only if all the overlapping elements of the
            source grid are masked. Defaults to 1.

        .. Note:
            Both sourge and target cubes must have an XY grid defined by
            separate X and Y dimensions with dimension coordinates.
            All of the XY dimension coordinates must also be bounded, and have
            the same cooordinate system.

        """
        if not (0 <= mdtol <= 1):
            msg = "Value for mdtol must be in range 0 - 1, got {}."
            raise ValueError(msg.format(mdtol))
        self.mdtol = mdtol

    def __repr__(self):
        return "AreaWeighted(mdtol={})".format(self.mdtol)

    def regridder(self, src_grid_cube, target_grid_cube):
        """
        Creates an area-weighted regridder to perform regridding from the
        source grid to the target grid.

        Typically you should use :meth:`iris.cube.Cube.regrid` for
        regridding a cube. There are, however, some situations when
        constructing your own regridder is preferable. These are detailed in
        the :ref:`user guide <caching_a_regridder>`.

        Args:

        * src_grid_cube:
            The :class:`~iris.cube.Cube` defining the source grid.
        * target_grid_cube:
            The :class:`~iris.cube.Cube` defining the target grid.

        Returns:
            A callable with the interface:

                `callable(cube)`

            where `cube` is a cube with the same grid as `src_grid_cube`
            that is to be regridded to the grid of `target_grid_cube`.

        """
        return AreaWeightedRegridder(
            src_grid_cube, target_grid_cube, mdtol=self.mdtol
        )


class Nearest:
    """
    This class describes the nearest-neighbour interpolation and regridding
    scheme for interpolating or regridding over one or more orthogonal
    coordinates, typically for use with :meth:`iris.cube.Cube.interpolate()`
    or :meth:`iris.cube.Cube.regrid()`.

    """

    def __init__(self, extrapolation_mode="extrapolate"):
        """
        Nearest-neighbour interpolation and regridding scheme suitable for
        interpolating or regridding over one or more orthogonal coordinates.

        Kwargs:

        * extrapolation_mode:
            Must be one of the following strings:

              * 'extrapolate' - The extrapolation points will take their
                value from the nearest source point.
              * 'nan' - The extrapolation points will be be set to NaN.
              * 'error' - A ValueError exception will be raised, notifying an
                attempt to extrapolate.
              * 'mask' - The extrapolation points will always be masked, even
                if the source data is not a MaskedArray.
              * 'nanmask' - If the source data is a MaskedArray the
                extrapolation points will be masked. Otherwise they will be
                set to NaN.

            The default mode of extrapolation is 'extrapolate'.

        """
        if extrapolation_mode not in EXTRAPOLATION_MODES:
            msg = "Extrapolation mode {!r} not supported."
            raise ValueError(msg.format(extrapolation_mode))
        self.extrapolation_mode = extrapolation_mode

    def __repr__(self):
        return "Nearest({!r})".format(self.extrapolation_mode)

    def interpolator(self, cube, coords):
        """
        Creates a nearest-neighbour interpolator to perform
        interpolation over the given :class:`~iris.cube.Cube` specified
        by the dimensions of the specified coordinates.

        Typically you should use :meth:`iris.cube.Cube.interpolate` for
        interpolating a cube. There are, however, some situations when
        constructing your own interpolator is preferable. These are detailed
        in the :ref:`user guide <caching_an_interpolator>`.

        Args:

        * cube:
            The source :class:`iris.cube.Cube` to be interpolated.
        * coords:
            The names or coordinate instances that are to be
            interpolated over.

        Returns:
            A callable with the interface:

                `callable(sample_points, collapse_scalar=True)`

            where `sample_points` is a sequence containing an array of values
            for each of the coordinates passed to this method, and
            `collapse_scalar` determines whether to remove length one
            dimensions in the result cube caused by scalar values in
            `sample_points`.

            The values for coordinates that correspond to date/times
            may optionally be supplied as datetime.datetime or
            cftime.datetime instances.

            For example, for the callable returned by:
            `Nearest().interpolator(cube, ['latitude', 'longitude'])`,
            sample_points must have the form
            `[new_lat_values, new_lon_values]`.

        """
        return RectilinearInterpolator(
            cube, coords, "nearest", self.extrapolation_mode
        )

    def regridder(self, src_grid, target_grid):
        """
        Creates a nearest-neighbour regridder to perform regridding from the
        source grid to the target grid.

        Typically you should use :meth:`iris.cube.Cube.regrid` for
        regridding a cube. There are, however, some situations when
        constructing your own regridder is preferable. These are detailed in
        the :ref:`user guide <caching_a_regridder>`.

        Args:

        * src_grid:
            The :class:`~iris.cube.Cube` defining the source grid.
        * target_grid:
            The :class:`~iris.cube.Cube` defining the target grid.

        Returns:
            A callable with the interface:

                `callable(cube)`

            where `cube` is a cube with the same grid as `src_grid`
            that is to be regridded to the `target_grid`.

        """
        return RectilinearRegridder(
            src_grid, target_grid, "nearest", self.extrapolation_mode
        )


class UnstructuredNearest:
    """
    This is a nearest-neighbour regridding scheme for regridding data whose
    horizontal (X- and Y-axis) coordinates are mapped to the *same* dimensions,
    rather than being orthogonal on independent dimensions.

    For latitude-longitude coordinates, the nearest-neighbour distances are
    computed on the sphere, otherwise flat Euclidean distances are used.

    The source X and Y coordinates can have any shape.

    The target grid must be of the "normal" kind, i.e. it has separate,
    1-dimensional X and Y coordinates.

    Source and target XY coordinates must have the same coordinate system,
    which may also be None.
    If any of the XY coordinates are latitudes or longitudes, then they *all*
    must be.  Otherwise, the corresponding X and Y coordinates must have the
    same units in the source and grid cubes.

    .. Note::
        Currently only supports regridding, not interpolation.

    .. Note::
          This scheme performs essentially the same job as
          :class:`iris.experimental.regrid.ProjectedUnstructuredNearest`.
          That scheme is faster, but only works well on data in a limited
          region of the globe, covered by a specified projection.
          This approach is more rigorously correct and can be applied to global
          datasets.

    """

    # Note: the argument requirements are simply those of the underlying
    # regridder class,
    # :class:`iris.analysis.trajectory.UnstructuredNearestNeigbourRegridder`.
    def __init__(self):
        """
        Nearest-neighbour interpolation and regridding scheme suitable for
        interpolating or regridding from un-gridded data such as trajectories
        or other data where the X and Y coordinates share the same dimensions.

        """
        pass

    def __repr__(self):
        return "UnstructuredNearest()"

    # TODO: add interpolator usage
    # def interpolator(self, cube):

    def regridder(self, src_cube, target_grid):
        """
        Creates a nearest-neighbour regridder, of the
        :class:`~iris.analysis.trajectory.UnstructuredNearestNeigbourRegridder`
        type, to perform regridding from the source grid to the target grid.

        This can then be applied to any source data with the same structure as
        the original 'src_cube'.

        Typically you should use :meth:`iris.cube.Cube.regrid` for
        regridding a cube. There are, however, some situations when
        constructing your own regridder is preferable. These are detailed in
        the :ref:`user guide <caching_a_regridder>`.

        Args:

        * src_cube:
            The :class:`~iris.cube.Cube` defining the source grid.
            The X and Y coordinates can have any shape, but must be mapped over
            the same cube dimensions.

        * target_grid:
            The :class:`~iris.cube.Cube` defining the target grid.
            The X and Y coordinates must be one-dimensional dimension
            coordinates, mapped to different dimensions.
            All other cube components are ignored.

        Returns:
            A callable with the interface:

                `callable(cube)`

            where `cube` is a cube with the same grid as `src_cube`
            that is to be regridded to the `target_grid`.

        """
        from iris.analysis.trajectory import (
            UnstructuredNearestNeigbourRegridder,
        )

        return UnstructuredNearestNeigbourRegridder(src_cube, target_grid)


class PointInCell:
    """
    This class describes the point-in-cell regridding scheme for use
    typically with :meth:`iris.cube.Cube.regrid()`.

    The PointInCell regridder can regrid data from a source grid of any
    dimensionality and in any coordinate system.
    The location of each source point is specified by X and Y coordinates
    mapped over the same cube dimensions, aka "grid dimensions" : the grid may
    have any dimensionality.  The X and Y coordinates must also have the same,
    defined coord_system.
    The weights, if specified, must have the same shape as the X and Y
    coordinates.
    The output grid can be any 'normal' XY grid, specified by *separate* X
    and Y coordinates :  That is, X and Y have two different cube dimensions.
    The output X and Y coordinates must also have a common, specified
    coord_system.

    """

    def __init__(self, weights=None):
        """
        Point-in-cell regridding scheme suitable for regridding over one
        or more orthogonal coordinates.

        Optional Args:

        * weights:
            A :class:`numpy.ndarray` instance that defines the weights
            for the grid cells of the source grid. Must have the same shape
            as the data of the source grid.
            If unspecified, equal weighting is assumed.

        """
        self.weights = weights

    def regridder(self, src_grid, target_grid):
        """
        Creates a point-in-cell regridder to perform regridding from the
        source grid to the target grid.

        Typically you should use :meth:`iris.cube.Cube.regrid` for
        regridding a cube. There are, however, some situations when
        constructing your own regridder is preferable. These are detailed in
        the :ref:`user guide <caching_a_regridder>`.

        Args:

        * src_grid:
            The :class:`~iris.cube.Cube` defining the source grid.
        * target_grid:
            The :class:`~iris.cube.Cube` defining the target grid.

        Returns:
            A callable with the interface:

                `callable(cube)`

            where `cube` is a cube with the same grid as `src_grid`
            that is to be regridded to the `target_grid`.

        """
        return CurvilinearRegridder(src_grid, target_grid, self.weights)
