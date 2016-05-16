.. _loading_iris_cubes:

===================
Loading Iris cubes
===================

To load a single file into a **list** of Iris cubes 
the :py:func:`iris.load` function is used::

    import iris
    filename = '/path/to/file'
    cubes = iris.load(filename)

Iris will attempt to return **as few cubes as possible** 
by collecting together multiple fields with a shared standard name 
into a single multidimensional cube. 

The :py:func:`iris.load` function automatically recognises the format 
of the given files and attempts to produce Iris Cubes from their contents.

.. note::

    Currently there is support for CF NetCDF, GRIB 1 & 2, PP and FieldsFiles 
    file formats with a framework for this to be extended to custom formats.


In order to find out what has been loaded, the result can be printed:

    >>> import iris
    >>> filename = iris.sample_data_path('uk_hires.pp')
    >>> cubes = iris.load(filename)
    >>> print(cubes)
    0: air_potential_temperature / (K)     (time: 3; model_level_number: 7; grid_latitude: 204; grid_longitude: 187)
    1: surface_altitude / (m)              (grid_latitude: 204; grid_longitude: 187)


This shows that there were 2 cubes as a result of loading the file, they were: 
``air_potential_temperature`` and ``surface_altitude``.

The ``surface_altitude`` cube was 2 dimensional with:
   * the two dimensions have extents of 204 and 187 respectively and are 
     represented by the ``grid_latitude`` and ``grid_longitude`` coordinates.

The ``air_potential_temperature`` cubes were 4 dimensional with:
   * the same length ``grid_latitude`` and ``grid_longitude`` dimensions as 
     ``surface_altitide``
   * a ``time`` dimension of length 3
   * a ``model_level_number`` dimension of length 7

.. note::

     The result of :func:`iris.load` is **always** a 
     :class:`list of cubes <iris.cube.CubeList>`. 
     Anything that can be done with a Python :class:`list` can be done 
     with the resultant list of cubes.

.. hint::

    Throughout this user guide you will see the function 
    ``iris.sample_data_path`` being used to get the filename for the resources 
    used in the examples. The result of this function is just a string.
     
    Using this function allows us to provide examples which will work 
    across platforms and with data installed in different locations, 
    however in practice you will want to use your own strings::
    
        filename = '/path/to/file'
        cubes = iris.load(filename)

To get the air potential temperature cube from the list of cubes 
returned by :py:func:`iris.load` in the previous example, 
list indexing can be used:

    >>> import iris
    >>> filename = iris.sample_data_path('uk_hires.pp')
    >>> cubes = iris.load(filename)
    >>> # get the first cube (list indexing is 0 based)
    >>> air_potential_temperature = cubes[0]
    >>> print(air_potential_temperature)
    air_potential_temperature / (K)     (time: 3; model_level_number: 7; grid_latitude: 204; grid_longitude: 187)
         Dimension coordinates:
              time                           x                      -                 -                    -
              model_level_number             -                      x                 -                    -
              grid_latitude                  -                      -                 x                    -
              grid_longitude                 -                      -                 -                    x
         Auxiliary coordinates:
              forecast_period                x                      -                 -                    -
              level_height                   -                      x                 -                    -
              sigma                          -                      x                 -                    -
              surface_altitude               -                      -                 x                    x
         Derived coordinates:
              altitude                       -                      x                 x                    x
         Scalar coordinates:
              forecast_reference_time: 2009-11-19 04:00:00
         Attributes:
              STASH: m01s00i004
              source: Data from Met Office Unified Model
              um_version: 7.3

Notice that the result of printing a **cube** is a little more verbose than 
it was when printing a **list of cubes**. In addition to the very short summary 
which is provided when printing a list of cubes, information is provided 
on the coordinates which constitute the cube in question. 
This was the output discussed at the end of the :doc:`iris_cubes` section.

.. note::

     Dimensioned coordinates will have a dimension marker ``x`` in the 
     appropriate column for each cube data dimension that they describe. 


Loading multiple files
-----------------------

To load more than one file into a list of cubes, a list of filenames can be 
provided to :py:func:`iris.load`::

    filenames = [iris.sample_data_path('uk_hires.pp'),
                 iris.sample_data_path('air_temp.pp')]
    cubes = iris.load(filenames)


It is also possible to load one or more files with wildcard substitution 
using the expansion rules defined :py:mod:`fnmatch`.

For example, to match **zero or more characters** in the filename, 
star wildcards can be used::

    filename = iris.sample_data_path('GloSea4', '*.pp')
    cubes = iris.load(filename)


Constrained loading
-----------------------
Given a large dataset, it is possible to restrict or constrain the load 
to match specific Iris cube metadata. 
Constrained loading provides the ability to generate a cube 
from a specific subset of data that is of particular interest.

As we have seen, loading the following file creates several Cubes::

    filename = iris.sample_data_path('uk_hires.pp')
    cubes = iris.load(filename)

Specifying a name as a constraint argument to :py:func:`iris.load` will mean 
only cubes with a matching :meth:`name <iris.cube.Cube.name>` 
will be returned::

    filename = iris.sample_data_path('uk_hires.pp')
    cubes = iris.load(filename, 'specific_humidity')

To constrain the load to multiple distinct constraints, a list of constraints 
can be provided.  This is equivalent to running load once for each constraint 
but is likely to be more efficient::

    filename = iris.sample_data_path('uk_hires.pp')
    cubes = iris.load(filename, ['air_potential_temperature', 'specific_humidity'])

The :class:`iris.Constraint` class can be used to restrict coordinate values 
on load. For example, to constrain the load to match 
a specific ``model_level_number``::

    filename = iris.sample_data_path('uk_hires.pp')
    level_10 = iris.Constraint(model_level_number=10)
    cubes = iris.load(filename, level_10)

Constraints can be combined using ``&`` to represent a more restrictive 
constraint to ``load``::

    filename = iris.sample_data_path('uk_hires.pp')
    forecast_6 = iris.Constraint(forecast_period=6)
    level_10 = iris.Constraint(model_level_number=10)
    cubes = iris.load(filename, forecast_6 & level_10)

As well as being able to combine constraints using ``&``, 
the :class:`iris.Constraint` class can accept multiple arguments, 
and a list of values can be given to constrain a coordinate to one of 
a collection of values::

    filename = iris.sample_data_path('uk_hires.pp')
    level_10_or_16_fp_6 = iris.Constraint(model_level_number=[10, 16], forecast_period=6)
    cubes = iris.load(filename, level_10_or_16_fp_6)

A common requirement is to limit the value of a coordinate to a specific range, 
this can be achieved by passing the constraint a function::

    def bottom_16_levels(cell):
       # return True or False as to whether the cell in question should be kept
       return cell <= 16

    filename = iris.sample_data_path('uk_hires.pp')
    level_lt_16 = iris.Constraint(model_level_number=bottom_16_levels)
    cubes = iris.load(filename, level_lt_16)
     
.. note::

    As with many of the examples later in this documentation, the 
    simple function above can be conveniently written as a lambda function 
    on a single line::

        bottom_16_levels = lambda cell: cell <= 16


Note also the :ref:`warning on equality constraints with floating point coordinates <floating-point-warning>`.


Cube attributes can also be part of the constraint criteria. Supposing a 
cube attribute of ``STASH`` existed, as is the case when loading ``PP`` files, 
then specific STASH codes can be filtered::

    filename = iris.sample_data_path('uk_hires.pp')
    level_10_with_stash = iris.AttributeConstraint(STASH='m01s00i004') & iris.Constraint(model_level_number=10)
    cubes = iris.load(filename, level_10_with_stash)

.. seealso::

    For advanced usage there are further examples in the 
    :class:`iris.Constraint` reference documentation. 

.. _using-time-constraints:

Constraining on Time
^^^^^^^^^^^^^^^^^^^^
Iris follows NetCDF-CF rules in representing time coordinate values as normalised,
purely numeric, values which are normalised by the calendar specified in the coordinate's
units (e.g. "days since 1970-01-01").
However, when constraining by time we usually want to test calendar-related
aspects such as hours of the day or months of the year, so Iris
provides special features to facilitate this:

Firstly, Iris can be configured so that when it evaluates Constraint
expressions, it will convert time-coordinate values (points and bounds) from
numbers into :class:`~datetime.datetime`-like objects for ease of calendar-based
testing.  This feature is not backwards compatible so for now it must be
explicitly enabled by setting the "cell_datetime_objects" option in :class:`iris.Future`:

    >>> filename = iris.sample_data_path('uk_hires.pp')
    >>> cube_all = iris.load_cube(filename, 'air_potential_temperature')
    >>> print('All times :\n' + str(cube_all.coord('time')))
    All times :
    DimCoord([2009-11-19 10:00:00, 2009-11-19 11:00:00, 2009-11-19 12:00:00], standard_name='time', calendar='gregorian')
    >>> # Define a function which accepts a datetime as its argument (this is simplified in later examples).
    >>> hour_11 = iris.Constraint(time=lambda cell: cell.point.hour == 11)
    >>> with iris.FUTURE.context(cell_datetime_objects=True):
    ...     cube_11 = cube_all.extract(hour_11)
    ... 
    >>> print('Selected times :\n' + str(cube_11.coord('time')))
    Selected times :
    DimCoord([2009-11-19 11:00:00], standard_name='time', calendar='gregorian')

Secondly, the :class:`iris.time` module provides flexible time comparison
facilities.  An :class:`iris.time.PartialDateTime` object can be compared to
objects such as :class:`datetime.datetime` instances, and this comparison will
then test only those 'aspects' which the PartialDateTime instance defines:

    >>> import datetime
    >>> from iris.time import PartialDateTime
    >>> dt = datetime.datetime(2011, 3, 7)
    >>> print(dt > PartialDateTime(year=2010, month=6))
    True
    >>> print(dt > PartialDateTime(month=6))
    False
    >>> 

These two facilities can be combined to provide straightforward calendar-based
time selections when loading or extracting data.

The previous constraint example can now be written as:

   >>> the_11th_hour = iris.Constraint(time=iris.time.PartialDateTime(hour=11))
   >>> with iris.FUTURE.context(cell_datetime_objects=True):
   ...     print(iris.load_cube(
   ...         iris.sample_data_path('uk_hires.pp'),
   ...         'air_potential_temperature' & the_11th_hour).coord('time'))
   DimCoord([2009-11-19 11:00:00], standard_name='time', calendar='gregorian')

A more complex example might be when there exists a time sequence representing the first day of every week
for many years:


.. testsetup:: timeseries_range

    import numpy as np
    from iris.time import PartialDateTime
    long_ts = iris.cube.Cube(np.arange(150), long_name='data', units='1')
    _mondays = iris.coords.DimCoord(7 * np.arange(150), standard_name='time', units='days since 2007-04-09')
    long_ts.add_dim_coord(_mondays, 0)


.. doctest:: timeseries_range
    :options: +NORMALIZE_WHITESPACE, +ELLIPSIS
    
    >>> print(long_ts.coord('time'))
    DimCoord([2007-04-09 00:00:00, 2007-04-16 00:00:00, 2007-04-23 00:00:00,
              ...
              2010-02-01 00:00:00, 2010-02-08 00:00:00, 2010-02-15 00:00:00],
             standard_name='time', calendar='gregorian')

We can select points within a certain part of the year, in this case between
the 15th of July through to the 25th of August, by combining the datetime cell
functionality with PartialDateTime:

.. doctest:: timeseries_range

    >>> st_swithuns_daterange = iris.Constraint(
    ...     time=lambda cell: PartialDateTime(month=7, day=15) < cell < PartialDateTime(month=8, day=25))
    >>> with iris.FUTURE.context(cell_datetime_objects=True):
    ...   within_st_swithuns = long_ts.extract(st_swithuns_daterange)
    ... 
    >>> print(within_st_swithuns.coord('time'))
    DimCoord([2007-07-16 00:00:00, 2007-07-23 00:00:00, 2007-07-30 00:00:00,
           2007-08-06 00:00:00, 2007-08-13 00:00:00, 2007-08-20 00:00:00,
           2008-07-21 00:00:00, 2008-07-28 00:00:00, 2008-08-04 00:00:00,
           2008-08-11 00:00:00, 2008-08-18 00:00:00, 2009-07-20 00:00:00,
           2009-07-27 00:00:00, 2009-08-03 00:00:00, 2009-08-10 00:00:00,
           2009-08-17 00:00:00, 2009-08-24 00:00:00], standard_name='time', calendar='gregorian')

Notice how the dates printed are between the range specified in the ``st_swithuns_daterange``
and that they span multiple years.


Strict loading
--------------

The :py:func:`iris.load_cube` and :py:func:`iris.load_cubes` functions are
similar to :py:func:`iris.load` except they can only return 
*one cube per constraint*.
The :func:`iris.load_cube` function accepts a single constraint and
returns a single cube. The :func:`iris.load_cubes` function accepts any
number of constraints and returns a list of cubes (as an `iris.cube.CubeList`).
Providing no constraints to :func:`iris.load_cube` or :func:`iris.load_cubes`
is equivalent to requesting exactly one cube of any type. 

A single cube is loaded in the following example::

    >>> filename = iris.sample_data_path('air_temp.pp')
    >>> cube = iris.load_cube(filename)
    >>> print(cube)
    air_temperature / (K)                 (latitude: 73; longitude: 96)
         Dimension coordinates:
              latitude                           x              -
              longitude                          -              x
    ...
         Cell methods:
              mean: time

However, when attempting to load data which would result in anything other than 
one cube, an exception is raised::

    >>> filename = iris.sample_data_path('uk_hires.pp')
    >>> cube = iris.load_cube(filename)
    Traceback (most recent call last):
    ...
    iris.exceptions.ConstraintMismatchError: Expected exactly one cube, found 2.

.. note::
 
    All the load functions share many of the same features, hence 
    multiple files could be loaded with wildcard filenames 
    or by providing a list of filenames.

The strict nature of :func:`iris.load_cube` and :func:`iris.load_cubes`
means that, when combined with constrained loading, it is possible to 
ensure that precisely what was asked for on load is given 
- otherwise an exception is raised. 
This fact can be utilised to make code only run successfully if 
the data provided has the expected criteria.

For example, suppose that code needed ``air_potential_temperature`` 
in order to run::

    import iris
    filename = iris.sample_data_path('uk_hires.pp')
    air_pot_temp = iris.load_cube(filename, 'air_potential_temperature')
    print(air_pot_temp)

Should the file not produce exactly one cube with a standard name of 
'air_potential_temperature', an exception will be raised.

Similarly, supposing a routine needed both 'surface_altitude' and 
'air_potential_temperature' to be able to run::

    import iris
    filename = iris.sample_data_path('uk_hires.pp')
    altitude_cube, pot_temp_cube = iris.load_cubes(filename, ['surface_altitude', 'air_potential_temperature'])

The result of :func:`iris.load_cubes` in this case will be a list of 2 cubes 
ordered by the constraints provided. Multiple assignment has been used to put 
these two cubes into separate variables.

.. note::

    In Python, lists of a pre-known length and order can be exploited 
    using *multiple assignment*:

        >>> number_one, number_two = [1, 2]
        >>> print(number_one)
        1
        >>> print(number_two)
        2

