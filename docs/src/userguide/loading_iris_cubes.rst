.. _loading_iris_cubes:

===================
Loading Iris Cubes
===================

To load a single file into a **list** of Iris cubes
the :py:func:`iris.load` function is used::

    import iris
    filename = '/path/to/file'
    cubes = iris.load(filename)

Iris will attempt to return **as few cubes as possible**
by collecting together multiple fields with a shared standard name
into a single multidimensional cube.

.. hint::

    There are details at :class:`iris.CombineOptions` on how Iris works to load
    fewer and larger cubes :  The :data:`iris.COMBINE_POLICY` object allows the user to
    control how cubes are combined during the loading process.  See the documentation
    of the :class:`iris.CombineOptions` class for details.

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

     The result of :func:`iris.load` is **always** a :class:`iris.cube.CubeList`
     (even if it only contains one :class:`iris.cube.Cube` - see
     :ref:`strict-loading`). Anything that can be done with a Python
     :class:`list` can be done with an :class:`iris.cube.CubeList`.

     The order of this list should not be relied upon. Ways of loading a
     specific cube or cubes are covered in :ref:`constrained-loading` and
     :ref:`strict-loading`.

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
            time                             x                      -                 -                    -
            model_level_number               -                      x                 -                    -
            grid_latitude                    -                      -                 x                    -
            grid_longitude                   -                      -                 -                    x
        Auxiliary coordinates:
            forecast_period                  x                      -                 -                    -
            level_height                     -                      x                 -                    -
            sigma                            -                      x                 -                    -
            surface_altitude                 -                      -                 x                    x
        Derived coordinates:
            altitude                         -                      x                 x                    x
        Scalar coordinates:
            forecast_reference_time     2009-11-19 04:00:00
        Attributes:
            STASH                       m01s00i004
            source                      'Data from Met Office Unified Model'
            um_version                  '7.3'

Notice that the result of printing a **cube** is a little more verbose than
it was when printing a **list of cubes**. In addition to the very short summary
which is provided when printing a list of cubes, information is provided
on the coordinates which constitute the cube in question.
This was the output discussed at the end of the :doc:`iris_cubes` section.

.. note::

     Dimensioned coordinates will have a dimension marker ``x`` in the
     appropriate column for each cube data dimension that they describe.


Loading Multiple Files
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


.. note::

     The cubes returned will not necessarily be in the same order as the
     order of the filenames.

Lazy Loading
------------

In fact when Iris loads data from most file types, it normally only reads the
essential descriptive information or metadata :  the bulk of the actual data
content will only be loaded later, as it is needed.
This is referred to as 'lazy' data.  It allows loading to be much quicker, and to occupy less memory.

For more on the benefits, handling and uses of lazy data, see :doc:`Real and Lazy Data </userguide/real_and_lazy_data>`.


.. _constrained-loading:

Constrained Loading
-----------------------
Given a large dataset, it is possible to restrict or constrain the load
to match specific Iris cube metadata.
Constrained loading provides the ability to generate a cube
from a specific subset of data that is of particular interest.

As we have seen, loading the following file creates several Cubes::

    filename = iris.sample_data_path('uk_hires.pp')
    cubes = iris.load(filename)

Specifying a name as a constraint argument to :py:func:`iris.load` will mean
only cubes with matching :meth:`name <iris.cube.Cube.names>`
will be returned::

    filename = iris.sample_data_path('uk_hires.pp')
    cubes = iris.load(filename, 'surface_altitude')

Note that, the provided name will match against either the standard name,
long name, NetCDF variable name or STASH metadata of a cube. Therefore, the
previous example using the ``surface_altitude`` standard name constraint can
also be achieved using the STASH value of ``m01s00i033``::

    filename = iris.sample_data_path('uk_hires.pp')
    cubes = iris.load(filename, 'm01s00i033')

If further specific name constraint control is required i.e., to constrain
against a combination of standard name, long name, NetCDF variable name and/or
STASH metadata, consider using the :class:`iris.NameConstraint`. For example,
to constrain against both a standard name of ``surface_altitude`` **and** a STASH
of ``m01s00i033``::

    filename = iris.sample_data_path('uk_hires.pp')
    constraint = iris.NameConstraint(standard_name='surface_altitude', STASH='m01s00i033')
    cubes = iris.load(filename, constraint)

To constrain the load to multiple distinct constraints, a list of constraints
can be provided.  This is equivalent to running load once for each constraint
but is likely to be more efficient::

    filename = iris.sample_data_path('uk_hires.pp')
    cubes = iris.load(filename, ['air_potential_temperature', 'surface_altitude'])

The :class:`iris.Constraint` class can be used to restrict coordinate values
on load. For example, to constrain the load to match
a specific ``model_level_number``::

    filename = iris.sample_data_path('uk_hires.pp')
    level_10 = iris.Constraint(model_level_number=10)
    cubes = iris.load(filename, level_10)

Further details on using :class:`iris.Constraint` are
discussed later in :ref:`cube_extraction`.

.. _strict-loading:

Strict Loading
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
              0                           time: mean

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

