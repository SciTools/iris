=================
Navigating a cube
=================

.. testsetup::

        import iris
        filename = iris.sample_data_path('rotated_pole.nc')
        # pot_temp = iris.load_cube(filename, 'air_potential_temperature')
        cube = iris.load_cube(filename)
        coord_names = [coord.name() for coord in cube.coords()]
        coord = cube.coord('grid_latitude')


After loading any cube, you will want to investigate precisely what it contains. This section is all about accessing 
and manipulating the metadata contained within a cube.

Cube string representations
---------------------------

We have already seen a basic string representation of a cube when printing:

    >>> import iris
    >>> filename = iris.sample_data_path('rotated_pole.nc')
    >>> cube = iris.load_cube(filename)
    >>> print(cube)
    air_pressure_at_sea_level / (Pa)    (grid_latitude: 22; grid_longitude: 36)
         Dimension coordinates:
              grid_latitude                           x                   -
              grid_longitude                          -                   x
         Scalar coordinates:
              forecast_period: 0.0 hours
              forecast_reference_time: 2006-06-15 00:00:00
              time: 2006-06-15 00:00:00
         Attributes:
              Conventions: CF-1.5
              STASH: m01s16i222
              source: Data from Met Office Unified Model 6.01


This representation is equivalent to passing the cube to the :func:`str` function.  This function can be used on 
any Python variable to get a string representation of that variable. 
Similarly there exist other standard functions for interrogating your variable: :func:`repr`, :func:`type` for example::

    print(str(cube))
    print(repr(cube))
    print(type(cube))

Other, more verbose, functions also exist which give information on **what** you can do with *any* given 
variable. In most cases it is reasonable to ignore anything starting with a "``_``" (underscore) or a "``__``" (double underscore)::

    dir(cube)
    help(cube)

Working with cubes
------------------

Every cube has a standard name, long name and units which are accessed with 
:attr:`Cube.standard_name <iris.cube.Cube.standard_name>`,
:attr:`Cube.long_name <iris.cube.Cube.long_name>` 
and :attr:`Cube.units <iris.cube.Cube.units>` respectively::

    print(cube.standard_name)
    print(cube.long_name)
    print(cube.units)
    
Interrogating these with the standard :func:`type` function will tell you that ``standard_name`` and ``long_name`` 
are either a string or ``None``, and ``units`` is an instance of :class:`iris.unit.Unit`.

You can access a string representing the "name" of a cube with the :meth:`Cube.name() <iris.cube.Cube.name>` method::

    print(cube.name())
    
The result of which is **always** a string.

Each cube also has a :mod:`numpy` array which represents the phenomenon of the cube which can be accessed with the 
:attr:`Cube.data <iris.cube.Cube.data>` attribute. As you can see the type is a :class:`numpy n-dimensional array <numpy.ndarray>`::

    print(type(cube.data))

.. note::

    When loading from most file formats in Iris, the data itself is not loaded until the **first** time that the data is requested. 
    Hence you may have noticed that running the previous command for the first time takes a little longer than it does for 
    subsequent calls.

    For this reason, when you have a large cube it is strongly recommended that you do not access the cube's data unless 
    you need to. 
    For convenience :attr:`~iris.cube.Cube.shape` and :attr:`~iris.cube.Cube.ndim` attributes exists on a cube, which 
    can tell you the shape of the cube's data without loading it::

       print(cube.shape)
       print(cube.ndim)

    For more on the benefits, handling and uses of lazy data, see
    :doc:`Real and Lazy Data </userguide/real_and_lazy_data>`


You can change the units of a cube using the :meth:`~iris.cube.Cube.convert_units` method. For example::

    cube.convert_units('celsius')

As well as changing the value of the :attr:`~iris.cube.Cube.units` attribute this will also convert the values in
:attr:`~iris.cube.Cube.data`. To replace the units without modifying the data values one can change the
:attr:`~iris.cube.Cube.units` attribute directly.

Some cubes represent a processed phenomenon which are represented with cell methods, these can be accessed on a 
cube with the :attr:`Cube.cell_methods <iris.cube.Cube.cell_methods>` attribute::

    print(cube.cell_methods)


Accessing coordinates on the cube
---------------------------------

A cube's coordinates can be retrieved via :meth:`Cube.coords <iris.cube.Cube.coords>`. 
A simple for loop over the coords can print a coordinate's :meth:`~iris.coords.Coord.name`::

     for coord in cube.coords():
         print(coord.name())

Alternatively, we can use *list comprehension* to store the names in a list::

     coord_names = [coord.name() for coord in cube.coords()]

The result is a basic Python list which could be sorted alphabetically and joined together:

     >>> print(', '.join(sorted(coord_names)))
     forecast_period, forecast_reference_time, grid_latitude, grid_longitude, time

To get an individual coordinate given its name, the :meth:`Cube.coord <iris.cube.Cube.coord>` method can be used::

     coord = cube.coord('grid_latitude')
     print(type(coord))

Every coordinate has a :attr:`Coord.standard_name <iris.coords.Coord.standard_name>`, 
:attr:`Coord.long_name <iris.coords.Coord.long_name>`, and :attr:`Coord.units <iris.coords.Coord.units>` attribute::

     print(coord.standard_name)
     print(coord.long_name)
     print(coord.units)

Additionally every coordinate can provide its :attr:`~iris.coords.Coord.points` and :attr:`~iris.coords.Coord.bounds` 
numpy array. If the coordinate has no bounds ``None`` will be returned::

     print(type(coord.points))
     print(type(coord.bounds))


Adding metadata to a cube
-------------------------

We can add and remove coordinates via :func:`Cube.add_dim_coord<iris.cube.Cube.add_dim_coord>`, 
:func:`Cube.add_aux_coord<iris.cube.Cube.add_aux_coord>`, and :meth:`Cube.remove_coord <iris.cube.Cube.remove_coord>`.


    >>> import iris.coords
    >>> new_coord = iris.coords.AuxCoord(1, long_name='my_custom_coordinate', units='no_unit')
    >>> cube.add_aux_coord(new_coord)
    >>> print(cube)
    air_pressure_at_sea_level / (Pa)    (grid_latitude: 22; grid_longitude: 36)
         Dimension coordinates:
              grid_latitude                           x                   -
              grid_longitude                          -                   x
         Scalar coordinates:
              forecast_period: 0.0 hours
              forecast_reference_time: 2006-06-15 00:00:00
              my_custom_coordinate: 1
              time: 2006-06-15 00:00:00
         Attributes:
              Conventions: CF-1.5
              STASH: m01s16i222
              source: Data from Met Office Unified Model 6.01


The coordinate ``my_custom_coordinate`` now exists on the cube and is listed under the non-dimensioned single valued scalar coordinates.


Adding and removing metadata to the cube at load time
-----------------------------------------------------

Sometimes when loading a cube problems occur when the amount of metadata is more or less than expected.
This is often caused by one of the following:

 * The file does not contain enough metadata, and therefore the cube cannot know everything about the file.
 * Some of the metadata of the file is contained in the filename, but is not part of the actual file.
 * There is not enough metadata loaded from the original file as Iris has not handled the format fully. *(in which case, 
   please let us know about it)*

To solve this, all of :func:`iris.load`, :func:`iris.load_cube`, and :func:`iris.load_cubes` support a callback keyword. 

The callback is a user defined function which must have the calling sequence ``function(cube, field, filename)`` 
which can make any modifications to the cube in-place, or alternatively return a completely new cube instance.

Suppose we wish to load a lagged ensemble dataset from the Met Office's GloSea4 model. 
The data for this example represents 13 ensemble members of 6 one month timesteps; the logistics of the 
model mean that the run is spread over several days. 

If we try to load the data directly for ``surface_temperature``:

    >>> filename = iris.sample_data_path('GloSea4', '*.pp')
    >>> print(iris.load(filename, 'surface_temperature'))
    0: surface_temperature / (K)           (time: 6; forecast_reference_time: 2; latitude: 145; longitude: 192)
    1: surface_temperature / (K)           (time: 6; forecast_reference_time: 2; latitude: 145; longitude: 192)
    2: surface_temperature / (K)           (realization: 9; time: 6; latitude: 145; longitude: 192)




We get multiple cubes some with more dimensions than expected, some without a ``realization`` (i.e. ensemble member) dimension. 
In this case, two of the PP files have been encoded without the appropriate ``realization`` number attribute, which means that
the appropriate coordinate cannot be added to the resultant cube. Fortunately, the missing attribute has been encoded in the filename
which, given the filename, we could extract::

    filename = iris.sample_data_path('GloSea4', 'ensemble_001.pp')
    realization = int(filename[-6:-3])
    print(realization)

We can solve this problem by adding the appropriate metadata, on load, by using a callback function, which runs on a field
by field basis *before* they are automatically merged together:

.. testcode::

    import numpy as np
    import iris
    import iris.coords as icoords

    def lagged_ensemble_callback(cube, field, filename):
        # Add our own realization coordinate if it doesn't already exist.
        if not cube.coords('realization'):
            realization = np.int32(filename[-6:-3])
            ensemble_coord = icoords.AuxCoord(realization, standard_name='realization', units="1")
            cube.add_aux_coord(ensemble_coord)

    filename = iris.sample_data_path('GloSea4', '*.pp')

    print(iris.load(filename, 'surface_temperature', callback=lagged_ensemble_callback))


The result is a single cube which represents the data in a form that was expected:

.. testoutput::

    0: surface_temperature / (K)           (realization: 13; time: 6; latitude: 145; longitude: 192)
