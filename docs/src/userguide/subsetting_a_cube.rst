.. _subsetting_a_cube:

=================
Subsetting a Cube
=================

The :doc:`loading_iris_cubes` section of the user guide showed how to load data into multidimensional Iris cubes.
However it is often necessary to reduce the dimensionality of a cube down to something more appropriate and/or manageable.

Iris provides several ways of reducing both the amount of data and/or the number of dimensions in your cube depending on the circumstance.
In all cases **the subset of a valid cube is itself a valid cube**.


Cube Extraction
^^^^^^^^^^^^^^^^
A subset of a cube can be "extracted" from a multi-dimensional cube in order to reduce its dimensionality:

    >>> import iris
    >>> filename = iris.sample_data_path('space_weather.nc')
    >>> cube = iris.load_cube(filename, 'electron density')
    >>> equator_slice = cube.extract(iris.Constraint(grid_latitude=0))
    >>> print(equator_slice)
    electron density / (1E11 e/m^3)     (height: 29; grid_longitude: 31)
        Dimension coordinates:
            height                             x                   -
            grid_longitude                     -                   x
        Auxiliary coordinates:
            latitude                           -                   x
            longitude                          -                   x
        Scalar coordinates:
            grid_latitude               0.0 degrees
        Attributes:
            Conventions                 'CF-1.5'


In this example we start with a 3 dimensional cube, with dimensions of ``height``, ``grid_latitude`` and ``grid_longitude``,
and extract every point where the latitude is 0, resulting in a 2d cube with axes of ``height`` and ``grid_longitude``.


.. _floating-point-warning:
.. warning::

    Caution is required when using equality constraints with floating point coordinates such as ``grid_latitude``.
    Printing the points of a coordinate does not necessarily show the full precision of the underlying number and it
    is very easy return no matches to a constraint when one was expected.
    This can be avoided by using a function as the argument to the constraint::

       def near_zero(cell):
          """Returns true if the cell is between -0.1 and 0.1."""
          return -0.1 < cell < 0.1

       equator_constraint = iris.Constraint(grid_latitude=near_zero)

    Often you will see this construct in shorthand using a lambda function definition::

        equator_constraint = iris.Constraint(grid_latitude=lambda cell: -0.1 < cell < 0.1)


The extract method could be applied again to the *equator_slice* cube to get a further subset.

For example to get a ``height`` of 9000 metres at the equator the following line extends the previous example::

	equator_height_9km_slice = equator_slice.extract(iris.Constraint(height=9000))
	print(equator_height_9km_slice)

The two steps required to get ``height`` of 9000 m at the equator can be simplified into a single constraint::

	equator_height_9km_slice = cube.extract(iris.Constraint(grid_latitude=0, height=9000))
	print(equator_height_9km_slice)

As we saw in :doc:`loading_iris_cubes` the result of :func:`iris.load` is a :class:`CubeList <iris.cube.CubeList>`.
The ``extract`` method also exists on a :class:`CubeList <iris.cube.CubeList>` and behaves in exactly the
same way as loading with constraints:

    >>> import iris
    >>> air_temp_and_fp_6 = iris.Constraint('air_potential_temperature', forecast_period=6)
    >>> level_10 = iris.Constraint(model_level_number=10)
    >>> filename = iris.sample_data_path('uk_hires.pp')
    >>> cubes = iris.load(filename).extract(air_temp_and_fp_6 & level_10)
    >>> print(cubes)
    0: air_potential_temperature / (K)     (grid_latitude: 204; grid_longitude: 187)
    >>> print(cubes[0])
    air_potential_temperature / (K)     (grid_latitude: 204; grid_longitude: 187)
        Dimension coordinates:
            grid_latitude                             x                    -
            grid_longitude                            -                    x
        Auxiliary coordinates:
            surface_altitude                          x                    x
        Derived coordinates:
            altitude                                  x                    x
        Scalar coordinates:
            forecast_period             6.0 hours
            forecast_reference_time     2009-11-19 04:00:00
            level_height                395.0 m, bound=(360.0, 433.3332) m
            model_level_number          10
            sigma                       0.9549927, bound=(0.9589389, 0.95068014)
            time                        2009-11-19 10:00:00
        Attributes:
            STASH                       m01s00i004
            source                      'Data from Met Office Unified Model'
            um_version                  '7.3'


Cube Iteration
^^^^^^^^^^^^^^^
It is not possible to directly iterate over an Iris cube. That is, you cannot use code such as
``for x in cube:``. However, you can iterate over cube slices, as this section details.

A useful way of dealing with a Cube in its **entirety** is by iterating over its layers or slices.
For example, to deal with a 3 dimensional cube (z,y,x) you could iterate over all 2 dimensional slices in y and x
which make up the full 3d cube.::

	import iris
	filename = iris.sample_data_path('hybrid_height.nc')
	cube = iris.load_cube(filename)
	print(cube)
	for yx_slice in cube.slices(['grid_latitude', 'grid_longitude']):
	   print(repr(yx_slice))

As the original cube had the shape (15, 100, 100) there were 15 latitude longitude slices and hence the
line ``print(repr(yx_slice))`` was run 15 times.

.. note::

	The order of latitude and longitude in the list is important; had they been swapped the resultant cube slices
	would have been transposed.

	For further information see :py:meth:`Cube.slices <iris.cube.Cube.slices>`.


This method can handle n-dimensional slices by providing more or fewer coordinate names in the list to **slices**::

	import iris
	filename = iris.sample_data_path('hybrid_height.nc')
	cube = iris.load_cube(filename)
	print(cube)
	for i, x_slice in enumerate(cube.slices(['grid_longitude'])):
	   print(i, repr(x_slice))

The Python function :py:func:`enumerate` is used in this example to provide an incrementing variable **i** which is
printed with the summary of each cube slice. Note that there were 1500 1d longitude cubes as a result of
slicing the 3 dimensional cube (15, 100, 100) by longitude (i starts at 0 and 1500 = 15 * 100).

.. hint::
    It is often useful to get a single 2d slice from a multidimensional cube in order to develop a 2d plot function, for example.
    This can be achieved by using the ``next()`` function on the result of
    slices::

         first_slice = next(cube.slices(['grid_latitude', 'grid_longitude']))

    Once the your code can handle a 2d slice, it is then an easy step to loop over **all** 2d slices within the bigger
    cube using the slices method.


Cube Indexing
^^^^^^^^^^^^^
In the same way that you would expect a numeric multidimensional array to be **indexed** to take a subset of your
original array, you can **index** a Cube for the same purpose.


Here are some examples of array indexing in :py:mod:`numpy`::

	import numpy as np
	# create an array of 12 consecutive integers starting from 0
	a = np.arange(12)
	print(a)

	print(a[0])     # first element of the array

	print(a[-1])    # last element of the array

	print(a[0:4])   # first four elements of the array (the same as a[:4])

	print(a[-4:])   # last four elements of the array

	print(a[::-1])  # gives all of the array, but backwards

	# Make a 2d array by reshaping a
	b = a.reshape(3, 4)
	print(b)

	print(b[0, 0])  # first element of the first and second dimensions

	print(b[0])     # first element of the first dimension (+ every other dimension)

	# get the second element of the first dimension and all of the second dimension
	# in reverse, by steps of two.
	print(b[1, ::-2])


Similarly, Iris cubes have indexing capability::

	import iris
        filename = iris.sample_data_path('hybrid_height.nc')
	cube = iris.load_cube(filename)

	print(cube)

	# get the first element of the first dimension (+ every other dimension)
	print(cube[0])

	# get the last element of the first dimension (+ every other dimension)
	print(cube[-1])

	# get the first 4 elements of the first dimension (+ every other dimension)
	print(cube[0:4])

	# Get the first element of the first and third dimension (+ every other dimension)
	print(cube[0, :, 0])

	# Get the second element of the first dimension and all of the second dimension
	# in reverse, by steps of two.
	print(cube[1, ::-2])
