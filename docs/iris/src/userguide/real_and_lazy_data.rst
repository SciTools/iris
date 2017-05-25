.. _real_and_lazy_data:

==================
Real and Lazy Data
==================

What is Real and Lazy Data?
---------------------------

Every Iris cube contains an n-dimensional  data array, which could be real or
lazy.

Real data is contained in an array which has a shape, a data type, some other
useful information and many data points, each of which use up a small
allocation of memory.  This generally takes the form of a numpy array.

Lazy data is contained in a conceptual array which retains the information
about its real counterpart but has no actual data points, so its memory
allocation is much smaller.  This will be in the form of a Dask array.

Arrays in Iris can be converted flexibly(?) between their real and lazy states,
although there are some limits to this process.  The advantage of using lazy
data is that it has a small memory footprint, so certain operations
(such as...?) can be much faster.  However, in order to perform other
operations (such as calculations on actual data values) the real data must be
realized.

* When a cube is loaded, are the data arrays always lazy to begin with? *

You can check whether the data array on your cube is lazy using the Iris
function 'has_lazy_data'.  For example:

.. doctest::

    >>> import iris
    >>> cube = iris.load_cube(filename, 'air_temp.pp')
    >>> cube.has_lazy_data()
    True
    >>> _ = cube.data
    >>> cube.has_lazy_data()
    False

Assigning the data array to a variable causes the real data to be realized, at
which point the array ceases to be lazy.  Any action which requires the use of
actual data values (such as cube maths) will have this effect, although data
realization is always deferred until the last possible moment:

.. doctest::

    >>> cube = iris.load_cube(iris.sample_data_path('air_temp.pp'))
    >>> cube.has_lazy_data()
    True
    >>> cube += 5
    >>> cube.has_lazy_data()
    True
    >>> cube.data
    >>> cube.has_lazy_data()
    False

You can also convert realized data back into a lazy array:

.. doctest::

    >>> cube.has_lazy_data()
    False
    >>> cube.data = cube.lazy_data()
    >>> cube.has_lazy_data()
    True



Core data refers to the current state of the cube's data, be it real or
lazy.  This can be used if you wish to refer to the data array but are
indifferent to its current state.  If the cube's data is lazy, it will not be
realized when you reference the core data attribute (?):

.. doctest::

    >>> cube = iris.load_cube(iris.sample_data_path('air_temp.pp'))
    >>> cube.has_lazy_data()
    True
    >>> the_data = cube.core_data
    >>> cube.has_lazy_data()
    True
    >>> real_data = cube.data
    >>> cube.has_lazy_data()
    False


Changing a Cube's Data
----------------------

There are several methods of modifying a cube's data array, each one subtly
different from the others.

Maths
^^^^^

You can use :ref:`cube maths <cube_maths>` to make in-place modifications to
each point in a cube's existing data array.  Provided you do not directly
reference the cube's data, the array will remain lazy:

.. doctest::

    >>> cube = iris.load_cube(iris.sample_data_path('air_temp.pp'))
    >>> cube.has_lazy_data()
    True
    >>> cube *= 10
    >>> cube.has_lazy_data()
    True

Copy
^^^^

You can copy a cube and assign a completely new data array to the copy. All the
original cube's metadata will be the same as the new cube's metadata.  However,
the new cube's data array will not be lazy if you replace it with a real array:

.. doctest::

    >>> import numpy as np
    >>> data = np.zeros((73, 96))
    >>> new_cube = cube.copy(data=data)
    >>> new_cube.has_lazy_data()
    False

Replace
^^^^^^^

This does essentially the same thing as `cube.copy()`, except that it provides
a safe method of doing so for the specific edge case of a lazy masked integer
array:

.. doctest::

    >>> values = np.zeros((73, 96), dtype=int)
    >>> data =np.ma.masked_values(values, 0)
    >>> print(data)
    [[-- -- -- ..., -- -- --]
     [-- -- -- ..., -- -- --]
     [-- -- -- ..., -- -- --]
     ...,
     [-- -- -- ..., -- -- --]
     [-- -- -- ..., -- -- --]
     [-- -- -- ..., -- -- --]]
    >>> new_cube = cube.copy(data=data)
    >>> new_cube.has_lazy_data()
    False
    >>> new_cube.data = new_cube.lazy_data()
    >>> new_cube.has_lazy_data()
    True

This method is necessary as Dask is currently unable to handle masked arrays.
Please refer to the Whitepaper for further details.


Dask Processing Options
-----------------------

As well as Dask offering the benefit of a smaller memory footprint through the
handling of lazy arrays, it can significantly speed up performance by allowing
Iris to use multiprocessing.

There are some default values which are set by Dask and passed through to Iris.
If you wish to change these options, you can override them globally or using a
context manager.

Here are some examples of the options that you may wish to change:

You can set the number of threads on which to work like this:

    >>> from multiprocessing.pool import ThreadPool
    >>> with dask.set_options(pool=ThreadPool(4)):
    ...     x.compute()

Multiple threads work well with heavy computation.


You can change the default option between threaded scheduler and
multiprocessing scheduler, for example:

    >>> with dask.set_options(get=dask.multiprocessing.get):
    ...     x.sum().compute()

Multiprocessing works well with strings, lists or custom Dask objects.


You can choose to run all processes in serial (which is currently the Iris
default):

    >>> dask.set_options(get=dask.get)

This option is particularly good for debugging scripts.


Further Reading
---------------



Stuff still to add (?):
- Links to dask docs and distributed docs


