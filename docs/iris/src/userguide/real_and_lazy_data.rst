.. _real_and_lazy_data:

==================
Real and Lazy Data
==================

What is real and lazy data?
---------------------------

Every Iris cube contains an n-dimensional data array, which could be real or
lazy.

Real data is contained in a NumPy array which has a shape, a data type, some
other useful information and many data points, each of which use up a small
allocation of memory.

Lazy data is contained in a conceptual array which retains the information
about its real counterpart but has no actual data points, so its memory
allocation is much smaller.  This will be in the form of a Dask array.

Arrays in Iris can be converted between their real and lazy states,
although there are some limits to this process (*explain this better - what
limits?*).

The advantage of using lazy data is that it has a small memory footprint which
enables the user to load and manipulate datasets that you would otherwise not
be able to fit into memory.

However, in order to execute certain operations (such as calculations on actual
data values) the real data must be realized.  Using Dask, the operation will
be deferred until you request the result, at which point it will be executed
using Dask's parallel processing schedulers.  The combination of these two
behaviours can offer a significant performance boost.

You can check whether the data array on your cube is lazy using the Iris
function 'has_lazy_data'.  For example:

.. doctest::

    >>> import iris
    >>> cube = iris.load_cube(filename, 'air_temp.pp')
    >>> cube.has_lazy_data()
    True
    >>> cube.data
    >>> cube.has_lazy_data()
    False

When does my data become real?
------------------------------

If the data on your cube is in its lazy state, it will only become real if you
'touch' the data.  This means any way of directly accessing the data, such as
assigning it to a variable or simply using 'cube.data' as in the example above.

Any action which requires the use of actual data values (such as cube maths)
will also cause the data to be loaded into memory, although data realization
is always deferred until the result is requested:

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
    >>> the_data = cube.core_data()
    >>> cube.has_lazy_data()
    True
    >>> real_data = cube.data
    >>> cube.has_lazy_data()
    False


Changing a cube's data
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


Coordinates
-----------

Cubes possess coordinate arrays as well as data arrays, so these also benefit
from Dask's functionality, although there are some distinctions between how
the different coordinate types are treated.

Auxiliary coordinates can now contain lazy arrays, so they will adhere to the
same rules and behaviour as the data arrays.  Dimension coordinates, however,
undergo monotonicity checks which cause the arrays to be realized upon
construction, so they can only contain real arrays.


Dask processing options
-----------------------

Dask applies some default values to certain aspects of the parallel processing
that it offers with Iris. It is possible to change these values and override
the defaults by using 'dask.set_options(option)' in your script.

You can use this as a global variable if you wish to use your chosen option for
the full length of the script, or you can use it with a context manager to
control the span of the option.

Here are some examples of the options that you may wish to change.

You can set the number of threads on which to work like this:

.. doctest::

    >>> import dask
    >>> from multiprocessing.pool import ThreadPool
    >>> with dask.set_options(pool=ThreadPool(4)):
    ...     x.compute()

Multiple threads work well with heavy computation.


You can change the default option between threaded scheduler and
multiprocessing scheduler, for example:

.. doctest::

    >>> with dask.set_options(get=dask.multiprocessing.get):
    ...     x.sum().compute()

Multiprocessing works well with strings, lists or custom Dask objects.


You can choose to run all processes in serial (which is currently the Iris
default):

.. doctest::

    >>> dask.set_options(get=dask.async.get_sync)

This option is particularly good for debugging scripts.


Further reading
---------------

Dask offers much more fine control than is described in this user guide,
although a good understanding of the package would be required to properly
utilize it.

For example, it is possible to write callback functions to customize processing
options, of which there are many more than we have outlined.  Also, you may
wish to use some of the available Dask functionality regarding deferred
operations for your own scripts and objects.

For more information about these tools, how they work and what you can do with
them, please visit the following package documentation pages:

.. _Dask: http://dask.pydata.org/en/latest/
.. _Dask.distributed: http://distributed.readthedocs.io/en/latest/

`Dask`_
`Dask.distributed`_



