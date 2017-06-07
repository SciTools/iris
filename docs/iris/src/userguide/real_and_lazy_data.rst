.. _real_and_lazy_data:


.. testsetup:: *

    import dask.array as da
    import iris
    import numpy as np


==================
Real and Lazy Data
==================

We have seen in the :doc:`user_guide_introduction` section of the user guide that
Iris cubes contain data and metadata about a phenomenon. The data element of a cube
is always an array, but the array may be either "real" or "lazy".

In this section of the user guide we will look specifically at the concepts of
real and lazy data as they apply to the cube and other data structures in Iris.


What is real and lazy data?
---------------------------

In Iris, we use the term **real data** to describe data arrays that are loaded
into memory. Real data is typically provided as a
`NumPy array <https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html>`_,
which has a shape and data type that are used to describe the array's data points.
Each data point takes up a small amount of memory, which means large NumPy arrays can
take up a large amount of memory.

Conversely, we use the term **lazy data** to describe data that is not loaded into memory.
(This is sometimes also referred to as **deferred data**.)
In Iris, lazy data is provided as a
`dask array <http://dask.pydata.org/en/latest/array-overview.html>`_.
A dask array also has a shape and data type
but typically the dask array's data points are not loaded into memory.
Instead the data points are stored on disk and only loaded into memory in
small chunks when absolutely necessary (see the section :ref:`when_real_data`
for examples of when this might happen).

The primary advantage of using lazy data is that it enables
`out-of-core processing <https://en.wikipedia.org/wiki/Out-of-core_algorithm>`_;
that is, the loading and manipulating of datasets that otherwise would not fit into memory.

You can check whether a cube has real data or lazy data by using the method
:meth:`~iris.cube.Cube.has_lazy_data`. For example::

    >>> cube = iris.load_cube(iris.sample_data_path('air_temp.pp'))
    >>> cube.has_lazy_data()
    True
    # Realise the lazy data.
    >>> cube.data
    >>> cube.has_lazy_data()
    False


.. _when_real_data:

When does my data become real?
------------------------------

When you load a dataset using Iris the data array will almost always initially be
a lazy array. This section details some operations that will realise lazy data
as well as some operations that will maintain lazy data. We use the term **realise**
to mean converting lazy data into real data.

Most operations on data arrays can be run equivalently on both real and lazy data.
If the data array is real then the operation will be run on the data array
immediately. The results of the operation will be available as soon as processing is completed.
If the data array is lazy then the operation will be deferred and the data array will
remain lazy until you request the result (such as when you call ``cube.data``)::

    >>> cube = iris.load_cube(iris.sample_data_path('air_temp.pp'))
    >>> cube.has_lazy_data()
    True
    >>> cube += 5
    >>> cube.has_lazy_data()
    True

The process by which the operation is deferred until the result is requested is
referred to as **lazy evaluation**.

Certain operations, including regridding and plotting, can only be run on real data.
Calling such operations on lazy data will automatically realise your lazy data.

You can also realise (and so load into memory) your cube's lazy data if you 'touch' the data.
To 'touch' the data means directly accessing the data by calling ``cube.data``,
as in the previous example.

Core data
^^^^^^^^^

Cubes have the concept of "core data". This returns the cube's data in its
current state:

 * If a cube has lazy data, calling the cube's :meth:`~iris.cube.Cube.core_data` method
   will return the cube's lazy dask array. Calling the cube's
   :meth:`~iris.cube.Cube.core_data` method **will never realise** the cube's data.
 * If a cube has real data, calling the cube's :meth:`~iris.cube.Cube.core_data` method
   will return the cube's real NumPy array.

For example::

    >>> cube = iris.load_cube(iris.sample_data_path('air_temp.pp'))
    >>> cube.has_lazy_data()
    True

    >>> the_data = cube.core_data()
    >>> type(the_data)
    <class 'dask.array.core.Array'>
    >>> cube.has_lazy_data()
    True

    # Realise the lazy data.
    >>> cube.data
    >>> the_data = cube.core_data()
    >>> type(the_data)
    <type 'numpy.ndarray'>
    >>> cube.has_lazy_data()
    False


Coordinates
-----------

In the same way that Iris cubes contain a data array, Iris coordinates contain a
points array and an optional bounds array.
Coordinate points and bounds arrays can also be real or lazy:

 * A :class:`~iris.coords.DimCoord` will only ever have **real** points and bounds
   arrays because of monotonicity checks that realise lazy arrays.
 * An :class:`~iris.coords.AuxCoord` can have **real or lazy** points and bounds.
 * An :class:`~iris.aux_factory.AuxCoordFactory` (or derived coordinate)
   can have **real or lazy** points and bounds. If all of the
   :class:`~iris.coords.AuxCoord` instances used to construct the derived coordinate
   have real points and bounds then the derived coordinate will have real points
   and bounds, otherwise the derived coordinate will have lazy points and bounds.

Iris cubes and coordinates have very similar interfaces, which extends to accessing
coordinates' lazy points and bounds:

.. doctest::

    >>> cube = iris.load_cube(iris.sample_data_path('hybrid_height.nc'))

    >>> dim_coord = cube.coord('model_level_number')
    >>> print(dim_coord.has_lazy_points())
    False
    >>> print(dim_coord.has_bounds())
    False
    >>> print(dim_coord.has_lazy_bounds())
    False

    >>> aux_coord = cube.coord('sigma')
    >>> print(aux_coord.has_lazy_points())
    True
    >>> print(aux_coord.has_bounds())
    True
    >>> print(aux_coord.has_lazy_bounds())
    True

    # Realise the lazy points. This will **not** realise the lazy bounds.
    >>> points = aux_coord.points
    >>> print(aux_coord.has_lazy_points())
    False
    >>> print(aux_coord.has_lazy_bounds())
    True

    >>> derived_coord = cube.coord('altitude')
    >>> print(derived_coord.has_lazy_points())
    True
    >>> print(derived_coord.has_bounds())
    True
    >>> print(derived_coord.has_lazy_bounds())
    True

.. note::
    Printing a lazy :class:`~iris.coords.AuxCoord` will realise its points and bounds arrays!


Dask processing options
-----------------------

As stated earlier in this user guide section, Iris uses dask to provide
lazy data arrays for both Iris cubes and coordinates. Iris also uses dask
functionality for processing deferred operations on lazy arrays.

Dask provides processing options to control how deferred operations on lazy arrays
are computed. This is provided via the ``dask.set_options`` interface.
We can make use of this functionality in Iris. This means we can
control how dask arrays in Iris are processed, for example giving us power to
run Iris processing in parallel.

Iris by default applies a single dask processing option. This specifies that
all dask processing in Iris should be run in serial (that is, without any
parallel processing enabled).

The dask processing option applied by Iris can be overridden by manually setting
dask processing options for either or both of:

 * the number of parallel workers to use,
 * the scheduler to use.

This must be done **before** importing Iris. For example, to specify that dask
processing within Iris should use four workers in a thread pool::

    >>> from multiprocessing.pool import ThreadPool
    >>> import dask
    >>> dask.set_options(get=dask.threaded.get, pool=ThreadPool(4))

    >>> import iris
    >>> # Iris processing here...

.. note::
    These dask processing options will last for the lifetime of the Python session
    and must be re-applied in other or subsequent sessions.

Other dask processing options are also available. See the
`dask documentation <http://dask.pydata.org/en/latest/scheduler-overview.html>`_
for more information on setting dask processing options.


Further reading
---------------

This section of the Iris user guide provides a quick overview of real and lazy
data within Iris. For more details on these and related concepts,
see the whitepaper on lazy data.
