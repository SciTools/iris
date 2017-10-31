
.. _real_and_lazy_data:


.. testsetup:: *

    import dask.array as da
    import iris
    import numpy as np


==================
Real and Lazy Data
==================

We have seen in the :doc:`iris_cubes` section of the user guide that
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
but the dask array's data points remain on disk and only loaded into memory in
small chunks when absolutely necessary.  This has key performance benefits for
handling large amounts of data, where both calculation time and storage
requirements can be significantly reduced.

In Iris, when actual data values are needed from a lazy data array, it is
*'realised'* : this means that all the actual values are read in from the file,
and a 'real'
(i.e. `numpy <https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html>`_)
array replaces the lazy array within the Iris object.

Following realisation, the Iris object just contains the actual ('real')
data, so the time cost of reading all the data is not incurred again.
From here on, access to the data is fast, but it now occupies its full memory space.

In particular, any direct reference to a `cube.data` will realise the cube data
content : any lazy content is lost as the data is read from file, and the cube
content is replaced with a real array.
This is also referred to simply as "touching" the data.

See the section :ref:`when_real_data`
for more examples of this.

You can check whether a cube has real data or lazy data by using the method
:meth:`~iris.cube.Cube.has_lazy_data`. For example::

    >>> cube = iris.load_cube(iris.sample_data_path('air_temp.pp'))
    >>> cube.has_lazy_data()
    True
    # Realise the lazy data.
    >>> cube.data
    >>> cube.has_lazy_data()
    False


Benefits
--------

The primary advantage of using lazy data is that it enables
`out-of-core processing <https://en.wikipedia.org/wiki/Out-of-core_algorithm>`_;
that is, the loading and manipulating of datasets without loading the full data into memory.

There are two key benefits from this :

**Firstly**, the result of a calculation on a large dataset often occupies much
less storage space than the source data -- such as for instance a maximum data
value calculated over a large number of datafiles.
In these cases the result can be computed in sections, without ever requiring the
entire source dataset to be loaded, thus drastically reducing memory footprint.
This strategy of task division can also enable reduced execution time through the effective
use of parallel processing capabilities.

**Secondly**, it is often simply convenient to form a calculation on a large
dataset, of which only a certain portion is required at any one time
-- for example, plotting individual timesteps from a large sequence.
In such cases, a required portion can be extracted and realised without calculating the entire result.

.. _when_real_data:

When does my data become real?
------------------------------

Certain operations, such as cube indexing and statistics, can be
performed in a lazy fashion, producing a 'lazy' result from a lazy input, so
that no realisation immediately occurs.
However other operations, such as plotting or printing data values, will always
trigger the 'realisation' of data.

When you load a dataset using Iris the data array will almost always initially be
a lazy array. This section details some operations that will realise lazy data
as well as some operations that will maintain lazy data. We use the term **realise**
to mean converting lazy data into real data.

Most operations on data arrays can be run equivalently on both real and lazy data.
If the data array is real then the operation will be run on the data array
immediately. The results of the operation will be available as soon as processing is completed.
If the data array is lazy then the operation will be deferred and the data array will
remain lazy until you request the result (such as when you read from ``cube.data``)::

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

    >>> cube = iris.load_cube(iris.sample_data_path('hybrid_height.nc'), 'air_potential_temperature')

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

Iris uses dask to provide lazy data arrays for both Iris cubes and coordinates,
and for computing deferred operations on lazy arrays.

Dask provides processing options to control how deferred operations on lazy arrays
are computed. This is provided via the ``dask.set_options`` interface. See the
`dask documentation <http://dask.pydata.org/en/latest/scheduler-overview.html>`_
for more information on setting dask processing options.
