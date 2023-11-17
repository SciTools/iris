.. _dask best practices:

Dask Best Practices
*******************

This section outlines some of the best practices when using Dask with Iris. These
practices involve improving performance through rechunking, making the best use of
computing clusters and avoiding parallelisation conflicts between Dask and NumPy.


.. note::

    Here, we have collated advice and a handful of examples, from the topics most
    relevant when using Dask with Iris, that we hope will assist users to make
    the best start when using Dask. It is *not* a fully comprehensive guide
    encompassing all best practices. You can find more general dask information in the
    `official Dask Documentation <https://docs.dask.org/en/stable/>`_.


Introduction
============

`Dask <https://dask.org/>`_ is a powerful tool for speeding up data handling
via lazy loading and parallel processing. To get the full benefit of using
Dask, it is important to configure it correctly and supply it with
appropriately structured data. For example, we may need to "chunk" data arrays
into smaller pieces to process, read and write it; getting the "chunking" right
can make a significant different to performance!


.. _numpy_threads:

NumPy Threads
=============

In certain scenarios NumPy will attempt to perform threading using an
external library - typically OMP, MKL or openBLAS - making use of **every**
CPU available. This interacts badly with Dask:

* Dask may create multiple instances of NumPy, each generating enough
  threads to use **all** the available CPUs. The resulting sharing of CPUs
  between threads greatly reduces performance. The more cores there are, the
  more pronounced this problem is.
* NumPy will generate enough threads to use all available CPUs even
  if Dask is deliberately configured to only use a subset of CPUs. The
  resulting sharing of CPUs between threads greatly reduces performance.
* `Dask is already designed to parallelise with NumPy arrays <https://docs
  .dask.org/en/latest/array.html>`_, so adding NumPy's 'competing' layer of
  parallelisation could cause unpredictable performance.

Therefore it is best to prevent NumPy performing its own parallelisation, `a
suggestion made in Dask's own documentation <https://docs.dask
.org/en/stable/array-best-practices.html#avoid-oversubscribing-threads>`_.
The following commands will ensure this in all scenarios:

in Python...

::

    # Must be run before importing NumPy.
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

or in Linux command line...

::

    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1


.. _multi-pro_systems:

Dask on Computing Clusters
==========================

Dask is well suited for use on computing clusters, but there are some important factors you must be
aware of. In particular, you will always need to explicitly control parallel
operation, both in Dask and likewise in NumPy.


.. _multi-pro_slurm:

CPU Allocation
--------------

When running on a computing cluster, unless configured otherwise, Dask will attempt to create
one parallel 'worker' task for each CPU. However, when using a job scheduler such as Slurm, only *some* of
these CPUs are actually accessible -- often, and by default, only one. This leads to a serious
over-commitment unless it is controlled.

So, **whenever Iris is used on a computing cluster, you must always control the number
of dask workers to a sensible value**, matching the slurm allocation.  You do
this with::

    dask.config.set(num_workers=N)

For an example, see :doc:`dask_bags_and_greed`.

Alternatively, when there is only one CPU allocated, it may actually be more
efficient to use a "synchronous" scheduler instead, with::

    dask.config.set(scheduler='synchronous')

See the Dask documentation on `Single thread synchronous scheduler
<https://docs.dask.org/en/latest/scheduling.html?highlight=single-threaded#single-thread>`_.


.. _multi-pro_numpy:

NumPy Threading
---------------

NumPy also interrogates the visible number of CPUs to multi-thread its operations.
The large number of CPUs available in a computing cluster will thus cause confusion if NumPy
attempts its own parallelisation, so this must be prevented. Refer back to
:ref:`numpy_threads` for more detail.


Distributed
-----------

Even though allocations on a computing cluster are generally restricted to a single node, there
are still good reasons for using 'dask.distributed' in many cases. See `Single Machine: dask.distributed
<https://docs.dask.org/en/latest/setup/single-distributed.html>`_ in the Dask documentation.


Chunking
========

Dask breaks down large data arrays into chunks, allowing efficient
parallelisation by processing several smaller chunks simultaneously. For more
information, see the documentation on
`Dask Array <https://docs.dask.org/en/latest/array.html>`_.

Iris provides a basic chunking shape to Dask, attempting to set the shape for
best performance. The chunking that is used can depend on the file format that
is being loaded. See below for how chunking is performed for:

* :ref:`chunking_netcdf`
* :ref:`chunking_pp_ff`

It can in some cases be beneficial to re-chunk the arrays in Iris cubes.
For information on how to do this, see :ref:`dask_rechunking`.


.. _chunking_netcdf:

NetCDF Files
------------

NetCDF files can include their own chunking specification. This is either
specified when creating the file, or is automatically assigned if one or
more of the dimensions is `unlimited <https://www.unidata.ucar
.edu/software/netcdf/docs/unlimited_dims.html>`_.
Importantly, netCDF chunk shapes are **not optimised for Dask
performance**.

Chunking can be set independently for any variable in a netCDF file.
When a netCDF variable uses an unlimited dimension, it is automatically
chunked: the chunking is the shape of the whole variable, but with '1' instead
of the length in any unlimited dimensions.

When chunking is specified for netCDF data, Iris will set the dask chunking
to an integer multiple or fraction of that shape, such that the data size is
near to but not exceeding the dask array chunk size.


.. _chunking_pp_ff:

PP and Fieldsfiles
------------------

PP and Fieldsfiles contain multiple 2D fields of data. When loading PP or
Fieldsfiles into Iris cubes, the chunking will automatically be set to a chunk
per field.

For example, if a PP file contains 2D lat-lon fields for each of the
85 model level numbers, it will load in a cube that looks as follows::

    (model_level_number: 85; latitude: 144; longitude: 192)

The data in this cube will be partitioned with chunks of shape
:code:`(1, 144, 192)`.

If the file(s) being loaded contain multiple fields, this can lead to an
excessive amount of chunks which will result in poor performance.

When the default chunking is not appropriate, it is possible to rechunk.
:doc:`dask_pp_to_netcdf` provides a detailed demonstration of how Dask can optimise
that process.


Examples
========

We have written some examples of use cases for using Dask, that come with advice and
explanations for why and how the tasks are performed the way they are.

If you feel you have an example of a Dask best practice that you think may be helpful to others,
please share them with us by raising a new `discussion on the Iris repository <https://github.com/SciTools/iris
/discussions/>`_.

* :doc:`dask_pp_to_netcdf`
* :doc:`dask_parallel_loop`
* :doc:`dask_bags_and_greed`

.. toctree::
   :hidden:
   :maxdepth: 1

   dask_pp_to_netcdf
   dask_parallel_loop
   dask_bags_and_greed
