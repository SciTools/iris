.. _examples_pp_to_ff:

1. Speed up Converting PP Files to NetCDF
-----------------------------------------

Here is an example of how dask objects can be tuned for better performance.

1.1 The Problem - Slow Saving
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We have ~300 PP files which we load as follows:

.. code-block:: python

    import iris
    import glob

    files = glob.glob("pp_files/*.pp")
    cube = iris.load_cube(files, "mass_fraction_of_ozone_in_air")

Note that loading here may also be parallelised in a similar manner as
described in :ref:`examples_bags_greed`. Either way, the resulting cube looks
as follows:

.. code-block:: text

    mass_fraction_of_ozone_in_air / (kg kg-1) (time: 276; model_level_number: 85; latitude: 144; longitude: 192)
         Dimension coordinates:
              time                                 x                        -             -               -
              model_level_number                   -                        x             -               -
              latitude                             -                        -             x               -
              longitude                            -                        -             -               x
         Auxiliary coordinates:
              forecast_period                      x                        -             -               -
              level_height                         -                        x             -               -
              sigma                                -                        x             -               -
         Scalar coordinates:
              forecast_reference_time: 1850-01-01 00:00:00
         Attributes:
              STASH: m01s34i001
              source: Data from Met Office Unified Model
              um_version: 10.9
         Cell methods:
              mean: time (1 hour)

The cube is then immediately saved as a netCDF file.

.. code-block:: python

    nc_chunks = [chunk[0] for chunk in cube.lazy_data().chunks]
    iris.save(cube, "outfile.nc", nc_chunks)

This operation was taking longer than expected and we would like to improve
performance. Note that when this cube is being saved, the data is still lazy,
data is both read and written at the saving step and is done so in chunks.
The way this data is divided into chunks can affect performance. By tweaking
the way these chunks are structured it may be possible to improve performance
when saving.


.. _dask_rechunking:

1.2 Rechunking
^^^^^^^^^^^^^^
We may inspect the cube's lazy data before saving:

.. code-block:: python

    # We can access the cubes Dask array
    lazy_data = cube.lazy_data()
    # We can find the shape of the chunks
    # Note that the chunksize of a Dask array is the shape of the chunk
    # as a tuple.
    print(lazy_data.chunksize)

Doing so, we find that the chunks currently have the shape::

(1, 1, 144, 192)

This is significantly smaller than the `size which Dask recommends
<https://docs.dask.org/en/latest/array-chunks.html>`_. Bear in mind that the
ideal chunk size depends on the platform you are running on (for this example,
the code is being run on a desktop with 8 CPUs). In this case, we have 23460
small chunks. We can reduce the number of chunks by rechunking before saving:

.. code-block:: python

    lazy_data = cube.lazy_data()
    lazy_data = lazy_data.rechunk(1, 85, 144, 192)
    cube.data = lazy_data

We now have 276 moderately sized chunks. When we try saving again, we find
that it is approximately 4 times faster, saving in 2m13s rather than 10m33s.
