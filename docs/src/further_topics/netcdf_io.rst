.. testsetup:: chunk_control

    import iris
    from iris.fileformats.netcdf.loader import CHUNK_CONTROL

    from pathlib import Path
    import dask
    import shutil
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_filepath = tmp_dir / "tmp.nc"

    cube = iris.load(iris.sample_data_path("E1_north_america.nc"))[0]
    iris.save(cube, tmp_filepath, chunksizes=(120, 37, 49))
    old_dask = dask.config.get("array.chunk-size")
    dask.config.set({'array.chunk-size': '500KiB'})


.. testcleanup:: chunk_control

    dask.config.set({'array.chunk-size': old_dask})
    shutil.rmtree(tmp_dir)

.. _netcdf_io:

=============================
NetCDF I/O Handling in Iris
=============================

This document provides a basic account of how Iris loads and saves NetCDF files.

.. admonition:: Under Construction

    This document is still a work in progress, so might include blank or unfinished sections,
    watch this space!


Chunk Control
--------------

Default Chunking
^^^^^^^^^^^^^^^^

Chunks are, by default, optimised by Iris on load. This will automatically
decide the best chunksize for your data without any user input. This is
calculated based on a number of factors, including:

- File Variable Chunking
- Full Variable Shape
- Dask Default Chunksize
- Dimension Order: Earlier (outer) dimensions will be prioritised to be split over later (inner) dimensions.

.. doctest:: chunk_control

    >>> cube = iris.load_cube(tmp_filepath)
    >>>
    >>> print(cube.shape)
    (240, 37, 49)
    >>> print(cube.core_data().chunksize)
    (60, 37, 49)

For more user control, functionality was updated in :pull:`5588`, with the
creation of the :data:`iris.fileformats.netcdf.loader.CHUNK_CONTROL` class.

Custom Chunking: Set
^^^^^^^^^^^^^^^^^^^^

There are three context managers within :data:`~iris.fileformats.netcdf.loader.CHUNK_CONTROL`. The most basic is
:meth:`~iris.fileformats.netcdf.loader.ChunkControl.set`. This allows you to specify the chunksize for each dimension,
and to specify a ``var_name`` specifically to change.

Using ``-1`` in place of a chunksize will ensure the chunksize stays the same
as the shape, i.e. no optimisation occurs on that dimension.

.. doctest:: chunk_control

    >>> with CHUNK_CONTROL.set("air_temperature", time=180, latitude=-1, longitude=25):
    ...     cube = iris.load_cube(tmp_filepath)
    >>>
    >>> print(cube.core_data().chunksize)
    (180, 37, 25)

Note that ``var_name`` is optional, and that you don't need to specify every dimension. If you
specify only one dimension, the rest will be optimised using Iris' default behaviour.

.. doctest:: chunk_control

    >>> with CHUNK_CONTROL.set(longitude=25):
    ...     cube = iris.load_cube(tmp_filepath)
    >>>
    >>> print(cube.core_data().chunksize)
    (120, 37, 25)

Custom Chunking: From File
^^^^^^^^^^^^^^^^^^^^^^^^^^

The second context manager is :meth:`~iris.fileformats.netcdf.loader.ChunkControl.from_file`.
This takes chunksizes as defined in the NetCDF file. Any dimensions without specified chunks
will default to Iris optimisation.

.. doctest:: chunk_control

    >>> with CHUNK_CONTROL.from_file():
    ...     cube = iris.load_cube(tmp_filepath)
    >>>
    >>> print(cube.core_data().chunksize)
    (120, 37, 49)

Custom Chunking: As Dask
^^^^^^^^^^^^^^^^^^^^^^^^

The final context manager, :meth:`~iris.fileformats.netcdf.loader.ChunkControl.as_dask`, bypasses
Iris' optimisation all together, and will take its chunksizes from Dask's behaviour.

.. doctest:: chunk_control

    >>> with CHUNK_CONTROL.as_dask():
    ...    cube = iris.load_cube(tmp_filepath)
    >>>
    >>> print(cube.core_data().chunksize)
    (70, 37, 49)


Split Attributes
-----------------

TBC


Deferred Saving
----------------

TBC


Guessing Coordinate Axes
------------------------

Iris will attempt to add an ``axis`` attribute when saving any coordinate
variable in a NetCDF file. E.g:

::

    float longitude(longitude) ;
        longitude:axis = "X" ;

This is achieved by calling :func:`iris.util.guess_coord_axis` on each
coordinate being saved.

Disabling Axis-Guessing
^^^^^^^^^^^^^^^^^^^^^^^

For some coordinates, :func:`~iris.util.guess_coord_axis` will derive an
axis that is not appropriate. If you have such a coordinate, you can disable
axis-guessing by setting the coordinate's
:attr:`~iris.coords.Coord.ignore_axis` property to ``True``.

One example (from https://github.com/SciTools/iris/issues/5003) is a
coordinate describing pressure thresholds, measured in hecto-pascals.
Iris interprets pressure units as indicating a Z-dimension coordinate, since
pressure is most commonly used to describe altitude/depth. But a
**pressure threshold** coordinate is instead describing alternate
**scenarios** - not a spatial dimension at all - and it is therefore
inappropriate to assign an axis to it.

Worked example:

.. doctest::

    >>> from iris.coords import DimCoord
    >>> from iris.util import guess_coord_axis
    >>> my_coord = DimCoord(
    ...    points=[1000, 1010, 1020],
    ...    long_name="pressure_threshold",
    ...    units="hPa",
    ... )
    >>> print(guess_coord_axis(my_coord))
    Z
    >>> my_coord.ignore_axis = True
    >>> print(guess_coord_axis(my_coord))
    None
