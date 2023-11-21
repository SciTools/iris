=============================
NetCDF I/O Handling in Iris
=============================

This document provides a basic account of how NetCDF files work within Iris.

This document is still a work in progress, so might include blank or unfinished sections,
watch this space!


Chunk Control
--------------

Chunks are, by default, optimised by iris on load. This will automatically
decide the best chunksize for your data without any user input.

For more user control, functionality was updated in :pull:`5588`, with the
creation of the CHUNK_CONTROL class.

There are three context manangers within CHUNK_CONTROL. The most basic is
CHUNK_CONTROL.set(). This allows you to specify the chunksize for each dimension,
and to specify a variable specifically to change.

Example:
    >>> import iris
    >>> from iris.file_formats.netcdf.loader import CHUNK_CONTROL
    >>> from iris import sample_data_path
    >>> with CHUNK_CONTROL.set('var1', model_level=1, time=50):
    ...     cubes = iris.load(sample_data_path("toa_brightness_stereographic.nc"))
    >>> print(cubes.chunksizes)
    <EXAMPLE_VALUES_HERE>

The second context manager is from_file(). This takes chunksizes as defined in
the NetCDF file. Any dimensions without specified chunks will default to iris optimisation.

The final context manager, as_dask(), bypasses Iris' optimisation all together, and
will take its chunksizes from dask's behaviour.


Split Attributes
----------------

TBC


Other Operations
----------------

TBC
