.. explanation:: Zarr I/O Handling in Iris
   :tags: topic_load_save

   Read about how Iris loads Zarr stores.

.. include:: ../../common_links.inc

.. _zarr_io:

=========================
Zarr I/O Handling in Iris
=========================

.. readingtime::

`Zarr <https://zarr.dev/>`_ stores an array as many independently compressed
chunks under a prefix, rather than as one file. Iris reads Zarr version 2 and
version 3 stores natively, through
`zarr-python <https://zarr.readthedocs.io/>`_.

``zarr`` is an **optional dependency**, and Iris requires at least version
3.0.8. Install it to load Zarr stores::

    conda install -c conda-forge "zarr>=3.0.8"

Iris without it installs and runs normally, raising an ``ImportError`` that
names the package and its minimum required version if you ask it to load a
store.

.. _zarr_io_loading:

Loading
-------

A store is a directory, or a URL, so pass it where you would pass a file::

    cubes = iris.load("/data/forecast.zarr")

Iris recognises a local store by the metadata file at its root -- ``zarr.json``
for version 3, ``.zgroup`` for version 2 -- not by the ``.zarr`` suffix, so a
directory that merely has the suffix is not mistaken for one.

Saving
------

Iris does not save to a native Zarr store. ``iris.save(cube, "out.zarr")``
raises ``ValueError: Cannot save; no saver``, naming neither Zarr nor an
alternative. Save to netCDF instead, or write Zarr through netCDF-c's NCZarr
driver, covered below in :ref:`zarr_io_nczarr`.

A cube loaded from Zarr may carry an attribute netCDF cannot hold, which
fails the save. See :ref:`zarr_io_attributes`.

Remote stores
-------------

A URL with one of the ``s3``, ``gs``, ``az`` or ``abfs`` schemes, ending in
``.zarr``, is passed to zarr-python unaltered::

    cubes = iris.load("s3://noaa-gfs-bdp-pds/forecast.zarr")

Iris adds no store library of its own and handles no credentials. Install the
one your scheme needs -- ``s3fs``, ``gcsfs``, ``adlfs`` -- and configure it as
that package documents. zarr-python raises an ``ImportError`` naming the
package if it is missing.

Groups
------

A Zarr store may hold several groups, and they need not describe the same
thing. Iris loads **one** group per call, like xarray, rather than merging
them: ``iris.load`` reads the root group, and a store whose root holds no data
variables raises a message naming the groups that do.

To read another group, call the Zarr loader directly::

    from iris.cube import CubeList
    import iris.fileformats.zarr

    cubes = CubeList(
        iris.fileformats.zarr.load_cubes(
            "/data/sentinel.zarr", group="measurements/reflectance/r10m"
        )
    )

Dimension names
---------------

A netCDF dimension has a name, one length, and a meaning shared across the
whole file. A Zarr dimension name promises less. An array need not name its
dimensions at all, and the same name in two arrays guarantees nothing -- not
even the same length. Both are permitted by the specification.

So Iris treats two arrays as sharing a dimension only when they name it
identically **and** it is the same length in both. Anything else loads as an
**anonymous** dimension, which is an ordinary Iris cube dimension with no
coordinate on it. That covers both a dimension the store never named, and a
name the store used at two different lengths.

A coordinate left with nothing to attach to is not discarded. It goes to
:data:`iris.loading.LOAD_PROBLEMS`, and the load warns that not every object
was parsed.

Version 2 stores have no dimension-name field. They carry names in an
``_ARRAY_DIMENSIONS`` attribute instead -- the convention xarray established
-- which Iris reads, so dimensions are named on both versions. The attribute
describes the store rather than the data, so it does not itself reach
``cube.attributes``.

Missing data
------------

Zarr returns a plain array and applies no masking, so Iris does it, from the
CF attributes: ``_FillValue`` and ``missing_value`` for sentinels,
``valid_min``/``valid_max``/``valid_range`` for a range, and
``scale_factor``/``add_offset`` for packed data.

A Zarr array also carries a ``fill_value`` in its metadata. That says what
unwritten storage reads back as, not which values are missing, and
zarr-python defaults it to zero. Iris reads it as missing data only on a
version 2 store whose array declares no CF attribute -- which is what version
2 writers meant by it, and what xarray does.

That one case can mask a genuine zero, so Iris warns when it happens, naming
the variable. Setting a ``_FillValue`` attribute on the array prevents both
the masking and the warning.

Chunking
--------

Iris chooses Dask chunks from the store's own chunking, as it does for
netCDF, and :data:`~iris.fileformats.cf.loader.CHUNK_CONTROL` works the same
way for both formats.

One rule is specific to Zarr. A Zarr chunk is the atomic unit of storage:
reading any part of one fetches and decompresses the whole object. So Iris
rounds the computed chunking **up** to a whole multiple of the store's
chunking, never below it. When a single store chunk is already larger than
Dask's target chunk size, the store's layout is the binding constraint on
memory per task, and Iris warns to say so.

A **sharded** array is chunked on its inner chunks, not its shards. Zarr's
sharding codec is indexed, so a reader fetches only the inner chunks it
needs.

zarr-python fetches chunks concurrently, and Dask's worker count multiplies
with that concurrency. If a remote store rate-limits you, lower either Dask's
worker count or zarr-python's own ``zarr.config`` setting
``async.concurrency``.

.. _zarr_io_attributes:

Attributes
----------

A Zarr attribute value may be any JSON literal, including many things CF
cannot express, such as multidimensional arrays, dictionaries or lists of
lists. Iris carries such a value onto the cube unchanged and does not
interpret it. See :issue:`7288` for the wider question of how the CF
attribute model should map onto JSON.

.. note::

    A cube carrying such an attribute **cannot be saved as netCDF**, because
    netCDF attribute values are scalars, strings and one-dimensional arrays.
    The netCDF library raises ``TypeError: illegal data type for attribute
    b'stac_discovery'``, which does not name the cube. Delete the attribute
    before saving::

        del cube.attributes["stac_discovery"]

    See :issue:`7329`.

.. _zarr_io_nczarr:

NCZarr
------

Iris also reads and writes Zarr through netCDF-c's NCZarr driver, reached by
way of a URL fragment::

    iris.load("file:///data/forecast.zarr#mode=nczarr,file")

NCZarr implements Zarr version 2, and writes scalar variables as length-one
arrays. The native reader described above handles both versions, and is
what a plain path or URL -- one with no ``#mode=`` fragment -- reaches.
