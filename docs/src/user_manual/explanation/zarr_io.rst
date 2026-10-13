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
names the package and the version floor if you ask it to load a store.

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
whole file. A Zarr dimension name is weaker in three ways, and all three are
permitted by the version 3 specification:

* an array need not name its dimensions at all;
* it may name some and not others;
* the same name in two arrays guarantees nothing -- not even the same length.

So Iris matches dimensions on the pair *(name, length)*. Two arrays share a
dimension when they name it identically **and** it is the same length in
both. A dimension with no name is **anonymous**, and never matches anything;
the cube simply has a dimension with no coordinate on it, which is an
ordinary Iris cube. A dimension name used at two different lengths is
reported with a warning and links nothing.

Version 2 stores have no dimension-name field, and carry names in an
``_ARRAY_DIMENSIONS`` attribute instead -- the convention xarray established.
Iris reads it, and does not pass it on to the cube.

Storage bookkeeping is not a CF attribute
-----------------------------------------

Two attributes never reach ``cube.attributes``, because they describe the
store rather than the data: ``_ARRAY_DIMENSIONS``, covered above, and
``_NCProperties``, the provenance stamp netCDF-c's NCZarr writer stamps onto
the root group of every store it writes. netCDF4-python has always hidden
``_NCProperties`` from a cube loaded through netCDF, and Iris hides it on the
Zarr path too, for the same reason. A cube loaded from a version 2 store and
the same cube loaded from a version 3 store therefore carry the same
attributes.

Missing data
------------

Zarr applies no masking of its own, so Iris does it, from the CF attributes:
``_FillValue`` and ``missing_value`` for sentinels,
``valid_min``/``valid_max``/``valid_range`` for a range, and
``scale_factor``/``add_offset`` for packed data.

A Zarr array also has a ``fill_value`` in its metadata. This is **not** a
missing-value declaration: it says what a chunk that was never written reads
back as, and zarr-python defaults it to zero. Iris therefore ignores it as a
mask source on version 3, where it is a required field, and uses it on
version 2 only when the array declares no CF attribute -- which is what
version 2 writers meant by it, and what xarray does.

A version 2 store that declares neither ``_FillValue`` nor ``missing_value``
therefore has its storage ``fill_value`` treated as missing data, so a value
equal to it is masked -- including a genuine zero, since zarr-python defaults
``fill_value`` to zero on every array it creates, version 2 included. Iris
warns when this happens, naming the variable and the value. Setting a
``_FillValue`` attribute on the array prevents it.

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

zarr-python fetches chunks concurrently, under its own ``zarr.config``
setting ``async.concurrency``. Iris does not change it. Dask's worker count
and that concurrency multiply, so a large Dask pool over a remote store can
open far more connections than either number suggests; if you are
rate-limited, lower one of them.

Attributes
----------

A Zarr attribute value may be any JSON literal, including a nested object or
an array of arrays, which CF cannot express at any rank. Iris carries such a
value onto the cube unchanged and does not interpret it. See
:issue:`7288` for the wider question of how the CF attribute model should map
onto JSON.

.. note::

    A cube carrying such an attribute **cannot be saved as netCDF**, because
    netCDF attribute values are scalars, strings and one-dimensional arrays.
    The failure comes from the netCDF library, as a ``TypeError`` naming the
    attribute::

        TypeError: illegal data type for attribute b'stac_discovery', ...

    It does not name the cube; see :issue:`7329`. Delete the attribute
    before saving::

        del cube.attributes["stac_discovery"]

.. _zarr_io_nczarr:

NCZarr
------

Iris also reads and writes Zarr through netCDF-c's NCZarr driver, reached by
way of a URL fragment::

    iris.load("file:///data/forecast.zarr#mode=nczarr,file")

NCZarr implements Zarr version 2, and writes scalar variables as length-one
arrays. The native reader described above handles both versions, and is
what a plain path or URL -- one with no ``#mode=`` fragment -- reaches.
