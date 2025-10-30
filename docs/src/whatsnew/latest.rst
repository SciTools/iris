.. include:: ../common_links.inc

|iris_version| |build_date| [unreleased]
****************************************

This document explains the changes made to Iris for this release
(:doc:`View all changes <index>`.)


.. dropdown:: |iris_version| Release Highlights
   :color: primary
   :icon: info
   :animate: fade-in
   :open:

   The highlights for this major/minor release of Iris include:

   * N/A

   And finally, get in touch with us on :issue:`GitHub<new/choose>` if you have
   any issues or feature requests for improving Iris. Enjoy!


📢 Announcements
================

#. N/A


✨ Features
===========

#. `@pp-mo`_ added a new utility function for making a test cube with a specified 2D
   horizontal grid.
   (:issue:`5770`, :pull:`6581`)

#. `@bjlittle`_ extended ``zlib`` compression of :class:`~iris.cube.Cube` data
   payload when saving to NetCDF to also include any attached `CF-UGRID`_
   :class:`~iris.mesh.components.MeshXY`. Additionally,
   :func:`~iris.fileformats.netcdf.saver.save_mesh` also supports ``zlib``
   compression. (:issue:`6565`, :pull:`6728`)

#. `@pp-mo`_ made it possible to save 'dataless' cubes to a netcdf file, and load them
   back again. (:issue:`6727`, :pull:`6739`)

#. `@ukmo-ccbunney`_ added a new :class:`~iris.util.CMLSettings` class to control
   the formatting of Cube CML output via a context manager.
   (:issue:`6244`, :pull:`6743`)

#. `@ESadek-MO`_ added functionality to allow :func:`~iris.cube.Cube.extract`,
   :func:`~iris.cube.Cube.collapsed`, :func:`~iris.cube.Cube.aggregated_by`,
   :func:`~iris.cube.Cube.convert_units`, :func:`~iris.cube.Cube.subset` and
   :func:`~iris.cube.Cube.slices` to work with dataless cubes.
   (:issue:`6725`, :pull:`6724`)


🐛 Bugs Fixed
=============

#. `@trexfeathers`_ corrected the ESMF/ESMPy import in
   :mod:`iris.experimental.regrid_conservative` (the module was renamed to ESMPy
   in v8.4). Note that :mod:`~iris.experimental.regrid_conservative`
   is already deprecated and will be removed in a future release. (:pull:`6643`)

#. `@rcomer`_ fixed a bug in merging cubes with cell measures or ancillary
   variables. The merged cube now has the cell measures and ancillary variables
   on the correct dimensions, and merge no longer fails when trying to add
   them to a dimension of the wrong length. (:issue:`2076`, :pull:`6688`)

#. `@bjlittle`_ added support for preserving masked auxiliary coordinates when
   using :meth:`~iris.cube.Cube.aggregated_by` or :meth:`~iris.cube.Cube.collapsed`.
   (:issue:`6473`, :pull:`6706`, :pull:`6719`)


💣 Incompatible Changes
=======================

#. N/A


🚀 Performance
==============

#. `@trexfeathers`_ investigated a significant performance regression in NetCDF
   loading and saving, caused by ``libnetcdf`` version ``4.9.3``.
   The regression is equal to several milliseconds per chunk
   of parallel operation; so a dataset containing ~100 chunks could be around
   0.5 seconds slower to load or save. This regression will NOT be fixed within
   Iris - doing so would introduce unacceptable complexity and potential
   concurrency problems. The regession has been reported to the NetCDF team; it
   is hoped that a future ``libnetcdf`` release will recover the original
   performance. See `netcdf-c#3183`_ for more details. (:pull:`6747`)

#. `@stephenworsley`_ made NetCDF loading more efficient by filtering variables
   before they become instantiated as cubes in the case where multiple name
   constraints are given. This was previously only implemented where one such
   constraint was given. (:issue:`6228`, :pull:`6754`)

#. `@stephenworsley`_ reduced the memory load for regridding and other operations
   using :func:`~iris._lazy_data.map_complete_blocks` when the output chunks would
   exceed the optimum chunksize set in dask. (:pull:`6730`)


🔥 Deprecations
===============

#. N/A


🔗 Dependencies
===============

#. N/A


📚 Documentation
================

#. `@rcomer`_ updated all Cartopy references to point to the new location at
   https://cartopy.readthedocs.io (:pull:`6636`)


💼 Internal
===========

#. `@trexfeathers`_ fixed benchmark result comparison to inspect the results
   for the current machine only. This is useful for setups where a single
   home-space is shared between multiple machines, as with some virtual desktop
   arrangements. (:pull:`6550`)

#. `@melissaKG`_ upgraded Iris' tests to no longer use the deprecated
   ``git whatchanged`` command. (:pull:`6672`)

#. `@ukmo-ccbunney`_ merged functionality of ``assert_CML_approx_data`` into
   ``assert_CML`` via the use of a new ``approx_data`` keyword. (:pull:`6713`)

#. `@ukmo-ccbunney`_ ``assert_CML`` now uses stricter array formatting to avoid
   changes in tests due to Numpy version changes. (:pull:`6743`)

#. `@stephenworsley`_ added a private switch :obj:`~iris.loading._CONCRETE_DERIVED_LOADING`
   for controlling laziness of coordinates from pp loading, avoiding a
   slowdown due to merging. Note: this object is temporary and is likely
   to be replaced by a permanent solution or else be renamed.
   (:issue:`6755`, :pull:`6767`)


.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:

.. _@melissaKG: https://github.com/melissaKG



.. comment
    Whatsnew resources in alphabetical order:

.. _netcdf-c#3183: https://github.com/Unidata/netcdf-c/issues/3183
