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

#. `@pp-mo`_ added the :func:`~iris.util.make_gridcube` utility function, for making a
   dataless test-cube with a specified 2D horizontal grid.
   (:issue:`5770`, :pull:`6581`, :pull:`6741`)

#. `@hsteptoe`_ and `@trexfeathers`_ (reviewer) added :func:`iris.util.mask_cube_from_shape`
   to handle additional Point and Line shape types.  This change also facilitates the use of
   shapefiles that use a different projection system to the cube that they are being applied to, 
   and makes performance improvements to the mask weighting calculations. 
   (:issue:`6126`, :pull:`6129`).
   
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

#. `@pp-mo`_ added the ability to merge dataless cubes.  This also means they can be
   re-loaded normally with :meth:`iris.load`.  See: :ref:`dataless_merge`.
   Also added a new documentation section on dataless cubes.
   (:issue:`6740`, :pull:`6741`)

#. `@trexfeathers`_ and `@jrackham-mo`_ added support for lazy calculation in
   :func:`iris.analysis.calculus.cube_delta` (used in the
   :func:`~iris.analysis.calculus.differentiate` function). (:issue:`6734`,
   :pull:`6772`)


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

#. `@trexfeathers`_ protected the NetCDF saving code from a transient I/O
   error, caused by bad synchronisation between Python-layer and HDF-layer
   file locking on certain filesystems. (:pull:`6760`).


💣 Incompatible Changes
=======================

#. Existing users of :func:`iris.util.mask_cube_from_shapefile` will need to
   install the additional dependencies `rasterio`_ and `affine`_ to continue
   using this function. These dependencies are necessary to support bug fixes 
   implemented in (:issue:`6126`, :pull:`6129`).  Note that this function will 
   be deprecated in a future version of Iris in favour of the new 
   :func:`iris.util.mask_cube_from_shape`, which offers richer shape handling.


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

#. `@hsteptoe`_ added `rasterio`_ and `affine`_ as optional dependencies that facilitate
   :func:`iris.util.mask_cube_from_shape`. These packages support new functionality that 
   handles additional shapefile types and projections. (:issue:`6126`, :pull:`6129`)

#. `@pp-mo`_ added a temporary dependency pins for Python<3.14, dask<2025.10.0 and
   netCDF4<1.7.3.  All of these introduce problems that won't necessarily be fixed soon,
   so we anticipate that these pins will be wanted for the v3.14 release.
   (:issue:`6775`, :issue:`6776`, :issue:`6777`, :pull:`6773`)


📚 Documentation
================

#. `@rcomer`_ updated all Cartopy references to point to the new location at
   https://cartopy.readthedocs.io (:pull:`6636`)

#. `@hsteptoe`_ added additional worked examples to the :func:`iris.util.mask_cube_from_shape` 
   documentation, to demonstrate how to use the function with different types of shapefiles. 
   (:pull:`6129`)


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

.. _@hsteptoe: https://github.com/hsteptoe
.. _@melissaKG: https://github.com/melissaKG



.. comment
    Whatsnew resources in alphabetical order:

.. _affine: https://affine.readthedocs.io/en/latest/
.. _netcdf-c#3183: https://github.com/Unidata/netcdf-c/issues/3183
.. _rasterio: https://rasterio.readthedocs.io/en/stable/index.html
