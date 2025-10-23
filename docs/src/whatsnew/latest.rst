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


🔥 Deprecations
===============

#. N/A


🔗 Dependencies
===============

#. `@hsteptoe`_ added `rasterio`_ and `affine`_ as optional dependencies that facilitate
   :func:`iris.util.mask_cube_from_shape`. These packages support new functionality that 
   handles additional shapefile types and projections. (:issue:`6126`, :pull:`6129`)

📚 Documentation
================

#. `@rcomer`_ updated all Cartopy references to point to the new location at
   https://cartopy.readthedocs.io (:pull:`6636`)

#. `@hsteptoe`_ added additional worked examples to the :ref:`iris.util.mask_cube_from_shape` 
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


.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:

.. _@hsteptoe: https://github.com/hsteptoe
.. _@melissaKG: https://github.com/melissaKG



.. comment
    Whatsnew resources in alphabetical order:

<<<<<<< Updated upstream
.. _netcdf-c#3183: https://github.com/Unidata/netcdf-c/issues/3183
=======
.. _affine: https://affine.readthedocs.io/en/latest/
.. _netcdf-c#3183: https://github.com/Unidata/netcdf-c/issues/3183
.. _rasterio: https://rasterio.readthedocs.io/en/stable/index.html
>>>>>>> Stashed changes
