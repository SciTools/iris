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

#. `@hsteptoe <https://github.com/hsteptoe>`_ and `@...`_ (reviewer) extended :func:`iris.util.mask_cube_from_shapefile`
   to handle additional Point and Line shape types.  This change also facilitates the use of shapefiles that 
   use a different projection system to the cube that they are being applied to, and makes performance improvements
   to the mask weighting calculations. (:issue:`6126`, :pull:`6129`).  

🐛 Bugs Fixed
=============

#. `@trexfeathers`_ corrected the ESMF/ESMPy import in
   :mod:`iris.experimental.regrid_conservative` (the module was renamed to ESMPy
   in v8.4). Note that :mod:`~iris.experimental.regrid_conservative`
   is already deprecated and will be removed in a future release. (:pull:`6643`)


💣 Incompatible Changes
=======================

#. N/A


🚀 Performance Enhancements
===========================

#. N/A


🔥 Deprecations
===============

#. N/A


🔗 Dependencies
===============

#. `@hsteptoe <https://github.com/hsteptoe>`_ added `rasterio <https://rasterio.readthedocs.io/en/stable/index.html>`_ 
   and `affiine <https://affine.readthedocs.io/en/latest/>`_ as a dependency for :func:`iris.util.mask_cube_from_shapefile`. 
   This is to support the new functionality of handling additional shapefiles and projections. (:issue:`6126`, :pull:`6129`)

📚 Documentation
================

#. `@rcomer`_ updated all Cartopy references to point to the new location at
   https://cartopy.readthedocs.io (:pull:`6636`)

#. `@hsteptoe <https://github.com/hsteptoe>`_ added additional worked examples
   to the :ref:`iris.util.mask_cube_from_shapefile` documentation, to demonstrate
   how to use the function with different types of shapefiles. (:pull:`6129`)


💼 Internal
===========

#. `@trexfeathers`_ fixed benchmark result comparison to inspect the results
   for the current machine only. This is useful for setups where a single
   home-space is shared between multiple machines, as with some virtual desktop
   arrangements. (:pull:`6550`)

#. `@melissaKG`_ upgraded Iris' tests to no longer use the deprecated
   ``git whatchanged`` command. (:pull:`6672`)


.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:

.. _@melissaKG: https://github.com/melissaKG



.. comment
    Whatsnew resources in alphabetical order: