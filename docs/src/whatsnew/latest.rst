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

#. `@rcomer`_ rewrote :func:`~iris.util.broadcast_to_shape` so it now handles
   lazy data. (:pull:`5307`)
   
#. `@acchamber`_   added error and warning messages about coordinate overlaps to 
   :func:`~iris.cube.concatenate` to improve the concatenation process. (:pull:`5382`)

#. `@trexfeathers`_ included mesh location coordinates
   (e.g. :attr:`~iris.experimental.ugrid.Mesh.face_coords`) in
   the data variable's ``coordinates`` attribute when saving to NetCDF.
   (:issue:`5206`, :pull:`5389`)

#. `@pp-mo`_ modified the install process to record the release version of the CF
   standard-names table, when it creates the ``iris/std_names.py`` module.
   The release number is also now available as
   ``iris.std_names.CF_STANDARD_NAMES_TABLE_VERSION``.
   (:pull:`5423`)


🐛 Bugs Fixed
=============

#. `@acchamber`_ fixed a bug with :func:`~iris.util.unify_time_units` so it does not block
   concatenation through different data types in rare instances. (:pull:`5372`) 

#. `@acchamber`_ removed some obsolete code that prevented extraction of time points
   from cubes with bounded times (:pull:`5175`)
   
#. `@rcomer`_ modified pp-loading to avoid a ``cftime`` warning for non-standard
   calendars. (:pull:`5357`)


💣 Incompatible Changes
=======================

#. N/A


🚀 Performance Enhancements
===========================

#. `@rcomer`_ made :meth:`~iris.cube.Cube.aggregated_by` faster. (:pull:`4970`)
#. `@rsdavies`_ modified the CF compliant standard name for m01s00i023 (:issue:`4566`)

🔥 Deprecations
===============

#. N/A


🔗 Dependencies
===============

#. N/A


📚 Documentation
================

#. `@tkknight`_ prepared the documentation for dark mode and enable the option
   to use it.  By default the theme will be based on the users system settings,
   defaulting to ``light`` if no system setting is found.  (:pull:`5299`)

#. `@HGWright`_ added a :doc:`/further_topics/dask_best_practices/index`
   section into the user guide, containing advice and use cases to help users
   get the best out of Dask with Iris.  (:pull:`5190`)

#. `@acchamber`_ improved documentation for :meth:`~iris.cube.Cube.convert_units`
   and :meth:`~iris.coords.Coord.convert_units` by including a link to the UDUNITS-2
   documentation which contains lists of compatible units and aliases for them.
  (:pull:`5388`)


#. `@rcomer`_ updated the :ref:`Installation Guide<installing_iris>` to reflect
   that some things are now simpler.  (:pull:`5416`)

💼 Internal
===========

#. `@pp-mo`_ supported loading and saving netcdf :class:`netCDF4.Dataset` compatible
   objects in place of file-paths, as hooks for a forthcoming
   `"Xarray bridge" <https://github.com/SciTools/iris/issues/4994>`_ facility.
   (:pull:`5214`, :pull:`5212`)

#. `@rcomer`_ updated :func:`~iris.plot.contourf` to avoid using functionality
   that is deprecated in Matplotlib v3.8 (:pull:`5405`)



.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:
.. _@rsdavies: https://github.com/rsdavies
.. _@acchamber: https://github.com/acchamber



.. comment
    Whatsnew resources in alphabetical order:
