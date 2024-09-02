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

#. N/A


🐛 Bugs Fixed
=============

#. N/A

#. `@rcomer`_ enabled partial collapse of multi-dimensional string coordinates,
   fixing :issue:`3653`. (:pull:`5955`)


💣 Incompatible Changes
=======================

#. N/A


🚀 Performance Enhancements
===========================

#. `@bouweandela`_ made the time coordinate categorisation functions in
   :mod:`~iris.coord_categorisation` faster. Anyone using
   :func:`~iris.coord_categorisation.add_categorised_coord`
   with cftime :class:`~cftime.datetime` objects can benefit from the same
   improvement by adding a type hint to their category funcion. (:pull:`5999`)

#. `@bouweandela`_ made :meth:`iris.cube.CubeList.concatenate` faster if more
   than two cubes are concatenated. (:pull:`5926`)

🔥 Deprecations
===============

#. N/A


🔗 Dependencies
===============

#. N/A


📚 Documentation
================

#. N/A


💼 Internal
===========

#. `@trexfeathers`_ improved the new ``tracemalloc`` benchmarking (introduced
   in Iris v3.10.0, :pull:`5948`) to use the same statistical repeat strategy
   as timing benchmarks. (:pull:`5981`)


.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:




.. comment
    Whatsnew resources in alphabetical order:
