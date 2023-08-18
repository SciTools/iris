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


💣 Incompatible Changes
=======================

#. N/A


🚀 Performance Enhancements
===========================

#. `@rcomer`_ made :meth:`~iris.cube.Cube.aggregated_by` faster. (:pull:`4970`)
#. `@rsdavies`_ modified the CF compliant standard name for m01s00i023 :issue:`4566`

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


💼 Internal
===========

#. `@pp-mo`_ supported loading and saving netcdf :class:`netCDF4.Dataset` compatible
   objects in place of file-paths, as hooks for a forthcoming
   `"Xarray bridge" <https://github.com/SciTools/iris/issues/4994>`_ facility.
   (:pull:`5214`)

#. `@trexfeathers`_ refactored benchmarking scripts to better support on-demand
   benchmarking of pull requests. Results are now posted as a new comment.
   (feature branch: :pull:`5430`)

#. `@trexfeathers`_ changed pull request benchmarking to compare: GitHub's
   simulated merge-commit (which it uses for all PR CI by default) versus the
   merge target (e.g. `main`). This should provide the most 'atomic' account
   of how the pull request changes affect performance. (feature branch:
   :pull:`5431`)

#. `@trexfeathers`_ added a catch to the overnight benchmark workflow to raise
   an issue if the overnight run fails - this was previously an 'invisible'
   problem. (feature branch: :pull:`5432`)

#. `@trexfeathers`_ set `bm_runner.py` to error when the called processes
   error. This fixes an oversight introduced in :pull:`5215`. (feature branch:
   :pull`5434`)

#. `@trexfeathers`_ inflated some benchmark data sizes to compensate for
   :pull:`5229`. (feature branch: :pull:`5436`)


.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:
.. _@rsdavies: https://github.com/rsdavies



.. comment
    Whatsnew resources in alphabetical order:
