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

#. `@trexfeathers`_ and `@HGWright`_ (reviewer) sub-categorised all Iris'
   :class:`UserWarning`\s for richer filtering. The full index of
   sub-categories can be seen here: :mod:`iris.exceptions` . (:pull:`5498`)


🐛 Bugs Fixed
=============

#. `@scottrobinson02`_ fixed the output units when dividing a coordinate by a
   cube. (:issue:`5305`, :pull:`5331`)

#. `@ESadek-MO`_ has updated :mod:`iris.tests.graphics.idiff` to stop duplicated file names
   preventing acceptance. (:issue:`5098`, :pull:`5482`)


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

#. N/A


📚 Documentation
================

#. N/A


💼 Internal
===========

#. `@trexfeathers`_ and `@ESadek-MO`_ (reviewer) performed a suite of fixes and
   improvements for benchmarking, primarily to get
   :ref:`on demand pull request benchmarking <on_demand_pr_benchmark>`
   working properly. (Main pull request: :pull:`5437`, more detail:
   :pull:`5430`, :pull:`5431`, :pull:`5432`, :pull:`5434`, :pull:`5436`)

#. `@acchamber`_ and `@ESadek-MO`_ resolved several deprecation to reduce
    number of warnings raised during tests.
   (:pull:`5493`, :pull:`5510`)

#. `@trexfeathers`_ replaced all uses of the ``logging.WARNING`` level, in
   favour of using Python warnings, following team agreement. (:pull:`5488`)

#. `@trexfeathers`_ adapted benchmarking to work with ASV ``>=v0.6`` by no
   longer using the ``--strict`` argument. (:pull:`5496`)




.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:

.. _@scottrobinson02: https://github.com/scottrobinson02
.. _@acchamber: https://github.com/acchamber


.. comment
    Whatsnew resources in alphabetical order:
