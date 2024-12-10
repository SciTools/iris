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

#. :class:`iris.tests.IrisTest` is being replaced by :mod:`iris.tests._shared_utils`.
   Once conversion from unittest to pytest is completed, :class:`iris.tests.IrisTest`
   class will be deprecated.


🚀 Performance Enhancements
===========================

#. `@fnattino`_ enabled lazy cube interpolation using the linear and
   nearest-neighbour interpolators (:class:`iris.analysis.Linear` and
   :class:`iris.analysis.Nearest`). (:pull:`6084`)


🔥 Deprecations
===============

#. N/A


🔗 Dependencies
===============

#. N/A


📚 Documentation
================

#. `@ESadek-MO`_ and `@trexfeathers`_ created :ref:`contributing_pytest_conversions`
   as a guide for converting from ``unittest`` to ``pytest``. (:pull:`5785`)

#. `@ESadek-MO`_ and `@trexfeathers`_ created a style guide for ``pytest`` tests,
   and consolidated ``Test Categories`` and ``Testing Tools`` into
   :ref:`contributing_tests` (:issue:`5574`, :pull:`5785`)


💼 Internal
===========

#. `@ESadek-MO`_ `@pp-mo`_ `@bjlittle`_ `@trexfeathers`_ and `@HGWright`_ have
   converted around a third of Iris' ``unittest`` style tests to ``pytest``. This is
   part of an ongoing effort to move from ``unittest`` to ``pytest``. (:pull:`6207`,
   part of :issue:`6212`)

#. `@trexfeathers`_, `@ESadek-MO`_ and `@HGWright`_ heavily re-worked
   :doc:`/developers_guide/release_do_nothing` to be more thorough and apply
   lessons learned from recent releases. (:pull:`6062`)


.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:

.. _@fnattino: httsps://github.com/fnattino


.. comment
    Whatsnew resources in alphabetical order: