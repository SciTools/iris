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

#. `@pp-mo`_ added a new utility function :func:`~iris.util.equalise_cubes`, to help
   with aligning cubes so they can merge / concatenate.
   (:issue:`6248`, :pull:`6257`)


#. `@fnattino`_ added the lazy median aggregator :class:`iris.analysis.MEDIAN`
   based on the implementation discussed by `@rcomer`_ and `@stefsmeets`_ in
   :issue:`4039` (:pull:`6167`).

#. `@ESadek-MO`_ made :attr:`~iris.cube.Cube.data` optional in a
   :class:`~iris.cube.Cube`, when :attr:`~iris.cube.Cube.shape` is provided
   instead. `dataless cubes` can currently be used as targets in regridding, or
   for templates to add data to at a later time.

   This is the first step in making `dataless cubes`. Currently, most cube methods
   don't work on `dataless cubes`, and will raise in an error if attempted.
   :meth:`~iris.cube.Cube.transpose` will work, as will :meth:`~iris.cube.Cube.copy`.
   `my_cube.copy(data = iris.DATALESS)` will copy the cube and remove data in
   the process.
   (:issue:`4447`, :pull:`6253`)

🐛 Bugs Fixed
=============

#. `@rcomer`_ added handling for string stash codes when saving pp files.
   (:issue:`6239`, :pull:`6289`)


💣 Incompatible Changes
=======================

#. :class:`iris.tests.IrisTest` is being replaced by :mod:`iris.tests._shared_utils`.
   Once conversion from unittest to pytest is completed, :class:`iris.tests.IrisTest`
   class will be deprecated.


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

.. _@fnattino: https://github.com/fnattino
.. _@jrackham-mo: https://github.com/jrackham-mo
.. _@stefsmeets: https://github.com/stefsmeets

.. comment
    Whatsnew resources in alphabetical order: