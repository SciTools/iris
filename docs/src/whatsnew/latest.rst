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

#. `@bouweandela`_ made :func:`iris.util.rolling_window` work with lazy arrays.
   (:pull:`5775`)

#. `@stephenworsley`_ fixed a potential memory leak for Iris uses of
   :func:`dask.array.map_blocks`; known specifically to be a problem in the
   :class:`iris.analysis.AreaWeighted` regridder. (:pull:`5767`)


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

#. `@trexfeathers`_ setup automatic benchmarking on pull requests that modify
   files likely to affect performance or performance testing. Such pull
   requests are also labelled using the `Pull Request Labeler Github action`_
   to increase visibility. (:pull:`5763`, :pull:`5776`)

#. `@tkknight`_ updated codebase to comply with a new enforced rule `NPY002`_ for
   `ruff`_.  (:pull:`5786`)

#. `@tkknight`_ enabled `numpydoc validation`_ via the pre-commit hook.  The docstrings
   have been updated to comply and some rules have been ignored for now.
   (:pull:`5762`)

.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:




.. comment
    Whatsnew resources in alphabetical order:

.. _Pull Request Labeler GitHub action: https://github.com/actions/labeler
.. _NPY002: https://docs.astral.sh/ruff/rules/numpy-legacy-random/
.. _numpydoc validation: https://numpydoc.readthedocs.io/en/latest/validation.html#