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

#. ⏱️ Performance benchmarking has shown that loading
   :term:`Fields File (FF) Format` with a large number of fields via
   :func:`iris.fileformats.um.structured_um_loading` has become ~30% slower
   since `Dask version 2024.2.1`_.


✨ Features
===========

#. `@HGWright`_ and `@trexfeathers`_ added the
   :mod:`iris.experimental.geovista` module, providing conveniences for using
   :ref:`ugrid geovista` with Iris. To see some of this in action, check out
   :ref:`ugrid operations`. Note that GeoVista is an **optional** dependency
   so you will need to explicitly install it into your environment.
   (:pull:`5740`)


🐛 Bugs Fixed
=============

#. N/A


💣 Incompatible Changes
=======================

#. Warnings are no longer produced for fill value 'collisions' in NetCDF
   saving. :ref:`Read more <missing_data_saving>`. (:pull:`5833`)


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
.. _Dask version 2024.2.1: https://docs.dask.org/en/stable/changelog.html#v2024-2-1