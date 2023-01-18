.. include:: ../common_links.inc

|iris_version| |build_date| [unreleased]
****************************************

This document explains the changes made to Iris for this release
(:doc:`View all changes <index>`.)


.. dropdown:: :opticon:`report` |iris_version| Release Highlights
   :container: + shadow
   :title: text-primary text-center font-weight-bold
   :body: bg-light
   :animate: fade-in
   :open:

   The highlights for this major/minor release of Iris include:

   * N/A

   And finally, get in touch with us on :issue:`GitHub<new/choose>` if you have
   any issues or feature requests for improving Iris. Enjoy!


📢 Announcements
================

#. Congratulations to `@ESadek-MO`_ who has become a core developer for Iris! 🎉


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

#. N/A


🔥 Deprecations
===============

#. N/A


🔗 Dependencies
===============

#. N/A


📚 Documentation
================

#. `@rcomer`_ clarified instructions for updating gallery tests. (:pull:`5100`)
#. `@tkknight`_ unpinned ``pydata-sphinx-theme`` and set the default to use
   the light version (not dark) while we make the docs dark mode friendly
   (:pull:`5129`)

#. `@jonseddon`_ updated the citation to a more recent version of Iris. (:pull:`5116`)

#. `@rcomer`_ linked the :obj:`~iris.analysis.PERCENTILE` aggregator from the
   :obj:`~iris.analysis.MEDIAN` docstring, noting that the former handles lazy
   data. (:pull:`5128`)

#. `@trexfeathers`_ updated the WSL link to Microsoft's latest documentation,
   and removed an ECMWF link in the ``v1.0`` What's New that was failing the
   linkcheck CI. (:pull:`5109`)

#. `@Esadek-MO`_ added notes to function docstings to
   to clarify if the function preserves laziness or not. (:pull:`5137`)

💼 Internal
===========

#. `@fnattino`_ changed the order of ``ncgen`` arguments in the command to
   create NetCDF files for testing  (caused errors on OS X). (:pull:`5105`)

#. `@rcomer`_ removed some old infrastructure that printed test timings.
   (:pull:`5101`)


.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:

.. _@fnattino: https://github.com/fnattino


.. comment
    Whatsnew resources in alphabetical order:

