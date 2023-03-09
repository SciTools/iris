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

   * We're so proud to fully support `@ed-hawkins`_ and `#ShowYourStripes`_ ❤️

   And finally, get in touch with us on :issue:`GitHub<new/choose>` if you have
   any issues or feature requests for improving Iris. Enjoy!


📢 Announcements
================

#. Congratulations to `@ESadek-MO`_ who has become a core developer for Iris! 🎉
#. Welcome and congratulations to `@HGWright`_ for making his first contribution to Iris! 🎉


✨ Features
===========

#. `@bsherratt`_ added support for plugins - see the corresponding
   :ref:`documentation page<community_plugins>` for further information.
   (:pull:`5144`)

#. `@rcomer`_ enabled lazy evaluation of :obj:`~iris.analysis.RMS` calcuations
   with weights. (:pull:`5017`)

#. `@lbdreyer`_ added :func:`iris.plot.hist` and :func:`iris.quickplot.hist`.
   (:pull:`5189)


🐛 Bugs Fixed
=============

#. `@trexfeathers`_ and `@pp-mo`_ made Iris' use of the `netCDF4`_ library
   thread-safe. (:pull:`5095`)

#. `@ESadek-MO`_ removed check and error raise for saving
   cubes with masked :class:`iris.coords.CellMeasure`.
   (:issue:`5147`, :pull:`5181`)


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

#. `@trexfeathers`_ added a new top-level :doc:`/community/index` section,
   as a one-stop place to find out about getting involved, and how we relate
   to other projects. (:pull:`5025`)

#. The **Iris community**, with help from the **Xarray community**, produced
   the :doc:`/community/iris_xarray` page, highlighting the similarities and
   differences between the two packages. (:pull:`5025`)

#. `@bjlittle`_ added a new section to the `README.md`_ to show our support
   for the outstanding work of `@ed-hawkins`_ et al for `#ShowYourStripes`_.
   (:pull:`5141`)

#. `@HGWright`_ fixed some typo's from Gitwash. (:pull:`5145`)

💼 Internal
===========

#. `@fnattino`_ changed the order of ``ncgen`` arguments in the command to
   create NetCDF files for testing  (caused errors on OS X). (:pull:`5105`)

#. `@rcomer`_ removed some old infrastructure that printed test timings.
   (:pull:`5101`)

#. `@lbdreyer`_ and `@trexfeathers`_ (reviewer) added coverage testing. This
   can be enabled by using the "--coverage" flag when running the tests with
   nox i.e. ``nox --session tests -- --coverage``. (:pull:`4765`)

#. `@lbdreyer`_ and `@trexfeathers`_ (reviewer) removed the ``--coding-tests``
   option from Iris' test runner. (:pull:`4765`)

.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:

.. _@fnattino: https://github.com/fnattino
.. _@ed-hawkins: https://github.com/ed-hawkins

.. comment
    Whatsnew resources in alphabetical order:

.. _#ShowYourStripes: https://showyourstripes.info/s/globe/
.. _README.md: https://github.com/SciTools/iris#-----
