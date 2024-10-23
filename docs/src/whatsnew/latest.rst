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

#. `@jrackham-mo`_ added :meth:`~iris.io.format_picker.FormatAgent.copy` and
   equality methods to :class:`iris.io.format_picker.FormatAgent`, as requested
   in :issue:`6108`, actioned in :pull:`6119`.

#. `@ukmo-ccbunney`_ added ``colorbar`` keyword to allow optional creation of
   the colorbar in the following quickplot methods:

   * :meth:`iris.quickplot.contourf`

   * :meth:`iris.quickplot.pcolor`

   * :meth:`iris.quickplot.pcolormesh`

   Requested in :issue:`5970`, actioned in :pull:`6169`.


🐛 Bugs Fixed
=============

#. `@rcomer`_ enabled partial collapse of multi-dimensional string coordinates,
   fixing :issue:`3653`. (:pull:`5955`)

#. `@ukmo-ccbunney`_ improved error handling for malformed `cell_method`
   attribute. Also made cell_method string parsing more lenient w.r.t.
   whitespace. (:pull:`6083`)

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
   than two cubes are concatenated with equality checks on the values of
   auxiliary coordinates, derived coordinates, cell measures, or ancillary
   variables enabled.
   In some cases, this may lead to higher memory use. This can be remedied by
   reducing the number of Dask workers.
   In rare cases, the new implementation could potentially be slower. This
   may happen when there are very many or large auxiliary coordinates, derived
   coordinates, cell measures, or ancillary variables to be checked that span
   the concatenation axis. This issue can be avoided by disabling the
   problematic check. (:pull:`5926`)

🔥 Deprecations
===============

#. N/A


🔗 Dependencies
===============

#. N/A


📚 Documentation
================

#. `@bouweandela`_ added type hints for :class:`~iris.cube.Cube`. (:pull:`6037`)

💼 Internal
===========

#. `@trexfeathers`_ improved the new ``tracemalloc`` benchmarking (introduced
   in Iris v3.10.0, :pull:`5948`) to use the same statistical repeat strategy
   as timing benchmarks. (:pull:`5981`)

#. `@trexfeathers`_ adapted Iris to work with Cartopy v0.24. (:pull:`6171`,
   :pull:`6172`)


.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:

.. _@jrackham-mo: https://github.com/jrackham-mo


.. comment
    Whatsnew resources in alphabetical order:

.. _cartopy#2390: https://github.com/SciTools/cartopy/issues/2390
