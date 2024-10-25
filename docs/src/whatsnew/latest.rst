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

#. Iris is now compliant with NumPy v2. This may affect your scripts.
   :ref:`See the full What's New entry for more details <numpy2>`.


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

#. `@pp-mo`_ and `@stephenworsley`_ added support for hybrid coordinates whose
   references are split across multiple input fields, and :meth:`~iris.LOAD_POLICY` to
   control it, as requested in :issue:`5369`, actioned in :pull:`6168`.


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
   problematic check. (:pull:`5926` and :pull:`6187`)

🔥 Deprecations
===============

#. N/A


🔗 Dependencies
===============

.. _numpy2:

#. `@trexfeathers`_ adapted the Iris codebase to work with NumPy v2. The
   `NumPy v2 full release notes`_ have the exhaustive details. Notable
   changes that may affect your Iris scripts are below. (:pull:`6035`)

   * `NumPy v2 changed data type promotion`_

   * `NumPy v2 changed scalar printing`_


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
.. _NumPy v2 changed data type promotion: https://numpy.org/doc/stable/numpy_2_0_migration_guide.html#changes-to-numpy-data-type-promotion
.. _NumPy v2 changed scalar printing: https://numpy.org/doc/stable/release/2.0.0-notes.html#representation-of-numpy-scalars-changed
.. _NumPy v2 full release notes: https://numpy.org/doc/stable/release/2.0.0-notes.html
