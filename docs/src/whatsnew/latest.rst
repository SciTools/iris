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

#. `@bsherratt`_ added support for plugins - see the corresponding
   :ref:`documentation page<community_plugins>` for further information.
   (:pull:`5144`)

#. `@rcomer`_ enabled lazy evaluation of :obj:`~iris.analysis.RMS` calcuations
   with weights. (:pull:`5017`)

#. `@schlunma`_ allowed the usage of cubes, coordinates, cell measures, or
   ancillary variables as weights for cube aggregations
   (:meth:`iris.cube.Cube.collapsed`, :meth:`iris.cube.Cube.aggregated_by`, and
   :meth:`iris.cube.Cube.rolling_window`). This automatically adapts cube units
   if necessary. (:pull:`5084`)

#. `@lbdreyer`_ and `@trexfeathers`_ (reviewer)  added :func:`iris.plot.hist` 
   and :func:`iris.quickplot.hist`. (:pull:`5189`)

#. `@tinyendian`_ edited :func:`~iris.analysis.cartography.rotate_winds` to
   enable lazy computation of rotated wind vector components (:issue:`4934`,
   :pull:`4972`)

#. `@ESadek-MO`_ updated to the latest CF Standard Names Table v80
   (07 February 2023). (:pull:`5244`)

#. `@pp-mo`_ and  `@lbdreyer`_ supported delayed saving of lazy data, when writing to
   the netCDF file format.  See : :ref:`delayed netCDF saves <delayed_netcdf_save>`.
   Also with significant input from `@fnattino`_.
   (:pull:`5191`)


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

#. `@tkknight`_ migrated to `sphinx-design`_ over the legacy `sphinx-panels`_.
   (:pull:`5127`)

#. `@tkknight`_ updated the ``make`` target for ``help`` and added
   ``livehtml`` to auto generate the documentation when changes are detected
   during development. (:pull:`5258`)


💼 Internal
===========

#. `@bjlittle`_ added the `codespell`_ `pre-commit`_ ``git-hook`` to automate
   spell checking within the code-base. (:pull:`5186`)

#. `@bjlittle`_ and `@trexfeathers`_ (reviewer) added a `check-manifest`_
   GitHub Action and `pre-commit`_ ``git-hook`` to automate verification
   of assets bundled within a ``sdist`` and binary ``wheel`` of our
   `scitools-iris`_ PyPI package. (:pull:`5259`)

.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:

.. _@fnattino: https://github.com/fnattino
.. _@tinyendian: https://github.com/tinyendian


.. comment
    Whatsnew resources in alphabetical order:

.. _sphinx-panels: https://github.com/executablebooks/sphinx-panels
.. _sphinx-design: https://github.com/executablebooks/sphinx-design
.. _check-manifest: https://github.com/mgedmin/check-manifest
