.. include:: ../common_links.inc

<unreleased>
************

This document explains the changes made to Iris for this release
(:doc:`View all changes <index>`.)


📢 Announcements
================

* N/A


✨ Features
===========

* N/A


🐛 Bugs Fixed
=============

* `@gcaria`_ fixed :meth:`~iris.cube.Cube.cell_measure_dims` to also accept the string name of a :class:`~iris.coords.CellMeasure`. (:pull:`3931`)
* `@gcaria`_ fixed :meth:`~iris.cube.Cube.ancillary_variable_dims` to also accept the string name of a :class:`~iris.coords.AncillaryVariable`. (:pull:`3931`)


💣 Incompatible Changes
=======================

* N/A


🔥 Deprecations
===============

* N/A


🔗 Dependencies
===============

* N/A


📚 Documentation
================

* `@rcomer`_ updated the "Seasonal ensemble model plots" Gallery example.
  (:pull:`3933`)


💼 Internal
===========

* `@rcomer`_ removed an old unused test file. (:pull:`3913`)


.. _@gcaria: https://github.com/gcaria
.. _@rcomer: https://github.com/rcomer
