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

* `@pelson`_ and `@trexfeathers`_ enhanced :meth:iris.plot.plot and
  :meth:iris.quickplot.plot to automatically place the cube on the x axis if
  the primary coordinate being plotted against is a vertical coordinate. E.g.
  ``iris.plot.plot(z_cube)`` will produce a z-vs-phenomenon plot, where before
  it would have produced a phenomenon-vs-z plot. (:pull:`3906`)


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



.. _@pelson: https://github.com/pelson
.. _@trexfeathers: https://github.com/trexfeathers
.. _@gcaria: https://github.com/gcaria
.. _@rcomer: https://github.com/rcomer
