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

   The highlights for this minor release of Iris include:

   * N/A

   And finally, get in touch with us on :issue:`GitHub<new/choose>` if you have
   any issues or feature requests for improving Iris. Enjoy!


📢 Announcements
================

#. N/A


✨ Features
===========

#. `@wjbenfold`_ added support for ``false_easting`` and ``false_northing`` to
   :class:`~iris.coord_system.Mercator`. (:issue:`3107`, :pull:`4524`)


🐛 Bugs Fixed
=============

#. `@rcomer`_ reverted part of the change from :pull:`3906` so that
   :func:`iris.plot.plot` no longer defaults to placing a "Y" coordinate (e.g.
   latitude) on the y-axis of the plot. (:issue:`4493`, :pull:`4601`)

#. `@rcomer`_ enabled passing of scalar objects to :func:`~iris.plot.plot` and
   :func:`~iris.plot.scatter`. (:pull:`4616`)

#. `@rcomer`_ fixed :meth:`~iris.cube.Cube.aggregated_by` with `mdtol` for 1D
   cubes where an aggregated section is entirely masked, reported at
   :issue:`3190`.  (:pull:`4246`)


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

#. `@rcomer`_ introduced the ``nc-time-axis >=1.4`` minimum pin, reflecting that
   we no longer use the deprecated :class:`nc_time_axis.CalendarDateTime`
   when plotting against time coordinates. (:pull:`4584`)


📚 Documentation
================

#. `@tkknight`_ added a page to show the issues that have been voted for.  See
   :ref:`voted_issues_top`. (:issue:`3307`, :pull:`4617`)


💼 Internal
===========

#. `@trexfeathers`_ and `@pp-mo`_ finished implementing a mature benchmarking
   infrastructure (see :ref:`contributing.benchmarks`), building on 2 hard
   years of lessons learned 🎉. (:pull:`4477`, :pull:`4562`, :pull:`4571`,
   :pull:`4583`, :pull:`4621`)
#. `@wjbenfold`_ made :func:`iris.tests.stock.simple_1d` respect the
   ``with_bounds`` argument. (:pull:`4658`)


.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:




.. comment
    Whatsnew resources in alphabetical order:


