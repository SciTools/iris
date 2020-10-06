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

* `@znicholls`_ fixed :meth:`~iris.quickplot._title` to only check ``units.is_time_reference`` if the ``units`` symbol is not used. (:pull:`3902`)


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

* N/A


💼 Internal
===========

* `@znicholls`_ added a test for plotting with the label being taken from the unit's symbol, see :meth:`~iris.tests.test_quickplot.TestLabels.test_pcolormesh_str_symbol` (:pull:`3902`).

* `@znicholls`_ made :func:`~iris.tests.idiff.step_over_diffs` robust to hyphens (``-``) in the input path (i.e. the ``result_dir`` argument) (:pull:`3902`).
