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

#. `@trexfeathers`_ and `@tkknight`_ removed the maximum pin for the
   PyData Sphinx Theme (used in the docs). (:issue:`6885`, :pull:`7053`)

#. `@tkknight`_ added a minimum pin for the PyData Sphinxc Theme as we use the
   collapse sidebar feature introduced in 0.17.0. (:pull:`7060`)


📚 Documentation
================

#. `@trexfeathers`_ and `@tkknight`_ made the docs compatible with the latest
   versions of PyData Sphinx Theme (>=0.16). (:issue:`6885`, :pull:`7053`)

#. `@tkknight`_ enabled the theme option to collapse the sidebar.  Note, it only
   appears once you click on a link away from the landing page.  Also moved
   the search box to the top navigation bar. (:pull:`7060`)


💼 Internal
===========

#. `@trexfeathers`_ altered the messaging for 'stale' issues and pull requests,
   to reduce the negative connotations. We now use ``needs-checkin`` for the
   initial prompt, and ``not-resourced`` if the issue/PR ends up closed.
   (:issue:`6993`, :pull:`7036`)

#. `@trexfeathers`_ fixed the benchmarking ``asv_delegated.py`` to work with
   Nox release ``2026.04.10`` (which adds more files to the environment parent
   directory, breaking previous assumptions). (:pull:`7046`)


#. `@ESadek-MO` and `@pp-mo`_ removed unit test reliance on all optional dependencies
   except for mo_pack.
   (:issue:`6832`, :pull:`6976`)

.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:




.. comment
    Whatsnew resources in alphabetical order: