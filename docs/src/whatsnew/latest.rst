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

#. `@trexfeathers`_ removed the custom ``setup.py develop`` command, since
   Setuptools are deprecating ``develop``; developers should instead
   use ``pip install -e .``. See `Running setuptools commands`_ for more.
   (:pull:`6424`)


💣 Incompatible Changes
=======================

#. N/A


🚀 Performance Enhancements
===========================

#. `@pp-mo`_ implemented automatic rechunking of hybrid (aka factory/derived)
   coordinates to avoid excessive memory usage. (:issue:`6404`, :pull:`6516`)


🔥 Deprecations
===============

#. N/A


🔗 Dependencies
===============

#. N/A


📚 Documentation
================

#. N/A


💼 Internal
===========

#. `@pp-mo`_ replaced the PR-based linkchecks with a daily scheduled link checker based
   on `lychee <https://github.com/lycheeverse/lychee-action>`__.
   (:issue:`4140`, :pull:`6386`)


.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:




.. comment
    Whatsnew resources in alphabetical order:

.. _Running setuptools commands: https://setuptools.pypa.io/en/latest/deprecated/commands.html