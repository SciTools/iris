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

#. `@ukmo-ccbunney`_ added new *cube component* convenience methods that allow
   for manipulation of any named dimensional component that can be attached to a
   cube (i.e. coordinates, cell measures and ancillary variables) via a common
   interface. The following methods are provided:

   * :func:`~iris.cube.Cube.component` and :func:`~iris.cube.Cube.components`:
     get one or more components from a cube
   * :func:`~iris.cube.Cube.add_component`: add a component to a cube
   * :func:`~iris.cube.Cube.remove_component`: remove a component from a cube
   * :func:`~iris.cube.Cube.component_dims`: return the cube dimension(s)
     spanned by a component.

   (:issue:`5819`, :pull:`6854`)

#. `@ESadek-MO`_ added functionality to allow :func:`~iris.cube.Cube.rolling_window` and
   :func:`~iris.cube.Cube.intersection` to work with dataless cubes. (:pull:`6757`)


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

#. N/A


💼 Internal
===========

#. `@trexfeathers`_ and `@hdyson`_ updated ``_ff_replacement.py`` to clarify
   that Iris supports Ancillaries. (:pull:`6792`)


.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:

.. _@hdyson: https://github.com/hdyson



.. comment
    Whatsnew resources in alphabetical order:
