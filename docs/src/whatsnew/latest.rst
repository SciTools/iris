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

#. `@ESadek-MO`_ added functionality to allow :func:`~iris.cube.Cube.concatenate`,
   :func:`~iris.cube.Cube.rolling_window` and :func:`~iris.cube.Cube.intersection`
   to work with dataless cubes. (:pull:`6860`, :pull:`6757`)

#. `@HGWright`_ added to the Nimrod loader to expand the types of Nimrod files it can load. This includes selecting which Nimrod table to use the data entry headers from. (:issue:`4505`, :pull:`6763`)

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

#. `@ESadek-MO`_ has deprecated the :class:`~iris.tests.IrisTest` class, and other unittest-based
   testing conveniences in favour of the conveniences found in :mod:`iris/tests/_shared_utils.py`.
   (:pull:`6950`)


🔗 Dependencies
===============

#. N/A


📚 Documentation
================

#. `@tkknight`_ reduced the space used on the documentation homepage by the quick
   link cards to allow for easier reading.  (:pull:`6886`)

#. `@tkknight`_ added a gallery carousel to the documentation homepage. (:pull:`6884`)

#. :user:`bjlittle` added the ``:user:`` `extlinks`_ ``github`` user convenience.
   (:pull:`6931`)


💼 Internal
===========

#. `@trexfeathers`_ and `@hdyson`_ updated ``_ff_replacement.py`` to clarify
   that Iris supports Ancillaries. (:pull:`6792`)

#. `@trexfeathers`_ adapted ``test_OceanSigmaZFactory`` for NumPy 2.4 - only
   0-dimensional arrays can now be converted to scalars. (:pull:`6876`)

#. `@trexfeathers`_ updated benchmarking to source Mule from its new home:
   https://github.com/MetOffice/mule . (:pull:`6879`)

#. `@tkknight`_ removed flake8, we have ruff now instead.  (:pull:`6889`)

#. `@trexfeathers`_ and `@ukmo-ccbunney`_ updated CI to support Python 3.14
   inline with `SPEC0 Minimum Supported Dependencies`_. Note: `pyvista` (and
   hence `geovista`) is not yet compatible with Python 3.14, so
   `:module:~iris.experimental.geovista` is currently only available for
   Python \<3.14.  (:pull:`6816`, :issue:`6775`)

#. `@ESadek-MO`_, `@trexfeathers`_, `@bjlittle`_, `@HGWright`_, `@pp-mo`_,
   `@stephenworsley`_ and `@ukmo-ccbunney`_ converted the entirity of the tests
   from unittest to pytest. Iris is now also ruff-PT compliant, save for PT019.
   (:issue:`6212`, :pull:`6939`)

.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:

.. _@hdyson: https://github.com/hdyson

.. _SPEC0 Minimum Supported Dependencies: https://scientific-python.org/specs/spec-0000/

.. comment
    Whatsnew resources in alphabetical order:
