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

#. `@rcomer`_ removed the *target* parameter from
   :func:`~iris.fileformats.pp.as_fields` and
   :func:`~iris.fileformats.pp.save_pairs_from_cube` because it had no effect.
   (:pull:`5783`)


🚀 Performance Enhancements
===========================

#. N/A

#. `@bouweandela`_ added the option to specify the Dask chunks of the target
   array in :func:`iris.util.broadcast_to_shape`. (:pull:`5620`)

#. `@schlunma`_ allowed :func:`iris.analysis.cartography.area_weights` to
   return dask arrays with arbitrary chunks. (:pull:`5658`)


🔥 Deprecations
===============

#. N/A


🔗 Dependencies
===============

#. `@tkknight`_ removed the pin for ``sphinx <=5.3``, so the latest should 
   now be used, currently being v7.2.6.
   (:pull:`5901`)


📚 Documentation
================

#. N/A


💼 Internal
===========

#. N/A


.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:




.. comment
    Whatsnew resources in alphabetical order:
