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

#. `@pp-mo`_ added a new utility function :func:`~iris.util.equalise_cubes`, to help
   with aligning cubes so they can merge / concatenate.
   (:issue:`6248`, :pull:`6257`)

#. `@fnattino`_ added the lazy median aggregator :class:`iris.analysis.MEDIAN`
   based on the implementation discussed by `@rcomer`_ and `@stefsmeets`_ in
   :issue:`4039` (:pull:`6167`).

#. `@ESadek-MO`_ made :attr:`~iris.cube.Cube.data` optional in a
   :class:`~iris.cube.Cube`, when :attr:`~iris.cube.Cube.shape` is provided. A
   `dataless cube` may be used as a target in regridding, or as a template cube
   to add data to at a later time.

   This is the first step in providing `dataless cube` support. Currently, most
   cube methods won't work with a `dataless cube` and will raise an exception.
   However, :meth:`~iris.cube.Cube.transpose` will work, as will
   :meth:`~iris.cube.Cube.copy`. Note that, ``cube.copy(data=iris.DATALESS)``
   will provide a dataless copy of a cube. (:issue:`4447`, :pull:`6253`)
   
#. `@ESadek-MO`_ added the :mod:`iris.quickplot` ``footer`` kwarg to
   render text in the bottom right of the plot figure.
   (:issue:`6247`, :pull:`6332`)
   

🐛 Bugs Fixed
=============

#. `@rcomer`_ added handling for string stash codes when saving pp files.
   (:issue:`6239`, :pull:`6289`)

#. `@trexfeathers`_ and `@jrackham-mo`_ added a check for dtype castability when
   saving NetCDF ``valid_range``, ``valid_min`` and ``valid_max`` attributes -
   older NetCDF formats e.g. ``NETCDF4_CLASSIC`` support a maximum precision of
   32-bit. (:issue:`6178`, :pull:`6343`)


💣 Incompatible Changes
=======================

#. :class:`iris.tests.IrisTest` is being replaced by :mod:`iris.tests._shared_utils`.
   Once conversion from unittest to pytest is completed, :class:`iris.tests.IrisTest`
   class will be deprecated.


🚀 Performance Enhancements
===========================

#. `@bouweandela`_ made loading :class:`~iris.cube.Cube`\s from NetCDF files
   faster. (:pull:`6229` and :pull:`6252`)

#. `@fnattino`_ enabled lazy cube interpolation using the linear and
   nearest-neighbour interpolators (:class:`iris.analysis.Linear` and
   :class:`iris.analysis.Nearest`). Note that this implementation removes
   performance benefits linked to caching an interpolator object. While this does
   not break previously suggested code (instantiating and re-using an interpolator
   object remains possible), this is no longer an advertised feature. (:pull:`6084`)


🔥 Deprecations
===============

#. N/A


🔗 Dependencies
===============

#. `@stephenworsley`_ dropped support for ``py310`` and adopted support for ``py313``
   as per the `SPEC 0`_ schedule. (:pull:`6195`)

📚 Documentation
================

#. `@ESadek-MO`_ and `@trexfeathers`_ created :ref:`contributing_pytest_conversions`
   as a guide for converting from ``unittest`` to ``pytest``. (:pull:`5785`)

#. `@ESadek-MO`_ and `@trexfeathers`_ created a style guide for ``pytest`` tests,
   and consolidated ``Test Categories`` and ``Testing Tools`` into
   :ref:`contributing_tests`. (:issue:`5574`, :pull:`5785`)

#. `@jfrost-mo`_ corrected ``unit`` to ``units`` in the docstring for
   :class:`iris.coords.AuxCoord`. (:issue:`6347`, :pull:`6348`)


💼 Internal
===========

#. `@ESadek-MO`_ `@pp-mo`_ `@bjlittle`_ `@trexfeathers`_ and `@HGWright`_ have
   converted around a third of Iris' ``unittest`` style tests to ``pytest``. This is
   part of an ongoing effort to move from ``unittest`` to ``pytest``. (:pull:`6207`,
   part of :issue:`6212`)

#. `@trexfeathers`_, `@ESadek-MO`_ and `@HGWright`_ heavily re-worked
   :doc:`/developers_guide/release_do_nothing` to be more thorough and apply
   lessons learned from recent releases. (:pull:`6062`)

#. `@schlunma`_ made lazy [smart
   weights](https://github.com/SciTools/iris/pull/5084) used for cube
   aggregations have the same chunks as their parent cube if broadcasting is
   necessary. (:issue:`6285`, :pull:`6288`)

#. `@trexfeathers`_ improved the handling of benchmark environments, especially
    when working across Python versions. (:pull:`6329`)

#. `@trexfeathers`_ temporarily pinned Sphinx to `<8.2`.
   (:pull:`6344`, :issue:`6345`)


.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:

.. _@fnattino: https://github.com/fnattino
.. _@jfrost-mo: https://github.com/jfrost-mo
.. _@jrackham-mo: https://github.com/jrackham-mo
.. _@stefsmeets: https://github.com/stefsmeets

.. comment
    Whatsnew resources in alphabetical order:

.. _SPEC 0: https://scientific-python.org/specs/spec-0000/
