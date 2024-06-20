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

#. `@ESadek-MO`_ updated the error messages in :meth:`iris.cube.CubeList.concatenate`
   to better explain the error. (:pull:`6005`)

#. `@trexfeathers`_ added the
   :meth:`~iris.experimental.ugrid.mesh.MeshCoord.collapsed` method to
   :class:`~iris.experimental.ugrid.mesh.MeshCoord`, enabling collapsing of
   the :class:`~iris.cube.Cube` :attr:`~iris.cube.Cube.mesh_dim` (see
   :ref:`cube-statistics-collapsing`). (:issue:`5377`, :pull:`6003`)


🐛 Bugs Fixed
=============

#. `@bouweandela`_ updated the ``chunktype`` of Dask arrays, so it corresponds
   to the array content. (:pull:`5801`)

#. `@rcomer`_ made the :obj:`~iris.analysis.WPERCENTILE` aggregator work with
   :func:`~iris.cube.Cube.rolling_window`.  (:issue:`5777`, :pull:`5825`)


#. `@pp-mo`_ corrected the use of mesh dimensions when saving with multiple
   meshes.  (:issue:`5908`, :pull:`6004`)


💣 Incompatible Changes
=======================

#. `@rcomer`_ removed the *target* parameter from
   :func:`~iris.fileformats.pp.as_fields` and
   :func:`~iris.fileformats.pp.save_pairs_from_cube` because it had no effect.
   (:pull:`5783`)

#. `@stephenworsley`_ made masked arrays on Iris objects now compare as equal
   precisely when all unmasked points are equal and when the masks are identical.
   This is due to changes in :func:`~iris.util.array_equal` which previously
   ignored masks entirely. (:pull:`4457`)


🚀 Performance Enhancements
===========================

#. `@bouweandela`_ added the option to specify the Dask chunks of the target
   array in :func:`iris.util.broadcast_to_shape`. (:pull:`5620`)

#. `@schlunma`_ allowed :func:`iris.analysis.cartography.area_weights` to
   return dask arrays with arbitrary chunks. (:pull:`5658`)

#. `@bouweandela`_ made :meth:`iris.cube.Cube.rolling_window` work with lazy
   data. (:pull:`5795`)

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

#. `@hsteptoe`_ added more detailed examples to :class:`~iris.cube.Cube` functions :func:`~iris.cube.Cube.slices` and :func:`~iris.cube.Cube.slices_over`. (:pull:`5735`)


💼 Internal
===========

#. `@bouweandela`_ removed a workaround in :meth:`~iris.cube.CubeList.merge` for an
   issue with :func:`dask.array.stack` which has been solved since 2017. (:pull:`5923`)

#. `@trexfeathers`_ introduced a temporary fix for Airspeed Velocity's
   deprecated use of the ``conda --force`` argument. To be removed once
   `airspeed-velocity/asv#1397`_ is merged and released. (:pull:`5931`)

#. `@trexfeathers`_ created :func:`iris.tests.stock.realistic_4d_w_everything`;
   providing a :class:`~iris.cube.Cube` aimed to exercise as much of Iris as
   possible. (:pull:`5949`)

#. `@trexfeathers`_ deactivated any small 'unit-style' benchmarks for default
   benchmark runs, and introduced larger more 'real world' benchmarks where
   coverage was needed. (:pull:`5949`).

#. `@trexfeathers`_ made a Nox `benchmarks` session as the recommended entry
   point for running benchmarks. (:pull:`5951`)

#. `@ESadek-MO`_ added further `benchmarks` for aggregation and collapse.
   (:pull:`5954`)

#. `@trexfeathers`_ set the benchmark data generation environment to
   automatically install iris-test-data during setup. (:pull:`5958`)

#. `@pp-mo`_ reworked benchmark peak-memory measurement to use the
   `tracemalloc <https://docs.python.org/3.12/library/tracemalloc.html>`_
   package.
   (:pull:`5948`)

#. `@pp-mo`_ added a benchmark 'trialrun' sub-command, to quickly test
   benchmarks during development. (:pull:`5957`)

#. `@pp-mo`_ moved several memory-measurement benchmarks from 'on-demand' to
   the standard set, in hopes that use of 'tracemalloc' (:pull:`5948`) makes
   the results consistent enough to monitor for performance changes.
   (:pull:`5959`)

#. `@rcomer`_ made some :meth:`~iris.cube.Cube.slices_over` tests go faster (:pull:`5973`)


.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:

.. _@hsteptoe: https://github.com/hsteptoe


.. comment
    Whatsnew resources in alphabetical order:

.. _airspeed-velocity/asv#1397: https://github.com/airspeed-velocity/asv/pull/1397
