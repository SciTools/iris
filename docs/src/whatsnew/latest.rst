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

#. `@pp-mo`_ made a MeshCoord inherit a coordinate system from its location coord,
   as it does its metadata.  N.B. mesh location coords can not however load a
   coordinate system from netcdf at present, as this needs the 'extended'
   grid-mappping syntax -- see : :issue:`3388`.
   (:issue:`5562`, :pull:`6016`)

#. `@HGWright`_ added the `monthly` and `yearly` options to the
   :meth:`~iris.coords.guess_bounds` method. (:issue:`4864`, :pull:`6090`)


🐛 Bugs Fixed
=============

#. `@bouweandela`_ updated the ``chunktype`` of Dask arrays, so it corresponds
   to the array content. (:pull:`5801`)

#. `@rcomer`_ made the :obj:`~iris.analysis.WPERCENTILE` aggregator work with
   :func:`~iris.cube.Cube.rolling_window`.  (:issue:`5777`, :pull:`5825`)


#. `@pp-mo`_ corrected the use of mesh dimensions when saving with multiple
   meshes.  (:issue:`5908`, :pull:`6004`)

#. `@trexfeathers`_ fixed the datum :class:`python:FutureWarning` to only be raised if
   the ``datum_support`` :class:`~iris.Future` flag is disabled AND a datum is
   present on the loaded NetCDF grid mapping. (:issue:`5749`, :pull:`6050`)


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

#. `@trexfeathers`_ renamed the ``Mesh`` class to
   :class:`~iris.experimental.ugrid.mesh.MeshXY`, in preparation for a future
   more flexible parent class (:class:`~iris.experimental.ugrid.mesh.Mesh`).
   (:issue:`6052` :pull:`6056`)

#. `@stephenworsley`_ replaced the ``include_nodes``, ``include_edges`` and
   ``include_faces`` arguments with a single ``location`` argument in the
   :class:`~iris.experimental.ugrid.Mesh` methods
   :meth:`~iris.experimental.ugrid.Mesh.coord`, :meth:`~iris.experimental.ugrid.Mesh.coords`
   and :meth:`~iris.experimental.ugrid.Mesh.remove_coords`. (:pull:`6055`)

#. `@pp-mo`_ moved all the mesh API from the :mod:`iris.experimental.ugrid` module to
   to :mod:`iris.mesh`, making this public supported API.  Note that the
   :class:`iris.experimental.ugrid.Mesh` class is renamed as :class:`iris.mesh.MeshXY`,
   to allow for possible future mesh types with different properties to exist as
   subclasses of a common generic :class:`~iris.mesh.components.Mesh` class.
   (:issue:`6057`, :pull:`6061`, :pull:`6077`)

#. `@pp-mo`_ and `@stephenworsley`_ Turned on UGRID loading by default, effectively removing
   the need for and deprecating the :func:`~iris.ugrid.experimental.PARSE_UGRID_ON_LOAD`
   context manager. (:pull:`6054`, :pull:`6088`)


🚀 Performance Enhancements
===========================

#. `@bouweandela`_ added the option to specify the Dask chunks of the target
   array in :func:`iris.util.broadcast_to_shape`. (:pull:`5620`)

#. `@schlunma`_ allowed :func:`iris.analysis.cartography.area_weights` to
   return dask arrays with arbitrary chunks. (:pull:`5658`)

#. `@bouweandela`_ made :meth:`iris.cube.Cube.rolling_window` work with lazy
   data. (:pull:`5795`)

#. `@bouweandela`_ updated :meth:`iris.cube.CubeList.concatenate` so it keeps
   ancillary variables and cell measures lazy. (:pull:`6010`)

#. `@bouweandela`_ made :meth:`iris.cube.CubeList.concatenate` faster for cubes
   that have coordinate factories. (:pull:`6038`)

🔥 Deprecations
===============

#. N/A


🔗 Dependencies
===============

#. `@tkknight`_ removed the pin for ``sphinx <=5.3``, so the latest should
   now be used, currently being v7.2.6.
   (:pull:`5901`)

#. `@trexfeathers`_ updated the :mod:`iris.experimental.geovista`
   documentation's use of :class:`geovista.geodesic.BBox`
   to be compatible with GeoVista v0.5, as well as previous versions.
   (:pull:`6064`)

#. `@pp-mo`_ temporarily pinned matplotlib to ">=3.5, !=3.9.1", to avoid current CI
   test failures on plot results, apparently due to a matplotlib bug.
   See : https://github.com/matplotlib/matplotlib/issues/28567
   (:pull:`6065`)



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

#. `@bouweandela`_ enabled mypy checks for type hints.
   The entire team would like to thank Bouwe for putting in the hard
   work on an unglamorous but highly valuable contribution. (:pull:`5956`)

#. `@trexfeathers`_ re-wrote the delegated ASV environment plugin to reduce
   complexity, remove unnecessary slow operations, apply the least-surprise
   principle, be more robust against failures, and improve the ability to
   benchmark historic commits (especially older Python versions).
   (:pull:`5963`)

#. `@bouweandela`_ made some tests for :func:`~iris.iterate.izip` faster. (:pull:`6041`)

.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:

.. _@hsteptoe: https://github.com/hsteptoe


.. comment
    Whatsnew resources in alphabetical order:

.. _airspeed-velocity/asv#1397: https://github.com/airspeed-velocity/asv/pull/1397
