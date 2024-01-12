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

#. `@lbdreyer`_ relicensed Iris from LGPL-3 to BSD-3. (:pull:`5577`)

#. `@HGWright`_, `@bjlittle`_ and `@trexfeathers`_ (reviewers) added a
   CITATION.cff file to Iris and updated the :ref:`citation documentation <Citing_Iris>`
   , to help users cite Iris in their work. (:pull:`5483`)


✨ Features
===========
#. `@pp-mo`_, `@lbdreyer`_ and `@trexfeathers`_ improved
   :class:`~iris.cube.Cube` :attr:`~iris.cube.Cube.attributes` handling to
   better preserve the distinction between dataset-level and variable-level
   attributes, allowing file-Cube-file round-tripping of NetCDF attributes. See
   :class:`~iris.cube.CubeAttrsDict`, NetCDF
   :func:`~iris.fileformats.netcdf.saver.save` and :data:`~iris.Future` for more.
   (:pull:`5152`, `split attributes project`_)

#. `@rcomer`_ rewrote :func:`~iris.util.broadcast_to_shape` so it now handles
   lazy data. (:pull:`5307`)

#. `@trexfeathers`_ and `@HGWright`_ (reviewer) sub-categorised all Iris'
   :class:`UserWarning`\s for richer filtering. The full index of
   sub-categories can be seen here: :mod:`iris.exceptions` . (:pull:`5498`)

#. `@trexfeathers`_ added the :class:`~iris.coord_systems.ObliqueMercator`
   and :class:`~iris.coord_systems.RotatedMercator` coordinate systems,
   complete with NetCDF loading and saving. (:pull:`5548`)

#. `@trexfeathers`_ added the ``use_year_at_season_start`` parameter to
   :func:`iris.coord_categorisation.add_season_year`. When
   ``use_year_at_season_start==True``: seasons spanning the year boundary (e.g.
   Winter - December to February) will be assigned to the preceding year (e.g.
   the year of December) instead of the following year (the default behaviour).
   (:pull:`5573`)

#. `@HGWright`_ added :attr:`~iris.coords.Coord.ignore_axis` to allow manual
   intervention preventing :func:`~iris.util.guess_coord_axis` from acting on a
   coordinate. (:pull:`5551`)

#. `@pp-mo`_, `@trexfeathers`_ and `@ESadek-MO`_ added more control over
   NetCDF chunking with the use of the :data:`iris.fileformats.netcdf.loader.CHUNK_CONTROL`
   context manager. (:pull:`5588`)


🐛 Bugs Fixed
=============

#. `@scottrobinson02`_ fixed the output units when dividing a coordinate by a
   cube. (:issue:`5305`, :pull:`5331`)

#. `@ESadek-MO`_ has updated :mod:`iris.tests.graphics.idiff` to stop duplicated file names
   preventing acceptance. (:issue:`5098`, :pull:`5482`)

#. `@acchamber`_ and `@rcomer`_ modified 2D plots so that time axes and their
   ticks have more sensible default labels.  (:issue:`5426`, :pull:`5561`)

#. `@rcomer`_ and `@trexfeathers`_ (reviewer) added handling for realization
   coordinates when saving pp files (:issue:`4747`, :pull:`5568`)

#. `@ESadek-MO`_ has updated
   :mod:`iris.fileformats._nc_load_rules.helpers` to lessen warning duplication.
   (:issue:`5536`, :pull:`5685`)


💣 Incompatible Changes
=======================

#. N/A


🚀 Performance Enhancements
===========================

#. `@stephenworsley`_ improved the speed of :class:`~iris.analysis.AreaWeighted`
   regridding. (:pull:`5543`)

#. `@bouweandela`_ made :func:`iris.util.array_equal` faster when comparing
   lazy data from file. This will also speed up coordinate comparison.
   (:pull:`5610`)


🔥 Deprecations
===============

#. N/A


🔗 Dependencies
===============

#. `@bjlittle`_ enforced the minimum pin of ``numpy>1.21`` in accordance with the `NEP29 Drop Schedule`_.
   (:pull:`5525`)

#. `@bjlittle`_ enforced the minimum pin of ``numpy>1.22`` in accordance with the `NEP29 Drop Schedule`_.
   (:pull:`5668`)


📚 Documentation
================

#. `@trexfeathers`_ documented the intended use of warnings filtering with
   Iris. See :ref:`filtering-warnings`. (:pull:`5509`)

#. `@rcomer`_ updated the
   :ref:`sphx_glr_generated_gallery_meteorology_plot_COP_maps.py` to show how
   a colourbar may steal space from multiple axes. (:pull:`5537`)

#. `@tkknight`_ improved the top navgation bar alignment and amount of
   links shown.  Also improved how the warning banner is implemented.
   (:pull:`5505` and :pull:`5508`)

#. `@tkknight`_  removed broken git links. (:pull:`5569`)

#. `@ESadek-MO`_ added a phrasebook for synonymous terms used in similar
   packages. (:pull:`5564`)

#. `@ESadek-MO`_ and `@trexfeathers`_ created a technical paper for NetCDF
   saving and loading, :ref:`netcdf_io` with a section on chunking, and placeholders
   for further topics. (:pull:`5588`)

#. `@bouweandela`_ updated all hyperlinks to https. (:pull:`5621`)

#. `@ESadek-MO`_ created an index page for :ref:`further_topics_index`, and
   relocated all 'Technical Papers' into
   :ref:`further_topics_index`. (:pull:`5602`)

#. `@trexfeathers`_ made drop-down icons visible to show which pages link to
   'sub-pages'. (:pull:`5684`)

#. `@trexfeathers`_ improved the documentation of acceptable
   :class:`~iris.cube.Cube` standard names in
   :func:`iris.analysis.calculus.curl`. (:pull:`5680`)


💼 Internal
===========

#. `@trexfeathers`_ and `@ESadek-MO`_ (reviewer) performed a suite of fixes and
   improvements for benchmarking, primarily to get
   :ref:`on demand pull request benchmarking <on_demand_pr_benchmark>`
   working properly. (Main pull request: :pull:`5437`, more detail:
   :pull:`5430`, :pull:`5431`, :pull:`5432`, :pull:`5434`, :pull:`5436`)

#. `@trexfeathers`_ set a number of memory benchmarks to be on-demand, as they
   were vulnerable to false positives in CI runs. (:pull:`5481`)

#. `@acchamber`_ and `@ESadek-MO`_ resolved several deprecation to reduce
   number of warnings raised during tests.
   (:pull:`5493`, :pull:`5511`)

#. `@trexfeathers`_ replaced all uses of the ``logging.WARNING`` level, in
   favour of using Python warnings, following team agreement. (:pull:`5488`)

#. `@trexfeathers`_ adapted benchmarking to work with ASV ``>=v0.6`` by no
   longer using the ``--strict`` argument. (:pull:`5496`)

#. `@fazledyn-or`_ replaced ``NotImplementedError`` with ``NotImplemented`` as
   a proper method call. (:pull:`5544`)

#. `@bjlittle`_ corrected various comment spelling mistakes detected by
   `codespell`_. (:pull:`5546`)

#. `@rcomer`_ reduced the size of the conda environment used for testing.
   (:pull:`5606`)

#. `@trexfeathers`_ and `@pp-mo`_ improved how the conda-forge feedstock
   release candidate branch is managed, via:
   :doc:`../developers_guide/release_do_nothing`.
   (:pull:`5515`)

#. `@bjlittle`_ adopted and configured the `ruff`_ linter. (:pull:`5623`)

#. `@bjlittle`_ configured the ``line-length = 88`` for `black`_, `isort`_
   and `ruff`_. (:pull:`5632`)

#. `@bjlittle`_ replaced `isort`_ with `ruff`_. (:pull:`5633`)

#. `@bjlittle`_ replaced `black`_ with `ruff`_. (:pull:`5634`)


.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:

.. _@scottrobinson02: https://github.com/scottrobinson02
.. _@acchamber: https://github.com/acchamber
.. _@fazledyn-or: https://github.com/fazledyn-or


.. comment
    Whatsnew resources in alphabetical order:

.. _NEP29 Drop Schedule: https://numpy.org/neps/nep-0029-deprecation_policy.html#drop-schedule
.. _codespell: https://github.com/codespell-project/codespell
.. _split attributes project: https://github.com/orgs/SciTools/projects/5?pane=info
