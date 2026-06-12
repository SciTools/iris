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

#. :user:`bjlittle` extended the :meth:`~iris.coords.Coord.cell` and
   :meth:`~iris.coords.Coord.cells` methods to allow users to specify that they
   want :class:`~datetime.datetime` compatible objects returned within each
   generated :class:`~iris.coords.Coord.Cell` from a temporal coordinate. This
   is achieved by passing the keyword argument ``pydate=True``.
   (:issue:`7112`, :pull:`7146`)


🐛 Bugs Fixed
=============

#. :user:`gaoflow` fixed an error when computing (e.g. saving) a scalar lazy
   cube whose units had been converted with
   :meth:`~iris.cube.Cube.convert_units`. The unit conversion could yield a
   plain Python scalar for the 0-dimensional block, which Dask was then unable
   to store. (:issue:`6965`)


💣 Incompatible Changes
=======================

#. N/A


🚀 Performance Enhancements
===========================

#. `@trexfeathers`_ improved the speed of field iteration when reading PP files.
   Up to 3x speed up has been seen, dependending on the circumstances.
   (:pull:`7089`)


🔥 Deprecations
===============

#. :user:`bjlittle` deprecated the :mod:`iris.analysis.calculus` module containing
   the following public functions:

   * :func:`~iris.analysis.calculus.cube_delta`
   * :func:`~iris.analysis.calculus.curl`
   * :func:`~iris.analysis.calculus.differentiate`
   * :func:`~iris.analysis.calculus.spatial_vectors_with_phenom_name`

   Native :class:`~iris.cube.Cube` calculus functionality will not be replaced
   and is scheduled for removal in ``Iris`` 4.0.0. (:issue:`6262`, :pull:`7102`)


🔗 Dependencies
===============

#. `@trexfeathers`_ and `@tkknight`_ removed the maximum pin for the
   PyData Sphinx Theme (used in the docs). (:issue:`6885`, :pull:`7053`)

#. `@tkknight`_ added a minimum pin for the PyData Sphinx Theme as we use the
   collapse sidebar feature introduced in 0.17.0. (:pull:`7060`)

#. `@tkknight`_ updated a dependency in the Read The Docs configuration file to
   use the latest python. (:pull:`7084`)

#. `@tkknight`_ added a dependency named sphinx-llm to generate summaries
   that LLMs can understand, `llms.txt` and `llms-full.txt`.  (:pull:`7105`)

#. `@tkknight`_ added a dependency named sphinx-sitemap to generate sitemap.xml for
   the documentaiton. (:pull:`7100`)


📚 Documentation
================

#. `@trexfeathers`_ and `@tkknight`_ made the docs compatible with the latest
   versions of PyData Sphinx Theme (>=0.16). (:issue:`6885`, :pull:`7053`)

#. `@tkknight`_ enabled the theme option to collapse the sidebar.  Note, it only
   appears once you click on a link away from the landing page.  Also moved
   the search box to the top navigation bar. (:pull:`7060`)

#. `@trexfeathers`_ switched to using the official URL of the `cf-checker`_, after
   our previous URL of choice was taken down. (:pull:`7072`)

#. `@tkknight`_ updated the voted table that uses datatables to not highlight the
   sorted column or row as is uses the incorrect theme color (light). Also updated
   the datatables version from 2.3.2 to 2.3.8. (:pull:`7079`)

#. `@tkknight`_ updated the documentation to generate summaries
   that LLMs can understand, `llms.txt` and `llms-full.txt`. (:pull:`7105`)

#. `@tkknight`_ updated the documentation to generate a sitemap.xml files for the
   stable version. (:pull:`7100`)

#. `@trexfeathers`_ and `@hdyson`_ clarified that
   :func:`iris.analysis.cartography.area_weights` requires 1-dimensional lat and lon 
   coordinates on the input :class:`~iris.cube.Cube`. (:pull:`7118`)

#. :user:`bjlittle` Added the custom `sphinx`_ ``readingtime`` directive to
   automatically estimate the audiance reading time of a page and render a
   branded banner in-situ. (:pull:`7150`)


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

#. `@SgtVarmint`_ migrated codebase from ``os.path`` to ``pathlib.Path`` where possible
   (:issue:`4523`, :pull:`7087`)

#. `@bjlittle`_ and `@trexfeathers`_ added the `Zizmor`_ pre-commit hook to monitor for
   security vulnerabilities in Iris' GitHub Actions workflows, and then actioned
   Zizmor's recommendations to harden the workflows. (:pull:`7138`)

#. `@trexfeathers` set the link checking workflow to accept redirect HTTP codes, as
   the reports were getting too noisy. (:pull:`7148`)

#. `@HGWright`_ changed the default of the private switch :obj:`~iris.loading._LAZY_DERIVED_LOADING` (formerly `.CONCRETE_DERIVED_LOADING`)
   for controlling laziness of coordinates from pp loading, now the switch must be set to True for lazy loading to be enabled.
   Note: this object is temporary and is likely to be replaced by a permanent solution or else be renamed.
   (:issue:`7094`, :pull:`7134`)

.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:

.. _@hdyson: https://github.com/hdyson


.. comment
    Whatsnew resources in alphabetical order:
.. _cf-checker: https://github.com/cedadev/cf-checker
.. _Zizmor: https://github.com/zizmorcore/zizmor
