.. include:: ../common_links.inc

|iris_version| |build_date| [unreleased]
****************************************

This document explains the changes made to Iris for this release
(:doc:`View all changes <index>`.)


.. dropdown:: :opticon:`report` |iris_version| Release Highlights
   :container: + shadow
   :title: text-primary text-center font-weight-bold
   :body: bg-light
   :animate: fade-in
   :open:

   The highlights for this major/minor release of Iris include:

   * We're so proud to fully support `@ed-hawkins`_ and `#ShowYourStripes`_ ❤️

   And finally, get in touch with us on :issue:`GitHub<new/choose>` if you have
   any issues or feature requests for improving Iris. Enjoy!


📢 Announcements
================

#. Congratulations to `@ESadek-MO`_ who has become a core developer for Iris! 🎉
#. Welcome and congratulations to `@HGWright`_, `@scottrobinson02`_ and
   `@agriyakhetarpal`_ who made their first contributions to Iris! 🎉


✨ Features
===========

#. `@bsherratt`_ added support for plugins - see the corresponding
   :ref:`documentation page<community_plugins>` for further information.
   (:pull:`5144`)

#. `@rcomer`_ enabled lazy evaluation of :obj:`~iris.analysis.RMS` calcuations
   with weights. (:pull:`5017`)

#. `@schlunma`_ allowed the usage of cubes, coordinates, cell measures, or
   ancillary variables as weights for cube aggregations
   (:meth:`iris.cube.Cube.collapsed`, :meth:`iris.cube.Cube.aggregated_by`, and
   :meth:`iris.cube.Cube.rolling_window`). This automatically adapts cube units
   if necessary. (:pull:`5084`)

#. `@lbdreyer`_ and `@trexfeathers`_ (reviewer)  added :func:`iris.plot.hist` 
   and :func:`iris.quickplot.hist`. (:pull:`5189`)

#. `@tinyendian`_ edited :func:`~iris.analysis.cartography.rotate_winds` to
   enable lazy computation of rotated wind vector components (:issue:`4934`,
   :pull:`4972`)

#. `@pp-mo`_ and  `@lbdreyer`_ supported delayed saving of lazy data, when writing to
   the netCDF file format.  See : :ref:`delayed netCDF saves <delayed_netcdf_save>`.
   Also with significant input from `@fnattino`_.
   (:pull:`5191`)


🐛 Bugs Fixed
=============

#. `@trexfeathers`_ and `@pp-mo`_ made Iris' use of the `netCDF4`_ library
   thread-safe. (:pull:`5095`)

#. `@ESadek-MO`_ removed check and error raise for saving
   cubes with masked :class:`iris.coords.CellMeasure`.
   (:issue:`5147`, :pull:`5181`)

#. `@scottrobinson02`_ fixed :class:`iris.util.new_axis` creating an anonymous new
   dimension, when the scalar coord provided is already a dim coord.
   (:issue:`4415`, :pull:`5194`)

#. `@HGWright`_ and `@trexfeathers`_ (reviewer) changed the way
   :class:`~iris.coords.CellMethod` are printed to be more CF compliant.
   (:pull:`5224`)


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

#. `@rcomer`_ clarified instructions for updating gallery tests. (:pull:`5100`)
#. `@tkknight`_ unpinned ``pydata-sphinx-theme`` and set the default to use
   the light version (not dark) while we make the docs dark mode friendly
   (:pull:`5129`)

#. `@jonseddon`_ updated the citation to a more recent version of Iris. (:pull:`5116`)

#. `@rcomer`_ linked the :obj:`~iris.analysis.PERCENTILE` aggregator from the
   :obj:`~iris.analysis.MEDIAN` docstring, noting that the former handles lazy
   data. (:pull:`5128`)

#. `@trexfeathers`_ updated the WSL link to Microsoft's latest documentation,
   and removed an ECMWF link in the ``v1.0`` What's New that was failing the
   linkcheck CI. (:pull:`5109`)

#. `@trexfeathers`_ added a new top-level :doc:`/community/index` section,
   as a one-stop place to find out about getting involved, and how we relate
   to other projects. (:pull:`5025`)

#. The **Iris community**, with help from the **Xarray community**, produced
   the :doc:`/community/iris_xarray` page, highlighting the similarities and
   differences between the two packages. (:pull:`5025`)

#. `@bjlittle`_ added a new section to the `README.md`_ to show our support
   for the outstanding work of `@ed-hawkins`_ et al for `#ShowYourStripes`_.
   (:pull:`5141`)

#. `@HGWright`_ fixed some typo's from Gitwash. (:pull:`5145`)


💼 Internal
===========

#. `@fnattino`_ changed the order of ``ncgen`` arguments in the command to
   create NetCDF files for testing  (caused errors on OS X). (:pull:`5105`)

#. `@rcomer`_ removed some old infrastructure that printed test timings.
   (:pull:`5101`)

#. `@lbdreyer`_ and `@trexfeathers`_ (reviewer) added coverage testing. This
   can be enabled by using the "--coverage" flag when running the tests with
   nox i.e. ``nox --session tests -- --coverage``. (:pull:`4765`)

#. `@lbdreyer`_ and `@trexfeathers`_ (reviewer) removed the ``--coding-tests``
   option from Iris' test runner. (:pull:`4765`)

#. `@lbdreyer`_ removed the Iris TestRunner. Tests are now run via nox or
   pytest. (:pull:`5205`)

#. `@agriyakhetarpal`_ prevented the GitHub action for publishing releases to
   PyPI from running in forks. (:pull:`5220`)

#. `@trexfeathers`_ moved the benchmark runner conveniences from ``noxfile.py``
   to a dedicated ``benchmarks/bm_runner.py``. (:pull:`5215`)

#. `@bjlittle`_ follow-up to :pull:`4972`, enforced ``dask>=2022.09.0`` minimum
   pin for first use of `dask.array.ma.empty_like`_ and replaced `@tinyendian`_
   workaround. (:pull:`5225`)


.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:

.. _@fnattino: https://github.com/fnattino
.. _@ed-hawkins: https://github.com/ed-hawkins
.. _@scottrobinson02: https://github.com/scottrobinson02
.. _@agriyakhetarpal: https://github.com/agriyakhetarpal
.. _@tinyendian: https://github.com/tinyendian


.. comment
    Whatsnew resources in alphabetical order:

.. _#ShowYourStripes: https://showyourstripes.info/s/globe/
.. _README.md: https://github.com/SciTools/iris#-----
.. _dask.array.ma.empty_like: https://docs.dask.org/en/stable/generated/dask.array.ma.empty_like.html
