<unreleased>
************

This document explains the changes made to Iris for this release
(:doc:`View all changes <index>`.)


.. contents:: Skip to section:
   :local:
   :depth: 3


Features
========

* Stephen Moseley greatly enhanced the :mod:`~iris.fileformats.nimrod` module
  to provide richer meta-data translation when loading ``Nimrod`` data into
  cubes. This covers most known operational use-cases. (:pull:`3647`)

* Stephen Worsley improved the handling of :class:`iris.coord.CellMeasure` in
  the statistical operations :meth:`iris.cube.Cube.collapsed`,
  :meth:`iris.cube.Cube.aggregated_by` and
  :meth:`iris.cube.Cube.rolling_window`. These previously removed every
  :class:`iris.coord.CellMeasure` attached to the cube.  Now, a
  :class:`iris.coord.CellMeasure` will only be removed if it is associated with
  an axis over which the statistic is being run. (:pull:`3549`)

* Stephen Worsley, Patrick Peglar and Anna Booton added support for
  `CF Ancillary Data`_ variables, which can be loaded from and saved to
  NetCDF-CF files. Support for `Quality Flags`_ is also provided to ensure they
  load and save with appropriate units. (:pull:`3800`)

* Bouwe Andela implemented lazy regridding for the
  :class:`~iris.analysis.Linear`, :class:`~iris.analysis.Nearest`, and
  :class:`~iris.analysis.AreaWeighted` regridding schemes. (:pull:`3701`)


Dependency Updates
==================

* Bill Little unpinned Iris to use the latest version of `Matplotlib`_.
  Supporting ``Iris`` for both ``Python2`` and ``Python3`` had resulted in
  pinning our dependency on `Matplotlib`_ at ``v2.x``.  But this is no longer
  necessary now that ``Python2`` support has been dropped. (:pull:`3468`)

* Stephen Worsley and Martin Yeo unpinned Iris to use the latest version of
  `Proj <https://github.com/OSGeo/PROJ>`_. (:pull:`3762`)

* Stephen Worsley and Martin Yeo pinned Iris to require
  `Cartopy <https://github.com/SciTools/cartopy>`_ >= 0.18, in
  order to remain compatible with the latest version of `Matplotlib`_.
  (:pull:`3762`)

* Stephen Worsley and Martin Yeo removed GDAL from the extensions dependency
  group. We no longer consider it to be an extension. (:pull:`3762`)

* Bill Little improved the developer set up process. Configuring Iris and
  :ref:`installing_from_source` as a developer, with all the required package
  dependencies is now easier with our curated conda environment YAML files.
  (:pull:`3812`)


Bugs Fixed
==========

* Stephen Worsley fixed :meth:`~iris.Cube.cube.remove_coord` to now also remove
  derived coordinates, by removing aux_factories. (:pull:`3641`)

* Jon Seddon fixed ``isinstance(cube, collections.Iterable)`` to now behave as
  expected if a :class:`~iris.cube.Cube` is iterated over, while also ensuring
  that ``TypeError`` is still raised. (Fixed by setting the ``__iter__()``
  method in :class:`~iris.cube.Cube` to ``None``). (:pull:`3656`)

* Stephen Worsley enabled cube concatenation along an axis shared by cell
  measures; these cell measures are now concatenated together in the resulting
  cube. Such a scenario would previously cause concatenation to inappropriately
  fail. (:pull:`3566`)

* Stephen Worsley newly included :class:`~iris.coords.CellMeasure`s in
  :class:`~iris.cube.Cube` copy operations. Previously copying a
  :class:`~iris.cube.Cube` would ignore any attached
  :class:`~iris.coords.CellMeasure`. (:pull:`3546`)

* Bill Little set a :class:`~iris.coords.CellMeasure`'s
  ``measure`` attribute to have a default value of ``area``.
  Previously, the ``measure`` was provided as a keyword argument to
  :class:`~iris.coords.CellMeasure` with a default value of ``None``, which
  caused a ``TypeError`` when no ``measure`` was provided, since ``area`` or
  ``volume`` are the only accepted values. (:pull:`3533`)

* Martin Yeo set **all** plot types in `iris.plot` to now use
  `matplotlib.dates.date2num
  <https://matplotlib.org/api/dates_api.html#matplotlib.dates.date2num>`_
  to format date/time coordinates for use on a plot axis (previously
  :meth:`~iris.plot.pcolor` and :meth:`~iris.plot.pcolormesh` did not include
  this behaviour). (:pull:`3762`)

* Martin Yeo changed date/time axis labels in `iris.quickplot` to now
  **always** be based on the ``epoch`` used in `matplotlib.dates.date2num
  <https://matplotlib.org/api/dates_api.html#matplotlib.dates.date2num>`_
  (previously would take the unit from a time coordinate, if present, even
  though the coordinate's value had been changed via ``date2num``).
  (:pull:`3762`)

* Patrick Peglar newly included attributes of cell measures in NETCDF-CF file
  loading; they were previously being discarded. They are now available on the
  :class:`~iris.coords.CellMeasure` in the loaded :class:`~iris.cube.Cube`.
  (:pull:`3800`)


Incompatible Changes
====================

* Patrick Peglar rationalised :class:`~iris.cube.CubeList` extraction
  methods:

  The method :meth:`~iris.cube.CubeList.extract_strict`, and the ``strict``
  keyword to :meth:`~iris.cube.CubeList.extract` method have been removed, and
  are replaced by the new routines :meth:`~iris.cube.CubeList.extract_cube` and
  :meth:`~iris.cube.CubeList.extract_cubes`.
  The new routines perform the same operation, but in a style more like other
  ``Iris`` functions such as :meth:`~iris.load_cube` and :meth:`~iris.load_cubes`.
  Unlike ``strict`` extraction, the type of return value is now completely
  consistent : :meth:`~iris.cube.CubeList.extract_cube` always returns a
  :class:`~iris.cube.Cube`, and :meth:`~iris.cube.CubeList.extract_cubes`
  always returns an :class:`iris.cube.CubeList` of a length equal to the
  number of constraints. (:pull:`3715`)

* Patrick Peglar removed the former function ``iris.analysis.coord_comparison``.
  (:pull:`3562`)

* Bill Little moved the :func:`iris.experimental.equalise_cubes.equalise_attributes`
  function from the :mod:`iris.experimental` module into the
  :mod:`iris.util` module.  Please use the :func:`iris.util.equalise_attributes`
  function instead. (:pull:`3527`)

* Bill Little removed the :mod:`iris.experimental.concatenate` module. In
  ``v1.6.0`` the experimental ``concatenate`` functionality was moved to the
  :meth:`iris.cube.CubeList.concatenate` method.  Since then, calling the
  :func:`iris.experimental.concatenate.concatenate` function raised an
  exception. (:pull:`3523`)

* Stephen Worsley changed Iris objects loaded from NetCDF-CF files to have
  ``units='unknown'`` where the corresponding NetCDF variable has no ``units``
  property. Previously these cases defaulted to ``units='1'``. (:pull:`3795`)


Internal
========

* Stephen Worsley changed the numerical values in tests involving the Robinson
  projection due to improvements made in `Proj <https://github.com/OSGeo/PROJ>`_.
  (:pull:`3762`) (see also
  `proj#1292 <https://github.com/OSGeo/PROJ/pull/1292>`_ and
  `proj#2151 <https://github.com/OSGeo/PROJ/pull/2151>`_)

* Stephen Worsley changed tests to account for more detailed descriptions of
  projections in `GDAL <https://github.com/OSGeo/gdal>`_. (:pull:`3762`)
  (`see also GDAL#1185 <https://github.com/OSGeo/gdal/pull/1185>`_)

* Stephen Worsley changed tests to account for
  `GDAL <https://github.com/OSGeo/gdal>`_ now saving fill values for data
  without masked points. (:pull:`3762`)

* Martin Yeo changed every graphics test that includes `Cartopy's coastlines
  <https://scitools.org.uk/cartopy/docs/latest/matplotlib/
  geoaxes.html?highlight=coastlines#cartopy.mpl.geoaxes.GeoAxes.coastlines>`_
  to account for new adaptive coastline scaling. (:pull:`3762`) (`see also
  cartopy#1105 <https://github.com/SciTools/cartopy/pull/1105>`_)

* Martin Yeo changed graphics tests to account for some new default grid-line
  spacing in `Cartopy <https://github.com/SciTools/cartopy>`_. (:pull:`3762`)
  (`see also cartopy#1117 <https://github.com/SciTools/cartopy/pull/1117>`_)

* Martin Yeo added additional acceptable graphics test targets to account for
  very minor changes in `Matplotlib`_ version 3.3 (colormaps, fonts and axes
  borders). (:pull:`3762`)


Deprecations
============

* Stephen Worsley removed the deprecated :class:`iris.Future` flags
  ``cell_date_time_objects``, ``netcdf_promote``, ``netcdf_no_unlimited`` and
  ``clip_latitudes``. (:pull:`3459`)

* Stephen Worsley changed :attr:`iris.fileformats.pp.PPField.lbproc` is be an
  ``int``. The deprecated attributes ``flag1``, ``flag2`` etc. have been
  removed from it. (:pull:`3461`).


Documentation
=============

* Tremain Knight moved the
  :ref:`sphx_glr_generated_gallery_oceanography_plot_orca_projection.py`
  from the general part of the gallery to oceanography. (:pull:`3761`)

* Tremain Knight updated documentation to use a modern sphinx theme and be
  served from https://scitools-iris.readthedocs.io/en/latest/. (:pull:`3752`)

* Bill Little added support for the
  `black <https://black.readthedocs.io/en/stable/>`_ code formatter. This is
  now automatically checked on GitHub PRs, replacing the older, unittest-based
  "iris.tests.test_coding_standards.TestCodeFormat". Black provides automatic
  code format correction for most IDEs.  See the new developer guide section on
  :ref:`iris_code_format`. (:pull:`3518`)

* Tremain Knight refreshed the :ref:`whats_new_contributions` for the
  :ref:`iris_whatsnew`. This includes always creating the ``latest`` what's new
  page so it appears on the latest documentation at
  https://scitools-iris.readthedocs.io/en/latest/whatsnew. This resolves
  :issue:`2104` and :issue:`3451`.  Also updated the
  :ref:`iris_development_releases_steps` to follow when making a release.
  (:pull:`3769`)

* Tremain Knight enabled the PDF creation of the documentation on the
  `Read the Docs`_ service. The PDF may be accessed by clicking on the version
  at the bottom of the side bar, then selecting ``PDF`` from the ``Downloads``
  section. (:pull:`3765`)

* Stephen Worsley added a warning to the
  :func:`iris.analysis.cartography.project` function regarding its behaviour on
  projections with non-rectangular boundaries. (:pull:`3762`)

* Stephen Worsley added the :ref:`cube_maths_combining_units` section to the
  user guide to clarify how ``Units`` are handled during cube arithmetic.
  (:pull:`3803`)

.. _Read the Docs: https://scitools-iris.readthedocs.io/en/latest/
.. _Matplotlib: https://matplotlib.org/
.. _CF Ancillary Data: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#ancillary-data
.. _Quality Flags: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#flags
