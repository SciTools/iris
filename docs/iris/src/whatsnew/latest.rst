<unreleased>
************

This document explains the changes made to Iris for this release
(:doc:`View all changes <index>`.)


.. contents:: Skip to section:
   :local:
   :depth: 3


Features
========

* The :mod:`~iris.fileformats.nimrod` module provides richer meta-data translation
  when loading ``Nimrod`` data into cubes. This covers most known
  operational use-cases.

* Statistical operations :meth:`iris.cube.Cube.collapsed`,
  :meth:`iris.cube.Cube.aggregated_by` and :meth:`iris.cube.Cube.rolling_window`
  previously removed every :class:`iris.coord.CellMeasure` attached to the
  cube.  Now, a :class:`iris.coord.CellMeasure` will only be removed if it is
  associated with an axis over which the statistic is being run.

* Supporting ``Iris`` for both ``Python2`` and ``Python3`` resulted in pinning our
  dependency on `Matplotlib`_ at ``v2.x``.  Now that ``Python2`` support has
  been dropped, ``Iris`` is free to use the latest version of `Matplotlib`_.

* `CF Ancillary Data`_ variables are now supported, and can be loaded from and
  saved to NetCDF-CF files. Support for `Quality Flags`_ is also provided to
  ensure they load and save with appropriate units. See :pull:`3800`.

* Lazy regridding with the :class:`~iris.analysis.Linear`,
  :class:`~iris.analysis.Nearest`, and
  :class:`~iris.analysis.AreaWeighted` regridding schemes.
  See :pull:`3701`.


Dependency Updates
==================

* Iris now supports the latest version of `Proj <https://github.com/OSGeo/PROJ>`_.

* Iris now requires `Cartopy <https://github.com/SciTools/cartopy>`_ >= 0.18 in
  order to remain compatible with the latest version of `Matplotlib`_.

* GDAL is removed from the extensions dependency group. We no longer consider it to
  be an extension.

* Configuring Iris and :ref:`installing_from_source` as a developer, with all the
  required package dependencies is now easier with our curated conda environment
  YAML files. See :pull:`3812`.

Bugs Fixed
==========

* The method :meth:`~iris.Cube.cube.remove_coord` would fail to remove derived
  coordinates, will now remove derived coordinates by removing aux_factories.

* The ``__iter__()`` method in :class:`~iris.cube.Cube` was set to ``None``.
  ``TypeError`` is still raised if a :class:`~iris.cube.Cube` is iterated over
  but ``isinstance(cube, collections.Iterable)`` now behaves as expected.

* Concatenating cubes along an axis shared by cell measures would cause
  concatenation to inappropriately fail.  These cell measures are now
  concatenated together in the resulting cube.

* Copying a cube would previously ignore any attached
  :class:`~iris.coords.CellMeasure`.  These are now copied over.

* A :class:`~iris.coords.CellMeasure` requires a string ``measure`` attribute
  to be defined, which can only have a value of ``area`` or ``volume``.
  Previously, the ``measure`` was provided as a keyword argument to
  :class:`~iris.coords.CellMeasure` with an default value of ``None``, which
  caused a ``TypeError`` when no ``measure`` was provided.  The default value
  of ``area`` is now used.

* **All** plot types in `iris.plot` now use `matplotlib.dates.date2num
  <https://matplotlib.org/api/dates_api.html#matplotlib.dates.date2num>`_
  to format date/time coordinates for use on a plot axis (previously
  :meth:`~iris.plot.pcolor` and :meth:`~iris.plot.pcolormesh` did not include
  this behaviour).

* Date/time axis labels in `iris.quickplot` are now **always** based on the
  ``epoch`` used in `matplotlib.dates.date2num
  <https://matplotlib.org/api/dates_api.html#matplotlib.dates.date2num>`_
  (previously would take the unit from a time coordinate, if present, even
  though the coordinate's value had been changed via ``date2num``).

* Attributes of cell measures in NetCDF-CF files were being discarded during
  loading. They are now available on the :class:`~iris.coords.CellMeasure` in
  the loaded :class:`~iris.cube.Cube`. See :pull:`3800`.

* the netcdf loader can now handle any grid-mapping variables with missing
  ``false_easting`` and ``false_northing`` properties, which was previously
  failing for some coordinate systems.
  See :issue:`3629`.


Incompatible Changes
====================

* The method :meth:`~iris.cube.CubeList.extract_strict`, and the ``strict``
  keyword to :meth:`~iris.cube.CubeList.extract` method have been removed, and
  are replaced by the new routines :meth:`~iris.cube.CubeList.extract_cube` and
  :meth:`~iris.cube.CubeList.extract_cubes`.
  The new routines perform the same operation, but in a style more like other
  ``Iris`` functions such as :meth:`~iris.load_cube` and :meth:`~iris.load_cubes`.
  Unlike ``strict`` extraction, the type of return value is now completely
  consistent : :meth:`~iris.cube.CubeList.extract_cube` always returns a
  :class:`~iris.cube.Cube`, and :meth:`~iris.cube.CubeList.extract_cubes`
  always returns an :class:`iris.cube.CubeList` of a length equal to the
  number of constraints.

* The former function ``iris.analysis.coord_comparison`` has been removed.

* The :func:`iris.experimental.equalise_cubes.equalise_attributes` function
  has been moved from the :mod:`iris.experimental` module into the
  :mod:`iris.util` module.  Please use the :func:`iris.util.equalise_attributes`
  function instead.

* The :mod:`iris.experimental.concatenate` module has now been removed. In
  ``v1.6.0`` the experimental ``concatenate`` functionality was moved to the
  :meth:`iris.cube.CubeList.concatenate` method.  Since then, calling the
  :func:`iris.experimental.concatenate.concatenate` function raised an
  exception.

* When loading data from NetCDF-CF files, where a variable has no ``units``
  property, the corresponding Iris object will have ``units='unknown'``.
  Prior to Iris ``3.0.0``, these cases defaulted to ``units='1'``.
  See :pull:`3795`.

* `Simon Peatman <https://github.com/SimonPeatman>`_ added attribute
  ``var_name`` to coordinates created by the
  :func:`iris.analysis.trajectory.interpolate` function.  This prevents
  duplicate coordinate errors in certain circumstances.  (:pull:`3718`).


Internal
========

* Changed the numerical values in tests involving the Robinson projection due
  to improvements made in `Proj <https://github.com/OSGeo/PROJ>`_ (see
  `proj#1292 <https://github.com/OSGeo/PROJ/pull/1292>`_ and
  `proj#2151 <https://github.com/OSGeo/PROJ/pull/2151>`_).

* Change tests to account for more detailed descriptions of projections in
  `GDAL <https://github.com/OSGeo/gdal>`_ 
  (`see GDAL#1185 <https://github.com/OSGeo/gdal/pull/1185>`_).

* Change tests to account for `GDAL <https://github.com/OSGeo/gdal>`_ now
  saving fill values for data without masked points.

* Changed every graphics test that includes `Cartopy's coastlines
  <https://scitools.org.uk/cartopy/docs/latest/matplotlib/
  geoaxes.html?highlight=coastlines#cartopy.mpl.geoaxes.GeoAxes.coastlines>`_
  to account for new adaptive coastline scaling (`see cartopy#1105
  <https://github.com/SciTools/cartopy/pull/1105>`_).

* Changed graphics tests to account for some new default grid-line spacing in
  `Cartopy <https://github.com/SciTools/cartopy>`_ (`part of cartopy#1117
  <https://github.com/SciTools/cartopy/pull/1117>`_).

* Additional acceptable graphics test targets to account for very minor changes
  in `Matplotlib`_ version 3.3 (colormaps, fonts and axes borders).


Deprecations
============

* The deprecated :class:`iris.Future` flags ``cell_date_time_objects``,
  ``netcdf_promote``, ``netcdf_no_unlimited`` and ``clip_latitudes`` have
  been removed.

* :attr:`iris.fileformats.pp.PPField.lbproc` is now an ``int``. The
  deprecated attributes ``flag1``, ``flag2`` etc. have been removed from it.


Documentation
=============

* Moved the :ref:`sphx_glr_generated_gallery_oceanography_plot_orca_projection.py`
  from the general part of the gallery to oceanography.

* Updated documentation to use a modern sphinx theme and be served from
  https://scitools-iris.readthedocs.io/en/latest/.

* Added support for the `black <https://black.readthedocs.io/en/stable/>`_ code
  formatter.  This is now automatically checked on GitHub PRs, replacing the
  older, unittest-based "iris.tests.test_coding_standards.TestCodeFormat".
  Black provides automatic code format correction for most IDEs.  See the new
  developer guide section on :ref:`iris_code_format`.

* Refreshed the :ref:`whats_new_contributions` for the :ref:`iris_whatsnew`.
  This includes always creating the ``latest`` what's new page so it appears
  on the latest documentation at
  https://scitools-iris.readthedocs.io/en/latest/whatsnew.  This resolves
  :issue:`2104` and :issue:`3451`.  Also updated the
  :ref:`iris_development_releases_steps` to follow when making a release.

* Enabled the PDF creation of the documentation on the `Read the Docs`_ service.
  The PDF may be accessed by clicking on the version at the bottom of the side
  bar, then selecting ``PDF`` from the ``Downloads`` section.

* Added a warning to the :func:`iris.analysis.cartography.project` function
  regarding its behaviour on projections with non-rectangular boundaries.

* Added the :ref:`cube_maths_combining_units` section to the user guide to
  clarify how ``Units`` are handled during cube arithmetic.

.. _Read the Docs: https://scitools-iris.readthedocs.io/en/latest/
.. _Matplotlib: https://matplotlib.org/
.. _CF Ancillary Data: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#ancillary-data
.. _Quality Flags: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#flags
