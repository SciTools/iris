.. include:: ../common_links.inc

<unreleased>
************

This document explains the changes made to Iris for this release
(:doc:`View all changes <index>`.)


📢 Announcements
================

* Congratulations to `@bouweandela`_, `@jvegasbsc`_, and `@zklaus`_ who
  recently became Iris core developers. They bring a wealth of expertise to the
  team, and are using Iris to underpin `ESMValTool`_ - "*A community diagnostic
  and performance metrics tool for routine evaluation of Earth system models
  in CMIP*". Welcome aboard! 🎉


✨ Features
===========

* `@MoseleyS`_ greatly enhanced  the :mod:`~iris.fileformats.nimrod`
  module to provide richer meta-data translation when loading ``Nimrod`` data
  into cubes. This covers most known operational use-cases. (:pull:`3647`)

* `@stephenworsley`_ improved the handling of
  :class:`iris.coords.CellMeasure`\ s in the :class:`~iris.cube.Cube`
  statistical operations :meth:`~iris.cube.Cube.collapsed`,
  :meth:`~iris.cube.Cube.aggregated_by` and
  :meth:`~iris.cube.Cube.rolling_window`. These previously removed every
  :class:`~iris.coords.CellMeasure` attached to the cube.  Now, a
  :class:`~iris.coords.CellMeasure` will only be removed if it is associated
  with an axis over which the statistic is being run. (:pull:`3549`)

* `@stephenworsley`_, `@pp-mo`_ and `@abooton`_ added support for
  `CF Ancillary Data`_ variables.  These are created as
  :class:`iris.coords.AncillaryVariable`, and appear as components of cubes
  much like :class:`~iris.coords.AuxCoord`\ s, with the new
  :class:`~iris.cube.Cube` methods
  :meth:`~iris.cube.Cube.add_ancillary_variable`,
  :meth:`~iris.cube.Cube.remove_ancillary_variable`,
  :meth:`~iris.cube.Cube.ancillary_variable`,
  :meth:`~iris.cube.Cube.ancillary_variables` and
  :meth:`~iris.cube.Cube.ancillary_variable_dims`.
  They are loaded from and saved to NetCDF-CF files.  Special support for
  `Quality Flags`_ is also provided, to ensure they load and save with
  appropriate units. (:pull:`3800`)

* `@bouweandela`_ implemented lazy regridding for the
  :class:`~iris.analysis.Linear`, :class:`~iris.analysis.Nearest`, and
  :class:`~iris.analysis.AreaWeighted` regridding schemes. (:pull:`3701`)


🐛 Bugs Fixed
=============

* `@stephenworsley`_ fixed :meth:`~iris.cube.Cube.remove_coord` to now also
  remove derived coordinates by removing aux_factories. (:pull:`3641`)

* `@jonseddon`_ fixed ``isinstance(cube, collections.Iterable)`` to now behave
  as expected if a :class:`~iris.cube.Cube` is iterated over, while also
  ensuring that ``TypeError`` is still raised. (Fixed by setting the
  ``__iter__()`` method in :class:`~iris.cube.Cube` to ``None``).
  (:pull:`3656`)

* `@stephenworsley`_ enabled cube concatenation along an axis shared by cell
  measures; these cell measures are now concatenated together in the resulting
  cube. Such a scenario would previously cause concatenation to inappropriately
  fail. (:pull:`3566`)

* `@stephenworsley`_ newly included :class:`~iris.coords.CellMeasure`\ s in
  :class:`~iris.cube.Cube` copy operations. Previously copying a
  :class:`~iris.cube.Cube` would ignore any attached
  :class:`~iris.coords.CellMeasure`. (:pull:`3546`)

* `@bjlittle`_ set a :class:`~iris.coords.CellMeasure`'s
  ``measure`` attribute to have a default value of ``area``.
  Previously, the ``measure`` was provided as a keyword argument to
  :class:`~iris.coords.CellMeasure` with a default value of ``None``, which
  caused a ``TypeError`` when no ``measure`` was provided, since ``area`` or
  ``volume`` are the only accepted values. (:pull:`3533`)

* `@trexfeathers`_ set **all** plot types in :mod:`iris.plot` to now use
  `matplotlib.dates.date2num`_ to format date/time coordinates for use on a plot
  axis (previously :meth:`~iris.plot.pcolor` and :meth:`~iris.plot.pcolormesh`
  did not include this behaviour). (:pull:`3762`)

* `@trexfeathers`_ changed date/time axis labels in :mod:`iris.quickplot` to
  now **always** be based on the ``epoch`` used in `matplotlib.dates.date2num`_
  (previously would take the unit from a time coordinate, if present, even
  though the coordinate's value had been changed via ``date2num``).
  (:pull:`3762`)

* `@pp-mo`_ newly included attributes of cell measures in NETCDF-CF
  file loading; they were previously being discarded. They are now available on
  the :class:`~iris.coords.CellMeasure` in the loaded :class:`~iris.cube.Cube`.
  (:pull:`3800`)

* `@pp-mo`_ fixed the netcdf loader to now handle any grid-mapping
  variables with missing ``false_easting`` and ``false_northing`` properties,
  which was previously failing for some coordinate systems. See :issue:`3629`.
  (:pull:`3804`)

* `@stephenworsley`_ changed the way tick labels are assigned from string coords.
  Previously, the first tick label would occasionally be duplicated. This also
  removes the use of Matplotlib's deprecated ``IndexFormatter``. (:pull:`3857`)


💣 Incompatible Changes
=======================

* `@pp-mo`_ rationalised :class:`~iris.cube.CubeList` extraction
  methods:

  The former method ``iris.cube.CubeList.extract_strict``, and the ``strict``
  keyword of the :meth:`~iris.cube.CubeList.extract` method have been removed,
  and are replaced by the new routines :meth:`~iris.cube.CubeList.extract_cube`
  and :meth:`~iris.cube.CubeList.extract_cubes`.
  The new routines perform the same operation, but in a style more like other
  ``Iris`` functions such as :meth:`~iris.load_cube` and :meth:`~iris.load_cubes`.
  Unlike ``strict`` extraction, the type of return value is now completely
  consistent : :meth:`~iris.cube.CubeList.extract_cube` always returns a
  :class:`~iris.cube.Cube`, and :meth:`~iris.cube.CubeList.extract_cubes`
  always returns an :class:`iris.cube.CubeList` of a length equal to the
  number of constraints. (:pull:`3715`)

* `@pp-mo`_ removed the former function
  ``iris.analysis.coord_comparison``. (:pull:`3562`)

* `@bjlittle`_ moved the
  :func:`iris.experimental.equalise_cubes.equalise_attributes` function from
  the :mod:`iris.experimental` module into the :mod:`iris.util` module.  Please
  use the :func:`iris.util.equalise_attributes` function instead.
  (:pull:`3527`)

* `@bjlittle`_ removed the module ``iris.experimental.concatenate``. In
  ``v1.6.0`` the experimental ``concatenate`` functionality was moved to the
  :meth:`iris.cube.CubeList.concatenate` method.  Since then, calling the
  :func:`iris.experimental.concatenate.concatenate` function raised an
  exception. (:pull:`3523`)

* `@stephenworsley`_ changed Iris objects loaded from NetCDF-CF files to have
  ``units='unknown'`` where the corresponding NetCDF variable has no ``units``
  property. Previously these cases defaulted to ``units='1'``.
  This affects loading of coordinates whose file variable has no "units"
  attribute (not valid, under `CF units rules`_):  These will now have units
  of `"unknown"`, rather than `"1"`, which **may prevent the creation of
  a hybrid vertical coordinate**.  While these cases used to "work", this was
  never really correct behaviour. (:pull:`3795`)

* `@SimonPeatman`_ added attribute ``var_name`` to coordinates created by the
  :func:`iris.analysis.trajectory.interpolate` function.  This prevents
  duplicate coordinate errors in certain circumstances. (:pull:`3718`)


🔥 Deprecations
===============

* `@stephenworsley`_ removed the deprecated :class:`iris.Future` flags
  ``cell_date_time_objects``, ``netcdf_promote``, ``netcdf_no_unlimited`` and
  ``clip_latitudes``. (:pull:`3459`)

* `@stephenworsley`_ changed :attr:`iris.fileformats.pp.PPField.lbproc` to be an
  ``int``. The deprecated attributes ``flag1``, ``flag2`` etc. have been
  removed from it. (:pull:`3461`)


🔗 Dependencies
===============

* `@stephenworsley`_, `@trexfeathers`_ and `@bjlittle`_ removed ``Python2``
  support, modernising the codebase by switching to exclusive ``Python3``
  support. (:pull:`3513`)

* `@bjlittle`_ improved the developer set up process. Configuring Iris and
  :ref:`installing_from_source` as a developer with all the required package
  dependencies is now easier with our curated conda environment YAML files.
  (:pull:`3812`)

* `@stephenworsley`_ pinned Iris to require `Dask`_ ``>=2.0``. (:pull:`3460`)

* `@stephenworsley`_ and `@trexfeathers`_ pinned Iris to require
  `Cartopy`_ ``>=0.18``, in order to remain compatible with the latest version
  of `Matplotlib`_. (:pull:`3762`)

* `@bjlittle`_ unpinned Iris to use the latest version of `Matplotlib`_.
  Supporting ``Iris`` for both ``Python2`` and ``Python3`` had resulted in
  pinning our dependency on `Matplotlib`_ at ``v2.x``.  But this is no longer
  necessary now that ``Python2`` support has been dropped. (:pull:`3468`)

* `@stephenworsley`_ and `@trexfeathers`_ unpinned Iris to use the latest version
  of `Proj`_. (:pull:`3762`)

* `@stephenworsley`_ and `@trexfeathers`_ removed GDAL from the extensions
  dependency group. We no longer consider it to be an extension. (:pull:`3762`)


📚 Documentation
================

* `@tkknight`_ moved the
  :ref:`sphx_glr_generated_gallery_oceanography_plot_orca_projection.py`
  from the general part of the gallery to oceanography. (:pull:`3761`)

* `@tkknight`_ updated documentation to use a modern sphinx theme and be
  served from https://scitools-iris.readthedocs.io/en/latest/. (:pull:`3752`)

* `@bjlittle`_ added support for the `black`_ code formatter. This is
  now automatically checked on GitHub PRs, replacing the older, unittest-based
  ``iris.tests.test_coding_standards.TestCodeFormat``. Black provides automatic
  code format correction for most IDEs.  See the new developer guide section on
  :ref:`code_formatting`. (:pull:`3518`)

* `@tkknight`_ and `@trexfeathers`_ refreshed the :ref:`whats_new_contributions`
  for the :ref:`iris_whatsnew`. This includes always creating the ``latest``
  what's new page so it appears on the latest documentation at
  https://scitools-iris.readthedocs.io/en/latest/whatsnew. This resolves
  :issue:`2104`, :issue:`3451`, :issue:`3818`, :issue:`3837`.  Also updated the
  :ref:`iris_development_releases_steps` to follow when making a release.
  (:pull:`3769`, :pull:`3838`, :pull:`3843`)

* `@tkknight`_ enabled the PDF creation of the documentation on the
  `Read the Docs`_ service. The PDF may be accessed by clicking on the version
  at the bottom of the side bar, then selecting ``PDF`` from the ``Downloads``
  section. (:pull:`3765`)

* `@stephenworsley`_ added a warning to the
  :func:`iris.analysis.cartography.project` function regarding its behaviour on
  projections with non-rectangular boundaries. (:pull:`3762`)

* `@stephenworsley`_ added the :ref:`cube_maths_combining_units` section to the
  user guide to clarify how ``Units`` are handled during cube arithmetic.
  (:pull:`3803`)

* `@tkknight`_ overhauled the :ref:`developers_guide` including information on
  getting involved in becoming a contributor and general structure of the
  guide.  This resolves :issue:`2170`, :issue:`2331`, :issue:`3453`,
  :issue:`314`, :issue:`2902`. (:pull:`3852`)

* `@rcomer`_ added argument descriptions to the :class:`~iris.coords.DimCoord`
  docstring. (:pull:`3681`)

* `@tkknight`_ added two url's to be ignored for the ``make linkcheck``.  This
  will ensure the Iris github project is not repeatedly hit during the
  linkcheck for issues and pull requests as it can result in connection
  refused and thus travis-ci_ job failures.  For more information on linkcheck,
  see :ref:`contributing.documentation.testing`.  (:pull:`3873`)

* `@tkknight`_ enabled the napolean_ package that is used by sphinx_ to cater
  for the existing google style docstrings and to also allow for numpy
  docstrings.  This resolves :issue:`3841`. (:pull:`3871`)


💼 Internal
===========

* `@pp-mo`_ and `@lbdreyer`_ removed all Iris test dependencies on `iris-grib`_
  by transferring all relevant content to the `iris-grib`_ repository. (:pull:`3662`,
  :pull:`3663`, :pull:`3664`, :pull:`3665`, :pull:`3666`, :pull:`3669`,
  :pull:`3670`, :pull:`3671`, :pull:`3672`, :pull:`3742`, :pull:`3746`)

* `@lbdreyer`_ and `@pp-mo`_ overhauled the handling of dimensional
  metadata to remove duplication. (:pull:`3422`, :pull:`3551`)

* `@trexfeathers`_ simplified the standard license header for all files, which
  removes the need to repeatedly update year numbers in the header.
  (:pull:`3489`)

* `@stephenworsley`_ changed the numerical values in tests involving the
  Robinson projection due to improvements made in
  `Proj`_. (:pull:`3762`) (see also `Proj#1292`_ and `Proj#2151`_)

* `@stephenworsley`_ changed tests to account for more detailed descriptions of
  projections in `GDAL`_. (:pull:`3762`) (see also `GDAL#1185`_)

* `@stephenworsley`_ changed tests to account for `GDAL`_ now saving fill values
  for data without masked points. (:pull:`3762`)

* `@trexfeathers`_ changed every graphics test that includes `Cartopy's coastlines`_
  to account for new adaptive coastline scaling. (:pull:`3762`)
  (see also `Cartopy#1105`_)

* `@trexfeathers`_ changed graphics tests to account for some new default
  grid-line spacing in `Cartopy`_. (:pull:`3762`) (see also `Cartopy#1117`_)

* `@trexfeathers`_ added additional acceptable graphics test targets to account
  for very minor changes in `Matplotlib`_ version ``3.3`` (colormaps, fonts and
  axes borders). (:pull:`3762`)

* `@rcomer`_ corrected the Matplotlib backend in Iris tests to ignore
  `matplotlib.rcdefaults <https://matplotlib.org/3.1.1/api/matplotlib_configuration_api.html?highlight=rcdefaults#matplotlib.rcdefaults>`_,
  instead the tests will **always** use ``agg``. (:pull:`3846`)

* `@bjlittle`_ migrated the `black`_ support from ``19.10b0`` to ``20.8b1``.
  (:pull:`3866`)

* `@lbdreyer`_ updated the CF standard name table to the latest version: `v75`_.
  (:pull:`3867`)


.. _Read the Docs: https://scitools-iris.readthedocs.io/en/latest/
.. _Matplotlib: https://matplotlib.org/
.. _CF units rules: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#units
.. _CF Ancillary Data: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#ancillary-data
.. _Quality Flags: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#flags
.. _iris-grib: https://github.com/SciTools/iris-grib
.. _Cartopy: https://github.com/SciTools/cartopy
.. _Cartopy's coastlines: https://scitools.org.uk/cartopy/docs/latest/matplotlib/geoaxes.html?highlight=coastlines#cartopy.mpl.geoaxes.GeoAxes.coastlines
.. _Cartopy#1105: https://github.com/SciTools/cartopy/pull/1105
.. _Cartopy#1117: https://github.com/SciTools/cartopy/pull/1117
.. _Dask: https://github.com/dask/dask
.. _matplotlib.dates.date2num: https://matplotlib.org/api/dates_api.html#matplotlib.dates.date2num
.. _Proj: https://github.com/OSGeo/PROJ
.. _black: https://black.readthedocs.io/en/stable/
.. _Proj#1292: https://github.com/OSGeo/PROJ/pull/1292
.. _Proj#2151: https://github.com/OSGeo/PROJ/pull/2151
.. _GDAL: https://github.com/OSGeo/gdal
.. _GDAL#1185: https://github.com/OSGeo/gdal/pull/1185
.. _@MoseleyS: https://github.com/MoseleyS
.. _@stephenworsley: https://github.com/stephenworsley
.. _@pp-mo: https://github.com/pp-mo
.. _@abooton: https://github.com/abooton
.. _@bouweandela: https://github.com/bouweandela
.. _@bjlittle: https://github.com/bjlittle
.. _@trexfeathers: https://github.com/trexfeathers
.. _@jonseddon: https://github.com/jonseddon
.. _@tkknight: https://github.com/tkknight
.. _@lbdreyer: https://github.com/lbdreyer
.. _@SimonPeatman: https://github.com/SimonPeatman
.. _@rcomer: https://github.com/rcomer
.. _@jvegasbsc: https://github.com/jvegasbsc
.. _@zklaus: https://github.com/zklaus
.. _ESMValTool: https://github.com/ESMValGroup/ESMValTool
.. _v75: https://cfconventions.org/Data/cf-standard-names/75/build/cf-standard-name-table.html
