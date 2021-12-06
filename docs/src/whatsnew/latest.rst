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

   The highlights for this minor release of Iris include:

   * We've added support for `UGRID`_ meshes which can now be loaded and attached
     to a cube.

   And finally, get in touch with us on `GitHub`_ if you have any issues or
   feature requests for improving Iris. Enjoy!


📢 Announcements
================

#. Welcome to `@wjbenfold`_, `@tinyendian`_, `@larsbarring`_, `@akuhnregnier`_,
   `@bsherratt`_ and `@aaronspring`_ who made their first contributions to Iris.
   The first of many we hope!
#. Congratulations to `@wjbenfold`_ who has become a core developer for Iris! 🎉


✨ Features
===========

#. `@bjlittle`_, `@pp-mo`_, `@trexfeathers`_ and `@stephenworsley`_ added
   support for unstructured meshes, as described by `UGRID`_. This involved
   adding a data model (:pull:`3968`, :pull:`4014`, :pull:`4027`, :pull:`4036`,
   :pull:`4053`, :pull:`4439`) and API (:pull:`4063`, :pull:`4064`), and
   supporting representation (:pull:`4033`, :pull:`4054`) of data on meshes.
   Most of this new API can be found in :mod:`iris.experimental.ugrid`. The key
   objects introduced are :class:`iris.experimental.ugrid.mesh.Mesh`,
   :class:`iris.experimental.ugrid.mesh.MeshCoord` and
   :obj:`iris.experimental.ugrid.load.PARSE_UGRID_ON_LOAD`.
   A :class:`iris.experimental.ugrid.mesh.Mesh` contains a full description of a UGRID
   type mesh. :class:`~iris.experimental.ugrid.mesh.MeshCoord`\ s are coordinates that
   reference and represent a :class:`~iris.experimental.ugrid.mesh.Mesh` for use
   on a :class:`~iris.cube.Cube`. :class:`~iris.cube.Cube`\ s are also given the
   property :attr:`~iris.cube.Cube.mesh` which returns a
   :class:`~iris.experimental.ugrid.mesh.Mesh` if one is attached to the
   :class:`~iris.cube.Cube` via a :class:`~iris.experimental.ugrid.mesh.MeshCoord`.

#. `@trexfeathers`_ added support for loading unstructured mesh data from netcdf data,
   for files using the `UGRID`_ conventions.
   The context manager :obj:`~iris.experimental.ugrid.load.PARSE_UGRID_ON_LOAD`
   provides a way to load UGRID files so that :class:`~iris.cube.Cube`\ s can be
   returned with a :class:`~iris.experimental.ugrid.mesh.Mesh` attached.
   (:pull:`4058`).

#. `@pp-mo`_ added support to save cubes with meshes to netcdf files, using the
   `UGRID`_ conventions.
   The existing :meth:`iris.save` function now does this, when saving cubes with meshes.
   A routine :meth:`iris.experimental.ugrid.save.save_mesh` allows saving
   :class:`~iris.experimental.ugrid.mesh.Mesh` objects to netcdf *without* any associated data
   (i.e. not attached to cubes).
   (:pull:`4318` and :pull:`4339`).

#. `@trexfeathers`_ added :meth:`iris.experimental.ugrid.mesh.Mesh.from_coords`
   for inferring a :class:`~iris.experimental.ugrid.mesh.Mesh` from an
   appropriate collection of :class:`iris.coords.Coord`\ s.

#. `@larsbarring`_ updated :func:`~iris.util.equalise_attributes` to return a list of dictionaries
   containing the attributes removed from each :class:`~iris.cube.Cube`. (:pull:`4357`)

#. `@trexfeathers`_ enabled streaming of **all** lazy arrays when saving to
   NetCDF files (was previously just :class:`~iris.cube.Cube`
   :attr:`~iris.cube.Cube.data`). This is
   important given the much greater size of
   :class:`~iris.coords.AuxCoord` :attr:`~iris.coords.AuxCoord.points` and
   :class:`~iris.experimental.ugrid.mesh.Connectivity`
   :attr:`~iris.experimental.ugrid.mesh.Connectivity.indices` under the
   `UGRID`_ model. (:pull:`4375`)

#. `@bsherratt`_ added a `threshold` parameter to
   :meth:`~iris.cube.Cube.intersection` (:pull:`4363`)

#. `@wjbenfold`_ added test data to ci benchmarks so that it is accessible to
   benchmark scripts. Also added a regridding benchmark that uses this data
   (:pull:`4402`)


🐛 Bugs Fixed
=============

#. `@rcomer`_ fixed :meth:`~iris.cube.Cube.intersection` for special cases where
   one cell's bounds align with the requested maximum and negative minimum, fixing
   :issue:`4221`. (:pull:`4278`)

#. `@bsherratt`_ fixed further edge cases in
   :meth:`~iris.cube.Cube.intersection`, including :issue:`3698` (:pull:`4363`)

#. `@tinyendian`_ fixed the error message produced by :meth:`~iris.cube.CubeList.concatenate_cube`
   when a cube list contains cubes with different names, which will no longer report
   "Cube names differ: var1 != var1" if var1 appears multiple times in the list
   (:issue:`4342`, :pull:`4345`)

#. `@larsbarring`_ fixed :class:`~iris.coord_systems.GeoCS` to handle spherical ellipsoid
   parameter inverse_flattening = 0 (:issue: `4146`, :pull:`4348`)

#. `@pdearnshaw`_ fixed an error in the call to :class:`cftime.datetime` in
   :mod:`~iris.fileformats.pp_save_rules` that prevented the saving to PP of climate
   means for DJF (:pull:`4391`)

#. `@wjbenfold`_ improved the error message for failure of :meth:`~iris.cube.CubeList.concatenate`
   to indicate that the value of a scalar coordinate may be mismatched, rather than the metadata
   (:issue:`4096`, :pull:`4387`)

#. `@bsherratt`_ fixed a regression to the NAME file loader introduced in 3.0.4,
   as well as some long-standing bugs with vertical coordinates and number
   formats. (:pull:`4411`)

#. `@rcomer`_ fixed :meth:`~iris.cube.Cube.subset` to alway return ``None`` if
   no value match is found.  (:pull:`4417`)

#. `@wjbenfold`_ resolved an issue that previously caused regridding with lazy
   data to take significantly longer than with real data. Relevant benchmark
   shows a time decrease from >10s to 625ms. (:issue:`4280`, :pull:`4400`)

#. `@wjbenfold`_ changed :meth:`iris.util.points_step` to stop it from warning
   when applied to a single point (:issue:`4250`, :pull:`4367`)

#. `@trexfeathers`_ changed :class:`~iris.coords._DimensionalMetadata` and
   :class:`~iris.experimental.ugrid.Connectivity` equality methods to preserve
   array laziness, allowing efficient comparisons even with larger-than-memory
   objects. (:pull:`4439`)


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

#. `@bjlittle`_ introduced the ``cartopy >=0.20`` minimum pin.
   (:pull:`4331`)

#. `@trexfeathers`_ introduced the ``cf-units >=3`` and ``nc-time-axis >=1.3``
   minimum pins. (:pull:`4356`)

#. `@bjlittle`_ introduced the ``numpy >=1.19`` minimum pin, in
   accordance with `NEP-29`_ deprecation policy. (:pull:`4386`)


📚 Documentation
================

#. `@rcomer`_ updated the "Plotting Wind Direction Using Quiver" Gallery
   example. (:pull:`4120`)

#. `@trexfeathers`_ included `Iris GitHub Discussions`_ in
   :ref:`get involved <development_where_to_start>`. (:pull:`4307`)

#. `@wjbenfold`_ improved readability in :ref:`userguide interpolation
   section <interpolation>`. (:pull:`4314`)

#. `@wjbenfold`_ added explanation about the absence of | operator for
   :class:`iris.Constraint` to :ref:`userguide loading section
   <constrained-loading>` and to api reference documentation. (:pull:`4321`)

#. `@trexfeathers`_ added more detail on making `iris-test-data`_ available
   during :ref:`developer_running_tests`. (:pull:`4359`)

#. `@lbdreyer`_ added a section to the release documentation outlining the role
   of the :ref:`release_manager`. (:pull:`4413`)

#. `@trexfeathers`_ encouraged contributors to include type hinting in code
   they are working on - :ref:`code_formatting`. (:pull:`4390`)


💼 Internal
===========

#. `@trexfeathers`_ set the linkcheck to ignore
   http://www.nationalarchives.gov.uk/doc/open-government-licence since this
   always works locally, but never within CI. (:pull:`4307`)

#. `@wjbenfold`_ netCDF integration tests now skip ``TestConstrainedLoad`` if
   test data is missing (:pull:`4319`)

#. `@wjbenfold`_ excluded ``Good First Issue`` labelled issues from being
   marked stale. (:pull:`4317`)

#. `@tkknight`_ added additional make targets for reducing the time of the
   documentation build including ``html-noapi`` and ``html-quick``.
   Useful for development purposes only.  For more information see
   :ref:`contributing.documentation.building` the documentation. (:pull:`4333`)

#. `@rcomer`_ modified the ``animation`` test to prevent it throwing a warning
   that sometimes interferes with unrelated tests. (:pull:`4330`)

#. `@rcomer`_ removed a now redundant workaround in :func:`~iris.plot.contourf`.
   (:pull:`4349`)

#. `@trexfeathers`_ refactored :mod:`iris.experimental.ugrid` into sub-modules.
   (:pull:`4347`).

#. `@bjlittle`_ enabled the `sort-all`_ `pre-commit`_ hook to automatically
   sort ``__all__`` entries into alphabetical order. (:pull:`4353`)

#. `@rcomer`_ modified a NetCDF saver test to prevent it triggering a numpy
   deprecation warning.  (:issue:`4374`, :pull:`4376`)

#. `@akuhnregnier`_ removed addition of period from
   :func:`~iris.analysis.cartography.wrap_lons` and updated affected tests
   using assertArrayAllClose following :issue:`3993`.
   (:pull:`4421`)
   
#. `@rcomer`_ updated some tests to work with Matplotlib v3.5. (:pull:`4428`)

#. `@rcomer`_ applied minor fixes to some regridding tests. (:pull:`4432`)

#. `@lbdreyer`_ corrected the license PyPI classifier. (:pull:`4435`)

#. `@aaronspring <https://github.com/aaronspring>`_ exchanged `dask` with
   `dask-core` in testing environments reducing the number of dependencies
   installed for testing. (:pull:`4434`)

#. `@wjbenfold`_ prevented github action runs in forks (:issue:`4441`,
   :pull:`4444`)

#. `@wjbenfold`_ fixed tests for hybrid formulae that weren't being found by
   nose (:issue:`4431`, :pull:`4450`)

.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:

.. _@aaronspring: https://github.com/aaronspring
.. _@akuhnregnier: https://github.com/akuhnregnier
.. _@bsherratt: https://github.com/bsherratt
.. _@larsbarring: https://github.com/larsbarring
.. _@pdearnshaw: https://github.com/pdearnshaw
.. _@tinyendian: https://github.com/tinyendian

.. comment
    Whatsnew resources in alphabetical order:

.. _GitHub: https://github.com/SciTools/iris/issues/new/choose
.. _NEP-29: https://numpy.org/neps/nep-0029-deprecation_policy.html
.. _UGRID: http://ugrid-conventions.github.io/ugrid-conventions/
.. _sort-all: https://github.com/aio-libs/sort-all
