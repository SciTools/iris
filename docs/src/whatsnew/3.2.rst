.. include:: ../common_links.inc

v3.2 (15 Feb 2022)
******************

This document explains the changes made to Iris for this release
(:doc:`View all changes <index>`.)


.. dropdown:: :opticon:`report` v3.2.0 Release Highlights
   :container: + shadow
   :title: text-primary text-center font-weight-bold
   :body: bg-light
   :animate: fade-in
   :open:

   The highlights for this minor release of Iris include:

   * We've added experimental support for
     :ref:`Meshes <ugrid>`, which can now be loaded and
     attached to a cube. Mesh support is based on the `CF-UGRID`_ model.
   * We've also dropped support for ``Python 3.7``.

   And finally, get in touch with us on :issue:`GitHub<new/choose>` if you have
   any issues or feature requests for improving Iris. Enjoy!


📢 Announcements
================

#. Welcome to `@wjbenfold`_, `@tinyendian`_, `@larsbarring`_, `@bsherratt`_ and
   `@aaronspring`_ who made their first contributions to Iris. The first of
   many we hope!
#. Congratulations to `@wjbenfold`_ who has become a core developer for Iris! 🎉


✨ Features
===========

#. `@bjlittle`_, `@pp-mo`_, `@trexfeathers`_ and `@stephenworsley`_ added
   support for :ref:`unstructured meshes <ugrid>`. This involved
   adding a data model (:pull:`3968`, :pull:`4014`, :pull:`4027`, :pull:`4036`,
   :pull:`4053`, :pull:`4439`) and API (:pull:`4063`, :pull:`4064`), and
   supporting representation (:pull:`4033`, :pull:`4054`) of data on meshes.
   Most of this new API can be found in :mod:`iris.experimental.ugrid`. The key
   objects introduced are :class:`iris.experimental.ugrid.mesh.Mesh`,
   :class:`iris.experimental.ugrid.mesh.MeshCoord` and
   :obj:`iris.experimental.ugrid.load.PARSE_UGRID_ON_LOAD`.
   A :class:`~iris.experimental.ugrid.mesh.Mesh` contains a full description of a UGRID
   type mesh. :class:`~iris.experimental.ugrid.mesh.MeshCoord`\ s are coordinates that
   reference and represent a :class:`~iris.experimental.ugrid.mesh.Mesh` for use
   on a :class:`~iris.cube.Cube`. :class:`~iris.cube.Cube`\ s are also given the
   property :attr:`~iris.cube.Cube.mesh` which returns a
   :class:`~iris.experimental.ugrid.mesh.Mesh` if one is attached to the
   :class:`~iris.cube.Cube` via a :class:`~iris.experimental.ugrid.mesh.MeshCoord`.

#. `@trexfeathers`_ added support for loading unstructured mesh data from netcdf data,
   for files using the `CF-UGRID`_ conventions.
   The context manager :obj:`~iris.experimental.ugrid.load.PARSE_UGRID_ON_LOAD`
   provides a way to load UGRID files so that :class:`~iris.cube.Cube`\ s can be
   returned with a :class:`~iris.experimental.ugrid.mesh.Mesh` attached.
   (:pull:`4058`).

#. `@pp-mo`_ added support to save cubes with :ref:`meshes <ugrid>` to netcdf
   files, using the `CF-UGRID`_ conventions.
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
   :ref:`mesh model <ugrid model>`. (:pull:`4375`)

#. `@bsherratt`_ added a ``threshold`` parameter to
   :meth:`~iris.cube.Cube.intersection` (:pull:`4363`)

#. `@wjbenfold`_ added test data to ci benchmarks so that it is accessible to
   benchmark scripts. Also added a regridding benchmark that uses this data
   (:pull:`4402`)

#. `@pp-mo`_ updated to the latest CF Standard Names Table ``v78`` (21 Sept 2021).
   (:issue:`4479`, :pull:`4483`)

#. `@SimonPeatman`_ added support for filenames in the form of a :class:`~pathlib.PurePath`
   in :func:`~iris.load`, :func:`~iris.load_cube`, :func:`~iris.load_cubes`,
   :func:`~iris.load_raw` and :func:`~iris.save` (:issue:`3411`, :pull:`3917`).
   Support for :class:`~pathlib.PurePath` is yet to be implemented across the rest
   of Iris (:issue:`4523`).

#. `@pp-mo`_ removed broken tooling for deriving Iris metadata translations
   from `Metarelate`_.  From now we intend to manage phenonemon translation
   in Iris itself.  (:pull:`4484`)

#. `@pp-mo`_ improved printout of various cube data component objects :
   :class:`~iris.coords.Coord`, :class:`~iris.coords.CellMeasure`,
   :class:`~iris.coords.AncillaryVariable`,
   :class:`~iris.experimental.ugrid.mesh.MeshCoord` and
   :class:`~iris.experimental.ugrid.mesh.Mesh`.
   These now all provide a more controllable ``summary()`` method, and
   more convenient and readable ``str()`` and ``repr()`` output in the style of
   the :class:`iris.cube.Cube`.
   They also no longer realise lazy data.  (:pull:`4499`).


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
   parameter inverse_flattening = 0 (:issue:`4146`, :pull:`4348`)

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

#. `@wjbenfold`_ changed :meth:`iris.util.points_step` to stop it from warning
   when applied to a single point (:issue:`4250`, :pull:`4367`)

#. `@trexfeathers`_ changed :class:`~iris.coords._DimensionalMetadata` and
   :class:`~iris.experimental.ugrid.Connectivity` equality methods to preserve
   array laziness, allowing efficient comparisons even with larger-than-memory
   objects. (:pull:`4439`)

#. `@rcomer`_ modified :meth:`~iris.cube.Cube.aggregated_by` to calculate new
   coordinate bounds using minimum and maximum for unordered coordinates,
   fixing :issue:`1528`. (:pull:`4315`)

#. `@wjbenfold`_ changed how a delayed unit conversion is performed on a cube
   so that a cube with lazy data awaiting a unit conversion can be pickled.
   (:issue:`4354`, :pull:`4377`)

#. `@pp-mo`_ fixed a bug in netcdf loading, whereby *any* rotated latlon coordinate
   was mistakenly interpreted as a latitude, usually resulting in two 'latitude's
   instead of one latitude and one longitude.
   (:issue:`4460`, :pull:`4470`)

#. `@wjbenfold`_ stopped :meth:`iris.coord_systems.GeogCS.as_cartopy_projection`
   from assuming the globe to be the Earth (:issue:`4408`, :pull:`4497`)

#. `@rcomer`_ corrected the ``long_name`` mapping from UM stash code ``m01s09i215``
   to indicate cloud fraction greater than 7.9 oktas, rather than 7.5 
   (:issue:`3305`, :pull:`4535`)

#. `@lbdreyer`_ fixed a bug in :class:`iris.io.load_http` which was missing an import
   (:pull:`4580`)


💣 Incompatible Changes
=======================

#. N/A


🚀 Performance Enhancements
===========================

#. `@wjbenfold`_ resolved an issue that previously caused regridding with lazy
   data to take significantly longer than with real data. Benchmark
   :class:`benchmarks.HorizontalChunkedRegridding` shows a time decrease
   from >10s to 625ms. (:issue:`4280`, :pull:`4400`)

#. `@bjlittle`_ included an optimisation to :class:`~iris.cube.Cube.coord_dims`
   to avoid unnecessary processing whenever a coordinate instance that already
   exists within the cube is provided. (:pull:`4549`)


🔥 Deprecations
===============

#. `@wjbenfold`_ removed :mod:`iris.experimental.equalise_cubes`. In ``v3.0``
   the experimental ``equalise_attributes`` functionality was moved to the
   :mod:`iris.util.equalise_attributes` function. Since then, calling the
   :func:`iris.experimental.equalise_cubes.equalise_attributes` function raised
   an exception. (:issue:`3528`, :pull:`4496`)

#. `@wjbenfold`_ deprecated :func:`iris.util.approx_equal` in preference for
   :func:`math.isclose`. The :func:`~iris.util.approx_equal` function will be
   removed in a future release of Iris. (:pull:`4514`)

#. `@wjbenfold`_ deprecated :mod:`iris.experimental.raster` as it is not
   believed to still be in use. The deprecation warnings invite users to contact
   the Iris Developers if this isn't the case. (:pull:`4525`)

#. `@wjbenfold`_ deprecated :mod:`iris.fileformats.abf` and
   :mod:`iris.fileformats.dot` as they are not believed to still be in use. The
   deprecation warnings invite users to contact the Iris Developers if this
   isn't the case. (:pull:`4515`)

#. `@wjbenfold`_ removed the :func:`iris.util.as_compatible_shape` function,
   which was deprecated in ``v3.0``. Instead use
   :class:`iris.common.resolve.Resolve`. For example, rather than calling
   ``as_compatible_shape(src_cube, target_cube)`` replace with
   ``Resolve(src_cube, target_cube)(target_cube.core_data())``. (:pull:`4513`)

#. `@wjbenfold`_ deprecated :func:`iris.analysis.maths.intersection_of_cubes` in
   preference for :meth:`iris.cube.CubeList.extract_overlapping`. The
   :func:`~iris.analysis.maths.intersection_of_cubes` function will be removed in
   a future release of Iris. (:pull:`4541`)

#. `@pp-mo`_ deprecated :mod:`iris.experimental.regrid_conservative`.  This is
   now replaced by `iris-emsf-regrid`_.  (:pull:`4551`)

#. `@pp-mo`_ deprecated everything in :mod:`iris.experimental.regrid`.
   Most features have a preferred exact alternative, as suggested, *except*
   :class:`iris.experimental.regrid.ProjectedUnstructuredLinear` : that has no
   identical equivalent, but :class:`iris.analysis.UnstructuredNearest` is
   suggested as being quite close (though possibly slower).  (:pull:`4548`)


🔗 Dependencies
===============

#. `@bjlittle`_ introduced the ``cartopy >=0.20`` minimum pin.
   (:pull:`4331`)

#. `@trexfeathers`_ introduced the ``cf-units >=3`` and ``nc-time-axis >=1.3``
   minimum pins. (:pull:`4356`)

#. `@bjlittle`_ introduced the ``numpy >=1.19`` minimum pin, in
   accordance with `NEP-29`_ deprecation policy. (:pull:`4386`)

#. `@bjlittle`_ dropped support for ``Python 3.7``, as per the `NEP-29`_
   backwards compatibility and deprecation policy schedule. (:pull:`4481`)


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

#. `@wjbenfold`_ updated Cartopy documentation links to point to the renamed
   :class:`cartopy.mpl.geoaxes.GeoAxes`. (:pull:`4464`)

#. `@wjbenfold`_ clarified behaviour of :func:`iris.load` in :ref:`userguide
   loading section <loading_iris_cubes>`. (:pull:`4462`)

#. `@bjlittle`_ migrated readthedocs to use mambaforge for `faster documentation building`_.
   (:pull:`4476`)

#. `@wjbenfold`_ contributed `@alastair-gemmell`_'s :ref:`step-by-step guide to
   contributing to the docs <contributing.documentation_easy>` to the docs.
   (:pull:`4461`)

#. `@pp-mo`_ improved and corrected docstrings of
   :class:`iris.analysis.PointInCell`, making it clear what is the actual
   calculation performed.  (:pull:`4548`)

#. `@pp-mo`_ removed reference in docstring of
   :class:`iris.analysis.UnstructuredNearest` to the obsolete (deprecated)
   :class:`iris.experimental.regrid.ProjectedUnstructuredNearest`.
   (:pull:`4548`)


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
   using ``assertArrayAllClose`` following :issue:`3993`.
   (:pull:`4421`)

#. `@rcomer`_ updated some tests to work with Matplotlib v3.5. (:pull:`4428`)

#. `@rcomer`_ applied minor fixes to some regridding tests. (:pull:`4432`)

#. `@lbdreyer`_ corrected the license PyPI classifier. (:pull:`4435`)

#. `@aaronspring`_ exchanged ``dask`` with
   ``dask-core`` in testing environments reducing the number of dependencies
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
.. _@SimonPeatman: https://github.com/SimonPeatman
.. _@tinyendian: https://github.com/tinyendian

.. comment
    Whatsnew resources in alphabetical order:

.. _NEP-29: https://numpy.org/neps/nep-0029-deprecation_policy.html
.. _Metarelate: http://www.metarelate.net/
.. _UGRID: http://ugrid-conventions.github.io/ugrid-conventions/
.. _iris-emsf-regrid: https://github.com/SciTools-incubator/iris-esmf-regrid
.. _faster documentation building: https://docs.readthedocs.io/en/stable/guides/conda.html#making-builds-faster-with-mamba
.. _sort-all: https://github.com/aio-libs/sort-all
