.. include:: ../common_links.inc

|iris_version| |build_date| [unreleased]
****************************************

This document explains the changes made to Iris for this release
(:doc:`View all changes <index>`.)


.. dropdown:: :opticon:`report` Release Highlights
   :container: + shadow
   :title: text-primary text-center font-weight-bold
   :body: bg-light
   :animate: fade-in
   :open:

   The highlights for this minor release of Iris include:

   * We've added support for UGRID meshes which can now be loaded and attached
     to a cube.
   * We've dropped support for `Python 3.6`_

   And finally, get in touch with us on `GitHub`_ if you have any issues or
   feature requests for improving Iris. Enjoy!


📢 Announcements
================

#. Congratulations to `@jamesp`_ who recently became an Iris core developer
   after joining the Iris development team at the `Met Office`_. 🎉

#. A special thanks goes to `@akuhnregnier`_, `@gcaria`_, `@jamesp`_, `@MHBalsmeier`_
   and `@Badboy-16`_ all of whom made their first contributions to Iris, which
   were gratefully received and included in this release. Keep up the awesome
   work! 🍻


✨ Features
===========

#. `@bjlittle`_, `@pp-mo`_ and `@trexfeathers`_ added support for unstructured
   meshes, as described by `UGRID`_. This involved adding a data model (:pull:`3968`,
   :pull:`4014`, :pull:`4036`, :pull:`4053`) and API (:pull:`4063`, :pull:`4064`),
   and supporting representation (:pull:`4033`, :pull:`4054`) and loading
   (:pull:`4058`, :pull:`4074`) of data on meshes.
   Most of this new API can be found in :mod:`iris.experimental.ugrid`. The key
   objects introduced are :class:`iris.experimental.ugrid.Mesh`,
   :class:`iris.experimental.ugrid.MeshCoord` and
   :obj:`iris.experimental.ugrid.PARSE_UGRID_ON_LOAD`.
   A :class:`iris.experimental.ugrid.Mesh` contains a full description of a UGRID
   type mesh. :class:`~iris.experimental.ugrid.MeshCoord`\ s are coordinates that
   reference and represent a :class:`~iris.experimental.ugrid.Mesh` for use
   on a :class:`~iris.cube.Cube`. :class:`~iris.cube.Cube`\ s are also given the
   property :attr:`~iris.cube.Cube.mesh` which returns a
   :class:`~iris.experimental.ugrid.Mesh` if one is attached to the
   :class:`~iris.cube.Cube` via a :class:`~iris.experimental.ugrid.MeshCoord`.
   Finally, the context manager :obj:`~iris.experimental.ugrid.PARSE_UGRID_ON_LOAD`
   provides a way to load UGRID files so that :class:`~iris.cube.Cube`\ s can be
   returned with a :class:`~iris.experimental.ugrid.Mesh` attached.

#. `@pelson`_ and `@trexfeathers`_ enhanced :meth:`iris.plot.plot` and
   :meth:`iris.quickplot.plot` to automatically place the cube on the x axis if
   the primary coordinate being plotted against is a vertical coordinate. E.g.
   ``iris.plot.plot(z_cube)`` will produce a z-vs-phenomenon plot, where before
   it would have produced a phenomenon-vs-z plot. (:pull:`3906`)

#. `@bjlittle`_ introduced :func:`iris.common.metadata.hexdigest` to the
   public API. Previously it was a private function introduced in ``v3.0.0``.
   Given any object, :func:`~iris.common.metadata.hexdigest` returns a string
   representation of the 64-bit non-cryptographic hash of the object using the
   extremely fast `xxhash`_ hashing algorithm. (:pull:`4020`)

#. `@rcomer`_ implemented a ``__str__`` method for metadata classes, so
   printing these objects skips metadata elements that are set to None or an
   empty string or dictionary. (:pull:`4040`)

#. `@Badboy-16`_ implemented a ``CubeList.copy()`` method to return a
   ``CubeList`` object instead of a ``list``. (:pull:`4094`)


🐛 Bugs Fixed
=============

#. `@gcaria`_ fixed :class:`~iris.coords.Cell` comparison with
   0-dimensional arrays and 1-dimensional arrays with len=1. (:pull:`4083`)

#. `@gcaria`_ fixed :meth:`~iris.cube.Cube.cell_measure_dims` to also accept the
   string name of a :class:`~iris.coords.CellMeasure`. (:pull:`3931`)

#. `@gcaria`_ fixed :meth:`~iris.cube.Cube.ancillary_variable_dims` to also accept
   the string name of a :class:`~iris.coords.AncillaryVariable`. (:pull:`3931`)

#. `@rcomer`_ modified :func:`~iris.plot.contourf` to skip the special handling for
   antialiasing when data values are too low for it to have an effect.  This caused
   unexpected artifacts in some edge cases, as shown at :issue:`4086`. (:pull:`4150`)

#. `@MHBalsmeier`_ modified :func:`~iris.plot.contourf` to generalize :pull:`4150`
   for the cases where NaN values occur in the plot array (:pull:`4263`)

💣 Incompatible Changes
=======================

#. N/A


🔥 Deprecations
===============

#. N/A


🔗 Dependencies
===============

#. `@bjlittle`_ dropped both `black`_ and `flake8`_ package dependencies
   from our `conda`_ YAML and ``setup.cfg`` PyPI requirements. (:pull:`4181`)


📚 Documentation
================

#. `@rcomer`_ updated the "Seasonal ensemble model plots" and "Global average
   annual temperature maps" Gallery examples. (:pull:`3933` and :pull:`3934`)

#. `@MHBalsmeier`_ described non-conda installation on Debian-based distros.
   (:pull:`3958`)

#. `@bjlittle`_ clarified in the doc-string that :class:`~iris.coords.Coord`
   is now an `abstract base class`_ since Iris ``3.0.0``, and it is **not**
   possible to create an instance of it. (:pull:`3971`)

#. `@bjlittle`_ added automated Iris version discovery for the ``latest.rst``
   in the ``whatsnew`` documentation. (:pull:`3981`)

#. `@tkknight`_ stated the Python version used to build the documentation
   on :ref:`installing_iris` and to the footer of all pages.  Also added the
   copyright years to the footer. (:pull:`3989`)

#. `@bjlittle`_ updated the ``intersphinx_mapping`` and fixed documentation
   to use ``stable`` URLs for `matplotlib`_. (:pull:`4003`)

#. `@bjlittle`_ added the |PyPI|_ badge to the `README.md`_. (:pull:`4004`)

#. `@tkknight`_ added a banner at the top of every page of the unreleased
   development documentation if being viewed on `Read the Docs`_.
   (:pull:`3999`)

#. `@bjlittle`_ added post-release instructions on how to :ref:`update_pypi`
   with `scitools-iris`_. (:pull:`4038`)

#. `@bjlittle`_ added the |pre-commit.ci|_ badge to the `README.md`_.
   See :ref:`pre_commit_ci` for further details. (:pull:`4061`)

#. `@rcomer`_ tweaked docstring layouts in the :mod:`iris.plot` module, so
   they render better in the published documentation.  See :issue:`4085`.
   (:pull:`4100`)

#. `@tkknight`_ documented the ``--force`` command line option when creating
   a conda development environment. See :ref:`installing_from_source`.
   (:pull:`4240`)

#. `@MHBalsmeier`_ updated and simplified non-conda installation on Debian-based distros.
   (:pull:`4260`)


💼 Internal
===========

#. `@rcomer`_ removed an old unused test file. (:pull:`3913`)

#. `@tkknight`_ moved the ``docs/iris`` directory to be in the parent
   directory ``docs``.  (:pull:`3975`)

#. `@jamesp`_ updated a test for `numpy`_ ``1.20.0``. (:pull:`3977`)

#. `@bjlittle`_ and `@jamesp`_ extended the `cirrus-ci`_ testing and `nox`_
   testing automation to support `Python 3.8`_. (:pull:`3976`)

#. `@bjlittle`_ rationalised the ``noxfile.py``, and added the ability for
   each ``nox`` session to list its ``conda`` environment packages and
   environment info. (:pull:`3990`)

#. `@bjlittle`_ enabled `cirrus-ci`_ compute credits for non-draft pull-requests
   from collaborators targeting the Iris ``main`` branch. (:pull:`4007`)

#. `@akuhnregnier`_ replaced `deprecated numpy 1.20 aliases for builtin types`_.
   (:pull:`3997`)

#. `@bjlittle`_ added conditional task execution to `.cirrus.yml`_ to allow
   developers to easily disable `cirrus-ci`_ tasks. See
   :ref:`skipping Cirrus-CI tasks`. (:pull:`4019`)

#. `@bjlittle`_ and `@jamesp`_ addressed a regression in behaviour when using
   `conda`_ 4.10.0 within `cirrus-ci`_. (:pull:`4084`)

#. `@bjlittle`_ updated the perceptual imagehash graphical test support for
   `matplotlib`_ 3.4.1. (:pull:`4087`)

#. `@jamesp`_ switched `cirrus-ci`_ testing and `nox`_
   testing to use `conda-lock`_ files for static test environments. (:pull:`4108`)

#. `@bjlittle`_ updated the ``bug-report`` and ``feature-request`` GitHub issue
   templates to remove an external URL reference that caused un-posted user issue
   content to be lost in the browser when followed. (:pull:`4147`)

#. `@bjlittle`_ dropped `Python 3.6`_ support, and automated the discovery of
   supported Python versions tested by `cirrus-ci`_ for documentation.
   (:pull:`4163`)

#. `@bjlittle`_ refactored ``setup.py`` into ``setup.cfg``. (:pull:`4168`)

#. `@bjlittle`_ consolidated the ``.flake8`` configuration into ``setup.cfg``.
   (:pull:`4200`)

#. `@bjlittle`_ added support for automated ``import`` linting with `isort`_.
   (:pull:`4174`)

#. `@bjlittle`_ renamed ``iris/master`` branch to ``iris/main`` and migrated
   references of ``master`` to ``main`` within codebase. (:pull:`4202`)

#. `@bjlittle`_ added the `blacken-docs`_ ``pre-commit`` hook to automate
   ``black`` linting of documentation code blocks. (:pull:`4205`)

#. `@bjlittle`_ consolidated `nox`_ ``black``, ``flake8`` and ``isort`` sessions
   into one ``lint`` session using ``pre-commit``. (:pull:`4181`)

#. `@bjlittle`_ streamlined the `cirrus-ci`_ testing by removing the ``minimal``
   tests, which are a subset of the ``full`` tests. (:pull:`4218`)

#. `@bjlittle`_ consolidated the `cirrus-ci`_ documentation ``doctest`` and
   ``gallery`` tasks into a single task and associated `nox`_ session.
   (:pull:`4219`)

#. `@jamesp`_ and `@trexfeathers`_ implmented a benchmarking CI check
   using `asv`_. (:pull:`4253`)

.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:

.. _@akuhnregnier: https://github.com/akuhnregnier
.. _@Badboy-16: https://github.com/Badboy-16
.. _@gcaria: https://github.com/gcaria
.. _@MHBalsmeier: https://github.com/MHBalsmeier


.. comment
    Whatsnew resources in alphabetical order:

.. _abstract base class: https://docs.python.org/3/library/abc.html
.. _blacken-docs: https://github.com/asottile/blacken-docs
.. _deprecated numpy 1.20 aliases for builtin types: https://numpy.org/doc/1.20/release/1.20.0-notes.html#using-the-aliases-of-builtin-types-like-np-int-is-deprecated
.. _GitHub: https://github.com/SciTools/iris/issues/new/choose
.. _Met Office: https://www.metoffice.gov.uk/
.. _numpy: https://numpy.org/doc/stable/release/1.20.0-notes.html
.. |pre-commit.ci| image:: https://results.pre-commit.ci/badge/github/SciTools/iris/main.svg
.. _pre-commit.ci: https://results.pre-commit.ci/latest/github/SciTools/iris/main
.. |PyPI| image:: https://img.shields.io/pypi/v/scitools-iris?color=orange&label=pypi%7Cscitools-iris
.. _PyPI: https://pypi.org/project/scitools-iris/
.. _Python 3.8: https://www.python.org/downloads/release/python-380/
.. _Python 3.6: https://www.python.org/downloads/release/python-360/
.. _README.md: https://github.com/SciTools/iris#-----
.. _xxhash: http://cyan4973.github.io/xxHash/
.. _conda-lock: https://github.com/conda-incubator/conda-lock
.. _asv: https://asv.readthedocs.io/en/stable/
.. _UGRID: http://ugrid-conventions.github.io/ugrid-conventions/
