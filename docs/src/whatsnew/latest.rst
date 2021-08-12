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

   The highlights for this major/minor release of Iris include:

   * N/A

   And finally, get in touch with us on `GitHub`_ if you have any issues or
   feature requests for improving Iris. Enjoy!


📢 Announcements
================

#. Congratulations to `@jamesp`_ who recently became an Iris core developer
   after joining the Iris development team at the `Met Office`_. 🎉

#. A special thanks goes to `@akuhnregnier`_, `@gcaria`_, `@jamesp`_ and
   `@MHBalsmeier`_ all of whom made their first contributions to Iris, which
   were gratefully received and included in this release. Keep up the awesome
   work! 🍻


✨ Features
===========

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


🐛 Bugs Fixed
=============

#. `@gcaria`_ fixed :meth:`~iris.cube.Cube.cell_measure_dims` to also accept the
   string name of a :class:`~iris.coords.CellMeasure`. (:pull:`3931`)

#. `@gcaria`_ fixed :meth:`~iris.cube.Cube.ancillary_variable_dims` to also accept
   the string name of a :class:`~iris.coords.AncillaryVariable`. (:pull:`3931`)


💣 Incompatible Changes
=======================

#. N/A


🔥 Deprecations
===============

#. N/A


🔗 Dependencies
===============

#. N/A


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
   from collaborators targeting the Iris ``master`` branch. (:pull:`4007`)

#. `@akuhnregnier`_ replaced `deprecated numpy 1.20 aliases for builtin types`_.
   (:pull:`3997`)

#. `@bjlittle`_ added conditional task execution to `.cirrus.yml`_ to allow
   developers to easily disable `cirrus-ci`_ tasks. See
   :ref:`skipping Cirrus-CI tasks`. (:pull:`4019`)


.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:

.. _@akuhnregnier: https://github.com/akuhnregnier
.. _@gcaria: https://github.com/gcaria
.. _@MHBalsmeier: https://github.com/MHBalsmeier


.. comment
    Whatsnew resources in alphabetical order:

.. _abstract base class: https://docs.python.org/3/library/abc.html
.. _deprecated numpy 1.20 aliases for builtin types: https://numpy.org/doc/1.20/release/1.20.0-notes.html#using-the-aliases-of-builtin-types-like-np-int-is-deprecated
.. _GitHub: https://github.com/SciTools/iris/issues/new/choose
.. _Met Office: https://www.metoffice.gov.uk/
.. _numpy: https://numpy.org/doc/stable/release/1.20.0-notes.html
.. |pre-commit.ci| image:: https://results.pre-commit.ci/badge/github/SciTools/iris/master.svg
.. _pre-commit.ci: https://results.pre-commit.ci/latest/github/SciTools/iris/master
.. |PyPI| image:: https://img.shields.io/pypi/v/scitools-iris?color=orange&label=pypi%7Cscitools-iris
.. _PyPI: https://pypi.org/project/scitools-iris/
.. _Python 3.8: https://www.python.org/downloads/release/python-380/
.. _README.md: https://github.com/SciTools/iris#-----
.. _xxhash: http://cyan4973.github.io/xxHash/
