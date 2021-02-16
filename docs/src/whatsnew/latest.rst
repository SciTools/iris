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


✨ Features
===========

#. `@pelson`_ and `@trexfeathers`_ enhanced :meth:`iris.plot.plot` and
   :meth:`iris.quickplot.plot` to automatically place the cube on the x axis if
   the primary coordinate being plotted against is a vertical coordinate. E.g.
   ``iris.plot.plot(z_cube)`` will produce a z-vs-phenomenon plot, where before
   it would have produced a phenomenon-vs-z plot. (:pull:`3906`)


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

#. `@tkknight`_ added a banner at the top of every page if the unreleased
   development documentatiion is being viewed. (:pull:`3999`)


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


.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:

.. _@gcaria: https://github.com/gcaria
.. _@MHBalsmeier: https://github.com/MHBalsmeier


.. comment
    Whatsnew resources in alphabetical order:

.. _abstract base class: https://docs.python.org/3/library/abc.html
.. _GitHub: https://github.com/SciTools/iris/issues/new/choose
.. _Met Office: https://www.metoffice.gov.uk/
.. _numpy: https://numpy.org/doc/stable/release/1.20.0-notes.html
.. |PyPI| image:: https://img.shields.io/pypi/v/scitools-iris?color=orange&label=pypi%7Cscitools-iris
.. _PyPI: https://pypi.org/project/scitools-iris/
.. _Python 3.8: https://www.python.org/downloads/release/python-380/
.. _README.md: https://github.com/SciTools/iris#-----
