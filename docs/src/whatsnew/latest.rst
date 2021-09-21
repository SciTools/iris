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

#. Welcome to `@wjbenfold`_ who has made their first contribution to Iris,
   the first of many we hope!


✨ Features
===========

#. `@bjlittle`_, `@pp-mo`_ and `@trexfeathers`_ added support for unstructured
   meshes, as described by `UGRID`_. This involved adding a data model (:pull:`3968`,
   :pull:`4014`, :pull:`4027`, :pull:`4036`, :pull:`4053`) and API (:pull:`4063`,
   :pull:`4064`), and supporting representation (:pull:`4033`, :pull:`4054`) and
   loading (:pull:`4058`) of data on meshes.
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


🐛 Bugs Fixed
=============

#. N/A


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

#. `@rcomer`_ updated the "Plotting Wind Direction Using Quiver" Gallery
   example. (:pull:`4120`)

#. `@trexfeathers`_ included `Iris GitHub Discussions`_ in
   :ref:`get involved <development_where_to_start>`. (:pull:`4307`)

#. `@wjbenfold`_ improved readability in :ref:`userguide interpolation
   section <interpolation>`. (:pull:`4314`)

#. `@wjbenfold`_ added explanation about the absence of | operator for
   :class:`iris.Constraint` to :ref:`userguide loading section
   <constrained-loading>` and to api reference documentation. (:pull:`4321`)


💼 Internal
===========

#. `@trexfeathers`_ set the linkcheck to ignore
   http://www.nationalarchives.gov.uk/doc/open-government-licence since this
   always works locally, but never within CI. (:pull:`4307`)

#. `@wjbenfold`_ netCDF integration tests now skip ``TestConstrainedLoad`` if
   test data is missing (:pull:`4319`)


#. `@wjbenfold`_ excluded "Good First Issue" labelled issues from being
   marked stale. (:pull:`4317`)

#. `@tkknight`_ added additional make targets for reducing the time of the
   documentation build including ``html-noapi`` and ``html-quick``.
   Useful for development purposes only.  For more information see
   :ref:`contributing.documentation.building` the documentation. (:pull:`4333`)

.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:




.. comment
    Whatsnew resources in alphabetical order:

.. _GitHub: https://github.com/SciTools/iris/issues/new/choose
.. _UGRID: http://ugrid-conventions.github.io/ugrid-conventions/
