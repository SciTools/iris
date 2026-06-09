.. include:: /common_links.inc

.. comment:
   now that User Manual is the official top-level, and the User Guide is a
   sub-section, the original labels have been relocated here.

.. _user_guide_index:
.. _user_guide_introduction:
.. _user_manual_index:

User Manual
===========

.. hint::

   If you are new to Iris: check out :ref:`getting_started_index` first.

Welcome to the Iris User Manual!

This is designed as a searchable index of **all** our user documentation. Try
the Topic and `Diataxis`_ filters below to find the information you need today.
Alternatively, you can use the sidebar to navigate by section.

.. tip::

   - :doc:`/user_manual/index`: a searchable index of **all** user
     documentation.
   - :doc:`User Guide </user_manual/section_indexes/userguide>`: a linear
     narrative introduction to Iris' data model and functionality.

.. comment:
   The tree structure for user_manual is specified here. As mentioned in the
   text, we prefer readers to use the tabbed sections below, so the toctree is
   hidden - not rendered in the text, only in the sidebar. This toctree is
   expected to be section_indexes/* pages; with each of those pages
   providing the remaining sub-structure.


.. toctree::
   :maxdepth: 1
   :hidden:

   section_indexes/get_started
   section_indexes/userguide
   /generated/gallery/index
   Iris API </generated/api/iris>
   section_indexes/dask_best_practices
   section_indexes/mesh_support
   section_indexes/metadata_arithmetic
   section_indexes/community
   section_indexes/general

.. _topic_all:

All
---

.. diataxis-page-list:: topic_all

By Topic
--------

.. _topic_data_model:

topic: ``data_model``
^^^^^^^^^^^^^^^^^^^^^

Pages about the :class:`~iris.cube.Cube` class and its associated components
such as :class:`~iris.coords.Coord` and :class:`~iris.mesh.MeshXY`.

.. diataxis-page-list:: topic_data_model


.. _topic_slice_combine:

topic: ``slice_combine``
^^^^^^^^^^^^^^^^^^^^^^^^

Pages about subsetting and combining :class:`~iris.cube.Cube` and
:class:`~iris.cube.CubeList` data. Examples include slicing, indexing, merging,
concatenating.

.. diataxis-page-list:: topic_slice_combine


.. _topic_load_save:

topic: ``load_save``
^^^^^^^^^^^^^^^^^^^^

Pages about reading from files into the data model, and writing from the data
model to files.

.. diataxis-page-list:: topic_load_save


.. _topic_lazy_data:

topic: ``lazy_data``
^^^^^^^^^^^^^^^^^^^^

Pages about Iris' implementation of parallel and out-of-core data handling, via
Dask. See :term:`Lazy Data`.

.. diataxis-page-list:: topic_lazy_data


.. _topic_plotting:

topic: ``plotting``
^^^^^^^^^^^^^^^^^^^

Pages about Iris' use of :term:`Cartopy` or :ref:`ugrid geovista` to plot
:class:`~iris.cube.Cube` data.

.. diataxis-page-list:: topic_plotting


.. _topic_maths_stats:

topic: ``maths_stats``
^^^^^^^^^^^^^^^^^^^^^^

Pages about statistical and mathematical operations on :class:`~iris.cube.Cube`
data, e.g. computing means, differences, etc.

.. diataxis-page-list:: topic_maths_stats


.. _topic_regrid:

topic: ``regrid``
^^^^^^^^^^^^^^^^^

Pages about regridding (2D to 2D) and interpolation (N-D to 1D) of data from one
set of coordinates to another. Commonly used to move between different XY grids.

.. diataxis-page-list:: topic_regrid


.. _topic_customisation:

topic: ``customisation``
^^^^^^^^^^^^^^^^^^^^^^^^

Pages about configurable Iris behaviour.

.. diataxis-page-list:: topic_customisation


.. _topic_troubleshooting:

topic: ``troubleshooting``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Pages about problems/exceptions you may encounter when using Iris, and how to
best handle them.

.. diataxis-page-list:: topic_troubleshooting


.. _topic_experimental:

topic: ``experimental``
^^^^^^^^^^^^^^^^^^^^^^^

Pages about API that is still subject to change.

.. diataxis-page-list:: topic_experimental


.. _topic_interoperability:

topic: ``interoperability``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pages about using Iris alongside other libraries and tools.

.. diataxis-page-list:: topic_interoperability


.. _topic_mesh:

topic: ``mesh``
^^^^^^^^^^^^^^^

Pages about Iris' support for unstructured mesh data.

.. diataxis-page-list:: topic_mesh


.. _topic_about:

topic: ``about``
^^^^^^^^^^^^^^^^

Pages about the non-code aspects of Iris: philosophy, installation, etc.

.. diataxis-page-list:: topic_about
