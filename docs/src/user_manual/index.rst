.. comment:
   now that User Manual is the official top-level, and the User Guide is a
   sub-section, the original labels have been relocated here.

.. _user_guide_index:
.. _user_guide_introduction:
.. _user_manual_index:

User Manual
===========

Welcome to the Iris User Manual!

We encourage exploring our User Manual pages using the tabbed sections below,
which combine the `Diataxis`_ framework and topic-based filters to find content
best suited to your purpose today. Alternatively, you can use the sidebar to
navigate by section.

.. todo:
   Should the sections also be offered as another Diataxis tab? This would allow
   topic-based filtering, and allow all pages to be found through the same
   route.

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

Pages about Iris' use of :term:`Cartopy` to plot :class:`~iris.cube.Cube` data.

.. diataxis-page-list:: topic_plotting

.. _topic_statistics:

topic: ``statistics``
^^^^^^^^^^^^^^^^^^^^^

.. todo: not sure about this topic - very unclear scope.

Pages about statistical and mathematical operations on :class:`~iris.cube.Cube`
data, e.g. computing means, differences, etc.

.. diataxis-page-list:: topic_statistics

.. _topic_regrid:

topic: ``regrid``
^^^^^^^^^^^^^^^^^

Pages about interpolating (1D) and regridding (2D) data from one set of
coordinates to another. Commonly used to move between different XY grids.

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


.. _Diataxis: https://diataxis.fr/
