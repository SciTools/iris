.. include:: ../../common_links.inc

.. _ugrid:

Mesh Support
************

Iris includes specialised handling of mesh-located data (as opposed to
grid-located data). Iris and its :ref:`partner packages <ugrid partners>` are
designed to make working with mesh-located data as simple as possible, with new
capabilities being added all the time. More detail is in this section and in
the :mod:`iris.experimental.ugrid` API documentation.

This mesh support is based on the `CF-UGRID Conventions`__; UGRID-conformant
meshes + data can be loaded from a file into Iris' data model, and meshes +
data represented in Iris' data model can be saved as a UGRID-conformant file.

----

Meshes are different
  Mesh-located data is fundamentally different to grid-located data.
  Many of Iris' existing operations need adapting before they can work with
  mesh-located data, and in some cases entirely new concepts are needed.
  **Read the detail in these pages before jumping into your own code.**
Iris' mesh support is experimental
  This is a rapidly evolving part of the codebase at time of writing
  (``Jan 2022``), as we continually expand the operations that work with mesh
  data. **Be prepared for breaking changes even in minor releases.**
:ref:`Get involved! <development_where_to_start>`
  We know meshes are an exciting new area for much of Earth science, so we hope
  there are a lot of you with new files/ideas/wishlists, and we'd love to hear
  more 🙂.

----

Read on to find out more...

* :doc:`data_model` - learn why the mesh experience is so different.
* :doc:`partner_packages` - meet some optional dependencies that provide powerful mesh operations.
* :doc:`operations` - experience how your workflows will look when written for mesh data.

..
    Need an actual TOC to get Sphinx working properly, but have hidden it in
     favour of the custom bullets above.

.. toctree::
   :hidden:
   :maxdepth: 1

   data_model
   partner_packages
   operations

__ CF-UGRID_
