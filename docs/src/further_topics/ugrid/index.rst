.. include:: ../../common_links.inc

.. _ugrid:

UGRID Support
*************

Iris includes specialised handling of data that follows the
`CF-UGRID Conventions`__. UGRID-conformant data can be loaded from a file,
represented in Iris' data model, worked with, and saved as a UGRID-conformant
file. More detail is in this section and in the :mod:`iris.experimental.ugrid`
API documentation.


UGRID is different
  UGRID's mesh-located data is fundamentally different to grid-located data.
  Many of Iris' existing operations need adapting before they can work with
  mesh-located data, and in some cases entirely new concepts are needed.
  **Read the detail here before jumping into your own code.**
Iris' UGRID support is experimental
  This is a rapidly evolving part of the codebase at time of writing
  (``Jan 2021``), as we continually expand the operations that work with mesh
  data. **Be prepared for breaking changes even in minor releases.**
:ref:`Get involved! <development_where_to_start>`
  We know meshes are an exciting new area for much of Earth science, so we hope
  there are a lot of you with new files/ideas/wishlists, and we'd love to hear
  more 🙂.

.. toctree::
   :hidden:
   :maxdepth: 1

   data_model
   partner_packages
   operations

* :doc:`data_model` - learn why the mesh experience is so different.
* :doc:`partner_packages` - meet Iris' partner packages providing powerful mesh operations.
* :doc:`operations` - experience how your workflows will look when written for UGRID data.




__ CF-UGRID_
