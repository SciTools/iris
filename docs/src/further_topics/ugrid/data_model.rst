.. include:: ../../common_links.inc

.. _ugrid model:

The Mesh Data Model
*******************

.. important::

        This page is intended to summarise the essentials that Iris users need
        to know about meshes. For exhaustive details on UGRID itself:
        `visit the official UGRID conventions site`__.

Evolution, not revolution
=========================
Mesh support has been designed wherever possible to fit within the existing
Iris model. Meshes concern only the spatial geography of data, and can
optionally be limited to just the horizontal geography (e.g. X and Y). Other
dimensions such as time or ensemble member (and often vertical levels)
retain their familiar structured format.

The UGRID conventions themselves are designed as an addition to the existing CF
conventions, which are at the core of Iris' philosophy.

What's Different?
=================

The mesh format represents data's geography using an **unstructured
mesh**. This has significant pros and cons when compared to a structured grid.

.. contents::
   :local:

The Detail
----------
..
    The diagram images are SVG's, so editable by any graphical software
     (e.g. Inkscape). They were originally made in MS PowerPoint.

    Uses the IBM Colour Blind Palette (see
     http://ibm-design-language.eu-de.mybluemix.net/design/language/resources/color-library
     )

Structured Grids (the old world)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Assigning data to locations using a structured grid is essentially an act of
matching coordinate arrays to each dimension of the data array. The data can
also be represented as an area (instead of a point) by including a bounds array
for each coordinate array.

.. figure:: images/data_structured_grid.svg
   :alt: Diagram of how data is represented on a structured grid

   Data on a structured grid

   :download:`full size image <images/data_structured_grid.svg>`

Unstructured Meshes (the new world)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A mesh is made up of different types of **element**:

.. list-table::
    :widths: 15, 15, 70

    * - 0D
      - ``node``
      - The 'core' of the mesh. A point position in space, constructed from
        2 or 3 coordinates (2D or 3D space).
    * - 1D
      - ``edge``
      - Constructed by connecting 2 nodes.
    * - 2D
      - ``face``
      - Constructed by connecting 3 or more nodes.
    * - 3D
      - ``volume``
      - Constructed by connecting 4 or more nodes (which must each have 3
        coordinates - 3D space).

Every node in the mesh is defined by indexing the 1-dimensional X and Y (and
optionally Z) coordinate arrays (the ``node_coordinates``) - e.g.
``(x[3], y[3])`` gives the position of the fourth node. Note that this means
each node has its own coordinates, independent of every other node.

Any higher dimensional element - an edge/face/volume - is described by a
sequence of the indices of the nodes that make up that element. E.g. a
triangular face made from connecting the first, third and fourth nodes:
``[0, 2, 3]``. These 1D sequences combine into a 2D array enumerating **all**
the elements of that type - edge/face/volume - called a **connectivity**.
E.g. we could make a mesh of 4 nodes, with 2 triangles described using this
``face_node_connectivity``: ``[[0, 2, 3], [3, 2, 1]]`` (note the shared nodes).

.. note:: More on Connectivities:

        * The element type described by a connectivity is known as its
          **location**; ``edge`` in ``edge_node_connectivity``.
        * According to the UGRID conventions, the nodes in a face should be
          listed in "anti-clockwise order from above".
        * Connectivities also exist to connect the higher dimensional elements,
          e.g. ``face_edge_connectivity``. These are optional conveniences to
          speed up certain operations and will not be discussed here.

.. important::

        **Meshes are unstructured**. Elements are enumerated along a single
        **unstructured dimension** represented by either the coordinate or
        connectivity arrays detailed above - and an element's position within
        its respective array has nothing to do with its spatial position.

A data variable associated with a mesh has a **location** of either ``node``,
``edge``, ``face`` or ``volume``. The data is stored in a 1D array with one
datum per element, matched to its element by matching the datum index with the
coordinate or connectivity index. So for an example data array called ``foo``:
``foo[3]`` would be at position ``(x[3], y[3])`` if it were node-located, or at
``faces[3]`` if it were face-located.

.. figure:: images/data_ugrid_mesh.svg
   :alt: Diagram of how data is represented on an unstructured mesh

   Data on an Unstructured Mesh

   :download:`full size image <images/data_ugrid_mesh.svg>`

----

The mesh model also supports edges/faces/volumes having associated 'centre'
coordinates - to allow point data to be assigned to these elements. 'Centre' is
just a convenience term - the points can exist anywhere within their respective
elements.

.. figure:: images/ugrid_element_centres.svg
   :alt: Diagram demonstrating mesh face-centred data.

   Data can be assigned to mesh edge/face/volume 'centres'

Mesh Flexibility
++++++++++++++++
Above we have seen how one could replicate data on a structured grid using
a mesh instead. But the utility of a mesh is the extra flexibility it offers.
Here are the main examples:

* Every node is completely independent - every one can have unique X and
  Y (and Z) coordinate values.

.. figure:: images/ugrid_node_independence.svg
   :alt: Diagram demonstrating the independence of each mesh node
   :align: center

   Every mesh node is completely independent

* Faces and volumes can have variable node counts, i.e. different numbers of
  sides. This is achieved by masking the unused 'slots' in the connectivity array.

.. figure:: images/ugrid_variable_faces.svg
   :alt: Diagram demonstrating mesh faces with variable node counts
   :align: center

   Mesh faces can have different node counts (using masking)

* Data can be assigned to lines (edges) just as easily as points (nodes) or
  areas (faces).

.. figure:: images/ugrid_edge_data.svg
   :alt: Diagram demonstrating data assigned to mesh edges
   :align: center

   Data can be assigned to mesh edges

.. _ugrid implications:

What does this mean?
--------------------
Meshes can represent much more varied spatial arrangements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The highly specific way of recording position (geometry) and shape
(topology) allows meshes to represent essentially **any** spatial arrangement
of data. There are therefore many new applications that aren't possible using a
structured grid, including:

* `The UK Met Office's LFRic cubed-sphere <https://hps.vi4io.org/_media/events/2018/sig-io-uk-adams.pdf>`_
* `Oceanic model outputs <https://doi.org/10.3390/jmse2010194>`_

.. todo:
        a third example!

Mesh 'payload' is much larger than with structured grids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Coordinates are recorded per-node, and connectivities are recorded per-element.
This is opposed to a structured grid, where a single coordinate value is shared
by every data point/area along that line.

For example: representing a cubed-sphere using a mesh leads to coordinates and
connectivities being **~8 times larger than the data itself**, as opposed to a
small fraction of the data size when dividing a sphere using a structured grid.

This further increases the emphasis on lazy loading and processing of data
using packages such as Dask.

.. note::

        The large, 1D data arrays associated with meshes are a very different
        shape to what Iris users and developers are used to. It is suspected
        that optimal performance will need new chunking strategies, but at time
        of writing (``Jan 2022``) experience is still limited.

.. todo:
        Revisit when we have more information.

Spatial operations on mesh data are more complex
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Detail: :doc:`operations`

Indexing a mesh data array cannot be used for:

#. Region selection
#. Neighbour identification

This is because - unlike with a structured data array - relative position in
a mesh's 1-dimensional data arrays has no relation to relative position in
space. We must instead perform specialised operations using the information in
the connectivities present, or by translating the mesh into a format designed
for mesh analysis such as VTK.

Such calculations can still be optimised to avoid them slowing workflows, but
the important take-away here is that **adaptation is needed when working mesh
data**.


How Iris Represents This
========================

..
    Include API links to the various classes

    Include Cube/Mesh printout(s)

.. seealso::

        Remember this is a prose summary. Precise documentation is at:
        :mod:`iris.experimental.ugrid`.

.. note::

        At time of writing (``Jan 2022``), neither 3D meshes nor 3D elements
        (volumes) are supported.

The Basics
----------
The Iris :class:`~iris.cube.Cube` has several new members:

* | :attr:`~iris.cube.Cube.mesh`
  | The :class:`iris.experimental.ugrid.Mesh` that describes the
    :class:`~iris.cube.Cube`\'s horizontal geography.
* | :attr:`~iris.cube.Cube.location`
  | ``node``/``edge``/``face`` - the mesh element type with which this
    :class:`~iris.cube.Cube`\'s :attr:`~iris.cube.Cube.data` is associated.
* | :meth:`~iris.cube.Cube.mesh_dim`
  | The :class:`~iris.cube.Cube`\'s **unstructured dimension** - the one that
    indexes over the horizontal :attr:`~iris.cube.Cube.data` positions.

These members will all be ``None`` for a :class:`~iris.cube.Cube` with no
associated :class:`~iris.experimental.ugrid.Mesh`.

This :class:`~iris.cube.Cube`\'s unstructured dimension has multiple attached
:class:`iris.experimental.ugrid.MeshCoord`\s (one for each axis e.g.
``x``/``y``), which can be used to infer the points and bounds of any index on
the :class:`~iris.cube.Cube`\'s unstructured dimension.

.. todo: Cube printout

The Detail
----------
How UGRID information is stored
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. todo: Mesh printout?

* | :class:`iris.experimental.ugrid.Mesh`
  | Contains all information about the mesh.
  | Includes:

  * 1-3 collections of :class:`iris.coords.AuxCoord`\s:

    * | **Required**: :attr:`~iris.experimental.ugrid.Mesh.node_coords`
      | The nodes that are the basis for the mesh.
    * | Optional: :attr:`~iris.experimental.ugrid.Mesh.edge_coords`,
        :attr:`~iris.experimental.ugrid.Mesh.face_coords`
      | For indicating the 'centres' of the edges/faces.
      | **NOTE:** generating a :class:`~iris.experimental.ugrid.MeshCoord` from
        a :class:`~iris.experimental.ugrid.Mesh` currently (``Jan 2022``)
        requires centre coordinates for the given ``location``; to be rectified
        in future.

  * 1 or more :class:`iris.experimental.ugrid.Connectivity`\s:

    * | **Required for 1D (edge) elements**:
        :attr:`~iris.experimental.ugrid.Mesh.edge_node_connectivity`
      | Define the edges by connecting nodes.
    * | **Required for 2D (face) elements**:
        :attr:`~iris.experimental.ugrid.Mesh.face_node_connectivity`
      | Define the faces by connecting nodes.
    * Optional: any other connectivity type. See
      :attr:`iris.experimental.ugrid.mesh.Connectivity.UGRID_CF_ROLES` for the
      full list of types.

.. todo: MeshCoord printout?

* | :class:`iris.experimental.ugrid.MeshCoord`
  | Described in detail in `MeshCoords`_.
  | Stores the following information:

    * | :attr:`~iris.experimental.ugrid.MeshCoord.mesh`
      | The :class:`~iris.experimental.ugrid.Mesh` associated with this
        :class:`~iris.experimental.ugrid.MeshCoord`. Mirrored by the
        :attr:`~iris.cube.Cube.mesh` attribute of any :class:`~iris.cube.Cube`
        this :class:`~iris.experimental.ugrid.MeshCoord` is attached to (see
        `The Basics`_)

    * | :attr:`~iris.experimental.ugrid.MeshCoord.location`
      | ``node``/``edge``/``face`` - the element detailed by this
        :class:`~iris.experimental.ugrid.MeshCoord`. Mirrored by the
        :attr:`~iris.cube.Cube.location` attribute of any
        :class:`~iris.cube.Cube` this
        :class:`~iris.experimental.ugrid.MeshCoord` is attached to (see
        `The Basics`_).

.. _ugrid MeshCoords:

MeshCoords
~~~~~~~~~~
Links a :class:`~iris.cube.Cube` to a :class:`~iris.experimental.ugrid.Mesh` by
attaching to the :class:`~iris.cube.Cube`\'s unstructured dimension, in the
same way that all :class:`~iris.coords.Coord`\s attach to
:class:`~iris.cube.Cube` dimensions. This allows a single
:class:`~iris.cube.Cube` to have a combination of unstructured and structured
dimensions (e.g. horizontal mesh plus vertical levels and a time series),
using the same logic for every dimension.

:class:`~iris.experimental.ugrid.MeshCoord`\s are instantiated using a given
:class:`~iris.experimental.ugrid.Mesh`, ``location``
("node"/"edge"/"face") and ``axis``. The process interprets the
:class:`~iris.experimental.ugrid.Mesh`\'s
:attr:`~iris.experimental.ugrid.Mesh.node_coords` and if appropriate the
:attr:`~iris.experimental.ugrid.Mesh.edge_node_connectivity`/
:attr:`~iris.experimental.ugrid.Mesh.face_node_connectivity` and
:attr:`~iris.experimental.ugrid.Mesh.edge_coords`/
:attr:`~iris.experimental.ugrid.Mesh.face_coords`
to produce a :class:`~iris.coords.Coord`
:attr:`~iris.coords.Coord.points` and :attr:`~iris.coords.Coord.bounds`
representation of all the :class:`~iris.experimental.ugrid.Mesh`\'s
nodes/edges/faces for the given axis.

The method :meth:`iris.experimental.ugrid.Mesh.to_MeshCoords` is available to
create a :class:`~iris.experimental.ugrid.MeshCoord` for
every axis represented by that :class:`~iris.experimental.ugrid.Mesh`,
given only the ``location`` argument


__ CF-UGRID_