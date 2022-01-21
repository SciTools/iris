.. include:: ../../common_links.inc

.. _ugrid model:

..
    The conventions page is [necessarily] not for a layperson. This is an
     opportunity to explain UGRID at an Iris user's level - doesn't need to be
     exhaustive, just get across what is needed for the user to understand why
     their experience will be different to 'normal'.


The UGRID Data Model
********************

.. important::

        This page is intended to summarise what Iris users need to know about
        the UGRID model. For full detail
        `visit the official UGRID conventions site`__.

Something to note straight away is that UGRID is designed as an addition to the
existing CF model. It concerns only spatial location of data, and even there
it can be limited just the horizontal location - X and Y - which is so far the
most popular use for UGRID. Other dimensions such as time and experimental run
numbers remain formatted as they always have been.

What's Different?
=================

The UGRID format represents data's geographic location using an **unstructured
mesh**. This has significant pros and cons when compared to a structured grid.

.. contents::
   :local:

The Detail
----------
Structured Grids
~~~~~~~~~~~~~~~~
Assigning data to locations using a structured grid is essentially an act of
matching coordinate arrays to each dimension of the data array. The data can
also be represented as an area (instead of a point) by including a bounds array
for each coordinate array.

.. figure:: images/data_structured_grid.svg
   :alt: Diagram of how data is represented on a structured grid

   Data on a structured grid

   :download:`full size image <images/data_structured_grid.svg>`

UGRID Unstructured Meshes
~~~~~~~~~~~~~~~~~~~~~~~~~
UGRID is based on a **mesh** instead of a grid. The most basic element in a
mesh is the 0-dimensional **node**: a single location in space. Every node in
the mesh is defined by indexing the 1-dimensional X and Y (and optionally Z)
coordinate arrays (the ``node_coordinates``) - e.g. ``(x[3], y[3])`` gives the
position of the fourth node. Since nodes can be anywhere in this space -
**unstructured** - the position in the array has nothing to do with spatial
position.

If data is assigned to node location it must be stored in a 1D array of equal
length to the coordinate arrays. ``data[3]`` is at the position:
``(x[3], y[3])``.

Data can also be assigned to higher dimensional elements - **edges**, **faces**
or **volumes**. These elements are constructed by connecting nodes together
using a 2-dimensional **connectivity** array. One dimension varies over each
element, while the other dimension varies over each node that makes up that
element; the values in the array are the node indices. E.g. we could make 2
square faces from 6 nodes using this ``face_node_connectivity``:
``[[0, 1, 3, 2], [2, 3, 5, 4]]``. Remember that UGRID is **unstructured**, so
there is no significance to the order of the faces in the array. Data assigned
to a higher dimensional location must be stored in a 1D array of equal length
to that connectivity array, e.g. ``my_data = [0.33, 4.02]`` for our example.

.. note::

        Connectivities also exist to connect the higher dimensional elements,
        e.g. ``face_edge_connectivity``. These are optional conveniences to
        speed up certain operations and will not be discussed here.

.. figure:: images/data_ugrid_mesh.svg
   :alt: Diagram of how data is represented on a UGRID unstructured mesh

   Data on a UGRID Unstructured Mesh

   :download:`full size image <images/data_ugrid_mesh.svg>`

----

UGRID also includes support for edges/faces/volumes to have associated 'centre'
coordinates - to allow point data to be assigned to these elements. 'Centre' is
just a convenience term - the points can exist anywhere within their respective
elements.

.. figure:: images/ugrid_element_centres.svg
   :alt: Diagram demonstrating UGRID face-centred data.

   Data can be assigned to UGRID edge/face/volume 'centres'

UGRID's Flexibility
~~~~~~~~~~~~~~~~~~~
Above we have seen how one could replicate data on a structured grid using
UGRID instead. But the utility of UGRID is the extra flexibility it offers.
Here are the main examples:

* Every UGRID node is completely independent - every one can have unique X and
  Y (and Z) coordinate values.

.. figure:: images/ugrid_node_independence.svg
   :alt: Diagram demonstrating the independence of each node in UGRID
   :align: center

   Every UGRID node is completely independent

* Faces and volumes can have variable node counts, i.e. different numbers of
  sides. This is achieved by masking the unused 'slots' in the connectivity array.

.. figure:: images/ugrid_variable_faces.svg
   :alt: Diagram demonstrating UGRID faces with variable node counts
   :align: center

   UGRID faces can have different node counts (using masking)

* Data can be assigned to lines (edges) just as easily as points (nodes) or
  areas (faces).

.. figure:: images/ugrid_edge_data.svg
   :alt: Diagram demonstrating data assigned to UGRID edges
   :align: center

   Data can be assigned to UGRID edges

What does this mean?
--------------------
UGRID can represent much more varied spatial arrangements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
UGRID's highly specific way of recording location (geometry) and shape
(topology) allows it to represent essentially **any** spatial arrangement of
data. There are therefore many new applications that aren't possible using a
structured grid, including:

* `The UK Met Office's LFRic cubed-sphere <https://hps.vi4io.org/_media/events/2018/sig-io-uk-adams.pdf>`_
* `Oceanic model outputs <https://doi.org/10.3390/jmse2010194>`_

.. todo:
        a third example!

UGRID 'payload' is much larger than with structured grids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Coordinates are recorded per-node, and connectivities are recorded per-element.
This is opposed to a structured grid, where a single coordinate value is shared
by every data point/area along that line.

For example: representing the Earth as a cubed-sphere leads to coordinates and
connectivities being **~8 times larger than the data itself**, as opposed to a
small fraction of the data size when using a structured grid.

This further increases the emphasis on lazy loading and processing of data
using packages such as Dask.

.. note::

        UGRID's large, 1D data arrays are a very different shape to what Iris
        users and developers are used to. It is suspected that optimal
        performance will need new chunking strategies, but at time of writing
        (``Jan 2022``) experience is still limited.

.. todo:
        Revisit when we have more information.

Spatial operations on UGRID data are more complex
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Detail: :doc:`operations`

Indexing a UGRID data array cannot be used for:

#. Region selection
#. Neighbour identification

This is because - unlike with a structured data array - relative position in
UGRID's 1-dimensional data arrays has no relation to relative position in
space. We must instead perform specialised operations using the information in
the connectivities present, or by translating the mesh into a format designed
for mesh analysis such as VTK.

Such calculations can still be optimised to avoid them slowing workflows, but
the important take-away here is that **adaptation is needed when working UGRID
data**.


How Iris Represents This
========================

..
    Include API links to the various classes

    Include Cube/Mesh printout(s)

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

A :class:`~iris.experimental.ugrid.Mesh`
method is available to create a :class:`~iris.experimental.ugrid.MeshCoord` for
every axis represented by that :class:`~iris.experimental.ugrid.Mesh`,
requiring only the ``location`` argument:
:meth:`~iris.experimental.ugrid.Mesh.to_MeshCoords`.


__ CF-UGRID_