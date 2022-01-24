.. _ugrid operations:

Working with Mesh Data
***********************

.. note:: Several of the operations below rely on the optional dependencies
          mentioned in :doc:`partner_packages`.

.. contents::
   :local:

..
    Below: use demo code over prose wherever workable. Headings aren't an
     exhaustive list (can you think of any other popular operations?).

Making a Mesh
-------------
|new|
~~~~~
Creating Iris objects from scratch is a highly useful skill for testing code
and improving understanding of how Iris works. This knowledge will likely prove
particularly useful when converting data into the Iris mesh data model from
structured formats and non-UGRID mesh formats.

The objects created in this example will be used where possible in the
subsequent example operations on this page.

.. doctest:: ugrid_operations

    >>> import numpy as np

    >>> from iris.coords import AuxCoord
    >>> from iris.experimental.ugrid import Connectivity, Mesh

    # Going to create the following mesh
    #  (node indices are shown to aid understanding):
    #
    #  0----1
    #  |    |\
    #  | +  |+\
    #  2----3--4

    >>> node_x = AuxCoord(
    ...     points=[3.0, 3.0, 0.0, 0.0, 0.0],
    ...     standard_name="longitude",
    ...     units="degrees_east",
    ...     long_name="node_x_coordinates",
    ... )
    >>> node_y = AuxCoord(points=[0.0, 5.0, 0.0, 5.0, 8.0], standard_name="latitude")

    >>> face_x = AuxCoord([2.0, 6.0], "longitude")
    >>> face_y = AuxCoord([1.0, 1.0], "latitude")

    >>> edge_node_c = Connectivity(
    ...     indices=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 3], [3, 4]],
    ...     cf_role="edge_node_connectivity",
    ...     attributes={"demo": "Supports every standard CF property"},
    ... )

    # Create some dead-centre edge coordinates.
    >>> edge_x, edge_y = [
    ...     AuxCoord(
    ...         node_coord.points[edge_node_c.indices_by_src()].mean(axis=1),
    ...         node_coord.standard_name,
    ...     )
    ...     for node_coord in (node_x, node_y)
    ... ]

    >>> face_indices = np.ma.masked_where(999, [[0, 1, 3, 2], [1, 4, 3, 999]])
    >>> face_node_c = Connectivity(
    ...     indices=face_indices, cf_role="face_node_connectivity"
    ... )

    >>> my_mesh = Mesh(
    ...     long_name="my_mesh",
    ...     topology_dimension=2,  # Supports 2D (face) elements.
    ...     node_coords_and_axes=[(node_x, "x"), (node_y, "y")],
    ...     connectivities=[edge_node_c, face_node_c],
    ...     edge_coords_and_axes=[(edge_x, "x"), (edge_y, "y")],
    ...     face_coords_and_axes=[(face_x, "x"), (face_y, "y")],
    ... )

Making a Cube (with a Mesh)
---------------------------
|unchanged|
~~~~~~~~~~~
Creating a :class:`~iris.cube.Cube` is unchanged; the
:class:`~iris.experimental.ugrid.Mesh` is linked via a
:class:`~iris.experimental.ugrid.MeshCoord` (see :ref:`ugrid MeshCoords`):

.. doctest:: ugrid_operations

    >>> import numpy as np

    >>> from iris.coords import DimCoord
    >>> from iris.cube import Cube, CubeList

    >>> vertical_levels = DimCoord([0, 1, 2], "height")

    >>> my_cubelist = CubeList()
    >>> for conn in (edge_node_c, face_node_c):
    ...    location = conn.src_location
    ...    mesh_coord_x, mesh_coord_y = my_mesh.to_MeshCoords(location)
    ...    data_shape = (len(conn.indices_by_src()), len(vertical_levels.points))
    ...    data_array = np.arange(np.prod(data_shape)).reshape(data_shape)
    ...
    ...    my_cubelist.append(
    ...        Cube(
    ...            data=data_array,
    ...            long_name=f"{location}_data",
    ...            units="K",
    ...            dim_coords_and_dims=[(vertical_levels, 1)],
    ...            aux_coords_and_dims=[(mesh_coord_x, 0), (mesh_coord_y, 0)],
    ...        )
    ...    )

    >>> print(my_cubelist)
    0: edge_data / (K)                     (-- : 6; height: 3)
    1: face_data / (K)                     (-- : 2; height: 3)

    >>> for cube in my_cubelist:
    ...     print(f"{cube.name()}: {cube.mesh.name()}, {cube.location}")
    edge_data: my_mesh, edge
    face_data: my_mesh, face

Save
----
|unchanged|
~~~~~~~~~~~
.. note:: UGRID saving support is limited to the NetCDF file format.

The Iris saving process automatically detects if the :class:`~iris.cube.Cube`
has an associated :class:`~iris.experimental.ugrid.Mesh` and automatically
saves the file in a UGRID-conformant format:

.. doctest:: ugrid_operations

    >>> from subprocess import run

    >>> from iris import save

    >>> cubelist_path = "my_cubelist.nc"
    >>> save(my_cubelist, cubelist_path)

    >>> ncdump_result = run(["ncdump", "-h", cubelist_path], capture_output=True)
    >>> print(ncdump_result.stdout.decode().replace("\t", "    "))
    netcdf my_cubelist {
    dimensions:
        Mesh2d_node = 5 ;
        Mesh2d_edge = 6 ;
        Mesh2d_face = 2 ;
        height = 3 ;
        my_mesh_face_N_nodes = 4 ;
        my_mesh_edge_N_nodes = 2 ;
    variables:
        int my_mesh ;
            my_mesh:cf_role = "mesh_topology" ;
            my_mesh:topology_dimension = 2 ;
            my_mesh:long_name = "my_mesh" ;
            my_mesh:node_coordinates = "longitude latitude" ;
            my_mesh:edge_coordinates = "longitude_0 latitude_0" ;
            my_mesh:face_coordinates = "longitude_1 latitude_1" ;
            my_mesh:face_node_connectivity = "mesh2d_face" ;
            my_mesh:edge_node_connectivity = "mesh2d_edge" ;
        double longitude(Mesh2d_node) ;
            longitude:units = "degrees_east" ;
            longitude:standard_name = "longitude" ;
            longitude:long_name = "node_x_coordinates" ;
        double latitude(Mesh2d_node) ;
            latitude:standard_name = "latitude" ;
        double longitude_0(Mesh2d_edge) ;
            longitude_0:standard_name = "longitude" ;
        double latitude_0(Mesh2d_edge) ;
            latitude_0:standard_name = "latitude" ;
        double longitude_1(Mesh2d_face) ;
            longitude_1:standard_name = "longitude" ;
        double latitude_1(Mesh2d_face) ;
            latitude_1:standard_name = "latitude" ;
        int64 mesh2d_face(Mesh2d_face, my_mesh_face_N_nodes) ;
            mesh2d_face:_FillValue = -1LL ;
            mesh2d_face:cf_role = "face_node_connectivity" ;
            mesh2d_face:start_index = 0LL ;
        int64 mesh2d_edge(Mesh2d_edge, my_mesh_edge_N_nodes) ;
            mesh2d_edge:demo = "Supports every standard CF property" ;
            mesh2d_edge:cf_role = "edge_node_connectivity" ;
            mesh2d_edge:start_index = 0LL ;
        int64 edge_data(Mesh2d_edge, height) ;
            edge_data:long_name = "edge_data" ;
            edge_data:units = "K" ;
            edge_data:mesh = "my_mesh" ;
            edge_data:location = "edge" ;
        int64 height(height) ;
            height:standard_name = "height" ;
        int64 face_data(Mesh2d_face, height) ;
            face_data:long_name = "face_data" ;
            face_data:units = "K" ;
            face_data:mesh = "my_mesh" ;
            face_data:location = "face" ;
    <BLANKLINE>
    // global attributes:
            :Conventions = "CF-1.7" ;
    }
    <BLANKLINE>

The :func:`iris.experimental.ugrid.save_mesh` function allows
:class:`~iris.experimental.ugrid.Mesh`\es to be saved to file without
associated :class:`~iris.cube.Cube`\s:

.. doctest:: ugrid_operations

    >>> from subprocess import run

    >>> from iris.experimental.ugrid import save_mesh

    >>> mesh_path = "my_mesh.nc"
    >>> save_mesh(my_mesh, mesh_path)

    >>> ncdump_result = run(["ncdump", "-h", mesh_path], capture_output=True)
    >>> print(ncdump_result.stdout.decode().replace("\t", "    "))
    netcdf my_mesh {
    dimensions:
        Mesh2d_node = 5 ;
        Mesh2d_edge = 6 ;
        Mesh2d_face = 2 ;
        my_mesh_face_N_nodes = 4 ;
        my_mesh_edge_N_nodes = 2 ;
    variables:
        int my_mesh ;
            my_mesh:cf_role = "mesh_topology" ;
            my_mesh:topology_dimension = 2 ;
            my_mesh:long_name = "my_mesh" ;
            my_mesh:node_coordinates = "longitude latitude" ;
            my_mesh:edge_coordinates = "longitude_0 latitude_0" ;
            my_mesh:face_coordinates = "longitude_1 latitude_1" ;
            my_mesh:face_node_connectivity = "mesh2d_face" ;
            my_mesh:edge_node_connectivity = "mesh2d_edge" ;
        double longitude(Mesh2d_node) ;
            longitude:units = "degrees_east" ;
            longitude:standard_name = "longitude" ;
            longitude:long_name = "node_x_coordinates" ;
        double latitude(Mesh2d_node) ;
            latitude:standard_name = "latitude" ;
        double longitude_0(Mesh2d_edge) ;
            longitude_0:standard_name = "longitude" ;
        double latitude_0(Mesh2d_edge) ;
            latitude_0:standard_name = "latitude" ;
        double longitude_1(Mesh2d_face) ;
            longitude_1:standard_name = "longitude" ;
        double latitude_1(Mesh2d_face) ;
            latitude_1:standard_name = "latitude" ;
        int64 mesh2d_face(Mesh2d_face, my_mesh_face_N_nodes) ;
            mesh2d_face:_FillValue = -1LL ;
            mesh2d_face:cf_role = "face_node_connectivity" ;
            mesh2d_face:start_index = 0LL ;
        int64 mesh2d_edge(Mesh2d_edge, my_mesh_edge_N_nodes) ;
            mesh2d_edge:demo = "Supports every standard CF property" ;
            mesh2d_edge:cf_role = "edge_node_connectivity" ;
            mesh2d_edge:start_index = 0LL ;
    <BLANKLINE>
    // global attributes:
            :Conventions = "CF-1.7" ;
    }
    <BLANKLINE>

Load
----
|different| - UGRID parsing is opt-in
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. note:: UGRID loading support is limited to the NetCDF file format.

While Iris' UGRID support remains :mod:`~iris.experimental`, parsing UGRID when
loading a file remains **optional**. To load UGRID data from a file into the
Iris mesh data model, use the
:const:`iris.experimental.ugrid.PARSE_UGRID_ON_LOAD` context manager:

.. doctest:: ugrid_operations

    >>> from iris import load
    >>> from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD

    >>> with PARSE_UGRID_ON_LOAD.context():
    ...     loaded_cubelist = load(cubelist_path)

    # Sort CubeList to ensure consistent result.
    >>> loaded_cubelist.sort(key=lambda cube: cube.name())
    >>> print(loaded_cubelist)
    0: edge_data / (K)                     (-- : 6; height: 3)
    1: face_data / (K)                     (-- : 2; height: 3)

All the existing loading functionality still operates on UGRID-compliant
data - :class:`~iris.Constraint`\s, callbacks, :func:`~iris.load_cube`
etcetera:

.. doctest:: ugrid_operations

    >>> from iris import Constraint, load_cube

    >>> with PARSE_UGRID_ON_LOAD.context():
    ...     ground_cubelist = load(cubelist_path, Constraint(height=0))
    ...     face_cube = load_cube(cubelist_path, "face_data")

    # Sort CubeList to ensure consistent result.
    >>> ground_cubelist.sort(key=lambda cube: cube.name())
    >>> print(ground_cubelist)
    0: edge_data / (K)                     (-- : 6)
    1: face_data / (K)                     (-- : 2)

    >>> print(face_cube)
    face_data / (K)                     (-- : 2; height: 3)
        Dimension coordinates:
            height                          -          x
        Mesh coordinates:
            latitude                        x          -
            longitude                       x          -
        Attributes:
            Conventions                 CF-1.7

.. note::

    We recommend caution if constraining on coordinates associated with a
    :class:`~iris.experimental.ugrid.Mesh`. An individual coordinate value
    might not be shared by any other data points, and using a coordinate range
    will demand notably higher performance given the size of the dimension
    versus structured grids
    (:ref:`see the data model detail <ugrid implications>`).

The :func:`iris.experimental.ugrid.load_mesh` and
:func:`~iris.experimental.ugrid.load_meshes` functions allow only
:class:`~iris.experimental.ugrid.Mesh`\es to be loaded from a file without
creating any associated :class:`~iris.cube.Cube`\s:

.. doctest:: ugrid_operations

    >>> from iris.experimental.ugrid import load_mesh

    >>> with PARSE_UGRID_ON_LOAD.context():
    ...     loaded_mesh = load_mesh(cubelist_path)

.. todo: print(loaded_mesh) - once printouts have been improved.

Summary
-------
.. todo: populate or remove

..
    Possibly covered by the data_model page?

Plotting
--------
|different| - plot with GeoVista
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. todo: populate!

Regional Extraction
-------------------
|different| - use GeoVista for mesh analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. todo: populate!

..
    Highlight the uselessness of indexing.

Regridding
----------
|different| - use iris-esmf-regrid for mesh regridders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. todo: populate!

Equality
--------
|unchanged|
~~~~~~~~~~~
:class:`~iris.experimental.ugrid.Mesh` comparison is supported:

.. doctest:: ugrid_operations

    >>> from copy import deepcopy

    >>> same_mesh = deepcopy(my_mesh)
    >>> print(my_mesh == same_mesh)
    True

    >>> different_mesh = deepcopy(my_mesh)
    >>> different_mesh.edge_node_connectivity.indices[0] = [0, 4]
    >>> print(my_mesh == different_mesh)
    False

Associated :class:`~iris.experimental.ugrid.Mesh`\es are included in
:class:`~iris.cube.Cube` comparisons:

.. doctest:: ugrid_operations

    >>> edge_cube = my_cubelist.extract_cube("edge_data")
    >>> different_cube = deepcopy(edge_cube)
    >>> for coord in different_cube.coords(mesh_coords=True):
    ...     different_cube.remove_coord(coord.name())
    >>> different_coords = different_mesh.to_MeshCoords(location="edge")
    >>> for coord in different_coords:
    ...     different_cube.add_aux_coord(coord, edge_cube.mesh_dim())

    >>> print(edge_cube == different_cube)
    False

.. note::

    Keep an eye on memory demand when comparing large
    :class:`~iris.experimental.ugrid.Mesh`\es, but note that
    :class:`~iris.experimental.ugrid.Mesh`\ equality is enabled for lazy
    processing (:doc:`/userguide/real_and_lazy_data`), so if the
    :class:`~iris.experimental.ugrid.Mesh`\es being compared are lazy the
    process will use less memory than their total size.

Combining Cubes
---------------
|different|
~~~~~~~~~~~
.. todo: populate!

Arithmetic
----------
|pending|
~~~~~~~~~
:class:`~iris.cube.Cube` Arithmetic (described in :doc:`/userguide/cube_maths`)
has not yet been adapted to handle :class:`~iris.cube.Cube`\s that include
:class:`~iris.experimental.ugrid.MeshCoord`\s.


.. todo:
    Enumerate other popular operations that aren't yet possible
     (and are they planned soon?)

.. |new| replace:: ✨ New
.. |unchanged| replace:: ♻️ Unchanged
.. |different| replace:: ⚠️ Different
.. |pending| replace:: 🚧 Support Pending