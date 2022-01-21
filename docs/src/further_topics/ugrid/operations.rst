.. _ugrid operations:

Working with Mesh Data
***********************

.. note:: Several of the operations below rely on the optional dependencies
          mentioned in :doc:`partner_packages`.

..
    Have a table here that lists the headings below, including a small note
     about whether it's unchanged (e.g. Saving), has to be done differently
     (e.g. Extraction), or isn't yet possible (e.g. Arithmetic). Each row
     should link to the section below. Basically an enhanced TOC.

..
    Below: use demo code over prose wherever workable. Headings aren't an
     exhaustive list (can you think of any other popular operations?).

Making a Cube with a Mesh
-------------------------
Creating Iris objects from scratch is a highly useful skill for testing code
and improving understanding of how Iris works. This knowledge will likely prove
particularly useful when converting data into the Iris mesh data model from
structured formats and non-UGRID mesh formats.

The objects created in this example will be used where possible in the
subsequent example operations on this page.

Create a :class:`~iris.experimental.ugrid.Mesh` :

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

Link the :class:`~iris.experimental.ugrid.Mesh` to new
:class:`~iris.cube.Cube`\s :

.. doctest:: ugrid_operations

    >>> import numpy as np

    >>> from iris.coords import DimCoord
    >>> from iris.cube import Cube, CubeList

    >>> vertical_levels = DimCoord([0, 1, 2], "height")

    >>> my_cube_list = CubeList()
    >>> for conn in (edge_node_c, face_node_c):
    ...    location = conn.src_location
    ...    mesh_coord_x, mesh_coord_y = my_mesh.to_MeshCoords(location)
    ...    data_shape = (len(conn.indices_by_src()), len(vertical_levels.points))
    ...    data_array = np.arange(np.prod(data_shape)).reshape(data_shape)
    ...
    ...    my_cube_list.append(
    ...        Cube(
    ...            data=data_array,
    ...            long_name=f"{location}_data",
    ...            units="K",
    ...            dim_coords_and_dims=[(vertical_levels, 1)],
    ...            aux_coords_and_dims=[(mesh_coord_x, 0), (mesh_coord_y, 0)],
    ...        )
    ...    )

    >>> print(my_cube_list)
    0: edge_data / (K)                     (-- : 6; height: 3)
    1: face_data / (K)                     (-- : 2; height: 3)
    >>> for cube in my_cube_list:
    ...     print(f"{cube.name()}: {cube.mesh.name()}, {cube.location}")
    edge_data: my_mesh, edge
    face_data: my_mesh, face

Save
----
.. doctest:: ugrid_operations

    >>> from subprocess import run

    >>> from iris import save
    >>> from iris.experimental.ugrid import save_mesh

    >>> target_path = "my_cube_list.nc"
    >>> save(my_cube_list, target_path)
    >>> ncdump_result = run(["ncdump", "-h", target_path], capture_output=True)
    >>> print(ncdump_result.stdout.decode().replace("\t", "    "))
    netcdf my_cube_list {
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

Load
----

Summary
-------
..
    Possibly covered by the data_model page?

Plotting
--------

Regional Extraction
-------------------
..
    Highlight the uselessness of indexing.

Regridding
----------

Equality
--------
..
    Is this worth mentioning, given it just works the way it always has?

Recombination
-------------

Arithmetic
----------
..
    Not possible yet - mention this.

..
    Headings for other popular operations that aren't yet possible, including
     if they're planned soon.
