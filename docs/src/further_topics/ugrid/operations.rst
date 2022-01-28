.. _ugrid operations:

Working with Mesh Data
**********************

.. note:: Several of the operations below rely on the optional dependencies
          mentioned in :doc:`partner_packages`.

Operations Summary
------------------
.. list-table::
    :align: left
    :widths: 35, 75

    * - `Making a Mesh`_
      - |tagline: making a mesh|
    * - `Making a Cube`_
      - |tagline: making a cube|
    * - `Save`_
      - |tagline: save|
    * - `Load`_
      - |tagline: load|
    * - `Plotting`_
      - |tagline: plotting|
    * - `Region Extraction`_
      - |tagline: region extraction|
    * - `Regridding`_
      - |tagline: regridding|
    * - `Equality`_
      - |tagline: equality|
    * - `Combining Cubes`_
      - |tagline: combining cubes|
    * - `Arithmetic`_
      - |tagline: arithmetic|

..
    Below: use demo code over prose wherever workable. Headings aren't an
     exhaustive list (can you think of any other popular operations?).

Making a Mesh
-------------
.. |tagline: making a mesh| replace:: |new|

.. rubric:: |tagline: making a mesh|

**Already have a file?** Consider skipping to `Load`_.

Creating Iris objects from scratch is a highly useful skill for testing code
and improving understanding of how Iris works. This knowledge will likely prove
particularly useful when converting data into the Iris mesh data model from
structured formats and non-UGRID mesh formats.

The objects created in this example will be used where possible in the
subsequent example operations on this page.

.. dropdown:: :opticon:`code`

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
        ...     points=[0.0, 5.0, 0.0, 5.0, 8.0],
        ...     standard_name="longitude",
        ...     units="degrees_east",
        ...     long_name="node_x_coordinates",
        ... )
        >>> node_y = AuxCoord(points=[3.0, 3.0, 0.0, 0.0, 0.0], standard_name="latitude")

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
        ...         node_coord.points[edge_node_c.indices_by_location()].mean(axis=1),
        ...         node_coord.standard_name,
        ...     )
        ...     for node_coord in (node_x, node_y)
        ... ]

        >>> face_indices = np.ma.masked_equal([[0, 1, 3, 2], [1, 4, 3, 999]], 999)
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

        >>> print(my_mesh)
        Mesh : 'my_mesh'
            topology_dimension: 2
            node
                node_dimension: 'Mesh2d_node'
                node coordinates
                    <AuxCoord: longitude / (degrees_east)  [...]  shape(5,)>
                    <AuxCoord: latitude / (unknown)  [...]  shape(5,)>
            edge
                edge_dimension: 'Mesh2d_edge'
                edge_node_connectivity: <Connectivity: unknown / (unknown)  [...]  shape(6, 2)>
                edge coordinates
                    <AuxCoord: longitude / (unknown)  [...]  shape(6,)>
                    <AuxCoord: latitude / (unknown)  [...]  shape(6,)>
            face
                face_dimension: 'Mesh2d_face'
                face_node_connectivity: <Connectivity: unknown / (unknown)  [...]  shape(2, 4)>
                face coordinates
                    <AuxCoord: longitude / (unknown)  [...]  shape(2,)>
                    <AuxCoord: latitude / (unknown)  [...]  shape(2,)>
            long_name: 'my_mesh'


.. _making a cube:

Making a Cube (with a Mesh)
---------------------------
.. |tagline: making a cube| replace:: |unchanged|

.. rubric:: |tagline: making a cube|

Creating a :class:`~iris.cube.Cube` is unchanged; the
:class:`~iris.experimental.ugrid.Mesh` is linked via a
:class:`~iris.experimental.ugrid.MeshCoord` (see :ref:`ugrid MeshCoords`):

.. dropdown:: :opticon:`code`

    .. doctest:: ugrid_operations

        >>> import numpy as np

        >>> from iris.coords import DimCoord
        >>> from iris.cube import Cube, CubeList

        >>> vertical_levels = DimCoord([0, 1, 2], "height")

        >>> my_cubelist = CubeList()
        >>> for conn in (edge_node_c, face_node_c):
        ...    location = conn.location
        ...    mesh_coord_x, mesh_coord_y = my_mesh.to_MeshCoords(location)
        ...    data_shape = (len(conn.indices_by_location()), len(vertical_levels.points))
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

        >>> print(my_cubelist.extract_cube("edge_data"))
        edge_data / (K)                     (-- : 6; height: 3)
            Dimension coordinates:
                height                          -          x
            Mesh coordinates:
                latitude                        x          -
                longitude                       x          -


Save
----
.. |tagline: save| replace:: |unchanged|

.. rubric:: |tagline: save|

.. note:: UGRID saving support is limited to the NetCDF file format.

The Iris saving process automatically detects if the :class:`~iris.cube.Cube`
has an associated :class:`~iris.experimental.ugrid.Mesh` and automatically
saves the file in a UGRID-conformant format:

.. dropdown:: :opticon:`code`

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

.. dropdown:: :opticon:`code`

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
.. |tagline: load| replace:: |different| - UGRID parsing is opt-in

.. rubric:: |tagline: load|

.. note:: UGRID loading support is limited to the NetCDF file format.

While Iris' UGRID support remains :mod:`~iris.experimental`, parsing UGRID when
loading a file remains **optional**. To load UGRID data from a file into the
Iris mesh data model, use the
:const:`iris.experimental.ugrid.PARSE_UGRID_ON_LOAD` context manager:

.. dropdown:: :opticon:`code`

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

.. dropdown:: :opticon:`code`

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
                Conventions                 'CF-1.7'

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

.. dropdown:: :opticon:`code`

    .. doctest:: ugrid_operations

        >>> from iris.experimental.ugrid import load_mesh

        >>> with PARSE_UGRID_ON_LOAD.context():
        ...     loaded_mesh = load_mesh(cubelist_path)

        >>> print(loaded_mesh)
        Mesh : 'my_mesh'
            topology_dimension: 2
            node
                node_dimension: 'Mesh2d_node'
                node coordinates
                    <AuxCoord: longitude / (degrees)  <lazy>  shape(5,)>
                    <AuxCoord: latitude / (unknown)  <lazy>  shape(5,)>
            edge
                edge_dimension: 'Mesh2d_edge'
                edge_node_connectivity: <Connectivity: mesh2d_edge / (unknown)  <lazy>  shape(6, 2)>
                edge coordinates
                    <AuxCoord: longitude / (unknown)  <lazy>  shape(6,)>
                    <AuxCoord: latitude / (unknown)  <lazy>  shape(6,)>
            face
                face_dimension: 'Mesh2d_face'
                face_node_connectivity: <Connectivity: mesh2d_face / (unknown)  <lazy>  shape(2, 4)>
                face coordinates
                    <AuxCoord: longitude / (unknown)  <lazy>  shape(2,)>
                    <AuxCoord: latitude / (unknown)  <lazy>  shape(2,)>
            long_name: 'my_mesh'
            var_name: 'my_mesh'

Plotting
--------
.. |tagline: plotting| replace:: |different| - plot with GeoVista

.. rubric:: |tagline: plotting|

The Cartopy-Matplotlib combination is not optimised for displaying the high
number of irregular shapes associated with meshes. Thankfully mesh
visualisation is already popular in many other fields (e.g. CGI, gaming,
SEM microscopy), so there is a wealth of tooling available, which
:ref:`ugrid geovista` harnesses for cartographic plotting.

GeoVista's default behaviour is to convert lat-lon information into full XYZ
coordinates so the data is visualised on the surface of a 3D globe. The plots
are interactive by default, so it's easy to explore the data in detail.

2D projections have also been demonstrated in proofs of concept, and will
be added to API in the near future.

This first example uses GeoVista to plot the ``face_cube`` that we created
earlier:

.. dropdown:: :opticon:`code`

    .. code-block:: python

        >>> from geovista import GeoPlotter, Transform
        >>> from geovista.common import to_xyz


        # We'll re-use this to plot some real global data later.
        >>> def cube_faces_to_polydata(cube):
        ...     lons, lats = cube.mesh.node_coords
        ...     face_node = cube.mesh.face_node_connectivity
        ...     indices = face_node.indices_by_location()
        ...
        ...     mesh = Transform.from_unstructured(
        ...         lons.points,
        ...         lats.points,
        ...         indices,
        ...         data=cube.data,
        ...         name=f"{cube.name()} / {cube.units}",
        ...         start_index=face_node.start_index,
        ...     )
        ...     return mesh

        >>> print(face_cube)
        face_data / (K)                     (-- : 2; height: 3)
            Dimension coordinates:
                height                          -          x
            Mesh coordinates:
                latitude                        x          -
                longitude                       x          -
            Attributes:
                Conventions                 'CF-1.7'

        # Convert our mesh+data to a PolyData object.
        # Just plotting a single height level.
        >>> face_polydata = cube_faces_to_polydata(face_cube[:, 0])
        >>> print(face_polydata)
        PolyData (0x7ff4861ff4c0)
          N Cells:	2
          N Points:	5
          X Bounds:	9.903e-01, 1.000e+00
          Y Bounds:	0.000e+00, 1.392e-01
          Z Bounds:	6.123e-17, 5.234e-02
          N Arrays:	2

        # Create the GeoVista plotter and add our mesh+data to it.
        >>> my_plotter = GeoPlotter()
        >>> my_plotter.add_coastlines(color="black")
        >>> my_plotter.add_base_layer(color="grey")
        >>> my_plotter.add_mesh(face_polydata)

        # Centre the camera on the data.
        >>> camera_region = to_xyz(
        ...     face_cube.coord("longitude").points,
        ...     face_cube.coord("latitude").points,
        ...     radius=3,
        ... )
        >>> camera_pos = camera_region.mean(axis=0)
        >>> my_plotter.camera.position = camera_pos

        >>> my_plotter.show()

    ..  image:: images/plotting_basic.png
        :alt: A GeoVista plot of the basic example Mesh.

    This artificial data makes West Africa rather chilly!

Here's another example using a global cubed-sphere data set:

.. dropdown:: :opticon:`code`

    .. code-block:: python

        >>> from iris import load_cube
        >>> from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD

        # Demonstrating with a global data set.
        # You could also download this file from github.com/SciTools/iris-test-data.
        >>> from iris.tests import get_data_path
        >>> file_path = get_data_path(
        ...     [
        ...         "NetCDF",
        ...         "unstructured_grid",
        ...         "lfric_surface_mean.nc",
        ...     ]
        ... )
        >>> with PARSE_UGRID_ON_LOAD.context():
        ...     global_cube = load_cube(file_path, "tstar_sea")
        >>> print(global_cube)
        sea_surface_temperature / (K)       (-- : 1; -- : 13824)
            Mesh coordinates:
                latitude                        -       x
                longitude                       -       x
            Auxiliary coordinates:
                time                            x       -
            Cell methods:
                mean                        time (300 s)
                mean                        time_counter
            Attributes:
                Conventions                 UGRID
                description                 Created by xios
                interval_operation          300 s
                interval_write              1 d
                name                        lfric_surface
                online_operation            average
                timeStamp                   2020-Feb-07 16:23:14 GMT
                title                       Created by xios
                uuid                        489bcef5-3d1c-4529-be42-4ab5f8c8497b

        >>> global_polydata = cube_faces_to_polydata(global_cube)
        >>> print(global_polydata)
        PolyData (0x7f761b536160)
          N Cells:	13824
          N Points:	13826
          X Bounds:	-1.000e+00, 1.000e+00
          Y Bounds:	-1.000e+00, 1.000e+00
          Z Bounds:	-1.000e+00, 1.000e+00
          N Arrays:	2

        >>> my_plotter = GeoPlotter()
        >>> my_plotter.add_coastlines()
        >>> my_plotter.add_mesh(global_polydata, show_edges=True)

        >>> my_plotter.show()

    ..  image:: images/plotting_global.png
        :alt: A GeoVista plot of a global sea surface temperature Mesh.

Region Extraction
-----------------
.. |tagline: region extraction| replace:: |different| - use GeoVista for mesh analysis

.. rubric:: |tagline: region extraction|

As described in :doc:`data_model`, indexing for a range along a
:class:`~iris.cube.Cube`\'s :meth:`~iris.cube.Cube.mesh_dim` will not provide
a contiguous region, since **position on the unstructured dimension is
unrelated to spatial position**. This means that subsetted
:class:`~iris.experimental.ugrid.MeshCoord`\s cannot be reliably interpreted
as intended, and subsetting a :class:`~iris.experimental.ugrid.MeshCoord` is
therefore set to return an :class:`~iris.coords.AuxCoord` instead - breaking
the link between :class:`~iris.cube.Cube` and
:class:`~iris.experimental.ugrid.Mesh`:

.. dropdown:: :opticon:`code`

    .. doctest:: ugrid_operations

        >>> edge_cube = my_cubelist.extract_cube("edge_data")
        >>> print(edge_cube)
        edge_data / (K)                     (-- : 6; height: 3)
            Dimension coordinates:
                height                          -          x
            Mesh coordinates:
                latitude                        x          -
                longitude                       x          -

        # Sub-setted MeshCoords have become AuxCoords.
        >>> print(edge_cube[:-1])
        edge_data / (K)                     (-- : 5; height: 3)
            Dimension coordinates:
                height                          -          x
            Auxiliary coordinates:
                latitude                        x          -
                longitude                       x          -

Extracting a region therefore requires extra steps - to determine the spatial
position of the data points before they can be analysed as inside/outside the
selected region. The recommended way to do this is using tools provided by
:ref:`ugrid geovista`, which is optimised for performant mesh analysis.

This approach centres around using :meth:`geovista.geodesic.BBox.enclosed` to
get the subset of the original mesh that is inside the
:class:`~geovista.geodesic.BBox`. This subset :class:`pyvista.PolyData` object
includes the original indices of each datapoint - the ``vtkOriginalCellIds``
array, which can be used to index the original :class:`~iris.cube.Cube`. Since
we **know** that this subset :class:`~iris.cube.Cube` represents a regional
mesh, we then reconstruct a :class:`~iris.experimental.ugrid.Mesh` from the
:class:`~iris.cube.Cube`\'s :attr:`~iris.cube.Cube.aux_coords` using
:meth:`iris.experimental.ugrid.Mesh.from_coords`:

..
    Not using doctest here as want to keep GeoVista as optional dependency.

.. dropdown:: :opticon:`code`

    .. code-block:: python

        >>> from geovista import Transform
        >>> from geovista.geodesic import BBox
        >>> from iris import load_cube
        >>> from iris.experimental.ugrid import Mesh, PARSE_UGRID_ON_LOAD

        # Need a larger dataset to demonstrate this operation.
        # You could also download this file from github.com/SciTools/iris-test-data.
        >>> from iris.tests import get_data_path
        >>> file_path = get_data_path(
        ...     [
        ...         "NetCDF",
        ...         "unstructured_grid",
        ...         "lfric_ngvat_2D_72t_face_half_levels_main_conv_rain.nc",
        ...     ]
        ... )

        >>> with PARSE_UGRID_ON_LOAD.context():
        ...     global_cube = load_cube(file_path, "conv_rain")
        >>> print(global_cube)
        surface_convective_rainfall_rate / (kg m-2 s-1) (-- : 72; -- : 864)
            Mesh coordinates:
                latitude                                    -        x
                longitude                                   -        x
            Auxiliary coordinates:
                time                                        x        -
            Cell methods:
                point                                   time
            Attributes:
                Conventions                             UGRID
                description                             Created by xios
                interval_operation                      300 s
                interval_write                          300 s
                name                                    lfric_ngvat_2D_72t_face_half_levels_main_conv_rain
                online_operation                        instant
                timeStamp                               2020-Oct-18 21:18:35 GMT
                title                                   Created by xios
                uuid                                    b3dc0fb4-9828-4663-a5ac-2a5763280159

        # Convert the Mesh to a GeoVista PolyData object.
        >>> lons, lats = global_cube.mesh.node_coords
        >>> face_node = global_cube.mesh.face_node_connectivity
        >>> indices = face_node.indices_by_location()
        >>> global_polydata = Transform.from_unstructured(
        ...     lons.points, lats.points, indices, start_index=face_node.start_index
        ... )

        # Define a region of 4 corners connected by great circles.
        #  Specialised sub-classes of BBox are also available e.g. panel/wedge.
        >>> region = BBox(lons=[0, 70, 70, 0], lats=[-25, -25, 45, 45])
        # 'Apply' the region to the PolyData object.
        >>> region_polydata = region.enclosed(global_polydata, preference="center")
        # Get the remaining face indices, to use for indexing the Cube.
        >>> indices = region_polydata["vtkOriginalCellIds"]

        >>> print(type(indices))
        <class 'numpy.ndarray'>
        # 101 is smaller than the original 864.
        >>> print(len(indices))
        101
        >>> print(indices[:10])
        [ 6  7  8  9 10 11 18 19 20 21]

        # Use the face indices to subset the global cube.
        >>> region_cube = global_cube[:, indices]

        # In this case we **know** the indices correspond to a contiguous
        #  region, so we will convert the sub-setted Cube back into a
        #  Cube-with-Mesh.
        >>> new_mesh = Mesh.from_coords(*region_cube.coords(dimensions=1))
        >>> new_mesh_coords = new_mesh.to_MeshCoords(global_cube.location)
        >>> for coord in new_mesh_coords:
        ...     region_cube.remove_coord(coord.name())
        ...     region_cube.add_aux_coord(coord, 1)

        # A Mesh-Cube with a subset (101) of the original 864 faces.
        >>> print(region_cube)
        surface_convective_rainfall_rate / (kg m-2 s-1) (-- : 72; -- : 101)
            Mesh coordinates:
                latitude                                    -        x
                longitude                                   -        x
            Auxiliary coordinates:
                time                                        x        -
            Cell methods:
                point                                   time
            Attributes:
                Conventions                             UGRID
                description                             Created by xios
                interval_operation                      300 s
                interval_write                          300 s
                name                                    lfric_ngvat_2D_72t_face_half_levels_main_conv_rain
                online_operation                        instant
                timeStamp                               2020-Oct-18 21:18:35 GMT
                title                                   Created by xios
                uuid                                    b3dc0fb4-9828-4663-a5ac-2a5763280159

Regridding
----------
.. |tagline: regridding| replace:: |different| - use iris-esmf-regrid for mesh regridders

.. rubric:: |tagline: regridding|

Regridding to or from a mesh requires different logic than Iris' existing
regridders, which are designed for structured grids. For this we recommend
ESMF's powerful regridding tools, which integrate with Iris' mesh data model
via the :ref:`ugrid iris-esmf-regrid` package.

.. todo: inter-sphinx links when available.

Regridding is achieved via the
:class:`esmf_regrid.experimental.unstructured_scheme.MeshToGridESMFRegridder`
and
:class:`~esmf_regrid.experimental.unstructured_scheme.GridToMeshESMFRegridder`
classes. Regridding from a source :class:`~iris.cube.Cube` to a target
:class:`~iris.cube.Cube` involves initialising and then calling one of these
classes. Initialising is done by passing in the source and target
:class:`~iris.cube.Cube` as arguments. The regridder is then called by passing
the source :class:`~iris.cube.Cube` as an argument. We can demonstrate this
with the
:class:`~esmf_regrid.experimental.unstructured_scheme.MeshToGridESMFRegridder`:

..
    Not using doctest here as want to keep iris-esmf-regrid as optional dependency.

.. dropdown:: :opticon:`code`

    .. code-block:: python

        >>> from esmf_regrid.experimental.unstructured_scheme import MeshToGridESMFRegridder
        >>> from iris import load, load_cube
        >>> from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD

        # You could also download these files from github.com/SciTools/iris-test-data.
        >>> from iris.tests import get_data_path
        >>> mesh_file = get_data_path(
        ...     ["NetCDF", "unstructured_grid", "lfric_surface_mean.nc"]
        ... )
        >>> grid_file = get_data_path(
        ...     ["NetCDF", "regrid", "regrid_template_global_latlon.nc"]
        ... )

        # Load a list of cubes defined on the same Mesh.
        >>> with PARSE_UGRID_ON_LOAD.context():
        ...     mesh_cubes = load(mesh_file)

        # Extract a specific cube.
        >>> mesh_cube1 = mesh_cubes.extract_cube("sea_surface_temperature")
        >>> print(mesh_cube1)
        sea_surface_temperature / (K)       (-- : 1; -- : 13824)
            Mesh coordinates:
                latitude                        -       x
                longitude                       -       x
            Auxiliary coordinates:
                time                            x       -
            Cell methods:
                mean                        time (300 s)
                mean                        time_counter
            Attributes:
                Conventions                 UGRID
                description                 Created by xios
                interval_operation          300 s
                interval_write              1 d
                name                        lfric_surface
                online_operation            average
                timeStamp                   2020-Feb-07 16:23:14 GMT
                title                       Created by xios
                uuid                        489bcef5-3d1c-4529-be42-4ab5f8c8497b

        # Load the target grid.
        >>> sample_grid = load_cube(grid_file)
        >>> print(sample_grid)
        sample_grid / (unknown)             (latitude: 180; longitude: 360)
            Dimension coordinates:
                latitude                             x               -
                longitude                            -               x
            Attributes:
                Conventions                 'CF-1.7'

        # Initialise the regridder.
        >>> rg = MeshToGridESMFRegridder(mesh_cube1, sample_grid)

        # Regrid the mesh cube cube.
        >>> result1 = rg(mesh_cube1)
        >>> print(result1)
        sea_surface_temperature / (K)       (-- : 1; latitude: 180; longitude: 360)
            Dimension coordinates:
                latitude                        -            x               -
                longitude                       -            -               x
            Auxiliary coordinates:
                time                            x            -               -
            Cell methods:
                mean                        time (300 s)
                mean                        time_counter
            Attributes:
                Conventions                 UGRID
                description                 Created by xios
                interval_operation          300 s
                interval_write              1 d
                name                        lfric_surface
                online_operation            average
                timeStamp                   2020-Feb-07 16:23:14 GMT
                title                       Created by xios
                uuid                        489bcef5-3d1c-4529-be42-4ab5f8c8497b

.. note::

    **All** :class:`~iris.cube.Cube` :attr:`~iris.cube.Cube.attributes` are
    retained when regridding, so watch out for any attributes that reference
    the format (there are several in these examples) - you may want to manually
    remove them to avoid later confusion.

The initialisation process is computationally expensive so we use caching to
improve performance. Once a regridder has been initialised, it can be used on
any :class:`~iris.cube.Cube` which has been defined on the same
:class:`~iris.experimental.ugrid.Mesh` (or on the same **grid** in the case of
:class:`~esmf_regrid.experimental.unstructured_scheme.GridToMeshESMFRegridder`).
Since calling a regridder is usually a lot faster than initialising, reusing
regridders can save a lot of time. We can demonstrate the reuse of the
previously initialised regridder:

.. dropdown:: :opticon:`code`

    .. code-block:: python

        # Extract a different cube defined on te same Mesh.
        >>> mesh_cube2 = mesh_cubes.extract_cube("precipitation_flux")
        >>> print(mesh_cube2)
        precipitation_flux / (kg m-2 s-1)   (-- : 1; -- : 13824)
            Mesh coordinates:
                latitude                        -       x
                longitude                       -       x
            Auxiliary coordinates:
                time                            x       -
            Cell methods:
                mean                        time (300 s)
                mean                        time_counter
            Attributes:
                Conventions                 UGRID
                description                 Created by xios
                interval_operation          300 s
                interval_write              1 d
                name                        lfric_surface
                online_operation            average
                timeStamp                   2020-Feb-07 16:23:14 GMT
                title                       Created by xios
                uuid                        489bcef5-3d1c-4529-be42-4ab5f8c8497b

        # Regrid the new mesh cube using the same regridder.
        >>> result2 = rg(mesh_cube2)
        >>> print(result2)
        precipitation_flux / (kg m-2 s-1)   (-- : 1; latitude: 180; longitude: 360)
            Dimension coordinates:
                latitude                        -            x               -
                longitude                       -            -               x
            Auxiliary coordinates:
                time                            x            -               -
            Cell methods:
                mean                        time (300 s)
                mean                        time_counter
            Attributes:
                Conventions                 UGRID
                description                 Created by xios
                interval_operation          300 s
                interval_write              1 d
                name                        lfric_surface
                online_operation            average
                timeStamp                   2020-Feb-07 16:23:14 GMT
                title                       Created by xios
                uuid                        489bcef5-3d1c-4529-be42-4ab5f8c8497b

Support also exists for saving and loading previously initialised regridders -
:func:`esmf_regrid.experimental.io.save_regridder` and
:func:`~esmf_regrid.experimental.io.load_regridder` - so that they can be
re-used by future scripts.

Equality
--------
.. |tagline: equality| replace:: |unchanged|

.. rubric:: |tagline: equality|

:class:`~iris.experimental.ugrid.Mesh` comparison is supported, and comparing
two ':class:`~iris.experimental.ugrid.Mesh`-:class:`~iris.cube.Cube`\s' will
include a comparison of the respective
:class:`~iris.experimental.ugrid.Mesh`\es, with no extra action needed by the
user.

.. note::

    Keep an eye on memory demand when comparing large
    :class:`~iris.experimental.ugrid.Mesh`\es, but note that
    :class:`~iris.experimental.ugrid.Mesh`\ equality is enabled for lazy
    processing (:doc:`/userguide/real_and_lazy_data`), so if the
    :class:`~iris.experimental.ugrid.Mesh`\es being compared are lazy the
    process will use less memory than their total size.

Combining Cubes
---------------
.. |tagline: combining cubes| replace:: |pending|

.. rubric:: |tagline: combining cubes|

Merging or concatenating :class:`~iris.cube.Cube`\s (described in
:doc:`/userguide/merge_and_concat`) with two different
:class:`~iris.experimental.ugrid.Mesh`\es is not possible - a
:class:`~iris.cube.Cube` must be associated with just a single
:class:`~iris.experimental.ugrid.Mesh`, and merge/concatenate are not yet
capable of combining multiple :class:`~iris.experimental.ugrid.Mesh`\es into
one.

:class:`~iris.cube.Cube`\s that include
:class:`~iris.experimental.ugrid.MeshCoord`\s can still be merged/concatenated
on dimensions other than the :meth:`~iris.cube.Cube.mesh_dim`, since such
:class:`~iris.cube.Cube`\s will by definition share the same
:class:`~iris.experimental.ugrid.Mesh`.

.. seealso::

    You may wish to investigate
    :func:`iris.experimental.ugrid.recombine_submeshes`, which can be used
    for a very specific type of :class:`~iris.experimental.ugrid.Mesh`
    combination not detailed here.

Arithmetic
----------
.. |tagline: arithmetic| replace:: |pending|

.. rubric:: |tagline: arithmetic|

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