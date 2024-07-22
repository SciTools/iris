# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Experimental module for using some GeoVista operations with Iris cubes."""

from geovista import Transform
from geovista.common import VTK_CELL_IDS, VTK_POINT_IDS

from iris.exceptions import CoordinateNotFoundError
from iris.experimental.ugrid import MeshXY


def _get_coord(cube, axis):
    """Get the axis coordinates from the cube."""
    try:
        coord = cube.coord(axis=axis, dim_coords=True)
    except CoordinateNotFoundError:
        coord = cube.coord(axis=axis)
    return coord


def cube_to_polydata(cube, **kwargs):
    r"""Create a :class:`pyvista.PolyData` object from a :class:`~iris.cube.Cube`.

    The resulting :class:`~pyvista.PolyData` object can be plotted using
    a :class:`geovista.geoplotter.GeoPlotter`.

    Uses :class:`geovista.bridge.Transform` to parse the cube's information - one
    of: :meth:`~geovista.bridge.Transform.from_1d` /
    :meth:`~geovista.bridge.Transform.from_2d` /
    :meth:`~geovista.bridge.Transform.from_unstructured`.

    Parameters
    ----------
    cube : :class:`~iris.cube.Cube`
        The Cube containing the spatial information and data for creating the
        class:`~pyvista.PolyData`.

    **kwargs : dict, optional
        Additional keyword arguments to be passed to the relevant
        :class:`~geovista.bridge.Transform` method (e.g ``zlevel``).

    Returns
    -------
    :class:`~pyvista.PolyData`
        The PolyData object representing the cube's spatial information and data.

    Raises
    ------
    NotImplementedError
        If a :class:`~iris.cube.Cube` with too many dimensions is passed. Only
        the horizontal data can be represented, meaning a 2D Cube, or 1D Cube
        if the horizontal space is described by
        :class:`~iris.experimental.ugrid.MeshCoord`\ s.

    Examples
    --------
    .. testsetup::

        from iris import load_cube, sample_data_path
        from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD

        cube = load_cube(sample_data_path("air_temp.pp"))
        cube_w_time = load_cube(sample_data_path("A1B_north_america.nc"))
        with PARSE_UGRID_ON_LOAD.context():
            cube_mesh = load_cube(sample_data_path("mesh_C4_synthetic_float.nc"))

    >>> from iris.experimental.geovista import cube_to_polydata

    Converting a standard 2-dimensional :class:`~iris.cube.Cube` with
    1-dimensional coordinates:

    >>> print(cube.summary(shorten=True))
    air_temperature / (K)               (latitude: 73; longitude: 96)
    >>> print(cube_to_polydata(cube))
    PolyData (...
      N Cells:    7008
      N Points:   7178
      N Strips:   0
      X Bounds:   -9.992e-01, 9.992e-01
      Y Bounds:   -9.992e-01, 9.992e-01
      Z Bounds:   -1.000e+00, 1.000e+00
      N Arrays:   4

    Configure the conversion by passing additional keyword arguments:

    >>> print(cube_to_polydata(cube, radius=2))
    PolyData (...
      N Cells:    7008
      N Points:   7178
      N Strips:   0
      X Bounds:   -1.998e+00, 1.998e+00
      Y Bounds:   -1.998e+00, 1.998e+00
      Z Bounds:   -2.000e+00, 2.000e+00
      N Arrays:   4

    Converting a :class:`~iris.cube.Cube` that has a
    :attr:`~iris.cube.Cube.mesh` describing its horizontal space:

    >>> print(cube_mesh.summary(shorten=True))
    synthetic / (1)                     (-- : 96)
    >>> print(cube_to_polydata(cube_mesh))
    PolyData (...
      N Cells:    96
      N Points:   98
      N Strips:   0
      X Bounds:   -1.000e+00, 1.000e+00
      Y Bounds:   -1.000e+00, 1.000e+00
      Z Bounds:   -1.000e+00, 1.000e+00
      N Arrays:   4

    Remember to reduce the dimensionality of your :class:`~iris.cube.Cube` to
    just be the horizontal space:

    >>> print(cube_w_time.summary(shorten=True))
    air_temperature / (K)               (time: 240; latitude: 37; longitude: 49)
    >>> print(cube_to_polydata(cube_w_time[0, :, :]))
    PolyData (...
      N Cells:    1813
      N Points:   1900
      N Strips:   0
      X Bounds:   -6.961e-01, 6.961e-01
      Y Bounds:   -9.686e-01, -3.411e-01
      Z Bounds:   2.483e-01, 8.714e-01
      N Arrays:   4

    """
    if cube.mesh:
        if cube.ndim != 1:
            raise NotImplementedError("Cubes with a mesh must be one dimensional")
        lons, lats = cube.mesh.node_coords
        face_node = cube.mesh.face_node_connectivity
        indices = face_node.indices_by_location()

        polydata = Transform.from_unstructured(
            xs=lons.points,
            ys=lats.points,
            connectivity=indices,
            data=cube.data,
            name=f"{cube.name()} / ({cube.units})",
            start_index=face_node.start_index,
            **kwargs,
        )
    # TODO: Add support for point clouds
    elif cube.ndim == 2:
        x_coord = _get_coord(cube, "X")
        y_coord = _get_coord(cube, "Y")
        transform_kwargs = dict(
            xs=x_coord.contiguous_bounds(),
            ys=y_coord.contiguous_bounds(),
            data=cube.data,
            name=f"{cube.name()} / ({cube.units})",
            **kwargs,
        )
        coord_system = cube.coord_system()
        if coord_system:
            transform_kwargs["crs"] = coord_system.as_cartopy_crs().proj4_init

        if x_coord.ndim == 2 and y_coord.ndim == 2:
            polydata = Transform.from_2d(**transform_kwargs)

        elif x_coord.ndim == 1 and y_coord.ndim == 1:
            polydata = Transform.from_1d(**transform_kwargs)

        else:
            raise NotImplementedError("Only 1D and 2D coordinates are supported")
    else:
        raise NotImplementedError("Cube must have a mesh or have 2 dimensions")

    return polydata


def extract_unstructured_region(cube, polydata, region, **kwargs):
    """Index a :class:`~iris.cube.Cube` with a :attr:`~iris.cube.Cube.mesh` to a specific region.

    Uses :meth:`geovista.geodesic.BBox.enclosed` to identify the `cube` indices
    that are within the specified region (`region` being a
    :class:`~geovista.geodesic.BBox` class).

    Parameters
    ----------
    cube : :class:`~iris.cube.Cube`
        The cube to be indexed (must have a :attr:`~iris.cube.Cube.mesh`).
    polydata : :class:`pyvista.PolyData`
        A :class:`~pyvista.PolyData` representing the same horizontal space as
        `cube`. The region extraction is first applied to `polydata`, with the
        resulting indices then applied to `cube`. In many cases `polydata` can
        be created by applying :func:`cube_to_polydata` to `cube`.
    region : :class:`geovista.geodesic.BBox`
        A :class:`~geovista.geodesic.BBox` representing the region to be
        extracted.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the
        :meth:`geovista.geodesic.BBox.enclosed` method (e.g ``preference``).

    Returns
    -------
    :class:`~iris.cube.Cube`
        The region extracted cube.

    Raises
    ------
    ValueError
        If `polydata` and the :attr:`~iris.cube.Cube.mesh` on `cube` do not
        have the same shape.

    Examples
    --------
    .. testsetup::

        from iris import load_cube, sample_data_path
        from iris.coords import AuxCoord
        from iris.cube import CubeList
        from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD

        file_path = sample_data_path("mesh_C4_synthetic_float.nc")
        with PARSE_UGRID_ON_LOAD.context():
            cube_w_mesh = load_cube(file_path)

        level_cubes = CubeList()
        for height_level in range(72):
            height_coord = AuxCoord([height_level], standard_name="height")
            level_cube = cube_w_mesh.copy()
            level_cube.add_aux_coord(height_coord)
            level_cubes.append(level_cube)

        cube_w_mesh = level_cubes.merge_cube()
        other_cube_w_mesh = cube_w_mesh[:20, :]

    The parameters of :func:`extract_unstructured_region` have been designed with
    flexibility and reuse in mind. This is demonstrated below.

    >>> from geovista.geodesic import BBox
    >>> from iris.experimental.geovista import cube_to_polydata, extract_unstructured_region
    >>> print(cube_w_mesh.shape)
    (72, 96)
    >>> # The mesh dimension represents the horizontal space of the cube.
    >>> print(cube_w_mesh.shape[cube_w_mesh.mesh_dim()])
    96
    >>> cube_polydata = cube_to_polydata(cube_w_mesh[0, :])
    >>> extracted_cube = extract_unstructured_region(
    ...     cube=cube_w_mesh,
    ...     polydata=cube_polydata,
    ...     region=BBox(lons=[0, 70, 70, 0], lats=[-25, -25, 45, 45]),
    ... )
    >>> print(extracted_cube.shape)
    (72, 11)

    Now reuse the same `cube` and `polydata` to extract a different region:

    >>> new_region = BBox(lons=[0, 35, 35, 0], lats=[-25, -25, 45, 45])
    >>> extracted_cube = extract_unstructured_region(
    ...     cube=cube_w_mesh,
    ...     polydata=cube_polydata,
    ...     region=new_region,
    ... )
    >>> print(extracted_cube.shape)
    (72, 6)

    Now apply the same region extraction to a different `cube` that has the
    same horizontal shape:

    >>> print(other_cube_w_mesh.shape)
    (20, 96)
    >>> extracted_cube = extract_unstructured_region(
    ...     cube=other_cube_w_mesh,
    ...     polydata=cube_polydata,
    ...     region=new_region,
    ... )
    >>> print(extracted_cube.shape)
    (20, 6)

    Arbitrary keywords can be passed down to
    :meth:`geovista.geodesic.BBox.enclosed` (``outside`` in this example):

    >>> extracted_cube = extract_unstructured_region(
    ...     cube=other_cube_w_mesh,
    ...     polydata=cube_polydata,
    ...     region=new_region,
    ...     outside=True,
    ... )
    >>> print(extracted_cube.shape)
    (20, 90)

    """
    if cube.mesh:
        # Find what dimension the mesh is in on the cube
        mesh_dim = cube.mesh_dim()
        recreate_mesh = False

        if cube.location == "face":
            polydata_length = polydata.GetNumberOfCells()
            indices_key = VTK_CELL_IDS
            recreate_mesh = True
        elif cube.location == "node":
            polydata_length = polydata.GetNumberOfPoints()
            indices_key = VTK_POINT_IDS
        else:
            raise NotImplementedError(
                f"cube.location must be `face` or `node`. Found: {cube.location}."
            )

        if cube.shape[mesh_dim] != polydata_length:
            raise ValueError(
                f"The mesh on the cube and the polydata"
                f"must have the same shape."
                f" Found Mesh: {cube.shape[mesh_dim]},"
                f" Polydata: {polydata_length}."
            )

        region_polydata = region.enclosed(polydata, **kwargs)
        indices = region_polydata[indices_key]
        if len(indices) == 0:
            raise IndexError("No part of `polydata` falls within `region`.")

        my_tuple = tuple(
            [slice(None) if i != mesh_dim else indices for i in range(cube.ndim)]
        )

        region_cube = cube[my_tuple]

        if recreate_mesh:
            coords_on_mesh_dim = region_cube.coords(dimensions=mesh_dim)
            new_mesh = MeshXY.from_coords(
                *[c for c in coords_on_mesh_dim if c.has_bounds()]
            )

            new_mesh_coords = new_mesh.to_MeshCoords(cube.location)

            for coord in new_mesh_coords:
                region_cube.remove_coord(coord.name())
                region_cube.add_aux_coord(coord, mesh_dim)

    # TODO: Support unstructured point based data without a mesh
    else:
        raise ValueError("Cube must have a mesh")

    return region_cube
