# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Experimental module for using some GeoVista operations with Iris cubes."""

from geovista import Transform
from geovista.common import VTK_CELL_IDS, VTK_POINT_IDS

from iris.exceptions import CoordinateNotFoundError
from iris.experimental.ugrid import Mesh


def _get_coord(cube, axis):
    """
    Helper function to get the coordinates from the cube.

    """
    try:
        coord = cube.coord(axis=axis, dim_coords=True)
    except CoordinateNotFoundError:
        coord = cube.coord(axis=axis)
    return coord


def cube_faces_to_polydata(cube, **kwargs):
    """
    Function to convert a cube or the mesh attached to a cube into a polydata
    object that can be used by GeoVista to generate plots.

    Parameters
    ----------
    cube : :class:`~iris.cube.Cube`
        Incoming cube containing the arrays or the mesh to be converted into the
        polydata object.

    **kwargs : dict
        Additional keyword arguments to be passed to the Transform method.

    """
    if cube.mesh:
        if cube.ndim != 1:
            raise NotImplementedError(
                "Cubes with a mesh must be one dimensional"
            )
        lons, lats = cube.mesh.node_coords
        face_node = cube.mesh.face_node_connectivity
        indices = face_node.indices_by_location()

        polydata = Transform.from_unstructured(
            xs=lons.points,
            ys=lats.points,
            connectivity=indices,
            data=cube.data,
            name=f"{cube.name()} / {cube.units}",
            start_index=face_node.start_index,
            **kwargs,
        )
    elif cube.ndim == 2:
        x_coord = _get_coord(cube, "X")
        y_coord = _get_coord(cube, "Y")
        transform_kwargs = dict(
            xs=x_coord.contiguous_bounds(),
            ys=y_coord.contiguous_bounds(),
            data=cube.data,
            name=f"{cube.name()} / {cube.units}",
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
            raise NotImplementedError(
                "Only 1D and 2D coordinates are supported"
            )
    else:
        raise NotImplementedError("Cube must have a mesh or have 2 dimensions")

    return polydata


def region_extraction(cube, polydata, region, **kwargs):
    """
    Function to extract a region from a cube and its associated mesh and return
    a new cube containing the region.

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
            raise NotImplementedError("Must be on face or node.")

        if cube.shape[mesh_dim] != polydata_length:
            raise ValueError(
                "The mesh on the cube and the polydata must have the"
                " same shape."
            )

        region_polydata = region.enclosed(polydata, **kwargs)
        indices = region_polydata[indices_key]
        if len(indices) == 0:
            raise IndexError("No part of `polydata` falls within `region`.")

        my_tuple = tuple(
            [
                slice(None) if i != mesh_dim else indices
                for i in range(cube.ndim)
            ]
        )

        region_cube = cube[my_tuple]

        if recreate_mesh:
            coords_on_mesh_dim = region_cube.coords(dimensions=mesh_dim)
            new_mesh = Mesh.from_coords(
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
