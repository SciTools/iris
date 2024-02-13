from geovista import Transform
from geovista.common import VTK_CELL_IDS

from iris.exceptions import CoordinateNotFoundError
from iris.experimental.ugrid import Mesh


def _get_coord(cube, axis):
    try:
        coord = cube.coord(axis=axis, dim_coords=True)
    except CoordinateNotFoundError:
        coord = cube.coord(axis=axis)
    return coord


def cube_faces_to_polydata(cube):
    if cube.mesh:
        lons, lats = cube.mesh.node_coords
        face_node = cube.mesh.face_node_connectivity
        indices = face_node.indices_by_location()

        polydata = Transform.from_unstructured(
            lons.points,
            lats.points,
            indices,
            data=cube.data,
            name=f"{cube.name()} / {cube.units}",
            start_index=face_node.start_index,
        )
    elif cube.ndim == 2:
        x_coord = _get_coord(cube, "X")
        y_coord = _get_coord(cube, "Y")
        kwargs = dict(
            xs=x_coord.contiguous_bounds(),
            ys=y_coord.contiguous_bounds(),
            data=cube.data,
            name=f"{cube.name()} / {cube.units}",
        )
        coord_system = cube.coord_system()
        if coord_system:
            kwargs["crs"] = coord_system.as_cartopy_crs().proj4_init

        if x_coord.ndim == 2 and y_coord.ndim == 2:
            polydata = Transform.from_2d(**kwargs)

        elif x_coord.ndim == 1 and y_coord.ndim == 1:
            polydata = Transform.from_1d(**kwargs)

        else:
            raise NotImplementedError("Only 1D and 2D coordinates are supported")
    else:
        raise NotImplementedError("Cube must have a mesh or have 2 dimensions")

    return polydata


def region_extraction(cube, mesh, region, preference):
    region_polydata = region.enclosed(mesh, preference=preference)
    indices = region_polydata[VTK_CELL_IDS]

    region_cube = cube[:, indices]

    new_mesh = Mesh.from_coords(*region_cube.coords(dimensions=cube.mesh_dim()))
    new_mesh_coords = new_mesh.to_MeshCoords(cube.location)

    for coord in new_mesh_coords:
        region_cube.remove_coord(coord.name())
        region_cube.add_aux_coord(coord, 1)

    return region_cube
