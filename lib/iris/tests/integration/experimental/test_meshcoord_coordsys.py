# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for MeshCoord.coord_system() behaviour."""

import pytest

import iris
from iris.coord_systems import GeogCS
from iris.experimental.ugrid.load import PARSE_UGRID_ON_LOAD
from iris.tests.stock.netcdf import ncgen_from_cdl

TEST_CDL = """
netcdf mesh_test {
    dimensions:
        node = 3 ;
        face = 1 ;
        vertex = 3 ;
    variables:
        int mesh ;
            mesh:cf_role = "mesh_topology" ;
            mesh:topology_dimension = 2 ;
            mesh:node_coordinates = "node_x node_y" ;
            mesh:face_node_connectivity = "face_nodes" ;
        float node_x(node) ;
            node_x:standard_name = "longitude" ;
        float node_y(node) ;
            node_y:standard_name = "latitude" ;
        int face_nodes(face, vertex) ;
            face_nodes:cf_role = "face_node_connectivity" ;
            face_nodes:start_index = 0 ;
        float node_data(node) ;
            node_data:coordinates = "node_x node_y" ;
            node_data:location = "node" ;
            node_data:mesh = "mesh" ;
}
"""

BASIC_CRS_STR = """
        int crs ;
          crs:grid_mapping_name = "latitude_longitude" ;
          crs:semi_major_axis = 6371000.0 ;
"""


def find_i_line(lines, match):
    (i_line,) = [i for i, line in enumerate(lines) if match in line]
    return i_line


def make_file(nc_path, node_x_crs=False, node_y_crs=False):
    lines = TEST_CDL.split("\n")

    includes = ["node_x"] if node_x_crs else []
    includes += ["node_y"] if node_y_crs else []
    includes = " ".join(includes)
    if includes:
        # insert a grid-mapping
        i_line = find_i_line(lines, "variables:")
        lines[i_line + 1 : i_line + 1] = BASIC_CRS_STR.split("\n")

        i_line = find_i_line(lines, "float node_data(")
        ref_lines = [
            # NOTE: space to match the surrounding indent
            f'            node_data:grid_mapping = "crs: {includes}" ;',
            # NOTE: this is already present
            # f'node_data:coordinates = "{includes}" ;'
        ]
        lines[i_line + 1 : i_line + 1] = ref_lines

    cdl_str = "\n".join(lines)
    ncgen_from_cdl(cdl_str=cdl_str, cdl_path=None, nc_path=nc_path)


@pytest.mark.parametrize("cs_axes", ["--", "x-", "-y", "xy"])
def test_default_mesh_cs(tmp_path, cs_axes):
    """Test coord-systems of mesh cube and coords, if location coords have a crs."""
    nc_path = tmp_path / "test_temp.nc"
    do_x = "x" in cs_axes
    do_y = "y" in cs_axes
    make_file(nc_path, node_x_crs=do_x, node_y_crs=do_y)
    with PARSE_UGRID_ON_LOAD.context():
        cube = iris.load_cube(nc_path, "node_data")
    meshco_x, meshco_y = [cube.coord(mesh_coords=True, axis=ax) for ax in ("x", "y")]
    # NOTE: at present, none of these load with a coordinate system,
    #  because we don't support the extended grid-mapping syntax.
    #  see: https://github.com/SciTools/iris/issues/3388
    assert meshco_x.coord_system is None
    assert meshco_y.coord_system is None


def test_assigned_mesh_cs(tmp_path):
    # Check that when a coord system is manually assigned to a location coord,
    # the corresponding meshcoord reports the same cs.
    nc_path = tmp_path / "test_temp.nc"
    make_file(nc_path)
    with PARSE_UGRID_ON_LOAD.context():
        cube = iris.load_cube(nc_path, "node_data")
    nodeco_x = cube.mesh.coord(location="node", axis="x")
    meshco_x, meshco_y = [cube.coord(axis=ax) for ax in ("x", "y")]
    assert nodeco_x.coord_system is None
    assert meshco_x.coord_system is None
    assert meshco_y.coord_system is None
    assigned_cs = GeogCS(1.0)
    nodeco_x.coord_system = assigned_cs
    assert meshco_x.coord_system is assigned_cs
    assert meshco_y.coord_system is None
    # This also affects cube.coord_system(), even though it is an auxcoord,
    #  since there are no dim-coords, or any other coord with a c-s.
    # TODO: this may be a mistake -- see https://github.com/SciTools/iris/issues/6051
    assert cube.coord_system() is assigned_cs


def test_meshcoord_coordsys_copy(tmp_path):
    # Check that copying a meshcoord with a coord system works properly.
    nc_path = tmp_path / "test_temp.nc"
    make_file(nc_path)
    with PARSE_UGRID_ON_LOAD.context():
        cube = iris.load_cube(nc_path, "node_data")
    node_coord = cube.mesh.coord(location="node", axis="x")
    assigned_cs = GeogCS(1.0)
    node_coord.coord_system = assigned_cs
    mesh_coord = cube.coord(axis="x")
    assert mesh_coord.coord_system is assigned_cs
    meshco_copy = mesh_coord.copy()
    assert meshco_copy == mesh_coord
    # Note: still the same object, because it is derived from the same node_coord
    assert meshco_copy.coord_system is assigned_cs
