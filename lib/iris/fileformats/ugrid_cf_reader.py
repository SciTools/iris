# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Adds a UGRID extension layer to netCDF file loading.

"""
from collections import namedtuple
import os

import netCDF4

from gridded.pyugrid.ugrid import UGrid
from gridded.pyugrid.read_netcdf import (
    find_mesh_names,
    load_grid_from_nc_dataset,
)
from iris.fileformats.cf import CFReader


_UGRID_ELEMENT_TYPE_NAMES = ("node", "edge", "face", "volume")

# Generate all possible UGRID structural property names.
# These are the UGRID mesh properties that contain variable names for linkage,
# which may appear as recognised properties of the main mesh variable.

# Start with coordinate variables for each element type (aka "mesh_location").
_UGRID_LINK_PROPERTIES = [
    "{}_coordinates".format(elem) for elem in _UGRID_ELEMENT_TYPE_NAMES
]

# Add in all possible type-to-type_connectivity elements.
# NOTE: this actually generates extra unused names, such as
# "node_face_connectivity", because we are not bothering to distinguish
# between lower- and higher-order elements.
# For now just don't worry about that, as long as we get all the ones which
# *are* needed.
_UGRID_LINK_PROPERTIES += [
    "{}_{}_connectivity".format(e1, e2)
    for e1 in _UGRID_ELEMENT_TYPE_NAMES
    for e2 in _UGRID_ELEMENT_TYPE_NAMES
]


class CubeUgrid(
    namedtuple("CubeUgrid", ["cube_dim", "grid", "mesh_location"])
):
    """
    Object recording the unstructured grid dimension of a cube.

    * cube_dim (int):
        The cube dimension which maps the unstructured grid.
        There can be only one.

    * grid (`gridded.pyugrid.UGrid`):
        A 'gridded' description of a UGRID mesh.

    * mesh_location (str):
        Which element of the mesh the cube is mapped to.
        Can be 'face', 'edge' or 'node'.  A 'volume' is not supported.

    """

    def __str__(self):
        result = "Cube unstructured-grid dimension:"
        result += "\n   cube dimension = {}".format(self.cube_dim)
        result += '\n   mesh_location = "{}"'.format(self.mesh_location)
        result += '\n   mesh "{}" :\n'.format(self.grid.mesh_name)
        try:
            mesh_str = str(self.grid.info)
        except TypeError:
            mesh_str = "<unprintable mesh>"
        result += "\n".join(["     " + line for line in mesh_str.split("\n")])
        result += "\n"
        return result


class UGridCFReader:
    """
    A CFReader extension to add UGRID information to netcdf cube loading.

    Identifies UGRID-specific parts of a netcdf file, providing:

    * `self.cfreader` : a CFReader object to interpret the CF data from the
      file for cube creation, while ignoring the UGRID mesh data.

    * `self.complete_ugrid_cube(cube)` a call to add the relevant UGRID
      information to a cube created from the cfreader data.

    This allows us to decouple UGRID from CF support with minimal changes to
    the existing `iris.fileformats.netcdf` code, which is intimately coupled to
    both the CFReader class and the netCDF4 file interface.

    """

    def __init__(self, filename, *args, **kwargs):
        self.filename = os.path.expanduser(filename)
        dataset = netCDF4.Dataset(self.filename, mode="r")
        self.dataset = dataset
        meshes = {}
        for meshname in find_mesh_names(self.dataset):
            mesh = UGrid()
            load_grid_from_nc_dataset(dataset, mesh, mesh_name=meshname)
            meshes[meshname] = mesh
        self.meshes = meshes

        # Generate list of excluded variable names.
        exclude_vars = list(meshes.keys())

        temp_xios_fix = kwargs.pop("temp_xios_fix", False)
        if not temp_xios_fix:
            # This way *ought* to work, but maybe problems with the test file ?
            for mesh in meshes.values():
                mesh_var = dataset.variables[mesh.mesh_name]
                for attr in mesh_var.ncattrs():
                    if attr in _UGRID_LINK_PROPERTIES:
                        exclude_vars.extend(mesh_var.getncattr(attr).split())
        else:
            # A crude and XIOS-specific alternative ..
            exclude_vars += [
                name
                for name in dataset.variables.keys()
                if any(name.startswith(meshname) for meshname in meshes.keys())
            ]

        # Identify possible mesh dimensions and make a map of them.
        meshdims_map = {}  # Maps {dimension-name: (mesh, mesh-location)}
        for mesh in meshes.values():
            mesh_var = dataset.variables[mesh.mesh_name]
            if mesh.faces is not None:
                # Work out name of faces dimension and record it.
                faces_varname = mesh_var.face_node_connectivity
                faces_var = dataset.variables[faces_varname]
                if "face_dimension" in faces_var.ncattrs():
                    faces_dim_index = faces_var.getncattr("face_dimension")
                else:
                    faces_dim_index = 0
                faces_dim_name = faces_var.dimensions[faces_dim_index]
                meshdims_map[faces_dim_name] = (mesh, "face")
            if mesh.edges is not None:
                # Work out name of edges dimension and record it.
                edges_varname = mesh_var.edge_node_connectivity
                edges_var = dataset.variables[edges_varname]
                if "edge_dimension" in edges_var.ncattrs():
                    edges_dim_index = faces_var.getncattr("edge_dimension")
                else:
                    edges_dim_index = 0
                edges_dim_name = edges_var.dimensions[edges_dim_index]
                meshdims_map[edges_dim_name] = (mesh, "edge")
            if mesh.nodes is not None:
                # Work out name of nodes dimension and record it.
                nodes_varname = mesh_var.node_coordinates.split()[0]
                nodes_var = dataset.variables[nodes_varname]
                nodes_dim_name = nodes_var.dimensions[0]
                meshdims_map[nodes_dim_name] = (mesh, "node")
        self.meshdims_map = meshdims_map

        # Create a CFReader object which skips the UGRID-related variables.
        kwargs["exclude_var_names"] = exclude_vars
        self.cfreader = CFReader(self.dataset, *args, **kwargs)

    def complete_ugrid_cube(self, cube):
        """
        Add the ".ugrid" property to a cube loaded with the `self.cfreader`.

        We identify the unstructured-grid dimension of the cube (if any), and
        attach a suitable CubeUgrid object, linking the cube mesh dimension to
        an element-type (aka "mesh_location") of a mesh.

        """
        # Set a 'cube.ugrid' property.
        data_var = self.dataset.variables[cube.var_name]
        meshes_info = [
            (i_dim, self.meshdims_map.get(dim_name))
            for i_dim, dim_name in enumerate(data_var.dimensions)
            if dim_name in self.meshdims_map
        ]
        if len(meshes_info) > 1:
            msg = "Cube maps more than one mesh dimension: {}"
            raise ValueError(msg.format(meshes_info))
        if meshes_info:
            i_dim, (mesh, mesh_location) = meshes_info[0]
            cube.ugrid = CubeUgrid(
                cube_dim=i_dim, grid=mesh, mesh_location=mesh_location
            )
        else:
            # Add an empty 'cube.ugrid' to all cubes otherwise.
            cube.ugrid = None
        return

    def __del__(self):
        # Explicitly close dataset to prevent file remaining open.
        self.dataset.close()
