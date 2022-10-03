# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

"""
Extensions to Iris' NetCDF saving to allow
:class:`~iris.experimental.ugrid.mesh.Mesh` saving in UGRID format.

Eventual destination: :mod:`iris.fileformats.netcdf`.

"""
from collections.abc import Iterable

from ...fileformats import netcdf


def save_mesh(mesh, filename, netcdf_format="NETCDF4"):
    """
    Save mesh(es) to a netCDF file.

    Args:

    * mesh (:class:`iris.experimental.ugrid.Mesh` or iterable):
        mesh(es) to save.

    * filename (string):
        Name of the netCDF file to create.

    Kwargs:

    * netcdf_format (string):
        Underlying netCDF file format, one of 'NETCDF4', 'NETCDF4_CLASSIC',
        'NETCDF3_CLASSIC' or 'NETCDF3_64BIT'. Default is 'NETCDF4' format.

    """
    # TODO: integrate with standard saving API when no longer 'experimental'.

    if isinstance(mesh, Iterable):
        meshes = mesh
    else:
        meshes = [mesh]

    # Initialise Manager for saving
    with netcdf.Saver(filename, netcdf_format) as sman:
        # Iterate through the list.
        for mesh in meshes:
            # Get suitable dimension names.
            mesh_dimensions, _ = sman._get_dim_names(mesh)

            # Create dimensions.
            sman._create_cf_dimensions(
                cube=None, dimension_names=mesh_dimensions
            )

            # Create the mesh components.
            sman._add_mesh(mesh)

        # Add a conventions attribute.
        # TODO: add 'UGRID' to conventions, when this is agreed with CF ?
        sman.update_global_attributes(
            Conventions=netcdf.CF_CONVENTIONS_VERSION
        )
