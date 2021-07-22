# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Routines for generating synthetic NetCDF files from template headers."""

from pathlib import Path
from string import Template
import subprocess

import netCDF4
import numpy as np


def _file_from_cdl_template(
    temp_file_dir, dataset_name, dataset_type, template_subs
):
    """Shared template filling behaviour.

    Substitutes placeholders in the appropriate CDL template, saves to a
    NetCDF file.

    """
    nc_write_path = (
        Path(temp_file_dir).joinpath(dataset_name).with_suffix(".nc")
    )
    # Fetch the specified CDL template type.
    templates_dir = Path(__file__).parent / "file_headers"
    template_filepath = templates_dir.joinpath(dataset_type).with_suffix(
        ".cdl"
    )
    # Substitute placeholders.
    with open(template_filepath) as file:
        template_string = Template(file.read())
    cdl = template_string.substitute(template_subs)

    # Spawn an "ncgen" command to create an actual NetCDF file from the
    # CDL string.
    subprocess.run(
        ["ncgen", "-o" + str(nc_write_path)],
        input=cdl,
        encoding="ascii",
        check=True,
    )

    return nc_write_path


def _add_standard_data(nc_path, unlimited_dim_size=0):
    """Shared data populating behaviour.

    Adds placeholder data to the variables in a NetCDF file, accounting for
    dimension size, 'dimension coordinates' and a possible unlimited dimension.

    """

    ds = netCDF4.Dataset(nc_path, "r+")

    unlimited_dim_names = [
        dim for dim in ds.dimensions if ds.dimensions[dim].size == 0
    ]
    # Data addition dependent on this assumption:
    assert len(unlimited_dim_names) < 2

    # Fill variables data with placeholder numbers.
    for var in ds.variables.values():
        shape = list(var.shape)
        dims = var.dimensions
        # Fill the unlimited dimension with the input size.
        shape = [
            unlimited_dim_size if dim == unlimited_dim_names[0] else size
            for dim, size in zip(dims, shape)
        ]
        data = np.zeros(shape, dtype=var.dtype)
        if len(var.dimensions) == 1 and var.dimensions[0] == var.name:
            # Fill the var with ascending values (not all zeroes),
            # so it can be a dim-coord.
            data = np.arange(data.size, dtype=data.dtype).reshape(data.shape)
        var[:] = data

    ds.close()


def create_file__xios_2d_face_half_levels(
    temp_file_dir, dataset_name, n_faces=866, n_times=1
):
    """Create a synthetic NetCDF file with XIOS-like content.

    Starts from a template CDL headers string, modifies to the input
    dimensions then adds data of the correct size.

    Parameters
    ----------
    temp_file_dir : str or pathlib.Path
        The directory in which to place the created file.
    dataset_name : str
        The name for the NetCDF dataset and also the created file.
    n_faces, n_times: int
        Dimension sizes for the dataset.

    Returns
    -------
    str
        Path of the created NetCDF file.

    """
    dataset_type = "xios_2D_face_half_levels"

    # Set the placeholder substitutions.
    template_subs = {
        "DATASET_NAME": dataset_name,
        "NUM_NODES": n_faces + 2,
        "NUM_FACES": n_faces,
    }

    # Create a NetCDF file based on the dataset type template and substitutions.
    nc_path = _file_from_cdl_template(
        temp_file_dir, dataset_name, dataset_type, template_subs
    )

    # Populate with the standard set of data, sized correctly.
    _add_standard_data(nc_path, unlimited_dim_size=n_times)

    return str(nc_path)


def create_file__xios_3d_face_half_levels(
    temp_file_dir, dataset_name, n_faces=866, n_times=1, n_levels=38
):
    """Create a synthetic NetCDF file with XIOS-like content.

    Starts from a template CDL headers string, modifies to the input
    dimensions then adds data of the correct size.

    Parameters
    ----------
    temp_file_dir : str or pathlib.Path
        The directory in which to place the created file.
    dataset_name : str
        The name for the NetCDF dataset and also the created file.
    n_faces, n_times, n_levels: int
        Dimension sizes for the dataset.

    Returns
    -------
    str
        Path of the created NetCDF file.

    """
    dataset_type = "xios_3D_face_half_levels"

    # Set the placeholder substitutions.
    template_subs = {
        "DATASET_NAME": dataset_name,
        "NUM_NODES": n_faces + 2,
        "NUM_FACES": n_faces,
        "NUM_LEVELS": n_levels,
    }

    # Create a NetCDF file based on the dataset type template and
    # substitutions.
    nc_path = _file_from_cdl_template(
        temp_file_dir, dataset_name, dataset_type, template_subs
    )

    # Populate with the standard set of data, sized correctly.
    _add_standard_data(nc_path, unlimited_dim_size=n_times)

    return str(nc_path)


def create_file__xios_3d_face_full_levels(
    temp_file_dir, dataset_name, n_faces=866, n_times=1, n_levels=39
):
    """Create a synthetic NetCDF file with XIOS-like content.

    Starts from a template CDL headers string, modifies to the input
    dimensions then adds data of the correct size.

    Parameters
    ----------
    temp_file_dir : str or pathlib.Path
        The directory in which to place the created file.
    dataset_name : str
        The name for the NetCDF dataset and also the created file.
    n_faces, n_times, n_levels: int
        Dimension sizes for the dataset.

    Returns
    -------
    str
        Path of the created NetCDF file.

    """
    dataset_type = "xios_3D_face_full_levels"

    # Set the placeholder substitutions.
    template_subs = {
        "DATASET_NAME": dataset_name,
        "NUM_NODES": n_faces + 2,
        "NUM_FACES": n_faces,
        "NUM_LEVELS": n_levels,
    }

    # Create a NetCDF file based on the dataset type template and
    # substitutions.
    nc_path = _file_from_cdl_template(
        temp_file_dir, dataset_name, dataset_type, template_subs
    )

    # Populate with the standard set of data, sized correctly.
    _add_standard_data(nc_path, unlimited_dim_size=n_times)

    return str(nc_path)
