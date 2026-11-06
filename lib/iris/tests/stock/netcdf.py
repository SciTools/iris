# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Routines for generating synthetic NetCDF files from template headers."""

from pathlib import Path
from string import Template
import subprocess
from typing import Any, Optional

import dask
from dask import array as da
import numpy as np

from iris.fileformats.netcdf import _thread_safe_nc
from iris.tests import env_bin_path

NCGEN_PATHSTR = str(env_bin_path("ncgen"))


def ncgen_from_cdl(cdl_str: Optional[str], cdl_path: Optional[str], nc_path: str):
    """Generate a test netcdf file from cdl.

    Source is CDL in either a string or a file.
    If given a string, will either save a CDL file, or pass text directly.
    A netcdf output file is always created, at the given path.

    Parameters
    ----------
    cdl_str : str or None
        String containing a CDL description of a netcdf file.
        If None, 'cdl_path' must be an existing file.
    cdl_path : str or None
        Path of temporary text file where cdl_str is written.
        If None, 'cdl_str' must be given, and is piped direct to ncgen.
    nc_path : str
        Path of temporary netcdf file where converted result is put.

    Notes
    -----
    For legacy reasons, the path args are 'str's not 'Path's.

    """
    if cdl_str and cdl_path:
        with open(cdl_path, "w") as f_out:
            f_out.write(cdl_str)
    if cdl_path:
        # Create netcdf from stored CDL file.
        call_args = [NCGEN_PATHSTR, "-k3", "-o", nc_path, cdl_path]
        call_kwargs: dict[str, Any] = {}
    else:
        # No CDL file : pipe 'cdl_str' directly into the ncgen program.
        if not cdl_str:
            raise ValueError("Must provide either 'cdl_str' or 'cdl_path'.")
        call_args = [NCGEN_PATHSTR, "-k3", "-o", nc_path]
        call_kwargs = dict(input=cdl_str, encoding="ascii")

    subprocess.run(call_args, check=True, **call_kwargs)


def _file_from_cdl_template(temp_file_dir, dataset_name, dataset_type, template_subs):
    """Shared template filling behaviour.

    Substitutes placeholders in the appropriate CDL template, saves to a
    NetCDF file.

    """
    nc_write_path = Path(temp_file_dir).joinpath(dataset_name).with_suffix(".nc")
    # Fetch the specified CDL template type.
    templates_dir = Path(__file__).parent / "file_headers"
    template_filepath = templates_dir.joinpath(dataset_type).with_suffix(".cdl")
    # Substitute placeholders.
    with open(template_filepath) as file:
        template_string = Template(file.read())
    cdl = template_string.substitute(template_subs)

    # Spawn an "ncgen" command to create an actual NetCDF file from the
    # CDL string.
    ncgen_from_cdl(cdl_str=cdl, cdl_path=None, nc_path=nc_write_path)

    return nc_write_path


def _add_standard_data(nc_path, unlimited_dim_size=0):
    """Shared data populating behaviour.

    Adds placeholder data to the variables in a NetCDF file, accounting for
    dimension size, 'dimension coordinates' and a possible unlimited dimension.

    """
    ds = _thread_safe_nc.DatasetWrapper(nc_path, "r+")

    unlimited_dim_names = [
        dim for dim in ds.dimensions if ds.dimensions[dim].isunlimited()
    ]
    # Data addition dependent on this assumption:
    assert len(unlimited_dim_names) < 2
    if len(unlimited_dim_names) == 0:
        unlimited_dim_names = ["*unused*"]

    # Fill variables data with placeholder numbers.
    for var in ds.variables.values():
        shape = list(var.shape)
        dims = var.dimensions
        # Fill the unlimited dimension with the input size.
        shape = [
            unlimited_dim_size if dim == unlimited_dim_names[0] else size
            for dim, size in zip(dims, shape)
        ]
        if len(var.dimensions) == 1 and var.dimensions[0] == var.name:
            # Fill the var with ascending values (not all zeroes),
            # so it can be a dim-coord.
            data_size = np.prod(shape)
            data = np.arange(1, data_size + 1, dtype=var.dtype).reshape(shape)
            var[:] = data
        else:
            # Fill with a plain value.  But avoid zeros, so we can simulate
            # valid mesh connectivities even when start_index=1.
            with dask.config.set({"array.chunk-size": "2048MiB"}):
                data = da.ones(shape, dtype=var.dtype)  # Do not use zero
            da.store(data, var)

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
