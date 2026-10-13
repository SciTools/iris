# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Builders for the small Zarr stores these tests ask questions of.

One canonical store, deliberately the same contents as the shared CFDataset
contract writes, so that a Zarr answer and a netCDF answer can be compared
without translating between two sets of fixtures.
"""

import numpy as np
import zarr

#: The canonical store's dimensions, in order.
DIMENSIONS = {"y": 3, "x": 2}
DATA = np.arange(6, dtype="f8").reshape(3, 2)
VARIABLE_ATTRS = {"units": "K", "long_name": "surface temperature"}
GLOBALS = {"Conventions": "CF-1.7", "title": "contract"}

#: The version 2 attribute carrying dimension names.
ARRAY_DIMENSIONS = "_ARRAY_DIMENSIONS"


def make_array(
    group,
    name,
    shape,
    dtype,
    zarr_format,
    *,
    dimensions=None,
    chunks="auto",
    shards=None,
    fill_value=None,
    attributes=None,
):
    """Create one array, recording dimension names the way its version does.

    zarr-python refuses `dimension_names` on a version 2 array, so version 2
    names are written to the `_ARRAY_DIMENSIONS` attribute by hand - which is
    what xarray and NCZarr both do.
    """
    extra = {}
    if zarr_format == 3 and dimensions is not None:
        extra["dimension_names"] = dimensions
    array = group.create_array(
        name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        shards=shards,
        fill_value=fill_value,
        **extra,
    )
    if zarr_format == 2 and dimensions is not None:
        array.attrs[ARRAY_DIMENSIONS] = list(dimensions)
    for key, value in (attributes or {}).items():
        array.attrs[key] = value
    return array


def make_store(path, *, zarr_format=3, chunks="auto", shards=None):
    """Write the canonical store at `path`, and return `path` as a string."""
    root = zarr.open_group(store=path, mode="w", zarr_format=zarr_format)
    for name, value in GLOBALS.items():
        root.attrs[name] = value

    array = make_array(
        root,
        "air_temperature",
        DATA.shape,
        "f8",
        zarr_format,
        dimensions=tuple(DIMENSIONS),
        chunks=chunks,
        shards=shards,
        fill_value=-1.0,
        attributes=VARIABLE_ATTRS,
    )
    array[:] = DATA

    # A dimensionless variable: what a grid mapping is, and the one shape a
    # CFDatasetVariable must handle without a leading dimension.
    scalar = make_array(
        root,
        "grid",
        (),
        "i4",
        zarr_format,
        dimensions=(),
        attributes={"grid_mapping_name": "latitude_longitude"},
    )
    scalar[()] = 0

    return str(path)
