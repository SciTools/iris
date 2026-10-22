# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Builders for the small Zarr stores these tests ask questions of.

One canonical store, built from the constants the shared CFDataset contract
writes, so that a Zarr answer and a netCDF answer can be compared without
translating between two sets of fixtures - and so that the two cannot drift
apart.
"""

import numpy as np
import zarr

from iris.tests.unit.fileformats.cf.dataset.contract import (
    CONTRACT_DATA as DATA,
)
from iris.tests.unit.fileformats.cf.dataset.contract import (
    CONTRACT_DIMENSIONS as DIMENSIONS,
)
from iris.tests.unit.fileformats.cf.dataset.contract import (
    CONTRACT_GLOBALS as GLOBALS,
)
from iris.tests.unit.fileformats.cf.dataset.contract import (
    CONTRACT_VARIABLE_ATTRS,
)

#: The canonical store's variable attributes.
#:
#: The contract's own, plus the `grid_mapping` that makes the `grid` array
#: below mean something. The contract checks its attributes are present
#: rather than that they are all there is, so the extra is legitimate.
VARIABLE_ATTRS = {**CONTRACT_VARIABLE_ATTRS, "grid_mapping": "grid"}

#: The version 2 attribute carrying dimension names.
ARRAY_DIMENSIONS = "_ARRAY_DIMENSIONS"

#: `make_array`'s own default for `fill_value`, distinct from `None`.
#:
#: `Group.create_array` treats an explicit `fill_value=None` as "this array
#: has no fill value" on a version 2 store, which is not the same thing as
#: "the caller did not ask for one". Passing this sentinel through instead
#: means `make_array`'s own default lets zarr-python choose its own default
#: fill value, on both versions, the way omitting the argument would.
_DEFAULT_FILL_VALUE = object()


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
    fill_value=_DEFAULT_FILL_VALUE,
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
    if fill_value is not _DEFAULT_FILL_VALUE:
        extra["fill_value"] = fill_value
    array = group.create_array(
        name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        shards=shards,
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
        attributes={
            "grid_mapping_name": "latitude_longitude",
            "earth_radius": 6371229.0,
        },
    )
    scalar[()] = 0

    return str(path)


#: The UGRID store's mesh topology variable, as CF attributes.
MESH_ATTRS = {
    "cf_role": "mesh_topology",
    "topology_dimension": 2,
    "node_coordinates": "node_x node_y",
    "face_node_connectivity": "face_nodes",
    "face_coordinates": "face_x face_y",
    "long_name": "topology of a 2D mesh",
}

#: The UGRID store's arrays: name -> (values, dimension names, attributes).
#:
#: Two quadrilateral faces over six nodes, with a time dimension on the data
#: so that the mesh dimension is not the only one. Every dimension is a
#: different size, so a coordinate cannot be attached to the right axis by
#: accident - see `_DimensionIdentity`.
MESH_ARRAYS = {
    "node_x": (
        np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
        ("node",),
        {"standard_name": "longitude", "units": "degrees_east"},
    ),
    "node_y": (
        np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        ("node",),
        {"standard_name": "latitude", "units": "degrees_north"},
    ),
    "face_x": (
        np.array([0.5, 1.5]),
        ("face",),
        {"standard_name": "longitude", "units": "degrees_east"},
    ),
    "face_y": (
        np.array([0.5, 0.5]),
        ("face",),
        {"standard_name": "latitude", "units": "degrees_north"},
    ),
    "face_nodes": (
        np.array([[0, 1, 4, 3], [1, 2, 5, 4]], dtype="i4"),
        ("face", "vertex"),
        {"cf_role": "face_node_connectivity", "start_index": 0},
    ),
    "time": (
        np.array([0.0, 1.0, 2.0]),
        ("time",),
        {"standard_name": "time", "units": "days since 2000-01-01"},
    ),
    "temp": (
        np.arange(6, dtype="f8").reshape(3, 2),
        ("time", "face"),
        {
            "standard_name": "air_temperature",
            "units": "K",
            "mesh": "mesh",
            "location": "face",
            "coordinates": "face_x face_y",
        },
    ),
}


def make_mesh_store(path, *, zarr_format=3):
    """Write a UGRID store at `path`, and return `path` as a string.

    The same topology this module's netCDF counterpart writes, so that the
    two backends' meshes can be compared directly.

    Every array is written with no fill value. A mesh is full of legitimate
    zeros - node coordinates on an axis, and connectivity indices under
    `start_index = 0` - and on a version 2 store zarr-python's default fill
    value is zero, which Iris reads as "mask the zeros". That rule has its
    own test; a mesh fixture that tripped it would be testing it twice and
    reading its own topology back full of masks.
    """
    root = zarr.open_group(store=path, mode="w", zarr_format=zarr_format)
    root.attrs["Conventions"] = "CF-1.7, UGRID-1.0"

    mesh = make_array(
        root, "mesh", (), "i4", zarr_format, dimensions=(), fill_value=None
    )
    for key, value in MESH_ATTRS.items():
        mesh.attrs[key] = value
    mesh[()] = 0

    for name, (values, dimensions, attributes) in MESH_ARRAYS.items():
        array = make_array(
            root,
            name,
            values.shape,
            values.dtype,
            zarr_format,
            dimensions=dimensions,
            fill_value=None,
            attributes=attributes,
        )
        array[:] = values

    return str(path)
