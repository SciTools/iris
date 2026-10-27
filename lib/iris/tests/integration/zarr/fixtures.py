# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""A version 3 store carrying the CF features a real one does.

Modelled on the NOAA GFS archive at dynamical.org, whose properties the Zarr
I/O design names as the ones worth exercising: version 3 `dimension_names`, a
sharded codec, a scalar grid-mapping variable, a two-dimensional auxiliary
coordinate, and the base64 `_FillValue` that released xarray writes for every
floating dtype.

A cut-down copy of that real store sits beside this one, and `test_gfs_sample.py`
is where it is loaded. The two are not redundant. This one needs no external
data, so the tests here run wherever `iris-test-data` does not; and it is the
more CF-conforming of the pair, because it gives its data variable a
`grid_mapping` attribute, which the published store does not have.

Bounds on `latitude` are not part of that real store, but earn their place
anyway: `CFBoundaryVariable.spans` is one of the overrides this pull request
rewrote against the dimension contract, so a bounded coordinate exercises
code this branch actually changed.
"""

import base64
import struct

import numpy as np
import zarr

SHAPE = (2, 4, 6, 5)
DIMENSIONS = ("time", "level", "latitude", "longitude")

#: Contiguous bounds for `latitude`, half a cell either side of each centre.
LATITUDE_BOUNDS = np.stack(
    [np.arange(SHAPE[2], dtype="f8") - 0.5, np.arange(SHAPE[2], dtype="f8") + 0.5],
    axis=1,
)


def float_fill_value(value):
    """Return `value` in the encoding released xarray writes for a float."""
    return base64.b64encode(struct.pack("<d", value)).decode("ascii")


def make_gfs_like_store(path):
    """Write the store and return its path as a string."""
    root = zarr.open_group(store=path, mode="w", zarr_format=3)
    root.attrs["Conventions"] = "CF-1.12"
    root.attrs["title"] = "A store shaped like a real one"
    root.attrs["license"] = "CC-BY-4.0"

    sizes = dict(zip(DIMENSIONS, SHAPE))
    for name, size in sizes.items():
        coordinate = root.create_array(
            name, shape=(size,), dtype="f8", dimension_names=(name,)
        )
        coordinate[:] = np.arange(size, dtype="f8")
    root["time"].attrs.update(
        {"units": "hours since 2000-01-01 00:00:00", "calendar": "standard"}
    )
    root["level"].attrs.update({"units": "hPa", "positive": "down"})
    root["latitude"].attrs.update(
        {"units": "degrees_north", "axis": "Y", "bounds": "latitude_bnds"}
    )
    root["longitude"].attrs.update({"units": "degrees_east", "axis": "X"})

    # Bounds on a dimension coordinate, exercising `CFBoundaryVariable.spans`:
    # one more dimension than `latitude`, named for a dimension `latitude`
    # does not have, at the size CF bounds always are.
    bounds = root.create_array(
        "latitude_bnds",
        shape=(SHAPE[2], 2),
        dtype="f8",
        dimension_names=("latitude", "bnds"),
    )
    bounds[:] = LATITUDE_BOUNDS

    # A scalar grid mapping, as the GFS store's `spatial_ref` is.
    grid = root.create_array("spatial_ref", shape=(), dtype="i4")
    grid[()] = 0
    grid.attrs.update(
        {"grid_mapping_name": "latitude_longitude", "earth_radius": 6371229.0}
    )

    # A two-dimensional auxiliary coordinate, as the GFS store's valid_time is.
    valid_time = root.create_array(
        "valid_time",
        shape=SHAPE[:2],
        dtype="f8",
        dimension_names=DIMENSIONS[:2],
    )
    valid_time[:] = np.arange(np.prod(SHAPE[:2]), dtype="f8").reshape(SHAPE[:2])
    valid_time.attrs.update(
        {
            "units": "hours since 2000-01-01 00:00:00",
            "standard_name": "time",
            "statistics_approximate": {"min": 0.0, "max": 7.0},
        }
    )

    # The data variable: sharded, packed, with a base64 fill value.
    data = root.create_array(
        "temperature_2m",
        shape=SHAPE,
        dtype="i2",
        chunks=(1, 1, 3, 5),
        shards=(2, 4, 6, 5),
        dimension_names=DIMENSIONS,
        fill_value=-32768,
    )
    data[:] = np.arange(np.prod(SHAPE), dtype="i2").reshape(SHAPE)
    data.attrs.update(
        {
            "standard_name": "air_temperature",
            "units": "K",
            "scale_factor": 0.01,
            "add_offset": 273.15,
            # An integer variable's sentinel is a JSON number. The base64
            # encoding is for floating dtypes, and is on `pressure_surface`.
            "_FillValue": -32768,
            "coordinates": "valid_time",
            "grid_mapping": "spatial_ref",
        }
    )

    # A floating variable, carrying the base64 `_FillValue` that released
    # xarray writes for every floating dtype.
    flat_shape = SHAPE[:1] + SHAPE[2:]
    pressure = root.create_array(
        "pressure_surface",
        shape=flat_shape,
        dtype="f4",
        dimension_names=DIMENSIONS[:1] + DIMENSIONS[2:],
    )
    pressure[:] = np.arange(int(np.prod(flat_shape)), dtype="f4").reshape(flat_shape)
    pressure.attrs.update(
        {
            "standard_name": "surface_air_pressure",
            "units": "Pa",
            "_FillValue": float_fill_value(float("nan")),
            "grid_mapping": "spatial_ref",
        }
    )

    return str(path)
