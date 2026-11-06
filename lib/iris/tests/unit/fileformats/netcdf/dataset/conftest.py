# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""A real netCDF4 file to test the netCDF CFDataset implementation against.

Real, rather than mocked, because the members being implemented are precisely
the ones whose netCDF4 behaviour is easy to misremember - what ``chunking()``
returns for a contiguous variable, what ``dimensions`` is for char data, what
``ncattrs()`` includes once ``fill_value=`` has been passed.
"""

import numpy as np
import pytest

from iris.fileformats.netcdf import _thread_safe_nc

#: How the sample file's dimensions are created. None means unlimited.
SAMPLE_DIMENSION_SIZES = {"time": 3, "lat": 4, "nchars": 8, "record": None}

#: How they should then read back. An unlimited dimension reports the number
#: of records actually written, which here is none.
SAMPLE_DIMENSIONS = {"time": 3, "lat": 4, "nchars": 8, "record": 0}

#: The sample file's global attributes.
SAMPLE_GLOBALS = {"Conventions": "CF-1.7", "title": "sample"}

#: The sample file's air_temperature payload.
SAMPLE_AIR = np.arange(12, dtype="f4").reshape(3, 4)

#: The sample file's label payload, as the strings it encodes.
SAMPLE_LABELS = ["alpha", "beta", "gamma"]


@pytest.fixture(scope="session")
def sample_path(tmp_path_factory):
    """Write, once per session, a small netCDF4 file and return its path."""
    path = tmp_path_factory.mktemp("cf_dataset") / "sample.nc"
    dataset = _thread_safe_nc.DatasetWrapper(path, mode="w", format="NETCDF4")
    try:
        for name, size in SAMPLE_DIMENSION_SIZES.items():
            dataset.createDimension(name, size)
        for name, value in SAMPLE_GLOBALS.items():
            dataset.setncattr(name, value)

        air = dataset.createVariable(
            "air_temperature",
            "f4",
            ("time", "lat"),
            fill_value=-999.0,
            zlib=True,
            chunksizes=(1, 4),
        )
        air.setncattr("units", "K")
        air.setncattr("standard_name", "air_temperature")
        air.setncattr("coordinates", "height")
        air[:] = SAMPLE_AIR

        # Scalar, and deliberately unchunked: chunking() answers differently.
        height = dataset.createVariable("height", "f4", ())
        height.setncattr("units", "m")
        height[:] = 1.5

        # Char data: the one case where EncodedVariable changes shape, dtype
        # and dimensions out from under the wrapper.
        label = dataset.createVariable("label", "S1", ("time", "nchars"))
        label[:] = np.array([list(f"{text:<8}") for text in SAMPLE_LABELS], dtype="S1")
    finally:
        dataset.close()

    return path


@pytest.fixture
def sample_location(sample_path):
    """The sample file's path, as the string a CFDataset reports."""
    return str(sample_path)
