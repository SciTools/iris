# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Stored Zarr stores, read back with their fixed-length unicode intact.

The unit tests in `lib/iris/tests/unit/fileformats/zarr/` write a `<U` array
and read it back, so both halves move whenever zarr-python's encoding moves.
These stores do not move. They were written once and committed, so they pin
that Iris still reads an encoding a published file may have been written with
years ago, which is the claim `<U` support actually makes. :issue:`6961` is
the dtype question.
"""

import json
from pathlib import Path

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

import iris
from iris.coords import AuxCoord
from iris.tests import _shared_utils

#: The dimension each string coordinate spans, and the dtype it is stored as.
EXPECTED_COORDS = {
    "Spectral band": ("<U3", 0, ["b01", "b02", "b03"]),
    "Detector": ("<U3", 1, ["d01", "d02", "d03"]),
    "Angle name": ("<U7", 2, ["zenith", "azimuth"]),
}

STORES = ["unicode_coords_v2.zarr", "unicode_coords_v3.zarr"]


@pytest.fixture(params=STORES)
def sample(request):
    return _shared_utils.get_data_path(("Zarr", request.param))


@_shared_utils.skip_data
def test_the_string_coordinates_keep_their_dtype_and_values(sample):
    cube = iris.load_cube(sample)
    assert cube.shape == (13, 5, 2)

    for name, (dtype, dimension, first_values) in EXPECTED_COORDS.items():
        coord = cube.coord(name)
        assert isinstance(coord, AuxCoord)
        assert coord.dtype == np.dtype(dtype)
        assert cube.coord_dims(coord) == (dimension,)
        assert list(coord.points[: len(first_values)]) == first_values


@_shared_utils.skip_data
def test_both_versions_give_the_same_cube():
    """Version 2 and version 3 record dimension names and dtypes differently.

    They are expected to reach an identical cube by different routes, so a
    difference here is a version-handling bug rather than a data difference.
    """
    cubes = [
        iris.load_cube(_shared_utils.get_data_path(("Zarr", store))) for store in STORES
    ]
    assert cubes[0] == cubes[1]


@_shared_utils.skip_data
def test_the_stores_still_hold_the_encoding_they_were_written_with():
    """These stores are worthless if regenerated, so say what they must be.

    A newer zarr-python may encode `<U` differently. That is exactly what
    these stores exist to outlive, so this fails if someone refreshes them
    rather than leaving the committed bytes alone.
    """
    version_2 = Path(_shared_utils.get_data_path(("Zarr", STORES[0])))
    metadata = json.loads((version_2 / "angle" / ".zarray").read_text())
    assert metadata["dtype"] == "<U7"

    version_3 = Path(_shared_utils.get_data_path(("Zarr", STORES[1])))
    metadata = json.loads((version_3 / "angle" / "zarr.json").read_text())
    assert metadata["data_type"]["name"] == "fixed_length_utf32"
    assert metadata["data_type"]["configuration"]["length_bytes"] == 28
