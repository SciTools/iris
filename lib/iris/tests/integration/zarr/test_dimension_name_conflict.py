# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""A dimension name used at two different lengths, loaded end to end.

Zarr "does not place any restrictions on the use of the same dimension name
across multiple arrays", so a store may name two arrays' dimensions ``t`` and
give them different lengths. Iris cannot link those, and the arrays load with
an anonymous dimension instead.

The warning is the only thing that explains why. A coordinate that could not
attach reaches the user through `LOAD_PROBLEMS`, which names the coordinate
and not the cause. These tests go through `iris.load` rather than building a
`ZarrDataset`, because the warning comes from a dataset member that the CF
layer has no other reason to touch - the unit tests pass either way.
"""

import numpy as np
import pytest
import zarr

import iris
from iris.loading import LOAD_PROBLEMS
from iris.warnings import IrisCfLoadWarning


@pytest.fixture(autouse=True)
def _reset_load_problems():
    """These loads record problems, so neither leaks into another test."""
    LOAD_PROBLEMS.reset()
    yield
    LOAD_PROBLEMS.reset()


@pytest.fixture
def conflicting_store(tmp_path):
    """A store naming ``t`` at two lengths, with a coordinate for the shorter."""
    location = tmp_path / "conflict.zarr"
    root = zarr.open_group(store=location, mode="w", zarr_format=3)
    root.attrs["Conventions"] = "CF-1.12"

    def array(name, size, **attributes):
        written = root.create_array(
            name, shape=(size,), dtype="f4", dimension_names=("t",)
        )
        written[:] = np.arange(size, dtype="f4")
        written.attrs.update(attributes)

    array("t", 4, standard_name="time", units="days since 2000-01-01")
    array("air_temperature", 4, standard_name="air_temperature", units="K")
    array("air_pressure", 5, standard_name="air_pressure", units="Pa")
    return str(location)


def test_the_conflict_reaches_a_load(conflicting_store):
    with pytest.warns(IrisCfLoadWarning, match=r"more than one size, \[4, 5\]: t"):
        iris.load(conflicting_store)


def test_the_unlinkable_dimension_is_anonymous(conflicting_store):
    cubes = iris.load(conflicting_store)

    by_name = {cube.name(): cube for cube in cubes}
    assert by_name["air_pressure"].coords() == []
    assert [coord.name() for coord in by_name["air_temperature"].coords()] == ["time"]
