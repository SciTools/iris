# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.zarr._dataset.ZarrDatasetVariable`."""

import numpy as np
import pytest
import zarr

from iris.fileformats.zarr._dataset import ZarrDatasetVariable
from iris.warnings import IrisCfLoadWarning

from .. import stores


def wrap(array, location="store.zarr"):
    """Return the ZarrDatasetVariable over `array`, naming its own format."""
    zarr_format = array.metadata.zarr_format
    return ZarrDatasetVariable(array, location, f"ZARR_V{zarr_format}")


@pytest.fixture(params=[2, 3], ids=["v2", "v3"])
def zarr_format(request):
    return request.param


@pytest.fixture
def group(tmp_path, zarr_format):
    return zarr.open_group(
        store=tmp_path / "probe.zarr", mode="w", zarr_format=zarr_format
    )


def test_named_dimensions(group, zarr_format):
    array = stores.make_array(
        group, "v", (3, 2), "f8", zarr_format, dimensions=("y", "x")
    )
    assert wrap(array).dimensions == ("y", "x")


def test_unnamed_dimensions_are_anonymous(group, zarr_format):
    array = stores.make_array(group, "v", (3, 2), "f8", zarr_format)
    assert wrap(array).dimensions == (None, None)


def test_scalar_has_no_dimensions(group, zarr_format):
    array = stores.make_array(group, "v", (), "i4", zarr_format)
    assert wrap(array).dimensions == ()


def test_partially_named_dimensions_version_3(tmp_path):
    # The version 3 core specification: "an unnamed dimension is indicated by
    # the null object". A store may name some dimensions and not others.
    group = zarr.open_group(store=tmp_path / "p.zarr", mode="w", zarr_format=3)
    array = group.create_array(
        "v", shape=(3, 2), dtype="f8", dimension_names=("y", None)
    )
    assert wrap(array).dimensions == ("y", None)


def test_repeated_name_within_one_array(tmp_path):
    # "It is recommended but not required that all non-null dimension names
    # are distinct." Recommended is not required, so this must not raise.
    group = zarr.open_group(store=tmp_path / "p.zarr", mode="w", zarr_format=3)
    array = group.create_array(
        "v", shape=(2, 2), dtype="f8", dimension_names=("d", "d")
    )
    assert wrap(array).dimensions == ("d", "d")


def test_wrong_length_dimension_names_warns_and_loads_anonymous(tmp_path):
    """A dimension-name count that disagrees with `ndim` is a malformation.

    Version 2 only: the version 3 core specification says `dimension_names`
    "must" be the same length as `shape`, and zarr-python enforces that on
    every read, not just on write - a hand-edited `zarr.json` with the wrong
    length raises `ValueError` from `Array.open` itself, before Iris sees an
    array at all. `_ARRAY_DIMENSIONS` carries no such validation, so this is
    reachable only through a version 2 store.
    """
    group = zarr.open_group(store=tmp_path / "bad.zarr", mode="w", zarr_format=2)
    array = stores.make_array(group, "v", (3, 2), "f8", 2, dimensions=("y", "x"))
    array.attrs[stores.ARRAY_DIMENSIONS] = ["y"]

    with pytest.warns(IrisCfLoadWarning, match="dimension_names"):
        dimensions = wrap(array).dimensions
    assert dimensions == (None, None)


def test_plain_properties(tmp_path, zarr_format):
    location = stores.make_store(tmp_path / "s.zarr", zarr_format=zarr_format)
    array = zarr.open_group(store=location, mode="r")["air_temperature"]
    variable = wrap(array, location)

    assert variable.name == "air_temperature"
    assert variable.location == location
    assert variable.format == f"ZARR_V{zarr_format}"
    assert variable.shape == stores.DATA.shape
    assert variable.dtype == np.dtype("f8")
    assert variable.size == stores.DATA.size
    assert variable.ndim == stores.DATA.ndim
    assert len(variable) == stores.DATA.shape[0]
    np.testing.assert_array_equal(variable[:], stores.DATA)
    np.testing.assert_array_equal(variable[1], stores.DATA[1])


def test_fill_value_is_the_storage_field(tmp_path, zarr_format):
    """`fill_value` is what unwritten storage reads as, not `_FillValue`.

    zarr-python defaults the field to zero, so an array that declares no CF
    sentinel still reports one here. Reading it as a mask source is what
    masks every valid zero.
    """
    group = zarr.open_group(
        store=tmp_path / "f.zarr", mode="w", zarr_format=zarr_format
    )
    array = stores.make_array(group, "v", (3,), "i2", zarr_format)
    assert wrap(array).fill_value == 0

    declared = stores.make_array(group, "w", (3,), "i2", zarr_format, fill_value=-999)
    assert wrap(declared).fill_value == -999


def test_chunking_is_the_inner_chunk_not_the_shard(tmp_path):
    """A sharded array reports `Array.chunks`.

    Zarr's sharding codec is indexed: a reader fetches the shard index, then
    byte ranges for only the inner chunks it needs. The inner chunk is the
    independently decodable unit.
    """
    group = zarr.open_group(store=tmp_path / "sh.zarr", mode="w", zarr_format=3)
    array = group.create_array(
        "v", shape=(1024,), dtype="i4", chunks=(8,), shards=(1024,)
    )
    variable = wrap(array)
    assert variable.chunking == (8,)
    assert variable.encoding["shards"] == (1024,)


def test_encoding_reports_the_store(tmp_path, zarr_format):
    location = stores.make_store(tmp_path / "e.zarr", zarr_format=zarr_format)
    array = zarr.open_group(store=location, mode="r")["air_temperature"]
    encoding = wrap(array, location).encoding
    assert encoding["chunks"] == array.chunks
    assert encoding["zarr_format"] == zarr_format


def test_array_dimensions_is_not_a_cf_attribute(tmp_path):
    """`_ARRAY_DIMENSIONS` is storage, so it must not reach `attributes`.

    It is how a version 2 store records dimension names. Every attribute the
    loading rules do not consume reaches `cube.attributes`, so leaving it in
    would put it on every cube loaded from a version 2 store, and make that
    cube unequal to the same data loaded from version 3.
    """
    group = zarr.open_group(store=tmp_path / "v2.zarr", mode="w", zarr_format=2)
    array = stores.make_array(
        group,
        "v",
        (3, 2),
        "f8",
        2,
        dimensions=("y", "x"),
        attributes={"units": "K"},
    )
    variable = wrap(array)

    assert variable.dimensions == ("y", "x")
    assert stores.ARRAY_DIMENSIONS in array.attrs
    assert stores.ARRAY_DIMENSIONS not in variable.attributes
    assert dict(variable.attributes) == {"units": "K"}


def test_version_2_and_version_3_present_the_same_attributes(tmp_path):
    v2 = stores.make_store(tmp_path / "a2.zarr", zarr_format=2)
    v3 = stores.make_store(tmp_path / "a3.zarr", zarr_format=3)
    from_2 = zarr.open_group(store=v2, mode="r")["air_temperature"]
    from_3 = zarr.open_group(store=v3, mode="r")["air_temperature"]
    assert dict(wrap(from_2, v2).attributes) == dict(wrap(from_3, v3).attributes)


def test_structured_attribute_values_are_carried_unchanged(tmp_path):
    """A JSON object or array attribute is carried, not interpreted.

    The version 3 specification allows an attribute to be any JSON literal,
    and CF cannot express a nested one at any rank. Two independent producers
    write them - NOAA GFS gives `valid_time` a `statistics_approximate`
    object, and the ESA EOPF product root carries `stac_discovery`.
    A flat numeric array is an ordinary CF vector attribute and is not special.
    """
    group = zarr.open_group(store=tmp_path / "n.zarr", mode="w", zarr_format=3)
    array = stores.make_array(
        group,
        "v",
        (2,),
        "f8",
        3,
        attributes={
            "statistics_approximate": {"min": 0.0, "max": 1.0},
            "valid_range": [0.0, 1.0],
            "nested_list": [[1, 2], [3, 4]],
        },
    )
    attributes = wrap(array).attributes
    assert attributes["statistics_approximate"] == {"min": 0.0, "max": 1.0}
    assert attributes["valid_range"] == [0.0, 1.0]
    assert attributes["nested_list"] == [[1, 2], [3, 4]]
