# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :meth:`ZarrDatasetVariable.read_data`."""

import dask
import dask.array as da
import dask.config
import numpy as np
import pytest
import zarr

from iris import _lazy_data
from iris.fileformats.cf import loader as cf_loader
from iris.fileformats.zarr._dataset import ZarrDataset
from iris.warnings import IrisLoadWarning

from .. import stores


def variable(location, name="air_temperature"):
    return ZarrDataset(location).variables[name]


def policy():
    """The chunking policy a loader outside any CHUNK_CONTROL supplies."""
    return None, None


def test_a_small_variable_is_read_whole(tmp_path):
    location = stores.make_store(tmp_path / "s.zarr")
    result = variable(location).read_data(policy)
    assert not isinstance(result, da.Array)
    np.testing.assert_array_equal(np.asarray(result), stores.DATA)


def test_a_small_variable_never_consults_the_policy(tmp_path):
    """Under CHUNK_CONTROL.from_file() the policy raises for an unchunked store.

    A variable small enough to read whole must reach its data without asking,
    or `from_file` starts rejecting small cubes that load today.
    """

    def raising_policy():
        raise AssertionError("chunking policy consulted for a small variable")

    location = stores.make_store(tmp_path / "s.zarr")
    variable(location).read_data(raising_policy)


def test_a_large_variable_is_lazy(tmp_path, monkeypatch):
    monkeypatch.setattr(cf_loader, "_LAZYVAR_MIN_BYTES", 1)
    location = stores.make_store(tmp_path / "s.zarr")
    result = variable(location).read_data(policy)
    assert isinstance(result, da.Array)
    np.testing.assert_array_equal(result.compute(), stores.DATA)


def test_a_plain_array_gets_an_unmasked_meta(tmp_path, monkeypatch):
    # No decode layer, so no masked meta: a masked meta with nothing beneath
    # it is a meta that lies about its own graph.
    monkeypatch.setattr(cf_loader, "_LAZYVAR_MIN_BYTES", 1)
    location = stores.make_store(tmp_path / "s.zarr")
    result = variable(location).read_data(policy)
    assert not isinstance(result._meta, np.ma.MaskedArray)


def test_a_decoded_array_gets_a_masked_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(cf_loader, "_LAZYVAR_MIN_BYTES", 1)
    group = zarr.open_group(store=tmp_path / "m.zarr", mode="w", zarr_format=3)
    array = stores.make_array(
        group,
        "v",
        (4,),
        "i2",
        3,
        dimensions=("t",),
        attributes={"_FillValue": -999},
    )
    array[:] = [-999, 1, 2, 3]

    result = ZarrDataset(str(tmp_path / "m.zarr")).variables["v"].read_data(policy)
    assert isinstance(result._meta, np.ma.MaskedArray)
    realised = result.compute()
    assert realised.mask.tolist() == [True, False, False, False]


def key_of(location, name="v"):
    return ZarrDataset(location).variables[name].cache_key


def test_the_key_is_stable_across_a_reopen(tmp_path):
    group = zarr.open_group(store=tmp_path / "k.zarr", mode="w", zarr_format=3)
    stores.make_array(group, "v", (4,), "i4", 3, dimensions=("t",))
    location = str(tmp_path / "k.zarr")
    assert key_of(location) == key_of(location)


def test_the_key_changes_when_the_shape_does(tmp_path):
    """A store rewritten in place keeps its address and changes its contents.

    Keyed on the address alone, an array reopened after its shape changed
    from (4,) to (6,) returns the cached four values, silently truncating -
    and that is weaker than the netCDF key it replaces, since
    repr(NetCDFDataProxy) already carries shape and dtype.
    """
    location = str(tmp_path / "k.zarr")
    group = zarr.open_group(store=location, mode="w", zarr_format=3)
    stores.make_array(group, "v", (4,), "i4", 3, dimensions=("t",), chunks=(2,))
    before = key_of(location)

    group = zarr.open_group(store=location, mode="a", zarr_format=3)
    group.create_array(
        "v",
        shape=(6,),
        dtype="i4",
        chunks=(2,),
        dimension_names=("t",),
        overwrite=True,
    )
    assert key_of(location) != before


def test_the_key_changes_when_the_fill_value_does(tmp_path):
    location = str(tmp_path / "k.zarr")
    group = zarr.open_group(store=location, mode="w", zarr_format=3)
    group.create_array("v", shape=(4,), dtype="i4", fill_value=-1)
    before = key_of(location)

    group = zarr.open_group(store=location, mode="a", zarr_format=3)
    group.create_array("v", shape=(4,), dtype="i4", fill_value=-2, overwrite=True)
    assert key_of(location) != before


@pytest.mark.parametrize(
    "change",
    [
        {"shape": (8,)},
        {"chunks": (4,)},
        {"shards": (8,)},
        {"dtype": "f8"},
        {"fill_value": -7},
        {"dimension_names": ("other",)},
    ],
)
def test_the_key_is_sensitive_to_every_metadata_field(tmp_path, change):
    """`Array.metadata.to_dict()` is zarr-python's rendering, not a guarantee.

    Nothing in the Zarr specification defines what that method returns, so
    these are what fail if a release stops carrying a field the key needs.
    """
    base = dict(
        shape=(4,),
        dtype="i4",
        chunks=(2,),
        fill_value=-1,
        dimension_names=("t",),
        shards=None,
    )
    location = str(tmp_path / "k.zarr")
    group = zarr.open_group(store=location, mode="w", zarr_format=3)
    group.create_array("v", **base)
    before = key_of(location)

    group = zarr.open_group(store=location, mode="a", zarr_format=3)
    group.create_array("v", **{**base, **change}, overwrite=True)
    assert key_of(location) != before


def test_the_key_is_sensitive_to_attributes(tmp_path):
    location = str(tmp_path / "k.zarr")
    group = zarr.open_group(store=location, mode="w", zarr_format=3)
    array = group.create_array("v", shape=(4,), dtype="i4")
    before = key_of(location)
    array.attrs["units"] = "K"
    assert key_of(location) != before


def test_two_cubes_sharing_an_array_share_one_dask_array(tmp_path, monkeypatch):
    """What the cache is for: many cubes in one store sharing a coordinate.

    Comparing the two Dask names proves nothing on its own: Dask tokenises a
    zarr.Array deterministically, so two independently opened arrays get the
    same name whether or not `read_data` passes a cache key at all. What has
    to be shown is that the first read puts an entry in the process-wide
    cache and the second read comes back out of it.

    The cache is replaced for the test rather than measured in place, because
    it is process-wide and holds only its last hundred entries - a test that
    counted the real one would depend on what ran before it.
    """
    monkeypatch.setattr(cf_loader, "_LAZYVAR_MIN_BYTES", 1)
    cache = _lazy_data.LRUCache(100)
    monkeypatch.setattr(_lazy_data, "CACHE", cache)
    location = stores.make_store(tmp_path / "s.zarr")

    first = variable(location).read_data(policy)
    assert len(cache._cache) == 1, "the first read did not populate the cache"

    second = variable(location).read_data(policy)
    assert len(cache._cache) == 1, "the second read built a second array"
    assert first.name == second.name


class CountingStore(zarr.storage.WrapperStore):
    """Wrap a store, adding every byte it returns to a caller-owned tally.

    zarr-python copies a store when it opens it read-only, through
    `with_read_only` and `_with_store`, so a tally kept on the instance is
    never the one the live store writes to, and the test passes vacuously.
    """

    def __init__(self, store, tally):
        super().__init__(store=store)
        self.tally = tally

    def _with_store(self, store):
        return type(self)(store, self.tally)

    async def get(self, key, prototype, byte_range=None):
        buffer = await self._store.get(key, prototype, byte_range)
        if buffer is not None:
            self.tally.append(len(buffer))
        return buffer

    async def get_partial_values(self, prototype, key_ranges):
        buffers = await self._store.get_partial_values(prototype, key_ranges)
        self.tally.extend(len(b) for b in buffers if b is not None)
        return buffers


def bytes_to_read_first_chunk(location, dask_chunks):
    """Return the bytes fetched reading the first eight values of "v"."""
    tally = []
    store = CountingStore(zarr.storage.LocalStore(location), tally)
    array = zarr.open_group(store=store, mode="r")["v"]
    tally.clear()
    lazy = da.from_array(array, chunks=dask_chunks, meta=np.empty((0,), "i4"))
    lazy = da.map_blocks(lambda block: np.ma.masked_array(block), lazy, dtype="i4")
    lazy[:8].compute()
    return sum(tally)


def test_a_sharded_array_is_read_by_the_inner_chunk(tmp_path):
    """Shard-sized Dask blocks fetch the whole shard, because of the layer.

    Bare Dask slicing can push a slice down into the array's own __getitem__,
    but a map_blocks layer in between cannot, so the block is materialised in
    full. The absolute byte counts are zarr-version-dependent; the comparison
    is not.
    """
    location = str(tmp_path / "sh.zarr")
    group = zarr.open_group(store=location, mode="w", zarr_format=3)
    array = group.create_array(
        "v", shape=(1024,), dtype="i4", chunks=(8,), shards=(1024,)
    )
    array[:] = np.arange(1024, dtype="i4")

    inner = bytes_to_read_first_chunk(location, 8)
    whole_shard = bytes_to_read_first_chunk(location, 1024)
    assert inner < whole_shard

    # And the variable chooses the inner chunk for exactly this reason.
    assert ZarrDataset(location).variables["v"].chunking == (8,)


def test_the_declared_dtype_matches_what_the_decode_layer_produces(
    tmp_path, monkeypatch
):
    """A packed variable's lazy dtype must equal its realised dtype.

    On the netCDF path netCDF4 does the arithmetic and reports the dtype. On
    the Zarr path Iris does both, in two separately written statements, and
    Dask does not check them against each other: a mismatch builds a graph
    and surfaces later as a wrong dtype on the cube.
    """
    monkeypatch.setattr(cf_loader, "_LAZYVAR_MIN_BYTES", 1)
    group = zarr.open_group(store=tmp_path / "p.zarr", mode="w", zarr_format=3)
    array = stores.make_array(
        group,
        "b02",
        (4,),
        "u2",
        3,
        dimensions=("p",),
        attributes={"scale_factor": 0.0001, "add_offset": -0.1},
    )
    array[:] = [1000, 2000, 3000, 4000]

    lazy = ZarrDataset(str(tmp_path / "p.zarr")).variables["b02"].read_data(policy)
    realised = lazy.compute()

    assert lazy.dtype == realised.dtype
    assert lazy.dtype == np.dtype("f8")
    np.testing.assert_allclose(realised, [0.0, 0.1, 0.2, 0.3])


@pytest.mark.parametrize(
    ("scale_factor", "expected"),
    # A Zarr attribute is JSON, so a scale factor reaches Iris as a Python
    # float or a Python int and never as a NumPy scalar - zarr-python refuses
    # to write one. Both forms are here because they promote differently:
    # a weak Python int leaves a uint16 variable uint16.
    [(0.01, "f8"), (1, "u2")],
)
def test_the_declared_dtype_matches_the_cf_loader_calculation(
    tmp_path, scale_factor, expected
):
    """And it is the same answer `_get_actual_dtype` gives.

    The cube's data variable and its coordinates must agree about what a
    packed value becomes, and they reach that answer through different code.
    """
    group = zarr.open_group(store=tmp_path / "p.zarr", mode="w", zarr_format=3)
    stores.make_array(
        group,
        "v",
        (4,),
        "u2",
        3,
        attributes={"scale_factor": scale_factor},
    )
    variable = ZarrDataset(str(tmp_path / "p.zarr")).variables["v"]
    from iris.fileformats.zarr import _decode

    parameters = _decode.decode_parameters(
        variable.attributes,
        dtype=variable.dtype,
        storage_fill_value=variable.fill_value,
        zarr_format=variable.zarr_format,
        variable_name=variable.name,
    )
    assert parameters["dtype"] == cf_loader._get_actual_dtype(variable)
    assert parameters["dtype"] == np.dtype(expected)


def test_a_lazy_read_aligns_to_the_store_chunks(tmp_path, monkeypatch):
    """The alignment reaches the array, not just the function that computes it.

    Dask's target is shrunk below one store chunk, so an unaligned result
    would be the 8 elements the target asks for, subdividing the store's 16.
    """
    monkeypatch.setattr(cf_loader, "_LAZYVAR_MIN_BYTES", 1)
    location = str(tmp_path / "b.zarr")
    group = zarr.open_group(store=location, mode="w", zarr_format=3)
    array = group.create_array(
        "v", shape=(64,), dtype="i4", chunks=(16,), dimension_names=("t",)
    )
    array[:] = np.arange(64, dtype="i4")

    with dask.config.set({"array.chunk-size": "32B"}):
        with pytest.warns(IrisLoadWarning, match="v"):
            lazy = ZarrDataset(location).variables["v"].read_data(policy)

    assert lazy.chunks == ((16, 16, 16, 16),)
    np.testing.assert_array_equal(lazy.compute(), np.arange(64))
