# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""The write-alignment invariant.

Spec §4.5: every Zarr array Iris writes has a write grid that the lazy
source's dask chunking tiles exactly, where the write grid is the shards when
the array is sharded and the chunks otherwise. Each concurrent task owns at
least one whole cell; no cell is touched by two.

The invariant is what licenses ``lock=False``. Stated over the chunk grid
alone it is insufficient, and the gap is not theoretical: eight inner chunks
sharing one shard means eight tasks read-modify-writing one stored object,
which lost 72-96 of 128 values in 12 of 12 trials.
"""

import dask.array as da
import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from iris.coords import DimCoord
from iris.cube import Cube
from iris.fileformats.zarr.saver import Saver, _tiles_the_grid


class TestTilesTheGrid:
    def test_equal_chunks_tile(self):
        assert _tiles_the_grid(((8, 8, 8),), (8,))

    def test_a_whole_multiple_tiles(self):
        # 128-value chunks over a 64-value grid: each task owns two cells.
        assert _tiles_the_grid(((128, 128),), (64,))

    def test_a_smaller_chunk_does_not(self):
        # The corruption case: eight chunks per shard.
        assert not _tiles_the_grid(((8,) * 16,), (64,))

    def test_a_ragged_final_chunk_is_allowed(self):
        # "A truncated final cell is permitted."
        assert _tiles_the_grid(((64, 64, 7),), (64,))

    def test_an_unaligned_boundary_does_not(self):
        assert not _tiles_the_grid(((70, 58),), (64,))

    def test_every_dimension_must_tile(self):
        assert not _tiles_the_grid(((64, 64), (8, 8)), (64, 64))


class TestTheInvariantHolds:
    def test_a_sharded_target_keeps_every_value(self, tmp_path):
        # The §4.5 reproduction, through the saver. Without the rechunk this
        # loses 72-96 of 128 values, every time.
        values = np.arange(128, dtype=np.float32)
        cube = Cube(da.from_array(values, chunks=8), var_name="data")
        cube.add_dim_coord(DimCoord(np.arange(128, dtype=np.float32), var_name="x"), 0)
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(cube, chunks=(8,), shards=(64,))
        written = zarr.open_group(store=tmp_path / "out.zarr", mode="r")["data"]
        np.testing.assert_array_equal(written[:], values)

    def test_the_source_is_rechunked_to_the_shard(self, tmp_path):
        cube = Cube(
            da.from_array(np.arange(128, dtype=np.float32), chunks=8),
            var_name="data",
        )
        cube.add_dim_coord(DimCoord(np.arange(128, dtype=np.float32), var_name="x"), 0)
        with Saver(tmp_path / "out.zarr", compute=False) as saver:
            saver.write(cube, chunks=(8,), shards=(64,))
            sources = [source for source, _ in saver._delayed_writes]
        data_source = [s for s in sources if s.shape == (128,)][0]
        assert data_source.chunks == ((64, 64),)

    def test_an_aligned_source_is_left_alone(self, tmp_path):
        # A needless rechunk is a needless graph layer.
        cube = Cube(
            da.from_array(np.arange(128, dtype=np.float32), chunks=64),
            var_name="data",
        )
        cube.add_dim_coord(DimCoord(np.arange(128, dtype=np.float32), var_name="x"), 0)
        with Saver(tmp_path / "out.zarr", compute=False) as saver:
            saver.write(cube, chunks=(8,), shards=(64,))
            sources = [source for source, _ in saver._delayed_writes]
        data_source = [s for s in sources if s.shape == (128,)][0]
        assert data_source.chunks == ((64, 64),)

    def test_an_unsharded_target_grids_on_chunks(self, tmp_path):
        cube = Cube(
            da.from_array(np.arange(64, dtype=np.float32), chunks=3),
            var_name="data",
        )
        cube.add_dim_coord(DimCoord(np.arange(64, dtype=np.float32), var_name="x"), 0)
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(cube, chunks=(16,))
        written = zarr.open_group(store=tmp_path / "out.zarr", mode="r")["data"]
        np.testing.assert_array_equal(written[:], np.arange(64, dtype=np.float32))
