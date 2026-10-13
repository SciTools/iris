# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the Zarr read-side chunk alignment.

No store is built here: the guard is a function of chunk shapes, which is
most of the point of having it as one.
"""

import warnings

import numpy as np
import pytest

from iris.fileformats.zarr._dataset import _align_chunks
from iris.warnings import IrisLoadWarning

F8 = np.dtype("f8")


def test_a_whole_multiple_is_left_alone():
    assert _align_chunks((4096, 4096), (512, 512), (4096, 4096), F8, None, "v") == (
        4096,
        4096,
    )


def test_a_sub_multiple_is_rounded_up():
    """A dask chunk that subdivides a store chunk costs N re-reads.

    A read of any part of a Zarr chunk fetches and decompresses the whole
    object, so N dask blocks over one store chunk is N times the bytes and N
    times the CPU - and, over a remote store, N HTTP GETs for one object.
    """
    with pytest.warns(IrisLoadWarning, match="v"):
        result = _align_chunks(
            (2000, 8000), (8000, 8000), (16000, 16000), F8, None, "v"
        )
    assert result == (8000, 8000)


def test_rounding_up_is_capped_at_the_shape():
    # Store chunks small enough to stay under Dask's target, so that this
    # case tests only the capping it is named for. Rounding 2000 up to a
    # multiple of 3000 overshoots the 2500 extent, so the result is 2500.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _align_chunks((2000, 2500), (3000, 2500), (2500, 2500), F8, None, "v")
    assert result == (2500, 2500)


def test_an_unchunked_store_is_left_alone():
    assert _align_chunks((10,), None, (100,), F8, None, "v") == (10,)


def test_an_explicit_chunk_control_setting_wins_but_warns():
    """A deliberate instruction is honoured, on the same terms.

    CHUNK_CONTROL marks the dimensions it set as fixed. Overriding them would
    make the keyword a suggestion.
    """
    with pytest.warns(IrisLoadWarning, match="v"):
        result = _align_chunks(
            (100, 8000), (8000, 8000), (16000, 16000), F8, (True, False), "v"
        )
    assert result[0] == 100


def test_an_oversize_store_chunk_warns():
    """One of the two things worth telling the user about.

    A store chunk larger than Dask's target makes the store's layout the
    binding constraint on memory per task, whatever chunking is asked for.
    """
    with pytest.warns(IrisLoadWarning, match="v"):
        _align_chunks((8000, 8000), (8000, 8000), (16000, 16000), F8, None, "v")


def test_an_ordinary_layout_does_not_warn():
    """And rounding up on its own is not worth telling the user about.

    A year of daily (500, 1000) fields chunked (1, 300, 1000) is an ordinary
    store. Its middle dimension rounds 500 up to 600 and is capped back to
    500, which leaves one block spanning the dimension - a block that reads
    each store chunk exactly once and so subdivides nothing. Warning here
    would warn on every load of a store with nothing wrong with it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _align_chunks(
            (31, 500, 1000), (1, 300, 1000), (365, 500, 1000), F8, None, "v"
        )
    assert result == (31, 500, 1000)


def test_a_small_store_chunk_does_not_warn():
    # Dask's target is the binding constraint here, which is the ordinary
    # case and nothing for the user to act on.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _align_chunks((1024, 1024), (512, 512), (4096, 4096), F8, None, "v")
    assert result == (1024, 1024)


@pytest.mark.parametrize(
    ("chunks", "store_chunks", "shape"),
    [
        ((2000, 8000), (8000, 8000), (16000, 16000)),
        ((1024, 1024), (512, 512), (4096, 4096)),
        ((3000, 10000), (1000, 1000), (10000, 10000)),
    ],
)
def test_the_result_tiles_the_store_grid_exactly(chunks, store_chunks, shape):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IrisLoadWarning)
        result = _align_chunks(chunks, store_chunks, shape, F8, None, "v")
    for size, store_size, extent in zip(result, store_chunks, shape):
        assert size % store_size == 0 or size == extent
