# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""The warning raised when a stored chunk is larger than Dask's target.

Iris sizes Dask chunks to Dask's target and does not round back up to the
store's own chunking, so a large stored chunk is divided and then fetched once
per division. The warning is what tells a caller there is a `CHUNK_CONTROL`
decision to make. It has to reach both storage backends, or netCDF and Zarr
would behave differently over physics they share.
"""

import warnings

import dask.config
import numpy as np
import pytest

from iris.fileformats.cf import CFDataVariable
from iris.fileformats.cf.loader import CHUNK_CONTROL, _chunks_from_chunk_control
from iris.warnings import IrisLoadWarning

zarr = pytest.importorskip("zarr")

from iris.fileformats.netcdf._dataset import NetCDFDataset  # noqa: E402
from iris.fileformats.zarr._dataset import ZarrDataset  # noqa: E402

SHAPE = (64,)
#: Stored chunk of 32 float64 values, which is 256 bytes.
BIG_CHUNK = (32,)
#: Stored chunk of 4 float64 values, which is 32 bytes.
SMALL_CHUNK = (4,)
#: Sits between the two, so one choice of stored chunk crosses it and the
#: other does not.
TARGET = "128B"


def build_netcdf(path, chunks):
    dataset = NetCDFDataset(str(path), mode="w")
    dataset.create_dimension("t", SHAPE[0])
    dataset.create_variable("v", np.dtype("f8"), ("t",), chunksizes=chunks)
    dataset.close()
    return NetCDFDataset(str(path))


def build_zarr(path, chunks):
    group = zarr.open_group(store=path, mode="w", zarr_format=3)
    group.create_array(
        "v", shape=SHAPE, dtype="f8", chunks=chunks, dimension_names=("t",)
    )
    return ZarrDataset(str(path))


@pytest.fixture(params=[build_netcdf, build_zarr], ids=["netcdf", "zarr"])
def build(request, tmp_path):
    """Return a callable giving the `CFDataVariable` for a stored chunking."""
    suffix = ".nc" if request.param is build_netcdf else ".zarr"

    def make(chunks):
        dataset = request.param(tmp_path / f"chunked{suffix}", chunks)
        return CFDataVariable("v", dataset.variables["v"])

    return make


def test_a_stored_chunk_over_the_target_warns(build):
    cf_var = build(BIG_CHUNK)
    with dask.config.set({"array.chunk-size": TARGET}):
        with pytest.warns(IrisLoadWarning, match="CHUNK_CONTROL"):
            _chunks_from_chunk_control(cf_var)


def test_the_warning_names_the_variable(build):
    cf_var = build(BIG_CHUNK)
    with dask.config.set({"array.chunk-size": TARGET}):
        with pytest.warns(IrisLoadWarning, match="v$"):
            _chunks_from_chunk_control(cf_var)


def test_a_stored_chunk_under_the_target_is_quiet(build):
    cf_var = build(SMALL_CHUNK)
    with dask.config.set({"array.chunk-size": TARGET}):
        with warnings.catch_warnings():
            warnings.simplefilter("error", IrisLoadWarning)
            _chunks_from_chunk_control(cf_var)


def test_a_pinned_chunking_is_not_second_guessed(build):
    """A caller who has set the chunking has already made this decision."""
    cf_var = build(BIG_CHUNK)
    with dask.config.set({"array.chunk-size": TARGET}):
        with warnings.catch_warnings():
            warnings.simplefilter("error", IrisLoadWarning)
            with CHUNK_CONTROL.set("v", t=8):
                _chunks_from_chunk_control(cf_var)
