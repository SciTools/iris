# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""The Zarr saver is reachable from :func:`iris.save`."""

import sys

import numpy as np
import pytest

import iris
from iris.cube import Cube, CubeList
import iris.io


@pytest.fixture
def cubes():
    return CubeList(
        [Cube(np.arange(4, dtype=np.float32), var_name=name) for name in ("a", "b")]
    )


class TestFindSaver:
    @pytest.mark.parametrize(
        "target", ["out.zarr", "out.zarr/", "s3://bucket/out.zarr", "OUT.ZARR"]
    )
    def test_the_suffix_finds_a_saver(self, target):
        assert iris.io.find_saver(target) is not None

    def test_it_is_the_zarr_one(self):
        saver = iris.io.find_saver("out.zarr")
        assert saver is iris.io._zarr_save

    def test_a_netcdf_target_is_unaffected(self):
        saver = iris.io.find_saver("out.nc")
        assert saver.__module__ == "iris.fileformats.netcdf.saver"


class TestRegistrationIsLazy:
    def test_finding_the_saver_does_not_import_zarr(self):
        # The optional-dependency promise: `iris.save(cube, "out.pp")` must
        # not import zarr on the way past.
        code = (
            "import sys, iris.io;"
            "iris.io.find_saver('out.zarr');"
            "print('zarr' in sys.modules)"
        )
        import subprocess

        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert completed.stdout.strip() == "False", completed.stderr


class TestSavingACubeList:
    def test_the_whole_list_goes_into_one_store(self, tmp_path, cubes):
        # Without the saver in _CUBELIST_SAVERS, iris.save loops over the
        # cubes, and the second one meets a store that mode="w-" refuses.
        pytest.importorskip("zarr")
        import zarr

        iris.save(cubes, tmp_path / "out.zarr")
        group = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        assert {"a", "b"} <= set(group.array_keys())

    def test_keywords_reach_the_saver(self, tmp_path, cubes):
        pytest.importorskip("zarr")

        with pytest.raises(TypeError, match="chunks"):
            iris.save(cubes, tmp_path / "out.zarr", chunksizes=(2,))
