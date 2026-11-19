# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the dataset :class:`iris.fileformats.netcdf.saver.Saver` owns."""

import numpy as np
import pytest

from iris.fileformats.netcdf import _thread_safe_nc as threadsafe_nc
from iris.fileformats.netcdf._dataset import NetCDFDataset
from iris.fileformats.netcdf.saver import Saver


class TestOwnedDataset:
    @pytest.fixture
    def saver(self, tmp_path):
        with Saver(tmp_path / "test.nc", "NETCDF4") as saver:
            yield saver

    def test_dataset_is_a_netcdf_cfdataset(self, saver):
        assert isinstance(saver._dataset, NetCDFDataset)

    def test_filepath_is_still_a_path(self, saver, tmp_path):
        # Public API: Saver.filepath has always been a Path for a local file.
        assert saver.filepath == (tmp_path / "test.nc").absolute()

    def test_the_write_lock_is_the_datasets_own(self, saver):
        # Every variable of the dataset holds this same lock. A second lock
        # would exclude nothing - see NetCDFDataset.write_lock.
        assert saver.file_write_lock is saver._dataset.write_lock

    def test_format_reaches_the_file(self, tmp_path):
        # Not through the context manager: __exit__ syncs, and a netCDF3 file
        # with nothing written to it is still in define mode, where sync()
        # raises. That is as true of the saver before this change as after.
        saver = Saver(tmp_path / "classic.nc", "NETCDF3_CLASSIC")
        try:
            assert saver._dataset.dataset.file_format == "NETCDF3_CLASSIC"
        finally:
            saver._dataset.close()

    def test_exit_closes_an_owned_dataset(self, tmp_path):
        with Saver(tmp_path / "test.nc", "NETCDF4") as saver:
            pass
        assert saver._dataset.closed


class TestBorrowedDataset:
    @pytest.fixture
    def raw(self, tmp_path):
        dataset = threadsafe_nc.DatasetWrapper(
            tmp_path / "user.nc", mode="w", format="NETCDF4"
        )
        yield dataset
        if dataset.isopen():
            dataset.close()

    def test_dataset_is_a_netcdf_cfdataset(self, raw):
        with Saver(raw, "NETCDF4", compute=False) as saver:
            assert isinstance(saver._dataset, NetCDFDataset)

    def test_exit_leaves_a_borrowed_dataset_open(self, raw):
        with Saver(raw, "NETCDF4", compute=False) as saver:
            pass
        assert raw.isopen()
        assert not saver._dataset.closed

    def test_complete_refuses_while_the_file_is_open(self, raw):
        with Saver(raw, "NETCDF4", compute=False) as saver:
            pass
        with pytest.raises(ValueError, match="until its dataset is closed"):
            saver.complete()

    def test_complete_sees_a_dataset_closed_behind_its_back(self, raw):
        # The caller owns the dataset and closes it themselves, so the flag
        # NetCDFDataset.close() sets is never set. complete() has to ask the
        # file, not the flag.
        with Saver(raw, "NETCDF4", compute=False) as saver:
            pass
        raw.close()
        assert saver._dataset.closed
        saver.complete()  # must not raise


class TestWritingThroughTheDataset:
    """The four verbs, through CFDataset rather than netCDF4."""

    @pytest.fixture
    def saver(self, tmp_path):
        with Saver(tmp_path / "test.nc", "NETCDF4") as saver:
            yield saver

    def test_create_dimension(self, saver):
        saver._dataset.create_dimension("x", 3)
        assert saver._dataset.dimensions["x"] == 3

    def test_create_variable_returns_a_cf_dataset_variable(self, saver):
        from iris.fileformats.netcdf._dataset import NetCDFDatasetVariable

        saver._dataset.create_dimension("x", 3)
        variable = saver._dataset.create_variable("a", np.dtype("f4"), ("x",))
        assert isinstance(variable, NetCDFDatasetVariable)
        assert saver._dataset.variables["a"] is variable
