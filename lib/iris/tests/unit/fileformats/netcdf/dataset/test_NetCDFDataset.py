# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.netcdf._dataset.NetCDFDataset`."""

import warnings

import numpy as np
import pytest

from iris.fileformats.cf.dataset import CFDataset
from iris.fileformats.netcdf import _bytecoding_datasets, _dataset, _thread_safe_nc
from iris.warnings import IrisLoadWarning

from .conftest import SAMPLE_DIMENSIONS, SAMPLE_GLOBALS


@pytest.fixture
def reader(sample_path):
    with _dataset.NetCDFDataset(sample_path) as dataset:
        yield dataset


class TestOpening:
    def test_is_a_cf_dataset(self, reader):
        assert isinstance(reader, CFDataset)

    def test_location_is_the_string_path(self, reader, sample_location):
        assert reader.location == sample_location

    def test_default_mode_is_read(self, reader):
        assert reader.mode == "r"

    def test_starts_open(self, reader):
        assert reader.closed is False

    def test_close_is_idempotent(self, sample_path):
        dataset = _dataset.NetCDFDataset(sample_path)
        dataset.close()
        assert dataset.closed is True
        dataset.close()
        assert dataset.closed is True

    def test_context_manager_closes(self, sample_path):
        with _dataset.NetCDFDataset(sample_path) as dataset:
            assert dataset.closed is False
        assert dataset.closed is True

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            _dataset.NetCDFDataset(tmp_path / "nope.nc")

    def test_decoding_setting_chooses_the_wrapper(self, sample_path):
        with _bytecoding_datasets.DECODE_TO_STRINGS_ON_READ.context(True):
            with _dataset.NetCDFDataset(sample_path) as dataset:
                assert isinstance(dataset.dataset, _bytecoding_datasets.EncodedDataset)
        with _bytecoding_datasets.DECODE_TO_STRINGS_ON_READ.context(False):
            with _dataset.NetCDFDataset(sample_path) as dataset:
                assert not isinstance(
                    dataset.dataset, _bytecoding_datasets.EncodedDataset
                )
                assert isinstance(dataset.dataset, _thread_safe_nc.DatasetWrapper)


class TestContents:
    def test_variables(self, reader, sample_location):
        assert sorted(reader.variables) == ["air_temperature", "height", "label"]
        air = reader.variables["air_temperature"]
        assert isinstance(air, _dataset.NetCDFDatasetVariable)
        assert air.name == "air_temperature"
        assert air.location == sample_location

    def test_variables_are_built_once(self, reader):
        assert (
            reader.variables["air_temperature"] is reader.variables["air_temperature"]
        )

    def test_variables_share_one_write_lock(self, reader):
        # _dask_locks.get_worker_lock() returns a FRESH threading.Lock under
        # the threaded scheduler, so a per-variable call would hand out locks
        # that exclude nothing.
        air = reader.variables["air_temperature"]
        height = reader.variables["height"]
        assert air._write_lock is height._write_lock
        assert air._write_lock is reader.write_lock

    def test_dimensions_are_names_and_lengths(self, reader):
        assert dict(reader.dimensions) == SAMPLE_DIMENSIONS

    def test_unlimited_dimension_reports_its_current_length(self, reader):
        # Nothing was written along "record", so it is zero-length - which is
        # what netCDF4 reports, and what saver.py's membership tests need.
        assert reader.dimensions["record"] == 0
        assert "record" in reader.dimensions

    def test_global_attributes(self, reader):
        assert dict(reader.attributes) == SAMPLE_GLOBALS

    def test_global_attributes_are_built_once(self, reader):
        assert reader.attributes is reader.attributes


class TestBorrowing:
    def test_from_existing_wraps_an_open_dataset(self, sample_path, sample_location):
        raw = _bytecoding_datasets.EncodedDataset(sample_path, mode="r")
        try:
            dataset = _dataset.NetCDFDataset.from_existing(raw)
            assert dataset.dataset is raw
            assert dataset.location == sample_location
            assert sorted(dataset.variables) == [
                "air_temperature",
                "height",
                "label",
            ]
        finally:
            raw.close()

    def test_from_existing_does_not_close_what_it_borrowed(self, sample_path):
        raw = _bytecoding_datasets.EncodedDataset(sample_path, mode="r")
        try:
            dataset = _dataset.NetCDFDataset.from_existing(raw)
            dataset.close()
            assert dataset.closed is True
            # Still usable: the borrower released nothing.
            assert raw.isopen()
        finally:
            raw.close()

    def test_from_existing_wraps_a_bare_netcdf4_dataset(self, sample_path):
        # What the Xarray bridge hands iris.save / CFReader: an object with
        # the netCDF4 API but no thread-safe wrapper around it.
        import netCDF4

        raw = netCDF4.Dataset(sample_path, mode="r")
        try:
            dataset = _dataset.NetCDFDataset.from_existing(raw)
            assert isinstance(dataset.dataset, _bytecoding_datasets.EncodedDataset)
            assert dataset.dataset._contained_instance is raw
        finally:
            raw.close()


class TestAutoChartostring:
    """Iris decodes byte data itself, so netCDF4 must not do it first.

    CFReader turned this off on every dataset it opened (_reader.py:182).
    The dataset now does it, so no caller has to remember.
    """

    def test_turned_off_on_an_opened_dataset(self, sample_path, mocker):
        spy = mocker.spy(_thread_safe_nc.DatasetWrapper, "set_auto_chartostring")
        with _bytecoding_datasets.DECODE_TO_STRINGS_ON_READ.context(False):
            with _dataset.NetCDFDataset(sample_path):
                pass
        spy.assert_called_once_with(mocker.ANY, False)

    def test_turned_off_on_a_borrowed_dataset(self, sample_path, mocker):
        # Spied on the class, not the instance: DatasetWrapper's __setattr__
        # forwards instance-level patching to the contained netCDF4 object
        # instead of shadowing the method, so mocker.spy(raw, ...) cannot
        # observe the call.
        raw = _thread_safe_nc.DatasetWrapper(sample_path, mode="r")
        spy = mocker.spy(_thread_safe_nc.DatasetWrapper, "set_auto_chartostring")
        try:
            _dataset.NetCDFDataset.from_existing(raw)
            spy.assert_called_once_with(raw, False)
        finally:
            raw.close()

    def test_an_encoded_dataset_blocks_it_rather_than_forwarding(self, sample_path):
        # EncodedDataset does its own decoding, so the call is inert there -
        # which is why making it unconditionally is safe.
        with _dataset.NetCDFDataset(sample_path) as dataset:
            assert isinstance(dataset.dataset, _bytecoding_datasets.EncodedDataset)
            with pytest.raises(TypeError, match="not supported"):
                dataset.dataset.set_auto_chartostring(True)


class TestLegacyFormatWarning:
    def test_silent_by_default(self, tmp_path, sample_path):
        legacy = _write_netcdf3(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with _dataset.NetCDFDataset(legacy):
                pass

    def test_warns_when_asked(self, tmp_path):
        legacy = _write_netcdf3(tmp_path)
        with pytest.warns(IrisLoadWarning, match="nccopy"):
            with _dataset.NetCDFDataset(legacy, warn_legacy_format=True):
                pass

    def test_does_not_warn_for_netcdf4(self, sample_path):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with _dataset.NetCDFDataset(sample_path, warn_legacy_format=True):
                pass


def _write_netcdf3(tmp_path):
    path = tmp_path / "legacy.nc"
    dataset = _thread_safe_nc.DatasetWrapper(path, mode="w", format="NETCDF3_CLASSIC")
    dataset.createDimension("time", 2)
    variable = dataset.createVariable("time", "f4", ("time",))
    variable[:] = np.arange(2, dtype="f4")
    dataset.close()
    return path
