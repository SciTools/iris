# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for :class:`iris.fileformats.cf.CFReader` reading a real file.

The rest of the CFReader tests drive it with mocks, which is the right shape
for its classification logic and the wrong shape for the question here: what
type the reader actually hands out, and what a load does with it. These use a
small file on disk instead.
"""

import numpy as np
import pytest

import iris
from iris.fileformats.cf import CFReader
from iris.fileformats.cf.dataset import CFDatasetVariable
from iris.fileformats.netcdf import _dataset, _thread_safe_nc

UNREAD_COMMENT = "an attribute nothing in Iris reads"


@pytest.fixture
def sample_path(tmp_path):
    """Write a small CF file: a data variable, a coordinate and its bounds."""
    path = tmp_path / "sample.nc"
    dataset = _thread_safe_nc.DatasetWrapper(path, mode="w")
    dataset.title = "a sample file"
    dataset.createDimension("time", 3)
    dataset.createDimension("bnds", 2)

    time = dataset.createVariable("time", "f8", ("time",))
    time.standard_name = "time"
    time.units = "days since 1970-01-01"
    time.bounds = "time_bnds"
    time.comment = UNREAD_COMMENT
    time[:] = np.arange(3, dtype="f8")

    bounds = dataset.createVariable("time_bnds", "f8", ("time", "bnds"))
    bounds[:] = np.zeros((3, 2))

    air = dataset.createVariable("air", "f4", ("time",))
    air.standard_name = "air_temperature"
    air.units = "K"
    air.coordinates = "time"
    air.comment = UNREAD_COMMENT
    air[:] = np.arange(3, dtype="f4")

    dataset.close()
    return path


@pytest.fixture
def char_path(tmp_path):
    """Write a small CF file with a character auxiliary coordinate.

    A separate fixture from ``sample_path``: adding a char variable there
    would change what ``test_only_unread_attributes_reach_the_cube`` and its
    neighbours load.
    """
    path = tmp_path / "char.nc"
    dataset = _thread_safe_nc.DatasetWrapper(path, mode="w")
    dataset.createDimension("x", 3)
    dataset.createDimension("strlen", 3)

    labels = dataset.createVariable("labels", "S1", ("x", "strlen"))
    labels._Encoding = "utf-8"
    labels[:] = np.array([list(s) for s in ("aaa", "bbb", "ccc")], dtype="S1")

    temp = dataset.createVariable("temp", "f4", ("x",))
    temp.standard_name = "air_temperature"
    temp.units = "K"
    temp.coordinates = "labels"
    temp[:] = np.arange(3, dtype="f4")

    dataset.close()
    return path


class TestABorrowedBareDataset:
    """A behaviour change from before this branch.

    Before this branch, ``CFReader`` stored a borrowed dataset untouched
    (``self._dataset = file_source``), so a bare ``netCDF4.Dataset`` - what
    the Xarray bridge hands to ``iris.load``, per ``loader.py``'s docstring -
    kept its character data as raw bytes. ``NetCDFDataset.from_existing`` now
    wraps anything lacking ``THREAD_SAFE_FLAG`` in an ``EncodedDataset``
    regardless of the caller's own decoding preference, so the same bare
    dataset's character data is decoded to Python strings instead. This is
    deliberate - the new behaviour is better, and a char auxiliary coordinate
    that used to be silently dropped by ``iris.load`` now loads - but it is a
    genuine, previously-undocumented change on a public entry point, so this
    pins it.
    """

    def test_char_data_is_decoded(self, char_path):
        # A genuinely bare, unwrapped dataset - not a DatasetWrapper, not an
        # EncodedDataset - is the point of this test.
        import netCDF4

        raw = netCDF4.Dataset(char_path, mode="r")
        try:
            with CFReader(raw) as reader:
                labels = reader.cf_group["labels"]
                assert labels.dtype.kind == "U"
                assert labels.shape == (3,)
                np.testing.assert_array_equal(labels[:], ["aaa", "bbb", "ccc"])
        finally:
            raw.close()


class TestTheSwap:
    def test_the_reader_owns_a_netcdf_dataset(self, sample_path):
        with CFReader(str(sample_path)) as reader:
            assert isinstance(reader._dataset, _dataset.NetCDFDataset)

    def test_cf_data_is_a_cf_dataset_variable(self, sample_path):
        # The whole point of the PR: nothing downstream of here needs to know
        # the file is netCDF.
        with CFReader(str(sample_path)) as reader:
            assert isinstance(reader.cf_group["air"].cf_data, CFDatasetVariable)

    def test_attributes_come_from_the_file(self, sample_path):
        with CFReader(str(sample_path)) as reader:
            air = reader.cf_group["air"]
            assert air.attributes["units"] == "K"
            assert air.units == "K"
            assert air.dimensions == ("time",)

    def test_global_attributes_come_from_the_dataset(self, sample_path):
        with CFReader(str(sample_path)) as reader:
            assert reader.cf_group.global_attributes["title"] == "a sample file"

    def test_filename_is_the_variables_location(self, sample_path):
        with CFReader(str(sample_path)) as reader:
            assert reader.cf_group["air"].filename == str(sample_path)

    def test_the_bounds_variable_was_classified(self, sample_path):
        # identify() now reads through .attributes, so a miss here means the
        # classification pass lost sight of the file's attributes entirely.
        with CFReader(str(sample_path)) as reader:
            assert "time_bnds" in reader.cf_group.bounds

    def test_a_borrowed_dataset_is_used_and_not_closed(self, sample_path):
        raw = _thread_safe_nc.DatasetWrapper(sample_path, mode="r")
        try:
            with CFReader(raw) as reader:
                assert reader.cf_group["air"].units == "K"
            assert raw.isopen()
        finally:
            raw.close()


class TestAttributesAreASnapshot:
    """Finding F11.

    ``NetCDFDatasetVariable.attributes`` writes through to the file, which is
    right for the saver and wrong for the reader: CFReader synthesises a
    "bounds" link during load, on a file it opened read-only. CFVariable
    therefore takes a copy.
    """

    def test_a_write_does_not_reach_the_file(self, sample_path):
        with CFReader(str(sample_path)) as reader:
            air = reader.cf_group["air"]
            air.attributes["bounds"] = "invented"

            assert "bounds" not in air.cf_data.attributes


class TestAttributeTrackingAcrossReset:
    """Review Focus 3.

    CFReader reads attributes while classifying variables, then calls
    ``cf_attrs_reset()`` so the rules start from a clean record. Until this
    PR, ``__getattr__`` cached the value on the instance, so a read after the
    reset found the cache and was never recorded - and an attribute Iris had
    in fact consumed was still reported unused, and so was copied onto the
    cube as if the file had volunteered it.
    """

    def test_the_reader_resets_what_it_read_while_classifying(self, sample_path):
        with CFReader(str(sample_path)) as reader:
            time = reader.cf_group["time"]
            # CFBoundaryVariable.identify() read "bounds" during __init__.
            # Ask through .untracked, so that asking does not itself record.
            assert "bounds" in time.attributes.untracked
            assert dict(time.cf_attrs_used()) == dict(time.cf_attrs_ignored())

    def test_a_read_after_the_reset_is_recorded(self, sample_path):
        with CFReader(str(sample_path)) as reader:
            time = reader.cf_group["time"]

            assert time.units == "days since 1970-01-01"
            assert "units" in dict(time.cf_attrs_used())
            assert "units" not in dict(time.cf_attrs_unused())

    def test_an_unread_attribute_reaches_the_cube(self, sample_path):
        # Named for what this pins, not the stronger claim it does not: see
        # the comment below on the "only" direction.
        cube = iris.load_cube(str(sample_path))

        # "title" is a global attribute: build_and_add_global_attributes
        # copies cf_group.global_attributes onto the cube unconditionally,
        # untouched by the per-variable used/unused tracking this test
        # otherwise pins - so it is expected here alongside the coordinate
        # variable's genuinely-unread "comment".
        assert cube.attributes == {"title": "a sample file", "comment": UNREAD_COMMENT}
        assert cube.coord("time").attributes == {"comment": UNREAD_COMMENT}

        # Known gap: this does not pin the complementary "only" direction -
        # that *no* read attribute reaches the cube. Every attribute this
        # fixture's variables read ("units", "standard_name", "coordinates",
        # "bounds") is also in saver.py's _CF_ATTRS, so _add_unused_attributes
        # (loader.py) filters them out before tracking is ever consulted;
        # mutating cf_attrs_unused() to cf_attrs_used() fails this test, but
        # mutating it to attributes.untracked.items() - tracking made blind,
        # so everything counts as unused - still passes. Pinning "only" would
        # need a non-_CF_ATTRS attribute the load rules genuinely read and
        # record, e.g. a UGRID or grid-mapping attribute, which means adding
        # a mesh or a grid mapping to a fixture whose subject is attribute
        # tracking, and would change what the file loads into.
