# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.dataset.TrackedAttributes`."""

import pytest

from iris.fileformats.cf.dataset import TrackedAttributes

# The set CFVariable seeds tracking with: attributes netCDF4 handles itself,
# and which are therefore "used" before anyone reads them.
IGNORED = ("_FillValue", "add_offset", "missing_value", "scale_factor")


@pytest.fixture
def source():
    return {"units": "K", "standard_name": "air_temperature", "_FillValue": -999}


@pytest.fixture
def tracked(source):
    return TrackedAttributes(source, ignored=IGNORED)


class TestWhatRecordsARead:
    def test_starts_with_only_the_present_ignored_names(self, tracked):
        # "scale_factor" is ignored but absent, so it is not seeded.
        assert tracked.read == frozenset(["_FillValue"])
        assert tracked.unread == frozenset(["units", "standard_name"])

    def test_getitem_records(self, tracked):
        assert tracked["units"] == "K"
        assert tracked.read == frozenset(["_FillValue", "units"])
        assert tracked.unread == frozenset(["standard_name"])

    def test_get_records(self, tracked):
        assert tracked.get("units") == "K"
        assert "units" in tracked.read

    def test_contains_records_a_hit(self, tracked):
        # hasattr(cf_var, name) goes through __getattr__ today and marks the
        # attribute used; the mapping form has to do the same.
        assert "units" in tracked
        assert "units" in tracked.read

    def test_contains_does_not_record_a_miss(self, tracked):
        assert "nonesuch" not in tracked
        assert tracked.read == frozenset(["_FillValue"])

    def test_getitem_of_a_missing_key_raises_before_recording(self, tracked):
        with pytest.raises(KeyError, match="nonesuch"):
            tracked["nonesuch"]
        assert tracked.read == frozenset(["_FillValue"])

    def test_get_of_a_missing_key_records_nothing(self, tracked):
        assert tracked.get("nonesuch") is None
        assert tracked.read == frozenset(["_FillValue"])


class TestWhatDoesNotRecordARead:
    def test_untracked_getitem(self, tracked):
        assert tracked.untracked["units"] == "K"
        assert tracked.read == frozenset(["_FillValue"])

    def test_untracked_contains(self, tracked):
        # The form helpers.py needs for its deliberately-unmarked flag probe.
        assert "units" in tracked.untracked
        assert tracked.read == frozenset(["_FillValue"])

    def test_untracked_is_read_only(self, tracked):
        with pytest.raises(TypeError):
            tracked.untracked["units"] = "m"

    def test_iteration(self, tracked):
        assert sorted(tracked) == ["_FillValue", "standard_name", "units"]
        assert tracked.read == frozenset(["_FillValue"])

    def test_keys_values_items(self, tracked):
        assert sorted(tracked.keys()) == ["_FillValue", "standard_name", "units"]
        assert sorted(tracked.values(), key=str) == [-999, "K", "air_temperature"]
        assert dict(tracked.items())["units"] == "K"
        assert tracked.read == frozenset(["_FillValue"])

    def test_len(self, tracked):
        assert len(tracked) == 3
        assert tracked.read == frozenset(["_FillValue"])


class TestMutation:
    def test_setitem_writes_through_and_does_not_record(self, tracked, source):
        tracked["comment"] = "hello"
        assert source["comment"] == "hello"
        assert tracked.read == frozenset(["_FillValue"])

    def test_delitem_writes_through_and_forgets_the_read(self, tracked, source):
        _ = tracked["units"]
        del tracked["units"]
        assert "units" not in source
        assert tracked.read == frozenset(["_FillValue"])


class TestReset:
    def test_reset_returns_to_the_present_ignored_names(self, tracked):
        _ = tracked["units"]
        _ = tracked["standard_name"]
        assert tracked.read == frozenset(["_FillValue", "units", "standard_name"])
        tracked.reset()
        assert tracked.read == frozenset(["_FillValue"])

    def test_reset_sees_attributes_added_since_construction(self, tracked):
        tracked["scale_factor"] = 2.0
        tracked.reset()
        assert tracked.read == frozenset(["_FillValue", "scale_factor"])


class TestEmptySource:
    def test_a_variable_with_no_attributes_is_not_an_error(self):
        tracked = TrackedAttributes({}, ignored=IGNORED)
        assert tracked.read == frozenset()
        assert tracked.unread == frozenset()
        assert len(tracked) == 0
        assert "units" not in tracked
