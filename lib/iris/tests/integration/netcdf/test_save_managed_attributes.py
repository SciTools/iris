# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for netcdf saving of attributes with "special" handling."""

try:
    import iris_grib
    from iris_grib.grib_phenom_translation._gribcode import GRIBCode
except ImportError:
    iris_grib = None

import pytest

import iris
from iris.cube import Cube
from iris.fileformats.netcdf._thread_safe_nc import DatasetWrapper as NcDataset
from iris.fileformats.pp import STASH


class SaveTestCommon:
    @pytest.fixture(autouse=True)
    def tmp_filepath(self, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("tmp_nc")
        # We can reuse the same path all over, as it is recreated for each test.
        self.tmp_ncpath = tmp_dir / "tmp.nc"
        yield

    def _check_save_inner(self, iris_name, nc_name, value):
        cube = Cube([1], var_name="x", attributes={iris_name: value})
        # Save : NB can fail
        with iris.FUTURE.context(save_split_attrs=True):
            iris.save(cube, self.tmp_ncpath)

        ds = NcDataset(self.tmp_ncpath)
        result = ds.variables["x"].getncattr(nc_name)
        return result


class TestStash(SaveTestCommon):
    def _check_save(self, value, succeed=True):
        # If succeed=False, expect failed translation writes a "STASH" attribute
        return self._check_save_inner(
            iris_name="STASH",
            nc_name="um_stash_source" if succeed else "STASH",
            value=value,
        )

    def test_simple_object(self):
        stash = STASH(1, 2, 324)
        result = self._check_save(stash)
        assert result == "m01s02i324"

    def test_simple_string(self):
        stash_str = "m1s2i324"
        result = self._check_save(stash_str)
        assert result == "m01s02i324"

    def test_bad_string__fail(self):
        bad_str = "xxx"
        with pytest.warns(UserWarning, match="Invalid value in managed.* attribute"):
            result = self._check_save(bad_str, succeed=False)
        assert result == "xxx"

    def test_empty_string__fail(self):
        with pytest.warns(UserWarning, match="Invalid value in managed.* attribute"):
            result = self._check_save("", succeed=False)
        assert result == ""

    def test_bad_object__fail(self):
        with pytest.warns(UserWarning, match="Invalid value in managed.* attribute"):
            result = self._check_save({}, succeed=False)
        assert result == "{}"

    def test_none_object__fail(self):
        with pytest.warns(UserWarning, match="Invalid value in managed.* attribute"):
            result = self._check_save(None, succeed=False)
        assert result == "None"


class TestUkmoProcessFlags(SaveTestCommon):
    def _check_save(self, value):
        return self._check_save_inner(
            "ukmo__process_flags", "ukmo__process_flags", value
        )

    def test_simple_object(self):
        flags = ("one", "two")
        result = self._check_save(flags)
        assert result == "one two"

    def test_simple_string(self):
        string = "one two three"
        result = self._check_save(string)
        assert result == string

    def test_empty_tuple(self):
        obj = ()
        result = self._check_save(obj)
        assert result == ""

    def test_string_w_spaces(self):
        obj = ("one", "two three")
        result = self._check_save(obj)
        assert result == "one two_three"

    def test_string_w_underscores(self):
        obj = ("one", "two_three")
        result = self._check_save(obj)
        assert result == "one two_three"

    def test_tuple_w_empty_string(self):
        obj = ("one", "", "two")
        result = self._check_save(obj)
        assert result == "one <EMPTY> two"

    def test_bad_object(self):
        obj = {}
        with pytest.warns(UserWarning, match="Invalid value in managed.* attribute"):
            result = self._check_save(obj)
        assert result == "{}"

    def test_none_object(self):
        obj = None
        with pytest.warns(UserWarning, match="Invalid value in managed.* attribute"):
            result = self._check_save(obj)
        assert result == "None"


@pytest.mark.skipif(iris_grib is None, reason="iris_grib is not available")
class TestGribParam(SaveTestCommon):
    def _check_save(self, value):
        return self._check_save_inner("GRIB_PARAM", "GRIB_PARAM", value)

    def test_simple_object(self):
        code = GRIBCode(1, 2, 3, 4)
        result = self._check_save(code)
        assert (
            result == "GRIBCode(edition=1, table_version=2, centre_number=3, number=4)"
        )

    def test_simple_string(self):
        code_string = "1, 2, 3,,,  4"  # the converter is highly tolerant
        result = self._check_save(code_string)
        assert (
            result == "GRIBCode(edition=1, table_version=2, centre_number=3, number=4)"
        )

    _encode_fail_msg = (
        r"Invalid value in managed.* attribute.* set to raw \(string\) value"
    )

    def test_bad_string_toofew__fail(self):
        code_string = "1, 2,  3"
        with pytest.warns(UserWarning, match=self._encode_fail_msg):
            result = self._check_save(code_string)
        assert result == "1, 2,  3"

    def test_bad_string_junk__fail(self):
        code_string = "xxx"
        with pytest.warns(UserWarning, match=self._encode_fail_msg):
            result = self._check_save(code_string)
        assert result == "xxx"

    def test_bad_object__fail(self):
        obj = {}
        with pytest.warns(UserWarning, match=self._encode_fail_msg):
            result = self._check_save(obj)
        assert result == "{}"

    def test_none_object__fail(self):
        obj = None
        with pytest.warns(UserWarning, match=self._encode_fail_msg):
            result = self._check_save(obj)
        assert result == "None"
