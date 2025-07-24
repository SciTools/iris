# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for netcdf loading of attributes with "special" handling."""

# Annoyingly, this import is *not* redundant, as pytest import fails without it.
import warnings

try:
    import iris_grib  # noqa: F401
    from iris_grib.grib_phenom_translation._gribcode import (
        GenericConcreteGRIBCode,
        GRIBCode,
    )
except ImportError:
    iris_grib = None

import numpy as np
import pytest

import iris
from iris.cube import Cube
from iris.fileformats.netcdf._thread_safe_nc import DatasetWrapper as NcDataset
from iris.fileformats.pp import STASH
from iris.warnings import IrisLoadWarning


class LoadTestCommon:
    @pytest.fixture(autouse=True)
    def tmp_filepath(self, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("tmp_nc")
        # We can reuse the same path all over, as it is recreated for each test.
        self.tmp_ncpath = tmp_dir / "tmp.nc"
        yield

    def _check_load_inner(self, iris_name, nc_name, value):
        # quickly create a valid netcdf file with a simple cube in it.
        cube = Cube([1], var_name="x")
        # Save : NB can fail
        with iris.FUTURE.context(save_split_attrs=True):
            iris.save(cube, self.tmp_ncpath)

        # Reopen for updating with netcdf
        ds = NcDataset(self.tmp_ncpath, "r+")
        # Add the test attribute content.
        ds.variables["x"].setncattr(nc_name, value)
        ds.close()
        # Now load back + see what Iris loader makes of the attribute value.
        cube = iris.load_cube(self.tmp_ncpath, "x")
        # NB can be absent -> None result.
        result = cube.attributes.get(iris_name)
        return result

    _LOAD_FAIL_MSG = "Invalid content for managed attribute.*untranslated"


class TestStash(LoadTestCommon):
    def _check_load(self, value, succeed=True):
        # When succeed=False, expect translation fail + return original 'raw' attribute
        return self._check_load_inner(
            iris_name="STASH" if succeed else "um_stash_source",
            nc_name="um_stash_source",
            value=value,
        )

    def test_simple_string(self):
        stash_string = "m01s02i324"
        result = self._check_load(stash_string)
        assert isinstance(result, STASH)
        assert result == STASH(1, 2, 324)

    def test_legacy_name(self):
        # Write using the old legacy name of the attribute.
        # Presumably may still occur in some old files.
        stash_string = "m01s02i324"
        result = self._check_load_inner(
            nc_name="ukmo__um_stash_source", iris_name="STASH", value=stash_string
        )
        assert isinstance(result, STASH)
        assert result == STASH(1, 2, 324)

    def test_dual_names(self):
        # Test the highly unusual case where both names occur.
        result = self._check_load(value="xxx")
        # Modify the file variable to have *both* attributes
        ds = NcDataset(self.tmp_ncpath, "r+")
        var = ds.variables["x"]
        var.um_stash_source = "m1s2i3"
        var.ukmo__um_stash_source = "m2s3i4"
        ds.close()

        # When re-loaded, this should raise a warning.
        msg = "Multiple file attributes would set .*STASH"
        with pytest.warns(UserWarning, match=msg):
            result_cube = iris.load_cube(self.tmp_ncpath, "x")
        result = result_cube.attributes["STASH"]
        assert isinstance(result, STASH)
        assert result == STASH(2, 3, 4)  # because the "legacy" name takes precedence

    def test_alternate_format(self):
        stash_string = "  m1s2i3  "  # slight tolerance in STASH conversion function
        result = self._check_load(stash_string)
        assert isinstance(result, STASH)
        assert result == STASH(1, 2, 3)

    def test_bad_string__fail(self):
        stash_string = "xxx"
        with pytest.warns(IrisLoadWarning, match=self._LOAD_FAIL_MSG):
            result = self._check_load(stash_string, succeed=False)
        assert result == "xxx"

    def test_empty_string__fail(self):
        stash_string = ""
        with pytest.warns(IrisLoadWarning, match=self._LOAD_FAIL_MSG):
            result = self._check_load(stash_string, succeed=False)
        assert result == ""

    def test_numeric_value__fail(self):
        value = 3
        with pytest.warns(IrisLoadWarning, match=self._LOAD_FAIL_MSG):
            result = self._check_load(value, succeed=False)
        # written directly, comes back as an array scalar of int64
        assert result.dtype == np.int64
        assert result == 3

    def test_tuple_value__fail(self):
        # As they cast to arrays, we expect lists to behave the same
        value = (2, 3, 7)
        with pytest.warns(IrisLoadWarning, match=self._LOAD_FAIL_MSG):
            result = self._check_load(value, succeed=False)
        # written directly, comes back as an array of int64, shape (3,)
        assert result.dtype == np.int64
        assert result.shape == (3,)
        assert np.all(result == value)


class TestUkmoProcessFlags(LoadTestCommon):
    def _check_load(self, value):
        return self._check_load_inner(
            "ukmo__process_flags", "ukmo__process_flags", value
        )

    def test_simple_string(self):
        flag_string = "one two"
        result = self._check_load(flag_string)
        assert isinstance(result, tuple)
        assert result == ("one", "two")

    def test_alternate_string(self):
        string = " one two  t   hree "  # merges multiple separators
        result = self._check_load(string)
        assert result == ("", "one", "two", "", "t", "", "", "hree", "")

    def test_special_elements(self):
        string = "one <EMPTY> two_three"
        result = self._check_load(string)
        assert result == ("one", "", "two three")

    def test_empty_string(self):
        string = ""
        # This is NOT an error
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = self._check_load(string)
        assert result == ()

    def test_numeric_value__fail(self):
        value = 3
        # Note: not a failure, because conversion forces to a string ...
        result = self._check_load(value)
        # ... but the answer isn't what you might want.
        assert result == ("3",)

    def test_tuple_value__fail(self):
        value = (2, 3, 7)
        result = self._check_load(value)
        # force to a string, the result is not pretty !
        assert result == ("[2", "3", "7]")


@pytest.mark.skipif(iris_grib is None, reason="iris_grib is not available")
class TestGribParam(LoadTestCommon):
    def _check_load(self, value):
        return self._check_load_inner("GRIB_PARAM", "GRIB_PARAM", value)

    def test_standard_string(self):
        string = "GRIBCode(edition=1, table_version=2, centre_number=3, number=4)"
        result = self._check_load(string)
        assert isinstance(result, GenericConcreteGRIBCode)
        assert result == GRIBCode(1, 2, 3, 4)

    def test_confused_string(self):
        string = "GRIBCode(edition=1, centre=2, nonsense=3, table_version=4)"
        result = self._check_load(string)
        assert isinstance(result, GenericConcreteGRIBCode)
        assert result == GRIBCode(1, 2, 3, 4)

    def test_alternate_format_string(self):
        string = "grib(1, 2, 3, 4)"
        result = self._check_load(string)
        assert isinstance(result, GenericConcreteGRIBCode)
        assert result == GRIBCode(1, 2, 3, 4)

    def test_minimal_string(self):
        string = "1 2 3 4"
        result = self._check_load(string)
        assert isinstance(result, GenericConcreteGRIBCode)
        assert result == GRIBCode(1, 2, 3, 4)

    def test_invalid_string__fail(self):
        string = "grib(1, 2, 3)"
        with pytest.warns(UserWarning, match=self._LOAD_FAIL_MSG):
            result = self._check_load(string)
        assert result == string

    def test_junk_string__fail(self):
        string = "xxx"
        with pytest.warns(UserWarning, match=self._LOAD_FAIL_MSG):
            result = self._check_load(string)
        assert result == string

    def test_empty_string__fail(self):
        string = ""
        with pytest.warns(UserWarning, match=self._LOAD_FAIL_MSG):
            result = self._check_load(string)
        assert result == string

    def test_numeric_value__fail(self):
        value = 3
        with pytest.warns(UserWarning, match=self._LOAD_FAIL_MSG):
            result = self._check_load(value)
        # written directly, comes back as an array scalar of int64
        assert result.dtype == np.int64
        assert result == 3

    def test_tuple_value__fail(self):
        # As they cast to arrays, we expect lists to behave the same
        value = (2, 3, 7)
        with pytest.warns(UserWarning, match=self._LOAD_FAIL_MSG):
            result = self._check_load(value)
        # written directly, comes back as an array of int64, shape (3,)
        assert result.dtype == np.int64
        assert result.shape == (3,)
        assert np.all(result == value)
