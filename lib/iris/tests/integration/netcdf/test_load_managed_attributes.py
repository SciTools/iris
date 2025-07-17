# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for netcdf loading of attributes with "special" handling."""

import warnings

from iris_grib.grib_phenom_translation._gribcode import (
    GenericConcreteGRIBCode,
    GRIBCode,
)
import netCDF4 as nc
import numpy as np
import pytest

import iris
from iris.cube import Cube
from iris.fileformats.pp import STASH


@pytest.fixture(autouse=True, scope="session")
def iris_futures():
    iris.FUTURE.save_split_attrs = True


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
        iris.save(cube, self.tmp_ncpath)
        # Reopen for updating with netcdf
        ds = nc.Dataset(self.tmp_ncpath, "r+")
        # Add the test attribute content.
        ds.variables["x"].setncattr(nc_name, value)
        ds.close()
        # Now load back + see what Iris loader makes of the attribute value.
        cube = iris.load_cube(self.tmp_ncpath, "x")
        # NB can be absent -> None result.
        result = cube.attributes.get(iris_name)
        return result

    _LOAD_FAIL_MSG = "Invalid content for attribute.* set to.* untranslated raw value"


class TestStash(LoadTestCommon):
    def _check_load(self, value):
        return self._check_load_inner("STASH", "um_stash_source", value)

    def test_simple_object(self):
        stash_string = "m01s02i324"
        result = self._check_load(stash_string)
        assert isinstance(result, STASH)
        assert result == STASH(1, 2, 324)

    def test_alternate_format(self):
        stash_string = "  m1s2i3  "  # slight tolerance in STASH conversion function
        result = self._check_load(stash_string)
        assert isinstance(result, STASH)
        assert result == STASH(1, 2, 3)

    def test_bad_string__fail(self):
        stash_string = "xxx"
        with pytest.warns(UserWarning, match=self._LOAD_FAIL_MSG):
            result = self._check_load(stash_string)
        assert result == "xxx"

    def test_empty_string__fail(self):
        stash_string = ""
        with pytest.warns(UserWarning, match=self._LOAD_FAIL_MSG):
            result = self._check_load(stash_string)
        assert result == ""

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
