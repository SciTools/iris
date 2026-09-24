# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFVariable`."""

import numpy as np
import pytest

from iris.fileformats import cf as cf
from iris.fileformats.cf import _variables


class CFVariableSub(cf.CFVariable):
    """A subclass of CFVariable for testing purposes."""

    def identify(self, variables, ignore=None, target=None, warn=True):
        return super().identify(variables, ignore=ignore, target=target, warn=warn)


def make_nc_var(mocker):
    nc_var = mocker.MagicMock()
    nc_var.ncattrs.return_value = ["coordinates", "standard_name", "_FillValue"]
    nc_var.getncattr.side_effect = {
        "coordinates": "x y",
        "standard_name": "air_temperature",
        "_FillValue": -999,
    }.__getitem__
    nc_var.coordinates = "x y"
    nc_var.standard_name = "air_temperature"
    nc_var.dimensions = ("time", "lat")
    nc_var.__len__.return_value = 4
    nc_var.__getitem__.return_value = "payload"
    nc_var.group.return_value.filepath.return_value = "/tmp/file.nc"

    return nc_var


@pytest.fixture
def nc_var(mocker):
    return make_nc_var(mocker)


@pytest.fixture
def nc_var_without_group(nc_var):
    del nc_var.group

    return nc_var


@pytest.fixture
def nc_vars(mocker):
    # Three is the maximum number of independent mock variables needed in one test.
    return tuple(make_nc_var(mocker) for _ in range(3))


class TestInit:
    def test_records_filename_from_group(self, nc_var):
        cf_var = CFVariableSub("foo", nc_var)

        assert cf_var.filename == "/tmp/file.nc"
        assert cf_var.cf_name == "foo"
        assert cf_var.cf_data is nc_var
        assert cf_var.cf_group is None
        assert cf_var.cf_terms_by_root == {}
        assert cf_var._to_be_promoted is False

    def test_falls_back_to_unknown_filename_without_group(self, nc_var_without_group):
        cf_var = CFVariableSub("foo", nc_var_without_group)

        assert cf_var.filename == "<unknown_filename>"


class TestIdentifyCommon:
    def test_handles_defaults_and_target_selection(self):
        variables = {"a": object(), "b": object()}

        ignore, target = CFVariableSub._identify_common(variables, None, None)
        assert ignore == []
        assert target is variables

        ignore, target = CFVariableSub._identify_common(variables, ["a"], "b")
        assert ignore == ["a"]
        assert target == {"b": variables["b"]}

    def test_raises_for_unknown_target(self):
        with pytest.raises(ValueError, match="Cannot identify unknown target"):
            CFVariableSub._identify_common({"a": object()}, None, "missing")

    def test_raises_for_invalid_target_type(self):
        with pytest.raises(TypeError, match="Expect a target CF-netCDF variable name"):
            CFVariableSub._identify_common({"a": object()}, None, object())


class TestSpans:
    def test_scalar_dimension_always_true(self, mocker, nc_var):
        nc_var.dimensions = (_variables._NCZARR_SCALAR_DIMENSION,)
        cf_var = CFVariableSub("scalar", nc_var)

        other = mocker.MagicMock()
        other.dimensions = ("time",)

        assert cf_var.spans(other)

    def test_is_subset_check(self, mocker, nc_vars):
        lhs_nc_var, other_nc_var, _ = nc_vars
        lhs_nc_var.dimensions = ("time",)
        lhs = CFVariableSub("lhs", lhs_nc_var)

        rhs = mocker.MagicMock()
        rhs.dimensions = ("time", "lat")

        assert lhs.spans(rhs)

        other_nc_var.dimensions = ("height",)
        other = CFVariableSub("other", other_nc_var)
        assert not other.spans(rhs)


class TestComparisonAndRepresentation:
    def test_equality_inequality_and_hash_by_name(self, nc_vars):
        first, second, third = nc_vars
        one = CFVariableSub("same", first)
        two = CFVariableSub("same", second)
        other = CFVariableSub("different", third)

        assert one == two
        assert one != other
        assert hash(one) == hash(two)
        assert hash(one) != hash(other)

    def test_repr_contains_class_name_name_and_data_repr(self, nc_var):
        cf_var = CFVariableSub("foo", nc_var)

        assert repr(cf_var) == f"CFVariableSub('foo', {nc_var!r})"


class TestAttributeAccess:
    def test_reads_are_not_cached_on_the_instance(self, nc_var):
        # The setattr cache is gone. It made a re-read after cf_attrs_reset()
        # invisible to attribute tracking, which decided which attributes
        # reached the cube - see TestReadAfterReset below.
        cf_var = CFVariableSub("foo", nc_var)

        assert "coordinates" not in cf_var.__dict__
        assert cf_var.coordinates == "x y"
        assert "coordinates" not in cf_var.__dict__
        assert cf_var.coordinates == "x y"
        assert "coordinates" not in cf_var.__dict__

    def test_attributes_are_read_from_the_file_once(self, nc_var):
        # Materialised at construction, so repeated reads cost nothing and
        # cf_attrs_unused() does not have to go back to the file to answer.
        cf_var = CFVariableSub("foo", nc_var)
        assert nc_var.ncattrs.call_count == 1

        _ = cf_var.coordinates
        _ = cf_var.standard_name
        assert nc_var.ncattrs.call_count == 1
        assert nc_var.getncattr.call_count == 3

    def test_getattr_of_a_non_attribute_reaches_the_variable(self, nc_var):
        # The one-cycle compatibility route: a netCDF4 member that is not a
        # CF attribute still resolves, and is still not marked as used.
        nc_var.not_an_ncattr = 42
        cf_var = CFVariableSub("foo", nc_var)

        assert cf_var.not_an_ncattr == 42
        assert "not_an_ncattr" not in cf_var.__dict__
        assert "not_an_ncattr" not in dict(cf_var.cf_attrs())
        assert "not_an_ncattr" not in dict(cf_var.cf_attrs_used())

    def test_getattr_of_nothing_at_all_raises_attribute_error(self, nc_var):
        cf_var = CFVariableSub("foo", nc_var)
        # A MagicMock invents any member, so ask a real object instead.
        cf_var.cf_data = object()
        with pytest.raises(AttributeError, match="nonesuch"):
            cf_var.nonesuch

    def test_getitem_and_len_delegate_to_underlying_variable(self, nc_var):
        cf_var = CFVariableSub("foo", nc_var)

        assert len(cf_var) == 4
        assert cf_var[0] == "payload"
        nc_var.__len__.assert_called_once_with()
        nc_var.__getitem__.assert_called_once_with(0)

    def test_cf_attrs_access_helpers_and_reset(self, nc_var):
        cf_var = CFVariableSub("foo", nc_var)

        assert cf_var.cf_attrs() == (
            ("_FillValue", -999),
            ("coordinates", "x y"),
            ("standard_name", "air_temperature"),
        )
        assert cf_var.cf_attrs_ignored() == (("_FillValue", -999),)
        assert cf_var.cf_attrs_used() == (("_FillValue", -999),)
        assert cf_var.cf_attrs_unused() == (
            ("coordinates", "x y"),
            ("standard_name", "air_temperature"),
        )

        _ = cf_var.coordinates
        assert cf_var.cf_attrs_used() == (("_FillValue", -999), ("coordinates", "x y"))

        cf_var.cf_attrs_reset()
        assert cf_var.cf_attrs_used() == (("_FillValue", -999),)


class TestFormulaTerms:
    def test_registration_and_presence(self, nc_var):
        cf_var = CFVariableSub("foo", nc_var)

        assert not cf_var.has_formula_terms()
        cf_var.add_formula_term("root", "a")
        assert cf_var.has_formula_terms()
        assert cf_var.cf_terms_by_root == {"root": "a"}


class TestIdentify:
    def test_subclass_stub_returns_none(self, nc_var):
        cf_var = CFVariableSub("foo", nc_var)

        assert cf_var.identify({}) is None


class TestAttributesMapping:
    def test_attributes_contents(self, nc_var):
        cf_var = CFVariableSub("foo", nc_var)

        assert dict(cf_var.attributes) == {
            "coordinates": "x y",
            "standard_name": "air_temperature",
            "_FillValue": -999,
        }

    def test_getattr_and_mapping_are_the_same_read(self, nc_var):
        cf_var = CFVariableSub("foo", nc_var)

        _ = cf_var.coordinates
        assert cf_var.attributes.read == frozenset(["_FillValue", "coordinates"])

        cf_var.cf_attrs_reset()
        _ = cf_var.attributes["coordinates"]
        assert cf_var.attributes.read == frozenset(["_FillValue", "coordinates"])

    def test_untracked_read_is_not_recorded(self, nc_var):
        # What helpers.py's flag-attribute probe needs: a look that does not
        # count, so the attribute still reaches the cube.
        cf_var = CFVariableSub("foo", nc_var)

        assert cf_var.attributes.untracked["coordinates"] == "x y"
        assert "coordinates" in dict(cf_var.cf_attrs_unused())


class TestTypedProperties:
    def test_dimensions_is_a_tuple(self, nc_var):
        cf_var = CFVariableSub("foo", nc_var)
        assert cf_var.dimensions == ("time", "lat")

    def test_shape_ndim_dtype_size(self, mocker, nc_var):
        nc_var.shape = (3, 4)
        nc_var.dtype = np.dtype("f4")
        nc_var.size = 12
        cf_var = CFVariableSub("foo", nc_var)

        assert cf_var.shape == (3, 4)
        assert cf_var.ndim == 2
        assert cf_var.dtype == np.dtype("f4")
        assert cf_var.size == 12

    def test_a_file_attribute_does_not_reach_the_typed_properties(self, nc_var):
        # The properties are looked up before __getattr__ runs, so a file
        # attribute called "shape" cannot displace the variable's shape.
        nc_var.ncattrs.return_value = ["shape"]
        nc_var.getncattr.side_effect = {"shape": "not a shape"}.__getitem__
        nc_var.shape = (2,)
        cf_var = CFVariableSub("foo", nc_var)

        assert cf_var.shape == (2,)
        assert cf_var.attributes["shape"] == "not a shape"
