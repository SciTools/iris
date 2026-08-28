# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFVariable`."""

import pytest

from iris.fileformats import cf as cf


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


def test_init_records_filename_from_group(nc_var):
    cf_var = CFVariableSub("foo", nc_var)

    assert cf_var.filename == "/tmp/file.nc"
    assert cf_var.cf_name == "foo"
    assert cf_var.cf_data is nc_var
    assert cf_var.cf_group is None
    assert cf_var.cf_terms_by_root == {}
    assert cf_var._to_be_promoted is False


def test_init_falls_back_to_unknown_filename_without_group(nc_var_without_group):
    cf_var = CFVariableSub("foo", nc_var_without_group)

    assert cf_var.filename == "<unknown_filename>"


def test_identify_common_handles_defaults_and_target_selection():
    variables = {"a": object(), "b": object()}

    ignore, target = CFVariableSub._identify_common(variables, None, None)
    assert ignore == []
    assert target is variables

    ignore, target = CFVariableSub._identify_common(variables, ["a"], "b")
    assert ignore == ["a"]
    assert target == {"b": variables["b"]}


def test_identify_common_raises_for_unknown_target():
    with pytest.raises(ValueError, match="Cannot identify unknown target"):
        CFVariableSub._identify_common({"a": object()}, None, "missing")


def test_identify_common_raises_for_invalid_target_type():
    with pytest.raises(TypeError, match="Expect a target CF-netCDF variable name"):
        CFVariableSub._identify_common({"a": object()}, None, object())


def test_spans_scalar_dimension_always_true(mocker, nc_var):
    nc_var.dimensions = (cf._NCZARR_SCALAR_DIMENSION,)
    cf_var = CFVariableSub("scalar", nc_var)

    other = mocker.MagicMock()
    other.dimensions = ("time",)

    assert cf_var.spans(other)


def test_spans_is_subset_check(mocker, nc_vars):
    lhs_nc_var, other_nc_var, _ = nc_vars
    lhs_nc_var.dimensions = ("time",)
    lhs = CFVariableSub("lhs", lhs_nc_var)

    rhs = mocker.MagicMock()
    rhs.dimensions = ("time", "lat")

    assert lhs.spans(rhs)

    other_nc_var.dimensions = ("height",)
    other = CFVariableSub("other", other_nc_var)
    assert not other.spans(rhs)


def test_equality_inequality_and_hash_by_name(nc_vars):
    first, second, third = nc_vars
    one = CFVariableSub("same", first)
    two = CFVariableSub("same", second)
    other = CFVariableSub("different", third)

    assert one == two
    assert one != other
    assert hash(one) == hash(two)
    assert hash(one) != hash(other)


def test_cached(nc_var):
    # Make sure attribute access to the underlying netCDF4.Variable
    # is cached.
    name = "foo"
    cf_var = CFVariableSub(name, nc_var)
    assert nc_var.ncattrs.call_count == 1

    # Accessing a netCDF attribute should result in no further calls
    # to nc_var.ncattrs() and the creation of an attribute on the
    # cf_var.
    # NB. Can't use hasattr() because that triggers the attribute
    # to be created!
    assert "coordinates" not in cf_var.__dict__
    _ = cf_var.coordinates
    assert nc_var.ncattrs.call_count == 1
    assert "coordinates" in cf_var.__dict__

    # Trying again results in no change.
    _ = cf_var.coordinates
    assert nc_var.ncattrs.call_count == 1
    assert "coordinates" in cf_var.__dict__

    # Trying another attribute results in just a new attribute.
    assert "standard_name" not in cf_var.__dict__
    _ = cf_var.standard_name
    assert nc_var.ncattrs.call_count == 1
    assert "standard_name" in cf_var.__dict__


def test_getattr_non_ncattr_value_is_cached_but_not_marked_used(nc_var):
    nc_var.not_an_ncattr = 42
    cf_var = CFVariableSub("foo", nc_var)

    assert cf_var.not_an_ncattr == 42
    assert "not_an_ncattr" in cf_var.__dict__
    assert "not_an_ncattr" not in cf_var.cf_attrs()


def test_getitem_and_len_delegate_to_underlying_variable(nc_var):
    cf_var = CFVariableSub("foo", nc_var)

    assert len(cf_var) == 4
    assert cf_var[0] == "payload"
    nc_var.__len__.assert_called_once_with()
    nc_var.__getitem__.assert_called_once_with(0)


def test_repr_contains_class_name_name_and_data_repr(nc_var):
    cf_var = CFVariableSub("foo", nc_var)

    assert repr(cf_var) == f"CFVariableSub('foo', {nc_var!r})"


def test_cf_attrs_access_helpers_and_reset(nc_var):
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


def test_formula_term_registration_and_presence(nc_var):
    cf_var = CFVariableSub("foo", nc_var)

    assert not cf_var.has_formula_terms()
    cf_var.add_formula_term("root", "a")
    assert cf_var.has_formula_terms()
    assert cf_var.cf_terms_by_root == {"root": "a"}


def test_identify_subclass_stub_returns_none(nc_var):
    cf_var = CFVariableSub("foo", nc_var)

    assert cf_var.identify({}) is None
