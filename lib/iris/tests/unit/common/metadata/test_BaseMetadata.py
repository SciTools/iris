# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.common.metadata.BaseMetadata`."""

from collections import OrderedDict

import numpy as np
import numpy.ma as ma
import pytest

from iris.common.lenient import _LENIENT, _qualname
from iris.common.metadata import BaseMetadata, CubeMetadata
from iris.tests._shared_utils import assert_dict_equal


class Test:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.standard_name = mocker.sentinel.standard_name
        self.long_name = mocker.sentinel.long_name
        self.var_name = mocker.sentinel.var_name
        self.units = mocker.sentinel.units
        self.attributes = mocker.sentinel.attributes
        self.cls = BaseMetadata

    def test_repr(self):
        metadata = self.cls(
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
            units=self.units,
            attributes=self.attributes,
        )
        fmt = (
            "BaseMetadata(standard_name={!r}, long_name={!r}, "
            "var_name={!r}, units={!r}, attributes={!r})"
        )
        expected = fmt.format(
            self.standard_name,
            self.long_name,
            self.var_name,
            self.units,
            self.attributes,
        )
        assert repr(metadata) == expected

    def test_str(self):
        metadata = self.cls(
            standard_name="",
            long_name=None,
            var_name=self.var_name,
            units=self.units,
            attributes={},
        )
        expected = f"BaseMetadata(var_name={self.var_name!r}, units={self.units!r})"
        assert str(metadata) == expected

    def test__fields(self):
        expected = (
            "standard_name",
            "long_name",
            "var_name",
            "units",
            "attributes",
        )
        assert self.cls._fields == expected


class Test___eq__:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.kwargs = dict(
            standard_name=mocker.sentinel.standard_name,
            long_name=mocker.sentinel.long_name,
            var_name=mocker.sentinel.var_name,
            units=mocker.sentinel.units,
            attributes=mocker.sentinel.attributes,
        )
        self.cls = BaseMetadata
        self.metadata = self.cls(**self.kwargs)

    def test_lenient_service(self):
        qualname___eq__ = _qualname(self.cls.__eq__)
        assert qualname___eq__ in _LENIENT
        assert _LENIENT[qualname___eq__]
        assert _LENIENT[self.cls.__eq__]

    def test_cannot_compare_non_class(self):
        result = self.metadata.__eq__(None)
        assert result is NotImplemented

    def test_cannot_compare_different_class(self):
        other = CubeMetadata(*(None,) * len(CubeMetadata._fields))
        result = self.metadata.__eq__(other)
        assert result is NotImplemented

    def test_lenient(self, mocker):
        return_value = mocker.sentinel.return_value
        mlenient = mocker.patch("iris.common.metadata._LENIENT", return_value=True)
        mcompare = mocker.patch.object(
            self.cls, "_compare_lenient", return_value=return_value
        )
        result = self.metadata.__eq__(self.metadata)

        assert result == return_value
        assert mcompare.call_count == 1
        (arg,), kwargs = mcompare.call_args
        assert arg is self.metadata
        assert kwargs == {}

        assert mlenient.call_count == 1
        (arg,), kwargs = mlenient.call_args
        assert _qualname(arg) == _qualname(self.cls.__eq__)
        assert kwargs == {}

    def test_strict_same(self):
        assert self.metadata.__eq__(self.metadata)
        other = self.cls(**self.kwargs)
        assert self.metadata.__eq__(other)
        assert other.__eq__(self.metadata)

    def test_strict_different(self):
        self.kwargs["var_name"] = None
        other = self.cls(**self.kwargs)
        assert not self.metadata.__eq__(other)
        assert not other.__eq__(self.metadata)


class Test___lt__:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cls = BaseMetadata
        self.one = self.cls(1, 1, 1, 1, 1)
        self.two = self.cls(1, 1, 1, 2, 1)
        self.none = self.cls(1, 1, 1, None, 1)
        self.attributes = self.cls(1, 1, 1, 1, 10)

    def test__ascending_lt(self):
        result = self.one < self.two
        assert result

    def test__descending_lt(self):
        result = self.two < self.one
        assert not result

    def test__none_rhs_operand(self):
        result = self.one < self.none
        assert not result

    def test__none_lhs_operand(self):
        result = self.none < self.one
        assert result

    def test__ignore_attributes(self):
        result = self.one < self.attributes
        assert not result
        result = self.attributes < self.one
        assert not result


class Test___ne__:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.cls = BaseMetadata
        self.metadata = self.cls(*(None,) * len(self.cls._fields))
        self.other = mocker.sentinel.other

    def test_notimplemented(self, mocker):
        return_value = NotImplemented
        patcher = mocker.patch.object(self.cls, "__eq__", return_value=return_value)
        result = self.metadata.__ne__(self.other)

        assert result is return_value
        assert patcher.call_count == 1
        (arg,), kwargs = patcher.call_args
        assert arg == self.other
        assert kwargs == {}

    def test_negate_true(self, mocker):
        return_value = True
        patcher = mocker.patch.object(self.cls, "__eq__", return_value=return_value)
        result = self.metadata.__ne__(self.other)

        assert not result
        (arg,), kwargs = patcher.call_args
        assert arg == self.other
        assert kwargs == {}

    def test_negate_false(self, mocker):
        return_value = False
        patcher = mocker.patch.object(self.cls, "__eq__", return_value=return_value)
        result = self.metadata.__ne__(self.other)

        assert result
        (arg,), kwargs = patcher.call_args
        assert arg == self.other
        assert kwargs == {}


class Test__combine:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.kwargs = dict(
            standard_name="standard_name",
            long_name="long_name",
            var_name="var_name",
            units="units",
            attributes=dict(one=mocker.sentinel.one, two=mocker.sentinel.two),
        )
        self.cls = BaseMetadata
        self.metadata = self.cls(**self.kwargs)

    def test_lenient(self, mocker):
        return_value = mocker.sentinel._combine_lenient
        other = mocker.sentinel.other
        mlenient = mocker.patch("iris.common.metadata._LENIENT", return_value=True)
        mcombine = mocker.patch.object(
            self.cls, "_combine_lenient", return_value=return_value
        )
        result = self.metadata._combine(other)

        assert mlenient.call_count == 1
        (arg,), kwargs = mlenient.call_args
        assert arg == self.metadata.combine
        assert kwargs == {}

        assert result == return_value
        assert mcombine.call_count == 1
        (arg,), kwargs = mcombine.call_args
        assert arg == other
        assert kwargs == {}

    def test_strict(self, mocker):
        dummy = mocker.sentinel.dummy
        values = self.kwargs.copy()
        values["standard_name"] = dummy
        values["var_name"] = dummy
        values["attributes"] = dummy
        other = self.cls(**values)
        mocker.patch("iris.common.metadata._LENIENT", return_value=False)
        result = self.metadata._combine(other)

        expected = [
            None if values[field] == dummy else values[field]
            for field in self.cls._fields
        ]
        assert result == expected


class Test__combine_lenient:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.cls = BaseMetadata
        self.none = self.cls(*(None,) * len(self.cls._fields))._asdict()
        self.names = dict(
            standard_name=mocker.sentinel.standard_name,
            long_name=mocker.sentinel.long_name,
            var_name=mocker.sentinel.var_name,
        )

    def test_strict_units(self):
        left = self.none.copy()
        left["units"] = "K"
        right = left.copy()
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)

        expected = list(left.values())
        assert lmetadata._combine_lenient(rmetadata) == expected
        assert rmetadata._combine_lenient(lmetadata) == expected

    def test_strict_units_different(self):
        left = self.none.copy()
        right = self.none.copy()
        left["units"] = "K"
        right["units"] = "km"
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)

        result = lmetadata._combine_lenient(rmetadata)
        expected = list(self.none.values())
        assert result == expected
        result = rmetadata._combine_lenient(lmetadata)
        assert result == expected

    def test_strict_units_different_none(self):
        left = self.none.copy()
        right = self.none.copy()
        left["units"] = "K"
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)

        result = lmetadata._combine_lenient(rmetadata)
        expected = list(self.none.values())
        assert result == expected

        result = rmetadata._combine_lenient(lmetadata)
        assert result == expected

    def test_attributes(self, mocker):
        left = self.none.copy()
        right = self.none.copy()
        ldict = dict(item=mocker.sentinel.left)
        rdict = dict(item=mocker.sentinel.right)
        left["attributes"] = ldict
        right["attributes"] = rdict
        rmetadata = self.cls(**right)
        return_value = mocker.sentinel.return_value
        patcher = mocker.patch.object(
            self.cls,
            "_combine_lenient_attributes",
            return_value=return_value,
        )
        lmetadata = self.cls(**left)
        result = lmetadata._combine_lenient(rmetadata)

        expected = self.none.copy()
        expected["attributes"] = return_value
        expected = list(expected.values())
        assert result == expected

        assert patcher.call_count == 1
        args, kwargs = patcher.call_args
        expected = (ldict, rdict)
        assert args == expected
        assert kwargs == {}

    def test_attributes_non_mapping_different(self, mocker):
        left = self.none.copy()
        right = self.none.copy()
        ldict = dict(item=mocker.sentinel.left)
        rdict = mocker.sentinel.right
        left["attributes"] = ldict
        right["attributes"] = rdict
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)

        expected = list(self.none.copy().values())
        assert lmetadata._combine_lenient(rmetadata) == expected
        assert rmetadata._combine_lenient(lmetadata) == expected

    def test_attributes_non_mapping_different_none(self, mocker):
        left = self.none.copy()
        right = self.none.copy()
        ldict = dict(item=mocker.sentinel.left)
        left["attributes"] = ldict
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)

        result = lmetadata._combine_lenient(rmetadata)
        expected = self.none.copy()
        expected["attributes"] = ldict
        expected = list(expected.values())
        assert result == expected

        result = rmetadata._combine_lenient(lmetadata)
        assert result == expected

    def test_names(self):
        left = self.none.copy()
        left.update(self.names)
        right = left.copy()
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)

        expected = list(left.values())
        assert lmetadata._combine_lenient(rmetadata) == expected
        assert rmetadata._combine_lenient(lmetadata) == expected

    def test_names_different(self, mocker):
        dummy = mocker.sentinel.dummy
        left = self.none.copy()
        right = self.none.copy()
        left.update(self.names)
        right["standard_name"] = dummy
        right["long_name"] = dummy
        right["var_name"] = dummy
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)

        expected = list(self.none.copy().values())
        assert lmetadata._combine_lenient(rmetadata) == expected
        assert rmetadata._combine_lenient(lmetadata) == expected

    def test_names_different_none(self):
        left = self.none.copy()
        right = self.none.copy()
        left.update(self.names)
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)

        result = lmetadata._combine_lenient(rmetadata)
        expected = list(left.values())
        assert result == expected

        result = rmetadata._combine_lenient(lmetadata)
        assert result == expected


class Test__combine_lenient_attributes:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.values = OrderedDict(
            one="one",
            two="two",
            three=np.int16(123),
            four=np.arange(10),
            five=ma.arange(10),
        )
        self.cls = BaseMetadata
        self.metadata = self.cls(*(None,) * len(self.cls._fields))
        self.dummy = mocker.sentinel.dummy

    def test_same(self):
        left = self.values.copy()
        right = self.values.copy()

        result = self.metadata._combine_lenient_attributes(left, right)
        expected = left
        assert_dict_equal(result, expected)

        result = self.metadata._combine_lenient_attributes(right, left)
        assert_dict_equal(result, expected)

    def test_different(self):
        left = self.values.copy()
        right = self.values.copy()
        left["two"] = left["four"] = self.dummy

        result = self.metadata._combine_lenient_attributes(left, right)
        expected = self.values.copy()
        for key in ["two", "four"]:
            del expected[key]
        assert_dict_equal(result, expected)

        result = self.metadata._combine_lenient_attributes(right, left)
        assert_dict_equal(result, expected)

    def test_different_none(self):
        left = self.values.copy()
        right = self.values.copy()
        left["one"] = left["three"] = left["five"] = None

        result = self.metadata._combine_lenient_attributes(left, right)
        expected = self.values.copy()
        for key in ["one", "three", "five"]:
            del expected[key]
        assert_dict_equal(result, expected)

        result = self.metadata._combine_lenient_attributes(right, left)
        assert_dict_equal(result, expected)

    def test_extra(self):
        left = self.values.copy()
        right = self.values.copy()
        left["extra_left"] = "extra_left"
        right["extra_right"] = "extra_right"

        result = self.metadata._combine_lenient_attributes(left, right)
        expected = self.values.copy()
        expected["extra_left"] = left["extra_left"]
        expected["extra_right"] = right["extra_right"]
        assert_dict_equal(result, expected)

        result = self.metadata._combine_lenient_attributes(right, left)
        assert_dict_equal(result, expected)


class Test__combine_strict_attributes:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.values = OrderedDict(
            one="one",
            two="two",
            three=np.int32(123),
            four=np.arange(10),
            five=ma.arange(10),
        )
        self.cls = BaseMetadata
        self.metadata = self.cls(*(None,) * len(self.cls._fields))
        self.dummy = mocker.sentinel.dummy

    def test_same(self):
        left = self.values.copy()
        right = self.values.copy()

        result = self.metadata._combine_strict_attributes(left, right)
        expected = left
        assert_dict_equal(result, expected)

        result = self.metadata._combine_strict_attributes(right, left)
        assert_dict_equal(result, expected)

    def test_different(self):
        left = self.values.copy()
        right = self.values.copy()
        left["one"] = left["three"] = self.dummy

        result = self.metadata._combine_strict_attributes(left, right)
        expected = self.values.copy()
        for key in ["one", "three"]:
            del expected[key]
        assert_dict_equal(result, expected)

        result = self.metadata._combine_strict_attributes(right, left)
        assert_dict_equal(result, expected)

    def test_different_none(self):
        left = self.values.copy()
        right = self.values.copy()
        left["one"] = left["three"] = left["five"] = None

        result = self.metadata._combine_strict_attributes(left, right)
        expected = self.values.copy()
        for key in ["one", "three", "five"]:
            del expected[key]
        assert_dict_equal(result, expected)

        result = self.metadata._combine_strict_attributes(right, left)
        assert_dict_equal(result, expected)

    def test_extra(self):
        left = self.values.copy()
        right = self.values.copy()
        left["extra_left"] = "extra_left"
        right["extra_right"] = "extra_right"

        result = self.metadata._combine_strict_attributes(left, right)
        expected = self.values.copy()
        assert_dict_equal(result, expected)

        result = self.metadata._combine_strict_attributes(right, left)
        assert_dict_equal(result, expected)


class Test__compare_lenient:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.cls = BaseMetadata
        self.none = self.cls(*(None,) * len(self.cls._fields))._asdict()
        self.names = dict(
            standard_name=mocker.sentinel.standard_name,
            long_name=mocker.sentinel.long_name,
            var_name=mocker.sentinel.var_name,
        )

    def test_name_same(self, mocker):
        left = self.none.copy()
        left.update(self.names)
        right = left.copy()
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)

        patcher = mocker.patch.object(self.cls, "_is_attributes", return_value=False)
        assert lmetadata._compare_lenient(rmetadata)
        assert rmetadata._compare_lenient(lmetadata)

        # mocker not called for "units" nor "var_name" members.
        expected = (len(self.cls._fields) - 2) * 2
        assert patcher.call_count == expected

    def test_name_same_lenient_false__long_name_different(self, mocker):
        left = self.none.copy()
        left.update(self.names)
        right = left.copy()
        right["long_name"] = mocker.sentinel.dummy
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)

        patcher = mocker.patch.object(self.cls, "_is_attributes", return_value=False)
        assert not lmetadata._compare_lenient(rmetadata)
        assert not rmetadata._compare_lenient(lmetadata)

        # mocker not called for "units" nor "var_name" members.
        expected = (len(self.cls._fields) - 2) * 2
        assert patcher.call_count == expected

    def test_name_same_lenient_true__var_name_different(self, mocker):
        left = self.none.copy()
        left.update(self.names)
        right = left.copy()
        right["var_name"] = mocker.sentinel.dummy
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)

        patcher = mocker.patch.object(self.cls, "_is_attributes", return_value=False)
        assert lmetadata._compare_lenient(rmetadata)
        assert rmetadata._compare_lenient(lmetadata)

        # mocker not called for "units" nor "var_name" members.
        expected = (len(self.cls._fields) - 2) * 2
        assert patcher.call_count == expected

    def test_name_different(self, mocker):
        left = self.none.copy()
        left.update(self.names)
        right = left.copy()
        right["standard_name"] = None
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)

        patcher = mocker.patch.object(self.cls, "_is_attributes")
        assert not lmetadata._compare_lenient(rmetadata)
        assert not rmetadata._compare_lenient(lmetadata)

        assert patcher.call_count == 0

    def test_strict_units(self, mocker):
        left = self.none.copy()
        left.update(self.names)
        left["units"] = "K"
        right = left.copy()
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)

        patcher = mocker.patch.object(self.cls, "_is_attributes", return_value=False)
        assert lmetadata._compare_lenient(rmetadata)
        assert rmetadata._compare_lenient(lmetadata)

        # mocker not called for "units" nor "var_name" members.
        expected = (len(self.cls._fields) - 2) * 2
        assert patcher.call_count == expected

    def test_strict_units_different(self, mocker):
        left = self.none.copy()
        left.update(self.names)
        left["units"] = "K"
        right = left.copy()
        right["units"] = "m"
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)

        patcher = mocker.patch.object(self.cls, "_is_attributes", return_value=False)
        assert not lmetadata._compare_lenient(rmetadata)
        assert not rmetadata._compare_lenient(lmetadata)

        # mocker not called for "units" nor "var_name" members.
        expected = (len(self.cls._fields) - 2) * 2
        assert patcher.call_count == expected

    def test_attributes(self, mocker):
        left = self.none.copy()
        left.update(self.names)
        right = left.copy()
        ldict = dict(item=mocker.sentinel.left)
        rdict = dict(item=mocker.sentinel.right)
        left["attributes"] = ldict
        right["attributes"] = rdict
        rmetadata = self.cls(**right)
        patcher = mocker.patch.object(
            self.cls,
            "_compare_lenient_attributes",
            return_value=True,
        )
        lmetadata = self.cls(**left)
        assert lmetadata._compare_lenient(rmetadata)
        assert rmetadata._compare_lenient(lmetadata)

        assert patcher.call_count == 2
        expected = [((ldict, rdict),), ((rdict, ldict),)]
        assert patcher.call_args_list == expected

    def test_attributes_non_mapping_different(self, mocker):
        left = self.none.copy()
        left.update(self.names)
        right = left.copy()
        ldict = dict(item=mocker.sentinel.left)
        rdict = mocker.sentinel.right
        left["attributes"] = ldict
        right["attributes"] = rdict
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)

        assert not lmetadata._compare_lenient(rmetadata)
        assert not rmetadata._compare_lenient(lmetadata)

    def test_attributes_non_mapping_different_none(self, mocker):
        left = self.none.copy()
        left.update(self.names)
        right = left.copy()
        ldict = dict(item=mocker.sentinel.left)
        left["attributes"] = ldict
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)

        assert lmetadata._compare_lenient(rmetadata)
        assert rmetadata._compare_lenient(lmetadata)

    def test_names(self):
        left = self.none.copy()
        left.update(self.names)
        left["long_name"] = None
        right = self.none.copy()
        right["long_name"] = left["standard_name"]
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)

        assert lmetadata._compare_lenient(rmetadata)
        assert rmetadata._compare_lenient(lmetadata)


class Test__compare_lenient_attributes:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.values = OrderedDict(
            one=mocker.sentinel.one,
            two=mocker.sentinel.two,
            three=np.int16(123),
            four=np.arange(10),
            five=ma.arange(5),
        )
        self.cls = BaseMetadata
        self.metadata = self.cls(*(None,) * len(self.cls._fields))
        self.dummy = mocker.sentinel.dummy

    def test_same(self):
        left = self.values.copy()
        right = self.values.copy()

        assert self.metadata._compare_lenient_attributes(left, right)
        assert self.metadata._compare_lenient_attributes(right, left)

    def test_different(self):
        left = self.values.copy()
        right = self.values.copy()
        left["two"] = left["four"] = self.dummy

        assert not self.metadata._compare_lenient_attributes(left, right)
        assert not self.metadata._compare_lenient_attributes(right, left)

    def test_different_none(self):
        left = self.values.copy()
        right = self.values.copy()
        left["one"] = left["three"] = left["five"] = None

        assert not self.metadata._compare_lenient_attributes(left, right)
        assert not self.metadata._compare_lenient_attributes(right, left)

    def test_extra(self, mocker):
        left = self.values.copy()
        right = self.values.copy()
        left["extra_left"] = mocker.sentinel.extra_left
        right["extra_right"] = mocker.sentinel.extra_right

        assert self.metadata._compare_lenient_attributes(left, right)
        assert self.metadata._compare_lenient_attributes(right, left)


class Test__compare_strict_attributes:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.values = OrderedDict(
            one=mocker.sentinel.one,
            two=mocker.sentinel.two,
            three=np.int16(123),
            four=np.arange(10),
            five=ma.arange(5),
        )
        self.cls = BaseMetadata
        self.metadata = self.cls(*(None,) * len(self.cls._fields))
        self.dummy = mocker.sentinel.dummy

    def test_same(self):
        left = self.values.copy()
        right = self.values.copy()

        assert self.metadata._compare_strict_attributes(left, right)
        assert self.metadata._compare_strict_attributes(right, left)

    def test_different(self):
        left = self.values.copy()
        right = self.values.copy()
        left["two"] = left["four"] = self.dummy

        assert not self.metadata._compare_strict_attributes(left, right)
        assert not self.metadata._compare_strict_attributes(right, left)

    def test_different_none(self):
        left = self.values.copy()
        right = self.values.copy()
        left["one"] = left["three"] = left["five"] = None

        assert not self.metadata._compare_strict_attributes(left, right)
        assert not self.metadata._compare_strict_attributes(right, left)

    def test_extra(self, mocker):
        left = self.values.copy()
        right = self.values.copy()
        left["extra_left"] = mocker.sentinel.extra_left
        right["extra_right"] = mocker.sentinel.extra_right

        assert not self.metadata._compare_strict_attributes(left, right)
        assert not self.metadata._compare_strict_attributes(right, left)


class Test__difference:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.kwargs = dict(
            standard_name="standard_name",
            long_name="long_name",
            var_name="var_name",
            units="units",
            attributes=dict(one=mocker.sentinel.one, two=mocker.sentinel.two),
        )
        self.cls = BaseMetadata
        self.metadata = self.cls(**self.kwargs)

    def test_lenient(self, mocker):
        return_value = mocker.sentinel._difference_lenient
        other = mocker.sentinel.other
        mlenient = mocker.patch("iris.common.metadata._LENIENT", return_value=True)
        mdifference = mocker.patch.object(
            self.cls, "_difference_lenient", return_value=return_value
        )
        result = self.metadata._difference(other)

        assert mlenient.call_count == 1
        (arg,), kwargs = mlenient.call_args
        assert arg == self.metadata.difference
        assert kwargs == {}

        assert result == return_value
        assert mdifference.call_count == 1
        (arg,), kwargs = mdifference.call_args
        assert arg == other
        assert kwargs == {}

    def test_strict(self, mocker):
        dummy = mocker.sentinel.dummy
        values = self.kwargs.copy()
        values["long_name"] = dummy
        values["units"] = dummy
        other = self.cls(**values)
        method = "_difference_strict_attributes"
        mocker.patch("iris.common.metadata._LENIENT", return_value=False)
        mdifference = mocker.patch.object(self.cls, method, return_value=None)
        result = self.metadata._difference(other)

        expected = [
            (self.kwargs[field], dummy) if values[field] == dummy else None
            for field in self.cls._fields
        ]
        assert result == expected
        assert mdifference.call_count == 1
        args, kwargs = mdifference.call_args
        expected = (self.kwargs["attributes"], values["attributes"])
        assert args == expected
        assert kwargs == {}

        mdifference = mocker.patch.object(self.cls, method, return_value=None)
        result = other._difference(self.metadata)

        expected = [
            (dummy, self.kwargs[field]) if values[field] == dummy else None
            for field in self.cls._fields
        ]
        assert result == expected
        assert mdifference.call_count == 1
        args, kwargs = mdifference.call_args
        expected = (self.kwargs["attributes"], values["attributes"])
        assert args == expected
        assert kwargs == {}


class Test__difference_lenient:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.cls = BaseMetadata
        self.none = self.cls(*(None,) * len(self.cls._fields))._asdict()
        self.names = dict(
            standard_name=mocker.sentinel.standard_name,
            long_name=mocker.sentinel.long_name,
            var_name=mocker.sentinel.var_name,
        )

    def test_strict_units(self):
        left = self.none.copy()
        left["units"] = "km"
        right = left.copy()
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)
        expected = list(self.none.values())
        assert lmetadata._difference_lenient(rmetadata) == expected
        assert rmetadata._difference_lenient(lmetadata) == expected

    def test_strict_units_different(self):
        left = self.none.copy()
        right = self.none.copy()
        lunits, runits = "m", "km"
        left["units"] = lunits
        right["units"] = runits
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)

        result = lmetadata._difference_lenient(rmetadata)
        expected = self.none.copy()
        expected["units"] = (lunits, runits)
        expected = list(expected.values())
        assert result == expected

        result = rmetadata._difference_lenient(lmetadata)
        expected = self.none.copy()
        expected["units"] = (runits, lunits)
        expected = list(expected.values())
        assert result == expected

    def test_strict_units_different_none(self):
        left = self.none.copy()
        right = self.none.copy()
        lunits, runits = "m", None
        left["units"] = lunits
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)

        result = lmetadata._difference_lenient(rmetadata)
        expected = self.none.copy()
        expected["units"] = (lunits, runits)
        expected = list(expected.values())

        assert result == expected
        result = rmetadata._difference_lenient(lmetadata)
        expected = self.none.copy()
        expected["units"] = (runits, lunits)
        expected = list(expected.values())
        assert result == expected

    def test_attributes(self, mocker):
        left = self.none.copy()
        right = self.none.copy()
        ldict = dict(item=mocker.sentinel.left)
        rdict = dict(item=mocker.sentinel.right)
        left["attributes"] = ldict
        right["attributes"] = rdict
        rmetadata = self.cls(**right)
        return_value = mocker.sentinel.return_value
        patcher = mocker.patch.object(
            self.cls,
            "_difference_lenient_attributes",
            return_value=return_value,
        )
        lmetadata = self.cls(**left)
        result = lmetadata._difference_lenient(rmetadata)

        expected = self.none.copy()
        expected["attributes"] = return_value
        expected = list(expected.values())
        assert result == expected

        assert patcher.call_count == 1
        args, kwargs = patcher.call_args
        expected = (ldict, rdict)
        assert args == expected
        assert kwargs == {}

    def test_attributes_non_mapping_different(self, mocker):
        left = self.none.copy()
        right = self.none.copy()
        ldict = dict(item=mocker.sentinel.left)
        rdict = mocker.sentinel.right
        left["attributes"] = ldict
        right["attributes"] = rdict
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)

        result = lmetadata._difference_lenient(rmetadata)
        expected = self.none.copy()
        expected["attributes"] = (ldict, rdict)
        expected = list(expected.values())
        assert result == expected

        result = rmetadata._difference_lenient(lmetadata)
        expected = self.none.copy()
        expected["attributes"] = (rdict, ldict)
        expected = list(expected.values())
        assert result == expected

    def test_attributes_non_mapping_different_none(self, mocker):
        left = self.none.copy()
        right = self.none.copy()
        ldict = dict(item=mocker.sentinel.left)
        left["attributes"] = ldict
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)

        result = lmetadata._difference_lenient(rmetadata)
        expected = list(self.none.copy().values())
        assert result == expected

        result = rmetadata._difference_lenient(lmetadata)
        assert result == expected

    def test_names(self):
        left = self.none.copy()
        left.update(self.names)
        right = left.copy()
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)

        expected = list(self.none.values())
        assert lmetadata._difference_lenient(rmetadata) == expected
        assert rmetadata._difference_lenient(lmetadata) == expected

    def test_names_different(self, mocker):
        dummy = mocker.sentinel.dummy
        left = self.none.copy()
        right = self.none.copy()
        left.update(self.names)
        right["standard_name"] = dummy
        right["long_name"] = dummy
        right["var_name"] = dummy
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)

        result = lmetadata._difference_lenient(rmetadata)
        expected = self.none.copy()
        expected["standard_name"] = (
            left["standard_name"],
            right["standard_name"],
        )
        expected["long_name"] = (left["long_name"], right["long_name"])
        expected["var_name"] = (left["var_name"], right["var_name"])
        expected = list(expected.values())
        assert result == expected

        result = rmetadata._difference_lenient(lmetadata)
        expected = self.none.copy()
        expected["standard_name"] = (
            right["standard_name"],
            left["standard_name"],
        )
        expected["long_name"] = (right["long_name"], left["long_name"])
        expected["var_name"] = (right["var_name"], left["var_name"])
        expected = list(expected.values())
        assert result == expected

    def test_names_different_none(self):
        left = self.none.copy()
        right = self.none.copy()
        left.update(self.names)
        lmetadata = self.cls(**left)
        rmetadata = self.cls(**right)

        result = lmetadata._difference_lenient(rmetadata)
        expected = list(self.none.values())
        assert result == expected

        result = rmetadata._difference_lenient(lmetadata)
        assert result == expected


class Test__difference_lenient_attributes:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.values = OrderedDict(
            one=mocker.sentinel.one,
            two=mocker.sentinel.two,
            three=np.float64(3.14),
            four=np.arange(10, dtype=np.float64),
            five=ma.arange(10, dtype=np.int16),
        )
        self.cls = BaseMetadata
        self.metadata = self.cls(*(None,) * len(self.cls._fields))
        self.dummy = mocker.sentinel.dummy

    def test_same(self):
        left = self.values.copy()
        right = self.values.copy()

        result = self.metadata._difference_lenient_attributes(left, right)
        assert result is None

        result = self.metadata._difference_lenient_attributes(right, left)
        assert result is None

    def test_different(self):
        left = self.values.copy()
        right = self.values.copy()
        left["two"] = left["four"] = self.dummy

        result = self.metadata._difference_lenient_attributes(left, right)
        for key in ["one", "three", "five"]:
            del left[key]
            del right[key]
        expected_left, expected_right = (left, right)
        result_left, result_right = result
        assert_dict_equal(result_left, expected_left)
        assert_dict_equal(result_right, expected_right)

        result = self.metadata._difference_lenient_attributes(right, left)
        result_left, result_right = result
        assert_dict_equal(result_left, expected_right)
        assert_dict_equal(result_right, expected_left)

    def test_different_none(self):
        left = self.values.copy()
        right = self.values.copy()
        left["one"] = left["three"] = left["five"] = None

        result = self.metadata._difference_lenient_attributes(left, right)
        for key in ["two", "four"]:
            del left[key]
            del right[key]
        expected_left, expected_right = (left, right)
        result_left, result_right = result
        assert_dict_equal(result_left, expected_left)
        assert_dict_equal(result_right, expected_right)

        result = self.metadata._difference_lenient_attributes(right, left)
        result_left, result_right = result
        assert_dict_equal(result_left, expected_right)
        assert_dict_equal(result_right, expected_left)

    def test_extra(self, mocker):
        left = self.values.copy()
        right = self.values.copy()
        left["extra_left"] = mocker.sentinel.extra_left
        right["extra_right"] = mocker.sentinel.extra_right
        result = self.metadata._difference_lenient_attributes(left, right)
        assert result is None

        result = self.metadata._difference_lenient_attributes(right, left)
        assert result is None


class Test__difference_strict_attributes:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.values = OrderedDict(
            one=mocker.sentinel.one,
            two=mocker.sentinel.two,
            three=np.int32(123),
            four=np.arange(10),
            five=ma.arange(10),
        )
        self.cls = BaseMetadata
        self.metadata = self.cls(*(None,) * len(self.cls._fields))
        self.dummy = mocker.sentinel.dummy

    def test_same(self):
        left = self.values.copy()
        right = self.values.copy()

        result = self.metadata._difference_strict_attributes(left, right)
        assert result is None
        result = self.metadata._difference_strict_attributes(right, left)
        assert result is None

    def test_different(self):
        left = self.values.copy()
        right = self.values.copy()
        left["one"] = left["three"] = left["five"] = self.dummy

        result = self.metadata._difference_strict_attributes(left, right)
        expected_left = left.copy()
        expected_right = right.copy()
        for key in ["two", "four"]:
            del expected_left[key]
            del expected_right[key]
        result_left, result_right = result
        assert_dict_equal(result_left, expected_left)
        assert_dict_equal(result_right, expected_right)

        result = self.metadata._difference_strict_attributes(right, left)
        result_left, result_right = result
        assert_dict_equal(result_left, expected_right)
        assert_dict_equal(result_right, expected_left)

    def test_different_none(self):
        left = self.values.copy()
        right = self.values.copy()
        left["one"] = left["three"] = left["five"] = None

        result = self.metadata._difference_strict_attributes(left, right)
        expected_left = left.copy()
        expected_right = right.copy()
        for key in ["two", "four"]:
            del expected_left[key]
            del expected_right[key]
        result_left, result_right = result
        assert_dict_equal(result_left, expected_left)
        assert_dict_equal(result_right, expected_right)

        result = self.metadata._difference_strict_attributes(right, left)
        result_left, result_right = result
        assert_dict_equal(result_left, expected_right)
        assert_dict_equal(result_right, expected_left)

    def test_extra(self, mocker):
        left = self.values.copy()
        right = self.values.copy()
        left["extra_left"] = mocker.sentinel.extra_left
        right["extra_right"] = mocker.sentinel.extra_right

        result = self.metadata._difference_strict_attributes(left, right)
        expected_left = dict(extra_left=left["extra_left"])
        expected_right = dict(extra_right=right["extra_right"])
        result_left, result_right = result
        assert_dict_equal(result_left, expected_left)
        assert_dict_equal(result_right, expected_right)

        result = self.metadata._difference_strict_attributes(right, left)
        result_left, result_right = result
        assert_dict_equal(result_left, expected_right)
        assert_dict_equal(result_right, expected_left)


class Test__is_attributes:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cls = BaseMetadata
        self.metadata = self.cls(*(None,) * len(self.cls._fields))
        self.field = "attributes"

    def test_field(self):
        assert self.metadata._is_attributes(self.field, {}, {})

    def test_field_not_attributes(self):
        assert not self.metadata._is_attributes(None, {}, {})

    def test_left_not_mapping(self):
        assert not self.metadata._is_attributes(self.field, None, {})

    def test_right_not_mapping(self):
        assert not self.metadata._is_attributes(self.field, {}, None)


class Test_combine:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        kwargs = dict(
            standard_name="standard_name",
            long_name="long_name",
            var_name="var_name",
            units="units",
            attributes="attributes",
        )
        self.cls = BaseMetadata
        self.metadata = self.cls(**kwargs)
        self.mock_kwargs = OrderedDict(
            standard_name=mocker.sentinel.standard_name,
            long_name=mocker.sentinel.long_name,
            var_name=mocker.sentinel.var_name,
            units=mocker.sentinel.units,
            attributes=mocker.sentinel.attributes,
        )

    def test_lenient_service(self):
        qualname_combine = _qualname(self.cls.combine)
        assert qualname_combine in _LENIENT
        assert _LENIENT[qualname_combine]
        assert _LENIENT[self.cls.combine]

    def test_cannot_combine_non_class(self):
        emsg = "Cannot combine"
        with pytest.raises(TypeError, match=emsg):
            _ = self.metadata.combine(None)

    def test_cannot_combine_different_class(self):
        other = CubeMetadata(*(None,) * len(CubeMetadata._fields))
        emsg = "Cannot combine"
        with pytest.raises(TypeError, match=emsg):
            _ = self.metadata.combine(other)

    def test_lenient_default(self, mocker):
        return_value = self.mock_kwargs.values()
        patcher = mocker.patch.object(self.cls, "_combine", return_value=return_value)
        result = self.metadata.combine(self.metadata)

        assert result._asdict() == self.mock_kwargs
        assert patcher.call_count == 1
        (arg,), kwargs = patcher.call_args
        assert arg is self.metadata
        assert kwargs == {}

    def test_lenient_true(self, mocker):
        return_value = self.mock_kwargs.values()
        mcombine = mocker.patch.object(self.cls, "_combine", return_value=return_value)
        mcontext = mocker.patch.object(_LENIENT, "context")
        result = self.metadata.combine(self.metadata, lenient=True)

        assert mcontext.call_count == 1
        (arg,), kwargs = mcontext.call_args
        assert arg == _qualname(self.cls.combine)
        assert kwargs == {}

        assert result._asdict() == self.mock_kwargs
        assert mcombine.call_count == 1
        (arg,), kwargs = mcombine.call_args
        assert arg is self.metadata
        assert kwargs == {}

    def test_lenient_false(self, mocker):
        return_value = self.mock_kwargs.values()
        mcombine = mocker.patch.object(self.cls, "_combine", return_value=return_value)
        mcontext = mocker.patch.object(_LENIENT, "context")
        result = self.metadata.combine(self.metadata, lenient=False)

        assert mcontext.call_count == 1
        args, kwargs = mcontext.call_args
        assert args == ()
        assert kwargs == {_qualname(self.cls.combine): False}

        assert result._asdict() == self.mock_kwargs
        assert mcombine.call_count == 1
        (arg,), kwargs = mcombine.call_args
        assert arg is self.metadata
        assert kwargs == {}


class Test_difference:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        kwargs = dict(
            standard_name="standard_name",
            long_name="long_name",
            var_name="var_name",
            units="units",
            attributes="attributes",
        )
        self.cls = BaseMetadata
        self.metadata = self.cls(**kwargs)
        self.mock_kwargs = OrderedDict(
            standard_name=mocker.sentinel.standard_name,
            long_name=mocker.sentinel.long_name,
            var_name=mocker.sentinel.var_name,
            units=mocker.sentinel.units,
            attributes=mocker.sentinel.attributes,
        )

    def test_lenient_service(self):
        qualname_difference = _qualname(self.cls.difference)
        assert qualname_difference in _LENIENT
        assert _LENIENT[qualname_difference]
        assert _LENIENT[self.cls.difference]

    def test_cannot_differ_non_class(self):
        emsg = "Cannot differ"
        with pytest.raises(TypeError, match=emsg):
            _ = self.metadata.difference(None)

    def test_cannot_differ_different_class(self):
        other = CubeMetadata(*(None,) * len(CubeMetadata._fields))
        emsg = "Cannot differ"
        with pytest.raises(TypeError, match=emsg):
            _ = self.metadata.difference(other)

    def test_lenient_default(self, mocker):
        return_value = self.mock_kwargs.values()
        patcher = mocker.patch.object(
            self.cls, "_difference", return_value=return_value
        )
        result = self.metadata.difference(self.metadata)

        assert result._asdict() == self.mock_kwargs
        assert patcher.call_count == 1
        (arg,), kwargs = patcher.call_args
        assert arg is self.metadata
        assert kwargs == {}

    def test_lenient_true(self, mocker):
        return_value = self.mock_kwargs.values()
        mdifference = mocker.patch.object(
            self.cls, "_difference", return_value=return_value
        )
        mcontext = mocker.patch.object(_LENIENT, "context")
        result = self.metadata.difference(self.metadata, lenient=True)

        assert mcontext.call_count == 1
        (arg,), kwargs = mcontext.call_args
        assert arg == _qualname(self.cls.difference)
        assert kwargs == {}

        assert result._asdict() == self.mock_kwargs
        assert mdifference.call_count == 1
        (arg,), kwargs = mdifference.call_args
        assert arg is self.metadata
        assert kwargs == {}

    def test_lenient_false(self, mocker):
        return_value = self.mock_kwargs.values()
        mdifference = mocker.patch.object(
            self.cls, "_difference", return_value=return_value
        )
        mcontext = mocker.patch.object(_LENIENT, "context")
        result = self.metadata.difference(self.metadata, lenient=False)

        assert mcontext.call_count == 1
        args, kwargs = mcontext.call_args
        assert args == ()
        assert kwargs == {_qualname(self.cls.difference): False}

        assert result._asdict() == self.mock_kwargs
        assert mdifference.call_count == 1
        (arg,), kwargs = mdifference.call_args
        assert arg is self.metadata
        assert kwargs == {}


class Test_equal:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        kwargs = dict(
            standard_name=mocker.sentinel.standard_name,
            long_name=mocker.sentinel.long_name,
            var_name=mocker.sentinel.var_name,
            units=mocker.sentinel.units,
            attributes=mocker.sentinel.attributes,
        )
        self.cls = BaseMetadata
        self.metadata = self.cls(**kwargs)

    def test_lenient_service(self):
        qualname_equal = _qualname(self.cls.equal)
        assert qualname_equal in _LENIENT
        assert _LENIENT[qualname_equal]
        assert _LENIENT[self.cls.equal]

    def test_cannot_compare_non_class(self):
        emsg = "Cannot compare"
        with pytest.raises(TypeError, match=emsg):
            _ = self.metadata.equal(None)

    def test_cannot_compare_different_class(self):
        other = CubeMetadata(*(None,) * len(CubeMetadata._fields))
        emsg = "Cannot compare"
        with pytest.raises(TypeError, match=emsg):
            _ = self.metadata.equal(other)

    def test_lenient_default(self, mocker):
        return_value = mocker.sentinel.return_value
        patcher = mocker.patch.object(self.cls, "__eq__", return_value=return_value)
        result = self.metadata.equal(self.metadata)

        assert result == return_value
        assert patcher.call_count == 1
        (arg,), kwargs = patcher.call_args
        assert arg is self.metadata
        assert kwargs == {}

    def test_lenient_true(self, mocker):
        return_value = mocker.sentinel.return_value
        m__eq__ = mocker.patch.object(self.cls, "__eq__", return_value=return_value)
        mcontext = mocker.patch.object(_LENIENT, "context")
        result = self.metadata.equal(self.metadata, lenient=True)

        assert result == return_value
        assert mcontext.call_count == 1
        (arg,), kwargs = mcontext.call_args
        assert arg == _qualname(self.cls.equal)
        assert kwargs == {}

        assert m__eq__.call_count == 1
        (arg,), kwargs = m__eq__.call_args
        assert arg is self.metadata
        assert kwargs == {}

    def test_lenient_false(self, mocker):
        return_value = mocker.sentinel.return_value
        m__eq__ = mocker.patch.object(self.cls, "__eq__", return_value=return_value)
        mcontext = mocker.patch.object(_LENIENT, "context")
        result = self.metadata.equal(self.metadata, lenient=False)

        assert mcontext.call_count == 1
        args, kwargs = mcontext.call_args
        assert args == ()
        assert kwargs == {_qualname(self.cls.equal): False}

        assert result == return_value
        assert m__eq__.call_count == 1
        (arg,), kwargs = m__eq__.call_args
        assert arg is self.metadata
        assert kwargs == {}


class Test_name:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cls = BaseMetadata
        self.default = self.cls.DEFAULT_NAME

    @staticmethod
    def _make(standard_name=None, long_name=None, var_name=None):
        return BaseMetadata(
            standard_name=standard_name,
            long_name=long_name,
            var_name=var_name,
            units=None,
            attributes=None,
        )

    def test_standard_name(self):
        token = "standard_name"
        metadata = self._make(standard_name=token)

        result = metadata.name()
        assert result == token
        result = metadata.name(token=True)
        assert result == token

    def test_standard_name__invalid_token(self):
        token = "nope nope"
        metadata = self._make(standard_name=token)

        result = metadata.name()
        assert result == token
        result = metadata.name(token=True)
        assert result == self.default

    def test_long_name(self):
        token = "long_name"
        metadata = self._make(long_name=token)

        result = metadata.name()
        assert result == token
        result = metadata.name(token=True)
        assert result == token

    def test_long_name__invalid_token(self):
        token = "nope nope"
        metadata = self._make(long_name=token)

        result = metadata.name()
        assert result == token
        result = metadata.name(token=True)
        assert result == self.default

    def test_var_name(self):
        token = "var_name"
        metadata = self._make(var_name=token)

        result = metadata.name()
        assert result == token
        result = metadata.name(token=True)
        assert result == token

    def test_var_name__invalid_token(self):
        token = "nope nope"
        metadata = self._make(var_name=token)

        result = metadata.name()
        assert result == token
        result = metadata.name(token=True)
        assert result == self.default

    def test_default(self):
        metadata = self._make()

        result = metadata.name()
        assert result == self.default
        result = metadata.name(token=True)
        assert result == self.default

    def test_default__invalid_token(self):
        token = "nope nope"
        metadata = self._make()

        result = metadata.name(default=token)
        assert result == token

        emsg = "Cannot retrieve a valid name token"
        with pytest.raises(ValueError, match=emsg):
            _ = metadata.name(default=token, token=True)


class Test_token:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cls = BaseMetadata

    def test_passthru_none(self):
        result = self.cls.token(None)
        assert result is None

    def test_fail_leading_underscore(self):
        result = self.cls.token("_nope")
        assert result is None

    def test_fail_leading_dot(self):
        result = self.cls.token(".nope")
        assert result is None

    def test_fail_leading_plus(self):
        result = self.cls.token("+nope")
        assert result is None

    def test_fail_leading_at(self):
        result = self.cls.token("@nope")
        assert result is None

    def test_fail_space(self):
        result = self.cls.token("nope nope")
        assert result is None

    def test_fail_colon(self):
        result = self.cls.token("nope:")
        assert result is None

    def test_pass_simple(self):
        token = "simple"
        result = self.cls.token(token)
        assert result == token

    def test_pass_leading_digit(self):
        token = "123simple"
        result = self.cls.token(token)
        assert result == token

    def test_pass_mixture(self):
        token = "S.imple@one+two_3"
        result = self.cls.token(token)
        assert result == token
