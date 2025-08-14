# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.common.metadata.ConnectivityMetadata`."""

from copy import deepcopy

import pytest

from iris.common.lenient import _LENIENT, _qualname
from iris.common.metadata import BaseMetadata, ConnectivityMetadata


class Test:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.standard_name = mocker.sentinel.standard_name
        self.long_name = mocker.sentinel.long_name
        self.var_name = mocker.sentinel.var_name
        self.units = mocker.sentinel.units
        self.attributes = mocker.sentinel.attributes
        self.cf_role = mocker.sentinel.cf_role
        self.start_index = mocker.sentinel.start_index
        self.location_axis = mocker.sentinel.location_axis
        self.cls = ConnectivityMetadata

    def test_repr(self, mocker):
        metadata = self.cls(
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
            units=self.units,
            attributes=self.attributes,
            cf_role=self.cf_role,
            start_index=self.start_index,
            location_axis=self.location_axis,
        )
        fmt = (
            "ConnectivityMetadata(standard_name={!r}, long_name={!r}, "
            "var_name={!r}, units={!r}, attributes={!r}, cf_role={!r}, "
            "start_index={!r}, location_axis={!r})"
        )
        expected = fmt.format(
            self.standard_name,
            self.long_name,
            self.var_name,
            self.units,
            self.attributes,
            self.cf_role,
            self.start_index,
            self.location_axis,
        )
        assert expected == repr(metadata)

    def test__fields(self, mocker):
        expected = (
            "standard_name",
            "long_name",
            "var_name",
            "units",
            "attributes",
            "cf_role",
            "start_index",
            "location_axis",
        )
        assert self.cls._fields == expected

    def test_bases(self, mocker):
        assert issubclass(self.cls, BaseMetadata)


class Test__eq__:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.values = dict(
            standard_name=mocker.sentinel.standard_name,
            long_name=mocker.sentinel.long_name,
            var_name=mocker.sentinel.var_name,
            units=mocker.sentinel.units,
            attributes=mocker.sentinel.attributes,
            cf_role=mocker.sentinel.cf_role,
            start_index=mocker.sentinel.start_index,
            location_axis=mocker.sentinel.location_axis,
        )
        self.dummy = mocker.sentinel.dummy
        self.cls = ConnectivityMetadata
        # The "location_axis" member is stateful only, and does not participate in
        # lenient/strict equivalence.
        self.members_no_location_axis = filter(
            lambda member: member != "location_axis", self.cls._members
        )

    def test_wraps_docstring(self, mocker):
        assert BaseMetadata.__eq__.__doc__ == self.cls.__eq__.__doc__

    def test_lenient_service(self, mocker):
        qualname___eq__ = _qualname(self.cls.__eq__)
        assert qualname___eq__ in _LENIENT
        assert _LENIENT[qualname___eq__]
        assert _LENIENT[self.cls.__eq__]

    def test_call(self, mocker):
        other = mocker.sentinel.other
        return_value = mocker.sentinel.return_value
        metadata = self.cls(*(None,) * len(self.cls._fields))
        mocked = mocker.patch.object(BaseMetadata, "__eq__", return_value=return_value)
        result = metadata.__eq__(other)

        assert return_value == result
        assert 1 == mocked.call_count
        (arg,), kwargs = mocked.call_args
        assert other == arg
        assert dict() == kwargs

    def test_op_lenient_same(self, mocker):
        lmetadata = self.cls(**self.values)
        rmetadata = self.cls(**self.values)

        mocker.patch("iris.common.metadata._LENIENT", return_value=True)
        assert lmetadata.__eq__(rmetadata)
        assert rmetadata.__eq__(lmetadata)

    def test_op_lenient_same_none(self, mocker):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["var_name"] = None
        rmetadata = self.cls(**right)

        mocker.patch("iris.common.metadata._LENIENT", return_value=True)
        assert lmetadata.__eq__(rmetadata)
        assert rmetadata.__eq__(lmetadata)

    def test_op_lenient_same_members_none(self, mocker):
        for member in self.members_no_location_axis:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = None
            rmetadata = self.cls(**right)

            mocker.patch("iris.common.metadata._LENIENT", return_value=True)
            assert not lmetadata.__eq__(rmetadata)
            assert not rmetadata.__eq__(lmetadata)

    def test_op_lenient_same_location_axis_none(self, mocker):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["location_axis"] = None
        rmetadata = self.cls(**right)

        mocker.patch("iris.common.metadata._LENIENT", return_value=True)
        assert lmetadata.__eq__(rmetadata)
        assert rmetadata.__eq__(lmetadata)

    def test_op_lenient_different(self, mocker):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["units"] = self.dummy
        rmetadata = self.cls(**right)

        mocker.patch("iris.common.metadata._LENIENT", return_value=True)
        assert not lmetadata.__eq__(rmetadata)
        assert not rmetadata.__eq__(lmetadata)

    def test_op_lenient_different_members(self, mocker):
        for member in self.members_no_location_axis:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = self.dummy
            rmetadata = self.cls(**right)

            mocker.patch("iris.common.metadata._LENIENT", return_value=True)
            assert not lmetadata.__eq__(rmetadata)
            assert not rmetadata.__eq__(lmetadata)

    def test_op_lenient_different_location_axis(self, mocker):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["location_axis"] = self.dummy
        rmetadata = self.cls(**right)

        mocker.patch("iris.common.metadata._LENIENT", return_value=True)
        assert lmetadata.__eq__(rmetadata)
        assert rmetadata.__eq__(lmetadata)

    def test_op_strict_same(self, mocker):
        lmetadata = self.cls(**self.values)
        rmetadata = self.cls(**self.values)

        mocker.patch("iris.common.metadata._LENIENT", return_value=False)
        assert lmetadata.__eq__(rmetadata)
        assert rmetadata.__eq__(lmetadata)

    def test_op_strict_different(self, mocker):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["long_name"] = self.dummy
        rmetadata = self.cls(**right)

        mocker.patch("iris.common.metadata._LENIENT", return_value=False)
        assert not lmetadata.__eq__(rmetadata)
        assert not rmetadata.__eq__(lmetadata)

    def test_op_strict_different_members(self, mocker):
        for member in self.members_no_location_axis:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = self.dummy
            rmetadata = self.cls(**right)

            mocker.patch("iris.common.metadata._LENIENT", return_value=False)
            assert not lmetadata.__eq__(rmetadata)
            assert not rmetadata.__eq__(lmetadata)

    def test_op_strict_different_location_axis(self, mocker):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["location_axis"] = self.dummy
        rmetadata = self.cls(**right)

        mocker.patch("iris.common.metadata._LENIENT", return_value=False)
        assert lmetadata.__eq__(rmetadata)
        assert rmetadata.__eq__(lmetadata)

    def test_op_strict_different_none(self, mocker):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["long_name"] = None
        rmetadata = self.cls(**right)

        mocker.patch("iris.common.metadata._LENIENT", return_value=False)
        assert not lmetadata.__eq__(rmetadata)
        assert not rmetadata.__eq__(lmetadata)

    def test_op_strict_different_members_none(self, mocker):
        for member in self.members_no_location_axis:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = None
            rmetadata = self.cls(**right)

            mocker.patch("iris.common.metadata._LENIENT", return_value=False)
            assert not lmetadata.__eq__(rmetadata)
            assert not rmetadata.__eq__(lmetadata)

    def test_op_strict_different_location_axis_none(self, mocker):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["location_axis"] = None
        rmetadata = self.cls(**right)

        mocker.patch("iris.common.metadata._LENIENT", return_value=False)
        assert lmetadata.__eq__(rmetadata)
        assert rmetadata.__eq__(lmetadata)


class Test___lt__:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.cls = ConnectivityMetadata
        self.one = self.cls(1, 1, 1, 1, 1, 1, 1, 1)
        self.two = self.cls(1, 1, 1, 2, 1, 1, 1, 1)
        self.none = self.cls(1, 1, 1, None, 1, 1, 1, 1)
        self.attributes = self.cls(1, 1, 1, 1, 10, 1, 1, 1)

    def test__ascending_lt(self, mocker):
        result = self.one < self.two
        assert result

    def test__descending_lt(self, mocker):
        result = self.two < self.one
        assert not result

    def test__none_rhs_operand(self, mocker):
        result = self.one < self.none
        assert not result

    def test__none_lhs_operand(self, mocker):
        result = self.none < self.one
        assert result

    def test__ignore_attributes(self, mocker):
        result = self.one < self.attributes
        assert not result
        result = self.attributes < self.one
        assert not result


class Test_combine:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.values = dict(
            standard_name=mocker.sentinel.standard_name,
            long_name=mocker.sentinel.long_name,
            var_name=mocker.sentinel.var_name,
            units=mocker.sentinel.units,
            attributes=mocker.sentinel.attributes,
            cf_role=mocker.sentinel.cf_role,
            start_index=mocker.sentinel.start_index,
            location_axis=mocker.sentinel.location_axis,
        )
        self.dummy = mocker.sentinel.dummy
        self.cls = ConnectivityMetadata
        self.none = self.cls(*(None,) * len(self.cls._fields))

    def test_wraps_docstring(self, mocker):
        assert BaseMetadata.combine.__doc__ == self.cls.combine.__doc__

    def test_lenient_service(self, mocker):
        qualname_combine = _qualname(self.cls.combine)
        assert qualname_combine in _LENIENT
        assert _LENIENT[qualname_combine]
        assert _LENIENT[self.cls.combine]

    def test_lenient_default(self, mocker):
        other = mocker.sentinel.other
        return_value = mocker.sentinel.return_value
        mocked = mocker.patch.object(BaseMetadata, "combine", return_value=return_value)
        result = self.none.combine(other)

        assert return_value == result
        assert 1 == mocked.call_count
        (arg,), kwargs = mocked.call_args
        assert other == arg
        assert dict(lenient=None) == kwargs

    def test_lenient(self, mocker):
        other = mocker.sentinel.other
        lenient = mocker.sentinel.lenient
        return_value = mocker.sentinel.return_value
        mocked = mocker.patch.object(BaseMetadata, "combine", return_value=return_value)
        result = self.none.combine(other, lenient=lenient)

        assert return_value == result
        assert 1 == mocked.call_count
        (arg,), kwargs = mocked.call_args
        assert other == arg
        assert dict(lenient=lenient) == kwargs

    def test_op_lenient_same(self, mocker):
        lmetadata = self.cls(**self.values)
        rmetadata = self.cls(**self.values)
        expected = self.values

        mocker.patch("iris.common.metadata._LENIENT", return_value=True)
        assert expected == lmetadata.combine(rmetadata)._asdict()
        assert expected == rmetadata.combine(lmetadata)._asdict()

    def test_op_lenient_same_none(self, mocker):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["var_name"] = None
        rmetadata = self.cls(**right)
        expected = self.values

        mocker.patch("iris.common.metadata._LENIENT", return_value=True)
        assert expected == lmetadata.combine(rmetadata)._asdict()
        assert expected == rmetadata.combine(lmetadata)._asdict()

    def test_op_lenient_same_members_none(self, mocker):
        for member in self.cls._members:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = None
            rmetadata = self.cls(**right)
            expected = right.copy()

            mocker.patch("iris.common.metadata._LENIENT", return_value=True)
            assert expected == lmetadata.combine(rmetadata)._asdict()
            assert expected == rmetadata.combine(lmetadata)._asdict()

    def test_op_lenient_different(self, mocker):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["units"] = self.dummy
        rmetadata = self.cls(**right)
        expected = self.values.copy()
        expected["units"] = None

        mocker.patch("iris.common.metadata._LENIENT", return_value=True)
        assert expected == lmetadata.combine(rmetadata)._asdict()
        assert expected == rmetadata.combine(lmetadata)._asdict()

    def test_op_lenient_different_members(self, mocker):
        for member in self.cls._members:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = self.dummy
            rmetadata = self.cls(**right)
            expected = self.values.copy()
            expected[member] = None

            mocker.patch("iris.common.metadata._LENIENT", return_value=True)
            assert expected == lmetadata.combine(rmetadata)._asdict()
            assert expected == rmetadata.combine(lmetadata)._asdict()

    def test_op_strict_same(self, mocker):
        lmetadata = self.cls(**self.values)
        rmetadata = self.cls(**self.values)
        expected = self.values.copy()

        mocker.patch("iris.common.metadata._LENIENT", return_value=False)
        assert expected == lmetadata.combine(rmetadata)._asdict()
        assert expected == rmetadata.combine(lmetadata)._asdict()

    def test_op_strict_different(self, mocker):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["long_name"] = self.dummy
        rmetadata = self.cls(**right)
        expected = self.values.copy()
        expected["long_name"] = None

        mocker.patch("iris.common.metadata._LENIENT", return_value=False)
        assert expected == lmetadata.combine(rmetadata)._asdict()
        assert expected == rmetadata.combine(lmetadata)._asdict()

    def test_op_strict_different_members(self, mocker):
        for member in self.cls._members:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = self.dummy
            rmetadata = self.cls(**right)
            expected = self.values.copy()
            expected[member] = None

            mocker.patch("iris.common.metadata._LENIENT", return_value=False)
            assert expected == lmetadata.combine(rmetadata)._asdict()
            assert expected == rmetadata.combine(lmetadata)._asdict()

    def test_op_strict_different_none(self, mocker):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["long_name"] = None
        rmetadata = self.cls(**right)
        expected = self.values.copy()
        expected["long_name"] = None

        mocker.patch("iris.common.metadata._LENIENT", return_value=False)
        assert expected == lmetadata.combine(rmetadata)._asdict()
        assert expected == rmetadata.combine(lmetadata)._asdict()

    def test_op_strict_different_members_none(self, mocker):
        for member in self.cls._members:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = None
            rmetadata = self.cls(**right)
            expected = self.values.copy()
            expected[member] = None

            mocker.patch("iris.common.metadata._LENIENT", return_value=False)
            assert expected == lmetadata.combine(rmetadata)._asdict()
            assert expected == rmetadata.combine(lmetadata)._asdict()


class Test_difference:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.values = dict(
            standard_name=mocker.sentinel.standard_name,
            long_name=mocker.sentinel.long_name,
            var_name=mocker.sentinel.var_name,
            units=mocker.sentinel.units,
            attributes=mocker.sentinel.attributes,
            cf_role=mocker.sentinel.cf_role,
            start_index=mocker.sentinel.start_index,
            location_axis=mocker.sentinel.location_axis,
        )
        self.dummy = mocker.sentinel.dummy
        self.cls = ConnectivityMetadata
        self.none = self.cls(*(None,) * len(self.cls._fields))

    def test_wraps_docstring(self, mocker):
        assert BaseMetadata.difference.__doc__ == self.cls.difference.__doc__

    def test_lenient_service(self, mocker):
        qualname_difference = _qualname(self.cls.difference)
        assert qualname_difference in _LENIENT
        assert _LENIENT[qualname_difference]
        assert _LENIENT[self.cls.difference]

    def test_lenient_default(self, mocker):
        other = mocker.sentinel.other
        return_value = mocker.sentinel.return_value
        mocked = mocker.patch.object(
            BaseMetadata, "difference", return_value=return_value
        )
        result = self.none.difference(other)

        assert return_value == result
        assert 1 == mocked.call_count
        (arg,), kwargs = mocked.call_args
        assert other == arg
        assert dict(lenient=None) == kwargs

    def test_lenient(self, mocker):
        other = mocker.sentinel.other
        lenient = mocker.sentinel.lenient
        return_value = mocker.sentinel.return_value
        mocked = mocker.patch.object(
            BaseMetadata, "difference", return_value=return_value
        )
        result = self.none.difference(other, lenient=lenient)

        assert return_value == result
        assert 1 == mocked.call_count
        (arg,), kwargs = mocked.call_args
        assert other == arg
        assert dict(lenient=lenient) == kwargs

    def test_op_lenient_same(self, mocker):
        lmetadata = self.cls(**self.values)
        rmetadata = self.cls(**self.values)

        mocker.patch("iris.common.metadata._LENIENT", return_value=True)
        assert lmetadata.difference(rmetadata) is None
        assert rmetadata.difference(lmetadata) is None

    def test_op_lenient_same_none(self, mocker):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["var_name"] = None
        rmetadata = self.cls(**right)

        mocker.patch("iris.common.metadata._LENIENT", return_value=True)
        assert lmetadata.difference(rmetadata) is None
        assert rmetadata.difference(lmetadata) is None

    def test_op_lenient_same_members_none(self, mocker):
        for member in self.cls._members:
            lmetadata = self.cls(**self.values)
            member_value = getattr(lmetadata, member)
            right = self.values.copy()
            right[member] = None
            rmetadata = self.cls(**right)
            lexpected = deepcopy(self.none)._asdict()
            lexpected[member] = (member_value, None)
            rexpected = deepcopy(self.none)._asdict()
            rexpected[member] = (None, member_value)

            mocker.patch("iris.common.metadata._LENIENT", return_value=True)
            assert lexpected == lmetadata.difference(rmetadata)._asdict()
            assert rexpected == rmetadata.difference(lmetadata)._asdict()

    def test_op_lenient_different(self, mocker):
        left = self.values.copy()
        lmetadata = self.cls(**left)
        right = self.values.copy()
        right["units"] = self.dummy
        rmetadata = self.cls(**right)
        lexpected = deepcopy(self.none)._asdict()
        lexpected["units"] = (left["units"], right["units"])
        rexpected = deepcopy(self.none)._asdict()
        rexpected["units"] = lexpected["units"][::-1]

        mocker.patch("iris.common.metadata._LENIENT", return_value=True)
        assert lexpected == lmetadata.difference(rmetadata)._asdict()
        assert rexpected == rmetadata.difference(lmetadata)._asdict()

    def test_op_lenient_different_members(self, mocker):
        for member in self.cls._members:
            left = self.values.copy()
            lmetadata = self.cls(**left)
            right = self.values.copy()
            right[member] = self.dummy
            rmetadata = self.cls(**right)
            lexpected = deepcopy(self.none)._asdict()
            lexpected[member] = (left[member], right[member])
            rexpected = deepcopy(self.none)._asdict()
            rexpected[member] = lexpected[member][::-1]

            mocker.patch("iris.common.metadata._LENIENT", return_value=True)
            assert lexpected == lmetadata.difference(rmetadata)._asdict()
            assert rexpected == rmetadata.difference(lmetadata)._asdict()

    def test_op_strict_same(self, mocker):
        lmetadata = self.cls(**self.values)
        rmetadata = self.cls(**self.values)

        mocker.patch("iris.common.metadata._LENIENT", return_value=False)
        assert lmetadata.difference(rmetadata) is None
        assert rmetadata.difference(lmetadata) is None

    def test_op_strict_different(self, mocker):
        left = self.values.copy()
        lmetadata = self.cls(**left)
        right = self.values.copy()
        right["long_name"] = self.dummy
        rmetadata = self.cls(**right)
        lexpected = deepcopy(self.none)._asdict()
        lexpected["long_name"] = (left["long_name"], right["long_name"])
        rexpected = deepcopy(self.none)._asdict()
        rexpected["long_name"] = lexpected["long_name"][::-1]

        mocker.patch("iris.common.metadata._LENIENT", return_value=False)
        assert lexpected == lmetadata.difference(rmetadata)._asdict()
        assert rexpected == rmetadata.difference(lmetadata)._asdict()

    def test_op_strict_different_members(self, mocker):
        for member in self.cls._members:
            left = self.values.copy()
            lmetadata = self.cls(**left)
            right = self.values.copy()
            right[member] = self.dummy
            rmetadata = self.cls(**right)
            lexpected = deepcopy(self.none)._asdict()
            lexpected[member] = (left[member], right[member])
            rexpected = deepcopy(self.none)._asdict()
            rexpected[member] = lexpected[member][::-1]

            mocker.patch("iris.common.metadata._LENIENT", return_value=False)
            assert lexpected == lmetadata.difference(rmetadata)._asdict()
            assert rexpected == rmetadata.difference(lmetadata)._asdict()

    def test_op_strict_different_none(self, mocker):
        left = self.values.copy()
        lmetadata = self.cls(**left)
        right = self.values.copy()
        right["long_name"] = None
        rmetadata = self.cls(**right)
        lexpected = deepcopy(self.none)._asdict()
        lexpected["long_name"] = (left["long_name"], right["long_name"])
        rexpected = deepcopy(self.none)._asdict()
        rexpected["long_name"] = lexpected["long_name"][::-1]

        mocker.patch("iris.common.metadata._LENIENT", return_value=False)
        assert lexpected == lmetadata.difference(rmetadata)._asdict()
        assert rexpected == rmetadata.difference(lmetadata)._asdict()

    def test_op_strict_different_members_none(self, mocker):
        for member in self.cls._members:
            left = self.values.copy()
            lmetadata = self.cls(**left)
            right = self.values.copy()
            right[member] = None
            rmetadata = self.cls(**right)
            lexpected = deepcopy(self.none)._asdict()
            lexpected[member] = (left[member], right[member])
            rexpected = deepcopy(self.none)._asdict()
            rexpected[member] = lexpected[member][::-1]

            mocker.patch("iris.common.metadata._LENIENT", return_value=False)
            assert lexpected == lmetadata.difference(rmetadata)._asdict()
            assert rexpected == rmetadata.difference(lmetadata)._asdict()


class Test_equal:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.cls = ConnectivityMetadata
        self.none = self.cls(*(None,) * len(self.cls._fields))

    def test_wraps_docstring(self, mocker):
        assert BaseMetadata.equal.__doc__ == self.cls.equal.__doc__

    def test_lenient_service(self, mocker):
        qualname_equal = _qualname(self.cls.equal)
        assert qualname_equal in _LENIENT
        assert _LENIENT[qualname_equal]
        assert _LENIENT[self.cls.equal]

    def test_lenient_default(self, mocker):
        other = mocker.sentinel.other
        return_value = mocker.sentinel.return_value
        mocked = mocker.patch.object(BaseMetadata, "equal", return_value=return_value)
        result = self.none.equal(other)

        assert return_value == result
        assert 1 == mocked.call_count
        (arg,), kwargs = mocked.call_args
        assert other == arg
        assert dict(lenient=None) == kwargs

    def test_lenient(self, mocker):
        other = mocker.sentinel.other
        lenient = mocker.sentinel.lenient
        return_value = mocker.sentinel.return_value
        mocked = mocker.patch.object(BaseMetadata, "equal", return_value=return_value)
        result = self.none.equal(other, lenient=lenient)

        assert return_value == result
        assert 1 == mocked.call_count
        (arg,), kwargs = mocked.call_args
        assert other == arg
        assert dict(lenient=lenient) == kwargs
