# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.common.metadata.CoordMetadata`."""

from copy import deepcopy

import pytest

from iris.common.lenient import _LENIENT, _qualname
from iris.common.metadata import BaseMetadata, CoordMetadata


class Test:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.standard_name = mocker.sentinel.standard_name
        self.long_name = mocker.sentinel.long_name
        self.var_name = mocker.sentinel.var_name
        self.units = mocker.sentinel.units
        self.attributes = mocker.sentinel.attributes
        self.coord_system = mocker.sentinel.coord_system
        self.climatological = mocker.sentinel.climatological
        self.cls = CoordMetadata

    def test_repr(self):
        metadata = self.cls(
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
            units=self.units,
            attributes=self.attributes,
            coord_system=self.coord_system,
            climatological=self.climatological,
        )
        fmt = (
            "CoordMetadata(standard_name={!r}, long_name={!r}, "
            "var_name={!r}, units={!r}, attributes={!r}, coord_system={!r}, "
            "climatological={!r})"
        )
        expected = fmt.format(
            self.standard_name,
            self.long_name,
            self.var_name,
            self.units,
            self.attributes,
            self.coord_system,
            self.climatological,
        )
        assert repr(metadata) == expected

    def test__fields(self):
        expected = (
            "standard_name",
            "long_name",
            "var_name",
            "units",
            "attributes",
            "coord_system",
            "climatological",
        )
        assert self.cls._fields == expected

    def test_bases(self):
        assert issubclass(self.cls, BaseMetadata)


class Test___eq__:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.values = dict(
            standard_name=mocker.sentinel.standard_name,
            long_name=mocker.sentinel.long_name,
            var_name=mocker.sentinel.var_name,
            units=mocker.sentinel.units,
            attributes=mocker.sentinel.attributes,
            coord_system=mocker.sentinel.coord_system,
            climatological=mocker.sentinel.climatological,
        )
        self.dummy = mocker.sentinel.dummy
        self.cls = CoordMetadata

    def test_wraps_docstring(self):
        assert self.cls.__eq__.__doc__ == BaseMetadata.__eq__.__doc__

    def test_lenient_service(self):
        qualname___eq__ = _qualname(self.cls.__eq__)
        assert qualname___eq__ in _LENIENT
        assert _LENIENT[qualname___eq__]
        assert _LENIENT[self.cls.__eq__]

    def test_call(self, mocker):
        other = mocker.sentinel.other
        return_value = mocker.sentinel.return_value
        metadata = self.cls(*(None,) * len(self.cls._fields))
        patcher = mocker.patch.object(BaseMetadata, "__eq__", return_value=return_value)
        result = metadata.__eq__(other)

        assert result == return_value
        assert patcher.call_count == 1
        (arg,), kwargs = patcher.call_args
        assert arg == other
        assert kwargs == {}

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
        mocker.patch("iris.common.metadata._LENIENT", return_value=True)

        for member in self.cls._members:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = None
            rmetadata = self.cls(**right)
            assert not lmetadata.__eq__(rmetadata)
            assert not rmetadata.__eq__(lmetadata)

    def test_op_lenient_different(self, mocker):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["units"] = self.dummy
        rmetadata = self.cls(**right)

        mocker.patch("iris.common.metadata._LENIENT", return_value=True)
        assert not lmetadata.__eq__(rmetadata)
        assert not rmetadata.__eq__(lmetadata)

    def test_op_lenient_different_members(self, mocker):
        mocker.patch("iris.common.metadata._LENIENT", return_value=True)

        for member in self.cls._members:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = self.dummy
            rmetadata = self.cls(**right)
            assert not lmetadata.__eq__(rmetadata)
            assert not rmetadata.__eq__(lmetadata)

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
        mocker.patch("iris.common.metadata._LENIENT", return_value=False)

        for member in self.cls._members:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = self.dummy
            rmetadata = self.cls(**right)
            assert not lmetadata.__eq__(rmetadata)
            assert not rmetadata.__eq__(lmetadata)

    def test_op_strict_different_none(self, mocker):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["long_name"] = None
        rmetadata = self.cls(**right)

        mocker.patch("iris.common.metadata._LENIENT", return_value=False)
        assert not lmetadata.__eq__(rmetadata)
        assert not rmetadata.__eq__(lmetadata)

    def test_op_strict_different_members_none(self, mocker):
        mocker.patch("iris.common.metadata._LENIENT", return_value=False)

        for member in self.cls._members:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = None
            rmetadata = self.cls(**right)
            assert not lmetadata.__eq__(rmetadata)
            assert not rmetadata.__eq__(lmetadata)


class Test___lt__:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cls = CoordMetadata
        self.one = self.cls(1, 1, 1, 1, 1, 1, 1)
        self.two = self.cls(1, 1, 1, 2, 1, 1, 1)
        self.none = self.cls(1, 1, 1, None, 1, 1, 1)
        self.attributes_cs = self.cls(1, 1, 1, 1, 10, 10, 1)

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

    def test__ignore_attributes_coord_system(self):
        result = self.one < self.attributes_cs
        assert not result
        result = self.attributes_cs < self.one
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
            coord_system=mocker.sentinel.coord_system,
            climatological=mocker.sentinel.climatological,
        )
        self.dummy = mocker.sentinel.dummy
        self.cls = CoordMetadata
        self.none = self.cls(*(None,) * len(self.cls._fields))

    def test_wraps_docstring(self):
        assert self.cls.combine.__doc__ == BaseMetadata.combine.__doc__

    def test_lenient_service(self):
        qualname_combine = _qualname(self.cls.combine)
        assert qualname_combine in _LENIENT
        assert _LENIENT[qualname_combine]
        assert _LENIENT[self.cls.combine]

    def test_lenient_default(self, mocker):
        other = mocker.sentinel.other
        return_value = mocker.sentinel.return_value
        patcher = mocker.patch.object(
            BaseMetadata, "combine", return_value=return_value
        )
        result = self.none.combine(other)

        assert result == return_value
        assert patcher.call_count == 1
        (arg,), kwargs = patcher.call_args
        assert arg == other
        assert kwargs == dict(lenient=None)

    def test_lenient(self, mocker):
        other = mocker.sentinel.other
        lenient = mocker.sentinel.lenient
        return_value = mocker.sentinel.return_value
        patcher = mocker.patch.object(
            BaseMetadata, "combine", return_value=return_value
        )
        result = self.none.combine(other, lenient=lenient)

        assert result == return_value
        assert patcher.call_count == 1
        (arg,), kwargs = patcher.call_args
        assert arg == other
        assert kwargs == dict(lenient=lenient)

    def test_op_lenient_same(self, mocker):
        lmetadata = self.cls(**self.values)
        rmetadata = self.cls(**self.values)
        expected = self.values

        mocker.patch("iris.common.metadata._LENIENT", return_value=True)
        assert lmetadata.combine(rmetadata)._asdict() == expected
        assert rmetadata.combine(lmetadata)._asdict() == expected

    def test_op_lenient_same_none(self, mocker):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["var_name"] = None
        rmetadata = self.cls(**right)
        expected = self.values

        mocker.patch("iris.common.metadata._LENIENT", return_value=True)
        assert lmetadata.combine(rmetadata)._asdict() == expected
        assert rmetadata.combine(lmetadata)._asdict() == expected

    def test_op_lenient_same_members_none(self, mocker):
        mocker.patch("iris.common.metadata._LENIENT", return_value=True)

        for member in self.cls._members:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = None
            rmetadata = self.cls(**right)
            expected = right.copy()
            assert lmetadata.combine(rmetadata)._asdict() == expected
            assert rmetadata.combine(lmetadata)._asdict() == expected

    def test_op_lenient_different(self, mocker):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["units"] = self.dummy
        rmetadata = self.cls(**right)
        expected = self.values.copy()
        expected["units"] = None

        mocker.patch("iris.common.metadata._LENIENT", return_value=True)
        assert lmetadata.combine(rmetadata)._asdict() == expected
        assert rmetadata.combine(lmetadata)._asdict() == expected

    def test_op_lenient_different_members(self, mocker):
        mocker.patch("iris.common.metadata._LENIENT", return_value=True)

        for member in self.cls._members:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = self.dummy
            rmetadata = self.cls(**right)
            expected = self.values.copy()
            expected[member] = None
            assert lmetadata.combine(rmetadata)._asdict() == expected
            assert rmetadata.combine(lmetadata)._asdict() == expected

    def test_op_strict_same(self, mocker):
        lmetadata = self.cls(**self.values)
        rmetadata = self.cls(**self.values)
        expected = self.values.copy()

        mocker.patch("iris.common.metadata._LENIENT", return_value=False)
        assert lmetadata.combine(rmetadata)._asdict() == expected
        assert rmetadata.combine(lmetadata)._asdict() == expected

    def test_op_strict_different(self, mocker):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["long_name"] = self.dummy
        rmetadata = self.cls(**right)
        expected = self.values.copy()
        expected["long_name"] = None

        mocker.patch("iris.common.metadata._LENIENT", return_value=False)
        assert lmetadata.combine(rmetadata)._asdict() == expected
        assert rmetadata.combine(lmetadata)._asdict() == expected

    def test_op_strict_different_members(self, mocker):
        mocker.patch("iris.common.metadata._LENIENT", return_value=False)

        for member in self.cls._members:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = self.dummy
            rmetadata = self.cls(**right)
            expected = self.values.copy()
            expected[member] = None
            assert lmetadata.combine(rmetadata)._asdict() == expected
            assert rmetadata.combine(lmetadata)._asdict() == expected

    def test_op_strict_different_none(self, mocker):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["long_name"] = None
        rmetadata = self.cls(**right)
        expected = self.values.copy()
        expected["long_name"] = None

        mocker.patch("iris.common.metadata._LENIENT", return_value=False)
        assert lmetadata.combine(rmetadata)._asdict() == expected
        assert rmetadata.combine(lmetadata)._asdict() == expected

    def test_op_strict_different_members_none(self, mocker):
        mocker.patch("iris.common.metadata._LENIENT", return_value=False)

        for member in self.cls._members:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = None
            rmetadata = self.cls(**right)
            expected = self.values.copy()
            expected[member] = None
            assert lmetadata.combine(rmetadata)._asdict() == expected
            assert rmetadata.combine(lmetadata)._asdict() == expected


class Test_difference:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.values = dict(
            standard_name=mocker.sentinel.standard_name,
            long_name=mocker.sentinel.long_name,
            var_name=mocker.sentinel.var_name,
            units=mocker.sentinel.units,
            attributes=mocker.sentinel.attributes,
            coord_system=mocker.sentinel.coord_system,
            climatological=mocker.sentinel.climatological,
        )
        self.dummy = mocker.sentinel.dummy
        self.cls = CoordMetadata
        self.none = self.cls(*(None,) * len(self.cls._fields))

    def test_wraps_docstring(self):
        assert self.cls.difference.__doc__ == BaseMetadata.difference.__doc__

    def test_lenient_service(self):
        qualname_difference = _qualname(self.cls.difference)
        assert qualname_difference in _LENIENT
        assert _LENIENT[qualname_difference]
        assert _LENIENT[self.cls.difference]

    def test_lenient_default(self, mocker):
        other = mocker.sentinel.other
        return_value = mocker.sentinel.return_value
        patcher = mocker.patch.object(
            BaseMetadata, "difference", return_value=return_value
        )
        result = self.none.difference(other)

        assert result == return_value
        assert patcher.call_count == 1
        (arg,), kwargs = patcher.call_args
        assert arg == other
        assert kwargs == dict(lenient=None)

    def test_lenient(self, mocker):
        other = mocker.sentinel.other
        lenient = mocker.sentinel.lenient
        return_value = mocker.sentinel.return_value
        patcher = mocker.patch.object(
            BaseMetadata, "difference", return_value=return_value
        )
        result = self.none.difference(other, lenient=lenient)

        assert result == return_value
        assert patcher.call_count == 1
        (arg,), kwargs = patcher.call_args
        assert arg == other
        assert kwargs == dict(lenient=lenient)

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
        mocker.patch("iris.common.metadata._LENIENT", return_value=True)

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
            assert lmetadata.difference(rmetadata)._asdict() == lexpected
            assert rmetadata.difference(lmetadata)._asdict() == rexpected

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
        assert lmetadata.difference(rmetadata)._asdict() == lexpected
        assert rmetadata.difference(lmetadata)._asdict() == rexpected

    def test_op_lenient_different_members(self, mocker):
        mocker.patch("iris.common.metadata._LENIENT", return_value=True)

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
            assert lmetadata.difference(rmetadata)._asdict() == lexpected
            assert rmetadata.difference(lmetadata)._asdict() == rexpected

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
        assert lmetadata.difference(rmetadata)._asdict() == lexpected
        assert rmetadata.difference(lmetadata)._asdict() == rexpected

    def test_op_strict_different_members(self, mocker):
        mocker.patch("iris.common.metadata._LENIENT", return_value=False)

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
            assert lmetadata.difference(rmetadata)._asdict() == lexpected
            assert rmetadata.difference(lmetadata)._asdict() == rexpected

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
        assert lmetadata.difference(rmetadata)._asdict() == lexpected
        assert rmetadata.difference(lmetadata)._asdict() == rexpected

    def test_op_strict_different_members_none(self, mocker):
        mocker.patch("iris.common.metadata._LENIENT", return_value=False)

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
            assert lmetadata.difference(rmetadata)._asdict() == lexpected
            assert rmetadata.difference(lmetadata)._asdict() == rexpected


class Test_equal:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cls = CoordMetadata
        self.none = self.cls(*(None,) * len(self.cls._fields))

    def test_wraps_docstring(self):
        assert self.cls.equal.__doc__ == BaseMetadata.equal.__doc__

    def test_lenient_service(self):
        qualname_equal = _qualname(self.cls.equal)
        assert qualname_equal in _LENIENT
        assert _LENIENT[qualname_equal]
        assert _LENIENT[self.cls.equal]

    def test_lenient_default(self, mocker):
        other = mocker.sentinel.other
        return_value = mocker.sentinel.return_value
        patcher = mocker.patch.object(BaseMetadata, "equal", return_value=return_value)
        result = self.none.equal(other)

        assert result == return_value
        assert patcher.call_count == 1
        (arg,), kwargs = patcher.call_args
        assert arg == other
        assert kwargs == dict(lenient=None)

    def test_lenient(self, mocker):
        other = mocker.sentinel.other
        lenient = mocker.sentinel.lenient
        return_value = mocker.sentinel.return_value
        patcher = mocker.patch.object(BaseMetadata, "equal", return_value=return_value)
        result = self.none.equal(other, lenient=lenient)

        assert result == return_value
        assert patcher.call_count == 1
        (arg,), kwargs = patcher.call_args
        assert arg == other
        assert kwargs == dict(lenient=lenient)
