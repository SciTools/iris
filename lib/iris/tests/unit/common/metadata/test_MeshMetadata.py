# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.common.metadata.MeshMetadata`."""

from copy import deepcopy

import pytest

from iris.common.lenient import _LENIENT, _qualname
from iris.common.metadata import BaseMetadata, MeshMetadata


class Test:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.standard_name = mocker.sentinel.standard_name
        self.long_name = mocker.sentinel.long_name
        self.var_name = mocker.sentinel.var_name
        self.units = mocker.sentinel.units
        self.attributes = mocker.sentinel.attributes
        self.topology_dimension = mocker.sentinel.topology_dimension
        self.node_dimension = mocker.sentinel.node_dimension
        self.edge_dimension = mocker.sentinel.edge_dimension
        self.face_dimension = mocker.sentinel.face_dimension
        self.cls = MeshMetadata

    def test_repr(self, mocker):
        metadata = self.cls(
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
            units=self.units,
            attributes=self.attributes,
            topology_dimension=self.topology_dimension,
            node_dimension=self.node_dimension,
            edge_dimension=self.edge_dimension,
            face_dimension=self.face_dimension,
        )
        fmt = (
            "MeshMetadata(standard_name={!r}, long_name={!r}, "
            "var_name={!r}, units={!r}, attributes={!r}, "
            "topology_dimension={!r}, node_dimension={!r}, "
            "edge_dimension={!r}, face_dimension={!r})"
        )
        expected = fmt.format(
            self.standard_name,
            self.long_name,
            self.var_name,
            self.units,
            self.attributes,
            self.topology_dimension,
            self.node_dimension,
            self.edge_dimension,
            self.face_dimension,
        )
        assert expected == repr(metadata)

    def test__fields(self, mocker):
        expected = (
            "standard_name",
            "long_name",
            "var_name",
            "units",
            "attributes",
            "topology_dimension",
            "node_dimension",
            "edge_dimension",
            "face_dimension",
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
            topology_dimension=mocker.sentinel.topology_dimension,
            node_dimension=mocker.sentinel.node_dimension,
            edge_dimension=mocker.sentinel.edge_dimension,
            face_dimension=mocker.sentinel.face_dimension,
        )
        self.dummy = mocker.sentinel.dummy
        self.cls = MeshMetadata
        # The "node_dimension", "edge_dimension" and "face_dimension" members
        # are stateful only; they do not participate in lenient/strict equivalence.
        self.members_dim_names = filter(
            lambda member: member
            in ("node_dimension", "edge_dimension", "face_dimension"),
            self.cls._members,
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

    def test_op_lenient_same_topology_dim_none(self, mocker):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["topology_dimension"] = None
        rmetadata = self.cls(**right)

        mocker.patch("iris.common.metadata._LENIENT", return_value=True)
        assert not lmetadata.__eq__(rmetadata)
        assert not rmetadata.__eq__(lmetadata)

    def test_op_lenient_same_dim_names_none(self, mocker):
        for member in self.members_dim_names:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = None
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

    def test_op_lenient_different_topology_dim(self, mocker):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["topology_dimension"] = self.dummy
        rmetadata = self.cls(**right)

        mocker.patch("iris.common.metadata._LENIENT", return_value=True)
        assert not lmetadata.__eq__(rmetadata)
        assert not rmetadata.__eq__(lmetadata)

    def test_op_lenient_different_dim_names(self, mocker):
        for member in self.members_dim_names:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = self.dummy
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

    def test_op_strict_different_topology_dim(self, mocker):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["topology_dimension"] = self.dummy
        rmetadata = self.cls(**right)

        mocker.patch("iris.common.metadata._LENIENT", return_value=False)
        assert not lmetadata.__eq__(rmetadata)
        assert not rmetadata.__eq__(lmetadata)

    def test_op_strict_different_dim_names(self, mocker):
        for member in self.members_dim_names:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = self.dummy
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

    def test_op_strict_different_topology_dim_none(self, mocker):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["topology_dimension"] = None
        rmetadata = self.cls(**right)

        mocker.patch("iris.common.metadata._LENIENT", return_value=False)
        assert not lmetadata.__eq__(rmetadata)
        assert not rmetadata.__eq__(lmetadata)

    def test_op_strict_different_dim_names_none(self, mocker):
        for member in self.members_dim_names:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = None
            rmetadata = self.cls(**right)

            mocker.patch("iris.common.metadata._LENIENT", return_value=False)
            assert lmetadata.__eq__(rmetadata)
            assert rmetadata.__eq__(lmetadata)


class Test___lt__:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.cls = MeshMetadata
        self.one = self.cls(1, 1, 1, 1, 1, 1, 1, 1, 1)
        self.two = self.cls(1, 1, 1, 2, 1, 1, 1, 1, 1)
        self.none = self.cls(1, 1, 1, None, 1, 1, 1, 1, 1)
        self.attributes = self.cls(1, 1, 1, 1, 10, 1, 1, 1, 1)

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
            topology_dimension=mocker.sentinel.topology_dimension,
            node_dimension=mocker.sentinel.node_dimension,
            edge_dimension=mocker.sentinel.edge_dimension,
            face_dimension=mocker.sentinel.face_dimension,
        )
        self.dummy = mocker.sentinel.dummy
        self.cls = MeshMetadata
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
            topology_dimension=mocker.sentinel.topology_dimension,
            node_dimension=mocker.sentinel.node_dimension,
            edge_dimension=mocker.sentinel.edge_dimension,
            face_dimension=mocker.sentinel.face_dimension,
        )
        self.dummy = mocker.sentinel.dummy
        self.cls = MeshMetadata
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
        self.cls = MeshMetadata
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
