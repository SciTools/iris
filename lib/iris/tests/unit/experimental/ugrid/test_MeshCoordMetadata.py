# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :class:`iris.experimental.ugrid.MeshCoordMetadata`.

"""

from copy import deepcopy
import unittest.mock as mock
from unittest.mock import sentinel

from iris.common.lenient import _LENIENT, _qualname
from iris.common.metadata import BaseMetadata
from iris.experimental.ugrid import MeshCoordMetadata

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests


class Test__identity(tests.IrisTest):
    def setUp(self):
        self.standard_name = mock.sentinel.standard_name
        self.long_name = mock.sentinel.long_name
        self.var_name = mock.sentinel.var_name
        self.units = mock.sentinel.units
        self.attributes = mock.sentinel.attributes
        self.location = mock.sentinel.location
        self.axis = mock.sentinel.axis
        self.cls = MeshCoordMetadata

    def test_repr(self):
        metadata = self.cls(
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
            units=self.units,
            attributes=self.attributes,
            location=self.location,
            axis=self.axis,
        )
        fmt = (
            "MeshCoordMetadata(standard_name={!r}, long_name={!r}, "
            "var_name={!r}, units={!r}, attributes={!r}, "
            "location={!r}, axis={!r})"
        )
        expected = fmt.format(
            self.standard_name,
            self.long_name,
            self.var_name,
            self.units,
            self.attributes,
            self.location,
            self.axis,
        )
        self.assertEqual(expected, repr(metadata))

    def test__fields(self):
        expected = (
            "standard_name",
            "long_name",
            "var_name",
            "units",
            "attributes",
            "location",
            "axis",
        )
        self.assertEqual(self.cls._fields, expected)

    def test_bases(self):
        self.assertTrue(issubclass(self.cls, BaseMetadata))


class Test__eq__(tests.IrisTest):
    def setUp(self):
        self.values = dict(
            standard_name=sentinel.standard_name,
            long_name=sentinel.long_name,
            var_name=sentinel.var_name,
            units=sentinel.units,
            attributes=sentinel.attributes,
            location=sentinel.location,
            axis=sentinel.axis,
        )
        self.dummy = sentinel.dummy
        self.cls = MeshCoordMetadata

    def test_wraps_docstring(self):
        self.assertEqual(BaseMetadata.__eq__.__doc__, self.cls.__eq__.__doc__)

    def test_lenient_service(self):
        qualname___eq__ = _qualname(self.cls.__eq__)
        self.assertIn(qualname___eq__, _LENIENT)
        self.assertTrue(_LENIENT[qualname___eq__])
        self.assertTrue(_LENIENT[self.cls.__eq__])

    def test_call(self):
        other = sentinel.other
        return_value = sentinel.return_value
        metadata = self.cls(*(None,) * len(self.cls._fields))
        with mock.patch.object(
            BaseMetadata, "__eq__", return_value=return_value
        ) as mocker:
            result = metadata.__eq__(other)

        self.assertEqual(return_value, result)
        self.assertEqual(1, mocker.call_count)
        (arg,), kwargs = mocker.call_args
        self.assertEqual(other, arg)
        self.assertEqual(dict(), kwargs)

    def test_op_lenient_same(self):
        lmetadata = self.cls(**self.values)
        rmetadata = self.cls(**self.values)

        with mock.patch("iris.common.metadata._LENIENT", return_value=True):
            self.assertTrue(lmetadata.__eq__(rmetadata))
            self.assertTrue(rmetadata.__eq__(lmetadata))

    def test_op_lenient_same_none_nonmember(self):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["long_name"] = None
        rmetadata = self.cls(**right)

        with mock.patch("iris.common.metadata._LENIENT", return_value=True):
            self.assertTrue(lmetadata.__eq__(rmetadata))
            self.assertTrue(rmetadata.__eq__(lmetadata))

    def test_op_lenient_same_members_none(self):
        for member in self.cls._members:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = None
            rmetadata = self.cls(**right)

            with mock.patch(
                "iris.common.metadata._LENIENT", return_value=True
            ):
                self.assertFalse(lmetadata.__eq__(rmetadata))
                self.assertFalse(rmetadata.__eq__(lmetadata))

    def test_op_lenient_different(self):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["long_name"] = self.dummy
        rmetadata = self.cls(**right)

        with mock.patch("iris.common.metadata._LENIENT", return_value=True):
            self.assertFalse(lmetadata.__eq__(rmetadata))
            self.assertFalse(rmetadata.__eq__(lmetadata))

    def test_op_lenient_different_members(self):
        for member in self.cls._members:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = self.dummy
            rmetadata = self.cls(**right)

            with mock.patch(
                "iris.common.metadata._LENIENT", return_value=True
            ):
                self.assertFalse(lmetadata.__eq__(rmetadata))
                self.assertFalse(rmetadata.__eq__(lmetadata))

    def test_op_strict_same(self):
        lmetadata = self.cls(**self.values)
        rmetadata = self.cls(**self.values)

        with mock.patch("iris.common.metadata._LENIENT", return_value=False):
            self.assertTrue(lmetadata.__eq__(rmetadata))
            self.assertTrue(rmetadata.__eq__(lmetadata))

    def test_op_strict_different(self):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["long_name"] = self.dummy
        rmetadata = self.cls(**right)

        with mock.patch("iris.common.metadata._LENIENT", return_value=False):
            self.assertFalse(lmetadata.__eq__(rmetadata))
            self.assertFalse(rmetadata.__eq__(lmetadata))

    def test_op_strict_different_members(self):
        for member in self.cls._members:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = self.dummy
            rmetadata = self.cls(**right)

            with mock.patch(
                "iris.common.metadata._LENIENT", return_value=False
            ):
                self.assertFalse(lmetadata.__eq__(rmetadata))
                self.assertFalse(rmetadata.__eq__(lmetadata))

    def test_op_strict_different_none(self):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["long_name"] = None
        rmetadata = self.cls(**right)

        with mock.patch("iris.common.metadata._LENIENT", return_value=False):
            self.assertFalse(lmetadata.__eq__(rmetadata))
            self.assertFalse(rmetadata.__eq__(lmetadata))

    def test_op_strict_different_members_none(self):
        for member in self.cls._members:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = None
            rmetadata = self.cls(**right)

            with mock.patch(
                "iris.common.metadata._LENIENT", return_value=False
            ):
                self.assertFalse(lmetadata.__eq__(rmetadata))
                self.assertFalse(rmetadata.__eq__(lmetadata))


class Test___lt__(tests.IrisTest):
    def setUp(self):
        self.cls = MeshCoordMetadata
        values = [1] * len(self.cls._fields)
        self.one = self.cls(*values)

        values_two = values[:]
        values_two[2] = 2
        self.two = self.cls(*values_two)

        values_none = values[:]
        values_none[2] = None
        self.none = self.cls(*values_none)

        values_attrs = values[:]
        values_attrs[4] = 10
        self.attributes = self.cls(*values_attrs)

    def test__ascending_lt(self):
        result = self.one < self.two
        self.assertTrue(result)

    def test__descending_lt(self):
        result = self.two < self.one
        self.assertFalse(result)

    def test__none_rhs_operand(self):
        result = self.one < self.none
        self.assertFalse(result)

    def test__none_lhs_operand(self):
        result = self.none < self.one
        self.assertTrue(result)

    def test__ignore_attributes(self):
        result = self.one < self.attributes
        self.assertFalse(result)
        result = self.attributes < self.one
        self.assertFalse(result)


class Test_combine(tests.IrisTest):
    def setUp(self):
        self.cls = MeshCoordMetadata
        self.values = dict(
            standard_name=sentinel.standard_name,
            long_name=sentinel.long_name,
            var_name=sentinel.var_name,
            units=sentinel.units,
            attributes=sentinel.attributes,
            location=sentinel.location,
            axis=sentinel.axis,
        )
        self.dummy = sentinel.dummy
        self.none = self.cls(*(None,) * len(self.cls._fields))

    def test_wraps_docstring(self):
        self.assertEqual(
            BaseMetadata.combine.__doc__, self.cls.combine.__doc__
        )

    def test_lenient_service(self):
        qualname_combine = _qualname(self.cls.combine)
        self.assertIn(qualname_combine, _LENIENT)
        self.assertTrue(_LENIENT[qualname_combine])
        self.assertTrue(_LENIENT[self.cls.combine])

    def test_lenient_default(self):
        other = sentinel.other
        return_value = sentinel.return_value
        with mock.patch.object(
            BaseMetadata, "combine", return_value=return_value
        ) as mocker:
            result = self.none.combine(other)

        self.assertEqual(return_value, result)
        self.assertEqual(1, mocker.call_count)
        (arg,), kwargs = mocker.call_args
        self.assertEqual(other, arg)
        self.assertEqual(dict(lenient=None), kwargs)

    def test_lenient(self):
        other = sentinel.other
        lenient = sentinel.lenient
        return_value = sentinel.return_value
        with mock.patch.object(
            BaseMetadata, "combine", return_value=return_value
        ) as mocker:
            result = self.none.combine(other, lenient=lenient)

        self.assertEqual(return_value, result)
        self.assertEqual(1, mocker.call_count)
        (arg,), kwargs = mocker.call_args
        self.assertEqual(other, arg)
        self.assertEqual(dict(lenient=lenient), kwargs)

    def test_op_lenient_same(self):
        lmetadata = self.cls(**self.values)
        rmetadata = self.cls(**self.values)
        expected = self.values

        with mock.patch("iris.common.metadata._LENIENT", return_value=True):
            self.assertEqual(expected, lmetadata.combine(rmetadata)._asdict())
            self.assertEqual(expected, rmetadata.combine(lmetadata)._asdict())

    def test_op_lenient_same_none(self):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["long_name"] = None
        rmetadata = self.cls(**right)
        expected = self.values

        with mock.patch("iris.common.metadata._LENIENT", return_value=True):
            self.assertEqual(expected, lmetadata.combine(rmetadata)._asdict())
            self.assertEqual(expected, rmetadata.combine(lmetadata)._asdict())

    def test_op_lenient_same_members_none(self):
        for member in self.cls._members:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = None
            rmetadata = self.cls(**right)
            expected = right.copy()

            with mock.patch(
                "iris.common.metadata._LENIENT", return_value=True
            ):
                self.assertEqual(
                    expected, lmetadata.combine(rmetadata)._asdict()
                )
                self.assertEqual(
                    expected, rmetadata.combine(lmetadata)._asdict()
                )

    def test_op_lenient_different(self):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["long_name"] = self.dummy
        rmetadata = self.cls(**right)
        expected = self.values.copy()
        expected["long_name"] = None

        with mock.patch("iris.common.metadata._LENIENT", return_value=True):
            self.assertEqual(expected, lmetadata.combine(rmetadata)._asdict())
            self.assertEqual(expected, rmetadata.combine(lmetadata)._asdict())

    def test_op_lenient_different_members(self):
        for member in self.cls._members:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = self.dummy
            rmetadata = self.cls(**right)
            expected = self.values.copy()
            expected[member] = None

            with mock.patch(
                "iris.common.metadata._LENIENT", return_value=True
            ):
                self.assertEqual(
                    expected, lmetadata.combine(rmetadata)._asdict()
                )
                self.assertEqual(
                    expected, rmetadata.combine(lmetadata)._asdict()
                )

    def test_op_strict_same(self):
        lmetadata = self.cls(**self.values)
        rmetadata = self.cls(**self.values)
        expected = self.values.copy()

        with mock.patch("iris.common.metadata._LENIENT", return_value=False):
            self.assertEqual(expected, lmetadata.combine(rmetadata)._asdict())
            self.assertEqual(expected, rmetadata.combine(lmetadata)._asdict())

    def test_op_strict_different(self):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["long_name"] = self.dummy
        rmetadata = self.cls(**right)
        expected = self.values.copy()
        expected["long_name"] = None

        with mock.patch("iris.common.metadata._LENIENT", return_value=False):
            self.assertEqual(expected, lmetadata.combine(rmetadata)._asdict())
            self.assertEqual(expected, rmetadata.combine(lmetadata)._asdict())

    def test_op_strict_different_members(self):
        for member in self.cls._members:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = self.dummy
            rmetadata = self.cls(**right)
            expected = self.values.copy()
            expected[member] = None

            with mock.patch(
                "iris.common.metadata._LENIENT", return_value=False
            ):
                self.assertEqual(
                    expected, lmetadata.combine(rmetadata)._asdict()
                )
                self.assertEqual(
                    expected, rmetadata.combine(lmetadata)._asdict()
                )

    def test_op_strict_different_none(self):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["long_name"] = None
        rmetadata = self.cls(**right)
        expected = self.values.copy()
        expected["long_name"] = None

        with mock.patch("iris.common.metadata._LENIENT", return_value=False):
            self.assertEqual(expected, lmetadata.combine(rmetadata)._asdict())
            self.assertEqual(expected, rmetadata.combine(lmetadata)._asdict())

    def test_op_strict_different_members_none(self):
        for member in self.cls._members:
            lmetadata = self.cls(**self.values)
            right = self.values.copy()
            right[member] = None
            rmetadata = self.cls(**right)
            expected = self.values.copy()
            expected[member] = None

            with mock.patch(
                "iris.common.metadata._LENIENT", return_value=False
            ):
                self.assertEqual(
                    expected, lmetadata.combine(rmetadata)._asdict()
                )
                self.assertEqual(
                    expected, rmetadata.combine(lmetadata)._asdict()
                )


class Test_difference(tests.IrisTest):
    def setUp(self):
        self.cls = MeshCoordMetadata
        self.values = dict(
            standard_name=sentinel.standard_name,
            long_name=sentinel.long_name,
            var_name=sentinel.var_name,
            units=sentinel.units,
            attributes=sentinel.attributes,
            location=sentinel.location,
            axis=sentinel.axis,
        )
        self.dummy = sentinel.dummy
        self.none = self.cls(*(None,) * len(self.cls._fields))

    def test_wraps_docstring(self):
        self.assertEqual(
            BaseMetadata.difference.__doc__, self.cls.difference.__doc__
        )

    def test_lenient_service(self):
        qualname_difference = _qualname(self.cls.difference)
        self.assertIn(qualname_difference, _LENIENT)
        self.assertTrue(_LENIENT[qualname_difference])
        self.assertTrue(_LENIENT[self.cls.difference])

    def test_lenient_default(self):
        other = sentinel.other
        return_value = sentinel.return_value
        with mock.patch.object(
            BaseMetadata, "difference", return_value=return_value
        ) as mocker:
            result = self.none.difference(other)

        self.assertEqual(return_value, result)
        self.assertEqual(1, mocker.call_count)
        (arg,), kwargs = mocker.call_args
        self.assertEqual(other, arg)
        self.assertEqual(dict(lenient=None), kwargs)

    def test_lenient(self):
        other = sentinel.other
        lenient = sentinel.lenient
        return_value = sentinel.return_value
        with mock.patch.object(
            BaseMetadata, "difference", return_value=return_value
        ) as mocker:
            result = self.none.difference(other, lenient=lenient)

        self.assertEqual(return_value, result)
        self.assertEqual(1, mocker.call_count)
        (arg,), kwargs = mocker.call_args
        self.assertEqual(other, arg)
        self.assertEqual(dict(lenient=lenient), kwargs)

    def test_op_lenient_same(self):
        lmetadata = self.cls(**self.values)
        rmetadata = self.cls(**self.values)

        with mock.patch("iris.common.metadata._LENIENT", return_value=True):
            self.assertIsNone(lmetadata.difference(rmetadata))
            self.assertIsNone(rmetadata.difference(lmetadata))

    def test_op_lenient_same_none(self):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["long_name"] = None
        rmetadata = self.cls(**right)

        with mock.patch("iris.common.metadata._LENIENT", return_value=True):
            self.assertIsNone(lmetadata.difference(rmetadata))
            self.assertIsNone(rmetadata.difference(lmetadata))

    def test_op_lenient_same_members_none(self):
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

            with mock.patch(
                "iris.common.metadata._LENIENT", return_value=True
            ):
                self.assertEqual(
                    lexpected, lmetadata.difference(rmetadata)._asdict()
                )
                self.assertEqual(
                    rexpected, rmetadata.difference(lmetadata)._asdict()
                )

    def test_op_lenient_different(self):
        left = self.values.copy()
        lmetadata = self.cls(**left)
        right = self.values.copy()
        right["long_name"] = self.dummy
        rmetadata = self.cls(**right)
        lexpected = deepcopy(self.none)._asdict()
        lexpected["long_name"] = (left["long_name"], right["long_name"])
        rexpected = deepcopy(self.none)._asdict()
        rexpected["long_name"] = lexpected["long_name"][::-1]

        with mock.patch("iris.common.metadata._LENIENT", return_value=True):
            self.assertEqual(
                lexpected, lmetadata.difference(rmetadata)._asdict()
            )
            self.assertEqual(
                rexpected, rmetadata.difference(lmetadata)._asdict()
            )

    def test_op_lenient_different_members(self):
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

            with mock.patch(
                "iris.common.metadata._LENIENT", return_value=True
            ):
                self.assertEqual(
                    lexpected, lmetadata.difference(rmetadata)._asdict()
                )
                self.assertEqual(
                    rexpected, rmetadata.difference(lmetadata)._asdict()
                )

    def test_op_strict_same(self):
        lmetadata = self.cls(**self.values)
        rmetadata = self.cls(**self.values)

        with mock.patch("iris.common.metadata._LENIENT", return_value=False):
            self.assertIsNone(lmetadata.difference(rmetadata))
            self.assertIsNone(rmetadata.difference(lmetadata))

    def test_op_strict_different(self):
        left = self.values.copy()
        lmetadata = self.cls(**left)
        right = self.values.copy()
        right["long_name"] = self.dummy
        rmetadata = self.cls(**right)
        lexpected = deepcopy(self.none)._asdict()
        lexpected["long_name"] = (left["long_name"], right["long_name"])
        rexpected = deepcopy(self.none)._asdict()
        rexpected["long_name"] = lexpected["long_name"][::-1]

        with mock.patch("iris.common.metadata._LENIENT", return_value=False):
            self.assertEqual(
                lexpected, lmetadata.difference(rmetadata)._asdict()
            )
            self.assertEqual(
                rexpected, rmetadata.difference(lmetadata)._asdict()
            )

    def test_op_strict_different_members(self):
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

            with mock.patch(
                "iris.common.metadata._LENIENT", return_value=False
            ):
                self.assertEqual(
                    lexpected, lmetadata.difference(rmetadata)._asdict()
                )
                self.assertEqual(
                    rexpected, rmetadata.difference(lmetadata)._asdict()
                )

    def test_op_strict_different_none(self):
        left = self.values.copy()
        lmetadata = self.cls(**left)
        right = self.values.copy()
        right["long_name"] = None
        rmetadata = self.cls(**right)
        lexpected = deepcopy(self.none)._asdict()
        lexpected["long_name"] = (left["long_name"], right["long_name"])
        rexpected = deepcopy(self.none)._asdict()
        rexpected["long_name"] = lexpected["long_name"][::-1]

        with mock.patch("iris.common.metadata._LENIENT", return_value=False):
            self.assertEqual(
                lexpected, lmetadata.difference(rmetadata)._asdict()
            )
            self.assertEqual(
                rexpected, rmetadata.difference(lmetadata)._asdict()
            )

    def test_op_strict_different_members_none(self):
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

            with mock.patch(
                "iris.common.metadata._LENIENT", return_value=False
            ):
                self.assertEqual(
                    lexpected, lmetadata.difference(rmetadata)._asdict()
                )
                self.assertEqual(
                    rexpected, rmetadata.difference(lmetadata)._asdict()
                )


class Test_equal(tests.IrisTest):
    def setUp(self):
        self.cls = MeshCoordMetadata
        self.none = self.cls(*(None,) * len(self.cls._fields))

    def test_wraps_docstring(self):
        self.assertEqual(BaseMetadata.equal.__doc__, self.cls.equal.__doc__)

    def test_lenient_service(self):
        qualname_equal = _qualname(self.cls.equal)
        self.assertIn(qualname_equal, _LENIENT)
        self.assertTrue(_LENIENT[qualname_equal])
        self.assertTrue(_LENIENT[self.cls.equal])

    def test_lenient_default(self):
        other = sentinel.other
        return_value = sentinel.return_value
        with mock.patch.object(
            BaseMetadata, "equal", return_value=return_value
        ) as mocker:
            result = self.none.equal(other)

        self.assertEqual(return_value, result)
        self.assertEqual(1, mocker.call_count)
        (arg,), kwargs = mocker.call_args
        self.assertEqual(other, arg)
        self.assertEqual(dict(lenient=None), kwargs)

    def test_lenient(self):
        other = sentinel.other
        lenient = sentinel.lenient
        return_value = sentinel.return_value
        with mock.patch.object(
            BaseMetadata, "equal", return_value=return_value
        ) as mocker:
            result = self.none.equal(other, lenient=lenient)

        self.assertEqual(return_value, result)
        self.assertEqual(1, mocker.call_count)
        (arg,), kwargs = mocker.call_args
        self.assertEqual(other, arg)
        self.assertEqual(dict(lenient=lenient), kwargs)


if __name__ == "__main__":
    tests.main()
