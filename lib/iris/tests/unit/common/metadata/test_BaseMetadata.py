# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :class:`iris.common.metadata.BaseMetadata`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from collections import OrderedDict
import unittest.mock as mock
from unittest.mock import sentinel

from iris.common.lenient import LENIENT, qualname
from iris.common.metadata import BaseMetadata, CubeMetadata


class Test(tests.IrisTest):
    def setUp(self):
        self.standard_name = mock.sentinel.standard_name
        self.long_name = mock.sentinel.long_name
        self.var_name = mock.sentinel.var_name
        self.units = mock.sentinel.units
        self.attributes = mock.sentinel.attributes

    def test_repr(self):
        metadata = BaseMetadata(
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
        self.assertEqual(expected, repr(metadata))

    def test__fields(self):
        expected = (
            "standard_name",
            "long_name",
            "var_name",
            "units",
            "attributes",
        )
        self.assertEqual(BaseMetadata._fields, expected)


class Test__combine(tests.IrisTest):
    def setUp(self):
        self.kwargs = dict(
            standard_name="standard_name",
            long_name="long_name",
            var_name="var_name",
            units="units",
            attributes=dict(one=1, two=2),
        )
        self.metadata = BaseMetadata(**self.kwargs)

    def test_lenient(self):
        return_value = sentinel._combine_lenient
        other = sentinel.other
        with mock.patch(
            "iris.common.metadata.LENIENT", return_value=True
        ) as mlenient:
            with mock.patch.object(
                BaseMetadata, "_combine_lenient", return_value=return_value
            ) as mcombine:
                result = self.metadata._combine(other)

                self.assertEqual(mlenient.call_count, 1)
                (args,), kwargs = mlenient.call_args
                self.assertEqual(args, self.metadata.combine)
                self.assertEqual(kwargs, dict())

                self.assertEqual(result, return_value)
                self.assertEqual(mcombine.call_count, 1)
                (args,), kwargs = mcombine.call_args
                self.assertEqual(args, other)
                self.assertEqual(kwargs, dict())

    def test_strict(self):
        dummy = sentinel.dummy
        values = self.kwargs.copy()
        values["standard_name"] = dummy
        values["var_name"] = dummy
        values["attributes"] = dummy
        other = BaseMetadata(**values)
        with mock.patch("iris.common.metadata.LENIENT", return_value=False):
            result = self.metadata._combine(other)

        expected = [
            None if values[field] == dummy else values[field]
            for field in BaseMetadata._fields
        ]
        self.assertEqual(result, expected)


class Test__combine_lenient(tests.IrisTest):
    def setUp(self):
        self.none = BaseMetadata(
            *(None,) * len(BaseMetadata._fields)
        )._asdict()
        self.names = dict(
            standard_name=sentinel.standard_name,
            long_name=sentinel.long_name,
            var_name=sentinel.var_name,
        )

    def test_strict_units(self):
        left = self.none.copy()
        left["units"] = "K"
        right = left.copy()
        lmetadata = BaseMetadata(**left)
        rmetadata = BaseMetadata(**right)
        result = lmetadata._combine_lenient(rmetadata)
        expected = list(left.values())
        self.assertEqual(expected, result)

    def test_strict_units_different(self):
        left = self.none.copy()
        right = self.none.copy()
        left["units"] = "K"
        right["units"] = "km"
        lmetadata = BaseMetadata(**left)
        rmetadata = BaseMetadata(**right)
        result = lmetadata._combine_lenient(rmetadata)
        expected = list(self.none.values())
        self.assertEqual(expected, result)

    def test_strict_units_different_none(self):
        left = self.none.copy()
        right = self.none.copy()
        left["units"] = "K"
        lmetadata = BaseMetadata(**left)
        rmetadata = BaseMetadata(**right)
        result = lmetadata._combine_lenient(rmetadata)
        expected = list(self.none.values())
        self.assertEqual(expected, result)
        result = rmetadata._combine_lenient(lmetadata)
        self.assertEqual(expected, result)

    def test_attributes(self):
        left = self.none.copy()
        right = self.none.copy()
        ldict = dict(item=sentinel.left)
        rdict = dict(item=sentinel.right)
        left["attributes"] = ldict
        right["attributes"] = rdict
        rmetadata = BaseMetadata(**right)
        return_value = sentinel.return_value
        with mock.patch.object(
            BaseMetadata,
            "_combine_lenient_attributes",
            return_value=return_value,
        ) as mocker:
            lmetadata = BaseMetadata(**left)
            result = lmetadata._combine_lenient(rmetadata)

        expected = self.none.copy()
        expected["attributes"] = return_value
        expected = list(expected.values())
        self.assertEqual(expected, result)

        self.assertEqual(mocker.call_count, 1)
        args, kwargs = mocker.call_args
        expected = (ldict, rdict)
        self.assertEqual(expected, args)

    def test_attributes_non_mapping_different(self):
        left = self.none.copy()
        right = self.none.copy()
        ldict = dict(item=sentinel.left)
        rdict = sentinel.right
        left["attributes"] = ldict
        right["attributes"] = rdict
        lmetadata = BaseMetadata(**left)
        rmetadata = BaseMetadata(**right)
        result = lmetadata._combine_lenient(rmetadata)
        expected = list(self.none.copy().values())
        self.assertEqual(expected, result)

    def test_attributes_non_mapping_different_none(self):
        left = self.none.copy()
        right = self.none.copy()
        ldict = dict(item=sentinel.left)
        left["attributes"] = ldict
        lmetadata = BaseMetadata(**left)
        rmetadata = BaseMetadata(**right)

        result = lmetadata._combine_lenient(rmetadata)
        expected = self.none.copy()
        expected["attributes"] = ldict
        expected = list(expected.values())
        self.assertEqual(expected, result)

        result = rmetadata._combine_lenient(lmetadata)
        self.assertEqual(expected, result)

    def test_names(self):
        left = self.none.copy()
        left.update(self.names)
        right = left.copy()
        lmetadata = BaseMetadata(**left)
        rmetadata = BaseMetadata(**right)
        result = lmetadata._combine_lenient(rmetadata)
        expected = list(left.values())
        self.assertEqual(expected, result)

    def test_names_different(self):
        dummy = sentinel.dummy
        left = self.none.copy()
        right = self.none.copy()
        left.update(self.names)
        right["standard_name"] = dummy
        right["long_name"] = dummy
        right["var_name"] = dummy
        lmetadata = BaseMetadata(**left)
        rmetadata = BaseMetadata(**right)
        result = lmetadata._combine_lenient(rmetadata)
        expected = list(self.none.copy().values())
        self.assertEqual(expected, result)

    def test_names_different_none(self):
        left = self.none.copy()
        right = self.none.copy()
        left.update(self.names)
        lmetadata = BaseMetadata(**left)
        rmetadata = BaseMetadata(**right)

        result = lmetadata._combine_lenient(rmetadata)
        expected = list(left.values())
        self.assertEqual(expected, result)

        result = rmetadata._combine_lenient(lmetadata)
        self.assertEqual(expected, result)


class Test__combine_lenient_attributes(tests.IrisTest):
    def setUp(self):
        self.values = OrderedDict(
            one=sentinel.one,
            two=sentinel.two,
            three=sentinel.three,
            four=sentinel.four,
            five=sentinel.five,
        )
        self.metadata = BaseMetadata(*(None,) * len(BaseMetadata._fields))
        self.dummy = sentinel.dummy

    def test_same(self):
        left = self.values.copy()
        right = self.values.copy()
        result = self.metadata._combine_lenient_attributes(left, right)
        expected = dict(**left)
        self.assertEqual(expected, result)

    def test_different(self):
        left = self.values.copy()
        right = self.values.copy()
        left["two"] = left["four"] = self.dummy
        result = self.metadata._combine_lenient_attributes(left, right)
        expected = self.values.copy()
        for key in ["two", "four"]:
            del expected[key]
        self.assertEqual(dict(expected), result)

    def test_different_none(self):
        left = self.values.copy()
        right = self.values.copy()
        left["one"] = left["three"] = left["five"] = self.dummy
        result = self.metadata._combine_lenient_attributes(left, right)
        expected = self.values.copy()
        for key in ["one", "three", "five"]:
            del expected[key]
        self.assertEqual(dict(expected), result)

    def test_extra(self):
        left = self.values.copy()
        right = self.values.copy()
        left["extra_left"] = sentinel.extra_left
        right["extra_right"] = sentinel.extra_right
        result = self.metadata._combine_lenient_attributes(left, right)
        expected = self.values.copy()
        expected["extra_left"] = left["extra_left"]
        expected["extra_right"] = right["extra_right"]
        self.assertEqual(dict(expected), result)


class Test_combine(tests.IrisTest):
    def setUp(self):
        kwargs = dict(
            standard_name="standard_name",
            long_name="long_name",
            var_name="var_name",
            units="units",
            attributes="attributes",
        )
        self.metadata = BaseMetadata(**kwargs)
        self.mock_kwargs = OrderedDict(
            standard_name=sentinel.standard_name,
            long_name=sentinel.long_name,
            var_name=sentinel.var_name,
            units=sentinel.units,
            attributes=sentinel.attributes,
        )

    def test_lenient_service(self):
        qualname_combine = qualname(BaseMetadata.combine)
        self.assertIn(qualname_combine, LENIENT)
        self.assertTrue(LENIENT[qualname_combine])

    def test_cannot_combine_non_class(self):
        emsg = "Cannot combine"
        with self.assertRaisesRegex(ValueError, emsg):
            self.metadata.combine(None)

    def test_cannot_combine(self):
        other = CubeMetadata(*(None,) * len(CubeMetadata._fields))
        emsg = "Cannot combine"
        with self.assertRaisesRegex(ValueError, emsg):
            self.metadata.combine(other)

    def test_lenient_default(self):
        return_value = self.mock_kwargs.values()
        with mock.patch.object(
            BaseMetadata, "_combine", return_value=return_value
        ) as mocker:
            result = self.metadata.combine(self.metadata)

        self.assertEqual(result._asdict(), self.mock_kwargs)
        self.assertEqual(mocker.call_count, 1)
        (args,), kwargs = mocker.call_args
        self.assertEqual(id(args), id(self.metadata))
        self.assertEqual(kwargs, dict())

    def test_lenient_true(self):
        return_value = self.mock_kwargs.values()
        with mock.patch.object(
            BaseMetadata, "_combine", return_value=return_value
        ) as mcombine:
            with mock.patch.object(LENIENT, "context") as mcontext:
                result = self.metadata.combine(self.metadata, lenient=True)

        self.assertEqual(mcontext.call_count, 1)
        (args,), kwargs = mcontext.call_args
        self.assertEqual(args, qualname(BaseMetadata.combine))
        self.assertEqual(kwargs, dict())

        self.assertEqual(result._asdict(), self.mock_kwargs)
        self.assertEqual(mcombine.call_count, 1)
        (args,), kwargs = mcombine.call_args
        self.assertEqual(id(args), id(self.metadata))
        self.assertEqual(kwargs, dict())

    def test_lenient_false(self):
        return_value = self.mock_kwargs.values()
        with mock.patch.object(
            BaseMetadata, "_combine", return_value=return_value
        ) as mcombine:
            with mock.patch.object(LENIENT, "context") as mcontext:
                result = self.metadata.combine(self.metadata, lenient=False)

        self.assertEqual(mcontext.call_count, 1)
        args, kwargs = mcontext.call_args
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {qualname(BaseMetadata.combine): False})

        self.assertEqual(result._asdict(), self.mock_kwargs)
        self.assertEqual(mcombine.call_count, 1)
        (args,), kwargs = mcombine.call_args
        self.assertEqual(id(args), id(self.metadata))
        self.assertEqual(kwargs, dict())


class Test__is_attributes(tests.IrisTest):
    def setUp(self):
        self.metadata = BaseMetadata(*(None,) * len(BaseMetadata._fields))
        self.field = "attributes"

    def test_field(self):
        self.assertTrue(self.metadata._is_attributes(self.field, {}, {}))

    def test_field_not_attributes(self):
        self.assertFalse(self.metadata._is_attributes(None, {}, {}))

    def test_left_not_mapping(self):
        self.assertFalse(self.metadata._is_attributes(self.field, None, {}))

    def test_right_not_mapping(self):
        self.assertFalse(self.metadata._is_attributes(self.field, {}, None))


class Test___lt__(tests.IrisTest):
    def setUp(self):
        self.one = BaseMetadata(1, 1, 1, 1, 1)
        self.two = BaseMetadata(1, 1, 1, 1, 2)
        self.none = BaseMetadata(1, 1, 1, 1, None)

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


class Test_name(tests.IrisTest):
    def setUp(self):
        self.default = BaseMetadata.DEFAULT_NAME

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
        self.assertEqual(result, token)
        result = metadata.name(token=True)
        self.assertEqual(result, token)

    def test_standard_name__invalid_token(self):
        token = "nope nope"
        metadata = self._make(standard_name=token)
        result = metadata.name()
        self.assertEqual(result, token)
        result = metadata.name(token=True)
        self.assertEqual(result, self.default)

    def test_long_name(self):
        token = "long_name"
        metadata = self._make(long_name=token)
        result = metadata.name()
        self.assertEqual(result, token)
        result = metadata.name(token=True)
        self.assertEqual(result, token)

    def test_long_name__invalid_token(self):
        token = "nope nope"
        metadata = self._make(long_name=token)
        result = metadata.name()
        self.assertEqual(result, token)
        result = metadata.name(token=True)
        self.assertEqual(result, self.default)

    def test_var_name(self):
        token = "var_name"
        metadata = self._make(var_name=token)
        result = metadata.name()
        self.assertEqual(result, token)
        result = metadata.name(token=True)
        self.assertEqual(result, token)

    def test_var_name__invalid_token(self):
        token = "nope nope"
        metadata = self._make(var_name=token)
        result = metadata.name()
        self.assertEqual(result, token)
        result = metadata.name(token=True)
        self.assertEqual(result, self.default)

    def test_default(self):
        metadata = self._make()
        result = metadata.name()
        self.assertEqual(result, self.default)
        result = metadata.name(token=True)
        self.assertEqual(result, self.default)

    def test_default__invalid_token(self):
        token = "nope nope"
        metadata = self._make()
        result = metadata.name(default=token)
        self.assertEqual(result, token)
        emsg = "Cannot retrieve a valid name token"
        with self.assertRaisesRegex(ValueError, emsg):
            metadata.name(default=token, token=True)


class Test_token(tests.IrisTest):
    def test_passthru_None(self):
        result = BaseMetadata.token(None)
        self.assertIsNone(result)

    def test_fail_leading_underscore(self):
        result = BaseMetadata.token("_nope")
        self.assertIsNone(result)

    def test_fail_leading_dot(self):
        result = BaseMetadata.token(".nope")
        self.assertIsNone(result)

    def test_fail_leading_plus(self):
        result = BaseMetadata.token("+nope")
        self.assertIsNone(result)

    def test_fail_leading_at(self):
        result = BaseMetadata.token("@nope")
        self.assertIsNone(result)

    def test_fail_space(self):
        result = BaseMetadata.token("nope nope")
        self.assertIsNone(result)

    def test_fail_colon(self):
        result = BaseMetadata.token("nope:")
        self.assertIsNone(result)

    def test_pass_simple(self):
        token = "simple"
        result = BaseMetadata.token(token)
        self.assertEqual(result, token)

    def test_pass_leading_digit(self):
        token = "123simple"
        result = BaseMetadata.token(token)
        self.assertEqual(result, token)

    def test_pass_mixture(self):
        token = "S.imple@one+two_3"
        result = BaseMetadata.token(token)
        self.assertEqual(result, token)


if __name__ == "__main__":
    tests.main()
