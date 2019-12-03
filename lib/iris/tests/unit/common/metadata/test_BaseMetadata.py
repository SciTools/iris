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

import unittest.mock as mock

from iris.common.metadata import BaseMetadata


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

        token = "nope nope"
        result = metadata.name(default=token)
        self.assertEqual(result, token)
        emsg = "Cannot retrieve a valid name token"
        with self.assertRaisesRegex(ValueError, emsg):
            metadata.name(default=token, token=True)


if __name__ == "__main__":
    tests.main()
