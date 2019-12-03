# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :class:`iris.common.metadata.CubeMetadata`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import unittest.mock as mock

from iris.common.metadata import BaseMetadata, CubeMetadata


def _make_metadata(
    standard_name=None,
    long_name=None,
    var_name=None,
    attributes=None,
    force_mapping=True,
):
    if force_mapping:
        if attributes is None:
            attributes = {}
        else:
            attributes = dict(STASH=attributes)

    return CubeMetadata(
        standard_name=standard_name,
        long_name=long_name,
        var_name=var_name,
        units=None,
        attributes=attributes,
        cell_methods=None,
    )


class Test(tests.IrisTest):
    def setUp(self):
        self.standard_name = mock.sentinel.standard_name
        self.long_name = mock.sentinel.long_name
        self.var_name = mock.sentinel.var_name
        self.units = mock.sentinel.units
        self.attributes = mock.sentinel.attributes
        self.cell_methods = mock.sentinel.cell_methods

    def test_repr(self):
        metadata = CubeMetadata(
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
            units=self.units,
            attributes=self.attributes,
            cell_methods=self.cell_methods,
        )
        fmt = (
            "CubeMetadata(standard_name={!r}, long_name={!r}, var_name={!r}, "
            "units={!r}, attributes={!r}, cell_methods={!r})"
        )
        expected = fmt.format(
            self.standard_name,
            self.long_name,
            self.var_name,
            self.units,
            self.attributes,
            self.cell_methods,
        )
        self.assertEqual(expected, repr(metadata))

    def test__fields(self):
        expected = (
            "standard_name",
            "long_name",
            "var_name",
            "units",
            "attributes",
            "cell_methods",
        )
        self.assertEqual(CubeMetadata._fields, expected)

    def test_bases(self):
        self.assertTrue(issubclass(CubeMetadata, BaseMetadata))


class Test_name(tests.IrisTest):
    def setUp(self):
        self.default = CubeMetadata.DEFAULT_NAME

    def test_standard_name(self):
        token = "standard_name"
        metadata = _make_metadata(standard_name=token)
        result = metadata.name()
        self.assertEqual(result, token)
        result = metadata.name(token=True)
        self.assertEqual(result, token)

        token = "nope nope"
        metadata = _make_metadata(standard_name=token)
        result = metadata.name()
        self.assertEqual(result, token)
        result = metadata.name(token=True)
        self.assertEqual(result, self.default)

    def test_long_name(self):
        token = "long_name"
        metadata = _make_metadata(long_name=token)
        result = metadata.name()
        self.assertEqual(result, token)
        result = metadata.name(token=True)
        self.assertEqual(result, token)

        token = "nope nope"
        metadata = _make_metadata(long_name=token)
        result = metadata.name()
        self.assertEqual(result, token)
        result = metadata.name(token=True)
        self.assertEqual(result, self.default)

    def test_var_name(self):
        token = "var_name"
        metadata = _make_metadata(var_name=token)
        result = metadata.name()
        self.assertEqual(result, token)
        result = metadata.name(token=True)
        self.assertEqual(result, token)

        token = "nope nope"
        metadata = _make_metadata(var_name=token)
        result = metadata.name()
        self.assertEqual(result, token)
        result = metadata.name(token=True)
        self.assertEqual(result, self.default)

    def test_attributes(self):
        token = "stash"
        metadata = _make_metadata(attributes=token)
        result = metadata.name()
        self.assertEqual(result, token)
        result = metadata.name(token=True)
        self.assertEqual(result, token)

        token = "nope nope"
        metadata = _make_metadata(attributes=token)
        result = metadata.name()
        self.assertEqual(result, token)
        result = metadata.name(token=True)
        self.assertEqual(result, self.default)

    def test_attributes__non_mapping(self):
        metadata = _make_metadata(force_mapping=False)
        self.assertIsNone(metadata.attributes)
        emsg = "Invalid 'CubeMetadata.attributes' member, must be a mapping."
        with self.assertRaisesRegex(AttributeError, emsg):
            _ = metadata.name()

    def test_default(self):
        metadata = _make_metadata()
        result = metadata.name()
        self.assertEqual(result, self.default)
        result = metadata.name(token=True)
        self.assertEqual(result, self.default)

        token = "nope nope"
        result = metadata.name(default=token)
        self.assertEqual(result, token)
        emsg = "Cannot retrieve a valid name token"
        with self.assertRaisesRegex(ValueError, emsg):
            _ = metadata.name(default=token, token=True)


class Test_names(tests.IrisTest):
    def test_standard_name(self):
        token = "standard_name"
        metadata = _make_metadata(standard_name=token)
        expected = (token, None, None, None)
        result = metadata._names
        self.assertEqual(expected, result)

    def test_long_name(self):
        token = "long_name"
        metadata = _make_metadata(long_name=token)
        expected = (None, token, None, None)
        result = metadata._names
        self.assertEqual(expected, result)

    def test_var_name(self):
        token = "var_name"
        metadata = _make_metadata(var_name=token)
        expected = (None, None, token, None)
        result = metadata._names
        self.assertEqual(expected, result)

    def test_attributes(self):
        token = "stash"
        metadata = _make_metadata(attributes=token)
        expected = (None, None, None, token)
        result = metadata._names
        self.assertEqual(expected, result)

    def test_attributes__non_mapping(self):
        metadata = _make_metadata(force_mapping=False)
        self.assertIsNone(metadata.attributes)
        emsg = "Invalid 'CubeMetadata.attributes' member, must be a mapping."
        with self.assertRaisesRegex(AttributeError, emsg):
            _ = metadata._names

    def test_None(self):
        metadata = _make_metadata()
        expected = (None, None, None, None)
        result = metadata._names
        self.assertEqual(expected, result)


if __name__ == "__main__":
    tests.main()
