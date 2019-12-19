# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :class:`iris.common.metadata.AncillaryVariableMetadata`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import unittest.mock as mock

from iris.common.metadata import BaseMetadata, AncillaryVariableMetadata


class Test(tests.IrisTest):
    def setUp(self):
        self.standard_name = mock.sentinel.standard_name
        self.long_name = mock.sentinel.long_name
        self.var_name = mock.sentinel.var_name
        self.units = mock.sentinel.units
        self.attributes = mock.sentinel.attributes

    def test_repr(self):
        metadata = AncillaryVariableMetadata(
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
            units=self.units,
            attributes=self.attributes,
        )
        fmt = (
            "AncillaryVariableMetadata(standard_name={!r}, long_name={!r}, "
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
        self.assertEqual(AncillaryVariableMetadata._fields, expected)

    def test_bases(self):
        self.assertTrue(issubclass(AncillaryVariableMetadata, BaseMetadata))


if __name__ == "__main__":
    tests.main()
