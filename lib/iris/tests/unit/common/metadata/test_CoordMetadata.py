# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :class:`iris.common.metadata.CoordMetadata`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import unittest.mock as mock

from iris.common.metadata import BaseMetadata, CoordMetadata


class Test(tests.IrisTest):
    def setUp(self):
        self.standard_name = mock.sentinel.standard_name
        self.long_name = mock.sentinel.long_name
        self.var_name = mock.sentinel.var_name
        self.units = mock.sentinel.units
        self.attributes = mock.sentinel.attributes
        self.coord_system = mock.sentinel.coord_system
        self.climatological = mock.sentinel.climatological

    def test_repr(self):
        metadata = CoordMetadata(
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
        self.assertEqual(expected, repr(metadata))

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
        self.assertEqual(CoordMetadata._fields, expected)

    def test_bases(self):
        self.assertTrue(issubclass(CoordMetadata, BaseMetadata))


if __name__ == "__main__":
    tests.main()
