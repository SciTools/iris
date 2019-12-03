# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :class:`iris.common.mixin.CFVariableMixin`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from collections import OrderedDict, namedtuple
from unittest import mock

from cf_units import Unit

from iris.common.metadata import BaseMetadata
from iris.common.mixin import CFVariableMixin, LimitedAttributeDict


class Test__getter(tests.IrisTest):
    def setUp(self):
        self.standard_name = mock.sentinel.standard_name
        self.long_name = mock.sentinel.long_name
        self.var_name = mock.sentinel.var_name
        self.units = mock.sentinel.units
        self.attributes = mock.sentinel.attributes
        self.metadata = mock.sentinel.metadata

        metadata = mock.MagicMock(
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
            units=self.units,
            attributes=self.attributes,
            values=self.metadata,
        )

        self.item = CFVariableMixin()
        self.item._metadata = metadata

    def test_standard_name(self):
        self.assertEqual(self.item.standard_name, self.standard_name)

    def test_long_name(self):
        self.assertEqual(self.item.long_name, self.long_name)

    def test_var_name(self):
        self.assertEqual(self.item.var_name, self.var_name)

    def test_units(self):
        self.assertEqual(self.item.units, self.units)

    def test_attributes(self):
        self.assertEqual(self.item.attributes, self.attributes)

    def test_metadata(self):
        self.assertEqual(self.item.metadata, self.metadata)


class Test__setter(tests.IrisTest):
    def setUp(self):
        metadata = mock.MagicMock(
            standard_name=mock.sentinel.standard_name,
            long_name=mock.sentinel.long_name,
            var_name=mock.sentinel.var_name,
            units=mock.sentinel.units,
            attributes=mock.sentinel.attributes,
            token=lambda name: name,
        )

        self.item = CFVariableMixin()
        self.item._metadata = metadata

    def test_standard_name(self):
        standard_name = "air_temperature"
        self.item.standard_name = standard_name
        self.assertEqual(self.item._metadata.standard_name, standard_name)

        self.item.standard_name = None
        self.assertIsNone(self.item._metadata.standard_name)

        standard_name = "nope nope"
        emsg = f"{standard_name!r} is not a valid standard_name"
        with self.assertRaisesRegex(ValueError, emsg):
            self.item.standard_name = standard_name

    def test_long_name(self):
        long_name = "long_name"
        self.item.long_name = long_name
        self.assertEqual(self.item._metadata.long_name, long_name)

        self.item.long_name = None
        self.assertIsNone(self.item._metadata.long_name)

    def test_var_name(self):
        var_name = "var_name"
        self.item.var_name = var_name
        self.assertEqual(self.item._metadata.var_name, var_name)

        self.item.var_name = None
        self.assertIsNone(self.item._metadata.var_name)

        var_name = "nope nope"
        self.item._metadata.token = lambda name: None
        emsg = f"{var_name!r} is not a valid NetCDF variable name."
        with self.assertRaisesRegex(ValueError, emsg):
            self.item.var_name = var_name

    def test_attributes(self):
        attributes = dict(hello="world")
        self.item.attributes = attributes
        self.assertEqual(self.item._metadata.attributes, attributes)
        self.assertIsNot(self.item._metadata.attributes, attributes)
        self.assertIsInstance(
            self.item._metadata.attributes, LimitedAttributeDict
        )

        self.item.attributes = None
        self.assertEqual(self.item._metadata.attributes, {})


class Test__metadata_setter(tests.IrisTest):
    def setUp(self):
        class Metadata:
            def __init__(self):
                self.cls = BaseMetadata
                self.fields = BaseMetadata._fields
                self.standard_name = mock.sentinel.standard_name
                self.long_name = mock.sentinel.long_name
                self.var_name = mock.sentinel.var_name
                self.units = mock.sentinel.units
                self.attributes = mock.sentinel.attributes
                self.token = lambda name: name

            @property
            def values(self):
                return dict(
                    standard_name=self.standard_name,
                    long_name=self.long_name,
                    var_name=self.var_name,
                    units=self.units,
                    attributes=self.attributes,
                )

        metadata = Metadata()
        self.item = CFVariableMixin()
        self.item._metadata = metadata
        self.attributes = dict(one=1, two=2, three=3)
        self.args = OrderedDict(
            standard_name="air_temperature",
            long_name="long_name",
            var_name="var_name",
            units=Unit("1"),
            attributes=self.attributes,
        )

    def test_dict(self):
        metadata = dict(**self.args)
        self.item.metadata = metadata
        self.assertEqual(self.item._metadata.values, metadata)
        self.assertIsNot(self.item._metadata.attributes, self.attributes)

    def test_dict__missing(self):
        metadata = dict(**self.args)
        del metadata["standard_name"]
        emsg = "Invalid .* metadata, require 'standard_name' to be specified."
        with self.assertRaisesRegex(TypeError, emsg):
            self.item.metadata = metadata

    def test_ordereddict(self):
        metadata = self.args
        self.item.metadata = metadata
        self.assertEqual(self.item._metadata.values, metadata)
        self.assertIsNot(self.item._metadata.attributes, self.attributes)

    def test_ordereddict__missing(self):
        metadata = self.args
        del metadata["long_name"]
        del metadata["units"]
        emsg = "Invalid .* metadata, require 'long_name', 'units' to be specified."
        with self.assertRaisesRegex(TypeError, emsg):
            self.item.metadata = metadata

    def test_tuple(self):
        metadata = tuple(self.args.values())
        self.item.metadata = metadata
        result = tuple(
            [
                getattr(self.item._metadata, field)
                for field in self.item._metadata.fields
            ]
        )
        self.assertEqual(result, metadata)
        self.assertIsNot(self.item._metadata.attributes, self.attributes)

    def test_tuple__missing(self):
        metadata = list(self.args.values())
        del metadata[2]
        emsg = "Invalid .* metadata, require .* to be specified."
        with self.assertRaisesRegex(TypeError, emsg):
            self.item.metadata = tuple(metadata)

    def test_namedtuple(self):
        Metadata = namedtuple(
            "Metadata",
            ("standard_name", "long_name", "var_name", "units", "attributes"),
        )
        metadata = Metadata(**self.args)
        self.item.metadata = metadata
        self.assertEqual(self.item._metadata.values, metadata._asdict())
        self.assertIsNot(self.item._metadata.attributes, metadata.attributes)

    def test_namedtuple__missing(self):
        Metadata = namedtuple(
            "Metadata", ("standard_name", "long_name", "var_name", "units")
        )
        metadata = Metadata(standard_name=1, long_name=2, var_name=3, units=4)
        emsg = "Invalid .* metadata, require 'attributes' to be specified."
        with self.assertRaisesRegex(TypeError, emsg):
            self.item.metadata = metadata

    def test_class(self):
        metadata = BaseMetadata(**self.args)
        self.item.metadata = metadata
        self.assertEqual(self.item._metadata.values, metadata._asdict())
        self.assertIsNot(self.item._metadata.attributes, metadata.attributes)


class Test_rename(tests.IrisTest):
    def setUp(self):
        metadata = mock.MagicMock(
            standard_name=mock.sentinel.standard_name,
            long_name=mock.sentinel.long_name,
            var_name=mock.sentinel.var_name,
            units=mock.sentinel.units,
            attributes=mock.sentinel.attributes,
            values=mock.sentinel.metadata,
            token=lambda name: name,
        )

        self.item = CFVariableMixin()
        self.item._metadata = metadata

    def test(self):
        name = "air_temperature"
        self.item.rename(name)
        self.assertEqual(self.item._metadata.standard_name, name)
        self.assertIsNone(self.item._metadata.long_name)
        self.assertIsNone(self.item._metadata.var_name)

        name = "nope nope"
        self.item.rename(name)
        self.assertIsNone(self.item._metadata.standard_name)
        self.assertEqual(self.item._metadata.long_name, name)
        self.assertIsNone(self.item._metadata.var_name)


class Test_name(tests.IrisTest):
    def setUp(self):
        class Metadata:
            def __init__(self, name):
                self.name = mock.MagicMock(return_value=name)

        self.name = mock.sentinel.name
        metadata = Metadata(self.name)

        self.item = CFVariableMixin()
        self.item._metadata = metadata

    def test(self):
        default = mock.sentinel.default
        token = mock.sentinel.token
        result = self.item.name(default=default, token=token)
        self.assertEqual(result, self.name)
        self.item._metadata.name.assert_called_with(
            default=default, token=token
        )


if __name__ == "__main__":
    tests.main()
