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
import iris.tests as tests  # isort:skip

from collections import OrderedDict, namedtuple
from unittest import mock

from cf_units import Unit

from iris.common.metadata import (
    AncillaryVariableMetadata,
    BaseMetadata,
    CellMeasureMetadata,
    CoordMetadata,
    CubeMetadata,
)
from iris.common.mixin import CFVariableMixin, LimitedAttributeDict
from iris.experimental.ugrid.metadata import ConnectivityMetadata


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
        self.item._metadata_manager = metadata

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
        self.item._metadata_manager = metadata

    def test_standard_name__valid(self):
        standard_name = "air_temperature"
        self.item.standard_name = standard_name
        self.assertEqual(
            self.item._metadata_manager.standard_name, standard_name
        )

    def test_standard_name__none(self):
        self.item.standard_name = None
        self.assertIsNone(self.item._metadata_manager.standard_name)

    def test_standard_name__invalid(self):
        standard_name = "nope nope"
        emsg = f"{standard_name!r} is not a valid standard_name"
        with self.assertRaisesRegex(ValueError, emsg):
            self.item.standard_name = standard_name

    def test_long_name(self):
        long_name = "long_name"
        self.item.long_name = long_name
        self.assertEqual(self.item._metadata_manager.long_name, long_name)

    def test_long_name__none(self):
        self.item.long_name = None
        self.assertIsNone(self.item._metadata_manager.long_name)

    def test_var_name(self):
        var_name = "var_name"
        self.item.var_name = var_name
        self.assertEqual(self.item._metadata_manager.var_name, var_name)

    def test_var_name__none(self):
        self.item.var_name = None
        self.assertIsNone(self.item._metadata_manager.var_name)

    def test_var_name__invalid_token(self):
        var_name = "nope nope"
        self.item._metadata_manager.token = lambda name: None
        emsg = f"{var_name!r} is not a valid NetCDF variable name."
        with self.assertRaisesRegex(ValueError, emsg):
            self.item.var_name = var_name

    def test_attributes(self):
        attributes = dict(hello="world")
        self.item.attributes = attributes
        self.assertEqual(self.item._metadata_manager.attributes, attributes)
        self.assertIsNot(self.item._metadata_manager.attributes, attributes)
        self.assertIsInstance(
            self.item._metadata_manager.attributes, LimitedAttributeDict
        )

    def test_attributes__none(self):
        self.item.attributes = None
        self.assertEqual(self.item._metadata_manager.attributes, {})


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
        self.item._metadata_manager = metadata
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
        self.assertEqual(self.item._metadata_manager.values, metadata)
        self.assertIsNot(
            self.item._metadata_manager.attributes, self.attributes
        )

    def test_dict__partial(self):
        metadata = dict(**self.args)
        del metadata["standard_name"]
        self.item.metadata = metadata
        metadata["standard_name"] = mock.sentinel.standard_name
        self.assertEqual(self.item._metadata_manager.values, metadata)
        self.assertIsNot(
            self.item._metadata_manager.attributes, self.attributes
        )

    def test_ordereddict(self):
        metadata = self.args
        self.item.metadata = metadata
        self.assertEqual(self.item._metadata_manager.values, metadata)
        self.assertIsNot(
            self.item._metadata_manager.attributes, self.attributes
        )

    def test_ordereddict__partial(self):
        metadata = self.args
        del metadata["long_name"]
        del metadata["units"]
        self.item.metadata = metadata
        metadata["long_name"] = mock.sentinel.long_name
        metadata["units"] = mock.sentinel.units
        self.assertEqual(self.item._metadata_manager.values, metadata)

    def test_tuple(self):
        metadata = tuple(self.args.values())
        self.item.metadata = metadata
        result = tuple(
            [
                getattr(self.item._metadata_manager, field)
                for field in self.item._metadata_manager.fields
            ]
        )
        self.assertEqual(result, metadata)
        self.assertIsNot(
            self.item._metadata_manager.attributes, self.attributes
        )

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
        self.assertEqual(
            self.item._metadata_manager.values, metadata._asdict()
        )
        self.assertIsNot(
            self.item._metadata_manager.attributes, metadata.attributes
        )

    def test_namedtuple__partial(self):
        Metadata = namedtuple(
            "Metadata", ("standard_name", "long_name", "var_name", "units")
        )
        del self.args["attributes"]
        metadata = Metadata(**self.args)
        self.item.metadata = metadata
        expected = metadata._asdict()
        expected.update(dict(attributes=mock.sentinel.attributes))
        self.assertEqual(self.item._metadata_manager.values, expected)

    def test_class_ancillaryvariablemetadata(self):
        metadata = AncillaryVariableMetadata(**self.args)
        self.item.metadata = metadata
        self.assertEqual(
            self.item._metadata_manager.values, metadata._asdict()
        )
        self.assertIsNot(
            self.item._metadata_manager.attributes, metadata.attributes
        )

    def test_class_basemetadata(self):
        metadata = BaseMetadata(**self.args)
        self.item.metadata = metadata
        self.assertEqual(
            self.item._metadata_manager.values, metadata._asdict()
        )
        self.assertIsNot(
            self.item._metadata_manager.attributes, metadata.attributes
        )

    def test_class_cellmeasuremetadata(self):
        self.args["measure"] = None
        metadata = CellMeasureMetadata(**self.args)
        self.item.metadata = metadata
        expected = metadata._asdict()
        del expected["measure"]
        self.assertEqual(self.item._metadata_manager.values, expected)
        self.assertIsNot(
            self.item._metadata_manager.attributes, metadata.attributes
        )

    def test_class_connectivitymetadata(self):
        self.args.update(
            dict(cf_role=None, start_index=None, location_axis=None)
        )
        metadata = ConnectivityMetadata(**self.args)
        self.item.metadata = metadata
        expected = metadata._asdict()
        del expected["cf_role"]
        del expected["start_index"]
        del expected["location_axis"]
        self.assertEqual(self.item._metadata_manager.values, expected)
        self.assertIsNot(
            self.item._metadata_manager.attributes, metadata.attributes
        )

    def test_class_coordmetadata(self):
        self.args.update(dict(coord_system=None, climatological=False))
        metadata = CoordMetadata(**self.args)
        self.item.metadata = metadata
        expected = metadata._asdict()
        del expected["coord_system"]
        del expected["climatological"]
        self.assertEqual(self.item._metadata_manager.values, expected)
        self.assertIsNot(
            self.item._metadata_manager.attributes, metadata.attributes
        )

    def test_class_cubemetadata(self):
        self.args["cell_methods"] = None
        metadata = CubeMetadata(**self.args)
        self.item.metadata = metadata
        expected = metadata._asdict()
        del expected["cell_methods"]
        self.assertEqual(self.item._metadata_manager.values, expected)
        self.assertIsNot(
            self.item._metadata_manager.attributes, metadata.attributes
        )


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
        self.item._metadata_manager = metadata

    def test__valid_standard_name(self):
        name = "air_temperature"
        self.item.rename(name)
        self.assertEqual(self.item._metadata_manager.standard_name, name)
        self.assertIsNone(self.item._metadata_manager.long_name)
        self.assertIsNone(self.item._metadata_manager.var_name)

    def test__invalid_standard_name(self):
        name = "nope nope"
        self.item.rename(name)
        self.assertIsNone(self.item._metadata_manager.standard_name)
        self.assertEqual(self.item._metadata_manager.long_name, name)
        self.assertIsNone(self.item._metadata_manager.var_name)


class Test_name(tests.IrisTest):
    def setUp(self):
        class Metadata:
            def __init__(self, name):
                self.name = mock.MagicMock(return_value=name)

        self.name = mock.sentinel.name
        metadata = Metadata(self.name)

        self.item = CFVariableMixin()
        self.item._metadata_manager = metadata

    def test(self):
        default = mock.sentinel.default
        token = mock.sentinel.token
        result = self.item.name(default=default, token=token)
        self.assertEqual(result, self.name)
        self.item._metadata_manager.name.assert_called_with(
            default=default, token=token
        )


if __name__ == "__main__":
    tests.main()
