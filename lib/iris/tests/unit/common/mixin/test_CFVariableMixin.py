# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.common.mixin.CFVariableMixin`."""

from collections import OrderedDict, namedtuple

from cf_units import Unit
import pytest

from iris.common.metadata import (
    AncillaryVariableMetadata,
    BaseMetadata,
    CellMeasureMetadata,
    ConnectivityMetadata,
    CoordMetadata,
    CubeMetadata,
)
from iris.common.mixin import CFVariableMixin, LimitedAttributeDict


class Test__getter:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.standard_name = mocker.sentinel.standard_name
        self.long_name = mocker.sentinel.long_name
        self.var_name = mocker.sentinel.var_name
        self.units = mocker.sentinel.units
        self.attributes = mocker.sentinel.attributes
        self.metadata = mocker.sentinel.metadata

        metadata = mocker.MagicMock(
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
        assert self.item.standard_name == self.standard_name

    def test_long_name(self):
        assert self.item.long_name == self.long_name

    def test_var_name(self):
        assert self.item.var_name == self.var_name

    def test_units(self):
        assert self.item.units == self.units

    def test_attributes(self):
        assert self.item.attributes == self.attributes

    def test_metadata(self):
        assert self.item.metadata == self.metadata


class Test__setter:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        metadata = mocker.MagicMock(
            standard_name=mocker.sentinel.standard_name,
            long_name=mocker.sentinel.long_name,
            var_name=mocker.sentinel.var_name,
            units=mocker.sentinel.units,
            attributes=mocker.sentinel.attributes,
            token=lambda name: name,
        )

        self.item = CFVariableMixin()
        self.item._metadata_manager = metadata

    def test_standard_name__valid(self):
        standard_name = "air_temperature"
        self.item.standard_name = standard_name
        assert self.item._metadata_manager.standard_name == standard_name

    def test_standard_name__none(self):
        self.item.standard_name = None
        assert self.item._metadata_manager.standard_name is None

    def test_standard_name__invalid(self):
        standard_name = "nope nope"
        emsg = f"{standard_name!r} is not a valid standard_name"
        with pytest.raises(ValueError, match=emsg):
            self.item.standard_name = standard_name

    def test_long_name(self):
        long_name = "long_name"
        self.item.long_name = long_name
        assert self.item._metadata_manager.long_name == long_name

    def test_long_name__none(self):
        self.item.long_name = None
        assert self.item._metadata_manager.long_name is None

    def test_var_name(self):
        var_name = "var_name"
        self.item.var_name = var_name
        assert self.item._metadata_manager.var_name == var_name

    def test_var_name__none(self):
        self.item.var_name = None
        assert self.item._metadata_manager.var_name is None

    def test_var_name__invalid_token(self):
        var_name = "nope nope"
        self.item._metadata_manager.token = lambda name: None
        emsg = f"{var_name!r} is not a valid NetCDF variable name."
        with pytest.raises(ValueError, match=emsg):
            self.item.var_name = var_name

    def test_attributes(self):
        attributes = dict(hello="world")
        self.item.attributes = attributes
        assert self.item._metadata_manager.attributes == attributes
        assert self.item._metadata_manager.attributes is not attributes
        assert isinstance(self.item._metadata_manager.attributes, LimitedAttributeDict)

    def test_attributes__none(self):
        self.item.attributes = None
        assert self.item._metadata_manager.attributes == {}


class Test__metadata_setter:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        class Metadata:
            def __init__(self):
                self.cls = BaseMetadata
                self.fields = BaseMetadata._fields
                self.standard_name = mocker.sentinel.standard_name
                self.long_name = mocker.sentinel.long_name
                self.var_name = mocker.sentinel.var_name
                self.units = mocker.sentinel.units
                self.attributes = mocker.sentinel.attributes
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
        assert self.item._metadata_manager.values == metadata
        assert self.item._metadata_manager.attributes is not self.attributes

    def test_dict__partial(self, mocker):
        metadata = dict(**self.args)
        del metadata["standard_name"]
        self.item.metadata = metadata
        metadata["standard_name"] = mocker.sentinel.standard_name
        assert self.item._metadata_manager.values == metadata
        assert self.item._metadata_manager.attributes is not self.attributes

    def test_ordereddict(self):
        metadata = self.args
        self.item.metadata = metadata
        assert self.item._metadata_manager.values == metadata
        assert self.item._metadata_manager.attributes is not self.attributes

    def test_ordereddict__partial(self, mocker):
        metadata = self.args
        del metadata["long_name"]
        del metadata["units"]
        self.item.metadata = metadata
        metadata["long_name"] = mocker.sentinel.long_name
        metadata["units"] = mocker.sentinel.units
        assert self.item._metadata_manager.values == metadata

    def test_tuple(self):
        metadata = tuple(self.args.values())
        self.item.metadata = metadata
        result = tuple(
            [
                getattr(self.item._metadata_manager, field)
                for field in self.item._metadata_manager.fields
            ]
        )
        assert result == metadata
        assert self.item._metadata_manager.attributes is not self.attributes

    def test_tuple__missing(self):
        metadata = list(self.args.values())
        del metadata[2]
        emsg = "Invalid .* metadata, require .* to be specified."
        with pytest.raises(TypeError, match=emsg):
            self.item.metadata = tuple(metadata)

    def test_namedtuple(self):
        Metadata = namedtuple(
            "Metadata",
            ("standard_name", "long_name", "var_name", "units", "attributes"),
        )
        metadata = Metadata(**self.args)
        self.item.metadata = metadata
        assert self.item._metadata_manager.values == metadata._asdict()
        assert self.item._metadata_manager.attributes is not metadata.attributes

    def test_namedtuple__partial(self, mocker):
        Metadata = namedtuple(
            "Metadata", ("standard_name", "long_name", "var_name", "units")
        )
        del self.args["attributes"]
        metadata = Metadata(**self.args)
        self.item.metadata = metadata
        expected = metadata._asdict()
        expected.update(dict(attributes=mocker.sentinel.attributes))
        assert self.item._metadata_manager.values == expected

    def test_class_ancillaryvariablemetadata(self):
        metadata = AncillaryVariableMetadata(**self.args)
        self.item.metadata = metadata
        assert self.item._metadata_manager.values == metadata._asdict()
        assert self.item._metadata_manager.attributes is not metadata.attributes

    def test_class_basemetadata(self):
        metadata = BaseMetadata(**self.args)
        self.item.metadata = metadata
        assert self.item._metadata_manager.values == metadata._asdict()
        assert self.item._metadata_manager.attributes is not metadata.attributes

    def test_class_cellmeasuremetadata(self):
        self.args["measure"] = None
        metadata = CellMeasureMetadata(**self.args)
        self.item.metadata = metadata
        expected = metadata._asdict()
        del expected["measure"]
        assert self.item._metadata_manager.values == expected
        assert self.item._metadata_manager.attributes is not metadata.attributes

    def test_class_connectivitymetadata(self):
        self.args.update(dict(cf_role=None, start_index=None, location_axis=None))
        metadata = ConnectivityMetadata(**self.args)
        self.item.metadata = metadata
        expected = metadata._asdict()
        del expected["cf_role"]
        del expected["start_index"]
        del expected["location_axis"]
        assert self.item._metadata_manager.values == expected
        assert self.item._metadata_manager.attributes is not metadata.attributes

    def test_class_coordmetadata(self):
        self.args.update(dict(coord_system=None, climatological=False))
        metadata = CoordMetadata(**self.args)
        self.item.metadata = metadata
        expected = metadata._asdict()
        del expected["coord_system"]
        del expected["climatological"]
        assert self.item._metadata_manager.values == expected
        assert self.item._metadata_manager.attributes is not metadata.attributes

    def test_class_cubemetadata(self):
        self.args["cell_methods"] = None
        metadata = CubeMetadata(**self.args)
        self.item.metadata = metadata
        expected = metadata._asdict()
        del expected["cell_methods"]
        assert self.item._metadata_manager.values == expected
        assert self.item._metadata_manager.attributes is not metadata.attributes


class Test_rename:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        metadata = mocker.MagicMock(
            standard_name=mocker.sentinel.standard_name,
            long_name=mocker.sentinel.long_name,
            var_name=mocker.sentinel.var_name,
            units=mocker.sentinel.units,
            attributes=mocker.sentinel.attributes,
            values=mocker.sentinel.metadata,
            token=lambda name: name,
        )

        self.item = CFVariableMixin()
        self.item._metadata_manager = metadata

    def test__valid_standard_name(self):
        name = "air_temperature"
        self.item.rename(name)
        assert self.item._metadata_manager.standard_name == name
        assert self.item._metadata_manager.long_name is None
        assert self.item._metadata_manager.var_name is None

    def test__invalid_standard_name(self):
        name = "nope nope"
        self.item.rename(name)
        assert self.item._metadata_manager.standard_name is None
        assert self.item._metadata_manager.long_name == name
        assert self.item._metadata_manager.var_name is None


class Test_name:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        class Metadata:
            def __init__(self, name):
                self.name = mocker.MagicMock(return_value=name)

        self.name = mocker.sentinel.name
        metadata = Metadata(self.name)

        self.item = CFVariableMixin()
        self.item._metadata_manager = metadata

    def test(self, mocker):
        default = mocker.sentinel.default
        token = mocker.sentinel.token
        result = self.item.name(default=default, token=token)
        assert result == self.name
        self.item._metadata_manager.name.assert_called_with(
            default=default, token=token
        )
