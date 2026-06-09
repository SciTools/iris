# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :func:`iris.common.metadata.metadata_manager_factory`."""

import pickle

from cf_units import Unit
import pytest

from iris.common.metadata import (
    AncillaryVariableMetadata,
    BaseMetadata,
    CellMeasureMetadata,
    ConnectivityMetadata,
    CoordMetadata,
    CubeMetadata,
    metadata_manager_factory,
)

BASES = [
    AncillaryVariableMetadata,
    BaseMetadata,
    CellMeasureMetadata,
    ConnectivityMetadata,
    CoordMetadata,
    CubeMetadata,
]


class Test_factory:
    def test__kwargs_invalid(self):
        emsg = "Invalid 'BaseMetadata' field parameters, got 'wibble'."
        with pytest.raises(ValueError, match=emsg):
            _ = metadata_manager_factory(BaseMetadata, wibble="nope")


class Test_instance:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.bases = BASES

    def test__namespace(self):
        namespace = [
            "DEFAULT_NAME",
            "__init__",
            "__eq__",
            "__getstate__",
            "__ne__",
            "__reduce__",
            "__repr__",
            "__setstate__",
            "fields",
            "name",
            "token",
            "values",
        ]
        for base in self.bases:
            metadata = metadata_manager_factory(base)
            for name in namespace:
                assert hasattr(metadata, name)
            if base is CubeMetadata:
                assert hasattr(metadata, "_names")
            assert metadata.cls is base

    def test__kwargs_default(self):
        for base in self.bases:
            kwargs = dict(zip(base._fields, [None] * len(base._fields)))
            metadata = metadata_manager_factory(base)
            assert metadata.values._asdict() == kwargs

    def test__kwargs(self):
        for base in self.bases:
            kwargs = dict(zip(base._fields, range(len(base._fields))))
            metadata = metadata_manager_factory(base, **kwargs)
            assert metadata.values._asdict() == kwargs


class Test_instance___eq__:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.metadata = metadata_manager_factory(BaseMetadata)

    def test__not_implemented(self):
        assert self.metadata != 1

    def test__not_is_cls(self):
        base = BaseMetadata
        other = metadata_manager_factory(base)
        assert other.cls is base
        other.cls = CoordMetadata
        assert other != self.metadata

    def test__not_values(self, mocker):
        standard_name = mocker.sentinel.standard_name
        other = metadata_manager_factory(BaseMetadata, standard_name=standard_name)
        assert other.standard_name == standard_name
        assert other.long_name is None
        assert other.var_name is None
        assert other.units is None
        assert other.attributes is None
        assert other != self.metadata

    def test__same_default(self):
        other = metadata_manager_factory(BaseMetadata)
        assert other == self.metadata

    def test__same(self):
        kwargs = dict(standard_name=1, long_name=2, var_name=3, units=4, attributes=5)
        metadata = metadata_manager_factory(BaseMetadata, **kwargs)
        other = metadata_manager_factory(BaseMetadata, **kwargs)
        assert metadata.values._asdict() == kwargs
        assert metadata == other


class Test_instance____repr__:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.metadata = metadata_manager_factory(BaseMetadata)

    def test(self, mocker):
        standard_name = mocker.sentinel.standard_name
        long_name = mocker.sentinel.long_name
        var_name = mocker.sentinel.var_name
        units = mocker.sentinel.units
        attributes = mocker.sentinel.attributes
        values = (standard_name, long_name, var_name, units, attributes)

        for field, value in zip(self.metadata.fields, values):
            setattr(self.metadata, field, value)

        result = repr(self.metadata)
        expected = (
            "MetadataManager(standard_name={!r}, long_name={!r}, var_name={!r}, "
            "units={!r}, attributes={!r})"
        )
        assert result == expected.format(*values)


class Test_instance__pickle:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.standard_name = "standard_name"
        self.long_name = "long_name"
        self.var_name = "var_name"
        self.units = Unit("1")
        self.attributes = dict(hello="world")
        values = (
            self.standard_name,
            self.long_name,
            self.var_name,
            self.units,
            self.attributes,
        )
        kwargs = dict(zip(BaseMetadata._fields, values))
        self.metadata = metadata_manager_factory(BaseMetadata, **kwargs)

    def test_pickle(self, tmp_path):
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            fname = tmp_path / f"pickle_{protocol}.pkl"
            with open(fname, "wb") as fout:
                pickle.dump(self.metadata, fout, protocol=protocol)
            with open(fname, "rb") as fin:
                metadata = pickle.load(fin)
                assert metadata == self.metadata


class Test_instance__fields:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.bases = BASES

    def test(self):
        for base in self.bases:
            fields = base._fields
            metadata = metadata_manager_factory(base)
            assert metadata.fields == fields
            for field in fields:
                assert hasattr(metadata, field)


class Test_instance__values:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.bases = BASES

    def test(self):
        for base in self.bases:
            metadata = metadata_manager_factory(base)
            result = metadata.values
            assert isinstance(result, base)
            assert result._fields == base._fields
