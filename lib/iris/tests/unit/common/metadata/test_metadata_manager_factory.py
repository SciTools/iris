# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :func:`iris.common.metadata.metadata_manager_factory`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import pickle
import unittest.mock as mock

from cf_units import Unit

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


class Test_factory(tests.IrisTest):
    def test__kwargs_invalid(self):
        emsg = "Invalid 'BaseMetadata' field parameters, got 'wibble'."
        with self.assertRaisesRegex(ValueError, emsg):
            metadata_manager_factory(BaseMetadata, wibble="nope")


class Test_instance(tests.IrisTest):
    def setUp(self):
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
                self.assertTrue(hasattr(metadata, name))
            if base is CubeMetadata:
                self.assertTrue(hasattr(metadata, "_names"))
            self.assertIs(metadata.cls, base)

    def test__kwargs_default(self):
        for base in self.bases:
            kwargs = dict(zip(base._fields, [None] * len(base._fields)))
            metadata = metadata_manager_factory(base)
            self.assertEqual(metadata.values._asdict(), kwargs)

    def test__kwargs(self):
        for base in self.bases:
            kwargs = dict(zip(base._fields, range(len(base._fields))))
            metadata = metadata_manager_factory(base, **kwargs)
            self.assertEqual(metadata.values._asdict(), kwargs)


class Test_instance___eq__(tests.IrisTest):
    def setUp(self):
        self.metadata = metadata_manager_factory(BaseMetadata)

    def test__not_implemented(self):
        self.assertNotEqual(self.metadata, 1)

    def test__not_is_cls(self):
        base = BaseMetadata
        other = metadata_manager_factory(base)
        self.assertIs(other.cls, base)
        other.cls = CoordMetadata
        self.assertNotEqual(self.metadata, other)

    def test__not_values(self):
        standard_name = mock.sentinel.standard_name
        other = metadata_manager_factory(BaseMetadata, standard_name=standard_name)
        self.assertEqual(other.standard_name, standard_name)
        self.assertIsNone(other.long_name)
        self.assertIsNone(other.var_name)
        self.assertIsNone(other.units)
        self.assertIsNone(other.attributes)
        self.assertNotEqual(self.metadata, other)

    def test__same_default(self):
        other = metadata_manager_factory(BaseMetadata)
        self.assertEqual(self.metadata, other)

    def test__same(self):
        kwargs = dict(standard_name=1, long_name=2, var_name=3, units=4, attributes=5)
        metadata = metadata_manager_factory(BaseMetadata, **kwargs)
        other = metadata_manager_factory(BaseMetadata, **kwargs)
        self.assertEqual(metadata.values._asdict(), kwargs)
        self.assertEqual(metadata, other)


class Test_instance____repr__(tests.IrisTest):
    def setUp(self):
        self.metadata = metadata_manager_factory(BaseMetadata)

    def test(self):
        standard_name = mock.sentinel.standard_name
        long_name = mock.sentinel.long_name
        var_name = mock.sentinel.var_name
        units = mock.sentinel.units
        attributes = mock.sentinel.attributes
        values = (standard_name, long_name, var_name, units, attributes)

        for field, value in zip(self.metadata.fields, values):
            setattr(self.metadata, field, value)

        result = repr(self.metadata)
        expected = (
            "MetadataManager(standard_name={!r}, long_name={!r}, var_name={!r}, "
            "units={!r}, attributes={!r})"
        )
        self.assertEqual(result, expected.format(*values))


class Test_instance__pickle(tests.IrisTest):
    def setUp(self):
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

    def test_pickle(self):
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.temp_filename(suffix=".pkl") as fname:
                with open(fname, "wb") as fout:
                    pickle.dump(self.metadata, fout, protocol=protocol)
                with open(fname, "rb") as fin:
                    metadata = pickle.load(fin)
                    self.assertEqual(metadata, self.metadata)


class Test_instance__fields(tests.IrisTest):
    def setUp(self):
        self.bases = BASES

    def test(self):
        for base in self.bases:
            fields = base._fields
            metadata = metadata_manager_factory(base)
            self.assertEqual(metadata.fields, fields)
            for field in fields:
                hasattr(metadata, field)


class Test_instance__values(tests.IrisTest):
    def setUp(self):
        self.bases = BASES

    def test(self):
        for base in self.bases:
            metadata = metadata_manager_factory(base)
            result = metadata.values
            self.assertIsInstance(result, base)
            self.assertEqual(result._fields, base._fields)


if __name__ == "__main__":
    tests.main()
