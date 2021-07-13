# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :func:`iris.common.metadata.metadata_manager_factory`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import pickle
import unittest.mock as mock

from cf_units import Unit

from iris.common.metadata import (
    AncillaryVariableMetadata,
    BaseMetadata,
    BaseMetadataManager,
    CellMeasureMetadata,
    CoordMetadata,
    CubeMetadata,
    make_metadata_manager_class,
)

BASES = [
    AncillaryVariableMetadata,
    BaseMetadata,
    CellMeasureMetadata,
    CoordMetadata,
    CubeMetadata,
]


class Test_factory(tests.IrisTest):
    def test__subclass_invalid(self):
        class Other:
            pass

        emsg = "Require a subclass of 'BaseMetadata'"
        with self.assertRaisesRegex(TypeError, emsg):
            _ = make_metadata_manager_class(Other)


class Test_instance(tests.IrisTest):
    def setUp(self):
        self.bases = BASES

    def test__namespace(self):
        namespace = [
            "DEFAULT_NAME",
            "__init__",
            "__eq__",
            "__ne__",
            "__repr__",
            "cls",
            "fields",
            "name",
            "token",
            "values",
        ]
        for base in self.bases:
            manager_cls = make_metadata_manager_class(base)
            for name in namespace:
                self.assertTrue(hasattr(manager_cls, name))
            if base is CubeMetadata:
                self.assertTrue(hasattr(manager_cls, "_names"))
            self.assertIs(manager_cls.cls, base)


class Test_instance___eq__(tests.IrisTest):
    def setUp(self):
        self.manager = make_metadata_manager_class(BaseMetadata)()

    def test__not_implemented(self):
        self.assertNotEqual(self.manager, 1)

    def test__not_is_cls(self):
        base = BaseMetadata
        other = make_metadata_manager_class(base)()
        self.assertIs(other.cls, base)
        other.cls = CoordMetadata
        self.assertNotEqual(self.manager, other)

    def test__not_values(self):
        standard_name = mock.sentinel.standard_name
        other = make_metadata_manager_class(BaseMetadata)()
        other.standard_name = standard_name
        self.assertEqual(other.standard_name, standard_name)
        self.assertIsNone(other.long_name)
        self.assertIsNone(other.var_name)
        self.assertIsNone(other.units)
        self.assertIsNone(other.attributes)
        self.assertNotEqual(self.manager, other)

    def test__same_default(self):
        other = make_metadata_manager_class(BaseMetadata)()
        self.assertEqual(self.manager, other)

    def test__same(self):
        kwargs = dict(
            standard_name=1, long_name=2, var_name=3, units=4, attributes=5
        )
        this = make_metadata_manager_class(BaseMetadata)()
        other = make_metadata_manager_class(BaseMetadata)()
        for key, val in kwargs.items():
            for manager in (this, other):
                setattr(manager, key, val)
        self.assertEqual(manager.values._asdict(), kwargs)
        self.assertEqual(manager, other)


class Test_instance____repr__(tests.IrisTest):
    def setUp(self):
        self.manager = make_metadata_manager_class(BaseMetadata)()

    def test(self):
        standard_name = mock.sentinel.standard_name
        long_name = mock.sentinel.long_name
        var_name = mock.sentinel.var_name
        units = mock.sentinel.units
        attributes = mock.sentinel.attributes
        values = (standard_name, long_name, var_name, units, attributes)

        for field, value in zip(self.manager.fields, values):
            setattr(self.manager, field, value)

        result = repr(self.manager)
        expected = (
            "BaseMetadataManager(standard_name={!r}, long_name={!r}, "
            "var_name={!r}, units={!r}, attributes={!r})"
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
        self.kwargs = dict(zip(BaseMetadata._fields, values))
        # NOTE: for pickle we now need an actual reference class
        # So we can't pickle something "instantly made up", it needs to be
        # available as a property of a module.
        # self.manager = make_metadata_manager_class(BaseMetadata)()
        self.manager = BaseMetadataManager()
        for key, val in self.kwargs.items():
            setattr(self.manager, key, val)

    def test_pickle(self):
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.temp_filename(suffix=".pkl") as fname:
                with open(fname, "wb") as fo:
                    pickle.dump(self.manager, fo, protocol=protocol)
                with open(fname, "rb") as fi:
                    metadata = pickle.load(fi)
                    self.assertEqual(metadata, self.manager)


class Test_instance__fields(tests.IrisTest):
    def setUp(self):
        self.bases = BASES

    def test(self):
        for base in self.bases:
            fields = base._fields
            metadata = make_metadata_manager_class(base)()
            self.assertEqual(metadata.fields, fields)
            for field in fields:
                hasattr(metadata, field)


class Test_instance__values(tests.IrisTest):
    def setUp(self):
        self.bases = BASES

    def test(self):
        for base in self.bases:
            metadata = make_metadata_manager_class(base)()
            result = metadata.values
            self.assertIsInstance(result, base)
            self.assertEqual(result._fields, base._fields)


if __name__ == "__main__":
    tests.main()
