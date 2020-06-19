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
from unittest.mock import sentinel

from iris.common.lenient import LENIENT, qualname
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


class Test___eq__(tests.IrisTest):
    def test_wraps_docstring(self):
        self.assertEqual(
            BaseMetadata.__eq__.__doc__,
            AncillaryVariableMetadata.__eq__.__doc__,
        )

    def test_lenient_service(self):
        qualname___eq__ = qualname(AncillaryVariableMetadata.__eq__)
        self.assertIn(qualname___eq__, LENIENT)
        self.assertTrue(LENIENT[qualname___eq__])
        self.assertTrue(LENIENT[AncillaryVariableMetadata.__eq__])

    def test(self):
        other = sentinel.other
        return_value = sentinel.return_value
        metadata = AncillaryVariableMetadata(
            *(None,) * len(AncillaryVariableMetadata._fields)
        )
        with mock.patch.object(
            BaseMetadata, "__eq__", return_value=return_value
        ) as mocker:
            result = metadata.__eq__(other)

        self.assertEqual(return_value, result)
        self.assertEqual(1, mocker.call_count)
        (arg,), kwargs = mocker.call_args
        self.assertEqual(other, arg)
        self.assertEqual(dict(), kwargs)


class Test_combine(tests.IrisTest):
    def test_wraps_docstring(self):
        self.assertEqual(
            BaseMetadata.combine.__doc__,
            AncillaryVariableMetadata.combine.__doc__,
        )

    def test_lenient_service(self):
        qualname_combine = qualname(AncillaryVariableMetadata.combine)
        self.assertIn(qualname_combine, LENIENT)
        self.assertTrue(LENIENT[qualname_combine])
        self.assertTrue(LENIENT[AncillaryVariableMetadata.combine])

    def test_lenient_default(self):
        other = sentinel.other
        return_value = sentinel.return_value
        metadata = AncillaryVariableMetadata(
            *(None,) * len(AncillaryVariableMetadata._fields)
        )
        with mock.patch.object(
            BaseMetadata, "combine", return_value=return_value
        ) as mocker:
            result = metadata.combine(other)

        self.assertEqual(return_value, result)
        self.assertEqual(1, mocker.call_count)
        (arg,), kwargs = mocker.call_args
        self.assertEqual(other, arg)
        self.assertEqual(dict(lenient=None), kwargs)

    def test_lenient(self):
        other = sentinel.other
        lenient = sentinel.lenient
        return_value = sentinel.return_value
        metadata = AncillaryVariableMetadata(
            *(None,) * len(AncillaryVariableMetadata._fields)
        )
        with mock.patch.object(
            BaseMetadata, "combine", return_value=return_value
        ) as mocker:
            result = metadata.combine(other, lenient=lenient)

        self.assertEqual(return_value, result)
        self.assertEqual(1, mocker.call_count)
        (arg,), kwargs = mocker.call_args
        self.assertEqual(other, arg)
        self.assertEqual(dict(lenient=lenient), kwargs)


class Test_difference(tests.IrisTest):
    def test_wraps_docstring(self):
        self.assertEqual(
            BaseMetadata.difference.__doc__,
            AncillaryVariableMetadata.difference.__doc__,
        )

    def test_lenient_service(self):
        qualname_difference = qualname(AncillaryVariableMetadata.difference)
        self.assertIn(qualname_difference, LENIENT)
        self.assertTrue(LENIENT[qualname_difference])
        self.assertTrue(LENIENT[AncillaryVariableMetadata.difference])

    def test_lenient_default(self):
        other = sentinel.other
        return_value = sentinel.return_value
        metadata = AncillaryVariableMetadata(
            *(None,) * len(AncillaryVariableMetadata._fields)
        )
        with mock.patch.object(
            BaseMetadata, "difference", return_value=return_value
        ) as mocker:
            result = metadata.difference(other)

        self.assertEqual(return_value, result)
        self.assertEqual(1, mocker.call_count)
        (arg,), kwargs = mocker.call_args
        self.assertEqual(other, arg)
        self.assertEqual(dict(lenient=None), kwargs)

    def test_lenient(self):
        other = sentinel.other
        lenient = sentinel.lenient
        return_value = sentinel.return_value
        metadata = AncillaryVariableMetadata(
            *(None,) * len(AncillaryVariableMetadata._fields)
        )
        with mock.patch.object(
            BaseMetadata, "difference", return_value=return_value
        ) as mocker:
            result = metadata.difference(other, lenient=lenient)

        self.assertEqual(return_value, result)
        self.assertEqual(1, mocker.call_count)
        (arg,), kwargs = mocker.call_args
        self.assertEqual(other, arg)
        self.assertEqual(dict(lenient=lenient), kwargs)


class Test_equal(tests.IrisTest):
    def test_wraps_docstring(self):
        self.assertEqual(
            BaseMetadata.equal.__doc__, AncillaryVariableMetadata.equal.__doc__
        )

    def test_lenient_service(self):
        qualname_equal = qualname(AncillaryVariableMetadata.equal)
        self.assertIn(qualname_equal, LENIENT)
        self.assertTrue(LENIENT[qualname_equal])
        self.assertTrue(LENIENT[AncillaryVariableMetadata.equal])

    def test_lenient_default(self):
        other = sentinel.other
        return_value = sentinel.return_value
        metadata = AncillaryVariableMetadata(
            *(None,) * len(AncillaryVariableMetadata._fields)
        )
        with mock.patch.object(
            BaseMetadata, "equal", return_value=return_value
        ) as mocker:
            result = metadata.equal(other)

        self.assertEqual(return_value, result)
        self.assertEqual(1, mocker.call_count)
        (arg,), kwargs = mocker.call_args
        self.assertEqual(other, arg)
        self.assertEqual(dict(lenient=None), kwargs)

    def test_lenient(self):
        other = sentinel.other
        lenient = sentinel.lenient
        return_value = sentinel.return_value
        metadata = AncillaryVariableMetadata(
            *(None,) * len(AncillaryVariableMetadata._fields)
        )
        with mock.patch.object(
            BaseMetadata, "equal", return_value=return_value
        ) as mocker:
            result = metadata.equal(other, lenient=lenient)

        self.assertEqual(return_value, result)
        self.assertEqual(1, mocker.call_count)
        (arg,), kwargs = mocker.call_args
        self.assertEqual(other, arg)
        self.assertEqual(dict(lenient=lenient), kwargs)


if __name__ == "__main__":
    tests.main()
