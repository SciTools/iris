# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :class:`iris.common.metadata.CellMeasureMetadata`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from copy import deepcopy
import unittest.mock as mock
from unittest.mock import sentinel

from iris.common.lenient import LENIENT, qualname
from iris.common.metadata import BaseMetadata, CellMeasureMetadata


class Test(tests.IrisTest):
    def setUp(self):
        self.standard_name = mock.sentinel.standard_name
        self.long_name = mock.sentinel.long_name
        self.var_name = mock.sentinel.var_name
        self.units = mock.sentinel.units
        self.attributes = mock.sentinel.attributes
        self.measure = mock.sentinel.measure

    def test_repr(self):
        metadata = CellMeasureMetadata(
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
            units=self.units,
            attributes=self.attributes,
            measure=self.measure,
        )
        fmt = (
            "CellMeasureMetadata(standard_name={!r}, long_name={!r}, "
            "var_name={!r}, units={!r}, attributes={!r}, measure={!r})"
        )
        expected = fmt.format(
            self.standard_name,
            self.long_name,
            self.var_name,
            self.units,
            self.attributes,
            self.measure,
        )
        self.assertEqual(expected, repr(metadata))

    def test__fields(self):
        expected = (
            "standard_name",
            "long_name",
            "var_name",
            "units",
            "attributes",
            "measure",
        )
        self.assertEqual(CellMeasureMetadata._fields, expected)

    def test_bases(self):
        self.assertTrue(issubclass(CellMeasureMetadata, BaseMetadata))


class Test___eq__(tests.IrisTest):
    def setUp(self):
        self.values = dict(
            standard_name=sentinel.standard_name,
            long_name=sentinel.long_name,
            var_name=sentinel.var_name,
            units=sentinel.units,
            attributes=sentinel.attributes,
            measure=sentinel.measure,
        )
        self.dummy = sentinel.dummy

    def test_wraps_docstring(self):
        self.assertEqual(
            BaseMetadata.__eq__.__doc__, CellMeasureMetadata.__eq__.__doc__,
        )

    def test_lenient_service(self):
        qualname___eq__ = qualname(CellMeasureMetadata.__eq__)
        self.assertIn(qualname___eq__, LENIENT)
        self.assertTrue(LENIENT[qualname___eq__])
        self.assertTrue(LENIENT[CellMeasureMetadata.__eq__])

    def test_call(self):
        other = sentinel.other
        return_value = sentinel.return_value
        metadata = CellMeasureMetadata(
            *(None,) * len(CellMeasureMetadata._fields)
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

    def test_op_lenient_same(self):
        lmetadata = CellMeasureMetadata(**self.values)
        rmetadata = CellMeasureMetadata(**self.values)

        with mock.patch("iris.common.metadata.LENIENT", return_value=True):
            self.assertTrue(lmetadata.__eq__(rmetadata))
            self.assertTrue(rmetadata.__eq__(lmetadata))

    def test_op_lenient_same_none(self):
        lmetadata = CellMeasureMetadata(**self.values)
        right = self.values.copy()
        right["var_name"] = None
        rmetadata = CellMeasureMetadata(**right)

        with mock.patch("iris.common.metadata.LENIENT", return_value=True):
            self.assertTrue(lmetadata.__eq__(rmetadata))
            self.assertTrue(rmetadata.__eq__(lmetadata))

    def test_op_lenient_same_measure_none(self):
        lmetadata = CellMeasureMetadata(**self.values)
        right = self.values.copy()
        right["measure"] = None
        rmetadata = CellMeasureMetadata(**right)

        with mock.patch("iris.common.metadata.LENIENT", return_value=True):
            self.assertFalse(lmetadata.__eq__(rmetadata))
            self.assertFalse(rmetadata.__eq__(lmetadata))

    def test_op_lenient_different(self):
        lmetadata = CellMeasureMetadata(**self.values)
        right = self.values.copy()
        right["units"] = self.dummy
        rmetadata = CellMeasureMetadata(**right)

        with mock.patch("iris.common.metadata.LENIENT", return_value=True):
            self.assertFalse(lmetadata.__eq__(rmetadata))
            self.assertFalse(rmetadata.__eq__(lmetadata))

    def test_op_lenient_different_measure(self):
        lmetadata = CellMeasureMetadata(**self.values)
        right = self.values.copy()
        right["measure"] = self.dummy
        rmetadata = CellMeasureMetadata(**right)

        with mock.patch("iris.common.metadata.LENIENT", return_value=True):
            self.assertFalse(lmetadata.__eq__(rmetadata))
            self.assertFalse(rmetadata.__eq__(lmetadata))

    def test_op_strict_same(self):
        lmetadata = CellMeasureMetadata(**self.values)
        rmetadata = CellMeasureMetadata(**self.values)

        with mock.patch("iris.common.metadata.LENIENT", return_value=False):
            self.assertTrue(lmetadata.__eq__(rmetadata))
            self.assertTrue(rmetadata.__eq__(lmetadata))

    def test_op_strict_different(self):
        lmetadata = CellMeasureMetadata(**self.values)
        right = self.values.copy()
        right["long_name"] = self.dummy
        rmetadata = CellMeasureMetadata(**right)

        with mock.patch("iris.common.metadata.LENIENT", return_value=False):
            self.assertFalse(lmetadata.__eq__(rmetadata))
            self.assertFalse(rmetadata.__eq__(lmetadata))

    def test_op_strict_different_measure(self):
        lmetadata = CellMeasureMetadata(**self.values)
        right = self.values.copy()
        right["measure"] = self.dummy
        rmetadata = CellMeasureMetadata(**right)

        with mock.patch("iris.common.metadata.LENIENT", return_value=False):
            self.assertFalse(lmetadata.__eq__(rmetadata))
            self.assertFalse(rmetadata.__eq__(lmetadata))

    def test_op_strict_different_none(self):
        lmetadata = CellMeasureMetadata(**self.values)
        right = self.values.copy()
        right["long_name"] = None
        rmetadata = CellMeasureMetadata(**right)

        with mock.patch("iris.common.metadata.LENIENT", return_value=False):
            self.assertFalse(lmetadata.__eq__(rmetadata))
            self.assertFalse(rmetadata.__eq__(lmetadata))

    def test_op_strict_different_measure_none(self):
        lmetadata = CellMeasureMetadata(**self.values)
        right = self.values.copy()
        right["measure"] = None
        rmetadata = CellMeasureMetadata(**right)

        with mock.patch("iris.common.metadata.LENIENT", return_value=False):
            self.assertFalse(lmetadata.__eq__(rmetadata))
            self.assertFalse(rmetadata.__eq__(lmetadata))


class Test_combine(tests.IrisTest):
    def setUp(self):
        self.values = dict(
            standard_name=sentinel.standard_name,
            long_name=sentinel.long_name,
            var_name=sentinel.var_name,
            units=sentinel.units,
            attributes=sentinel.attributes,
            measure=sentinel.measure,
        )
        self.dummy = sentinel.dummy
        self.none = CellMeasureMetadata(
            *(None,) * len(CellMeasureMetadata._fields)
        )

    def test_wraps_docstring(self):
        self.assertEqual(
            BaseMetadata.combine.__doc__, CellMeasureMetadata.combine.__doc__,
        )

    def test_lenient_service(self):
        qualname_combine = qualname(CellMeasureMetadata.combine)
        self.assertIn(qualname_combine, LENIENT)
        self.assertTrue(LENIENT[qualname_combine])
        self.assertTrue(LENIENT[CellMeasureMetadata.combine])

    def test_lenient_default(self):
        other = sentinel.other
        return_value = sentinel.return_value
        with mock.patch.object(
            BaseMetadata, "combine", return_value=return_value
        ) as mocker:
            result = self.none.combine(other)

        self.assertEqual(return_value, result)
        self.assertEqual(1, mocker.call_count)
        (arg,), kwargs = mocker.call_args
        self.assertEqual(other, arg)
        self.assertEqual(dict(lenient=None), kwargs)

    def test_lenient(self):
        other = sentinel.other
        lenient = sentinel.lenient
        return_value = sentinel.return_value
        with mock.patch.object(
            BaseMetadata, "combine", return_value=return_value
        ) as mocker:
            result = self.none.combine(other, lenient=lenient)

        self.assertEqual(return_value, result)
        self.assertEqual(1, mocker.call_count)
        (arg,), kwargs = mocker.call_args
        self.assertEqual(other, arg)
        self.assertEqual(dict(lenient=lenient), kwargs)

    def test_op_lenient_same(self):
        lmetadata = CellMeasureMetadata(**self.values)
        rmetadata = CellMeasureMetadata(**self.values)
        expected = self.values

        with mock.patch("iris.common.metadata.LENIENT", return_value=True):
            self.assertEqual(expected, lmetadata.combine(rmetadata)._asdict())
            self.assertEqual(expected, rmetadata.combine(lmetadata)._asdict())

    def test_op_lenient_same_none(self):
        lmetadata = CellMeasureMetadata(**self.values)
        right = self.values.copy()
        right["var_name"] = None
        rmetadata = CellMeasureMetadata(**right)
        expected = self.values

        with mock.patch("iris.common.metadata.LENIENT", return_value=True):
            self.assertEqual(expected, lmetadata.combine(rmetadata)._asdict())
            self.assertEqual(expected, rmetadata.combine(lmetadata)._asdict())

    def test_op_lenient_same_measure_none(self):
        lmetadata = CellMeasureMetadata(**self.values)
        right = self.values.copy()
        right["measure"] = None
        rmetadata = CellMeasureMetadata(**right)
        expected = right.copy()

        with mock.patch("iris.common.metadata.LENIENT", return_value=True):
            self.assertTrue(expected, lmetadata.combine(rmetadata)._asdict())
            self.assertTrue(expected, rmetadata.combine(lmetadata)._asdict())

    def test_op_lenient_different(self):
        lmetadata = CellMeasureMetadata(**self.values)
        right = self.values.copy()
        right["units"] = self.dummy
        rmetadata = CellMeasureMetadata(**right)
        expected = self.values.copy()
        expected["units"] = None

        with mock.patch("iris.common.metadata.LENIENT", return_value=True):
            self.assertEqual(expected, lmetadata.combine(rmetadata)._asdict())
            self.assertEqual(expected, rmetadata.combine(lmetadata)._asdict())

    def test_op_lenient_different_measure(self):
        lmetadata = CellMeasureMetadata(**self.values)
        right = self.values.copy()
        right["measure"] = self.dummy
        rmetadata = CellMeasureMetadata(**right)
        expected = self.values.copy()
        expected["measure"] = None

        with mock.patch("iris.common.metadata.LENIENT", return_value=True):
            self.assertEqual(expected, lmetadata.combine(rmetadata)._asdict())
            self.assertEqual(expected, rmetadata.combine(lmetadata)._asdict())

    def test_op_strict_same(self):
        lmetadata = CellMeasureMetadata(**self.values)
        rmetadata = CellMeasureMetadata(**self.values)
        expected = self.values.copy()

        with mock.patch("iris.common.metadata.LENIENT", return_value=False):
            self.assertEqual(expected, lmetadata.combine(rmetadata)._asdict())
            self.assertEqual(expected, rmetadata.combine(lmetadata)._asdict())

    def test_op_strict_different(self):
        lmetadata = CellMeasureMetadata(**self.values)
        right = self.values.copy()
        right["long_name"] = self.dummy
        rmetadata = CellMeasureMetadata(**right)
        expected = self.values.copy()
        expected["long_name"] = None

        with mock.patch("iris.common.metadata.LENIENT", return_value=False):
            self.assertEqual(expected, lmetadata.combine(rmetadata)._asdict())
            self.assertEqual(expected, rmetadata.combine(lmetadata)._asdict())

    def test_op_strict_different_measure(self):
        lmetadata = CellMeasureMetadata(**self.values)
        right = self.values.copy()
        right["measure"] = self.dummy
        rmetadata = CellMeasureMetadata(**right)
        expected = self.values.copy()
        expected["measure"] = None

        with mock.patch("iris.common.metadata.LENIENT", return_value=False):
            self.assertEqual(expected, lmetadata.combine(rmetadata)._asdict())
            self.assertEqual(expected, rmetadata.combine(lmetadata)._asdict())

    def test_op_strict_different_none(self):
        lmetadata = CellMeasureMetadata(**self.values)
        right = self.values.copy()
        right["long_name"] = None
        rmetadata = CellMeasureMetadata(**right)
        expected = self.values.copy()
        expected["long_name"] = None

        with mock.patch("iris.common.metadata.LENIENT", return_value=False):
            self.assertEqual(expected, lmetadata.combine(rmetadata)._asdict())
            self.assertEqual(expected, rmetadata.combine(lmetadata)._asdict())

    def test_op_strict_different_measure_none(self):
        lmetadata = CellMeasureMetadata(**self.values)
        right = self.values.copy()
        right["measure"] = None
        rmetadata = CellMeasureMetadata(**right)
        expected = self.values.copy()
        expected["measure"] = None

        with mock.patch("iris.common.metadata.LENIENT", return_value=False):
            self.assertEqual(expected, lmetadata.combine(rmetadata)._asdict())
            self.assertEqual(expected, rmetadata.combine(lmetadata)._asdict())


class Test_difference(tests.IrisTest):
    def setUp(self):
        self.values = dict(
            standard_name=sentinel.standard_name,
            long_name=sentinel.long_name,
            var_name=sentinel.var_name,
            units=sentinel.units,
            attributes=sentinel.attributes,
            measure=sentinel.measure,
        )
        self.dummy = sentinel.dummy
        self.none = CellMeasureMetadata(
            *(None,) * len(CellMeasureMetadata._fields)
        )

    def test_wraps_docstring(self):
        self.assertEqual(
            BaseMetadata.difference.__doc__,
            CellMeasureMetadata.difference.__doc__,
        )

    def test_lenient_service(self):
        qualname_difference = qualname(CellMeasureMetadata.difference)
        self.assertIn(qualname_difference, LENIENT)
        self.assertTrue(LENIENT[qualname_difference])
        self.assertTrue(LENIENT[CellMeasureMetadata.difference])

    def test_lenient_default(self):
        other = sentinel.other
        return_value = sentinel.return_value
        with mock.patch.object(
            BaseMetadata, "difference", return_value=return_value
        ) as mocker:
            result = self.none.difference(other)

        self.assertEqual(return_value, result)
        self.assertEqual(1, mocker.call_count)
        (arg,), kwargs = mocker.call_args
        self.assertEqual(other, arg)
        self.assertEqual(dict(lenient=None), kwargs)

    def test_lenient(self):
        other = sentinel.other
        lenient = sentinel.lenient
        return_value = sentinel.return_value
        with mock.patch.object(
            BaseMetadata, "difference", return_value=return_value
        ) as mocker:
            result = self.none.difference(other, lenient=lenient)

        self.assertEqual(return_value, result)
        self.assertEqual(1, mocker.call_count)
        (arg,), kwargs = mocker.call_args
        self.assertEqual(other, arg)
        self.assertEqual(dict(lenient=lenient), kwargs)

    def test_op_lenient_same(self):
        lmetadata = CellMeasureMetadata(**self.values)
        rmetadata = CellMeasureMetadata(**self.values)
        expected = deepcopy(self.none)._asdict()

        with mock.patch("iris.common.metadata.LENIENT", return_value=True):
            self.assertEqual(
                expected, lmetadata.difference(rmetadata)._asdict()
            )
            self.assertEqual(
                expected, rmetadata.difference(lmetadata)._asdict()
            )

    def test_op_lenient_same_none(self):
        lmetadata = CellMeasureMetadata(**self.values)
        right = self.values.copy()
        right["var_name"] = None
        rmetadata = CellMeasureMetadata(**right)
        expected = deepcopy(self.none)._asdict()

        with mock.patch("iris.common.metadata.LENIENT", return_value=True):
            self.assertEqual(
                expected, lmetadata.difference(rmetadata)._asdict()
            )
            self.assertEqual(
                expected, rmetadata.difference(lmetadata)._asdict()
            )

    def test_op_lenient_same_measure_none(self):
        lmetadata = CellMeasureMetadata(**self.values)
        right = self.values.copy()
        right["measure"] = None
        rmetadata = CellMeasureMetadata(**right)
        lexpected = deepcopy(self.none)._asdict()
        lexpected["measure"] = (sentinel.measure, None)
        rexpected = deepcopy(self.none)._asdict()
        rexpected["measure"] = (None, sentinel.measure)

        with mock.patch("iris.common.metadata.LENIENT", return_value=True):
            self.assertEqual(
                lexpected, lmetadata.difference(rmetadata)._asdict()
            )
            self.assertEqual(
                rexpected, rmetadata.difference(lmetadata)._asdict()
            )

    def test_op_lenient_different(self):
        left = self.values.copy()
        lmetadata = CellMeasureMetadata(**left)
        right = self.values.copy()
        right["units"] = self.dummy
        rmetadata = CellMeasureMetadata(**right)
        lexpected = deepcopy(self.none)._asdict()
        lexpected["units"] = (left["units"], right["units"])
        rexpected = deepcopy(self.none)._asdict()
        rexpected["units"] = lexpected["units"][::-1]

        with mock.patch("iris.common.metadata.LENIENT", return_value=True):
            self.assertEqual(
                lexpected, lmetadata.difference(rmetadata)._asdict()
            )
            self.assertEqual(
                rexpected, rmetadata.difference(lmetadata)._asdict()
            )

    def test_op_lenient_different_measure(self):
        left = self.values.copy()
        lmetadata = CellMeasureMetadata(**left)
        right = self.values.copy()
        right["measure"] = self.dummy
        rmetadata = CellMeasureMetadata(**right)
        lexpected = deepcopy(self.none)._asdict()
        lexpected["measure"] = (left["measure"], right["measure"])
        rexpected = deepcopy(self.none)._asdict()
        rexpected["measure"] = lexpected["measure"][::-1]

        with mock.patch("iris.common.metadata.LENIENT", return_value=True):
            self.assertEqual(
                lexpected, lmetadata.difference(rmetadata)._asdict()
            )
            self.assertEqual(
                rexpected, rmetadata.difference(lmetadata)._asdict()
            )

    def test_op_strict_same(self):
        lmetadata = CellMeasureMetadata(**self.values)
        rmetadata = CellMeasureMetadata(**self.values)
        expected = deepcopy(self.none)._asdict()

        with mock.patch("iris.common.metadata.LENIENT", return_value=False):
            self.assertEqual(
                expected, lmetadata.difference(rmetadata)._asdict()
            )
            self.assertEqual(
                expected, rmetadata.difference(lmetadata)._asdict()
            )

    def test_op_strict_different(self):
        left = self.values.copy()
        lmetadata = CellMeasureMetadata(**left)
        right = self.values.copy()
        right["long_name"] = self.dummy
        rmetadata = CellMeasureMetadata(**right)
        lexpected = deepcopy(self.none)._asdict()
        lexpected["long_name"] = (left["long_name"], right["long_name"])
        rexpected = deepcopy(self.none)._asdict()
        rexpected["long_name"] = lexpected["long_name"][::-1]

        with mock.patch("iris.common.metadata.LENIENT", return_value=False):
            self.assertEqual(
                lexpected, lmetadata.difference(rmetadata)._asdict()
            )
            self.assertEqual(
                rexpected, rmetadata.difference(lmetadata)._asdict()
            )

    def test_op_strict_different_measure(self):
        left = self.values.copy()
        lmetadata = CellMeasureMetadata(**left)
        right = self.values.copy()
        right["measure"] = self.dummy
        rmetadata = CellMeasureMetadata(**right)
        lexpected = deepcopy(self.none)._asdict()
        lexpected["measure"] = (left["measure"], right["measure"])
        rexpected = deepcopy(self.none)._asdict()
        rexpected["measure"] = lexpected["measure"][::-1]

        with mock.patch("iris.common.metadata.LENIENT", return_value=False):
            self.assertEqual(
                lexpected, lmetadata.difference(rmetadata)._asdict()
            )
            self.assertEqual(
                rexpected, rmetadata.difference(lmetadata)._asdict()
            )

    def test_op_strict_different_none(self):
        left = self.values.copy()
        lmetadata = CellMeasureMetadata(**left)
        right = self.values.copy()
        right["long_name"] = None
        rmetadata = CellMeasureMetadata(**right)
        lexpected = deepcopy(self.none)._asdict()
        lexpected["long_name"] = (left["long_name"], right["long_name"])
        rexpected = deepcopy(self.none)._asdict()
        rexpected["long_name"] = lexpected["long_name"][::-1]

        with mock.patch("iris.common.metadata.LENIENT", return_value=False):
            self.assertEqual(
                lexpected, lmetadata.difference(rmetadata)._asdict()
            )
            self.assertEqual(
                rexpected, rmetadata.difference(lmetadata)._asdict()
            )

    def test_op_strict_different_measure_none(self):
        left = self.values.copy()
        lmetadata = CellMeasureMetadata(**left)
        right = self.values.copy()
        right["measure"] = None
        rmetadata = CellMeasureMetadata(**right)
        lexpected = deepcopy(self.none)._asdict()
        lexpected["measure"] = (left["measure"], right["measure"])
        rexpected = deepcopy(self.none)._asdict()
        rexpected["measure"] = lexpected["measure"][::-1]

        with mock.patch("iris.common.metadata.LENIENT", return_value=False):
            self.assertEqual(
                lexpected, lmetadata.difference(rmetadata)._asdict()
            )
            self.assertEqual(
                rexpected, rmetadata.difference(lmetadata)._asdict()
            )


class Test_equal(tests.IrisTest):
    def setUp(self):
        self.none = CellMeasureMetadata(
            *(None,) * len(CellMeasureMetadata._fields)
        )

    def test_wraps_docstring(self):
        self.assertEqual(
            BaseMetadata.equal.__doc__, CellMeasureMetadata.equal.__doc__
        )

    def test_lenient_service(self):
        qualname_equal = qualname(CellMeasureMetadata.equal)
        self.assertIn(qualname_equal, LENIENT)
        self.assertTrue(LENIENT[qualname_equal])
        self.assertTrue(LENIENT[CellMeasureMetadata.equal])

    def test_lenient_default(self):
        other = sentinel.other
        return_value = sentinel.return_value
        with mock.patch.object(
            BaseMetadata, "equal", return_value=return_value
        ) as mocker:
            result = self.none.equal(other)

        self.assertEqual(return_value, result)
        self.assertEqual(1, mocker.call_count)
        (arg,), kwargs = mocker.call_args
        self.assertEqual(other, arg)
        self.assertEqual(dict(lenient=None), kwargs)

    def test_lenient(self):
        other = sentinel.other
        lenient = sentinel.lenient
        return_value = sentinel.return_value
        with mock.patch.object(
            BaseMetadata, "equal", return_value=return_value
        ) as mocker:
            result = self.none.equal(other, lenient=lenient)

        self.assertEqual(return_value, result)
        self.assertEqual(1, mocker.call_count)
        (arg,), kwargs = mocker.call_args
        self.assertEqual(other, arg)
        self.assertEqual(dict(lenient=lenient), kwargs)


if __name__ == "__main__":
    tests.main()
