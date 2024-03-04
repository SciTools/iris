# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the class
:class:`iris.fileformats.um._fast_load_structured_fields.BasicFieldCollation`.

"""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests  # isort:skip

from cftime import datetime
import numpy as np

from iris._lazy_data import as_lazy_data
import iris.fileformats.pp
from iris.fileformats.um._fast_load_structured_fields import (
    BasicFieldCollation,
)


class Test___init__(tests.IrisTest):
    def test_no_fields(self):
        with self.assertRaises(AssertionError):
            BasicFieldCollation([])


class Test_fields(tests.IrisTest):
    def test_preserve_members(self):
        fields = ("foo", "bar", "wibble")
        collation = BasicFieldCollation(fields)
        self.assertEqual(collation.fields, fields)


def _make_field(
    lbyr=None, lbyrd=None, lbft=None, blev=None, bhlev=None, data=None
):
    header = [0] * 64
    if lbyr is not None:
        header[0] = lbyr
        header[1] = 1
        header[2] = 1
    if lbyrd is not None:
        header[6] = lbyrd
        header[7] = 1
        header[8] = 1
    if lbft is not None:
        header[13] = lbft
    if blev is not None:
        header[51] = blev
    if bhlev is not None:
        header[53] = bhlev
    field = iris.fileformats.pp.PPField3(header)
    if data is not None:
        _data = _make_data(data)
        field.data = _data
    return field


def _make_data(fill_value):
    shape = (10, 10)
    return as_lazy_data(np.ones(shape) * fill_value)


class Test_data(tests.IrisTest):
    # Test order of the data attribute when fastest-varying element is changed.
    def test_t1_varies_faster(self):
        collation = BasicFieldCollation(
            [
                _make_field(lbyr=2013, lbyrd=2000, data=0),
                _make_field(lbyr=2014, lbyrd=2000, data=1),
                _make_field(lbyr=2015, lbyrd=2000, data=2),
                _make_field(lbyr=2013, lbyrd=2001, data=3),
                _make_field(lbyr=2014, lbyrd=2001, data=4),
                _make_field(lbyr=2015, lbyrd=2001, data=5),
            ]
        )
        result = collation.data[:, :, 0, 0]
        expected = [[0, 1, 2], [3, 4, 5]]
        self.assertArrayEqual(result, expected)

    def test_t2_varies_faster(self):
        collation = BasicFieldCollation(
            [
                _make_field(lbyr=2013, lbyrd=2000, data=0),
                _make_field(lbyr=2013, lbyrd=2001, data=1),
                _make_field(lbyr=2013, lbyrd=2002, data=2),
                _make_field(lbyr=2014, lbyrd=2000, data=3),
                _make_field(lbyr=2014, lbyrd=2001, data=4),
                _make_field(lbyr=2014, lbyrd=2002, data=5),
            ]
        )
        result = collation.data[:, :, 0, 0]
        expected = [[0, 1, 2], [3, 4, 5]]
        self.assertArrayEqual(result, expected)


class Test_element_arrays_and_dims(tests.IrisTest):
    def test_single_field(self):
        field = _make_field(2013)
        collation = BasicFieldCollation([field])
        self.assertEqual(collation.element_arrays_and_dims, {})

    def test_t1(self):
        collation = BasicFieldCollation(
            [_make_field(lbyr=2013), _make_field(lbyr=2014)]
        )
        result = collation.element_arrays_and_dims
        self.assertEqual(list(result.keys()), ["t1"])
        values, dims = result["t1"]
        self.assertArrayEqual(
            values, [datetime(2013, 1, 1), datetime(2014, 1, 1)]
        )
        self.assertEqual(dims, (0,))

    def test_t1_and_t2(self):
        collation = BasicFieldCollation(
            [
                _make_field(lbyr=2013, lbyrd=2000),
                _make_field(lbyr=2014, lbyrd=2001),
                _make_field(lbyr=2015, lbyrd=2002),
            ]
        )
        result = collation.element_arrays_and_dims
        self.assertEqual(set(result.keys()), set(["t1", "t2"]))
        values, dims = result["t1"]
        self.assertArrayEqual(
            values,
            [datetime(2013, 1, 1), datetime(2014, 1, 1), datetime(2015, 1, 1)],
        )
        self.assertEqual(dims, (0,))
        values, dims = result["t2"]
        self.assertArrayEqual(
            values,
            [datetime(2000, 1, 1), datetime(2001, 1, 1), datetime(2002, 1, 1)],
        )
        self.assertEqual(dims, (0,))

    def test_t1_and_t2_and_lbft(self):
        collation = BasicFieldCollation(
            [
                _make_field(lbyr=1, lbyrd=15, lbft=6),
                _make_field(lbyr=1, lbyrd=16, lbft=9),
                _make_field(lbyr=11, lbyrd=25, lbft=6),
                _make_field(lbyr=11, lbyrd=26, lbft=9),
            ]
        )
        result = collation.element_arrays_and_dims
        self.assertEqual(set(result.keys()), set(["t1", "t2", "lbft"]))
        values, dims = result["t1"]
        self.assertArrayEqual(values, [datetime(1, 1, 1), datetime(11, 1, 1)])
        self.assertEqual(dims, (0,))
        values, dims = result["t2"]
        self.assertArrayEqual(
            values,
            [
                [datetime(15, 1, 1), datetime(16, 1, 1)],
                [datetime(25, 1, 1), datetime(26, 1, 1)],
            ],
        )
        self.assertEqual(dims, (0, 1))
        values, dims = result["lbft"]
        self.assertArrayEqual(values, [6, 9])
        self.assertEqual(dims, (1,))

    def test_blev(self):
        collation = BasicFieldCollation(
            [_make_field(blev=1), _make_field(blev=2)]
        )
        result = collation.element_arrays_and_dims
        keys = set(
            ["blev", "brsvd1", "brsvd2", "brlev", "bhrlev", "lblev", "bhlev"]
        )
        self.assertEqual(set(result.keys()), keys)
        values, dims = result["blev"]
        self.assertArrayEqual(values, [1, 2])
        self.assertEqual(dims, (0,))

    def test_bhlev(self):
        collation = BasicFieldCollation(
            [_make_field(blev=0, bhlev=1), _make_field(blev=1, bhlev=2)]
        )
        result = collation.element_arrays_and_dims
        keys = set(
            ["blev", "brsvd1", "brsvd2", "brlev", "bhrlev", "lblev", "bhlev"]
        )
        self.assertEqual(set(result.keys()), keys)
        values, dims = result["bhlev"]
        self.assertArrayEqual(values, [1, 2])
        self.assertEqual(dims, (0,))


class Test__time_comparable_int(tests.IrisTest):
    def test(self):
        # Define a list of date-time tuples, which should remain both all
        # distinct and in ascending order when converted...
        test_date_tuples = [
            # Increment each component in turn to check that all are handled.
            (2004, 1, 1, 0, 0, 0),
            (2004, 1, 1, 0, 0, 1),
            (2004, 1, 1, 0, 1, 0),
            (2004, 1, 1, 1, 0, 0),
            (2004, 1, 2, 0, 0, 0),
            (2004, 2, 1, 0, 0, 0),
            # Go across 2004-02-29 leap-day, and on to "Feb 31 .. Mar 1".
            (2004, 2, 27, 0, 0, 0),
            (2004, 2, 28, 0, 0, 0),
            (2004, 2, 29, 0, 0, 0),
            (2004, 2, 30, 0, 0, 0),
            (2004, 2, 31, 0, 0, 0),
            (2004, 3, 1, 0, 0, 0),
            (2005, 1, 1, 0, 0, 0),
        ]

        collation = BasicFieldCollation(["foo", "bar"])
        test_date_ints = [
            collation._time_comparable_int(*test_tuple)
            for test_tuple in test_date_tuples
        ]
        # Check all values are distinct.
        self.assertEqual(len(test_date_ints), len(set(test_date_ints)))
        # Check all values are in order.
        self.assertEqual(test_date_ints, sorted(test_date_ints))


if __name__ == "__main__":
    tests.main()
