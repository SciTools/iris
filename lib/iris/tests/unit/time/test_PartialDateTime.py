# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.time.PartialDateTime` class."""

import datetime
import operator

import cftime
import pytest

from iris.time import PartialDateTime


class Test___init__:
    def test_positional(self):
        # Test that we can define PartialDateTimes with positional arguments.
        pd = PartialDateTime(1066, None, 10)
        assert pd.year == 1066
        assert pd.month is None
        assert pd.day == 10

    def test_keyword_args(self):
        # Test that we can define PartialDateTimes with keyword arguments.
        pd = PartialDateTime(microsecond=10)
        assert pd.year is None
        assert pd.microsecond == 10


class Test___repr__:
    def test_full(self):
        pd = PartialDateTime(*list(range(7)))
        result = repr(pd)
        assert (
            result == "PartialDateTime(year=0, month=1, day=2,"
            " hour=3, minute=4, second=5,"
            " microsecond=6)"
        )

    def test_partial(self):
        pd = PartialDateTime(month=2, day=30)
        result = repr(pd)
        assert result == "PartialDateTime(month=2, day=30)"

    def test_empty(self):
        pd = PartialDateTime()
        result = repr(pd)
        assert result == "PartialDateTime()"


class Test_timetuple:
    def test_exists(self):
        # Check that the PartialDateTime class implements a timetuple (needed
        # because of https://bugs.python.org/issue8005).
        pd = PartialDateTime(*list(range(7)))
        assert hasattr(pd, "timetuple")


class _Test_operator:
    def test_invalid_type(self):
        pdt = PartialDateTime()
        with pytest.raises(TypeError):
            self.op(pdt, 1)

    def _test(self, pdt, other, name):
        expected = self.expected_value[name]
        if isinstance(expected, type):
            with pytest.raises(expected):
                result = self.op(pdt, other)
        else:
            result = self.op(pdt, other)
            assert result is expected

    def _test_dt(self, pdt, name, mocker):
        other = mocker.Mock(
            name="datetime",
            spec=datetime.datetime,
            year=2013,
            month=3,
            day=20,
            second=2,
        )
        self._test(pdt, other, name)

    def test_no_difference(self, mocker):
        self._test_dt(
            PartialDateTime(year=2013, month=3, day=20, second=2),
            "no_difference",
            mocker,
        )

    def test_null(self, mocker):
        self._test_dt(PartialDateTime(), "null", mocker)

    def test_item1_lo(self, mocker):
        self._test_dt(PartialDateTime(year=2011, month=3, second=2), "item1_lo", mocker)

    def test_item1_hi(self, mocker):
        self._test_dt(PartialDateTime(year=2015, month=3, day=24), "item1_hi", mocker)

    def test_item2_lo(self, mocker):
        self._test_dt(PartialDateTime(year=2013, month=1, second=2), "item2_lo", mocker)

    def test_item2_hi(self, mocker):
        self._test_dt(PartialDateTime(year=2013, month=5, day=24), "item2_hi", mocker)

    def test_item3_lo(self, mocker):
        self._test_dt(PartialDateTime(year=2013, month=3, second=1), "item3_lo", mocker)

    def test_item3_hi(self, mocker):
        self._test_dt(
            PartialDateTime(year=2013, month=3, second=42), "item3_hi", mocker
        )

    def test_mix_hi_lo(self, mocker):
        self._test_dt(PartialDateTime(year=2015, month=1, day=24), "mix_hi_lo", mocker)

    def test_mix_lo_hi(self, mocker):
        self._test_dt(PartialDateTime(year=2011, month=5, day=24), "mix_lo_hi", mocker)

    def _test_pdt(self, other, name):
        pdt = PartialDateTime(year=2013, day=24)
        self._test(pdt, other, name)

    def test_pdt_same(self):
        self._test_pdt(PartialDateTime(year=2013, day=24), "pdt_same")

    def test_pdt_diff(self):
        self._test_pdt(PartialDateTime(year=2013, day=25), "pdt_diff")

    def test_pdt_diff_fewer_fields(self):
        self._test_pdt(PartialDateTime(year=2013), "pdt_diff_fewer")

    def test_pdt_diff_more_fields(self):
        self._test_pdt(PartialDateTime(year=2013, day=24, hour=12), "pdt_diff_more")

    def test_pdt_diff_no_fields(self):
        pdt1 = PartialDateTime()
        pdt2 = PartialDateTime(month=3, day=24)
        self._test(pdt1, pdt2, "pdt_empty")


def negate_expectations(expectations):
    def negate(expected):
        if not isinstance(expected, type):
            expected = not expected
        return expected

    return {name: negate(value) for name, value in expectations.items()}


EQ_EXPECTATIONS = {
    "no_difference": True,
    "item1_lo": False,
    "item1_hi": False,
    "item2_lo": False,
    "item2_hi": False,
    "item3_lo": False,
    "item3_hi": False,
    "mix_hi_lo": False,
    "mix_lo_hi": False,
    "null": True,
    "pdt_same": True,
    "pdt_diff": False,
    "pdt_diff_fewer": False,
    "pdt_diff_more": False,
    "pdt_empty": False,
}

GT_EXPECTATIONS = {
    "no_difference": False,
    "item1_lo": False,
    "item1_hi": True,
    "item2_lo": False,
    "item2_hi": True,
    "item3_lo": False,
    "item3_hi": True,
    "mix_hi_lo": True,
    "mix_lo_hi": False,
    "null": False,
    "pdt_same": TypeError,
    "pdt_diff": TypeError,
    "pdt_diff_fewer": TypeError,
    "pdt_diff_more": TypeError,
    "pdt_empty": TypeError,
}

LT_EXPECTATIONS = {
    "no_difference": False,
    "item1_lo": True,
    "item1_hi": False,
    "item2_lo": True,
    "item2_hi": False,
    "item3_lo": True,
    "item3_hi": False,
    "mix_hi_lo": False,
    "mix_lo_hi": True,
    "null": False,
    "pdt_same": TypeError,
    "pdt_diff": TypeError,
    "pdt_diff_fewer": TypeError,
    "pdt_diff_more": TypeError,
    "pdt_empty": TypeError,
}


class Test___eq__(_Test_operator):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.op = operator.eq
        self.expected_value = EQ_EXPECTATIONS

    def test_cftime_equal(self):
        pdt = PartialDateTime(month=3, second=2)
        other = cftime.datetime(year=2013, month=3, day=20, second=2)
        assert pdt == other

    def test_cftime_not_equal(self):
        pdt = PartialDateTime(month=3, second=2)
        other = cftime.datetime(year=2013, month=4, day=20, second=2)
        assert not pdt == other


class Test___ne__(_Test_operator):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.op = operator.ne
        self.expected_value = negate_expectations(EQ_EXPECTATIONS)


class Test___gt__(_Test_operator):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.op = operator.gt
        self.expected_value = GT_EXPECTATIONS

    def test_cftime_greater(self):
        pdt = PartialDateTime(month=3, microsecond=2)
        other = cftime.datetime(year=2013, month=2, day=20, second=3)
        assert pdt > other

    def test_cftime_not_greater(self):
        pdt = PartialDateTime(month=3, microsecond=2)
        other = cftime.datetime(year=2013, month=3, day=20, second=3)
        assert not pdt > other


class Test___le__(_Test_operator):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.op = operator.le
        self.expected_value = negate_expectations(GT_EXPECTATIONS)


class Test___lt__(_Test_operator):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.op = operator.lt
        self.expected_value = LT_EXPECTATIONS


class Test___ge__(_Test_operator):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.op = operator.ge
        self.expected_value = negate_expectations(LT_EXPECTATIONS)
