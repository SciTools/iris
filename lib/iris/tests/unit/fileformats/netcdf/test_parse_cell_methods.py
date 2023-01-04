# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for :func:`iris.fileformats.netcdf.parse_cell_methods`.

"""
import warnings

import pytest

from iris.coords import CellMethod
from iris.fileformats.netcdf import (
    UnknownCellMethodWarning,
    parse_cell_methods,
)


class TestParseCellMethods:
    def _check_answers(self, test_string_or_strings, result):
        """
        Compare a list of test strings against a single expected result.

        Done this way so that any failures produce intelligible Pytest messages.
        """
        if isinstance(test_string_or_strings, str):
            test_string_or_strings = [test_string_or_strings]
        expected_tests_and_results = [
            (cell_method_str, result)
            for cell_method_str in test_string_or_strings
        ]
        actual_tests_and_results = [
            (cell_method_str, parse_cell_methods(cell_method_str))
            for cell_method_str in test_string_or_strings
        ]
        assert actual_tests_and_results == expected_tests_and_results

    def test_simple(self):
        # Some simple testcases which should all have the same result
        cell_method_strings = [
            "time: mean",
            "time : mean",
        ]
        expected = (CellMethod(method="mean", coords="time"),)
        self._check_answers(cell_method_strings, expected)

    def test_with_interval(self):
        cell_method_strings = [
            "time: variance (interval: 1 hr)",
            "time : variance (interval: 1 hr)",
        ]
        expected = (
            CellMethod(method="variance", coords="time", intervals="1 hr"),
        )
        self._check_answers(cell_method_strings, expected)

    def test_multiple_axes(self):
        cell_method_strings = [
            "lat: lon: standard_deviation",
            "lat: lon : standard_deviation",
            "lat : lon: standard_deviation",
            "lat : lon : standard_deviation",
        ]
        expected = (
            CellMethod(method="standard_deviation", coords=["lat", "lon"]),
        )
        self._check_answers(cell_method_strings, expected)

    def test_multiple(self):
        cell_method_strings = [
            "time: maximum (interval: 1 hr) time: mean (interval: 1 day)",
            "time : maximum (interval: 1 hr) time: mean (interval: 1 day)",
            "time: maximum (interval: 1 hr) time : mean (interval: 1 day)",
            "time : maximum (interval: 1 hr) time : mean (interval: 1 day)",
        ]
        expected = (
            CellMethod(method="maximum", coords="time", intervals="1 hr"),
            CellMethod(method="mean", coords="time", intervals="1 day"),
        )
        self._check_answers(cell_method_strings, expected)

    def test_comment(self):
        cell_method_strings = [
            (
                "time: maximum (interval: 1 hr comment: first bit) "
                "time: mean (interval: 1 day comment: second bit)"
            ),
            (
                "time : maximum (interval: 1 hr comment: first bit) "
                "time: mean (interval: 1 day comment: second bit)"
            ),
            (
                "time: maximum (interval: 1 hr comment: first bit) "
                "time : mean (interval: 1 day comment: second bit)"
            ),
            (
                "time : maximum (interval: 1 hr comment: first bit) "
                "time : mean (interval: 1 day comment: second bit)"
            ),
        ]
        expected = (
            CellMethod(
                method="maximum",
                coords="time",
                intervals="1 hr",
                comments="first bit",
            ),
            CellMethod(
                method="mean",
                coords="time",
                intervals="1 day",
                comments="second bit",
            ),
        )
        self._check_answers(cell_method_strings, expected)

    def test_comment_brackets(self):
        cell_method_strings = [
            "time: minimum within days (comment: 18h(day-1)-18h)",
            "time : minimum within days (comment: 18h(day-1)-18h)",
        ]
        expected = (
            CellMethod(
                method="minimum within days",
                coords="time",
                intervals=None,
                comments="18h(day-1)-18h",
            ),
        )
        self._check_answers(cell_method_strings, expected)

    def test_comment_bracket_mismatch_warning(self):
        cell_method_strings = [
            "time: minimum within days (comment: 18h day-1)-18h)",
            "time : minimum within days (comment: 18h day-1)-18h)",
        ]
        expected = (
            CellMethod(
                method="minimum within days",
                coords="time",
                intervals=None,
                comments="18h day-1)-18h",
            ),
        )
        msg = (
            "Cell methods may be incorrectly parsed due to mismatched brackets"
        )
        for cell_method_str in cell_method_strings:
            with pytest.warns(UserWarning, match=msg):
                self._check_answers(cell_method_strings, expected)

    def test_badly_formatted__warns(self):
        cell_method_strings = [
            (
                "time: (interval: 1 hr comment: first bit) "
                "time: mean (interval: 1 day comment: second bit)"
            ),
            (
                "time: maximum (interval: 1 hr comment: first bit) "
                "time: (interval: 1 day comment: second bit)"
            ),
        ]
        for cell_method_str in cell_method_strings[1:]:
            with pytest.warns(
                UserWarning,
                match="Failed to fully parse cell method string: time: ",
            ):
                _ = parse_cell_methods(cell_method_str)

    def test_portions_of_cells(self):
        cell_method_strings = [
            "area: mean where sea_ice over sea",
            "area : mean where sea_ice over sea",
        ]
        expected = (
            CellMethod(method="mean where sea_ice over sea", coords="area"),
        )
        self._check_answers(cell_method_strings, expected)

    def test_climatology(self):
        cell_method_strings = [
            "time: minimum within days time: mean over days",
            "time : minimum within days time: mean over days",
            "time: minimum within days time : mean over days",
            "time : minimum within days time : mean over days",
        ]
        expected = (
            CellMethod(method="minimum within days", coords="time"),
            CellMethod(method="mean over days", coords="time"),
        )
        self._check_answers(cell_method_strings, expected)

    def test_climatology_with_unknown_method__warns(self):
        cell_method_strings = [
            "time: min within days time: mean over days",
            "time : min within days time: mean over days",
            "time: min within days time : mean over days",
            "time : min within days time : mean over days",
        ]
        expected = (
            CellMethod(method="min within days", coords="time"),
            CellMethod(method="mean over days", coords="time"),
        )
        msg = "NetCDF variable contains unknown cell method 'min'"
        for cell_method_str in cell_method_strings:
            with pytest.warns(UnknownCellMethodWarning, match=msg):
                res = parse_cell_methods(cell_method_str)
                assert res == expected

    def test_empty__warns(self):
        cm_str = ""
        msg = "contains no valid cell methods"
        with pytest.warns(UserWarning, match=msg):
            result = parse_cell_methods(cm_str)
        assert result == ()

    def test_whitespace__warns(self):
        cm_str = " \t "
        msg = "contains no valid cell methods"
        with pytest.warns(UserWarning, match=msg):
            result = parse_cell_methods(cm_str)
        assert result == ()

    def test_barename__warns(self):
        cm_str = "time"
        msg = "contains no valid cell methods"
        with pytest.warns(UserWarning, match=msg):
            result = parse_cell_methods(cm_str)
        assert result == ()

    def test_missedspace__warns(self):
        cm_str = "time:mean"
        msg = "contains no valid cell methods"
        with pytest.warns(UserWarning, match=msg):
            result = parse_cell_methods(cm_str)
        assert result == ()

    def test_random_junk__warns(self):
        cm_str = "y:12+4#?x:this"
        msg = "contains no valid cell methods"
        with pytest.warns(UserWarning, match=msg):
            result = parse_cell_methods(cm_str)
        assert result == ()

    def test_junk_after__silentlyignores(self):
        cm_str = "time: mean -?-"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = parse_cell_methods(cm_str)
        expected = (CellMethod("mean", ("time",)),)
        assert result == expected

    def test_junk_before__silentlyignores(self):
        cm_str = "-?- time: mean"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = parse_cell_methods(cm_str)
        expected = (CellMethod("mean", ("time",)),)
        assert result == expected

    def test_embeddedcolon__silentlyignores(self):
        cm_str = "time:any: mean"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = parse_cell_methods(cm_str)
        # N.B. treats the initial "time:" as plain junk + discards it
        expected = (CellMethod("mean", ("any",)),)
        assert result == expected
