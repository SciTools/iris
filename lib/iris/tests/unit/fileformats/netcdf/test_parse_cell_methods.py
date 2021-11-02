# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for :func:`iris.fileformats.netcdf.parse_cell_methods`.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

from unittest import mock

from iris.coords import CellMethod
from iris.fileformats.netcdf import parse_cell_methods


class Test(tests.IrisTest):
    def test_simple(self):
        cell_method_strings = [
            "time: mean",
            "time : mean",
        ]
        expected = (CellMethod(method="mean", coords="time"),)
        for cell_method_str in cell_method_strings:
            res = parse_cell_methods(cell_method_str)
            self.assertEqual(res, expected)

    def test_with_interval(self):
        cell_method_strings = [
            "time: variance (interval: 1 hr)",
            "time : variance (interval: 1 hr)",
        ]
        expected = (
            CellMethod(method="variance", coords="time", intervals="1 hr"),
        )
        for cell_method_str in cell_method_strings:
            res = parse_cell_methods(cell_method_str)
            self.assertEqual(res, expected)

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
        for cell_method_str in cell_method_strings:
            res = parse_cell_methods(cell_method_str)
            self.assertEqual(res, expected)

    def test_comment(self):
        cell_method_strings = [
            "time: maximum (interval: 1 hr comment: first bit) "
            "time: mean (interval: 1 day comment: second bit)",
            "time : maximum (interval: 1 hr comment: first bit) "
            "time: mean (interval: 1 day comment: second bit)",
            "time: maximum (interval: 1 hr comment: first bit) "
            "time : mean (interval: 1 day comment: second bit)",
            "time : maximum (interval: 1 hr comment: first bit) "
            "time : mean (interval: 1 day comment: second bit)",
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
        for cell_method_str in cell_method_strings:
            res = parse_cell_methods(cell_method_str)
            self.assertEqual(res, expected)

    def test_portions_of_cells(self):
        cell_method_strings = [
            "area: mean where sea_ice over sea",
            "area : mean where sea_ice over sea",
        ]
        expected = (
            CellMethod(method="mean where sea_ice over sea", coords="area"),
        )
        for cell_method_str in cell_method_strings:
            res = parse_cell_methods(cell_method_str)
            self.assertEqual(res, expected)

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
        for cell_method_str in cell_method_strings:
            res = parse_cell_methods(cell_method_str)
            self.assertEqual(res, expected)

    def test_climatology_with_unknown_method(self):
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
        for cell_method_str in cell_method_strings:
            with mock.patch("warnings.warn") as warn:
                res = parse_cell_methods(cell_method_str)
            self.assertIn(
                "NetCDF variable contains unknown cell method 'min'",
                warn.call_args[0][0],
            )
            self.assertEqual(res, expected)


if __name__ == "__main__":
    tests.main()
