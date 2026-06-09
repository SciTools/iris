# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.analysis.Nearest`."""

import pytest

from iris.analysis import Nearest


def create_scheme(mode=None):
    kwargs = {}
    if mode is not None:
        kwargs["extrapolation_mode"] = mode
    return Nearest(**kwargs)


class Test___init__:
    def test_invalid(self):
        with pytest.raises(ValueError, match="Extrapolation mode"):
            Nearest("bogus")


class Test_extrapolation_mode:
    def check_mode(self, mode):
        scheme = create_scheme(mode)
        assert scheme.extrapolation_mode == mode

    def test_default(self):
        scheme = Nearest()
        assert scheme.extrapolation_mode == "extrapolate"

    def test_extrapolate(self):
        self.check_mode("extrapolate")

    def test_nan(self):
        self.check_mode("nan")

    def test_error(self):
        self.check_mode("error")

    def test_mask(self):
        self.check_mode("mask")

    def test_nanmask(self):
        self.check_mode("nanmask")


class Test_interpolator:
    def check_mode(self, mocker, mode=None):
        scheme = create_scheme(mode)

        # Check that calling `scheme.interpolator(...)` returns an
        # instance of RectilinearInterpolator which has been created
        # using the correct arguments.
        ri = mocker.patch(
            "iris.analysis.RectilinearInterpolator",
            return_value=mocker.sentinel.interpolator,
        )
        interpolator = scheme.interpolator(mocker.sentinel.cube, mocker.sentinel.coords)
        if mode is None:
            expected_mode = "extrapolate"
        else:
            expected_mode = mode
        ri.assert_called_once_with(
            mocker.sentinel.cube, mocker.sentinel.coords, "nearest", expected_mode
        )
        assert interpolator is mocker.sentinel.interpolator

    def test_default(self, mocker):
        self.check_mode(mocker)

    def test_extrapolate(self, mocker):
        self.check_mode(mocker, "extrapolate")

    def test_nan(self, mocker):
        self.check_mode(mocker, "nan")

    def test_error(self, mocker):
        self.check_mode(mocker, "error")

    def test_mask(self, mocker):
        self.check_mode(mocker, "mask")

    def test_nanmask(self, mocker):
        self.check_mode(mocker, "nanmask")


class Test_regridder:
    def check_mode(self, mocker, mode=None):
        scheme = create_scheme(mode)

        # Ensure that calling the regridder results in an instance of
        # RectilinearRegridder being returned, which has been created with
        # the expected arguments.
        rr = mocker.patch(
            "iris.analysis.RectilinearRegridder",
            return_value=mocker.sentinel.regridder,
        )
        regridder = scheme.regridder(mocker.sentinel.src_grid, mocker.sentinel.tgt_grid)

        expected_mode = "extrapolate" if mode is None else mode
        rr.assert_called_once_with(
            mocker.sentinel.src_grid,
            mocker.sentinel.tgt_grid,
            "nearest",
            expected_mode,
        )
        assert regridder is mocker.sentinel.regridder

    def test_default(self, mocker):
        self.check_mode(mocker)

    def test_extrapolate(self, mocker):
        self.check_mode(mocker, "extrapolate")

    def test_nan(self, mocker):
        self.check_mode(mocker, "nan")

    def test_error(self, mocker):
        self.check_mode(mocker, "error")

    def test_mask(self, mocker):
        self.check_mode(mocker, "mask")

    def test_nanmask(self, mocker):
        self.check_mode(mocker, "nanmask")
