# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for :class:`iris.analysis.Nearest`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

from iris.analysis import Nearest


def create_scheme(mode=None):
    kwargs = {}
    if mode is not None:
        kwargs["extrapolation_mode"] = mode
    return Nearest(**kwargs)


class Test___init__(tests.IrisTest):
    def test_invalid(self):
        with self.assertRaisesRegex(ValueError, "Extrapolation mode"):
            Nearest("bogus")


class Test_extrapolation_mode(tests.IrisTest):
    def check_mode(self, mode):
        scheme = create_scheme(mode)
        self.assertEqual(scheme.extrapolation_mode, mode)

    def test_default(self):
        scheme = Nearest()
        self.assertEqual(scheme.extrapolation_mode, "extrapolate")

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


class Test_interpolator(tests.IrisTest):
    def check_mode(self, mode=None):
        scheme = create_scheme(mode)

        # Check that calling `scheme.interpolator(...)` returns an
        # instance of RectilinearInterpolator which has been created
        # using the correct arguments.
        with mock.patch(
            "iris.analysis.RectilinearInterpolator",
            return_value=mock.sentinel.interpolator,
        ) as ri:
            interpolator = scheme.interpolator(
                mock.sentinel.cube, mock.sentinel.coords
            )
        if mode is None:
            expected_mode = "extrapolate"
        else:
            expected_mode = mode
        ri.assert_called_once_with(
            mock.sentinel.cube, mock.sentinel.coords, "nearest", expected_mode
        )
        self.assertIs(interpolator, mock.sentinel.interpolator)

    def test_default(self):
        self.check_mode()

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


class Test_regridder(tests.IrisTest):
    def check_mode(self, mode=None):
        scheme = create_scheme(mode)

        # Ensure that calling the regridder results in an instance of
        # RectilinearRegridder being returned, which has been created with
        # the expected arguments.
        with mock.patch(
            "iris.analysis.RectilinearRegridder",
            return_value=mock.sentinel.regridder,
        ) as rr:
            regridder = scheme.regridder(
                mock.sentinel.src_grid, mock.sentinel.tgt_grid
            )

        expected_mode = "extrapolate" if mode is None else mode
        rr.assert_called_once_with(
            mock.sentinel.src_grid,
            mock.sentinel.tgt_grid,
            "nearest",
            expected_mode,
        )
        self.assertIs(regridder, mock.sentinel.regridder)

    def test_default(self):
        self.check_mode()

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


if __name__ == "__main__":
    tests.main()
