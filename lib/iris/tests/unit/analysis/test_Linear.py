# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for :class:`iris.analysis.Linear`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

from iris.analysis import Linear


def create_scheme(mode=None):
    kwargs = {}
    if mode is not None:
        kwargs["extrapolation_mode"] = mode
    return Linear(**kwargs)


class Test_extrapolation_mode(tests.IrisTest):
    def check_mode(self, mode):
        linear = create_scheme(mode)
        self.assertEqual(linear.extrapolation_mode, mode)

    def test_default(self):
        linear = Linear()
        self.assertEqual(linear.extrapolation_mode, "linear")

    def test_extrapolate(self):
        self.check_mode("extrapolate")

    def test_linear(self):
        self.check_mode("linear")

    def test_nan(self):
        self.check_mode("nan")

    def test_error(self):
        self.check_mode("error")

    def test_mask(self):
        self.check_mode("mask")

    def test_nanmask(self):
        self.check_mode("nanmask")

    def test_invalid(self):
        with self.assertRaisesRegex(ValueError, "Extrapolation mode"):
            Linear("bogus")


class Test_interpolator(tests.IrisTest):
    def check_mode(self, mode=None):
        linear = create_scheme(mode)

        # Check that calling `linear.interpolator(...)` returns an
        # instance of RectilinearInterpolator which has been created
        # using the correct arguments.
        with mock.patch(
            "iris.analysis.RectilinearInterpolator",
            return_value=mock.sentinel.interpolator,
        ) as ri:
            interpolator = linear.interpolator(
                mock.sentinel.cube, mock.sentinel.coords
            )
        if mode is None or mode == "linear":
            expected_mode = "extrapolate"
        else:
            expected_mode = mode
        ri.assert_called_once_with(
            mock.sentinel.cube, mock.sentinel.coords, "linear", expected_mode
        )
        self.assertIs(interpolator, mock.sentinel.interpolator)

    def test_default(self):
        self.check_mode()

    def test_extrapolate(self):
        self.check_mode("extrapolate")

    def test_linear(self):
        self.check_mode("linear")

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
        linear = create_scheme(mode)

        # Check that calling `linear.regridder(...)` returns an instance
        # of RectilinearRegridder which has been created using the correct
        # arguments.
        with mock.patch(
            "iris.analysis.RectilinearRegridder",
            return_value=mock.sentinel.regridder,
        ) as lr:
            regridder = linear.regridder(
                mock.sentinel.src, mock.sentinel.target
            )
        if mode is None or mode == "linear":
            expected_mode = "extrapolate"
        else:
            expected_mode = mode
        lr.assert_called_once_with(
            mock.sentinel.src, mock.sentinel.target, "linear", expected_mode
        )
        self.assertIs(regridder, mock.sentinel.regridder)

    def test_default(self):
        self.check_mode()

    def test_extrapolate(self):
        self.check_mode("extrapolate")

    def test_linear(self):
        self.check_mode("linear")

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
