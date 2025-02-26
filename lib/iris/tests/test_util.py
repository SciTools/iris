# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test iris.util."""

import inspect
from io import StringIO

import cf_units
import numpy as np
import pytest

import iris.analysis
import iris.coords
from iris.tests import _shared_utils
import iris.tests.stock as stock
import iris.util


class TestMonotonic:
    def assert_monotonic(self, array, direction=None, **kwargs):
        if direction is not None:
            mono, dir = iris.util.monotonic(array, return_direction=True, **kwargs)
            if not mono:
                pytest.fail("Array was not monotonic:/n %r" % array)
            if dir != np.sign(direction):
                pytest.fail(
                    "Array was monotonic but not in the direction expected:"
                    "/n  + requested direction: %s/n  + resultant direction: %s"
                    % (direction, dir)
                )
        else:
            mono = iris.util.monotonic(array, **kwargs)
            if not mono:
                pytest.fail("Array was not monotonic:/n %r" % array)

    def assert_not_monotonic(self, array, **kwargs):
        mono = iris.util.monotonic(array, **kwargs)
        if mono:
            pytest.fail("Array was monotonic when it shouldn't be:/n %r" % array)

    def test_monotonic_pve(self):
        a = np.array([3, 4, 5.3])
        self.assert_monotonic(a)
        self.assert_monotonic(a, direction=1)

        # test the reverse for negative monotonic.
        a = a[::-1]
        self.assert_monotonic(a)
        self.assert_monotonic(a, direction=-1)

    def test_not_monotonic(self):
        b = np.array([3, 5.3, 4])
        self.assert_not_monotonic(b)

    def test_monotonic_strict(self):
        b = np.array([3, 5.3, 4])
        self.assert_not_monotonic(b, strict=True)
        self.assert_not_monotonic(b)

        b = np.array([3, 5.3, 5.3])
        self.assert_not_monotonic(b, strict=True)
        self.assert_monotonic(b, direction=1)

        b = b[::-1]
        self.assert_not_monotonic(b, strict=True)
        self.assert_monotonic(b, direction=-1)

        b = np.array([0.0])
        pytest.raises(ValueError, iris.util.monotonic, b)
        pytest.raises(ValueError, iris.util.monotonic, b, strict=True)

        b = np.array([0.0, 0.0])
        self.assert_not_monotonic(b, strict=True)
        self.assert_monotonic(b)


class TestClipString:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.test_string = "Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
        self.rider = "**^^**$$..--__"  # A good chance at being unique and not in the string to be tested!

    def test_oversize_string(self):
        # Test with a clip length that means the string will be clipped

        clip_length = 109
        result = iris.util.clip_string(self.test_string, clip_length, self.rider)

        # Check the length is between what we requested ( + rider length) and the length of the original string
        fail_message = "String was not clipped."
        clip_rider_len = clip_length + len(self.rider)
        assert clip_rider_len <= len(result) < len(self.test_string), fail_message

        # Also test the rider was added
        fail_message = "Rider was not added to the string when it should have been."
        assert self.rider in result, fail_message

    def test_undersize_string(self):
        # Test with a clip length that is longer than the string

        clip_length = 10999
        result = iris.util.clip_string(self.test_string, clip_length, self.rider)
        fail_message = "String was clipped when it should not have been."
        assert len(result) == len(self.test_string), fail_message

        # Also test that no rider was added on the end if the string was not clipped
        fail_message = "Rider was adding to the string when it should not have been."
        assert self.rider not in result, fail_message

    def test_invalid_clip_lengths(self):
        # Clip values less than or equal to zero are not valid
        for clip_length in [0, -100]:
            result = iris.util.clip_string(self.test_string, clip_length, self.rider)
            fail_message = "String was clipped when it should not have been."
            assert len(result) == len(self.test_string), fail_message

    def test_default_values(self):
        # Get the default values specified in the function
        argspec = inspect.getfullargspec(iris.util.clip_string)
        arg_dict = dict(zip(argspec.args[-2:], argspec.defaults))

        result = iris.util.clip_string(
            self.test_string, arg_dict["clip_length"], arg_dict["rider"]
        )

        assert len(result) < len(self.test_string), "String was not clipped."

        rider_returned = result[-len(arg_dict["rider"]) :]
        assert rider_returned == arg_dict["rider"], "Default rider was not applied."

    def test_trim_string_with_no_spaces(self):
        clip_length = 200
        no_space_string = "a" * 500

        # Since this string has no spaces, clip_string will not be able to gracefully clip it
        # but will instead clip it exactly where the user specified
        result = iris.util.clip_string(no_space_string, clip_length, self.rider)

        expected_length = clip_length + len(self.rider)

        # Check the length of the returned string is equal to clip length + length of rider
        assert len(result) == expected_length, (
            "Mismatch in expected length of clipped string. Length was %s, "
            "expected value is %s" % (len(result), expected_length)
        )


@_shared_utils.skip_data
class TestDescribeDiff:
    @pytest.fixture(autouse=True)
    def _setup(self, request):
        self.request = request

    def test_identical(self):
        test_cube_a = stock.realistic_4d()
        test_cube_b = stock.realistic_4d()

        return_sio = StringIO()
        iris.util.describe_diff(test_cube_a, test_cube_b, output_file=return_sio)
        return_str = return_sio.getvalue()

        _shared_utils.assert_string(
            self.request, return_str, "compatible_cubes.str.txt"
        )

    def test_different(self):
        # test incompatible attributes
        test_cube_a = stock.realistic_4d()
        test_cube_b = stock.realistic_4d()

        test_cube_a.attributes["Conventions"] = "CF-1.5"
        test_cube_b.attributes["Conventions"] = "CF-1.6"

        return_sio = StringIO()
        iris.util.describe_diff(test_cube_a, test_cube_b, output_file=return_sio)
        return_str = return_sio.getvalue()

        _shared_utils.assert_string(
            self.request, return_str, "incompatible_attr.str.txt"
        )

        # test incompatible names
        test_cube_a = stock.realistic_4d()
        test_cube_b = stock.realistic_4d()

        test_cube_a.standard_name = "relative_humidity"

        return_sio = StringIO()
        iris.util.describe_diff(test_cube_a, test_cube_b, output_file=return_sio)
        return_str = return_sio.getvalue()

        _shared_utils.assert_string(
            self.request, return_str, "incompatible_name.str.txt"
        )

        # test incompatible unit
        test_cube_a = stock.realistic_4d()
        test_cube_b = stock.realistic_4d()

        test_cube_a.units = cf_units.Unit("m")

        return_sio = StringIO()
        iris.util.describe_diff(test_cube_a, test_cube_b, output_file=return_sio)
        return_str = return_sio.getvalue()

        _shared_utils.assert_string(
            self.request, return_str, "incompatible_unit.str.txt"
        )

        # test incompatible methods
        test_cube_a = stock.realistic_4d()
        test_cube_b = stock.realistic_4d().collapsed(
            "model_level_number", iris.analysis.MEAN
        )

        return_sio = StringIO()
        iris.util.describe_diff(test_cube_a, test_cube_b, output_file=return_sio)
        return_str = return_sio.getvalue()

        _shared_utils.assert_string(
            self.request, return_str, "incompatible_meth.str.txt"
        )

    def test_output_file(self, tmp_path):
        # test incompatible attributes
        test_cube_a = stock.realistic_4d()
        test_cube_b = stock.realistic_4d().collapsed(
            "model_level_number", iris.analysis.MEAN
        )

        test_cube_a.attributes["Conventions"] = "CF-1.5"
        test_cube_b.attributes["Conventions"] = "CF-1.6"
        test_cube_a.standard_name = "relative_humidity"
        test_cube_a.units = cf_units.Unit("m")

        filename = tmp_path / "tmp"
        with open(filename, "w") as f:
            iris.util.describe_diff(test_cube_a, test_cube_b, output_file=f)
            f.close()

        _shared_utils.assert_files_equal(filename, "incompatible_cubes.str.txt")
