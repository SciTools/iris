# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.util.describe_diff`."""

from io import StringIO

import numpy as np
import pytest

import iris.cube
from iris.tests import _shared_utils
from iris.util import describe_diff


class Test:
    @pytest.fixture(autouse=True)
    def _setup(self, request):
        self.request = request
        self.cube_a = iris.cube.Cube([])
        self.cube_b = self.cube_a.copy()

    def _compare_result(self, cube_a, cube_b):
        result_sio = StringIO()
        describe_diff(cube_a, cube_b, output_file=result_sio)
        return result_sio.getvalue()

    def test_noncommon_array_attributes(self):
        # test non-common array attribute
        self.cube_a.attributes["test_array"] = np.array([1, 2, 3])
        return_str = self._compare_result(self.cube_a, self.cube_b)
        _shared_utils.assert_string(
            self.request, return_str, ["compatible_cubes.str.txt"]
        )

    def test_same_array_attributes(self):
        # test matching array attribute
        self.cube_a.attributes["test_array"] = np.array([1, 2, 3])
        self.cube_b.attributes["test_array"] = np.array([1, 2, 3])
        return_str = self._compare_result(self.cube_a, self.cube_b)
        _shared_utils.assert_string(
            self.request, return_str, ["compatible_cubes.str.txt"]
        )

    def test_different_array_attributes(self):
        # test non-matching array attribute
        self.cube_a.attributes["test_array"] = np.array([1, 2, 3])
        self.cube_b.attributes["test_array"] = np.array([1, 7, 3])
        return_str = self._compare_result(self.cube_a, self.cube_b)
        _shared_utils.assert_string(
            self.request,
            return_str,
            [
                "unit",
                "util",
                "describe_diff",
                "incompatible_array_attrs.str.txt",
            ],
        )
