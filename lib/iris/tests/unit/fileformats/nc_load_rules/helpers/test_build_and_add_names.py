# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.fileformats._nc_load_rules.helpers.build_and_add_names`."""

from unittest import mock

import numpy as np
import pytest

from iris.cube import Cube
from iris.fileformats._nc_load_rules.helpers import build_and_add_names
from iris.loading import LOAD_PROBLEMS


@pytest.fixture
def mock_engine():
    global_attributes = {
        "Conventions": "CF-1.5",
        "comment": "Mocked test object",
    }
    cf_group = mock.Mock(global_attributes=global_attributes)
    cf_var = mock.MagicMock(
        cf_name="wibble",
        standard_name=None,
        long_name=None,
        units="m",
        dtype=np.float64,
        cell_methods=None,
        cf_group=cf_group,
    )
    engine = mock.Mock(cube=Cube([23]), cf_var=cf_var, filename="foo.nc")
    yield engine


class TestCubeName:
    cf_name: str

    @pytest.fixture(autouse=True)
    def _setup(self, mock_engine):
        LOAD_PROBLEMS.reset()
        self.engine = mock_engine

    def check_cube_names(self, inputs, expected):
        # Inputs - attributes on the fake CF Variable.
        standard_name, long_name = inputs
        # Expected - The expected cube attributes.
        exp_standard_name, exp_long_name = expected

        self.engine.cf_var.standard_name = standard_name
        self.engine.cf_var.long_name = long_name
        # engine = _make_engine(standard_name=standard_name, long_name=long_name)
        build_and_add_names(self.engine)
        self.cf_name = self.engine.cf_var.cf_name

        # Check the cube's standard name and long name are as expected.
        assert self.engine.cube.standard_name == exp_standard_name
        assert self.engine.cube.long_name == exp_long_name

    def check_load_problems(self, invalid_standard_name=None):
        if invalid_standard_name is None:
            assert LOAD_PROBLEMS.problems == []
        else:
            load_problem = LOAD_PROBLEMS.problems[-1]
            assert load_problem.loaded == {"standard_name": invalid_standard_name}
            assert load_problem.handled

            destination = load_problem.destination
            assert destination.iris_class == Cube
            assert destination.identifier == self.cf_name

    def test_standard_name_none_long_name_none(self):
        inputs = (None, None)
        expected = (None, None)
        self.check_cube_names(inputs, expected)
        self.check_load_problems()

    def test_standard_name_none_long_name_set(self):
        inputs = (None, "ice_thickness_long_name")
        expected = (None, "ice_thickness_long_name")
        self.check_cube_names(inputs, expected)
        self.check_load_problems()

    def test_standard_name_valid_long_name_none(self):
        inputs = ("sea_ice_thickness", None)
        expected = ("sea_ice_thickness", None)
        self.check_cube_names(inputs, expected)
        self.check_load_problems()

    def test_standard_name_valid_long_name_set(self):
        inputs = ("sea_ice_thickness", "ice_thickness_long_name")
        expected = ("sea_ice_thickness", "ice_thickness_long_name")
        self.check_cube_names(inputs, expected)
        self.check_load_problems()

    def test_standard_name_invalid_long_name_none(self):
        inputs = ("not_a_standard_name", None)
        expected = (
            None,
            "not_a_standard_name",
        )
        self.check_cube_names(inputs, expected)
        self.check_load_problems("not_a_standard_name")

    def test_standard_name_invalid_long_name_set(self):
        inputs = ("not_a_standard_name", "ice_thickness_long_name")
        expected = (None, "ice_thickness_long_name")
        self.check_cube_names(inputs, expected)
        self.check_load_problems("not_a_standard_name")
