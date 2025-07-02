# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.fileformats._nc_load_rules.helpers._add_or_capture`."""

from unittest.mock import MagicMock

import pytest

from iris.cube import Cube
from iris.fileformats._nc_load_rules import helpers
from iris.fileformats.cf import CFVariable
from iris.loading import LOAD_PROBLEMS, LoadProblems


class Mixin:
    build_func: MagicMock
    add_method: MagicMock
    cf_var: MagicMock

    filename: str = "test__add_or_capture.nc"
    attr_key: str = "attr_key"
    attr_value: str = "attr_value"
    destination: LoadProblems.Problem.Destination = LoadProblems.Problem.Destination(
        Cube, "foo"
    )

    @pytest.fixture
    def make_args(self, mocker):
        self.build_func = mocker.MagicMock()
        self.build_func.return_value = "BUILT"
        self.add_method = mocker.MagicMock()
        self.cf_var = mocker.MagicMock(spec=CFVariable)
        self.cf_var.filename = self.filename
        setattr(self.cf_var, self.attr_key, self.attr_value)

    def call(
        self,
        filename=None,
        attr_key=None,
    ):
        if filename is not None:
            self.cf_var.filename = filename

        result = helpers._add_or_capture(
            build_func=self.build_func,
            add_method=self.add_method,
            cf_var=self.cf_var,
            destination=self.destination,
            attr_key=attr_key,
        )
        return result


class TestBuildProblems(Mixin):
    @pytest.fixture(autouse=True)
    def _setup(self, make_args):
        LOAD_PROBLEMS.reset()
        self.failure_string = "FAILED: BUILD"
        self.build_func.side_effect = ValueError(self.failure_string)

    @pytest.fixture
    def patch_build_raw_cube(self, mocker):
        patch = mocker.patch.object(helpers, "build_raw_cube", return_value="RAW_CUBE")
        yield patch

    @pytest.fixture
    def cause_build_raw_cube_error(self, patch_build_raw_cube):
        patch_build_raw_cube.side_effect = ValueError("FAILED")
        yield
        patch_build_raw_cube.side_effect = None

    def common_test(self, attr_key, expected_loaded):
        result = self.call(attr_key=attr_key)
        self.build_func.assert_called_once()

        assert isinstance(result, LoadProblems.Problem)
        assert result.filename == self.filename
        assert result.loaded == expected_loaded
        assert str(result.stack_trace) == self.failure_string
        assert result.destination is self.destination
        assert result is LOAD_PROBLEMS.problems[-1]

    def test_w_o_attr_can_build(self, patch_build_raw_cube):
        self.common_test(
            attr_key=None,
            expected_loaded=patch_build_raw_cube.return_value,
        )

    def test_w_o_attr_cannot_build(self, cause_build_raw_cube_error):
        self.common_test(
            attr_key=None,
            expected_loaded=None,
        )

    def test_w_attr_can_find(self):
        self.common_test(
            attr_key=self.attr_key, expected_loaded={self.attr_key: self.attr_value}
        )

    def test_w_attr_cannot_find(self):
        self.common_test(
            attr_key="standard_name",
            expected_loaded={"standard_name": None},
        )

    def test_multiple_problems_same_file(self):
        results = [self.call() for _ in range(3)]
        for ix, problem in enumerate(LOAD_PROBLEMS.problems):
            assert problem.filename == self.filename
            assert problem is results[ix]

    def test_multiple_problems_diff_file(self):
        names = [f"test__add_or_capture_{ix}.nc" for ix in range(3)]
        results = [self.call(filename=name) for name in names]
        problems_by_file = LOAD_PROBLEMS.problems_by_file
        for ix, (problem_file, problems) in enumerate(problems_by_file.items()):
            assert problem_file == names[ix]
            for jx, problem in enumerate(problems):
                assert problem is results[ix]


class TestAddProblems(Mixin):
    @pytest.fixture(autouse=True)
    def _setup(self, make_args):
        LOAD_PROBLEMS.reset()
        self.failure_string = "FAILED: ADD"
        self.add_method.side_effect = ValueError(self.failure_string)

    @pytest.mark.parametrize(
        "attr_key", [None, Mixin.attr_key], ids=["w_o_attr", "w_attr"]
    )
    def test_standard(self, attr_key):
        result = self.call(attr_key=attr_key)
        self.build_func.assert_called_once()
        self.add_method.assert_called_once_with(self.build_func.return_value)
        built = self.build_func.return_value
        if attr_key is None:
            expected_loaded = built
        else:
            expected_loaded = {attr_key: built}

        assert isinstance(result, LoadProblems.Problem)
        assert result.filename == self.filename
        assert result.loaded == expected_loaded
        assert str(result.stack_trace) == self.failure_string
        assert result is LOAD_PROBLEMS.problems[-1]

    def test_multiple_problems_same_file(self):
        results = [self.call() for _ in range(3)]
        for ix, problem in enumerate(LOAD_PROBLEMS.problems):
            assert problem.filename == self.filename
            assert problem is results[ix]

    def test_multiple_problems_diff_file(self):
        names = [f"test__add_or_capture_{ix}.nc" for ix in range(3)]
        results = [self.call(filename=name) for name in names]
        problems_by_file = LOAD_PROBLEMS.problems_by_file
        for ix, (problem_file, problems) in enumerate(problems_by_file.items()):
            assert problem_file == names[ix]
            for jx, problem in enumerate(problems):
                assert problem is results[ix]


class TestSuccess(Mixin):
    @pytest.fixture(autouse=True)
    def _setup(self, make_args):
        LOAD_PROBLEMS.reset()

    @pytest.mark.parametrize(
        "attr_key", [None, Mixin.attr_key], ids=["w_o_attr", "w_attr"]
    )
    def test(self, attr_key):
        result = self.call(attr_key=attr_key)
        self.build_func.assert_called_once()
        self.add_method.assert_called_once_with(self.build_func.return_value)
        assert LOAD_PROBLEMS.problems == []
        assert result is None
