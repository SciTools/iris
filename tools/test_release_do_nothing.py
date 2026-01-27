# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the ``release_do_nothing.py`` file."""

from typing import NamedTuple

import pytest

import nothing
from release_do_nothing import IrisRelease


@pytest.fixture(autouse=True)
def mock_fast_print(mocker) -> None:
    """Prevent the mod:`nothing` print methods from sleeping."""
    mocker.patch.object(nothing, "sleep", return_value=None)


@pytest.fixture(autouse=True)
def mock_git_commands(mocker) -> None:
    """Detach testing from reliance on .git directory."""
    mocker.patch.object(
        IrisRelease,
        "_git_remote_v",
        return_value="origin\nupstream\nfoo\n",
    )

    mocker.patch.object(
        IrisRelease,
        "_git_remote_get_url",
        return_value="git@github.com:foo/iris.git",
    )

    mocker.patch.object(
        IrisRelease,
        "_git_ls_remote_tags",
        # TODO: make this as minimal as possible while still enabling the tests.
        return_value=(
            "abcd1234        refs/tags/1.0.0\n"
            "abcd1235        refs/tags/1.0.1\n"
            "abcd1236        refs/tags/1.0.2\n"
            "abcd1237        refs/tags/1.1.0rc1\n"
            "abcd1238        refs/tags/1.1.0rc2\n"
            "abcd1239        refs/tags/1.1.0\n"
            "abcd1240        refs/tags/1.2.0rc0\n"
        ),
    )


def mock_input(mocker, input_str: str) -> None:
    """Mock :func:`input` to return a specific value."""
    mocker.patch("builtins.input", return_value=input_str)


class TestValidate:
    """Tests for the :func:`release_do_nothing.validate` function."""
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.instance = IrisRelease(
            _dry_run=True,
            latest_complete_step=IrisRelease.get_steps().index(IrisRelease.validate) - 1,
            github_user="user",
            patch_min_max_tag=("8.0.0", "9.0.0")
        )

    class Case(NamedTuple):
        git_tag: str
        match: str

    @pytest.fixture(params=[
        pytest.param(
            Case("9.1.dev0", "development release.*cannot handle"),
            id="dev release",
        ),
        pytest.param(
            Case("9.1.post0", "post release.*cannot handle"),
            id="post release",
        ),
        pytest.param(
            Case("9.1.alpha0", "release candidate.*got 'a'"),
            id="pre-release non-rc",
        ),
        pytest.param(
            Case("9.1.1rc0", "PATCH release AND a release candidate.*cannot handle"),
            id="patch release rc",
        ),
        pytest.param(
            Case("9.1.1", "No previous releases.*cannot handle a PATCH"),
            id="first in series patch",
        ),
    ])
    def unhandled_cases(self, request) -> Case:
        case = request.param
        self.instance.git_tag = case.git_tag
        return case

    def test_unhandled_cases(self, unhandled_cases):
        case = unhandled_cases
        with pytest.raises(RuntimeError, match=case.match):
            self.instance.validate()
        pass

    @pytest.fixture
    def first_in_series_not_rc(self) -> None:
        self.instance.git_tag = "9.1.0"

    def test_first_in_series_not_rc_message(self, first_in_series_not_rc, capfd, mocker):
        mock_input(mocker, "y")
        self.instance.validate()
        out, err = capfd.readouterr()
        assert "No previous releases" in out
        assert "expected to be a release candidate" in out
        assert "sure you want to continue" in out

    def test_first_in_series_not_rc_exit(self, first_in_series_not_rc, mocker):
        mock_input(mocker, "n")
        with pytest.raises(SystemExit):
            self.instance.validate()

    def test_first_in_series_not_rc_continue(self, first_in_series_not_rc, mocker):
        mock_input(mocker, "y")
        self.instance.validate()

    # Not an exhaustive list, just the inverse of the unhandled cases.
    @pytest.fixture(params=[
        pytest.param("9.0.0rc0", id="major release RC"),
        pytest.param("9.1.0rc0", id="minor release RC"),
        pytest.param("1.2.0", id="minor release existing series"),
        pytest.param("1.1.1", id="patch release existing series"),
        pytest.param("9.1.0", id="first in series not RC"),
    ])
    def handled_cases(self, request) -> None:
        self.instance.git_tag = request.param

    def test_handled_cases(self, handled_cases, mocker):
        message = "Confirm that the details above are correct"
        mock_input(mocker, "y")
        mocked = mocker.patch.object(IrisRelease, "wait_for_done")
        self.instance.validate()
        mocked.assert_called_once()
        assert message in mocked.call_args[0][0]
