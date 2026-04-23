# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the ``release_do_nothing.py`` file."""
import enum
from datetime import datetime
from pathlib import Path
import re
from typing import Any, NamedTuple

import pytest
from pytest_mock import MockType

import nothing
from release_do_nothing import IrisRelease, IrisVersion


@pytest.fixture(autouse=True)
def mock_fast_print(mocker) -> None:
    """Prevent the mod:`nothing` print methods from sleeping."""
    mocker.patch.object(nothing, "sleep", return_value=None)


@pytest.fixture(autouse=True)
def mock_git_remote_v(mocker) -> MockType:
    """Mock :meth:`IrisRelease._git_remote_v`.

    Assumes return_value will be overridden by any calling test (the default
    empty string will always error downstream).
    """
    return mocker.patch.object(
        IrisRelease,
        "_git_remote_v",
        return_value="",
    )


@pytest.fixture(autouse=True)
def mock_git_remote_get_url(mocker) -> MockType:
    """Mock :meth:`IrisRelease._git_remote_get_url`.

    Assumes return_value will be overridden by any calling test (the default
    empty string will always error downstream).
    """
    return mocker.patch.object(
        IrisRelease,
        "_git_remote_get_url",
        return_value="",
    )


@pytest.fixture(autouse=True)
def mock_git_ls_remote_tags(mocker) -> MockType:
    """Mock :meth:`IrisRelease._git_ls_remote_tags`.

    Assumes return_value will be overridden by any calling test (the default
    empty string will always error downstream).
    """
    return mocker.patch.object(
        IrisRelease,
        "_git_ls_remote_tags",
        return_value="",
    )


@pytest.fixture
def mock_wait_for_done(mocker) -> MockType:
    """Mock :meth:`IrisRelease.wait_for_done` to not wait, and to count calls."""
    return mocker.patch.object(IrisRelease, "wait_for_done", return_value=None)


@pytest.fixture
def mock_report_problem(mocker) -> MockType:
    return mocker.patch.object(IrisRelease, "report_problem")


def mock_inputs(mocker, *inputs: str) -> None:
    """Mock :func:`input` to return chosen values, specified in a sequence."""
    mocker.patch("builtins.input", side_effect=inputs)


def assert_input_msg_regex(call: Any, expected: re.Pattern[str] | str) -> None:
    # TODO: use this for testing ALL messages that include dynamic content?
    if isinstance(expected, str):
        expected = re.compile(expected, re.DOTALL)
    assert hasattr(call, "args") and len(call.args) > 0
    message = call.args[0]
    assert isinstance(message, str)
    assert expected.search(message) is not None, f"Expected message matching {expected!r} in {message!r}"


class TestIrisVersion:
    """Tests for the :class:`IrisVersion` class."""
    @pytest.fixture(params=["9.0.0", "9.0.1", "9.1.0"], autouse=True)
    def _setup(self, request):
        self.version = IrisVersion(request.param)
        self.input_str = request.param

    def test_str(self):
        expecteds = {"9.0.0": "v9.0.0", "9.0.1": "v9.0.1", "9.1.0": "v9.1.0"}
        assert str(self.version) == expecteds[self.input_str]

    def test_series(self):
        expecteds = {"9.0.0": "v9.0", "9.0.1": "v9.0", "9.1.0": "v9.1"}
        assert self.version.series == expecteds[self.input_str]

    def test_branch(self):
        expecteds = {"9.0.0": "v9.0.x", "9.0.1": "v9.0.x", "9.1.0": "v9.1.x"}
        assert self.version.branch == expecteds[self.input_str]


class TestProperties:
    """Tests for the properties of the :class:`IrisRelease` class."""
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.instance = IrisRelease(
            _dry_run=True,
            latest_complete_step=len(IrisRelease.get_steps()) - 1,
            github_scitools="foo",
            github_fork="bar",
            github_user="user",
            patch_min_max_tag=("8.0.0", "9.0.0"),
            git_tag="9.1.1",
            sha256="abcd1234",
        )

    def test_version(self):
        assert self.instance.version == IrisVersion("9.1.1")

    @pytest.mark.parametrize("git_tag", ["1.1.0", "1.1.2"])
    def test_is_latest_tag(self, git_tag, mock_git_ls_remote_tags):
        mock_git_ls_remote_tags.return_value = (
            "abcd1234 refs/tags/1.1.1\n"
        )
        expecteds = {"1.1.0": False, "1.1.2": True}
        expected = expecteds[git_tag]
        self.instance.git_tag = git_tag
        assert self.instance.is_latest_tag is expected

    @pytest.mark.parametrize("git_tag", ["9.0.0", "9.1.0", "9.1.1"])
    def test_release_type(self, git_tag):
        expecteds = {
            "9.0.0": IrisRelease.ReleaseTypes.MAJOR,
            "9.1.0": IrisRelease.ReleaseTypes.MINOR,
            "9.1.1": IrisRelease.ReleaseTypes.PATCH,
        }
        expected = expecteds[git_tag]
        self.instance.git_tag = git_tag
        assert self.instance.release_type is expected

    @pytest.mark.parametrize("git_tag", ["9.1.0rc1", "9.1.0"])
    def test_is_release_candidate(self, git_tag):
        expecteds = {"9.1.0rc1": True, "9.1.0": False}
        expected = expecteds[git_tag]
        self.instance.git_tag = git_tag
        assert self.instance.is_release_candidate is expected

    @pytest.mark.parametrize("git_tag", ["1.0.0", "1.0.1", "1.1.0"])
    def test_first_in_series(self, git_tag, mock_git_ls_remote_tags):
        mock_git_ls_remote_tags.return_value = (
            "abcd1234 refs/tags/1.0.0\n"
            "abcd1235 refs/tags/1.0.1\n"
        )
        expecteds = {"1.0.0": False, "1.0.1": False, "1.1.0": True}
        expected = expecteds[git_tag]
        self.instance.git_tag = git_tag
        assert self.instance.first_in_series is expected

    def test_patch_min_max(self):
        assert self.instance.patch_min_max == (
            IrisVersion("8.0.0"),
            IrisVersion("9.0.0"),
        )
        with pytest.raises(AssertionError, match="^$"):
            self.instance.patch_min_max_tag = ("9.0.0",)
            _ = self.instance.patch_min_max

    @pytest.mark.parametrize("git_tag", ["8.1.0", "8.1.1", "9.0.1", "9.1.1"])
    def test_more_patches_after_this_one(self, git_tag):
        expecteds = {
            "8.1.0": False,     # Not a PATCH release.
            "8.1.1": True,      # 9.0.0 still to patch.
            "9.0.1": False,     # Last PATCH in series.
            "9.1.1": False,     # Beyond max series.
        }
        expected = expecteds[git_tag]
        self.instance.git_tag = git_tag
        assert self.instance.more_patches_after_this_one is expected


    def test_whats_news(self):
        whatsnew_dir = Path(__file__).parents[1] / "docs" / "src" / "whatsnew"
        expected = IrisRelease.WhatsNewRsts(
            latest=whatsnew_dir / "latest.rst",
            release=whatsnew_dir / "9.1.rst",
            index_=whatsnew_dir / "index.rst",
            template=whatsnew_dir / "latest.rst.template",
        )
        assert self.instance.whats_news == expected


class TestAnalyseRemotes:
    """Tests for the :meth:`IrisRelease.analyse_remotes` method."""
    @pytest.fixture(autouse=True)
    def _setup(self, mock_git_remote_get_url, mock_git_remote_v) -> None:
        self.instance = IrisRelease(_dry_run=True)
        mock_git_remote_get_url.return_value = "git@github.com:myself/iris.git"
        mock_git_remote_v.return_value = (
            "origin    git@github.com:myself/iris.git (fetch)\n"
            "origin    git@github.com:myself/iris.git (push)\n"
            "upstream  git@github.com:SciTools/iris.git (fetch)\n"
            "upstream  no_push (push)\n"
            "foo       git@github.com:foo/iris.git (fetch)\n"
            "foo       git@github.com:foo/iris.git (push)\n"
        )

    def test_github_scitools(self, mocker):
        # The input is irrelevant to this test, we just need a valid input to
        #  get past that line so we can test the line that sets github_scitools.
        mock_inputs(mocker, "0")
        self.instance.analyse_remotes()
        assert self.instance.github_scitools == "upstream"

    def test_no_forks(self, mock_git_remote_v):
        # The only remote is 'upstream', so error.
        #  (Also confirms that upstream has been successfully ignored).
        mock_git_remote_v.return_value = (
            "upstream  git@github.com:SciTools/iris.git (fetch)\n"
            "upstream  no_push (push)\n"
        )
        with pytest.raises(AssertionError, match="^$"):
            self.instance.analyse_remotes()

    def test_choose_fork(self, mocker):
        # Developer chooses a fork other than `myself`.
        mock_inputs(mocker, "1")
        self.instance.analyse_remotes()
        assert self.instance.github_fork == "foo"

    def test_choose_fork_invalid(self, mocker, mock_report_problem):
        # Mock an invalid input followed by a valid one.
        mock_inputs(mocker, "99", "1")
        self.instance.analyse_remotes()
        mock_report_problem.assert_called_once_with(
            "Invalid number. Please try again ..."
        )

    def test_derive_username(self, mocker):
        mock_inputs(mocker, "0")
        self.instance.analyse_remotes()
        assert self.instance.github_user == "myself"

    def test_error_deriving_username(self, mocker, mock_git_remote_get_url):
        mock_git_remote_get_url.return_value = "bad_url"
        mock_inputs(mocker, "0")
        with pytest.raises(RuntimeError, match="Error deriving GitHub username"):
            self.instance.analyse_remotes()

    def test_default_fork_preserved(self, mocker):
        self.instance.github_fork = "bar"
        mock_inputs(mocker, "")
        self.instance.analyse_remotes()
        assert self.instance.github_fork == "bar"


class TestGetReleaseTag:
    """Tests for the :meth:`IrisRelease.get_release_tag` method."""
    @pytest.fixture(autouse=True)
    def _setup(self, mock_git_ls_remote_tags) -> None:
        self.instance = IrisRelease(_dry_run=True)
        mock_git_ls_remote_tags.return_value = "abcd1234 refs/tags/1.0.0"

    def test_valid_tag(self, mocker):
        # User inputs a valid, non-existing tag
        mock_inputs(mocker, "v1.1.0")
        self.instance.get_release_tag()
        assert self.instance.git_tag == "v1.1.0"

    def test_existing_tag(self, mocker, mock_report_problem):
        # User tries an existing tag, then provides a valid one
        mock_inputs(mocker, "v1.0.0", "v1.1.0")
        self.instance.get_release_tag()
        mock_report_problem.assert_called_once_with(
            "Version v1.0.0 already exists as a git tag. Please try again ..."
        )
        assert self.instance.git_tag == "v1.1.0"

    def test_invalid_version_format(self, mocker, mock_report_problem):
        # User inputs invalid version format, then valid one
        mock_inputs(mocker, "not-a-version", "v1.1.0")
        self.instance.get_release_tag()
        assert mock_report_problem.call_count == 1
        (call,) = mock_report_problem.call_args_list
        (message,) = call.args
        assert "Packaging error" in message
        assert "Please try again" in message
        assert self.instance.git_tag == "v1.1.0"

    def test_default_value_preserved(self, mocker):
        # When loading from saved state, existing git_tag should be offered as default
        self.instance.git_tag = "v1.1.0"
        mock_inputs(mocker, "")  # User accepts default
        self.instance.get_release_tag()
        assert self.instance.git_tag == "v1.1.0"


class TestGetAllPatches:
    """Tests for the :meth:`IrisRelease.get_all_patches` method."""
    @pytest.fixture(autouse=True)
    def _setup(self, mock_git_ls_remote_tags) -> None:
        self.instance = IrisRelease(
            _dry_run=True,
            git_tag="v1.1.1",
        )
        mock_git_ls_remote_tags.return_value = (
            "abcd1234 refs/tags/v1.0.0\n"
            "abcd1235 refs/tags/v1.0.1\n"
            "abcd1237 refs/tags/v1.1.0rc1\n"
            "abcd1239 refs/tags/v1.1.0\n"
            "abcd1240 refs/tags/v1.2.0\n"
        )

    def test_not_patch_release(self):
        # Non-PATCH releases skip this step
        self.instance.git_tag = "v1.3.0"
        self.instance.get_all_patches()
        assert self.instance.patch_min_max_tag is None

    def test_patch_single_series(self, mocker):
        # PATCH release, user doesn't want to patch multiple series
        mock_inputs(mocker, "1,1")
        self.instance.get_all_patches()
        assert self.instance.patch_min_max_tag == ("v1.1.1", "v1.1.1")

    def test_patch_multiple_series(self, mocker):
        # User selects a range of series to patch
        mock_inputs(mocker, "1,2")
        self.instance.get_all_patches()
        assert self.instance.patch_min_max_tag == ("v1.1.1", "v1.2.1")
        assert self.instance.git_tag == "v1.1.1"

    def test_invalid_format(self, mocker, mock_report_problem):
        # User inputs invalid format, then valid input
        mock_inputs(mocker, "not-numbers", "1,2")
        self.instance.get_all_patches()
        mock_report_problem.assert_called_once_with(
            "Invalid input, expected two integers comma-separated. "
            "Please try again ..."
        )
        assert self.instance.patch_min_max_tag == ("v1.1.1", "v1.2.1")
        assert self.instance.git_tag == "v1.1.1"

    def test_invalid_numbers(self, mocker, mock_report_problem):
        # User inputs out-of-range numbers, then valid input
        mock_inputs(mocker, "99,100", "1,2")
        self.instance.get_all_patches()
        mock_report_problem.assert_called_once_with(
            "Invalid numbers. Please try again ..."
        )
        assert self.instance.patch_min_max_tag == ("v1.1.1", "v1.2.1")
        assert self.instance.git_tag == "v1.1.1"

    def test_starts_with_earlier_patch(self, mocker, capfd):
        # When patch_min is earlier than current git_tag, git_tag is updated
        mock_inputs(mocker, "0,2")
        self.instance.get_all_patches()
        out, err = capfd.readouterr()
        assert "Starting with v1.0.2. (v1.1.1 will be covered in sequence)" in out
        assert self.instance.git_tag == "v1.0.2"
        assert self.instance.patch_min_max_tag == ("v1.0.2", "v1.2.1")

    def test_default_value_preserved(self, mocker):
        # When loading from saved state, existing patch_min_max_tag should work
        self.instance.patch_min_max_tag = ("v1.0.2", "v1.2.1")
        mock_inputs(mocker, "")
        self.instance.get_all_patches()
        assert self.instance.patch_min_max_tag == ("v1.0.2", "v1.2.1")


class TestApplyPatches:
    """Tests for the :meth:`IrisRelease.apply_patches` method."""
    @pytest.fixture(autouse=True)
    def _setup(self, mock_wait_for_done) -> None:
        self.instance = IrisRelease(
            _dry_run=True,
            git_tag="v1.1.1",
        )
        self.mock_wait_for_done = mock_wait_for_done

    def get_wait_for_done_call(self) -> Any:
        self.mock_wait_for_done.assert_called_once()
        (call,) = self.mock_wait_for_done.call_args_list
        return call

    def test_not_patch_release(self):
        # Non-PATCH releases skip this step entirely.
        self.instance.git_tag = "v1.2.0"
        self.instance.apply_patches()
        self.mock_wait_for_done.assert_not_called()

    def test_patch_branch_is_release_branch(self, mocker):
        # User inputs the ideal branch - message confirms it is optimal.
        mock_inputs(mocker, self.instance.version.branch)
        self.instance.apply_patches()
        call = self.get_wait_for_done_call()
        branch = re.escape(self.instance.version.branch)
        assert_input_msg_regex(
            call, rf"patch change\(s\) are on the ideal branch.*{branch}.*"
        )

    def test_patch_branch_empty(self, mocker):
        # User inputs nothing - message instructs them to create a PR.
        mock_inputs(mocker, "")
        self.instance.apply_patches()
        call = self.get_wait_for_done_call()
        branch = re.escape(self.instance.version.branch)
        assert_input_msg_regex(
            call, rf"Propose the patch change\(s\).*{branch}.*"
        )

    def test_patch_branch_other(self, mocker):
        # User inputs a different branch - message warns about cherry-pick conflicts.
        mock_inputs(mocker, "some-other-branch")
        self.instance.apply_patches()
        call = self.get_wait_for_done_call()
        branch = re.escape(self.instance.version.branch)
        assert_input_msg_regex(
            call,
            rf"cherry-picking the patch change\(s\).*some-other-branch.*{branch}.*"
        )


class TestValidate:
    """Tests for the :meth:`IrisRelease.validate` method."""
    @pytest.fixture(autouse=True)
    def _setup(self, mock_git_ls_remote_tags) -> None:
        self.instance = IrisRelease(
            _dry_run=True,
            github_user="user",
            patch_min_max_tag=("1.0.0", "1.1.0")
        )
        mock_git_ls_remote_tags.return_value = "abcd1234 refs/tags/1.0.0"

    class Case(NamedTuple):
        git_tag: str
        match: str

    @pytest.fixture(params=[
        pytest.param(
            Case("1.1.dev0", "development release.*cannot handle"),
            id="dev release",
        ),
        pytest.param(
            Case("1.1.post0", "post release.*cannot handle"),
            id="post release",
        ),
        pytest.param(
            Case("1.1.alpha0", "release candidate.*got 'a'"),
            id="pre-release non-rc",
        ),
        pytest.param(
            Case("1.1.1rc0", "PATCH release AND a release candidate.*cannot handle"),
            id="patch release rc",
        ),
        pytest.param(
            Case("1.1.1", "No previous releases.*cannot handle a PATCH"),
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
        self.instance.git_tag = "1.1.0"

    def test_first_in_series_not_rc_message(self, first_in_series_not_rc, capfd, mocker):
        # Two "yes" answers to arrive at the appropriate decision node.
        mock_inputs(mocker, "y", "y")
        self.instance.validate()
        out, err = capfd.readouterr()
        assert "No previous releases" in out
        assert "expected to be a release candidate" in out
        assert "sure you want to continue" in out

    def test_first_in_series_not_rc_exit(self, first_in_series_not_rc, mocker):
        # One "no" answer to arrive at the appropriate decision node.
        mock_inputs(mocker, "n")
        with pytest.raises(SystemExit):
            self.instance.validate()

    def test_first_in_series_not_rc_continue(self, first_in_series_not_rc, mocker):
        # Two "yes" answers to arrive at the appropriate decision node.
        mock_inputs(mocker, "y", "y")
        self.instance.validate()

    # Not an exhaustive list, just the inverse of the unhandled cases.
    @pytest.fixture(params=[
        pytest.param("2.0.0rc0", id="major release RC"),
        pytest.param("1.1.0rc0", id="minor release RC"),
        pytest.param("1.1.0", id="minor release existing major"),
        pytest.param("1.0.1", id="patch release existing minor"),
        pytest.param("1.1.0", id="first in series not RC"),
    ])
    def handled_cases(self, request) -> None:
        self.instance.git_tag = request.param

    def test_handled_cases(self, handled_cases, mocker, mock_wait_for_done):
        # One "yes" answer to arrive at the appropriate decision node.
        mock_inputs(mocker, "y")
        self.instance.validate()
        mock_wait_for_done.assert_called_once()
        (call,) = mock_wait_for_done.call_args_list
        assert_input_msg_regex(call, "Confirm that the details above are correct")


class TestUpdateStandardNames:
    """Tests for the :meth:`IrisRelease.update_standard_names` method."""
    @pytest.fixture(autouse=True)
    def _setup(self, mock_wait_for_done, mock_git_ls_remote_tags) -> None:
        self.instance = IrisRelease(_dry_run=True)
        self.mock_wait_for_done = mock_wait_for_done
        mock_git_ls_remote_tags.return_value = (
            "abcd1234  refs/tags/v1.0.0\n"
            "abcd1235  refs/tags/v1.0.1\n"
        )

    def test_not_first_in_series(self):
        # Not first in series - method does nothing.
        self.instance.git_tag = "v1.0.2"
        self.instance.update_standard_names()
        self.mock_wait_for_done.assert_not_called()

    def test_wait_messages(self):
        # First in series. No other branching behaviour, so just a cursory check
        #  for the expected messages.
        self.instance.git_tag = "v1.1.0"
        self.instance.update_standard_names()
        assert self.mock_wait_for_done.call_count == 5
        delete, checkout, update, pr, merge = self.mock_wait_for_done.call_args_list
        message_fragments = [
            (delete, "avoid a name clash by deleting any existing local branch"),
            (checkout, "Checkout a local branch from the official"),
            (update, "Update the CF standard names table"),
            (pr, "Create a Pull Request for your changes"),
            (merge, "Work with the development team to get the PR merged"),
        ]
        for call, expected in message_fragments:
            assert_input_msg_regex(call, expected)


class TestCheckDeprecations:
    """Tests for the :meth:`IrisRelease.check_deprecations` method."""
    @pytest.fixture(autouse=True)
    def _setup(self, mock_wait_for_done) -> None:
        self.instance = IrisRelease(_dry_run=True)
        self.mock_wait_for_done = mock_wait_for_done

    @pytest.mark.parametrize("git_tag", ["v1.1.0", "v1.1.1"])
    def test_not_major_release(self, git_tag):
        # Not a MAJOR release - method does nothing.
        self.instance.git_tag = git_tag
        self.instance.check_deprecations()
        self.mock_wait_for_done.assert_not_called()

    def test_major_release(self):
        # MAJOR release - code block is active.
        self.instance.git_tag = "v1.0.0"
        self.instance.check_deprecations()
        self.mock_wait_for_done.assert_called_once()
        (call,) = self.mock_wait_for_done.call_args_list
        assert_input_msg_regex(call, "be sure to finalise all deprecations")


class TestCreateReleaseBranch:
    """Tests for the :meth:`IrisRelease.create_release_branch` method."""
    @pytest.fixture(autouse=True)
    def _setup(self, mock_wait_for_done, mock_git_ls_remote_tags) -> None:
        self.instance = IrisRelease(_dry_run=True)
        self.mock_wait_for_done = mock_wait_for_done
        mock_git_ls_remote_tags.return_value = (
            "abcd1234  refs/tags/v1.0.0\n"
            "abcd1235  refs/tags/v1.0.1\n"
        )

    def test_first_in_series(self):
        self.instance.git_tag = "v1.1.0"
        self.instance.create_release_branch()
        self.mock_wait_for_done.assert_called_once()
        (call,) = self.mock_wait_for_done.call_args_list
        assert_input_msg_regex(
            call,
            f"create the ``{self.instance.version.branch}`` release branch"
        )

    def test_not_first_in_series(self):
        self.instance.git_tag = "v1.0.2"
        self.instance.create_release_branch()
        self.mock_wait_for_done.assert_called_once()
        (call,) = self.mock_wait_for_done.call_args_list
        assert_input_msg_regex(
            call,
            "If necessary: cherry-pick any specific commits that are needed",
        )


class TestFinaliseWhatsNew:
    """Tests for the :meth:`IrisRelease.finalise_whats_new` method."""
    class WaitMessages(enum.StrEnum):
        DELETE = "avoid a name clash by deleting any existing local branch"
        CHECKOUT = "Checkout a local branch from the official"
        CUT = "'Cut' the What's New for the release"
        REFS = r"Replace references to.*latest\.rst with.*{series}"
        TITLE = r"set the page title to.*\nv{series}"
        UNDERLINE = "ensure the page title underline is the exact same length"
        DROPDOWN_HIGHLIGHT = r"set the sphinx-design dropdown title.*\nv{series}"
        REFLECTION = "ensure it is a good reflection of what is new"
        HIGHLIGHTS = "populate the Release Highlights dropdown"
        DROPDOWN_PATCH = "Create a patch dropdown section"
        TEMPLATE = "Remove the What's New template file"
        PUSH = "Commit and push all the What's New changes"
        PR = "Create a Pull Request for your changes"
        MERGE = "Work with the development team to get the PR merged"

    @pytest.fixture(autouse=True)
    def _setup(self, mock_wait_for_done, mock_git_ls_remote_tags) -> None:
        self.instance = IrisRelease(_dry_run=True)
        self.mock_wait_for_done = mock_wait_for_done
        mock_git_ls_remote_tags.return_value = (
            "abcd1234  refs/tags/v1.0.0\n"
            "abcd1235  refs/tags/v1.0.1\n"
            "abcd1236  refs/tags/v1.1.0rc0\n"
        )

    def common_test(self, git_tag, expected_messages):
        self.instance.git_tag = git_tag
        self.instance.finalise_whats_new()
        assert self.mock_wait_for_done.call_count == len(expected_messages)
        for call, expected in zip(
            self.mock_wait_for_done.call_args_list,
            expected_messages,
        ):
            expected = expected.format(series=re.escape(self.instance.version.series[1:]))
            assert_input_msg_regex(call, expected)

    def test_first_in_series(self):
        expected_messages = [
            self.WaitMessages.DELETE,
            self.WaitMessages.CHECKOUT,
            self.WaitMessages.CUT,
            self.WaitMessages.REFS,
            self.WaitMessages.TITLE,
            self.WaitMessages.UNDERLINE,
            self.WaitMessages.DROPDOWN_HIGHLIGHT,
            self.WaitMessages.REFLECTION,
            self.WaitMessages.HIGHLIGHTS,
            self.WaitMessages.TEMPLATE,
            self.WaitMessages.PUSH,
            self.WaitMessages.PR,
            self.WaitMessages.MERGE,
        ]
        self.common_test("v1.2.0", expected_messages)

    def test_minor_not_first(self):
        expected_messages = [
            self.WaitMessages.DELETE,
            self.WaitMessages.CHECKOUT,
            self.WaitMessages.TITLE,
            self.WaitMessages.UNDERLINE,
            self.WaitMessages.DROPDOWN_HIGHLIGHT,
            self.WaitMessages.REFLECTION,
            self.WaitMessages.HIGHLIGHTS,
            self.WaitMessages.PUSH,
            self.WaitMessages.PR,
            self.WaitMessages.MERGE,
        ]
        self.common_test("v1.1.0", expected_messages)

    def test_patch(self):
        expected_messages = [
            self.WaitMessages.DELETE,
            self.WaitMessages.CHECKOUT,
            self.WaitMessages.DROPDOWN_PATCH,
            self.WaitMessages.PUSH,
            self.WaitMessages.PR,
            self.WaitMessages.MERGE,
        ]
        self.common_test("v1.0.2", expected_messages)


class TestCutRelease:
    """Tests for the :meth:`IrisRelease.cut_release` method."""
    class WaitMessages(enum.StrEnum):
        WEBPAGE = "Visit https://github.com/SciTools/iris/releases/new"
        TAG = "as the new tag to create, and also as the Release title"
        TEXT = "Populate the main text box"
        INSTALL_RC = "This is a release candidate - include the following instructions"
        TICK_RC = "This is a release candidate - tick the box"
        LATEST = "Tick the box to set this as the latest release"
        NOT_LATEST = "Un-tick the latest release box."
        PUBLISH = "Click: Publish release !"
        URL = "Visit https://github.com/SciTools/iris/actions/workflows/ci-wheels.yml"

    @pytest.fixture(autouse=True)
    def _setup(self, mock_wait_for_done, mock_git_ls_remote_tags) -> None:
        self.instance = IrisRelease(_dry_run=True)
        self.mock_wait_for_done = mock_wait_for_done
        mock_git_ls_remote_tags.return_value = (
            "abcd1234  refs/tags/v1.0.0\n"
            "abcd1235  refs/tags/v1.0.1\n"
            "abcd1236  refs/tags/v1.1.0\n"
        )

    def common_test(self, git_tag, expected_messages):
        self.instance.git_tag = git_tag
        self.instance.cut_release()
        assert self.mock_wait_for_done.call_count == len(expected_messages)
        for call, expected in zip(
            self.mock_wait_for_done.call_args_list,
            expected_messages,
        ):
            assert_input_msg_regex(call, expected)

    def test_latest(self):
        self.instance.git_tag = "v1.2.0"
        expected_messages = [
            self.WaitMessages.WEBPAGE,
            self.WaitMessages.TAG,
            self.WaitMessages.TEXT,
            self.WaitMessages.LATEST,
            self.WaitMessages.PUBLISH,
            self.WaitMessages.URL,
        ]
        self.common_test("v1.2.0", expected_messages)

    def test_not_latest(self):
        expected_messages = [
            self.WaitMessages.WEBPAGE,
            self.WaitMessages.TAG,
            self.WaitMessages.TEXT,
            self.WaitMessages.NOT_LATEST,
            self.WaitMessages.PUBLISH,
            self.WaitMessages.URL,
        ]
        self.common_test("v1.0.2", expected_messages)

    def test_release_candidate(self):
        expected_messages = [
            self.WaitMessages.WEBPAGE,
            self.WaitMessages.TAG,
            self.WaitMessages.TEXT,
            self.WaitMessages.INSTALL_RC,
            self.WaitMessages.TICK_RC,
            self.WaitMessages.PUBLISH,
            self.WaitMessages.URL,
        ]
        self.common_test("v1.2.0rc0", expected_messages)


class TestCheckRtd:
    """Tests for the :meth:`IrisRelease.check_rtd` method."""
    @pytest.fixture(autouse=True)
    def _setup(self, mock_wait_for_done, mock_git_ls_remote_tags) -> None:
        self.instance = IrisRelease(_dry_run=True)
        self.mock_wait_for_done = mock_wait_for_done
        mock_git_ls_remote_tags.return_value = (
            "abcd1234  refs/tags/v1.0.0\n"
            "abcd1235  refs/tags/v1.0.1\n"
            "abcd1236  refs/tags/v1.1.0\n"
        )

    @pytest.mark.parametrize("latest", [True, False], ids=["is_latest", "not_latest"])
    @pytest.mark.parametrize("rc", [True, False], ids=["is_rc", "not_rc"])
    def test_default(self, latest: bool, rc: bool):
        if latest:
            git_tag = "v1.2.0"
        else:
            git_tag = "v1.0.2"
        if rc:
            git_tag += "rc0"
        self.instance.git_tag = git_tag
        self.instance.check_rtd()
        series = re.escape(self.instance.version.series)
        expected_messages = [
            "Visit https://readthedocs.org/projects/scitools-iris/versions/",
            rf"{series}.* to Active, un-Hidden",
            rf"{series}.* to Active, Hidden",
            "Keep only the latest 2 branch doc builds active",
            rf"{series}.* is available in RTD's version switcher",
            rf"{series}.* is NOT available in RTD's version switcher",
        ]
        call_args_list = self.mock_wait_for_done.call_args_list
        assert self.mock_wait_for_done.call_count == len(expected_messages)
        for call, expected in zip(call_args_list, expected_messages):
            assert_input_msg_regex(call, expected)

        (check_message,) = call_args_list[4][0]
        check_expected = "Selecting 'stable' in the version switcher"
        if latest and not rc:
            assert check_expected in check_message
        else:
            assert check_expected not in check_message


class TestCheckPyPI:
    """Tests for the :meth:`IrisRelease.check_pypi` method."""
    class WaitMessages(enum.StrEnum):
        URL = "Confirm that the following URL is correctly populated"
        TOP = "{public} is at the top of this page"
        PRE_RELEASE = "{public} is marked as a pre-release on this page"
        TAG = "{public} is the tag shown on the scitools-iris PyPI homepage"
        INSTALL = "Confirm that pip install works as expected"

    @pytest.fixture(autouse=True)
    def _setup(self, mock_wait_for_done, mock_git_ls_remote_tags, mocker) -> None:
        self.instance = IrisRelease(_dry_run=True)
        self.mock_wait_for_done = mock_wait_for_done
        # For the PyPI SHA256 input.
        mock_inputs(
            mocker,
            "ccc8025d24b74d86ab780266cb9f708c468ac53426a45fab20bfc315c68383f7",
        )
        mock_git_ls_remote_tags.return_value = (
            "abcd1234  refs/tags/v1.0.0\n"
            "abcd1235  refs/tags/v1.0.1\n"
            "abcd1236  refs/tags/v1.2.0\n"
        )

    def common_test(self, git_tag, expected_messages):
        self.instance.git_tag = git_tag
        self.instance.check_pypi()
        assert self.mock_wait_for_done.call_count == len(expected_messages)
        for call, expected in zip(
            self.mock_wait_for_done.call_args_list,
            expected_messages,
        ):
            expected = expected.format(public=re.escape(self.instance.version.public))
            assert_input_msg_regex(call, expected)

    def test_latest(self):
        expected_messages = [
            self.WaitMessages.URL,
            self.WaitMessages.TOP,
            self.WaitMessages.TAG,
            self.WaitMessages.INSTALL,
        ]
        self.common_test("v1.3.0", expected_messages)

    def test_not_latest(self):
        expected_messages = [
            self.WaitMessages.URL,
            self.WaitMessages.INSTALL,
        ]
        self.common_test("v1.0.2", expected_messages)

    def test_release_candidate(self):
        expected_messages = [
            self.WaitMessages.URL,
            self.WaitMessages.PRE_RELEASE,
            self.WaitMessages.INSTALL,
        ]
        self.common_test("v1.1.0rc0", expected_messages)

    def test_latest_and_rc(self):
        expected_messages = [
            self.WaitMessages.URL,
            self.WaitMessages.TOP,
            self.WaitMessages.PRE_RELEASE,
            self.WaitMessages.INSTALL,
        ]
        self.common_test("v1.3.0rc0", expected_messages)

    def test_sha256_input(self, mocker, capfd):
        self.instance.git_tag = "v1.3.0"
        fake_sha = "3b2f4091883d1e401192b4f64aead9e4bbdb84854b74c984614d79742b2fab96"
        mock_inputs(mocker, fake_sha)
        self.instance.check_pypi()
        out, err = capfd.readouterr()
        assert "Visit the below and click `view details`" in out
        assert self.instance.sha256 == fake_sha

    def test_invalid_sha(self, mocker, mock_report_problem):
        self.instance.git_tag = "v1.3.0"
        fake_sha = "3b2f4091883d1e401192b4f64aead9e4bbdb84854b74c984614d79742b2fab96"
        mock_inputs(mocker, "not-a-sha", fake_sha)
        self.instance.check_pypi()
        mock_report_problem.assert_called_once_with(
            "Invalid SHA256 hash. Please try again ..."
        )
        assert self.instance.sha256 == fake_sha

    def test_sha_default_value_preserved(self, mocker):
        self.instance.git_tag = "v1.3.0"
        fake_sha = "3b2f4091883d1e401192b4f64aead9e4bbdb84854b74c984614d79742b2fab96"
        self.instance.sha256 = fake_sha
        mock_inputs(mocker, "")
        self.instance.check_pypi()
        assert self.instance.sha256 == fake_sha


class TestUpdateCondaForge:
    """Tests for the :meth:`IrisRelease.update_conda_forge` method."""
    # TODO: Confirming this one behaves correctly is a nightmare. There is more
    #  conditional branching than elsewhere. Suggestions welcome.
    class WaitMessages(enum.StrEnum):
        FORK = "Make sure you have a GitHub fork of"
        RC_BRANCHES = "Visit the conda-forge feedstock branches page"
        # `rc-original` = just the value used in these tests
        RC_ARCHIVE = "Archive the rc-original branch"
        CHECKOUT = "Checkout a new branch for the conda-forge"

        UPDATE = re.escape("Update ./recipe/meta.yaml:") + ".*unsure\.$"
        UPDATE_NOT_LATEST = re.escape("Update ./recipe/meta.yaml:") + ".*unsure\..*{version} is not the latest Iris release"

        PUSH = "push up the changes to prepare for a Pull Request"
        PR = "Create a Pull Request for your changes"

        AUTO = "Follow the automatic conda-forge guidance.*Pull Request\.$"
        AUTO_RC = "Follow the automatic conda-forge guidance.*Pull Request\..*release candidate"

        MAINTAINERS = "Work with your fellow feedstock maintainers"
        CI = "wait for the CI to complete"
        LIST = r"Confirm that {public} appears in this list:"
        LATEST = "is displayed on this page as the latest available"
        TESTING = "The new release will now undergo testing and validation"
        INSTALL = re.escape("Confirm that conda (or mamba) install works as expected")
        PATCH = r"{version} is not the latest Iris release"

    @pytest.fixture(autouse=True)
    def _setup(self, mock_wait_for_done, mock_git_ls_remote_tags) -> None:
        self.instance = IrisRelease(_dry_run=True)
        self.mock_wait_for_done = mock_wait_for_done
        mock_git_ls_remote_tags.return_value = (
            "abcd1234  refs/tags/v1.0.0\n"
            "abcd1235  refs/tags/v1.0.1\n"
            "abcd1236  refs/tags/v1.1.0\n"
            "abcd1237  refs/tags/v2.0.0\n"
        )

    @pytest.mark.parametrize("latest", [True, False], ids=["is_latest", "not_latest"])
    @pytest.mark.parametrize("rc", [True, False], ids=["is_rc", "not_rc"])
    @pytest.mark.parametrize("more_patches", [True, False], ids=["more_patches", "no_more_patches"])
    def test_waits(self, latest: bool, rc: bool, more_patches: bool, mocker):
        if latest:
            git_tag = "v2.1"
        else:
            git_tag = "v1.2"
        if more_patches:
            git_tag += ".1"
        else:
            git_tag += ".0"
        if rc:
            git_tag += "rc0"
        self.instance.git_tag = git_tag
        if more_patches:
            self.instance.patch_min_max_tag = (git_tag, "v2.2.1")

        # All inputs relate to handling of the release candidate branch. We
        #  choose the inputs that allow exercising every wait message.
        mock_inputs(mocker, "rc-original", "y", "rc-new")

        expected_messages = list(self.WaitMessages)
        if not rc:
            expected_messages.remove(self.WaitMessages.RC_BRANCHES)
            expected_messages.remove(self.WaitMessages.RC_ARCHIVE)
            expected_messages.remove(self.WaitMessages.AUTO_RC)
        else:
            expected_messages.remove(self.WaitMessages.AUTO)

        if latest:
            expected_messages.remove(self.WaitMessages.UPDATE_NOT_LATEST)
        else:
            expected_messages.remove(self.WaitMessages.UPDATE)

        if rc or not latest:
            expected_messages.remove(self.WaitMessages.LATEST)

        if latest or more_patches:
            expected_messages.remove(self.WaitMessages.PATCH)

        self.instance.update_conda_forge()
        assert self.mock_wait_for_done.call_count == len(expected_messages)
        for call, expected in zip(
            self.mock_wait_for_done.call_args_list,
            expected_messages,
        ):
            expected_str = expected.format(
                public=re.escape(self.instance.version.public),
                version=re.escape(str(self.instance.version)),
            )
            assert_input_msg_regex(call, expected_str)

    def test_original_rc_branch_name(self, mocker):
        self.instance.git_tag = "v2.1.0rc0"
        mock_inputs(mocker, "my-special-rc-branch", "y", "rc-new")
        self.instance.update_conda_forge()
        wait_messages = [
            call.args[0] for call in self.mock_wait_for_done.call_args_list
        ]
        expected = self.WaitMessages.RC_ARCHIVE.replace("rc-original", "my-special-rc-branch")
        not_expected = self.WaitMessages.RC_ARCHIVE
        assert any(re.search(expected, m) for m in wait_messages)
        assert not any(re.search(not_expected, m) for m in wait_messages)

    @pytest.mark.parametrize("rc", [True, False], ids=["is_rc", "not_rc"])
    def test_new_rc_branch_name(self, rc, mocker):
        git_tag = "v1.2.0"
        if rc:
            git_tag += "rc0"
        self.instance.git_tag = git_tag
        mock_inputs(mocker, "rc-original", "y", "rc-new")
        self.instance.update_conda_forge()
        all_calls = [call.args[0] for call in self.mock_wait_for_done.call_args_list]
        calls = [
            call for call in all_calls
            if any(phrase in call for phrase in [
                "Checkout a new branch",
                "Create a Pull Request",
                "branch needs to be restored",
            ])
        ]
        expected = "rc-new" if rc else "main"
        assert all(expected in c for c in calls)

    def test_young_rc_branch(self, mocker):
        self.instance.git_tag = "v2.1.0rc0"
        mock_inputs(mocker, "rc-original", "n")
        self.instance.update_conda_forge()
        wait_messages = [
            call.args[0] for call in self.mock_wait_for_done.call_args_list
        ]
        regex = re.compile(self.WaitMessages.RC_ARCHIVE)
        assert all(regex.search(m) is None for m in wait_messages)

    def test_invalid_rc_branch_age(self, mocker, mock_report_problem):
        self.instance.git_tag = "v2.1.0rc0"
        # Invalid entry, then valid "n".
        mock_inputs(mocker, "rc-original", "maybe", "n")
        self.instance.update_conda_forge()
        mock_report_problem.assert_called_once_with(
            "Invalid entry. Please try again ..."
        )

    @pytest.mark.parametrize("rc", [True, False], ids=["is_rc", "not_rc"])
    def test_channel_command(self, rc, mocker):
        git_tag = "v1.2.0"
        if rc:
            git_tag += "rc0"
        self.instance.git_tag = git_tag
        mock_inputs(mocker, "rc-original", "n")
        self.instance.update_conda_forge()
        if rc:
            assert any("label/rc_iris" in call.args[0] for call in self.mock_wait_for_done.call_args_list)
        else:
            assert not any("label/rc_iris" in call.args[0] for call in self.mock_wait_for_done.call_args_list)


class TestUpdateLinks:
    """Tests for the :meth:`IrisRelease.update_links` method."""
    @pytest.fixture(autouse=True)
    def _setup(self, mock_wait_for_done) -> None:
        self.instance = IrisRelease(_dry_run=True, git_tag="v1.2.0")
        self.mock_wait_for_done = mock_wait_for_done

    def test_waits(self, mocker):
        mock_inputs(mocker, "some-url")
        self.instance.update_links()
        assert self.mock_wait_for_done.call_count == 3
        revisit, update, comment = self.mock_wait_for_done.call_args_list
        message_fragments = [
            (revisit, "Revisit the GitHub release:"),
            (update, "Update .* with the above links and anything else appropriate"),
            (comment, "notify anyone watching"),
        ]
        for call, expected in message_fragments:
            assert_input_msg_regex(call, expected)

    def test_url_input(self, mocker, capfd):
        mock_inputs(mocker, "some-url")
        self.instance.update_links()
        out, err = capfd.readouterr()
        assert "What is the URL for the GitHub discussions page" in out
        revisit, update, comment = self.mock_wait_for_done.call_args_list
        assert_input_msg_regex(update, "some-url")
        assert_input_msg_regex(comment, "some-url")


class TestBlueskyAnnounce:
    """Tests for the :meth:`IrisRelease.bluesky_announce` method."""
    @pytest.fixture(autouse=True)
    def _setup(self, mock_wait_for_done, mock_git_ls_remote_tags) -> None:
        self.instance = IrisRelease(_dry_run=True)
        self.mock_wait_for_done = mock_wait_for_done
        mock_git_ls_remote_tags.return_value = (
            "abcd1234  refs/tags/v1.0.0\n"
            "abcd1235  refs/tags/v1.0.1\n"
        )

    @pytest.mark.parametrize("first_in_series", [True, False], ids=["first_in_series", "not_first_in_series"])
    def test_wait(self, first_in_series: bool):
        if first_in_series:
            git_tag = "v1.1.0"
        else:
            git_tag = "v1.0.2"
        self.instance.git_tag = git_tag
        self.instance.bluesky_announce()
        self.mock_wait_for_done.assert_called_once()
        (call,) = self.mock_wait_for_done.call_args_list
        assert_input_msg_regex(call, "Announce the release")
        if not first_in_series:
            assert_input_msg_regex(call, "Consider replying within an existing")


class TestMergeBack:
    """Tests for the :meth:`IrisRelease.merge_back` method."""
    # TODO: figure out how to test this one - more complex than the rest.

    class WaitMessages(enum.StrEnum):
        DELETE = "avoid a name clash by deleting any existing local branch"
        CHECKOUT = "Checkout a local branch from the official"
        MERGE_IN = "Merge in the commits from {branch}"
        TEMPLATE = "Recreate the What's New template"
        LATEST = "Recreate the What's New latest"
        GUIDANCE = "Follow any guidance in .*latest\.rst"
        INDEX = "Add .*latest\.rst to the top of the list"
        PUSH = "Commit and push all the What's New changes"
        PR = "Create a Pull Request for your changes"
        RISKY = "COMBINING BRANCHES CAN BE RISKY"
        PR_MERGE = "Work with the development team to get the PR merged"
        NEXT_PATCH = "Run the following command in a new terminal"

    @pytest.fixture(autouse=True)
    def _setup(self, mock_wait_for_done, mock_git_ls_remote_tags):
        self.instance = IrisRelease(_dry_run=True)
        self.mock_wait_for_done = mock_wait_for_done
        mock_git_ls_remote_tags.return_value = (
            "abcd1234  refs/tags/v1.0.0\n"
            "abcd1235  refs/tags/v1.1.0\n"
            "abcd1236  refs/tags/v1.2.0\n"
        )

    @pytest.mark.parametrize("first", [True, False], ids=["first_in_series", "not_first_in_series"])
    @pytest.mark.parametrize("more_patches", [True, False], ids=["more_patches", "no_more_patches"])
    def test_waits(self, first, more_patches):
        if first and more_patches:
            pytest.skip("first_in_series and more_patches are mutually exclusive in reality.")
        if first:
            git_tag = "v1.3.0"
        else:
            git_tag = "v1.0.1"
        self.instance.git_tag = git_tag
        if more_patches:
            self.instance.patch_min_max_tag = (git_tag, "v1.2.1")

        expected_messages = list(self.WaitMessages)
        if not first:
            expected_messages.remove(self.WaitMessages.TEMPLATE)
            expected_messages.remove(self.WaitMessages.LATEST)
            expected_messages.remove(self.WaitMessages.GUIDANCE)
            expected_messages.remove(self.WaitMessages.INDEX)
            expected_messages.remove(self.WaitMessages.PUSH)
        if not more_patches:
            expected_messages.remove(self.WaitMessages.NEXT_PATCH)

        self.instance.merge_back()
        assert self.mock_wait_for_done.call_count == len(expected_messages)
        for call, expected in zip(
            self.mock_wait_for_done.call_args_list,
            expected_messages,
        ):
            expected = expected.format(branch=re.escape(self.instance.version.branch))
            assert_input_msg_regex(call, expected)

    @pytest.mark.parametrize("more_patches", [True, False], ids=["more_patches", "no_more_patches"])
    def test_branches(self, more_patches):
        self.instance.git_tag = "v1.0.1"
        if more_patches:
            self.instance.patch_min_max_tag = ("v1.0.1", "v1.2.1")
            target_branch = "v1.1.x"
            working_branch = "v1.0.1-to-v1.1.x"
        else:
            target_branch = "main"
            working_branch = "v1.0.x.mergeback"

        self.instance.merge_back()
        wait_messages = [
            call.args[0] for call in self.mock_wait_for_done.call_args_list
        ]
        # Use CHECKOUT as the test since it contains target_ and working_branch.
        (checkout_message,) = [
            m for m in wait_messages if re.search(self.WaitMessages.CHECKOUT, m)
        ]
        pattern = re.compile(rf"git checkout .*{target_branch} -b {working_branch}")
        assert pattern.search(checkout_message) is not None

    def test_next_series_error(self, mocker):
        self.instance.git_tag = "v1.0.1"
        self.instance.patch_min_max_tag = ("v1.0.1", "v1.2.1")
        _ = mocker.patch.object(
            IrisRelease,
            "_get_tagged_versions",
            return_value=[IrisVersion("v1.0.0")],
        )
        with pytest.raises(RuntimeError, match="Error finding next series"):
            self.instance.merge_back()

    def test_next_patch_file(self):
        self.instance.git_tag = "v1.0.1"
        self.instance.patch_min_max_tag = ("v1.0.1", "v1.2.1")
        expected_file = self.instance._get_file_stem().with_name("v1_1_1.json")
        self.instance.merge_back()
        assert expected_file.exists()
        next_patch = IrisRelease.load(expected_file, dry_run=True)
        assert next_patch.latest_complete_step == IrisRelease.get_steps().index(IrisRelease.validate) - 1
        assert next_patch.git_tag == "v1.1.1"
        assert next_patch.patch_min_max == (IrisVersion("v1.0.1"), IrisVersion("v1.2.1"))


class TestNextRelease:
    """Tests for the :meth:`IrisRelease.next_release` method."""
    @pytest.fixture(autouse=True)
    def _setup(self, mock_wait_for_done) -> None:
        self.instance = IrisRelease(_dry_run=True)
        self.mock_wait_for_done = mock_wait_for_done

    @pytest.mark.parametrize("patch", [True, False], ids=["patch", "not_patch"])
    @pytest.mark.parametrize("rc", [True, False], ids=["rc", "not_rc"])
    def test_waits(self, patch: bool, rc: bool):
        if patch:
            git_tag = "v1.1.1"
        else:
            git_tag = "v1.2.0"
        if rc:
            git_tag += "rc0"
        self.instance.git_tag = git_tag
        self.instance.next_release()
        if not patch and not rc:
            assert self.mock_wait_for_done.call_count == 5
            manager, milestone, discussion, sprints, champion = self.mock_wait_for_done.call_args_list
            message_fragments = [
                (manager, "Confirm that there is a release manager"),
                (milestone, "has set up a milestone for their release"),
                (discussion, "has set up a discussion page for their release"),
                (sprints, "has arranged some team development time"),
                (champion, "importance of regularly championing their release"),
            ]
            for call, expected in message_fragments:
                assert_input_msg_regex(call, expected)
        else:
            self.mock_wait_for_done.assert_not_called()
