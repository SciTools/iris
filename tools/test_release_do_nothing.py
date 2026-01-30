# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the ``release_do_nothing.py`` file."""

from pathlib import Path
from typing import NamedTuple

import pytest
from pytest_mock import MockType

import nothing
from release_do_nothing import IrisRelease, IrisVersion


@pytest.fixture(autouse=True)
def mock_fast_print(mocker) -> None:
    """Prevent the mod:`nothing` print methods from sleeping."""
    mocker.patch.object(nothing, "sleep", return_value=None)


@pytest.fixture(autouse=True)
def mock_git_commands(mocker) -> None:
    """Detach testing from reliance on .git directory."""
    return_value = (
        "origin    git@github.com:myself/iris.git (fetch)\n"
        "origin    git@github.com:myself/iris.git (push)\n"
        "upstream  git@github.com:SciTools/iris.git (fetch)\n"
        "upstream  no_push (push)\n"
        "foo       git@github.com:foo/iris.git (fetch)\n"
        "foo       git@github.com:foo/iris.git (push)\n"
    )
    mocker.patch.object(
        IrisRelease,
        "_git_remote_v",
        return_value=return_value,
    )

    mocker.patch.object(
        IrisRelease,
        "_git_remote_get_url",
        return_value="git@github.com:myself/iris.git",
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

    @pytest.mark.parametrize("git_tag", ["0.0.1", "9.1.1"])
    def test_is_latest_tag(self, git_tag):
        expecteds = {"0.0.1": False, "9.1.1": True}
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

    @pytest.mark.parametrize("git_tag", ["9.1.0", "1.1.1"])
    def test_first_in_series(self, git_tag):
        expecteds = {"9.1.0": True, "1.1.1": False}
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
    def _setup(self) -> None:
        self.instance = IrisRelease(
            _dry_run=True,
        )

    def test_github_scitools(self, mocker):
        # Developer is asked to select their Iris fork from a list.
        #  See mock_git_commands()
        mock_inputs(mocker, "0")
        self.instance.analyse_remotes()
        assert self.instance.github_scitools == "upstream"

    def test_no_forks(self, mocker):
        # The only remote is 'upstream', so error.
        return_value = (
            "upstream  git@github.com:SciTools/iris.git (fetch)\n"
            "upstream  no_push (push)\n"
        )
        mocker.patch.object(
            IrisRelease,
            "_git_remote_v",
            return_value=return_value,
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

    def test_error_deriving_username(self, mocker):
        mocker.patch.object(
            IrisRelease,
            "_git_remote_get_url",
            return_value="bad_url",
        )
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
    def _setup(self) -> None:
        self.instance = IrisRelease(
            _dry_run=True,
            github_scitools="upstream",
            github_fork="origin",
            github_user="myself",
        )

    def test_valid_tag(self, mocker):
        # User inputs a valid, non-existing tag
        mock_inputs(mocker, "v9.2.0")
        self.instance.get_release_tag()
        assert self.instance.git_tag == "v9.2.0"

    def test_existing_tag(self, mocker, mock_report_problem):
        # User tries an existing tag, then provides a valid one
        mock_inputs(mocker, "v1.1.0", "v9.2.0")
        self.instance.get_release_tag()
        mock_report_problem.assert_called_once_with(
            "Version v1.1.0 already exists as a git tag. Please try again ..."
        )
        assert self.instance.git_tag == "v9.2.0"

    def test_invalid_version_format(self, mocker, mock_report_problem):
        # User inputs invalid version format, then valid one
        mock_inputs(mocker, "not-a-version", "v9.2.0")
        self.instance.get_release_tag()
        assert mock_report_problem.call_count == 1
        (call,) = mock_report_problem.call_args_list
        (message,) = call.args
        assert "Packaging error" in message
        assert "Please try again" in message
        assert self.instance.git_tag == "v9.2.0"

    def test_default_value_preserved(self, mocker):
        # When loading from saved state, existing git_tag should be offered as default
        self.instance.git_tag = "v9.2.0"
        mock_inputs(mocker, "")  # User accepts default
        self.instance.get_release_tag()
        assert self.instance.git_tag == "v9.2.0"


class TestGetAllPatches:
    """Tests for the :meth:`IrisRelease.get_all_patches` method."""
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.instance = IrisRelease(
            _dry_run=True,
            github_scitools="upstream",
            github_fork="origin",
            github_user="myself",
            git_tag="v1.1.1",
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

    def test_starts_with_earlier_patch(self, mocker):
        # When patch_min is earlier than current git_tag, git_tag is updated
        mock_inputs(mocker, "0,2")
        self.instance.get_all_patches()
        # TODO: assert for message.
        assert self.instance.git_tag == "v1.0.3"
        assert self.instance.patch_min_max_tag == ("v1.0.3", "v1.2.1")

    def test_default_value_preserved(self, mocker):
        # When loading from saved state, existing patch_min_max_tag should work
        self.instance.patch_min_max_tag = ("v1.0.3", "v1.2.1")
        mock_inputs(mocker, "")
        self.instance.get_all_patches()
        assert self.instance.patch_min_max_tag == ("v1.0.3", "v1.2.1")


class TestApplyPatches:
    """Tests for the :meth:`IrisRelease.apply_patches` method."""
    pass


class TestValidate:
    """Tests for the :meth:`IrisRelease.validate` method."""
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.instance = IrisRelease(
            _dry_run=True,
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
        mock_inputs(mocker, "y", "y")
        self.instance.validate()
        out, err = capfd.readouterr()
        assert "No previous releases" in out
        assert "expected to be a release candidate" in out
        assert "sure you want to continue" in out

    def test_first_in_series_not_rc_exit(self, first_in_series_not_rc, mocker):
        mock_inputs(mocker, "n")
        with pytest.raises(SystemExit):
            self.instance.validate()

    def test_first_in_series_not_rc_continue(self, first_in_series_not_rc, mocker):
        mock_inputs(mocker, "y", "y")
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

    def test_handled_cases(self, handled_cases, mocker, mock_wait_for_done):
        sub_message = "Confirm that the details above are correct"
        mock_inputs(mocker, "y")
        self.instance.validate()
        mock_wait_for_done.assert_called_once()
        (call,) = mock_wait_for_done.call_args_list
        (message,) = call.args
        assert sub_message in message


class TestUpdateStandardNames:
    """Tests for the :meth:`IrisRelease.update_standard_names` method."""
    pass


class TestCheckDeprecations:
    """Tests for the :meth:`IrisRelease.check_deprecations` method."""
    pass


class TestCreateReleaseBranch:
    """Tests for the :meth:`IrisRelease.create_release_branch` method."""
    pass


class TestFinaliseWhatsNew:
    """Tests for the :meth:`IrisRelease.finalise_whats_new` method."""
    pass


class TestCutRelease:
    """Tests for the :meth:`IrisRelease.cut_release` method."""
    pass


class TestCheckRtd:
    """Tests for the :meth:`IrisRelease.check_rtd` method."""
    pass


class TestCheckPyPI:
    """Tests for the :meth:`IrisRelease.check_pypi` method."""
    pass


class TestUpdateCondaForge:
    """Tests for the :meth:`IrisRelease.update_conda_forge` method."""
    pass


class TestUpdateLinks:
    """Tests for the :meth:`IrisRelease.update_links` method."""
    pass


class TestBlueskyAnnounce:
    """Tests for the :meth:`IrisRelease.bluesky_announce` method."""
    pass


class TestMergeBack:
    """Tests for the :meth:`IrisRelease.merge_back` method."""
    pass


class TestNextRelease:
    """Tests for the :meth:`IrisRelease.next_release` method."""
    pass
